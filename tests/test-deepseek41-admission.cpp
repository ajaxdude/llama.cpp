#include "../src/llama-dsv41-admission.h"
#include "../src/llama-dsv41.h"

#include "ggml.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#define REQUIRE(cond) do { if (!(cond)) { throw std::runtime_error("requirement failed: " #cond); } } while (0)

namespace {

struct temp_procfs {
    std::filesystem::path path;

    temp_procfs() {
        static uint64_t sequence = 0;
        const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
        path = std::filesystem::temp_directory_path() /
                ("llama-dsv41-admission-" + std::to_string(stamp) + "-" + std::to_string(++sequence));
        std::filesystem::create_directories(path);
    }

    ~temp_procfs() {
        std::error_code ec;
        std::filesystem::remove_all(path, ec);
    }

    void write(const char * name, const std::string & value) {
        std::ofstream file(path / name);
        REQUIRE((bool) file);
        file << value;
        REQUIRE((bool) file);
    }
};

template<typename F>
std::string thrown(F && fn) {
    try {
        fn();
    } catch (const std::exception & e) {
        return e.what();
    }
    throw std::runtime_error("expected exception");
}

std::vector<llama_expert_store_tensor> published_tensors() {
    std::vector<llama_expert_store_tensor> result;
    uint64_t offset = 4096;
    for (int32_t il = 0; il < (int32_t) LLAMA_DSV41_N_LAYER; ++il) {
        for (llama_expert_projection projection : {
                    LLAMA_EXPERT_PROJECTION_GATE,
                    LLAMA_EXPERT_PROJECTION_UP,
                    LLAMA_EXPERT_PROJECTION_DOWN }) {
            llama_expert_store_tensor tensor;
            tensor.name = "blk." + std::to_string(il) + ".expert." + std::to_string((int) projection);
            tensor.fname = "published.gguf";
            tensor.layer = il;
            tensor.projection = projection;
            tensor.type = projection == LLAMA_EXPERT_PROJECTION_DOWN ? GGML_TYPE_Q2_K : GGML_TYPE_IQ2_XXS;
            tensor.ne[0] = projection == LLAMA_EXPERT_PROJECTION_DOWN ? 2304 : 5120;
            tensor.ne[1] = projection == LLAMA_EXPERT_PROJECTION_DOWN ? 5120 : 2304;
            tensor.ne[2] = LLAMA_DSV41_N_EXPERT;
            tensor.nb[0] = ggml_type_size(tensor.type);
            tensor.nb[1] = ggml_row_size(tensor.type, tensor.ne[0]);
            tensor.nb[2] = tensor.nb[1]*tensor.ne[1];
            tensor.file_offset = offset;
            tensor.file_size = offset + tensor.nb[2]*tensor.ne[2];
            offset = tensor.file_size;
            result.push_back(std::move(tensor));
        }
    }
    return result;
}

llama_dsv41_host_memory host_with_used(uint64_t used) {
    llama_dsv41_host_memory host;
    host.total = 128ULL << 30;
    host.available = host.total - used;
    host.used = used;
    return host;
}

llama_dsv41_admission_params base_params() {
    llama_dsv41_admission_params params;
    params.n_ubatch = 32;
    params.n_vocab = LLAMA_DSV41_N_VOCAB;
    params.n_expert_used = LLAMA_DSV41_N_EXPERT_USED;
    params.state_bytes = 2ULL << 30;
    return params;
}

void test_procfs() {
    temp_procfs procfs;
    procfs.write("meminfo",
            "MemTotal:       131072000 kB\n"
            "MemFree:          100000 kB\n"
            "MemAvailable:  120000000 kB\n");
    procfs.write("swaps", "Filename Type Size Used Priority\n");
    const auto memory = llama_dsv41_read_host_memory(procfs.path.string());
    REQUIRE(memory.total == 131072000ULL*1024);
    REQUIRE(memory.available == 120000000ULL*1024);
    REQUIRE(memory.used == 11072000ULL*1024);
    REQUIRE(memory.swap_entries == 0);

    procfs.write("swaps",
            "Filename Type Size Used Priority\n"
            "/swapfile file 33554428 0 -2\n");
    const auto swapped = llama_dsv41_read_host_memory(procfs.path.string());
    REQUIRE(swapped.swap_entries == 1);
    REQUIRE(swapped.swap_bytes == 33554428ULL*1024);
    REQUIRE(thrown([&]() {
        llama_dsv41_admit(swapped, 0, published_tensors(), base_params());
    }).find("category=swap") != std::string::npos);
}

void test_procfs_fail_closed() {
    temp_procfs procfs;
    procfs.write("meminfo", "MemTotal: 10 kB\n");
    procfs.write("swaps", "Filename Type Size Used Priority\n");
    REQUIRE(!thrown([&]() {
        llama_dsv41_read_host_memory(procfs.path.string());
    }).empty());

    procfs.write("meminfo",
            "MemTotal: 18446744073709551615 kB\n"
            "MemAvailable: 1 kB\n");
    REQUIRE(thrown([&]() {
        llama_dsv41_read_host_memory(procfs.path.string());
    }).find("overflow") != std::string::npos);

    procfs.write("meminfo", "MemTotal: 10 bytes\nMemAvailable: 1 kB\n");
    REQUIRE(!thrown([&]() {
        llama_dsv41_read_host_memory(procfs.path.string());
    }).empty());
}

void test_published_slot_fit() {
    const auto tensors = published_tensors();
    auto params = base_params();
    params.device_reported_bytes = 64ULL << 30;
    const auto result = llama_dsv41_admit(
            host_with_used(8ULL << 30),
            9376ULL << 20,
            tensors,
            params);
    REQUIRE(result.expert_slot_bytes == 398131200);
    REQUIRE(result.expert_staging_slot_bytes == 9953280);
    REQUIRE(result.expert_slots >= LLAMA_DSV41_N_EXPERT_USED);
    REQUIRE(result.expert_slots < LLAMA_DSV41_N_EXPERT);
    REQUIRE(result.expert_cache_bytes == result.expert_slot_bytes*result.expert_slots);
    REQUIRE(result.projected_bytes <= result.soft_bytes);
    REQUIRE(result.device_reported_bytes_ignored == 64ULL << 30);
}

void test_configured_cache() {
    const auto tensors = published_tensors();
    auto params = base_params();
    params.n_ubatch = 2;
    params.configured_cache_slots = 12;
    params.configured_cache_bytes = 12*398131200ULL + 1024;
    const auto result = llama_dsv41_admit(host_with_used(0), 0, tensors, params);
    REQUIRE(result.expert_slots == 12);
    REQUIRE(result.expert_cache_bytes == 12*398131200ULL);

    params.configured_cache_slots = 13;
    REQUIRE(thrown([&]() {
        llama_dsv41_admit(host_with_used(0), 0, tensors, params);
    }).find("disagree") != std::string::npos);

    params.configured_cache_slots = 5;
    params.configured_cache_bytes = 0;
    REQUIRE(thrown([&]() {
        llama_dsv41_admit(host_with_used(0), 0, tensors, params);
    }).find("routed expert union") != std::string::npos);

    params.configured_cache_slots = LLAMA_DSV41_N_EXPERT + 1;
    REQUIRE(thrown([&]() {
        llama_dsv41_admit(host_with_used(0), 0, tensors, params);
    }).find("published expert count") != std::string::npos);
}

void test_threshold_boundaries() {
    const auto tensors = published_tensors();
    auto params = base_params();
    params.n_ubatch = 1;
    params.configured_cache_slots = LLAMA_DSV41_N_EXPERT_USED;
    params.configured_cache_bytes = params.configured_cache_slots*398131200ULL;
    params.safety_margin_bytes = 1;

    const auto zero = llama_dsv41_admit(host_with_used(0), 0, tensors, params);
    const uint64_t planned_without_host = zero.projected_bytes;
    const auto exact_soft = llama_dsv41_admit(
            host_with_used(params.soft_bytes - planned_without_host), 0, tensors, params);
    REQUIRE(exact_soft.projected_bytes == params.soft_bytes);

    const std::string watchdog_error = thrown([&]() {
        llama_dsv41_admit(
                host_with_used(params.watchdog_bytes - planned_without_host), 0, tensors, params);
    });
    REQUIRE(watchdog_error.find("category=cache") != std::string::npos);
    REQUIRE(watchdog_error.find("watchdog=126701535232") != std::string::npos);

    REQUIRE(thrown([&]() {
        llama_dsv41_admit(host_with_used(params.hard_bytes), 0, tensors, params);
    }).find("category=hard") != std::string::npos);

    params.soft_bytes = LLAMA_DSV41_ADMISSION_SOFT_BYTES + 1;
    REQUIRE(thrown([&]() {
        llama_dsv41_admit(host_with_used(0), 0, tensors, params);
    }).find("category=thresholds") != std::string::npos);

    params.soft_bytes = LLAMA_DSV41_ADMISSION_SOFT_BYTES;
    params.watchdog_bytes = LLAMA_DSV41_WATCHDOG_EMERGENCY_BYTES + 1;
    REQUIRE(thrown([&]() {
        llama_dsv41_admit(host_with_used(0), 0, tensors, params);
    }).find("category=thresholds") != std::string::npos);
}

void test_context_progression() {
    const auto tensors = published_tensors();
    uint64_t previous_state = 0;
    uint64_t previous_workspace = 0;
    for (uint32_t n_ctx : { 32768U, 65536U, 98304U, 131072U }) {
        auto params = base_params();
        params.n_ctx = n_ctx;
        params.state_bytes = static_cast<uint64_t>(n_ctx)*65536;
        const auto result = llama_dsv41_admit(host_with_used(0), 0, tensors, params);
        REQUIRE(result.n_ctx == n_ctx);
        REQUIRE(result.state_bytes > previous_state);
        REQUIRE(result.graph_workspace_bytes > previous_workspace);
        previous_state = result.state_bytes;
        previous_workspace = result.graph_workspace_bytes;
    }

    auto params = base_params();
    params.n_ctx = 49152;
    REQUIRE(thrown([&]() {
        llama_dsv41_admit(host_with_used(0), 0, tensors, params);
    }).find("32768, 65536, 98304, or 131072") != std::string::npos);
}

void test_diagnostics_and_guards() {
    const auto tensors = published_tensors();
    auto params = base_params();
    params.direct_io = false;
    REQUIRE(thrown([&]() {
        llama_dsv41_admit(host_with_used(0), 0, tensors, params);
    }).find("category=direct_io") != std::string::npos);

    params.direct_io = true;
    params.unified_memory = false;
    REQUIRE(thrown([&]() {
        llama_dsv41_admit(host_with_used(0), 0, tensors, params);
    }).find("category=unified_memory") != std::string::npos);

    params.unified_memory = true;
    const auto result = llama_dsv41_admit(host_with_used(0), 0, tensors, params);
    const std::string diagnostic = result.describe();
    for (const char * field : {
                "category=", "current=", "fixed=", "dense=", "state=", "workspace=",
                "host_total=", "host_available=", "batch=", "outputs=", "outputs_per_seq=",
                "expert_slots=", "required_expert_slots=", "expert_ubatch_capacity=",
                "expert_cache=", "expert_staging=",
                "output_bytes=", "soft=", "watchdog=", "hard=" }) {
        REQUIRE(diagnostic.find(field) != std::string::npos);
    }

    REQUIRE(thrown([&]() {
        llama_dsv41_admit(host_with_used(0), UINT64_MAX, tensors, params);
    }).find("overflow") != std::string::npos);

    auto small_host = host_with_used(0);
    small_host.total = 8ULL << 30;
    small_host.available = small_host.total;
    REQUIRE(thrown([&]() {
        llama_dsv41_admit(small_host, 0, tensors, params);
    }).find("physical host memory") != std::string::npos);
}

void test_expert_union_and_outputs() {
    const auto tensors = published_tensors();
    auto params = base_params();
    params.n_ubatch = 2;
    params.configured_cache_slots = 11;
    REQUIRE(thrown([&]() {
        llama_dsv41_admit(host_with_used(0), 0, tensors, params);
    }).find("worst-case routed expert union") != std::string::npos);

    params.configured_cache_slots = 12;
    const auto result = llama_dsv41_admit(host_with_used(0), 0, tensors, params);
    REQUIRE(result.required_expert_slots == 12);
    REQUIRE(result.expert_slots == 12);

    params = base_params();
    params.n_ubatch = 37;
    params.configured_cache_slots = 224;
    const auto bounded = llama_dsv41_admit(host_with_used(0), 0, tensors, params);
    REQUIRE(bounded.required_expert_slots == 222);
    REQUIRE(bounded.expert_slots == 224);
    REQUIRE(bounded.expert_ubatch_capacity == 37);

    params.n_ubatch = 38;
    REQUIRE(thrown([&]() {
        llama_dsv41_admit(host_with_used(0), 0, tensors, params);
    }).find("worst-case routed expert union") != std::string::npos);

    const uint64_t expected =
            3*100ULL*10*sizeof(float) +
            (100ULL + 1)*10*sizeof(int32_t) +
            16*sizeof(int32_t) +
            3*10*sizeof(size_t);
    REQUIRE(llama_dsv41_output_bytes(100, 16, 10) == expected);
}

void test_runtime_memory_validation() {
    auto params = base_params();
    params.n_ubatch = 1;
    params.configured_cache_slots = LLAMA_DSV41_N_EXPERT_USED;
    const auto admitted = llama_dsv41_admit(host_with_used(0), 0, published_tensors(), params);

    const auto measured = llama_dsv41_validate_runtime_memory(
            admitted, admitted.state_bytes, admitted.graph_workspace_bytes - 1);
    REQUIRE(measured.projected_bytes == admitted.projected_bytes - 1);

    const uint64_t over_soft = admitted.graph_workspace_bytes +
        (admitted.soft_bytes - admitted.projected_bytes) + 1;
    REQUIRE(thrown([&]() {
        llama_dsv41_validate_runtime_memory(admitted, admitted.state_bytes, over_soft);
    }).find("category=runtime_workspace") != std::string::npos);

    REQUIRE(thrown([&]() {
        llama_dsv41_validate_runtime_memory(admitted, 0, admitted.graph_workspace_bytes);
    }).find("category=runtime") != std::string::npos);
}

void test_unified_topology() {
    REQUIRE(!llama_dsv41_has_unified_topology({}));
    REQUIRE(!llama_dsv41_has_unified_topology({ GGML_BACKEND_DEVICE_TYPE_CPU }));
    REQUIRE(!llama_dsv41_has_unified_topology({ GGML_BACKEND_DEVICE_TYPE_GPU }));
    REQUIRE(!llama_dsv41_has_unified_topology({ GGML_BACKEND_DEVICE_TYPE_META }));
    REQUIRE(llama_dsv41_has_unified_topology({ GGML_BACKEND_DEVICE_TYPE_IGPU }));
    REQUIRE(llama_dsv41_has_unified_topology({
            GGML_BACKEND_DEVICE_TYPE_IGPU,
            GGML_BACKEND_DEVICE_TYPE_IGPU }));
    REQUIRE(!llama_dsv41_has_unified_topology({
            GGML_BACKEND_DEVICE_TYPE_IGPU,
            GGML_BACKEND_DEVICE_TYPE_GPU }));
}

}

int main() {
    test_procfs();
    test_procfs_fail_closed();
    test_published_slot_fit();
    test_configured_cache();
    test_threshold_boundaries();
    test_context_progression();
    test_diagnostics_and_guards();
    test_expert_union_and_outputs();
    test_runtime_memory_validation();
    test_unified_topology();
    return 0;
}
