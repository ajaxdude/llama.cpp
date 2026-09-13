#include "llama-dsv41-admission.h"

#include "llama-dsv41.h"
#include "llama-impl.h"

#include <algorithm>
#include <charconv>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace {

uint64_t checked_add(uint64_t a, uint64_t b, const char * category) {
    if (b > std::numeric_limits<uint64_t>::max() - a) {
        throw std::runtime_error(std::string("DeepSeek V4.1 memory admission overflow: ") + category);
    }
    return a + b;
}

uint64_t checked_mul(uint64_t a, uint64_t b, const char * category) {
    if (a != 0 && b > std::numeric_limits<uint64_t>::max()/a) {
        throw std::runtime_error(std::string("DeepSeek V4.1 memory admission overflow: ") + category);
    }
    return a*b;
}

uint64_t parse_u64(const std::string & value, const char * field) {
    uint64_t result = 0;
    const char * begin = value.data();
    const char * end = begin + value.size();
    const auto parsed = std::from_chars(begin, end, result);
    if (parsed.ec != std::errc() || parsed.ptr != end) {
        throw std::runtime_error(std::string("DeepSeek V4.1 procfs malformed integer: ") + field);
    }
    return result;
}

uint64_t read_meminfo_value(
        const std::string & line,
        const char * expected_key) {
    std::istringstream stream(line);
    std::string key;
    std::string value;
    std::string unit;
    std::string extra;
    if (!(stream >> key >> value >> unit) || stream >> extra ||
            key != std::string(expected_key) + ":" || unit != "kB") {
        throw std::runtime_error(std::string("DeepSeek V4.1 procfs malformed field: ") + expected_key);
    }
    return checked_mul(parse_u64(value, expected_key), 1024, expected_key);
}

void validate_context(uint32_t n_ctx) {
    switch (n_ctx) {
        case 32768:
        case 65536:
        case 98304:
        case 131072:
            return;
        default:
            throw std::runtime_error(
                    "DeepSeek V4.1 memory admission context must be one of 32768, 65536, 98304, or 131072");
    }
}

[[noreturn]] void reject(
        const char * category,
        const llama_dsv41_admission_result & result,
        const std::string & detail) {
    llama_dsv41_admission_result failure = result;
    failure.category = category;
    throw std::runtime_error(failure.describe() + ", detail=" + detail);
}

}

llama_dsv41_host_memory llama_dsv41_read_host_memory(const std::string & procfs_root) {
#ifndef __linux__
    if (procfs_root == "/proc") {
        throw std::runtime_error("DeepSeek V4.1 memory admission requires Linux procfs");
    }
#endif
    const std::string root = procfs_root.empty() ? "/proc" : procfs_root;
    std::ifstream meminfo(root + "/meminfo");
    if (!meminfo) {
        throw std::runtime_error("DeepSeek V4.1 memory admission cannot read " + root + "/meminfo");
    }

    llama_dsv41_host_memory result;
    bool have_total = false;
    bool have_available = false;
    std::string line;
    while (std::getline(meminfo, line)) {
        if (line.rfind("MemTotal:", 0) == 0) {
            if (have_total) {
                throw std::runtime_error("DeepSeek V4.1 procfs has duplicate MemTotal");
            }
            result.total = read_meminfo_value(line, "MemTotal");
            have_total = true;
        } else if (line.rfind("MemAvailable:", 0) == 0) {
            if (have_available) {
                throw std::runtime_error("DeepSeek V4.1 procfs has duplicate MemAvailable");
            }
            result.available = read_meminfo_value(line, "MemAvailable");
            have_available = true;
        }
    }
    if (!meminfo.eof() || !have_total || !have_available || result.available > result.total) {
        throw std::runtime_error("DeepSeek V4.1 procfs meminfo is missing or invalid");
    }
    result.used = result.total - result.available;

    std::ifstream swaps(root + "/swaps");
    if (!swaps) {
        throw std::runtime_error("DeepSeek V4.1 memory admission cannot read " + root + "/swaps");
    }
    if (!std::getline(swaps, line)) {
        throw std::runtime_error("DeepSeek V4.1 procfs swaps header is missing");
    }
    {
        std::istringstream header(line);
        std::string filename;
        std::string type;
        std::string size;
        std::string used;
        std::string priority;
        std::string extra;
        if (!(header >> filename >> type >> size >> used >> priority) || header >> extra ||
                filename != "Filename" || type != "Type" || size != "Size" ||
                used != "Used" || priority != "Priority") {
            throw std::runtime_error("DeepSeek V4.1 procfs swaps header is malformed");
        }
    }
    while (std::getline(swaps, line)) {
        if (line.empty()) {
            continue;
        }
        std::istringstream entry(line);
        std::string filename;
        std::string type;
        std::string size;
        std::string used;
        std::string priority;
        std::string extra;
        if (!(entry >> filename >> type >> size >> used >> priority) || entry >> extra) {
            throw std::runtime_error("DeepSeek V4.1 procfs swaps entry is malformed");
        }
        const uint64_t size_bytes = checked_mul(parse_u64(size, "swap size"), 1024, "swap size");
        parse_u64(used, "swap used");
        int64_t priority_value = 0;
        const auto parsed_priority = std::from_chars(
                priority.data(), priority.data() + priority.size(), priority_value);
        if (parsed_priority.ec != std::errc() ||
                parsed_priority.ptr != priority.data() + priority.size()) {
            throw std::runtime_error("DeepSeek V4.1 procfs malformed integer: swap priority");
        }
        result.swap_entries = checked_add(result.swap_entries, 1, "swap entries");
        result.swap_bytes = checked_add(result.swap_bytes, size_bytes, "swap bytes");
    }
    if (!swaps.eof()) {
        throw std::runtime_error("DeepSeek V4.1 procfs swaps read failed");
    }
    return result;
}

uint64_t llama_dsv41_estimate_graph_workspace(uint32_t n_ctx, uint32_t n_ubatch) {
    validate_context(n_ctx);
    if (n_ubatch == 0 || n_ubatch > 2048) {
        throw std::runtime_error("DeepSeek V4.1 bounded admission requires n_ubatch in 1..2048");
    }
    // Conservative ds4 graph bound: 7.884 GiB total state at 32K and 8.951 GiB at 131K.
    // Replace this estimate when the full graph can report exact no-alloc reserve bytes before model allocation.
    const uint64_t base = 7688ULL << 20;
    return checked_add(base, checked_mul(n_ctx, 7424, "graph workspace"), "graph workspace");
}

uint64_t llama_dsv41_engram_staging_bytes(uint32_t n_ubatch) {
    const uint64_t ids = checked_mul(
            checked_mul(n_ubatch, LLAMA_ENGRAM_LAYERS*LLAMA_ENGRAM_COLS, "Engram row IDs"),
            sizeof(uint32_t),
            "Engram row IDs");
    const uint64_t decoded = checked_mul(
            checked_mul(n_ubatch, LLAMA_ENGRAM_LAYERS*LLAMA_ENGRAM_COLS*LLAMA_ENGRAM_DIM, "Engram decoded rows"),
            sizeof(float),
            "Engram decoded rows");
    return checked_add(checked_add(ids, decoded, "Engram staging"), n_ubatch, "Engram staging");
}

uint64_t llama_dsv41_output_bytes(uint32_t n_vocab, uint32_t n_ubatch) {
    if (n_vocab == 0 || n_ubatch == 0) {
        throw std::runtime_error("DeepSeek V4.1 output accounting dimensions must be non-zero");
    }
    const uint64_t floats = checked_mul(
            checked_mul(n_vocab, n_ubatch, "output floats"),
            2*sizeof(float),
            "output floats");
    const uint64_t token_rows = checked_add(n_vocab, 1, "output tokens");
    const uint64_t tokens = checked_mul(
            checked_mul(token_rows, n_ubatch, "output tokens"),
            sizeof(int32_t),
            "output tokens");
    return checked_add(floats, tokens, "outputs");
}

llama_dsv41_admission_result llama_dsv41_admit(
        const llama_dsv41_host_memory & host,
        uint64_t dense_tensor_bytes,
        const std::vector<llama_expert_store_tensor> & expert_tensors,
        const llama_dsv41_admission_params & params) {
    llama_dsv41_admission_result result;
    result.host_total = host.total;
    result.host_available = host.available;
    result.host_used = host.used;
    result.dense_tensor_bytes = dense_tensor_bytes;
    result.soft_bytes = params.soft_bytes;
    result.watchdog_bytes = params.watchdog_bytes;
    result.hard_bytes = params.hard_bytes;
    result.safety_margin_bytes = params.safety_margin_bytes;
    result.device_reported_bytes_ignored = params.device_reported_bytes;
    result.n_ctx = params.n_ctx;
    result.n_seq = params.n_seq;
    result.n_ubatch = params.n_ubatch;

    if (host.total == 0 || host.available > host.total || host.used != host.total - host.available) {
        reject("host", result, "host memory snapshot is invalid");
    }
    if (host.swap_entries != 0 || host.swap_bytes != 0) {
        reject("swap", result, format(
                "%llu configured swap entries (%llu bytes)",
                (unsigned long long) host.swap_entries,
                (unsigned long long) host.swap_bytes));
    }
    if (!params.direct_io) {
        reject("direct_io", result, "buffered expert or Engram I/O is not bounded");
    }
    if (!params.unified_memory) {
        reject("unified_memory", result, "Strix admission requires one unified host/GPU memory pool");
    }
    if (params.soft_bytes == 0 || params.soft_bytes > LLAMA_DSV41_ADMISSION_SOFT_BYTES ||
            params.watchdog_bytes > LLAMA_DSV41_WATCHDOG_EMERGENCY_BYTES ||
            params.soft_bytes >= params.watchdog_bytes ||
            params.watchdog_bytes >= params.hard_bytes ||
            params.hard_bytes > LLAMA_DSV41_ADMISSION_HARD_BYTES) {
        reject("thresholds", result, "require soft <= 116 GiB, watchdog <= 118 GiB, and soft < watchdog < hard <= 120 GiB");
    }
    if (params.safety_margin_bytes == 0) {
        reject("thresholds", result, "safety margin must be non-zero");
    }
    validate_context(params.n_ctx);
    if (params.n_seq != 1) {
        reject("context", result, "bounded DeepSeek V4.1 admission currently requires one sequence");
    }
    if (params.n_expert_used == 0 || params.n_expert_used > LLAMA_DSV41_N_EXPERT) {
        reject("cache", result, "expert top-k is invalid");
    }

    std::vector<uint64_t> layer_slot_bytes(LLAMA_DSV41_N_LAYER, 0);
    for (const auto & tensor : expert_tensors) {
        llama_expert_store_validate_tensor(tensor);
        if (tensor.layer < 0 || tensor.layer >= (int32_t) LLAMA_DSV41_N_LAYER ||
                tensor.ne[2] != LLAMA_DSV41_N_EXPERT) {
            reject("cache", result, "expert tensor geometry is invalid");
        }
        layer_slot_bytes[tensor.layer] = checked_add(
                layer_slot_bytes[tensor.layer], tensor.nb[2], "expert slot");
        result.expert_slot_bytes = checked_add(
                result.expert_slot_bytes, tensor.nb[2], "expert slot");
    }
    if (expert_tensors.size() != LLAMA_DSV41_N_LAYER*3) {
        reject("cache", result, "expected 40 gate/up/down expert tensor sets");
    }
    for (uint64_t bytes : layer_slot_bytes) {
        if (bytes == 0) {
            reject("cache", result, "expert layer has no tensor plane bytes");
        }
        result.expert_staging_slot_bytes = std::max(result.expert_staging_slot_bytes, bytes);
    }

    const uint64_t max_cache_bytes = checked_mul(
            result.expert_slot_bytes, LLAMA_DSV41_N_EXPERT, "maximum expert cache");
    if (params.configured_cache_slots > LLAMA_DSV41_N_EXPERT ||
            params.configured_cache_bytes > max_cache_bytes) {
        reject("cache", result, "configured cache exceeds the published expert count");
    }
    const uint64_t bytes_slots = params.configured_cache_bytes == 0 ?
            LLAMA_DSV41_N_EXPERT : params.configured_cache_bytes/result.expert_slot_bytes;
    if (params.configured_cache_slots != 0 && params.configured_cache_bytes != 0 &&
            params.configured_cache_slots != bytes_slots) {
        reject("cache", result, "configured cache slots and bytes disagree");
    }
    uint64_t slot_cap = LLAMA_DSV41_N_EXPERT;
    if (params.configured_cache_slots != 0) {
        slot_cap = std::min<uint64_t>(slot_cap, params.configured_cache_slots);
    }
    if (params.configured_cache_bytes != 0) {
        slot_cap = std::min(slot_cap, bytes_slots);
    }
    if (slot_cap < params.n_expert_used) {
        reject("cache", result, "configured cache is smaller than expert top-k");
    }

    const auto state = llama_dsv41_account_memory(
            params.n_ctx,
            params.n_seq,
            params.n_ubatch,
            params.kv_element_size,
            params.index_element_size,
            0);
    result.state_bytes = state.total();
    result.graph_workspace_bytes = llama_dsv41_estimate_graph_workspace(params.n_ctx, params.n_ubatch);
    result.engram_staging_bytes = llama_dsv41_engram_staging_bytes(params.n_ubatch);
    result.output_bytes = llama_dsv41_output_bytes(params.n_vocab, params.n_ubatch);

    result.fixed_bytes = result.host_used;
    result.fixed_bytes = checked_add(result.fixed_bytes, result.dense_tensor_bytes, "fixed bytes");
    result.fixed_bytes = checked_add(result.fixed_bytes, result.state_bytes, "fixed bytes");
    result.fixed_bytes = checked_add(result.fixed_bytes, result.graph_workspace_bytes, "fixed bytes");
    result.fixed_bytes = checked_add(result.fixed_bytes, result.engram_staging_bytes, "fixed bytes");
    result.fixed_bytes = checked_add(result.fixed_bytes, result.output_bytes, "fixed bytes");
    result.fixed_bytes = checked_add(result.fixed_bytes, result.safety_margin_bytes, "fixed bytes");

    if (result.host_used >= result.hard_bytes) {
        reject("hard", result, "current host use is at or above the strict hard limit");
    }
    if (result.fixed_bytes > result.soft_bytes) {
        result.projected_bytes = result.fixed_bytes;
        reject("fixed", result, "fixed startup categories exceed the soft limit");
    }
    if (result.fixed_bytes > result.host_total) {
        result.projected_bytes = result.fixed_bytes;
        reject("host", result, "fixed startup categories exceed physical host memory");
    }

    const uint64_t bytes_per_slot = checked_add(
            result.expert_slot_bytes, result.expert_staging_slot_bytes, "expert slot and staging");
    const uint64_t fit_slots = (result.soft_bytes - result.fixed_bytes)/bytes_per_slot;
    const uint64_t selected = std::min(slot_cap, fit_slots);
    if (selected < params.n_expert_used) {
        result.projected_bytes = result.fixed_bytes;
        reject("cache", result, "remaining budget cannot hold the minimum expert top-k");
    }

    result.expert_slots = static_cast<uint32_t>(selected);
    result.expert_cache_bytes = checked_mul(result.expert_slot_bytes, selected, "expert cache");
    result.expert_staging_bytes = checked_mul(result.expert_staging_slot_bytes, selected, "expert staging");
    result.projected_bytes = checked_add(
            checked_add(result.fixed_bytes, result.expert_cache_bytes, "projected bytes"),
            result.expert_staging_bytes,
            "projected bytes");
    if (result.projected_bytes > result.soft_bytes) {
        reject("soft", result, "projected startup exceeds the soft limit");
    }
    if (result.projected_bytes > result.host_total) {
        reject("host", result, "projected startup exceeds physical host memory");
    }
    if (result.projected_bytes >= result.hard_bytes) {
        reject("hard", result, "projected startup is not strictly below the hard limit");
    }
    result.category = "accepted";
    return result;
}

std::string llama_dsv41_admission_result::describe() const {
    return format(
            "DeepSeek V4.1 memory admission: category=%s, context=%u, sequences=%u, ubatch=%u, "
            "host_total=%llu, host_available=%llu, current=%llu, fixed=%llu, "
            "dense=%llu, state=%llu, workspace=%llu, engram_staging=%llu, expert_slots=%u, "
            "expert_cache=%llu, expert_staging=%llu, outputs=%llu, safety_margin=%llu, projected=%llu, "
            "soft=%llu, watchdog=%llu, hard=%llu, device_reported_ignored=%llu",
            category.c_str(),
            n_ctx,
            n_seq,
            n_ubatch,
            (unsigned long long) host_total,
            (unsigned long long) host_available,
            (unsigned long long) host_used,
            (unsigned long long) fixed_bytes,
            (unsigned long long) dense_tensor_bytes,
            (unsigned long long) state_bytes,
            (unsigned long long) graph_workspace_bytes,
            (unsigned long long) engram_staging_bytes,
            expert_slots,
            (unsigned long long) expert_cache_bytes,
            (unsigned long long) expert_staging_bytes,
            (unsigned long long) output_bytes,
            (unsigned long long) safety_margin_bytes,
            (unsigned long long) projected_bytes,
            (unsigned long long) soft_bytes,
            (unsigned long long) watchdog_bytes,
            (unsigned long long) hard_bytes,
            (unsigned long long) device_reported_bytes_ignored);
}
