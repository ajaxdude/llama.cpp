#include "arg.h"
#include "build-info.h"
#include "common.h"
#include "ggml-backend.h"
#include "ggml.h"
extern "C" {
#include "hash/sha256/sha256.h"
}
#include "llama.h"
#include "llama-ext.h"
#include "host-attestation.h"
#include "trace-components.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cerrno>
#include <cctype>
#include <clocale>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#if defined(__linux__)
#include <fcntl.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
#endif

#if !defined(_WIN32)
#include <sys/utsname.h>
#endif

namespace fs = std::filesystem;
using json = nlohmann::ordered_json;

static constexpr int TRACE_VERSION = 2;
#if defined(__linux__)
static constexpr const char * WATCHDOG_SCRIPT_SHA256 =
    "d2781a25f978dd2bc14fc113079aa2dbf513aa157b44da9d0d51d750daa6c94f";
#endif

static std::string sha256_hex(const unsigned char digest[SHA256_DIGEST_SIZE]) {
    std::ostringstream stream;
    stream << std::hex << std::setfill('0');
    for (size_t i = 0; i < SHA256_DIGEST_SIZE; ++i) {
        stream << std::setw(2) << static_cast<unsigned>(digest[i]);
    }
    return stream.str();
}

static std::string sha256_data(const void * data, size_t size) {
    unsigned char digest[SHA256_DIGEST_SIZE];
    sha256_hash(digest, static_cast<const unsigned char *>(data), size);
    return sha256_hex(digest);
}

static std::string sha256_file(const fs::path & path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("cannot open for SHA-256: " + path.string());
    }

    sha256_t state;
    sha256_init(&state);
    std::vector<unsigned char> buffer(1024 * 1024);
    while (input) {
        input.read(reinterpret_cast<char *>(buffer.data()), static_cast<std::streamsize>(buffer.size()));
        const std::streamsize count = input.gcount();
        if (count > 0) {
            sha256_update(&state, buffer.data(), static_cast<size_t>(count));
        }
    }
    if (!input.eof()) {
        throw std::runtime_error("failed while hashing: " + path.string());
    }

    unsigned char digest[SHA256_DIGEST_SIZE];
    sha256_final(&state, digest);
    return sha256_hex(digest);
}

static std::vector<uint8_t> read_file(const fs::path & path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) {
        throw std::runtime_error("cannot open: " + path.string());
    }
    const std::streamsize size = input.tellg();
    if (size < 0) {
        throw std::runtime_error("cannot determine file size: " + path.string());
    }
    std::vector<uint8_t> result(static_cast<size_t>(size));
    input.seekg(0);
    if (size != 0 && !input.read(reinterpret_cast<char *>(result.data()), size)) {
        throw std::runtime_error("cannot read: " + path.string());
    }
    return result;
}

static std::string required_environment(const char * name) {
    const char * value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        throw std::runtime_error(std::string("required environment variable is missing: ") + name);
    }
    return value;
}

#if defined(__linux__)
static constexpr uint64_t DSV41_GIB = UINT64_C(1024)*1024*1024;

static std::pair<int64_t, uint64_t> proc_identity(int64_t pid) {
    const std::vector<uint8_t> bytes = read_file("/proc/" + std::to_string(pid) + "/stat");
    const std::string stat(bytes.begin(), bytes.end());
    const size_t command_end = stat.rfind(')');
    if (command_end == std::string::npos) {
        throw std::runtime_error("watchdog process stat is invalid");
    }
    std::istringstream fields(stat.substr(command_end + 2));
    std::string value;
    int64_t parent = 0;
    for (int field = 3; field <= 22; ++field) {
        if (!(fields >> value)) {
            throw std::runtime_error("watchdog process stat is truncated");
        }
        if (field == 4) {
            parent = std::stoll(value);
        }
    }
    return { parent, std::stoull(value) };
}

static bool process_is_descendant(int64_t pid, int64_t ancestor) {
    std::vector<int64_t> seen;
    while (pid > 1 && std::find(seen.begin(), seen.end(), pid) == seen.end()) {
        if (pid == ancestor) {
            return true;
        }
        seen.push_back(pid);
        pid = proc_identity(pid).first;
    }
    return false;
}

static void validate_watchdog(const json & data) {
    const int64_t pid = data.value("watchdog_pid", INT64_C(0));
    const int64_t guardian_pid = data.value("guardian_pid", INT64_C(0));
    const int64_t child_pid = data.value("child_pid", INT64_C(0));
    const int64_t child_pgid = data.value("child_process_group_id", INT64_C(0));
    if (pid <= 1 || !fs::exists("/proc/" + std::to_string(pid))) {
        throw std::runtime_error("watchdog process is not running");
    }
    if (data.value("format", "") != "strix-memory-watchdog-lease" || data.value("version", 0) != 2) {
        throw std::runtime_error("watchdog lease format is invalid");
    }
    if (data.value("soft_bytes", UINT64_C(0)) != 116*DSV41_GIB ||
            data.value("emergency_bytes", UINT64_C(0)) != 118*DSV41_GIB ||
            data.value("strict_ceiling_bytes", UINT64_C(0)) != 120*DSV41_GIB ||
            data.value("grace_seconds", 0.0) != 30.0 ||
            data.value("sample_interval_seconds", 0.0) != 1.0 ||
            data.value("procfs_root", "") != "/proc") {
        throw std::runtime_error("watchdog execution policy is invalid");
    }
    if (guardian_pid <= 1 || child_pid <= 1 || child_pgid <= 1 || getpgrp() != child_pgid ||
            !process_is_descendant(getpid(), child_pid)) {
        throw std::runtime_error("trace exporter is outside the watchdog-monitored process group");
    }
    if (proc_identity(guardian_pid).first != pid ||
            proc_identity(child_pid).first != guardian_pid ||
            getpgid(guardian_pid) != child_pgid ||
            getpgid(child_pid) != child_pgid ||
            child_pgid != guardian_pid) {
        throw std::runtime_error("watchdog guardian or child process identity is invalid");
    }
    if (proc_identity(pid).second != data.value("watchdog_start_time_ticks", UINT64_C(0))) {
        throw std::runtime_error("watchdog process start time changed");
    }
    const std::vector<uint8_t> command = read_file("/proc/" + std::to_string(pid) + "/cmdline");
    if (sha256_data(command.data(), command.size()) != data.value("watchdog_command_sha256", "")) {
        throw std::runtime_error("watchdog process command changed");
    }
    const fs::path executable_path = data.value("watchdog_executable_path", "");
    if (executable_path.empty() ||
            fs::canonical("/proc/" + std::to_string(pid) + "/exe") != fs::canonical(executable_path)) {
        throw std::runtime_error("watchdog executable identity changed");
    }
    const fs::path script_path = data.value("watchdog_script_path", "");
    if (script_path.empty() || data.value("watchdog_revision", "") !=
    "778db6f50eae04e6c232c69b9575bdbd0747962b" ||
            data.value("watchdog_script_sha256", "") != WATCHDOG_SCRIPT_SHA256 ||
            sha256_file(script_path) != WATCHDOG_SCRIPT_SHA256) {
        throw std::runtime_error("watchdog script identity changed");
    }
    std::vector<std::string> watchdog_arguments;
    size_t argument_start = 0;
    while (argument_start < command.size()) {
        const auto * begin = reinterpret_cast<const char *>(command.data() + argument_start);
        const size_t argument_size = std::char_traits<char>::length(begin);
        watchdog_arguments.emplace_back(begin, argument_size);
        argument_start += argument_size + 1;
    }
    fs::path command_script;
    if (watchdog_arguments.size() >= 2) {
        command_script = watchdog_arguments[1];
        if (!command_script.is_absolute()) {
            command_script = fs::canonical("/proc/" + std::to_string(pid) + "/cwd") / command_script;
        }
    }
    if (watchdog_arguments.size() < 2 || fs::canonical(command_script) != fs::canonical(script_path)) {
        throw std::runtime_error("watchdog script is not in executable argv position");
    }
    const fs::path heartbeat_path = data.value("heartbeat_path", "");
    const double max_age = data.value("max_heartbeat_age_seconds", 0.0);
    if (heartbeat_path.empty() || max_age <= 0 || max_age > 30) {
        throw std::runtime_error("watchdog heartbeat configuration is invalid");
    }
    if (fs::canonical(required_environment("STRIX_MEMORY_WATCHDOG_LEASE_PATH")) !=
            fs::canonical(fs::path(data.value("lease_path", ""))) ||
            fs::canonical(required_environment("STRIX_MEMORY_WATCHDOG_HEARTBEAT_PATH")) !=
                fs::canonical(heartbeat_path) ||
            fs::canonical(required_environment("STRIX_MEMORY_WATCHDOG_AUDIT_PATH")) !=
                fs::canonical(fs::path(data.value("audit_live_path", ""))) ||
            std::stod(required_environment("STRIX_MEMORY_WATCHDOG_HEARTBEAT_MAX_AGE_SECONDS")) != max_age) {
        throw std::runtime_error("watchdog lease does not match the inherited environment");
    }
    const json child_command = data.value("command", json::array());
    if (!child_command.is_array() || child_command.empty()) {
        throw std::runtime_error("watchdog child command is invalid");
    }
    for (const json & argument : child_command) {
        if (!argument.is_string()) {
            throw std::runtime_error("watchdog child command is invalid");
        }
    }
    const std::string child_command_json = child_command.dump(-1, ' ', true);
    if (sha256_data(
                reinterpret_cast<const uint8_t *>(child_command_json.data()),
                child_command_json.size()) != data.value("child_command_sha256", "")) {
        throw std::runtime_error("watchdog child command SHA-256 is invalid");
    }
    const std::vector<uint8_t> heartbeat_bytes = read_file(heartbeat_path);
    json heartbeat;
    try {
        heartbeat = json::parse(heartbeat_bytes.begin(), heartbeat_bytes.end());
    } catch (const json::exception & error) {
        throw std::runtime_error(std::string("watchdog heartbeat is invalid: ") + error.what());
    }
    if (heartbeat.value("format", "") != "strix-memory-watchdog-heartbeat" ||
            heartbeat.value("version", 0) != 2 ||
            heartbeat.value("lease_id", "") != data.value("lease_id", "") ||
            heartbeat.value("sequence", INT64_C(-1)) < 0 ||
            heartbeat.value("state", "") != "active" ||
            heartbeat.value("updated_at", "").empty() ||
            heartbeat.value("watchdog_pid", INT64_C(0)) != pid ||
            heartbeat.value("watchdog_start_time_ticks", UINT64_C(0)) !=
                data.value("watchdog_start_time_ticks", UINT64_C(0)) ||
            heartbeat.value("child_pid", INT64_C(0)) != child_pid ||
            heartbeat.value("child_process_group_id", INT64_C(0)) != child_pgid) {
        throw std::runtime_error("watchdog heartbeat identity is invalid");
    }
    const json heartbeat_sample = heartbeat.value("sample", json::object());
    const std::string audit_record_sha256 = heartbeat_sample.value("audit_record_sha256", "");
    if (audit_record_sha256.size() != 64) {
        throw std::runtime_error("watchdog heartbeat audit identity is invalid");
    }
    const uint64_t updated_monotonic_ns = heartbeat.value("updated_monotonic_ns", UINT64_C(0));
    struct timespec now;
    if (updated_monotonic_ns == 0 || clock_gettime(CLOCK_MONOTONIC, &now) != 0) {
        throw std::runtime_error("watchdog heartbeat monotonic timestamp is invalid");
    }
    const uint64_t now_monotonic_ns =
        static_cast<uint64_t>(now.tv_sec)*UINT64_C(1000000000) + static_cast<uint64_t>(now.tv_nsec);
    const uint64_t max_age_ns = static_cast<uint64_t>(max_age*1000000000.0);
    if (updated_monotonic_ns > now_monotonic_ns ||
            now_monotonic_ns - updated_monotonic_ns > max_age_ns) {
        throw std::runtime_error("watchdog heartbeat is stale");
    }
    const fs::path audit_path = data.value("audit_live_path", "");
    if (audit_path.empty()) {
        throw std::runtime_error("watchdog audit path is invalid");
    }
    struct stat audit_stat;
    struct stat descriptor_stat;
    const int audit_fd = data.value("audit_fd", -1);
    const fs::path descriptor_path =
        "/proc/" + std::to_string(pid) + "/fd/" + std::to_string(audit_fd);
    if (audit_fd < 0 || lstat(audit_path.c_str(), &audit_stat) != 0 ||
            stat(descriptor_path.c_str(), &descriptor_stat) != 0 ||
            !S_ISREG(audit_stat.st_mode) ||
            static_cast<uint64_t>(audit_stat.st_dev) != data.value("audit_device", UINT64_C(0)) ||
            static_cast<uint64_t>(audit_stat.st_ino) != data.value("audit_inode", UINT64_C(0)) ||
            static_cast<uint64_t>(descriptor_stat.st_dev) != data.value("audit_device", UINT64_C(0)) ||
            static_cast<uint64_t>(descriptor_stat.st_ino) != data.value("audit_inode", UINT64_C(0)) ||
            audit_stat.st_uid != getuid() ||
            static_cast<uint64_t>(audit_stat.st_uid) != data.value("audit_uid", UINT64_C(0)) ||
            (audit_stat.st_mode & 0777) != 0600 ||
            data.value("audit_mode", UINT64_C(0)) != 0600) {
        throw std::runtime_error("watchdog persistent audit identity changed");
    }
    const int local_audit_fd = open(audit_path.c_str(), O_RDONLY | O_NOFOLLOW);
    if (local_audit_fd < 0) {
        throw std::runtime_error("cannot open watchdog persistent audit");
    }
    const int lock_result = flock(local_audit_fd, LOCK_EX | LOCK_NB);
    if (lock_result == 0) {
        flock(local_audit_fd, LOCK_UN);
        close(local_audit_fd);
        throw std::runtime_error("watchdog does not hold the persistent audit lock");
    }
    if (errno != EWOULDBLOCK && errno != EAGAIN) {
        close(local_audit_fd);
        throw std::runtime_error("cannot inspect watchdog persistent audit lock");
    }
    close(local_audit_fd);
    std::ifstream audit_stream(audit_path);
    if (!audit_stream) {
        throw std::runtime_error("cannot read watchdog persistent audit");
    }
    std::string audit_line;
    bool found_audit_record = false;
    while (std::getline(audit_stream, audit_line)) {
        audit_line.push_back('\n');
        if (sha256_data(audit_line.data(), audit_line.size()) == audit_record_sha256) {
            found_audit_record = true;
            break;
        }
    }
    if (!found_audit_record) {
        throw std::runtime_error("watchdog heartbeat audit record is missing");
    }
}
#endif

static void bind_memory_audit_metadata(json & result, const json & audit) {
    result["accelerator"] = audit.value("accelerator", json::object());
    result["storage"] = audit.value("storage", json::object());
    result["storage_policy"] = audit.value("storage_policy", json::object());
}

static json audit_reference(const char * environment_name, const char * expected_kind) {
    const fs::path path = required_environment(environment_name);
    dsv41::require_nvme_path(path, "audit");
    const std::vector<uint8_t> bytes = read_file(path);
    json audit;
    try {
        audit = json::parse(bytes.begin(), bytes.end());
    } catch (const json::exception & error) {
        throw std::runtime_error(std::string("invalid audit JSON: ") + error.what());
    }
    if (audit.value("kind", "") != expected_kind) {
        throw std::runtime_error(std::string("audit kind mismatch for ") + expected_kind);
    }
    if (audit.value("environment", json::object()).value("HIP_LAUNCH_BLOCKING", "") != "1") {
        throw std::runtime_error(std::string("audit environment mismatch for ") + expected_kind);
    }
    const int64_t created = audit.value("created_unix", INT64_C(0));
    const int64_t now = static_cast<int64_t>(std::time(nullptr));
    if (created <= 0 || now < created || now - created > 300) {
        throw std::runtime_error(std::string("audit is stale: ") + expected_kind);
    }
    if (std::string(expected_kind) == "swap" && audit["data"].value("enabled", true)) {
        throw std::runtime_error("swap audit reports enabled swap");
    }
#if defined(__linux__)
    if (std::string(expected_kind) == "watchdog") {
        validate_watchdog(audit["data"]);
    }
#endif
    json result = {
        {"path", fs::absolute(path).lexically_normal().string()},
        {"sha256", sha256_data(bytes.data(), bytes.size())},
        {"created_unix", created},
    };
    if (std::string(expected_kind) == "watchdog") {
        result["data"] = audit["data"];
    } else if (std::string(expected_kind) == "memory") {
        bind_memory_audit_metadata(result, audit);
    }
    return result;
}

static std::string tensor_dtype(const ggml_tensor * tensor) {
    switch (tensor->type) {
        case GGML_TYPE_F32:  return "f32";
        case GGML_TYPE_BF16: return "bf16";
        case GGML_TYPE_I32:  return "i32";
        case GGML_TYPE_I8:   return "i8";
        default: throw std::runtime_error(
            std::string("unsupported trace tensor type: ") + ggml_type_name(tensor->type));
    }
}

static std::vector<int64_t> tensor_shape(const ggml_tensor * tensor) {
    int rank = GGML_MAX_DIMS;
    while (rank > 2 && tensor->ne[rank - 1] == 1) {
        --rank;
    }
    std::vector<int64_t> result;
    result.reserve(rank);
    for (int i = 0; i < rank; ++i) {
        result.push_back(tensor->ne[i]);
    }
    return result;
}

class trace_writer {
public:
    trace_writer(fs::path root, json manifest) :
        root(std::move(root)),
        blobs(this->root / "blobs"),
        manifest(std::move(manifest)) {
        if (fs::exists(this->root) && !fs::is_empty(this->root)) {
            throw std::runtime_error("trace output directory is not empty: " + this->root.string());
        }
        fs::create_directories(blobs);
        events.open(this->root / "events.jsonl", std::ios::binary | std::ios::trunc);
        if (!events) {
            throw std::runtime_error("cannot create events.jsonl");
        }
        this->manifest["trace_format"] = "dsv41-trace";
        this->manifest["trace_version"] = TRACE_VERSION;
    }

    void set_execution(std::string phase, int step, int64_t token_start, int64_t token_count) {
        this->phase = std::move(phase);
        this->step = step;
        this->token_start = token_start;
        this->token_count = token_count;
    }

    void add(
            const std::string & component,
            int layer,
            const std::string & dtype,
            const std::vector<int64_t> & shape,
            const void * data,
            size_t size,
            const char * semantic_id_space = nullptr) {
        if (error_message.size() != 0) {
            return;
        }
        try {
            const std::string digest = sha256_data(data, size);
            const fs::path blob = blobs / (digest + ".bin");
            if (!fs::exists(blob)) {
                const fs::path temp = blob.string() + ".tmp";
                {
                    std::ofstream output(temp, std::ios::binary | std::ios::trunc);
                    if (!output || (size != 0 && !output.write(static_cast<const char *>(data), size))) {
                        throw std::runtime_error("cannot write trace blob");
                    }
                }
                fs::rename(temp, blob);
            }

            json event = {
                {"trace_version", TRACE_VERSION},
                {"component", component},
                {"phase", phase},
                {"step", step},
                {"token_start", token_start},
                {"token_count", token_count},
                {"layer", layer >= 0 ? json(layer) : json(nullptr)},
                {"dtype", dtype},
                {"shape", shape},
                {"byte_order", "little"},
                {"byte_count", size},
                {"sha256", digest},
                {"blob", "blobs/" + digest + ".bin"},
            };
            if (semantic_id_space != nullptr) {
                event["semantic_id_space"] = semantic_id_space;
            }
            events << event.dump() << '\n';
            events.flush();
            if (!events) {
                throw std::runtime_error("cannot append trace event");
            }
            ++event_count;
        } catch (const std::exception & error) {
            error_message = error.what();
        }
    }

    void add_tensor(const ggml_tensor * tensor) {
        const std::string name = tensor->name;
        const auto descriptor = dsv41_trace_parse_name(name);
        if (!descriptor) {
            return;
        }
        const size_t size = ggml_nbytes(tensor);
        buffer.resize(size);
        ggml_backend_tensor_get(tensor, buffer.data(), 0, size);
        add(descriptor->component, descriptor->layer, tensor_dtype(tensor), tensor_shape(tensor),
                buffer.data(), size, descriptor->semantic_id_space);
    }

    bool has_error() const {
        return !error_message.empty();
    }

    const std::string & error() const {
        return error_message;
    }

    void fail(const std::string & message) {
        if (error_message.empty()) {
            error_message = message;
        }
    }

    void finish() {
        if (has_error()) {
            throw std::runtime_error(error_message);
        }
        events.close();
        manifest["event_count"] = event_count;
        const fs::path output = root / "manifest.json";
        const fs::path temp = output.string() + ".tmp";
        {
            std::ofstream stream(temp, std::ios::binary | std::ios::trunc);
            stream << manifest.dump() << '\n';
            if (!stream) {
                throw std::runtime_error("cannot write trace manifest");
            }
        }
        fs::rename(temp, output);
    }

private:
    fs::path root;
    fs::path blobs;
    std::ofstream events;
    json manifest;
    std::vector<uint8_t> buffer;
    std::string phase = "unknown";
    std::string error_message;
    int step = 0;
    int64_t token_start = 0;
    int64_t token_count = 0;
    uint64_t event_count = 0;
};

static bool trace_callback(ggml_tensor * tensor, bool ask, void * user_data) {
    auto * writer = static_cast<trace_writer *>(user_data);
    try {
        if (ask) {
            return dsv41_trace_select_name(tensor->name);
        }
        writer->add_tensor(tensor);
        return !writer->has_error();
    } catch (const std::exception & error) {
        writer->fail(error.what());
        std::fprintf(stderr, "trace callback failed: %s\n", error.what());
        return false;
    }
}

static std::vector<float> copy_logits(llama_context * ctx, int32_t n_vocab) {
    const float * logits = llama_get_logits_ith(ctx, -1);
    if (logits == nullptr) {
        throw std::runtime_error("runtime did not produce final-token logits");
    }
    return std::vector<float>(logits, logits + n_vocab);
}

static int32_t greedy_token(const std::vector<float> & logits) {
    if (logits.empty()) {
        throw std::runtime_error("cannot select from empty logits");
    }
    return static_cast<int32_t>(std::max_element(logits.begin(), logits.end()) - logits.begin());
}

static void decode_tokens(
        llama_context * ctx,
        trace_writer & writer,
        const std::vector<llama_token> & tokens,
        int32_t n_ubatch) {
    int64_t offset = 0;
    while (offset < static_cast<int64_t>(tokens.size())) {
        const int32_t count = static_cast<int32_t>(std::min<int64_t>(n_ubatch, tokens.size() - offset));
        llama_batch batch = llama_batch_init(count, 0, 1);
        for (int32_t i = 0; i < count; ++i) {
            const bool logits = offset + i + 1 == static_cast<int64_t>(tokens.size());
            common_batch_add(batch, tokens[offset + i], offset + i, {0}, logits);
        }
        writer.set_execution("prefill", 0, offset, count);
        const int result = llama_decode(ctx, batch);
        llama_batch_free(batch);
        if (result != 0) {
            throw std::runtime_error("prefill failed at token " + std::to_string(offset));
        }
        if (writer.has_error()) {
            throw std::runtime_error(writer.error());
        }
        offset += count;
    }
}

static std::string model_architecture(const llama_model * model) {
    std::array<char, 128> buffer = {};
    const int32_t count = llama_model_meta_val_str(
            model, "general.architecture", buffer.data(), buffer.size());
    if (count < 0) {
        throw std::runtime_error("model has no general.architecture metadata");
    }
    return buffer.data();
}

static std::vector<std::string> command_line(int argc, char ** argv) {
    std::vector<std::string> result;
    result.reserve(argc);
    for (int i = 0; i < argc; ++i) {
        result.emplace_back(argv[i]);
    }
    return result;
}

static std::string command_line_json(int argc, char ** argv) {
    return json(command_line(argc, argv)).dump();
}

static bool flash_attention_enabled(enum llama_flash_attn_type value) {
    return value == LLAMA_FLASH_ATTN_TYPE_ENABLED;
}

static std::string runtime_system_info(const common_params & params) {
#if defined(_WIN32)
    const std::string platform = "Windows";
#else
    struct utsname info = {};
    if (uname(&info) != 0) {
        throw std::runtime_error("cannot query operating system identity");
    }
    const std::string platform =
        std::string(info.sysname) + " " + info.release + " " + info.machine;
#endif
    return platform + "; " + common_params_get_system_info(params);
}

static json accelerator_json(const dsv41::accelerator_attestation & accelerator) {
    return {
        {"format", "dsv41-accelerator-attestation"},
        {"version", 2},
        {"runtime_kind", "strix-rocm"},
        {"platform", "linux"},
        {"backend", "ROCm"},
        {"backend_device", accelerator.backend_device},
        {"backend_description", accelerator.backend_description},
        {"pci_device_id", accelerator.pci_device_id},
        {"kfd_node", accelerator.kfd_node},
        {"gpu_id", accelerator.gpu_id},
        {"gfx_target_version", accelerator.gfx_target_version},
        {"architecture", accelerator.architecture},
        {"source", "linux-kfd-sysfs"},
    };
}

static json storage_json(const dsv41::storage_attestation & storage) {
    return {
        {"format", "dsv41-storage-attestation"},
        {"version", 2},
        {"runtime_kind", "strix-rocm"},
        {"platform", "linux"},
        {"storage_kind", "linux-nvme"},
        {"resolved_path", storage.resolved_path.string()},
        {"existing_path", storage.existing_path.string()},
        {"mount_point", storage.mount_point},
        {"filesystem_type", storage.filesystem_type},
        {"mount_source", storage.mount_source},
        {"device_number", storage.device_number},
        {"block_device_path", storage.block_device_path.string()},
        {"nvme_device", storage.nvme_device},
        {"rotational", false},
        {"source", "linux-mountinfo-sysfs"},
    };
}

static json storage_policy_json() {
    return {
        {"format", "dsv41-state-storage-policy"},
        {"version", 1},
        {"expert_cache", "memory-resident"},
        {"kv_cache", "memory-resident"},
        {"external_cache_paths", json::array()},
        {"external_state_paths", json::array()},
    };
}

static void write_manifest_type_probe(const fs::path & path, int argc, char ** argv) {
    const std::vector<uint8_t> bytes = read_file(path);
    json manifest = json::parse(bytes.begin(), bytes.end());
    if (!manifest.is_object()) {
        throw std::runtime_error("manifest type probe input is not a JSON object");
    }
    common_params params;
    params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
#if defined(__linux__)
    manifest["environment"]["system_info"] = runtime_system_info(params);
#endif
    manifest["environment"]["command"] = command_line_json(argc, argv);
    manifest["config"]["flash_attention"] = flash_attention_enabled(params.flash_attn_type);
    manifest["storage_policy"] = storage_policy_json();

    const fs::path temp = path.string() + ".tmp";
    {
        std::ofstream stream(temp, std::ios::binary | std::ios::trunc);
        stream << manifest.dump() << '\n';
        if (!stream) {
            throw std::runtime_error("cannot write manifest type probe");
        }
    }
    fs::rename(temp, path);
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");
    try {
        if (argc >= 2 && std::string(argv[1]) == "--dsv41-manifest-type-probe") {
            if (argc != 3) {
                throw std::runtime_error("--dsv41-manifest-type-probe requires a manifest path");
            }
            write_manifest_type_probe(argv[2], argc, argv);
            return 0;
        }
        if (argc == 3 && std::string(argv[1]) == "--dsv41-attest-device") {
            common_init();
            ggml_backend_load_all();
            const dsv41::accelerator_attestation accelerator =
                dsv41::require_gfx1151_device(ggml_backend_dev_by_name(argv[2]));
            std::cout << accelerator_json(accelerator).dump() << '\n';
            return 0;
        }

        const uint16_t endian = 1;
        if (*reinterpret_cast<const uint8_t *>(&endian) != 1) {
            throw std::runtime_error("trace writer requires a little-endian host");
        }

        common_params params;
        params.escape = false;
        params.warmup = false;
        params.ctx_shift = false;
        common_init();
        if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_RESULTS)) {
            return 1;
        }
        if (params.model.path.empty() || params.prompt_file.empty() || params.out_file.empty()) {
            throw std::runtime_error("-m, -bf, and -o are required");
        }
        if (params.n_predict < 1) {
            throw std::runtime_error("-n must request at least one deterministic decode step");
        }

        const dsv41::storage_attestation model_storage =
            dsv41::require_nvme_path(params.model.path, "model");
        const dsv41::storage_attestation prompt_storage =
            dsv41::require_nvme_path(params.prompt_file, "prompt");
        const dsv41::storage_attestation output_storage =
            dsv41::require_nvme_path(params.out_file, "trace output");
        const fs::path temporary_directory = required_environment("TMPDIR");
        dsv41::require_usable_directory(temporary_directory, "TMPDIR");
        const dsv41::storage_attestation temporary_storage =
            dsv41::require_nvme_path(temporary_directory, "temporary directory");
        if (required_environment("HIP_LAUNCH_BLOCKING") != "1") {
            throw std::runtime_error("HIP_LAUNCH_BLOCKING=1 is required for gfx1151 correctness runs");
        }
        if (params.devices.size() != 1 || params.devices[0] == nullptr) {
            throw std::runtime_error("trace tool requires exactly one selected execution device");
        }
        const dsv41::accelerator_attestation configured_accelerator =
            dsv41::require_gfx1151_device(params.devices[0]);
        const json memory_audit = audit_reference("DSV41_TRACE_MEMORY_AUDIT", "memory");
        if (memory_audit.value("accelerator", json::object()) != accelerator_json(configured_accelerator)) {
            throw std::runtime_error("preflight accelerator audit does not match the selected execution device");
        }
        if (memory_audit.value("storage_policy", json::object()) != storage_policy_json()) {
            throw std::runtime_error("preflight external cache/state storage policy is invalid");
        }
        const json audited_storage = memory_audit.value("storage", json::object());
        if (audited_storage.value("model", json::object()) != storage_json(model_storage) ||
                audited_storage.value("prompt", json::object()) != storage_json(prompt_storage) ||
                audited_storage.value("output", json::object()) != storage_json(output_storage) ||
                audited_storage.value("temporary_directory", json::object()) != storage_json(temporary_storage)) {
            throw std::runtime_error("preflight storage audit does not match the selected execution paths");
        }
        const json swap_audit = audit_reference("DSV41_TRACE_SWAP_AUDIT", "swap");
        const json watchdog_audit = audit_reference("DSV41_TRACE_WATCHDOG_AUDIT", "watchdog");

        const std::vector<uint8_t> prompt_bytes = read_file(params.prompt_file);
        if (params.prompt.size() != prompt_bytes.size() ||
                !std::equal(prompt_bytes.begin(), prompt_bytes.end(), params.prompt.begin())) {
            throw std::runtime_error("parsed prompt differs from exact prompt file bytes");
        }

        const fs::path model_path = model_storage.resolved_path;
        const fs::path prompt_path = prompt_storage.resolved_path;
        const fs::path output_path = output_storage.resolved_path;
        llama_backend_init();
        llama_numa_init(params.numa);
        common_init_result_ptr init = common_init_from_params(params);
        llama_model * model = init->model();
        llama_context * ctx = init->context();
        if (model == nullptr || ctx == nullptr) {
            throw std::runtime_error("failed to initialize llama.cpp");
        }
        if (model_architecture(model) != "deepseek41") {
            throw std::runtime_error("trace tool requires general.architecture=deepseek41");
        }
        std::vector<std::string> model_devices;
        for (int32_t index = 0; index < llama_model_n_devices(model); ++index) {
            model_devices.emplace_back(ggml_backend_dev_name(llama_model_get_device(model, index)));
        }
        if (model_devices != std::vector<std::string>{"ROCm0"}) {
            throw std::runtime_error("trace tool requires the loaded model to use only ROCm0");
        }
        const dsv41::accelerator_attestation accelerator =
            dsv41::require_gfx1151_device(llama_model_get_device(model, 0));
        if (accelerator_json(accelerator) != accelerator_json(configured_accelerator)) {
            throw std::runtime_error("loaded model device differs from the pre-allocation accelerator attestation");
        }
        const llama_vocab * vocab = llama_model_get_vocab(model);
        const bool add_bos = llama_vocab_get_add_bos(vocab);
        const std::vector<llama_token> tokens = common_tokenize(ctx, params.prompt, add_bos, true);
        const int32_t n_vocab = llama_vocab_n_tokens(vocab);
        if (tokens.empty()) {
            throw std::runtime_error("prompt tokenization produced no tokens");
        }
        if (tokens.size() > llama_n_ctx(ctx)) {
            throw std::runtime_error("prompt token count exceeds the configured context");
        }
        if (tokens.size() + static_cast<size_t>(params.n_predict) > llama_n_ctx(ctx)) {
            throw std::runtime_error("prompt plus decode steps exceed the configured context");
        }

        std::vector<int32_t> all_layers(40);
        for (int32_t layer = 0; layer < 40; ++layer) {
            all_layers[layer] = layer;
        }

        json manifest = {
            {"runtime", "llama.cpp"},
            {"revision", llama_commit()},
            {"build", {
                {"number", llama_build_number()},
                {"info", llama_build_info()},
                {"compiler", llama_compiler()},
                {"target", llama_build_target()},
                {"path", fs::absolute(argv[0]).lexically_normal().string()},
                {"sha256", sha256_file(fs::absolute(argv[0]).lexically_normal())},
            }},
            {"model", {
                {"path", model_path.string()},
                {"architecture", "deepseek41"},
                {"byte_count", fs::file_size(model_path)},
                {"sha256", sha256_file(model_path)},
            }},
            {"prompt", {
                {"path", prompt_path.string()},
                {"byte_count", prompt_bytes.size()},
                {"sha256", sha256_data(prompt_bytes.data(), prompt_bytes.size())},
            }},
            {"accelerator", accelerator_json(accelerator)},
            {"paths", {
                {"model", model_path.string()},
                {"prompt", prompt_path.string()},
                {"output", output_path.string()},
                {"repository", audited_storage["repository"].value("resolved_path", "")},
                {"temporary_directory", temporary_storage.resolved_path.string()},
            }},
            {"storage_policy", storage_policy_json()},
            {"config", {
                {"context", llama_n_ctx(ctx)},
                {"batch", params.n_batch},
                {"ubatch", params.n_ubatch},
                {"device", model_devices[0]},
                {"device_architecture", accelerator.architecture},
                {"device_pci_id", accelerator.pci_device_id},
                {"decode_steps", params.n_predict},
                {"kv_type_k", ggml_type_name(params.cache_type_k)},
                {"kv_type_v", ggml_type_name(params.cache_type_v)},
                {"flash_attention", flash_attention_enabled(params.flash_attn_type)},
                {"gpu_layers", params.n_gpu_layers},
                {"load_mode", static_cast<int>(params.load_mode)},
                {"expert_cache_slots", params.expert_cache_slots},
                {"expert_cache_bytes", static_cast<uint64_t>(params.expert_cache_mib) << 20},
                {"tokenizer_add_bos", add_bos},
                {"tokenizer_parse_special", true},
                {"deepseek41", {
                    {"layer_count", 40},
                    {"vocab_size", n_vocab},
                    {"engram_layers", {1, 14}},
                    {"engram_rows_per_token", 24},
                    {"expert_count", 384},
                    {"experts_used", 6},
                    {"candidate_source_layer", 20},
                    {"candidate_topk_blocks", 2048},
                    {"candidate_block_size", 8},
                    {"index_top_k", 512},
                    {"raw_attention_layers", {0, 1}},
                    {"raw_attention_width", 128},
                    {"candidate_propagation_layers", {24, 28, 32, 36}},
                }},
            }},
            {"comparison", {
                {"tokens", "exact"},
                {"engram_rows", "exact"},
                {"expert_ids", "exact-original-id-space"},
                {"expert_weights", "byte-identical-f32"},
                {"attention_candidates", "exact"},
                {"logits", "byte-identical-f32"},
            }},
            {"environment", {
                {"system_info", runtime_system_info(params)},
                {"command", command_line_json(argc, argv)},
            }},
            {"audits", {
                {"memory", memory_audit},
                {"swap", swap_audit},
                {"watchdog", watchdog_audit},
            }},
            {"expected", {
                {"prompt_tokens", tokens.size()},
                {"decode_steps", params.n_predict},
                {"components", {
                    {"prompt.bytes", {{"layers", nullptr}, {"input", "tokens"}}},
                    {"prompt.tokens", {{"layers", nullptr}, {"input", "tokens"}}},
                    {"engram.row_ids", {{"layers", {1, 14}}, {"prefill", "tokens"}, {"decode", "steps"}}},
                    {"expert.ids", {{"layers", all_layers}, {"prefill", "tokens"}, {"decode", "steps"}}},
                    {"expert.weights", {{"layers", all_layers}, {"prefill", "tokens"}, {"decode", "steps"}}},
                    {"attn.source", {{"layers", all_layers}, {"prefill", "tokens"}, {"decode", "steps"}}},
                    {"attn.candidate_blocks", {{"layers", {20}}, {"prefill", "tokens"}, {"decode", "steps"}}},
                    {"attn.candidates", {{"layers", {24, 28, 32, 36}}, {"prefill", "tokens"}, {"decode", "steps"}}},
                    {"logits.prefill", {{"layers", nullptr}, {"prefill", "final"}}},
                    {"logits.decode", {{"layers", nullptr}, {"decode", "steps"}}},
                    {"decode.greedy_token", {{"layers", nullptr}, {"decode", "steps"}}},
                }},
            }},
        };

        trace_writer writer(output_path, std::move(manifest));
        llama_set_eval_callback(ctx, trace_callback, &writer);

        writer.set_execution("input", 0, 0, tokens.size());
        writer.add("prompt.bytes", -1, "bytes", {static_cast<int64_t>(prompt_bytes.size())},
                prompt_bytes.data(), prompt_bytes.size());
        writer.add("prompt.tokens", -1, "i32", {static_cast<int64_t>(tokens.size())},
                tokens.data(), tokens.size()*sizeof(tokens[0]));

        decode_tokens(ctx, writer, tokens, params.n_ubatch);
        std::vector<float> logits = copy_logits(ctx, n_vocab);
        writer.set_execution("prefill", 0, tokens.size() - 1, 1);
        writer.add("logits.prefill", -1, "f32", {n_vocab}, logits.data(), logits.size()*sizeof(float));

        int64_t position = tokens.size();
        for (int32_t step = 0; step < params.n_predict; ++step) {
            const llama_token token = greedy_token(logits);
            writer.set_execution("decode", step, position, 1);
            writer.add("decode.greedy_token", -1, "i32", {1}, &token, sizeof(token));

            llama_batch batch = llama_batch_init(1, 0, 1);
            common_batch_add(batch, token, position, {0}, true);
            const int result = llama_decode(ctx, batch);
            llama_batch_free(batch);
            if (result != 0) {
                throw std::runtime_error("decode failed at step " + std::to_string(step));
            }
            if (writer.has_error()) {
                throw std::runtime_error(writer.error());
            }
            logits = copy_logits(ctx, n_vocab);
            writer.add("logits.decode", -1, "f32", {n_vocab}, logits.data(), logits.size()*sizeof(float));
            ++position;
        }

#if defined(__linux__)
        validate_watchdog(watchdog_audit["data"]);
#endif
        writer.finish();
        llama_backend_free();
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "llama-deepseek-v41-trace: %s\n", error.what());
        return 1;
    }
}
