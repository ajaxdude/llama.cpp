#include "arg.h"
#include "build-info.h"
#include "common.h"
#include "ggml-backend.h"
#include "ggml.h"
extern "C" {
#include "hash/sha256/sha256.h"
}
#include "llama.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <clocale>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using json = nlohmann::ordered_json;

static constexpr int TRACE_VERSION = 1;
static constexpr const char * TRACE_PREFIX = "dsv41.trace.";

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

static void require_nvme_path(const fs::path & path, const char * label) {
    const fs::path absolute = fs::absolute(path).lexically_normal();
    const std::string value = absolute.string();
    if (value == "/mnt/bigspace" || value.rfind("/mnt/bigspace/", 0) == 0) {
        throw std::runtime_error(std::string(label) + " must not use /mnt/bigspace");
    }
}

static std::string required_environment(const char * name) {
    const char * value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        throw std::runtime_error(std::string("required environment variable is missing: ") + name);
    }
    return value;
}

static json audit_reference(const char * environment_name, const char * expected_kind) {
    const fs::path path = required_environment(environment_name);
    require_nvme_path(path, "audit");
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
        const int64_t pid = audit["data"].value("pid", INT64_C(0));
        if (pid <= 1 || !fs::exists("/proc/" + std::to_string(pid))) {
            throw std::runtime_error("watchdog audit process is not running");
        }
    }
#endif
    return {
        {"path", fs::absolute(path).lexically_normal().string()},
        {"sha256", sha256_data(bytes.data(), bytes.size())},
        {"created_unix", created},
    };
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
    while (rank > 1 && tensor->ne[rank - 1] == 1) {
        --rank;
    }
    std::vector<int64_t> result;
    result.reserve(rank);
    for (int i = rank - 1; i >= 0; --i) {
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
        std::string component;
        const char * semantic_id_space = nullptr;
        if (name.rfind("dsv41.trace.engram.row_ids.l", 0) == 0) {
            component = "engram.row_ids";
        } else if (name.rfind("dsv41.trace.expert.ids.l", 0) == 0) {
            component = "expert.ids";
            semantic_id_space = "original";
        } else if (name.rfind("dsv41.trace.expert.weights.l", 0) == 0) {
            component = "expert.weights";
        } else if (name.rfind("dsv41.trace.attn.source.l", 0) == 0) {
            component = "attn.source";
        } else if (name.rfind("dsv41.trace.attn.candidate_blocks.l", 0) == 0) {
            component = "attn.candidate_blocks";
        } else if (name.rfind("dsv41.trace.attn.candidates.l", 0) == 0) {
            component = "attn.candidates";
        } else {
            return;
        }

        static const std::regex layer_pattern(R"(\.l([0-9]+)$)");
        std::smatch match;
        if (!std::regex_search(name, match, layer_pattern)) {
            throw std::runtime_error("trace tensor name has no layer suffix: " + name);
        }
        const int layer = std::stoi(match[1].str());
        const size_t size = ggml_nbytes(tensor);
        buffer.resize(size);
        ggml_backend_tensor_get(tensor, buffer.data(), 0, size);
        add(component, layer, tensor_dtype(tensor), tensor_shape(tensor), buffer.data(), size, semantic_id_space);
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
    if (ask) {
        return std::string(tensor->name).rfind(TRACE_PREFIX, 0) == 0;
    }
    try {
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
        int32_t n_batch) {
    int64_t offset = 0;
    while (offset < static_cast<int64_t>(tokens.size())) {
        const int32_t count = static_cast<int32_t>(std::min<int64_t>(n_batch, tokens.size() - offset));
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

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");
    try {
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
            throw std::runtime_error("-m, -f, and -o are required");
        }
        if (params.n_predict < 1) {
            throw std::runtime_error("-n must request at least one deterministic decode step");
        }

        require_nvme_path(params.model.path, "model");
        require_nvme_path(params.prompt_file, "prompt");
        require_nvme_path(params.out_file, "trace output");
        const json memory_audit = audit_reference("DSV41_TRACE_MEMORY_AUDIT", "memory");
        const json swap_audit = audit_reference("DSV41_TRACE_SWAP_AUDIT", "swap");
        const json watchdog_audit = audit_reference("DSV41_TRACE_WATCHDOG_AUDIT", "watchdog");

        const std::vector<uint8_t> prompt_bytes = read_file(params.prompt_file);
        if (params.prompt.size() != prompt_bytes.size() ||
                !std::equal(prompt_bytes.begin(), prompt_bytes.end(), params.prompt.begin())) {
            throw std::runtime_error("parsed prompt differs from exact prompt file bytes");
        }

        const fs::path model_path = fs::absolute(params.model.path).lexically_normal();
        const fs::path prompt_path = fs::absolute(params.prompt_file).lexically_normal();
        const fs::path output_path = fs::absolute(params.out_file).lexically_normal();
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
        const llama_vocab * vocab = llama_model_get_vocab(model);
        const bool add_bos = llama_vocab_get_add_bos(vocab);
        const std::vector<llama_token> tokens = common_tokenize(ctx, params.prompt, add_bos, true);
        if (tokens.empty()) {
            throw std::runtime_error("prompt tokenization produced no tokens");
        }
        if (tokens.size() > llama_n_ctx(ctx)) {
            throw std::runtime_error("prompt token count exceeds the configured context");
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
            }},
            {"model", {
                {"path", model_path.string()},
                {"byte_count", fs::file_size(model_path)},
                {"sha256", sha256_file(model_path)},
            }},
            {"prompt", {
                {"path", prompt_path.string()},
                {"byte_count", prompt_bytes.size()},
                {"sha256", sha256_data(prompt_bytes.data(), prompt_bytes.size())},
            }},
            {"config", {
                {"context", llama_n_ctx(ctx)},
                {"batch", params.n_batch},
                {"ubatch", params.n_ubatch},
                {"decode_steps", params.n_predict},
                {"kv_type_k", ggml_type_name(params.cache_type_k)},
                {"kv_type_v", ggml_type_name(params.cache_type_v)},
                {"flash_attention", static_cast<int>(params.flash_attn_type)},
                {"gpu_layers", params.n_gpu_layers},
                {"load_mode", static_cast<int>(params.load_mode)},
                {"expert_cache_slots", params.expert_cache_slots},
                {"expert_cache_bytes", static_cast<uint64_t>(params.expert_cache_mib) << 20},
                {"tokenizer_add_bos", add_bos},
                {"tokenizer_parse_special", true},
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
                {"system_info", common_params_get_system_info(params)},
                {"command", command_line(argc, argv)},
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

        decode_tokens(ctx, writer, tokens, params.n_batch);
        const int32_t n_vocab = llama_vocab_n_tokens(vocab);
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

        writer.finish();
        llama_backend_free();
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "llama-deepseek-v41-trace: %s\n", error.what());
        return 1;
    }
}
