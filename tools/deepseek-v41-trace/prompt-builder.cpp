#include "common.h"
#include "host-attestation.h"
#include "llama.h"

#include <nlohmann/json.hpp>

#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using json = nlohmann::ordered_json;

static std::string read_file(const fs::path & path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("cannot open: " + path.string());
    }
    return std::string(std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
}

static std::string argument(int argc, char ** argv, const std::string & name) {
    for (int index = 1; index + 1 < argc; ++index) {
        if (argv[index] == name) {
            return argv[index + 1];
        }
    }
    throw std::runtime_error("missing argument: " + name);
}

static std::string model_architecture(const llama_model * model) {
    char buffer[128] = {};
    if (llama_model_meta_val_str(model, "general.architecture", buffer, sizeof(buffer)) < 0) {
        throw std::runtime_error("model has no general.architecture metadata");
    }
    return buffer;
}

int main(int argc, char ** argv) {
    try {
        fs::path model_path = argument(argc, argv, "--model");
        fs::path corpus_path = argument(argc, argv, "--corpus");
        fs::path output_path = argument(argc, argv, "--output");
        const int64_t target_tokens = std::stoll(argument(argc, argv, "--tokens"));
        if (target_tokens < 2) {
            throw std::runtime_error("--tokens must be at least 2");
        }
        model_path = dsv41::require_nvme_path(model_path, "model").resolved_path;
        corpus_path = dsv41::require_nvme_path(corpus_path, "corpus").resolved_path;
        output_path = dsv41::require_nvme_path(output_path, "prompt output").resolved_path;
        const char * tmpdir_value = std::getenv("TMPDIR");
        if (tmpdir_value == nullptr || *tmpdir_value == '\0') {
            throw std::runtime_error("TMPDIR is required");
        }
        dsv41::require_usable_directory(tmpdir_value, "TMPDIR");
        const dsv41::storage_attestation temporary_storage =
            dsv41::require_nvme_path(tmpdir_value, "temporary directory");
        if (fs::exists(output_path)) {
            throw std::runtime_error("prompt output already exists: " + output_path.string());
        }

        const std::string corpus = read_file(corpus_path);
        if (corpus.empty()) {
            throw std::runtime_error("corpus is empty");
        }

        llama_backend_init();
        llama_model_params model_params = llama_model_default_params();
        model_params.vocab_only = true;
        llama_model * model = llama_model_load_from_file(model_path.string().c_str(), model_params);
        if (model == nullptr) {
            throw std::runtime_error("cannot load model vocabulary");
        }
        if (model_architecture(model) != "deepseek41") {
            llama_model_free(model);
            throw std::runtime_error("prompt builder requires general.architecture=deepseek41");
        }
        const llama_vocab * vocab = llama_model_get_vocab(model);
        const bool add_bos = llama_vocab_get_add_bos(vocab);

        std::string repeated = corpus;
        std::vector<llama_token> tokens = common_tokenize(vocab, repeated, add_bos, true);
        while (tokens.size() < static_cast<size_t>(target_tokens)) {
            if (repeated.size() > (size_t(1) << 31)) {
                llama_model_free(model);
                throw std::runtime_error("repeated prompt exceeds 2 GiB");
            }
            repeated += repeated;
            tokens = common_tokenize(vocab, repeated, add_bos, true);
        }
        tokens.resize(static_cast<size_t>(target_tokens));
        std::vector<llama_token> content_tokens = tokens;
        if (add_bos) {
            if (content_tokens.front() != llama_vocab_bos(vocab)) {
                llama_model_free(model);
                throw std::runtime_error("tokenized prompt does not begin with the configured BOS token");
            }
            content_tokens.erase(content_tokens.begin());
        }
        const std::string prompt = common_detokenize(vocab, content_tokens, true);
        const std::vector<llama_token> verified = common_tokenize(vocab, prompt, add_bos, true);
        if (verified != tokens) {
            llama_model_free(model);
            throw std::runtime_error("constructed prompt does not round-trip to the target token IDs");
        }

        if (!output_path.parent_path().empty()) {
            fs::create_directories(output_path.parent_path());
        }
        std::ofstream output(output_path, std::ios::binary | std::ios::trunc);
        if (!output || !output.write(prompt.data(), static_cast<std::streamsize>(prompt.size()))) {
            llama_model_free(model);
            throw std::runtime_error("cannot write prompt output");
        }
        output.close();
        std::printf("%s\n", json({
            {"target_tokens", target_tokens},
            {"actual_tokens", verified.size()},
            {"byte_count", prompt.size()},
            {"add_bos", add_bos},
            {"temporary_directory", temporary_storage.resolved_path.string()},
        }).dump().c_str());
        llama_model_free(model);
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "error: %s\n", error.what());
        return 1;
    }
}
