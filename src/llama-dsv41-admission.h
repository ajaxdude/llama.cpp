#pragma once

#include "llama-expert-store.h"

#include "ggml-backend.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

static constexpr uint64_t LLAMA_DSV41_ADMISSION_SOFT_BYTES     = 116ULL << 30;
static constexpr uint64_t LLAMA_DSV41_WATCHDOG_EMERGENCY_BYTES = 118ULL << 30;
static constexpr uint64_t LLAMA_DSV41_ADMISSION_HARD_BYTES     = 120ULL << 30;
static constexpr uint64_t LLAMA_DSV41_ADMISSION_MARGIN_BYTES   =   2ULL << 30;
static constexpr uint32_t LLAMA_DSV41_ADMISSION_CONTEXT        = 32768;

struct llama_dsv41_host_memory {
    uint64_t total = 0;
    uint64_t available = 0;
    uint64_t used = 0;
    uint64_t swap_entries = 0;
    uint64_t swap_bytes = 0;
};

struct llama_dsv41_admission_params {
    uint64_t soft_bytes = LLAMA_DSV41_ADMISSION_SOFT_BYTES;
    uint64_t watchdog_bytes = LLAMA_DSV41_WATCHDOG_EMERGENCY_BYTES;
    uint64_t hard_bytes = LLAMA_DSV41_ADMISSION_HARD_BYTES;
    uint64_t safety_margin_bytes = LLAMA_DSV41_ADMISSION_MARGIN_BYTES;
    uint64_t configured_cache_bytes = 0;
    uint64_t device_reported_bytes = 0;
    uint64_t state_bytes = 0;
    uint32_t configured_cache_slots = 0;
    uint32_t n_ctx = LLAMA_DSV41_ADMISSION_CONTEXT;
    uint32_t n_batch = 2048;
    uint32_t n_seq = 1;
    uint32_t n_ubatch = 2048;
    uint32_t n_outputs_max = 2048;
    uint32_t n_outputs_max_per_seq = 2048;
    uint32_t n_vocab = 0;
    uint32_t n_expert_used = 0;
    bool direct_io = true;
    bool unified_memory = true;
};

struct llama_dsv41_admission_result {
    uint64_t host_total = 0;
    uint64_t host_available = 0;
    uint64_t host_used = 0;
    uint64_t dense_tensor_bytes = 0;
    uint64_t state_bytes = 0;
    uint64_t graph_workspace_bytes = 0;
    uint64_t engram_staging_bytes = 0;
    uint64_t expert_staging_bytes = 0;
    uint64_t expert_cache_bytes = 0;
    uint64_t output_bytes = 0;
    uint64_t safety_margin_bytes = 0;
    uint64_t device_reported_bytes_ignored = 0;
    uint64_t fixed_bytes = 0;
    uint64_t projected_bytes = 0;
    uint64_t soft_bytes = 0;
    uint64_t watchdog_bytes = 0;
    uint64_t hard_bytes = 0;
    uint64_t expert_slot_bytes = 0;
    uint64_t expert_staging_slot_bytes = 0;
    uint32_t expert_slots = 0;
    uint32_t required_expert_slots = 0;
    uint32_t expert_ubatch_capacity = 0;
    uint32_t n_ctx = 0;
    uint32_t n_batch = 0;
    uint32_t n_seq = 0;
    uint32_t n_ubatch = 0;
    uint32_t n_outputs_max = 0;
    uint32_t n_outputs_max_per_seq = 0;
    std::string category;

    std::string describe() const;
};

llama_dsv41_host_memory llama_dsv41_read_host_memory(const std::string & procfs_root);

uint64_t llama_dsv41_estimate_graph_workspace(uint32_t n_ctx, uint32_t n_ubatch);
uint64_t llama_dsv41_engram_staging_bytes(uint32_t n_ubatch);
uint64_t llama_dsv41_output_bytes(
        uint32_t n_vocab,
        uint32_t n_batch,
        uint32_t n_outputs_max);
bool llama_dsv41_has_unified_topology(const std::vector<enum ggml_backend_dev_type> & device_types);

llama_dsv41_admission_result llama_dsv41_admit(
        const llama_dsv41_host_memory & host,
        uint64_t dense_tensor_bytes,
        const std::vector<llama_expert_store_tensor> & expert_tensors,
        const llama_dsv41_admission_params & params);

llama_dsv41_admission_result llama_dsv41_validate_runtime_memory(
        const llama_dsv41_admission_result & admitted,
        uint64_t state_bytes,
        uint64_t graph_workspace_bytes);
