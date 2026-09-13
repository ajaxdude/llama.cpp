#include "../src/llama-dsv41.h"
#include "../src/llama-arch.h"

#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

std::string llama_dsv41_graph_trace_name(const char * trace, uint32_t layer);
ggml_tensor * llama_dsv41_graph_append_zero_row(ggml_context * ctx, ggml_tensor * tensor);
ggml_tensor * llama_dsv41_graph_completion_zero(
        ggml_context * ctx,
        ggml_tensor * dependency,
        ggml_type type);

static void check(bool condition, const std::string & message) {
    if (!condition) {
        std::fprintf(stderr, "%s\n", message.c_str());
        std::exit(1);
    }
}

static void expect_throw(const std::function<void()> & fn, const std::string & message) {
    try {
        fn();
    } catch (const std::runtime_error &) {
        return;
    }
    check(false, message);
}

static llama_dsv41_config valid_config() {
    llama_dsv41_config config = {};
    config.n_ctx_train = LLAMA_DSV41_N_CTX;
    config.n_embd = LLAMA_DSV41_N_EMBD;
    config.n_layer = LLAMA_DSV41_N_LAYER;
    config.n_vocab = LLAMA_DSV41_N_VOCAB;
    config.n_head = LLAMA_DSV41_N_HEAD;
    config.n_head_kv = LLAMA_DSV41_N_HEAD_KV;
    config.n_head_dim = LLAMA_DSV41_N_HEAD_DIM;
    config.n_rot = LLAMA_DSV41_N_ROT;
    config.n_lora_q = LLAMA_DSV41_N_LORA_Q;
    config.n_lora_o = LLAMA_DSV41_N_LORA_O;
    config.n_o_group = LLAMA_DSV41_N_O_GROUP;
    config.n_ff_dense = LLAMA_DSV41_N_FF_DENSE;
    config.n_ff_expert = LLAMA_DSV41_N_FF_EXP;
    config.n_expert = LLAMA_DSV41_N_EXPERT;
    config.n_expert_used = LLAMA_DSV41_N_EXPERT_USED;
    config.n_expert_shared = LLAMA_DSV41_N_EXPERT_SHARED;
    config.indexer_n_head = LLAMA_DSV41_N_INDEX_HEAD;
    config.indexer_head_size = LLAMA_DSV41_N_INDEX_HEAD_DIM;
    config.indexer_top_k = LLAMA_DSV41_N_INDEX_TOP_K;
    config.hc_count = LLAMA_DSV41_HC_MULT;
    config.hc_sinkhorn_iters = LLAMA_DSV41_HC_SINKHORN_ITERS;
    config.raw_window = LLAMA_DSV41_N_SWA;
    config.candidate_source_layer = LLAMA_DSV41_CANDIDATE_SOURCE_LAYER;
    config.candidate_topk_blocks = LLAMA_DSV41_CANDIDATE_TOPK_BLOCKS;
    config.candidate_block_size = LLAMA_DSV41_CANDIDATE_BLOCK_SIZE;
    config.f_norm_rms_eps = 1.0e-20f;
    config.hc_eps = 1.0e-6f;
    config.swiglu_clamp = 10.0f;
    config.routed_scale = 1.5f;
    config.rope_theta = 10000.0f;
    config.compress_rope_theta = 160000.0f;
    config.yarn_factor = 16.0f;
    config.yarn_beta_fast = 32.0f;
    config.yarn_beta_slow = 1.0f;
    config.yarn_original_context = 65536;
    config.expert_weights_norm = true;
    config.hidden_act = "silu";
    config.scoring_func = "sqrtsoftplus";
    config.topk_method = "noaux_tc";
    for (uint32_t il = 0; il < LLAMA_DSV41_N_LAYER; ++il) {
        config.compress_ratios.push_back(llama_dsv41_compress_ratio(il));
    }
    config.kv_sources = { 2, 8, 14, 20 };
    config.index_sources = { 2, 8, 14, 20, 24, 28, 32, 36 };
    config.engram_layers = { 1, 14 };
    config.engram_rows = { 384006168, 384016682 };
    config.engram_encoding = LLAMA_DSV41_ENGRAM_ENCODING;
    config.engram_compressed_vocab_size = LLAMA_DSV41_ENGRAM_COMPRESSED_VOCAB;
    config.engram_pad_id = LLAMA_DSV41_ENGRAM_PAD_ID;
    config.engram_token_map_size = LLAMA_DSV41_N_VOCAB;
    config.engram_primes_size = LLAMA_DSV41_ENGRAM_PRIMES_COUNT;
    config.engram_multipliers_size = LLAMA_DSV41_ENGRAM_MULTIPLIERS_COUNT;
    return config;
}

static void test_hparams() {
    static_assert(std::is_same_v<decltype(llama_dsv41_config::rope_theta), uint32_t>);
    static_assert(std::is_same_v<decltype(llama_dsv41_config::compress_rope_theta), uint32_t>);
    static_assert(std::is_same_v<decltype(llama_dsv41_config::yarn_original_context), float>);

    llama_dsv41_validate_config(valid_config());
    check(valid_config().engram_rows == std::vector<uint32_t>({ 384006168, 384016682 }), "published Engram rows mismatch");

    llama_dsv41_config config = valid_config();
    config.compress_ratios.insert(config.compress_ratios.end(), 3, 0);
    expect_throw([&]() { llama_dsv41_validate_config(config); }, "43-entry source-config compression layout was accepted");

    config = valid_config();
    config.compress_ratios[20] = 2;
    expect_throw([&]() { llama_dsv41_validate_config(config); }, "invalid ratio-1 boundary was accepted");

    config = valid_config();
    config.kv_sources = { 2, 8, 20 };
    expect_throw([&]() { llama_dsv41_validate_config(config); }, "invalid KV source map was accepted");

    config = valid_config();
    config.engram_primes_size = 24;
    expect_throw([&]() { llama_dsv41_validate_config(config); }, "truncated Engram prime table was accepted");

    check(llm_arch_is_hybrid(LLM_ARCH_DEEPSEEK41), "DeepSeek V4.1 must use hybrid context handling");
    check(!llm_arch_supports_rs_rollback(LLM_ARCH_DEEPSEEK41),
          "DeepSeek V4.1 must not advertise partial rollback support");
}

static void test_source_maps() {
    check(llama_dsv41_compress_ratio(0) == 0, "layer 0 ratio mismatch");
    check(llama_dsv41_compress_ratio(1) == 0, "layer 1 ratio mismatch");
    check(llama_dsv41_compress_ratio(2) == 2, "layer 2 ratio mismatch");
    check(llama_dsv41_compress_ratio(19) == 2, "layer 19 ratio mismatch");
    check(llama_dsv41_compress_ratio(20) == 1, "layer 20 ratio mismatch");

    check(llama_dsv41_kv_source_layer(0) == -1, "layer 0 unexpectedly has a KV source");
    check(llama_dsv41_kv_source_layer(2) == 2, "layer 2 KV source mismatch");
    check(llama_dsv41_kv_source_layer(7) == 2, "layer 7 KV source mismatch");
    check(llama_dsv41_kv_source_layer(8) == 8, "layer 8 KV source mismatch");
    check(llama_dsv41_kv_source_layer(19) == 14, "layer 19 KV source mismatch");
    check(llama_dsv41_kv_source_layer(39) == 20, "layer 39 KV source mismatch");

    check(llama_dsv41_index_source_layer(19) == 14, "layer 19 index source mismatch");
    check(llama_dsv41_index_source_layer(20) == 20, "layer 20 index source mismatch");
    check(llama_dsv41_index_source_layer(23) == 20, "layer 23 index source mismatch");
    check(llama_dsv41_index_source_layer(24) == 24, "layer 24 index source mismatch");
    check(llama_dsv41_index_source_layer(39) == 36, "layer 39 index source mismatch");
}

static void test_compression() {
    const auto ratio_2 = llama_dsv41_build_compression_plan({ 0, 1, 2 }, 2, 1024);
    check(ratio_2.n_visible == std::vector<int32_t>({ 0, 1, 1 }), "ratio-2 visible counts mismatch");
    check(ratio_2.write_idxs == std::vector<int64_t>({ 0 }), "ratio-2 write index mismatch");
    check(ratio_2.write_pos == std::vector<int32_t>({ 0 }), "ratio-2 compressed position mismatch");
    check(ratio_2.state_persist_dst_idxs == std::vector<int32_t>({ 0, 1 }), "ratio-2 state rows mismatch");

    const auto ratio_1 = llama_dsv41_build_compression_plan({ 19, 20 }, 1, 1024);
    check(ratio_1.n_visible == std::vector<int32_t>({ 20, 21 }), "ratio-1 visible counts mismatch");
    check(ratio_1.write_idxs == std::vector<int64_t>({ 19, 20 }), "ratio-1 write indexes mismatch");
    check(ratio_1.write_pos == std::vector<int32_t>({ 19, 20 }), "ratio-1 compressed positions mismatch");

    const auto layer_0 = llama_dsv41_build_layer_plan(0, { 0 }, 1024);
    check(layer_0.ratio == 0 && layer_0.compression.write_idxs.empty(), "layer 0 must use raw attention only");
    const auto layer_2 = llama_dsv41_build_layer_plan(2, { 0, 1 }, 1024);
    check(layer_2.ratio == 2 && layer_2.owns_kv_source, "layer 2 compression ownership mismatch");
    check(layer_2.compression.write_idxs == std::vector<int64_t>({ 0 }), "layer 2 graph compression mismatch");
    const auto layer_20 = llama_dsv41_build_layer_plan(20, { 20 }, 1024);
    check(layer_20.ratio == 1 && layer_20.owns_kv_source, "layer 20 compression ownership mismatch");
    check(layer_20.builds_candidates && !layer_20.uses_candidates, "layer 20 candidate propagation mismatch");
    const auto layer_21 = llama_dsv41_build_layer_plan(21, { 21 }, 1024);
    check(!layer_21.uses_candidates && layer_21.reuses_index_selection, "layer 21 index reuse mismatch");
    const auto layer_24 = llama_dsv41_build_layer_plan(24, { 24 }, 1024);
    check(!layer_24.owns_kv_source && layer_24.owns_index_source, "layer 24 source ownership mismatch");
    check(layer_24.uses_candidates, "layer 24 must consume layer-20 candidates");
    check(llama_dsv41_build_layer_plan(39, { 39 }, 1024).collapses_output, "final layer output collapse missing");
    expect_throw([&]() { llama_dsv41_build_layer_plan(20, { 20, 22 }, 1024); }, "non-contiguous graph plan was accepted");
}

static void test_state() {
    llama_dsv41_cache_state state(1024);
    for (llama_pos pos = 0; pos <= 129; ++pos) {
        state.append(pos);
    }
    check(state.position() == 129, "cache position mismatch");
    check(state.raw_slots()[0] == 128 && state.raw_slots()[1] == 129, "raw ring state mismatch");
    check(state.compressed_slots(2)[0] == 0, "ratio-2 first compressed row mismatch");
    check(state.compressed_slots(2)[64] == 128, "ratio-2 boundary row mismatch");
    check(state.pending_slots(2) == std::vector<llama_pos>({ 128, 129 }), "ratio-2 pending rows mismatch");
    check(state.compressed_slots(20)[129] == 129, "ratio-1 direct row mismatch");
    state.set_candidate_blocks({ 4, 1 });
    check(state.candidate_blocks() == std::vector<int32_t>({ 4, 1 }), "candidate state mismatch");
    expect_throw([&]() { state.append(131); }, "non-contiguous cache append was accepted");
    state.clear();
    check(state.position() == -1 && state.raw_slots()[0] == -1, "cache clear mismatch");

    llama_dsv41_cache_state small(1);
    small.append(0);
    expect_throw([&]() { small.append(1); }, "compressed cache overflow was accepted");
    check(small.position() == 0 && small.raw_slots()[1] == -1, "failed cache append mutated state");

    const auto bytes = llama_dsv41_account_memory(32768, 1, 8192, 2, 2, 1234);
    check(bytes.raw_kv > 0 && bytes.compressed_kv > 0 && bytes.index_keys > 0, "cache memory accounting is incomplete");
    check(bytes.compressor_carry > 0 && bytes.candidate_scores > 0 && bytes.candidate_ids > 0, "state memory accounting is incomplete");
    check(bytes.total() == bytes.raw_kv + bytes.compressed_kv + bytes.index_keys + bytes.compressor_carry +
            bytes.candidate_scores + bytes.candidate_ids + bytes.position_state + bytes.graph_workspace,
            "memory accounting total mismatch");
}

static void test_raw_ring() {
    const auto at_127 = llama_dsv41_raw_ring_order(127, 128);
    check(at_127.size() == 128 && at_127.front() == 0 && at_127.back() == 127, "raw ring at 127 mismatch");
    const auto at_128 = llama_dsv41_raw_ring_order(128, 128);
    check(at_128.front() == 1 && at_128.back() == 0, "raw ring at 128 mismatch");
    const auto at_129 = llama_dsv41_raw_ring_order(129, 128);
    check(at_129.front() == 2 && at_129.back() == 1, "raw ring at 129 mismatch");
}

static void test_candidates() {
    for (uint32_t n_visible : { 1u, 7u, 8u, 9u, 127u, 16385u, 16392u, 17017u }) {
        std::vector<float> scores(n_visible);
        for (uint32_t i = 0; i < n_visible; ++i) {
            scores[i] = -(float) i;
        }
        const auto blocks = llama_dsv41_select_candidate_blocks(scores, n_visible, 8, 2048);
        const int32_t final_block = (int32_t) ((n_visible - 1)/8);
        check(std::find(blocks.begin(), blocks.end(), final_block) != blocks.end(), "final visible candidate block was dropped");
        check(blocks.size() == std::min<uint32_t>(2048, (n_visible + 7)/8), "candidate block count mismatch");
        const auto rows = llama_dsv41_candidate_rows(blocks, n_visible, 8);
        check(std::find(rows.begin(), rows.end(), (int32_t) n_visible - 1) != rows.end(), "final visible row was filtered");
        check(std::all_of(rows.begin(), rows.end(), [&](int32_t row) { return row >= 0 && (uint32_t) row < n_visible; }), "candidate rows crossed causal visibility");
    }

    const auto tie = llama_dsv41_select_candidate_blocks(std::vector<float>(24, 1.0f), 24, 8, 2);
    check(tie == std::vector<int32_t>({ 2, 0 }), "candidate tie-break or final block retention mismatch");

    std::vector<float> partial_scores(9, -100.0f);
    partial_scores[0] = 100.0f;
    const auto partial = llama_dsv41_select_candidate_blocks(partial_scores, 9, 8, 1);
    check(partial == std::vector<int32_t>({ 1 }), "final partial candidate block was not forced");

    std::vector<float> full_scores(16392, -100.0f);
    full_scores[0] = 100.0f;
    const auto full = llama_dsv41_select_candidate_blocks(full_scores, 16392, 8, 2048);
    check(std::find(full.begin(), full.end(), 2048) != full.end(), "final full candidate block was not forced");
}

static void test_output_collapse() {
    const std::vector<float> residual = {
        1.0f, 2.0f,
        3.0f, 4.0f,
        5.0f, 6.0f,
        7.0f, 8.0f,
    };
    const auto result = llama_dsv41_output_collapse(residual, { 0.1f, 0.2f, 0.3f, 0.4f }, 2, 4);
    check(result.size() == 2, "output collapse width mismatch");
    check(std::abs(result[0] - 5.0f) < 1.0e-6f, "output collapse first value mismatch");
    check(std::abs(result[1] - 6.0f) < 1.0e-6f, "output collapse second value mismatch");
}

static void test_graph_contract() {
    enum stage {
        STAGE_ENGRAM,
        STAGE_ATTN_HC,
        STAGE_CARRIED_PRE,
        STAGE_ATTN,
        STAGE_ATTN_POST,
        STAGE_FFN_HC,
        STAGE_ATTN_PRE,
        STAGE_ROUTED_EXPERTS,
        STAGE_SHARED_EXPERT,
        STAGE_FFN_POST,
        STAGE_CARRY_FFN_PRE,
    };

    const std::vector<stage> common = {
        STAGE_ATTN_HC,
        STAGE_CARRIED_PRE,
        STAGE_ATTN,
        STAGE_ATTN_POST,
        STAGE_FFN_HC,
        STAGE_ATTN_PRE,
        STAGE_ROUTED_EXPERTS,
        STAGE_SHARED_EXPERT,
        STAGE_FFN_POST,
        STAGE_CARRY_FFN_PRE,
    };
    uint32_t ratio_count[3] = {};
    uint32_t kv_sources = 0;
    uint32_t index_sources = 0;
    uint32_t candidate_sources = 0;
    std::vector<uint32_t> candidate_trace_layers;
    for (uint32_t il = 0; il < LLAMA_DSV41_N_LAYER; ++il) {
        std::vector<stage> stages = common;
        if (il == 1 || il == 14) {
            stages.insert(stages.begin(), STAGE_ENGRAM);
            check(stages[0] == STAGE_ENGRAM && stages[1] == STAGE_ATTN_HC,
                    "Engram must precede attention HC");
        }
        check(stages[stages.size() - 1] == STAGE_CARRY_FFN_PRE,
                "FFN pre must be carried to the next layer");
        check(stages[1 + (il == 1 || il == 14)] == STAGE_CARRIED_PRE,
                "attention must collapse with carried pre");
        check(stages[5 + (il == 1 || il == 14)] == STAGE_ATTN_PRE,
                "FFN must collapse with current attention pre");
        check(stages[7 + (il == 1 || il == 14)] == STAGE_SHARED_EXPERT,
                "shared expert must be added after routed experts");

        const uint32_t ratio = llama_dsv41_compress_ratio(il);
        ratio_count[ratio]++;
        kv_sources += llama_dsv41_kv_source_layer(il) == (int32_t) il;
        index_sources += llama_dsv41_index_source_layer(il) == (int32_t) il;
        candidate_sources += il == LLAMA_DSV41_CANDIDATE_SOURCE_LAYER;

        check(
                llama_dsv41_graph_trace_name("expert.ids", il) ==
                    "dsv41.trace.expert.ids.l" + std::to_string(il),
                "expert ID trace name is unstable");
        check(
                llama_dsv41_graph_trace_name("expert.weights", il) ==
                    "dsv41.trace.expert.weights.l" + std::to_string(il),
                "expert weight trace name is unstable");
        check(
                llama_dsv41_graph_trace_name("attn.source", il) ==
                    "dsv41.trace.attn.source.l" + std::to_string(il),
                "attention source trace name is unstable");
        if (il > LLAMA_DSV41_CANDIDATE_SOURCE_LAYER &&
                llama_dsv41_index_source_layer(il) == (int32_t) il) {
            candidate_trace_layers.push_back(il);
            check(
                    llama_dsv41_graph_trace_name("attn.candidates", il) ==
                        "dsv41.trace.attn.candidates.l" + std::to_string(il),
                    "attention candidate trace name is unstable");
        }
    }
    check(ratio_count[0] == 2 && ratio_count[1] == 20 && ratio_count[2] == 18,
            "ratio 0/1/2 layer counts mismatch");
    check(kv_sources == 4, "KV source ownership count mismatch");
    check(index_sources == 8, "index source ownership count mismatch");
    check(candidate_sources == 1, "candidate source ownership count mismatch");
    check(candidate_trace_layers == std::vector<uint32_t>({ 24, 28, 32, 36 }),
            "attention candidate trace layer coverage mismatch");
    check(
            llama_dsv41_graph_trace_name("attn.candidate_blocks", 20) ==
                "dsv41.trace.attn.candidate_blocks.l20",
            "candidate block trace name is unstable");
    check(
            llama_dsv41_graph_trace_name("engram.row_ids", 1) ==
                "dsv41.trace.engram.row_ids.l1" &&
            llama_dsv41_graph_trace_name("engram.row_ids", 14) ==
                "dsv41.trace.engram.row_ids.l14",
            "Engram row trace names are unstable");
    check(llama_dsv41_build_layer_plan(39, { 39 }, 1024).collapses_output,
            "final layer must preserve streams for carried-pre output collapse");
}

static void test_graph_construction() {
    ggml_init_params params = {
        /*.mem_size   =*/ 4*1024*1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ false,
    };
    ggml_context * ctx = ggml_init(params);
    check(ctx != nullptr, "failed to create graph test context");

    ggml_tensor * kv = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 512, 2, 3);
    ggml_tensor * gate = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 512, 2, 3);
    ggml_tensor * pooled = llama_dsv41_build_ratio_pool(ctx, kv, gate, 2);
    check(pooled->ne[0] == 512 && pooled->ne[1] == 3, "ratio-2 graph output shape mismatch");

    ggml_tensor * direct = llama_dsv41_build_ratio_pool(
            ctx, ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 512, 1, 3), nullptr, 1);
    check(direct->ne[0] == 512 && direct->ne[1] == 3, "ratio-1 graph output shape mismatch");

    ggml_tensor * compressed_scores = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 32, 64, 2);
    ggml_tensor * raw_scores = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 128, 64, 2);
    ggml_tensor * probs = llama_dsv41_build_shared_softmax(ctx, raw_scores, compressed_scores);
    check(probs->ne[0] == 160 && probs->ne[1] == 64 && probs->ne[2] == 2, "shared-softmax graph shape mismatch");

    ggml_tensor * raw_order = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 2);
    ggml_tensor * compressed_order = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    const float raw_values[] = { 1.0f, 2.0f };
    const float compressed_value = 3.0f;
    std::memcpy(raw_order->data, raw_values, sizeof(raw_values));
    std::memcpy(compressed_order->data, &compressed_value, sizeof(compressed_value));
    ggml_tensor * ordered_probs = llama_dsv41_build_shared_softmax(ctx, raw_order, compressed_order);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, ordered_probs);
    check(ggml_graph_compute_with_ctx(ctx, gf, 1) == GGML_STATUS_SUCCESS, "shared-softmax graph execution failed");
    const float * ordered = static_cast<const float *>(ordered_probs->data);
    check(ordered[0] < ordered[1] && ordered[1] < ordered[2], "shared-softmax segment order mismatch");

    ggml_tensor * f16_cache = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 32, 1);
    std::vector<float> cache_values(32, 1.0f);
    ggml_fp32_to_fp16_row(
            cache_values.data(),
            static_cast<ggml_fp16_t *>(f16_cache->data),
            cache_values.size());
    ggml_tensor * cache_with_sentinel =
        llama_dsv41_graph_append_zero_row(ctx, f16_cache);
    ggml_tensor * sentinel_ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 2);
    static_cast<int32_t *>(sentinel_ids->data)[0] = 1;
    static_cast<int32_t *>(sentinel_ids->data)[1] = 0;
    ggml_tensor * sentinel_rows = ggml_get_rows(ctx, cache_with_sentinel, sentinel_ids);
    ggml_tensor * completion_zero =
        llama_dsv41_graph_completion_zero(ctx, sentinel_rows, GGML_TYPE_F16);
    ggml_cgraph * support_gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(support_gf, sentinel_rows);
    ggml_build_forward_expand(support_gf, completion_zero);
    ggml_backend_t backend = ggml_backend_cpu_init();
    check(backend != nullptr, "failed to create graph support backend");
    for (int i = 0; i < ggml_graph_n_nodes(support_gf); ++i) {
        ggml_tensor * node = ggml_graph_node(support_gf, i);
        if (node->op == GGML_OP_SCALE) {
            check(node->src[0]->type == GGML_TYPE_F32,
                  "DeepSeek V4.1 graph contains a non-F32 SCALE input");
        }
        check(ggml_backend_supports_op(backend, node),
              "CPU backend does not support a DeepSeek V4.1 dependency node");
    }
    ggml_backend_free(backend);
    check(cache_with_sentinel->ne[1] == 2,
          "compressed cache sentinel row was not allocated");
    check(ggml_graph_compute_with_ctx(ctx, support_gf, 1) == GGML_STATUS_SUCCESS,
          "compressed sentinel graph execution failed");
    for (uint32_t i = 0; i < 32; ++i) {
        check(ggml_get_f32_1d(sentinel_rows, i) == 0.0f,
              "compressed sentinel row is not zero");
        check(ggml_get_f32_1d(sentinel_rows, 32 + i) == 1.0f,
              "compressed real row changed");
    }

    ggml_tensor * candidate_scores = ggml_new_tensor_2d(
            ctx, GGML_TYPE_F32, 24, 2);
    ggml_tensor * candidate_bias = ggml_new_tensor_2d(
            ctx, GGML_TYPE_F32, 3, 2);
    float * score_data = static_cast<float *>(candidate_scores->data);
    float * bias_data = static_cast<float *>(candidate_bias->data);
    std::fill_n(score_data, 48, -100.0f);
    std::fill_n(bias_data, 6, 0.0f);
    score_data[0] = 100.0f;
    score_data[24] = 100.0f;
    std::fill(score_data + 9, score_data + 24, -std::numeric_limits<float>::infinity());
    std::fill(score_data + 24 + 17, score_data + 48, -std::numeric_limits<float>::infinity());
    bias_data[1] = std::numeric_limits<float>::infinity();
    bias_data[2] = -std::numeric_limits<float>::infinity();
    bias_data[3 + 2] = std::numeric_limits<float>::infinity();
    ggml_tensor * block_scores = ggml_pool_1d(
            ctx, candidate_scores, GGML_OP_POOL_MAX, 8, 8, 0);
    block_scores = ggml_add(ctx, block_scores, candidate_bias);
    ggml_tensor * candidate_ids = ggml_cont(
            ctx, ggml_argsort_top_k(ctx, block_scores, 2));
    ggml_set_name(
            candidate_ids,
            llama_dsv41_graph_trace_name(
                "attn.candidate_blocks", 20).c_str());
    ggml_cgraph * candidate_gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(candidate_gf, candidate_ids);
    check(
            ggml_graph_compute_with_ctx(ctx, candidate_gf, 1) ==
                GGML_STATUS_SUCCESS,
            "candidate block graph execution failed");
    const int32_t * ids = static_cast<const int32_t *>(candidate_ids->data);
    check(
            ids[0] == 1 && ids[1] == 0 &&
            ids[2] == 2 && ids[3] == 0,
            "candidate blocks are not pinned and sorted");
    check(
            ggml_is_contiguous(candidate_ids) &&
            std::string(ggml_get_name(candidate_ids)) ==
                "dsv41.trace.attn.candidate_blocks.l20",
            "candidate block trace is not stable and contiguous");

    ggml_tensor * prior_ring = ggml_new_tensor_2d(
            ctx, GGML_TYPE_F32, 1, 2);
    ggml_tensor * current_k = ggml_new_tensor_2d(
            ctx, GGML_TYPE_F32, 1, 1);
    ggml_tensor * read_idx = ggml_new_tensor_1d(
            ctx, GGML_TYPE_I32, 1);
    ggml_tensor * write_idx = ggml_new_tensor_1d(
            ctx, GGML_TYPE_I64, 1);
    static_cast<float *>(prior_ring->data)[0] = 10.0f;
    static_cast<float *>(prior_ring->data)[1] = 20.0f;
    static_cast<float *>(current_k->data)[0] = 30.0f;
    static_cast<int32_t *>(read_idx->data)[0] = 0;
    static_cast<int64_t *>(write_idx->data)[0] = 0;
    ggml_tensor * prior_read = ggml_get_rows(
            ctx, prior_ring, read_idx);
    ggml_tensor * attention = ggml_add(
            ctx, prior_read, current_k);
    ggml_tensor * completion = ggml_argsort_top_k(
            ctx, ggml_view_1d(ctx, attention, 1, 0), 1);
    completion = ggml_scale(
            ctx, ggml_cast(ctx, completion, GGML_TYPE_F32), 0.0f);
    ggml_tensor * delayed_k = ggml_add(
            ctx, current_k, completion);
    ggml_tensor * ring_update = ggml_set_rows(
            ctx, prior_ring, delayed_k, write_idx);
    ggml_cgraph * ring_gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(ring_gf, ring_update);
    check(
            ggml_graph_compute_with_ctx(ctx, ring_gf, 1) ==
                GGML_STATUS_SUCCESS,
            "ordered raw-ring update graph execution failed");
    check(
            static_cast<float *>(attention->data)[0] == 40.0f &&
            static_cast<float *>(prior_ring->data)[0] == 30.0f,
            "raw-ring write did not wait for the prior-ring read");

    ggml_tensor * selected_ids = ggml_new_tensor_2d(
            ctx, GGML_TYPE_I32, 3, 2);
    const int32_t selected_values[] = { 5, 1, 3, 4, 0, 2 };
    std::memcpy(selected_ids->data, selected_values, sizeof(selected_values));
    ggml_tensor * selected_order = ggml_argsort(
            ctx, ggml_cast(ctx, selected_ids, GGML_TYPE_F32),
            GGML_SORT_ORDER_ASC);
    ggml_tensor * selected_sorted = ggml_get_rows(
            ctx, ggml_reshape_3d(ctx, selected_ids, 1, 3, 2),
            selected_order);
    selected_sorted = ggml_cont(
            ctx, ggml_reshape_2d(ctx, selected_sorted, 3, 2));
    ggml_tensor * routing_probs = ggml_new_tensor_2d(
            ctx, GGML_TYPE_F32, 6, 2);
    const float routing_values[] = {
        10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f,
        20.0f, 21.0f, 22.0f, 23.0f, 24.0f, 25.0f,
    };
    std::memcpy(
            routing_probs->data, routing_values,
            sizeof(routing_values));
    ggml_tensor * selected_weights = ggml_get_rows(
            ctx,
            ggml_reshape_3d(ctx, routing_probs, 1, 6, 2),
            selected_sorted);
    selected_weights = ggml_cont(
            ctx, ggml_reshape_2d(ctx, selected_weights, 3, 2));
    ggml_cgraph * selected_gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(selected_gf, selected_sorted);
    ggml_build_forward_expand(selected_gf, selected_weights);
    check(
            ggml_graph_compute_with_ctx(ctx, selected_gf, 1) ==
                GGML_STATUS_SUCCESS,
            "selected ID ordering graph execution failed");
    const int32_t selected_expected[] = { 1, 3, 5, 0, 2, 4 };
    check(
            std::memcmp(
                selected_sorted->data, selected_expected,
                sizeof(selected_expected)) == 0,
            "selected IDs are not accumulated in original ID order");
    const float selected_weight_expected[] = {
        11.0f, 13.0f, 15.0f, 20.0f, 22.0f, 24.0f,
    };
    check(
            std::memcmp(
                selected_weights->data, selected_weight_expected,
                sizeof(selected_weight_expected)) == 0,
            "routing weights are not paired with sorted original IDs");

    ggml_tensor * residual = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 32, 4, 2);
    ggml_tensor * pre = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 2);
    ggml_tensor * collapsed = llama_dsv41_build_output_collapse(ctx, residual, pre, 32, 4, 2);
    check(collapsed->type == GGML_TYPE_BF16, "output collapse BF16 boundary is missing");
    check(collapsed->ne[0] == 32 && collapsed->ne[1] == 2, "output collapse graph shape mismatch");

    ggml_tensor * output_norm = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 32);
    ggml_tensor * output = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 32, 64);
    ggml_tensor * logits = llama_dsv41_build_output(ctx, residual, pre, output_norm, output, 1.0e-20f, 4);
    check(logits->ne[0] == 64 && logits->ne[1] == 2, "final output graph shape mismatch");

    ggml_tensor * original_ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 6, 2);
    ggml_tensor * slot_ids = ggml_cont(ctx, original_ids);
    check(ggml_is_contiguous(original_ids), "original expert IDs must be contiguous");
    check(ggml_is_contiguous(slot_ids) && slot_ids != original_ids,
            "slot IDs must be a distinct contiguous remap");

    ggml_tensor * routed = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 32, 2);
    ggml_tensor * shared = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 32, 2);
    ggml_tensor * combined = ggml_add(ctx, routed, shared);
    check(combined->src[0] == routed && combined->src[1] == shared,
            "shared expert output is not added to routed output");

    ggml_free(ctx);
}

int main() {
    test_hparams();
    test_source_maps();
    test_compression();
    test_state();
    test_raw_ring();
    test_candidates();
    test_output_collapse();
    test_graph_contract();
    test_graph_construction();
    return 0;
}
