#include "llama-dsv41.h"
#include "llama-hparams.h"
#include "models.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

static float dsv41_rope_attn_factor(float freq_scale) {
    return 1.0f/(1.0f + 0.1f*logf(1.0f/freq_scale));
}

void llama_model_deepseek41::load_arch_hparams(llama_model_loader & ml) {
    llama_dsv41_config config = {};
    std::string raw_config;
    ml.get_key(LLM_KV_DSV41_CONFIG, raw_config);
    ml.get_key(LLM_KV_DSV41_MAX_POSITION_EMBEDDINGS, config.n_ctx_train);
    ml.get_key(LLM_KV_DSV41_HIDDEN_SIZE, config.n_embd);
    ml.get_key(LLM_KV_DSV41_NUM_HIDDEN_LAYERS, config.n_layer);
    ml.get_key(LLM_KV_DSV41_VOCAB_SIZE, config.n_vocab);
    ml.get_key(LLM_KV_DSV41_NUM_ATTENTION_HEADS, config.n_head);
    ml.get_key(LLM_KV_DSV41_NUM_KEY_VALUE_HEADS, config.n_head_kv);
    ml.get_key(LLM_KV_DSV41_HEAD_DIM, config.n_head_dim);
    ml.get_key(LLM_KV_DSV41_QK_ROPE_HEAD_DIM, config.n_rot);
    ml.get_key(LLM_KV_DSV41_Q_LORA_RANK, config.n_lora_q);
    ml.get_key(LLM_KV_DSV41_O_LORA_RANK, config.n_lora_o);
    ml.get_key(LLM_KV_DSV41_O_GROUPS, config.n_o_group);
    ml.get_key(LLM_KV_DSV41_MOE_INTERMEDIATE_SIZE, config.n_ff_expert);
    ml.get_key(LLM_KV_DSV41_N_ROUTED_EXPERTS, config.n_expert);
    ml.get_key(LLM_KV_DSV41_NUM_EXPERTS_PER_TOK, config.n_expert_used);
    ml.get_key(LLM_KV_DSV41_N_SHARED_EXPERTS, config.n_expert_shared);
    ml.get_key(LLM_KV_DSV41_INDEX_N_HEADS, config.indexer_n_head);
    ml.get_key(LLM_KV_DSV41_INDEX_HEAD_DIM, config.indexer_head_size);
    ml.get_key(LLM_KV_DSV41_INDEX_TOPK, config.indexer_top_k);
    ml.get_key(LLM_KV_DSV41_HC_MULT, config.hc_count);
    ml.get_key(LLM_KV_DSV41_HC_SINKHORN_ITERS, config.hc_sinkhorn_iters);
    ml.get_key(LLM_KV_DSV41_SLIDING_WINDOW, config.raw_window);
    ml.get_key(LLM_KV_DSV41_CANDIDATE_SOURCE_LAYER_ID, config.candidate_source_layer);
    ml.get_key(LLM_KV_DSV41_CANDIDATE_TOPK_BLOCKS, config.candidate_topk_blocks);
    ml.get_key(LLM_KV_DSV41_CANDIDATE_BLOCK_SIZE, config.candidate_block_size);
    ml.get_key(LLM_KV_DSV41_RMS_NORM_EPS, config.f_norm_rms_eps);
    ml.get_key(LLM_KV_DSV41_HC_EPS, config.hc_eps);
    ml.get_key(LLM_KV_DSV41_SWIGLU_LIMIT, config.swiglu_clamp);
    ml.get_key(LLM_KV_DSV41_ROUTED_SCALING_FACTOR, config.routed_scale);
    ml.get_key(LLM_KV_DSV41_ROPE_THETA, config.rope_theta);
    ml.get_key(LLM_KV_DSV41_COMPRESS_ROPE_THETA, config.compress_rope_theta);
    ml.get_key(LLM_KV_DSV41_ROPE_SCALING_FACTOR, config.yarn_factor);
    ml.get_key(LLM_KV_DSV41_ROPE_SCALING_BETA_FAST, config.yarn_beta_fast);
    ml.get_key(LLM_KV_DSV41_ROPE_SCALING_BETA_SLOW, config.yarn_beta_slow);
    ml.get_key(LLM_KV_DSV41_ROPE_SCALING_ORIG_CTX_LEN, config.yarn_original_context);
    ml.get_key(LLM_KV_DSV41_NORM_TOPK_PROB, config.expert_weights_norm);
    ml.get_key(LLM_KV_DSV41_HIDDEN_ACT, config.hidden_act);
    ml.get_key(LLM_KV_DSV41_SCORING_FUNC, config.scoring_func);
    ml.get_key(LLM_KV_DSV41_TOPK_METHOD, config.topk_method);
    ml.get_arr(LLM_KV_DSV41_COMPRESS_RATIOS, config.compress_ratios);
    ml.get_arr(LLM_KV_DSV41_KV_SOURCE_LAYER_IDS, config.kv_sources);
    ml.get_arr(LLM_KV_DSV41_INDEX_SOURCE_LAYER_IDS, config.index_sources);
    ml.get_key(LLM_KV_DSV41_ENGRAM_ENCODING, config.engram_encoding);
    ml.get_arr(LLM_KV_DSV41_ENGRAM_LAYER_IDS, config.engram_layers);
    ml.get_arr(LLM_KV_DSV41_ENGRAM_ROWS, config.engram_rows);
    ml.get_key(LLM_KV_DSV41_ENGRAM_COMPRESSED_VOCAB_SIZE, config.engram_compressed_vocab_size);
    ml.get_key(LLM_KV_DSV41_ENGRAM_PAD_ID, config.engram_pad_id);
    ml.get_arr_n(LLM_KV_DSV41_ENGRAM_TOKEN_MAP, config.engram_token_map_size);
    ml.get_arr_n(LLM_KV_DSV41_ENGRAM_PRIMES, config.engram_primes_size);
    ml.get_arr_n(LLM_KV_DSV41_ENGRAM_MULTIPLIERS, config.engram_multipliers_size);

    config.n_ff_dense = LLAMA_DSV41_N_FF_DENSE;
    llama_dsv41_validate_config(config);

    if (raw_config.empty()) {
        throw std::runtime_error("DeepSeek V4.1 metadata: config must not be empty");
    }
    hparams.n_ctx_train = config.n_ctx_train;
    hparams.n_embd = config.n_embd;
    hparams.n_embd_out_impl = config.n_embd;
    hparams.n_layer_all = config.n_layer;
    hparams.n_layer_nextn = 0;
    hparams.n_expert = config.n_expert;
    hparams.n_expert_shared = config.n_expert_shared;
    hparams.n_lora_q = config.n_lora_q;
    hparams.n_ff_shexp = config.n_ff_expert;
    hparams.n_embd_head_k_full = config.n_head_dim;
    hparams.n_embd_head_v_full = config.n_head_dim;
    hparams.n_embd_head_k_swa = config.n_head_dim;
    hparams.n_embd_head_v_swa = config.n_head_dim;
    hparams.n_rot_full = config.n_rot;
    hparams.n_rot_swa = config.n_rot;
    hparams.n_swa = config.raw_window;
    hparams.indexer_n_head = config.indexer_n_head;
    hparams.indexer_head_size = config.indexer_head_size;
    hparams.indexer_top_k = config.indexer_top_k;
    hparams.dsv4_o_group_count = config.n_o_group;
    hparams.dsv4_o_lora_rank = config.n_lora_o;
    hparams.dsv4_hc_mult = config.hc_count;
    hparams.dsv4_hc_sinkhorn_iters = config.hc_sinkhorn_iters;
    hparams.dsv4_compress_rope_base = config.compress_rope_theta;
    hparams.dsv4_hc_eps = config.hc_eps;
    hparams.dsv41_candidate_source_layer = config.candidate_source_layer;
    hparams.dsv41_candidate_topk_blocks = config.candidate_topk_blocks;
    hparams.dsv41_candidate_block_size = config.candidate_block_size;
    hparams.f_norm_rms_eps = config.f_norm_rms_eps;
    hparams.expert_weights_scale = config.routed_scale;
    hparams.expert_weights_norm = config.expert_weights_norm;
    hparams.expert_gating_func = LLAMA_EXPERT_GATING_FUNC_TYPE_SQRT_SOFTPLUS;
    hparams.rope_freq_base_train = config.rope_theta;
    hparams.rope_freq_base_train_swa = config.rope_theta;
    hparams.rope_freq_scale_train = 1.0f/config.yarn_factor;
    hparams.rope_freq_scale_train_swa = hparams.rope_freq_scale_train;
    hparams.n_ctx_orig_yarn = config.yarn_original_context;
    hparams.yarn_beta_fast = config.yarn_beta_fast;
    hparams.yarn_beta_slow = config.yarn_beta_slow;
    hparams.yarn_ext_factor = 1.0f;
    hparams.rope_attn_factor = dsv41_rope_attn_factor(hparams.rope_freq_scale_train);
    hparams.swa_type = LLAMA_SWA_TYPE_STANDARD;
    hparams.causal_attn = true;

    for (uint32_t il = 0; il < config.n_layer; ++il) {
        hparams.n_head_arr[il] = config.n_head;
        hparams.n_head_kv_arr[il] = config.n_head_kv;
        hparams.n_ff_arr[il] = config.n_ff_dense;
        hparams.n_ff_exp_arr[il] = config.n_ff_expert;
        hparams.n_expert_used_arr[il] = config.n_expert_used;
        hparams.swiglu_clamp_exp[il] = config.swiglu_clamp;
        hparams.swiglu_clamp_shexp[il] = config.swiglu_clamp;
        hparams.dsv4_compress_ratios[il] = config.compress_ratios[il];
        hparams.dsv41_kv_source_layer[il] = llama_dsv41_kv_source_layer(il);
        hparams.dsv41_index_source_layer[il] = llama_dsv41_index_source_layer(il);
        hparams.is_swa_impl[il] = 1;
    }
    for (uint32_t il : config.engram_layers) {
        hparams.dsv41_engram_layers.set(il);
    }

    type = LLM_TYPE_UNKNOWN;
}

[[noreturn]] void llama_model_deepseek41::load_arch_tensors(llama_model_loader & ml) {
    LLAMA_LOAD_LOCALS;

    const int64_t q_lora_rank     = hparams.n_lora_q;
    const int64_t n_ff_exp        = hparams.n_ff_exp();
    const int64_t n_expert_shared = hparams.n_expert_shared;
    const int64_t n_embd_head     = hparams.n_embd_head_k();
    const int64_t o_groups        = hparams.dsv4_o_group_count;
    const int64_t o_lora_rank     = hparams.dsv4_o_lora_rank;
    const int64_t hc_mult         = hparams.dsv4_hc_mult;
    const int64_t hc_dim          = hc_mult*n_embd;
    const int64_t hc_mix_dim      = (2 + hc_mult)*hc_mult;

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, 0);
    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), { n_embd }, 0);
    output = create_tensor(tn(LLM_TENSOR_OUTPUT, "weight"), { n_embd, n_vocab }, 0);

    for (int32_t il = 0; il < n_layer; ++il) {
        auto & layer = layers[il];

        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", il), { n_embd }, 0);
        layer.attn_sinks = create_tensor(tn(LLM_TENSOR_ATTN_SINKS, "weight", il), { n_head }, 0);
        layer.wq_a = create_tensor(tn(LLM_TENSOR_ATTN_Q_A, "weight", il), { n_embd, q_lora_rank }, 0);
        layer.attn_q_a_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_A_NORM, "weight", il), { q_lora_rank }, 0);
        layer.wq_b = create_tensor(tn(LLM_TENSOR_ATTN_Q_B, "weight", il), { q_lora_rank, n_head*n_embd_head }, 0);
        layer.wkv = create_tensor(tn(LLM_TENSOR_ATTN_KV, "weight", il), { n_embd, n_embd_head }, 0);
        layer.attn_kv_a_norm = create_tensor(tn(LLM_TENSOR_ATTN_KV_A_NORM, "weight", il), { n_embd_head }, 0);
        layer.wo_a = create_tensor(tn(LLM_TENSOR_ATTN_OUT_A, "weight", il), { n_head*n_embd_head/o_groups, o_lora_rank, o_groups }, TENSOR_ALLOW_RESHAPE);
        layer.wo_b = create_tensor(tn(LLM_TENSOR_ATTN_OUT_B, "weight", il), { o_groups*o_lora_rank, n_embd }, 0);

        layer.hc_attn_fn = create_tensor(tn(LLM_TENSOR_HC_ATTN_FN, "weight", il), { hc_dim, hc_mix_dim }, 0);
        layer.hc_attn_base = create_tensor(tn(LLM_TENSOR_HC_ATTN_BASE, "weight", il), { hc_mix_dim }, 0);
        layer.hc_attn_scale = create_tensor(tn(LLM_TENSOR_HC_ATTN_SCALE, "weight", il), { 3 }, 0);
        layer.hc_ffn_fn = create_tensor(tn(LLM_TENSOR_HC_FFN_FN, "weight", il), { hc_dim, hc_mix_dim }, 0);
        layer.hc_ffn_base = create_tensor(tn(LLM_TENSOR_HC_FFN_BASE, "weight", il), { hc_mix_dim }, 0);
        layer.hc_ffn_scale = create_tensor(tn(LLM_TENSOR_HC_FFN_SCALE, "weight", il), { 3 }, 0);

        if (hparams.dsv41_is_kv_source(il)) {
            layer.attn_comp_wkv = create_tensor(tn(LLM_TENSOR_ATTN_COMPRESSOR_WKV, "weight", il), { n_embd, n_embd_head }, 0);
            layer.attn_comp_norm = create_tensor(tn(LLM_TENSOR_ATTN_COMPRESSOR_NORM, "weight", il), { n_embd_head }, 0);
            if (hparams.dsv4_compress_ratios[il] == 2) {
                layer.attn_comp_wgate = create_tensor(tn(LLM_TENSOR_ATTN_COMPRESSOR_WGATE, "weight", il), { n_embd, n_embd_head }, 0);
            }
            layer.indexer_attn_k = create_tensor(tn(LLM_TENSOR_INDEXER_ATTN_K, "weight", il), { n_embd_head, hparams.indexer_head_size }, 0);
            layer.indexer_k_norm = create_tensor(tn(LLM_TENSOR_INDEXER_K_NORM, "weight", il), { hparams.indexer_head_size }, 0);
        }
        if (hparams.dsv41_is_index_source(il)) {
            layer.indexer_proj = create_tensor(tn(LLM_TENSOR_INDEXER_PROJ, "weight", il), { n_embd, hparams.indexer_n_head }, 0);
            layer.indexer_attn_q_b = create_tensor(tn(LLM_TENSOR_INDEXER_ATTN_Q_B, "weight", il), { q_lora_rank, hparams.indexer_n_head*hparams.indexer_head_size }, 0);
        }

        layer.ffn_gate_inp = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP, "weight", il), { n_embd, n_expert }, 0);
        layer.ffn_exp_probs_b = create_tensor(tn(LLM_TENSOR_FFN_EXP_PROBS_B, "bias", il), { n_expert }, 0);
        layer.ffn_exp_probs_b_vl = create_tensor(tn(LLM_TENSOR_FFN_EXP_PROBS_B_VL, "bias", il), { n_expert }, TENSOR_NOT_REQUIRED);
        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", il), { n_embd }, 0);
        layer.ffn_gate_exps = create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", il), { n_embd, n_ff_exp, n_expert }, 0);
        layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", il), { n_ff_exp, n_embd, n_expert }, 0);
        layer.ffn_up_exps = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS, "weight", il), { n_embd, n_ff_exp, n_expert }, 0);
        layer.ffn_gate_shexp = create_tensor(tn(LLM_TENSOR_FFN_GATE_SHEXP, "weight", il), { n_embd, n_ff_exp*n_expert_shared }, 0);
        layer.ffn_down_shexp = create_tensor(tn(LLM_TENSOR_FFN_DOWN_SHEXP, "weight", il), { n_ff_exp*n_expert_shared, n_embd }, 0);
        layer.ffn_up_shexp = create_tensor(tn(LLM_TENSOR_FFN_UP_SHEXP, "weight", il), { n_embd, n_ff_exp*n_expert_shared }, 0);

        if (hparams.dsv41_engram_layers.test(il)) {
            const llm_tensor engram_tensors[] = {
                LLM_TENSOR_ENGRAM_EMBD,
                LLM_TENSOR_ENGRAM_Q_NORM,
                LLM_TENSOR_ENGRAM_K_NORM,
                LLM_TENSOR_ENGRAM_KV,
            };
            for (llm_tensor tensor : engram_tensors) {
                const std::string name = tn(tensor, "weight", il).str();
                if (ml.get_weight(name.c_str()) == nullptr) {
                    throw std::runtime_error("DeepSeek V4.1 is missing required Engram tensor " + name);
                }
            }
        }
    }

    throw std::runtime_error(
            std::string("DeepSeek V4.1 tensor metadata is valid, but tensor payloads cannot be mapped: ") +
            llama_dsv41_runtime_dependency_error());
}

[[noreturn]] std::unique_ptr<llm_graph_context> llama_model_deepseek41::build_arch_graph(const llm_graph_params &) const {
    throw std::runtime_error(llama_dsv41_runtime_dependency_error());
}
