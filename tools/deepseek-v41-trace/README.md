# DeepSeek V4.1 correctness traces

This directory defines the versioned cross-runtime trace format used by issue #48. It compares the unchanged published GGUF between llama.cpp and ds4 revision `bd66c402070042bf0a79ad6ece8242de4c93680c`.

Each trace is a directory:

- `manifest.json` records the model and prompt SHA-256 values, exact runtime revision/build, inference configuration, environment, and content-addressed memory/swap/watchdog audit references.
- `events.jsonl` is an ordered stream of content-addressed event records.
- `blobs/<sha256>.bin` stores canonical little-endian tensor bytes. This keeps complete logits and per-token state exact without embedding large numeric arrays in JSON.
- `audits/pre/<sha256>.json` and `audits/post/<sha256>.json` store immutable safety evidence from both sides of execution.
- `provenance/<sha256>.json` binds the exact prompt to its fixed corpus, published model, target token count, and prompt-builder executable.

The required hard-failure event components are `prompt.bytes`, `prompt.tokens`, `engram.row_ids`, `expert.ids`, `expert.weights`, `attn.source`, `attn.candidate_blocks`, `attn.candidates`, `logits.prefill`, `logits.decode`, and `decode.greedy_token`. `expert.ids` must declare `semantic_id_space: "original"`; cache slot IDs are rejected. Any graph tensor in the reserved `dsv41.trace.*` namespace with an unknown component, malformed suffix, or unexpected layer fails the exporter.

Internal tensors use raw ggml dimension order, and every dimension must be positive. The validator requires Engram rows as i32 `[24, token_count]`, original expert IDs as i32 `[6, token_count]`, router weights as f32 `[6, token_count]`, attention-source IDs as nonempty rank-2 i32 with width at most 512 and `token_count` in the second dimension, layer-20 candidate blocks as nonempty rank-2 i32 with width at most 2048, propagated candidates as nonempty rank-2 i32 with width at most 512, and complete f32 logits as `[129280]`. Original expert IDs must be within `0..383`.

Validate or compare bundles:

```sh
python3 tools/deepseek-v41-trace/trace_format.py validate TRACE
python3 tools/deepseek-v41-trace/trace_format.py compare DS4_TRACE LLAMA_TRACE --report report.json
```

The first mismatch is reported by phase, decode step, exact token, layer, component, byte offset, flat element index, and per-token component element index. All required components use exact byte comparison. There is no tolerance mode. A ds4 bundle is invalid unless it reports revision `bd66c402070042bf0a79ad6ece8242de4c93680c`.

## Corpus matrix

Use only the target model and these repository files:

```text
tests/corpus/correctness-prose.txt
tests/corpus/correctness-code.txt
tests/corpus/correctness-structured.txt
tests/corpus/correctness-numeric.txt
```

Their SHA-256 values are fixed in `trace_format.py`; the matrix refuses modified corpus bytes. The published GGUF must have SHA-256 `1ce6a8f8806205c13330d7ca287bd198331dc5ca35ccc5d8a9a92a188a6f6f42`.

Start at context 32768. `llama-deepseek-v41-prompt-builder` loads only the GGUF vocabulary, repeats each repository corpus deterministically, truncates the token sequence to `context - decode_steps`, detokenizes it, and requires exact token round-trip before saving the prompt. `run_matrix.py` creates each prompt once before either runtime executes and reuses its exact bytes for every chunk-boundary case. Run prefill plus at least eight greedy decode steps in one reused context.

## Strix execution gate

`run_ds4.py` verifies that the pinned ds4 checkout has no tracked or untracked changes and refuses model execution when swap is enabled, the watchdog lease or heartbeat is missing/stale, another matching DS4 workload is active, or any model/prompt/trace path resolves under `/mnt/bigspace`.

The watchdog lease is JSON, not a bare PID:

```json
{"pid":1234,"start_time_ticks":5678,"command_sha256":"<sha256-of-/proc/1234/cmdline>","heartbeat_path":"/run/user/1000/dsv41-watchdog.heartbeat","max_heartbeat_age_seconds":30}
```

The watchdog must update the heartbeat file with the current Unix timestamp at least every 30 seconds. The wrappers verify the PID, Linux process start time, exact command bytes, and heartbeat before and after execution. The llama exporter repeats the same identity and heartbeat check before finalizing its trace.

```sh
python3 tools/deepseek-v41-trace/run_ds4.py \
  --model /mnt/models/deepseek-v41/DeepSeek-V4.1-Flash-Q2.gguf \
  --prompt /path/on/nvme/correctness-prose-32768.txt \
  --corpus-name correctness-prose.txt \
  --corpus-sha256 2da590a37e3297767336c10b024a0de732d64bee4da5792596f8ddf49ea408d2 \
  --prompt-provenance /path/on/nvme/correctness-prose-32768.txt.provenance.json \
  --output /path/on/nvme/traces/ds4-prose-32768 \
  --watchdog-pid-file /run/user/$(id -u)/dsv41-watchdog.pid \
  --exporter /path/to/pinned-ds4-trace-exporter \
  --exporter-sha256 <trusted-build-sha256>
```

The exporter is intentionally external to the canonical ds4 checkout. It must be built from the pinned revision and emit this trace format without changing the canonical checkout. The launcher requires its trusted SHA-256 and rejects a bundle unless the exporter reports the pinned revision and its build SHA-256 matches the executed file.

The llama.cpp exporter is built as `llama-deepseek-v41-trace`. It accepts the normal model, context, batch, ubatch, KV, Flash Attention, offload, and expert-cache arguments. `-bf` supplies the exact prompt bytes, `-n` is the number of greedy decode steps, and `-o` is the trace directory. It also requires `DSV41_TRACE_MEMORY_AUDIT`, `DSV41_TRACE_SWAP_AUDIT`, and `DSV41_TRACE_WATCHDOG_AUDIT` so every run points to its safety evidence.

Use `run_llama.py` on the validation host instead of calling the exporter directly. It applies the same zero-swap, watchdog, active-workload, and NVMe gates and embeds content-addressed preflight and postflight evidence in the trace. It requires the exact candidate revision, full-graph base revision, expected base-to-candidate binary diff SHA-256, and repository path. It rejects tracked or untracked checkout changes and rejects an exporter whose embedded build revision or executable hash does not match that attestation.

`run_matrix.py` copies the four repository corpora byte-for-byte into the NVMe result directory, verifies their fixed hashes, builds exact-length prompt artifacts and content-addressed provenance, runs ds4 and llama.cpp with matched context/decode settings, compares each bundle immediately, and stops at the first divergence. Pass both `--llama-exporter` and `--llama-prompt-builder` from the same build, plus the candidate revision, full-graph base revision, and expected binary diff SHA-256. Its default context matrix is 32768. Pass `--contexts 32768 65536 98304 131072` only after the 32K target passes.
