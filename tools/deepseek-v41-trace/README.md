# DeepSeek V4.1 correctness traces

This directory defines the versioned cross-runtime trace format used by issue #48. It compares the unchanged published GGUF between llama.cpp and ds4 revision `bd66c402070042bf0a79ad6ece8242de4c93680c`.

Each trace is a directory:

- `manifest.json` records the model and prompt SHA-256 values, exact runtime revision/build, inference configuration, environment, and memory/swap/watchdog audit references.
- `events.jsonl` is an ordered stream of content-addressed event records.
- `blobs/<sha256>.bin` stores canonical little-endian tensor bytes. This keeps complete logits and per-token state exact without embedding large numeric arrays in JSON.

The required hard-failure event components are `prompt.tokens`, `engram.row_ids`, `expert.ids`, `expert.weights`, `attn.source`, `attn.candidate_blocks`, `attn.candidates`, `logits.prefill`, `logits.decode`, and `decode.greedy_token`. `expert.ids` must declare `semantic_id_space: "original"`; cache slot IDs are rejected.

Validate or compare bundles:

```sh
python3 tools/deepseek-v41-trace/trace_format.py validate TRACE
python3 tools/deepseek-v41-trace/trace_format.py compare DS4_TRACE LLAMA_TRACE --report report.json
```

The first mismatch is reported by phase, decode step, token range, layer, component, byte offset, and logical element index. All required components use exact byte comparison. There is no tolerance mode.

## Corpus matrix

Use only the target model and these repository files:

```text
tests/corpus/correctness-prose.txt
tests/corpus/correctness-code.txt
tests/corpus/correctness-structured.txt
tests/corpus/correctness-numeric.txt
```

Start at context 32768. Repeat a corpus deterministically when more prompt tokens are required, and save the exact repeated bytes before either runtime executes. Run prefill plus at least eight greedy decode steps in one reused context. Boundary runs must vary the prefill chunk size without changing prompt bytes or model settings.

## Strix execution gate

`run_ds4.py` verifies the pinned ds4 checkout and refuses model execution when swap is enabled, the watchdog PID file is missing/stale, another matching DS4 workload is active, or any model/prompt/trace path resolves under `/mnt/bigspace`.

```sh
python3 tools/deepseek-v41-trace/run_ds4.py \
  --model /mnt/models/deepseek-v41/DeepSeek-V4.1-Flash-Q2.gguf \
  --prompt /path/on/nvme/correctness-prose-32768.txt \
  --output /path/on/nvme/traces/ds4-prose-32768 \
  --watchdog-pid-file /run/user/$(id -u)/dsv41-watchdog.pid \
  --exporter /path/to/pinned-ds4-trace-exporter \
  --exporter-sha256 <trusted-build-sha256>
```

The exporter is intentionally external to the canonical ds4 checkout. It must be built from the pinned revision and emit this trace format without changing the canonical checkout. The launcher requires its trusted SHA-256 and rejects a bundle unless the exporter reports the pinned revision.

The llama.cpp exporter is built as `llama-deepseek-v41-trace`. It accepts the normal model, context, batch, ubatch, KV, Flash Attention, offload, and expert-cache arguments. `-f` supplies the exact prompt bytes, `-n` is the number of greedy decode steps, and `-o` is the trace directory. It also requires `DSV41_TRACE_MEMORY_AUDIT`, `DSV41_TRACE_SWAP_AUDIT`, and `DSV41_TRACE_WATCHDOG_AUDIT` so every run points to its safety evidence.

Use `run_llama.py` on the validation host instead of calling the exporter directly. It applies the same zero-swap, watchdog, active-workload, and NVMe gates and writes separate audit files next to the trace directory.

`run_matrix.py` copies the four repository corpora byte-for-byte into the NVMe result directory, records their hashes, runs ds4 and llama.cpp with matched context/decode settings, compares each bundle immediately, and stops at the first divergence. Its default context matrix is 32768. Pass `--contexts 32768 65536 98304 131072` only after the 32K target passes.
