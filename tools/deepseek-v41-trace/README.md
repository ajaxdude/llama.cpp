# DeepSeek V4.1 correctness traces

This directory defines version 2 of the cross-runtime trace format used by issue #48. It compares the unchanged published GGUF between llama.cpp and ds4 revision `bd66c402070042bf0a79ad6ece8242de4c93680c`.

Each trace is a directory:

- `manifest.json` records the model and prompt SHA-256 values, exact runtime revision/build, inference configuration, environment, runtime-specific host evidence, exact execution paths, and content-addressed audit references.
- `events.jsonl` is an ordered stream of content-addressed event records.
- `blobs/<sha256>.bin` stores canonical little-endian tensor bytes. This keeps complete logits and per-token state exact without embedding large numeric arrays in JSON.
- `audits/pre/<sha256>.json` and `audits/post/<sha256>.json` store immutable safety evidence from both sides of execution.
- `provenance/<sha256>.json` binds the exact prompt to its fixed corpus, published model, target token count, and prompt-builder executable.

Every manifest and memory audit declares that the expert cache and KV cache are memory-resident and that there are no external cache or state paths. Missing, substituted, or additional file-backed cache/state declarations fail closed.

The required hard-failure event components are `prompt.bytes`, `prompt.tokens`, `engram.row_ids`, `expert.ids`, `expert.weights`, `attn.source`, `attn.candidate_blocks`, `attn.candidates`, `logits.prefill`, `logits.decode`, and `decode.greedy_token`. `expert.ids` must declare `semantic_id_space: "original"`; cache slot IDs are rejected. Any graph tensor in the reserved `dsv41.trace.*` namespace with an unknown component, malformed suffix, or unexpected layer fails the exporter.

Every bundle carries one strict runtime-discriminated accelerator attestation. A llama.cpp candidate uses the `strix-rocm` kind: the selected backend device must map through its PCI identity and Linux KFD topology to `gfx_target_version=110501` (`gfx1151`). A ds4 oracle uses the `apple-metal` kind: the selected Metal device records its registry ID, reported architecture, unified-memory property, and recommended working-set size. Missing kinds, unknown kinds, cross-kind fields, duplicate JSON keys, and mixed evidence fail closed. Cross-runtime comparison does not require the two physical accelerators or PCI identities to match; each runtime proves its own execution environment, while the model, prompt, inference semantics, and complete output artifacts remain exact comparison inputs.

Internal tensors use raw ggml dimension order, and every dimension must be positive. The validator requires Engram rows as i32 `[24, token_count]`, original expert IDs as i32 `[6, token_count]`, router weights as f32 `[6, token_count]`, layer-0/1 raw attention-source rows as i32 `[128, token_count]`, compressed attention-source IDs as nonempty rank-2 i32 with width at most 512, layer-20 candidate blocks as nonempty rank-2 i32 with width at most 2048, propagated candidates as nonempty rank-2 i32 with width at most 512, and complete f32 logits as `[129280]`. Raw attention rows use physical ring IDs `0..127`, visible current-ubatch IDs `128..128+token_index`, and unavailable sentinel `128+token_count`; layers 0 and 1 must be byte-identical for each execution step. Original expert IDs must be within `0..383`.

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

The initial correctness matrix uses the final admitted physical ubatch of 32. The worst-case routed expert union is `min(384, 6*32) = 192` experts per layer. The published GGUF metadata yields 398131200 bytes per cross-layer expert slot, so the required cache is exactly 76441190400 bytes, or 72900 MiB. The launchers reject other ubatch, slot, or cache-byte values. Host-memory admission still accounts the cache, staging, model, graph, state, outputs, current host use, and safety margin together before allocation.

Run the oracle matrix only from the final integration commit that contains the canonical watchdog, memory admission, and this correctness harness. Supply its exact revision, its immutable merge-base, and a newly computed base-to-head binary diff SHA-256 to `run_matrix.py`; do not reuse the standalone correctness PR head or diff identity.

ROCm on `gfx1151` is the primary acceptance backend:

```sh
HIPCXX="$(hipconfig -l)/clang" HIP_PATH="$(hipconfig -R)" \
  cmake -S . -B build-dsv41-trace-rocm \
    -DGGML_HIP=ON \
    -DGPU_TARGETS=gfx1151 \
    -DGGML_NATIVE=ON \
    -DCMAKE_BUILD_TYPE=Release
cmake --build build-dsv41-trace-rocm --config Release -j "$(nproc)" --target \
  llama-deepseek-v41-trace \
  llama-deepseek-v41-prompt-builder \
  test-backend-ops \
  test-deepseek41-schema \
  test-deepseek41-engram \
  test-deepseek41-expert \
  test-deepseek41-memory \
  test-deepseek41-runtime \
  test-deepseek41-trace-host

build-dsv41-trace-rocm/bin/test-backend-ops -b ROCm0 -o MUL_MAT_ID
build-dsv41-trace-rocm/bin/test-backend-ops -b ROCm0 -o MUL_MAT
build-dsv41-trace-rocm/bin/test-backend-ops -b ROCm0 -o GET_ROWS
build-dsv41-trace-rocm/bin/test-backend-ops -b ROCm0 -o SET_ROWS
build-dsv41-trace-rocm/bin/test-backend-ops -b ROCm0 -o CPY
```

Do not change host ROCm packages for this run. Vulkan can provide secondary coverage, but it cannot replace the required ROCm low-level and oracle evidence. The llama runner selects `ROCm0` explicitly, invokes the exact exporter for a pre-allocation device attestation, and rejects the run unless the backend PCI identity maps to exactly one KFD node reporting `gfx1151`. The native exporter repeats the query before model allocation and verifies that the loaded model still uses the same device.

Set `HIP_LAUNCH_BLOCKING=1` on the canonical watchdog command that owns the complete Strix matrix process group. The llama.cpp wrapper fails closed if this variable is absent or different, and every embedded Strix memory, swap, and watchdog audit records it. This Linux/ROCm setting is not an Apple Metal oracle requirement.

## Strix candidate execution gate

`run_llama.py` refuses model execution when swap is enabled, the canonical watchdog lease, heartbeat, or JSONL audit is missing or stale, another unrelated matching model workload is active, or any model/prompt/trace path fails the storage gate.

The approved watchdog revision is exactly `778db6f50eae04e6c232c69b9575bdbd0747962b`, with `scripts/strix_memory_watchdog.py` SHA-256 `d2781a25f978dd2bc14fc113079aa2dbf513aa157b44da9d0d51d750daa6c94f`. Both Python validators and the native exporter reject every other revision or script hash.

The approved watchdog must own the complete matrix process group and expose its canonical validation and process-group lease-guard APIs. The wrappers verify its pinned script identity, Python executable and argv position, PID and Linux start time, exact command bytes, 116/118/120 GiB thresholds, `/proc` source, watchdog/guardian/matrix topology, current process group, child command hash, atomic lease/heartbeat identities, heartbeat freshness and persistent-audit record hash, and the watchdog-held audit lock. The direct matrix payload starts the canonical process-group lease guard before inference. The wrappers repeat validation before and after each runtime.

Use one empty directory on verified non-rotational NVMe for every Strix input and output. The Python launcher and both native tools resolve the nearest existing output parent through `/proc/self/mountinfo`, `/sys/dev/block`, and `/sys/class/block`. They require a resolvable local NVMe block device with `queue/rotational=0`; tmpfs, network filesystems, rotational disks, unknown devices, lexical or resolved `/mnt/bigspace` paths, and forbidden-root symlink escapes fail closed. Btrfs subvolume sources such as `/dev/nvme0n1p3[/home]` are resolved through the parent block device. `TMPDIR` is mandatory, its original lexical pathname must be an existing writable non-symlink directory on verified NVMe, and there is no `/tmp` fallback. Literal shell shorthand such as `~/tmp` is rejected because environment values are not shell-expanded for the launched process. These metadata commands do not execute the model:

```sh
MODEL=/mnt/models/DeepSeek-V4.1-Flash-Q2.gguf
REPO=/home/papa/src/strix-llama-integration
RUN_ROOT=/home/papa/dsv41-correctness
TMPDIR=/home/papa/tmp/dsv41
export TMPDIR

test "$(stat -c %s "$MODEL")" = 365713686528
test "$(realpath "$MODEL")" = /mnt/models/DeepSeek-V4.1-Flash-Q2.gguf
test "$(realpath -m "$REPO")" = /home/papa/src/strix-llama-integration
test "$(realpath -m "$RUN_ROOT")" = /home/papa/dsv41-correctness
test "$(realpath -m "$TMPDIR")" = /home/papa/tmp/dsv41
mkdir -p "$RUN_ROOT" "$TMPDIR"
findmnt -no SOURCE,FSTYPE,TARGET -T "$MODEL"
findmnt -no SOURCE,FSTYPE,TARGET -T /home
lsblk -d -o NAME,ROTA,TYPE,SIZE,MODEL
df -B1 /mnt/models /home
sha256sum "$MODEL"
PYTHONPATH=gguf-py python3 -m gguf.scripts.gguf_dump "$MODEL" > "$RUN_ROOT/model-metadata.txt"
git -C /home/papa/src/ds4-v41 status --short
git -C /home/papa/src/ds4-v41 rev-parse HEAD
test "$(awk 'NR > 1 { count++ } END { print count + 0 }' /proc/swaps)" = 0
```

The expected model digest is `1ce6a8f8806205c13330d7ca287bd198331dc5ca35ccc5d8a9a92a188a6f6f42`, both selected block devices report `ROTA=0`, and `/proc/swaps` has zero entries. The observed planning snapshot had 76366495744 bytes free on `/mnt/models` and 679635001344 bytes free under `/home`; recheck before every run. Keep the unchanged 365713686528-byte GGUF in place on `/mnt/models`. Do not copy the model or place builds, logs, traces, audit files, or temporary files there. Put all of those under `/home`, and never use `/mnt/bigspace`.

The exporter is intentionally external to the canonical ds4 checkout. It must be built from the pinned revision and emit this trace format without changing the canonical checkout. The launcher requires its trusted SHA-256 and rejects a bundle unless the exporter reports the pinned revision and its build SHA-256 matches the executed file.

The llama.cpp exporter is built as `llama-deepseek-v41-trace`. It accepts the normal model, context, batch, ubatch, KV, Flash Attention, offload, and expert-cache arguments. `-bf` supplies the exact prompt bytes, `-n` is the number of greedy decode steps, and `-o` is the trace directory. It also requires `DSV41_TRACE_MEMORY_AUDIT`, `DSV41_TRACE_SWAP_AUDIT`, and `DSV41_TRACE_WATCHDOG_AUDIT` so every run points to its safety evidence. The content-addressed memory audit binds the preflight accelerator and storage attestations; the manifest binds the independently repeated native accelerator attestation.

Use `run_llama.py` on the validation host instead of calling the exporter directly. It applies the same zero-swap, watchdog, active-workload, exact-`gfx1151`, and proven-NVMe gates and embeds content-addressed preflight and postflight evidence in the trace. It requires the exact final integration revision, immutable oracle revision, expected oracle-to-candidate binary diff SHA-256, and repository path. It rejects tracked or untracked checkout changes and rejects an exporter whose embedded build revision, executable hash, accelerator identity, or loaded model device does not match that attestation.

`run_matrix.py --llama-only` copies the four repository corpora byte-for-byte into the NVMe result directory, verifies their fixed hashes, builds exact-length prompt artifacts and content-addressed provenance, and captures the llama.cpp side. Pass both `--llama-exporter` and `--llama-prompt-builder` from the same build, plus the final integration revision, immutable oracle revision, and expected binary diff SHA-256. Its default context matrix is 32768. Pass later contexts only after the 32K target passes. The Apple oracle is captured separately with `run_ds4.py`; compare completed per-case bundles with `trace_format.py compare`.

## Apple Metal oracle execution gate

The external ds4 exporter is not present in the pinned canonical checkout. It remains blocked until a separately built executable is reviewed and its exact SHA-256 is added to the otherwise empty `APPROVED_EXPORTERS` map. Approval is checked before the exporter can run, including device-only preflight. `run_ds4.py` requires macOS arm64, at least 128 GiB of measured host memory, zero swap, no unrelated matching workload, an exact selected Metal device query, and an existing writable non-symlink `TMPDIR`. The model, prompt, output, harness repository, ds4 checkout, temporary directory, Python executable, runner script, and exporter must resolve through `df -P` to a volume that `diskutil info -plist` proves is internal solid-state storage backed by NVMe or Apple Fabric. SATA, network, virtual, disk-image, external/non-internal, non-solid-state, and incomplete device identities fail closed, as do lexical or resolved forbidden paths.

The ds4 memory audit binds the exact Metal accelerator, host model/OS/memory identity, and every storage record. The runner audit binds the Python runner process, UID, executable/script paths and hashes, exporter path/hash, pinned checkout path/revision, and exact command hash. The runner script must be inside the attested harness repository. The preflight and postflight accelerator and host identities must remain unchanged. These Apple audits replace Linux KFD, `/proc`, HIP, and Strix watchdog claims; the oracle must never fabricate those fields.

After the exporter is independently reviewed on an authorized 128 GiB or larger Apple oracle host, add its exact executable SHA-256 and pinned ds4 revision to `APPROVED_EXPORTERS` in `run_ds4.py`; a caller-provided digest alone is not sufficient oracle provenance. The exporter must answer `--dsv41-attest-device Metal0` without loading the model and emit the strict `apple-metal` attestation. Its trace command interface is:

The unpublished `ds4gguf` documentation revision `e13893ffcb33e90c8852929303e188102df7a8f5` is provenance only. It is not an executable dependency, exporter approval, or fixture source. Executable tests and fixtures stay in this `strix-llama.cpp` stack.

```text
--model PATH --prompt-file PATH --output PATH --context N --decode-steps N --prefill-chunk 32 --device Metal0
```

It must emit a complete valid `dsv41-trace` bundle, report ds4 revision `bd66c402070042bf0a79ad6ece8242de4c93680c`, put its own executable SHA-256 in `manifest.json`, and report the same selected Metal device before and after execution. `run_ds4.py` embeds the platform-native memory, swap, runner, accelerator, host, storage, and exact-path evidence.

Capture the first llama.cpp matrix under the watchdog:

```sh
REPO=/home/papa/src/strix-llama-integration
MODEL=/mnt/models/DeepSeek-V4.1-Flash-Q2.gguf
RUN_ROOT=/home/papa/dsv41-correctness
CASE_ROOT="$RUN_ROOT/c32768"
CANDIDATE_REV="$(git -C "$REPO" rev-parse HEAD)"
BASE_REV=<full-immutable-oracle-revision>
DIFF_SHA256="$(git -C "$REPO" diff --binary --no-ext-diff "$BASE_REV" "$CANDIDATE_REV" -- | sha256sum | awk '{print $1}')"

mkdir -p "$CASE_ROOT/watchdog"
cd "$REPO"
HIP_LAUNCH_BLOCKING=1 python3 scripts/strix_memory_watchdog.py \
  --procfs-root /proc \
  --soft-gib 116 \
  --emergency-gib 118 \
  --grace-seconds 30 \
  --sample-interval-seconds 1 \
  --lease-path "$CASE_ROOT/watchdog/lease.json" \
  --heartbeat-path "$CASE_ROOT/watchdog/heartbeat.json" \
  --audit-path "$CASE_ROOT/watchdog/audit.jsonl" \
  --heartbeat-max-age-seconds 5 \
  -- \
  python3 tools/deepseek-v41-trace/run_matrix.py \
    --repo "$REPO" \
    --model "$MODEL" \
    --output "$CASE_ROOT/matrix" \
    --llama-runner "$REPO/tools/deepseek-v41-trace/run_llama.py" \
    --llama-exporter "$REPO/build-dsv41-trace-rocm/bin/llama-deepseek-v41-trace" \
    --llama-prompt-builder "$REPO/build-dsv41-trace-rocm/bin/llama-deepseek-v41-prompt-builder" \
    --candidate-revision "$CANDIDATE_REV" \
    --base-revision "$BASE_REV" \
    --candidate-diff-sha256 "$DIFF_SHA256" \
    --llama-only \
    --contexts 32768 \
    --ubatches 32 \
    --batch 2048 \
    --device ROCm0 \
    --expert-cache-slots 192 \
    --expert-cache-mib 72900
```

Use a new empty output and watchdog directory for each later context. Repeat the same command after setting `CASE_ROOT` and changing `--contexts`:

```sh
CASE_ROOT="$RUN_ROOT/c65536"  # then use --contexts 65536
CASE_ROOT="$RUN_ROOT/c98304"  # then use --contexts 98304
CASE_ROOT="$RUN_ROOT/c131072" # then use --contexts 131072
```

On the separately authorized Apple oracle, place the unchanged GGUF, exact prompt, prompt-provenance record, harness checkout, pinned ds4 checkout, exporter, trace output, and `TMPDIR` on internal solid-state storage. Then run one case at a time:

```sh
export TMPDIR=/Users/oracle/dsv41/tmp
python3 tools/deepseek-v41-trace/run_ds4.py \
  --repo /Users/oracle/src/strix-llama.cpp \
  --checkout /Users/oracle/src/ds4-v41 \
  --exporter /Users/oracle/bin/dsv41-trace-exporter \
  --exporter-sha256 <approved-exact-sha256> \
  --model /Users/oracle/models/DeepSeek-V4.1-Flash-Q2.gguf \
  --prompt /Users/oracle/dsv41/inputs/correctness-prose-c32768.txt \
  --prompt-provenance /Users/oracle/dsv41/inputs/correctness-prose-c32768.txt.provenance.json \
  --output /Users/oracle/dsv41/traces/correctness-prose-c32768-ub32 \
  --corpus-name correctness-prose.txt \
  --corpus-sha256 2da590a37e3297767336c10b024a0de732d64bee4da5792596f8ddf49ea408d2 \
  --context 32768 \
  --decode-steps 8 \
  --prefill-chunk 32 \
  --device Metal0
```

Compare the completed bundle with the matching Strix bundle using `trace_format.py compare`. Repeat for all four corpora before expanding the context matrix.

## Strix bring-up evidence lane

The 128 GiB or larger Apple Metal run remains an external verification gate for the pinned ds4 cross-runtime oracle. A Strix bring-up can complete first, but it must report `BRINGUP PASS`, never `TARGET PASS`. It does not replace complete byte-identical ds4 logits.

The pinned `antirez/ds4@bd66c402070042bf0a79ad6ece8242de4c93680c` evidence anchors are:

| Evidence | SHA-256 | Limitation |
|---|---|---|
| `tests/test-vectors/README.md` | `0e59b2f2832bed8af0a91e6ff20962debf964cd2d1d141c086e52cfcc995a1c3` | Official-vector provenance and limitations |
| `tests/test-vectors/flash-0731/manifest.json` | `ebf237a5660a6851fb8085e77f532901a9d758208b25d7ed5af0b7af4b28f91b` | Official API provenance |
| `tests/test-vectors/flash-0731/official.vec` | `77ae699889bfaf1348768dcbe7ea2c72279ae86abb10470d3e1b08cd1fd82a83` | Official selected-token/top-logprob slice, not full logits |
| `tests/test-vectors/flash-0731/local-golden.vec` | `23d942ff3b9bb2a3f82927d11aa3ed1461e1f302071e788d0d95a5c165e47d3b` | Local tolerant top-64 drift anchor, not exact |
| `tests/test_engram.c` | `198a561d981f62518a9d28035480a7e220b99c156cde6d248b8baabd684cc74b` | Model-free Engram oracle |
| `ds4_engram.c` | `2b6ca468510ebf45ee298a905525bc7234dacad9a384bf2011eba19ba2c0bdf7` | Engram implementation under test |
| `ds4_engram.h` | `f84a264e0fe199d23a6f0c56fbbd19e222adc7009eed13c185af6403f68f7f5c` | Engram schema |
| `tests/test_deepseek41_metal.c` | `9197c2f9d65b380ce25be5334991e4bfaf40e82e6708e64f2b9329552412d28c` | Synthetic exact candidate/top-k oracle; source anchor only on Strix |
| `tests/test_deepseek41_graph.c` | `6dc786f831c93ae7f5aa56e7518f67657125f7c0fd35cead3c646eff3d3f9e09` | Synthetic routing and graph oracle |
| `tests/test_deepseek41_prefill.c` | `452774b9332d393822d84288eeb2d71ca1fb25f1ba30d20a49160d1787de734f` | Prefill boundary fixture |
| `tests/test_deepseek41_manifest.py` | `2d7aa1fc93805d9c97839c5de9eccc856628f13e6913c3bbc3f0c99b0827aeae` | Model manifest/schema oracle |
| `tests/test_deepseek41_conversion.py` | `a40b83062a9b91338773addd77fe62650296f4de607057cb4fd1329287f347b5c` | Conversion/schema fixture |
| `tests/test_deepseek41_gguf.c` | `8f41e049d5ec179c38a1a00ac61712db0306902f0ef74bcdb989d8993316ff35` | GGUF schema oracle |
| `gguf-tools/deepseek41_metadata.py` | `39300bbd504165b97de017edd72563377f50b8a5511ca7478252a86f31a0009b` | Conversion metadata source |
| `ds4.c` | `1776dbfed177ea14f3ce6cac1d8d0b1c1b44dfff2c2663769a9a5634aeec34e7` | Runtime schema source |

Verify these files from the pinned checkout before using their results. Do not copy an unpublished exporter or depend on a private ds4 remote.

```sh
python3 tools/deepseek-v41-trace/verify_ds4_anchors.py \
  --checkout /home/papa/src/ds4-v41
```

`run_matrix.py --llama-only` captures all four repository corpora without claiming cross-runtime success. Run the final integration build twice with separate output directories, then run an equivalently instrumented immutable base build once. Keep every run under its own canonical watchdog invocation and use the exact ubatch/cache/ROCm arguments above.

For each case (`correctness-prose-c32768-ub32`, `correctness-code-c32768-ub32`, `correctness-structured-c32768-ub32`, and `correctness-numeric-c32768-ub32`), require both comparisons:

```sh
python3 tools/deepseek-v41-trace/trace_format.py compare-local self-consistency \
  "$RUN_A/llama/$CASE" "$RUN_B/llama/$CASE" \
  --report "$REPORTS/$CASE-self.json"

python3 tools/deepseek-v41-trace/trace_format.py compare-local base-regression \
  "$BASE_RUN/llama/$CASE" "$RUN_A/llama/$CASE" \
  --report "$REPORTS/$CASE-base.json"
```

Both commands compare every exact trace component, including complete prefill/decode logits, tokens, Engram rows, original expert IDs and weights, raw/compressed attention sources, and candidate propagation. `base-regression` also requires the base trace revision to equal the integrated trace's attested oracle revision. The report includes `cross_runtime_status: "INCOMPLETE"` even when it returns `BRINGUP PASS`.

The pinned ds4 evidence commands are:

```sh
git -C /home/papa/src/ds4-v41 diff --quiet
git -C /home/papa/src/ds4-v41 diff --cached --quiet
test "$(git -C /home/papa/src/ds4-v41 rev-parse HEAD)" = bd66c402070042bf0a79ad6ece8242de4c93680c
/home/papa/src/ds4-v41/tests/test_engram
DS4_TEST_MODEL="$MODEL" \
DS4_TEST_VECTOR_FILE=/home/papa/src/ds4-v41/tests/test-vectors/flash-0731/official.vec \
  /home/papa/src/ds4-v41/ds4_test --logprob-vectors
```

Capture the exact commands, executable hashes, stdout/stderr hashes, exit status, and watchdog artifacts. The official-vector result and pinned fixtures are supporting bring-up evidence only. The first `TARGET PASS` still requires the external pinned ds4 exporter and the full cross-runtime trace comparison.
