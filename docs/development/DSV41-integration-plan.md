# DeepSeek V4.1 integration plan

The machine-readable release input is `tools/deepseek-v41-trace/integration-plan.json`. It keeps conversion, runtime correctness, and the published model as separate replaceable inputs until final integration.

## Dependency boundaries

- `ggml-org/llama.cpp#28696` at `b12818a24407175d941e9299e7b5fb7874a654d9` is the conversion and GGUF schema half. It does not provide runtime execution and is not a merge dependency of PR59.
- `halo-box/strix-llama.cpp#59` at `013bfa15b80f7dbc7760c4dd7202efbbcf74a952` is the complementary runtime correctness half. Its successor commit remains external to this branch until the current head passes the sealed native model-free closure.
- The ds4 exporter-v2 package is pinned at `650c4c937b06b90b7a06dec5d6bbe4fd1ba82d55` on branch `deepseek-v41-exporter-v2`. Its contract lock still targets the current integration head and must be regenerated after the final PR59 merge.
- The published GGUF remains `/mnt/models/DeepSeek-V4.1-Flash-Q2.gguf`, 365713686528 bytes, SHA-256 `1ce6a8f8806205c13330d7ca287bd198331dc5ca35ccc5d8a9a92a188a6f6f42`. Conversion experiments must use a separate output and must not replace or modify this file.

The conversion PR reports estimated complete outputs of 246.3 GiB for Q2_K and 323.4 GiB for Q3_K_M. Neither fits a 128 GiB unified-memory machine as a fully resident model. End-to-end usability combines the converter half with the runtime half. Runtime viability therefore depends on the separate lossless file-backed Engram and routed-expert paths, not on a smaller conversion estimate.

## NVMe runtime profile

The published GGUF is the backing store for both offload paths:

- Engram tensors are registered with `TENSOR_SKIP`. `llama_engram_table` reads their exact I8 GGUF extents with uncached aligned reads and decodes only selected rows.
- Routed expert tensors are registered as file extents. `llama_expert_store` uses direct I/O, rejects buffered-I/O fallback, and fills the admitted resident cache.
- Dense tensors, state, graph workspace, Engram staging, expert staging, replacement space, direct-I/O bounce space, outputs, and the safety margin remain part of one admission calculation.

No second model copy or generated offload file is required. The model, repository, temporary directory, prompt, trace output, watchdog artifacts, and reports must all resolve to non-rotational NVMe. The preflight rejects rotational, network, tmpfs, alias, and forbidden-root paths.

The fixed first validation profile is:

```text
-c 32768 -b 2048 -ub 32 -np 1 -ngl 99 -dev ROCm0
--expert-cache-slots 192 --expert-cache-mib 72900
```

Run it only inside the repository watchdog with 116 GiB soft and 118 GiB emergency limits. The preflight must establish `gfx1151`, the exact published model identity, zero configured swap or a kernel-enforced cgroup v2 no-swap scope, the active watchdog lease, and the exact arguments above before model loading.

## Integration gates

1. Verify the conversion dependency and tensor metadata without using its runtime.
2. Build exporter-v2 twice from the pinned ds4 source and require identical package evidence.
3. Require the PR59 current-head native model-free packet to pass all three selectors, trace host/install CTests, and its zero-skip marker.
4. Merge PR59 normally. Do not rebase or rewrite its preserved input.
5. Rebuild the combined integration head and rerun the focused admission, schema, Engram, expert, memory, runtime, no-allocation, trace, trace-host, and trace-install tests.
6. Relock exporter-v2 to the immutable integrated head and rerun its exact contract suite.
7. Only after an explicit host release, run the unchanged published GGUF under the watchdog and preserve all commands, hashes, logs, and cleanup evidence.

Until gate 3 passes, model-backed status is `INCOMPLETE`. No performance claim is authorized by model-free validation.
