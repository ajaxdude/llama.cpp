# DeepSeek V4.1 integration plan

The machine-readable release input is `tools/deepseek-v41-trace/integration-plan.json`. It keeps conversion, runtime correctness, and the published model as separate replaceable inputs until final integration.

## Dependency boundaries

- `ggml-org/llama.cpp#28696` at `b12818a24407175d941e9299e7b5fb7874a654d9` is the conversion and GGUF schema half. It does not provide runtime execution and is not a merge dependency of PR59.
- `halo-box/strix-llama.cpp#59` at `013bfa15b80f7dbc7760c4dd7202efbbcf74a952` is the complementary runtime correctness half. Its successor commit remains external to this branch until the current head passes the sealed native model-free closure.
- The ds4 exporter-v2 package is pinned at `650c4c937b06b90b7a06dec5d6bbe4fd1ba82d55` on branch `deepseek-v41-exporter-v2`. Its contract lock still targets the current integration head and must be regenerated after the final PR59 merge.
- The `antirez/ds4` support reference is pinned at `9139e2ae58a41503968a500f36f75895c1ba63fc`. The immutable 13-file documentation review has SHA-256 `c3f0694874941d0397dcfcd2151b4a53e0ba1ada183048ffa209a1355a71a0d1`.
- The published GGUF remains `/mnt/models/DeepSeek-V4.1-Flash-Q2.gguf`, 365713686528 bytes, SHA-256 `1ce6a8f8806205c13330d7ca287bd198331dc5ca35ccc5d8a9a92a188a6f6f42`. Conversion experiments must use a separate output and must not replace or modify this file.

The conversion PR reports estimated complete outputs of 246.3 GiB for Q2_K and 323.4 GiB for Q3_K_M. Neither fits a 128 GiB unified-memory machine as a fully resident model. End-to-end usability combines the converter half with the runtime half. Runtime viability therefore depends on the separate lossless file-backed Engram and routed-expert paths, not on a smaller conversion estimate.

## Schema alignment status

The exact pinned converter and runtime are not yet compatible. This is a release blocker, independent of native execution:

| Surface | Converter `b12818a` | Runtime `013bfa15` | Result |
|---|---|---|---|
| Core metadata | Standard GGUF names such as `deepseek41.context_length` and `deepseek41.embedding_length` | Requires custom names beginning with `deepseek41.config`, `deepseek41.max_position_embeddings`, and `deepseek41.hidden_size` | Runtime fails at the first required key |
| Engram layout | Does not write `engram.encoding`, `engram.rows`, or `engram.compressed_vocab_size` | Requires all three before allocation | Runtime rejects the layout |
| Engram primes | Writes a `UINT64` array | Reads a `uint32_t` array and rejects other element types | Type mismatch |
| Engram tensors | Writes `engram_q`, `engram_k`, and `engram_wkv` | Requires `engram_q_norm`, `engram_k_norm`, and `engram_kv` | Three tensors cannot be resolved |

The exact matches are `engram.layer_ids`, `engram.pad_id`, `engram.token_map`, `engram.multipliers`, and `engram_embd.weight`. These partial matches are insufficient. Final integration requires either a converter update that emits the runtime contract or a reviewed runtime compatibility adapter. Native validation cannot override this schema gate.

## Conversion provenance

Any replacement GGUF must be converted from the original safetensors at source revision `df42c109f1defefcbfcedbe7d905718a12266e40`. The conversion must preserve the native FP8 Engram values and scales in the disk-only GGUF tail. Requantizing the published or bootstrap GGUF is forbidden, including when adding an imatrix.

This provenance requirement is pending a generated conversion artifact; it is not attributed to the current converter pin by documentation alone. The requested DS4 reference suite includes `tests/test_deepseek41_conversion.py`, `tests/test_deepseek41_manifest.py`, `test-linux-memory`, `test-engram`, `test-deepseek41-gguf`, `test-frontends`, and `test-session-state`. These tests are release inputs, not substitutes for the generated-GGUF schema fixture or later inference correctness.

The exact DS4 suite at `9139e2ae` is currently incomplete: 6 of 8 requested commands passed on Darwin, one exact command failed, and one is blocked. `make -C gguf-tools libds4quants.so` failed because Darwin exposes `libds4quants.dylib`; the exact platform-equivalent target passed. The no-argument manifest command requires `--hf-dir`, and synthetic config, index, and metadata-only shard attempts cannot satisfy its model structure and valid payload-extent checks. It is classified `BLOCKED_BY_REQUIRED_MODEL_METADATA`, not as a runtime failure or a model-free pass.

The final evidence hashes are `6f75cc8ca6921cba19b99b92c6321eafb12cb40fecf468b01b2118869bbf7a92` for all logs, `5465ab6ae0ebfe1b663e668e14117f176305c0d1cec4e6365c297f42fdbab21c` for analysis documents, and `4c18dfec321a159e9c2b42bb4c04981dbd44a047aa73f2f0445217adfbd52bc5` for complete evidence excluding binaries. No model weights, conversion, inference, server, or Strix access occurred.

## Reference support boundary

The pinned DS4 documentation supports V4.1 Metal text and vision from `bd66c402070042bf0a79ad6ece8242de4c93680c` and CUDA Q2 text SSD from `a04f46fa423e45712c8c7e430eff422479f314a3`. It documents no V4.1 ROCm, pipeline, or speculative/DSpark support.

PR59 is an experimental Strix-specific ROCm candidate, not evidence that the upstream DS4 support matrix has changed. Its fixed validation profile therefore keeps pipeline parallelism, speculative decoding, and DSpark disabled. The candidate remains release-ineligible until native and model-backed correctness pass. Metal or CUDA measurements cannot be generalized to Strix.

## NVMe runtime profile

The published GGUF is the backing store for both offload paths:

- Engram tensors are registered with `TENSOR_SKIP`. `llama_engram_table` reads their exact I8 GGUF extents with uncached aligned reads and decodes only selected rows.
- Routed expert tensors are registered as file extents. `llama_expert_store` uses direct I/O, rejects buffered-I/O fallback, and fills the admitted resident cache.
- The approximately 189 GiB Engram structure remains disk-only. Its row-read and staging budget is separate from the 192-slot, 72900 MiB routed-expert memory cache.
- Dense tensors, state, graph workspace, Engram staging, expert staging, replacement space, direct-I/O bounce space, outputs, both independent budgets, and the safety margin remain part of one admission calculation.

No second model copy or generated offload file is required. The model, repository, temporary directory, prompt, trace output, watchdog artifacts, and reports must all resolve to non-rotational NVMe. The preflight rejects rotational, network, tmpfs, alias, and forbidden-root paths.

The fixed first validation profile is:

```text
-c 32768 -b 2048 -ub 32 -np 1 -ngl 99 -dev ROCm0
--expert-cache-slots 192 --expert-cache-mib 72900
```

Run it only inside the repository watchdog with 116 GiB soft and 118 GiB emergency limits. The preflight must establish `gfx1151`, the exact published model identity, zero configured swap or a kernel-enforced cgroup v2 no-swap scope, the active watchdog lease, and the exact arguments above before model loading.

## Integration gates

1. Verify the conversion dependency and tensor metadata without using its runtime.
2. Run the pinned DS4 conversion, manifest, Engram, sparse-GGUF, memory, frontend, and session-state model-free suites.
3. Reject ROCm pipeline, speculative decoding, and DSpark combinations. Keep the candidate single-device and non-speculative until separately implemented and validated.
4. Build exporter-v2 twice from the pinned ds4 source and require identical package evidence.
5. Resolve the converter/runtime schema blockers listed above and add an exact generated-GGUF schema fixture.
6. Require the PR59 current-head native model-free packet to pass all three selectors, trace host/install CTests, and its zero-skip marker.
7. Merge PR59 normally. Do not rebase or rewrite its preserved input.
8. Rebuild the combined integration head and rerun the focused admission, schema, Engram, expert, memory, runtime, no-allocation, trace, trace-host, and trace-install tests.
9. Relock exporter-v2 to the immutable integrated head and rerun its exact contract suite.
10. Only after an explicit host release, run the unchanged published GGUF under the watchdog and preserve all commands, hashes, logs, and cleanup evidence.

Current acceptance is `LOCAL_MODEL_FREE_PASS_REFERENCE_INCOMPLETE_NATIVE_INCOMPLETE`: local trace, watchdog, and focused CTests pass; the DS4 reference suite is blocked on model metadata; and A07 stopped during privileged preparation before the launcher. A08 is sealed for the same source revision but is not authorized; its scope is model-free build, install, trace, and containment evidence only. Rootless execution cannot satisfy the current immutable-owner, supplementary-group, UID-map, and ROCm trust contract; the review is bound by SHA-256 `98a941861d93ce7ccafc76824c609d495fa7cb032c74b1b13006e8a9f711bb31`. The shortest accepted workflow remains one human `sudo -v` after explicit release. Until the schema gate, reference manifest gate, and native closure all pass, model-backed status is `INCOMPLETE`. No performance claim is authorized by model-free validation.
