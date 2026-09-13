#!/usr/bin/env python3

import importlib.util
import json
import struct
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

TRACE_DIR = Path(__file__).parents[1] / "tools" / "deepseek-v41-trace"
sys.path.insert(0, str(TRACE_DIR))
MODULE_PATH = TRACE_DIR / "trace_format.py"
SPEC = importlib.util.spec_from_file_location("dsv41_trace_format", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
trace = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(trace)
import run_llama
import run_ds4
import preflight


AUDIT_RECORDS = {
    "memory": {
        "created_unix": 1,
        "kind": "memory",
        "data": {"mem_total_bytes": 128, "mem_available_bytes": 64, "mem_used_bytes": 64},
    },
    "swap": {
        "created_unix": 1,
        "kind": "swap",
        "data": {"enabled": False, "entries": []},
    },
    "watchdog": {
        "created_unix": 1,
        "kind": "watchdog",
        "data": {
            "pid": 123,
            "start_time_ticks": 456,
            "command_sha256": "7" * 64,
            "heartbeat_path": "/run/user/123/watchdog.heartbeat",
            "heartbeat_unix": 1,
            "max_heartbeat_age_seconds": 30,
        },
    },
}


def audit_bytes(kind: str) -> bytes:
    return (json.dumps(AUDIT_RECORDS[kind], sort_keys=True, separators=(",", ":")) + "\n").encode("ascii")


def provenance_bytes(prompt: bytes = b"abc") -> bytes:
    record = {
        "format": "dsv41-prompt-provenance",
        "version": 1,
        "corpus_name": "correctness-prose.txt",
        "corpus_sha256": trace.CORPUS_SHA256["correctness-prose.txt"],
        "model_sha256": trace.MODEL_SHA256,
        "prompt_sha256": trace.sha256_bytes(prompt),
        "prompt_byte_count": len(prompt),
        "target_tokens": 2,
        "actual_tokens": 2,
        "builder_sha256": "8" * 64,
    }
    return (json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n").encode("ascii")


def manifest(runtime: str = "llama.cpp", prompt: bytes = b"abc") -> dict:
    provenance_sha256 = trace.sha256_bytes(provenance_bytes(prompt))
    result = {
        "runtime": runtime,
        "revision": trace.DS4_REVISION if runtime == "ds4" else "a" * 40,
        "build": {"sha256": "3" * 64},
        "model": {"sha256": trace.MODEL_SHA256, "byte_count": 123, "architecture": "deepseek41"},
        "prompt": {
            "sha256": trace.sha256_bytes(prompt),
            "byte_count": len(prompt),
            "corpus_name": "correctness-prose.txt",
            "corpus_sha256": trace.CORPUS_SHA256["correctness-prose.txt"],
            "target_tokens": 2,
            "provenance": {
                "path": f"provenance/{provenance_sha256}.json",
                "sha256": provenance_sha256,
            },
        },
        "config": {
            "context": 3,
            "decode_steps": 1,
            "batch": 512,
            "ubatch": 128,
            "kv_type_k": "f16",
            "kv_type_v": "f16",
            "flash_attention": True,
            "expert_cache_slots": 8,
            "expert_cache_bytes": 4096,
            "deepseek41": {
                "layer_count": 40,
                "vocab_size": 129280,
                "engram_layers": [1, 14],
                "engram_rows_per_token": 24,
                "expert_count": 384,
                "experts_used": 6,
                "candidate_source_layer": 20,
                "candidate_topk_blocks": 2048,
                "candidate_block_size": 8,
                "index_top_k": 512,
                "candidate_propagation_layers": [24, 28, 32, 36],
            },
        },
        "comparison": {"logits": "byte-identical-f32"},
        "expected": {
            "prompt_tokens": 2,
            "decode_steps": 1,
            "components": {
                "prompt.bytes": {"layers": None, "input": "tokens"},
                "prompt.tokens": {"layers": None, "input": "tokens"},
                "engram.row_ids": {"layers": [1, 14], "prefill": "tokens", "decode": "steps"},
                "expert.ids": {"layers": list(range(40)), "prefill": "tokens", "decode": "steps"},
                "expert.weights": {"layers": list(range(40)), "prefill": "tokens", "decode": "steps"},
                "attn.source": {"layers": list(range(40)), "prefill": "tokens", "decode": "steps"},
                "attn.candidate_blocks": {"layers": [20], "prefill": "tokens", "decode": "steps"},
                "attn.candidates": {"layers": [24, 28, 32, 36], "prefill": "tokens", "decode": "steps"},
                "logits.prefill": {"layers": None, "prefill": "final"},
                "logits.decode": {"layers": None, "decode": "steps"},
                "decode.greedy_token": {"layers": None, "decode": "steps"},
            },
        },
        "environment": {},
        "audits": {
            phase: {
                kind: {
                    "path": f"audits/{phase}/{trace.sha256_bytes(audit_bytes(kind))}.json",
                    "sha256": trace.sha256_bytes(audit_bytes(kind)),
                    "created_unix": 1,
                }
                for kind in ("memory", "swap", "watchdog")
            }
            for phase in ("pre", "post")
        },
    }
    if runtime == "llama.cpp":
        result["candidate"] = {
            "repository": trace.REPOSITORY,
            "revision": "a" * 40,
            "base_revision": "b" * 40,
            "diff_sha256": "c" * 64,
            "executable_sha256": "3" * 64,
        }
    return result


def add_required_events(writer: object, logits: bytes | None = None, prompt: bytes = b"abc") -> None:
    for phase in ("pre", "post"):
        audit_root = writer.root / "audits" / phase
        audit_root.mkdir(parents=True, exist_ok=True)
        for kind in ("memory", "swap", "watchdog"):
            data = audit_bytes(kind)
            (audit_root / f"{trace.sha256_bytes(data)}.json").write_bytes(data)
    provenance_root = writer.root / "provenance"
    provenance_root.mkdir(exist_ok=True)
    data = provenance_bytes(prompt)
    (provenance_root / f"{trace.sha256_bytes(data)}.json").write_bytes(data)
    writer.add_event(
        component="prompt.bytes",
        phase="input",
        step=0,
        token_start=0,
        token_count=2,
        layer=None,
        dtype="bytes",
        shape=[len(prompt)],
        data=prompt,
    )
    writer.add_event(
        component="prompt.tokens",
        phase="input",
        step=0,
        token_start=0,
        token_count=2,
        layer=None,
        dtype="i32",
        shape=[2],
        data=struct.pack("<ii", 10, 20),
    )
    writer.add_event(
        component="engram.row_ids",
        phase="prefill",
        step=0,
        token_start=0,
        token_count=2,
        layer=1,
        dtype="i32",
        shape=[24, 2],
        data=struct.pack("<" + "i" * 48, *range(48)),
    )
    writer.add_event(
        component="expert.ids",
        phase="prefill",
        step=0,
        token_start=0,
        token_count=2,
        layer=0,
        dtype="i32",
        shape=[6, 2],
        data=struct.pack("<iiiiiiiiiiii", 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6),
        semantic_id_space="original",
    )
    writer.add_event(
        component="expert.weights",
        phase="prefill",
        step=0,
        token_start=0,
        token_count=2,
        layer=0,
        dtype="f32",
        shape=[6, 2],
        data=struct.pack("<ffffffffffff", 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6),
    )
    writer.add_event(
        component="attn.source",
        phase="prefill",
        step=0,
        token_start=0,
        token_count=2,
        layer=20,
        dtype="i32",
        shape=[1, 2],
        data=struct.pack("<ii", 20, 20),
    )
    writer.add_event(
        component="attn.candidate_blocks",
        phase="prefill",
        step=0,
        token_start=0,
        token_count=2,
        layer=20,
        dtype="i32",
        shape=[2, 2],
        data=struct.pack("<iiii", 4, 7, 4, 7),
    )
    writer.add_event(
        component="attn.candidates",
        phase="prefill",
        step=0,
        token_start=0,
        token_count=2,
        layer=24,
        dtype="i32",
        shape=[512, 2],
        data=struct.pack("<" + "i" * 1024, *([4, 7] * 512)),
    )
    writer.add_event(
        component="logits.prefill",
        phase="prefill",
        step=0,
        token_start=1,
        token_count=1,
        layer=None,
        dtype="f32",
        shape=[129280],
        data=logits if logits is not None else struct.pack("<" + "I" * 129280, *([0x3F800000] * 129280)),
    )
    writer.add_event(
        component="decode.greedy_token",
        phase="decode",
        step=0,
        token_start=2,
        token_count=1,
        layer=None,
        dtype="i32",
        shape=[1],
        data=struct.pack("<i", 3),
    )
    writer.add_event(
        component="engram.row_ids", phase="decode", step=0, token_start=2, token_count=1,
        layer=1, dtype="i32", shape=[24, 1], data=struct.pack("<" + "i" * 24, *range(24)))
    writer.add_event(
        component="expert.ids", phase="decode", step=0, token_start=2, token_count=1,
        layer=0, dtype="i32", shape=[6, 1], data=struct.pack("<iiiiii", 1, 2, 3, 4, 5, 6),
        semantic_id_space="original")
    writer.add_event(
        component="expert.weights", phase="decode", step=0, token_start=2, token_count=1,
        layer=0, dtype="f32", shape=[6, 1], data=struct.pack("<ffffff", 1, 2, 3, 4, 5, 6))
    writer.add_event(
        component="attn.source", phase="decode", step=0, token_start=2, token_count=1,
        layer=20, dtype="i32", shape=[1, 1], data=struct.pack("<i", 20))
    writer.add_event(
        component="attn.candidate_blocks", phase="decode", step=0, token_start=2, token_count=1,
        layer=20, dtype="i32", shape=[2, 1], data=struct.pack("<ii", 4, 7))
    writer.add_event(
        component="attn.candidates", phase="decode", step=0, token_start=2, token_count=1,
        layer=24, dtype="i32", shape=[512, 1],
        data=struct.pack("<" + "i" * 512, *([4, 7] * 256)))
    writer.add_event(
        component="logits.decode",
        phase="decode",
        step=0,
        token_start=2,
        token_count=1,
        layer=None,
        dtype="f32",
        shape=[129280],
        data=logits if logits is not None else struct.pack("<" + "I" * 129280, *([0x3F800000] * 129280)),
    )
    writer.add_event(
        component="engram.row_ids", phase="prefill", step=0, token_start=0, token_count=2,
        layer=14, dtype="i32", shape=[24, 2], data=struct.pack("<" + "i" * 48, *range(48)))
    writer.add_event(
        component="engram.row_ids", phase="decode", step=0, token_start=2, token_count=1,
        layer=14, dtype="i32", shape=[24, 1], data=struct.pack("<" + "i" * 24, *range(24)))
    for layer in range(1, 40):
        writer.add_event(
            component="expert.ids", phase="prefill", step=0, token_start=0, token_count=2,
            layer=layer, dtype="i32", shape=[6, 2],
            data=struct.pack("<iiiiiiiiiiii", 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6),
            semantic_id_space="original")
        writer.add_event(
            component="expert.ids", phase="decode", step=0, token_start=2, token_count=1,
            layer=layer, dtype="i32", shape=[6, 1], data=struct.pack("<iiiiii", 1, 2, 3, 4, 5, 6),
            semantic_id_space="original")
        writer.add_event(
            component="expert.weights", phase="prefill", step=0, token_start=0, token_count=2,
            layer=layer, dtype="f32", shape=[6, 2],
            data=struct.pack("<ffffffffffff", 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6))
        writer.add_event(
            component="expert.weights", phase="decode", step=0, token_start=2, token_count=1,
            layer=layer, dtype="f32", shape=[6, 1], data=struct.pack("<ffffff", 1, 2, 3, 4, 5, 6))
    for layer in list(range(20)) + list(range(21, 40)):
        writer.add_event(
            component="attn.source", phase="prefill", step=0, token_start=0, token_count=2,
            layer=layer, dtype="i32", shape=[1, 2], data=struct.pack("<ii", 20, 20))
        writer.add_event(
            component="attn.source", phase="decode", step=0, token_start=2, token_count=1,
            layer=layer, dtype="i32", shape=[1, 1], data=struct.pack("<i", 20))
    for layer in (28, 32, 36):
        writer.add_event(
            component="attn.candidates", phase="prefill", step=0, token_start=0, token_count=2,
            layer=layer, dtype="i32", shape=[512, 2],
            data=struct.pack("<" + "i" * 1024, *([4, 7] * 512)))
        writer.add_event(
            component="attn.candidates", phase="decode", step=0, token_start=2, token_count=1,
            layer=layer, dtype="i32", shape=[512, 1],
            data=struct.pack("<" + "i" * 512, *([4, 7] * 256)))


class TraceFormatTests(unittest.TestCase):
    def test_serialization_preserves_float_bits_and_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "trace"
            bits = (0x3F800000, 0x80000000, 0x7FC12345, 0)
            data = struct.pack("<IIII", *bits) + bytes((129280 - len(bits)) * 4)
            with trace.TraceBundleWriter(root, manifest()) as writer:
                add_required_events(writer, data)
            bundle = trace.TraceBundle(root)
            event = next(item for item in bundle.events if item["component"] == "logits.prefill")
            self.assertEqual(bundle.read_blob(event), data)
            self.assertEqual(event["sha256"], trace.sha256_bytes(data))

    def test_compare_reports_first_routing_divergence(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            left = Path(temp) / "left"
            right = Path(temp) / "right"
            with trace.TraceBundleWriter(left, manifest("ds4")) as writer:
                add_required_events(writer)
            with trace.TraceBundleWriter(right, manifest("llama.cpp")) as writer:
                add_required_events(writer)
            right_bundle = trace.TraceBundle(right)
            expert = next(item for item in right_bundle.events if item["component"] == "expert.ids")
            mutated = struct.pack(
                "<iiiiiiiiiiii", 1, 2, 3, 4, 5, 6, 1, 2, 9, 4, 5, 6)
            expert["sha256"] = trace.sha256_bytes(mutated)
            expert["blob"] = f"blobs/{expert['sha256']}.bin"
            new_blob = right / expert["blob"]
            new_blob.write_bytes(mutated)
            events = [
                expert if trace.event_key(item) == trace.event_key(expert) else item
                for item in right_bundle.events
            ]
            (right / trace.EVENTS_NAME).write_text(
                "".join(trace.canonical_json(item) + "\n" for item in events),
                encoding="ascii",
            )
            result = trace.report(trace.TraceBundle(left), trace.TraceBundle(right))
            self.assertEqual(result["status"], "FAIL")
            divergence = result["first_divergence"]
            self.assertEqual(divergence["classification"], "routing_original_expert")
            self.assertEqual(divergence["layer"], 0)
            self.assertEqual(divergence["element_index"], 8)
            self.assertEqual(divergence["token_index"], 1)
            self.assertEqual(divergence["component_element_index"], 2)

    def test_compare_selects_global_first_token_divergence(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            left = Path(temp) / "left"
            right = Path(temp) / "right"
            with trace.TraceBundleWriter(left, manifest("ds4")) as writer:
                add_required_events(writer)
            with trace.TraceBundleWriter(right, manifest("llama.cpp")) as writer:
                add_required_events(writer)
            events_path = right / trace.EVENTS_NAME
            events = [json.loads(line) for line in events_path.read_text(encoding="ascii").splitlines()]
            for layer, element in ((0, 8), (1, 2)):
                event = next(item for item in events if (
                    item["component"] == "expert.ids" and
                    item["phase"] == "prefill" and
                    item["layer"] == layer
                ))
                values = list(struct.unpack("<iiiiiiiiiiii", (right / event["blob"]).read_bytes()))
                values[element] = 9
                data = struct.pack("<iiiiiiiiiiii", *values)
                event["sha256"] = trace.sha256_bytes(data)
                event["blob"] = f"blobs/{event['sha256']}.bin"
                (right / event["blob"]).write_bytes(data)
            events_path.write_text(
                "".join(trace.canonical_json(event) + "\n" for event in events),
                encoding="ascii",
            )
            result = trace.report(trace.TraceBundle(left), trace.TraceBundle(right))
            divergence = result["first_divergence"]
            self.assertEqual(divergence["token_index"], 0)
            self.assertEqual(divergence["layer"], 1)

    def test_llama_runner_preserves_binary_prompt_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            model = root / "model.gguf"
            prompt = root / "prompt.txt"
            exporter = root / "exporter"
            output = root / "trace"
            model.write_bytes(b"model")
            prompt.write_bytes(b"line with trailing newline\n")
            args = Namespace(
                model=model,
                prompt=prompt,
                context=32768,
                decode_steps=8,
                batch=2048,
                ubatch=512,
                gpu_layers=99,
                expert_cache_slots=8,
                expert_cache_mib=4096,
            )
            command = run_llama.build_command(args, exporter, output)
            self.assertEqual(command[command.index("-bf") + 1], str(prompt.resolve()))
            self.assertNotIn("-f", command)

    def test_rejects_cache_slot_id_space(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "trace"
            writer = trace.TraceBundleWriter(root, manifest())
            with self.assertRaisesRegex(trace.TraceError, "original expert IDs"):
                writer.add_event(
                    component="expert.ids", phase="prefill", step=0, token_start=0, token_count=1,
                    layer=0, dtype="i32", shape=[1], data=struct.pack("<i", 3),
                    semantic_id_space="cache_slot")
            writer.events.close()

    def test_detects_truncated_and_corrupt_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "trace"
            with trace.TraceBundleWriter(root, manifest()) as writer:
                add_required_events(writer)
            events_path = root / trace.EVENTS_NAME
            events_path.write_bytes(events_path.read_bytes()[:-1])
            with self.assertRaisesRegex(trace.TraceError, "truncated"):
                trace.TraceBundle(root)

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "trace"
            with trace.TraceBundleWriter(root, manifest()) as writer:
                add_required_events(writer)
            bundle = trace.TraceBundle(root)
            event = bundle.events[0]
            (root / event["blob"]).write_bytes(b"bad")
            with self.assertRaisesRegex(trace.TraceError, "truncated blob|corrupt blob"):
                trace.TraceBundle(root)

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "trace"
            bad_manifest = manifest()
            bad_manifest["audits"]["pre"]["watchdog"] = "watchdog.json"
            with trace.TraceBundleWriter(root, bad_manifest) as writer:
                add_required_events(writer)
            with self.assertRaisesRegex(trace.TraceError, "watchdog audit reference"):
                trace.TraceBundle(root)

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "trace"
            trace_manifest = manifest()
            with trace.TraceBundleWriter(root, trace_manifest) as writer:
                add_required_events(writer)
            (root / trace_manifest["audits"]["pre"]["memory"]["path"]).unlink()
            with self.assertRaisesRegex(trace.TraceError, "pre memory audit evidence"):
                trace.TraceBundle(root)

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "trace"
            trace_manifest = manifest()
            with trace.TraceBundleWriter(root, trace_manifest) as writer:
                add_required_events(writer)
            (root / trace_manifest["audits"]["post"]["watchdog"]["path"]).unlink()
            with self.assertRaisesRegex(trace.TraceError, "post watchdog audit evidence"):
                trace.TraceBundle(root)

    def test_rejects_unpinned_ds4_revision(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "trace"
            trace_manifest = manifest("ds4")
            trace_manifest["revision"] = "a" * 40
            with trace.TraceBundleWriter(root, trace_manifest) as writer:
                add_required_events(writer)
            with self.assertRaisesRegex(trace.TraceError, "ds4 revision"):
                trace.TraceBundle(root)

    def test_rejects_wrong_component_schema_and_same_bundle_compare(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "trace"
            with trace.TraceBundleWriter(root, manifest("llama.cpp")) as writer:
                add_required_events(writer)
            events_path = root / trace.EVENTS_NAME
            events = [
                json.loads(line)
                for line in events_path.read_text(encoding="ascii").splitlines()
            ]
            expert = next(event for event in events if event["component"] == "expert.ids")
            expert["dtype"] = "f32"
            events_path.write_text(
                "".join(trace.canonical_json(event) + "\n" for event in events),
                encoding="ascii",
            )
            with self.assertRaisesRegex(trace.TraceError, "expert.ids dtype"):
                trace.TraceBundle(root)

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "trace"
            with trace.TraceBundleWriter(root, manifest("ds4")) as writer:
                add_required_events(writer)
            bundle = trace.TraceBundle(root)
            result = trace.report(bundle, bundle)
            self.assertEqual(result["first_divergence"]["classification"], "artifact_identity")

    def test_rejects_empty_variable_width_components(self) -> None:
        for component in ("attn.source", "attn.candidate_blocks", "attn.candidates"):
            with self.subTest(component=component), tempfile.TemporaryDirectory() as temp:
                writer = trace.TraceBundleWriter(Path(temp) / "trace", manifest())
                with self.assertRaisesRegex(trace.TraceError, "nonzero"):
                    writer.add_event(
                        component=component,
                        phase="prefill",
                        step=0,
                        token_start=0,
                        token_count=2,
                        layer=20,
                        dtype="i32",
                        shape=[0, 2],
                        data=b"",
                    )
                writer.events.close()

    def test_rejects_zero_dimensions_globally(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            writer = trace.TraceBundleWriter(Path(temp) / "trace", manifest())
            with self.assertRaisesRegex(trace.TraceError, "nonzero"):
                writer.add_event(
                    component="prompt.bytes",
                    phase="input",
                    step=0,
                    token_start=0,
                    token_count=1,
                    layer=None,
                    dtype="bytes",
                    shape=[0],
                    data=b"",
                )
            writer.events.close()

    def test_rejects_dirty_ds4_checkout(self) -> None:
        original = run_ds4.git_output
        try:
            run_ds4.git_output = lambda checkout, *args: (
                trace.DS4_REVISION if args == ("rev-parse", "HEAD") else " M runtime.py")
            with self.assertRaisesRegex(preflight.PreflightError, "tracked or untracked"):
                run_ds4.verify_checkout(Path("/tmp/ds4"))
        finally:
            run_ds4.git_output = original

    def test_rejects_weakened_coverage_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "trace"
            weakened = manifest()
            del weakened["expected"]["components"]["expert.ids"]["decode"]
            with trace.TraceBundleWriter(root, weakened) as writer:
                add_required_events(writer)
            with self.assertRaisesRegex(trace.TraceError, "coverage contract"):
                trace.TraceBundle(root)

    def test_rejects_prompt_token_count_not_bound_to_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "trace"
            mismatched = manifest()
            mismatched["expected"]["prompt_tokens"] = 1
            with trace.TraceBundleWriter(root, mismatched) as writer:
                add_required_events(writer)
            with self.assertRaisesRegex(trace.TraceError, "prompt provenance"):
                trace.TraceBundle(root)

    def test_watchdog_stat_parser_handles_parentheses(self) -> None:
        fields = ["S", *[str(value) for value in range(4, 23)]]
        self.assertEqual(preflight.proc_start_time_ticks(f"123 (watch) dog) {' '.join(fields)}"), 22)

    def test_report_generation_passes_identical_bundles(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            left = Path(temp) / "left"
            right = Path(temp) / "right"
            with trace.TraceBundleWriter(left, manifest("ds4")) as writer:
                add_required_events(writer)
            with trace.TraceBundleWriter(right, manifest("llama.cpp")) as writer:
                add_required_events(writer)
            result = trace.report(trace.TraceBundle(left), trace.TraceBundle(right))
            self.assertEqual(result["status"], "TARGET PASS")
            self.assertEqual(result["events_compared"], 259)
            self.assertIsNone(result["first_divergence"])

    def test_manifest_mismatch_is_classified(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            left = Path(temp) / "left"
            right = Path(temp) / "right"
            left_manifest = manifest("ds4")
            right_prompt = b"abd"
            right_manifest = manifest("llama.cpp", right_prompt)
            with trace.TraceBundleWriter(left, left_manifest) as writer:
                add_required_events(writer)
            with trace.TraceBundleWriter(right, right_manifest) as writer:
                add_required_events(writer, prompt=right_prompt)
            result = trace.report(trace.TraceBundle(left), trace.TraceBundle(right))
            self.assertEqual(result["first_divergence"]["classification"], "prompt_identity")

    def test_identically_incomplete_decode_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "trace"
            incomplete = manifest()
            incomplete["config"]["context"] = 4
            incomplete["config"]["decode_steps"] = 2
            incomplete["expected"]["decode_steps"] = 2
            with trace.TraceBundleWriter(root, incomplete) as writer:
                add_required_events(writer)
            with self.assertRaisesRegex(trace.TraceError, "decode step coverage"):
                trace.TraceBundle(root)


if __name__ == "__main__":
    unittest.main()
