#!/usr/bin/env python3

import importlib.util
import json
import struct
import tempfile
import unittest
from pathlib import Path

MODULE_PATH = Path(__file__).parents[1] / "tools" / "deepseek-v41-trace" / "trace_format.py"
SPEC = importlib.util.spec_from_file_location("dsv41_trace_format", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
trace = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(trace)


def manifest(runtime: str = "test") -> dict:
    return {
        "runtime": runtime,
        "revision": "a" * 40,
        "build": {"sha256": "3" * 64},
        "model": {"sha256": "1" * 64, "byte_count": 123},
        "prompt": {"sha256": "2" * 64, "byte_count": 3},
        "config": {
            "context": 32768,
            "decode_steps": 1,
            "batch": 512,
            "ubatch": 128,
            "kv_type_k": "f16",
            "kv_type_v": "f16",
            "flash_attention": True,
            "expert_cache_slots": 8,
            "expert_cache_bytes": 4096,
        },
        "comparison": {"logits": "byte-identical-f32"},
        "expected": {
            "prompt_tokens": 2,
            "decode_steps": 1,
            "components": {
                "prompt.tokens": {"layers": None, "input": "tokens"},
                "engram.row_ids": {"layers": [1], "prefill": "tokens", "decode": "steps"},
                "expert.ids": {"layers": [0], "prefill": "tokens", "decode": "steps"},
                "expert.weights": {"layers": [0], "prefill": "tokens", "decode": "steps"},
                "attn.source": {"layers": [20], "prefill": "tokens", "decode": "steps"},
                "attn.candidate_blocks": {"layers": [20], "prefill": "tokens", "decode": "steps"},
                "attn.candidates": {"layers": [24], "prefill": "tokens", "decode": "steps"},
                "logits.prefill": {"layers": None, "prefill": "final"},
                "logits.decode": {"layers": None, "decode": "steps"},
                "decode.greedy_token": {"layers": None, "decode": "steps"},
            },
        },
        "environment": {},
        "audits": {
            "memory": {"path": "memory.json", "sha256": "4" * 64, "created_unix": 1},
            "swap": {"path": "swap.json", "sha256": "5" * 64, "created_unix": 1},
            "watchdog": {"path": "watchdog.json", "sha256": "6" * 64, "created_unix": 1},
        },
    }


def add_required_events(writer: object, logits: bytes | None = None) -> None:
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
        dtype="u32",
        shape=[2, 4],
        data=struct.pack("<IIIIIIII", 1, 2, 3, 4, 5, 6, 7, 8),
    )
    writer.add_event(
        component="expert.ids",
        phase="prefill",
        step=0,
        token_start=0,
        token_count=2,
        layer=0,
        dtype="i32",
        shape=[2, 6],
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
        shape=[2, 6],
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
        shape=[2],
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
        shape=[2, 2],
        data=struct.pack("<iiii", 4, 7, 4, 7),
    )
    writer.add_event(
        component="logits.prefill",
        phase="prefill",
        step=0,
        token_start=1,
        token_count=1,
        layer=None,
        dtype="f32",
        shape=[4],
        data=logits if logits is not None else struct.pack("<IIII", 0x3F800000, 0x80000000, 0x7FC12345, 0),
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
        layer=1, dtype="u32", shape=[1, 4], data=struct.pack("<IIII", 9, 10, 11, 12))
    writer.add_event(
        component="expert.ids", phase="decode", step=0, token_start=2, token_count=1,
        layer=0, dtype="i32", shape=[1, 6], data=struct.pack("<iiiiii", 1, 2, 3, 4, 5, 6),
        semantic_id_space="original")
    writer.add_event(
        component="expert.weights", phase="decode", step=0, token_start=2, token_count=1,
        layer=0, dtype="f32", shape=[1, 6], data=struct.pack("<ffffff", 1, 2, 3, 4, 5, 6))
    writer.add_event(
        component="attn.source", phase="decode", step=0, token_start=2, token_count=1,
        layer=20, dtype="i32", shape=[1], data=struct.pack("<i", 20))
    writer.add_event(
        component="attn.candidate_blocks", phase="decode", step=0, token_start=2, token_count=1,
        layer=20, dtype="i32", shape=[2], data=struct.pack("<ii", 4, 7))
    writer.add_event(
        component="attn.candidates", phase="decode", step=0, token_start=2, token_count=1,
        layer=24, dtype="i32", shape=[2], data=struct.pack("<ii", 4, 7))
    writer.add_event(
        component="logits.decode",
        phase="decode",
        step=0,
        token_start=2,
        token_count=1,
        layer=None,
        dtype="f32",
        shape=[4],
        data=logits if logits is not None else struct.pack("<IIII", 0x3F800000, 0x80000000, 0x7FC12345, 0),
    )


class TraceFormatTests(unittest.TestCase):
    def test_serialization_preserves_float_bits_and_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "trace"
            bits = (0x3F800000, 0x80000000, 0x7FC12345, 0)
            data = struct.pack("<IIII", *bits)
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
            with trace.TraceBundleWriter(left, manifest("llama.cpp")) as writer:
                add_required_events(writer)
            with trace.TraceBundleWriter(right, manifest("ds4")) as writer:
                add_required_events(writer)
            right_bundle = trace.TraceBundle(right)
            expert = next(item for item in right_bundle.events if item["component"] == "expert.ids")
            expert_blob = right / expert["blob"]
            expert_blob.write_bytes(struct.pack(
                "<iiiiiiiiiiii", 1, 2, 9, 4, 5, 6, 1, 2, 3, 4, 5, 6))
            expert["sha256"] = trace.sha256_file(expert_blob)
            expert["blob"] = f"blobs/{expert['sha256']}.bin"
            new_blob = right / expert["blob"]
            expert_blob.rename(new_blob)
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
            self.assertEqual(divergence["element_index"], 2)

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
            bad_manifest["audits"]["watchdog"] = "watchdog.json"
            with trace.TraceBundleWriter(root, bad_manifest) as writer:
                add_required_events(writer)
            with self.assertRaisesRegex(trace.TraceError, "watchdog audit reference"):
                trace.TraceBundle(root)

    def test_report_generation_passes_identical_bundles(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            left = Path(temp) / "left"
            right = Path(temp) / "right"
            with trace.TraceBundleWriter(left, manifest("llama.cpp")) as writer:
                add_required_events(writer)
            with trace.TraceBundleWriter(right, manifest("ds4")) as writer:
                add_required_events(writer)
            result = trace.report(trace.TraceBundle(left), trace.TraceBundle(right))
            self.assertEqual(result["status"], "TARGET PASS")
            self.assertEqual(result["events_compared"], 16)
            self.assertIsNone(result["first_divergence"])

    def test_manifest_mismatch_is_classified(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            left = Path(temp) / "left"
            right = Path(temp) / "right"
            left_manifest = manifest("llama.cpp")
            right_manifest = manifest("ds4")
            right_manifest["prompt"]["sha256"] = "3" * 64
            with trace.TraceBundleWriter(left, left_manifest) as writer:
                add_required_events(writer)
            with trace.TraceBundleWriter(right, right_manifest) as writer:
                add_required_events(writer)
            result = trace.report(trace.TraceBundle(left), trace.TraceBundle(right))
            self.assertEqual(result["first_divergence"]["classification"], "prompt_identity")

    def test_identically_incomplete_decode_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "trace"
            incomplete = manifest()
            incomplete["config"]["decode_steps"] = 2
            incomplete["expected"]["decode_steps"] = 2
            with trace.TraceBundleWriter(root, incomplete) as writer:
                add_required_events(writer)
            with self.assertRaisesRegex(trace.TraceError, "decode step coverage"):
                trace.TraceBundle(root)


if __name__ == "__main__":
    unittest.main()
