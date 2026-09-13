#!/usr/bin/env python3

import argparse
import hashlib
import json
import os
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Iterable

TRACE_FORMAT = "dsv41-trace"
TRACE_VERSION = 1
MANIFEST_NAME = "manifest.json"
EVENTS_NAME = "events.jsonl"
BLOBS_DIR = "blobs"

DTYPE_SIZES = {
    "f32": 4,
    "bf16": 2,
    "i32": 4,
    "u32": 4,
    "i8": 1,
    "u8": 1,
    "bytes": 1,
}

HARD_FAILURE_COMPONENTS = (
    "prompt.tokens",
    "engram.row_ids",
    "expert.ids",
    "expert.weights",
    "attn.source",
    "attn.candidate_blocks",
    "attn.candidates",
    "logits.prefill",
    "logits.decode",
    "decode.greedy_token",
)

DEEPSEEK41_LAYERS = {
    "engram.row_ids": [1, 14],
    "expert.ids": list(range(40)),
    "expert.weights": list(range(40)),
    "attn.source": list(range(40)),
    "attn.candidate_blocks": [20],
    "attn.candidates": [24, 28, 32, 36],
}


class TraceError(RuntimeError):
    pass


@dataclass(frozen=True)
class Mismatch:
    classification: str
    component: str
    phase: str
    step: int
    token_start: int
    layer: int | None
    detail: str
    element_index: int | None = None
    byte_offset: int | None = None

    def as_dict(self) -> dict[str, Any]:
        result = {
            "classification": self.classification,
            "component": self.component,
            "phase": self.phase,
            "step": self.step,
            "token_start": self.token_start,
            "layer": self.layer,
            "detail": self.detail,
        }
        if self.element_index is not None:
            result["element_index"] = self.element_index
        if self.byte_offset is not None:
            result["byte_offset"] = self.byte_offset
        return result


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def element_count(shape: Iterable[int]) -> int:
    count = 1
    for dim in shape:
        if not isinstance(dim, int) or dim < 0:
            raise TraceError(f"invalid shape dimension: {dim!r}")
        count *= dim
    return count


def expected_bytes(event: dict[str, Any]) -> int:
    dtype = event.get("dtype")
    if dtype not in DTYPE_SIZES:
        raise TraceError(f"unsupported dtype: {dtype!r}")
    shape = event.get("shape")
    if not isinstance(shape, list):
        raise TraceError("event shape must be an array")
    return element_count(shape) * DTYPE_SIZES[dtype]


def event_key(event: dict[str, Any]) -> tuple[Any, ...]:
    return (
        event.get("phase", ""),
        int(event.get("step", -1)),
        int(event.get("token_start", -1)),
        event.get("layer"),
        event.get("component", ""),
    )


def event_order(key: tuple[Any, ...]) -> tuple[Any, ...]:
    phase_order = {"input": 0, "prefill": 1, "decode": 2}
    component_order = {
        "prompt.bytes": 0,
        "prompt.tokens": 1,
        "engram.row_ids": 2,
        "expert.ids": 3,
        "expert.weights": 4,
        "attn.source": 5,
        "attn.candidate_blocks": 6,
        "attn.candidates": 7,
        "decode.greedy_token": 8,
        "logits.prefill": 9,
        "logits.decode": 10,
    }
    return (
        phase_order.get(key[0], 99),
        key[2],
        key[1],
        -1 if key[3] is None else key[3],
        component_order.get(key[4], 99),
        key[4],
    )


def classify(component: str) -> str:
    if component == "prompt.tokens":
        return "tokenizer"
    if component == "engram.row_ids":
        return "engram_row"
    if component == "expert.ids":
        return "routing_original_expert"
    if component == "expert.weights":
        return "routing_weight"
    if component.startswith("attn.candidate"):
        return "attention_candidate"
    if component == "attn.source":
        return "attention_source"
    if component == "logits.prefill":
        return "prefill_logits"
    if component == "logits.decode":
        return "decode_logits"
    if component == "decode.greedy_token":
        return "decode_token"
    return "trace_data"


def validate_event(event: dict[str, Any]) -> None:
    required = {
        "trace_version",
        "component",
        "phase",
        "step",
        "token_start",
        "token_count",
        "layer",
        "dtype",
        "shape",
        "byte_order",
        "byte_count",
        "sha256",
        "blob",
    }
    missing = sorted(required - event.keys())
    if missing:
        raise TraceError(f"event is missing fields: {', '.join(missing)}")
    if event["trace_version"] != TRACE_VERSION:
        raise TraceError(f"unsupported event version: {event['trace_version']!r}")
    if event["byte_order"] != "little":
        raise TraceError("trace blobs must use little-endian byte order")
    if event["byte_count"] != expected_bytes(event):
        raise TraceError("event byte_count does not match dtype and shape")
    digest = event["sha256"]
    if not isinstance(digest, str) or len(digest) != 64:
        raise TraceError("event sha256 is invalid")
    if event["blob"] != f"{BLOBS_DIR}/{digest}.bin":
        raise TraceError("event blob path is not content addressed")
    component = event["component"]
    if component == "expert.ids":
        if event.get("semantic_id_space") != "original":
            raise TraceError("expert.ids must contain original expert IDs, not cache slot IDs")
    if "slot" in component or event.get("semantic_id_space") == "cache_slot":
        raise TraceError("cache slot IDs are forbidden in correctness traces")


class TraceBundleWriter:
    def __init__(self, root: Path, manifest: dict[str, Any]):
        self.root = root
        self.blobs = root / BLOBS_DIR
        if root.exists() and any(root.iterdir()):
            raise TraceError(f"trace output directory is not empty: {root}")
        self.blobs.mkdir(parents=True, exist_ok=True)
        self.events = (root / EVENTS_NAME).open("x", encoding="ascii", newline="\n")
        self.manifest = dict(manifest)
        self.manifest["trace_format"] = TRACE_FORMAT
        self.manifest["trace_version"] = TRACE_VERSION
        self.event_count = 0

    def add_event(
        self,
        *,
        component: str,
        phase: str,
        step: int,
        token_start: int,
        token_count: int,
        layer: int | None,
        dtype: str,
        shape: list[int],
        data: bytes,
        semantic_id_space: str | None = None,
    ) -> dict[str, Any]:
        digest = sha256_bytes(data)
        blob_rel = f"{BLOBS_DIR}/{digest}.bin"
        event = {
            "trace_version": TRACE_VERSION,
            "component": component,
            "phase": phase,
            "step": step,
            "token_start": token_start,
            "token_count": token_count,
            "layer": layer,
            "dtype": dtype,
            "shape": shape,
            "byte_order": "little",
            "byte_count": len(data),
            "sha256": digest,
            "blob": blob_rel,
        }
        if semantic_id_space is not None:
            event["semantic_id_space"] = semantic_id_space
        validate_event(event)
        blob_path = self.root / blob_rel
        if not blob_path.exists():
            temp = blob_path.with_suffix(".tmp")
            temp.write_bytes(data)
            os.replace(temp, blob_path)
        self.events.write(canonical_json(event) + "\n")
        self.events.flush()
        self.event_count += 1
        return event

    def close(self) -> None:
        if self.events.closed:
            return
        self.events.close()
        self.manifest["event_count"] = self.event_count
        manifest_path = self.root / MANIFEST_NAME
        temp = manifest_path.with_suffix(".tmp")
        temp.write_text(canonical_json(self.manifest) + "\n", encoding="ascii")
        os.replace(temp, manifest_path)

    def __enter__(self) -> "TraceBundleWriter":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.events.close()
        if exc_type is None:
            self.manifest["event_count"] = self.event_count
            manifest_path = self.root / MANIFEST_NAME
            temp = manifest_path.with_suffix(".tmp")
            temp.write_text(canonical_json(self.manifest) + "\n", encoding="ascii")
            os.replace(temp, manifest_path)


class TraceBundle:
    def __init__(self, root: Path, verify_blobs: bool = True):
        self.root = root
        try:
            self.manifest = json.loads((root / MANIFEST_NAME).read_text(encoding="ascii"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise TraceError(f"cannot read manifest: {error}") from error
        if self.manifest.get("trace_format") != TRACE_FORMAT:
            raise TraceError("manifest trace_format mismatch")
        if self.manifest.get("trace_version") != TRACE_VERSION:
            raise TraceError("manifest trace_version mismatch")
        self.events = self._read_events(verify_blobs)
        if self.manifest.get("event_count") != len(self.events):
            raise TraceError("manifest event_count mismatch")
        self._validate_coverage()

    def _read_events(self, verify_blobs: bool) -> list[dict[str, Any]]:
        result = []
        try:
            stream: BinaryIO
            with (self.root / EVENTS_NAME).open("rb") as stream:
                for line_number, raw in enumerate(stream, 1):
                    if not raw.endswith(b"\n"):
                        raise TraceError(f"events.jsonl is truncated at line {line_number}")
                    try:
                        event = json.loads(raw.decode("ascii"))
                    except (UnicodeError, json.JSONDecodeError) as error:
                        raise TraceError(f"invalid event at line {line_number}: {error}") from error
                    validate_event(event)
                    if verify_blobs:
                        data = self.read_blob(event)
                        if len(data) != event["byte_count"]:
                            raise TraceError(f"truncated blob for event line {line_number}")
                        if sha256_bytes(data) != event["sha256"]:
                            raise TraceError(f"corrupt blob for event line {line_number}")
                    result.append(event)
        except OSError as error:
            raise TraceError(f"cannot read events: {error}") from error
        return result

    def read_blob(self, event: dict[str, Any]) -> bytes:
        try:
            return (self.root / event["blob"]).read_bytes()
        except OSError as error:
            raise TraceError(f"cannot read blob {event['blob']}: {error}") from error

    def _validate_coverage(self) -> None:
        expected = self.manifest.get("expected")
        if not isinstance(expected, dict):
            raise TraceError("manifest expected coverage is missing")
        prompt_tokens = expected.get("prompt_tokens")
        decode_steps = expected.get("decode_steps")
        components = expected.get("components")
        if not isinstance(prompt_tokens, int) or prompt_tokens <= 0:
            raise TraceError("expected prompt_tokens is invalid")
        if not isinstance(decode_steps, int) or decode_steps <= 0:
            raise TraceError("expected decode_steps is invalid")
        if self.manifest.get("config", {}).get("decode_steps") != decode_steps:
            raise TraceError("expected decode_steps does not match config")
        if not isinstance(components, dict):
            raise TraceError("expected components are invalid")
        if self.manifest.get("model", {}).get("architecture") == "deepseek41":
            for component, layers in DEEPSEEK41_LAYERS.items():
                if components.get(component, {}).get("layers") != layers:
                    raise TraceError(f"DeepSeek V4.1 expected layers are invalid for {component}")

        by_component: dict[str, list[dict[str, Any]]] = {}
        for event in self.events:
            by_component.setdefault(event["component"], []).append(event)
        for component in HARD_FAILURE_COMPONENTS:
            if component not in components:
                raise TraceError(f"expected coverage lacks {component}")
            rules = components[component]
            events = by_component.get(component, [])
            if not events:
                raise TraceError(f"trace lacks required component: {component}")
            expected_layers = rules.get("layers")
            if expected_layers is not None:
                actual_layers = sorted({event["layer"] for event in events})
                if actual_layers != expected_layers:
                    raise TraceError(
                        f"{component} layer coverage mismatch: expected {expected_layers}, found {actual_layers}")
            input_coverage = rules.get("input")
            prefill_coverage = rules.get("prefill")
            decode_coverage = rules.get("decode")
            if input_coverage == "tokens":
                input_events = [event for event in events if event["phase"] == "input"]
                if len(input_events) != 1 or input_events[0]["token_start"] != 0 or input_events[0]["token_count"] != prompt_tokens:
                    raise TraceError(f"{component} input coverage is incomplete")
            for layer in expected_layers or [None]:
                layer_events = [event for event in events if event["layer"] == layer]
                if prefill_coverage == "tokens":
                    prefill = sorted(
                        (event for event in layer_events if event["phase"] == "prefill"),
                        key=lambda event: event["token_start"],
                    )
                    frontier = 0
                    for event in prefill:
                        if event["token_start"] != frontier or event["token_count"] <= 0:
                            raise TraceError(f"{component} prefill coverage has a gap or overlap at token {frontier}")
                        frontier += event["token_count"]
                    if frontier != prompt_tokens:
                        raise TraceError(
                            f"{component} prefill coverage ends at {frontier}, expected {prompt_tokens}")
                elif prefill_coverage == "final":
                    prefill = [event for event in layer_events if event["phase"] == "prefill"]
                    if len(prefill) != 1 or prefill[0]["token_start"] != prompt_tokens - 1 or prefill[0]["token_count"] != 1:
                        raise TraceError(f"{component} final prefill coverage is invalid")
                if decode_coverage == "steps":
                    decode = sorted(
                        (event for event in layer_events if event["phase"] == "decode"),
                        key=lambda event: event["step"],
                    )
                    if [event["step"] for event in decode] != list(range(decode_steps)):
                        raise TraceError(f"{component} decode step coverage is incomplete")
                    for event in decode:
                        if event["token_start"] != prompt_tokens + event["step"] or event["token_count"] != 1:
                            raise TraceError(f"{component} decode token coordinates are invalid")


def first_byte_difference(left: bytes, right: bytes) -> int | None:
    for index, (a, b) in enumerate(zip(left, right)):
        if a != b:
            return index
    if len(left) != len(right):
        return min(len(left), len(right))
    return None


def compare_manifests(left: TraceBundle, right: TraceBundle) -> Mismatch | None:
    checks = (
        ("model.sha256", "model_identity"),
        ("model.architecture", "model_identity"),
        ("prompt.sha256", "prompt_identity"),
        ("prompt.byte_count", "prompt_identity"),
        ("config.context", "configuration"),
        ("config.decode_steps", "configuration"),
        ("comparison.logits", "comparison_policy"),
    )
    for dotted, classification in checks:
        left_value: Any = left.manifest
        right_value: Any = right.manifest
        for key in dotted.split("."):
            left_value = left_value.get(key) if isinstance(left_value, dict) else None
            right_value = right_value.get(key) if isinstance(right_value, dict) else None
        if left_value != right_value:
            return Mismatch(
                classification,
                "manifest",
                "metadata",
                -1,
                -1,
                None,
                f"{dotted} differs: {left_value!r} != {right_value!r}",
            )
    return None


def compare_bundles(left: TraceBundle, right: TraceBundle) -> Mismatch | None:
    mismatch = compare_manifests(left, right)
    if mismatch is not None:
        return mismatch

    for bundle_name, bundle in (("left", left), ("right", right)):
        components = {event["component"] for event in bundle.events}
        missing = sorted(set(HARD_FAILURE_COMPONENTS) - components)
        if missing:
            return Mismatch(
                "artifact_missing",
                "manifest",
                "metadata",
                -1,
                -1,
                None,
                f"{bundle_name} trace is missing required components: {', '.join(missing)}",
            )

    left_map = {event_key(event): event for event in left.events}
    right_map = {event_key(event): event for event in right.events}
    if len(left_map) != len(left.events) or len(right_map) != len(right.events):
        return Mismatch(
            "artifact_duplicate",
            "events",
            "metadata",
            -1,
            -1,
            None,
            "a trace contains duplicate phase/step/token/layer/component coordinates",
        )
    all_keys = sorted(set(left_map) | set(right_map), key=event_order)
    for key in all_keys:
        left_event = left_map.get(key)
        right_event = right_map.get(key)
        template = left_event or right_event
        assert template is not None
        if left_event is None or right_event is None:
            return Mismatch(
                "artifact_missing",
                template["component"],
                template["phase"],
                template["step"],
                template["token_start"],
                template["layer"],
                "event is missing from " + ("left" if left_event is None else "right") + " trace",
            )
        for field in ("dtype", "shape", "token_count", "semantic_id_space"):
            if left_event.get(field) != right_event.get(field):
                return Mismatch(
                    "artifact_shape",
                    template["component"],
                    template["phase"],
                    template["step"],
                    template["token_start"],
                    template["layer"],
                    f"{field} differs: {left_event.get(field)!r} != {right_event.get(field)!r}",
                )
        if left_event["sha256"] == right_event["sha256"]:
            continue
        left_data = left.read_blob(left_event)
        right_data = right.read_blob(right_event)
        byte_offset = first_byte_difference(left_data, right_data)
        assert byte_offset is not None
        item_size = DTYPE_SIZES[left_event["dtype"]]
        detail = f"first byte mismatch at {byte_offset}"
        item_offset = byte_offset - byte_offset % item_size
        if item_offset + item_size <= min(len(left_data), len(right_data)):
            left_item = left_data[item_offset:item_offset + item_size]
            right_item = right_data[item_offset:item_offset + item_size]
            detail += f"; left=0x{left_item.hex()} right=0x{right_item.hex()}"
        return Mismatch(
            classify(template["component"]),
            template["component"],
            template["phase"],
            template["step"],
            template["token_start"],
            template["layer"],
            detail,
            element_index=byte_offset // item_size,
            byte_offset=byte_offset,
        )
    return None


def report(left: TraceBundle, right: TraceBundle) -> dict[str, Any]:
    mismatch = compare_bundles(left, right)
    if mismatch is None:
        return {
            "status": "TARGET PASS",
            "trace_version": TRACE_VERSION,
            "left_runtime": left.manifest.get("runtime"),
            "right_runtime": right.manifest.get("runtime"),
            "events_compared": len(left.events),
            "first_divergence": None,
        }
    return {
        "status": "FAIL",
        "trace_version": TRACE_VERSION,
        "left_runtime": left.manifest.get("runtime"),
        "right_runtime": right.manifest.get("runtime"),
        "events_compared": 0,
        "first_divergence": mismatch.as_dict(),
    }


def command_validate(args: argparse.Namespace) -> int:
    bundle = TraceBundle(args.bundle)
    print(canonical_json({
        "status": "valid",
        "runtime": bundle.manifest.get("runtime"),
        "events": len(bundle.events),
    }))
    return 0


def command_compare(args: argparse.Namespace) -> int:
    result = report(TraceBundle(args.left), TraceBundle(args.right))
    text = canonical_json(result) + "\n"
    if args.report:
        args.report.write_text(text, encoding="ascii")
    sys.stdout.write(text)
    return 0 if result["status"] == "TARGET PASS" else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate and compare DeepSeek V4.1 correctness traces")
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("bundle", type=Path)
    validate_parser.set_defaults(func=command_validate)
    compare_parser = subparsers.add_parser("compare")
    compare_parser.add_argument("left", type=Path)
    compare_parser.add_argument("right", type=Path)
    compare_parser.add_argument("--report", type=Path)
    compare_parser.set_defaults(func=command_compare)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
