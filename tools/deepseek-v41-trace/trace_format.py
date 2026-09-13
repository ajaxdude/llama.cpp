#!/usr/bin/env python3

import argparse
import hashlib
import json
import os
import re
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Iterable

TRACE_FORMAT = "dsv41-trace"
TRACE_VERSION = 1
DS4_REVISION = "bd66c402070042bf0a79ad6ece8242de4c93680c"
MODEL_SHA256 = "1ce6a8f8806205c13330d7ca287bd198331dc5ca35ccc5d8a9a92a188a6f6f42"
REPOSITORY = "halo-box/strix-llama.cpp"
SOFT_MEMORY_LIMIT = 116 * 1024 * 1024 * 1024
CORPUS_SHA256 = {
    "correctness-prose.txt": "2da590a37e3297767336c10b024a0de732d64bee4da5792596f8ddf49ea408d2",
    "correctness-code.txt": "41b4246ef4e6b4e3f9f23a3d02aa8cdab48f495b3af0ebeaccea255679c771f0",
    "correctness-structured.txt": "1278707adea5a953196c4cf5c04de301952813be3eac416a6aac4ff94f42f701",
    "correctness-numeric.txt": "ebd444cf70662cc09289af45ef654af0953b98e967a449d031627d8ea92bc2e0",
}
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
    "prompt.bytes",
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

DEEPSEEK41_EXPECTED_COMPONENTS = {
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
    token_index: int | None = None
    component_element_index: int | None = None

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
        if self.token_index is not None:
            result["token_index"] = self.token_index
        if self.component_element_index is not None:
            result["component_element_index"] = self.component_element_index
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
        if not isinstance(dim, int) or dim <= 0:
            raise TraceError(f"shape dimension must be a nonzero positive integer: {dim!r}")
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
    if event["phase"] not in ("input", "prefill", "decode"):
        raise TraceError("event phase is invalid")
    if not isinstance(event["step"], int) or event["step"] < 0:
        raise TraceError("event step is invalid")
    if not isinstance(event["token_start"], int) or event["token_start"] < 0:
        raise TraceError("event token_start is invalid")
    if not isinstance(event["token_count"], int) or event["token_count"] <= 0:
        raise TraceError("event token_count is invalid")
    if event["layer"] is not None and (not isinstance(event["layer"], int) or event["layer"] < 0):
        raise TraceError("event layer is invalid")
    if not isinstance(event["shape"], list) or not event["shape"] or any(dim <= 0 for dim in event["shape"]):
        raise TraceError("event shape dimensions must be nonzero")
    if event["byte_count"] != expected_bytes(event):
        raise TraceError("event byte_count does not match dtype and shape")
    digest = event["sha256"]
    if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
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
        self._validate_manifest()
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

    def _validate_manifest(self) -> None:
        for key in ("runtime", "revision", "build", "model", "prompt", "config", "comparison", "environment", "audits"):
            if key not in self.manifest:
                raise TraceError(f"manifest is missing {key}")
        if not isinstance(self.manifest["runtime"], str) or not self.manifest["runtime"]:
            raise TraceError("manifest runtime is invalid")
        if self.manifest["runtime"] not in ("ds4", "llama.cpp"):
            raise TraceError("manifest runtime must be ds4 or llama.cpp")
        if not isinstance(self.manifest["revision"], str) or not self.manifest["revision"]:
            raise TraceError("manifest revision is invalid")
        if self.manifest["runtime"] == "ds4" and self.manifest["revision"] != DS4_REVISION:
            raise TraceError(f"ds4 revision must be {DS4_REVISION}")
        if not isinstance(self.manifest["build"], dict):
            raise TraceError("manifest build is invalid")
        build_sha256 = self.manifest["build"].get("sha256", "")
        if not isinstance(build_sha256, str) or re.fullmatch(r"[0-9a-f]{64}", build_sha256) is None:
            raise TraceError("manifest build SHA-256 is invalid")
        for section in ("model", "prompt"):
            if not isinstance(self.manifest[section], dict):
                raise TraceError(f"manifest {section} is invalid")
            digest = self.manifest[section].get("sha256")
            if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
                raise TraceError(f"manifest {section} SHA-256 is invalid")
            if not isinstance(self.manifest[section].get("byte_count"), int) or self.manifest[section]["byte_count"] <= 0:
                raise TraceError(f"manifest {section} byte_count is invalid")
        for section in ("config", "comparison", "environment", "audits"):
            if not isinstance(self.manifest[section], dict):
                raise TraceError(f"manifest {section} is invalid")
        context = self.manifest["config"].get("context")
        decode_steps = self.manifest["config"].get("decode_steps")
        if not isinstance(context, int) or not isinstance(decode_steps, int) or (
                decode_steps <= 0 or context <= decode_steps):
            raise TraceError("manifest context or decode_steps is invalid")
        if self.manifest["model"]["sha256"] != MODEL_SHA256:
            raise TraceError(f"model SHA-256 must be {MODEL_SHA256}")
        if self.manifest["model"].get("architecture") != "deepseek41":
            raise TraceError("model architecture must be deepseek41")
        corpus_name = self.manifest["prompt"].get("corpus_name")
        if corpus_name not in CORPUS_SHA256:
            raise TraceError("prompt corpus is not in the fixed correctness corpus set")
        if self.manifest["prompt"].get("corpus_sha256") != CORPUS_SHA256[corpus_name]:
            raise TraceError(f"prompt corpus SHA-256 is invalid for {corpus_name}")
        provenance = self.manifest["prompt"].get("provenance")
        if not isinstance(provenance, dict):
            raise TraceError("prompt provenance reference is missing")
        provenance_sha256 = provenance.get("sha256", "")
        if not isinstance(provenance_sha256, str) or re.fullmatch(
                r"[0-9a-f]{64}", provenance_sha256) is None:
            raise TraceError("prompt provenance SHA-256 is invalid")
        if provenance.get("path") != f"provenance/{provenance_sha256}.json":
            raise TraceError("prompt provenance path is not content addressed")
        try:
            provenance_bytes = (self.root / provenance["path"]).read_bytes()
            provenance_record = json.loads(provenance_bytes.decode("ascii"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise TraceError(f"cannot read prompt provenance: {error}") from error
        if sha256_bytes(provenance_bytes) != provenance_sha256:
            raise TraceError("prompt provenance SHA-256 mismatch")
        expected_target = context - decode_steps
        provenance_checks = {
            "format": "dsv41-prompt-provenance",
            "version": 1,
            "corpus_name": corpus_name,
            "corpus_sha256": self.manifest["prompt"]["corpus_sha256"],
            "model_sha256": self.manifest["model"]["sha256"],
            "prompt_sha256": self.manifest["prompt"]["sha256"],
            "prompt_byte_count": self.manifest["prompt"]["byte_count"],
            "target_tokens": expected_target,
            "actual_tokens": expected_target,
        }
        for key, value in provenance_checks.items():
            if provenance_record.get(key) != value:
                raise TraceError(f"prompt provenance {key} mismatch")
        if re.fullmatch(r"[0-9a-f]{64}", provenance_record.get("builder_sha256", "")) is None:
            raise TraceError("prompt provenance builder SHA-256 is invalid")
        if self.manifest["prompt"].get("target_tokens") != expected_target:
            raise TraceError("prompt target token count does not fill the configured context")
        if self.manifest["runtime"] == "llama.cpp":
            candidate = self.manifest.get("candidate")
            if not isinstance(candidate, dict):
                raise TraceError("llama.cpp candidate attestation is missing")
            if candidate.get("repository") != REPOSITORY:
                raise TraceError(f"candidate repository must be {REPOSITORY}")
            for key in ("revision", "base_revision", "diff_sha256", "executable_sha256"):
                value = candidate.get(key, "")
                if not isinstance(value, str) or re.fullmatch(
                        r"[0-9a-f]{40}" if "revision" in key else r"[0-9a-f]{64}", value) is None:
                    raise TraceError(f"candidate {key} is invalid")
            if not candidate["revision"].startswith(self.manifest["revision"]):
                raise TraceError("candidate revision does not match the exporter build revision")
            if candidate["executable_sha256"] != self.manifest["build"]["sha256"]:
                raise TraceError("candidate executable SHA-256 does not match the trace build")
        expected_config = {
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
        }
        if self.manifest["config"].get("deepseek41") != expected_config:
            raise TraceError("DeepSeek V4.1 configuration is invalid")
        if self.manifest["comparison"].get("logits") != "byte-identical-f32":
            raise TraceError("logit comparison policy must be byte-identical-f32")
        for audit_phase in ("pre", "post"):
            phase_audits = self.manifest["audits"].get(audit_phase)
            if not isinstance(phase_audits, dict):
                raise TraceError(f"manifest {audit_phase} audit set is invalid")
            for kind in ("memory", "swap", "watchdog"):
                self._validate_audit_reference(audit_phase, kind, phase_audits.get(kind))

    def _validate_audit_reference(self, phase: str, kind: str, audit: Any) -> None:
        if not isinstance(audit, dict):
            raise TraceError(f"manifest {phase} {kind} audit reference is invalid")
        audit_path = audit.get("path")
        if not isinstance(audit_path, str) or not audit_path:
            raise TraceError(f"manifest {phase} {kind} audit path is invalid")
        digest = audit.get("sha256", "")
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise TraceError(f"manifest {phase} {kind} audit SHA-256 is invalid")
        if not isinstance(audit.get("created_unix"), int) or audit["created_unix"] <= 0:
            raise TraceError(f"manifest {phase} {kind} audit timestamp is invalid")
        expected_path = f"audits/{phase}/{digest}.json"
        if audit_path != expected_path:
            raise TraceError(f"manifest {phase} {kind} audit path is not content addressed")
        evidence_path = self.root / audit_path
        try:
            evidence = evidence_path.read_bytes()
        except OSError as error:
            raise TraceError(f"cannot read {phase} {kind} audit evidence: {error}") from error
        if sha256_bytes(evidence) != digest:
            raise TraceError(f"{phase} {kind} audit evidence SHA-256 mismatch")
        try:
            record = json.loads(evidence.decode("ascii"))
        except (UnicodeError, json.JSONDecodeError) as error:
            raise TraceError(f"{phase} {kind} audit evidence is invalid: {error}") from error
        if record.get("kind") != kind or record.get("created_unix") != audit["created_unix"]:
            raise TraceError(f"{phase} {kind} audit evidence metadata mismatch")
        if not isinstance(record.get("data"), dict):
            raise TraceError(f"{phase} {kind} audit evidence data is invalid")
        if kind == "memory":
            used = record["data"].get("mem_used_bytes")
            if not isinstance(used, int) or used < 0 or used >= SOFT_MEMORY_LIMIT:
                raise TraceError(f"{phase} memory audit evidence is invalid")
        if kind == "swap":
            if record["data"].get("enabled") is not False or record["data"].get("entries") != []:
                raise TraceError(f"{phase} swap audit evidence does not report zero configured swap")
        if kind == "watchdog":
            required = ("pid", "start_time_ticks", "command_sha256", "heartbeat_path", "heartbeat_unix")
            if any(key not in record["data"] for key in required):
                raise TraceError(f"{phase} watchdog audit evidence is incomplete")
            data = record["data"]
            if not isinstance(data["pid"], int) or data["pid"] <= 1:
                raise TraceError(f"{phase} watchdog audit PID is invalid")
            if not isinstance(data["start_time_ticks"], int) or data["start_time_ticks"] <= 0:
                raise TraceError(f"{phase} watchdog audit start time is invalid")
            if not isinstance(data["command_sha256"], str) or re.fullmatch(
                    r"[0-9a-f]{64}", data["command_sha256"]) is None:
                raise TraceError(f"{phase} watchdog audit command SHA-256 is invalid")
            if not isinstance(data["heartbeat_path"], str) or not data["heartbeat_path"]:
                raise TraceError(f"{phase} watchdog audit heartbeat path is invalid")
            if not isinstance(data["heartbeat_unix"], int) or data["heartbeat_unix"] <= 0:
                raise TraceError(f"{phase} watchdog audit heartbeat timestamp is invalid")
            max_age = data.get("max_heartbeat_age_seconds")
            if not isinstance(max_age, int) or max_age <= 0 or max_age > 30:
                raise TraceError(f"{phase} watchdog audit heartbeat age is invalid")
            if data["heartbeat_unix"] > record["created_unix"] or (
                    record["created_unix"] - data["heartbeat_unix"] > max_age):
                raise TraceError(f"{phase} watchdog audit heartbeat was stale when captured")

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
        if prompt_tokens != self.manifest.get("prompt", {}).get("target_tokens"):
            raise TraceError("expected prompt_tokens does not match prompt provenance")
        for event in self.events:
            self._validate_component_schema(event)
        if not isinstance(components, dict):
            raise TraceError("expected components are invalid")
        if self.manifest.get("model", {}).get("architecture") == "deepseek41":
            if components != DEEPSEEK41_EXPECTED_COMPONENTS:
                raise TraceError("DeepSeek V4.1 expected component coverage contract is invalid")

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

    def _validate_component_schema(self, event: dict[str, Any]) -> None:
        component = event["component"]
        phase = event["phase"]
        layer = event["layer"]
        dtype = event["dtype"]
        shape = event["shape"]
        token_count = event["token_count"]
        config = self.manifest["config"].get("deepseek41")
        if not isinstance(config, dict):
            raise TraceError("DeepSeek V4.1 component schema configuration is missing")

        if component == "prompt.bytes":
            if phase != "input" or layer is not None or dtype != "bytes" or len(shape) != 1:
                raise TraceError("prompt.bytes schema is invalid")
            if shape[0] != self.manifest["prompt"]["byte_count"]:
                raise TraceError("prompt.bytes length does not match the manifest")
            if event["sha256"] != self.manifest["prompt"]["sha256"]:
                raise TraceError("prompt.bytes SHA-256 does not match the manifest")
            return
        if component == "prompt.tokens":
            if phase != "input" or layer is not None or dtype != "i32" or shape != [token_count]:
                raise TraceError("prompt.tokens schema is invalid")
            return
        if component in ("logits.prefill", "logits.decode"):
            if phase not in ("prefill", "decode") or layer is not None or dtype != "f32":
                raise TraceError(f"{component} schema is invalid")
            if token_count != 1 or shape != [config.get("vocab_size")]:
                raise TraceError(f"{component} must contain one complete vocabulary-sized logit vector")
            return
        if component == "decode.greedy_token":
            if phase != "decode" or layer is not None or dtype != "i32" or token_count != 1 or shape != [1]:
                raise TraceError("decode.greedy_token schema is invalid")
            return

        if phase not in ("prefill", "decode") or layer is None:
            raise TraceError(f"{component} phase or layer is invalid")
        if len(shape) != 2 or shape[1] != token_count:
            raise TraceError(f"{component} second dimension must equal token_count")
        widths = {
            "engram.row_ids": ("i32", config.get("engram_rows_per_token")),
            "expert.ids": ("i32", config.get("experts_used")),
            "expert.weights": ("f32", config.get("experts_used")),
            "attn.source": ("i32", None),
            "attn.candidate_blocks": ("i32", None),
            "attn.candidates": ("i32", None),
        }
        if component not in widths:
            raise TraceError(f"unsupported trace component: {component}")
        expected_dtype, width = widths[component]
        if dtype != expected_dtype:
            raise TraceError(f"{component} dtype must be {expected_dtype}")
        if width is not None and shape[0] != width:
            raise TraceError(f"{component} shape must be [{width},token_count]")
        if component == "attn.candidate_blocks" and shape[0] > config.get("candidate_topk_blocks", 0):
            raise TraceError("attn.candidate_blocks width exceeds candidate_topk_blocks")
        if component in ("attn.source", "attn.candidates") and shape[0] > config.get("index_top_k", 0):
            raise TraceError(f"{component} width exceeds index_top_k")
        if component == "expert.ids":
            values = struct.iter_unpack("<i", self.read_blob(event))
            if any(value < 0 or value >= config.get("expert_count", 0) for value, in values):
                raise TraceError("expert.ids contains an out-of-range original expert ID")


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
        ("expected.prompt_tokens", "tokenizer"),
        ("config.context", "configuration"),
        ("config.decode_steps", "configuration"),
        ("config.deepseek41", "configuration"),
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
    if left.root.resolve() == right.root.resolve():
        return Mismatch(
            "artifact_identity",
            "manifest",
            "metadata",
            -1,
            -1,
            None,
            "cannot compare a trace bundle with itself",
        )
    if left.manifest["runtime"] != "ds4" or right.manifest["runtime"] != "llama.cpp":
        return Mismatch(
            "runtime_role",
            "manifest",
            "metadata",
            -1,
            -1,
            None,
            "left trace must be pinned ds4 and right trace must be llama.cpp candidate",
        )
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
    mismatches = []
    all_keys = sorted(set(left_map) | set(right_map), key=event_order)
    for key in all_keys:
        left_event = left_map.get(key)
        right_event = right_map.get(key)
        template = left_event or right_event
        assert template is not None
        if left_event is None or right_event is None:
            mismatches.append(Mismatch(
                "artifact_missing",
                template["component"],
                template["phase"],
                template["step"],
                template["token_start"],
                template["layer"],
                "event is missing from " + ("left" if left_event is None else "right") + " trace",
                token_index=template["token_start"],
            ))
            continue
        shape_mismatch = False
        for field in ("dtype", "shape", "token_count", "semantic_id_space"):
            if left_event.get(field) != right_event.get(field):
                mismatches.append(Mismatch(
                    "artifact_shape",
                    template["component"],
                    template["phase"],
                    template["step"],
                    template["token_start"],
                    template["layer"],
                    f"{field} differs: {left_event.get(field)!r} != {right_event.get(field)!r}",
                    token_index=template["token_start"],
                ))
                shape_mismatch = True
                break
        if shape_mismatch:
            continue
        if left_event["sha256"] == right_event["sha256"]:
            continue
        left_data = left.read_blob(left_event)
        right_data = right.read_blob(right_event)
        byte_offset = first_byte_difference(left_data, right_data)
        assert byte_offset is not None
        item_size = DTYPE_SIZES[left_event["dtype"]]
        flat_element_index = byte_offset // item_size
        token_index = None
        component_element_index = None
        token_count = left_event["token_count"]
        elements = element_count(left_event["shape"])
        if token_count > 0 and elements % token_count == 0:
            elements_per_token = elements // token_count
            token_index = left_event["token_start"] + flat_element_index // elements_per_token
            component_element_index = flat_element_index % elements_per_token
        detail = f"first byte mismatch at {byte_offset}"
        item_offset = byte_offset - byte_offset % item_size
        if item_offset + item_size <= min(len(left_data), len(right_data)):
            left_item = left_data[item_offset:item_offset + item_size]
            right_item = right_data[item_offset:item_offset + item_size]
            detail += f"; left=0x{left_item.hex()} right=0x{right_item.hex()}"
        mismatches.append(Mismatch(
            classify(template["component"]),
            template["component"],
            template["phase"],
            template["step"],
            template["token_start"],
            template["layer"],
            detail,
            element_index=flat_element_index,
            byte_offset=byte_offset,
            token_index=token_index,
            component_element_index=component_element_index,
        ))
    if not mismatches:
        return None
    phase_order = {"input": 0, "prefill": 1, "decode": 2, "metadata": -1}
    component_order = {component: index for index, component in enumerate(HARD_FAILURE_COMPONENTS)}
    return min(mismatches, key=lambda item: (
        phase_order.get(item.phase, 99),
        item.token_index if item.token_index is not None else item.token_start,
        item.step,
        -1 if item.layer is None else item.layer,
        component_order.get(item.component, 99),
        item.component,
    ))


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
