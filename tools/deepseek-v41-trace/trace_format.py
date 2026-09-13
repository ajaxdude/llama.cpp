#!/usr/bin/env python3

import argparse
import hashlib
import json
import math
import os
import re
import struct
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, Iterable

TRACE_FORMAT = "dsv41-trace"
TRACE_VERSION = 2
DS4_REVISION = "bd66c402070042bf0a79ad6ece8242de4c93680c"
APPROVED_EXPORTERS: dict[str, str] = {}
MODEL_SHA256 = "1ce6a8f8806205c13330d7ca287bd198331dc5ca35ccc5d8a9a92a188a6f6f42"
REPOSITORY = "halo-box/strix-llama.cpp"
SOFT_MEMORY_LIMIT = 116 * 1024 * 1024 * 1024
WATCHDOG_EMERGENCY_LIMIT = 118 * 1024 * 1024 * 1024
STRICT_MEMORY_LIMIT = 120 * 1024 * 1024 * 1024
WATCHDOG_LEASE_FORMAT = "strix-memory-watchdog-lease"
WATCHDOG_VERSION = 2
WATCHDOG_REVISION = "778db6f50eae04e6c232c69b9575bdbd0747962b"
WATCHDOG_SCRIPT_SHA256 = "d2781a25f978dd2bc14fc113079aa2dbf513aa157b44da9d0d51d750daa6c94f"
APPROVED_WATCHDOGS = {WATCHDOG_SCRIPT_SHA256: WATCHDOG_REVISION}
NO_EXTERNAL_STATE_STORAGE = {
    "format": "dsv41-state-storage-policy",
    "version": 1,
    "expert_cache": "memory-resident",
    "kv_cache": "memory-resident",
    "external_cache_paths": [],
    "external_state_paths": [],
}
ADMITTED_UBATCH = 32
ADMITTED_BATCH = 2048
EXPERT_COUNT = 384
EXPERTS_USED = 6
EXPERT_SLOT_BYTES = 398_131_200
REQUIRED_EXPERT_SLOTS = min(EXPERT_COUNT, EXPERTS_USED * ADMITTED_UBATCH)
REQUIRED_EXPERT_CACHE_BYTES = REQUIRED_EXPERT_SLOTS * EXPERT_SLOT_BYTES
REQUIRED_EXPERT_CACHE_MIB = REQUIRED_EXPERT_CACHE_BYTES // (1024 * 1024)
RAW_ATTENTION_LAYERS = (0, 1)
RAW_ATTENTION_WIDTH = 128
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


def strict_json_loads(data: str) -> Any:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result = {}
        for key, value in pairs:
            if key in result:
                raise TraceError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    try:
        return json.loads(data, object_pairs_hook=reject_duplicates)
    except json.JSONDecodeError as error:
        raise TraceError(f"invalid JSON: {error}") from error


def _require_exact_keys(record: dict[str, Any], keys: set[str], label: str) -> None:
    missing = sorted(keys - set(record))
    extra = sorted(set(record) - keys)
    if missing or extra:
        detail = []
        if missing:
            detail.append("missing " + ", ".join(missing))
        if extra:
            detail.append("unexpected " + ", ".join(extra))
        raise TraceError(f"{label} fields are invalid: {'; '.join(detail)}")


def validate_accelerator_attestation(runtime: str, accelerator: Any) -> dict[str, Any]:
    if not isinstance(accelerator, dict):
        raise TraceError("manifest accelerator attestation is invalid")
    common = {
        "format",
        "version",
        "runtime_kind",
        "platform",
        "backend",
        "backend_device",
        "backend_description",
        "architecture",
        "source",
    }
    if runtime == "llama.cpp":
        _require_exact_keys(
            accelerator,
            common | {"pci_device_id", "kfd_node", "gpu_id", "gfx_target_version"},
            "llama.cpp accelerator attestation",
        )
        expected = {
            "format": "dsv41-accelerator-attestation",
            "version": 2,
            "runtime_kind": "strix-rocm",
            "platform": "linux",
            "backend": "ROCm",
            "backend_device": "ROCm0",
            "architecture": "gfx1151",
            "gfx_target_version": 110501,
            "source": "linux-kfd-sysfs",
        }
        for key, value in expected.items():
            if accelerator.get(key) != value:
                raise TraceError(f"llama.cpp accelerator {key} mismatch")
        if re.fullmatch(
                r"[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]",
                accelerator.get("pci_device_id", "")) is None:
            raise TraceError("llama.cpp accelerator PCI identity is invalid")
        if not isinstance(accelerator.get("kfd_node"), str) or not accelerator["kfd_node"].isdigit():
            raise TraceError("llama.cpp accelerator KFD node is invalid")
        if type(accelerator.get("gpu_id")) is not int or accelerator["gpu_id"] <= 0:
            raise TraceError("llama.cpp accelerator GPU identity is invalid")
    elif runtime == "ds4":
        _require_exact_keys(
            accelerator,
            common | {
                "metal_registry_id",
                "recommended_max_working_set_bytes",
                "unified_memory",
            },
            "ds4 accelerator attestation",
        )
        expected = {
            "format": "dsv41-accelerator-attestation",
            "version": 2,
            "runtime_kind": "apple-metal",
            "platform": "macos",
            "backend": "Metal",
            "source": "metal-device-query",
            "unified_memory": True,
        }
        for key, value in expected.items():
            if accelerator.get(key) != value:
                raise TraceError(f"ds4 accelerator {key} mismatch")
        if type(accelerator.get("unified_memory")) is not bool:
            raise TraceError("ds4 accelerator unified-memory identity is invalid")
        registry_id = accelerator.get("metal_registry_id")
        if type(registry_id) is not int or registry_id <= 0:
            raise TraceError("ds4 accelerator Metal registry identity is invalid")
        working_set = accelerator.get("recommended_max_working_set_bytes")
        if type(working_set) is not int or working_set <= 0:
            raise TraceError("ds4 accelerator working-set identity is invalid")
    else:
        raise TraceError(f"unsupported runtime accelerator attestation: {runtime}")
    for key in ("backend_device", "backend_description", "architecture"):
        if not isinstance(accelerator.get(key), str) or not accelerator[key]:
            raise TraceError(f"manifest accelerator {key} is invalid")
    return accelerator


def validate_storage_attestation(runtime: str, item: Any) -> dict[str, Any]:
    if not isinstance(item, dict):
        raise TraceError("storage attestation is invalid")
    common = {
        "format",
        "version",
        "runtime_kind",
        "platform",
        "storage_kind",
        "resolved_path",
        "existing_path",
        "mount_point",
        "filesystem_type",
        "source",
    }
    if runtime == "llama.cpp":
        _require_exact_keys(
            item,
            common | {
                "mount_source",
                "device_number",
                "block_device_path",
                "nvme_device",
                "rotational",
            },
            "llama.cpp storage attestation",
        )
        expected = {
            "format": "dsv41-storage-attestation",
            "version": 2,
            "runtime_kind": "strix-rocm",
            "platform": "linux",
            "storage_kind": "linux-nvme",
            "source": "linux-mountinfo-sysfs",
            "rotational": False,
        }
        for key, value in expected.items():
            if item.get(key) != value:
                raise TraceError(f"llama.cpp storage {key} mismatch")
        if type(item.get("rotational")) is not bool:
            raise TraceError("llama.cpp storage rotational identity is invalid")
        if not isinstance(item.get("nvme_device"), str) or re.fullmatch(
                r"nvme[0-9]+(?:c[0-9]+)?n[0-9]+", item["nvme_device"]) is None:
            raise TraceError("llama.cpp storage NVMe device identity is invalid")
        if not isinstance(item.get("mount_source"), str) or not item["mount_source"].startswith("/dev/"):
            raise TraceError("llama.cpp storage mount source is not a local block device")
        if re.fullmatch(r"[0-9]+:[0-9]+", item.get("device_number", "")) is None:
            raise TraceError("llama.cpp storage device number is invalid")
        block_device_path = item.get("block_device_path")
        if not isinstance(block_device_path, str) or item["nvme_device"] not in Path(block_device_path).parts:
            raise TraceError("llama.cpp storage block device ancestry is invalid")
    elif runtime == "ds4":
        _require_exact_keys(
            item,
            common | {
                "device_identifier",
                "parent_whole_disk",
                "bus_protocol",
                "filesystem_device",
                "internal",
                "solid_state",
            },
            "ds4 storage attestation",
        )
        expected = {
            "format": "dsv41-storage-attestation",
            "version": 2,
            "runtime_kind": "apple-metal",
            "platform": "macos",
            "storage_kind": "darwin-local-solid-state",
            "source": "diskutil-info-plist",
            "internal": True,
            "solid_state": True,
        }
        for key, value in expected.items():
            if item.get(key) != value:
                raise TraceError(f"ds4 storage {key} mismatch")
        if type(item.get("internal")) is not bool or type(item.get("solid_state")) is not bool:
            raise TraceError("ds4 storage media identity is invalid")
        for key in ("device_identifier", "parent_whole_disk", "bus_protocol"):
            if not isinstance(item.get(key), str) or not item[key]:
                raise TraceError(f"ds4 storage {key} is invalid")
        if item["bus_protocol"].lower() not in {"nvme", "apple fabric"}:
            raise TraceError("ds4 storage is not NVMe-backed")
        if type(item.get("filesystem_device")) is not int or item["filesystem_device"] < 0:
            raise TraceError("ds4 storage filesystem device identity is invalid")
    else:
        raise TraceError(f"unsupported runtime storage attestation: {runtime}")
    for path_key in ("resolved_path", "existing_path", "mount_point"):
        value = item.get(path_key)
        if not isinstance(value, str) or not value.startswith("/"):
            raise TraceError(f"storage {path_key} is invalid")
    if not isinstance(item.get("filesystem_type"), str) or not item["filesystem_type"]:
        raise TraceError("storage filesystem type is invalid")
    try:
        Path(item["resolved_path"]).relative_to(Path(item["mount_point"]))
        Path(item["existing_path"]).relative_to(Path(item["mount_point"]))
    except ValueError as error:
        raise TraceError("storage mount ancestry is invalid") from error
    return item


def validate_host_attestation(host: Any) -> dict[str, Any]:
    if not isinstance(host, dict):
        raise TraceError("ds4 host attestation is invalid")
    _require_exact_keys(
        host,
        {
            "format",
            "version",
            "runtime_kind",
            "platform",
            "machine",
            "hardware_model",
            "os_version",
            "memory_bytes",
            "source",
        },
        "ds4 host attestation",
    )
    expected = {
        "format": "dsv41-host-attestation",
        "version": 1,
        "runtime_kind": "apple-metal",
        "platform": "macos",
        "machine": "arm64",
        "source": "darwin-sysctl",
    }
    for key, value in expected.items():
        if host.get(key) != value:
            raise TraceError(f"ds4 host {key} mismatch")
    for key in ("hardware_model", "os_version"):
        if not isinstance(host.get(key), str) or not host[key]:
            raise TraceError(f"ds4 host {key} is invalid")
    if type(host.get("memory_bytes")) is not int or host["memory_bytes"] < 128 * 1024 * 1024 * 1024:
        raise TraceError("ds4 host memory is below 128 GiB")
    return host


WATCHDOG_STATE_KEYS = {
    "total_bytes",
    "available_bytes",
    "used_bytes",
    "swap_entries",
    "peak_used_bytes",
    "child_pid",
    "child_status",
    "child_returncode",
    "process_group_id",
    "process_group_status",
    "threshold_reason",
}
WATCHDOG_REQUIRED_ERROR_CLASSIFICATIONS = {
    "configuration_error",
    "internal_error",
    "launch_error",
    "lease_error",
    "termination_timeout",
}
WATCHDOG_OPTIONAL_ERROR_CLASSIFICATIONS = {
    "procfs_error",
    "signal_error",
}


def validate_watchdog_event(event: Any) -> dict[str, Any]:
    if not isinstance(event, dict):
        raise TraceError("watchdog JSONL event is not an object")
    event_name = event.get("event")
    required = {"timestamp", "event"} | WATCHDOG_STATE_KEYS
    optional: set[str] = set()
    if event_name == "preflight":
        required |= {"soft_bytes", "emergency_bytes", "strict_ceiling_bytes"}
    elif event_name == "child_started":
        required.add("command")
    elif event_name == "sample":
        pass
    elif event_name == "process_group_signal":
        required.add("signal")
        optional.add("grace_deadline_monotonic")
    elif event_name == "final":
        required |= {"classification", "exit_code"}
        optional |= {"error", "secondary_errors"}
    else:
        raise TraceError(f"watchdog JSONL event name is invalid: {event_name}")
    missing = sorted(required - set(event))
    unexpected = sorted(set(event) - required - optional)
    if missing or unexpected:
        details = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if unexpected:
            details.append("unexpected " + ", ".join(unexpected))
        raise TraceError("watchdog JSONL event fields are invalid: " + "; ".join(details))
    if not isinstance(event["timestamp"], str) or re.fullmatch(
            r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}\.[0-9]{3}Z",
            event["timestamp"]) is None:
        raise TraceError("watchdog JSONL timestamp is invalid")
    for key in ("total_bytes", "available_bytes", "used_bytes", "swap_entries", "peak_used_bytes"):
        if event[key] is not None and (type(event[key]) is not int or event[key] < 0):
            raise TraceError(f"watchdog JSONL {key} is invalid")
    for key in ("child_pid", "process_group_id"):
        if event[key] is not None and (type(event[key]) is not int or event[key] <= 0):
            raise TraceError(f"watchdog JSONL {key} is invalid")
    if event["child_returncode"] is not None and type(event["child_returncode"]) is not int:
        raise TraceError("watchdog JSONL child_returncode is invalid")
    if event["child_status"] not in {"not_started", "running", "signaled", "exited"}:
        raise TraceError("watchdog JSONL child_status is invalid")
    if event["process_group_status"] not in {
            "not_created", "active", "leader_exited", "signal_error",
            "termination_timeout", "missing", "sighup_sent", "sigint_sent",
            "sigterm_sent", "sigkill_sent", "sigkill_timeout"}:
        raise TraceError("watchdog JSONL process_group_status is invalid")
    if not isinstance(event["threshold_reason"], str) or not event["threshold_reason"]:
        raise TraceError("watchdog JSONL threshold_reason is invalid")
    if event_name == "preflight":
        for key in ("soft_bytes", "emergency_bytes", "strict_ceiling_bytes"):
            if type(event[key]) is not int or event[key] <= 0:
                raise TraceError(f"watchdog JSONL {key} is invalid")
    elif event_name == "child_started":
        if not isinstance(event["command"], list) or not event["command"] or (
                not all(isinstance(value, str) and value for value in event["command"])):
            raise TraceError("watchdog JSONL command is invalid")
    elif event_name == "process_group_signal":
        if event["signal"] not in {"SIGHUP", "SIGINT", "SIGTERM", "SIGKILL"}:
            raise TraceError("watchdog JSONL signal is invalid")
        deadline = event.get("grace_deadline_monotonic")
        if deadline is not None and (
                event["signal"] != "SIGTERM" or
                not isinstance(deadline, (int, float)) or isinstance(deadline, bool) or
                not math.isfinite(deadline)):
            raise TraceError("watchdog JSONL grace deadline is invalid")
    elif event_name == "final":
        if event["classification"] not in {
                "child_exit", "soft_limit", "grace_timeout", "swap_appeared",
                "emergency_limit", "procfs_error", "signal_error", "termination_timeout",
                "lease_error", "parent_signal", "internal_error", "launch_error",
                "configuration_error", "startup_swap_active", "startup_emergency_limit",
                "startup_soft_limit"}:
            raise TraceError("watchdog JSONL classification is invalid")
        if type(event["exit_code"]) is not int:
            raise TraceError("watchdog JSONL exit code is invalid")
        classification = event["classification"]
        requires_error = classification in WATCHDOG_REQUIRED_ERROR_CLASSIFICATIONS
        allows_error = requires_error or classification in WATCHDOG_OPTIONAL_ERROR_CLASSIFICATIONS
        if requires_error and "error" not in event:
            raise TraceError("watchdog JSONL error presence does not match classification")
        if not allows_error and "error" in event:
            raise TraceError("watchdog JSONL error presence does not match classification")
        if "error" in event and (not isinstance(event["error"], str) or not event["error"]):
            raise TraceError("watchdog JSONL error is invalid")
        if "secondary_errors" in event:
            secondary_errors = event["secondary_errors"]
            if classification != "signal_error" or "error" not in event:
                raise TraceError("watchdog JSONL secondary errors require a primary signal error")
            if not isinstance(secondary_errors, list) or not secondary_errors:
                raise TraceError("watchdog JSONL secondary errors are invalid")
            for secondary_error in secondary_errors:
                _require_exact_keys(
                    secondary_error, {"component", "detail"}, "watchdog JSONL secondary error")
                if secondary_error["component"] not in {"audit", "lease", "stderr"} or (
                        not isinstance(secondary_error["detail"], str) or not secondary_error["detail"]):
                    raise TraceError("watchdog JSONL secondary error is invalid")
    return event


def element_count(shape: Iterable[int]) -> int:
    count = 1
    for dim in shape:
        if type(dim) is not int or dim <= 0:
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
    allowed = set(required)
    if event.get("component") == "expert.ids":
        allowed.add("semantic_id_space")
    extra = sorted(set(event) - allowed)
    if extra:
        raise TraceError(f"event has unexpected fields: {', '.join(extra)}")
    if event["trace_version"] != TRACE_VERSION:
        raise TraceError(f"unsupported event version: {event['trace_version']!r}")
    if event["byte_order"] != "little":
        raise TraceError("trace blobs must use little-endian byte order")
    if event["phase"] not in ("input", "prefill", "decode"):
        raise TraceError("event phase is invalid")
    if type(event["step"]) is not int or event["step"] < 0:
        raise TraceError("event step is invalid")
    if type(event["token_start"]) is not int or event["token_start"] < 0:
        raise TraceError("event token_start is invalid")
    if type(event["token_count"]) is not int or event["token_count"] <= 0:
        raise TraceError("event token_count is invalid")
    if event["layer"] is not None and (type(event["layer"]) is not int or event["layer"] < 0):
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
        if root.is_symlink():
            raise TraceError("trace root must not be a symlink")
        self.root = root.resolve()
        try:
            self.manifest = strict_json_loads(self._path(MANIFEST_NAME).read_text(encoding="ascii"))
        except (OSError, UnicodeError, TraceError) as error:
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
            with self._path(EVENTS_NAME).open("rb") as stream:
                for line_number, raw in enumerate(stream, 1):
                    if not raw.endswith(b"\n"):
                        raise TraceError(f"events.jsonl is truncated at line {line_number}")
                    try:
                        event = strict_json_loads(raw.decode("ascii"))
                    except (UnicodeError, TraceError) as error:
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
        if not isinstance(self.manifest.get("runtime"), str) or not self.manifest["runtime"]:
            raise TraceError("manifest runtime is invalid")
        if self.manifest["runtime"] not in ("ds4", "llama.cpp"):
            raise TraceError("manifest runtime must be ds4 or llama.cpp")
        top_level = {
            "trace_format",
            "trace_version",
            "event_count",
            "runtime",
            "revision",
            "build",
            "model",
            "prompt",
            "accelerator",
            "config",
            "comparison",
            "environment",
            "paths",
            "storage_policy",
            "audits",
            "expected",
        }
        top_level.add("candidate" if self.manifest["runtime"] == "llama.cpp" else "host")
        _require_exact_keys(self.manifest, top_level, f"{self.manifest['runtime']} manifest")
        if not isinstance(self.manifest["revision"], str) or not self.manifest["revision"]:
            raise TraceError("manifest revision is invalid")
        if self.manifest["runtime"] == "ds4" and self.manifest["revision"] != DS4_REVISION:
            raise TraceError(f"ds4 revision must be {DS4_REVISION}")
        if self.manifest["runtime"] == "llama.cpp" and re.fullmatch(
                r"[0-9a-f]{40}", self.manifest["revision"]) is None:
            raise TraceError("llama.cpp revision must be the exact full Git revision")
        if not isinstance(self.manifest["build"], dict):
            raise TraceError("manifest build is invalid")
        build_keys = (
            {"number", "info", "compiler", "target", "path", "sha256", "runtime_libraries"}
            if self.manifest["runtime"] == "llama.cpp"
            else {"compiler", "target", "path", "sha256"}
        )
        _require_exact_keys(self.manifest["build"], build_keys, "manifest build")
        build_sha256 = self.manifest["build"].get("sha256", "")
        if not isinstance(build_sha256, str) or re.fullmatch(r"[0-9a-f]{64}", build_sha256) is None:
            raise TraceError("manifest build SHA-256 is invalid")
        if self.manifest["runtime"] == "ds4" and (
                APPROVED_EXPORTERS.get(build_sha256) != DS4_REVISION):
            raise TraceError("ds4 exporter is not approved for the pinned ds4 revision")
        for key in build_keys - {"sha256", "number", "runtime_libraries"}:
            value = self.manifest["build"].get(key)
            if not isinstance(value, str) or not value:
                raise TraceError(f"manifest build {key} is invalid")
        if "number" in build_keys and type(self.manifest["build"]["number"]) is not int:
            raise TraceError("manifest build number is invalid")
        build_path = self.manifest["build"]["path"]
        if not build_path.startswith("/") or ".." in PurePosixPath(build_path).parts or (
                str(PurePosixPath(build_path)) != build_path):
            raise TraceError("manifest build path is not canonical")
        if self.manifest["runtime"] == "llama.cpp":
            libraries = self.manifest["build"]["runtime_libraries"]
            if not isinstance(libraries, list) or len(libraries) != 4:
                raise TraceError("manifest runtime library identities are invalid")
            roles = set()
            for library in libraries:
                _require_exact_keys(library, {"role", "path", "sha256"}, "manifest runtime library")
                role = library.get("role")
                path = library.get("path")
                digest = library.get("sha256")
                if role not in {"build-info", "llama", "ggml", "selected-backend"}:
                    raise TraceError("manifest runtime library role is invalid")
                if role in roles:
                    raise TraceError("manifest runtime library role is duplicated")
                roles.add(role)
                if not isinstance(path, str) or not path.startswith("/") or ".." in PurePosixPath(path).parts or (
                        str(PurePosixPath(path)) != path):
                    raise TraceError("manifest runtime library path is not canonical")
                if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
                    raise TraceError("manifest runtime library SHA-256 is invalid")
            if roles != {"build-info", "llama", "ggml", "selected-backend"}:
                raise TraceError("manifest runtime library identities are incomplete")
        for section in ("model", "prompt"):
            if not isinstance(self.manifest[section], dict):
                raise TraceError(f"manifest {section} is invalid")
        _require_exact_keys(
            self.manifest["model"],
            {"path", "sha256", "byte_count", "architecture"},
            "manifest model",
        )
        _require_exact_keys(
            self.manifest["prompt"],
            {
                "path",
                "sha256",
                "byte_count",
                "corpus_name",
                "corpus_sha256",
                "target_tokens",
                "provenance",
            },
            "manifest prompt",
        )
        for section in ("model", "prompt"):
            digest = self.manifest[section].get("sha256")
            if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
                raise TraceError(f"manifest {section} SHA-256 is invalid")
            if type(self.manifest[section].get("byte_count")) is not int or self.manifest[section]["byte_count"] <= 0:
                raise TraceError(f"manifest {section} byte_count is invalid")
        for section in ("config", "comparison", "environment", "audits"):
            if not isinstance(self.manifest[section], dict):
                raise TraceError(f"manifest {section} is invalid")
        _require_exact_keys(
            self.manifest["environment"],
            {"system_info", "command"},
            "manifest environment",
        )
        if not all(isinstance(value, str) and value for value in self.manifest["environment"].values()):
            raise TraceError("manifest environment values are invalid")
        system_info = self.manifest["environment"]["system_info"].lower()
        if self.manifest["runtime"] == "llama.cpp" and "linux" not in system_info:
            raise TraceError("llama.cpp environment is not Linux")
        if self.manifest["runtime"] == "ds4" and not any(
                name in system_info for name in ("darwin", "macos")):
            raise TraceError("ds4 environment is not macOS")
        context = self.manifest["config"].get("context")
        decode_steps = self.manifest["config"].get("decode_steps")
        if type(context) is not int or type(decode_steps) is not int or (
                decode_steps <= 0 or context <= decode_steps):
            raise TraceError("manifest context or decode_steps is invalid")
        if self.manifest["model"]["sha256"] != MODEL_SHA256:
            raise TraceError(f"model SHA-256 must be {MODEL_SHA256}")
        if self.manifest["model"].get("architecture") != "deepseek41":
            raise TraceError("model architecture must be deepseek41")
        if self.manifest["storage_policy"] != NO_EXTERNAL_STATE_STORAGE:
            raise TraceError("manifest external cache/state storage policy is invalid")
        accelerator = validate_accelerator_attestation(self.manifest["runtime"], self.manifest["accelerator"])
        if self.manifest["runtime"] == "ds4":
            if "host" not in self.manifest:
                raise TraceError("ds4 host attestation is missing")
            validate_host_attestation(self.manifest["host"])
        else:
            if "host" in self.manifest:
                raise TraceError("llama.cpp manifest must not contain Apple host attestation")
        required_paths = {
            "model", "prompt", "output", "repository", "temporary_directory",
        }
        if self.manifest["runtime"] == "ds4":
            required_paths.update({"runtime_checkout", "runner_executable", "runner_script", "exporter"})
        paths = self.manifest["paths"]
        if not isinstance(paths, dict) or set(paths) != required_paths:
            raise TraceError("manifest execution paths are invalid")
        for label, value in paths.items():
            if not isinstance(label, str) or not isinstance(value, str) or not value.startswith("/"):
                raise TraceError("manifest execution path is invalid")
        if self.manifest["model"].get("path") != paths["model"]:
            raise TraceError("manifest model path is not bound to execution paths")
        if self.manifest["prompt"].get("path") != paths["prompt"]:
            raise TraceError("manifest prompt path is not bound to execution paths")
        if self.manifest["runtime"] == "ds4" and self.manifest["build"]["path"] != paths["exporter"]:
            raise TraceError("ds4 build path is not bound to the executed exporter")
        corpus_name = self.manifest["prompt"].get("corpus_name")
        if corpus_name not in CORPUS_SHA256:
            raise TraceError("prompt corpus is not in the fixed correctness corpus set")
        if self.manifest["prompt"].get("corpus_sha256") != CORPUS_SHA256[corpus_name]:
            raise TraceError(f"prompt corpus SHA-256 is invalid for {corpus_name}")
        provenance = self.manifest["prompt"].get("provenance")
        if not isinstance(provenance, dict):
            raise TraceError("prompt provenance reference is missing")
        _require_exact_keys(provenance, {"path", "sha256"}, "prompt provenance reference")
        provenance_sha256 = provenance.get("sha256", "")
        if not isinstance(provenance_sha256, str) or re.fullmatch(
                r"[0-9a-f]{64}", provenance_sha256) is None:
            raise TraceError("prompt provenance SHA-256 is invalid")
        if provenance.get("path") != f"provenance/{provenance_sha256}.json":
            raise TraceError("prompt provenance path is not content addressed")
        try:
            provenance_bytes = self._path(provenance["path"]).read_bytes()
            provenance_record = strict_json_loads(provenance_bytes.decode("ascii"))
        except (OSError, UnicodeError, TraceError) as error:
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
        _require_exact_keys(
            provenance_record,
            set(provenance_checks) | {"builder_sha256"},
            "prompt provenance",
        )
        if self.manifest["prompt"].get("target_tokens") != expected_target:
            raise TraceError("prompt target token count does not fill the configured context")
        if self.manifest["runtime"] == "llama.cpp":
            candidate = self.manifest.get("candidate")
            if not isinstance(candidate, dict):
                raise TraceError("llama.cpp candidate attestation is missing")
            _require_exact_keys(
                candidate,
                {
                    "repository",
                    "revision",
                    "base_revision",
                    "diff_sha256",
                    "executable_path",
                    "executable_sha256",
                    "runtime_libraries_sha256",
                },
                "llama.cpp candidate attestation",
            )
            if candidate.get("repository") != REPOSITORY:
                raise TraceError(f"candidate repository must be {REPOSITORY}")
            for key in (
                    "revision",
                    "base_revision",
                    "diff_sha256",
                    "executable_sha256",
                    "runtime_libraries_sha256"):
                value = candidate.get(key, "")
                if not isinstance(value, str) or re.fullmatch(
                        r"[0-9a-f]{40}" if "revision" in key else r"[0-9a-f]{64}", value) is None:
                    raise TraceError(f"candidate {key} is invalid")
            executable_path = candidate.get("executable_path")
            if not isinstance(executable_path, str) or not executable_path.startswith("/") or (
                    ".." in PurePosixPath(executable_path).parts or
                    str(PurePosixPath(executable_path)) != executable_path):
                raise TraceError("candidate executable path is invalid")
            if candidate["revision"] != self.manifest["revision"]:
                raise TraceError("candidate revision does not match the exporter build revision")
            if candidate.get("executable_path") != self.manifest["build"]["path"]:
                raise TraceError("candidate executable path does not match the trace build")
            if candidate["executable_sha256"] != self.manifest["build"]["sha256"]:
                raise TraceError("candidate executable SHA-256 does not match the trace build")
            runtime_libraries_sha256 = sha256_bytes(
                canonical_json(self.manifest["build"]["runtime_libraries"]).encode("ascii"))
            if candidate["runtime_libraries_sha256"] != runtime_libraries_sha256:
                raise TraceError("candidate runtime library identities do not match the trace build")
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
            "raw_attention_layers": list(RAW_ATTENTION_LAYERS),
            "raw_attention_width": RAW_ATTENTION_WIDTH,
            "candidate_propagation_layers": [24, 28, 32, 36],
        }
        if self.manifest["config"].get("deepseek41") != expected_config:
            raise TraceError("DeepSeek V4.1 configuration is invalid")
        expected_comparison = {
            "tokens": "exact",
            "engram_rows": "exact",
            "expert_ids": "exact-original-id-space",
            "expert_weights": "byte-identical-f32",
            "attention_candidates": "exact",
            "logits": "byte-identical-f32",
        }
        if self.manifest["comparison"] != expected_comparison:
            raise TraceError("trace comparison policy is invalid")
        config = self.manifest["config"]
        if self.manifest["runtime"] == "llama.cpp":
            _require_exact_keys(
                config,
                {
                    "context",
                    "batch",
                    "ubatch",
                    "device",
                    "device_architecture",
                    "device_pci_id",
                    "decode_steps",
                    "kv_type_k",
                    "kv_type_v",
                    "flash_attention",
                    "gpu_layers",
                    "load_mode",
                    "expert_cache_slots",
                    "expert_cache_bytes",
                    "tokenizer_add_bos",
                    "tokenizer_parse_special",
                    "deepseek41",
                },
                "llama.cpp config",
            )
            if accelerator["backend_device"] != "ROCm0":
                raise TraceError("llama.cpp accelerator backend device must be ROCm0")
            if config.get("batch") != ADMITTED_BATCH or config.get("ubatch") != ADMITTED_UBATCH:
                raise TraceError("llama.cpp trace does not use the admitted batch and ubatch")
            if config.get("expert_cache_slots") != REQUIRED_EXPERT_SLOTS or (
                    config.get("expert_cache_bytes") != REQUIRED_EXPERT_CACHE_BYTES):
                raise TraceError("llama.cpp trace does not use the admitted expert cache")
            if config.get("device") != "ROCm0" or config.get("gpu_layers") != 99:
                raise TraceError("llama.cpp trace does not use the required ROCm0 offload")
            if config.get("device_architecture") != accelerator["architecture"] or (
                    config.get("device_pci_id") != accelerator["pci_device_id"]):
                raise TraceError("llama.cpp trace device identity is not bound to the accelerator attestation")
            if config.get("kv_type_k") != "f16" or config.get("kv_type_v") != "f16" or (
                    config.get("flash_attention") is not True) or config.get("load_mode") != 0:
                raise TraceError("llama.cpp trace inference configuration is invalid")
        if self.manifest["runtime"] == "ds4":
            _require_exact_keys(
                config,
                {
                    "context",
                    "decode_steps",
                    "prefill_chunk",
                    "device_backend",
                    "device_registry_id",
                    "deepseek41",
                },
                "ds4 config",
            )
            if config.get("prefill_chunk") != ADMITTED_UBATCH:
                raise TraceError("ds4 trace does not use the admitted prefill chunk")
            if config.get("device_backend") != "Metal" or (
                    config.get("device_registry_id") != accelerator["metal_registry_id"]):
                raise TraceError("ds4 trace device identity is not bound to the accelerator attestation")
        _require_exact_keys(self.manifest["audits"], {"pre", "post"}, "manifest audit envelope")
        for audit_phase in ("pre", "post"):
            phase_audits = self.manifest["audits"].get(audit_phase)
            if not isinstance(phase_audits, dict):
                raise TraceError(f"manifest {audit_phase} audit set is invalid")
            expected_kinds = (
                ("memory", "swap", "watchdog")
                if self.manifest["runtime"] == "llama.cpp"
                else ("memory", "swap", "runner")
            )
            if set(phase_audits) != set(expected_kinds):
                raise TraceError(f"manifest {audit_phase} audit kinds are invalid")
            for kind in expected_kinds:
                self._validate_audit_reference(audit_phase, kind, phase_audits.get(kind))

    def _validate_audit_reference(self, phase: str, kind: str, audit: Any) -> None:
        if not isinstance(audit, dict):
            raise TraceError(f"manifest {phase} {kind} audit reference is invalid")
        _require_exact_keys(
            audit,
            {"path", "sha256", "created_unix"},
            f"manifest {phase} {kind} audit reference",
        )
        audit_path = audit.get("path")
        if not isinstance(audit_path, str) or not audit_path:
            raise TraceError(f"manifest {phase} {kind} audit path is invalid")
        digest = audit.get("sha256", "")
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise TraceError(f"manifest {phase} {kind} audit SHA-256 is invalid")
        if type(audit.get("created_unix")) is not int or audit["created_unix"] <= 0:
            raise TraceError(f"manifest {phase} {kind} audit timestamp is invalid")
        expected_path = f"audits/{phase}/{digest}.json"
        if audit_path != expected_path:
            raise TraceError(f"manifest {phase} {kind} audit path is not content addressed")
        evidence_path = self._path(audit_path)
        try:
            evidence = evidence_path.read_bytes()
        except OSError as error:
            raise TraceError(f"cannot read {phase} {kind} audit evidence: {error}") from error
        if sha256_bytes(evidence) != digest:
            raise TraceError(f"{phase} {kind} audit evidence SHA-256 mismatch")
        try:
            record = strict_json_loads(evidence.decode("ascii"))
        except (UnicodeError, TraceError) as error:
            raise TraceError(f"{phase} {kind} audit evidence is invalid: {error}") from error
        if record.get("kind") != kind or record.get("created_unix") != audit["created_unix"]:
            raise TraceError(f"{phase} {kind} audit evidence metadata mismatch")
        record_keys = {"created_unix", "kind", "environment", "data"}
        if kind == "memory":
            record_keys |= {"storage", "storage_policy", "accelerator"}
            if self.manifest["runtime"] == "ds4":
                record_keys.add("host")
        _require_exact_keys(record, record_keys, f"{phase} {kind} audit evidence")
        expected_environment = (
            {"HIP_LAUNCH_BLOCKING": "1"}
            if self.manifest["runtime"] == "llama.cpp"
            else {}
        )
        if record.get("environment") != expected_environment:
            raise TraceError(f"{phase} {kind} audit environment is invalid")
        if not isinstance(record.get("data"), dict):
            raise TraceError(f"{phase} {kind} audit evidence data is invalid")
        if kind == "memory":
            _require_exact_keys(
                record["data"],
                {"mem_total_bytes", "mem_available_bytes", "mem_used_bytes"},
                f"{phase} memory audit data",
            )
            used = record["data"].get("mem_used_bytes")
            total = record["data"].get("mem_total_bytes")
            available = record["data"].get("mem_available_bytes")
            if type(used) is not int or used < 0:
                raise TraceError(f"{phase} memory audit evidence is invalid")
            if self.manifest["runtime"] == "llama.cpp" and used >= SOFT_MEMORY_LIMIT:
                raise TraceError(f"{phase} memory audit evidence is invalid")
            if self.manifest["runtime"] == "ds4" and (
                    type(total) is not int or total < 128 * 1024 * 1024 * 1024 or
                    type(available) is not int or available <= 0 or available > total or used > total):
                raise TraceError(f"{phase} ds4 memory audit evidence is invalid")
            storage = record.get("storage")
            if record.get("storage_policy") != self.manifest["storage_policy"]:
                raise TraceError(f"{phase} memory audit storage policy mismatch")
            required_storage = {
                "model", "prompt", "output", "repository", "temporary_directory",
            }
            if self.manifest["runtime"] == "ds4":
                required_storage.update({"runtime_checkout", "runner_executable", "runner_script", "exporter"})
            if not isinstance(storage, dict) or set(storage) != required_storage:
                raise TraceError(f"{phase} memory audit storage evidence is missing")
            for label, item in storage.items():
                if not isinstance(label, str):
                    raise TraceError(f"{phase} memory audit storage evidence is invalid")
                validate_storage_attestation(self.manifest["runtime"], item)
                if item["resolved_path"] != self.manifest["paths"][label]:
                    raise TraceError(f"{phase} memory audit {label} path differs from the manifest")
            if record.get("accelerator") != self.manifest["accelerator"]:
                raise TraceError(f"{phase} memory audit accelerator evidence mismatch")
            if self.manifest["runtime"] == "ds4":
                host = validate_host_attestation(record.get("host"))
                if host != self.manifest["host"]:
                    raise TraceError(f"{phase} ds4 host evidence mismatch")
                if host["memory_bytes"] != total:
                    raise TraceError(f"{phase} ds4 host memory differs from the memory audit")
        if kind == "swap":
            if self.manifest["runtime"] == "llama.cpp":
                _require_exact_keys(record["data"], {"enabled", "entries"}, f"{phase} swap audit data")
                if record["data"].get("enabled") is not False or record["data"].get("entries") != []:
                    raise TraceError(f"{phase} swap audit evidence does not report zero configured swap")
            elif record["data"] != {
                    "source": "darwin-sysctl-vm.swapusage",
                    "total_bytes": 0,
                    "used_bytes": 0,
                    "free_bytes": 0,
            }:
                raise TraceError(f"{phase} ds4 swap audit evidence does not report zero swap")
        if kind == "runner":
            data = record["data"]
            _require_exact_keys(
                data,
                {
                    "format",
                    "version",
                    "runtime_kind",
                    "source",
                    "runner_pid",
                    "runner_parent_pid",
                    "runner_uid",
                    "runner_executable",
                    "runner_executable_sha256",
                    "runner_script",
                    "runner_script_sha256",
                    "exporter_path",
                    "exporter_sha256",
                    "checkout_path",
                    "checkout_revision",
                    "command_sha256",
                },
                f"{phase} ds4 runner audit",
            )
            expected = {
                "format": "dsv41-runner-ownership",
                "version": 1,
                "runtime_kind": "apple-metal",
                "source": "python-subprocess",
                "exporter_sha256": self.manifest["build"]["sha256"],
                "checkout_revision": DS4_REVISION,
                "checkout_path": self.manifest["paths"]["runtime_checkout"],
                "runner_executable": self.manifest["paths"]["runner_executable"],
                "runner_script": self.manifest["paths"]["runner_script"],
                "exporter_path": self.manifest["paths"]["exporter"],
            }
            for key, value in expected.items():
                if data.get(key) != value:
                    raise TraceError(f"{phase} ds4 runner {key} mismatch")
            for key in ("runner_pid", "runner_parent_pid"):
                if type(data.get(key)) is not int or data[key] <= 0:
                    raise TraceError(f"{phase} ds4 runner {key} is invalid")
            if type(data.get("runner_uid")) is not int or data["runner_uid"] < 0:
                raise TraceError(f"{phase} ds4 runner UID is invalid")
            for key in (
                    "runner_executable_sha256",
                    "runner_script_sha256",
                    "exporter_sha256",
                    "command_sha256"):
                if re.fullmatch(r"[0-9a-f]{64}", data.get(key, "")) is None:
                    raise TraceError(f"{phase} ds4 runner {key} is invalid")
        if kind == "watchdog":
            required = (
                "format",
                "version",
                "lease_id",
                "state",
                "file_device",
                "file_inode",
                "file_uid",
                "file_mode",
                "lease_path",
                "watchdog_pid",
                "watchdog_start_time_utc",
                "watchdog_start_time_ticks",
                "watchdog_command",
                "watchdog_command_sha256",
                "watchdog_executable_path",
                "watchdog_script_path",
                "watchdog_script_sha256",
                "watchdog_revision",
                "soft_bytes",
                "emergency_bytes",
                "strict_ceiling_bytes",
                "grace_seconds",
                "sample_interval_seconds",
                "procfs_root",
                "guardian_pid",
                "child_pid",
                "child_process_group_id",
                "command",
                "child_command_sha256",
                "heartbeat_path",
                "heartbeat_unix",
                "max_heartbeat_age_seconds",
                "audit_live_path",
                "audit_device",
                "audit_inode",
                "audit_uid",
                "audit_mode",
                "audit_fd",
                "audit_sha256",
                "audit",
            )
            _require_exact_keys(record["data"], set(required), f"{phase} watchdog audit evidence")
            data = record["data"]
            if data["format"] != WATCHDOG_LEASE_FORMAT or data["version"] != WATCHDOG_VERSION:
                raise TraceError(f"{phase} watchdog audit format is invalid")
            if data["state"] != "active":
                raise TraceError(f"{phase} watchdog audit state is invalid")
            for key in ("file_device", "file_inode", "file_uid"):
                if type(data[key]) is not int or data[key] < 0:
                    raise TraceError(f"{phase} watchdog lease {key} is invalid")
            if data["file_mode"] != 0o600:
                raise TraceError(f"{phase} watchdog lease file mode is invalid")
            if not isinstance(data["lease_id"], str) or re.fullmatch(r"[0-9a-f]{32,64}", data["lease_id"]) is None:
                raise TraceError(f"{phase} watchdog audit lease ID is invalid")
            if type(data["watchdog_pid"]) is not int or data["watchdog_pid"] <= 1:
                raise TraceError(f"{phase} watchdog audit PID is invalid")
            if type(data["watchdog_start_time_ticks"]) is not int or data["watchdog_start_time_ticks"] <= 0:
                raise TraceError(f"{phase} watchdog audit start time is invalid")
            if not isinstance(data["watchdog_start_time_utc"], str) or re.fullmatch(
                    r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}\.[0-9]{3}Z",
                    data["watchdog_start_time_utc"]) is None:
                raise TraceError(f"{phase} watchdog audit UTC start time is invalid")
            for key in ("watchdog_command_sha256", "watchdog_script_sha256"):
                if not isinstance(data[key], str) or re.fullmatch(r"[0-9a-f]{64}", data[key]) is None:
                    raise TraceError(f"{phase} watchdog audit {key} is invalid")
            for key in ("lease_path", "watchdog_command", "watchdog_script_path", "heartbeat_path", "audit_live_path"):
                if not isinstance(data[key], str) or not data[key]:
                    raise TraceError(f"{phase} watchdog audit {key} is invalid")
            if APPROVED_WATCHDOGS.get(data["watchdog_script_sha256"]) != data["watchdog_revision"]:
                raise TraceError(f"{phase} watchdog audit revision is invalid")
            if not isinstance(data["watchdog_executable_path"], str) or not data["watchdog_executable_path"]:
                raise TraceError(f"{phase} watchdog executable path is invalid")
            if data["soft_bytes"] != SOFT_MEMORY_LIMIT or (
                    data["emergency_bytes"] != WATCHDOG_EMERGENCY_LIMIT) or (
                    data["strict_ceiling_bytes"] != STRICT_MEMORY_LIMIT):
                raise TraceError(f"{phase} watchdog audit thresholds are invalid")
            if data["grace_seconds"] != 30.0 or data["sample_interval_seconds"] != 1.0:
                raise TraceError(f"{phase} watchdog timing policy is invalid")
            if data["procfs_root"] != "/proc":
                raise TraceError(f"{phase} watchdog procfs root is invalid")
            if type(data["guardian_pid"]) is not int or data["guardian_pid"] <= 1 or (
                    type(data["child_pid"]) is not int or data["child_pid"] <= 1) or (
                    type(data["child_process_group_id"]) is not int or data["child_process_group_id"] <= 1):
                raise TraceError(f"{phase} watchdog child identity is invalid")
            for key in ("audit_device", "audit_inode", "audit_uid", "audit_fd"):
                if type(data[key]) is not int or data[key] < 0:
                    raise TraceError(f"{phase} watchdog audit {key} is invalid")
            if data["audit_mode"] != 0o600:
                raise TraceError(f"{phase} watchdog audit mode is invalid")
            if not isinstance(data["command"], list) or not data["command"] or (
                    not all(isinstance(argument, str) for argument in data["command"])):
                raise TraceError(f"{phase} watchdog child command is invalid")
            canonical_command = json.dumps(
                data["command"], ensure_ascii=True, separators=(",", ":")).encode("utf-8")
            if data["child_command_sha256"] != sha256_bytes(canonical_command):
                raise TraceError(f"{phase} watchdog child command SHA-256 is invalid")
            if not isinstance(data["watchdog_command_sha256"], str) or re.fullmatch(
                    r"[0-9a-f]{64}", data["watchdog_command_sha256"]) is None:
                raise TraceError(f"{phase} watchdog audit command SHA-256 is invalid")
            if type(data["heartbeat_unix"]) is not int or data["heartbeat_unix"] <= 0:
                raise TraceError(f"{phase} watchdog audit heartbeat timestamp is invalid")
            max_age = data.get("max_heartbeat_age_seconds")
            if not isinstance(max_age, (int, float)) or isinstance(max_age, bool) or (
                    max_age <= 0 or max_age > 30):
                raise TraceError(f"{phase} watchdog audit heartbeat age is invalid")
            if data["heartbeat_unix"] > record["created_unix"] or (
                    record["created_unix"] - data["heartbeat_unix"] > max_age):
                raise TraceError(f"{phase} watchdog audit heartbeat was stale when captured")
            audit_jsonl = data["audit"]
            if not isinstance(audit_jsonl, dict):
                raise TraceError(f"{phase} watchdog JSONL reference is invalid")
            _require_exact_keys(
                audit_jsonl,
                {"path", "sha256", "event_count"},
                f"{phase} watchdog JSONL reference",
            )
            jsonl_digest = audit_jsonl.get("sha256", "")
            if not isinstance(jsonl_digest, str) or re.fullmatch(r"[0-9a-f]{64}", jsonl_digest) is None:
                raise TraceError(f"{phase} watchdog JSONL SHA-256 is invalid")
            if data["audit_sha256"] != jsonl_digest:
                raise TraceError(f"{phase} watchdog live and embedded audit SHA-256 differ")
            if audit_jsonl.get("path") != f"audits/{phase}/{jsonl_digest}.jsonl":
                raise TraceError(f"{phase} watchdog JSONL path is invalid")
            if type(audit_jsonl.get("event_count")) is not int or audit_jsonl["event_count"] < 2:
                raise TraceError(f"{phase} watchdog JSONL event count is invalid")
            jsonl_path = self._path(audit_jsonl["path"])
            try:
                jsonl_bytes = jsonl_path.read_bytes()
            except OSError as error:
                raise TraceError(f"cannot read {phase} watchdog JSONL audit: {error}") from error
            if sha256_bytes(jsonl_bytes) != jsonl_digest:
                raise TraceError(f"{phase} watchdog JSONL SHA-256 mismatch")
            lines = jsonl_bytes.decode("ascii").splitlines()
            if len(lines) != audit_jsonl["event_count"]:
                raise TraceError(f"{phase} watchdog JSONL event count mismatch")
            try:
                events = [validate_watchdog_event(strict_json_loads(line)) for line in lines]
            except TraceError as error:
                raise TraceError(f"{phase} watchdog JSONL is invalid: {error}") from error
            if not any(event.get("event") == "preflight" for event in events) or (
                    not any(event.get("event") == "child_started" for event in events)):
                raise TraceError(f"{phase} watchdog JSONL lacks startup evidence")

    def read_blob(self, event: dict[str, Any]) -> bytes:
        try:
            return self._path(event["blob"]).read_bytes()
        except OSError as error:
            raise TraceError(f"cannot read blob {event['blob']}: {error}") from error

    def _path(self, relative: str) -> Path:
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise TraceError(f"trace path is outside the bundle: {relative}")
        candidate = self.root
        for part in relative_path.parts:
            candidate = candidate / part
            if candidate.is_symlink():
                raise TraceError(f"trace path must not use symlinks: {relative}")
        try:
            candidate.resolve().relative_to(self.root)
        except ValueError as error:
            raise TraceError(f"trace path is outside the bundle: {relative}") from error
        return candidate

    def _validate_coverage(self) -> None:
        expected = self.manifest.get("expected")
        if not isinstance(expected, dict):
            raise TraceError("manifest expected coverage is missing")
        _require_exact_keys(
            expected,
            {"prompt_tokens", "decode_steps", "components"},
            "manifest expected coverage",
        )
        prompt_tokens = expected.get("prompt_tokens")
        decode_steps = expected.get("decode_steps")
        components = expected.get("components")
        if type(prompt_tokens) is not int or prompt_tokens <= 0:
            raise TraceError("expected prompt_tokens is invalid")
        if type(decode_steps) is not int or decode_steps <= 0:
            raise TraceError("expected decode_steps is invalid")
        if self.manifest.get("config", {}).get("decode_steps") != decode_steps:
            raise TraceError("expected decode_steps does not match config")
        if prompt_tokens != self.manifest.get("prompt", {}).get("target_tokens"):
            raise TraceError("expected prompt_tokens does not match prompt provenance")
        for event in self.events:
            self._validate_component_schema(event)
        raw_attention = {
            (event["phase"], event["step"], event["token_start"], event["layer"]): event
            for event in self.events
            if event["component"] == "attn.source" and event["layer"] in RAW_ATTENTION_LAYERS
        }
        raw_coordinates = {
            (phase, step, token_start)
            for phase, step, token_start, _layer in raw_attention
        }
        for phase, step, token_start in raw_coordinates:
            layer0 = raw_attention.get((phase, step, token_start, 0))
            layer1 = raw_attention.get((phase, step, token_start, 1))
            if layer0 is None or layer1 is None:
                continue
            if layer0["shape"] != layer1["shape"] or self.read_blob(layer0) != self.read_blob(layer1):
                raise TraceError(
                    f"raw attn.source differs between layers 0 and 1 at "
                    f"{phase} step {step} token {token_start}")
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
                if len(input_events) != 1 or input_events[0]["token_start"] != 0 or (
                        input_events[0]["token_count"] != prompt_tokens):
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
        if component == "attn.source" and layer in RAW_ATTENTION_LAYERS:
            if shape[0] != RAW_ATTENTION_WIDTH:
                raise TraceError(f"raw attn.source shape must be [{RAW_ATTENTION_WIDTH},token_count]")
            values = list(struct.iter_unpack("<i", self.read_blob(event)))
            sentinel = RAW_ATTENTION_WIDTH + token_count
            for token in range(token_count):
                for row in range(RAW_ATTENTION_WIDTH):
                    value = values[token*RAW_ATTENTION_WIDTH + row][0]
                    if 0 <= value < RAW_ATTENTION_WIDTH or (
                            RAW_ATTENTION_WIDTH <= value <= RAW_ATTENTION_WIDTH + token) or value == sentinel:
                        continue
                    raise TraceError(
                        f"raw attn.source contains invalid row {value} for token {token}; "
                        f"expected physical row, visible ubatch row, or sentinel {sentinel}")
        if component == "attn.candidate_blocks" and shape[0] > config.get("candidate_topk_blocks", 0):
            raise TraceError("attn.candidate_blocks width exceeds candidate_topk_blocks")
        if component == "attn.source" and layer not in RAW_ATTENTION_LAYERS and (
                shape[0] > config.get("index_top_k", 0)):
            raise TraceError(f"{component} width exceeds index_top_k")
        if component == "attn.candidates" and shape[0] > config.get("index_top_k", 0):
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


def compare_bundles(
        left: TraceBundle,
        right: TraceBundle,
        *,
        runtime_roles: tuple[str, str] = ("ds4", "llama.cpp")) -> Mismatch | None:
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
    if left.manifest["runtime"] != runtime_roles[0] or right.manifest["runtime"] != runtime_roles[1]:
        return Mismatch(
            "runtime_role",
            "manifest",
            "metadata",
            -1,
            -1,
            None,
            f"left trace must be {runtime_roles[0]} and right trace must be {runtime_roles[1]}",
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


def report(
        left: TraceBundle,
        right: TraceBundle,
        *,
        runtime_roles: tuple[str, str] = ("ds4", "llama.cpp"),
        success_status: str = "TARGET PASS") -> dict[str, Any]:
    mismatch = compare_bundles(left, right, runtime_roles=runtime_roles)
    if mismatch is None:
        return {
            "status": success_status,
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


def local_report(left: TraceBundle, right: TraceBundle, mode: str) -> dict[str, Any]:
    result = report(
        left,
        right,
        runtime_roles=("llama.cpp", "llama.cpp"),
        success_status="BRINGUP PASS",
    )
    result["mode"] = mode
    result["cross_runtime_status"] = "INCOMPLETE"
    result["cross_runtime_requirement"] = (
        "Run the pinned ds4 exporter and trace_format.py compare before reporting TARGET PASS.")
    if mode == "self-consistency":
        if left.manifest.get("candidate") != right.manifest.get("candidate"):
            result = {
                **result,
                "status": "FAIL",
                "first_divergence": Mismatch(
                    "candidate_identity",
                    "manifest",
                    "metadata",
                    -1,
                    -1,
                    None,
                    "self-consistency traces use different candidate attestations",
                ).as_dict(),
            }
    else:
        base_revision = left.manifest.get("candidate", {}).get("revision")
        integrated_base = right.manifest.get("candidate", {}).get("base_revision")
        if base_revision != integrated_base:
            result = {
                **result,
                "status": "FAIL",
                "first_divergence": Mismatch(
                    "candidate_identity",
                    "manifest",
                    "metadata",
                    -1,
                    -1,
                    None,
                    f"base trace revision {base_revision!r} != integrated oracle {integrated_base!r}",
                ).as_dict(),
            }
    return result


def command_compare_local(args: argparse.Namespace) -> int:
    result = local_report(TraceBundle(args.left), TraceBundle(args.right), args.mode)
    text = canonical_json(result) + "\n"
    if args.report:
        args.report.write_text(text, encoding="ascii")
    sys.stdout.write(text)
    return 0 if result["status"] == "BRINGUP PASS" else 1


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
    local_parser = subparsers.add_parser("compare-local")
    local_parser.add_argument("mode", choices=("self-consistency", "base-regression"))
    local_parser.add_argument("left", type=Path)
    local_parser.add_argument("right", type=Path)
    local_parser.add_argument("--report", type=Path)
    local_parser.set_defaults(func=command_compare_local)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
