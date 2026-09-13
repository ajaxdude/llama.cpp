#!/usr/bin/env python3

import importlib.util
import io
import json
import struct
import sys
import tempfile
import unittest
from unittest import mock
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
import verify_ds4_anchors

trace.APPROVED_WATCHDOGS[trace.WATCHDOG_SCRIPT_SHA256] = trace.WATCHDOG_REVISION

WATCHDOG_EVENTS = [
    {
        "timestamp": "1970-01-01T00:00:01.000Z",
        "event": "preflight",
        "total_bytes": 128 * 1024 * 1024 * 1024,
        "available_bytes": 64 * 1024 * 1024 * 1024,
        "used_bytes": 64 * 1024 * 1024 * 1024,
        "swap_entries": 0,
        "peak_used_bytes": 64 * 1024 * 1024 * 1024,
        "child_pid": None,
        "child_status": "not_started",
        "child_returncode": None,
        "process_group_id": None,
        "process_group_status": "not_created",
        "threshold_reason": "none",
        "soft_bytes": trace.SOFT_MEMORY_LIMIT,
        "emergency_bytes": trace.WATCHDOG_EMERGENCY_LIMIT,
        "strict_ceiling_bytes": trace.STRICT_MEMORY_LIMIT,
    },
    {
        "timestamp": "1970-01-01T00:00:01.000Z",
        "event": "child_started",
        "total_bytes": 128 * 1024 * 1024 * 1024,
        "available_bytes": 64 * 1024 * 1024 * 1024,
        "used_bytes": 64 * 1024 * 1024 * 1024,
        "swap_entries": 0,
        "peak_used_bytes": 64 * 1024 * 1024 * 1024,
        "child_pid": 456,
        "child_status": "running",
        "child_returncode": None,
        "process_group_id": 455,
        "process_group_status": "active",
        "threshold_reason": "none",
        "command": ["python3", "run_matrix.py"],
    },
]
WATCHDOG_JSONL = "".join(
    json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n"
    for event in WATCHDOG_EVENTS
).encode("ascii")
WATCHDOG_JSONL_SHA256 = trace.sha256_bytes(WATCHDOG_JSONL)

ACCELERATOR_ATTESTATION = {
    "format": "dsv41-accelerator-attestation",
    "version": 2,
    "runtime_kind": "strix-rocm",
    "platform": "linux",
    "backend": "ROCm",
    "backend_device": "ROCm0",
    "backend_description": "AMD Radeon Graphics",
    "pci_device_id": "0000:c1:00.0",
    "kfd_node": "1",
    "gpu_id": 1234,
    "gfx_target_version": 110501,
    "architecture": "gfx1151",
    "source": "linux-kfd-sysfs",
}

METAL_ACCELERATOR_ATTESTATION = {
    "format": "dsv41-accelerator-attestation",
    "version": 2,
    "runtime_kind": "apple-metal",
    "platform": "macos",
    "backend": "Metal",
    "backend_device": "Metal0",
    "backend_description": "Apple M3 Ultra",
    "architecture": "Apple M3 Ultra",
    "metal_registry_id": 0x12345678,
    "recommended_max_working_set_bytes": 256 * 1024 * 1024 * 1024,
    "unified_memory": True,
    "source": "metal-device-query",
}


def storage_record(path: str) -> dict[str, object]:
    model_storage = path.startswith("/mnt/models")
    mount_point = "/mnt/models" if model_storage else "/home"
    source = "/dev/nvme1n1" if model_storage else "/dev/nvme0n1p3[/home]"
    device_number = "259:0" if model_storage else "259:3"
    nvme_device = "nvme1n1" if model_storage else "nvme0n1"
    return {
        "format": "dsv41-storage-attestation",
        "version": 2,
        "runtime_kind": "strix-rocm",
        "platform": "linux",
        "storage_kind": "linux-nvme",
        "resolved_path": path,
        "existing_path": path,
        "mount_point": mount_point,
        "filesystem_type": "xfs" if model_storage else "btrfs",
        "mount_source": source,
        "device_number": device_number,
        "block_device_path": f"/sys/devices/pci/block/{nvme_device}",
        "nvme_device": nvme_device,
        "rotational": False,
        "source": "linux-mountinfo-sysfs",
    }


STORAGE_ATTESTATION = {
    "model": storage_record("/mnt/models/model.gguf"),
    "prompt": storage_record("/home/prompt.txt"),
    "output": storage_record("/home"),
    "repository": storage_record("/home/repo"),
    "temporary_directory": storage_record("/home/tmp"),
}

def metal_storage_record(path: str, mount_point: str = "/Users") -> dict[str, object]:
    return {
        "format": "dsv41-storage-attestation",
        "version": 2,
        "runtime_kind": "apple-metal",
        "platform": "macos",
        "storage_kind": "darwin-local-solid-state",
        "resolved_path": path,
        "existing_path": path,
        "mount_point": mount_point,
        "filesystem_type": "apfs",
        "device_identifier": "disk3s1",
        "parent_whole_disk": "disk3",
        "bus_protocol": "Apple Fabric",
        "filesystem_device": 1,
        "internal": True,
        "solid_state": True,
        "source": "diskutil-info-plist",
    }


DS4_STORAGE_ATTESTATION = {
    "model": metal_storage_record("/Users/oracle/model.gguf"),
    "prompt": metal_storage_record("/Users/oracle/prompt.txt"),
    "output": metal_storage_record("/Users/oracle/output"),
    "repository": metal_storage_record("/Users/oracle/repo"),
    "runtime_checkout": metal_storage_record("/Users/oracle/ds4"),
    "temporary_directory": metal_storage_record("/Users/oracle/tmp"),
    "runner_executable": metal_storage_record("/usr/bin/python3", "/"),
    "runner_script": metal_storage_record("/Users/oracle/repo/tools/deepseek-v41-trace/run_ds4.py"),
    "exporter": metal_storage_record("/Users/oracle/bin/ds4-trace"),
}

DS4_HOST_ATTESTATION = {
    "format": "dsv41-host-attestation",
    "version": 1,
    "runtime_kind": "apple-metal",
    "platform": "macos",
    "machine": "arm64",
    "hardware_model": "Mac14,8",
    "os_version": "15.6",
    "memory_bytes": 256 * 1024 * 1024 * 1024,
    "source": "darwin-sysctl",
}

DS4_RUNNER_ATTESTATION = {
    "format": "dsv41-runner-ownership",
    "version": 1,
    "runtime_kind": "apple-metal",
    "source": "python-subprocess",
    "runner_pid": 100,
    "runner_parent_pid": 99,
    "runner_uid": 501,
    "runner_executable": "/usr/bin/python3",
    "runner_executable_sha256": "1" * 64,
    "runner_script": "/Users/oracle/repo/tools/deepseek-v41-trace/run_ds4.py",
    "runner_script_sha256": "2" * 64,
    "exporter_path": "/Users/oracle/bin/ds4-trace",
    "exporter_sha256": "3" * 64,
    "checkout_path": "/Users/oracle/ds4",
    "checkout_revision": trace.DS4_REVISION,
    "command_sha256": "4" * 64,
}

AUDIT_RECORDS = {
    "memory": {
        "created_unix": 1,
        "kind": "memory",
        "environment": {"HIP_LAUNCH_BLOCKING": "1"},
        "data": {"mem_total_bytes": 128, "mem_available_bytes": 64, "mem_used_bytes": 64},
        "storage": STORAGE_ATTESTATION,
        "accelerator": dict(ACCELERATOR_ATTESTATION),
    },
    "swap": {
        "created_unix": 1,
        "kind": "swap",
        "environment": {"HIP_LAUNCH_BLOCKING": "1"},
        "data": {"enabled": False, "entries": []},
    },
    "watchdog": {
        "created_unix": 1,
        "kind": "watchdog",
        "environment": {"HIP_LAUNCH_BLOCKING": "1"},
        "data": {
            "format": trace.WATCHDOG_LEASE_FORMAT,
            "version": trace.WATCHDOG_VERSION,
            "lease_id": "1" * 32,
            "lease_path": "/run/user/123/watchdog.lease",
            "watchdog_pid": 123,
            "watchdog_start_time_ticks": 456,
            "watchdog_command": "python3 /repo/scripts/strix_memory_watchdog.py",
            "watchdog_command_sha256": "7" * 64,
            "watchdog_executable_path": "/usr/bin/python3",
            "watchdog_script_path": "/repo/scripts/strix_memory_watchdog.py",
            "watchdog_script_sha256": trace.WATCHDOG_SCRIPT_SHA256,
            "watchdog_revision": trace.WATCHDOG_REVISION,
            "soft_bytes": trace.SOFT_MEMORY_LIMIT,
            "emergency_bytes": trace.WATCHDOG_EMERGENCY_LIMIT,
            "strict_ceiling_bytes": trace.STRICT_MEMORY_LIMIT,
            "grace_seconds": 30.0,
            "sample_interval_seconds": 1.0,
            "procfs_root": "/proc",
            "guardian_pid": 455,
            "child_pid": 456,
            "child_process_group_id": 455,
            "command": ["python3", "run_matrix.py"],
            "child_command_sha256": trace.sha256_bytes(b'["python3","run_matrix.py"]'),
            "heartbeat_path": "/run/user/123/watchdog.heartbeat",
            "heartbeat_unix": 1,
            "max_heartbeat_age_seconds": 5.0,
            "audit_live_path": "/run/user/123/watchdog.jsonl",
            "audit_device": 1,
            "audit_inode": 2,
            "audit_uid": 1000,
            "audit_mode": 0o600,
            "audit_fd": 3,
            "audit": {
                "path": "",
                "sha256": WATCHDOG_JSONL_SHA256,
                "event_count": len(WATCHDOG_EVENTS),
            },
        },
    },
}

DS4_AUDIT_RECORDS = {
    "memory": {
        "created_unix": 1,
        "kind": "memory",
        "environment": {},
        "data": {
            "mem_total_bytes": DS4_HOST_ATTESTATION["memory_bytes"],
            "mem_available_bytes": 128 * 1024 * 1024 * 1024,
            "mem_used_bytes": 128 * 1024 * 1024 * 1024,
        },
        "storage": DS4_STORAGE_ATTESTATION,
        "accelerator": dict(METAL_ACCELERATOR_ATTESTATION),
        "host": DS4_HOST_ATTESTATION,
    },
    "swap": {
        "created_unix": 1,
        "kind": "swap",
        "environment": {},
        "data": {
            "source": "darwin-sysctl-vm.swapusage",
            "total_bytes": 0,
            "used_bytes": 0,
            "free_bytes": 0,
        },
    },
    "runner": {
        "created_unix": 1,
        "kind": "runner",
        "environment": {},
        "data": DS4_RUNNER_ATTESTATION,
    },
}


def audit_bytes(kind: str, phase: str, runtime: str = "llama.cpp") -> bytes:
    records = DS4_AUDIT_RECORDS if runtime == "ds4" else AUDIT_RECORDS
    record = json.loads(json.dumps(records[kind]))
    if kind == "watchdog":
        record["data"]["audit"]["path"] = f"audits/{phase}/{WATCHDOG_JSONL_SHA256}.jsonl"
    return (json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n").encode("ascii")


def replace_audit_record(root: Path, phase: str, kind: str, record: dict[str, object]) -> None:
    data = (json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n").encode("ascii")
    digest = trace.sha256_bytes(data)
    path = root / "audits" / phase / f"{digest}.json"
    path.write_bytes(data)
    manifest_path = root / trace.MANIFEST_NAME
    manifest_record = json.loads(manifest_path.read_text(encoding="ascii"))
    manifest_record["audits"][phase][kind] = {
        "path": f"audits/{phase}/{digest}.json",
        "sha256": digest,
        "created_unix": record["created_unix"],
    }
    manifest_path.write_text(
        json.dumps(manifest_record, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="ascii",
    )


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
    is_ds4 = runtime == "ds4"
    storage = DS4_STORAGE_ATTESTATION if is_ds4 else STORAGE_ATTESTATION
    audit_kinds = ("memory", "swap", "runner") if is_ds4 else ("memory", "swap", "watchdog")
    result = {
        "runtime": runtime,
        "revision": trace.DS4_REVISION if is_ds4 else "a" * 40,
        "build": (
            {
                "compiler": "clang",
                "target": "arm64-apple-darwin",
                "path": "/Users/oracle/bin/ds4-trace",
                "sha256": "3" * 64,
            }
            if is_ds4
            else {
                "number": 1,
                "info": "test",
                "compiler": "clang",
                "target": "arm64-apple-darwin",
                "path": "/home/repo/build/bin/llama-deepseek-v41-trace",
                "sha256": "3" * 64,
            }
        ),
        "model": {
            "path": storage["model"]["resolved_path"],
            "sha256": trace.MODEL_SHA256,
            "byte_count": 123,
            "architecture": "deepseek41",
        },
        "accelerator": dict(METAL_ACCELERATOR_ATTESTATION if is_ds4 else ACCELERATOR_ATTESTATION),
        "prompt": {
            "path": storage["prompt"]["resolved_path"],
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
                "raw_attention_layers": list(trace.RAW_ATTENTION_LAYERS),
                "raw_attention_width": trace.RAW_ATTENTION_WIDTH,
                "candidate_propagation_layers": [24, 28, 32, 36],
            },
        },
        "paths": {
                label: record["resolved_path"]
                for label, record in storage.items()
        },
        "comparison": {
            "tokens": "exact",
            "engram_rows": "exact",
            "expert_ids": "exact-original-id-space",
            "expert_weights": "byte-identical-f32",
            "attention_candidates": "exact",
            "logits": "byte-identical-f32",
        },
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
        "environment": {
            "system_info": "Linux test system" if runtime == "llama.cpp" else "macOS test system",
            "command": "test command",
        },
        "audits": {
            phase: {
                kind: {
                    "path": f"audits/{phase}/{trace.sha256_bytes(audit_bytes(kind, phase, runtime))}.json",
                    "sha256": trace.sha256_bytes(audit_bytes(kind, phase, runtime)),
                    "created_unix": 1,
                }
                for kind in audit_kinds
            }
            for phase in ("pre", "post")
        },
    }
    if not is_ds4:
        result["config"].update({
            "batch": trace.ADMITTED_BATCH,
            "ubatch": trace.ADMITTED_UBATCH,
            "kv_type_k": "f16",
            "kv_type_v": "f16",
            "flash_attention": True,
            "expert_cache_slots": trace.REQUIRED_EXPERT_SLOTS,
            "expert_cache_bytes": trace.REQUIRED_EXPERT_CACHE_BYTES,
            "device": "ROCm0",
            "device_architecture": "gfx1151",
            "device_pci_id": "0000:c1:00.0",
            "gpu_layers": 99,
            "load_mode": 0,
            "tokenizer_add_bos": True,
            "tokenizer_parse_special": True,
        })
        result["candidate"] = {
            "repository": trace.REPOSITORY,
            "revision": "a" * 40,
            "base_revision": "b" * 40,
            "diff_sha256": "c" * 64,
            "executable_sha256": "3" * 64,
        }
    else:
        result["host"] = dict(DS4_HOST_ATTESTATION)
        result["config"]["prefill_chunk"] = trace.ADMITTED_UBATCH
        result["config"]["device_backend"] = "Metal"
        result["config"]["device_registry_id"] = METAL_ACCELERATOR_ATTESTATION["metal_registry_id"]
    return result


def add_required_events(writer: object, logits: bytes | None = None, prompt: bytes = b"abc") -> None:
    runtime = writer.manifest["runtime"]
    audit_kinds = ("memory", "swap", "runner") if runtime == "ds4" else ("memory", "swap", "watchdog")
    for phase in ("pre", "post"):
        audit_root = writer.root / "audits" / phase
        audit_root.mkdir(parents=True, exist_ok=True)
        for kind in audit_kinds:
            data = audit_bytes(kind, phase, runtime)
            (audit_root / f"{trace.sha256_bytes(data)}.json").write_bytes(data)
        if runtime == "llama.cpp":
            (audit_root / f"{WATCHDOG_JSONL_SHA256}.jsonl").write_bytes(WATCHDOG_JSONL)
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
    for layer in trace.RAW_ATTENTION_LAYERS:
        writer.add_event(
            component="attn.source",
            phase="prefill",
            step=0,
            token_start=0,
            token_count=2,
            layer=layer,
            dtype="i32",
            shape=[trace.RAW_ATTENTION_WIDTH, 2],
            data=struct.pack(
                "<" + "i" * (trace.RAW_ATTENTION_WIDTH * 2),
                *([trace.RAW_ATTENTION_WIDTH + 2] * (trace.RAW_ATTENTION_WIDTH * 2)),
            ),
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
    for layer in trace.RAW_ATTENTION_LAYERS:
        writer.add_event(
            component="attn.source",
            phase="decode",
            step=0,
            token_start=2,
            token_count=1,
            layer=layer,
            dtype="i32",
            shape=[trace.RAW_ATTENTION_WIDTH, 1],
            data=struct.pack(
                "<" + "i" * trace.RAW_ATTENTION_WIDTH,
                *([trace.RAW_ATTENTION_WIDTH + 1] * trace.RAW_ATTENTION_WIDTH),
            ),
        )
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
    for layer in list(range(2, 20)) + list(range(21, 40)):
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


def replace_event_blob(
        root: Path,
        *,
        component: str,
        phase: str,
        layer: int,
        shape: list[int],
        data: bytes) -> None:
    events_path = root / trace.EVENTS_NAME
    events = [json.loads(line) for line in events_path.read_text(encoding="ascii").splitlines()]
    event = next(
        item for item in events
        if item["component"] == component and item["phase"] == phase and item["layer"] == layer
    )
    digest = trace.sha256_bytes(data)
    blob = root / trace.BLOBS_DIR / f"{digest}.bin"
    blob.write_bytes(data)
    event.update({
        "shape": shape,
        "byte_count": len(data),
        "sha256": digest,
        "blob": f"{trace.BLOBS_DIR}/{digest}.bin",
    })
    events_path.write_text(
        "".join(json.dumps(item, sort_keys=True, separators=(",", ":")) + "\n" for item in events),
        encoding="ascii",
    )


class TraceFormatTests(unittest.TestCase):
    def setUp(self) -> None:
        self._require_nvme_path = preflight.require_nvme_path
        preflight.require_nvme_path = lambda path, label, **kwargs: preflight.resolved(path)

    def tearDown(self) -> None:
        preflight.require_nvme_path = self._require_nvme_path

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
                ubatch=trace.ADMITTED_UBATCH,
                device="ROCm0",
                gpu_layers=99,
                expert_cache_slots=trace.REQUIRED_EXPERT_SLOTS,
                expert_cache_mib=trace.REQUIRED_EXPERT_CACHE_MIB,
            )
            command = run_llama.build_command(args, exporter, output)
            self.assertEqual(command[command.index("-bf") + 1], str(prompt.resolve()))
            self.assertEqual(command[command.index("--device") + 1], "ROCm0")
            self.assertEqual(command[command.index("--load-mode") + 1], "none")
            self.assertNotIn("-f", command)

    def test_rejects_unadmitted_expert_cache_configuration(self) -> None:
        invalid = Namespace(
            batch=trace.ADMITTED_BATCH,
            ubatch=512,
            device="ROCm0",
            gpu_layers=99,
            expert_cache_slots=8,
            expert_cache_mib=4096,
        )
        with self.assertRaisesRegex(preflight.PreflightError, "admitted ubatch 32"):
            run_llama.validate_runtime_config(invalid)

        invalid.ubatch = trace.ADMITTED_UBATCH
        with self.assertRaisesRegex(preflight.PreflightError, "192 expert cache slots"):
            run_llama.validate_runtime_config(invalid)

        invalid.expert_cache_slots = trace.REQUIRED_EXPERT_SLOTS
        with self.assertRaisesRegex(preflight.PreflightError, "76441190400 expert cache bytes"):
            run_llama.validate_runtime_config(invalid)

        valid = Namespace(
            batch=trace.ADMITTED_BATCH,
            ubatch=trace.ADMITTED_UBATCH,
            device="ROCm0",
            gpu_layers=99,
            expert_cache_slots=trace.REQUIRED_EXPERT_SLOTS,
            expert_cache_mib=trace.REQUIRED_EXPERT_CACHE_MIB,
        )
        run_llama.validate_runtime_config(valid)
        self.assertEqual(trace.REQUIRED_EXPERT_CACHE_BYTES, 76_441_190_400)
        self.assertEqual(trace.REQUIRED_EXPERT_CACHE_MIB, 72_900)

    def test_accelerator_attestation_rejects_wrong_missing_and_spoofed_architecture(self) -> None:
        valid = dict(ACCELERATOR_ATTESTATION)
        self.assertEqual(
            run_llama.validate_accelerator_attestation(valid),
            valid,
        )
        for key, value, message in (
                ("architecture", "gfx1100", "architecture mismatch"),
                ("architecture", None, "fields are invalid"),
                ("backend_device", "ROCm1", "backend_device mismatch"),
                ("gfx_target_version", 110500, "gfx_target_version mismatch"),
                ("source", "environment", "source mismatch")):
            invalid = dict(valid)
            if value is None:
                del invalid[key]
            else:
                invalid[key] = value
            with self.assertRaisesRegex(preflight.PreflightError, message):
                run_llama.validate_accelerator_attestation(invalid)

        spoofed = dict(valid)
        spoofed["gfx_target_version"] = 110500
        spoofed["architecture"] = "gfx1151"
        with self.assertRaisesRegex(preflight.PreflightError, "gfx_target_version mismatch"):
            run_llama.validate_accelerator_attestation(spoofed)

    def test_accelerator_query_fails_closed(self) -> None:
        valid_result = run_llama.subprocess.CompletedProcess(
            ["exporter"], 0, json.dumps(ACCELERATOR_ATTESTATION), "")
        with mock.patch.object(run_llama.subprocess, "run", return_value=valid_result):
            self.assertEqual(
                run_llama.query_accelerator_attestation(Path("/exporter"), "ROCm0"),
                ACCELERATOR_ATTESTATION,
            )

        failed_result = run_llama.subprocess.CompletedProcess(["exporter"], 1, "", "query failed")
        with mock.patch.object(run_llama.subprocess, "run", return_value=failed_result):
            with self.assertRaisesRegex(preflight.PreflightError, "query failed"):
                run_llama.query_accelerator_attestation(Path("/exporter"), "ROCm0")

    def test_metal_accelerator_attestation_is_runtime_specific(self) -> None:
        valid = dict(METAL_ACCELERATOR_ATTESTATION)
        self.assertEqual(run_ds4.validate_accelerator_attestation(valid), valid)
        for mutation, message in (
                ({"runtime_kind": "strix-rocm"}, "runtime_kind mismatch"),
                ({"platform": "linux"}, "platform mismatch"),
                ({"backend": "ROCm"}, "backend mismatch"),
                ({"source": "environment"}, "source mismatch"),
                ({"pci_device_id": "0000:c1:00.0"}, "fields are invalid")):
            invalid = dict(valid)
            invalid.update(mutation)
            with self.assertRaisesRegex(preflight.PreflightError, message):
                run_ds4.validate_accelerator_attestation(invalid)

    def test_metal_accelerator_query_rejects_duplicate_keys(self) -> None:
        duplicate = json.dumps(METAL_ACCELERATOR_ATTESTATION).replace(
            '"backend": "Metal"',
            '"backend": "Metal", "backend": "Metal"',
        )
        result = run_ds4.subprocess.CompletedProcess(["exporter"], 0, duplicate, "")
        with mock.patch.object(run_ds4.subprocess, "run", return_value=result):
            with self.assertRaisesRegex(preflight.PreflightError, "duplicate JSON key"):
                run_ds4.query_accelerator_attestation(Path("/exporter"), "Metal0")

    def test_nvme_attestation_uses_mount_and_block_ancestry(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            xfs = root / "mnt" / "models"
            btrfs = root / "home"
            rotating = root / "rotating"
            ram = root / "ram"
            network = root / "network"
            missing = root / "missing"
            for directory in (xfs, btrfs, rotating, ram, network, missing):
                directory.mkdir(parents=True)
                (directory / "data").write_bytes(b"x")

            sys_root = root / "sys"
            nvme1 = sys_root / "devices" / "pci" / "block" / "nvme1n1"
            nvme0 = sys_root / "devices" / "pci" / "block" / "nvme0n1"
            nvme0p3 = nvme0 / "nvme0n1p3"
            sdb = sys_root / "devices" / "pci" / "block" / "sdb"
            sdb1 = sdb / "sdb1"
            for disk, rotational in ((nvme1, "0\n"), (nvme0, "0\n"), (sdb, "1\n")):
                (disk / "queue").mkdir(parents=True)
                (disk / "queue" / "rotational").write_text(rotational, encoding="ascii")
            nvme0p3.mkdir()
            sdb1.mkdir()
            dev_block = sys_root / "dev" / "block"
            dev_block.mkdir(parents=True)
            (dev_block / "259:0").symlink_to(nvme1, target_is_directory=True)
            (dev_block / "259:3").symlink_to(nvme0p3, target_is_directory=True)
            (dev_block / "8:17").symlink_to(sdb1, target_is_directory=True)
            class_block = sys_root / "class" / "block"
            (class_block / "nvme0n1p3").mkdir(parents=True)
            (class_block / "nvme0n1p3" / "dev").write_text("259:3\n", encoding="ascii")

            mountinfo = root / "mountinfo"
            mountinfo.write_text(
                f"1 0 259:0 / {xfs.resolve()} rw - xfs /dev/nvme1n1 rw\n"
                f"2 0 0:35 /home {btrfs.resolve()} rw - btrfs /dev/nvme0n1p3[/home] rw\n"
                f"3 0 8:17 / {rotating.resolve()} rw - ext4 /dev/sdb1 rw\n"
                f"4 0 0:42 / {ram.resolve()} rw - tmpfs tmpfs rw\n"
                f"5 0 0:43 / {network.resolve()} rw - nfs server:/share rw\n"
                f"6 0 240:1 / {missing.resolve()} rw - ext4 /dev/missing rw\n",
                encoding="ascii",
            )

            xfs_result = preflight.storage_attestation(
                xfs / "data",
                "xfs",
                mountinfo_path=mountinfo,
                sys_dev_block_root=dev_block,
                sys_class_block_root=class_block,
            )
            self.assertEqual(xfs_result["filesystem_type"], "xfs")
            self.assertEqual(xfs_result["nvme_device"], "nvme1n1")

            btrfs_result = preflight.storage_attestation(
                btrfs / "new" / "trace",
                "btrfs",
                mountinfo_path=mountinfo,
                sys_dev_block_root=dev_block,
                sys_class_block_root=class_block,
            )
            self.assertEqual(btrfs_result["filesystem_type"], "btrfs")
            self.assertEqual(btrfs_result["device_number"], "259:3")
            self.assertEqual(btrfs_result["existing_path"], str(btrfs.resolve()))
            self.assertEqual(btrfs_result["nvme_device"], "nvme0n1")

            for path, message in (
                    (rotating / "data", "non-rotational"),
                    (ram / "data", "local block device"),
                    (network / "data", "local block device"),
                    (missing / "data", "cannot be resolved")):
                with self.assertRaisesRegex(preflight.PreflightError, message):
                    preflight.storage_attestation(
                        path,
                        "invalid",
                        mountinfo_path=mountinfo,
                        sys_dev_block_root=dev_block,
                        sys_class_block_root=class_block,
                    )

            (xfs / "escape").symlink_to(ram, target_is_directory=True)
            with self.assertRaisesRegex(preflight.PreflightError, "local block device"):
                preflight.storage_attestation(
                    xfs / "escape" / "data",
                    "symlink escape",
                    mountinfo_path=mountinfo,
                    sys_dev_block_root=dev_block,
                    sys_class_block_root=class_block,
                )
            with self.assertRaisesRegex(preflight.PreflightError, "/mnt/bigspace"):
                preflight.storage_attestation(
                    Path("/mnt/bigspace/model.gguf"),
                    "forbidden",
                    mountinfo_path=mountinfo,
                    sys_dev_block_root=dev_block,
                    sys_class_block_root=class_block,
                )
            forbidden = root / "forbidden"
            forbidden.mkdir()
            (forbidden / "escape").symlink_to(xfs, target_is_directory=True)
            with self.assertRaisesRegex(preflight.PreflightError, "must not use"):
                preflight.storage_attestation(
                    forbidden / "escape" / "model.gguf",
                    "forbidden symlink",
                    mountinfo_path=mountinfo,
                    sys_dev_block_root=dev_block,
                    sys_class_block_root=class_block,
                    forbidden_root=forbidden,
                )

    def test_darwin_storage_and_host_preflight_are_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for name in ("repo", "ds4", "tmp", "output"):
                (root / name).mkdir()
            model = root / "model.gguf"
            prompt = root / "prompt.txt"
            model.write_bytes(b"model")
            prompt.write_bytes(b"prompt")
            runner_executable = root / "python3"
            runner_script = root / "repo" / "tools" / "deepseek-v41-trace" / "run_ds4.py"
            exporter = root / "ds4-trace"
            runner_script.parent.mkdir(parents=True)
            runner_executable.write_bytes(b"python")
            runner_script.write_bytes(b"runner")
            exporter.write_bytes(b"exporter")

            def disk_info(path: Path) -> dict[str, object]:
                return {
                    "MountPoint": str(root.resolve()),
                    "FilesystemType": "apfs",
                    "DeviceIdentifier": "disk3s1",
                    "ParentWholeDisk": "disk3",
                    "BusProtocol": "Apple Fabric",
                    "Internal": True,
                    "SolidState": True,
                    "VolumeNetwork": False,
                    "DiskImage": False,
                }

            def command_text(*args: str) -> str:
                commands = {
                    ("sysctl", "-n", "hw.memsize"): str(256 * 1024 * 1024 * 1024),
                    ("sysctl", "-n", "hw.model"): "Mac14,8",
                    ("sysctl", "-n", "kern.osproductversion"): "15.6",
                    ("sysctl", "-n", "vm.swapusage"): "total = 0.00M used = 0.00M free = 0.00M (encrypted)",
                    ("vm_stat",): (
                        "Mach Virtual Memory Statistics: (page size of 16384 bytes)\n"
                        "Pages free: 1000000.\n"
                        "Pages inactive: 1000000.\n"
                        "Pages speculative: 1000000.\n"
                    ),
                    ("ps", "-axo", "pid=,ppid=,command="): "",
                }
                return commands[args]

            runner = dict(DS4_RUNNER_ATTESTATION)
            runner["runner_executable"] = str(runner_executable.resolve())
            runner["runner_executable_sha256"] = trace.sha256_file(runner_executable)
            runner["runner_script"] = str(runner_script.resolve())
            runner["runner_script_sha256"] = trace.sha256_file(runner_script)
            runner["exporter_path"] = str(exporter.resolve())
            runner["exporter_sha256"] = trace.sha256_file(exporter)
            runner["checkout_path"] = str((root / "ds4").resolve())
            with mock.patch.dict(preflight.os.environ, {"TMPDIR": str(root / "tmp")}, clear=True):
                result = preflight.run_oracle_preflight(
                    model=model,
                    prompt=prompt,
                    output=root / "output",
                    repo=root / "repo",
                    checkout=root / "ds4",
                    busy_patterns=[],
                    accelerator=dict(METAL_ACCELERATOR_ATTESTATION),
                    runner=runner,
                    disk_info=disk_info,
                    command_text=command_text,
                    system="Darwin",
                    machine="arm64",
                )
            self.assertEqual(result["runtime_kind"], "apple-metal")
            self.assertEqual(result["storage"]["model"]["storage_kind"], "darwin-local-solid-state")
            self.assertEqual(result["host"]["memory_bytes"], 256 * 1024 * 1024 * 1024)

            for mutation, message in (
                    ({"Internal": False}, "internal non-rotational"),
                    ({"SolidState": False}, "internal non-rotational"),
                    ({"VolumeNetwork": True}, "local storage"),
                    ({"DiskImage": True}, "local storage"),
                    ({"BusProtocol": "Network"}, "bus protocol")):
                def invalid_info(path: Path, mutation: dict[str, object] = mutation) -> dict[str, object]:
                    result = disk_info(path)
                    result.update(mutation)
                    return result

                with self.assertRaisesRegex(preflight.PreflightError, message):
                    preflight.darwin_storage_attestation(
                        model,
                        "model",
                        disk_info=invalid_info,
                    )

            with self.assertRaisesRegex(preflight.PreflightError, "macOS on arm64"):
                preflight.darwin_host_and_memory_audit(
                    command_text=command_text,
                    system="Linux",
                    machine="x86_64",
                )

    def test_darwin_storage_queries_the_containing_mount(self) -> None:
        path = Path("/Users/test/model.gguf")
        disk_info = {
            "DeviceIdentifier": "disk3s5",
            "ParentWholeDisk": "disk3",
            "BusProtocol": "Apple Fabric",
            "Internal": True,
            "SolidState": True,
        }

        def check_output(command, **_kwargs):
            if command == ["df", "-P", str(path)]:
                return (
                    "Filesystem 512-blocks Used Available Capacity Mounted on\n"
                    "/dev/disk3s5 100 10 90 10% /System/Volumes/Data\n"
                )
            self.assertEqual(command, ["diskutil", "info", "-plist", "/System/Volumes/Data"])
            return preflight.plistlib.dumps(disk_info)

        with mock.patch.object(preflight.subprocess, "check_output", side_effect=check_output):
            result = preflight._diskutil_info(path)
        self.assertEqual(result["_dsv41_mount_point"], "/System/Volumes/Data")
        self.assertEqual({key: value for key, value in result.items() if not key.startswith("_")}, disk_info)

    def test_preflight_requires_explicit_nvme_tmpdir(self) -> None:
        with mock.patch.object(
                preflight,
                "storage_attestation",
                return_value=storage_record("/home/test")):
            with mock.patch.dict(preflight.os.environ, {"HIP_LAUNCH_BLOCKING": "1"}, clear=True):
                with self.assertRaisesRegex(preflight.PreflightError, "TMPDIR is required"):
                    preflight.run_strix_preflight(
                        model=Path("/home/model.gguf"),
                        prompt=Path("/home/prompt.txt"),
                        output=Path("/home/trace"),
                        repo=Path("/home/repo"),
                        busy_patterns=[],
                    )
            with tempfile.TemporaryDirectory() as temp:
                missing = Path(temp) / "missing"
                with mock.patch.dict(
                        preflight.os.environ,
                        {"HIP_LAUNCH_BLOCKING": "1", "TMPDIR": str(missing)},
                        clear=True):
                    with self.assertRaisesRegex(preflight.PreflightError, "existing writable directory"):
                        preflight.run_strix_preflight(
                            model=Path("/home/model.gguf"),
                            prompt=Path("/home/prompt.txt"),
                            output=Path("/home/trace"),
                            repo=Path("/home/repo"),
                            busy_patterns=[],
                        )
                actual = Path(temp) / "actual"
                actual.mkdir()
                link = Path(temp) / "link"
                link.symlink_to(actual, target_is_directory=True)
                with mock.patch.dict(
                        preflight.os.environ,
                        {"HIP_LAUNCH_BLOCKING": "1", "TMPDIR": str(link)},
                        clear=True):
                    with self.assertRaisesRegex(preflight.PreflightError, "must not be a symlink"):
                        preflight.run_strix_preflight(
                            model=Path("/home/model.gguf"),
                            prompt=Path("/home/prompt.txt"),
                            output=Path("/home/trace"),
                            repo=Path("/home/repo"),
                            busy_patterns=[],
                        )

    def test_watchdog_lease_rejects_arbitrary_heartbeat_process(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            repo = root / "repo"
            script = repo / "scripts" / "strix_memory_watchdog.py"
            script.parent.mkdir(parents=True)
            script.write_text("#!/usr/bin/env python3\n", encoding="ascii")

            procfs = root / "proc"
            watchdog_pid = 123
            child_pid = 456
            current_pid = 789
            for pid in (watchdog_pid, child_pid, current_pid):
                (procfs / str(pid)).mkdir(parents=True)

            def stat(pid: int, parent: int, start: int) -> str:
                fields = ["S", str(parent), *(["0"] * 17), str(start)]
                return f"{pid} (test) " + " ".join(fields) + "\n"

            (procfs / str(watchdog_pid) / "stat").write_text(
                stat(watchdog_pid, 1, 1000), encoding="ascii")
            (procfs / str(child_pid) / "stat").write_text(
                stat(child_pid, watchdog_pid, 2000), encoding="ascii")
            (procfs / str(current_pid) / "stat").write_text(
                stat(current_pid, child_pid, 3000), encoding="ascii")

            watchdog_command = (
                b"python3\0" + str(script.resolve()).encode("ascii") +
                b"\0--soft-gib\0" + b"116\0--emergency-gib\0" + b"118\0"
            )
            child_command = b"python3\0run_matrix.py\0"
            (procfs / str(watchdog_pid) / "cmdline").write_bytes(watchdog_command)
            (procfs / str(child_pid) / "cmdline").write_bytes(child_command)
            (procfs / str(watchdog_pid) / "cwd").symlink_to(repo)

            heartbeat = root / "heartbeat.json"
            audit = root / "watchdog.jsonl"
            lease = root / "lease.json"
            lease_id = "1" * 32
            heartbeat.write_text(json.dumps({
                "format": preflight.WATCHDOG_HEARTBEAT_FORMAT,
                "version": preflight.WATCHDOG_VERSION,
                "lease_id": lease_id,
                "sequence": 1,
                "state": "active",
                "updated_at": "1970-01-01T00:01:40.000Z",
                "updated_monotonic_ns": 1,
                "watchdog_pid": watchdog_pid,
                "watchdog_start_time_ticks": 1000,
                "child_pid": child_pid,
                "child_process_group_id": child_pid,
                "sample": {},
            }), encoding="ascii")
            child_argv = ["python3", "run_matrix.py"]
            watchdog_events = json.loads(json.dumps(WATCHDOG_EVENTS))
            watchdog_events[1]["child_pid"] = child_pid
            watchdog_events[1]["process_group_id"] = child_pid
            watchdog_events[1]["command"] = child_argv
            audit.write_text(
                "".join(json.dumps(event) + "\n" for event in watchdog_events),
                encoding="ascii",
            )
            lease_record = {
                "format": preflight.WATCHDOG_LEASE_FORMAT,
                "version": preflight.WATCHDOG_VERSION,
                "lease_id": lease_id,
                "state": "active",
                "watchdog_pid": watchdog_pid,
                "watchdog_start_time_ticks": 1000,
                "watchdog_command_sha256": preflight.sha256_bytes(watchdog_command),
                "watchdog_script_path": str(script.resolve()),
                "watchdog_script_sha256": preflight.sha256_bytes(script.read_bytes()),
                "soft_bytes": preflight.SOFT_MEMORY_LIMIT,
                "emergency_bytes": preflight.WATCHDOG_EMERGENCY_LIMIT,
                "strict_ceiling_bytes": preflight.STRICT_MEMORY_LIMIT,
                "procfs_root": "/proc",
                "child_pid": child_pid,
                "child_process_group_id": child_pid,
                "command": child_argv,
                "child_command_sha256": preflight.sha256_bytes(child_command),
                "heartbeat_path": str(heartbeat),
                "max_heartbeat_age_seconds": 5.0,
                "audit_path": str(audit),
            }
            lease_record["child_command_sha256"] = preflight.sha256_bytes(
                json.dumps(child_argv, ensure_ascii=True, separators=(",", ":")).encode("utf-8"))
            lease.write_text(json.dumps(lease_record), encoding="ascii")
            environment = {
                preflight.WATCHDOG_LEASE_ENV: str(lease),
                preflight.WATCHDOG_HEARTBEAT_ENV: str(heartbeat),
                preflight.WATCHDOG_AUDIT_ENV: str(audit),
                preflight.WATCHDOG_MAX_AGE_ENV: "5.0",
            }
            result = preflight.watchdog_audit(
                repo,
                environment=environment,
                procfs_root=procfs,
                current_pid=current_pid,
                current_pgid=child_pid,
                getpgid=lambda pid: child_pid,
                now=100,
                monotonic_ns=lambda: 1_000_000_001,
                timeout_seconds=0,
            )
            self.assertEqual(result["child_process_group_id"], child_pid)

            arbitrary = root / "arbitrary-heartbeat.py"
            arbitrary.write_text("#!/usr/bin/env python3\n", encoding="ascii")
            arbitrary_command = b"python3\0" + str(arbitrary.resolve()).encode("ascii") + b"\0"
            (procfs / str(watchdog_pid) / "cmdline").write_bytes(arbitrary_command)
            lease_record["watchdog_command_sha256"] = preflight.sha256_bytes(arbitrary_command)
            lease.write_text(json.dumps(lease_record), encoding="ascii")
            with self.assertRaisesRegex(preflight.PreflightError, "candidate repository script"):
                preflight.watchdog_audit(
                    repo,
                    environment=environment,
                    procfs_root=procfs,
                    current_pid=current_pid,
                    current_pgid=child_pid,
                    getpgid=lambda pid: child_pid,
                    now=100,
                    monotonic_ns=lambda: 1_000_000_001,
                    timeout_seconds=0,
                )

    def test_canonical_watchdog_validation_is_pinned_and_delegated(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            repo = root / "repo"
            script = repo / "scripts" / "strix_memory_watchdog.py"
            script.parent.mkdir(parents=True)
            script.write_text("# fixture\n", encoding="ascii")
            lease = root / "lease.json"
            heartbeat = root / "heartbeat.json"
            audit = root / "audit.jsonl"
            lease.write_text("{}\n", encoding="ascii")
            heartbeat.write_text(
                json.dumps({"updated_at": "1970-01-01T00:00:01.000Z"}) + "\n",
                encoding="ascii",
            )
            audit.write_bytes(WATCHDOG_JSONL)
            environment = {
                preflight.WATCHDOG_LEASE_ENV: str(lease),
                preflight.WATCHDOG_HEARTBEAT_ENV: str(heartbeat),
                preflight.WATCHDOG_AUDIT_ENV: str(audit),
                preflight.WATCHDOG_MAX_AGE_ENV: "5",
            }

            class FakeLeaseError(RuntimeError):
                pass

            class FakeWatchdog:
                LeaseValidationError = FakeLeaseError
                calls = []

                @classmethod
                def validate_active_lease(cls, lease_path: Path, **kwargs: object) -> dict[str, object]:
                    cls.calls.append((lease_path, kwargs))
                    return {
                        **AUDIT_RECORDS["watchdog"]["data"],
                        "audit_path": str(audit),
                        "heartbeat_path": str(heartbeat),
                        "child_pid": 456,
                    }

                @staticmethod
                def start_process_group_lease_guard(*args: object, **kwargs: object) -> None:
                    raise AssertionError("descendant validation must not start the direct-child guard")

            original_sha256 = preflight.WATCHDOG_SCRIPT_SHA256
            original_approved = dict(preflight.APPROVED_WATCHDOGS)
            try:
                preflight.WATCHDOG_SCRIPT_SHA256 = preflight.sha256_bytes(script.read_bytes())
                preflight.APPROVED_WATCHDOGS[preflight.WATCHDOG_SCRIPT_SHA256] = (
                    preflight.WATCHDOG_REVISION)
                result = preflight.watchdog_audit(
                    repo,
                    environment=environment,
                    procfs_root=Path("/proc"),
                    current_pid=789,
                    watchdog_module=FakeWatchdog,
                    monotonic=lambda: 2.0,
                    sleeper=lambda _: None,
                )
            finally:
                preflight.APPROVED_WATCHDOGS.clear()
                preflight.APPROVED_WATCHDOGS.update(original_approved)
                preflight.WATCHDOG_SCRIPT_SHA256 = original_sha256
            self.assertEqual(result["watchdog_revision"], preflight.WATCHDOG_REVISION)
            self.assertEqual(len(FakeWatchdog.calls), 1)
            kwargs = FakeWatchdog.calls[0][1]
            self.assertEqual(kwargs["expected_soft_bytes"], trace.SOFT_MEMORY_LIMIT)
            self.assertEqual(kwargs["expected_emergency_bytes"], trace.WATCHDOG_EMERGENCY_LIMIT)
            self.assertEqual(kwargs["expected_procfs_root"], Path("/proc"))

    def test_approved_watchdog_is_exact(self) -> None:
        expected = {
            "d2781a25f978dd2bc14fc113079aa2dbf513aa157b44da9d0d51d750daa6c94f":
                "778db6f50eae04e6c232c69b9575bdbd0747962b",
        }
        self.assertEqual(preflight.APPROVED_WATCHDOGS, expected)
        self.assertEqual(trace.APPROVED_WATCHDOGS, expected)
        self.assertEqual(preflight.WATCHDOG_VERSION, 2)
        self.assertEqual(trace.WATCHDOG_VERSION, 2)

    def test_workload_scan_ignores_guarded_process_ancestry(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            procfs = Path(temp)

            def add_process(pid: int, parent: int, command: bytes) -> None:
                process = procfs / str(pid)
                process.mkdir()
                process.joinpath("stat").write_text(
                    f"{pid} (test) S {parent} " + " ".join(["0"] * 18) + "\n",
                    encoding="ascii",
                )
                process.joinpath("cmdline").write_bytes(command)

            add_process(90, 1, b"python3\0scripts/strix_memory_watchdog.py\0DeepSeek-V4.1\0")
            add_process(100, 90, b"python3\0run_matrix.py\0--ds4-checkout\0/home/papa/src/ds4-v41\0")
            add_process(200, 1, b"/tmp/ds4-v41-worker\0")
            self.assertEqual(
                preflight.matching_workloads(
                    ["ds4-v41", "DeepSeek-V4.1"],
                    procfs_root=procfs,
                    current_pid=100,
                ),
                [{"pid": 200, "command": "/tmp/ds4-v41-worker"}],
            )

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

    def test_rejects_unattested_accelerator_identity(self) -> None:
        for key, value, message in (
                ("architecture", "gfx1100", "architecture mismatch"),
                ("gfx_target_version", 110500, "gfx_target_version mismatch"),
                ("pci_device_id", "ROCm0", "PCI identity")):
            with tempfile.TemporaryDirectory() as temp:
                root = Path(temp) / "trace"
                trace_manifest = manifest()
                trace_manifest["accelerator"][key] = value
                with trace.TraceBundleWriter(root, trace_manifest) as writer:
                    add_required_events(writer)
                with self.assertRaisesRegex(trace.TraceError, message):
                    trace.TraceBundle(root)

    def test_accepts_truthful_metal_vs_strix_bundles(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            left = Path(temp) / "ds4"
            right = Path(temp) / "llama"
            with trace.TraceBundleWriter(left, manifest("ds4")) as writer:
                add_required_events(writer)
            with trace.TraceBundleWriter(right, manifest("llama.cpp")) as writer:
                add_required_events(writer)
            left_bundle = trace.TraceBundle(left)
            right_bundle = trace.TraceBundle(right)
            self.assertNotEqual(
                left_bundle.manifest["accelerator"]["architecture"],
                right_bundle.manifest["accelerator"]["architecture"],
            )
            result = trace.report(left_bundle, right_bundle)
            self.assertEqual(result["status"], "TARGET PASS")

    def test_rejects_cross_runtime_attestation_substitution(self) -> None:
        for runtime, accelerator, message in (
                ("ds4", ACCELERATOR_ATTESTATION, "ds4 accelerator attestation fields"),
                ("llama.cpp", METAL_ACCELERATOR_ATTESTATION, "llama.cpp accelerator attestation fields")):
            with self.subTest(runtime=runtime), tempfile.TemporaryDirectory() as temp:
                root = Path(temp) / "trace"
                trace_manifest = manifest(runtime)
                trace_manifest["accelerator"] = dict(accelerator)
                with trace.TraceBundleWriter(root, trace_manifest) as writer:
                    add_required_events(writer)
                with self.assertRaisesRegex(trace.TraceError, message):
                    trace.TraceBundle(root)

        for runtime, records, replacement, message in (
                (
                    "ds4",
                    DS4_AUDIT_RECORDS,
                    storage_record("/mnt/models/model.gguf"),
                    "ds4 storage attestation fields",
                ),
                (
                    "llama.cpp",
                    AUDIT_RECORDS,
                    metal_storage_record("/Users/oracle/model.gguf"),
                    "llama.cpp storage attestation fields",
                )):
            with self.subTest(runtime=runtime), tempfile.TemporaryDirectory() as temp:
                root = Path(temp) / "trace"
                with trace.TraceBundleWriter(root, manifest(runtime)) as writer:
                    add_required_events(writer)
                record = json.loads(json.dumps(records["memory"]))
                record["storage"]["model"] = replacement
                replace_audit_record(root, "pre", "memory", record)
                with self.assertRaisesRegex(trace.TraceError, message):
                    trace.TraceBundle(root)

    def test_rejects_ds4_accelerator_audit_removal(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "trace"
            with trace.TraceBundleWriter(root, manifest("ds4")) as writer:
                add_required_events(writer)
            for phase in ("pre", "post"):
                record = json.loads(json.dumps(DS4_AUDIT_RECORDS["memory"]))
                del record["accelerator"]
                replace_audit_record(root, phase, "memory", record)
            with self.assertRaisesRegex(trace.TraceError, "missing accelerator"):
                trace.TraceBundle(root)

    def test_rejects_cross_runtime_host_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "trace"
            trace_manifest = manifest("llama.cpp")
            trace_manifest["host"] = dict(DS4_HOST_ATTESTATION)
            with trace.TraceBundleWriter(root, trace_manifest) as writer:
                add_required_events(writer)
            with self.assertRaisesRegex(trace.TraceError, "unexpected host"):
                trace.TraceBundle(root)

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "trace"
            trace_manifest = manifest("ds4")
            trace_manifest["host"]["runtime_kind"] = "strix-rocm"
            with trace.TraceBundleWriter(root, trace_manifest) as writer:
                add_required_events(writer)
            with self.assertRaisesRegex(trace.TraceError, "host runtime_kind mismatch"):
                trace.TraceBundle(root)

    def test_rejects_storage_audit_path_substitution(self) -> None:
        for runtime, records, different in (
                ("llama.cpp", AUDIT_RECORDS, "/mnt/models/different.gguf"),
                ("ds4", DS4_AUDIT_RECORDS, "/Users/oracle/different.gguf")):
            with self.subTest(runtime=runtime), tempfile.TemporaryDirectory() as temp:
                root = Path(temp) / "trace"
                with trace.TraceBundleWriter(root, manifest(runtime)) as writer:
                    add_required_events(writer)
                for phase in ("pre", "post"):
                    record = json.loads(json.dumps(records["memory"]))
                    record["storage"]["model"]["resolved_path"] = different
                    record["storage"]["model"]["existing_path"] = different
                    replace_audit_record(root, phase, "memory", record)
                with self.assertRaisesRegex(trace.TraceError, "model path differs from the manifest"):
                    trace.TraceBundle(root)

    def test_rejects_duplicate_and_unknown_runtime_attestation_keys(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "trace"
            with trace.TraceBundleWriter(root, manifest("ds4")) as writer:
                add_required_events(writer)
            manifest_path = root / trace.MANIFEST_NAME
            data = manifest_path.read_text(encoding="ascii")
            data = data.replace(
                '"runtime_kind":"apple-metal"',
                '"runtime_kind":"apple-metal","runtime_kind":"apple-metal"',
                1,
            )
            manifest_path.write_text(data, encoding="ascii")
            with self.assertRaisesRegex(trace.TraceError, "duplicate JSON key"):
                trace.TraceBundle(root)

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "trace"
            trace_manifest = manifest("ds4")
            trace_manifest["accelerator"]["runtime_kind"] = "unknown"
            with trace.TraceBundleWriter(root, trace_manifest) as writer:
                add_required_events(writer)
            with self.assertRaisesRegex(trace.TraceError, "runtime_kind mismatch"):
                trace.TraceBundle(root)

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "trace"
            trace_manifest = manifest("ds4")
            del trace_manifest["accelerator"]["runtime_kind"]
            with trace.TraceBundleWriter(root, trace_manifest) as writer:
                add_required_events(writer)
            with self.assertRaisesRegex(trace.TraceError, "fields are invalid"):
                trace.TraceBundle(root)

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "trace"
            trace_manifest = manifest("ds4")
            trace_manifest["unknown_top_level"] = True
            with trace.TraceBundleWriter(root, trace_manifest) as writer:
                add_required_events(writer)
            with self.assertRaisesRegex(trace.TraceError, "unexpected unknown_top_level"):
                trace.TraceBundle(root)

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "trace"
            trace_manifest = manifest("ds4")
            trace_manifest["config"]["unvalidated_mode"] = "unsafe"
            with trace.TraceBundleWriter(root, trace_manifest) as writer:
                add_required_events(writer)
            with self.assertRaisesRegex(trace.TraceError, "unexpected unvalidated_mode"):
                trace.TraceBundle(root)

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "trace"
            trace_manifest = manifest("ds4")
            trace_manifest["environment"]["system_info"] = "Linux test system"
            with trace.TraceBundleWriter(root, trace_manifest) as writer:
                add_required_events(writer)
            with self.assertRaisesRegex(trace.TraceError, "environment is not macOS"):
                trace.TraceBundle(root)

    def test_rejects_boolean_accelerator_identities(self) -> None:
        for runtime, field in (
                ("llama.cpp", "gpu_id"),
                ("ds4", "metal_registry_id"),
                ("ds4", "recommended_max_working_set_bytes")):
            with self.subTest(runtime=runtime, field=field), tempfile.TemporaryDirectory() as temp:
                root = Path(temp) / "trace"
                trace_manifest = manifest(runtime)
                trace_manifest["accelerator"][field] = True
                with trace.TraceBundleWriter(root, trace_manifest) as writer:
                    add_required_events(writer)
                with self.assertRaisesRegex(trace.TraceError, "invalid"):
                    trace.TraceBundle(root)

    def test_rejects_unknown_watchdog_event_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "trace"
            with trace.TraceBundleWriter(root, manifest("llama.cpp")) as writer:
                add_required_events(writer)
            events = json.loads(json.dumps(WATCHDOG_EVENTS))
            events[0]["unknown"] = True
            data = "".join(
                json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n"
                for event in events
            ).encode("ascii")
            digest = trace.sha256_bytes(data)
            for phase in ("pre", "post"):
                jsonl = root / "audits" / phase / f"{digest}.jsonl"
                jsonl.write_bytes(data)
                record = json.loads(json.dumps(AUDIT_RECORDS["watchdog"]))
                record["data"]["audit"]["path"] = f"audits/{phase}/{digest}.jsonl"
                record["data"]["audit"]["sha256"] = digest
                record["data"]["audit"]["event_count"] = len(events)
                replace_audit_record(root, phase, "watchdog", record)
            with self.assertRaisesRegex(trace.TraceError, "unexpected unknown"):
                trace.TraceBundle(root)

    def test_ds4_runner_path_must_match_attested_storage(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "trace"
            with trace.TraceBundleWriter(root, manifest("ds4")) as writer:
                add_required_events(writer)
            for phase in ("pre", "post"):
                record = json.loads(json.dumps(DS4_AUDIT_RECORDS["runner"]))
                record["data"]["runner_script"] = "/Users/oracle/other/run_ds4.py"
                replace_audit_record(root, phase, "runner", record)
            with self.assertRaisesRegex(trace.TraceError, "runner_script mismatch"):
                trace.TraceBundle(root)

    def test_unapproved_ds4_exporter_is_not_executed(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            exporter = root / "exporter"
            exporter.write_text("#!/bin/sh\nexit 0\n", encoding="ascii")
            exporter.chmod(0o755)
            argv = [
                "run_ds4.py",
                "--repo", str(root),
                "--model", str(root / "model.gguf"),
                "--prompt", str(root / "prompt.txt"),
                "--output", str(root / "output"),
                "--exporter", str(exporter),
                "--exporter-sha256", trace.sha256_file(exporter),
                "--corpus-name", "correctness-prose.txt",
                "--corpus-sha256", trace.CORPUS_SHA256["correctness-prose.txt"],
                "--prompt-provenance", str(root / "prompt.json"),
                "--preflight-only",
            ]
            with mock.patch.object(sys, "argv", argv), mock.patch.object(
                    sys, "stderr", io.StringIO()), mock.patch.object(
                    run_ds4, "query_accelerator_attestation") as query:
                self.assertEqual(run_ds4.main(), 1)
            query.assert_not_called()

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

    def test_rejects_malformed_raw_attention_sources(self) -> None:
        cases = (
            (
                "width",
                0,
                [trace.RAW_ATTENTION_WIDTH - 1, 2],
                [trace.RAW_ATTENTION_WIDTH + 2] * ((trace.RAW_ATTENTION_WIDTH - 1) * 2),
                "raw attn.source shape",
            ),
            (
                "future ubatch row",
                0,
                [trace.RAW_ATTENTION_WIDTH, 2],
                [trace.RAW_ATTENTION_WIDTH + 1] +
                [trace.RAW_ATTENTION_WIDTH + 2] * (trace.RAW_ATTENTION_WIDTH * 2 - 1),
                "invalid row",
            ),
            (
                "layer mismatch",
                1,
                [trace.RAW_ATTENTION_WIDTH, 2],
                [0] + [trace.RAW_ATTENTION_WIDTH + 2] * (trace.RAW_ATTENTION_WIDTH * 2 - 1),
                "differs between layers 0 and 1",
            ),
        )
        for name, layer, shape, values, message in cases:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temp:
                root = Path(temp) / "trace"
                with trace.TraceBundleWriter(root, manifest()) as writer:
                    add_required_events(writer)
                replace_event_blob(
                    root,
                    component="attn.source",
                    phase="prefill",
                    layer=layer,
                    shape=shape,
                    data=struct.pack("<" + "i" * len(values), *values),
                )
                with self.assertRaisesRegex(trace.TraceError, message):
                    trace.TraceBundle(root)

    def test_compares_raw_attention_sources_in_every_prefill_chunk(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "trace"
            with trace.TraceBundleWriter(root, manifest()) as writer:
                add_required_events(writer)
            events = [
                json.loads(line)
                for line in (root / trace.EVENTS_NAME).read_text(encoding="ascii").splitlines()
            ]
            split_events = []
            for event in events:
                if event["component"] != "attn.source" or event["phase"] != "prefill" or (
                        event["layer"] not in trace.RAW_ATTENTION_LAYERS):
                    split_events.append(event)
                    continue
                for token_start in (0, 1):
                    values = [0] * trace.RAW_ATTENTION_WIDTH
                    if event["layer"] == 0 and token_start == 0:
                        values[0] = 1
                    data = struct.pack("<" + "i" * len(values), *values)
                    digest = trace.sha256_bytes(data)
                    (root / trace.BLOBS_DIR / f"{digest}.bin").write_bytes(data)
                    split = dict(event)
                    split.update({
                        "token_start": token_start,
                        "token_count": 1,
                        "shape": [trace.RAW_ATTENTION_WIDTH, 1],
                        "byte_count": len(data),
                        "sha256": digest,
                        "blob": f"{trace.BLOBS_DIR}/{digest}.bin",
                    })
                    split_events.append(split)
            (root / trace.EVENTS_NAME).write_text(
                "".join(json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n"
                        for event in split_events),
                encoding="ascii",
            )
            manifest_record = json.loads((root / trace.MANIFEST_NAME).read_text(encoding="ascii"))
            manifest_record["event_count"] = len(split_events)
            (root / trace.MANIFEST_NAME).write_text(
                json.dumps(manifest_record, sort_keys=True, separators=(",", ":")) + "\n",
                encoding="ascii",
            )
            with self.assertRaisesRegex(trace.TraceError, "differs between layers 0 and 1"):
                trace.TraceBundle(root)

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

    def test_verifies_pinned_ds4_anchor_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            checkout = Path(temp)
            anchor = checkout / "tests" / "fixture.vec"
            anchor.parent.mkdir(parents=True)
            anchor.write_bytes(b"fixture")
            expected = trace.sha256_bytes(b"fixture")
            original = verify_ds4_anchors.git_output
            try:
                verify_ds4_anchors.git_output = lambda checkout, *args: (
                    trace.DS4_REVISION if args == ("rev-parse", "HEAD") else "")
                result = verify_ds4_anchors.verify(
                    checkout,
                    anchors={"tests/fixture.vec": expected},
                )
                self.assertEqual(result["status"], "ANCHORS VERIFIED")
                anchor.write_bytes(b"changed")
                with self.assertRaisesRegex(verify_ds4_anchors.AnchorError, "SHA-256 mismatch"):
                    verify_ds4_anchors.verify(
                        checkout,
                        anchors={"tests/fixture.vec": expected},
                    )
            finally:
                verify_ds4_anchors.git_output = original

    def test_embeds_rewritten_watchdog_audit_content(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source"
            source.mkdir()
            audits = {}
            for kind in ("memory", "swap", "watchdog"):
                record = json.loads(json.dumps(AUDIT_RECORDS[kind]))
                if kind == "watchdog":
                    jsonl = source / "watchdog-events.jsonl"
                    jsonl.write_bytes(WATCHDOG_JSONL)
                    record["data"].pop("audit")
                    record["data"]["audit_path"] = str(jsonl)
                    record["data"]["audit_sha256"] = WATCHDOG_JSONL_SHA256
                    record["data"]["audit_event_count"] = len(WATCHDOG_EVENTS)
                path = source / f"{kind}.json"
                path.write_text(
                    json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n",
                    encoding="ascii",
                )
                audits[kind] = str(path)
            references = preflight.embed_audits(root / "trace", "pre", audits)
            embedded = json.loads(
                (root / "trace" / references["watchdog"]["path"]).read_text(encoding="ascii"))
            self.assertIn("audit", embedded["data"])
            self.assertNotIn("audit_path", embedded["data"])
            self.assertEqual(
                trace.sha256_bytes(
                    (root / "trace" / references["watchdog"]["path"]).read_bytes()),
                references["watchdog"]["sha256"],
            )

    def test_rejects_unapproved_ds4_exporter(self) -> None:
        with self.assertRaisesRegex(preflight.PreflightError, "not approved"):
            run_ds4.verify_exporter_approval("a" * 64)

    def test_rejects_preflight_audit_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            audits = {}
            for kind in ("memory", "swap", "watchdog"):
                record = json.loads(json.dumps(AUDIT_RECORDS[kind]))
                if kind == "watchdog":
                    jsonl = root / "watchdog-events.jsonl"
                    jsonl.write_bytes(WATCHDOG_JSONL)
                    record["data"]["audit_path"] = str(jsonl)
                    record["data"]["audit_sha256"] = WATCHDOG_JSONL_SHA256
                    record["data"]["audit_event_count"] = len(WATCHDOG_EVENTS)
                path = root / f"{kind}.json"
                path.write_text(
                    json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n",
                    encoding="ascii",
                )
                audits[kind] = str(path)
            digests = preflight.seal_audits(audits)
            memory = Path(audits["memory"])
            memory.chmod(0o644)
            memory.write_text("{}\n", encoding="ascii")
            with self.assertRaisesRegex(preflight.PreflightError, "changed during runtime"):
                preflight.verify_sealed_audits(audits, digests)

    def test_rejects_symlinked_bundle_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            bundle = root / "bundle"
            with trace.TraceBundleWriter(bundle, manifest()) as writer:
                add_required_events(writer)
            provenance = next((bundle / "provenance").iterdir())
            outside = root / "outside.json"
            outside.write_bytes(provenance.read_bytes())
            provenance.unlink()
            provenance.symlink_to(outside)
            with self.assertRaisesRegex(trace.TraceError, "must not use symlinks"):
                trace.TraceBundle(bundle)

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

    def test_local_bringup_reports_do_not_claim_cross_runtime_pass(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            first = Path(temp) / "first"
            second = Path(temp) / "second"
            with trace.TraceBundleWriter(first, manifest("llama.cpp")) as writer:
                add_required_events(writer)
            with trace.TraceBundleWriter(second, manifest("llama.cpp")) as writer:
                add_required_events(writer)
            result = trace.local_report(
                trace.TraceBundle(first),
                trace.TraceBundle(second),
                "self-consistency",
            )
            self.assertEqual(result["status"], "BRINGUP PASS")
            self.assertNotEqual(result["status"], "TARGET PASS")
            self.assertEqual(result["cross_runtime_status"], "INCOMPLETE")

    def test_local_base_regression_requires_attested_oracle_revision(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp) / "base"
            integrated = Path(temp) / "integrated"
            base_manifest = manifest("llama.cpp")
            base_manifest["revision"] = "b" * 40
            base_manifest["candidate"]["revision"] = "b" * 40
            with trace.TraceBundleWriter(base, base_manifest) as writer:
                add_required_events(writer)
            with trace.TraceBundleWriter(integrated, manifest("llama.cpp")) as writer:
                add_required_events(writer)
            result = trace.local_report(
                trace.TraceBundle(base),
                trace.TraceBundle(integrated),
                "base-regression",
            )
            self.assertEqual(result["status"], "BRINGUP PASS")
            self.assertEqual(result["cross_runtime_status"], "INCOMPLETE")

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
