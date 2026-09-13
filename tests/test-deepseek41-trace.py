#!/usr/bin/env python3

import copy
import importlib.util
import io
import json
import os
import shutil
import subprocess
import struct
import sys
import tempfile
import time
import unittest
from argparse import Namespace
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

TRACE_DIR = Path(__file__).parents[1] / "tools" / "deepseek-v41-trace"
sys.path.insert(0, str(TRACE_DIR))
MODULE_PATH = TRACE_DIR / "trace_format.py"
SPEC = importlib.util.spec_from_file_location("dsv41_trace_format", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
trace = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(trace)
import run_llama
import run_ds4
import run_matrix
import preflight
import verify_ds4_anchors

trace.APPROVED_WATCHDOGS[trace.WATCHDOG_SCRIPT_SHA256] = trace.WATCHDOG_REVISION
FIXTURE_DS4_EXPORTER_SHA256 = "3" * 64
trace.APPROVED_EXPORTERS[FIXTURE_DS4_EXPORTER_SHA256] = trace.DS4_REVISION
run_ds4.APPROVED_EXPORTERS[FIXTURE_DS4_EXPORTER_SHA256] = trace.DS4_REVISION
TEST_AUTH_ISSUED = int(time.time()) - 60
TEST_AUTH_EXPIRES = TEST_AUTH_ISSUED + 3600
TEST_CHALLENGE = "d" * 64
TEST_RUN_IDS = {
    "llama.cpp": "strix-llama-test-run",
    "ds4": "apple-ds4-test-run",
}
TEST_CANDIDATE_EXPORTER_POLICY_ID = "test-candidate-exporter"
TEST_PROMPT_BUILDER_POLICY_ID = "test-prompt-builder"

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
    "exporter_sha256": FIXTURE_DS4_EXPORTER_SHA256,
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
        "storage_policy": json.loads(json.dumps(trace.NO_EXTERNAL_STATE_STORAGE)),
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
            "state": "active",
            "file_device": 1,
            "file_inode": 2,
            "file_uid": 1000,
            "file_mode": 0o600,
            "lease_path": "/run/user/123/watchdog.lease",
            "watchdog_pid": 123,
            "watchdog_start_time_utc": "1970-01-01T00:00:01.000Z",
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
            "audit_sha256": WATCHDOG_JSONL_SHA256,
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
        "storage_policy": json.loads(json.dumps(trace.NO_EXTERNAL_STATE_STORAGE)),
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


def replace_watchdog_events(root: Path, phase: str, events: list[dict[str, object]]) -> None:
    records = []
    for event in events:
        record = json.loads(json.dumps(event))
        records.append(record)
    data = b"".join(
        (json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n").encode("ascii")
        for record in records
    )
    digest = trace.sha256_bytes(data)
    path = root / "audits" / phase / f"{digest}.jsonl"
    path.write_bytes(data)
    record = json.loads(json.dumps(AUDIT_RECORDS["watchdog"]))
    record["data"]["audit_sha256"] = digest
    record["data"]["audit"]["path"] = f"audits/{phase}/{digest}.jsonl"
    record["data"]["audit"]["sha256"] = digest
    record["data"]["audit"]["event_count"] = len(records)
    replace_audit_record(root, phase, "watchdog", record)


def fixture_prompt_builder_policy(
        prompt: bytes,
        *,
        context: int = 3,
        decode_steps: int = 1,
        builder_path: str = "/home/repo/build/bin/llama-deepseek-v41-prompt-builder",
        builder_sha256: str = "8" * 64,
        source_root: str = "/home/repo") -> dict[str, object]:
    runtime_components = [
        ("ggml", "libggml.so", "4", None),
        ("ggml-base", "libggml-base.so", "5", "a" * 40),
        ("ggml-hip", "libggml-hip.so", "6", None),
        ("llama", "libllama.so", "7", None),
        ("llama-common", "libllama-common.so", "9", "a" * 40),
    ]
    runtime_profile = {
        "name": "sibling-lib",
        "components": sorted(component for component, *_rest in runtime_components),
        "selected_backend_component": "ggml-hip",
    }
    return {
        "runtime": "llama.cpp",
        "runtime_profile": runtime_profile,
        "repository": trace.REPOSITORY,
        "revision": "a" * 40,
        "install_root": str(Path(builder_path).parent.parent),
        "executable_path": builder_path,
        "executable_sha256": builder_sha256,
        "source_root": source_root,
        "runtime_receipt": {
            "format": "dsv41-runtime-receipt",
            "version": 1,
            "revision": "a" * 40,
            "profile": "sibling-lib",
            "components": [
                {
                    "component": component,
                    "filename": filename,
                    "sha256": digest * 64,
                    "revision": revision,
                }
                for component, filename, digest, revision in runtime_components
            ],
        },
        "model_sha256": trace.MODEL_SHA256,
        "corpora": dict(trace.CORPUS_SHA256),
        "prompts": [{
            "corpus_name": "correctness-prose.txt",
            "corpus_sha256": trace.CORPUS_SHA256["correctness-prose.txt"],
            "context": context,
            "decode_steps": decode_steps,
            "target_tokens": context - decode_steps,
            "prompt_sha256": trace.sha256_bytes(prompt),
            "prompt_byte_count": len(prompt),
            "add_bos": True,
        }],
    }


def materialize_policy_runtime(policy: dict[str, object]) -> None:
    library_root = Path(policy["install_root"]) / "lib"
    library_root.mkdir(parents=True, exist_ok=True)
    for component in policy["runtime_receipt"]["components"]:
        path = library_root / component["filename"]
        path.write_bytes(component["component"].encode("ascii"))
        component["sha256"] = trace.sha256_file(path)


def provenance_bytes(
        prompt: bytes = b"abc",
        *,
        context: int = 3,
        decode_steps: int = 1) -> bytes:
    policy = fixture_prompt_builder_policy(prompt, context=context, decode_steps=decode_steps)
    _validated, policy_sha256 = trace.prompt_builder_approval(
        TEST_PROMPT_BUILDER_POLICY_ID,
        policies={TEST_PROMPT_BUILDER_POLICY_ID: policy},
    )
    record = {
        "format": "dsv41-prompt-provenance",
        "version": 1,
        "corpus_name": "correctness-prose.txt",
        "corpus_sha256": trace.CORPUS_SHA256["correctness-prose.txt"],
        "corpus_path": "/home/repo/tests/corpus/correctness-prose.txt",
        "model_sha256": trace.MODEL_SHA256,
        "prompt_sha256": trace.sha256_bytes(prompt),
        "prompt_byte_count": len(prompt),
        "context": context,
        "decode_steps": decode_steps,
        "target_tokens": context - decode_steps,
        "actual_tokens": context - decode_steps,
        "builder_approval_id": TEST_PROMPT_BUILDER_POLICY_ID,
        "builder_approval_sha256": policy_sha256,
        "builder_path": policy["executable_path"],
        "builder_sha256": "8" * 64,
        "builder_revision": "a" * 40,
        "builder_runtime_profile": policy["runtime_profile"],
    }
    return (json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n").encode("ascii")


def manifest(
        runtime: str = "llama.cpp",
        prompt: bytes = b"abc",
        *,
        context: int = 3,
        decode_steps: int = 1) -> dict:
    provenance_sha256 = trace.sha256_bytes(
        provenance_bytes(prompt, context=context, decode_steps=decode_steps))
    is_ds4 = runtime == "ds4"
    storage = DS4_STORAGE_ATTESTATION if is_ds4 else STORAGE_ATTESTATION
    audit_kinds = ("memory", "swap", "runner") if is_ds4 else ("memory", "swap", "watchdog")
    runtime_components = [
        ("ggml", "libggml.so", "4", "runtime:ggml", None),
        ("ggml-base", "libggml-base.so", "5", "ggml", "a" * 40),
        ("ggml-hip", "libggml-hip.so", "6", "selected-backend", None),
        ("llama", "libllama.so", "7", "llama", None),
        ("llama-common", "libllama-common.so", "9", "build-info", "a" * 40),
    ]
    runtime_libraries = [
        {
            "component": component,
            "filename": filename,
            "path": f"/home/repo/build/lib/{filename}",
            "sha256": digest * 64,
            "role": role,
            "revision": revision,
        }
        for component, filename, digest, role, revision in runtime_components
    ]
    runtime_receipt = {
        "format": "dsv41-runtime-receipt",
        "version": 1,
        "revision": "a" * 40,
        "profile": "sibling-lib",
        "components": [
            {
                "component": component,
                "filename": filename,
                "sha256": digest * 64,
                "revision": revision,
            }
            for component, filename, digest, _role, revision in runtime_components
        ],
    }
    result = {
        "runtime": runtime,
        "revision": trace.DS4_REVISION if is_ds4 else "a" * 40,
        "build": (
            {
                "compiler": "clang",
                "target": "arm64-apple-darwin",
                "path": "/Users/oracle/bin/ds4-trace",
                "sha256": FIXTURE_DS4_EXPORTER_SHA256,
            }
            if is_ds4
            else {
                "number": 1,
                "info": "test",
                "compiler": "clang",
                "target": "arm64-apple-darwin",
                "path": "/home/repo/build/bin/llama-deepseek-v41-trace",
                "sha256": "3" * 64,
                "runtime_profile": {
                    "name": "sibling-lib",
                    "components": [component for component, *_rest in runtime_components],
                    "selected_backend_component": "ggml-hip",
                },
                "runtime_receipt_sha256": trace.sha256_bytes(
                    trace.canonical_json(runtime_receipt).encode("ascii")),
                "runtime_libraries": sorted(runtime_libraries, key=lambda library: library["path"]),
                "runtime_libraries_post": sorted(
                    copy.deepcopy(runtime_libraries), key=lambda library: library["path"]),
                "runtime_module_monitor": {
                    "mechanism": "pre-post-snapshot",
                    "checked_after_trace": True,
                    "project_additions": [],
                },
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
            "target_tokens": context - decode_steps,
            "provenance": {
                "path": f"provenance/{provenance_sha256}.json",
                "sha256": provenance_sha256,
            },
        },
        "config": {
            "context": context,
            "decode_steps": decode_steps,
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
        "storage_policy": json.loads(json.dumps(trace.NO_EXTERNAL_STATE_STORAGE)),
        "comparison": {
            "tokens": "exact",
            "engram_rows": "exact",
            "expert_ids": "exact-original-id-space",
            "expert_weights": "byte-identical-f32",
            "attention_candidates": "exact",
            "logits": "byte-identical-f32",
        },
        "expected": {
            "prompt_tokens": context - decode_steps,
            "decode_steps": decode_steps,
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
            "executable_path": result["build"]["path"],
            "executable_sha256": "3" * 64,
            "runtime_libraries_sha256": trace.sha256_bytes(
                trace.canonical_json({
                    "pre": result["build"]["runtime_libraries"],
                    "post": result["build"]["runtime_libraries_post"],
                }).encode("ascii")),
            "runtime_receipt_sha256": result["build"]["runtime_receipt_sha256"],
        }
    else:
        result["host"] = dict(DS4_HOST_ATTESTATION)
        result["config"]["prefill_chunk"] = trace.ADMITTED_UBATCH
        result["config"]["device_backend"] = "Metal"
        result["config"]["device_registry_id"] = METAL_ACCELERATOR_ATTESTATION["metal_registry_id"]
    prompt_policy = fixture_prompt_builder_policy(
        prompt, context=context, decode_steps=decode_steps)
    _prompt_policy, prompt_policy_sha256 = trace.prompt_builder_approval(
        TEST_PROMPT_BUILDER_POLICY_ID,
        policies={TEST_PROMPT_BUILDER_POLICY_ID: prompt_policy},
    )
    approvals = {
        "prompt_builder": trace.approval_binding(
            "prompt_builder", TEST_PROMPT_BUILDER_POLICY_ID, prompt_policy_sha256),
    }
    if not is_ds4:
        candidate_policy = {
            "runtime": "llama.cpp",
            "repository": trace.REPOSITORY,
            "revision": result["candidate"]["revision"],
            "base_revision": result["candidate"]["base_revision"],
            "diff_sha256": result["candidate"]["diff_sha256"],
            "install_root": "/home/repo/build",
            "executable_path": result["candidate"]["executable_path"],
            "executable_sha256": result["candidate"]["executable_sha256"],
            "runtime_profile": copy.deepcopy(result["build"]["runtime_profile"]),
            "runtime_receipt": runtime_receipt,
        }
        _candidate_policy, candidate_policy_sha256 = trace.candidate_exporter_approval(
            TEST_CANDIDATE_EXPORTER_POLICY_ID,
            policies={TEST_CANDIDATE_EXPORTER_POLICY_ID: candidate_policy},
        )
        result["candidate"]["exporter_approval_id"] = TEST_CANDIDATE_EXPORTER_POLICY_ID
        result["candidate"]["exporter_approval_sha256"] = candidate_policy_sha256
        approvals["candidate_exporter"] = trace.approval_binding(
            "candidate_exporter", TEST_CANDIDATE_EXPORTER_POLICY_ID, candidate_policy_sha256)
    result["authorization"] = trace.execution_authorization(
        lane=trace.ORACLE_LANE if is_ds4 else trace.CANDIDATE_LANE,
        challenge=TEST_CHALLENGE,
        run_id=TEST_RUN_IDS[runtime],
        issued_unix=TEST_AUTH_ISSUED,
        expires_unix=TEST_AUTH_EXPIRES,
        approval_policy_sha256="e" * 64,
        verifier_revision="a" * 40,
        approvals=approvals,
    )
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
    data = provenance_bytes(
        prompt,
        context=writer.manifest["config"]["context"],
        decode_steps=writer.manifest["config"]["decode_steps"],
    )
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
    @classmethod
    def setUpClass(cls) -> None:
        cls._signing_directory = tempfile.TemporaryDirectory()
        cls.ssh_keygen = trace.trusted_ssh_keygen_path()
        cls.signing_keys = {}
        cls.signer_principals = {
            "llama.cpp": "dsv41-test-candidate",
            "ds4": "dsv41-test-oracle",
        }
        cls.test_signers = {}
        for runtime, principal in cls.signer_principals.items():
            signing_key = Path(cls._signing_directory.name) / f"{runtime.replace('.', '-')}-key"
            subprocess.run(
                [
                    str(cls.ssh_keygen),
                    "-q",
                    "-t", "ed25519",
                    "-N", "",
                    "-f", str(signing_key),
                ],
                check=True,
            )
            signing_key.chmod(0o600)
            public_key = subprocess.check_output(
                [str(cls.ssh_keygen), "-y", "-f", str(signing_key)],
                text=True,
            ).strip()
            public_key = " ".join(public_key.split()[:2])
            lane = trace.ORACLE_LANE if runtime == "ds4" else trace.CANDIDATE_LANE
            profile = "apple-metal" if runtime == "ds4" else "sibling-lib"
            cls.signing_keys[runtime] = signing_key
            cls.test_signers[principal] = {
                "public_key": public_key,
                "lane": lane,
                "runtime": runtime,
                "runtime_profile": profile,
            }
        cls.signing_key = cls.signing_keys["llama.cpp"]
        cls.signer_principal = cls.signer_principals["llama.cpp"]
        cls.verifier = cls._verifier_for_runtime("llama.cpp")
        cls._trace_bundle_class = trace.TraceBundle

    @classmethod
    def _verifier_for_runtime(
            cls,
            runtime: str,
            *,
            manifest_record: dict[str, object] | None = None,
            expected_challenge: str = TEST_CHALLENGE,
            expected_run_id: str | None = None,
            verification_unix: int | None = None,
            seen_run_ids: set[str] | None = None) -> trace.TraceVerifier:
        principal = cls.signer_principals[runtime]
        policy = cls.test_signers[principal]
        manifest_record = manifest_record or manifest(runtime)
        prompt_policy = fixture_prompt_builder_policy(b"abc")
        prompt_policy["prompts"][0].update({
            "corpus_name": manifest_record["prompt"]["corpus_name"],
            "corpus_sha256": manifest_record["prompt"]["corpus_sha256"],
            "context": manifest_record["config"]["context"],
            "decode_steps": manifest_record["config"]["decode_steps"],
            "target_tokens": manifest_record["prompt"]["target_tokens"],
            "prompt_sha256": manifest_record["prompt"]["sha256"],
            "prompt_byte_count": manifest_record["prompt"]["byte_count"],
        })
        prompt_policies = {TEST_PROMPT_BUILDER_POLICY_ID: prompt_policy}
        candidate_policies = {}
        candidate_policy_id = None
        if runtime == "llama.cpp":
            receipt = {
                "format": "dsv41-runtime-receipt",
                "version": 1,
                "revision": manifest_record["candidate"]["revision"],
                "profile": manifest_record["build"]["runtime_profile"]["name"],
                "components": sorted(
                    [
                        {
                            "component": library["component"],
                            "filename": library["filename"],
                            "sha256": library["sha256"],
                            "revision": library["revision"],
                        }
                        for library in manifest_record["build"]["runtime_libraries"]
                    ],
                    key=lambda item: item["component"],
                ),
            }
            candidate_policy = {
                "runtime": "llama.cpp",
                "repository": manifest_record["candidate"]["repository"],
                "revision": manifest_record["candidate"]["revision"],
                "base_revision": manifest_record["candidate"]["base_revision"],
                "diff_sha256": manifest_record["candidate"]["diff_sha256"],
                "install_root": "/home/repo/build",
                "executable_path": manifest_record["candidate"]["executable_path"],
                "executable_sha256": manifest_record["candidate"]["executable_sha256"],
                "runtime_profile": copy.deepcopy(manifest_record["build"]["runtime_profile"]),
                "runtime_receipt": receipt,
            }
            candidate_policies[TEST_CANDIDATE_EXPORTER_POLICY_ID] = candidate_policy
            candidate_policy_id = TEST_CANDIDATE_EXPORTER_POLICY_ID
        return trace.TraceVerifier.for_tests(
            principal,
            policy["public_key"],
            lane=policy["lane"],
            runtime=runtime,
            runtime_profile=policy["runtime_profile"],
            expected_challenge=expected_challenge,
            expected_run_id=expected_run_id or TEST_RUN_IDS[runtime],
            candidate_exporter_policies=candidate_policies,
            prompt_builder_policies=prompt_policies,
            expected_candidate_exporter_policy_id=candidate_policy_id,
            expected_prompt_builder_policy_id=TEST_PROMPT_BUILDER_POLICY_ID,
            verification_unix=verification_unix or int(time.time()),
            ssh_keygen=cls.ssh_keygen,
            seen_run_ids=seen_run_ids,
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls._signing_directory.cleanup()

    def setUp(self) -> None:
        self._require_nvme_path = preflight.require_nvme_path
        preflight.require_nvme_path = lambda path, label, **kwargs: preflight.resolved(path)
        self._trace_bundle_symbol = trace.TraceBundle

        def test_bundle(root: Path, verify_blobs: bool = True, **_kwargs: object) -> object:
            self._prune_fixture_extras(Path(root))
            signature = Path(root) / trace.SIGNATURE_NAME
            if signature.exists() or signature.is_symlink():
                signature.unlink()
            manifest_record = trace.strict_json_loads(
                (Path(root) / trace.MANIFEST_NAME).read_text(encoding="ascii"))
            runtime = manifest_record["runtime"]
            principal = self.signer_principals[runtime]
            authorization = manifest_record["authorization"]
            verifier = self._verifier_for_runtime(
                runtime,
                manifest_record=manifest_record,
                expected_challenge=authorization["challenge"],
                expected_run_id=authorization["run_id"],
            )
            trace.seal_bundle(
                Path(root),
                private_key=self.signing_keys[runtime],
                principal=principal,
                expected_lane=authorization["lane"],
                expected_challenge=authorization["challenge"],
                expected_run_id=authorization["run_id"],
                candidate_exporter_policies=verifier.candidate_exporter_policies,
                prompt_builder_policies=verifier.prompt_builder_policies,
                expected_candidate_exporter_policy_id=verifier.expected_candidate_exporter_policy_id,
                expected_prompt_builder_policy_id=verifier.expected_prompt_builder_policy_id,
                expected_approval_policy_sha256=verifier.expected_approval_policy_sha256,
                expected_verifier_revision=verifier.expected_verifier_revision,
                trusted_signers=self.test_signers,
                ssh_keygen=self.ssh_keygen,
            )
            return self._trace_bundle_class(
                Path(root),
                verify_blobs,
                verifier=verifier,
            )

        trace.TraceBundle = test_bundle

    def tearDown(self) -> None:
        preflight.require_nvme_path = self._require_nvme_path
        trace.TraceBundle = self._trace_bundle_symbol

    def _seal_test_bundle(self, root: Path) -> str:
        signature = root / trace.SIGNATURE_NAME
        if signature.exists() or signature.is_symlink():
            signature.unlink()
        manifest_record = trace.strict_json_loads(
            (root / trace.MANIFEST_NAME).read_text(encoding="ascii"))
        runtime = manifest_record["runtime"]
        principal = self.signer_principals[runtime]
        authorization = manifest_record["authorization"]
        verifier = self._verifier_for_runtime(
            runtime,
            manifest_record=manifest_record,
            expected_challenge=authorization["challenge"],
            expected_run_id=authorization["run_id"],
        )
        return trace.seal_bundle(
            root,
            private_key=self.signing_keys[runtime],
            principal=principal,
            expected_lane=authorization["lane"],
            expected_challenge=authorization["challenge"],
            expected_run_id=authorization["run_id"],
            candidate_exporter_policies=verifier.candidate_exporter_policies,
            prompt_builder_policies=verifier.prompt_builder_policies,
            expected_candidate_exporter_policy_id=verifier.expected_candidate_exporter_policy_id,
            expected_prompt_builder_policy_id=verifier.expected_prompt_builder_policy_id,
            expected_approval_policy_sha256=verifier.expected_approval_policy_sha256,
            expected_verifier_revision=verifier.expected_verifier_revision,
            trusted_signers=self.test_signers,
            ssh_keygen=self.ssh_keygen,
        )

    def _prune_fixture_extras(self, root: Path) -> None:
        manifest_path = root / trace.MANIFEST_NAME
        events_path = root / trace.EVENTS_NAME
        if not manifest_path.is_file() or not events_path.is_file():
            return
        try:
            manifest_record = trace.strict_json_loads(manifest_path.read_text(encoding="ascii"))
            events = [
                trace.strict_json_loads(line)
                for line in events_path.read_text(encoding="ascii").splitlines()
            ]
        except (OSError, UnicodeError, trace.TraceError):
            return
        expected = {trace.MANIFEST_NAME, trace.EVENTS_NAME}
        if isinstance(manifest_record, dict):
            prompt = manifest_record.get("prompt")
            provenance = prompt.get("provenance") if isinstance(prompt, dict) else None
            if isinstance(provenance, dict) and isinstance(provenance.get("path"), str):
                expected.add(provenance["path"])
            audits = manifest_record.get("audits")
            if isinstance(audits, dict):
                for phase in audits.values():
                    if not isinstance(phase, dict):
                        continue
                    for reference in phase.values():
                        if not isinstance(reference, dict) or not isinstance(reference.get("path"), str):
                            continue
                        expected.add(reference["path"])
                        audit_path = root / reference["path"]
                        if not audit_path.is_file():
                            continue
                        try:
                            audit = trace.strict_json_loads(audit_path.read_text(encoding="ascii"))
                        except (OSError, UnicodeError, trace.TraceError):
                            continue
                        nested = audit.get("data", {}).get("audit") if isinstance(audit, dict) else None
                        if isinstance(nested, dict) and isinstance(nested.get("path"), str):
                            expected.add(nested["path"])
        for event in events:
            if isinstance(event, dict) and isinstance(event.get("blob"), str):
                expected.add(event["blob"])
        for path in root.rglob("*"):
            if path.is_file() and path.relative_to(root).as_posix() not in expected | {trace.SIGNATURE_NAME}:
                path.unlink()

    def _read_sealed_bundle(self, root: Path) -> object:
        return self._trace_bundle_class(root, verifier=self.verifier)

    def test_seal_requires_external_trust_and_fixed_verifier(self) -> None:
        self.assertEqual(trace.APPROVED_TRACE_SIGNERS, {})
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "trace"
            with trace.TraceBundleWriter(root, manifest()) as writer:
                add_required_events(writer)
            self._seal_test_bundle(root)
            self._read_sealed_bundle(root)
            with self.assertRaisesRegex(
                    trace.TraceError,
                    "external signer, lane, challenge, run ID, and prompt builder approval"):
                self._trace_bundle_class(root)
            with self.assertRaisesRegex(trace.TraceError, "not approved"):
                self._trace_bundle_class(
                    root,
                    signer_principal=self.signer_principal,
                    expected_lane=trace.CANDIDATE_LANE,
                    expected_challenge=TEST_CHALLENGE,
                    expected_run_id=TEST_RUN_IDS["llama.cpp"],
                    expected_candidate_exporter_policy_id=TEST_CANDIDATE_EXPORTER_POLICY_ID,
                    expected_prompt_builder_policy_id=TEST_PROMPT_BUILDER_POLICY_ID,
                )
            candidate_policy = self.test_signers[self.signer_principal]
            unknown = trace.TraceVerifier.for_tests(
                "unknown",
                candidate_policy["public_key"],
                lane=trace.CANDIDATE_LANE,
                runtime="llama.cpp",
                runtime_profile="sibling-lib",
                expected_challenge=TEST_CHALLENGE,
                expected_run_id=TEST_RUN_IDS["llama.cpp"],
                verification_unix=int(time.time()),
            )
            with self.assertRaisesRegex(trace.TraceError, "externally expected signer"):
                self._trace_bundle_class(root, verifier=unknown)

            fake_directory = Path(temp) / "fake-bin"
            fake_directory.mkdir()
            fake = fake_directory / "ssh-keygen"
            fake.write_text("#!/bin/sh\nexit 0\n", encoding="ascii")
            fake.chmod(0o755)
            with mock.patch.dict(os.environ, {"PATH": str(fake_directory)}):
                self._read_sealed_bundle(root)
            substituted = Path(temp) / "substituted-ssh-keygen"
            substituted.symlink_to(fake)
            verifier = trace.TraceVerifier.for_tests(
                self.signer_principal,
                candidate_policy["public_key"],
                lane=trace.CANDIDATE_LANE,
                runtime="llama.cpp",
                runtime_profile="sibling-lib",
                expected_challenge=TEST_CHALLENGE,
                expected_run_id=TEST_RUN_IDS["llama.cpp"],
                verification_unix=int(time.time()),
                ssh_keygen=substituted,
            )
            with self.assertRaisesRegex(trace.TraceError, "non-symlink"):
                self._trace_bundle_class(root, verifier=verifier)

            with self.assertRaisesRegex(trace.TraceError, "already exists"):
                trace.seal_bundle(
                    root,
                    private_key=self.signing_key,
                    principal=self.signer_principal,
                    expected_lane=trace.CANDIDATE_LANE,
                    expected_challenge=TEST_CHALLENGE,
                    expected_run_id=TEST_RUN_IDS["llama.cpp"],
                    trusted_signers=self.test_signers,
                    ssh_keygen=self.ssh_keygen,
                )

            other_key = Path(temp) / "other-key"
            subprocess.run(
                [
                    str(self.ssh_keygen),
                    "-q",
                    "-t", "ed25519",
                    "-N", "",
                    "-f", str(other_key),
                ],
                check=True,
            )
            other_key.chmod(0o600)
            with self.assertRaisesRegex(trace.TraceError, "does not match"):
                trace.validate_signing_identity(
                    other_key,
                    self.signer_principal,
                    trusted_signers=self.test_signers,
                    ssh_keygen=self.ssh_keygen,
                )

    def test_production_executable_approval_maps_fail_closed(self) -> None:
        self.assertEqual(trace.APPROVED_CANDIDATE_EXPORTERS, {})
        self.assertEqual(trace.APPROVED_PROMPT_BUILDERS, {})
        self.assertEqual(trace.APPROVED_EXECUTABLE_APPROVERS, {})
        with self.assertRaisesRegex(trace.TraceError, "candidate exporter approval is not trusted"):
            trace.candidate_exporter_approval(TEST_CANDIDATE_EXPORTER_POLICY_ID)
        with self.assertRaisesRegex(trace.TraceError, "prompt builder approval is not trusted"):
            trace.prompt_builder_approval(TEST_PROMPT_BUILDER_POLICY_ID)

    def test_external_executable_approval_signature_and_tamper(self) -> None:
        verifier = self._verifier_for_runtime("llama.cpp")
        principal = "dsv41-test-executable-approver"
        public_key = self.test_signers[self.signer_principal]["public_key"]
        policy = {
            "format": trace.EXECUTABLE_APPROVAL_FORMAT,
            "version": trace.EXECUTABLE_APPROVAL_VERSION,
            "principal": principal,
            "verifier_repository": trace.REPOSITORY,
            "verifier_revision": "a" * 40,
            "candidate_exporters": verifier.candidate_exporter_policies,
            "prompt_builders": verifier.prompt_builder_policies,
        }
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            policy_path = root / "approval.json"
            policy_path.write_text(trace.canonical_json(policy) + "\n", encoding="ascii")
            subprocess.run(
                [
                    str(self.ssh_keygen),
                    "-Y", "sign",
                    "-f", str(self.signing_key),
                    "-n", trace.EXECUTABLE_APPROVAL_NAMESPACE,
                    str(policy_path),
                ],
                check=True,
                stdin=subprocess.DEVNULL,
                capture_output=True,
            )
            signature_path = policy_path.with_suffix(".json.sig")
            loaded = trace.load_executable_approval_policy(
                policy_path,
                signature_path,
                expected_principal=principal,
                trusted_approvers={principal: public_key},
                ssh_keygen=self.ssh_keygen,
            )
            self.assertEqual(loaded.verifier_revision, "a" * 40)
            self.assertEqual(loaded.sha256, trace.sha256_file(policy_path))
            self.assertEqual(
                loaded.candidate_exporters,
                verifier.candidate_exporter_policies,
            )
            with self.assertRaisesRegex(trace.TraceError, "outside protected output roots"):
                trace.load_executable_approval_policy(
                    policy_path,
                    signature_path,
                    expected_principal=principal,
                    trusted_approvers={principal: public_key},
                    ssh_keygen=self.ssh_keygen,
                    forbidden_roots=(root,),
                )
            tampered = copy.deepcopy(policy)
            tampered["verifier_revision"] = "b" * 40
            policy_path.write_text(trace.canonical_json(tampered) + "\n", encoding="ascii")
            with self.assertRaisesRegex(trace.TraceError, "signature verification failed"):
                trace.load_executable_approval_policy(
                    policy_path,
                    signature_path,
                    expected_principal=principal,
                    trusted_approvers={principal: public_key},
                    ssh_keygen=self.ssh_keygen,
                )
            with self.assertRaisesRegex(trace.TraceError, "principal is not trusted"):
                trace.load_executable_approval_policy(
                    policy_path,
                    signature_path,
                    expected_principal=principal,
                    trusted_approvers={},
                    ssh_keygen=self.ssh_keygen,
                )

    def test_candidate_runner_rejects_unapproved_exporter_before_execution(self) -> None:
        argv = [
            "run_llama.py",
            "--exporter", "/usr/bin/true",
            "--repo", "/tmp/repo",
            "--candidate-revision", "a" * 40,
            "--base-revision", "b" * 40,
            "--candidate-diff-sha256", "c" * 64,
            "--candidate-exporter-policy-id", "unapproved-exporter",
            "--prompt-builder-policy-id", "unapproved-builder",
            "--approval-policy", "/tmp/approval.json",
            "--approval-signature", "/tmp/approval.sig",
            "--approval-principal", "unapproved",
            "--corpus-name", "correctness-prose.txt",
            "--corpus-sha256", trace.CORPUS_SHA256["correctness-prose.txt"],
            "--prompt-provenance", "/tmp/prompt.json",
            "--model", "/tmp/model.gguf",
            "--prompt", "/tmp/prompt.txt",
            "--output", "/tmp/output",
            "--signer-principal", "candidate",
            "--signing-key", "/tmp/key",
            "--execution-challenge", TEST_CHALLENGE,
            "--run-id", TEST_RUN_IDS["llama.cpp"],
            "--authorization-issued-unix", str(TEST_AUTH_ISSUED),
            "--authorization-expires-unix", str(TEST_AUTH_EXPIRES),
            "--preflight-only",
        ]
        with mock.patch.dict(os.environ, {}, clear=True), mock.patch.object(
                sys, "argv", argv), mock.patch.object(
                sys, "stderr", io.StringIO()), mock.patch.object(
                run_llama, "query_runtime_build_attestation") as build_query, mock.patch.object(
                run_llama, "query_accelerator_attestation") as accelerator_query:
            self.assertEqual(run_llama.main(), 1)
        build_query.assert_not_called()
        accelerator_query.assert_not_called()

    def test_prompt_builder_rejects_unapproved_identity_before_execution(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            builder = root / "actual" / "bin" / "llama-deepseek-v41-prompt-builder"
            approved_builder = root / "approved" / "bin" / "llama-deepseek-v41-prompt-builder"
            builder.parent.mkdir(parents=True)
            builder.write_bytes(b"builder")
            builder.chmod(0o755)
            policy = fixture_prompt_builder_policy(
                b"prompt",
                builder_path=str(approved_builder),
                builder_sha256=trace.sha256_file(builder),
                source_root=str(Path(__file__).parents[1].resolve()),
            )
            _validated, policy_sha256 = trace.prompt_builder_approval(
                TEST_PROMPT_BUILDER_POLICY_ID,
                policies={TEST_PROMPT_BUILDER_POLICY_ID: policy},
            )
            with mock.patch.object(run_matrix.subprocess, "run") as execute, self.assertRaisesRegex(
                    run_matrix.TraceError, "path differs from external approval"):
                run_matrix.prepare_prompt(
                    builder=builder,
                    builder_approval_id=TEST_PROMPT_BUILDER_POLICY_ID,
                    builder_policy=policy,
                    builder_policy_sha256=policy_sha256,
                    model=root / "model.gguf",
                    corpus=root / "corpus.txt",
                    source_corpus=Path(__file__).parents[1] / "tests" / "corpus" / "correctness-prose.txt",
                    corpus_name="correctness-prose.txt",
                    corpus_sha256=trace.CORPUS_SHA256["correctness-prose.txt"],
                    output=root / "prompt.txt",
                    context=3,
                    decode_steps=1,
                )
            execute.assert_not_called()

    def test_approved_executable_uses_linux_descriptor_path(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            executable = Path(temp).resolve() / "approved"
            executable.write_bytes(b"approved")
            executable.chmod(0o755)
            completed = subprocess.CompletedProcess([str(executable)], 0, "", "")
            with mock.patch.object(trace.sys, "platform", "linux"), mock.patch.object(
                    trace.subprocess, "run", return_value=completed) as execute:
                result, identity = trace.run_approved_executable(
                    [str(executable), "--version"],
                    path=executable,
                    expected_path=str(executable),
                    expected_sha256=trace.sha256_file(executable),
                    label="approved executable",
                    check=False,
                    capture_output=True,
                    text=True,
                )
            self.assertIs(result, completed)
            self.assertEqual(identity.path, str(executable))
            kwargs = execute.call_args.kwargs
            self.assertRegex(kwargs["executable"], r"^/proc/self/fd/[0-9]+$")
            self.assertEqual(len(kwargs["pass_fds"]), 1)

    def test_prompt_builder_rejects_runtime_receipt_before_execution(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            builder = root / "install" / "bin" / "llama-deepseek-v41-prompt-builder"
            builder.parent.mkdir(parents=True)
            builder.write_bytes(b"builder")
            builder.chmod(0o755)
            policy = fixture_prompt_builder_policy(
                b"prompt",
                builder_path=str(builder),
                builder_sha256=trace.sha256_file(builder),
                source_root=str(Path(__file__).parents[1].resolve()),
            )
            materialize_policy_runtime(policy)
            _validated, policy_sha256 = trace.prompt_builder_approval(
                TEST_PROMPT_BUILDER_POLICY_ID,
                policies={TEST_PROMPT_BUILDER_POLICY_ID: policy},
            )
            runtime_component = policy["runtime_receipt"]["components"][0]
            (Path(policy["install_root"]) / "lib" / runtime_component["filename"]).write_bytes(
                b"changed")
            with mock.patch.object(run_matrix, "run_approved_executable") as execute, self.assertRaisesRegex(
                    run_matrix.TraceError, "runtime component .* SHA-256 differs from external approval"):
                run_matrix.prepare_prompt(
                    builder=builder,
                    builder_approval_id=TEST_PROMPT_BUILDER_POLICY_ID,
                    builder_policy=policy,
                    builder_policy_sha256=policy_sha256,
                    model=root / "model.gguf",
                    corpus=root / "corpus.txt",
                    source_corpus=Path(__file__).parents[1] / "tests" / "corpus" / "correctness-prose.txt",
                    corpus_name="correctness-prose.txt",
                    corpus_sha256=trace.CORPUS_SHA256["correctness-prose.txt"],
                    output=root / "prompt.txt",
                    context=3,
                    decode_steps=1,
                )
            execute.assert_not_called()

    def test_prompt_builder_rejects_output_binary_and_corpus_mutation(self) -> None:
        for mutation, message in (
                ("output", "output differs from external approval"),
                ("builder", "SHA-256 differs from external approval"),
                ("corpus", "corpus changed during execution"),
                ("runtime", "runtime component SHA-256 differs from external approval"),
        ):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temp:
                root = Path(temp).resolve()
                builder = root / "install" / "bin" / "llama-deepseek-v41-prompt-builder"
                model = root / "model.gguf"
                corpus = root / "corpus.txt"
                source_root = Path(__file__).parents[1].resolve()
                source_corpus = source_root / "tests" / "corpus" / "correctness-prose.txt"
                output = root / "prompt.txt"
                tmpdir = root / "tmp"
                builder.parent.mkdir(parents=True)
                builder.write_bytes(b"builder")
                builder.chmod(0o755)
                model.write_bytes(b"model")
                shutil.copyfile(source_corpus, corpus)
                tmpdir.mkdir()
                approved_output = b"approved" if mutation == "output" else b"prompt"
                policy = fixture_prompt_builder_policy(
                    approved_output,
                    builder_path=str(builder),
                    builder_sha256=trace.sha256_file(builder),
                    source_root=str(source_root),
                )
                materialize_policy_runtime(policy)
                _validated, policy_sha256 = trace.prompt_builder_approval(
                    TEST_PROMPT_BUILDER_POLICY_ID,
                    policies={TEST_PROMPT_BUILDER_POLICY_ID: policy},
                )
                initial_identity = run_matrix.approved_executable_identity(
                    builder,
                    expected_path=policy["executable_path"],
                    expected_sha256=policy["executable_sha256"],
                    label="prompt builder",
                )

                def run_builder(command, **_kwargs):
                    output.write_bytes(b"prompt")
                    if mutation == "builder":
                        builder.write_bytes(b"changed")
                    if mutation == "corpus":
                        corpus.write_bytes(b"changed")
                    if mutation == "runtime":
                        runtime_component = policy["runtime_receipt"]["components"][0]
                        (Path(policy["install_root"]) / "lib" / runtime_component["filename"]).write_bytes(
                            b"changed")
                    return (
                        subprocess.CompletedProcess(
                            command,
                            0,
                            json.dumps({
                                "target_tokens": 2,
                                "actual_tokens": 2,
                                "byte_count": 6,
                                "add_bos": True,
                                "temporary_directory": str(tmpdir),
                            }),
                            "",
                        ),
                        initial_identity,
                    )

                with mock.patch.dict(os.environ, {"TMPDIR": str(tmpdir)}, clear=True), mock.patch.object(
                        run_matrix, "run_approved_executable", side_effect=run_builder), self.assertRaisesRegex(
                        (RuntimeError, run_matrix.TraceError), message):
                    run_matrix.prepare_prompt(
                        builder=builder,
                        builder_approval_id=TEST_PROMPT_BUILDER_POLICY_ID,
                        builder_policy=policy,
                        builder_policy_sha256=policy_sha256,
                        model=model,
                        corpus=corpus,
                        source_corpus=source_corpus,
                        corpus_name="correctness-prose.txt",
                        corpus_sha256=trace.CORPUS_SHA256["correctness-prose.txt"],
                        output=output,
                        context=3,
                        decode_steps=1,
                    )

    def test_signed_approval_binding_tamper_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "trace"
            with trace.TraceBundleWriter(root, manifest()) as writer:
                add_required_events(writer)
            trace_manifest = trace.strict_json_loads(
                (root / trace.MANIFEST_NAME).read_text(encoding="ascii"))
            trace_manifest["authorization"]["approvals"]["candidate_exporter"]["sha256"] = "f" * 64
            (root / trace.MANIFEST_NAME).write_text(
                trace.canonical_json(trace_manifest) + "\n", encoding="ascii")
            with self.assertRaisesRegex(
                    trace.TraceError, "candidate exporter approval differs from external policy"):
                self._seal_test_bundle(root)

    def test_seal_rejects_protected_bundle_mutations(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            baseline = Path(temp) / "baseline"
            with trace.TraceBundleWriter(baseline, manifest()) as writer:
                add_required_events(writer)
            self._seal_test_bundle(baseline)
            self._read_sealed_bundle(baseline)

            mutations = {
                "manifest": lambda root: (root / trace.MANIFEST_NAME).write_bytes(
                    (root / trace.MANIFEST_NAME).read_bytes().replace(b'"event_count":259', b'"event_count":258')),
                "events": lambda root: (root / trace.EVENTS_NAME).write_bytes(
                    (root / trace.EVENTS_NAME).read_bytes().replace(b'"step":0', b'"step":1', 1)),
                "audit": lambda root: next((root / "audits").rglob("*.json")).write_bytes(b"{}\n"),
                "blob": lambda root: next((root / trace.BLOBS_DIR).iterdir()).write_bytes(b"changed"),
                "provenance": lambda root: next((root / "provenance").iterdir()).write_bytes(b"{}\n"),
                "added": lambda root: (root / "unexpected").write_bytes(b"x"),
                "signature": lambda root: (root / trace.SIGNATURE_NAME).write_bytes(
                    (root / trace.SIGNATURE_NAME).read_bytes().replace(b"SSH SIGNATURE", b"SSH SIGNATURX", 1)),
            }
            for name, mutate in mutations.items():
                with self.subTest(name=name):
                    root = Path(temp) / name
                    shutil.copytree(baseline, root)
                    mutate(root)
                    with self.assertRaises(trace.TraceError):
                        self._read_sealed_bundle(root)

    def test_seal_binds_runtime_lane_challenge_and_run_id(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "trace"
            with trace.TraceBundleWriter(root, manifest()) as writer:
                add_required_events(writer)
            self._seal_test_bundle(root)

            with self.assertRaisesRegex(trace.TraceError, "external challenge"):
                self._trace_bundle_class(
                    root,
                    verifier=self._verifier_for_runtime("llama.cpp", expected_challenge="e" * 64),
                )
            with self.assertRaisesRegex(trace.TraceError, "external run ID"):
                self._trace_bundle_class(
                    root,
                    verifier=self._verifier_for_runtime(
                        "llama.cpp", expected_run_id="strix-llama-other-run"),
                )
            with self.assertRaisesRegex(trace.TraceError, "expected execution lane"):
                wrong_lane = trace.TraceVerifier.for_tests(
                    self.signer_principal,
                    self.test_signers[self.signer_principal]["public_key"],
                    lane=trace.ORACLE_LANE,
                    runtime="ds4",
                    runtime_profile="apple-metal",
                    expected_challenge=TEST_CHALLENGE,
                    expected_run_id=TEST_RUN_IDS["ds4"],
                    verification_unix=int(time.time()),
                    ssh_keygen=self.ssh_keygen,
                )
                self._trace_bundle_class(root, verifier=wrong_lane)

            seen_run_ids: set[str] = set()
            self._trace_bundle_class(
                root,
                verifier=self._verifier_for_runtime("llama.cpp", seen_run_ids=seen_run_ids),
            )
            with self.assertRaisesRegex(trace.TraceError, "reused"):
                self._trace_bundle_class(
                    root,
                    verifier=self._verifier_for_runtime("llama.cpp", seen_run_ids=seen_run_ids),
                )

            with self.assertRaisesRegex(trace.TraceError, "expired"):
                self._trace_bundle_class(
                    root,
                    verifier=self._verifier_for_runtime(
                        "llama.cpp", verification_unix=TEST_AUTH_EXPIRES + 1),
                )

    def test_seal_rejects_nonportable_signed_paths(self) -> None:
        invalid_paths = (
            "../escape.json",
            "./provenance.json",
            "audit//record.json",
            r"audit\record.json",
            "C:/audit/record.json",
            "//server/share.json",
            "%2e%2e/escape.json",
            "audit/\x01.json",
            "audit/\N{LATIN SMALL LETTER E WITH ACUTE}.json",
        )
        for index, invalid in enumerate(invalid_paths):
            with self.subTest(path=invalid), tempfile.TemporaryDirectory() as temp:
                root = Path(temp) / "trace"
                record = manifest()
                record["prompt"]["provenance"]["path"] = invalid
                with trace.TraceBundleWriter(root, record) as writer:
                    add_required_events(writer)
                with self.assertRaisesRegex(trace.TraceError, "trace path"):
                    self._seal_test_bundle(root)

    def test_signing_rejects_bundle_key_and_public_key_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "trace"
            with trace.TraceBundleWriter(root, manifest()) as writer:
                add_required_events(writer)
            bundled_key = root / "private-key"
            shutil.copyfile(self.signing_key, bundled_key)
            bundled_key.chmod(0o600)
            with self.assertRaisesRegex(trace.TraceError, "outside the bundle"):
                trace.seal_bundle(
                    root,
                    private_key=bundled_key,
                    principal=self.signer_principal,
                    expected_lane=trace.CANDIDATE_LANE,
                    expected_challenge=TEST_CHALLENGE,
                    expected_run_id=TEST_RUN_IDS["llama.cpp"],
                    trusted_signers=self.test_signers,
                    ssh_keygen=self.ssh_keygen,
                )
            public_key_path = self.signing_key.with_suffix(".pub")
            public_key_path.chmod(0o600)
            with self.assertRaisesRegex(trace.TraceError, "derive|match"):
                trace.validate_signing_identity(
                    public_key_path,
                    self.signer_principal,
                    trusted_signers=self.test_signers,
                    ssh_keygen=self.ssh_keygen,
                )
            with mock.patch.dict(os.environ, {"SSH_AUTH_SOCK": "/tmp/attacker-agent"}):
                trace.validate_signing_identity(
                    self.signing_key,
                    self.signer_principal,
                    trusted_signers=self.test_signers,
                    ssh_keygen=self.ssh_keygen,
                )

    def test_signer_policy_rejects_open_ssh_options_certificates_and_extra_fields(self) -> None:
        policy = copy.deepcopy(self.test_signers[self.signer_principal])
        public_key = policy["public_key"]
        invalid_keys = (
            f"cert-authority {public_key}",
            public_key.replace("ssh-ed25519", "ssh-ed25519-cert-v01@openssh.com", 1),
            f"{public_key} comment",
            f"{public_key}\n{public_key}",
        )
        for public_key_value in invalid_keys:
            with self.subTest(public_key=public_key_value):
                invalid_policy = copy.deepcopy(policy)
                invalid_policy["public_key"] = public_key_value
                with self.assertRaisesRegex(trace.TraceError, "OpenSSH Ed25519 key"):
                    trace.validate_signing_identity(
                        self.signing_key,
                        self.signer_principal,
                        trusted_signers={self.signer_principal: invalid_policy},
                        ssh_keygen=self.ssh_keygen,
                    )
        with self.assertRaisesRegex(trace.TraceError, "principal is invalid"):
            trace.validate_signing_identity(
                self.signing_key,
                "*",
                trusted_signers={"*": policy},
                ssh_keygen=self.ssh_keygen,
            )

    def test_seal_rejects_noncanonical_duplicate_and_incomplete_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            baseline = Path(temp) / "baseline"
            with trace.TraceBundleWriter(baseline, manifest()) as writer:
                add_required_events(writer)
            self._seal_test_bundle(baseline)

            noncanonical = Path(temp) / "noncanonical"
            shutil.copytree(baseline, noncanonical)
            record = json.loads((noncanonical / trace.MANIFEST_NAME).read_text(encoding="ascii"))
            (noncanonical / trace.MANIFEST_NAME).write_text(
                json.dumps(record, sort_keys=True, indent=2) + "\n",
                encoding="ascii",
            )
            with self.assertRaisesRegex(trace.TraceError, "not canonical"):
                self._read_sealed_bundle(noncanonical)

            duplicate = Path(temp) / "duplicate"
            shutil.copytree(baseline, duplicate)
            manifest_path = duplicate / trace.MANIFEST_NAME
            data = manifest_path.read_text(encoding="ascii")
            manifest_path.write_text(data.replace("{", '{"trace_format":"dsv41-trace",', 1), encoding="ascii")
            with self.assertRaisesRegex(trace.TraceError, "duplicate JSON key"):
                self._read_sealed_bundle(duplicate)

            truncated = Path(temp) / "truncated"
            shutil.copytree(baseline, truncated)
            events_path = truncated / trace.EVENTS_NAME
            events_path.write_bytes(events_path.read_bytes()[:-1])
            with self.assertRaisesRegex(trace.TraceError, "truncated"):
                self._read_sealed_bundle(truncated)

            missing = Path(temp) / "missing"
            shutil.copytree(baseline, missing)
            next((missing / trace.BLOBS_DIR).iterdir()).unlink()
            with self.assertRaises(trace.TraceError):
                self._read_sealed_bundle(missing)

            unsigned = Path(temp) / "unsigned"
            shutil.copytree(baseline, unsigned)
            (unsigned / trace.SIGNATURE_NAME).unlink()
            with self.assertRaisesRegex(trace.TraceError, "bundle-signature"):
                self._read_sealed_bundle(unsigned)

            for field, value in (
                    ("namespace", "wrong-namespace"),
                    ("principal", "other-principal"),
                    ("format", "other-format"),
                    ("version", 2)):
                with self.subTest(envelope_field=field):
                    root = Path(temp) / f"envelope-{field}"
                    shutil.copytree(baseline, root)
                    envelope_path = root / trace.SIGNATURE_NAME
                    envelope = json.loads(envelope_path.read_text(encoding="ascii"))
                    envelope[field] = value
                    envelope_path.write_text(
                        json.dumps(envelope, sort_keys=True, separators=(",", ":")) + "\n",
                        encoding="ascii",
                    )
                    with self.assertRaises(trace.TraceError):
                        self._read_sealed_bundle(root)

    def test_bundle_cannot_supply_signature_trust_inputs(self) -> None:
        for field in ("signer_public_key", "signer_principal", "verifier_path", "allowed_signers"):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as temp:
                root = Path(temp) / "trace"
                record = manifest()
                record[field] = "attacker-controlled"
                with trace.TraceBundleWriter(root, record) as writer:
                    add_required_events(writer)
                with self.assertRaisesRegex(trace.TraceError, f"unexpected {field}"):
                    trace.TraceBundle(root)

    def test_seal_rejects_hard_links_and_post_verify_swaps(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "trace"
            with trace.TraceBundleWriter(root, manifest()) as writer:
                add_required_events(writer)
            self._seal_test_bundle(root)
            bundle = self._read_sealed_bundle(root)
            event = bundle.events[0]
            blob = root / event["blob"]
            replacement = root / "replacement"
            replacement.write_bytes(blob.read_bytes())
            os.replace(replacement, blob)
            with self.assertRaisesRegex(trace.TraceError, "changed after signature verification"):
                bundle.read_blob(event)

        if os.name != "nt":
            with tempfile.TemporaryDirectory() as temp:
                root = Path(temp) / "trace"
                with trace.TraceBundleWriter(root, manifest()) as writer:
                    add_required_events(writer)
                blob = next((root / trace.BLOBS_DIR).iterdir())
                os.link(blob, Path(temp) / "hard-link")
                with self.assertRaisesRegex(trace.TraceError, "hard linked"):
                    self._seal_test_bundle(root)

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
            root = Path(temp).resolve()
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
                    ({"BusProtocol": "Network"}, "NVMe-backed"),
                    ({"BusProtocol": "SATA"}, "NVMe-backed")):
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

    def test_tmpdir_rejects_symlink_components(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            actual = root / "actual"
            child = actual / "child"
            child.mkdir(parents=True)
            link = root / "link"
            link.symlink_to(actual, target_is_directory=True)
            for path in (link, Path(str(link) + "/"), link / ".", link / "child"):
                with self.subTest(path=path), self.assertRaisesRegex(
                        preflight.PreflightError, "symlink"):
                    preflight.require_safe_tmpdir_path(path)
            with self.assertRaisesRegex(preflight.PreflightError, "must not use /mnt/bigspace"):
                preflight.require_safe_tmpdir_path(Path("/mnt/bigspace/escape"))

    def test_full_preflights_reject_unusable_lexical_tmpdir(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            (root / "good" / "tmp").mkdir(parents=True)
            unusable = root / "good" / "missing" / ".." / "tmp"
            actual = root / "actual"
            actual.mkdir()
            link = root / "link"
            link.symlink_to(actual, target_is_directory=True)
            home = root / "home"
            (home / "tmp").mkdir(parents=True)
            self.assertFalse(unusable.is_dir())
            self.assertFalse(os.access(unusable, os.W_OK | os.X_OK))
            self.assertTrue(link.is_symlink())

            def strix_storage(_path: Path, label: str) -> dict[str, object]:
                if label == "temporary directory":
                    raise AssertionError("unusable TMPDIR reached storage attestation")
                return storage_record("/home/test")

            def oracle_storage(
                    _path: Path,
                    label: str,
                    **_kwargs: object) -> dict[str, object]:
                if label == "temporary directory":
                    raise AssertionError("unusable TMPDIR reached storage attestation")
                return metal_storage_record("/Users/oracle/test")

            def assert_rejected(tmpdir: Path, message: str) -> None:
                with mock.patch.dict(
                        preflight.os.environ,
                        {"HIP_LAUNCH_BLOCKING": "1", "TMPDIR": str(tmpdir), "HOME": str(home)},
                        clear=True), mock.patch.object(
                        preflight,
                        "storage_attestation",
                        side_effect=strix_storage):
                    with self.assertRaisesRegex(preflight.PreflightError, message):
                        preflight.run_strix_preflight(
                            model=Path("/home/model.gguf"),
                            prompt=Path("/home/prompt.txt"),
                            output=Path("/home/trace"),
                            repo=Path("/home/repo"),
                            busy_patterns=[],
                        )
                with mock.patch.dict(
                        preflight.os.environ,
                        {"TMPDIR": str(tmpdir), "HOME": str(home)},
                        clear=True), mock.patch.object(
                        preflight,
                        "darwin_storage_attestation",
                        side_effect=oracle_storage):
                    with self.assertRaisesRegex(preflight.PreflightError, message):
                        preflight.run_oracle_preflight(
                            model=Path("/Users/oracle/model.gguf"),
                            prompt=Path("/Users/oracle/prompt.txt"),
                            output=Path("/Users/oracle/trace"),
                            repo=Path("/Users/oracle/repo"),
                            checkout=Path("/Users/oracle/ds4"),
                            busy_patterns=[],
                            accelerator={},
                            runner={},
                        )

            for tmpdir, message in (
                    (unusable, "original lexical path"),
                    (link, "symlink")):
                with self.subTest(tmpdir=tmpdir):
                    assert_rejected(tmpdir, message)

            literal_parent = root / "~"
            literal_target = root / "literal"
            (literal_target / "tmp").mkdir(parents=True)
            literal_parent.symlink_to(literal_target, target_is_directory=True)
            previous_cwd = Path.cwd()
            try:
                os.chdir(root)
                literal_home = Path("~/tmp")
                self.assertTrue(literal_home.is_dir())
                self.assertTrue(os.access(literal_home, os.W_OK | os.X_OK))
                self.assertNotEqual(literal_home.absolute(), literal_home.expanduser().absolute())
                assert_rejected(literal_home, "absolute literal path")
            finally:
                os.chdir(previous_cwd)

            model = root / "model.gguf"
            prompt = root / "prompt.txt"
            output = root / "output"
            repo = root / "repo"
            model.write_bytes(b"model")
            prompt.write_bytes(b"prompt")
            output.mkdir()
            repo.mkdir()
            valid_tmpdir = root / "good" / "tmp"

            def valid_strix_storage(path: Path, _label: str) -> dict[str, object]:
                return storage_record(str(path.resolve()))

            with mock.patch.dict(
                    preflight.os.environ,
                    {"HIP_LAUNCH_BLOCKING": "1", "TMPDIR": str(valid_tmpdir)},
                    clear=True), mock.patch.object(
                    preflight, "storage_attestation", side_effect=valid_strix_storage), mock.patch.object(
                    preflight, "swap_audit", return_value={"enabled": False, "entries": []}), mock.patch.object(
                    preflight, "watchdog_audit", return_value=copy.deepcopy(
                        AUDIT_RECORDS["watchdog"]["data"])), mock.patch.object(
                    preflight, "matching_workloads", return_value=[]), mock.patch.object(
                    preflight, "memory_audit", return_value=copy.deepcopy(
                        AUDIT_RECORDS["memory"]["data"])):
                result = preflight.run_strix_preflight(
                    model=model,
                    prompt=prompt,
                    output=output,
                    repo=repo,
                    busy_patterns=[],
                )
            self.assertEqual(result["storage"]["temporary_directory"]["resolved_path"], str(valid_tmpdir))

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
                root = Path(temp).resolve()
                missing = root / "missing"
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
                actual = root / "actual"
                (actual / "child").mkdir(parents=True)
                link = root / "link"
                link.symlink_to(actual, target_is_directory=True)
                for path in (link, Path(str(link) + "/"), link / ".", link / "child"):
                    with self.subTest(path=path), mock.patch.dict(
                            preflight.os.environ,
                            {"HIP_LAUNCH_BLOCKING": "1", "TMPDIR": str(path)},
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

    def test_canonical_watchdog_artifacts_embed_and_validate(self) -> None:
        revision = preflight.WATCHDOG_REVISION
        repository = Path(__file__).parents[1]
        script = repository / "scripts" / "strix_memory_watchdog.py"
        self.assertTrue(script.is_file())
        source = script.read_bytes()
        self.assertEqual(preflight.sha256_bytes(source), preflight.WATCHDOG_SCRIPT_SHA256)
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", revision, "HEAD"],
            cwd=repository,
            check=True,
        )

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            repo = repository
            watchdog = preflight._load_watchdog_module(script)

            lease_path = root / "watchdog.lease"
            heartbeat_path = root / "watchdog.heartbeat"
            audit_path = root / "watchdog.jsonl"
            child_command = ["python3", "run_matrix.py"]
            arguments = [
                "--soft-gib", "116",
                "--emergency-gib", "118",
                "--grace-seconds", "30",
                "--sample-interval-seconds", "1",
                "--heartbeat-max-age-seconds", "5",
                "--lease-path", str(lease_path),
                "--heartbeat-path", str(heartbeat_path),
                "--audit-path", str(audit_path),
                "--",
                *child_command,
            ]
            config = watchdog.parse_args(arguments)
            paths = config.validate()
            now = datetime.now(timezone.utc).replace(microsecond=0)
            monotonic_ns = 1_000_000_000
            procfs = root / "proc"
            watchdog_pid = os.getpid()
            guardian_pid = watchdog_pid + 100000
            child_pid = guardian_pid + 1

            def proc_stat(pid: int, parent: int, group: int, start: int) -> str:
                fields = ["S", str(parent), str(group), *(["0"] * 16), str(start)]
                return f"{pid} (test) " + " ".join(fields) + "\n"

            for pid in (watchdog_pid, guardian_pid, child_pid):
                (procfs / str(pid)).mkdir(parents=True)
            (procfs / str(watchdog_pid) / "stat").write_text(
                proc_stat(watchdog_pid, 1, watchdog_pid, 1000), encoding="ascii")
            (procfs / str(guardian_pid) / "stat").write_text(
                proc_stat(guardian_pid, watchdog_pid, guardian_pid, 2000), encoding="ascii")
            (procfs / str(child_pid) / "stat").write_text(
                proc_stat(child_pid, guardian_pid, guardian_pid, 3000), encoding="ascii")
            watchdog_argv = [sys.executable, str(script), *arguments]
            watchdog_cmdline = b"\0".join(os.fsencode(value) for value in watchdog_argv) + b"\0"
            (procfs / str(watchdog_pid) / "cmdline").write_bytes(watchdog_cmdline)
            (procfs / str(watchdog_pid) / "exe").symlink_to(Path(sys.executable).resolve())
            (procfs / str(watchdog_pid) / "cwd").symlink_to(repo)

            class FakeProcess:
                pid = guardian_pid

                @staticmethod
                def poll() -> None:
                    return None

                @staticmethod
                def wait(timeout: float | None = None) -> int:
                    del timeout
                    return 0

            guardian = watchdog.GuardianProcess(FakeProcess(), child_pid, -1)
            snapshot = watchdog.HostSnapshot(
                128 * 1024 * 1024 * 1024,
                64 * 1024 * 1024 * 1024,
                (),
            )
            canonical_finals = {}
            for classification, exit_code in (
                    ("procfs_error", watchdog.EXIT_PROCFS_ERROR),
                    ("signal_error", watchdog.EXIT_SIGNAL_ERROR)):
                stream = io.StringIO()
                final_logger = watchdog.AuditLogger(stream, wall_clock=lambda: now)
                with mock.patch.object(watchdog.signal, "pthread_sigmask", return_value=set()):
                    watchdog._kill_and_finish(
                        final_logger,
                        guardian,
                        snapshot,
                        snapshot.used_bytes,
                        classification,
                        exit_code,
                        f"test {classification}",
                        lambda _pid, _signal: "sigkill_sent",
                    )
                final_event = trace.strict_json_loads(stream.getvalue().splitlines()[-1])
                self.assertNotIn("error", final_event)
                trace.validate_watchdog_event(final_event)
                canonical_finals[classification] = final_event

            def fail_signal(_pid: int, _signal: int) -> str:
                raise watchdog.ProcessGroupError("test signal failure")

            signal_stream = io.StringIO()
            signal_logger = watchdog.AuditLogger(signal_stream, wall_clock=lambda: now)
            with mock.patch.object(watchdog.signal, "pthread_sigmask", return_value=set()):
                watchdog._kill_and_finish(
                    signal_logger,
                    guardian,
                    snapshot,
                    snapshot.used_bytes,
                    "procfs_error",
                    watchdog.EXIT_PROCFS_ERROR,
                    "test signal failure",
                    fail_signal,
                )
            signal_final = trace.strict_json_loads(signal_stream.getvalue().splitlines()[-1])
            self.assertEqual(signal_final["classification"], "signal_error")
            self.assertIsInstance(signal_final["error"], str)
            trace.validate_watchdog_event(signal_final)
            canonical_finals["signal-error-detail"] = signal_final

            secondary_stream = io.StringIO()
            secondary_logger = watchdog.AuditLogger(secondary_stream, wall_clock=lambda: now)
            with mock.patch.object(watchdog.signal, "pthread_sigmask", return_value=set()):
                watchdog._emit_final(
                    secondary_logger,
                    "signal_error",
                    watchdog.EXIT_SIGNAL_ERROR,
                    "test secondary error",
                    snapshot,
                    snapshot.used_bytes,
                    guardian,
                    0,
                    "signal_error",
                    "primary signal failure",
                    preserve_primary_on_artifact_error=True,
                    secondary_errors=[{"component": "audit", "detail": "secondary audit failure"}],
                )
            secondary_final = trace.strict_json_loads(secondary_stream.getvalue().splitlines()[-1])
            trace.validate_watchdog_event(secondary_final)
            canonical_finals["signal-error-secondary"] = secondary_final

            class TimeoutProcess(FakeProcess):
                @staticmethod
                def wait(timeout: float | None = None) -> int:
                    raise subprocess.TimeoutExpired(child_command, timeout)

            timeout_stream = io.StringIO()
            timeout_logger = watchdog.AuditLogger(timeout_stream, wall_clock=lambda: now)
            timeout_guardian = watchdog.GuardianProcess(TimeoutProcess(), child_pid, -1)
            with mock.patch.object(watchdog.signal, "pthread_sigmask", return_value=set()):
                watchdog._kill_and_finish(
                    timeout_logger,
                    timeout_guardian,
                    snapshot,
                    snapshot.used_bytes,
                    "procfs_error",
                    watchdog.EXIT_PROCFS_ERROR,
                    "test timeout",
                    lambda _pid, _signal: "sigkill_sent",
                )
            timeout_final = trace.strict_json_loads(timeout_stream.getvalue().splitlines()[-1])
            self.assertEqual(timeout_final["classification"], "termination_timeout")
            self.assertEqual(timeout_final["process_group_status"], "sigkill_timeout")
            self.assertIsInstance(timeout_final["error"], str)
            trace.validate_watchdog_event(timeout_final)
            canonical_finals["termination_timeout"] = timeout_final

            logger = watchdog.AuditLogger(io.StringIO(), wall_clock=lambda: now)
            logger.open_persistent(audit_path)
            state = watchdog._state_fields(
                snapshot, snapshot.used_bytes, None, None, "not_created", "none")
            logger.emit(
                "preflight",
                **state,
                soft_bytes=config.soft_bytes,
                emergency_bytes=config.emergency_bytes,
                strict_ceiling_bytes=watchdog.STRICT_CEILING_BYTES,
            )
            manager = watchdog.LeaseManager(
                config,
                paths,
                process_procfs_root=procfs,
                wall_clock=lambda: now,
                monotonic_ns=lambda: monotonic_ns,
            )
            manager.start(guardian, logger)
            logger.lease_manager = manager
            state = watchdog._state_fields(
                snapshot, snapshot.used_bytes, guardian, None, "active", "none")
            logger.emit("child_started", **state, command=child_command)
            logger.heartbeat(state)
            fd_path = procfs / str(watchdog_pid) / "fd"
            fd_path.mkdir()
            (fd_path / str(logger.persistent_identity()["fd"])).symlink_to(audit_path)

            validated = watchdog.validate_active_lease(
                lease_path,
                expected_script_path=script,
                expected_executable_path=Path(sys.executable),
                expected_soft_bytes=trace.SOFT_MEMORY_LIMIT,
                expected_emergency_bytes=trace.WATCHDOG_EMERGENCY_LIMIT,
                expected_procfs_root=Path("/proc"),
                expected_command=child_command,
                expected_heartbeat_path=heartbeat_path,
                expected_audit_path=audit_path,
                expected_max_heartbeat_age_seconds=5.0,
                current_process_id=child_pid,
                process_procfs_root=procfs,
                monotonic_ns=lambda: monotonic_ns,
                pidfd_open=lambda _: os.open("/dev/null", os.O_RDONLY),
            )
            watchdog_audit = preflight._watchdog_audit_result(
                validated,
                watchdog_revision=revision,
                lease_path=lease_path,
                heartbeat_path=heartbeat_path,
                audit_path=audit_path,
                audit_event_count=2,
                procfs_root=procfs,
            )
            created_unix = int(now.timestamp())
            audit = {
                "created_unix": created_unix,
                "runtime_kind": "strix-rocm",
                "memory": copy.deepcopy(AUDIT_RECORDS["memory"]["data"]),
                "swap": copy.deepcopy(AUDIT_RECORDS["swap"]["data"]),
                "watchdog": watchdog_audit,
                "environment": copy.deepcopy(AUDIT_RECORDS["memory"]["environment"]),
                "storage": copy.deepcopy(STORAGE_ATTESTATION),
                "storage_policy": copy.deepcopy(trace.NO_EXTERNAL_STATE_STORAGE),
                "accelerator": copy.deepcopy(ACCELERATOR_ATTESTATION),
            }
            trace_root = root / "trace"
            with trace.TraceBundleWriter(trace_root, manifest("llama.cpp")) as writer:
                add_required_events(writer)
            audit_sets = {}
            for phase in ("pre", "post"):
                paths = preflight.write_audits(root / f"{phase}-audits", audit)
                preflight.seal_audits(paths)
                audit_sets[phase] = paths
            preflight.bind_embedded_audits(trace_root, audit_sets)
            try:
                trace.TraceBundle(trace_root)
                for classification, final_event in canonical_finals.items():
                    final_root = root / f"trace-{classification}"
                    with trace.TraceBundleWriter(final_root, manifest("llama.cpp")) as writer:
                        add_required_events(writer)
                    for phase in ("pre", "post"):
                        replace_watchdog_events(
                            final_root, phase, [*WATCHDOG_EVENTS, final_event])
                    trace.TraceBundle(final_root)
            finally:
                logger.close()

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
            with self.assertRaisesRegex(trace.TraceError, "audit reference"):
                trace.TraceBundle(root)

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "trace"
            trace_manifest = manifest()
            with trace.TraceBundleWriter(root, trace_manifest) as writer:
                add_required_events(writer)
            (root / trace_manifest["audits"]["pre"]["memory"]["path"]).unlink()
            with self.assertRaisesRegex(trace.TraceError, "missing or not regular"):
                trace.TraceBundle(root)

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "trace"
            trace_manifest = manifest()
            with trace.TraceBundleWriter(root, trace_manifest) as writer:
                add_required_events(writer)
            (root / trace_manifest["audits"]["post"]["watchdog"]["path"]).unlink()
            with self.assertRaisesRegex(trace.TraceError, "missing or not regular"):
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

    def test_rejects_unbound_runtime_build_identity(self) -> None:
        cases = []

        ds4_manifest = manifest("ds4")
        ds4_manifest["build"]["path"] = "/Users/attacker/unrelated-exporter"
        cases.append((ds4_manifest, "ds4 build path"))

        short_revision_manifest = manifest()
        short_revision_manifest["revision"] = "a" * 9
        short_revision_manifest["candidate"]["revision"] = "a" * 9
        cases.append((short_revision_manifest, "candidate exporter approval revision"))

        revision_manifest = manifest()
        revision_manifest["candidate"]["revision"] = "d" * 40
        cases.append((revision_manifest, "candidate exporter approval"))

        executable_manifest = manifest()
        executable_manifest["candidate"]["executable_path"] = "/home/repo/build/bin/other-exporter"
        cases.append((executable_manifest, "candidate exporter approval executable path"))

        library_manifest = manifest()
        library_manifest["build"]["runtime_libraries"][0]["sha256"] = "e" * 64
        library_manifest["build"]["runtime_libraries_post"][0]["sha256"] = "e" * 64
        cases.append((library_manifest, "candidate exporter approval"))

        added_module_manifest = manifest()
        added_module_manifest["build"]["runtime_module_monitor"]["project_additions"] = [
            {"path": "lib/libggml-injected.module"}]
        cases.append((added_module_manifest, "runtime module addition during trace generation"))

        incomplete_monitor_manifest = manifest()
        incomplete_monitor_manifest["build"]["runtime_module_monitor"]["checked_after_trace"] = False
        cases.append((incomplete_monitor_manifest, "runtime module monitor did not complete"))

        library_path_manifest = manifest()
        library_path_manifest["build"]["runtime_libraries"][0]["path"] = (
            "/home/repo/build/bin/../substituted/libllama-common.so")
        library_path_manifest["build"]["runtime_libraries_post"][0]["path"] = (
            "/home/repo/build/bin/../substituted/libllama-common.so")
        cases.append((library_path_manifest, "runtime library path is not canonical"))

        external_library_manifest = manifest()
        external_library_manifest["build"]["runtime_libraries"][0]["path"] = (
            "/Users/attacker/libggml-injected.dylib")
        external_library_manifest["build"]["runtime_libraries"].sort(key=lambda item: item["path"])
        external_library_manifest["build"]["runtime_libraries_post"] = copy.deepcopy(
            external_library_manifest["build"]["runtime_libraries"])
        cases.append((external_library_manifest, "outside the exporter runtime directory"))

        omitted_library_manifest = manifest()
        omitted_library_manifest["build"]["runtime_libraries"] = [
            library
            for library in omitted_library_manifest["build"]["runtime_libraries"]
            if library["role"] != "selected-backend"
        ]
        omitted_library_manifest["build"]["runtime_libraries_post"] = copy.deepcopy(
            omitted_library_manifest["build"]["runtime_libraries"])
        cases.append((omitted_library_manifest, "candidate exporter approval receipt"))

        duplicate_path_manifest = manifest()
        duplicate_path_manifest["build"]["runtime_libraries"][1]["path"] = (
            duplicate_path_manifest["build"]["runtime_libraries"][0]["path"])
        duplicate_path_manifest["build"]["runtime_libraries_post"] = copy.deepcopy(
            duplicate_path_manifest["build"]["runtime_libraries"])
        cases.append((duplicate_path_manifest, "runtime library path is duplicated"))

        duplicate_role_manifest = manifest()
        duplicate_role_manifest["build"]["runtime_libraries"][1]["role"] = (
            duplicate_role_manifest["build"]["runtime_libraries"][0]["role"])
        duplicate_role_manifest["build"]["runtime_libraries_post"] = copy.deepcopy(
            duplicate_role_manifest["build"]["runtime_libraries"])
        cases.append((duplicate_role_manifest, "runtime library role is invalid"))

        unknown_component_manifest = manifest()
        unknown_component_manifest["build"]["runtime_libraries"][0]["component"] = "ggml-injected"
        unknown_component_manifest["build"]["runtime_libraries_post"][0]["component"] = "ggml-injected"
        cases.append((unknown_component_manifest, "candidate exporter approval runtime receipt component"))

        revision_library_manifest = manifest()
        revision_library = next(
            library
            for library in revision_library_manifest["build"]["runtime_libraries"]
            if library["role"] == "build-info")
        revision_library["revision"] = "b" * 40
        next(
            library
            for library in revision_library_manifest["build"]["runtime_libraries_post"]
            if library["role"] == "build-info")["revision"] = "b" * 40
        cases.append((revision_library_manifest, "candidate exporter approval runtime receipt revision"))

        unexpected_revision_manifest = manifest()
        unexpected_revision = next(
            library
            for library in unexpected_revision_manifest["build"]["runtime_libraries"]
            if library["role"].startswith("runtime:"))
        unexpected_revision["revision"] = "a" * 40
        next(
            library
            for library in unexpected_revision_manifest["build"]["runtime_libraries_post"]
            if library["role"].startswith("runtime:"))["revision"] = "a" * 40
        cases.append((unexpected_revision_manifest, "candidate exporter approval runtime receipt revision"))

        for trace_manifest, message in cases:
            with self.subTest(message=message), tempfile.TemporaryDirectory() as temp:
                root = Path(temp) / "trace"
                with trace.TraceBundleWriter(root, trace_manifest) as writer:
                    add_required_events(writer)
                with self.assertRaisesRegex(trace.TraceError, message):
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
                ("ds4", ACCELERATOR_ATTESTATION, "runtime profile"),
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

    def test_rejects_ds4_sata_storage_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "trace"
            with trace.TraceBundleWriter(root, manifest("ds4")) as writer:
                add_required_events(writer)
            for phase in ("pre", "post"):
                record = json.loads(json.dumps(DS4_AUDIT_RECORDS["memory"]))
                record["storage"]["model"]["bus_protocol"] = "SATA"
                replace_audit_record(root, phase, "memory", record)
            with self.assertRaisesRegex(trace.TraceError, "not NVMe-backed"):
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
            with self.assertRaisesRegex(trace.TraceError, "runtime profile"):
                trace.TraceBundle(root)

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "trace"
            trace_manifest = manifest("ds4")
            del trace_manifest["accelerator"]["runtime_kind"]
            with trace.TraceBundleWriter(root, trace_manifest) as writer:
                add_required_events(writer)
            with self.assertRaisesRegex(trace.TraceError, "runtime profile"):
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

    def test_native_complete_manifest_writer_validates(self) -> None:
        production_binary = Path(os.environ.get(
            "DSV41_NATIVE_TRACE_BINARY",
            Path(__file__).parents[1] / "build-harness" / "bin" / "llama-deepseek-v41-trace",
        ))
        manifest_binary = Path(os.environ.get(
            "DSV41_NATIVE_MANIFEST_BINARY",
            Path(__file__).parents[1] / "build-harness" / "bin" / "test-deepseek41-trace-manifest",
        ))
        if not production_binary.is_file() or not manifest_binary.is_file():
            self.skipTest("native trace exporter and manifest harness are not built")
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "trace with space"
            with trace.TraceBundleWriter(root, manifest("llama.cpp")) as writer:
                add_required_events(writer)
            manifest_path = root / trace.MANIFEST_NAME
            fixture = trace.strict_json_loads(manifest_path.read_text(encoding="ascii"))
            writer_input = {
                key: fixture[key]
                for key in ("model", "prompt", "audits", "expected", "event_count")
            }
            input_path = Path(temp) / "manifest-input.json"
            input_path.write_text(
                json.dumps(writer_input, sort_keys=True, separators=(",", ":")) + "\n",
                encoding="ascii",
            )
            manifest_path.unlink()
            command = [
                str(manifest_binary.resolve()),
                "--write-test-manifest",
                str(input_path),
                str(manifest_path),
            ]
            subprocess.run(command, check=True)
            native = trace.strict_json_loads(manifest_path.read_text(encoding="ascii"))
            revision = subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=Path(__file__).parents[1],
                text=True,
            ).strip()
            version = subprocess.run(
                [str(production_binary.resolve()), "--version"],
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertIn(f"commit {revision}", version.stdout)
            self.assertEqual(native["revision"], revision)
            self.assertEqual(native["build"]["path"], str(manifest_binary.resolve()))
            self.assertIn("test-only manifest harness", native["build"]["info"])
            self.assertEqual(native["accelerator"]["runtime_kind"], "native-test")
            self.assertEqual(native["accelerator"]["device_type"], "cpu")
            self.assertIs(native["accelerator"]["test_only"], True)
            components = native["build"]["runtime_profile"]["components"]
            self.assertEqual(components, sorted(components))
            self.assertEqual(
                native["build"]["runtime_libraries_post"],
                native["build"]["runtime_libraries"],
            )
            self.assertEqual(
                native["build"]["runtime_module_monitor"],
                {
                    "mechanism": "dyld-add-image" if sys.platform == "darwin" else "pre-post-snapshot",
                    "checked_after_trace": True,
                    "project_additions": [],
                },
            )
            self.assertEqual(
                {library["component"] for library in native["build"]["runtime_libraries"]},
                set(components),
            )
            self.assertEqual(
                len({library["role"] for library in native["build"]["runtime_libraries"]}),
                len(components),
            )
            self.assertEqual(
                [library["path"] for library in native["build"]["runtime_libraries"]],
                sorted(library["path"] for library in native["build"]["runtime_libraries"]),
            )
            for library in native["build"]["runtime_libraries"]:
                expected_revision = (
                    revision
                    if library["component"] in {"llama-common", "ggml-base"}
                    else None
                )
                self.assertEqual(library["revision"], expected_revision)
            receipt = {
                "format": "dsv41-runtime-receipt",
                "version": 1,
                "revision": revision,
                "profile": native["build"]["runtime_profile"]["name"],
                "components": sorted(
                    [
                        {
                            "component": library["component"],
                            "filename": library["filename"],
                            "sha256": library["sha256"],
                            "revision": library["revision"],
                        }
                        for library in native["build"]["runtime_libraries"]
                    ],
                    key=lambda item: item["component"],
                ),
            }
            self.assertEqual(
                native["build"]["runtime_receipt_sha256"],
                trace.sha256_bytes(trace.canonical_json(receipt).encode("ascii")),
            )
            self.assertIsInstance(native["environment"]["command"], str)
            self.assertEqual(json.loads(native["environment"]["command"]), command)
            self.assertIs(native["config"]["flash_attention"], False)
            self.assertEqual(native["config"]["runtime_kind"], "native-test")
            self.assertEqual(native["config"]["device_type"], "cpu")
            self.assertIs(native["config"]["test_only"], True)
            self.assertEqual(native["storage_policy"], trace.NO_EXTERNAL_STATE_STORAGE)
            attestation = fixture["candidate"]
            attestation["revision"] = revision
            attestation["executable_path"] = str(manifest_binary.resolve())
            attestation["executable_sha256"] = trace.sha256_file(manifest_binary)
            with self.assertRaisesRegex(run_llama.PreflightError, "test-only manifest harness"):
                run_llama.bind_candidate_attestation(
                    root,
                    attestation,
                    native["accelerator"],
                    manifest_binary,
                    trace.sha256_file(manifest_binary),
                    {
                        "runtime_profile": native["build"]["runtime_profile"],
                        "runtime_receipt": receipt,
                    },
                )
            with self.assertRaisesRegex(trace.TraceError, "execution authorization is missing"):
                trace.seal_bundle(
                    root,
                    private_key=self.signing_key,
                    principal=self.signer_principal,
                    expected_lane=trace.CANDIDATE_LANE,
                    expected_challenge=TEST_CHALLENGE,
                    expected_run_id=TEST_RUN_IDS["llama.cpp"],
                    trusted_signers=self.test_signers,
                    ssh_keygen=self.ssh_keygen,
                )

            for protected_field in (
                    "accelerator", "authorization", "build", "candidate", "comparison", "config",
                    "environment", "paths", "revision", "runtime", "storage_policy"):
                protected_input = dict(writer_input)
                protected_input[protected_field] = fixture.get(protected_field, {})
                input_path.write_text(
                    json.dumps(protected_input, sort_keys=True, separators=(",", ":")) + "\n",
                    encoding="ascii",
                )
                rejected = subprocess.run(command, check=False, capture_output=True, text=True)
                self.assertNotEqual(rejected.returncode, 0)
                self.assertIn(f"unexpected field: {protected_field}", rejected.stderr)

            for option in ("--dsv41-manifest-writer-probe", "--dsv41-runtime-module-path-probe"):
                rejected = subprocess.run(
                    [str(production_binary.resolve()), option],
                    check=False,
                    capture_output=True,
                    text=True,
                )
                self.assertNotEqual(rejected.returncode, 0)

            rejected = subprocess.run(
                [str(production_binary.resolve()), "--dsv41-attest-device", "CPU"],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn("selected execution device must be ROCm0", rejected.stderr)

    def test_runtime_build_validator_rejects_closure_substitution(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            binary_directory = root / "build" / "bin"
            library_directory = root / "build" / "lib"
            binary_directory.mkdir(parents=True)
            library_directory.mkdir()
            exporter = binary_directory / "llama-deepseek-v41-trace"
            exporter.write_bytes(b"exporter")
            records = []
            for component, name, role, content in (
                    ("ggml", "libggml.so", "runtime:ggml", b"ggml"),
                    ("ggml-base", "libggml-base.so", "ggml", b"base"),
                    ("ggml-blas", "libggml-blas.so", "runtime:ggml-blas", b"blas"),
                    ("ggml-hip", "libggml-hip.so", "selected-backend", b"backend"),
                    ("llama", "libllama.so", "llama", b"llama"),
                    ("llama-common", "libllama-common.so", "build-info", b"build")):
                path = library_directory / name
                path.write_bytes(content)
                records.append({
                    "component": component,
                    "filename": name,
                    "path": str(path.resolve()),
                    "sha256": trace.sha256_file(path),
                    "role": role,
                    "revision": "a" * 40 if component in {"llama-common", "ggml-base"} else None,
                })
            records.sort(key=lambda record: record["path"])
            components = sorted(record["component"] for record in records)
            receipt = {
                "format": "dsv41-runtime-receipt",
                "version": 1,
                "revision": "a" * 40,
                "profile": "sibling-lib",
                "components": sorted(
                    [
                        {
                            "component": record["component"],
                            "filename": record["filename"],
                            "sha256": record["sha256"],
                            "revision": record["revision"],
                        }
                        for record in records
                    ],
                    key=lambda item: item["component"],
                ),
            }
            build_manifest = {
                "revision": "a" * 40,
                "build": {
                    "path": str(exporter.resolve()),
                    "sha256": trace.sha256_file(exporter),
                    "info": "test",
                    "runtime_profile": {
                        "name": "sibling-lib",
                        "components": components,
                        "selected_backend_component": "ggml-hip",
                    },
                    "runtime_receipt_sha256": trace.sha256_bytes(
                        trace.canonical_json(receipt).encode("ascii")),
                    "runtime_libraries": records,
                    "runtime_libraries_post": copy.deepcopy(records),
                    "runtime_module_monitor": {
                        "mechanism": "pre-post-snapshot",
                        "checked_after_trace": True,
                        "project_additions": [],
                    },
                },
            }
            approval = {
                "runtime_profile": copy.deepcopy(build_manifest["build"]["runtime_profile"]),
                "runtime_receipt": copy.deepcopy(receipt),
            }
            libraries_digest, receipt_digest = run_llama.validate_runtime_build(
                build_manifest,
                exporter=exporter,
                exporter_sha256=trace.sha256_file(exporter),
                candidate_revision="a" * 40,
                approval=approval,
            )
            self.assertEqual(
                libraries_digest,
                trace.sha256_bytes(trace.canonical_json({
                    "pre": records,
                    "post": records,
                }).encode("ascii")),
            )
            self.assertEqual(receipt_digest, build_manifest["build"]["runtime_receipt_sha256"])

            cases = []
            omitted = copy.deepcopy(build_manifest)
            omitted["build"]["runtime_libraries"] = [
                library
                for library in omitted["build"]["runtime_libraries"]
                if library["role"] != "selected-backend"
            ]
            omitted["build"]["runtime_libraries_post"] = copy.deepcopy(
                omitted["build"]["runtime_libraries"])
            cases.append((omitted, "set differs from the runtime profile"))

            changed_hash = copy.deepcopy(build_manifest)
            changed_hash["build"]["runtime_libraries"][0]["sha256"] = "f" * 64
            changed_hash["build"]["runtime_libraries_post"][0]["sha256"] = "f" * 64
            cases.append((changed_hash, "SHA-256 mismatch"))

            changed_post = copy.deepcopy(build_manifest)
            changed_post["build"]["runtime_libraries_post"][0]["sha256"] = "f" * 64
            cases.append((changed_post, "closure changed during trace generation"))

            added_module = copy.deepcopy(build_manifest)
            added_module["build"]["runtime_module_monitor"]["project_additions"] = [
                {"path": "lib/libggml-injected.module"}]
            cases.append((added_module, "runtime module addition during trace generation"))

            incomplete_monitor = copy.deepcopy(build_manifest)
            incomplete_monitor["build"]["runtime_module_monitor"]["checked_after_trace"] = False
            cases.append((incomplete_monitor, "runtime module monitor did not complete"))

            changed_revision = copy.deepcopy(build_manifest)
            revision_record = next(
                library
                for library in changed_revision["build"]["runtime_libraries"]
                if library["role"] == "build-info")
            revision_record["revision"] = "b" * 40
            next(
                library
                for library in changed_revision["build"]["runtime_libraries_post"]
                if library["role"] == "build-info")["revision"] = "b" * 40
            cases.append((changed_revision, "revision mismatch"))

            duplicate_role = copy.deepcopy(build_manifest)
            role_records = duplicate_role["build"]["runtime_libraries"]
            next(library for library in role_records if library["role"].startswith("runtime:"))["role"] = "llama"
            duplicate_role["build"]["runtime_libraries_post"] = copy.deepcopy(role_records)
            cases.append((duplicate_role, "role is invalid"))

            duplicate_component = copy.deepcopy(build_manifest)
            duplicate_component["build"]["runtime_libraries"][1]["component"] = (
                duplicate_component["build"]["runtime_libraries"][0]["component"])
            duplicate_component["build"]["runtime_libraries_post"] = copy.deepcopy(
                duplicate_component["build"]["runtime_libraries"])
            cases.append((duplicate_component, "component is invalid"))

            selected_backend = copy.deepcopy(build_manifest)
            selected_backend["build"]["runtime_profile"]["selected_backend_component"] = "ggml-blas"
            cases.append((selected_backend, "selected backend component is not ggml-hip"))

            filename = copy.deepcopy(build_manifest)
            filename["build"]["runtime_libraries"][0]["filename"] = "other.so"
            filename["build"]["runtime_libraries_post"] = copy.deepcopy(
                filename["build"]["runtime_libraries"])
            cases.append((filename, "path differs from the exact runtime profile"))

            receipt_digest = copy.deepcopy(build_manifest)
            receipt_digest["build"]["runtime_receipt_sha256"] = "f" * 64
            cases.append((receipt_digest, "runtime receipt SHA-256 mismatch"))

            external = root / "external" / "libggml-blas.so"
            external.parent.mkdir()
            external.write_bytes(b"blas")
            external_manifest = copy.deepcopy(build_manifest)
            external_record = next(
                library
                for library in external_manifest["build"]["runtime_libraries"]
                if library["component"] == "ggml-blas")
            external_record["path"] = str(external.resolve())
            external_manifest["build"]["runtime_libraries"].sort(key=lambda record: record["path"])
            external_manifest["build"]["runtime_libraries_post"] = copy.deepcopy(
                external_manifest["build"]["runtime_libraries"])
            cases.append((external_manifest, "path differs from the exact runtime profile"))

            catalogued_not_profile = copy.deepcopy(build_manifest)
            injected = library_directory / "libggml-injected.so"
            injected.write_bytes(b"injected")
            catalogued_not_profile["build"]["runtime_libraries"].append({
                "component": "ggml-injected",
                "filename": injected.name,
                "path": str(injected.resolve()),
                "sha256": trace.sha256_file(injected),
                "role": "runtime:ggml-injected",
                "revision": None,
            })
            catalogued_not_profile["build"]["runtime_libraries"].sort(key=lambda record: record["path"])
            catalogued_not_profile["build"]["runtime_libraries_post"] = copy.deepcopy(
                catalogued_not_profile["build"]["runtime_libraries"])
            cases.append((catalogued_not_profile, "component is invalid"))

            duplicate_path = copy.deepcopy(build_manifest)
            duplicate_path["build"]["runtime_libraries"][1]["path"] = (
                duplicate_path["build"]["runtime_libraries"][0]["path"])
            duplicate_path["build"]["runtime_libraries_post"] = copy.deepcopy(
                duplicate_path["build"]["runtime_libraries"])
            cases.append((duplicate_path, "duplicated or unsorted"))

            for candidate, message in cases:
                with self.subTest(message=message), self.assertRaisesRegex(
                        run_llama.PreflightError, message):
                    run_llama.validate_runtime_build(
                        candidate,
                        exporter=exporter,
                        exporter_sha256=trace.sha256_file(exporter),
                        candidate_revision="a" * 40,
                        approval=approval,
                    )

    @unittest.skipUnless(sys.platform.startswith(("darwin", "linux")), "loader injection test")
    def test_native_rejects_injected_project_library(self) -> None:
        binary = Path(os.environ.get(
            "DSV41_NATIVE_TRACE_BINARY",
            Path(__file__).parents[1] / "build-harness" / "bin" / "llama-deepseek-v41-trace",
        ))
        injected = Path(os.environ.get(
            "DSV41_NATIVE_INJECT_LIBRARY",
            Path(__file__).parents[1] / "build-harness" / "bin" / "libggml-injected.module",
        ))
        if not binary.is_file() or not injected.is_file():
            self.skipTest("native trace exporter and injected test library are not built")
        with tempfile.TemporaryDirectory() as temp:
            external = Path(temp) / injected.name
            shutil.copy2(injected, external)
            variable = "DYLD_INSERT_LIBRARIES" if sys.platform == "darwin" else "LD_PRELOAD"
            inside = binary.parent / "renamed-injected.module"
            shutil.copy2(injected, inside)
            try:
                for name, injected_path in (("outside", external), ("renamed-inside", inside)):
                    with self.subTest(name=name):
                        environment = dict(os.environ)
                        environment[variable] = str(injected_path)
                        rejected = subprocess.run(
                            [str(binary.resolve()), "--version"],
                            check=False,
                            capture_output=True,
                            text=True,
                            env=environment,
                        )
                        self.assertNotEqual(rejected.returncode, 0)
                        self.assertIn("forbids loader override", rejected.stderr)
            finally:
                inside.unlink(missing_ok=True)

    def test_python_runner_rejects_loader_overrides(self) -> None:
        for variable in trace.FORBIDDEN_LOADER_ENVIRONMENT:
            with self.subTest(variable=variable), mock.patch.dict(
                    os.environ, {variable: "/tmp/untrusted-runtime"}, clear=True):
                with self.assertRaisesRegex(trace.TraceError, variable):
                    trace.reject_loader_overrides()

    def test_prompt_builder_result_becomes_strict_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            builder = root / "install" / "bin" / "llama-deepseek-v41-prompt-builder"
            model = root / "model.gguf"
            corpus = root / "corpus.txt"
            source_root = Path(__file__).parents[1]
            source_corpus = source_root / "tests" / "corpus" / "correctness-prose.txt"
            output = root / "prompt.txt"
            tmpdir = root / "tmp"
            builder.parent.mkdir(parents=True)
            builder.write_bytes(b"builder")
            builder.chmod(0o755)
            model.write_bytes(b"model")
            shutil.copyfile(source_corpus, corpus)
            tmpdir.mkdir()
            builder = builder.resolve()
            model = model.resolve()
            corpus = corpus.resolve()
            output = output.resolve()
            tmpdir = tmpdir.resolve()
            builder_policy = fixture_prompt_builder_policy(
                b"prompt",
                builder_path=str(builder.resolve()),
                builder_sha256=trace.sha256_file(builder),
                source_root=str(source_root.resolve()),
            )
            materialize_policy_runtime(builder_policy)
            _validated, builder_policy_sha256 = trace.prompt_builder_approval(
                TEST_PROMPT_BUILDER_POLICY_ID,
                policies={TEST_PROMPT_BUILDER_POLICY_ID: builder_policy},
            )
            builder_identity = run_matrix.approved_executable_identity(
                builder,
                expected_path=builder_policy["executable_path"],
                expected_sha256=builder_policy["executable_sha256"],
                label="prompt builder",
            )

            def run_builder(command, **_kwargs):
                output.write_bytes(b"prompt")
                return (
                    subprocess.CompletedProcess(
                        command,
                        0,
                        json.dumps({
                            "target_tokens": 2,
                            "actual_tokens": 2,
                            "byte_count": 6,
                            "add_bos": True,
                            "temporary_directory": str(tmpdir.resolve()),
                        }),
                        "",
                    ),
                    builder_identity,
                )

            with mock.patch.dict(os.environ, {"TMPDIR": str(tmpdir)}, clear=True), mock.patch.object(
                    run_matrix, "run_approved_executable", side_effect=run_builder), mock.patch.object(
                    sys, "stderr", io.StringIO()):
                result = run_matrix.prepare_prompt(
                    builder=builder,
                    builder_approval_id=TEST_PROMPT_BUILDER_POLICY_ID,
                    builder_policy=builder_policy,
                    builder_policy_sha256=builder_policy_sha256,
                    model=model,
                    corpus=corpus,
                    source_corpus=source_corpus,
                    corpus_name="correctness-prose.txt",
                    corpus_sha256=trace.CORPUS_SHA256["correctness-prose.txt"],
                    output=output,
                    context=3,
                    decode_steps=1,
                )
            provenance_path = Path(result["provenance_path"])
            provenance = trace.strict_json_loads(provenance_path.read_text(encoding="ascii"))
            self.assertEqual(
                set(provenance),
                {
                    "format", "version", "corpus_name", "corpus_sha256", "corpus_path",
                    "model_sha256", "prompt_sha256", "prompt_byte_count", "context",
                    "decode_steps", "builder_approval_id", "builder_approval_sha256",
                    "builder_path", "builder_sha256", "builder_revision",
                    "builder_runtime_profile", "target_tokens", "actual_tokens",
                },
            )
            preflight.validate_prompt_provenance(
                provenance_path,
                prompt=output,
                corpus_name="correctness-prose.txt",
                corpus_sha256=trace.CORPUS_SHA256["correctness-prose.txt"],
                model_sha256=trace.MODEL_SHA256,
                context=3,
                decode_steps=1,
                target_tokens=2,
                builder_approval_id=TEST_PROMPT_BUILDER_POLICY_ID,
                builder_policy=builder_policy,
                builder_policy_sha256=builder_policy_sha256,
                path_resolver=lambda path, _label: path.resolve(),
            )

    def test_rejects_cross_runtime_and_unknown_audit_envelopes(self) -> None:
        for mutation, message in (
                (lambda value: value["audits"].update({"unknown": {}}), "audit envelope"),
                (lambda value: value["audits"]["pre"].update({"watchdog": {}}), "audit reference")):
            with tempfile.TemporaryDirectory() as temp:
                root = Path(temp) / "trace"
                trace_manifest = manifest("ds4")
                mutation(trace_manifest)
                with trace.TraceBundleWriter(root, trace_manifest) as writer:
                    add_required_events(writer)
                with self.assertRaisesRegex(trace.TraceError, message):
                    trace.TraceBundle(root)

    def test_requires_no_external_cache_or_state_storage(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "trace"
            trace_manifest = manifest("ds4")
            trace_manifest["storage_policy"]["external_cache_paths"] = ["/Users/oracle/cache"]
            with trace.TraceBundleWriter(root, trace_manifest) as writer:
                add_required_events(writer)
            with self.assertRaisesRegex(trace.TraceError, "storage policy"):
                trace.TraceBundle(root)

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "trace"
            with trace.TraceBundleWriter(root, manifest("ds4")) as writer:
                add_required_events(writer)
            record = json.loads(json.dumps(DS4_AUDIT_RECORDS["memory"]))
            del record["storage_policy"]
            replace_audit_record(root, "pre", "memory", record)
            with self.assertRaisesRegex(trace.TraceError, "storage_policy"):
                trace.TraceBundle(root)

    def test_accepts_authentic_watchdog_lease_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "trace"
            with trace.TraceBundleWriter(root, manifest("llama.cpp")) as writer:
                add_required_events(writer)
            trace.TraceBundle(root)

        for field, value, message in (
                ("file_inode", True, "file_inode is invalid"),
                ("file_mode", 0o644, "file mode is invalid"),
                ("audit_sha256", "f" * 64, "live and embedded audit SHA-256 differ")):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as temp:
                root = Path(temp) / "trace"
                with trace.TraceBundleWriter(root, manifest("llama.cpp")) as writer:
                    add_required_events(writer)
                for phase in ("pre", "post"):
                    record = copy.deepcopy(AUDIT_RECORDS["watchdog"])
                    record["data"]["audit"]["path"] = (
                        f"audits/{phase}/{WATCHDOG_JSONL_SHA256}.jsonl")
                    record["data"][field] = value
                    replace_audit_record(root, phase, "watchdog", record)
                with self.assertRaisesRegex(trace.TraceError, message):
                    trace.TraceBundle(root)

    def test_enforces_watchdog_final_error_classification(self) -> None:
        self.assertEqual(trace.WATCHDOG_REQUIRED_ERROR_CLASSIFICATIONS, {
            "configuration_error",
            "internal_error",
            "launch_error",
            "lease_error",
            "termination_timeout",
        })
        self.assertEqual(trace.WATCHDOG_OPTIONAL_ERROR_CLASSIFICATIONS, {
            "procfs_error",
            "signal_error",
        })
        terminal = {
            key: copy.deepcopy(value)
            for key, value in WATCHDOG_EVENTS[1].items()
            if key not in {"command", "event_id", "parent_event_sha256", "record_sha256"}
        }
        terminal.update({
            "event": "final",
            "classification": "internal_error",
            "exit_code": 1,
            "error": "test internal error",
        })

        def assert_watchdog_final_rejected(candidate: dict[str, object], message: str) -> None:
            with tempfile.TemporaryDirectory() as temp:
                root = Path(temp) / "trace"
                with trace.TraceBundleWriter(root, manifest("llama.cpp")) as writer:
                    add_required_events(writer)
                for phase in ("pre", "post"):
                    replace_watchdog_events(root, phase, [*WATCHDOG_EVENTS, candidate])
                with self.assertRaisesRegex(trace.TraceError, message):
                    trace.TraceBundle(root)
                with self.assertRaisesRegex(trace.TraceError, message):
                    trace.command_validate(Namespace(bundle=root))

        def assert_watchdog_final_valid(candidate: dict[str, object]) -> None:
            with tempfile.TemporaryDirectory() as temp:
                root = Path(temp) / "trace"
                with trace.TraceBundleWriter(root, manifest("llama.cpp")) as writer:
                    add_required_events(writer)
                for phase in ("pre", "post"):
                    replace_watchdog_events(root, phase, [*WATCHDOG_EVENTS, candidate])
                trace.TraceBundle(root)
                with mock.patch("sys.stdout", new_callable=io.StringIO):
                    self.assertEqual(trace.command_validate(Namespace(bundle=root)), 0)

        assert_watchdog_final_valid(terminal)

        for classification, error in (
                ("procfs_error", None),
                ("procfs_error", "initial snapshot failed"),
                ("signal_error", None),
                ("signal_error", "cannot signal process group")):
            with self.subTest(classification=classification, error=error):
                candidate = copy.deepcopy(terminal)
                candidate["classification"] = classification
                if error is None:
                    candidate.pop("error")
                else:
                    candidate["error"] = error
                assert_watchdog_final_valid(candidate)

        candidate = copy.deepcopy(terminal)
        candidate.update({
            "classification": "signal_error",
            "error": "primary signal failure",
            "secondary_errors": [{
                "component": "audit",
                "detail": "secondary audit failure",
            }],
        })
        assert_watchdog_final_valid(candidate)

        for classification, error in (
                ("internal_error", "internal failure"),
                ("signal_error", None)):
            with self.subTest(classification=classification, secondary=True):
                candidate = copy.deepcopy(terminal)
                candidate["classification"] = classification
                candidate["secondary_errors"] = [{
                    "component": "audit",
                    "detail": "secondary audit failure",
                }]
                if error is None:
                    candidate.pop("error")
                else:
                    candidate["error"] = error
                assert_watchdog_final_rejected(candidate, "require a primary signal error")

        for classification, error, secondary_errors, message in (
                ("child_exit", None, None, "require a primary signal error"),
                ("internal_error", "internal failure", None, "require a primary signal error"),
                ("signal_error", None, None, "require a primary signal error"),
                ("signal_error", "primary signal failure", None, "secondary errors are invalid"),
                ("signal_error", "primary signal failure", [], "secondary errors are invalid")):
            with self.subTest(
                    classification=classification,
                    secondary_errors=secondary_errors):
                candidate = copy.deepcopy(terminal)
                candidate["classification"] = classification
                candidate["secondary_errors"] = secondary_errors
                if error is None:
                    candidate.pop("error")
                else:
                    candidate["error"] = error
                assert_watchdog_final_rejected(candidate, message)

        for classification, error in (
                ("internal_error", None),
                ("child_exit", "fabricated error")):
            with self.subTest(classification=classification):
                candidate = copy.deepcopy(terminal)
                candidate["classification"] = classification
                if error is None:
                    candidate.pop("error")
                else:
                    candidate["error"] = error
                assert_watchdog_final_rejected(candidate, "error presence does not match")

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
                record["data"]["audit_sha256"] = digest
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
                "--prompt-builder-policy-id", TEST_PROMPT_BUILDER_POLICY_ID,
                "--approval-policy", str(root / "approval.json"),
                "--approval-signature", str(root / "approval.sig"),
                "--approval-principal", "unapproved",
                "--signer-principal", self.signer_principals["ds4"],
                "--signing-key", str(self.signing_keys["ds4"]),
                "--execution-challenge", TEST_CHALLENGE,
                "--run-id", TEST_RUN_IDS["ds4"],
                "--authorization-issued-unix", str(TEST_AUTH_ISSUED),
                "--authorization-expires-unix", str(TEST_AUTH_EXPIRES),
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

    def test_bundle_validation_and_comparison_reject_unapproved_ds4_exporter(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            ds4_root = root / "ds4"
            llama_root = root / "llama"
            with trace.TraceBundleWriter(ds4_root, manifest("ds4")) as writer:
                add_required_events(writer)
            with trace.TraceBundleWriter(llama_root, manifest("llama.cpp")) as writer:
                add_required_events(writer)
            approved = dict(trace.APPROVED_EXPORTERS)
            try:
                trace.APPROVED_EXPORTERS.clear()
                with self.assertRaisesRegex(trace.TraceError, "exporter is not approved"):
                    trace.TraceBundle(ds4_root)
                with self.assertRaisesRegex(trace.TraceError, "exporter is not approved"):
                    trace.command_validate(Namespace(bundle=ds4_root))
                with self.assertRaisesRegex(trace.TraceError, "exporter is not approved"):
                    trace.command_compare(Namespace(left=ds4_root, right=llama_root, report=None))
            finally:
                trace.APPROVED_EXPORTERS.clear()
                trace.APPROVED_EXPORTERS.update(approved)

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
            second_manifest = manifest("llama.cpp")
            second_manifest["authorization"]["run_id"] = "strix-llama-test-run-2"
            with trace.TraceBundleWriter(second, second_manifest) as writer:
                add_required_events(writer)
            result = trace.local_report(
                trace.TraceBundle(first),
                trace.TraceBundle(second),
                "self-consistency",
            )
            self.assertEqual(result["status"], "BRINGUP PASS")
            self.assertNotEqual(result["status"], "TARGET PASS")
            self.assertEqual(result["cross_runtime_status"], "INCOMPLETE")
            with mock.patch("sys.stdout", new_callable=io.StringIO):
                self.assertEqual(
                    trace.command_compare_local(Namespace(
                        mode="self-consistency",
                        left=first,
                        right=second,
                        left_signer_principal=self.signer_principal,
                        right_signer_principal=self.signer_principal,
                        execution_challenge=TEST_CHALLENGE,
                        left_run_id=TEST_RUN_IDS["llama.cpp"],
                        right_run_id="strix-llama-test-run-2",
                        report=None,
                    )),
                    0,
                )

    def test_local_base_regression_requires_attested_oracle_revision(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp) / "base"
            integrated = Path(temp) / "integrated"
            base_manifest = manifest("llama.cpp")
            base_manifest["revision"] = "b" * 40
            base_manifest["candidate"]["revision"] = "b" * 40
            for library in base_manifest["build"]["runtime_libraries"]:
                if library["component"] in {"llama-common", "ggml-base"}:
                    library["revision"] = "b" * 40
            base_manifest["build"]["runtime_libraries_post"] = copy.deepcopy(
                base_manifest["build"]["runtime_libraries"])
            receipt = {
                "format": "dsv41-runtime-receipt",
                "version": 1,
                "revision": "b" * 40,
                "profile": base_manifest["build"]["runtime_profile"]["name"],
                "components": sorted(
                    [
                        {
                            "component": library["component"],
                            "filename": library["filename"],
                            "sha256": library["sha256"],
                            "revision": library["revision"],
                        }
                        for library in base_manifest["build"]["runtime_libraries"]
                    ],
                    key=lambda item: item["component"],
                ),
            }
            receipt_sha256 = trace.sha256_bytes(trace.canonical_json(receipt).encode("ascii"))
            base_manifest["build"]["runtime_receipt_sha256"] = receipt_sha256
            base_manifest["candidate"]["runtime_libraries_sha256"] = trace.sha256_bytes(
                trace.canonical_json({
                    "pre": base_manifest["build"]["runtime_libraries"],
                    "post": base_manifest["build"]["runtime_libraries_post"],
                }).encode("ascii"))
            base_manifest["candidate"]["runtime_receipt_sha256"] = receipt_sha256
            base_verifier = self._verifier_for_runtime(
                "llama.cpp", manifest_record=base_manifest)
            _candidate_policy, candidate_policy_sha256 = trace.candidate_exporter_approval(
                TEST_CANDIDATE_EXPORTER_POLICY_ID,
                policies=base_verifier.candidate_exporter_policies,
            )
            base_manifest["candidate"]["exporter_approval_sha256"] = candidate_policy_sha256
            base_manifest["authorization"]["approvals"]["candidate_exporter"] = (
                trace.approval_binding(
                    "candidate_exporter",
                    TEST_CANDIDATE_EXPORTER_POLICY_ID,
                    candidate_policy_sha256,
                )
            )
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
            incomplete = manifest(context=4, decode_steps=2)
            with trace.TraceBundleWriter(root, incomplete) as writer:
                add_required_events(writer)
            with self.assertRaisesRegex(trace.TraceError, "decode step coverage"):
                trace.TraceBundle(root)


if __name__ == "__main__":
    unittest.main()
