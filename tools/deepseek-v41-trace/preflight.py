#!/usr/bin/env python3

import json
import hashlib
import os
import re
import shutil
import time
from pathlib import Path

FORBIDDEN_ROOT = Path("/mnt/bigspace")
SOFT_MEMORY_LIMIT = 116 * 1024 * 1024 * 1024
MAX_WATCHDOG_HEARTBEAT_AGE = 30


class PreflightError(RuntimeError):
    pass


def resolved(path: Path) -> Path:
    return path.expanduser().resolve()


def require_nvme_path(path: Path, label: str) -> Path:
    path = resolved(path)
    try:
        path.relative_to(FORBIDDEN_ROOT)
    except ValueError:
        return path
    raise PreflightError(f"{label} must not use rotational storage: {path}")


def read_proc_lines(path: Path) -> list[str]:
    try:
        return path.read_text(encoding="ascii").splitlines()
    except OSError as error:
        raise PreflightError(f"cannot read {path}: {error}") from error


def swap_audit() -> dict[str, object]:
    lines = read_proc_lines(Path("/proc/swaps"))
    entries = []
    for line in lines[1:]:
        fields = line.split()
        if len(fields) >= 5:
            entries.append({
                "path": fields[0],
                "type": fields[1],
                "size_kib": int(fields[2]),
                "used_kib": int(fields[3]),
                "priority": int(fields[4]),
            })
    return {"enabled": bool(entries), "entries": entries}


def memory_audit() -> dict[str, int]:
    values: dict[str, int] = {}
    for line in read_proc_lines(Path("/proc/meminfo")):
        key, value = line.split(":", 1)
        fields = value.split()
        if fields:
            values[key] = int(fields[0]) * 1024
    required = ("MemTotal", "MemAvailable")
    if any(key not in values for key in required):
        raise PreflightError("/proc/meminfo lacks MemTotal or MemAvailable")
    result = {
        "mem_total_bytes": values["MemTotal"],
        "mem_available_bytes": values["MemAvailable"],
        "mem_used_bytes": values["MemTotal"] - values["MemAvailable"],
    }
    if result["mem_used_bytes"] >= SOFT_MEMORY_LIMIT:
        raise PreflightError(
            f"host memory use is at or above the 116 GiB soft limit: {result['mem_used_bytes']}")
    return result


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def proc_start_time_ticks(stat: str) -> int:
    command_end = stat.rfind(")")
    if command_end < 0:
        raise PreflightError("watchdog process stat is invalid")
    fields = stat[command_end + 2:].split()
    if len(fields) < 20:
        raise PreflightError("watchdog process stat is truncated")
    return int(fields[19])


def read_heartbeat(path: Path, max_age_seconds: int) -> int:
    try:
        heartbeat = int(path.read_text(encoding="ascii").strip())
    except (OSError, ValueError) as error:
        raise PreflightError(f"watchdog heartbeat is invalid: {error}") from error
    now = int(time.time())
    if heartbeat <= 0 or heartbeat > now or now - heartbeat > max_age_seconds:
        raise PreflightError("watchdog heartbeat is stale")
    return heartbeat


def watchdog_audit(pid_file: Path) -> dict[str, object]:
    pid_file = require_nvme_path(pid_file, "watchdog lease")
    try:
        lease = json.loads(pid_file.read_text(encoding="ascii"))
        pid = int(lease["pid"])
        expected_start = int(lease["start_time_ticks"])
        expected_command_sha256 = str(lease["command_sha256"])
        heartbeat_path = require_nvme_path(Path(lease["heartbeat_path"]), "watchdog heartbeat")
        max_age_seconds = int(lease.get("max_heartbeat_age_seconds", MAX_WATCHDOG_HEARTBEAT_AGE))
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as error:
        raise PreflightError(f"watchdog lease is invalid: {error}") from error
    if re.fullmatch(r"[0-9a-f]{64}", expected_command_sha256) is None:
        raise PreflightError("watchdog lease command SHA-256 is invalid")
    if max_age_seconds <= 0 or max_age_seconds > MAX_WATCHDOG_HEARTBEAT_AGE:
        raise PreflightError(f"watchdog heartbeat age must be within 1..{MAX_WATCHDOG_HEARTBEAT_AGE} seconds")
    if pid <= 1 or not Path(f"/proc/{pid}").exists():
        raise PreflightError(f"watchdog process {pid} is not running")
    try:
        command_bytes = Path(f"/proc/{pid}/cmdline").read_bytes()
        start_time_ticks = proc_start_time_ticks(Path(f"/proc/{pid}/stat").read_text(encoding="ascii"))
    except (OSError, ValueError) as error:
        raise PreflightError(f"cannot inspect watchdog process {pid}: {error}") from error
    command_sha256 = sha256_bytes(command_bytes)
    if start_time_ticks != expected_start or command_sha256 != expected_command_sha256:
        raise PreflightError("watchdog process identity does not match its lease")
    heartbeat = read_heartbeat(heartbeat_path, max_age_seconds)
    command = command_bytes.replace(b"\0", b" ").decode("utf-8", "replace").strip()
    if not command:
        raise PreflightError(f"watchdog process {pid} has no command line")
    return {
        "pid": pid,
        "pid_file": str(pid_file),
        "start_time_ticks": start_time_ticks,
        "command": command,
        "command_sha256": command_sha256,
        "heartbeat_path": str(heartbeat_path),
        "heartbeat_unix": heartbeat,
        "max_heartbeat_age_seconds": max_age_seconds,
    }


def matching_workloads(patterns: list[str]) -> list[dict[str, object]]:
    matches = []
    excluded = {os.getpid(), os.getppid()}
    lowered = [pattern.lower() for pattern in patterns if pattern]
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        if pid in excluded:
            continue
        try:
            command = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8", "replace").strip()
        except OSError:
            continue
        command_lower = command.lower()
        if any(pattern in command_lower for pattern in lowered):
            matches.append({"pid": pid, "command": command})
    return matches


def run_preflight(
        *,
        model: Path,
        prompt: Path,
        output: Path,
        watchdog_pid_file: Path,
        busy_patterns: list[str],
) -> dict[str, object]:
    model = require_nvme_path(model, "model")
    prompt = require_nvme_path(prompt, "prompt")
    output = require_nvme_path(output, "trace output")
    if not model.is_file():
        raise PreflightError(f"model is not a file: {model}")
    if not prompt.is_file():
        raise PreflightError(f"prompt is not a file: {prompt}")
    swap = swap_audit()
    if swap["enabled"]:
        raise PreflightError("swap is enabled; model execution is blocked")
    watchdog = watchdog_audit(watchdog_pid_file)
    workloads = matching_workloads(busy_patterns)
    if workloads:
        raise PreflightError("active model workload detected: " + json.dumps(workloads, ensure_ascii=True))
    return {
        "created_unix": int(time.time()),
        "model": str(model),
        "prompt": str(prompt),
        "output": str(output),
        "memory": memory_audit(),
        "swap": swap,
        "watchdog": watchdog,
        "active_workloads": [],
    }


def write_audits(root: Path, audit: dict[str, object]) -> dict[str, str]:
    root = resolved(root)
    if root.exists() and any(root.iterdir()):
        raise PreflightError(f"audit directory is not empty: {root}")
    root.mkdir(parents=True, exist_ok=True)
    result = {}
    for key in ("memory", "swap", "watchdog"):
        path = root / f"{key}.json"
        value = {
            "created_unix": audit["created_unix"],
            "kind": key,
            "data": audit[key],
        }
        path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n", encoding="ascii")
        result[key] = str(path)
    summary = root / "preflight.json"
    summary.write_text(json.dumps(audit, sort_keys=True, separators=(",", ":")) + "\n", encoding="ascii")
    result["preflight"] = str(summary)
    return result


def embed_audits(trace_root: Path, phase: str, audits: dict[str, str]) -> dict[str, dict[str, object]]:
    trace_root = resolved(trace_root)
    embedded_root = trace_root / "audits" / phase
    embedded_root.mkdir(parents=True, exist_ok=True)
    result = {}
    for kind in ("memory", "swap", "watchdog"):
        source = resolved(Path(audits[kind]))
        data = source.read_bytes()
        digest = sha256_bytes(data)
        destination = embedded_root / f"{digest}.json"
        if destination.exists() and destination.read_bytes() != data:
            raise PreflightError(f"content-addressed audit collision: {destination}")
        if not destination.exists():
            shutil.copyfile(source, destination)
        try:
            record = json.loads(data.decode("ascii"))
            created = int(record["created_unix"])
        except (UnicodeError, ValueError, TypeError, KeyError, json.JSONDecodeError) as error:
            raise PreflightError(f"cannot embed {kind} audit: {error}") from error
        result[kind] = {
            "path": f"audits/{phase}/{digest}.json",
            "sha256": digest,
            "created_unix": created,
        }
    return result


def bind_embedded_audits(trace_root: Path, audit_sets: dict[str, dict[str, str]]) -> None:
    trace_root = resolved(trace_root)
    manifest_path = trace_root / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise PreflightError(f"cannot bind trace audits: {error}") from error
    if set(audit_sets) != {"pre", "post"}:
        raise PreflightError("trace requires pre and post audit sets")
    manifest["audits"] = {
        phase: embed_audits(trace_root, phase, audits)
        for phase, audits in audit_sets.items()
    }
    temp = manifest_path.with_suffix(".tmp")
    temp.write_text(json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n", encoding="ascii")
    os.replace(temp, manifest_path)


def validate_prompt_provenance(
    path: Path,
    *,
    prompt: Path,
    corpus_name: str,
    corpus_sha256: str,
    model_sha256: str,
    target_tokens: int,
) -> dict[str, object]:
    path = require_nvme_path(path, "prompt provenance")
    try:
        data = path.read_bytes()
        record = json.loads(data.decode("ascii"))
        prompt_path = resolved(prompt)
        prompt_bytes = prompt_path.read_bytes()
        prompt_size = prompt_path.stat().st_size
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise PreflightError(f"prompt provenance is invalid: {error}") from error
    if not isinstance(record, dict):
        raise PreflightError("prompt provenance must be a JSON object")
    expected = {
        "format": "dsv41-prompt-provenance",
        "version": 1,
        "corpus_name": corpus_name,
        "corpus_sha256": corpus_sha256,
        "model_sha256": model_sha256,
        "prompt_sha256": sha256_bytes(prompt_bytes),
        "prompt_byte_count": prompt_size,
        "target_tokens": target_tokens,
        "actual_tokens": target_tokens,
    }
    for key, value in expected.items():
        if record.get(key) != value:
            raise PreflightError(f"prompt provenance {key} mismatch")
    builder_sha256 = record.get("builder_sha256", "")
    if not isinstance(builder_sha256, str) or re.fullmatch(r"[0-9a-f]{64}", builder_sha256) is None:
        raise PreflightError("prompt provenance builder SHA-256 is invalid")
    return {"path": str(path), "bytes": data, "record": record}


def bind_prompt_provenance(trace_root: Path, provenance: dict[str, object]) -> None:
    trace_root = resolved(trace_root)
    manifest_path = trace_root / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise PreflightError(f"cannot bind prompt provenance: {error}") from error
    prompt = manifest.get("prompt")
    if not isinstance(prompt, dict):
        raise PreflightError("trace manifest prompt is invalid")
    record = provenance["record"]
    data = provenance["bytes"]
    if not isinstance(record, dict) or not isinstance(data, bytes):
        raise PreflightError("validated prompt provenance is invalid")
    digest = sha256_bytes(data)
    provenance_root = trace_root / "provenance"
    provenance_root.mkdir(parents=True, exist_ok=True)
    destination = provenance_root / f"{digest}.json"
    if destination.exists() and destination.read_bytes() != data:
        raise PreflightError(f"content-addressed provenance collision: {destination}")
    if not destination.exists():
        destination.write_bytes(data)
    prompt["corpus_name"] = record["corpus_name"]
    prompt["corpus_sha256"] = record["corpus_sha256"]
    prompt["target_tokens"] = record["target_tokens"]
    prompt["provenance"] = {
        "path": f"provenance/{digest}.json",
        "sha256": digest,
    }
    temp = manifest_path.with_suffix(".tmp")
    temp.write_text(json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n", encoding="ascii")
    os.replace(temp, manifest_path)
