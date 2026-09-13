#!/usr/bin/env python3

import json
import os
import time
from pathlib import Path

FORBIDDEN_ROOT = Path("/mnt/bigspace")
SOFT_MEMORY_LIMIT = 116 * 1024 * 1024 * 1024


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


def watchdog_audit(pid_file: Path) -> dict[str, object]:
    pid_file = resolved(pid_file)
    try:
        pid = int(pid_file.read_text(encoding="ascii").strip())
    except (OSError, ValueError) as error:
        raise PreflightError(f"watchdog pid file is invalid: {error}") from error
    if pid <= 1 or not Path(f"/proc/{pid}").exists():
        raise PreflightError(f"watchdog process {pid} is not running")
    try:
        command = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8", "replace").strip()
    except OSError as error:
        raise PreflightError(f"cannot inspect watchdog process {pid}: {error}") from error
    if not command:
        raise PreflightError(f"watchdog process {pid} has no command line")
    return {"pid": pid, "pid_file": str(pid_file), "command": command}


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
