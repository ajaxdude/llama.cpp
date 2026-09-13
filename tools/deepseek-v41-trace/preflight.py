#!/usr/bin/env python3

import json
import hashlib
import importlib.util
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Callable

FORBIDDEN_ROOT = Path("/mnt/bigspace")
SOFT_MEMORY_LIMIT = 116 * 1024 * 1024 * 1024
WATCHDOG_EMERGENCY_LIMIT = 118 * 1024 * 1024 * 1024
STRICT_MEMORY_LIMIT = 120 * 1024 * 1024 * 1024
MAX_WATCHDOG_HEARTBEAT_AGE = 30.0
WATCHDOG_LEASE_FORMAT = "strix-memory-watchdog-lease"
WATCHDOG_HEARTBEAT_FORMAT = "strix-memory-watchdog-heartbeat"
WATCHDOG_VERSION = 2
WATCHDOG_STARTUP_TIMEOUT_SECONDS = 5.0
WATCHDOG_LEASE_ENV = "STRIX_MEMORY_WATCHDOG_LEASE_PATH"
WATCHDOG_HEARTBEAT_ENV = "STRIX_MEMORY_WATCHDOG_HEARTBEAT_PATH"
WATCHDOG_AUDIT_ENV = "STRIX_MEMORY_WATCHDOG_AUDIT_PATH"
WATCHDOG_MAX_AGE_ENV = "STRIX_MEMORY_WATCHDOG_HEARTBEAT_MAX_AGE_SECONDS"
WATCHDOG_REVISION = "778db6f50eae04e6c232c69b9575bdbd0747962b"
WATCHDOG_SCRIPT_SHA256 = "d2781a25f978dd2bc14fc113079aa2dbf513aa157b44da9d0d51d750daa6c94f"
APPROVED_WATCHDOGS = {WATCHDOG_SCRIPT_SHA256: WATCHDOG_REVISION}
_WATCHDOG_GUARD = None


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


def safe_trace_path(root: Path, relative: Path | str) -> Path:
    root = root.expanduser().absolute()
    if root.is_symlink():
        raise PreflightError("trace output root must not be a symlink")
    root = require_nvme_path(root, "trace output")
    relative = Path(relative)
    if relative.is_absolute() or ".." in relative.parts:
        raise PreflightError(f"trace output path is outside the bundle: {relative}")
    candidate = root
    for part in relative.parts:
        candidate = candidate / part
        if candidate.is_symlink():
            raise PreflightError(f"trace output path must not use symlinks: {relative}")
    try:
        candidate.resolve().relative_to(root)
    except ValueError as error:
        raise PreflightError(f"trace output path is outside the bundle: {relative}") from error
    require_nvme_path(candidate, "trace output")
    return candidate


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


def proc_parent_pid(stat: str) -> int:
    command_end = stat.rfind(")")
    if command_end < 0:
        raise PreflightError("process stat is invalid")
    fields = stat[command_end + 2:].split()
    if len(fields) < 2:
        raise PreflightError("process stat is truncated")
    return int(fields[1])


def is_descendant(pid: int, ancestor_pid: int, procfs_root: Path) -> bool:
    seen = set()
    while pid > 1 and pid not in seen:
        if pid == ancestor_pid:
            return True
        seen.add(pid)
        try:
            pid = proc_parent_pid((procfs_root / str(pid) / "stat").read_text(encoding="ascii"))
        except (OSError, ValueError):
            return False
    return False


def read_heartbeat(
        path: Path,
        max_age_seconds: float,
        *,
        lease_id: str,
        watchdog_pid: int,
        watchdog_start_time_ticks: int,
        child_pid: int,
        child_pgid: int,
        now: int | None = None,
        monotonic_ns: Callable[[], int] = time.monotonic_ns) -> int:
    try:
        record = json.loads(path.read_text(encoding="ascii"))
        updated = datetime.fromisoformat(str(record["updated_at"]).replace("Z", "+00:00"))
        heartbeat = int(updated.timestamp())
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as error:
        raise PreflightError(f"watchdog heartbeat is invalid: {error}") from error
    if record.get("format") != WATCHDOG_HEARTBEAT_FORMAT or record.get("version") != WATCHDOG_VERSION:
        raise PreflightError("watchdog heartbeat format is invalid")
    if record.get("lease_id") != lease_id or record.get("state") != "active":
        raise PreflightError("watchdog heartbeat lease identity or state is invalid")
    if not isinstance(record.get("sequence"), int) or record["sequence"] < 0:
        raise PreflightError("watchdog heartbeat sequence is invalid")
    if not isinstance(record.get("updated_monotonic_ns"), int) or record["updated_monotonic_ns"] <= 0:
        raise PreflightError("watchdog heartbeat monotonic timestamp is invalid")
    if record.get("watchdog_pid") != watchdog_pid or (
            record.get("watchdog_start_time_ticks") != watchdog_start_time_ticks):
        raise PreflightError("watchdog heartbeat owner does not match the lease")
    if record.get("child_pid") != child_pid or record.get("child_process_group_id") != child_pgid:
        raise PreflightError("watchdog heartbeat child identity does not match the lease")
    age_ns = monotonic_ns() - record["updated_monotonic_ns"]
    if age_ns < 0 or age_ns > int(max_age_seconds * 1_000_000_000):
        raise PreflightError("watchdog heartbeat is stale")
    now = int(time.time()) if now is None else now
    if heartbeat <= 0 or heartbeat > now:
        raise PreflightError("watchdog heartbeat wall-clock timestamp is invalid")
    return heartbeat


def _read_json_with_retry(
        path: Path,
        *,
        timeout_seconds: float,
        monotonic: Callable[[], float],
        sleeper: Callable[[float], None]) -> dict[str, object]:
    deadline = monotonic() + timeout_seconds
    last_error: Exception | None = None
    while True:
        try:
            record = json.loads(path.read_text(encoding="ascii"))
            if not isinstance(record, dict):
                raise ValueError("record is not an object")
            return record
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
            last_error = error
            if monotonic() >= deadline:
                raise PreflightError(f"watchdog lease did not become ready: {last_error}") from last_error
            sleeper(0.05)


def _read_watchdog_events(path: Path) -> list[dict[str, object]]:
    try:
        lines = path.read_text(encoding="ascii").splitlines()
    except OSError as error:
        raise PreflightError(f"cannot read watchdog audit: {error}") from error
    events = []
    for line_number, line in enumerate(lines, start=1):
        try:
            event = json.loads(line)
        except json.JSONDecodeError as error:
            raise PreflightError(f"watchdog audit line {line_number} is invalid: {error}") from error
        if not isinstance(event, dict) or not isinstance(event.get("event"), str):
            raise PreflightError(f"watchdog audit line {line_number} is not an event")
        events.append(event)
    if not events:
        raise PreflightError("watchdog audit is empty")
    return events


def _read_watchdog_startup_events(
        path: Path,
        *,
        timeout_seconds: float,
        monotonic: Callable[[], float],
        sleeper: Callable[[float], None]) -> list[dict[str, object]]:
    deadline = monotonic() + timeout_seconds
    last_error: Exception | None = None
    while True:
        try:
            events = _read_watchdog_events(path)
            if any(event.get("event") == "preflight" for event in events) and (
                    any(event.get("event") == "child_started" for event in events)):
                return events
            last_error = PreflightError("watchdog audit lacks preflight or child_started evidence")
        except PreflightError as error:
            last_error = error
        if monotonic() >= deadline:
            assert last_error is not None
            raise PreflightError(f"watchdog audit did not become ready: {last_error}") from last_error
        sleeper(0.05)


def _load_watchdog_module(script: Path) -> object:
    spec = importlib.util.spec_from_file_location("dsv41_strix_memory_watchdog", script)
    if spec is None or spec.loader is None:
        raise PreflightError(f"cannot load canonical watchdog module: {script}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except (ImportError, OSError, RuntimeError, SyntaxError) as error:
        raise PreflightError(f"cannot load canonical watchdog module: {error}") from error
    return module


def _canonical_watchdog_audit(
        repo: Path,
        *,
        environment: dict[str, str] | os._Environ[str],
        current_pid: int | None,
        timeout_seconds: float,
        monotonic: Callable[[], float],
        sleeper: Callable[[float], None],
        watchdog_module: object | None) -> dict[str, object]:
    global _WATCHDOG_GUARD
    try:
        lease_path = require_nvme_path(Path(environment[WATCHDOG_LEASE_ENV]), "watchdog lease")
        heartbeat_path = require_nvme_path(Path(environment[WATCHDOG_HEARTBEAT_ENV]), "watchdog heartbeat")
        audit_path = require_nvme_path(Path(environment[WATCHDOG_AUDIT_ENV]), "watchdog audit")
        max_age_seconds = float(environment[WATCHDOG_MAX_AGE_ENV])
    except (KeyError, ValueError) as error:
        raise PreflightError(f"canonical watchdog environment is incomplete: {error}") from error
    if max_age_seconds <= 0 or max_age_seconds > MAX_WATCHDOG_HEARTBEAT_AGE:
        raise PreflightError(f"watchdog heartbeat age must be within 1..{MAX_WATCHDOG_HEARTBEAT_AGE} seconds")
    expected_script = resolved(repo / "scripts" / "strix_memory_watchdog.py")
    if not expected_script.is_file():
        raise PreflightError(f"canonical watchdog script is missing: {expected_script}")
    script_sha256 = sha256_bytes(expected_script.read_bytes())
    watchdog_revision = APPROVED_WATCHDOGS.get(script_sha256)
    if watchdog_revision is None:
        raise PreflightError(
            "no independently reviewed watchdog revision is approved for correctness execution")
    module = watchdog_module or _load_watchdog_module(expected_script)
    validator = getattr(module, "validate_active_lease", None)
    validation_error = getattr(module, "LeaseValidationError", RuntimeError)
    guard = getattr(module, "start_process_group_lease_guard", None)
    if not callable(validator) or not isinstance(validation_error, type):
        raise PreflightError("canonical watchdog module lacks the lease validation API")
    process_id = os.getpid() if current_pid is None else current_pid
    try:
        lease = validator(
            lease_path,
            expected_script_path=expected_script,
            expected_soft_bytes=SOFT_MEMORY_LIMIT,
            expected_emergency_bytes=WATCHDOG_EMERGENCY_LIMIT,
            expected_procfs_root=Path("/proc"),
            expected_heartbeat_path=heartbeat_path,
            expected_audit_path=audit_path,
            expected_max_heartbeat_age_seconds=max_age_seconds,
            current_process_id=process_id,
            process_procfs_root=Path("/proc"),
        )
        if lease.get("child_pid") == process_id and _WATCHDOG_GUARD is None:
            if not callable(guard):
                raise PreflightError("canonical watchdog module lacks the process-group lease guard")
            _WATCHDOG_GUARD = guard(
                expected_script,
                startup_timeout_seconds=timeout_seconds,
                expected_procfs_root=Path("/proc"),
                process_procfs_root=Path("/proc"),
            )
    except validation_error as error:
        raise PreflightError(f"canonical watchdog lease is invalid: {error}") from error
    if not isinstance(lease, dict):
        raise PreflightError("canonical watchdog lease validator returned invalid data")
    if lease.get("grace_seconds") != 30.0 or lease.get("sample_interval_seconds") != 1.0:
        raise PreflightError("canonical watchdog timing policy is invalid")
    events = _read_watchdog_startup_events(
        audit_path,
        timeout_seconds=timeout_seconds,
        monotonic=monotonic,
        sleeper=sleeper,
    )
    try:
        heartbeat_record = json.loads(heartbeat_path.read_text(encoding="ascii"))
        updated = datetime.fromisoformat(str(heartbeat_record["updated_at"]).replace("Z", "+00:00"))
        heartbeat_unix = int(updated.timestamp())
    except (OSError, UnicodeError, ValueError, TypeError, KeyError, json.JSONDecodeError) as error:
        raise PreflightError(f"canonical watchdog heartbeat is invalid: {error}") from error
    try:
        audit_sha256 = sha256_bytes(audit_path.read_bytes())
    except OSError as error:
        raise PreflightError(f"cannot hash canonical watchdog audit: {error}") from error
    watchdog_command = lease.get("watchdog_command")
    if not isinstance(watchdog_command, str) or not watchdog_command:
        try:
            command_bytes = (
                Path("/proc") / str(lease["watchdog_pid"]) / "cmdline").read_bytes()
        except (OSError, KeyError) as error:
            raise PreflightError(f"cannot read canonical watchdog command: {error}") from error
        watchdog_command = command_bytes.replace(b"\0", b" ").decode("utf-8", "replace").strip()
        if not watchdog_command:
            raise PreflightError("canonical watchdog command is empty")
    result = dict(lease)
    result.update({
        "watchdog_revision": watchdog_revision,
        "lease_path": str(lease_path),
        "watchdog_command": watchdog_command,
        "heartbeat_unix": heartbeat_unix,
        "audit_path": str(audit_path),
        "audit_sha256": audit_sha256,
        "audit_event_count": len(events),
    })
    return result


def watchdog_audit(
        repo: Path,
        *,
        environment: dict[str, str] | os._Environ[str] = os.environ,
        procfs_root: Path = Path("/proc"),
        current_pid: int | None = None,
        current_pgid: int | None = None,
        getpgid: Callable[[int], int] = os.getpgid,
        now: int | None = None,
        monotonic_ns: Callable[[], int] = time.monotonic_ns,
        timeout_seconds: float = WATCHDOG_STARTUP_TIMEOUT_SECONDS,
        monotonic: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
        watchdog_module: object | None = None) -> dict[str, object]:
    if watchdog_module is not None or (
            procfs_root == Path("/proc") and current_pid is None and current_pgid is None):
        return _canonical_watchdog_audit(
            repo,
            environment=environment,
            current_pid=current_pid,
            timeout_seconds=timeout_seconds,
            monotonic=monotonic,
            sleeper=sleeper,
            watchdog_module=watchdog_module,
        )
    try:
        pid_file = Path(environment[WATCHDOG_LEASE_ENV])
        environment_heartbeat = require_nvme_path(
            Path(environment[WATCHDOG_HEARTBEAT_ENV]), "watchdog heartbeat")
        environment_audit = require_nvme_path(
            Path(environment[WATCHDOG_AUDIT_ENV]), "watchdog audit")
        environment_max_age = float(environment[WATCHDOG_MAX_AGE_ENV])
    except (KeyError, ValueError) as error:
        raise PreflightError(f"canonical watchdog environment is incomplete: {error}") from error
    pid_file = require_nvme_path(pid_file, "watchdog lease")
    repo = resolved(repo)
    expected_script = resolved(repo / "scripts" / "strix_memory_watchdog.py")
    if not expected_script.is_file():
        raise PreflightError(f"canonical watchdog script is missing: {expected_script}")
    expected_script_sha256 = sha256_bytes(expected_script.read_bytes())
    lease = _read_json_with_retry(
        pid_file,
        timeout_seconds=timeout_seconds,
        monotonic=monotonic,
        sleeper=sleeper,
    )
    try:
        pid = int(lease["watchdog_pid"])
        expected_start = int(lease["watchdog_start_time_ticks"])
        expected_command_sha256 = str(lease["watchdog_command_sha256"])
        child_pid = int(lease["child_pid"])
        child_pgid = int(lease["child_process_group_id"])
        script_path = resolved(Path(str(lease["watchdog_script_path"])))
        script_sha256 = str(lease["watchdog_script_sha256"])
        lease_id = str(lease["lease_id"])
        soft_bytes = int(lease["soft_bytes"])
        emergency_bytes = int(lease["emergency_bytes"])
        strict_bytes = int(lease["strict_ceiling_bytes"])
        procfs_path = str(lease["procfs_root"])
        heartbeat_path = require_nvme_path(Path(lease["heartbeat_path"]), "watchdog heartbeat")
        audit_path = require_nvme_path(Path(lease["audit_path"]), "watchdog audit")
        max_age_seconds = float(lease.get("max_heartbeat_age_seconds", MAX_WATCHDOG_HEARTBEAT_AGE))
    except (OSError, ValueError, TypeError, KeyError) as error:
        raise PreflightError(f"watchdog lease is invalid: {error}") from error
    if lease.get("format") != WATCHDOG_LEASE_FORMAT or lease.get("version") != WATCHDOG_VERSION:
        raise PreflightError("watchdog lease format is invalid")
    if lease.get("state") != "active" or re.fullmatch(r"[0-9a-f]{32,64}", lease_id) is None:
        raise PreflightError("watchdog lease identity or state is invalid")
    if script_path != expected_script or script_sha256 != expected_script_sha256:
        raise PreflightError("watchdog script identity does not match the candidate repository")
    if soft_bytes != SOFT_MEMORY_LIMIT or emergency_bytes != WATCHDOG_EMERGENCY_LIMIT or (
            strict_bytes != STRICT_MEMORY_LIMIT):
        raise PreflightError("watchdog memory thresholds are invalid")
    if procfs_path != "/proc":
        raise PreflightError("watchdog procfs root must be /proc")
    if heartbeat_path != environment_heartbeat or audit_path != environment_audit or (
            max_age_seconds != environment_max_age):
        raise PreflightError("watchdog lease paths or heartbeat age do not match the inherited environment")
    if re.fullmatch(r"[0-9a-f]{64}", expected_command_sha256) is None:
        raise PreflightError("watchdog lease command SHA-256 is invalid")
    if max_age_seconds <= 0 or max_age_seconds > MAX_WATCHDOG_HEARTBEAT_AGE:
        raise PreflightError(f"watchdog heartbeat age must be within 1..{MAX_WATCHDOG_HEARTBEAT_AGE} seconds")
    if pid <= 1 or child_pid <= 1 or child_pgid <= 1:
        raise PreflightError("watchdog or monitored child identity is invalid")
    if current_pgid is None:
        current_pgid = os.getpgrp()
    if current_pid is None:
        current_pid = os.getpid()
    if current_pgid != child_pgid:
        raise PreflightError("current process is outside the watchdog-monitored process group")
    if not is_descendant(current_pid, child_pid, procfs_root):
        raise PreflightError("current process is not a descendant of the watchdog-monitored child")
    try:
        child_parent = proc_parent_pid(
            (procfs_root / str(child_pid) / "stat").read_text(encoding="ascii"))
        if child_parent != pid or getpgid(child_pid) != child_pgid or child_pgid != child_pid:
            raise PreflightError("watchdog child process group does not match the lease")
    except (OSError, ValueError) as error:
        raise PreflightError(f"cannot inspect watchdog child process group: {error}") from error
    if not (procfs_root / str(pid)).exists():
        raise PreflightError(f"watchdog process {pid} is not running")
    try:
        command_bytes = (procfs_root / str(pid) / "cmdline").read_bytes()
        start_time_ticks = proc_start_time_ticks(
            (procfs_root / str(pid) / "stat").read_text(encoding="ascii"))
    except (OSError, ValueError) as error:
        raise PreflightError(f"cannot inspect watchdog process {pid}: {error}") from error
    command_sha256 = sha256_bytes(command_bytes)
    if start_time_ticks != expected_start or command_sha256 != expected_command_sha256:
        raise PreflightError("watchdog process identity does not match its lease")
    command_parts = [part.decode("utf-8", "replace") for part in command_bytes.split(b"\0") if part]
    try:
        watchdog_cwd = (procfs_root / str(pid) / "cwd").resolve()
    except OSError as error:
        raise PreflightError(f"cannot inspect watchdog process working directory: {error}") from error
    script_named = any(
        resolved(Path(argument) if Path(argument).is_absolute() else watchdog_cwd / argument) == expected_script
        for argument in command_parts
    )
    if not script_named:
        raise PreflightError("watchdog command does not execute the candidate repository script")
    child_command = lease.get("command")
    if not isinstance(child_command, list) or not child_command or (
            not all(isinstance(argument, str) for argument in child_command)):
        raise PreflightError("watchdog child command is invalid")
    child_command_sha256 = sha256_bytes(
        json.dumps(child_command, ensure_ascii=True, separators=(",", ":")).encode("utf-8"))
    if child_command_sha256 != lease.get("child_command_sha256"):
        raise PreflightError("watchdog child command does not match the lease")
    heartbeat = read_heartbeat(
        heartbeat_path,
        max_age_seconds,
        lease_id=lease_id,
        watchdog_pid=pid,
        watchdog_start_time_ticks=start_time_ticks,
        child_pid=child_pid,
        child_pgid=child_pgid,
        now=now,
        monotonic_ns=monotonic_ns,
    )
    command = command_bytes.replace(b"\0", b" ").decode("utf-8", "replace").strip()
    if not command:
        raise PreflightError(f"watchdog process {pid} has no command line")
    events = _read_watchdog_startup_events(
        audit_path,
        timeout_seconds=timeout_seconds,
        monotonic=monotonic,
        sleeper=sleeper,
    )
    preflight = next((event for event in events if event.get("event") == "preflight"), None)
    child_started = next((event for event in events if event.get("event") == "child_started"), None)
    if preflight is None or child_started is None:
        raise PreflightError("watchdog audit lacks preflight or child_started evidence")
    if preflight.get("soft_bytes") != SOFT_MEMORY_LIMIT or (
            preflight.get("emergency_bytes") != WATCHDOG_EMERGENCY_LIMIT) or (
            preflight.get("strict_ceiling_bytes") != STRICT_MEMORY_LIMIT):
        raise PreflightError("watchdog audit thresholds are invalid")
    if preflight.get("swap_entries") != 0:
        raise PreflightError("watchdog audit does not report zero swap")
    if child_started.get("child_pid") != child_pid or (
            child_started.get("process_group_id") != child_pgid):
        raise PreflightError("watchdog audit child identity does not match the lease")
    if child_started.get("command") != lease.get("command"):
        raise PreflightError("watchdog audit child command does not match the lease")
    return {
        "format": WATCHDOG_LEASE_FORMAT,
        "version": WATCHDOG_VERSION,
        "lease_id": lease_id,
        "lease_path": str(pid_file),
        "watchdog_pid": pid,
        "watchdog_start_time_ticks": start_time_ticks,
        "watchdog_command": command,
        "watchdog_command_sha256": command_sha256,
        "watchdog_script_path": str(script_path),
        "watchdog_script_sha256": script_sha256,
        "soft_bytes": soft_bytes,
        "emergency_bytes": emergency_bytes,
        "strict_ceiling_bytes": strict_bytes,
        "procfs_root": procfs_path,
        "child_pid": child_pid,
        "child_process_group_id": child_pgid,
        "command": child_command,
        "child_command_sha256": child_command_sha256,
        "heartbeat_path": str(heartbeat_path),
        "heartbeat_unix": heartbeat,
        "max_heartbeat_age_seconds": max_age_seconds,
        "audit_path": str(audit_path),
        "audit_sha256": sha256_bytes(audit_path.read_bytes()),
        "audit_event_count": len(events),
    }


def process_ancestry(pid: int, procfs_root: Path) -> set[int]:
    ancestors = {pid}
    while pid > 1:
        try:
            parent = proc_parent_pid((procfs_root / str(pid) / "stat").read_text(encoding="ascii"))
        except (OSError, UnicodeError, ValueError):
            break
        if parent <= 1 or parent in ancestors:
            break
        ancestors.add(parent)
        pid = parent
    return ancestors


def matching_workloads(
        patterns: list[str],
        *,
        procfs_root: Path = Path("/proc"),
        current_pid: int | None = None) -> list[dict[str, object]]:
    matches = []
    excluded = process_ancestry(os.getpid() if current_pid is None else current_pid, procfs_root)
    lowered = [pattern.lower() for pattern in patterns if pattern]
    for entry in procfs_root.iterdir():
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
        repo: Path,
        busy_patterns: list[str],
) -> dict[str, object]:
    model = require_nvme_path(model, "model")
    prompt = require_nvme_path(prompt, "prompt")
    output = require_nvme_path(output, "trace output")
    if not model.is_file():
        raise PreflightError(f"model is not a file: {model}")
    if not prompt.is_file():
        raise PreflightError(f"prompt is not a file: {prompt}")
    if os.environ.get("HIP_LAUNCH_BLOCKING") != "1":
        raise PreflightError("HIP_LAUNCH_BLOCKING=1 is required for gfx1151 correctness runs")
    swap = swap_audit()
    if swap["enabled"]:
        raise PreflightError("swap is enabled; model execution is blocked")
    watchdog = watchdog_audit(repo)
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
        "environment": {"HIP_LAUNCH_BLOCKING": "1"},
    }


def write_audits(root: Path, audit: dict[str, object]) -> dict[str, str]:
    root = resolved(root)
    if root.exists() and any(root.iterdir()):
        raise PreflightError(f"audit directory is not empty: {root}")
    root.mkdir(parents=True, exist_ok=True)
    result = {}
    for key in ("memory", "swap", "watchdog"):
        path = root / f"{key}.json"
        data = audit[key]
        if key == "watchdog":
            data = dict(data)
            live_audit_path = resolved(Path(str(data["audit_path"])))
            try:
                audit_bytes = live_audit_path.read_bytes()
                audit_text = audit_bytes.decode("ascii")
            except (OSError, UnicodeError) as error:
                raise PreflightError(f"cannot snapshot watchdog audit: {error}") from error
            events = []
            for line_number, line in enumerate(audit_text.splitlines(), start=1):
                try:
                    event = json.loads(line)
                except json.JSONDecodeError as error:
                    raise PreflightError(
                        f"watchdog audit line {line_number} is invalid while snapshotting: {error}") from error
                if not isinstance(event, dict) or not isinstance(event.get("event"), str):
                    raise PreflightError(
                        f"watchdog audit line {line_number} is not an event while snapshotting")
                events.append(event)
            if not events:
                raise PreflightError("watchdog audit snapshot is empty")
            snapshot = root / "watchdog-events.jsonl"
            snapshot.write_bytes(audit_bytes)
            data["audit_live_path"] = str(live_audit_path)
            data["audit_path"] = str(snapshot)
            data["audit_sha256"] = sha256_bytes(audit_bytes)
            data["audit_event_count"] = len(events)
        value = {
            "created_unix": audit["created_unix"],
            "kind": key,
            "data": data,
            "environment": audit["environment"],
        }
        path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n", encoding="ascii")
        result[key] = str(path)
    summary = root / "preflight.json"
    summary.write_text(json.dumps(audit, sort_keys=True, separators=(",", ":")) + "\n", encoding="ascii")
    result["preflight"] = str(summary)
    return result


def seal_audits(audits: dict[str, str]) -> dict[str, str]:
    digests = {}
    paths = []
    try:
        for kind in ("memory", "swap", "watchdog"):
            path = resolved(Path(audits[kind]))
            paths.append(path)
            data = path.read_bytes()
            digests[kind] = sha256_bytes(data)
            if kind == "watchdog":
                record = json.loads(data.decode("ascii"))
                jsonl_path = resolved(Path(record["data"]["audit_path"]))
                paths.append(jsonl_path)
                digests["watchdog_jsonl"] = sha256_bytes(jsonl_path.read_bytes())
        for path in paths:
            path.chmod(0o444)
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError) as error:
        raise PreflightError(f"cannot seal audit evidence: {error}") from error
    return digests


def verify_sealed_audits(audits: dict[str, str], digests: dict[str, str]) -> None:
    current = seal_audits(audits)
    if current != digests:
        raise PreflightError("preflight audit evidence changed during runtime execution")


def embed_audits(trace_root: Path, phase: str, audits: dict[str, str]) -> dict[str, dict[str, object]]:
    trace_root = safe_trace_path(trace_root, ".")
    embedded_root = safe_trace_path(trace_root, Path("audits") / phase)
    embedded_root.mkdir(parents=True, exist_ok=True)
    result = {}
    for kind in ("memory", "swap", "watchdog"):
        source = resolved(Path(audits[kind]))
        data = source.read_bytes()
        try:
            record = json.loads(data.decode("ascii"))
        except (UnicodeError, json.JSONDecodeError) as error:
            raise PreflightError(f"cannot embed {kind} audit: {error}") from error
        if kind == "watchdog":
            try:
                jsonl_source = resolved(Path(record["data"].pop("audit_path")))
                jsonl_data = jsonl_source.read_bytes()
            except (OSError, TypeError, KeyError) as error:
                raise PreflightError(f"cannot embed watchdog JSONL audit: {error}") from error
            jsonl_digest = sha256_bytes(jsonl_data)
            if record["data"].get("audit_sha256") != jsonl_digest:
                raise PreflightError("watchdog JSONL audit SHA-256 changed before embedding")
            jsonl_destination = safe_trace_path(
                trace_root, Path("audits") / phase / f"{jsonl_digest}.jsonl")
            if jsonl_destination.exists() and jsonl_destination.read_bytes() != jsonl_data:
                raise PreflightError(f"content-addressed watchdog audit collision: {jsonl_destination}")
            if not jsonl_destination.exists():
                jsonl_destination.write_bytes(jsonl_data)
            record["data"]["audit"] = {
                "path": f"audits/{phase}/{jsonl_digest}.jsonl",
                "sha256": jsonl_digest,
                "event_count": record["data"].pop("audit_event_count"),
            }
            data = (json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n").encode("ascii")
        digest = sha256_bytes(data)
        destination = safe_trace_path(trace_root, Path("audits") / phase / f"{digest}.json")
        if destination.exists() and destination.read_bytes() != data:
            raise PreflightError(f"content-addressed audit collision: {destination}")
        if not destination.exists():
            destination.write_bytes(data)
        try:
            created = int(record["created_unix"])
        except (ValueError, TypeError, KeyError) as error:
            raise PreflightError(f"cannot embed {kind} audit: {error}") from error
        result[kind] = {
            "path": f"audits/{phase}/{digest}.json",
            "sha256": digest,
            "created_unix": created,
        }
    return result


def bind_embedded_audits(trace_root: Path, audit_sets: dict[str, dict[str, str]]) -> None:
    trace_root = safe_trace_path(trace_root, ".")
    manifest_path = safe_trace_path(trace_root, "manifest.json")
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
    trace_root = safe_trace_path(trace_root, ".")
    manifest_path = safe_trace_path(trace_root, "manifest.json")
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
    provenance_root = safe_trace_path(trace_root, "provenance")
    provenance_root.mkdir(parents=True, exist_ok=True)
    destination = safe_trace_path(trace_root, Path("provenance") / f"{digest}.json")
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
