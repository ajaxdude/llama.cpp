#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import os
import re
import signal
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import IO, Protocol


GIB = 1024**3
STRICT_CEILING_BYTES = 120 * GIB
DEFAULT_SOFT_BYTES = 116 * GIB
DEFAULT_EMERGENCY_BYTES = 118 * GIB
DEFAULT_GRACE_SECONDS = 30.0
DEFAULT_SAMPLE_INTERVAL_SECONDS = 1.0

EXIT_PROCFS_ERROR = 2
EXIT_SWAP_ACTIVE = 3
EXIT_SOFT_LIMIT = 4
EXIT_EMERGENCY_LIMIT = 5
EXIT_GRACE_TIMEOUT = 6
EXIT_SIGNAL_ERROR = 7
EXIT_INTERNAL_ERROR = 70
EXIT_LAUNCH_ERROR = 127

MEMINFO_VALUE_RE = re.compile(r"([0-9]+) kB")
SWAPS_HEADER = ["Filename", "Type", "Size", "Used", "Priority"]
PARENT_SIGNALS = (signal.SIGINT, signal.SIGTERM)


class ProcfsError(RuntimeError):
    pass


class ProcessGroupError(RuntimeError):
    pass


class ParentSignal(RuntimeError):
    def __init__(self, signal_number: int):
        self.signal_number = signal_number
        super().__init__(signal.Signals(signal_number).name)


class ProcessHandle(Protocol):
    pid: int

    def poll(self) -> int | None:
        ...

    def wait(self, timeout: float | None = None) -> int:
        ...


@dataclass(frozen=True)
class HostSnapshot:
    total_bytes: int
    available_bytes: int
    active_swaps: tuple[str, ...]

    @property
    def used_bytes(self) -> int:
        return self.total_bytes - self.available_bytes


@dataclass
class RuntimeState:
    snapshot: HostSnapshot
    peak_used_bytes: int


@dataclass(frozen=True)
class WatchdogConfig:
    command: tuple[str, ...]
    procfs_root: Path = Path("/proc")
    soft_bytes: int = DEFAULT_SOFT_BYTES
    emergency_bytes: int = DEFAULT_EMERGENCY_BYTES
    grace_seconds: float = DEFAULT_GRACE_SECONDS
    sample_interval_seconds: float = DEFAULT_SAMPLE_INTERVAL_SECONDS

    def validate(self) -> None:
        if not self.command:
            raise ValueError("a command is required after --")
        if self.soft_bytes <= 0:
            raise ValueError("soft threshold must be greater than zero")
        if self.emergency_bytes <= self.soft_bytes:
            raise ValueError("emergency threshold must be greater than soft threshold")
        if self.emergency_bytes >= STRICT_CEILING_BYTES:
            raise ValueError("emergency threshold must be below 120 GiB")
        if not math.isfinite(self.grace_seconds) or self.grace_seconds <= 0:
            raise ValueError("grace period must be greater than zero")
        if (
            not math.isfinite(self.sample_interval_seconds)
            or self.sample_interval_seconds <= 0
        ):
            raise ValueError("sample interval must be greater than zero")


class ProcfsReader:
    def __init__(self, root: Path):
        self.root = root

    def _read_text(self, name: str) -> str:
        path = self.root / name
        try:
            return path.read_text(encoding="utf-8")
        except OSError as exc:
            detail = exc.strerror or str(exc)
            raise ProcfsError(f"cannot read {path}: {detail}") from exc

    def read_snapshot(self) -> HostSnapshot:
        active_swaps = self._parse_swaps(self._read_text("swaps"))
        total_bytes, available_bytes = self._parse_meminfo(
            self._read_text("meminfo")
        )
        return HostSnapshot(total_bytes, available_bytes, active_swaps)

    @staticmethod
    def _parse_meminfo(content: str) -> tuple[int, int]:
        values: dict[str, int] = {}
        required = {"MemTotal", "MemAvailable"}
        for line in content.splitlines():
            key, separator, raw_value = line.partition(":")
            if not separator or key not in required:
                continue
            if key in values:
                raise ProcfsError(f"duplicate {key} in meminfo")
            match = MEMINFO_VALUE_RE.fullmatch(raw_value.strip())
            if match is None:
                raise ProcfsError(f"malformed {key} in meminfo")
            values[key] = int(match.group(1)) * 1024

        missing = sorted(required - values.keys())
        if missing:
            raise ProcfsError(f"missing {', '.join(missing)} in meminfo")
        if values["MemAvailable"] > values["MemTotal"]:
            raise ProcfsError("MemAvailable exceeds MemTotal")
        return values["MemTotal"], values["MemAvailable"]

    @staticmethod
    def _parse_swaps(content: str) -> tuple[str, ...]:
        lines = content.splitlines()
        if not lines or lines[0].split() != SWAPS_HEADER:
            raise ProcfsError("malformed swaps header")

        entries: list[str] = []
        for line in lines[1:]:
            if not line.strip():
                continue
            fields = line.split()
            if len(fields) != len(SWAPS_HEADER):
                raise ProcfsError("malformed swaps entry")
            try:
                int(fields[2])
                int(fields[3])
                int(fields[4])
            except ValueError as exc:
                raise ProcfsError("malformed swaps entry") from exc
            entries.append(fields[0])
        return tuple(entries)


class AuditLogger:
    def __init__(
        self,
        stream: IO[str],
        wall_clock: Callable[[], datetime] | None = None,
    ):
        self.stream = stream
        self.wall_clock = wall_clock or (
            lambda: datetime.now(timezone.utc)
        )

    def emit(self, event: str, **fields: object) -> None:
        timestamp = self.wall_clock().astimezone(timezone.utc)
        record = {
            "timestamp": timestamp.isoformat(timespec="milliseconds").replace(
                "+00:00", "Z"
            ),
            "event": event,
            **fields,
        }
        self.stream.write(
            json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
        )
        self.stream.flush()


def _child_status(returncode: int | None, started: bool = True) -> str:
    if not started:
        return "not_started"
    if returncode is None:
        return "running"
    return "signaled" if returncode < 0 else "exited"


def _state_fields(
    snapshot: HostSnapshot | None,
    peak_used_bytes: int | None,
    child: ProcessHandle | None,
    child_returncode: int | None,
    process_group_status: str,
    threshold_reason: str,
) -> dict[str, object]:
    return {
        "total_bytes": snapshot.total_bytes if snapshot else None,
        "available_bytes": snapshot.available_bytes if snapshot else None,
        "used_bytes": snapshot.used_bytes if snapshot else None,
        "swap_entries": len(snapshot.active_swaps) if snapshot else None,
        "peak_used_bytes": peak_used_bytes,
        "child_pid": child.pid if child else None,
        "child_status": _child_status(
            child_returncode, started=child is not None
        ),
        "child_returncode": child_returncode,
        "process_group_id": child.pid if child else None,
        "process_group_status": process_group_status,
        "threshold_reason": threshold_reason,
    }


def _emit_final(
    audit: AuditLogger,
    classification: str,
    exit_code: int,
    reason: str,
    snapshot: HostSnapshot | None,
    peak_used_bytes: int | None,
    child: ProcessHandle | None = None,
    child_returncode: int | None = None,
    process_group_status: str = "not_created",
    error: str | None = None,
) -> int:
    fields = _state_fields(
        snapshot,
        peak_used_bytes,
        child,
        child_returncode,
        process_group_status,
        reason,
    )
    fields.update(classification=classification, exit_code=exit_code)
    if error:
        fields["error"] = error
    audit.emit("final", **fields)
    return exit_code


def _signal_process_group(process_group_id: int, signal_number: int) -> str:
    try:
        os.killpg(process_group_id, signal_number)
    except ProcessLookupError:
        return "missing"
    except OSError as exc:
        name = signal.Signals(signal_number).name
        detail = exc.strerror or str(exc)
        raise ProcessGroupError(
            f"cannot send {name} to process group {process_group_id}: {detail}"
        ) from exc
    return f"{signal.Signals(signal_number).name.lower()}_sent"


def _process_group_alive(process_group_id: int) -> bool:
    try:
        os.killpg(process_group_id, 0)
    except ProcessLookupError:
        return False
    except OSError as exc:
        detail = exc.strerror or str(exc)
        raise ProcessGroupError(
            f"cannot inspect process group {process_group_id}: {detail}"
        ) from exc
    return True


def _raise_parent_signal(signal_number: int, _frame: object) -> None:
    raise ParentSignal(signal_number)


def _set_parent_signal_handlers(
    handler: signal.Handlers,
) -> dict[int, signal.Handlers]:
    previous: dict[int, signal.Handlers] = {}
    for signal_number in PARENT_SIGNALS:
        previous[signal_number] = signal.signal(signal_number, handler)
    return previous


def _restore_parent_signal_handlers(
    previous: dict[int, signal.Handlers],
) -> None:
    for signal_number, handler in previous.items():
        signal.signal(signal_number, handler)


def _kill_and_finish(
    audit: AuditLogger,
    child: ProcessHandle,
    snapshot: HostSnapshot,
    peak_used_bytes: int,
    classification: str,
    exit_code: int,
    reason: str,
    signal_group: Callable[[int, int], str],
) -> int:
    try:
        group_status = signal_group(child.pid, signal.SIGKILL)
    except ProcessGroupError as exc:
        return _emit_final(
            audit,
            "signal_error",
            EXIT_SIGNAL_ERROR,
            reason,
            snapshot,
            peak_used_bytes,
            child,
            child.poll(),
            "signal_error",
            str(exc),
        )

    audit.emit(
        "process_group_signal",
        **_state_fields(
            snapshot,
            peak_used_bytes,
            child,
            child.poll(),
            group_status,
            reason,
        ),
        signal="SIGKILL",
    )
    try:
        child_returncode = child.wait(timeout=5.0)
    except subprocess.TimeoutExpired as exc:
        return _emit_final(
            audit,
            "termination_timeout",
            EXIT_SIGNAL_ERROR,
            reason,
            snapshot,
            peak_used_bytes,
            child,
            child.poll(),
            "sigkill_timeout",
            str(exc),
        )
    return _emit_final(
        audit,
        classification,
        exit_code,
        reason,
        snapshot,
        peak_used_bytes,
        child,
        child_returncode,
        group_status,
    )


def _graceful_cleanup(
    audit: AuditLogger,
    child: ProcessHandle,
    snapshot: HostSnapshot,
    peak_used_bytes: int,
    classification: str,
    exit_code: int,
    reason: str,
    graceful_signal: int | None,
    grace_seconds: float,
    signal_group: Callable[[int, int], str],
    group_alive: Callable[[int], bool],
    monotonic: Callable[[], float],
    sleeper: Callable[[float], None],
    process_group_status: str = "active",
    escalation_result: tuple[str, int, str] | None = None,
    error: str | None = None,
) -> int:
    escalated = False
    try:
        if graceful_signal is not None:
            process_group_status = signal_group(
                child.pid, graceful_signal
            )
            audit.emit(
                "process_group_signal",
                **_state_fields(
                    snapshot,
                    peak_used_bytes,
                    child,
                    child.poll(),
                    process_group_status,
                    reason,
                ),
                signal=signal.Signals(graceful_signal).name,
            )
        deadline = monotonic() + grace_seconds
        while monotonic() < deadline:
            child.poll()
            if not group_alive(child.pid):
                break
            sleeper(min(0.05, deadline - monotonic()))
        child.poll()
        if group_alive(child.pid):
            escalated = True
            process_group_status = signal_group(
                child.pid, signal.SIGKILL
            )
            signal_reason = (
                escalation_result[2]
                if escalation_result is not None
                else reason
            )
            audit.emit(
                "process_group_signal",
                **_state_fields(
                    snapshot,
                    peak_used_bytes,
                    child,
                    child.poll(),
                    process_group_status,
                    signal_reason,
                ),
                signal="SIGKILL",
            )
    except ProcessGroupError as exc:
        return _emit_final(
            audit,
            "signal_error",
            EXIT_SIGNAL_ERROR,
            reason,
            snapshot,
            peak_used_bytes,
            child,
            child.poll(),
            "signal_error",
            str(exc),
        )

    child_returncode = child.poll()
    if child_returncode is None:
        try:
            child_returncode = child.wait(timeout=5.0)
        except subprocess.TimeoutExpired as exc:
            return _emit_final(
                audit,
                "termination_timeout",
                EXIT_SIGNAL_ERROR,
                reason,
                snapshot,
                peak_used_bytes,
                child,
                child.poll(),
                "termination_timeout",
                str(exc),
            )

    if escalated and escalation_result is not None:
        classification, exit_code, reason = escalation_result
    return _emit_final(
        audit,
        classification,
        exit_code,
        reason,
        snapshot,
        peak_used_bytes,
        child,
        child_returncode,
        process_group_status,
        error,
    )


def _monitor_child(
    config: WatchdogConfig,
    reader: ProcfsReader,
    audit: AuditLogger,
    child: ProcessHandle,
    state: RuntimeState,
    signal_group: Callable[[int, int], str],
    group_alive: Callable[[int], bool],
    monotonic: Callable[[], float],
    sleeper: Callable[[float], None],
) -> int:
    soft_deadline: float | None = None

    while True:
        child_returncode = child.poll()
        if child_returncode is not None:
            soft_stop = soft_deadline is not None
            classification = "soft_limit" if soft_stop else "child_exit"
            exit_code = EXIT_SOFT_LIMIT if soft_stop else (
                128 - child_returncode
                if child_returncode < 0
                else child_returncode
            )
            reason = (
                "child exited during soft-threshold grace period"
                if soft_stop
                else "child exited"
            )
            if group_alive(child.pid):
                grace_seconds = config.grace_seconds
                graceful_signal: int | None = signal.SIGTERM
                group_status = "active"
                if soft_stop:
                    grace_seconds = max(
                        0.0, soft_deadline - monotonic()
                    )
                    graceful_signal = None
                    group_status = "sigterm_sent"
                return _graceful_cleanup(
                    audit,
                    child,
                    state.snapshot,
                    state.peak_used_bytes,
                    classification,
                    exit_code,
                    (
                        f"{reason}; process group members still running"
                    ),
                    graceful_signal,
                    grace_seconds,
                    signal_group,
                    group_alive,
                    monotonic,
                    sleeper,
                    group_status,
                    (
                        (
                            "grace_timeout",
                            EXIT_GRACE_TIMEOUT,
                            "soft-threshold grace period expired with "
                            "process group members still running",
                        )
                        if soft_stop
                        else None
                    ),
                )
            return _emit_final(
                audit,
                classification,
                exit_code,
                reason,
                state.snapshot,
                state.peak_used_bytes,
                child,
                child_returncode,
                "leader_exited",
            )

        now = monotonic()
        if soft_deadline is not None and now >= soft_deadline:
            return _kill_and_finish(
                audit,
                child,
                state.snapshot,
                state.peak_used_bytes,
                "grace_timeout",
                EXIT_GRACE_TIMEOUT,
                "soft-threshold grace period expired",
                signal_group,
            )

        try:
            state.snapshot = reader.read_snapshot()
        except ProcfsError as exc:
            return _kill_and_finish(
                audit,
                child,
                state.snapshot,
                state.peak_used_bytes,
                "procfs_error",
                EXIT_PROCFS_ERROR,
                str(exc),
                signal_group,
            )

        state.peak_used_bytes = max(
            state.peak_used_bytes, state.snapshot.used_bytes
        )
        audit.emit(
            "sample",
            **_state_fields(
                state.snapshot,
                state.peak_used_bytes,
                child,
                None,
                "active",
                "none",
            ),
        )

        if state.snapshot.active_swaps:
            return _kill_and_finish(
                audit,
                child,
                state.snapshot,
                state.peak_used_bytes,
                "swap_appeared",
                EXIT_SWAP_ACTIVE,
                "active swap appeared during execution",
                signal_group,
            )
        if state.snapshot.used_bytes >= config.emergency_bytes:
            return _kill_and_finish(
                audit,
                child,
                state.snapshot,
                state.peak_used_bytes,
                "emergency_limit",
                EXIT_EMERGENCY_LIMIT,
                "used_bytes >= emergency_bytes",
                signal_group,
            )
        if (
            soft_deadline is None
            and state.snapshot.used_bytes >= config.soft_bytes
        ):
            try:
                group_status = signal_group(child.pid, signal.SIGTERM)
            except ProcessGroupError as exc:
                return _emit_final(
                    audit,
                    "signal_error",
                    EXIT_SIGNAL_ERROR,
                    "used_bytes >= soft_bytes",
                    state.snapshot,
                    state.peak_used_bytes,
                    child,
                    child.poll(),
                    "signal_error",
                    str(exc),
                )
            soft_deadline = now + config.grace_seconds
            audit.emit(
                "process_group_signal",
                **_state_fields(
                    state.snapshot,
                    state.peak_used_bytes,
                    child,
                    child.poll(),
                    group_status,
                    "used_bytes >= soft_bytes",
                ),
                signal="SIGTERM",
                grace_deadline_monotonic=soft_deadline,
            )

        sleep_seconds = config.sample_interval_seconds
        if soft_deadline is not None:
            sleep_seconds = min(
                sleep_seconds,
                max(0.0, soft_deadline - monotonic()),
            )
        sleeper(sleep_seconds)


def run_watchdog(
    config: WatchdogConfig,
    *,
    reader: ProcfsReader | None = None,
    audit: AuditLogger | None = None,
    launcher: Callable[..., ProcessHandle] | None = None,
    signal_group: Callable[[int, int], str] | None = None,
    group_alive: Callable[[int], bool] | None = None,
    monotonic: Callable[[], float] | None = None,
    sleeper: Callable[[float], None] | None = None,
) -> int:
    config.validate()
    reader = reader or ProcfsReader(config.procfs_root)
    audit = audit or AuditLogger(sys.stderr)
    launcher = launcher or subprocess.Popen
    signal_group = signal_group or _signal_process_group
    group_alive = group_alive or _process_group_alive
    monotonic = monotonic or time.monotonic
    sleeper = sleeper or time.sleep

    try:
        snapshot = reader.read_snapshot()
    except ProcfsError as exc:
        return _emit_final(
            audit,
            "procfs_error",
            EXIT_PROCFS_ERROR,
            str(exc),
            None,
            None,
            error=str(exc),
        )

    audit.emit(
        "preflight",
        **_state_fields(
            snapshot,
            snapshot.used_bytes,
            None,
            None,
            "not_created",
            "none",
        ),
        soft_bytes=config.soft_bytes,
        emergency_bytes=config.emergency_bytes,
        strict_ceiling_bytes=STRICT_CEILING_BYTES,
    )

    if snapshot.active_swaps:
        return _emit_final(
            audit,
            "startup_swap_active",
            EXIT_SWAP_ACTIVE,
            "active swap present before command launch",
            snapshot,
            snapshot.used_bytes,
        )
    if snapshot.used_bytes >= config.emergency_bytes:
        return _emit_final(
            audit,
            "startup_emergency_limit",
            EXIT_EMERGENCY_LIMIT,
            "used_bytes >= emergency_bytes before launch",
            snapshot,
            snapshot.used_bytes,
        )
    if snapshot.used_bytes >= config.soft_bytes:
        return _emit_final(
            audit,
            "startup_soft_limit",
            EXIT_SOFT_LIMIT,
            "used_bytes >= soft_bytes before launch",
            snapshot,
            snapshot.used_bytes,
        )

    previous_mask = signal.pthread_sigmask(
        signal.SIG_BLOCK, PARENT_SIGNALS
    )
    mask_restored = False
    previous_handlers: dict[int, signal.Handlers] = {}
    child: ProcessHandle | None = None
    state = RuntimeState(snapshot, snapshot.used_bytes)
    try:
        launch_mask = previous_mask

        def restore_child_signal_mask() -> None:
            signal.pthread_sigmask(signal.SIG_SETMASK, launch_mask)

        try:
            child = launcher(
                config.command,
                start_new_session=True,
                preexec_fn=restore_child_signal_mask,
            )
        except (OSError, ValueError) as exc:
            detail = getattr(exc, "strerror", None) or str(exc)
            return _emit_final(
                audit,
                "launch_error",
                EXIT_LAUNCH_ERROR,
                "command launch failed",
                snapshot,
                snapshot.used_bytes,
                error=detail,
            )

        previous_handlers = _set_parent_signal_handlers(
            _raise_parent_signal
        )
        signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)
        mask_restored = True
        audit.emit(
            "child_started",
            **_state_fields(
                state.snapshot,
                state.peak_used_bytes,
                child,
                None,
                "active",
                "none",
            ),
            command=list(config.command),
        )
        return _monitor_child(
            config,
            reader,
            audit,
            child,
            state,
            signal_group,
            group_alive,
            monotonic,
            sleeper,
        )
    except ParentSignal as exc:
        _set_parent_signal_handlers(signal.SIG_IGN)
        signal_name = signal.Signals(exc.signal_number).name
        return _graceful_cleanup(
            audit,
            child,
            state.snapshot,
            state.peak_used_bytes,
            "parent_signal",
            128 + exc.signal_number,
            f"wrapper received {signal_name}",
            exc.signal_number,
            config.grace_seconds,
            signal_group,
            group_alive,
            monotonic,
            sleeper,
        )
    except Exception as exc:
        _set_parent_signal_handlers(signal.SIG_IGN)
        return _graceful_cleanup(
            audit,
            child,
            state.snapshot,
            state.peak_used_bytes,
            "internal_error",
            EXIT_INTERNAL_ERROR,
            "unexpected post-launch exception",
            signal.SIGTERM,
            config.grace_seconds,
            signal_group,
            group_alive,
            monotonic,
            sleeper,
            error=f"{type(exc).__name__}: {exc}",
        )
    finally:
        if not mask_restored:
            signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)
        if previous_handlers:
            _restore_parent_signal_handlers(previous_handlers)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def parse_args(argv: Sequence[str]) -> WatchdogConfig:
    parser = argparse.ArgumentParser(
        description=(
            "Launch a command in a new process group and stop it before "
            "host-wide memory use reaches the 120 GiB Strix validation ceiling."
        )
    )
    parser.add_argument(
        "--procfs-root",
        type=Path,
        default=Path("/proc"),
        help="procfs root containing meminfo and swaps (default: /proc)",
    )
    parser.add_argument(
        "--soft-gib",
        type=_positive_int,
        default=116,
        help="send SIGTERM at this many GiB used (default: 116)",
    )
    parser.add_argument(
        "--emergency-gib",
        type=_positive_int,
        default=118,
        help=(
            "send SIGKILL at this many GiB used (default: 118, leaving "
            "a 2 GiB sampling margin below 120 GiB)"
        ),
    )
    parser.add_argument(
        "--grace-seconds",
        type=_positive_float,
        default=DEFAULT_GRACE_SECONDS,
        help="maximum time after SIGTERM before SIGKILL (default: 30)",
    )
    parser.add_argument(
        "--sample-interval-seconds",
        type=_positive_float,
        default=DEFAULT_SAMPLE_INTERVAL_SECONDS,
        help="procfs sampling interval (default: 1)",
    )
    parser.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="command and arguments, preceded by --",
    )
    args = parser.parse_args(argv)
    command = tuple(args.command)
    if command and command[0] == "--":
        command = command[1:]
    return WatchdogConfig(
        command=command,
        procfs_root=args.procfs_root,
        soft_bytes=args.soft_gib * GIB,
        emergency_bytes=args.emergency_gib * GIB,
        grace_seconds=args.grace_seconds,
        sample_interval_seconds=args.sample_interval_seconds,
    )


def main(argv: Sequence[str] | None = None) -> int:
    config = parse_args(argv if argv is not None else sys.argv[1:])
    audit = AuditLogger(sys.stderr)
    try:
        return run_watchdog(config, audit=audit)
    except ValueError as exc:
        return _emit_final(
            audit,
            "configuration_error",
            EXIT_PROCFS_ERROR,
            "invalid configuration",
            None,
            None,
            error=str(exc),
        )


if __name__ == "__main__":
    sys.exit(main())
