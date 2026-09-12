#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import io
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "strix_memory_watchdog.py"
)
SPEC = importlib.util.spec_from_file_location(
    "strix_memory_watchdog", SCRIPT_PATH
)
assert SPEC is not None
assert SPEC.loader is not None
watchdog = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = watchdog
SPEC.loader.exec_module(watchdog)


def snapshot(
    used_bytes: int,
    *,
    total_bytes: int = 200,
    active_swaps: tuple[str, ...] = (),
) -> Any:
    return watchdog.HostSnapshot(
        total_bytes=total_bytes,
        available_bytes=total_bytes - used_bytes,
        active_swaps=active_swaps,
    )


class SequenceReader:
    def __init__(self, values: list[Any]):
        self.values = values
        self.index = 0

    def read_snapshot(self) -> Any:
        index = min(self.index, len(self.values) - 1)
        self.index += 1
        value = self.values[index]
        if isinstance(value, Exception):
            raise value
        return value


class FakeClock:
    def __init__(self):
        self.value = 0.0

    def monotonic(self) -> float:
        return self.value

    def sleep(self, seconds: float) -> None:
        self.value += seconds


class FakeProcess:
    def __init__(self, returncode: int | None = None):
        self.pid = 4321
        self.returncode = returncode

    def poll(self) -> int | None:
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        if self.returncode is None:
            raise subprocess.TimeoutExpired("fake", timeout)
        return self.returncode


class Harness:
    def __init__(
        self,
        values: list[Any],
        process: FakeProcess,
        signal_handler: Any | None = None,
    ):
        self.reader = SequenceReader(values)
        self.process = process
        self.signal_handler = signal_handler
        self.clock = FakeClock()
        self.stream = io.StringIO()
        self.launched = False
        self.signals: list[int] = []
        fixed_time = datetime(2026, 1, 1, tzinfo=timezone.utc)
        self.audit = watchdog.AuditLogger(
            self.stream, wall_clock=lambda: fixed_time
        )

    def launcher(self, command: tuple[str, ...], **kwargs: Any) -> FakeProcess:
        self.launched = True
        self.command = command
        self.launch_kwargs = kwargs
        return self.process

    def signal_group(self, process_group_id: int, signal_number: int) -> str:
        self.signals.append(signal_number)
        if self.signal_handler is not None:
            self.signal_handler(self.process, signal_number)
        return f"{signal.Signals(signal_number).name.lower()}_sent"

    def group_alive(self, process_group_id: int) -> bool:
        return self.process.returncode is None

    def run(self, **overrides: Any) -> int:
        config = watchdog.WatchdogConfig(
            command=("fake-command",),
            soft_bytes=100,
            emergency_bytes=150,
            grace_seconds=2,
            sample_interval_seconds=1,
            **overrides,
        )
        return watchdog.run_watchdog(
            config,
            reader=self.reader,
            audit=self.audit,
            launcher=self.launcher,
            signal_group=self.signal_group,
            group_alive=self.group_alive,
            monotonic=self.clock.monotonic,
            sleeper=self.clock.sleep,
        )

    def records(self) -> list[dict[str, Any]]:
        return [
            json.loads(line)
            for line in self.stream.getvalue().splitlines()
        ]


class TestProcfsParsing(unittest.TestCase):
    def test_parses_meminfo_as_integer_bytes_and_allows_zero_swap(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "meminfo").write_text(
                "MemTotal:       131072 kB\n"
                "MemFree:          4096 kB\n"
                "MemAvailable:    32768 kB\n",
                encoding="utf-8",
            )
            (root / "swaps").write_text(
                "Filename Type Size Used Priority\n",
                encoding="utf-8",
            )

            result = watchdog.ProcfsReader(root).read_snapshot()

        self.assertEqual(result.total_bytes, 131072 * 1024)
        self.assertEqual(result.available_bytes, 32768 * 1024)
        self.assertEqual(result.used_bytes, 98304 * 1024)
        self.assertEqual(result.active_swaps, ())

    def test_rejects_active_swap_entry(self) -> None:
        content = (
            "Filename Type Size Used Priority\n"
            "/swapfile file 1048572 0 -2\n"
        )
        self.assertEqual(
            watchdog.ProcfsReader._parse_swaps(content),
            ("/swapfile",),
        )

    def test_rejects_malformed_or_missing_procfs_data(self) -> None:
        with self.assertRaisesRegex(
            watchdog.ProcfsError, "malformed MemAvailable"
        ):
            watchdog.ProcfsReader._parse_meminfo(
                "MemTotal: 10 kB\nMemAvailable: unknown\n"
            )
        with self.assertRaisesRegex(
            watchdog.ProcfsError, "missing MemAvailable"
        ):
            watchdog.ProcfsReader._parse_meminfo("MemTotal: 10 kB\n")
        with self.assertRaisesRegex(
            watchdog.ProcfsError, "malformed swaps header"
        ):
            watchdog.ProcfsReader._parse_swaps("")
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(
                watchdog.ProcfsError, "cannot read"
            ):
                watchdog.ProcfsReader(
                    Path(temp_dir)
                ).read_snapshot()


class TestWatchdogBehavior(unittest.TestCase):
    @staticmethod
    def _process_is_running(process_id: int) -> bool:
        result = subprocess.run(
            ["ps", "-o", "stat=", "-p", str(process_id)],
            capture_output=True,
            check=False,
            text=True,
        )
        return result.returncode == 0 and not result.stdout.lstrip().startswith(
            "Z"
        )

    def test_parent_signals_leave_no_child_or_grandchild(self) -> None:
        child_code = (
            "import os,signal,sys,time;"
            "signal.signal(signal.SIGINT,signal.SIG_IGN);"
            "signal.signal(signal.SIGTERM,signal.SIG_IGN);"
            "grandchild=os.fork();"
            "\nif grandchild == 0:\n"
            " time.sleep(30)\n"
            "else:\n"
            " open(sys.argv[1],'w').write("
            "f'{os.getpid()} {grandchild}\\n');"
            " time.sleep(30)\n"
        )
        for signal_number in (signal.SIGINT, signal.SIGTERM):
            with self.subTest(signal=signal.Signals(signal_number).name):
                with tempfile.TemporaryDirectory() as temp_dir:
                    root = Path(temp_dir)
                    pid_file = root / "pids"
                    (root / "meminfo").write_text(
                        "MemTotal: 131072 kB\n"
                        "MemAvailable: 65536 kB\n",
                        encoding="utf-8",
                    )
                    (root / "swaps").write_text(
                        "Filename Type Size Used Priority\n",
                        encoding="utf-8",
                    )
                    audit_path = root / "audit.jsonl"
                    with audit_path.open("w", encoding="utf-8") as audit:
                        wrapper = subprocess.Popen(
                            [
                                sys.executable,
                                str(SCRIPT_PATH),
                                "--procfs-root",
                                str(root),
                                "--grace-seconds",
                                "0.2",
                                "--sample-interval-seconds",
                                "0.05",
                                "--",
                                sys.executable,
                                "-c",
                                child_code,
                                str(pid_file),
                            ],
                            stderr=audit,
                            text=True,
                        )
                        child_pid = None
                        grandchild_pid = None
                        try:
                            deadline = time.monotonic() + 5
                            while not pid_file.exists():
                                if time.monotonic() >= deadline:
                                    self.fail(
                                        "child process group did not start"
                                    )
                                time.sleep(0.01)
                            child_pid, grandchild_pid = (
                                int(value)
                                for value in pid_file.read_text(
                                    encoding="utf-8"
                                ).split()
                            )
                            time.sleep(0.05)
                            wrapper.send_signal(signal_number)
                            wrapper.wait(timeout=5)
                        finally:
                            if wrapper.poll() is None:
                                wrapper.kill()
                                wrapper.wait(timeout=5)
                            if child_pid is not None:
                                try:
                                    os.killpg(child_pid, signal.SIGKILL)
                                except ProcessLookupError:
                                    pass

                    self.assertEqual(
                        wrapper.returncode, 128 + signal_number
                    )
                    records = [
                        json.loads(line)
                        for line in audit_path.read_text(
                            encoding="utf-8"
                        ).splitlines()
                    ]
                    self.assertEqual(
                        records[-1]["classification"], "parent_signal"
                    )
                    forwarded = [
                        record["signal"]
                        for record in records
                        if record["event"] == "process_group_signal"
                    ]
                    self.assertEqual(
                        forwarded[0],
                        signal.Signals(signal_number).name,
                    )
                    self.assertEqual(forwarded[-1], "SIGKILL")
                    for process_id in (child_pid, grandchild_pid):
                        deadline = time.monotonic() + 2
                        while (
                            self._process_is_running(process_id)
                            and time.monotonic() < deadline
                        ):
                            time.sleep(0.01)
                        self.assertFalse(
                            self._process_is_running(process_id)
                        )

    def test_configuration_rejects_non_finite_timing(self) -> None:
        config = watchdog.WatchdogConfig(
            command=("fake-command",),
            grace_seconds=float("nan"),
        )
        with self.assertRaisesRegex(ValueError, "grace period"):
            config.validate()

    def test_cli_fixture_launches_command_and_propagates_exit(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "meminfo").write_text(
                "MemTotal: 131072 kB\nMemAvailable: 65536 kB\n",
                encoding="utf-8",
            )
            (root / "swaps").write_text(
                "Filename Type Size Used Priority\n",
                encoding="utf-8",
            )
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT_PATH),
                    "--procfs-root",
                    str(root),
                    "--sample-interval-seconds",
                    "0.01",
                    "--",
                    sys.executable,
                    "-c",
                    "raise SystemExit(23)",
                ],
                capture_output=True,
                check=False,
                text=True,
            )

        self.assertEqual(result.returncode, 23)
        records = [
            json.loads(line) for line in result.stderr.splitlines()
        ]
        self.assertEqual(records[-1]["classification"], "child_exit")
        self.assertEqual(records[-1]["child_returncode"], 23)

    def test_zero_swap_gate_launches_and_propagates_child_exit(self) -> None:
        harness = Harness([snapshot(50)], FakeProcess(returncode=37))

        result = harness.run()

        self.assertEqual(result, 37)
        self.assertTrue(harness.launched)
        self.assertTrue(harness.launch_kwargs["start_new_session"])
        final = harness.records()[-1]
        self.assertEqual(final["classification"], "child_exit")
        self.assertEqual(final["total_bytes"], 200)
        self.assertEqual(final["available_bytes"], 150)
        self.assertEqual(final["used_bytes"], 50)
        self.assertEqual(final["peak_used_bytes"], 50)
        self.assertEqual(final["child_status"], "exited")
        self.assertEqual(final["process_group_status"], "leader_exited")

    def test_signaled_child_exit_uses_shell_exit_convention(self) -> None:
        harness = Harness(
            [snapshot(50)],
            FakeProcess(returncode=-signal.SIGTERM),
        )

        result = harness.run()

        self.assertEqual(result, 128 + signal.SIGTERM)

    def test_active_swap_rejects_startup_without_launch(self) -> None:
        harness = Harness(
            [snapshot(50, active_swaps=("/swapfile",))],
            FakeProcess(),
        )

        result = harness.run()

        self.assertEqual(result, watchdog.EXIT_SWAP_ACTIVE)
        self.assertFalse(harness.launched)
        self.assertEqual(
            harness.records()[-1]["classification"],
            "startup_swap_active",
        )

    def test_soft_limit_sends_sigterm(self) -> None:
        def exit_on_term(process: FakeProcess, signal_number: int) -> None:
            if signal_number == signal.SIGTERM:
                process.returncode = -signal.SIGTERM

        harness = Harness(
            [snapshot(50), snapshot(110)],
            FakeProcess(),
            signal_handler=exit_on_term,
        )

        result = harness.run()

        self.assertEqual(result, watchdog.EXIT_SOFT_LIMIT)
        self.assertEqual(harness.signals, [signal.SIGTERM])
        self.assertEqual(
            harness.records()[-1]["classification"], "soft_limit"
        )

    def test_emergency_limit_sends_sigkill(self) -> None:
        def exit_on_kill(process: FakeProcess, signal_number: int) -> None:
            if signal_number == signal.SIGKILL:
                process.returncode = -signal.SIGKILL

        harness = Harness(
            [snapshot(50), snapshot(160)],
            FakeProcess(),
            signal_handler=exit_on_kill,
        )

        result = harness.run()

        self.assertEqual(result, watchdog.EXIT_EMERGENCY_LIMIT)
        self.assertEqual(harness.signals, [signal.SIGKILL])
        self.assertEqual(
            harness.records()[-1]["classification"], "emergency_limit"
        )

    def test_grace_timeout_escalates_to_sigkill(self) -> None:
        def ignore_term(process: FakeProcess, signal_number: int) -> None:
            if signal_number == signal.SIGKILL:
                process.returncode = -signal.SIGKILL

        harness = Harness(
            [snapshot(50), snapshot(110)],
            FakeProcess(),
            signal_handler=ignore_term,
        )

        result = harness.run()

        self.assertEqual(result, watchdog.EXIT_GRACE_TIMEOUT)
        self.assertEqual(
            harness.signals,
            [signal.SIGTERM, signal.SIGKILL],
        )
        self.assertEqual(harness.clock.value, 2.0)
        self.assertEqual(
            harness.records()[-1]["classification"], "grace_timeout"
        )

    def test_swap_appearing_during_execution_kills_group(self) -> None:
        def exit_on_kill(process: FakeProcess, signal_number: int) -> None:
            if signal_number == signal.SIGKILL:
                process.returncode = -signal.SIGKILL

        harness = Harness(
            [
                snapshot(50),
                snapshot(60, active_swaps=("/swapfile",)),
            ],
            FakeProcess(),
            signal_handler=exit_on_kill,
        )

        result = harness.run()

        self.assertEqual(result, watchdog.EXIT_SWAP_ACTIVE)
        self.assertEqual(harness.signals, [signal.SIGKILL])
        self.assertEqual(
            harness.records()[-1]["classification"], "swap_appeared"
        )

    def test_runtime_procfs_error_kills_group(self) -> None:
        def exit_on_kill(process: FakeProcess, signal_number: int) -> None:
            if signal_number == signal.SIGKILL:
                process.returncode = -signal.SIGKILL

        harness = Harness(
            [snapshot(50), watchdog.ProcfsError("missing meminfo")],
            FakeProcess(),
            signal_handler=exit_on_kill,
        )

        result = harness.run()

        self.assertEqual(result, watchdog.EXIT_PROCFS_ERROR)
        self.assertEqual(harness.signals, [signal.SIGKILL])
        self.assertEqual(
            harness.records()[-1]["classification"], "procfs_error"
        )

    def test_unexpected_monitor_error_cleans_up_process_group(self) -> None:
        def exit_on_kill(process: FakeProcess, signal_number: int) -> None:
            if signal_number == signal.SIGKILL:
                process.returncode = -signal.SIGKILL

        harness = Harness(
            [snapshot(50), RuntimeError("unexpected")],
            FakeProcess(),
            signal_handler=exit_on_kill,
        )

        result = harness.run()

        self.assertEqual(result, watchdog.EXIT_INTERNAL_ERROR)
        self.assertEqual(
            harness.signals,
            [signal.SIGTERM, signal.SIGKILL],
        )
        final = harness.records()[-1]
        self.assertEqual(final["classification"], "internal_error")
        self.assertIn("RuntimeError: unexpected", final["error"])

    def test_launch_failure_is_explicit(self) -> None:
        harness = Harness([snapshot(50)], FakeProcess())

        def fail_launch(
            command: tuple[str, ...], **kwargs: Any
        ) -> FakeProcess:
            raise FileNotFoundError(2, "No such file or directory")

        harness.launcher = fail_launch

        result = harness.run()

        self.assertEqual(result, watchdog.EXIT_LAUNCH_ERROR)
        self.assertEqual(
            harness.records()[-1]["classification"], "launch_error"
        )


if __name__ == "__main__":
    unittest.main()
