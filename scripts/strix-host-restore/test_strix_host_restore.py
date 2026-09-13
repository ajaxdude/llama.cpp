import copy
import hashlib
import hmac
import importlib.machinery
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock


SCRIPT = Path(__file__).with_name("strix-host-restore")
LOADER = importlib.machinery.SourceFileLoader("strix_host_restore", str(SCRIPT))
SPEC = importlib.util.spec_from_loader(LOADER.name, LOADER)
restore = importlib.util.module_from_spec(SPEC)
LOADER.exec_module(restore)


NOW = 2_000_000_000
NONCE = "restore_nonce_123456"
DEADLINE = NOW + 60
KEY = bytes(range(32))


def plan_data():
    return {
        "schema": 1,
        "platform": {
            "os": "fedora",
            "architecture": "x86_64",
            "device": "strix-halo",
        },
        "nonce": NONCE,
        "restore_deadline_epoch": DEADLINE,
        "expires_epoch": DEADLINE + 600,
        "target_uid": 1000,
        "components": {
            "swapfile": {"enabled": True},
            "docker_monerod": {
                "running": True,
                "health_timeout_seconds": 10,
            },
            "p2pool": {"active": True, "enabled": True},
            "xmrig": {"active": True, "enabled": True},
            "llama_swap": {"active": True, "enabled": True},
            "proxy": {"active": True, "enabled": True},
        },
    }


def make_plan(data=None):
    data = data or plan_data()
    raw = json.dumps(data, sort_keys=True).encode()
    return restore.parse_plan(
        raw,
        hashlib.sha256(raw).hexdigest(),
        NONCE,
        data["restore_deadline_epoch"],
        NOW,
        require_due=False,
    )


class PlanFiles:
    def __init__(self, root, raw=None):
        self.root = Path(root)
        self.raw = raw or json.dumps(plan_data(), sort_keys=True).encode()
        self.plan = self.root / "plan.json"
        self.signature = self.root / "plan.sig"
        self.key = self.root / "plan.key"
        self.plan.write_bytes(self.raw)
        self.signature.write_text(hmac.new(KEY, self.raw, hashlib.sha256).hexdigest() + "\n")
        self.key.write_text(KEY.hex() + "\n")
        for path in (self.plan, self.signature, self.key):
            path.chmod(0o600)

    def load(self, **overrides):
        arguments = {
            "plan_path": self.plan,
            "signature_path": self.signature,
            "key_path": self.key,
            "expected_uid": os.geteuid(),
            "expected_nonce": NONCE,
            "expected_deadline": DEADLINE,
            "now": NOW,
            "require_due": False,
        }
        arguments.update(overrides)
        return restore.load_authenticated_plan(**arguments)


class RecordingAudit:
    def __init__(self, fail_at=None):
        self.records = []
        self.fail_at = fail_at

    def write(self, state):
        if self.fail_at is not None and len(self.records) == self.fail_at:
            raise restore.RestoreError("injected audit failure")
        self.records.append(copy.deepcopy(state))


class FakeClock:
    def __init__(self, wall=NOW):
        self.monotonic_value = 0
        self.wall_value = wall

    def monotonic(self):
        return self.monotonic_value

    def wall(self):
        return self.wall_value

    def sleep(self, seconds):
        self.advance(seconds)

    def advance(self, seconds):
        self.monotonic_value += seconds
        self.wall_value += seconds


class FakeRunner:
    def __init__(self):
        self.calls = []
        self.timeouts = []
        self.swap_active = False
        self.docker_running = False
        self.docker_health_test = '["CMD","check"]'
        self.docker_health = ["starting", "healthy"]
        self.units = {
            "p2pool.service": {"enabled": "disabled", "active": "inactive"},
            "xmrig.service": {"enabled": "disabled", "active": "inactive"},
            "llama-swap.service": {"enabled": "disabled", "active": "inactive"},
            "llama-proxy.service": {"enabled": "disabled", "active": "inactive"},
        }
        self.fail_on = None

    def run(self, args, allowed_returncodes=(0,), timeout=30):
        args = tuple(args)
        self.calls.append(args)
        self.timeouts.append(timeout)
        if self.fail_on is not None and args[-len(self.fail_on):] == self.fail_on:
            raise restore.RestoreError("injected command failure")
        if args[:1] == (restore.SWAPON,):
            if args[1:] == ("--show=NAME", "--noheadings", "--raw"):
                return restore.SWAP_PATH if self.swap_active else ""
            if args[1:] == (restore.SWAP_PATH,):
                self.swap_active = True
                return ""
        if args[:2] == (restore.DOCKER, "start"):
            self.docker_running = True
            return restore.DOCKER_CONTAINER
        if args[:2] == (restore.DOCKER, "inspect"):
            template = args[2]
            if template == "--format={{.State.Running}}":
                return "true" if self.docker_running else "false"
            if template == "--format={{if .Config.Healthcheck}}{{json .Config.Healthcheck.Test}}{{end}}":
                return self.docker_health_test
            if template == "--format={{.State.Health.Status}}":
                if len(self.docker_health) > 1:
                    return self.docker_health.pop(0)
                return self.docker_health[0]
        try:
            index = args.index(restore.SYSTEMCTL)
        except ValueError:
            index = -1
        if index >= 0:
            offset = index + 1
            if args[offset] == "--user":
                offset += 1
            operation, unit = args[offset:offset + 2]
            state = self.units[unit]
            if operation == "is-enabled":
                return state["enabled"]
            if operation == "is-active":
                return state["active"]
            if operation == "enable":
                state["enabled"] = "enabled"
                return ""
            if operation == "start":
                state["active"] = "active"
                return ""
        raise AssertionError(f"unexpected command: {args}")


class AdvancingRunner(FakeRunner):
    def __init__(self, clock, suffix, advance_seconds):
        super().__init__()
        self.clock = clock
        self.suffix = tuple(suffix)
        self.advance_seconds = advance_seconds

    def run(self, args, allowed_returncodes=(0,), timeout=30):
        result = super().run(args, allowed_returncodes, timeout)
        args = tuple(args)
        if args[-len(self.suffix):] == self.suffix:
            self.clock.advance(self.advance_seconds)
        return result


def runtime_prefix(uid, username):
    return (
        restore.RUNUSER,
        "--user",
        username,
        "--",
        restore.ENV,
        f"XDG_RUNTIME_DIR=/run/user/{uid}",
        f"DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/{uid}/bus",
    )


class PlanValidationTests(unittest.TestCase):
    def test_authenticated_plan_and_action_order(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            plan = PlanFiles(directory).load()
        self.assertEqual(
            plan.intended_actions(),
            [
                "enable swap on /swapfile",
                "start Docker container monerod, then wait up to 10s for its configured health state",
                "enable system unit p2pool.service",
                "start system unit p2pool.service",
                "enable system unit xmrig.service",
                "start system unit xmrig.service",
                "enable user unit llama-swap.service as UID 1000",
                "start user unit llama-swap.service as UID 1000",
                "enable user unit llama-proxy.service as UID 1000",
                "start user unit llama-proxy.service as UID 1000",
            ],
        )

    def test_tampered_plan_is_rejected(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            files = PlanFiles(directory)
            files.plan.write_bytes(files.raw + b"\n")
            with self.assertRaisesRegex(restore.RestoreError, "signature is invalid"):
                files.load()

    def test_duplicate_unknown_missing_and_unsupported_values_are_rejected(self):
        cases = []
        duplicate = json.dumps(plan_data(), sort_keys=True)
        duplicate = duplicate.replace('"schema": 1,', '"schema": 1, "schema": 1,', 1)
        cases.append((duplicate.encode(), "duplicate JSON key"))

        unknown = plan_data()
        unknown["components"]["surprise"] = {"active": True}
        cases.append((json.dumps(unknown).encode(), "unknown keys"))

        unsafe_path = plan_data()
        unsafe_path["components"]["swapfile"]["path"] = "/tmp/swapfile"
        cases.append((json.dumps(unsafe_path).encode(), "unknown keys"))

        missing = plan_data()
        del missing["components"]["xmrig"]
        cases.append((json.dumps(missing).encode(), "missing keys"))

        bad_state = plan_data()
        bad_state["components"]["p2pool"]["active"] = "yes"
        cases.append((json.dumps(bad_state).encode(), "must be a boolean"))

        bad_schema = plan_data()
        bad_schema["schema"] = True
        cases.append((json.dumps(bad_schema).encode(), "schema must be an integer"))

        bad_platform = plan_data()
        bad_platform["platform"]["os"] = "ubuntu"
        cases.append((json.dumps(bad_platform).encode(), "must be Fedora"))

        for raw, message in cases:
            with self.subTest(message=message):
                with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
                    with self.assertRaisesRegex(restore.RestoreError, message):
                        PlanFiles(directory, raw).load()

    def test_expiry_deadline_and_nonce_are_bound(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            files = PlanFiles(directory)
            with self.assertRaisesRegex(restore.RestoreError, "nonce"):
                files.load(expected_nonce="different_nonce_1234")
            with self.assertRaisesRegex(restore.RestoreError, "deadline"):
                files.load(expected_deadline=DEADLINE + 1)
            with self.assertRaisesRegex(restore.RestoreError, "expired"):
                files.load(now=DEADLINE + 601)
            with self.assertRaisesRegex(restore.RestoreError, "has not been reached"):
                files.load(require_due=True)

    def test_plan_signature_and_key_symlinks_are_rejected(self):
        for selected in ("plan", "signature", "key"):
            with self.subTest(selected=selected):
                with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
                    files = PlanFiles(directory)
                    path = getattr(files, selected)
                    target = Path(directory) / f"{selected}.target"
                    path.rename(target)
                    path.symlink_to(target)
                    with self.assertRaisesRegex(restore.RestoreError, "securely open"):
                        files.load()

    def test_parent_directory_symlink_is_rejected(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            root = Path(directory)
            real = root / "real"
            real.mkdir(mode=0o700)
            files = PlanFiles(real)
            link = root / "linked"
            link.symlink_to(real, target_is_directory=True)
            with self.assertRaisesRegex(restore.RestoreError, "cannot inspect|unsafe directory"):
                files.load(
                    plan_path=link / "plan.json",
                    signature_path=link / "plan.sig",
                    key_path=link / "plan.key",
                )

    def test_untrusted_or_writable_ancestor_is_rejected(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            writable = Path(directory) / "writable"
            writable.mkdir(mode=0o700)
            writable.chmod(0o777)
            private = writable / "private"
            private.mkdir(mode=0o700)
            files = PlanFiles(private)
            with self.assertRaisesRegex(restore.RestoreError, "group-writable or world-writable"):
                files.load()
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            files = PlanFiles(directory)
            with self.assertRaisesRegex(restore.RestoreError, "untrusted owner"):
                files.load(expected_uid=os.geteuid() + 1)

    def test_plan_directory_must_be_private(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            files = PlanFiles(directory)
            Path(directory).chmod(0o755)
            with self.assertRaisesRegex(restore.RestoreError, "mode 0700"):
                files.load()

    def test_plan_directory_replacement_is_rejected(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            root = Path(directory)
            plan_directory = root / "plan"
            plan_directory.mkdir(mode=0o700)
            files = PlanFiles(plan_directory)
            original_read = restore.secure_read_at
            replaced = False

            def replace_after_plan(*args, **kwargs):
                nonlocal replaced
                result = original_read(*args, **kwargs)
                if args[4] == "plan" and not replaced:
                    replaced = True
                    plan_directory.rename(root / "original")
                    plan_directory.mkdir(mode=0o700)
                return result

            with mock.patch.object(restore, "secure_read_at", side_effect=replace_after_plan):
                with self.assertRaisesRegex(restore.RestoreError, "replaced during validation"):
                    files.load()

    def test_fifo_plan_is_rejected_without_blocking(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            files = PlanFiles(directory)
            files.plan.unlink()
            os.mkfifo(files.plan, mode=0o600)
            started = time.monotonic()
            with self.assertRaisesRegex(restore.RestoreError, "not a regular file"):
                files.load()
            self.assertLess(time.monotonic() - started, 1)

    def test_file_mode_and_hard_links_are_rejected(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            files = PlanFiles(directory)
            files.plan.chmod(0o640)
            with self.assertRaisesRegex(restore.RestoreError, "mode must be 0600"):
                files.load()
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            files = PlanFiles(directory)
            os.link(files.plan, Path(directory) / "plan-copy.json")
            with self.assertRaisesRegex(restore.RestoreError, "exactly one hard link"):
                files.load()

    def test_non_root_validate_command_executes_no_host_commands(self):
        now = int(time.time())
        data = plan_data()
        data["restore_deadline_epoch"] = now + 60
        data["expires_epoch"] = now + 600
        raw = json.dumps(data, sort_keys=True).encode()
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            files = PlanFiles(directory, raw)
            result = subprocess.run(
                (
                    sys.executable,
                    str(SCRIPT),
                    "validate",
                    "--plan",
                    str(files.plan),
                    "--signature",
                    str(files.signature),
                    "--key",
                    str(files.key),
                    "--expected-nonce",
                    NONCE,
                    "--expected-deadline",
                    str(now + 60),
                ),
                check=False,
                capture_output=True,
                text=True,
            )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("VALID plan_sha256=", result.stdout)
        self.assertTrue(result.stdout.rstrip().endswith("No commands were executed."))


class RestorationTests(unittest.TestCase):
    def create_restorer(self, plan=None, runner=None, audit=None, clock=None, wall_clock=None):
        plan = plan or make_plan()
        runner = runner or FakeRunner()
        audit = audit or RecordingAudit()
        clock = clock or FakeClock()
        restorer = restore.Restorer(
            plan,
            audit,
            runner=runner,
            monotonic=clock.monotonic,
            sleeper=clock.sleep,
            runtime_validator=lambda uid, username: runtime_prefix(uid, username),
            username_lookup=lambda uid: "modeluser",
            wall_clock=wall_clock or clock.wall,
        )
        return restorer, runner, audit

    def test_all_components_restore_in_required_order(self):
        restorer, runner, audit = self.create_restorer()
        restorer.restore()
        mutations = []
        for call in runner.calls:
            if call == (restore.SWAPON, restore.SWAP_PATH):
                mutations.append("swapfile")
            elif call[:2] == (restore.DOCKER, "start"):
                mutations.append("docker_monerod")
            elif restore.SYSTEMCTL in call:
                index = call.index(restore.SYSTEMCTL)
                offset = index + 2 if call[index + 1] == "--user" else index + 1
                operation, unit = call[offset:offset + 2]
                if operation in ("enable", "start"):
                    mutations.append(f"{unit}:{operation}")
        self.assertEqual(
            mutations,
            [
                "swapfile",
                "docker_monerod",
                "p2pool.service:enable",
                "p2pool.service:start",
                "xmrig.service:enable",
                "xmrig.service:start",
                "llama-swap.service:enable",
                "llama-swap.service:start",
                "llama-proxy.service:enable",
                "llama-proxy.service:start",
            ],
        )
        self.assertEqual(audit.records[-1]["status"], "complete")
        self.assertEqual(
            [event["component"] for event in audit.records[-1]["events"]],
            ["swapfile", "docker_monerod", "p2pool", "xmrig", "llama_swap", "proxy"],
        )

    def test_already_restored_state_is_idempotent(self):
        runner = FakeRunner()
        runner.swap_active = True
        runner.docker_running = True
        runner.docker_health = ["healthy"]
        for state in runner.units.values():
            state["enabled"] = "enabled"
            state["active"] = "active"
        restorer, runner, audit = self.create_restorer(runner=runner)
        restorer.restore()
        mutating = [
            call
            for call in runner.calls
            if call == (restore.SWAPON, restore.SWAP_PATH)
            or call[:2] == (restore.DOCKER, "start")
            or (
                restore.SYSTEMCTL in call
                and (
                    call[call.index(restore.SYSTEMCTL) + 1] in ("enable", "start")
                    or call[call.index(restore.SYSTEMCTL) + 2] in ("enable", "start")
                )
            )
        ]
        self.assertEqual(mutating, [])
        self.assertEqual(audit.records[-1]["status"], "complete")

    def test_false_baseline_never_stops_or_disables(self):
        data = plan_data()
        data["components"]["swapfile"]["enabled"] = False
        data["components"]["docker_monerod"]["running"] = False
        for component in ("p2pool", "xmrig", "llama_swap", "proxy"):
            data["components"][component] = {"active": False, "enabled": False}
        runner = FakeRunner()
        runner.swap_active = True
        runner.docker_running = True
        for state in runner.units.values():
            state["enabled"] = "enabled"
            state["active"] = "active"
        restorer, runner, audit = self.create_restorer(plan=make_plan(data), runner=runner)
        restorer.restore()
        self.assertEqual(runner.calls, [])
        self.assertEqual(audit.records[-1]["status"], "complete")

    def test_user_systemctl_uses_explicit_uid_runtime(self):
        restorer, runner, _ = self.create_restorer()
        restorer.restore()
        user_calls = [
            call
            for call in runner.calls
            if restore.SYSTEMCTL in call and call.index(restore.SYSTEMCTL) > 0
        ]
        prefix = runtime_prefix(1000, "modeluser")
        self.assertTrue(user_calls)
        self.assertTrue(
            all(
                call[:len(prefix)] == prefix
                and call[len(prefix):len(prefix) + 2] == (restore.SYSTEMCTL, "--user")
                for call in user_calls
            )
        )

    def test_linked_and_runtime_enabled_states_are_not_persistent(self):
        for current_state in ("linked", "linked-runtime", "enabled-runtime"):
            with self.subTest(current_state=current_state):
                runner = FakeRunner()
                runner.units["p2pool.service"]["enabled"] = current_state
                restorer, runner, _ = self.create_restorer(runner=runner)
                restorer.restore()
                self.assertIn(
                    (restore.SYSTEMCTL, "enable", "p2pool.service"),
                    runner.calls,
                )
                self.assertEqual(runner.units["p2pool.service"]["enabled"], "enabled")

    def test_linked_state_that_stays_linked_fails_closed(self):
        class LinkedRunner(FakeRunner):
            def run(self, args, allowed_returncodes=(0,), timeout=30):
                if tuple(args) == (restore.SYSTEMCTL, "enable", "p2pool.service"):
                    self.calls.append(tuple(args))
                    self.timeouts.append(timeout)
                    return ""
                return super().run(args, allowed_returncodes, timeout)

        runner = LinkedRunner()
        runner.units["p2pool.service"]["enabled"] = "linked"
        restorer, _, audit = self.create_restorer(runner=runner)
        with self.assertRaisesRegex(restore.RestoreError, "persistently enabled"):
            restorer.restore()
        self.assertEqual(audit.records[-1]["status"], "failed")

    def test_command_failure_stops_later_components_and_is_audited(self):
        runner = FakeRunner()
        runner.fail_on = (restore.SYSTEMCTL, "start", "xmrig.service")
        restorer, runner, audit = self.create_restorer(runner=runner)
        with self.assertRaisesRegex(restore.RestoreError, "command failure"):
            restorer.restore()
        self.assertEqual(audit.records[-1]["status"], "failed")
        self.assertFalse(
            any(
                restore.SYSTEMCTL in call
                and call[-1] in ("llama-swap.service", "llama-proxy.service")
                for call in runner.calls
            )
        )

    def test_docker_health_timeout_stops_following_components(self):
        data = plan_data()
        data["components"]["docker_monerod"]["health_timeout_seconds"] = 3
        runner = FakeRunner()
        runner.docker_health = ["starting"]
        clock = FakeClock()
        restorer, runner, audit = self.create_restorer(
            plan=make_plan(data),
            runner=runner,
            clock=clock,
        )
        with self.assertRaisesRegex(restore.RestoreError, "timed out"):
            restorer.restore()
        self.assertEqual(audit.records[-1]["status"], "failed")
        self.assertFalse(any(restore.SYSTEMCTL in call for call in runner.calls))

    def test_missing_docker_health_command_fails_closed(self):
        runner = FakeRunner()
        runner.docker_health_test = ""
        restorer, runner, audit = self.create_restorer(runner=runner)
        with self.assertRaisesRegex(restore.RestoreError, "no existing Docker health"):
            restorer.restore()
        self.assertEqual(audit.records[-1]["status"], "failed")

    def test_expiry_during_restoration_stops_later_components(self):
        clock = FakeClock()
        runner = AdvancingRunner(
            clock,
            (restore.SWAPON, restore.SWAP_PATH),
            DEADLINE + 601 - NOW,
        )
        restorer, runner, audit = self.create_restorer(
            runner=runner,
            clock=clock,
        )
        with self.assertRaisesRegex(restore.RestoreError, "expired during restoration"):
            restorer.restore()
        self.assertTrue(runner.swap_active)
        self.assertFalse(runner.docker_running)
        self.assertEqual(audit.records[-1]["status"], "failed")

    def test_healthy_result_crossing_expiry_is_rejected(self):
        clock = FakeClock()
        runner = AdvancingRunner(
            clock,
            (
                restore.DOCKER,
                "inspect",
                "--format={{.State.Health.Status}}",
                restore.DOCKER_CONTAINER,
            ),
            DEADLINE + 601 - NOW,
        )
        runner.swap_active = True
        runner.docker_running = True
        runner.docker_health = ["healthy"]
        restorer, _, audit = self.create_restorer(runner=runner, clock=clock)
        with self.assertRaisesRegex(restore.RestoreError, "expired during restoration"):
            restorer.restore()
        self.assertEqual(audit.records[-1]["status"], "failed")
        self.assertFalse(any(event["component"] == "docker_monerod" for event in audit.records[-1]["events"]))

    def test_final_proxy_completion_crossing_expiry_is_rejected(self):
        clock = FakeClock()
        runner = AdvancingRunner(
            clock,
            (restore.SYSTEMCTL, "--user", "start", "llama-proxy.service"),
            DEADLINE + 601 - NOW,
        )
        runner.swap_active = True
        runner.docker_running = True
        runner.docker_health = ["healthy"]
        for unit, state in runner.units.items():
            state["enabled"] = "enabled"
            state["active"] = "active"
        runner.units["llama-proxy.service"]["active"] = "inactive"
        restorer, _, audit = self.create_restorer(runner=runner, clock=clock)
        with self.assertRaisesRegex(restore.RestoreError, "expired during restoration"):
            restorer.restore()
        self.assertEqual(audit.records[-1]["status"], "failed")
        self.assertNotIn("completed_epoch", audit.records[-1])

    def test_command_timeout_is_bounded_by_signed_expiry(self):
        data = plan_data()
        data["restore_deadline_epoch"] = NOW - 10
        data["expires_epoch"] = NOW + 5
        clock = FakeClock()
        runner = FakeRunner()
        runner.swap_active = True
        runner.docker_running = True
        runner.docker_health = ["healthy"]
        for state in runner.units.values():
            state["enabled"] = "enabled"
            state["active"] = "active"
        restorer, runner, _ = self.create_restorer(
            plan=make_plan(data),
            runner=runner,
            clock=clock,
        )
        restorer.restore()
        self.assertTrue(runner.timeouts)
        self.assertTrue(all(0 < timeout <= 5 for timeout in runner.timeouts))

    def test_initial_audit_failure_prevents_commands(self):
        audit = RecordingAudit(fail_at=0)
        restorer, runner, _ = self.create_restorer(audit=audit)
        with self.assertRaisesRegex(restore.RestoreError, "audit failure"):
            restorer.restore()
        self.assertEqual(runner.calls, [])

    def test_audit_failure_after_action_stops_later_actions(self):
        audit = RecordingAudit(fail_at=1)
        restorer, runner, _ = self.create_restorer(audit=audit)
        with self.assertRaisesRegex(restore.RestoreError, "audit"):
            restorer.restore()
        self.assertTrue(runner.swap_active)
        self.assertFalse(runner.docker_running)

    def test_unsupported_service_state_fails_closed(self):
        runner = FakeRunner()
        runner.units["p2pool.service"]["enabled"] = "masked"
        restorer, runner, audit = self.create_restorer(runner=runner)
        with self.assertRaisesRegex(restore.RestoreError, "cannot safely enable"):
            restorer.restore()
        self.assertEqual(audit.records[-1]["status"], "failed")


class AuditWriterTests(unittest.TestCase):
    def test_atomic_audit_write_and_rewrite(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            Path(directory).chmod(0o700)
            with restore.AuditWriter(directory, NONCE, expected_uid=os.geteuid()) as audit:
                audit.write({"status": "running"})
                audit.write({"status": "complete"})
            record = Path(directory) / f"restore-{NONCE}.json"
            self.assertEqual(json.loads(record.read_text()), {"status": "complete"})
            self.assertEqual(record.stat().st_mode & 0o777, 0o600)
            self.assertEqual(list(Path(directory).glob("*.tmp")), [])

    def test_audit_directory_and_record_symlinks_are_rejected(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as root:
            real = Path(root) / "real"
            real.mkdir(mode=0o700)
            link = Path(root) / "audit"
            link.symlink_to(real, target_is_directory=True)
            with self.assertRaisesRegex(restore.RestoreError, "cannot inspect|unsafe directory"):
                with restore.AuditWriter(link, NONCE, expected_uid=os.geteuid()):
                    pass
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            directory = Path(directory)
            directory.chmod(0o700)
            target = directory / "target"
            target.write_text("do not replace")
            record = directory / f"restore-{NONCE}.json"
            record.symlink_to(target)
            with self.assertRaisesRegex(restore.RestoreError, "not a regular file"):
                with restore.AuditWriter(directory, NONCE, expected_uid=os.geteuid()):
                    pass
            self.assertEqual(target.read_text(), "do not replace")

    def test_audit_lock_symlink_is_rejected(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            directory = Path(directory)
            directory.chmod(0o700)
            target = directory / "target"
            target.write_text("do not alter")
            (directory / ".lock").symlink_to(target)
            with self.assertRaisesRegex(restore.RestoreError, "initialize audit writer"):
                with restore.AuditWriter(directory, NONCE, expected_uid=os.geteuid()):
                    pass
            self.assertEqual(target.read_text(), "do not alter")

    def test_unsafe_existing_audit_mode_is_rejected(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            directory = Path(directory)
            directory.chmod(0o700)
            record = directory / f"restore-{NONCE}.json"
            record.write_text("{}")
            record.chmod(0o644)
            with self.assertRaisesRegex(restore.RestoreError, "unsafe ownership or mode"):
                with restore.AuditWriter(directory, NONCE, expected_uid=os.geteuid()):
                    pass

    def test_preflight_plan_failure_is_recorded(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            directory = Path(directory)
            directory.chmod(0o700)
            restore.write_preflight_audit(
                directory,
                NONCE,
                DEADLINE,
                restore.RestoreError("missing plan"),
                expected_uid=os.geteuid(),
            )
            record = json.loads((directory / "restore-preflight.json").read_text())
            self.assertEqual(record["status"], "failed")
            self.assertEqual(record["phase"], "plan-validation")
            self.assertEqual(record["expected_nonce"], NONCE)
            self.assertEqual(record["error"], "missing plan")


class TemplateTests(unittest.TestCase):
    def test_readme_denies_model_execution_authorization(self):
        readme = SCRIPT.with_name("README.md").read_text()
        self.assertIn(
            "Arming this restoration timer is not authorization to load or execute a model.",
            readme,
        )

    def test_service_is_inert_until_timer_invocation(self):
        service = SCRIPT.with_name("strix-host-restore.service.in").read_text()
        self.assertNotIn("Wants=docker.service", service)
        self.assertNotIn("ConditionPathExists=", service)
        self.assertNotIn("Restart=", service)
        self.assertNotIn("[Install]", service)
        self.assertIn("User=root", service)
        self.assertIn("--expected-nonce @EXPECTED_NONCE@", service)
        self.assertIn("--expected-deadline @EXPECTED_DEADLINE_EPOCH@", service)

    def test_timer_has_exact_persistent_deadline_template(self):
        timer = SCRIPT.with_name("strix-host-restore.timer.in").read_text()
        self.assertIn("OnCalendar=@@EXPECTED_DEADLINE_EPOCH@", timer)
        self.assertIn("AccuracySec=1s", timer)
        self.assertIn("RandomizedDelaySec=0", timer)
        self.assertIn("Persistent=true", timer)
        self.assertIn("Unit=strix-host-restore.service", timer)


if __name__ == "__main__":
    unittest.main()
