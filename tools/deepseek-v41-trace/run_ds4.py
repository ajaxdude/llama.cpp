#!/usr/bin/env python3

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

from preflight import PreflightError, resolved, run_preflight, write_audits
from trace_format import TraceBundle, sha256_file

DS4_REVISION = "bd66c402070042bf0a79ad6ece8242de4c93680c"


def read_revision(checkout: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"],
        text=True,
    ).strip()


def preflight(args: argparse.Namespace) -> dict[str, object]:
    checkout = resolved(args.checkout)
    revision = read_revision(checkout)
    if revision != DS4_REVISION:
        raise PreflightError(f"ds4 revision mismatch: expected {DS4_REVISION}, found {revision}")
    result = run_preflight(
        model=args.model,
        prompt=args.prompt,
        output=args.output,
        watchdog_pid_file=args.watchdog_pid_file,
        busy_patterns=args.busy_pattern,
    )
    result.update({
        "runtime": "ds4",
        "ds4_revision": revision,
        "checkout": str(checkout),
        "config": {
            "context": args.context,
            "decode_steps": args.decode_steps,
            "prefill_chunk": args.prefill_chunk,
        },
    })
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Fail-closed launcher for the pinned ds4 trace exporter")
    parser.add_argument("--checkout", type=Path, default=Path("/home/papa/src/ds4-v41"))
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--prompt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--watchdog-pid-file", type=Path, required=True)
    parser.add_argument("--busy-pattern", action="append", default=["ds4-v41", "DeepSeek-V4.1"])
    parser.add_argument("--exporter", type=Path, required=True)
    parser.add_argument("--exporter-sha256", required=True)
    parser.add_argument("--context", type=int, default=32768)
    parser.add_argument("--decode-steps", type=int, default=8)
    parser.add_argument("--prefill-chunk", type=int, default=512)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()

    try:
        audit = preflight(args)
        if args.preflight_only:
            print(json.dumps(audit, sort_keys=True, separators=(",", ":")))
            return 0

        exporter = resolved(args.exporter)
        if not exporter.is_file() or not os.access(exporter, os.X_OK):
            raise PreflightError(f"trace exporter is not executable: {exporter}")
        exporter_sha256 = sha256_file(exporter)
        if exporter_sha256 != args.exporter_sha256:
            raise PreflightError(
                f"trace exporter SHA-256 mismatch: expected {args.exporter_sha256}, found {exporter_sha256}")
        audit["exporter"] = {"path": str(exporter), "sha256": exporter_sha256}
        output = resolved(args.output)
        if output.exists() and any(output.iterdir()):
            raise PreflightError(f"trace output directory is not empty: {output}")
        audits = write_audits(Path(str(output) + ".audit"), audit)
        command = [
            str(exporter),
            "--model", str(resolved(args.model)),
            "--prompt-file", str(resolved(args.prompt)),
            "--output", str(output),
            "--context", str(args.context),
            "--decode-steps", str(args.decode_steps),
            "--prefill-chunk", str(args.prefill_chunk),
            "--memory-audit", audits["memory"],
            "--swap-audit", audits["swap"],
            "--watchdog-audit", audits["watchdog"],
        ]
        print("exec:", shlex.join(command), file=sys.stderr)
        result = subprocess.run(command, cwd=resolved(args.checkout), check=False)
        if result.returncode != 0:
            return result.returncode
        bundle = TraceBundle(output)
        if bundle.manifest.get("runtime") != "ds4":
            raise PreflightError("ds4 exporter wrote a non-ds4 trace")
        if bundle.manifest.get("revision") != DS4_REVISION:
            raise PreflightError(
                f"ds4 trace revision mismatch: expected {DS4_REVISION}, found {bundle.manifest.get('revision')}")
        return 0
    except PreflightError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
