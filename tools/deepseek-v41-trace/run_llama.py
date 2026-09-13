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


def main() -> int:
    parser = argparse.ArgumentParser(description="Fail-closed launcher for llama.cpp DeepSeek V4.1 traces")
    parser.add_argument("--exporter", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--prompt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--watchdog-pid-file", type=Path, required=True)
    parser.add_argument("--busy-pattern", action="append", default=["ds4-v41", "DeepSeek-V4.1"])
    parser.add_argument("--context", type=int, default=32768)
    parser.add_argument("--decode-steps", type=int, default=8)
    parser.add_argument("--batch", type=int, default=2048)
    parser.add_argument("--ubatch", type=int, default=512)
    parser.add_argument("--expert-cache-slots", type=int, required=True)
    parser.add_argument("--expert-cache-mib", type=int, required=True)
    parser.add_argument("--gpu-layers", type=int, default=99)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()

    try:
        audit = run_preflight(
            model=args.model,
            prompt=args.prompt,
            output=args.output,
            watchdog_pid_file=args.watchdog_pid_file,
            busy_patterns=args.busy_pattern,
        )
        audit["runtime"] = "llama.cpp"
        audit["config"] = {
            "context": args.context,
            "decode_steps": args.decode_steps,
            "batch": args.batch,
            "ubatch": args.ubatch,
            "expert_cache_slots": args.expert_cache_slots,
            "expert_cache_mib": args.expert_cache_mib,
            "gpu_layers": args.gpu_layers,
        }
        if args.preflight_only:
            print(json.dumps(audit, sort_keys=True, separators=(",", ":")))
            return 0

        exporter = resolved(args.exporter)
        if not exporter.is_file() or not os.access(exporter, os.X_OK):
            raise PreflightError(f"trace exporter is not executable: {exporter}")
        exporter_sha256 = sha256_file(exporter)
        output = resolved(args.output)
        if output.exists() and any(output.iterdir()):
            raise PreflightError(f"trace output directory is not empty: {output}")
        audits = write_audits(Path(str(output) + ".audit"), audit)
        environment = os.environ.copy()
        environment["DSV41_TRACE_MEMORY_AUDIT"] = audits["memory"]
        environment["DSV41_TRACE_SWAP_AUDIT"] = audits["swap"]
        environment["DSV41_TRACE_WATCHDOG_AUDIT"] = audits["watchdog"]
        command = [
            str(exporter),
            "-m", str(resolved(args.model)),
            "-f", str(resolved(args.prompt)),
            "-o", str(output),
            "-c", str(args.context),
            "-n", str(args.decode_steps),
            "-b", str(args.batch),
            "-ub", str(args.ubatch),
            "-ngl", str(args.gpu_layers),
            "-fa", "on",
            "-ctk", "f16",
            "-ctv", "f16",
            "--expert-cache-slots", str(args.expert_cache_slots),
            "--expert-cache-mib", str(args.expert_cache_mib),
        ]
        print("exec:", shlex.join(command), file=sys.stderr)
        result = subprocess.run(command, env=environment, check=False)
        if result.returncode != 0:
            return result.returncode
        bundle = TraceBundle(output)
        if bundle.manifest.get("runtime") != "llama.cpp":
            raise PreflightError("llama exporter wrote a non-llama.cpp trace")
        if bundle.manifest.get("build", {}).get("sha256") != exporter_sha256:
            raise PreflightError("llama trace build SHA-256 does not match the executed exporter")
        return 0
    except PreflightError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
