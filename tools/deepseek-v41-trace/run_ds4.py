#!/usr/bin/env python3

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

from preflight import (
    PreflightError,
    bind_embedded_audits,
    bind_prompt_provenance,
    resolved,
    run_preflight,
    seal_audits,
    validate_prompt_provenance,
    verify_sealed_audits,
    write_audits,
)
from trace_format import ADMITTED_UBATCH, CORPUS_SHA256, MODEL_SHA256, TraceBundle, TraceError, sha256_file

DS4_REVISION = "bd66c402070042bf0a79ad6ece8242de4c93680c"
APPROVED_EXPORTERS: dict[str, str] = {}


def git_output(checkout: Path, *args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(checkout), *args],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise PreflightError(f"ds4 git {' '.join(args)} failed: {error}") from error


def verify_checkout(checkout: Path) -> str:
    revision = git_output(checkout, "rev-parse", "HEAD")
    status = git_output(checkout, "status", "--porcelain", "--untracked-files=all")
    if status:
        raise PreflightError("ds4 checkout has tracked or untracked changes")
    return revision


def verify_exporter_approval(exporter_sha256: str) -> None:
    if APPROVED_EXPORTERS.get(exporter_sha256) != DS4_REVISION:
        raise PreflightError(
            "ds4 trace exporter is not approved for the pinned ds4 revision; "
            "publish and review the exporter before cross-runtime execution")


def preflight(args: argparse.Namespace) -> dict[str, object]:
    checkout = resolved(args.checkout)
    revision = verify_checkout(checkout)
    if revision != DS4_REVISION:
        raise PreflightError(f"ds4 revision mismatch: expected {DS4_REVISION}, found {revision}")
    result = run_preflight(
        model=args.model,
        prompt=args.prompt,
        output=args.output,
        repo=args.repo,
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
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--prompt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--busy-pattern", action="append", default=["ds4-v41", "DeepSeek-V4.1"])
    parser.add_argument("--exporter", type=Path, required=True)
    parser.add_argument("--exporter-sha256", required=True)
    parser.add_argument("--corpus-name", choices=sorted(CORPUS_SHA256), required=True)
    parser.add_argument("--corpus-sha256", required=True)
    parser.add_argument("--prompt-provenance", type=Path, required=True)
    parser.add_argument("--context", type=int, default=32768)
    parser.add_argument("--decode-steps", type=int, default=8)
    parser.add_argument("--prefill-chunk", type=int, default=ADMITTED_UBATCH)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()

    try:
        if args.prefill_chunk != ADMITTED_UBATCH:
            raise PreflightError(
                f"DeepSeek V4.1 correctness runs require admitted prefill chunk {ADMITTED_UBATCH}, "
                f"found {args.prefill_chunk}")
        if args.corpus_sha256 != CORPUS_SHA256[args.corpus_name]:
            raise PreflightError(f"corpus SHA-256 mismatch for {args.corpus_name}")
        if args.preflight_only:
            audit = preflight(args)
            print(json.dumps(audit, sort_keys=True, separators=(",", ":")))
            return 0

        exporter = resolved(args.exporter)
        if not exporter.is_file() or not os.access(exporter, os.X_OK):
            raise PreflightError(f"trace exporter is not executable: {exporter}")
        exporter_sha256 = sha256_file(exporter)
        if exporter_sha256 != args.exporter_sha256:
            raise PreflightError(
                f"trace exporter SHA-256 mismatch: expected {args.exporter_sha256}, found {exporter_sha256}")
        verify_exporter_approval(exporter_sha256)
        model_sha256 = sha256_file(resolved(args.model))
        if model_sha256 != MODEL_SHA256:
            raise PreflightError(f"published model SHA-256 mismatch: expected {MODEL_SHA256}, found {model_sha256}")
        provenance = validate_prompt_provenance(
            args.prompt_provenance,
            prompt=args.prompt,
            corpus_name=args.corpus_name,
            corpus_sha256=args.corpus_sha256,
            model_sha256=model_sha256,
            target_tokens=args.context - args.decode_steps,
        )
        output = resolved(args.output)
        if output.exists() and any(output.iterdir()):
            raise PreflightError(f"trace output directory is not empty: {output}")
        preflight_audit = preflight(args)
        preflight_audit["exporter"] = {"path": str(exporter), "sha256": exporter_sha256}
        pre_audits = write_audits(Path(str(output) + ".audit") / "pre", preflight_audit)
        pre_audit_digests = seal_audits(pre_audits)
        command = [
            str(exporter),
            "--model", str(resolved(args.model)),
            "--prompt-file", str(resolved(args.prompt)),
            "--output", str(output),
            "--context", str(args.context),
            "--decode-steps", str(args.decode_steps),
            "--prefill-chunk", str(args.prefill_chunk),
            "--memory-audit", pre_audits["memory"],
            "--swap-audit", pre_audits["swap"],
            "--watchdog-audit", pre_audits["watchdog"],
        ]
        print("exec:", shlex.join(command), file=sys.stderr)
        result = subprocess.run(command, cwd=resolved(args.checkout), check=False)
        if result.returncode != 0:
            return result.returncode
        verify_sealed_audits(pre_audits, pre_audit_digests)
        postflight_audit = preflight(args)
        post_audits = write_audits(Path(str(output) + ".audit") / "post", postflight_audit)
        bind_embedded_audits(output, {"pre": pre_audits, "post": post_audits})
        bind_prompt_provenance(output, provenance)
        bundle = TraceBundle(output)
        if bundle.manifest.get("runtime") != "ds4":
            raise PreflightError("ds4 exporter wrote a non-ds4 trace")
        if bundle.manifest.get("revision") != DS4_REVISION:
            raise PreflightError(
                f"ds4 trace revision mismatch: expected {DS4_REVISION}, found {bundle.manifest.get('revision')}")
        if bundle.manifest.get("build", {}).get("sha256") != exporter_sha256:
            raise PreflightError("ds4 trace build SHA-256 does not match the executed exporter")
        if bundle.manifest.get("model", {}).get("sha256") != MODEL_SHA256:
            raise PreflightError("ds4 trace model SHA-256 does not match the published GGUF")
        return 0
    except (PreflightError, TraceError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
