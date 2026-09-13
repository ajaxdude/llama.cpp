#!/usr/bin/env python3

import argparse
import hashlib
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
    safe_trace_path,
    seal_audits,
    validate_prompt_provenance,
    verify_sealed_audits,
    write_audits,
)
from trace_format import (
    ADMITTED_BATCH,
    ADMITTED_UBATCH,
    CORPUS_SHA256,
    MODEL_SHA256,
    REPOSITORY,
    REQUIRED_EXPERT_CACHE_BYTES,
    REQUIRED_EXPERT_CACHE_MIB,
    REQUIRED_EXPERT_SLOTS,
    TraceBundle,
    TraceError,
    sha256_file,
)


def git_output(repo: Path, *args: str) -> bytes:
    try:
        return subprocess.check_output(["git", "-C", str(repo), *args], stderr=subprocess.STDOUT)
    except (OSError, subprocess.CalledProcessError) as error:
        raise PreflightError(f"git {' '.join(args)} failed: {error}") from error


def candidate_attestation(args: argparse.Namespace, exporter_sha256: str) -> dict[str, str]:
    repo = resolved(args.repo)
    revision = git_output(repo, "rev-parse", "HEAD").decode("ascii").strip()
    base_revision = git_output(repo, "rev-parse", args.base_revision).decode("ascii").strip()
    if revision != args.candidate_revision:
        raise PreflightError(
            f"candidate revision mismatch: expected {args.candidate_revision}, found {revision}")
    if base_revision != args.base_revision:
        raise PreflightError(
            f"base revision mismatch: expected {args.base_revision}, found {base_revision}")
    try:
        subprocess.run(
            ["git", "-C", str(repo), "merge-base", "--is-ancestor", base_revision, revision],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        subprocess.run(
            ["git", "-C", str(repo), "diff", "--quiet"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        subprocess.run(
            ["git", "-C", str(repo), "diff", "--cached", "--quiet"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise PreflightError(f"candidate repository is not cleanly based on {base_revision}: {error}") from error
    status = git_output(repo, "status", "--porcelain", "--untracked-files=all")
    if status:
        raise PreflightError("candidate repository has tracked or untracked changes")
    diff = git_output(repo, "diff", "--binary", "--no-ext-diff", base_revision, revision, "--")
    diff_sha256 = hashlib.sha256(diff).hexdigest()
    if diff_sha256 != args.candidate_diff_sha256:
        raise PreflightError(
            f"candidate diff SHA-256 mismatch: expected {args.candidate_diff_sha256}, found {diff_sha256}")
    return {
        "repository": REPOSITORY,
        "revision": revision,
        "base_revision": base_revision,
        "diff_sha256": diff_sha256,
        "executable_sha256": exporter_sha256,
    }


def bind_candidate_attestation(
        output: Path,
        attestation: dict[str, str]) -> None:
    manifest_path = safe_trace_path(output, "manifest.json")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise PreflightError(f"cannot bind candidate attestation: {error}") from error
    manifest["candidate"] = attestation
    temp = manifest_path.with_suffix(".tmp")
    temp.write_text(json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n", encoding="ascii")
    os.replace(temp, manifest_path)


def build_command(args: argparse.Namespace, exporter: Path, output: Path) -> list[str]:
    return [
        str(exporter),
        "-m", str(resolved(args.model)),
        "-bf", str(resolved(args.prompt)),
        "-o", str(output),
        "-c", str(args.context),
        "-n", str(args.decode_steps),
        "-b", str(args.batch),
        "-ub", str(args.ubatch),
        "--device", args.device,
        "-ngl", str(args.gpu_layers),
        "-fa", "on",
        "-ctk", "f16",
        "-ctv", "f16",
        "--load-mode", "none",
        "--expert-cache-slots", str(args.expert_cache_slots),
        "--expert-cache-mib", str(args.expert_cache_mib),
    ]


def validate_runtime_config(args: argparse.Namespace) -> None:
    if args.batch != ADMITTED_BATCH:
        raise PreflightError(
            f"DeepSeek V4.1 correctness runs require batch {ADMITTED_BATCH}, found {args.batch}")
    if args.ubatch != ADMITTED_UBATCH:
        raise PreflightError(
            f"DeepSeek V4.1 correctness runs require admitted ubatch {ADMITTED_UBATCH}, found {args.ubatch}")
    if args.expert_cache_slots != REQUIRED_EXPERT_SLOTS:
        raise PreflightError(
            f"DeepSeek V4.1 correctness runs require {REQUIRED_EXPERT_SLOTS} expert cache slots, "
            f"found {args.expert_cache_slots}")
    if args.expert_cache_mib != REQUIRED_EXPERT_CACHE_MIB:
        raise PreflightError(
            f"DeepSeek V4.1 correctness runs require {REQUIRED_EXPERT_CACHE_BYTES} expert cache bytes "
            f"({REQUIRED_EXPERT_CACHE_MIB} MiB), found {args.expert_cache_mib} MiB")
    if args.device != "ROCm0":
        raise PreflightError(f"DeepSeek V4.1 correctness runs require device ROCm0, found {args.device}")
    if args.gpu_layers != 99:
        raise PreflightError(f"DeepSeek V4.1 correctness runs require 99 GPU layers, found {args.gpu_layers}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Fail-closed launcher for llama.cpp DeepSeek V4.1 traces")
    parser.add_argument("--exporter", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--candidate-revision", required=True)
    parser.add_argument("--base-revision", required=True)
    parser.add_argument("--candidate-diff-sha256", required=True)
    parser.add_argument("--corpus-name", choices=sorted(CORPUS_SHA256), required=True)
    parser.add_argument("--corpus-sha256", required=True)
    parser.add_argument("--prompt-provenance", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--prompt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--busy-pattern", action="append", default=["ds4-v41", "DeepSeek-V4.1"])
    parser.add_argument("--context", type=int, default=32768)
    parser.add_argument("--decode-steps", type=int, default=8)
    parser.add_argument("--batch", type=int, default=ADMITTED_BATCH)
    parser.add_argument("--ubatch", type=int, default=ADMITTED_UBATCH)
    parser.add_argument("--device", default="ROCm0")
    parser.add_argument("--expert-cache-slots", type=int, default=REQUIRED_EXPERT_SLOTS)
    parser.add_argument("--expert-cache-mib", type=int, default=REQUIRED_EXPERT_CACHE_MIB)
    parser.add_argument("--gpu-layers", type=int, default=99)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()

    try:
        validate_runtime_config(args)
        if args.corpus_sha256 != CORPUS_SHA256[args.corpus_name]:
            raise PreflightError(f"corpus SHA-256 mismatch for {args.corpus_name}")
        if args.preflight_only:
            audit = run_preflight(
                model=args.model,
                prompt=args.prompt,
                output=args.output,
                repo=args.repo,
                busy_patterns=args.busy_pattern,
            )
            print(json.dumps(audit, sort_keys=True, separators=(",", ":")))
            return 0

        exporter = resolved(args.exporter)
        if not exporter.is_file() or not os.access(exporter, os.X_OK):
            raise PreflightError(f"trace exporter is not executable: {exporter}")
        exporter_sha256 = sha256_file(exporter)
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
        attestation = candidate_attestation(args, exporter_sha256)
        output = resolved(args.output)
        if output.exists() and any(output.iterdir()):
            raise PreflightError(f"trace output directory is not empty: {output}")
        preflight_audit = run_preflight(
            model=args.model,
            prompt=args.prompt,
            output=args.output,
            repo=args.repo,
            busy_patterns=args.busy_pattern,
        )
        preflight_audit["runtime"] = "llama.cpp"
        preflight_audit["config"] = {
            "context": args.context,
            "decode_steps": args.decode_steps,
            "batch": args.batch,
            "ubatch": args.ubatch,
            "device": args.device,
            "expert_cache_slots": args.expert_cache_slots,
            "expert_cache_mib": args.expert_cache_mib,
            "gpu_layers": args.gpu_layers,
        }
        pre_audits = write_audits(Path(str(output) + ".audit") / "pre", preflight_audit)
        pre_audit_digests = seal_audits(pre_audits)
        environment = os.environ.copy()
        environment["DSV41_TRACE_MEMORY_AUDIT"] = pre_audits["memory"]
        environment["DSV41_TRACE_SWAP_AUDIT"] = pre_audits["swap"]
        environment["DSV41_TRACE_WATCHDOG_AUDIT"] = pre_audits["watchdog"]
        command = build_command(args, exporter, output)
        print("exec:", shlex.join(command), file=sys.stderr)
        result = subprocess.run(command, env=environment, check=False)
        if result.returncode != 0:
            return result.returncode
        verify_sealed_audits(pre_audits, pre_audit_digests)
        postflight_audit = run_preflight(
            model=args.model,
            prompt=args.prompt,
            output=args.output,
            repo=args.repo,
            busy_patterns=args.busy_pattern,
        )
        postflight_audit["runtime"] = "llama.cpp"
        post_audits = write_audits(Path(str(output) + ".audit") / "post", postflight_audit)
        bind_embedded_audits(output, {"pre": pre_audits, "post": post_audits})
        bind_prompt_provenance(output, provenance)
        bind_candidate_attestation(output, attestation)
        bundle = TraceBundle(output)
        if bundle.manifest.get("runtime") != "llama.cpp":
            raise PreflightError("llama exporter wrote a non-llama.cpp trace")
        if bundle.manifest.get("build", {}).get("sha256") != exporter_sha256:
            raise PreflightError("llama trace build SHA-256 does not match the executed exporter")
        if bundle.manifest.get("model", {}).get("sha256") != MODEL_SHA256:
            raise PreflightError("llama trace model SHA-256 does not match the published GGUF")
        return 0
    except (PreflightError, TraceError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
