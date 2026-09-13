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
    darwin_storage_attestation,
    resolved,
    run_oracle_preflight,
    seal_audits,
    validate_prompt_provenance,
    verify_sealed_audits,
    write_audits,
)
from trace_format import (
    ADMITTED_UBATCH,
    APPROVED_EXPORTERS,
    APPROVED_PROMPT_BUILDERS,
    APPROVED_TRACE_SIGNERS,
    CORPUS_SHA256,
    DS4_REVISION,
    MODEL_SHA256,
    NO_EXTERNAL_STATE_STORAGE,
    ORACLE_LANE,
    TraceBundle,
    TraceError,
    TraceVerifier,
    approval_binding,
    bind_execution_authorization,
    canonical_json,
    execution_authorization,
    load_executable_approval_policy,
    prompt_builder_approval,
    reject_loader_overrides,
    seal_bundle,
    sha256_bytes,
    sha256_file,
    strict_json_loads,
    tokenizer_policy_sha256,
    validate_signing_identity,
)

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


def validate_accelerator_attestation(
        record: object,
        *,
        expected_device: str = "Metal0") -> dict[str, object]:
    if not isinstance(record, dict):
        raise PreflightError("accelerator attestation is not an object")
    required_keys = {
        "format",
        "version",
        "runtime_kind",
        "platform",
        "backend",
        "backend_device",
        "backend_description",
        "architecture",
        "metal_registry_id",
        "recommended_max_working_set_bytes",
        "unified_memory",
        "source",
    }
    if set(record) != required_keys:
        raise PreflightError("accelerator attestation fields are invalid")
    expected = {
        "format": "dsv41-accelerator-attestation",
        "version": 2,
        "runtime_kind": "apple-metal",
        "platform": "macos",
        "backend": "Metal",
        "backend_device": expected_device,
        "unified_memory": True,
        "source": "metal-device-query",
    }
    for key, value in expected.items():
        if record.get(key) != value:
            raise PreflightError(f"accelerator attestation {key} mismatch")
    if type(record.get("unified_memory")) is not bool:
        raise PreflightError("accelerator attestation unified-memory identity is invalid")
    for key in ("backend_description", "architecture"):
        if not isinstance(record.get(key), str) or not record[key]:
            raise PreflightError(f"accelerator attestation {key} is missing")
    if type(record.get("metal_registry_id")) is not int or record["metal_registry_id"] <= 0:
        raise PreflightError("accelerator attestation Metal registry identity is invalid")
    if type(record.get("recommended_max_working_set_bytes")) is not int or (
            record["recommended_max_working_set_bytes"] <= 0):
        raise PreflightError("accelerator attestation working-set identity is invalid")
    return dict(record)


def query_accelerator_attestation(exporter: Path, device: str) -> dict[str, object]:
    try:
        result = subprocess.run(
            [str(exporter), "--dsv41-attest-device", device],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as error:
        raise PreflightError(f"cannot query selected accelerator: {error}") from error
    if result.returncode != 0:
        detail = result.stderr.strip() or f"exit {result.returncode}"
        raise PreflightError(f"selected accelerator query failed: {detail}")
    try:
        record = strict_json_loads(result.stdout)
    except TraceError as error:
        raise PreflightError(f"selected accelerator query returned invalid JSON: {error}") from error
    return validate_accelerator_attestation(record, expected_device=device)


def runner_attestation(
        *,
        exporter: Path,
        exporter_sha256: str,
        checkout: Path,
        command: list[str]) -> dict[str, object]:
    runner_executable = resolved(Path(sys.executable))
    runner_script = resolved(Path(__file__))
    return {
        "format": "dsv41-runner-ownership",
        "version": 1,
        "runtime_kind": "apple-metal",
        "source": "python-subprocess",
        "runner_pid": os.getpid(),
        "runner_parent_pid": os.getppid(),
        "runner_uid": os.getuid(),
        "runner_executable": str(runner_executable),
        "runner_executable_sha256": sha256_file(runner_executable),
        "runner_script": str(runner_script),
        "runner_script_sha256": sha256_file(runner_script),
        "exporter_path": str(exporter),
        "exporter_sha256": exporter_sha256,
        "checkout_path": str(checkout),
        "checkout_revision": DS4_REVISION,
        "command_sha256": sha256_bytes(canonical_json(command).encode("ascii")),
    }


def bind_oracle_attestation(
        output: Path,
        audit: dict[str, object],
        accelerator: dict[str, object],
        command: list[str]) -> None:
    manifest_path = output / "manifest.json"
    try:
        manifest = strict_json_loads(manifest_path.read_text(encoding="ascii"))
    except (OSError, UnicodeError, TraceError) as error:
        raise PreflightError(f"cannot bind ds4 runtime attestation: {error}") from error
    if manifest.get("accelerator") != accelerator:
        raise PreflightError("ds4 trace accelerator attestation differs from the preflight query")
    storage = audit.get("storage")
    if not isinstance(storage, dict):
        raise PreflightError("ds4 storage attestation is missing")
    paths = {}
    for label, record in storage.items():
        if not isinstance(record, dict) or not isinstance(record.get("resolved_path"), str):
            raise PreflightError(f"ds4 storage attestation is invalid for {label}")
        paths[label] = record["resolved_path"]
    if manifest.get("model", {}).get("path") != paths["model"]:
        raise PreflightError("ds4 trace model path differs from the attested path")
    if manifest.get("prompt", {}).get("path") != paths["prompt"]:
        raise PreflightError("ds4 trace prompt path differs from the attested path")
    if "paths" in manifest and manifest["paths"] != paths:
        raise PreflightError("ds4 trace execution paths differ from preflight")
    manifest["paths"] = paths
    build = manifest.get("build")
    if not isinstance(build, dict) or build.get("path") != paths["exporter"]:
        raise PreflightError("ds4 trace build path differs from the executed exporter")
    runner = audit.get("runner")
    if not isinstance(runner, dict) or build.get("sha256") != runner.get("exporter_sha256"):
        raise PreflightError("ds4 trace build SHA-256 differs from the executed exporter")
    storage_policy = audit.get("storage_policy")
    if storage_policy != NO_EXTERNAL_STATE_STORAGE:
        raise PreflightError("ds4 external cache/state storage policy is invalid")
    if "storage_policy" in manifest and manifest["storage_policy"] != storage_policy:
        raise PreflightError("ds4 trace external cache/state storage policy differs from preflight")
    manifest["storage_policy"] = storage_policy
    host = audit.get("host")
    if not isinstance(host, dict):
        raise PreflightError("ds4 host attestation is missing")
    if "host" in manifest and manifest["host"] != host:
        raise PreflightError("ds4 trace host identity differs from preflight")
    manifest["host"] = host
    manifest["environment"] = {
        "system_info": f"macOS {host['os_version']} arm64 {host['hardware_model']}",
        "command": shlex.join(command),
    }
    config = manifest.get("config")
    if not isinstance(config, dict):
        raise PreflightError("ds4 trace config is invalid")
    for key, value in (
            ("device_backend", "Metal"),
            ("device_registry_id", accelerator["metal_registry_id"])):
        if key in config and config[key] != value:
            raise PreflightError(f"ds4 trace {key} differs from preflight")
        config[key] = value
    temp = manifest_path.with_suffix(".tmp")
    temp.write_text(json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n", encoding="ascii")
    os.replace(temp, manifest_path)


def preflight(
        args: argparse.Namespace,
        *,
        accelerator: dict[str, object],
        runner: dict[str, object]) -> dict[str, object]:
    checkout = resolved(args.checkout)
    revision = verify_checkout(checkout)
    if revision != DS4_REVISION:
        raise PreflightError(f"ds4 revision mismatch: expected {DS4_REVISION}, found {revision}")
    result = run_oracle_preflight(
        model=args.model,
        prompt=args.prompt,
        output=args.output,
        repo=args.repo,
        checkout=checkout,
        busy_patterns=args.busy_pattern,
        accelerator=accelerator,
        runner=runner,
    )
    result.update({
        "runtime": "ds4",
        "ds4_revision": revision,
        "checkout": str(checkout),
        "config": {
            "context": args.context,
            "decode_steps": args.decode_steps,
            "prefill_chunk": args.prefill_chunk,
            "device_backend": "Metal",
            "device_registry_id": accelerator["metal_registry_id"],
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
    parser.add_argument("--device", default="Metal0")
    parser.add_argument("--signer-principal", required=True)
    parser.add_argument("--signing-key", type=Path, required=True)
    parser.add_argument("--execution-challenge", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--authorization-issued-unix", type=int, required=True)
    parser.add_argument("--authorization-expires-unix", type=int, required=True)
    parser.add_argument("--prompt-builder-policy-id", required=True)
    parser.add_argument("--approval-policy", type=Path, required=True)
    parser.add_argument("--approval-signature", type=Path, required=True)
    parser.add_argument("--approval-principal", required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()

    try:
        if args.prefill_chunk != ADMITTED_UBATCH:
            raise PreflightError(
                f"DeepSeek V4.1 correctness runs require admitted prefill chunk {ADMITTED_UBATCH}, "
                f"found {args.prefill_chunk}")
        reject_loader_overrides()
        output = resolved(args.output)
        approval_policy = load_executable_approval_policy(
            args.approval_policy,
            args.approval_signature,
            expected_principal=args.approval_principal,
            forbidden_roots=(output,),
        )
        prompt_policy, prompt_policy_sha256 = prompt_builder_approval(
            args.prompt_builder_policy_id,
            policies=approval_policy.prompt_builders,
        )
        harness_repo = resolved(args.repo)
        harness_revision = subprocess.check_output(
            ["git", "-C", str(harness_repo), "rev-parse", "HEAD"],
            stderr=subprocess.STDOUT,
        ).decode("ascii").strip()
        if harness_revision != approval_policy.verifier_revision:
            raise PreflightError("ds4 verifier checkout differs from the external approval policy")
        if args.corpus_sha256 != CORPUS_SHA256[args.corpus_name]:
            raise PreflightError(f"corpus SHA-256 mismatch for {args.corpus_name}")
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
            context=args.context,
            decode_steps=args.decode_steps,
            builder_approval_id=args.prompt_builder_policy_id,
            builder_policy=prompt_policy,
            builder_policy_sha256=prompt_policy_sha256,
            path_resolver=lambda path, label: Path(
                str(darwin_storage_attestation(path, label)["resolved_path"])),
        )
        authorization = execution_authorization(
            lane=ORACLE_LANE,
            challenge=args.execution_challenge,
            run_id=args.run_id,
            issued_unix=args.authorization_issued_unix,
            expires_unix=args.authorization_expires_unix,
            approval_policy_sha256=approval_policy.sha256,
            verifier_revision=approval_policy.verifier_revision,
            tokenizer_policy_sha256_value=tokenizer_policy_sha256(prompt_policy["tokenizer"]),
            approvals={
                "prompt_builder": approval_binding(
                    "prompt_builder",
                    args.prompt_builder_policy_id,
                    prompt_policy_sha256,
                    provenance["record"]["builder_install_trust_sha256"],
                ),
            },
        )
        exporter = resolved(args.exporter)
        if not exporter.is_file() or not os.access(exporter, os.X_OK):
            raise PreflightError(f"trace exporter is not executable: {exporter}")
        exporter_sha256 = sha256_file(exporter)
        if exporter_sha256 != args.exporter_sha256:
            raise PreflightError(
                f"trace exporter SHA-256 mismatch: expected {args.exporter_sha256}, found {exporter_sha256}")
        verify_exporter_approval(exporter_sha256)
        validate_signing_identity(
            args.signing_key,
            args.signer_principal,
            trusted_signers=APPROVED_TRACE_SIGNERS,
            forbidden_root=output,
        )
        command = [
            str(exporter),
            "--model", str(resolved(args.model)),
            "--prompt-file", str(resolved(args.prompt)),
            "--output", str(output),
            "--context", str(args.context),
            "--decode-steps", str(args.decode_steps),
            "--prefill-chunk", str(args.prefill_chunk),
            "--device", args.device,
        ]
        accelerator = query_accelerator_attestation(exporter, args.device)
        runner = runner_attestation(
            exporter=exporter,
            exporter_sha256=exporter_sha256,
            checkout=resolved(args.checkout),
            command=command,
        )
        if args.preflight_only:
            audit = preflight(args, accelerator=accelerator, runner=runner)
            print(json.dumps(audit, sort_keys=True, separators=(",", ":")))
            return 0

        if output.exists() and any(output.iterdir()):
            raise PreflightError(f"trace output directory is not empty: {output}")
        preflight_audit = preflight(args, accelerator=accelerator, runner=runner)
        preflight_audit["exporter"] = {"path": str(exporter), "sha256": exporter_sha256}
        pre_audits = write_audits(Path(str(output) + ".audit") / "pre", preflight_audit)
        pre_audit_digests = seal_audits(pre_audits)
        print("exec:", shlex.join(command), file=sys.stderr)
        result = subprocess.run(command, cwd=resolved(args.checkout), check=False)
        if result.returncode != 0:
            return result.returncode
        verify_sealed_audits(pre_audits, pre_audit_digests)
        post_accelerator = query_accelerator_attestation(exporter, args.device)
        if post_accelerator != accelerator:
            raise PreflightError("selected accelerator identity changed during trace execution")
        postflight_audit = preflight(args, accelerator=post_accelerator, runner=runner)
        if postflight_audit.get("host") != preflight_audit.get("host"):
            raise PreflightError("ds4 host identity changed during trace execution")
        post_audits = write_audits(Path(str(output) + ".audit") / "post", postflight_audit)
        bind_embedded_audits(output, {"pre": pre_audits, "post": post_audits})
        bind_prompt_provenance(output, provenance)
        bind_oracle_attestation(output, preflight_audit, accelerator, command)
        bind_execution_authorization(output, authorization)
        seal_bundle(
            output,
            private_key=args.signing_key,
            principal=args.signer_principal,
            expected_lane=ORACLE_LANE,
            expected_challenge=args.execution_challenge,
            expected_run_id=args.run_id,
            candidate_exporter_policies={},
            prompt_builder_policies=approval_policy.prompt_builders,
            expected_candidate_exporter_policy_id=None,
            expected_prompt_builder_policy_id=args.prompt_builder_policy_id,
            expected_approval_policy_sha256=approval_policy.sha256,
            expected_verifier_revision=approval_policy.verifier_revision,
            trusted_signers=APPROVED_TRACE_SIGNERS,
        )
        bundle = TraceBundle(
            output,
            verifier=TraceVerifier.production(
                args.signer_principal,
                expected_lane=ORACLE_LANE,
                expected_challenge=args.execution_challenge,
                expected_run_id=args.run_id,
                expected_candidate_exporter_policy_id=None,
                expected_prompt_builder_policy_id=args.prompt_builder_policy_id,
                approval_policy=approval_policy,
                verification_unix=None,
            ),
        )
        if bundle.manifest.get("runtime") != "ds4":
            raise PreflightError("ds4 exporter wrote a non-ds4 trace")
        if bundle.manifest.get("revision") != DS4_REVISION:
            raise PreflightError(
                f"ds4 trace revision mismatch: expected {DS4_REVISION}, found {bundle.manifest.get('revision')}")
        if bundle.manifest.get("build", {}).get("sha256") != exporter_sha256:
            raise PreflightError("ds4 trace build SHA-256 does not match the executed exporter")
        if bundle.manifest.get("model", {}).get("sha256") != MODEL_SHA256:
            raise PreflightError("ds4 trace model SHA-256 does not match the published GGUF")
        if bundle.manifest.get("accelerator") != accelerator:
            raise PreflightError("ds4 trace accelerator attestation differs from the measured Metal device")
        return 0
    except (PreflightError, TraceError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
