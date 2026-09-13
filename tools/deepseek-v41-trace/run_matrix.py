#!/usr/bin/env python3

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from preflight import PreflightError, require_nvme_path, resolved, run_strix_preflight
from trace_format import (
    ADMITTED_BATCH,
    ADMITTED_UBATCH,
    CORPUS_SHA256,
    MODEL_SHA256,
    REQUIRED_EXPERT_CACHE_MIB,
    REQUIRED_EXPERT_SLOTS,
    TraceError,
    sha256_file,
    strict_json_loads,
)

CORPORA = (
    "correctness-prose.txt",
    "correctness-code.txt",
    "correctness-structured.txt",
    "correctness-numeric.txt",
)


def run(command: list[str]) -> None:
    print("exec:", " ".join(command), file=sys.stderr)
    result = subprocess.run(command, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"command failed with status {result.returncode}")


def prepare_prompt(
    *,
    builder: Path,
    model: Path,
    corpus: Path,
    corpus_name: str,
    corpus_sha256: str,
    output: Path,
    target_tokens: int,
) -> dict[str, object]:
    command = [
        str(builder),
        "--model", str(model),
        "--corpus", str(corpus),
        "--output", str(output),
        "--tokens", str(target_tokens),
    ]
    print("exec:", " ".join(command), file=sys.stderr)
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"prompt builder failed: {result.stderr.strip()}")
    try:
        native_record = strict_json_loads(result.stdout)
    except TraceError as error:
        raise RuntimeError(f"prompt builder returned invalid JSON: {error}") from error
    if not isinstance(native_record, dict) or set(native_record) != {
            "target_tokens", "actual_tokens", "byte_count", "add_bos", "temporary_directory"}:
        raise RuntimeError("prompt builder returned an invalid result schema")
    if native_record.get("target_tokens") != target_tokens or native_record.get("actual_tokens") != target_tokens:
        raise RuntimeError("prompt builder did not produce the requested token count")
    if type(native_record.get("byte_count")) is not int or native_record["byte_count"] != output.stat().st_size:
        raise RuntimeError("prompt builder byte count does not match its output")
    if type(native_record.get("add_bos")) is not bool:
        raise RuntimeError("prompt builder add_bos result is invalid")
    temporary_directory = native_record["temporary_directory"]
    expected_temporary_directory = os.environ.get("TMPDIR")
    if not expected_temporary_directory or temporary_directory != str(resolved(Path(expected_temporary_directory))):
        raise RuntimeError("prompt builder did not attest the selected temporary directory")
    record = {
        "format": "dsv41-prompt-provenance",
        "version": 1,
        "corpus_name": corpus_name,
        "corpus_sha256": corpus_sha256,
        "model_sha256": MODEL_SHA256,
        "prompt_sha256": sha256_file(output),
        "prompt_byte_count": output.stat().st_size,
        "builder_sha256": sha256_file(builder),
        "target_tokens": target_tokens,
        "actual_tokens": target_tokens,
    }
    provenance_path = output.with_suffix(output.suffix + ".provenance.json")
    provenance_path.write_text(
        json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="ascii",
    )
    record.update({
        "path": str(output),
        "provenance_path": str(provenance_path),
        "provenance_sha256": sha256_file(provenance_path),
    })
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description="Capture the DeepSeek V4.1 llama.cpp corpus matrix")
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--llama-runner", type=Path, required=True)
    parser.add_argument("--llama-exporter", type=Path, required=True)
    parser.add_argument("--llama-prompt-builder", type=Path, required=True)
    parser.add_argument("--candidate-revision", required=True)
    parser.add_argument("--base-revision", required=True)
    parser.add_argument("--candidate-diff-sha256", required=True)
    parser.add_argument("--llama-only", action="store_true")
    parser.add_argument("--contexts", type=int, nargs="+", default=[32768])
    parser.add_argument("--ubatches", type=int, nargs="+", default=[ADMITTED_UBATCH])
    parser.add_argument("--decode-steps", type=int, default=8)
    parser.add_argument("--batch", type=int, default=ADMITTED_BATCH)
    parser.add_argument("--device", default="ROCm0")
    parser.add_argument("--expert-cache-slots", type=int, default=REQUIRED_EXPERT_SLOTS)
    parser.add_argument("--expert-cache-mib", type=int, default=REQUIRED_EXPERT_CACHE_MIB)
    parser.add_argument("--busy-pattern", action="append", default=["ds4-v41", "DeepSeek-V4.1"])
    args = parser.parse_args()

    try:
        if args.ubatches != [ADMITTED_UBATCH]:
            raise PreflightError(
                f"DeepSeek V4.1 correctness matrix requires admitted ubatch [{ADMITTED_UBATCH}]")
        if args.batch != ADMITTED_BATCH:
            raise PreflightError(f"DeepSeek V4.1 correctness matrix requires batch {ADMITTED_BATCH}")
        if args.device != "ROCm0":
            raise PreflightError("DeepSeek V4.1 correctness matrix requires device ROCm0")
        if args.expert_cache_slots != REQUIRED_EXPERT_SLOTS:
            raise PreflightError(
                f"DeepSeek V4.1 correctness matrix requires {REQUIRED_EXPERT_SLOTS} expert cache slots")
        if args.expert_cache_mib != REQUIRED_EXPERT_CACHE_MIB:
            raise PreflightError(
                f"DeepSeek V4.1 correctness matrix requires {REQUIRED_EXPERT_CACHE_MIB} MiB expert cache")
        if not args.llama_only:
            raise PreflightError(
                "cross-runtime capture must run on separate Strix and Apple hosts; "
                "use --llama-only here and compare completed bundles with trace_format.py")
        repo = resolved(args.repo)
        output = require_nvme_path(args.output, "matrix output")
        model = require_nvme_path(args.model, "model")
        if not model.is_file():
            raise PreflightError(f"model is not a file: {model}")
        model_sha256 = sha256_file(model)
        if model_sha256 != MODEL_SHA256:
            raise PreflightError(f"published model SHA-256 mismatch: expected {MODEL_SHA256}, found {model_sha256}")
        initial_corpus = require_nvme_path(
            repo / "tests" / "corpus" / CORPORA[0],
            "repository corpus",
        )
        run_strix_preflight(
            model=model,
            prompt=initial_corpus,
            output=output,
            repo=repo,
            busy_patterns=args.busy_pattern,
        )
        prompt_builder = resolved(args.llama_prompt_builder)
        if not prompt_builder.is_file() or not os.access(prompt_builder, os.X_OK):
            raise PreflightError(f"prompt builder is not executable: {prompt_builder}")
        if output.exists() and any(output.iterdir()):
            raise PreflightError(f"matrix output directory is not empty: {output}")
        inputs = output / "inputs"
        sources = inputs / "sources"
        prompts = inputs / "prompts"
        sources.mkdir(parents=True, exist_ok=True)
        prompts.mkdir(parents=True, exist_ok=True)
        corpus_records = []
        for name in CORPORA:
            source = require_nvme_path(repo / "tests" / "corpus" / name, "repository corpus")
            if not source.is_file():
                raise PreflightError(f"repository corpus is missing: {source}")
            destination = sources / name
            shutil.copyfile(source, destination)
            source_sha256 = sha256_file(destination)
            if source_sha256 != CORPUS_SHA256[name]:
                raise PreflightError(
                    f"repository corpus SHA-256 mismatch for {name}: expected {CORPUS_SHA256[name]}, found {source_sha256}")
            corpus_records.append({
                "name": name,
                "source": str(source),
                "path": str(destination),
                "byte_count": destination.stat().st_size,
                "sha256": source_sha256,
            })

        results = []
        prompt_records = []
        for context in args.contexts:
            if context < 32768 or context > 131072:
                raise PreflightError(f"context is outside the supported 32768..131072 matrix: {context}")
            target_tokens = context - args.decode_steps
            if target_tokens < 1:
                raise PreflightError("decode steps leave no room for prompt tokens")
            prepared_prompts = {}
            for corpus in corpus_records:
                stem = Path(corpus["name"]).stem
                prompt = prompts / f"{stem}-c{context}.txt"
                run_strix_preflight(
                    model=model,
                    prompt=Path(corpus["path"]),
                    output=prompt,
                    repo=repo,
                    busy_patterns=args.busy_pattern,
                )
                prepared = prepare_prompt(
                    builder=resolved(args.llama_prompt_builder),
                    model=model,
                    corpus=Path(corpus["path"]),
                    corpus_name=corpus["name"],
                    corpus_sha256=corpus["sha256"],
                    output=prompt,
                    target_tokens=target_tokens,
                )
                prepared.update({"corpus": corpus["name"], "context": context})
                prepared_prompts[corpus["name"]] = prepared
                prompt_records.append(prepared)
            for ubatch in args.ubatches:
                for corpus in corpus_records:
                    stem = Path(corpus["name"]).stem
                    case = f"{stem}-c{context}-ub{ubatch}"
                    llama_output = output / "llama" / case
                    prompt = prepared_prompts[corpus["name"]]["path"]
                    provenance = prepared_prompts[corpus["name"]]["provenance_path"]
                    common = [
                        "--model", str(model),
                        "--prompt", prompt,
                        "--prompt-provenance", provenance,
                        "--corpus-name", corpus["name"],
                        "--corpus-sha256", corpus["sha256"],
                        "--context", str(context),
                        "--decode-steps", str(args.decode_steps),
                    ]
                    for pattern in args.busy_pattern:
                        common.extend(["--busy-pattern", pattern])
                    run([
                        sys.executable,
                        str(resolved(args.llama_runner)),
                        "--exporter", str(resolved(args.llama_exporter)),
                        "--repo", str(repo),
                        "--candidate-revision", args.candidate_revision,
                        "--base-revision", args.base_revision,
                        "--candidate-diff-sha256", args.candidate_diff_sha256,
                        "--output", str(llama_output),
                        "--batch", str(args.batch),
                        "--ubatch", str(ubatch),
                        "--device", args.device,
                        "--expert-cache-slots", str(args.expert_cache_slots),
                        "--expert-cache-mib", str(args.expert_cache_mib),
                        *common,
                    ])
                    results.append({
                        "case": case,
                        "status": "BRINGUP TRACE CAPTURED",
                        "cross_runtime_status": "INCOMPLETE",
                        "trace": str(llama_output),
                    })

        summary = {
            "status": "BRINGUP TRACE CAPTURED",
            "mode": "llama-only",
            "cross_runtime_status": "INCOMPLETE",
            "model": str(model),
            "model_sha256": model_sha256,
            "candidate_revision": args.candidate_revision,
            "base_revision": args.base_revision,
            "candidate_diff_sha256": args.candidate_diff_sha256,
            "corpora": corpus_records,
            "prompts": prompt_records,
            "contexts": args.contexts,
            "ubatches": args.ubatches,
            "decode_steps": args.decode_steps,
            "target_prompt_tokens": {
                str(context): context - args.decode_steps for context in args.contexts
            },
            "cases": results,
        }
        (output / "summary.json").write_text(
            json.dumps(summary, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="ascii",
        )
        return 0
    except (PreflightError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
