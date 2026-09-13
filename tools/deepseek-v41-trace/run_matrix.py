#!/usr/bin/env python3

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

from preflight import PreflightError, require_nvme_path, resolved
from trace_format import TraceBundle, report

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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the DeepSeek V4.1 cross-runtime corpus matrix")
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--watchdog-pid-file", type=Path, required=True)
    parser.add_argument("--llama-runner", type=Path, required=True)
    parser.add_argument("--llama-exporter", type=Path, required=True)
    parser.add_argument("--ds4-runner", type=Path, required=True)
    parser.add_argument("--ds4-exporter", type=Path, required=True)
    parser.add_argument("--ds4-exporter-sha256", required=True)
    parser.add_argument("--ds4-checkout", type=Path, default=Path("/home/papa/src/ds4-v41"))
    parser.add_argument("--contexts", type=int, nargs="+", default=[32768])
    parser.add_argument("--ubatches", type=int, nargs="+", default=[512])
    parser.add_argument("--decode-steps", type=int, default=8)
    parser.add_argument("--batch", type=int, default=2048)
    parser.add_argument("--expert-cache-slots", type=int, required=True)
    parser.add_argument("--expert-cache-mib", type=int, required=True)
    args = parser.parse_args()

    try:
        repo = resolved(args.repo)
        output = require_nvme_path(args.output, "matrix output")
        model = require_nvme_path(args.model, "model")
        if output.exists() and any(output.iterdir()):
            raise PreflightError(f"matrix output directory is not empty: {output}")
        inputs = output / "inputs"
        inputs.mkdir(parents=True, exist_ok=True)
        corpus_records = []
        for name in CORPORA:
            source = repo / "tests" / "corpus" / name
            if not source.is_file():
                raise PreflightError(f"repository corpus is missing: {source}")
            destination = inputs / name
            shutil.copyfile(source, destination)
            corpus_records.append({
                "name": name,
                "source": str(source),
                "path": str(destination),
                "byte_count": destination.stat().st_size,
                "sha256": sha256_file(destination),
            })

        results = []
        for context in args.contexts:
            if context < 32768 or context > 131072:
                raise PreflightError(f"context is outside the supported 32768..131072 matrix: {context}")
            for ubatch in args.ubatches:
                for corpus in corpus_records:
                    stem = Path(corpus["name"]).stem
                    case = f"{stem}-c{context}-ub{ubatch}"
                    llama_output = output / "llama" / case
                    ds4_output = output / "ds4" / case
                    common = [
                        "--model", str(model),
                        "--prompt", corpus["path"],
                        "--watchdog-pid-file", str(resolved(args.watchdog_pid_file)),
                        "--context", str(context),
                        "--decode-steps", str(args.decode_steps),
                    ]
                    run([
                        sys.executable,
                        str(resolved(args.ds4_runner)),
                        "--checkout", str(resolved(args.ds4_checkout)),
                        "--exporter", str(resolved(args.ds4_exporter)),
                        "--exporter-sha256", args.ds4_exporter_sha256,
                        "--output", str(ds4_output),
                        "--prefill-chunk", str(ubatch),
                        *common,
                    ])
                    run([
                        sys.executable,
                        str(resolved(args.llama_runner)),
                        "--exporter", str(resolved(args.llama_exporter)),
                        "--output", str(llama_output),
                        "--batch", str(args.batch),
                        "--ubatch", str(ubatch),
                        "--expert-cache-slots", str(args.expert_cache_slots),
                        "--expert-cache-mib", str(args.expert_cache_mib),
                        *common,
                    ])
                    comparison = report(TraceBundle(ds4_output), TraceBundle(llama_output))
                    result_path = output / "reports" / f"{case}.json"
                    result_path.parent.mkdir(parents=True, exist_ok=True)
                    result_path.write_text(
                        json.dumps(comparison, sort_keys=True, separators=(",", ":")) + "\n",
                        encoding="ascii",
                    )
                    results.append({"case": case, **comparison})
                    if comparison["status"] != "TARGET PASS":
                        raise RuntimeError(f"correctness mismatch in {case}: {comparison['first_divergence']}")

        summary = {
            "status": "TARGET PASS",
            "model": str(model),
            "corpora": corpus_records,
            "contexts": args.contexts,
            "ubatches": args.ubatches,
            "decode_steps": args.decode_steps,
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
