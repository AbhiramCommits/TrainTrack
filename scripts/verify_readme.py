"""Verify that the README results match docs/results/*.json exactly."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS = PROJECT_ROOT / "docs" / "results"
README = PROJECT_ROOT / "README.md"


def load(name: str) -> dict:
    return json.loads((RESULTS / name).read_text())


def main() -> int:
    text = README.read_text()
    checks: list[tuple[str, str]] = []

    training = load("training_runs.json")
    snapshot = training["dataset_snapshot"]
    checks.append(("snapshot hash", snapshot["content_hash"]))
    checks.append(("snapshot rows", f"{snapshot['row_count']:,}"))
    for run in training["runs"]:
        checks.append((f"run {run['run_id'][:8]}", f"`{run['run_id'][:8]}`"))
        checks.append(("val accuracy", f"{run['val_accuracy'] * 100:.2f}%"))
        checks.append(("test accuracy", f"{run['test_accuracy'] * 100:.2f}%"))
    evaluation = training["eval_on_best"]
    checks.append(("eval accuracy", f"{evaluation['accuracy'] * 100:.2f}%"))
    checks.append(("eval macro f1", f"{evaluation['macro_f1'] * 100:.2f}%"))
    checks.append(("eval p50", f"{evaluation['inference_p50_ms']:.6f} ms"))
    checks.append(("eval p95", f"{evaluation['inference_p95_ms']:.6f} ms"))
    artifacts = training["artifacts"]
    checks.append(("artifact count", str(artifacts["unique_artifacts"])))
    checks.append(("dedupe ratio", f"{artifacts['dedupe_ratio'] * 100:.2f}%"))
    checks.append(
        ("dedupe bytes", f"{artifacts['bytes_saved'] / (1024 * 1024):.2f} MiB")
    )

    k8s = load("k8s_runs.json")
    for run in k8s["runs"]:
        checks.append(("k8s scheduling", f"{run['scheduling_latency_seconds']:.2f}"))
        checks.append(("k8s end-to-end", f"{run['end_to_end_seconds']:.2f}"))
    checks.append(("gpu dry run", f"validated={k8s['gpu_manifest_dry_run']['validated']}"))

    benchmark = load("benchmark.json")
    for config in benchmark["configurations"]:
        checks.append(("benchmark drain", f"{config['drain_throughput_rps']:.2f}"))
        checks.append(("benchmark submit", f"{config['submission_throughput_rps']:.2f}"))
    checks.append(
        ("publish confirm p50", f"{benchmark['broker_publish_confirm_ms']['p50_ms']:.2f} ms")
    )

    reliability = load("reliability.json")
    checks.append(("retry attempts", str(reliability["attempts_observed"])))
    checks.append(("retry backoff", f"{reliability['observed_retry_delay_seconds']:.3f} s"))
    checks.append(("dlq", str(reliability["dlq_messages"])))

    tests = load("test_results.json")
    checks.append(("backend tests", str(tests["backend"]["tests"])))
    checks.append(("coverage", f"{tests['backend']['coverage_percent']:.2f}%"))
    checks.append(("frontend tests", str(tests["frontend"]["tests"])))
    checks.append(("total tests", str(tests["totals"]["tests"])))

    missing = [(label, value) for label, value in checks if value not in text]
    if missing:
        for label, value in missing:
            print(f"MISSING {label}: {value}", file=sys.stderr)
        return 1

    for screenshot in (
        "runs-dashboard.png",
        "run-detail.png",
        "live-logs.png",
        "compare.png",
        "submit-run.png",
        "queue-health.png",
        "kubernetes-runs.png",
        "kubernetes-run-detail.png",
    ):
        if not (PROJECT_ROOT / "docs" / "screenshots" / screenshot).exists():
            print(f"MISSING screenshot: {screenshot}", file=sys.stderr)
            return 1

    print(f"README matches docs/results/*.json ({len(checks)} checks) and all screenshots exist")
    return 0


if __name__ == "__main__":
    sys.exit(main())
