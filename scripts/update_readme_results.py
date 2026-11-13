"""Render the README RESULTS section from docs/results/*.json.

The README prose is hand-written; everything between the RESULTS markers is
generated here so every figure comes from measured result files.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS = PROJECT_ROOT / "docs" / "results"
README = PROJECT_ROOT / "README.md"
START = "<!-- RESULTS:START -->"
END = "<!-- RESULTS:END -->"


def load(name: str) -> dict[str, Any]:
    return json.loads((RESULTS / name).read_text())


def bytes_mb(value: int) -> str:
    return f"{value / (1024 * 1024):.2f} MiB"


def seconds(value: float | None) -> str:
    return "—" if value is None else f"{value:.2f}"


def percent(value: float | None) -> str:
    return "—" if value is None else f"{value * 100:.2f}%"


def render_training(training: dict[str, Any]) -> list[str]:
    environment = training["environment"]
    snapshot = training["dataset_snapshot"]
    lines: list[str] = []
    lines.append("### 4.1 Hardware and dataset snapshot")
    lines.append("")
    lines.append(
        "All training runs below ran on the local machine described by the registry: "
        f"`{environment['platform']}` ({environment['machine']}), Python "
        f"{environment['python']}, PyTorch {environment['torch']}. "
        f"CUDA available: **{environment['cuda_available']}**; MPS available: "
        f"**{environment['mps_available']}**. The devices actually used were: "
        f"{', '.join(f'`{device}`' for device in environment['devices_used'])}. "
        "No NVIDIA GPU was present. The `nvidia.com/gpu` request path is implemented "
        "and verified by rendered-manifest golden tests plus a Kubernetes server-side "
        "dry run (Section 4.5); no GPU benchmark is claimed."
    )
    lines.append("")
    lines.append(
        "| Snapshot | Content hash | Rows | Seed | Mean | Std |"
    )
    lines.append("|---|---|---|---|---|---|")
    lines.append(
        f"| `{snapshot['version']}` | `{snapshot['content_hash']}` | "
        f"{snapshot['row_count']:,} | {snapshot['seed']} | {snapshot['mean']:.6f} | "
        f"{snapshot['std']:.6f} |"
    )
    lines.append("")
    splits = snapshot.get("splits") or {}
    lines.append("| Split | Rows | Shape |")
    lines.append("|---|---|---|")
    for name in ("train", "val", "test"):
        split = splits.get(name, {})
        lines.append(
            f"| {name} | {int(split.get('rows', 0)):,} | `{split.get('shape')}` |"
        )
    lines.append("")
    lines.append(
        "The snapshot was built with a fixed seed, normalized with the train-split "
        "mean/std, and registered only after the validation gate recomputed every "
        "file digest and the aggregate content hash."
    )
    lines.append("")

    lines.append("### 4.2 Real training runs")
    lines.append("")
    lines.append(
        f"{training['completed_runs']} training runs were submitted through the REST API, "
        "scheduled by the RabbitMQ worker, and actually trained with PyTorch. "
        "Wall clock is the registry's `finished_at - started_at`; train loop is the "
        "job-internal timer, including test evaluation."
    )
    lines.append("")
    lines.append(
        "| Run | Optimizer | LR | Batch | Epochs | Wall clock (s) | Train loop (s) | "
        "Val accuracy | Test accuracy | Device | Requested GPUs |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for run in training["runs"]:
        hyper = run["hyperparams"]
        lines.append(
            f"| `{run['run_id'][:8]}` | {hyper.get('optimizer')} | {hyper.get('lr')} | "
            f"{hyper.get('batch_size')} | {run.get('epochs')} | "
            f"{seconds(run.get('wall_clock_seconds'))} | "
            f"{seconds(run.get('train_seconds'))} | "
            f"{percent(run.get('val_accuracy'))} | {percent(run.get('test_accuracy'))} | "
            f"{run.get('device')} | {run.get('requested_gpus')} |"
        )
    lines.append("")
    best = training["best_run"]
    lines.append(
        f"Best validation run: `{best['run_id']}` with val accuracy "
        f"**{percent(best['val_accuracy'])}** and test accuracy "
        f"**{percent(best['test_accuracy'])}**. Its checkpoint digest is "
        f"`{best['checkpoint_sha']}`."
    )
    lines.append("")

    evaluation = training.get("eval_on_best")
    lines.append("### 4.3 Best model evaluation report")
    lines.append("")
    if evaluation:
        lines.append(
            "A separate eval job loaded the best checkpoint, verified its dataset "
            "snapshot hash, and scored all "
            f"{evaluation['num_test_samples']:,} held-out test images."
        )
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|---|---|")
        lines.append(f"| Test accuracy | **{percent(evaluation['accuracy'])}** |")
        lines.append(f"| Macro precision | {percent(evaluation['macro_precision'])} |")
        lines.append(f"| Macro recall | {percent(evaluation['macro_recall'])} |")
        lines.append(f"| Macro F1 | **{percent(evaluation['macro_f1'])}** |")
        lines.append(f"| Test cross-entropy loss | {evaluation['test_loss']:.6f} |")
        lines.append(f"| Inference latency p50 | {evaluation['inference_p50_ms']:.6f} ms/sample |")
        lines.append(f"| Inference latency p95 | {evaluation['inference_p95_ms']:.6f} ms/sample |")
        lines.append(f"| Batch size | {evaluation['batch_size']} |")
        lines.append(f"| Evaluation wall clock | {seconds(evaluation.get('wall_clock_seconds'))} s |")
        lines.append(f"| Device | `{evaluation['device']}` |")
        lines.append("")
        lines.append(
            "Per-class precision, recall, F1, support, and the full 10×10 confusion "
            "matrix are stored in the `eval_report` artifact and returned by the "
            "lineage endpoint."
        )
    else:  # pragma: no cover - a failed build would show this
        lines.append("No evaluation report was recorded.")
    lines.append("")

    artifacts = training["artifacts"]
    lineage = training["lineage"]
    lines.append("### 4.4 Artifact store, dedupe, and lineage")
    lines.append("")
    lines.append("| Artifact metric | Value |")
    lines.append("|---|---|")
    lines.append(f"| Unique artifacts | **{artifacts['unique_artifacts']}** |")
    lines.append(f"| Artifact kinds | {', '.join(f'{k}: {v}' for k, v in artifacts['by_kind'].items())} |")
    lines.append(f"| Stored bytes | {bytes_mb(artifacts['stored_bytes'])} |")
    lines.append(f"| Attempted writes | {bytes_mb(artifacts['attempted_bytes'])} |")
    lines.append(f"| Deduped bytes saved | **{bytes_mb(artifacts['bytes_saved'])}** |")
    lines.append(f"| Dedupe ratio | **{artifacts['dedupe_ratio'] * 100:.2f}%** of attempted bytes |")
    lines.append(f"| Best-checkpoint lineage chain | {lineage['best_checkpoint_chain_length']} artifacts |")
    lines.append(
        f"| Eval report points at best checkpoint | {lineage['eval_report_parents_checkpoint']} |"
    )
    lines.append("")
    lines.append(
        "Checkpoints use a deterministic `.npz` serialization (tensor bytes plus "
        "canonical JSON metadata), so writing the final-epoch checkpoint when it is "
        "also the best epoch is byte-identical and is deduped by the content-addressed "
        "store instead of stored twice. Lineage is queryable with "
        "`GET /api/artifacts/<sha>/lineage`: `artifact -> producing run -> "
        "hyperparams/git_sha -> dataset snapshot`, following `parent_sha` for chains."
    )
    lines.append("")
    return lines


def render_k8s(k8s: dict[str, Any]) -> list[str]:
    summary = k8s["summary"]
    nodes = k8s.get("nodes", {}).get("nodes", [])
    node = nodes[0] if nodes else {}
    lines: list[str] = []
    lines.append("### 4.5 Kubernetes executor on kind")
    lines.append("")
    lines.append(
        f"Two runs were submitted through the API and executed as real `batch/v1` Jobs "
        f"on a kind cluster running Kubernetes {node.get('kubernetes_version', '—')} "
        f"({node.get('os', '—')}, {node.get('cpu', '—')} CPUs, "
        f"{node.get('memory', '—')} memory, {node.get('container_runtime', '—')}). "
        "Scheduling latency is submit → pod `Running`; queue wait is submit → scheduler "
        "claim. The second run was queued behind the first because the cluster ran one "
        "worker, so its submit-to-running latency includes that queue wait."
    )
    lines.append("")
    lines.append(
        "| Run | Pod | Submit → pod Running (s) | Queue wait (s) | Pod execution (s) | "
        "End-to-end (s) | Val acc | Test acc |"
    )
    lines.append("|---|---|---|---|---|---|---|---|")
    for run in k8s["runs"]:
        lines.append(
            f"| `{run['run_id'][:8]}` | `{run.get('pod_name') or '—'}` | "
            f"{seconds(run.get('scheduling_latency_seconds'))} | "
            f"{seconds(run.get('queue_wait_seconds'))} | "
            f"{seconds(run.get('pod_execution_seconds'))} | "
            f"**{seconds(run.get('end_to_end_seconds'))}** | "
            f"{percent(run.get('val_accuracy'))} | {percent(run.get('test_accuracy'))} |"
        )
    lines.append("")
    latencies = summary["scheduling_latency_seconds"]
    end_to_end = summary["end_to_end_seconds"]
    lines.append(
        f"Observed submit → pod Running: min **{seconds(latencies['min'])} s**, "
        f"median **{seconds(latencies['median'])} s**, max "
        f"**{seconds(latencies['max'])} s**. End-to-end wall time: min "
        f"**{seconds(end_to_end['min'])} s**, median **{seconds(end_to_end['median'])} s**, "
        f"max **{seconds(end_to_end['max'])} s**."
    )
    lines.append("")
    dry_run = k8s.get("gpu_manifest_dry_run") or {}
    lines.append(
        f"The GPU request manifest (one `nvidia.com/gpu` limit plus GPU node selector "
        f"and toleration) was validated with `{dry_run.get('command')}`: "
        f"**validated={dry_run.get('validated')}**. No GPU node existed, so the GPU "
        "path was not scheduled; the CPU path above is the real execution evidence."
    )
    prometheus = k8s.get("prometheus") or {}
    lines.append("")
    lines.append(
        f"Prometheus (the docker-compose instance, scraping "
        f"`host.docker.internal:8010`) scraped the API `/metrics` endpoint "
        f"(`up{{job=\"traintrack-api\"}} = {prometheus.get('target_up')}`). "
        "The Grafana dashboard `TrainTrack` is provisioned from "
        "`deploy/monitoring/dashboards/traintrack.json`; the cluster API exposes "
        "the same `/metrics` endpoint."
    )
    lines.append("")
    return lines


def render_benchmark(benchmark: dict[str, Any]) -> list[str]:
    publish = benchmark["broker_publish_confirm_ms"]
    lines: list[str] = []
    lines.append("### 4.6 Submission benchmark")
    lines.append("")
    lines.append(
        f"{benchmark['parameters']['runs_per_configuration']} no-op runs per configuration, "
        f"{benchmark['parameters']['submitters']} concurrent submitters. Runs are queued "
        "first; the configured number of workers is then started and the backlog drain "
        "time is measured. No-op jobs isolate API, broker-confirm, and scheduler overhead "
        "from model training. Raw latency samples are in `docs/results/benchmark.json`. "
        "Throughput is wall-clock, so absolute values depend on concurrent host load."
    )
    lines.append("")
    lines.append(
        "| Workers | Submitted | Submit throughput (runs/s) | Submit p50 (ms) | "
        "Submit p95 (ms) | Submit p99 (ms) | Drain (s) | Drain throughput (runs/s) | Statuses |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for config in benchmark["configurations"]:
        latency = config["submission_latency_ms"]
        lines.append(
            f"| {config['workers']} | {config['runs']} | "
            f"**{config['submission_throughput_rps']:.2f}** | "
            f"{latency['p50_ms']:.2f} | {latency['p95_ms']:.2f} | "
            f"{latency['p99_ms']:.2f} | **{config['drain_seconds']:.2f}** | "
            f"**{config['drain_throughput_rps']:.2f}** | "
            f"{', '.join(f'{k}: {v}' for k, v in config['statuses'].items())} |"
        )
    lines.append("")
    lines.append(
        f"RabbitMQ publisher-confirm latency over {publish['count']} persistent publishes: "
        f"p50 **{publish['p50_ms']:.2f} ms**, p95 **{publish['p95_ms']:.2f} ms**, "
        f"p99 **{publish['p99_ms']:.2f} ms**."
    )
    lines.append("")
    return lines


def render_reliability(reliability: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    lines.append("### 4.7 Retry and dead-letter behavior")
    lines.append("")
    lines.append(
        "A deliberately failing training job (snapshot path removed after registration) "
        "was submitted with `max_attempts=2` and observed through the real worker:"
    )
    lines.append("")
    lines.append("| Measurement | Value |")
    lines.append("|---|---|")
    lines.append(f"| Final status | `{reliability['status']}` |")
    lines.append(f"| Attempts observed / configured | **{reliability['attempts_observed']} / {reliability['attempts_configured']}** |")
    lines.append(f"| Retry transitions | {reliability['retry_transitions']} |")
    lines.append(f"| Observed retry backoff | **{reliability['observed_retry_delay_seconds']:.3f} s** |")
    lines.append(f"| Messages in `runs.dead` | **{reliability['dlq_messages']}** |")
    lines.append(f"| Failure reason | `{reliability['failure_reason']}` |")
    lines.append("")
    lines.append(
        "The same suite also exercises broker disconnect/reconnect (the worker reconnects "
        "and finishes a subsequent run), orphaned `RUNNING` run recovery after worker loss "
        "(the execution lock expires, the run is failed, requeued, and retried), and "
        "cancellation of a running job (the executor observes the cancel flag and the run "
        "ends `CANCELED`)."
    )
    lines.append("")
    return lines


def render_tests(tests: dict[str, Any]) -> list[str]:
    backend = tests["backend"]
    frontend = tests.get("frontend") or {}
    totals = tests["totals"]
    lines: list[str] = []
    lines.append("### 4.8 Test suite and coverage")
    lines.append("")
    lines.append("| Suite | Tests | Passed | Failed | Coverage |")
    lines.append("|---|---|---|---|---|")
    lines.append(
        f"| Backend (`pytest`) | {backend['tests']} | {backend['passed']} | "
        f"{backend['failed']} | **{backend['coverage_percent']:.2f}%** |"
    )
    if frontend:
        lines.append(
            f"| Frontend (`vitest`) | {frontend['tests']} | {frontend['passed']} | "
            f"{frontend['failed']} | — |"
        )
    lines.append(
        f"| **Total** | **{totals['tests']}** | **{totals['passed']}** | "
        f"**{totals['failed']}** | — |"
    )
    lines.append("")
    lines.append(
        f"Coverage is measured over `traintrack/` with `--cov-fail-under=80` enforced. "
        f"Scope: {backend['coverage_scope']}. The integration test exercises "
        "API → RabbitMQ → worker → LocalExecutor → PyTorch metrics → artifact lineage "
        "against the real compose services."
    )
    lines.append("")
    return lines


def build_section() -> str:
    training = load("training_runs.json")
    k8s = load("k8s_runs.json")
    benchmark = load("benchmark.json")
    reliability = load("reliability.json")
    tests = load("test_results.json")

    lines = [
        START,
        "",
        "## 4. Results (measured)",
        "",
        "Every figure in this section is rendered from `docs/results/*.json` by "
        "`scripts/update_readme_results.py`; those files are written by the build "
        "commands, not typed by hand. Files: "
        "[training](docs/results/training_runs.json), "
        "[Kubernetes](docs/results/k8s_runs.json), "
        "[benchmark](docs/results/benchmark.json), "
        "[reliability](docs/results/reliability.json), "
        "[tests](docs/results/test_results.json).",
        "",
    ]
    lines.extend(render_training(training))
    lines.extend(render_k8s(k8s))
    lines.extend(render_benchmark(benchmark))
    lines.extend(render_reliability(reliability))
    lines.extend(render_tests(tests))
    lines.append(END)
    return "\n".join(lines)


def main() -> None:
    text = README.read_text()
    if START not in text or END not in text:
        raise SystemExit("README is missing RESULTS markers")
    before, _, remainder = text.partition(START)
    _, _, after = remainder.partition(END)
    README.write_text(before + build_section() + after)
    print("README results section updated from docs/results/*.json")


if __name__ == "__main__":
    main()
