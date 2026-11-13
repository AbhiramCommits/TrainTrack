"""Submit a real MNIST hyperparameter sweep through the API and broker.

Waits for every run to reach a terminal state, then submits an evaluation job
against the best checkpoint. Writes docs/results/run_manifest.json (machine
generated) so later export steps can locate the experiment and eval job.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = PROJECT_ROOT / "docs" / "results" / "run_manifest.json"
EXPERIMENT_NAME = "mnist-sweep-v1"

SWEEP: list[dict[str, Any]] = [
    {
        "slug": "adam-lr001-b64",
        "hyperparams": {
            "lr": 0.001,
            "batch_size": 64,
            "epochs": 2,
            "optimizer": "adam",
            "dropout": 0.0,
            "seed": 42,
        },
    },
    {
        "slug": "adam-lr0005-b128",
        "hyperparams": {
            "lr": 0.0005,
            "batch_size": 128,
            "epochs": 2,
            "optimizer": "adam",
            "dropout": 0.1,
            "seed": 43,
        },
    },
    {
        "slug": "adam-lr002-b256",
        "hyperparams": {
            "lr": 0.002,
            "batch_size": 256,
            "epochs": 2,
            "optimizer": "adam",
            "dropout": 0.0,
            "seed": 44,
        },
    },
    {
        "slug": "sgd-lr01-b64",
        "hyperparams": {
            "lr": 0.01,
            "batch_size": 64,
            "epochs": 2,
            "optimizer": "sgd",
            "momentum": 0.9,
            "seed": 45,
        },
    },
    {
        "slug": "adamw-lr0003-b512",
        "hyperparams": {
            "lr": 0.0003,
            "batch_size": 512,
            "epochs": 2,
            "optimizer": "adamw",
            "weight_decay": 0.01,
            "seed": 46,
        },
    },
]


def git_sha() -> str | None:
    """Return the current short commit hash, or None outside git."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=PROJECT_ROOT, text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


class Api:
    """Tiny authenticated API helper."""

    def __init__(self, base_url: str, api_key: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.session.headers.update({"X-API-Key": api_key})

    def get(self, path: str, **params: Any) -> Any:
        for _attempt in range(5):
            response = self.session.get(
                f"{self.base_url}{path}", params=params, timeout=30
            )
            if response.status_code == 429:
                time.sleep(float(response.headers.get("Retry-After", "2")))
                continue
            response.raise_for_status()
            return response.json()
        raise RuntimeError(f"rate limited repeatedly on {path}")

    def post(
        self, path: str, payload: dict[str, Any], idempotency_key: str | None = None
    ) -> tuple[int, Any]:
        headers = {}
        if idempotency_key:
            headers["Idempotency-Key"] = idempotency_key
        for _attempt in range(5):
            response = self.session.post(
                f"{self.base_url}{path}", json=payload, headers=headers, timeout=30
            )
            if response.status_code != 429:
                return response.status_code, response.json()
            time.sleep(float(response.headers.get("Retry-After", "2")))
        raise RuntimeError(f"rate limited repeatedly on {path}")


def get_or_create_experiment(api: Api, name: str) -> dict[str, Any]:
    """Reuse an experiment by name or create it."""
    for experiment in api.get("/api/experiments")["experiments"]:
        if experiment["name"] == name:
            return experiment
    status, body = api.post(
        "/api/experiments",
        {"name": name, "description": "MNIST CNN lr/batch-size sweep"},
    )
    if status not in (200, 201):
        raise RuntimeError(f"could not create experiment: {body}")
    return body


def wait_for_runs(
    api: Api, run_ids: list[str], timeout: float = 3600.0
) -> dict[str, dict[str, Any]]:
    """Poll until every run is SUCCEEDED/FAILED/CANCELED."""
    terminal = {"SUCCEEDED", "FAILED", "CANCELED"}
    started = time.monotonic()
    last_status: dict[str, str] = {}
    while time.monotonic() - started < timeout:
        states = {}
        for run_id in run_ids:
            detail = api.get(f"/api/runs/{run_id}")
            states[run_id] = detail
            if last_status.get(run_id) != detail["status"]:
                print(f"  {run_id[:8]} -> {detail['status']}", flush=True)
                last_status[run_id] = detail["status"]
        if all(detail["status"] in terminal for detail in states.values()):
            return states
        time.sleep(3)
    raise TimeoutError("runs did not finish before the timeout")


def best_metric(api: Api, run_id: str, name: str, default: float = -1.0) -> float:
    """Maximum value of a metric for a run."""
    points = api.get(f"/api/runs/{run_id}/metrics", names=name)["points"]
    return max((float(point["value"]) for point in points), default=default)


def checkpoint_for_run(api: Api, run_id: str) -> str | None:
    """Best checkpoint digest, read from the run's train_summary artifact."""
    artifacts = api.get("/api/artifacts", run_id=run_id)["artifacts"]
    summaries = [a for a in artifacts if a["kind"] == "train_summary"]
    if not summaries:
        checkpoints = [a for a in artifacts if a["kind"] == "checkpoint"]
        return checkpoints[-1]["sha256"] if checkpoints else None
    return summaries[0]["parent_sha"]


def main(argv: list[str] | None = None) -> int:
    """Run the sweep end to end."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-url", default="http://localhost:8010")
    parser.add_argument(
        "--api-key", default="tt_dev_local_key_do_not_use_in_prod"
    )
    parser.add_argument("--experiment", default=EXPERIMENT_NAME)
    parser.add_argument("--timeout", type=float, default=3600.0)
    args = parser.parse_args(argv)

    api = Api(args.api_url, args.api_key)
    experiments = api.get("/api/experiments")["experiments"]
    experiment = next(
        (item for item in experiments if item["name"] == args.experiment), None
    )
    if experiment is None:
        experiment = get_or_create_experiment(api, args.experiment)
    snapshots = api.get("/api/snapshots")["snapshots"]
    if not snapshots:
        raise SystemExit("no validated dataset snapshot found; run `make snapshot` first")
    snapshot = snapshots[0]
    print(
        f"experiment {experiment['id']} snapshot {snapshot['version']} "
        f"hash={snapshot['content_hash'][:12]} rows={snapshot['row_count']}"
    )

    sha = git_sha()
    run_ids: list[str] = []
    for config in SWEEP:
        payload = {
            "experiment_id": experiment["id"],
            "job_kind": "train",
            "hyperparams": config["hyperparams"],
            "dataset_snapshot_id": snapshot["id"],
            "requested_gpus": 1,
            "git_sha": sha,
            "timeout_seconds": 1800,
        }
        status, body = api.post(
            "/api/runs", payload, idempotency_key=f"sweep-v2-{config['slug']}"
        )
        if status not in (200, 201):
            raise RuntimeError(f"submission failed for {config['slug']}: {body}")
        print(f"submitted {config['slug']}: {body['id']} ({status})", flush=True)
        run_ids.append(body["id"])

    print("waiting for training runs...", flush=True)
    states = wait_for_runs(api, run_ids, timeout=args.timeout)
    failed = [run_id for run_id, detail in states.items() if detail["status"] != "SUCCEEDED"]
    if failed:
        for run_id in failed:
            print(
                f"FAILED {run_id}: {states[run_id].get('failure_reason')}", file=sys.stderr
            )
        return 1

    scores = {
        run_id: best_metric(api, run_id, "val_accuracy") for run_id in run_ids
    }
    best_run_id = max(scores, key=scores.get)
    best_checkpoint = checkpoint_for_run(api, best_run_id)
    if not best_checkpoint:
        raise SystemExit("best run has no checkpoint artifact")
    print(
        f"best run {best_run_id} val_accuracy={scores[best_run_id]:.4f} "
        f"checkpoint={best_checkpoint[:12]}"
    )

    status, eval_run = api.post(
        "/api/runs",
        {
            "experiment_id": experiment["id"],
            "job_kind": "eval",
            "hyperparams": {"checkpoint_sha": best_checkpoint, "batch_size": 512},
            "git_sha": sha,
            "timeout_seconds": 900,
        },
        idempotency_key=f"sweep-v2-eval-{best_checkpoint[:16]}",
    )
    if status not in (200, 201):
        raise RuntimeError(f"eval submission failed: {eval_run}")
    print(f"submitted eval run {eval_run['id']} ({status})", flush=True)
    eval_states = wait_for_runs(api, [eval_run["id"]], timeout=args.timeout)
    if eval_states[eval_run["id"]]["status"] != "SUCCEEDED":
        raise RuntimeError(f"eval job failed: {eval_states[eval_run['id']]}")

    manifest = {
        "generated_at": datetime.now(UTC).isoformat(),
        "generated_by": "scripts/run_experiments.py",
        "git_sha": sha,
        "experiment": experiment,
        "snapshot": {
            "id": snapshot["id"],
            "version": snapshot["version"],
            "content_hash": snapshot["content_hash"],
            "row_count": snapshot["row_count"],
        },
        "runs": [
            {
                "run_id": run_id,
                "slug": SWEEP[index]["slug"],
                "hyperparams": SWEEP[index]["hyperparams"],
                "best_val_accuracy": scores[run_id],
                "checkpoint_sha": checkpoint_for_run(api, run_id),
                "started_at": states[run_id]["started_at"],
                "finished_at": states[run_id]["finished_at"],
                "duration_seconds": states[run_id]["duration_seconds"],
            }
            for index, run_id in enumerate(run_ids)
        ],
        "best_run_id": best_run_id,
        "best_checkpoint_sha": best_checkpoint,
        "eval_run_id": eval_run["id"],
    }
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {MANIFEST_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
