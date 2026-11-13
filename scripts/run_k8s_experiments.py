"""Submit two real runs to the TrainTrack API on kind and verify the GPU path.

The script can start a kubectl port-forward itself, submits CPU training jobs
through the Kubernetes executor, waits for completion, performs a server-side
dry run of a GPU-request manifest, and writes docs/results/k8s_run_manifest.json.
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
MANIFEST_PATH = PROJECT_ROOT / "docs" / "results" / "k8s_run_manifest.json"
EXPERIMENT_NAME = "k8s-kind-check"
TERMINAL = {"SUCCEEDED", "FAILED", "CANCELED"}


class Api:
    def __init__(self, base_url: str, api_key: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.session.headers.update({"X-API-Key": api_key})

    def get(self, path: str, **params: Any) -> Any:
        response = self.session.get(f"{self.base_url}{path}", params=params, timeout=30)
        if response.status_code == 429:
            time.sleep(float(response.headers.get("Retry-After", "2")))
            return self.get(path, **params)
        response.raise_for_status()
        return response.json()

    def post(self, path: str, payload: dict, key: str) -> tuple[int, Any]:
        response = self.session.post(
            f"{self.base_url}{path}",
            json=payload,
            headers={"Idempotency-Key": key},
            timeout=30,
        )
        return response.status_code, response.json()


def git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=PROJECT_ROOT, text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def start_port_forward(api_url: str, namespace: str) -> subprocess.Popen | None:
    """Start a port-forward when the API is not already reachable."""
    try:
        requests.get(f"{api_url}/healthz", timeout=2)
        return None
    except requests.RequestException:
        pass
    port = api_url.rsplit(":", 1)[-1].split("/")[0]
    process = subprocess.Popen(
        [
            "kubectl",
            "-n",
            namespace,
            "port-forward",
            "svc/traintrack-api",
            f"{port}:8010",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    for _ in range(60):
        try:
            requests.get(f"{api_url}/healthz", timeout=2)
            return process
        except requests.RequestException:
            time.sleep(1)
    process.terminate()
    raise RuntimeError("API did not become reachable through port-forward")


def gpu_dry_run() -> dict[str, Any]:
    """Render a GPU Job and ask the API server to validate it."""
    output = "/tmp/traintrack-gpu-job.yaml"
    subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts/render_gpu_manifest.py"), "--output", output],
        check=True,
        capture_output=True,
        text=True,
    )
    result = subprocess.run(
        ["kubectl", "apply", "--dry-run=server", "-f", output],
        capture_output=True,
        text=True,
    )
    return {
        "command": f"kubectl apply --dry-run=server -f {output}",
        "returncode": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
        "validated": result.returncode == 0,
        "manifest": output,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-url", default="http://localhost:18011")
    parser.add_argument("--api-key", default="tt_dev_local_key_do_not_use_in_prod")
    parser.add_argument("--namespace", default="traintrack")
    parser.add_argument("--experiment", default=EXPERIMENT_NAME)
    parser.add_argument("--timeout", type=float, default=1800.0)
    args = parser.parse_args(argv)

    forward = start_port_forward(args.api_url, args.namespace)
    api = Api(args.api_url, args.api_key)
    try:
        experiments = api.get("/api/experiments")["experiments"]
        experiment = next(
            (item for item in experiments if item["name"] == args.experiment), None
        )
        if experiment is None:
            status, experiment = api.post(
                "/api/experiments",
                {"name": args.experiment, "description": "kind Kubernetes executor check"},
                key=f"k8s-exp-{args.experiment}",
            )
            if status not in (200, 201):
                raise RuntimeError(f"experiment creation failed: {experiment}")

        snapshots = api.get("/api/snapshots")["snapshots"]
        if not snapshots:
            raise SystemExit("no validated snapshot in the cluster registry")
        snapshot = snapshots[0]

        dry_run = gpu_dry_run()
        print(f"GPU manifest dry-run validated={dry_run['validated']}")
        if not dry_run["validated"]:
            raise RuntimeError(f"GPU manifest dry run failed: {dry_run['stderr']}")

        sha = git_sha()
        configs = [
            {"lr": 0.001, "batch_size": 128, "epochs": 1, "optimizer": "adam", "seed": 42},
            {"lr": 0.002, "batch_size": 256, "epochs": 1, "optimizer": "adam", "seed": 43},
        ]
        run_ids = []
        for index, config in enumerate(configs):
            status, body = api.post(
                "/api/runs",
                {
                    "experiment_id": experiment["id"],
                    "job_kind": "train",
                    "hyperparams": config,
                    "dataset_snapshot_id": snapshot["id"],
                    "requested_gpus": 0,
                    "git_sha": sha,
                    "timeout_seconds": 900,
                },
                key=f"k8s-kind-v1-{index}",
            )
            if status not in (200, 201):
                raise RuntimeError(f"submission failed: {body}")
            print(f"submitted k8s run {body['id']} ({status})")
            run_ids.append(body["id"])

        started = time.monotonic()
        states: dict[str, dict[str, Any]] = {}
        last: dict[str, str] = {}
        while time.monotonic() - started < args.timeout:
            states = {run_id: api.get(f"/api/runs/{run_id}") for run_id in run_ids}
            for run_id, detail in states.items():
                if last.get(run_id) != detail["status"]:
                    print(f"  {run_id[:8]} -> {detail['status']}", flush=True)
                    last[run_id] = detail["status"]
            if all(detail["status"] in TERMINAL for detail in states.values()):
                break
            time.sleep(4)
        else:
            raise TimeoutError("Kubernetes runs did not finish in time")

        failed = [run_id for run_id, detail in states.items() if detail["status"] != "SUCCEEDED"]
        if failed:
            for run_id in failed:
                logs = api.get(f"/api/runs/{run_id}/logs", tail=200)
                print(f"--- logs for {run_id} ---")
                for entry in logs["logs"]:
                    print(entry["line"])
            return 1

        manifest = {
            "generated_at": datetime.now(UTC).isoformat(),
            "generated_by": "scripts/run_k8s_experiments.py",
            "git_sha": sha,
            "cluster": args.namespace,
            "experiment": experiment,
            "snapshot": {
                "id": snapshot["id"],
                "version": snapshot["version"],
                "content_hash": snapshot["content_hash"],
                "row_count": snapshot["row_count"],
            },
            "gpu_manifest_dry_run": dry_run,
            "run_ids": run_ids,
            "api_url": args.api_url,
        }
        MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
        MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"wrote {MANIFEST_PATH}")
        return 0
    finally:
        if forward is not None:
            forward.terminate()
            forward.wait(timeout=10)


if __name__ == "__main__":
    sys.exit(main())
