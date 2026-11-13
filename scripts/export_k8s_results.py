"""Export real kind run timings from the in-cluster registry.

Reads the cluster Postgres through ``kubectl exec`` so the numbers come from
the same database the Kubernetes worker wrote, computes scheduling latency and
end-to-end wall time, and records Prometheus scrape status if reachable.
"""

from __future__ import annotations

import json
import statistics
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = PROJECT_ROOT / "docs" / "results" / "k8s_run_manifest.json"
OUTPUT_PATH = PROJECT_ROOT / "docs" / "results" / "k8s_runs.json"
NAMESPACE = "traintrack"


def _psql_json(sql: str) -> Any:
    result = subprocess.run(
        [
            "kubectl",
            "-n",
            NAMESPACE,
            "exec",
            "deploy/postgres",
            "--",
            "psql",
            "-U",
            "traintrack",
            "-d",
            "traintrack",
            "-t",
            "-A",
            "-c",
            sql,
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    payload = result.stdout.strip()
    return json.loads(payload) if payload else None


def _node_info() -> dict[str, Any]:
    result = subprocess.run(
        ["kubectl", "get", "nodes", "-o", "json"],
        capture_output=True,
        text=True,
        check=True,
    )
    data = json.loads(result.stdout)
    nodes = []
    for item in data.get("items", []):
        capacity = item.get("status", {}).get("capacity", {})
        nodes.append(
            {
                "name": item.get("metadata", {}).get("name"),
                "kubernetes_version": item.get("status", {})
                .get("nodeInfo", {})
                .get("kubeletVersion"),
                "cpu": capacity.get("cpu"),
                "memory": capacity.get("memory"),
                "os": item.get("status", {}).get("nodeInfo", {}).get("osImage"),
                "container_runtime": item.get("status", {})
                .get("nodeInfo", {})
                .get("containerRuntimeVersion"),
            }
        )
    return {"nodes": nodes}


def _prometheus_status() -> dict[str, Any]:
    try:
        up = requests.get(
            "http://localhost:9090/api/v1/query",
            params={"query": 'up{job="traintrack-api"}'},
            timeout=5,
        ).json()
        queue = requests.get(
            "http://localhost:9090/api/v1/query",
            params={"query": "traintrack_queue_depth"},
            timeout=5,
        ).json()
        results = up.get("data", {}).get("result", [])
        return {
            "reachable": True,
            "target_up": float(results[0]["value"][1]) if results else None,
            "queue_depth_sample": queue.get("data", {}).get("result"),
        }
    except requests.RequestException as exc:
        return {"reachable": False, "error": str(exc)}


def _parse(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value else None


def _seconds(later: datetime | None, earlier: datetime | None) -> float | None:
    if later is None or earlier is None:
        return None
    return round((later - earlier).total_seconds(), 3)


def main() -> int:
    if not MANIFEST_PATH.exists():
        raise SystemExit("run scripts/run_k8s_experiments.py first")
    manifest = json.loads(MANIFEST_PATH.read_text())
    experiment_name = manifest["experiment"]["name"]
    sql = f"""
SELECT COALESCE(json_agg(row_to_json(t) ORDER BY t.created_at), '[]'::json)
FROM (
  SELECT r.id::text AS run_id, r.status, r.hyperparams, r.requested_gpus,
         r.executor, r.pod_name, r.attempt_count,
         r.created_at::text AS created_at,
         r.scheduled_at::text AS scheduled_at,
         r.started_at::text AS started_at,
         r.finished_at::text AS finished_at,
         r.scheduling_latency_seconds,
         (SELECT max(value) FROM metric_points m
           WHERE m.run_id = r.id AND m.name = 'val_accuracy') AS val_accuracy,
         (SELECT max(value) FROM metric_points m
           WHERE m.run_id = r.id AND m.name = 'test_accuracy') AS test_accuracy,
         (SELECT count(*) FROM artifacts a WHERE a.run_id = r.id) AS artifact_count
  FROM runs r JOIN experiments e ON e.id = r.experiment_id
  WHERE e.name = '{experiment_name}' AND r.job_kind = 'train'
) t;
"""
    rows = _psql_json(sql) or []
    if len(rows) < 2:
        raise SystemExit(f"expected at least 2 kind runs, found {len(rows)}")

    runs: list[dict[str, Any]] = []
    for row in rows:
        created = _parse(row["created_at"])
        scheduled = _parse(row["scheduled_at"])
        started = _parse(row["started_at"])
        finished = _parse(row["finished_at"])
        runs.append(
            {
                **row,
                "queue_wait_seconds": _seconds(scheduled, created),
                "pod_execution_seconds": _seconds(finished, started),
                "end_to_end_seconds": _seconds(finished, created),
                "scheduling_latency_seconds": (
                    round(float(row["scheduling_latency_seconds"]), 3)
                    if row["scheduling_latency_seconds"] is not None
                    else None
                ),
            }
        )

    latencies = [
        run["scheduling_latency_seconds"]
        for run in runs
        if run["scheduling_latency_seconds"] is not None
    ]
    end_to_end = [
        run["end_to_end_seconds"] for run in runs if run["end_to_end_seconds"] is not None
    ]
    payload = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "generated_by": "scripts/export_k8s_results.py",
        "cluster": manifest.get("cluster", NAMESPACE),
        "experiment": manifest["experiment"],
        "snapshot": manifest["snapshot"],
        "gpu_manifest_dry_run": manifest.get("gpu_manifest_dry_run"),
        "nodes": _node_info(),
        "runs": runs,
        "summary": {
            "run_count": len(runs),
            "succeeded": sum(1 for run in runs if run["status"] == "SUCCEEDED"),
            "scheduling_latency_seconds": {
                "min": min(latencies) if latencies else None,
                "median": round(statistics.median(latencies), 3) if latencies else None,
                "max": max(latencies) if latencies else None,
            },
            "end_to_end_seconds": {
                "min": min(end_to_end) if end_to_end else None,
                "median": round(statistics.median(end_to_end), 3)
                if end_to_end
                else None,
                "max": max(end_to_end) if end_to_end else None,
            },
            "executors": sorted({run["executor"] for run in runs if run["executor"]}),
            "pods": sorted({run["pod_name"] for run in runs if run["pod_name"]}),
        },
        "prometheus": _prometheus_status(),
    }
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {OUTPUT_PATH}")
    for run in runs:
        print(
            run["run_id"][:8],
            run["status"],
            "scheduling=", run["scheduling_latency_seconds"],
            "end_to_end=", run["end_to_end_seconds"],
            "pod=", run["pod_name"],
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
