"""Concurrent submission and scheduler-drain benchmark.

Measures:
  * API submission throughput and p50/p95/p99 latency with concurrent submitters
  * RabbitMQ publisher-confirm latency on a dedicated, temporary queue
  * drain time for N queued no-op runs with 1, 2 and 4 worker processes

Writes raw samples and summaries to docs/results/benchmark.json. No-op jobs are
used deliberately so the measurement isolates submission, broker, and scheduler
overhead rather than model training time.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pika
import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = PROJECT_ROOT / "docs" / "results" / "benchmark.json"
EXPERIMENT_NAME = "benchmark-noop"
TERMINAL = {"SUCCEEDED", "FAILED", "CANCELED"}


def git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=PROJECT_ROOT, text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def percentiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"p50_ms": None, "p95_ms": None, "p99_ms": None, "mean_ms": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "p50_ms": round(float(np.percentile(array, 50)), 3),
        "p95_ms": round(float(np.percentile(array, 95)), 3),
        "p99_ms": round(float(np.percentile(array, 99)), 3),
        "mean_ms": round(float(array.mean()), 3),
    }


class Api:
    def __init__(self, base_url: str, api_key: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.headers = {"X-API-Key": api_key}

    def get(self, path: str, **params: Any) -> Any:
        for _attempt in range(6):
            response = requests.get(
                f"{self.base_url}{path}", params=params, headers=self.headers, timeout=30
            )
            if response.status_code == 429:
                time.sleep(float(response.headers.get("Retry-After", "2")))
                continue
            response.raise_for_status()
            return response.json()
        raise RuntimeError(f"rate limited on {path}")

    def post(self, path: str, payload: dict[str, Any], key: str | None = None) -> tuple[int, Any, float]:
        headers = dict(self.headers)
        if key:
            headers["Idempotency-Key"] = key
        started = time.perf_counter()
        response = requests.post(
            f"{self.base_url}{path}", json=payload, headers=headers, timeout=30
        )
        elapsed_ms = (time.perf_counter() - started) * 1000
        return response.status_code, response.json(), elapsed_ms


def get_or_create_experiment(api: Api, name: str) -> dict[str, Any]:
    for experiment in api.get("/api/experiments")["experiments"]:
        if experiment["name"] == name:
            return experiment
    status, body, _ = api.post(
        "/api/experiments",
        {"name": name, "description": "No-op scheduler throughput benchmark"},
        key=f"bench-exp-{uuid.uuid4().hex[:8]}",
    )
    if status not in (200, 201):
        raise RuntimeError(f"experiment creation failed: {body}")
    return body


def broker_channel() -> tuple[pika.BlockingConnection, Any]:
    parameters = pika.URLParameters(
        os.environ.get(
            "AMQP_URL", "amqp://traintrack:traintrack@localhost:5672/%2F"
        )
    )
    connection = pika.BlockingConnection(parameters)
    channel = connection.channel()
    from traintrack.broker.topology import declare_topology

    declare_topology(channel)
    return connection, channel


def measure_publish_confirms(samples: int) -> dict[str, Any]:
    connection, channel = broker_channel()
    queue = f"bench.publish.{uuid.uuid4().hex[:10]}"
    try:
        channel.queue_declare(queue=queue, durable=False, auto_delete=True)
        channel.queue_bind(queue=queue, exchange="traintrack", routing_key=queue)
        channel.confirm_delivery()
        latencies: list[float] = []
        for index in range(samples):
            body = json.dumps({"bench": index}).encode()
            started = time.perf_counter()
            channel.basic_publish(
                exchange="traintrack",
                routing_key=queue,
                body=body,
                properties=pika.BasicProperties(delivery_mode=2),
                mandatory=True,
            )
            latencies.append((time.perf_counter() - started) * 1000)
        return {
            "samples": latencies,
            "count": len(latencies),
            **percentiles(latencies),
        }
    finally:
        try:
            channel.queue_delete(queue=queue)
        finally:
            connection.close()


def purge_queues() -> None:
    connection, channel = broker_channel()
    try:
        from traintrack.broker.topology import RUNS_DEAD, RUNS_RETRY, RUNS_SUBMITTED

        for queue in (RUNS_SUBMITTED, RUNS_RETRY, RUNS_DEAD):
            channel.queue_purge(queue=queue)
    finally:
        connection.close()


def consumer_count() -> int:
    connection, channel = broker_channel()
    try:
        from traintrack.broker.topology import RUNS_SUBMITTED

        declared = channel.queue_declare(queue=RUNS_SUBMITTED, passive=True)
        return int(declared.method.consumer_count)
    finally:
        connection.close()


def start_workers(count: int) -> list[subprocess.Popen]:
    processes = [
        subprocess.Popen(
            [sys.executable, "-m", "traintrack.worker"],
            cwd=str(PROJECT_ROOT),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=os.environ.copy(),
        )
        for _ in range(count)
    ]
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        if consumer_count() >= count:
            return processes
        time.sleep(0.5)
    stop_workers(processes)
    raise RuntimeError(f"workers did not register as consumers (expected {count})")


def stop_workers(processes: list[subprocess.Popen]) -> None:
    for process in processes:
        if process.poll() is None:
            process.send_signal(signal.SIGTERM)
    for process in processes:
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


def run_configuration(
    api: Api,
    experiment: dict[str, Any],
    workers: int,
    runs: int,
    submitters: int,
    timeout: float,
) -> dict[str, Any]:
    purge_queues()
    processes = start_workers(workers)
    snapshot = api.get("/api/snapshots")["snapshots"]
    snapshot_id = snapshot[0]["id"] if snapshot else None
    run_ids: list[str] = []
    latencies: list[float] = []
    token = uuid.uuid4().hex[:8]

    def submit(index: int) -> tuple[int, dict[str, Any], float]:
        return api.post(
            "/api/runs",
            {
                "experiment_id": experiment["id"],
                "job_kind": "noop",
                "hyperparams": {},
                "requested_gpus": 0,
                "timeout_seconds": 120,
            },
            key=f"bench-{workers}-{token}-{index}",
        )

    submit_started = time.perf_counter()
    try:
        with ThreadPoolExecutor(max_workers=submitters) as pool:
            results = list(pool.map(submit, range(runs)))
        submit_seconds = time.perf_counter() - submit_started
        for status, body, elapsed in results:
            if status not in (200, 201):
                raise RuntimeError(f"submission failed: {status} {body}")
            run_ids.append(body["id"])
            latencies.append(elapsed)

        deadline = time.monotonic() + timeout
        states: dict[str, dict[str, Any]] = {}
        while time.monotonic() < deadline:
            listing = api.get(
                "/api/runs",
                experiment_id=experiment["id"],
                limit=200,
            )
            states = {
                run["id"]: run
                for run in listing["runs"]
                if run["id"] in set(run_ids)
            }
            if len(states) == runs and all(
                run["status"] in TERMINAL for run in states.values()
            ):
                break
            time.sleep(0.5)
        else:
            raise TimeoutError("benchmark runs did not drain before timeout")

        finished = [
            datetime.fromisoformat(run["finished_at"])
            for run in states.values()
            if run.get("finished_at")
        ]
        created = [
            datetime.fromisoformat(run["created_at"]) for run in states.values()
        ]
        drain_seconds = (
            (max(finished) - min(created)).total_seconds()
            if finished and created
            else None
        )
        statuses: dict[str, int] = {}
        for run in states.values():
            statuses[run["status"]] = statuses.get(run["status"], 0) + 1
        all_succeeded = statuses.get("SUCCEEDED", 0) == runs
        return {
            "workers": workers,
            "runs": runs,
            "submitters": submitters,
            "submit_seconds": round(submit_seconds, 3),
            "submission_throughput_rps": round(runs / submit_seconds, 3),
            "submission_latency_ms": percentiles(latencies),
            "drain_seconds": round(drain_seconds, 3) if drain_seconds else None,
            "drain_throughput_rps": round(runs / drain_seconds, 3)
            if drain_seconds
            else None,
            "statuses": statuses,
            "all_succeeded": all_succeeded,
            "run_ids": run_ids,
            "snapshot_id": snapshot_id,
        }
    finally:
        stop_workers(processes)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-url", default="http://localhost:8010")
    parser.add_argument("--api-key", default="tt_dev_local_key_do_not_use_in_prod")
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--submitters", type=int, default=6)
    parser.add_argument("--workers", default="1,2,4")
    parser.add_argument("--publish-samples", type=int, default=40)
    parser.add_argument("--timeout", type=float, default=600.0)
    args = parser.parse_args(argv)

    api = Api(args.api_url, args.api_key)
    if consumer_count() > 0:
        raise SystemExit(
            "another worker is consuming runs.submitted; stop it before benchmarking"
        )
    experiment = get_or_create_experiment(api, EXPERIMENT_NAME)
    print(f"experiment {experiment['id']} ({experiment['name']})")

    publish_confirm = measure_publish_confirms(args.publish_samples)
    print(
        f"publish confirm p50={publish_confirm['p50_ms']}ms "
        f"p95={publish_confirm['p95_ms']}ms"
    )

    configurations = []
    for worker_count in [int(value) for value in args.workers.split(",")]:
        print(f"running with {worker_count} worker(s)…", flush=True)
        result = run_configuration(
            api,
            experiment,
            worker_count,
            args.runs,
            args.submitters,
            args.timeout,
        )
        configurations.append(result)
        print(
            f"  submit={result['submission_throughput_rps']} runs/s "
            f"drain={result['drain_seconds']}s "
            f"({result['drain_throughput_rps']} runs/s) "
            f"statuses={result['statuses']}"
        )
        if not result["all_succeeded"]:
            print("  WARNING: not all runs succeeded", file=sys.stderr)

    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "generated_by": "scripts/bench_submit.py",
        "git_sha": git_sha(),
        "api_url": args.api_url,
        "experiment": experiment,
        "environment": {
            "python": sys.version.split()[0],
            "cpu_count": os.cpu_count(),
            "platform": sys.platform,
        },
        "parameters": {
            "runs_per_configuration": args.runs,
            "submitters": args.submitters,
            "worker_counts": [int(value) for value in args.workers.split(",")],
        },
        "broker_publish_confirm_ms": publish_confirm,
        "configurations": configurations,
        "notes": (
            "No-op jobs isolate API, broker-confirm, and scheduler overhead; "
            "training time is measured separately in training_runs.json."
        ),
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
