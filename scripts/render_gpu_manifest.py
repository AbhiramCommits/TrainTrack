"""Render a GPU-request Job manifest for server-side dry-run validation."""

from __future__ import annotations

import argparse
import uuid
from pathlib import Path

import yaml
from traintrack.executors.base import JobContext
from traintrack.executors.kubernetes import K8sRenderOptions, render_job_manifest


def main() -> None:
    """Write a rendered GPU manifest to the requested path."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="/tmp/traintrack-gpu-job.yaml")
    args = parser.parse_args()
    job = JobContext(
        run_id=uuid.UUID("11111111-1111-4111-8111-111111111111"),
        experiment_id=uuid.UUID("22222222-2222-4222-8222-222222222222"),
        job_kind="train",
        hyperparams={"lr": 0.001},
        dataset_snapshot_path="/data/snapshots/mnist-v1",
        dataset_snapshot_id=None,
        requested_gpus=1,
        image="traintrack:dev",
        timeout_seconds=600,
    )
    manifest = render_job_manifest(
        job,
        K8sRenderOptions(
            namespace="traintrack",
            image="traintrack:dev",
            env={
                "DATABASE_URL": "postgresql+psycopg://traintrack:traintrack@postgres:5432/traintrack",
                "TRAINTRACK_API_URL": "http://traintrack-api:8010",
                "TRAINTRACK_API_KEY": "dry-run",
            },
        ),
    )
    Path(args.output).write_text(yaml.safe_dump(manifest, sort_keys=True))
    print(args.output)


if __name__ == "__main__":
    main()
