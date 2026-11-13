"""API blueprint for artifacts and lineage."""

from __future__ import annotations

from typing import Any

from flask import Blueprint, jsonify, request
from sqlalchemy import select

from traintrack.api.auth import require_api_key
from traintrack.api.deps import db_session
from traintrack.api.errors import ApiError
from traintrack.api.serializers import artifact_to_dict, run_to_dict, snapshot_to_dict
from traintrack.models.artifact import Artifact
from traintrack.models.base import utcnow
from traintrack.models.run import Run
from traintrack.models.snapshot import DatasetSnapshot

bp = Blueprint("artifacts", __name__, url_prefix="/api/artifacts")

MAX_LINEAGE_DEPTH = 20


@bp.get("")
@require_api_key
def list_artifacts():
    """List artifacts, optionally filtered by producing run."""
    session = db_session()
    query = select(Artifact)
    run_id = request.args.get("run_id")
    if run_id:
        query = query.where(Artifact.run_id == run_id)
    kind = request.args.get("kind")
    if kind:
        query = query.where(Artifact.kind == kind)
    artifacts = list(
        session.execute(query.order_by(Artifact.created_at.desc()).limit(500)).scalars()
    )
    return jsonify({"artifacts": [artifact_to_dict(a) for a in artifacts]})


def _lineage_node(session, artifact: Artifact) -> dict[str, Any]:
    run = session.get(Run, artifact.run_id) if artifact.run_id else None
    snapshot = (
        session.get(DatasetSnapshot, artifact.dataset_snapshot_id)
        if artifact.dataset_snapshot_id
        else None
    )
    return {
        "artifact": artifact_to_dict(artifact),
        "run": (
            run_to_dict(run, include_transitions=False) if run is not None else None
        ),
        "dataset_snapshot": snapshot_to_dict(snapshot) if snapshot else None,
    }


@bp.get("/<sha256>/lineage")
@require_api_key
def artifact_lineage(sha256: str):
    """Return the chain artifact -> run -> hyperparams/git_sha -> snapshot."""
    session = db_session()
    artifact = session.execute(
        select(Artifact).where(Artifact.sha256 == sha256)
    ).scalar_one_or_none()
    if artifact is None:
        raise ApiError("Artifact not found", 404, "not_found")

    chain: list[dict[str, Any]] = []
    seen: set[str] = set()
    current: Artifact | None = artifact
    while current is not None and len(chain) < MAX_LINEAGE_DEPTH:
        if current.sha256 in seen:
            break
        seen.add(current.sha256)
        node = _lineage_node(session, current)
        chain.append(node)
        if not current.parent_sha:
            break
        current = session.execute(
            select(Artifact).where(Artifact.sha256 == current.parent_sha)
        ).scalar_one_or_none()

    head = chain[0]
    return jsonify(
        {
            "sha256": sha256,
            "chain": chain,
            "artifact": head["artifact"],
            "run": head["run"],
            "dataset_snapshot": head["dataset_snapshot"],
            "generated_at": utcnow().isoformat(),
        }
    )
