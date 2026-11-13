"""Dataset snapshot endpoints for the dashboard picker."""

from __future__ import annotations

import uuid

from flask import Blueprint, jsonify, request
from sqlalchemy import select

from traintrack.api.auth import require_api_key
from traintrack.api.deps import db_session
from traintrack.api.errors import ApiError
from traintrack.api.serializers import snapshot_to_dict
from traintrack.models.snapshot import DatasetSnapshot

bp = Blueprint("snapshots", __name__, url_prefix="/api/snapshots")


@bp.get("")
@require_api_key
def list_snapshots():
    """List dataset snapshots, newest first."""
    session = db_session()
    query = select(DatasetSnapshot)
    if request.args.get("validated", "true").lower() == "true":
        query = query.where(DatasetSnapshot.validated.is_(True))
    snapshots = list(
        session.execute(
            query.order_by(DatasetSnapshot.created_at.desc()).limit(200)
        ).scalars()
    )
    return jsonify({"snapshots": [snapshot_to_dict(s) for s in snapshots]})


@bp.get("/<snapshot_id>")
@require_api_key
def get_snapshot(snapshot_id: str):
    """Fetch one snapshot by id."""
    session = db_session()
    try:
        parsed = uuid.UUID(snapshot_id)
    except ValueError as exc:
        raise ApiError("Invalid snapshot id", 400, "invalid_id") from exc
    snapshot = session.get(DatasetSnapshot, parsed)
    if snapshot is None:
        raise ApiError("Snapshot not found", 404, "not_found")
    return jsonify(snapshot_to_dict(snapshot))
