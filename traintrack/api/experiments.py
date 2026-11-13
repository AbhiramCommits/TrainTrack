"""Experiment endpoints, including the cached leaderboard."""

from __future__ import annotations

import uuid

from flask import Blueprint, jsonify
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError

from traintrack.api.auth import require_api_key
from traintrack.api.deps import db_session
from traintrack.api.errors import ApiError, json_body
from traintrack.api.serializers import iso, run_to_dict
from traintrack.models.experiment import Experiment
from traintrack.models.run import Run
from traintrack.services.leaderboard import (
    LEADERBOARD_METRIC,
    get_leaderboard,
)

bp = Blueprint("experiments", __name__, url_prefix="/api/experiments")


@bp.post("")
@require_api_key
def create_experiment():
    """Create an experiment."""
    from traintrack.api.schemas import ExperimentCreate

    payload = ExperimentCreate.model_validate(json_body())
    session = db_session()
    experiment = Experiment(name=payload.name, description=payload.description)
    session.add(experiment)
    try:
        session.commit()
    except IntegrityError as exc:
        session.rollback()
        raise ApiError(
            f"Experiment name already exists: {payload.name}", 409, "conflict"
        ) from exc
    return (
        jsonify(
            {
                "id": str(experiment.id),
                "name": experiment.name,
                "description": experiment.description,
                "created_at": iso(experiment.created_at),
            }
        ),
        201,
    )


@bp.get("")
@require_api_key
def list_experiments():
    """List experiments with run counts and status breakdown."""
    session = db_session()
    experiments = list(
        session.execute(select(Experiment).order_by(Experiment.created_at.desc())).scalars()
    )
    counts = dict(
        session.execute(
            select(Run.experiment_id, func.count(Run.id)).group_by(Run.experiment_id)
        ).all()
    )
    return jsonify(
        {
            "experiments": [
                {
                    "id": str(experiment.id),
                    "name": experiment.name,
                    "description": experiment.description,
                    "created_at": iso(experiment.created_at),
                    "run_count": int(counts.get(experiment.id, 0)),
                }
                for experiment in experiments
            ]
        }
    )


@bp.get("/<experiment_id>")
@require_api_key
def get_experiment(experiment_id: str):
    """Fetch a single experiment."""
    session = db_session()
    try:
        parsed = uuid.UUID(experiment_id)
    except ValueError as exc:
        raise ApiError("Invalid experiment id", 400, "invalid_id") from exc
    experiment = session.get(Experiment, parsed)
    if experiment is None:
        raise ApiError("Experiment not found", 404, "not_found")
    return jsonify(
        {
            "id": str(experiment.id),
            "name": experiment.name,
            "description": experiment.description,
            "created_at": iso(experiment.created_at),
            "runs": [run_to_dict(run) for run in experiment.runs],
        }
    )


@bp.get("/<experiment_id>/leaderboard")
@require_api_key
def leaderboard(experiment_id: str):
    """Rank runs by best validation accuracy, served from a TTL cache."""
    session = db_session()
    try:
        parsed = uuid.UUID(experiment_id)
    except ValueError as exc:
        raise ApiError("Invalid experiment id", 400, "invalid_id") from exc
    if session.get(Experiment, parsed) is None:
        raise ApiError("Experiment not found", 404, "not_found")
    entries, cache_hit = get_leaderboard(session, parsed)
    return jsonify(
        {
            "experiment_id": str(parsed),
            "metric": LEADERBOARD_METRIC,
            "cache_hit": cache_hit,
            "entries": entries,
        }
    )
