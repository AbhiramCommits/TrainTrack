"""Prometheus /metrics endpoint."""

from __future__ import annotations

from flask import Blueprint, Response

from traintrack.metrics.exporter import render_metrics

bp = Blueprint("metrics", __name__)


@bp.get("/metrics")
def metrics():
    """Expose Prometheus metrics for scraping (no API key required)."""
    return Response(render_metrics(), mimetype="text/plain; version=0.0.4")
