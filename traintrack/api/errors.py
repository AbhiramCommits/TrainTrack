"""Shared API helpers and error types."""

from __future__ import annotations

import base64
import binascii
import json
import uuid
from datetime import datetime
from typing import Any

from flask import Flask, jsonify
from pydantic import ValidationError
from werkzeug.exceptions import HTTPException


class ApiError(Exception):
    """Domain error carrying an HTTP status code."""

    def __init__(self, message: str, status: int = 400, code: str = "bad_request") -> None:
        super().__init__(message)
        self.message = message
        self.status = status
        self.code = code


def encode_cursor(created_at: datetime, run_id: uuid.UUID) -> str:
    """Encode a keyset pagination cursor."""
    payload = json.dumps(
        {"created_at": created_at.isoformat(), "id": str(run_id)}
    ).encode()
    return base64.urlsafe_b64encode(payload).decode()


def decode_cursor(cursor: str) -> tuple[datetime, uuid.UUID]:
    """Decode a keyset cursor, raising ApiError when malformed."""
    try:
        payload = json.loads(base64.urlsafe_b64decode(cursor.encode()))
        return datetime.fromisoformat(payload["created_at"]), uuid.UUID(payload["id"])
    except (ValueError, KeyError, TypeError, binascii.Error) as exc:
        raise ApiError("Invalid cursor", 400, "invalid_cursor") from exc


def register_error_handlers(app: Flask) -> None:
    """Attach uniform JSON error handlers to the app."""

    @app.errorhandler(ApiError)
    def _api_error(error: ApiError):
        return jsonify({"error": error.code, "message": error.message}), error.status

    @app.errorhandler(ValidationError)
    def _validation_error(error: ValidationError):
        return (
            jsonify(
                {
                    "error": "validation_error",
                    "message": "Request validation failed",
                    "details": json.loads(error.json()),
                }
            ),
            422,
        )

    @app.errorhandler(HTTPException)
    def _http_error(error: HTTPException):
        return (
            jsonify(
                {
                    "error": error.name.lower().replace(" ", "_"),
                    "message": error.description,
                }
            ),
            error.code or 500,
        )

    @app.errorhandler(Exception)
    def _unhandled(error: Exception):
        app.logger.exception("Unhandled API error")
        return jsonify({"error": "internal_error", "message": str(error)}), 500


def json_body() -> dict[str, Any]:
    """Return the JSON body, raising a 400 when absent or invalid."""
    from flask import request

    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        raise ApiError("Expected a JSON object body", 400, "invalid_json")
    return payload
