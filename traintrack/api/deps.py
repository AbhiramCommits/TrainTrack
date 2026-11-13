"""Per-request database session helpers."""

from __future__ import annotations

from flask import g

from traintrack.db import get_session


def db_session():
    """Return the request-scoped SQLAlchemy session."""
    session = getattr(g, "db_session", None)
    if session is None:
        session = get_session()
        g.db_session = session
    return session


def close_db_session(exception: BaseException | None = None) -> None:
    """Close the request-scoped session at teardown."""
    session = g.pop("db_session", None)
    if session is not None:
        if exception is not None:
            session.rollback()
        session.close()
