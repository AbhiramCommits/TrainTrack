"""SQLAlchemy engine and session management."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from traintrack.config import settings

_engine = None
_session_factory: sessionmaker[Session] | None = None


def init_db(url: str | None = None, *, echo: bool = False) -> None:
    """Create (or replace) the global engine for the given database URL."""
    global _engine, _session_factory
    db_url = url or settings().database_url
    if db_url.startswith("postgresql://"):
        db_url = db_url.replace("postgresql://", "postgresql+psycopg://", 1)
    if _engine is not None:
        _engine.dispose()
    _engine = create_engine(
        db_url,
        pool_pre_ping=True,
        pool_size=10,
        max_overflow=20,
        echo=echo,
        future=True,
    )
    _session_factory = sessionmaker(
        bind=_engine, autoflush=False, expire_on_commit=False, class_=Session
    )


def get_engine():
    """Return the global engine, initialising it lazily if needed."""
    if _engine is None:
        init_db()
    return _engine


def get_session() -> Session:
    """Return a new session bound to the global engine."""
    if _session_factory is None:
        init_db()
    assert _session_factory is not None
    return _session_factory()


@contextmanager
def session_scope() -> Iterator[Session]:
    """Transactional scope: commit on success, roll back on error."""
    session = get_session()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def dispose_engine() -> None:
    """Dispose of the global engine (used by tests and worker shutdown)."""
    global _engine, _session_factory
    if _engine is not None:
        _engine.dispose()
    _engine = None
    _session_factory = None
