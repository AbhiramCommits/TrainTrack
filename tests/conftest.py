"""Shared pytest fixtures backed by real Postgres and Redis from compose."""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterator

import pytest

TEST_DATABASE_URL = os.environ.get(
    "TEST_DATABASE_URL",
    "postgresql+psycopg://traintrack:traintrack@localhost:55432/traintrack_test",
)
TEST_REDIS_URL = os.environ.get(
    "TEST_REDIS_URL", "redis://localhost:56379/15"
)
os.environ["DATABASE_URL"] = TEST_DATABASE_URL
os.environ["REDIS_URL"] = TEST_REDIS_URL
os.environ["TRAINTRACK_API_URL"] = "http://localhost:8010"
os.environ["TRAINTRACK_ARTIFACT_ROOT"] = "/tmp/traintrack-test-artifacts"
# Keep the rate limiter generous in tests; dedicated tests override per-key limits.
os.environ["TRAINTRACK_RATE_LIMIT"] = "100000"

from sqlalchemy import create_engine, text  # noqa: E402
from traintrack.api.app import create_app  # noqa: E402
from traintrack.config import reset_settings_cache  # noqa: E402
from traintrack.db import dispose_engine, get_session, init_db  # noqa: E402
from traintrack.models import Base  # noqa: E402
from traintrack.services.auth import create_api_key  # noqa: E402
from traintrack.services.cache import get_redis, reset_redis_client  # noqa: E402


def _ensure_test_database() -> None:
    admin_url = TEST_DATABASE_URL.rsplit("/", 1)[0] + "/postgres"
    engine = create_engine(admin_url, isolation_level="AUTOCOMMIT")
    database = TEST_DATABASE_URL.rsplit("/", 1)[1]
    with engine.connect() as connection:
        exists = connection.execute(
            text("SELECT 1 FROM pg_database WHERE datname = :name"), {"name": database}
        ).scalar()
        if not exists:
            connection.execute(text(f'CREATE DATABASE "{database}"'))
    engine.dispose()


@pytest.fixture(scope="session", autouse=True)
def database_schema() -> Iterator[None]:
    """Create the test database and schema once per session."""
    _ensure_test_database()
    reset_settings_cache()
    dispose_engine()
    init_db(TEST_DATABASE_URL)
    from traintrack.db import get_engine

    Base.metadata.drop_all(get_engine())
    Base.metadata.create_all(get_engine())
    yield
    dispose_engine()


@pytest.fixture(autouse=True)
def clean_state() -> Iterator[None]:
    """Truncate all tables and flush the test Redis DB between tests."""
    from traintrack.db import get_engine

    yield
    with get_engine().begin() as connection:
        tables = ", ".join(
            f'"{table.name}"' for table in reversed(Base.metadata.sorted_tables)
        )
        connection.execute(text(f"TRUNCATE {tables} RESTART IDENTITY CASCADE"))
    reset_redis_client()
    get_redis().flushdb()


@pytest.fixture()
def app():
    """Flask app wired to the test database."""
    reset_settings_cache()
    reset_redis_client()
    application = create_app({"TESTING": True, "DATABASE_URL": TEST_DATABASE_URL})
    yield application


@pytest.fixture()
def client(app):
    """Flask test client."""
    return app.test_client()


@pytest.fixture()
def api_key(app) -> str:
    """Create a dedicated API key for a test and return plaintext."""
    session = get_session()
    try:
        record, raw = create_api_key(session, f"test-{uuid.uuid4().hex[:8]}")
        session.commit()
        return raw
    finally:
        session.close()


@pytest.fixture()
def auth_headers(api_key: str) -> dict[str, str]:
    """Headers carrying a valid API key."""
    return {"X-API-Key": api_key}


@pytest.fixture()
def experiment(client, auth_headers) -> dict:
    """Create a test experiment and return its JSON."""
    response = client.post(
        "/api/experiments",
        json={"name": f"exp-{uuid.uuid4().hex[:8]}", "description": "test"},
        headers=auth_headers,
    )
    assert response.status_code == 201, response.get_json()
    return response.get_json()


@pytest.fixture()
def idempotency_headers(auth_headers) -> dict[str, str]:
    """Auth headers with a fresh idempotency key."""
    return {**auth_headers, "Idempotency-Key": uuid.uuid4().hex}
