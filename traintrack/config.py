"""Application configuration loaded from environment variables."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except ValueError:
        return default


@lru_cache(maxsize=1)
def settings() -> Settings:
    """Return the process-wide settings snapshot (cached)."""
    load_dotenv(PROJECT_ROOT / ".env")
    return Settings()


class Settings:
    """Typed accessors over environment configuration.

    Values are read once at construction; tests override environment variables
    before calling :func:`settings` (and clear the cache).
    """

    def __init__(self) -> None:
        self.database_url = _env(
            "DATABASE_URL",
            "postgresql+psycopg://traintrack:traintrack@localhost:55432/traintrack",
        )
        self.redis_url = _env("REDIS_URL", "redis://localhost:56379/0")
        self.amqp_url = _env(
            "AMQP_URL", "amqp://traintrack:traintrack@localhost:5672/%2F"
        )
        self.rabbitmq_management_url = _env(
            "RABBITMQ_MANAGEMENT_URL", "http://localhost:15672"
        )
        self.api_url = _env("TRAINTRACK_API_URL", "http://localhost:8010")
        self.api_port = _env_int("TRAINTRACK_API_PORT", 8010)
        self.api_host = _env("TRAINTRACK_API_HOST", "0.0.0.0")
        self.secret_key = _env("TRAINTRACK_SECRET_KEY", "dev-insecure-key")
        self.dev_api_key = _env(
            "TRAINTRACK_DEV_API_KEY", "tt_dev_local_key_do_not_use_in_prod"
        )
        self.rate_limit = _env_int("TRAINTRACK_RATE_LIMIT", 120)
        self.executor = _env("TRAINTRACK_EXECUTOR", "local").lower()
        self.max_attempts = _env_int("TRAINTRACK_MAX_ATTEMPTS", 3)
        self.worker_prefetch = _env_int("TRAINTRACK_WORKER_PREFETCH", 1)
        self.job_timeout = _env_int("TRAINTRACK_JOB_TIMEOUT", 3600)
        self.log_maxlen = _env_int("REDIS_LOG_MAXLEN", 10000)
        self.artifact_root = Path(
            _env("TRAINTRACK_ARTIFACT_ROOT", str(PROJECT_ROOT / "artifacts"))
        ).resolve()
        self.data_root = Path(
            _env("TRAINTRACK_DATA_ROOT", str(PROJECT_ROOT / "data"))
        ).resolve()
        self.k8s_namespace = _env("TRAINTRACK_K8S_NAMESPACE", "traintrack")
        self.k8s_image = _env("TRAINTRACK_K8S_IMAGE", "traintrack:dev")
        self.k8s_cpu_request = _env("TRAINTRACK_K8S_CPU_REQUEST", "1")
        self.k8s_cpu_limit = _env("TRAINTRACK_K8S_CPU_LIMIT", "4")
        self.k8s_memory_request = _env("TRAINTRACK_K8S_MEMORY_REQUEST", "1Gi")
        self.k8s_memory_limit = _env("TRAINTRACK_K8S_MEMORY_LIMIT", "6Gi")
        self.k8s_gpu_node_selector_key = _env(
            "TRAINTRACK_K8S_GPU_NODE_SELECTOR_KEY", "runpod.io/gpu"
        )
        self.leaderboard_ttl = _env_int("TRAINTRACK_LEADERBOARD_TTL", 30)
        self.prometheus_port = _env_int("PROMETHEUS_PORT", 9090)
        self.retry_base_seconds = _env_float("TRAINTRACK_RETRY_BASE_SECONDS", 2.0)
        self.retry_max_seconds = _env_float("TRAINTRACK_RETRY_MAX_SECONDS", 60.0)

    @property
    def sync_database_url(self) -> str:
        """SQLAlchemy URL for the synchronous psycopg driver."""
        return self.database_url.replace(
            "postgresql://", "postgresql+psycopg://"
        )


def reset_settings_cache() -> None:
    """Clear the cached settings (used by tests)."""
    settings.cache_clear()
