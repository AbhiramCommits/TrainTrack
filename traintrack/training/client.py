"""HTTP client used by training/eval processes to post allowed metrics."""

from __future__ import annotations

import time
from typing import Any

import requests

from traintrack.config import settings


class ApiClient:
    """Posts metric and resource batches to the REST API with retries."""

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        *,
        timeout: float = 30.0,
        attempts: int = 4,
    ) -> None:
        self.base_url = (base_url or settings().api_url).rstrip("/")
        self.api_key = api_key or settings().dev_api_key
        self.timeout = timeout
        self.attempts = attempts
        self.session = requests.Session()
        self.session.headers.update({"X-API-Key": self.api_key})

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        last_error: Exception | None = None
        for attempt in range(1, self.attempts + 1):
            try:
                response = self.session.post(
                    f"{self.base_url}{path}", json=payload, timeout=self.timeout
                )
                if response.status_code in (200, 201):
                    data = response.json()
                    assert isinstance(data, dict)
                    return data
                if response.status_code in (401, 403, 404, 409, 422):
                    raise RuntimeError(
                        f"API rejected {path}: {response.status_code} {response.text[:300]}"
                    )
                last_error = RuntimeError(
                    f"API error {response.status_code} for {path}: {response.text[:300]}"
                )
            except (requests.RequestException, RuntimeError) as exc:
                last_error = exc
            if attempt < self.attempts:
                time.sleep(min(8.0, 0.5 * 2 ** (attempt - 1)))
        raise RuntimeError(f"failed to post {path}: {last_error}")

    def post_metrics(self, run_id: str, points: list[dict[str, Any]]) -> None:
        """Post a batch of metric points."""
        self._post(f"/api/runs/{run_id}/metrics", {"metrics": points})

    def post_resources(self, run_id: str, samples: list[dict[str, Any]]) -> None:
        """Post a batch of device utilization samples."""
        self._post(f"/api/runs/{run_id}/resources", {"samples": samples})
