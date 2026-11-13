"""Pydantic request/response schemas."""

from __future__ import annotations

import math
import re
import uuid
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

METRIC_NAME_RE = re.compile(r"^[A-Za-z0-9_./:-]{1,100}$")


class ExperimentCreate(BaseModel):
    """Payload for creating an experiment."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=200)
    description: str | None = Field(default=None, max_length=4000)

    @field_validator("name")
    @classmethod
    def _strip_name(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("name must not be blank")
        return stripped


class TrainHyperparams(BaseModel):
    """Validated training hyperparameters."""

    model_config = ConfigDict(extra="forbid")

    lr: float = Field(gt=0.0, le=1.0)
    batch_size: int = Field(ge=1, le=4096)
    epochs: int = Field(default=3, ge=1, le=50)
    optimizer: Literal["sgd", "adam", "adamw"] = "adam"
    dropout: float = Field(default=0.0, ge=0.0, le=0.9)
    seed: int = Field(default=42, ge=0, le=2**31 - 1)
    weight_decay: float = Field(default=0.0, ge=0.0, le=1.0)
    momentum: float = Field(default=0.0, ge=0.0, le=1.0)
    label_smoothing: float = Field(default=0.0, ge=0.0, le=0.5)
    max_batches: int | None = Field(default=None, ge=1, le=10000)


class EvalHyperparams(BaseModel):
    """Validated evaluation hyperparameters."""

    model_config = ConfigDict(extra="forbid")

    checkpoint_sha: str = Field(pattern=r"^[a-f0-9]{64}$")
    batch_size: int = Field(default=256, ge=1, le=4096)
    max_batches: int | None = Field(default=None, ge=1, le=10000)


class NoopHyperparams(BaseModel):
    """No-op benchmark jobs carry no hyperparameters."""

    model_config = ConfigDict(extra="forbid")


class RunCreate(BaseModel):
    """Payload for submitting a run."""

    model_config = ConfigDict(extra="forbid")

    experiment_id: uuid.UUID
    job_kind: Literal["train", "eval", "noop"] = "train"
    hyperparams: dict[str, Any] = Field(default_factory=dict)
    dataset_snapshot_id: uuid.UUID | None = None
    requested_gpus: int = Field(default=0, ge=0, le=8)
    priority: int = Field(default=5, ge=0, le=9)
    image: str | None = Field(default=None, max_length=300)
    timeout_seconds: int = Field(default=3600, ge=10, le=86400)
    max_attempts: int = Field(default=3, ge=1, le=10)
    git_sha: str | None = Field(default=None, max_length=64)

    @model_validator(mode="after")
    def _validate_hyperparams(self) -> RunCreate:
        if self.job_kind == "train":
            parsed = TrainHyperparams.model_validate(self.hyperparams)
            if self.dataset_snapshot_id is None:
                raise ValueError("dataset_snapshot_id is required for training runs")
            self.hyperparams = parsed.model_dump()
        elif self.job_kind == "eval":
            parsed_eval = EvalHyperparams.model_validate(self.hyperparams)
            self.hyperparams = parsed_eval.model_dump()
        else:
            NoopHyperparams.model_validate(self.hyperparams)
        return self


class MetricIn(BaseModel):
    """One metric sample in a batch."""

    model_config = ConfigDict(extra="forbid")

    name: str
    step: int = Field(ge=0, le=10_000_000)
    value: float
    wall_time: float | None = None

    @field_validator("name")
    @classmethod
    def _valid_name(cls, value: str) -> str:
        if not METRIC_NAME_RE.match(value):
            raise ValueError(f"invalid metric name: {value!r}")
        return value

    @field_validator("value")
    @classmethod
    def _finite(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("metric value must be finite")
        return value


class MetricBatch(BaseModel):
    """Batch metric ingest payload."""

    model_config = ConfigDict(extra="forbid")

    metrics: list[MetricIn] = Field(min_length=1, max_length=1000)


class ResourceSample(BaseModel):
    """One device utilization sample from the training loop."""

    model_config = ConfigDict(extra="forbid")

    step: int = Field(ge=0)
    device: str = Field(max_length=20)
    utilization_pct: float = Field(ge=0.0, le=100.0)
    memory_used_mb: float = Field(ge=0.0)
    cpu_pct: float = Field(ge=0.0, le=100.0 * 64)
    rss_mb: float = Field(ge=0.0)
    wall_time: float | None = None


class ResourceBatch(BaseModel):
    """Batch device-sample payload."""

    model_config = ConfigDict(extra="forbid")

    samples: list[ResourceSample] = Field(min_length=1, max_length=500)
