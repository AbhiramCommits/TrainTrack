"""Pure-Python run state machine.

The registry enforces this transition table on every status write. Illegal
transitions raise :class:`IllegalTransition` and are covered by unit tests.
"""

from __future__ import annotations

from enum import Enum

UNKNOWN_STATE = "UNKNOWN"


class RunStatus(str, Enum):
    """Lifecycle states a run can occupy."""

    QUEUED = "QUEUED"
    SCHEDULED = "SCHEDULED"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    CANCELED = "CANCELED"


TERMINAL_STATUSES = frozenset(
    {RunStatus.SUCCEEDED, RunStatus.FAILED, RunStatus.CANCELED}
)

#: Legal transitions. A failed run may be re-queued for a retry attempt.
LEGAL_TRANSITIONS: dict[RunStatus, frozenset[RunStatus]] = {
    RunStatus.QUEUED: frozenset({RunStatus.SCHEDULED, RunStatus.CANCELED}),
    RunStatus.SCHEDULED: frozenset(
        {RunStatus.RUNNING, RunStatus.CANCELED, RunStatus.FAILED}
    ),
    RunStatus.RUNNING: frozenset(
        {RunStatus.SUCCEEDED, RunStatus.FAILED, RunStatus.CANCELED}
    ),
    RunStatus.FAILED: frozenset({RunStatus.QUEUED}),
    RunStatus.SUCCEEDED: frozenset(),
    RunStatus.CANCELED: frozenset(),
}


class IllegalTransition(Exception):
    """Raised when a run is moved between states not present in the table."""

    def __init__(self, current: str, target: str) -> None:
        self.current = current
        self.target = target
        super().__init__(f"Illegal run transition: {current} -> {target}")


def coerce_status(value: RunStatus | str) -> RunStatus:
    """Convert a string or enum into a :class:`RunStatus`."""
    if isinstance(value, RunStatus):
        return value
    try:
        return RunStatus(value)
    except ValueError as exc:
        raise IllegalTransition(UNKNOWN_STATE, str(value)) from exc


def can_transition(current: RunStatus | str, target: RunStatus | str) -> bool:
    """Return True when ``current -> target`` is legal."""
    try:
        current_status = coerce_status(current)
        target_status = coerce_status(target)
    except IllegalTransition:
        return False
    return target_status in LEGAL_TRANSITIONS[current_status]


def ensure_transition(current: RunStatus | str, target: RunStatus | str) -> RunStatus:
    """Validate a transition, returning the target status or raising."""
    current_status = coerce_status(current)
    target_status = coerce_status(target)
    if target_status not in LEGAL_TRANSITIONS[current_status]:
        raise IllegalTransition(current_status.value, target_status.value)
    return target_status


def next_states(current: RunStatus | str) -> frozenset[RunStatus]:
    """Return the legal successors of a state."""
    return LEGAL_TRANSITIONS[coerce_status(current)]
