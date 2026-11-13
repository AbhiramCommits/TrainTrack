"""Unit tests for the run state machine."""

from __future__ import annotations

import pytest
from traintrack.models.state import (
    LEGAL_TRANSITIONS,
    IllegalTransition,
    RunStatus,
    can_transition,
    ensure_transition,
    next_states,
)


def test_happy_path_transitions_are_legal() -> None:
    path = [
        RunStatus.QUEUED,
        RunStatus.SCHEDULED,
        RunStatus.RUNNING,
        RunStatus.SUCCEEDED,
    ]
    for current, target in zip(path, path[1:], strict=False):
        assert can_transition(current, target)
        assert ensure_transition(current, target) is target


def test_running_can_fail_or_be_canceled() -> None:
    assert ensure_transition(RunStatus.RUNNING, RunStatus.FAILED) is RunStatus.FAILED
    assert ensure_transition(RunStatus.RUNNING, RunStatus.CANCELED) is RunStatus.CANCELED


def test_failed_run_can_be_requeued_for_retry() -> None:
    assert ensure_transition(RunStatus.FAILED, RunStatus.QUEUED) is RunStatus.QUEUED


def test_queued_and_scheduled_can_be_canceled() -> None:
    assert can_transition(RunStatus.QUEUED, RunStatus.CANCELED)
    assert can_transition(RunStatus.SCHEDULED, RunStatus.CANCELED)


@pytest.mark.parametrize(
    ("current", "target"),
    [
        (RunStatus.QUEUED, RunStatus.RUNNING),
        (RunStatus.QUEUED, RunStatus.SUCCEEDED),
        (RunStatus.SCHEDULED, RunStatus.SUCCEEDED),
        (RunStatus.RUNNING, RunStatus.QUEUED),
        (RunStatus.SUCCEEDED, RunStatus.RUNNING),
        (RunStatus.SUCCEEDED, RunStatus.FAILED),
        (RunStatus.CANCELED, RunStatus.QUEUED),
        (RunStatus.FAILED, RunStatus.RUNNING),
        (RunStatus.FAILED, RunStatus.SUCCEEDED),
    ],
)
def test_illegal_transitions_raise(current: RunStatus, target: RunStatus) -> None:
    assert not can_transition(current, target)
    with pytest.raises(IllegalTransition):
        ensure_transition(current, target)


def test_unknown_status_is_rejected() -> None:
    with pytest.raises(IllegalTransition):
        ensure_transition("QUEUED", "WOBBLING")
    with pytest.raises(IllegalTransition):
        ensure_transition("NOT_A_STATE", RunStatus.RUNNING)


def test_terminal_states_have_no_successors() -> None:
    assert next_states(RunStatus.SUCCEEDED) == frozenset()
    assert next_states(RunStatus.CANCELED) == frozenset()


def test_only_failed_is_retryable() -> None:
    retryable = {
        status for status in RunStatus if RunStatus.QUEUED in LEGAL_TRANSITIONS[status]
    }
    assert retryable == {RunStatus.FAILED}
