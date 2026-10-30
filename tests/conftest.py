"""Shared test helpers.

The clock helper lives here rather than in each test module because the injected
clock is a *contract* — as of D19 it must return an aware datetime, and a naive
one is refused. Two copies of a helper that encodes a contract drift the first
time the contract changes, and the reviewer who noticed the copies was right
before the contract had even moved.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

# Stands in for customer data. If this string ever appears in an audit file or an
# exception, locked decision #5 is broken.
SENTINEL = "patient-name-jane-doe-payload"


def make_clock(step_ms: int = 1) -> Callable[[], datetime]:
    """A clock that advances a fixed step per call, so timestamps are fixed.

    Aware, in UTC. `AuditLog` refuses a naive datetime rather than silently
    reading it as local time (D19), and this is the shape it expects.
    """
    moment = datetime(2026, 9, 20, 11, 2, 4, 881000, tzinfo=UTC)
    step = timedelta(milliseconds=step_ms)

    def now() -> datetime:
        nonlocal moment
        current = moment
        moment += step
        return current

    return now


def records_in(path: Path) -> list[dict[str, object]]:
    """Every record in a JSONL audit log, parsed."""
    text = path.read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


@pytest.fixture
def clock() -> Callable[[], datetime]:
    return make_clock()


_PROBE = "paveo-test-probe"
# Spellings of `paveo test` (src/paveo/bypasses.json) that a pattern in the
# starters cannot stop without refusing ordinary work, each with the words the
# README's "What the guard cannot do" uses for it.
KNOWN_GAPS = {
    f"git 'push' --force {_PROBE} main": "a quoted verb (`git 'push'`)",
    f"git 'reset' --hard {_PROBE}": "a quoted verb (`git 'push'`)",
    f"F=--force; git push $F {_PROBE} main": "a command hidden in a variable",
    f"R=rm; $R -rf /{_PROBE}": "a command hidden in a variable",
    f"rm -r /{_PROBE}": "`rm -r` without `-f`",
}


@pytest.fixture(autouse=True)
def _fresh_matcher_budget() -> None:
    """Each test is one run: a fresh budget for trying repository matchers
    (D118), read from ``TIMEOUT_S`` at its first try, so a test may patch it."""
    from paveo import _matcher_trial  # noqa: PLC0415 - fixture-local on purpose

    _matcher_trial.reset()
