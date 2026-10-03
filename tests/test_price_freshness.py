"""How old a price table may be (§12.5, P1, D43)."""

from __future__ import annotations

import logging
from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from paveo import Paveo
from paveo.prices import VERIFIED, stale_providers


def test_no_table_is_stale_on_the_day_it_was_read() -> None:
    newest = max(date.fromisoformat(d) for d in VERIFIED.values())

    assert stale_providers(newest, older_than_days=30) == []


def test_a_table_31_days_old_fails_the_release_window() -> None:
    """P1's done-when: the release gate refuses a table dated 31 days ago."""
    oldest = min(date.fromisoformat(d) for d in VERIFIED.values())
    day_31 = date.fromordinal(oldest.toordinal() + 31)

    assert stale_providers(day_31, older_than_days=30)


def test_every_provider_the_table_prices_has_a_verified_date() -> None:
    """Checked against the providers the table actually carries, so a fourth
    cannot be added without a date its age is judged by (/code-review)."""
    from paveo.prices import _ALL_LISTINGS, _PROVIDERS  # noqa: PLC0415

    assert set(VERIFIED) == set(_PROVIDERS)
    assert set(_ALL_LISTINGS) == {m for rows in _PROVIDERS.values() for m in rows}


def test_the_release_gate_itself_fails_at_31_days_and_passes_inside_30() -> None:
    """The gate, not just the function under it (/code-review, D43)."""
    from tests.price_freshness import RELEASE_WINDOW_DAYS, main  # noqa: PLC0415

    oldest = min(date.fromisoformat(d) for d in VERIFIED.values())
    newest = max(date.fromisoformat(d) for d in VERIFIED.values())

    assert RELEASE_WINDOW_DAYS == 30
    assert main(date.fromordinal(oldest.toordinal() + 31)) == 1
    assert main(newest) == 0


def policy() -> dict[str, object]:
    return {
        "version": 1,
        "policy_id": "fresh",
        "agents": [{"id": "bot", "models": {"allow": ["claude-sonnet-5"]}}],
    }


@pytest.fixture(autouse=True)
def _forget_warnings() -> None:
    from paveo import prices  # noqa: PLC0415

    prices._WARNED.clear()


def test_an_old_table_is_announced_when_paveo_starts(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A warning, not a refusal: an old table is not a wrong one (D43). Once per
    provider and process, so a service building many cannot drown it out."""
    later = datetime(2027, 3, 1, tzinfo=UTC)
    with (
        caplog.at_level(logging.WARNING, logger="paveo"),
        Paveo.from_policy(policy(), audit_path=tmp_path / "a.jsonl", now=lambda: later),
    ):
        pass

    with Paveo.from_policy(
        policy(), audit_path=tmp_path / "b.jsonl", now=lambda: later
    ):
        pass

    warned = [r.getMessage() for r in caplog.records]
    for provider in VERIFIED:
        assert sum(f"{provider} prices were last verified" in m for m in warned) == 1


def test_a_fresh_table_starts_quietly(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    today = datetime.fromisoformat(max(VERIFIED.values())).replace(tzinfo=UTC)
    with (
        caplog.at_level(logging.WARNING, logger="paveo"),
        Paveo.from_policy(policy(), audit_path=tmp_path / "a.jsonl", now=lambda: today),
    ):
        pass

    assert "prices were last verified" not in caplog.text
