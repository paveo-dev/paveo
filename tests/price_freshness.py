"""The release gate for prices: run ``python -m tests.price_freshness``.

Exits non-zero if any provider's table was last verified more than 30 days ago,
so a stale price table cannot be released by accident (§12.5, D43). Runs the
real clock on purpose: it is a gate on today, not a test of the logic, which
`test_price_freshness.py` covers with fixed dates. `make release-check` runs it.
"""

from __future__ import annotations

import sys
from datetime import UTC, date, datetime

from paveo.prices import VERIFIED, stale_providers

RELEASE_WINDOW_DAYS = 30


def main(today: date | None = None) -> int:
    """0 if every table is inside the window, 1 if not. ``today`` for the tests."""
    day = datetime.now(UTC).date() if today is None else today
    stale = stale_providers(day, older_than_days=RELEASE_WINDOW_DAYS)
    if stale:
        for provider in stale:
            print(
                f"FAIL: {provider} prices last verified {VERIFIED[provider]}, over "
                f"{RELEASE_WINDOW_DAYS} days ago. Re-read them (docs/PRICES.md) "
                f"before releasing.",
                file=sys.stderr,
            )
        return 1
    print(
        f"ok: every provider's prices were verified within {RELEASE_WINDOW_DAYS} days"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
