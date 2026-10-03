"""The examples run, and show what the README says they show (Rule 8).

A README that promises `make example` and an example that no longer runs is an
overclaim a new user finds in their first minute. Each runs in a fresh
interpreter, as a user would run it.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent


def run(example: str) -> str:
    done = subprocess.run(  # noqa: S603 - our own interpreter, our own example
        [sys.executable, str(ROOT / "examples" / example)],
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
    )
    assert done.returncode == 0, done.stderr
    return done.stdout


def test_the_runaway_loop_is_refused_before_the_call_that_breaches_the_ceiling() -> (
    None
):
    out = run("runaway_loop.py")

    assert "REFUSED before it was sent" in out
    assert "chain intact: True" in out
    assert "0E-" not in out


def test_the_refund_bot_runs() -> None:
    assert "chain" in run("refund_bot.py").lower()
