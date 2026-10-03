"""The overhead gate's verdict, on fixed figures (Rule 17, D73).

The measuring is ``python -m tests.overhead``, run by ``make check`` on the real
clock. What is tested here is only whether a figure over its budget fails.
"""

from __future__ import annotations

from tests.overhead import BUDGET_US, over_budget


def test_figures_inside_every_budget_pass() -> None:
    assert over_budget(dict(BUDGET_US)) == []


def test_one_figure_over_its_budget_fails_and_is_named() -> None:
    medians = dict(BUDGET_US)
    medians["check_llm + record"] += 1

    failures = over_budget(medians)

    assert len(failures) == 1
    assert failures[0].startswith("check_llm + record:")
