"""The two invariants the error taxonomy claims, checked rather than trusted.

Spec §8 asks for errors that say how to fix themselves and never carry a payload.
Both are easy to hold on the day they are written and easy to lose six months
later, so both are tests (Rule 5).
"""

from __future__ import annotations

import decimal
from decimal import Decimal

import pytest

from paveo import (
    BudgetExceeded,
    ConfigError,
    PaveoError,
    PolicyDenied,
    PolicyUnavailable,
    PricingUnknown,
)

SENTINEL = "patient-name-jane-doe-payload"


def test_every_error_is_catchable_as_one_thing() -> None:
    for error in (
        ConfigError("x.", remedy="y."),
        PolicyDenied(reason="r", rule="t.a.max", remedy="y."),
        BudgetExceeded(
            limit=Decimal("50"),
            spent=Decimal("10"),
            reserved=Decimal("5"),
            requested=Decimal("40"),
            remedy="y.",
        ),
        PolicyUnavailable("x.", remedy="y."),
        PricingUnknown(model="m", detail="d", remedy="y."),
    ):
        assert isinstance(error, PaveoError)


def test_every_message_says_how_to_fix_it() -> None:
    errors = (
        ConfigError("the policy is empty.", remedy="add an agent."),
        PolicyDenied(reason="tool_denied", rule="refund", remedy="remove the deny."),
        BudgetExceeded(
            limit=Decimal("50.00"),
            spent=Decimal("49.00"),
            reserved=Decimal("0.50"),
            requested=Decimal("1.00"),
            remedy="raise the ceiling.",
        ),
        PolicyUnavailable("the audit log is unwritable.", remedy="free some disk."),
        PricingUnknown(model="claude-x", detail="unknown model", remedy="update it."),
    )
    for error in errors:
        assert "Fix: " in str(error)
        assert error.remedy
        assert str(error).endswith(error.remedy)


def test_a_remedy_is_not_optional() -> None:
    """The type system, not a convention, is what stops an unhelpful error."""
    with pytest.raises(TypeError):
        PaveoError("something went wrong.")  # type: ignore[call-arg]


def test_policy_denied_carries_the_rule_and_not_the_value() -> None:
    error = PolicyDenied(
        reason="constraint_violated",
        rule="refund.amount_usd.max",
        remedy="the call must satisfy the max.",
    )
    assert error.rule == "refund.amount_usd.max"
    assert error.reason == "constraint_violated"
    assert "refund.amount_usd.max" in str(error)
    assert SENTINEL not in str(error)


def test_budget_exceeded_explains_the_arithmetic() -> None:
    error = BudgetExceeded(
        limit=Decimal("50.00"),
        spent=Decimal("30.00"),
        reserved=Decimal("5.00"),
        requested=Decimal("20.00"),
        remedy="raise agents['bot'].budget.limit_usd.",
    )
    assert error.limit == Decimal("50.00")
    assert error.spent == Decimal("30.00")
    assert error.reserved == Decimal("5.00")
    assert error.requested == Decimal("20.00")
    # The four numbers, and the one derived from them, all appear — so a refusal
    # can be understood without re-deriving it from the audit log.
    assert "15.00" in str(error)
    assert "20.00" in str(error)


def test_a_refusal_is_still_a_refusal_under_a_hostile_decimal_context() -> None:
    """Found by /security-review, 2026-09-23 (D35). The message does one
    subtraction; in the caller's context, with `Inexact` trapped and the
    precision lowered, that raised `decimal.Inexact` in place of the refusal,
    and nobody catching `PaveoError` to stop their agent would have caught it."""
    hostile = decimal.Context(prec=3, traps=[decimal.Inexact])

    with decimal.localcontext(hostile):
        error = BudgetExceeded(
            limit=Decimal("100"),
            spent=Decimal("99.99999999"),
            reserved=Decimal("0"),
            requested=Decimal("0.00000002"),
            remedy="raise agents['bot'].budget.limit_usd.",
        )

    assert isinstance(error, PaveoError)
    # Written as a plain decimal, not `1E-8`, which reads as an error to a person.
    assert "0.00000001 USD" in str(error)


def test_pricing_unknown_names_the_model() -> None:
    error = PricingUnknown(
        model="claude-not-in-the-table",
        detail="no entry in the price table",
        remedy="add it to prices.py or set a default base pair.",
    )
    assert error.model == "claude-not-in-the-table"
    assert "claude-not-in-the-table" in str(error)
