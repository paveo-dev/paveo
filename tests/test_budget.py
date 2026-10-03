"""The budget core, proved once with no lock in the picture (§11, §4.9.1).

The property test below is the reason this file exists. Everything under it is a
worked example of one behaviour the property covers in general — kept because a
failing property tells you *that* the ledger is wrong at step 37 of a shrunk
trace, and a failing named test tells you *which rule* you broke.

The clock is injected everywhere, so the 00:00 UTC rollover is exercised in
microseconds rather than never (Rule 14).
"""

from __future__ import annotations

import decimal
import subprocess
import sys
from collections.abc import Callable
from datetime import UTC, date, datetime, timedelta, timezone
from decimal import ROUND_FLOOR, Decimal
from pathlib import Path

import pytest
from hypothesis import strategies as st
from hypothesis.stateful import RuleBasedStateMachine, invariant, precondition, rule

from paveo._policy_document import _PERIODS as POLICY_PERIODS
from paveo.budget import _PERIODS as BUDGET_PERIODS
from paveo.budget import BudgetDenial, Reservation, _BudgetCore

# A call costs some share of the ceiling — up to a fifth more than all of it.
# Drawn as a share rather than an absolute amount so the interesting region, just
# under and just over, is reached in a step or two whatever the ceiling is.
SHARE = st.integers(min_value=0, max_value=120)

# Less than the ledger's smallest unit, so most draws also exercise the round-up.
DUST = st.integers(min_value=0, max_value=999_999)

# Written out rather than imported: a model that borrows the implementation's
# constant cannot disagree with it, and disagreeing is the model's whole job.
QUANTUM = Decimal("0.00000001")

BEFORE_MIDNIGHT = datetime(2026, 9, 21, 23, 0, tzinfo=UTC)


def core(
    limit: str = "10.00", period: str = "day", *, real_ids: bool = False
) -> _BudgetCore:
    """A ledger whose reservation ids count up, so tests can name them."""
    counter = 0

    def new_id() -> str:
        nonlocal counter
        counter += 1
        return str(counter)

    return _BudgetCore(
        limit=Decimal(limit), period=period, new_id=None if real_ids else new_id
    )


def reserve(ledger: _BudgetCore, amount: str, at: datetime) -> Reservation:
    """Reserve and insist it was allowed, so a test reads as what it is about."""
    outcome = ledger.try_reserve(Decimal(amount), at)
    assert isinstance(outcome, Reservation), f"expected an allow, got {outcome}"
    return outcome


# --------------------------------------------------------------------------
# The property (§11): across any interleaving, the ceiling holds and the hold
# comes back.
# --------------------------------------------------------------------------


class Ledger(RuleBasedStateMachine):
    """Reserve, settle, fail and pass midnight, in any order Hypothesis likes.

    The trace assumes exactly one thing, which is the pricing table's promise and
    not this module's (§4.1): a call never costs more than it reserved. ``settle``
    draws its actual as a fraction of the reservation, so the mispriced call that
    §4.8.3 exists to make impossible never appears here.

    **It keeps its own books and compares, rather than only checking the ceiling
    held.** §11 asks for ``spent <= limit``, which a ledger that records nothing
    at all satisfies perfectly — measured, and it does: deleting the line that
    charges spend passes that inequality on every seed. So the model below adds
    up what it expects independently, from §4.7 rather than from the code, and
    the invariant is equality.
    """

    def __init__(self) -> None:
        super().__init__()
        self.limit = Decimal("10.00")
        self.moment = BEFORE_MIDNIGHT
        self.ledger = core(str(self.limit), "day")
        self.held: dict[str, tuple[Decimal, date]] = {}
        self.today = self.moment.date()
        self.expected_spent = Decimal(0)

    def _roll(self) -> None:
        """§4.7 in the model's own words: a new UTC day starts empty.

        Lazily, exactly as the ledger does — only an operation carrying a time can
        notice the day has turned, so ``wait`` alone must not move this either, or
        the two would disagree about a boundary neither has reached yet.
        """
        if self.moment.date() != self.today:
            self.today = self.moment.date()
            self.expected_spent = Decimal(0)

    @rule(share=SHARE, dust=DUST)
    def reserve(self, share: int, dust: int) -> None:
        self._roll()
        worst_case = self.limit * share / 100 + Decimal(dust) / Decimal(10**12)

        outcome = self.ledger.try_reserve(worst_case, self.moment)

        if isinstance(outcome, BudgetDenial):
            assert outcome.spent + outcome.reserved + outcome.requested > outcome.limit
            return
        assert outcome.worst_case >= worst_case, "a hold must cover its call (§4.6)"
        self.held[outcome.reservation_id] = (outcome.worst_case, self.today)

    @precondition(lambda self: bool(self.held))
    @rule(
        pick=st.integers(min_value=0), percent=st.integers(min_value=0, max_value=100)
    )
    def settle(self, pick: int, percent: int) -> None:
        """The call returned, having cost some fraction of what it might have."""
        self._roll()
        reservation_id = sorted(self.held)[pick % len(self.held)]
        worst_case, reserved_on = self.held.pop(reservation_id)
        actual = (worst_case * percent / 100).quantize(QUANTUM, rounding=ROUND_FLOOR)

        self.ledger.settle(reservation_id, actual, self.moment)

        if reserved_on == self.today:
            self.expected_spent += actual

    @precondition(lambda self: bool(self.held))
    @rule(pick=st.integers(min_value=0))
    def fail(self, pick: int) -> None:
        """The call raised, so nothing was consumed and the hold comes back (§4.2)."""
        reservation_id = sorted(self.held)[pick % len(self.held)]
        released = self.ledger.release(reservation_id)
        assert released == self.held.pop(reservation_id)[0]

    @rule(minutes=st.integers(min_value=0, max_value=3000))
    def wait(self, minutes: int) -> None:
        """Enough of a range to cross midnight, twice, from where we start."""
        self.moment += timedelta(minutes=minutes)

    @invariant()
    def the_ceiling_holds(self) -> None:
        """§11, and the reason any of this exists."""
        assert self.ledger.spent <= self.limit
        assert self.ledger.spent + self.ledger.reserved <= self.limit

    @invariant()
    def the_books_agree(self) -> None:
        assert self.ledger.spent == self.expected_spent
        assert self.ledger.reserved == sum(
            (worst_case for worst_case, _ in self.held.values()), Decimal(0)
        )

    def teardown(self) -> None:
        for reservation_id in list(self.held):
            self.ledger.release(reservation_id)
            del self.held[reservation_id]
        assert self.ledger.reserved == 0, "a reservation leaked"


TestLedger = Ledger.TestCase


# --------------------------------------------------------------------------
# Reserve
# --------------------------------------------------------------------------


def test_a_reservation_holds_the_rounded_up_amount() -> None:
    """§4.6: the fraction is held too, so the ceiling is protected by it."""
    ledger = core()

    held = reserve(ledger, "0.000000001", BEFORE_MIDNIGHT)

    assert held.worst_case == Decimal("0.00000001")
    assert ledger.reserved == Decimal("0.00000001")


def test_a_call_that_would_breach_the_ceiling_is_refused_with_its_numbers() -> None:
    ledger = core("1.00")
    reserve(ledger, "0.60", BEFORE_MIDNIGHT)

    outcome = ledger.try_reserve(Decimal("0.50"), BEFORE_MIDNIGHT)

    assert outcome == BudgetDenial(
        limit=Decimal("1.00"),
        spent=Decimal(0),
        reserved=Decimal("0.60"),
        requested=Decimal("0.50"),
    )


def test_a_refusal_holds_nothing() -> None:
    """A denial that left a hold behind would deny the next call too, forever."""
    ledger = core("1.00")

    assert isinstance(
        ledger.try_reserve(Decimal("2.00"), BEFORE_MIDNIGHT), BudgetDenial
    )

    assert ledger.reserved == 0
    assert isinstance(ledger.try_reserve(Decimal("1.00"), BEFORE_MIDNIGHT), Reservation)


def test_the_ceiling_counts_calls_in_flight_not_just_settled_spend() -> None:
    """The concurrency case §4.1 exists for: four sub-agents, one ceiling."""
    ledger = core("1.00")
    for _ in range(4):
        reserve(ledger, "0.25", BEFORE_MIDNIGHT)

    outcome = ledger.try_reserve(Decimal("0.01"), BEFORE_MIDNIGHT)

    assert isinstance(outcome, BudgetDenial)
    assert ledger.spent == 0


def test_reserving_exactly_the_ceiling_is_allowed() -> None:
    """The limit is what may be spent, not what may not be reached."""
    ledger = core("1.00")

    assert isinstance(ledger.try_reserve(Decimal("1.00"), BEFORE_MIDNIGHT), Reservation)


# --------------------------------------------------------------------------
# Settle and release
# --------------------------------------------------------------------------


def test_settling_charges_the_actual_and_frees_the_rest() -> None:
    ledger = core()
    held = reserve(ledger, "2.00", BEFORE_MIDNIGHT)

    ledger.settle(held.reservation_id, Decimal("0.125"), BEFORE_MIDNIGHT)

    assert ledger.spent == Decimal("0.125")
    assert ledger.reserved == 0
    assert ledger.remaining(BEFORE_MIDNIGHT) == Decimal("9.875")


def test_settling_rounds_the_charge_up() -> None:
    ledger = core()
    held = reserve(ledger, "2.00", BEFORE_MIDNIGHT)

    ledger.settle(held.reservation_id, Decimal("0.000000001"), BEFORE_MIDNIGHT)

    assert ledger.spent == Decimal("0.00000001")


def test_releasing_gives_back_exactly_what_was_held() -> None:
    """D23 #3: re-rounding a release strands a sliver in `reserved` forever."""
    ledger = core()
    held = reserve(ledger, "0.000000001", BEFORE_MIDNIGHT)

    released = ledger.release(held.reservation_id)

    assert released == held.worst_case == Decimal("0.00000001")
    assert ledger.reserved == 0


def test_a_mispriced_call_is_recorded_in_full_rather_than_clamped() -> None:
    """The ledger tells the truth and then denies everything (§4.8.3).

    Only a stale price table can produce this, and hiding it would leave a
    breached ceiling looking exactly like an intact one.
    """
    ledger = core("1.00")
    held = reserve(ledger, "0.50", BEFORE_MIDNIGHT)

    ledger.settle(held.reservation_id, Decimal("4.00"), BEFORE_MIDNIGHT)

    assert ledger.spent == Decimal("4.00")
    assert ledger.remaining(BEFORE_MIDNIGHT) == Decimal("-3.00")
    assert isinstance(
        ledger.try_reserve(Decimal("0.01"), BEFORE_MIDNIGHT), BudgetDenial
    )


@pytest.mark.parametrize(
    "unwind",
    [
        pytest.param(
            lambda ledger, held: ledger.settle(held, Decimal("0.10"), BEFORE_MIDNIGHT),
            id="settle",
        ),
        pytest.param(lambda ledger, held: ledger.release(held), id="release"),
    ],
)
def test_a_reservation_unwinds_exactly_once(
    unwind: Callable[[_BudgetCore, str], object],
) -> None:
    """Settling twice would charge the ceiling twice for one call."""
    ledger = core()
    held = reserve(ledger, "1.00", BEFORE_MIDNIGHT)
    ledger.settle(held.reservation_id, Decimal("0.10"), BEFORE_MIDNIGHT)

    with pytest.raises(ValueError, match="not outstanding"):
        unwind(ledger, held.reservation_id)

    assert ledger.spent == Decimal("0.10")


def test_an_unknown_reservation_is_refused_rather_than_ignored() -> None:
    ledger = core()

    with pytest.raises(ValueError, match="not outstanding"):
        ledger.release("never-made")


# --------------------------------------------------------------------------
# Period boundaries (§4.7) — the whole reason the clock is injected
# --------------------------------------------------------------------------


def test_a_daily_ceiling_refills_at_midnight_utc() -> None:
    ledger = core("1.00", "day")
    held = reserve(ledger, "1.00", BEFORE_MIDNIGHT)
    ledger.settle(held.reservation_id, Decimal("1.00"), BEFORE_MIDNIGHT)
    assert isinstance(
        ledger.try_reserve(Decimal("0.01"), BEFORE_MIDNIGHT), BudgetDenial
    )

    after_midnight = datetime(2026, 9, 22, 0, 0, tzinfo=UTC)

    assert ledger.remaining(after_midnight) == Decimal("1.00")
    assert isinstance(ledger.try_reserve(Decimal("1.00"), after_midnight), Reservation)


def test_an_hourly_ceiling_refills_on_the_hour() -> None:
    ledger = core("1.00", "hour")
    held = reserve(ledger, "1.00", BEFORE_MIDNIGHT)
    ledger.settle(held.reservation_id, Decimal("1.00"), BEFORE_MIDNIGHT)

    assert ledger.remaining(datetime(2026, 9, 21, 23, 59, tzinfo=UTC)) == 0
    assert ledger.remaining(datetime(2026, 9, 22, 0, 0, tzinfo=UTC)) == Decimal("1.00")


def test_a_session_ceiling_never_refills() -> None:
    """`session` is the lifetime of the context manager, not a clock window."""
    ledger = core("1.00", "session")
    held = reserve(ledger, "1.00", BEFORE_MIDNIGHT)
    ledger.settle(held.reservation_id, Decimal("1.00"), BEFORE_MIDNIGHT)

    a_year_later = BEFORE_MIDNIGHT + timedelta(days=365)

    assert ledger.remaining(a_year_later) == 0


def test_a_call_that_outlives_its_window_is_not_charged_to_the_next_one() -> None:
    """It was admitted against yesterday's ceiling, and yesterday is closed."""
    ledger = core("1.00", "day")
    held = reserve(ledger, "1.00", BEFORE_MIDNIGHT)

    after_midnight = datetime(2026, 9, 22, 0, 30, tzinfo=UTC)
    ledger.settle(held.reservation_id, Decimal("0.90"), after_midnight)

    assert ledger.spent == 0
    assert ledger.reserved == 0
    assert ledger.remaining(after_midnight) == Decimal("1.00")


def test_a_breach_survives_the_period_boundary_that_its_call_did_not() -> None:
    """The part nobody authorised follows the money into the live window.

    Found by review. D24 #1 (attribute a straddling call to the window that
    admitted it) and D24 #5 (record a mispriced call in full) were decided apart
    and, together, erased a real overspend at midnight: a $400 call settled at
    00:01 left `spent` at zero and the ceiling refilled. See D25.
    """
    ledger = core("1.00", "day")
    held = reserve(ledger, "0.50", datetime(2026, 9, 21, 23, 59, tzinfo=UTC))

    after_midnight = datetime(2026, 9, 22, 0, 1, tzinfo=UTC)
    ledger.settle(held.reservation_id, Decimal("400.00"), after_midnight)

    # $0.50 was yesterday's to authorise. The other $399.50 was nobody's.
    assert ledger.spent == Decimal("399.50")
    assert isinstance(ledger.try_reserve(Decimal("0.01"), after_midnight), BudgetDenial)


def test_a_call_that_stayed_within_its_hold_is_forgiven_at_the_boundary() -> None:
    """The other half of the same rule: no excess, nothing carried forward."""
    ledger = core("1.00", "day")
    held = reserve(ledger, "0.50", datetime(2026, 9, 21, 23, 59, tzinfo=UTC))

    after_midnight = datetime(2026, 9, 22, 0, 1, tzinfo=UTC)
    ledger.settle(held.reservation_id, Decimal("0.50"), after_midnight)

    assert ledger.spent == 0
    assert ledger.remaining(after_midnight) == Decimal("1.00")


def test_a_call_in_flight_across_midnight_still_holds_todays_headroom() -> None:
    """Safe direction: the hold stands until we know what the call cost."""
    ledger = core("1.00", "day")
    reserve(ledger, "0.80", BEFORE_MIDNIGHT)

    after_midnight = datetime(2026, 9, 22, 0, 30, tzinfo=UTC)

    assert ledger.remaining(after_midnight) == Decimal("0.20")


def test_a_clock_stepped_backwards_does_not_refill_the_ceiling() -> None:
    """An NTP correction must not hand out a second day's budget."""
    ledger = core("1.00", "day")
    after_midnight = datetime(2026, 9, 22, 0, 30, tzinfo=UTC)
    held = reserve(ledger, "1.00", after_midnight)
    ledger.settle(held.reservation_id, Decimal("1.00"), after_midnight)

    assert ledger.remaining(BEFORE_MIDNIGHT) == 0


def test_the_window_is_utc_and_not_wherever_the_server_is_standing() -> None:
    """Same instant, written in another zone: same window, no refill."""
    ledger = core("1.00", "day")
    held = reserve(ledger, "1.00", BEFORE_MIDNIGHT)
    ledger.settle(held.reservation_id, Decimal("1.00"), BEFORE_MIDNIGHT)

    # 04:30 on the 22nd in +05:30 is 23:00 on the 21st in UTC — still yesterday.
    same_instant_elsewhere = datetime(
        2026, 9, 22, 4, 30, tzinfo=timezone(timedelta(hours=5, minutes=30))
    )

    assert ledger.remaining(same_instant_elsewhere) == 0


# --------------------------------------------------------------------------
# Inputs no correct caller produces
# --------------------------------------------------------------------------


@pytest.mark.parametrize("period", sorted(BUDGET_PERIODS))
def test_a_naive_datetime_is_refused_rather_than_read_as_local_time(
    period: str,
) -> None:
    """Every period, not two of three.

    Found by review: `session` has no window to truncate, so it returned before
    the clock was ever looked at and took a naive datetime happily. A guard that
    depends on which branch you took is the kind Rule 5 exists to remove.
    """
    ledger = core(period=period)

    with pytest.raises(ValueError, match="aware datetime"):
        ledger.try_reserve(Decimal("0.01"), datetime(2026, 9, 21, 23, 0))


@pytest.mark.parametrize(
    ("bad", "complaint"),
    [
        ("naive-clock", "aware datetime"),
        ("unaccountable-amount", "cannot be accounted"),
    ],
)
def test_a_refused_settle_leaves_the_reservation_where_it_was(
    bad: str, complaint: str
) -> None:
    """Half-applied is worse than refused: the hold would go back unspent.

    Found by re-reading the diff cold — `settle` took the reservation out before
    it had checked the clock, so a naive datetime released the hold and then
    raised, and the call it was holding for was never charged to anything.
    """
    ledger = core("1.00")
    held = reserve(ledger, "0.50", BEFORE_MIDNIGHT)
    naive = bad == "naive-clock"

    with pytest.raises(ValueError, match=complaint):
        ledger.settle(
            held.reservation_id,
            Decimal("0.10") if naive else Decimal("NaN"),
            datetime(2026, 9, 21, 23, 30) if naive else BEFORE_MIDNIGHT,
        )

    assert ledger.reserved == Decimal("0.50")
    assert ledger.spent == 0
    ledger.settle(held.reservation_id, Decimal("0.10"), BEFORE_MIDNIGHT)
    assert ledger.spent == Decimal("0.10")


@pytest.mark.parametrize("amount", ["NaN", "Infinity", "-1"])
def test_an_amount_that_cannot_be_accounted_is_refused(amount: str) -> None:
    """NaN is the one that matters: every comparison against it is False, so it
    would be *admitted* and then sit in `reserved` forever."""
    ledger = core()

    with pytest.raises(ValueError, match="cannot be accounted"):
        ledger.try_reserve(Decimal(amount), BEFORE_MIDNIGHT)


@pytest.mark.parametrize(
    ("amount", "complaint"),
    [
        pytest.param(0.5, "cannot be accounted", id="a-float-from-an-untyped-caller"),
        pytest.param(Decimal("1E+22"), "too large", id="more-than-money-goes-to"),
    ],
)
def test_an_amount_the_ledger_cannot_hold_raises_what_the_contract_promises(
    amount: object, complaint: str
) -> None:
    """Not AttributeError, and not a bare decimal.InvalidOperation.

    The store above translates a denial and a ValueError. Anything else it passes
    straight through to the customer's agent, which is where the contract has to
    hold rather than where it is documented.
    """
    ledger = _BudgetCore(limit=Decimal("1e30"), period="day")

    with pytest.raises(ValueError, match=complaint):
        ledger.try_reserve(amount, BEFORE_MIDNIGHT)  # type: ignore[arg-type]


# The public operations of the core. Written out so that a new one fails the
# guard below until it is added to the hostile-context test beside it (D35).
LEDGER_OPERATIONS = {"try_reserve", "settle", "release", "remaining"}


def test_every_ledger_operation_is_in_the_hostile_context_test() -> None:
    """Rule 5, found by /code-review: the private context is a `with` in each
    method body, so a new method written without one would bring D35's overspend
    back and no test would notice. This fails first, and says where to go."""
    public = {
        name
        for name, member in vars(_BudgetCore).items()
        if not name.startswith("_") and callable(member)
    }

    assert public == LEDGER_OPERATIONS, (
        "a public operation was added to or removed from _BudgetCore: run it "
        "under the hostile contexts in the test below, then update this set"
    )


@pytest.mark.parametrize(
    "callers",
    [
        pytest.param(decimal.Context(prec=3, traps=[]), id="traps-switched-off"),
        pytest.param(decimal.Context(prec=10), id="precision-lowered"),
        pytest.param(
            decimal.Context(prec=10, rounding=ROUND_FLOOR), id="rounding-down"
        ),
    ],
)
def test_the_callers_decimal_context_cannot_move_the_ceiling(
    callers: decimal.Context,
) -> None:
    """Found by /security-review on 2026-09-23, reproduced before fixing (D35).

    `decimal`'s context belongs to the calling thread or task, and anything in
    the customer's process can change it. With the traps off, an amount that
    cannot be quantised became NaN and every comparison admitted it: five $10.50
    holds against a $1.00 ceiling, and `reserved` NaN for ever. With only the
    precision lowered, the sum was rounded before it met the ceiling, and a call
    taking the true total to 100.00000001 against 100 was admitted.

    Every operation runs here, including the settle that crosses midnight, whose
    `charge - hold` is its own subtraction. Amounts carry more digits than any of
    these contexts can hold, so arithmetic that escaped the ledger's own context
    would round or turn to NaN and fail an equality below.
    """
    after_midnight = BEFORE_MIDNIGHT + timedelta(hours=1, minutes=30)
    with decimal.localcontext(callers):
        ledger = _BudgetCore(limit=Decimal("100"), period="day")
        held = reserve(ledger, "99.99999999", BEFORE_MIDNIGHT)
        ledger.settle(held.reservation_id, Decimal("99.99999999"), BEFORE_MIDNIGHT)
        over_by_one_unit = ledger.try_reserve(Decimal("0.00000002"), BEFORE_MIDNIGHT)
        far_over = ledger.try_reserve(Decimal("10.50"), BEFORE_MIDNIGHT)
        exactly_fits = reserve(ledger, "0.00000001", BEFORE_MIDNIGHT)
        released = ledger.release(exactly_fits.reservation_id)
        last_unit = ledger.remaining(BEFORE_MIDNIGHT)

        straddler = _BudgetCore(limit=Decimal("1"), period="day")
        late = reserve(straddler, "0.30000001", BEFORE_MIDNIGHT)
        straddler.settle(late.reservation_id, Decimal("0.50000003"), after_midnight)
        next_day = straddler.remaining(after_midnight)

    assert isinstance(over_by_one_unit, BudgetDenial)
    assert isinstance(far_over, BudgetDenial)
    assert released == Decimal("0.00000001")
    assert ledger.reserved == 0
    assert last_unit == Decimal("0.00000001")
    # Only the unauthorised excess follows the money into the new day (D25).
    assert next_day == Decimal("1") - Decimal("0.20000002")


def test_a_tampered_default_context_does_not_reach_the_ledger() -> None:
    """Found by /code-review (D35). A field left out of the ledger's context is
    copied from `decimal.DefaultContext` at import, and the customer's process
    can change that too. With `Emax` lowered, a settle raised `Overflow` after
    the hold was given back, and the charge was lost. A fresh interpreter, so
    the tampering happens before paveo is imported, as it would in real use."""
    script = f"""
import sys; sys.path.insert(0, {str(Path(__file__).parent.parent / "src")!r})
import decimal; decimal.DefaultContext.Emax = 3
from datetime import UTC, datetime
from decimal import Decimal
from paveo.budget import _BudgetCore
now = datetime(2026, 9, 21, 23, 0, tzinfo=UTC)
ledger = _BudgetCore(limit=Decimal("20000"), period="day")
for amount in ("5000", "6000"):
    held = ledger.try_reserve(Decimal(amount), now)
    ledger.settle(held.reservation_id, Decimal(amount), now)
assert ledger.spent == Decimal("11000"), ledger.spent
assert ledger.reserved == 0, ledger.reserved
"""
    done = subprocess.run(  # noqa: S603 - our own interpreter, fixed script
        [sys.executable, "-c", script], capture_output=True, text=True, check=False
    )

    assert done.returncode == 0, done.stderr


def test_a_repeated_reservation_id_is_refused() -> None:
    """Two reservations sharing an id would settle as one and lose a hold."""
    ledger = _BudgetCore(limit=Decimal("10"), period="day", new_id=lambda: "same")
    reserve(ledger, "1.00", BEFORE_MIDNIGHT)

    with pytest.raises(ValueError, match="already outstanding"):
        ledger.try_reserve(Decimal("1.00"), BEFORE_MIDNIGHT)


def test_an_unknown_period_is_refused_at_construction() -> None:
    with pytest.raises(ValueError, match="period is 'week'"):
        _BudgetCore(limit=Decimal("10"), period="week")


# --------------------------------------------------------------------------
# Seams (Rule 2)
# --------------------------------------------------------------------------


def test_every_period_a_policy_may_declare_has_a_window() -> None:
    """The period names live in two files. This is what stops them drifting."""
    assert BUDGET_PERIODS == POLICY_PERIODS


def test_reservation_ids_do_not_repeat_across_ledgers() -> None:
    """D23 #2 correlates two audit records through this id, so it must be unique
    across every ledger and every restart that shares one log — not just one."""
    first = core(real_ids=True)
    second = core(real_ids=True)

    minted = {
        reserve(ledger, "0.01", BEFORE_MIDNIGHT).reservation_id
        for ledger in (first, second)
        for _ in range(50)
    }

    assert len(minted) == 100
