"""The lock, and only the lock (§11, §4.9.1).

The guarantee that a ceiling cannot be exceeded is proved in ``test_budget.py``,
against the core, with no lock in the picture at all. So these tests ask one
question instead: **is the lock actually applied?** Eight threads against one
ceiling answer it under load, and a barrier answers it deterministically, and
the two fail in different ways — the first when the arithmetic races, the second
when two callers are simply inside the ledger at once. Both worlds get the same
questions, because one lock serves both (D33).

The rest is the wiring the core cannot do for itself: the clock it is not
allowed to read, and the ledger each agent gets.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from decimal import Decimal

import pytest

from paveo.budget import _PERIODS, Reservation, _BudgetCore
from paveo.errors import BudgetExceeded, ConfigError
from paveo.policy import BudgetPolicy
from paveo.stores import InMemoryBudgetStore
from paveo.stores.memory import _RESET

AGENT = "refund-bot"
OTHER_AGENT = "triage-bot"

LIMIT = Decimal("1.00")
DAILY = BudgetPolicy(period="day", limit_usd=LIMIT)

BEFORE_MIDNIGHT = datetime(2026, 9, 21, 23, 0, tzinfo=UTC)
AFTER_MIDNIGHT = datetime(2026, 9, 22, 0, 0, tzinfo=UTC)


class Clock:
    """A clock a test moves by hand, because a test cannot wait until midnight.

    The whole reason time is injected rather than read (Rule 14): §4.7 resets a
    daily ceiling at 00:00 UTC, and a store that called ``datetime.now()`` inline
    would have that logic exercised never.
    """

    def __init__(self, moment: datetime) -> None:
        self.moment = moment

    def __call__(self) -> datetime:
        return self.moment


def store(moment: datetime = BEFORE_MIDNIGHT) -> InMemoryBudgetStore:
    return InMemoryBudgetStore(now=Clock(moment))


def reserve(
    keeper: InMemoryBudgetStore, amount: str, *, agent_id: str = AGENT
) -> Reservation:
    return keeper.reserve(agent_id=agent_id, budget=DAILY, worst_case=Decimal(amount))


# --------------------------------------------------------------------------
# The lock
# --------------------------------------------------------------------------

THREADS = 8
CALLS_EACH = 25
HOLD = Decimal("0.05")
CHARGE = Decimal("0.01")


def spend_from_a_thread(keeper: InMemoryBudgetStore) -> int:
    """Reserve and settle until the attempts run out; return how many got in."""
    allowed = 0
    for _ in range(CALLS_EACH):
        try:
            held = keeper.reserve(agent_id=AGENT, budget=DAILY, worst_case=HOLD)
        except BudgetExceeded:
            continue
        keeper.settle(agent_id=AGENT, reservation_id=held.reservation_id, actual=CHARGE)
        allowed += 1
    return allowed


def test_eight_threads_against_one_ceiling_cannot_overspend_it() -> None:
    """The failure the library exists to prevent, and the one people describe.

    Eight workers, two hundred attempts, one dollar: settling every attempt would
    spend two. What the assertion checks is stronger than "the ceiling held" —
    a ledger that recorded nothing at all also never exceeds its limit (D24). So
    it asserts the *exact* remaining, which is only right if every allowed call
    was charged once and every hold came back.
    """
    keeper = store()
    start = threading.Barrier(THREADS)

    def spend() -> int:
        start.wait()
        return spend_from_a_thread(keeper)

    with ThreadPoolExecutor(max_workers=THREADS) as pool:
        allowed = sum(
            future.result() for future in [pool.submit(spend) for _ in range(THREADS)]
        )

    assert allowed < THREADS * CALLS_EACH, "nothing was ever refused; prove it again"
    remaining = keeper.remaining(agent_id=AGENT, budget=DAILY)
    # `remaining` is limit - spent - reserved, so this one equality says both
    # that spend is exact and that no hold leaked.
    assert remaining == LIMIT - CHARGE * allowed
    assert remaining >= 0


def from_a_thread(keeper: InMemoryBudgetStore, ready: threading.Barrier) -> None:
    ready.wait()
    reserve(keeper, "0.10")


def from_an_event_loop(keeper: InMemoryBudgetStore, ready: threading.Barrier) -> None:
    async def inside_the_loop() -> None:
        ready.wait()
        reserve(keeper, "0.10")

    asyncio.run(inside_the_loop())


@pytest.mark.parametrize(
    "second",
    [
        pytest.param(from_a_thread, id="two-threads"),
        pytest.param(from_an_event_loop, id="a-thread-and-a-running-loop"),
    ],
)
def test_two_callers_are_never_inside_the_ledger_at_once(
    monkeypatch: pytest.MonkeyPatch,
    second: Callable[[InMemoryBudgetStore, threading.Barrier], None],
) -> None:
    """The lock is held across the ledger call, not merely allocated.

    The test above races the arithmetic and would catch a missing lock the
    expensive way — eventually, on some interleaving, on some machine. This one
    asks the question directly: two callers meet at a barrier *inside* the
    ledger, and if the lock is held they cannot, so the barrier times out and
    breaks. A broken barrier is the pass.

    **The second case is the one §4.9 used to say was impossible:** a coroutine
    in a running event loop and a plain thread, excluded by the same
    ``threading.Lock``. It is what lets one store serve both worlds at once
    (D33), and it is the case the card's ``asyncio.Lock`` would have failed — a
    lock that is not thread-safe excludes nobody here.

    Both callers are released from a barrier of their own *before* either
    touches the store, so neither is still starting up when the race begins, and
    both are asserted to have reached the ledger.

    **What it does not prove**, stated because the previous version of this
    docstring claimed it did: arrival plus a broken barrier is strong evidence of
    exclusion, not proof of it. A thread descheduled for longer than the timeout
    would look the same. The eight-thread test above is what carries the
    guarantee; this one localises the failure.
    """
    ready = threading.Barrier(2)
    meeting = threading.Barrier(2, timeout=0.25)
    entered: list[str] = []
    met: list[str] = []
    ledger_reserve = _BudgetCore.try_reserve

    def meet_inside(self: _BudgetCore, worst_case: Decimal, now: datetime) -> object:
        entered.append(threading.current_thread().name)
        try:
            meeting.wait()
        except threading.BrokenBarrierError:
            pass
        else:
            met.append(threading.current_thread().name)
        return ledger_reserve(self, worst_case, now)

    monkeypatch.setattr(_BudgetCore, "try_reserve", meet_inside)
    keeper = store()

    with ThreadPoolExecutor(max_workers=2) as pool:
        racers = [from_a_thread, second]
        for future in [pool.submit(racer, keeper, ready) for racer in racers]:
            future.result()

    assert len(entered) == 2, "only one caller reached the ledger; nothing was tested"
    assert not met, "two callers were inside the ledger at once: the lock is not held"


# --------------------------------------------------------------------------
# Denial — the core returns it, the store raises it (§4.9.1)
# --------------------------------------------------------------------------


def test_a_call_that_would_breach_the_ceiling_raises_with_its_numbers() -> None:
    keeper = store()
    reserve(keeper, "0.60")

    with pytest.raises(BudgetExceeded) as refused:
        reserve(keeper, "0.50")

    assert refused.value.limit == LIMIT
    assert refused.value.spent == 0
    assert refused.value.reserved == Decimal("0.60")
    assert refused.value.requested == Decimal("0.50")


def test_a_shared_store_refuses_a_session_ceiling() -> None:
    """Found by review. D29 #3 said a session-scoped agent gets its own store and
    nothing enforced it, so a shared store turned a session ceiling into a
    process-lifetime one: the first session to exhaust it denied every session
    after, while the refusal text promised "a new session starts with a full one".
    A rule that depends on the caller remembering is the shape Rule 5 removes."""
    keeper = store()

    with pytest.raises(ConfigError, match="shared between sessions"):
        keeper.reserve(
            agent_id=AGENT,
            budget=BudgetPolicy(period="session", limit_usd=LIMIT),
            worst_case=Decimal("0.10"),
        )


def test_a_session_store_keeps_a_session_ceiling() -> None:
    """The other half: with its own store the ceiling is honoured, and it does
    not refill, because a session is not a clock window."""
    budget = BudgetPolicy(period="session", limit_usd=LIMIT)
    keeper = InMemoryBudgetStore(now=Clock(BEFORE_MIDNIGHT), one_session=True)
    held = keeper.reserve(agent_id=AGENT, budget=budget, worst_case=LIMIT)
    keeper.settle(agent_id=AGENT, reservation_id=held.reservation_id, actual=LIMIT)

    assert keeper.remaining(agent_id=AGENT, budget=budget) == 0
    with pytest.raises(BudgetExceeded):
        keeper.reserve(agent_id=AGENT, budget=budget, worst_case=Decimal("0.01"))


@pytest.mark.parametrize(
    ("period", "says"),
    [
        ("day", "00:00 UTC"),
        ("hour", "on the hour"),
        ("session", "never resets"),
    ],
)
def test_the_refusal_says_when_the_ceiling_it_names_comes_back(
    period: str, says: str
) -> None:
    """§8: an error message must say how to fix it, and which ceiling this is
    changes the answer. The ledger knows neither the period nor the agent."""
    budget = BudgetPolicy(period=period, limit_usd=LIMIT)
    keeper = InMemoryBudgetStore(
        now=Clock(BEFORE_MIDNIGHT), one_session=period == "session"
    )

    with pytest.raises(BudgetExceeded) as refused:
        keeper.reserve(agent_id=AGENT, budget=budget, worst_case=Decimal("2.00"))

    assert says in refused.value.remedy
    assert AGENT in refused.value.remedy


def test_a_refusal_holds_nothing_and_the_next_affordable_call_is_allowed() -> None:
    keeper = store()

    with pytest.raises(BudgetExceeded):
        reserve(keeper, "2.00")

    assert keeper.remaining(agent_id=AGENT, budget=DAILY) == LIMIT
    assert isinstance(reserve(keeper, "1.00"), Reservation)


# --------------------------------------------------------------------------
# One ledger per agent
# --------------------------------------------------------------------------


def test_each_agent_spends_its_own_ceiling() -> None:
    """A shared number would make a denial unattributable to anyone (§5)."""
    keeper = store()
    held = reserve(keeper, "1.00")
    keeper.settle(agent_id=AGENT, reservation_id=held.reservation_id, actual=LIMIT)

    assert keeper.remaining(agent_id=AGENT, budget=DAILY) == 0
    assert keeper.remaining(agent_id=OTHER_AGENT, budget=DAILY) == LIMIT
    assert isinstance(reserve(keeper, "1.00", agent_id=OTHER_AGENT), Reservation)


def test_an_agent_who_has_spent_nothing_has_the_whole_ceiling() -> None:
    keeper = store()

    assert keeper.remaining(agent_id=AGENT, budget=DAILY) == LIMIT


def test_a_second_ceiling_for_one_agent_is_refused_rather_than_ignored() -> None:
    """Two policies through one store: keeping the first would enforce a limit
    nobody can find in a file (§5.1)."""
    keeper = store()
    reserve(keeper, "0.10")

    with pytest.raises(ConfigError, match="already spending against"):
        keeper.reserve(
            agent_id=AGENT,
            budget=BudgetPolicy(period="day", limit_usd=Decimal("500.00")),
            worst_case=Decimal("0.10"),
        )


# --------------------------------------------------------------------------
# The clock the ledger is not allowed to read (Rule 14, §4.7)
# --------------------------------------------------------------------------


def test_a_daily_ceiling_refills_at_midnight_utc() -> None:
    """The store resolves the clock and passes the reading through, because only
    an operation carrying a time can notice the day has turned."""
    clock = Clock(BEFORE_MIDNIGHT)
    keeper = InMemoryBudgetStore(now=clock)
    held = reserve(keeper, "1.00")
    keeper.settle(agent_id=AGENT, reservation_id=held.reservation_id, actual=LIMIT)
    with pytest.raises(BudgetExceeded):
        reserve(keeper, "0.01")

    clock.moment = AFTER_MIDNIGHT

    assert keeper.remaining(agent_id=AGENT, budget=DAILY) == LIMIT
    assert isinstance(reserve(keeper, "1.00"), Reservation)


def test_the_default_clock_is_the_real_one() -> None:
    """`now=None` means the real clock, resolved here and not by the ledger."""
    keeper = InMemoryBudgetStore()
    held = reserve(keeper, "0.10")

    keeper.release(agent_id=AGENT, reservation_id=held.reservation_id)

    assert keeper.remaining(agent_id=AGENT, budget=DAILY) == LIMIT


@pytest.mark.parametrize(
    "reading",
    [
        pytest.param(datetime(2026, 9, 21, 23, 0), id="naive"),
        pytest.param("2026-09-21T23:00:00Z", id="not-a-datetime"),
    ],
)
def test_a_clock_that_cannot_be_accounted_is_wiring_and_says_so(
    reading: object,
) -> None:
    """ConfigError, not ValueError: the clock came from the caller, and someone
    catching PaveoError to stop their agent has to catch this too (§8)."""
    keeper = InMemoryBudgetStore(now=lambda: reading)

    with pytest.raises(ConfigError, match="aware datetime"):
        reserve(keeper, "0.10")


def test_a_clock_that_is_not_even_callable_is_still_wiring() -> None:
    """Found by review: a `TypeError` here escaped the documented taxonomy, so
    someone catching `PaveoError` to stop their agent would not have caught it."""
    keeper = InMemoryBudgetStore(now=datetime(2026, 9, 21, 23, 0, tzinfo=UTC))

    with pytest.raises(ConfigError, match="aware datetime"):
        reserve(keeper, "0.10")


def test_a_release_does_not_need_a_working_clock() -> None:
    """It runs in a `finally`, usually because something else has already failed.

    A release that can fail on its own account leaks the reservation, which is a
    denial of service against the customer's own agent (§4.2).
    """
    clock = Clock(BEFORE_MIDNIGHT)
    keeper = InMemoryBudgetStore(now=clock)
    held = reserve(keeper, "0.10")
    clock.moment = datetime(2026, 9, 21, 23, 30)  # naive: the clock breaks

    keeper.release(agent_id=AGENT, reservation_id=held.reservation_id)

    clock.moment = BEFORE_MIDNIGHT
    assert keeper.remaining(agent_id=AGENT, budget=DAILY) == LIMIT


# --------------------------------------------------------------------------
# Unwinding
# --------------------------------------------------------------------------


def test_settling_charges_the_actual_and_frees_the_rest() -> None:
    keeper = store()
    held = reserve(keeper, "0.80")

    keeper.settle(
        agent_id=AGENT, reservation_id=held.reservation_id, actual=Decimal("0.02")
    )

    assert keeper.remaining(agent_id=AGENT, budget=DAILY) == Decimal("0.98")


@pytest.mark.parametrize(
    "unwind",
    [
        pytest.param(
            lambda keeper: keeper.settle(
                agent_id=OTHER_AGENT, reservation_id="never-made", actual=CHARGE
            ),
            id="settle",
        ),
        pytest.param(
            lambda keeper: keeper.release(
                agent_id=OTHER_AGENT, reservation_id="never-made"
            ),
            id="release",
        ),
    ],
)
def test_unwinding_against_an_agent_who_holds_nothing_is_refused(
    unwind: Callable[[InMemoryBudgetStore], object],
) -> None:
    """Not a KeyError out of a dict: settling against the wrong agent would
    otherwise be indistinguishable from a bug inside the ledger itself."""
    keeper = store()
    reserve(keeper, "0.10")

    with pytest.raises(ValueError, match="not outstanding"):
        unwind(keeper)


def test_a_reservation_unwinds_exactly_once_through_the_store() -> None:
    keeper = store()
    held = reserve(keeper, "0.80")
    keeper.release(agent_id=AGENT, reservation_id=held.reservation_id)

    with pytest.raises(ValueError, match="not outstanding"):
        keeper.release(agent_id=AGENT, reservation_id=held.reservation_id)


# --------------------------------------------------------------------------
# Coroutines — the same lock, the same questions (§11, D33)
# --------------------------------------------------------------------------

COROUTINES = 32


@pytest.mark.parametrize(
    "threads",
    [
        pytest.param(0, id="coroutines-alone"),
        pytest.param(4, id="with-worker-threads-too"),
    ],
)
def test_coroutines_against_one_ceiling_cannot_overspend_it(threads: int) -> None:
    """§11's asyncio test, and the §10.11 case it used to rule out.

    Every coroutine holds its reservation across an ``await`` — the model call,
    in flight — so the ceiling is contested by holds other tasks have not yet
    settled: four sub-agents seeing the same headroom, in the world they
    actually run in. With worker threads, an async program's thread pool spends
    the same ceiling through the same store at the same time, and the books
    still come out exact to the eighth decimal.

    **What it cannot see**, stated because D24 and D29 both found tests that
    could not see their own subject: **a missing lock.** Measured with the lock
    swapped for a no-op, both cases passed 10 runs in 10. Coroutines alone
    cannot race inside a call — that is the point of D33, the loop is already
    the lock there — and four threads rarely race hard enough in a hundred
    attempts to show it. This test proves the books come out exact when both
    worlds spend at once; the barrier test above is what proves a coroutine and
    a thread are kept apart, and it caught the same removal 10 in 10.
    """
    keeper = store()
    in_flight = 0
    most_in_flight = 0

    async def agent() -> int:
        nonlocal in_flight, most_in_flight
        allowed = 0
        for _ in range(CALLS_EACH):
            try:
                held = keeper.reserve(agent_id=AGENT, budget=DAILY, worst_case=HOLD)
            except BudgetExceeded:
                await asyncio.sleep(0)
                continue
            in_flight += 1
            most_in_flight = max(most_in_flight, in_flight)
            await asyncio.sleep(0)
            in_flight -= 1
            keeper.settle(
                agent_id=AGENT, reservation_id=held.reservation_id, actual=CHARGE
            )
            allowed += 1
        return allowed

    async def program() -> int:
        workers = [
            asyncio.to_thread(spend_from_a_thread, keeper) for _ in range(threads)
        ]
        return sum(
            await asyncio.gather(*(agent() for _ in range(COROUTINES)), *workers)
        )

    allowed = asyncio.run(program())

    assert most_in_flight > 1, "no two holds were ever in flight; nothing contested"
    attempts = (COROUTINES + threads) * CALLS_EACH
    assert allowed < attempts, "nothing was ever refused; prove it again"
    remaining = keeper.remaining(agent_id=AGENT, budget=DAILY)
    assert remaining == LIMIT - CHARGE * allowed
    assert remaining >= 0


# --------------------------------------------------------------------------
# Seams (Rule 2)
# --------------------------------------------------------------------------


def test_every_period_the_ledger_keeps_can_be_explained_to_whoever_is_refused() -> None:
    """The period names now live in three files, and this is the third.

    A period with no remedy here would raise `KeyError` from inside the store at
    the worst possible moment — while explaining a refusal — instead of denying
    the call it was asked about.
    """
    assert set(_RESET) == _PERIODS
