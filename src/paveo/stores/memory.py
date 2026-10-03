"""The store v1 ships: one process, one lock, ledgers in a dict (§4.9.1, D33).

``_BudgetCore`` does the arithmetic and is deliberately not safe to share. This
is the object that makes it safe: a ``threading.Lock`` held across every call
into a ledger, so the check and the hold that follows it cannot be split by
another caller. Four sub-agents that each read the same headroom and are all
admitted is the failure this exists to prevent, and it is the one people running
agents described first.

**That is the whole of the concurrency story here.** The guarantee itself —
that a ceiling cannot be exceeded by any interleaving — was proved against the
core with no lock in the picture at all (§11, §4.9.1), so this file needs only
to be a lock that is actually applied.

**One ledger per agent**, created on first use from the ceiling the policy
declares for that agent (§5). Agents do not share a number: a shared ledger
would make a denial unattributable and is not what the policy file says.

**A ``session`` ceiling needs a store of its own.** §4.7 makes it the lifetime
of the context manager, and the ledgers here outlive any one session. Giving a
session-scoped agent its own store is the caller's decision, and a shared store
refuses a session ceiling rather than letting it be forgotten (D31).

**Thread safety, and coroutine safety, from the same lock.**
``InMemoryBudgetStore`` is safe to share across threads, across coroutines, and
across both at once (§4.9, D33). Every method takes the lock for the whole
read-decide-write sequence, and nothing it returns is mutable. One lock serves
both worlds because nothing inside it can await: the ledger is arithmetic with no
I/O, and every method here is a plain function. Inside one event loop a task runs
until it awaits, so no other task can reach the middle of a call; across threads
the lock excludes; and a lock never held across an ``await`` cannot deadlock a
loop, because that takes a task suspending while it holds one.

**What it costs a coroutine, measured rather than assumed (D33):** it waits while
a thread holds the lock. A reserve and settle from a coroutine takes about three
microseconds when nothing contends, and around a millisecond at the 99th
percentile when eight threads do nothing but hammer one ceiling. It is the same
trade §4.9.3 makes for the audit log, whose lock is held across a disk write.

**One store per ceiling, in either world.** Two store objects keep two sets of
books, and each will let the agent spend the whole ceiling (§10.11).

Its lock is **separate from the audit log's** and must stay that way (§6): they
protect different invariants, and merging them would put a disk write inside the
budget's critical section.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal

from paveo.budget import BudgetDenial, Reservation, _as_utc, _BudgetCore
from paveo.errors import BudgetExceeded, ConfigError
from paveo.policy import BudgetPolicy

# What a caller can do about a full ceiling depends on which one it is, and the
# ledger knows about none of this (§4.7). Written as sentences because a remedy
# is read by a person at three in the morning.
_RESET: Mapping[str, str] = {
    "day": "this ceiling resets at 00:00 UTC",
    "hour": "this ceiling resets on the hour, UTC",
    "session": (
        "a session ceiling never resets — it is the lifetime of this session, so "
        "a new session starts with a full one"
    ),
}


def _utc_now() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True, slots=True)
class _Ledger:
    """One agent's ceiling and the numbers standing against it.

    The ``budget`` is kept beside the core so that a second ceiling arriving for
    the same agent is refused rather than silently ignored. The core itself does
    not carry it: it is told a limit once and has no use for the policy object.
    """

    budget: BudgetPolicy
    core: _BudgetCore


class InMemoryBudgetStore:
    """Every agent's ceiling, in this process, behind one lock.

    Safe to share across threads and coroutines, including both at once. See the
    module docstring for what makes that true and what it costs.
    """

    def __init__(
        self,
        *,
        now: Callable[[], datetime] | None = None,
        one_session: bool = False,
    ) -> None:
        """``now`` is injected so §4.7's period reset is testable (Rule 14).

        ``one_session`` says this store belongs to a single session and dies with
        it. It exists because §4.7 makes a ``session`` ceiling the lifetime of the
        context manager, and a store shared between sessions cannot honour that:
        its ledgers outlive every session, so the ceiling silently becomes a
        process-lifetime one and the agent is denied forever after the first
        session exhausts it. **A shared store refuses a ``session`` budget rather
        than pretending**, which is D29 #3 made mechanical instead of remembered
        (Rule 5).

        The store resolves the clock, once per operation, and hands the reading
        to the core — which has none of its own, so a test can cross midnight in
        microseconds instead of never. ``None`` means the real clock, so a caller
        passing one through does not have to import our default.

        **The clock is read before the lock is taken, never inside it.** ``now``
        is a callable the customer supplied, and running arbitrary code inside
        the critical section of every call their agent makes is how a slow clock
        becomes our outage. The cost is that two threads at a period boundary can
        reach the ledger with their readings in the other order — which changes
        nothing, because a window never moves backwards (D24 #2) and both calls
        are still checked against a live ceiling.
        """
        self._now = _utc_now if now is None else now
        self._one_session = one_session
        self._lock = threading.Lock()
        self._ledgers: dict[str, _Ledger] = {}

    def reserve(
        self, *, agent_id: str, budget: BudgetPolicy, worst_case: Decimal
    ) -> Reservation:
        """Hold ``worst_case`` against the agent's ceiling. Raises if it does not fit.

        The refusal is raised here and nowhere else (§4.9.1): the core returns a
        value so a property test can inspect thousands of outcomes without a
        ``try``, and the caller gets an exception so it cannot forget to check
        one.
        """
        moment = self._reading()
        with self._lock:
            outcome = self._ledger(agent_id, budget).try_reserve(worst_case, moment)
        if isinstance(outcome, BudgetDenial):
            raise BudgetExceeded(
                limit=outcome.limit,
                spent=outcome.spent,
                reserved=outcome.reserved,
                requested=outcome.requested,
                remedy=(
                    f"{_RESET[budget.period]}. Until then, let the calls already "
                    f"in flight finish, or raise "
                    f"agents[{agent_id!r}].budget.limit_usd in the policy if the "
                    f"ceiling is wrong."
                ),
            )
        return outcome

    def settle(self, *, agent_id: str, reservation_id: str, actual: Decimal) -> None:
        """Replace the hold with what the call really cost (§4.2)."""
        moment = self._reading()
        with self._lock:
            core = self._holder(agent_id, reservation_id, "settle")
            core.settle(reservation_id, actual, moment)

    def release(self, *, agent_id: str, reservation_id: str) -> None:
        """Give back a hold whose call spent nothing (§4.2).

        Reads no clock, deliberately. This runs in a ``finally`` on every exit
        path, including the one taken because something else already failed, and
        a release that can fail on its own account leaks the reservation — a
        denial of service against the customer's agent, which is a bug of the
        same severity as an overspend.
        """
        with self._lock:
            core = self._holder(agent_id, reservation_id, "release")
            core.release(reservation_id)

    def remaining(self, *, agent_id: str, budget: BudgetPolicy) -> Decimal:
        """``limit - spent - reserved`` (D23 #6).

        Takes the lock like everything else: the ledger rolls its period on the
        operations that carry a time, so this one writes as well as reads and
        does not belong outside the lock (§4.7).
        """
        moment = self._reading()
        with self._lock:
            return self._ledger(agent_id, budget).remaining(moment)

    def _ledger(self, agent_id: str, budget: BudgetPolicy) -> _BudgetCore:
        """The agent's ledger, opened on first use. Called under the lock.

        **The caller has already established that the policy declares this
        agent** — a store holds no policy and cannot check it (§5.1 denies an
        undeclared agent before any of this is reached). That contract is also
        what bounds this dict: one ledger per agent the policy file names.

        A second, different ceiling for an agent this store is already keeping
        books for is refused. Policy is immutable once loaded and a reload builds
        a new object (§5.1), so this means one store is serving two policies —
        and quietly keeping the first ceiling would enforce a limit nobody in the
        file can see.
        """
        if budget.period == "session" and not self._one_session:
            raise ConfigError(
                f"agent {agent_id!r} declares a session ceiling, and this store "
                f"is shared between sessions, so its ledgers outlive every one of "
                f"them.",
                remedy=(
                    "give a session-scoped agent a store of its own: "
                    "`InMemoryBudgetStore(one_session=True)`, built when the "
                    "session opens and discarded when it closes. Sharing one "
                    "would make the ceiling last as long as the process, so the "
                    "first session to exhaust it would deny every session after."
                ),
            )
        existing = self._ledgers.get(agent_id)
        if existing is None:
            opened = _Ledger(
                budget=budget,
                core=_BudgetCore(limit=budget.limit_usd, period=budget.period),
            )
            self._ledgers[agent_id] = opened
            return opened.core
        if existing.budget != budget:
            raise ConfigError(
                f"agent {agent_id!r} is already spending against a "
                f"{existing.budget.period} ceiling of {existing.budget.limit_usd} "
                f"USD, and this call declares {budget.period} "
                f"{budget.limit_usd} USD.",
                remedy=(
                    "one store serves one loaded policy. Reloading a policy "
                    "builds a new Policy object, so give it a new store rather "
                    "than sharing this one."
                ),
            )
        return existing.core

    def _holder(self, agent_id: str, reservation_id: str, action: str) -> _BudgetCore:
        """The ledger holding this reservation. Called under the lock.

        An agent with no ledger cannot be holding one, so this says so in the
        words the core uses for the same mistake rather than raising ``KeyError``
        from inside a dict.
        """
        existing = self._ledgers.get(agent_id)
        if existing is None:
            raise ValueError(
                f"cannot {action} reservation {reservation_id!r}: it is not "
                f"outstanding, because agent {agent_id!r} has not reserved "
                f"anything from this store. Settle and release a reservation "
                f"against the agent that made it."
            )
        return existing.core

    def _reading(self) -> datetime:
        """What the clock says, refused here if it cannot be accounted.

        The core raises ``ValueError`` for a naive datetime, which is right for
        it — that is our own code being wrong. At this boundary the clock came
        from the customer's wiring, and §8 has a name for wiring that is wrong.
        Translated here so that a caller who catches ``PaveoError`` around their
        agent, which is the documented way to stop it, catches this too.
        """
        try:
            return _as_utc(self._now())
        except (AttributeError, TypeError, ValueError) as e:
            raise ConfigError(
                f"the clock supplied to the budget store did not produce an aware "
                f"datetime ({e}).",
                remedy=(
                    "pass a callable returning an aware datetime, for example "
                    "`lambda: datetime.now(UTC)`. A naive one would be read as "
                    "local time, which moves the instant the ceiling resets by "
                    "hours."
                ),
            ) from e
