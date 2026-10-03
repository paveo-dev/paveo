"""The public entry point and the session that carries identity (§3, §5.2).

``Paveo`` holds the loaded policy and the audit log. ``Session`` carries
*who is calling* — an ``agent_id`` and, optionally, the ``principal`` the agent
claims to act for. Both appear in every audit record, because an action should be
attributable to an agent **and** to the human it claims to act for; that pair is
the beginning of an answer to the confused-deputy problem. v1 records the pair
and enforces policy on the ``agent_id``. Delegation chains are v2.

**Thread safety.**

- ``Paveo`` is safe to share across threads. The policy is immutable and the
  audit log serialises its own writes.
- ``Session`` is **not** safe to share across threads or tasks. Use one per
  thread, per task, or per unit of work; they are cheap, and a session owns the
  reservations of the model calls it admitted, which must unwind on exactly one
  path. The budget they spend is shared all the same: every session of one
  ``Paveo`` reserves against **one store** (D33, §10.11), so a thread pool and an
  event loop spending one agent's ceiling cannot race or double it.
- ``LLMCall``, the handle ``check_llm`` returns, belongs to its session and is
  not safe to share either.

A session must be used as a context manager. Calling ``check_tool`` or
``check_llm`` on one that was never entered raises, rather than working now and
leaking a reservation later (Rule 5). **On exit, any call still open is charged
its whole worst case**: it may have been billed, and nothing came back to say
otherwise (Rule 6).

``async_session`` is not here yet. Not required: without the budget engine there
is nothing for it to await, and an async surface that is async in name only
teaches the wrong thing. The budget store will not supply that content either: one
store serves both worlds and none of its calls wait on anything (D33). *Trigger:
an async client to wrap (§2.1), which is the one thing a coroutine must await.*
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from types import TracebackType

from . import enforce
from ._anthropic_client import Wrapped
from ._licence import DEVELOPER, apply, read_key
from ._memory import Memory
from ._policy_document import load_document, load_file
from .audit import AuditLog, utc_now
from .errors import ConfigError
from .policy import Policy
from .prices import _ALL_LISTINGS, _PriceTable
from .stores import BudgetStore, InMemoryBudgetStore

# Where decisions go when the caller does not say. A relative path, so it lands
# beside the service rather than somewhere surprising, and it is gitignored in
# this repo because an audit log carries agent identities and spend.
DEFAULT_AUDIT_PATH = "paveo-audit.jsonl"


@dataclass(frozen=True, slots=True)
class Identity:
    """Who is calling, and on whose behalf (§5.2).

    A pair rather than two loose strings, because they are one idea and they
    always travel together: an action is attributable to an agent **and** to the
    human it claims to act for. Policy is enforced on ``agent_id``; ``principal``
    is an opaque string we record and never interpret.
    """

    agent_id: str
    principal: str | None = None


class Paveo:
    """A loaded policy, the log it writes decisions to, and the books it keeps.

    **One budget store for every session it opens**, sync or async (D33): a second
    store is a second set of books, and each would let an agent spend its whole
    ceiling. The price table is built here too, from the built-in listings and any
    prices the policy declares (§4.8.4, D39), so a policy that declares a price
    for a model the table already knows is refused as it loads.
    """

    def __init__(
        self,
        policy: Policy,
        audit: AuditLog,
        *,
        now: Callable[[], datetime] | None = None,
        licence: str | None = None,
    ) -> None:
        """Prefer ``from_file`` or ``from_policy``; this takes what they build.

        ``licence`` is a key for a paid plan; without one the Developer plan
        applies, which covers two agents per policy (D61, D62, D74). A key that
        is not one we issued raises ``ConfigError``.
        """
        self._clock = utc_now if now is None else now
        self._document = policy
        self._licence = licence
        self._applied_on: date | None = None
        self._policy = self._licensed()
        self._audit = audit
        self._table = _PriceTable(
            _ALL_LISTINGS, declared=policy.prices, now=self._clock
        )
        self._store = InMemoryBudgetStore(now=self._clock)

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        *,
        audit_path: str | Path = DEFAULT_AUDIT_PATH,
        now: Callable[[], datetime] | None = None,
        licence: str | None = None,
    ) -> Paveo:
        """Load a policy from JSON on disk.

        Raises ``ConfigError`` if the policy is missing or invalid, or if the
        audit log cannot be opened. Nothing starts with a policy we could not
        read: a guard that silently stops guarding is worse than no guard.
        """
        return cls._open(load_file(path), audit_path, now, licence)

    @classmethod
    def from_policy(
        cls,
        document: object,
        *,
        audit_path: str | Path = DEFAULT_AUDIT_PATH,
        now: Callable[[], datetime] | None = None,
        licence: str | None = None,
    ) -> Paveo:
        """Load a policy from an already-parsed object, validated identically."""
        return cls._open(load_document(document), audit_path, now, licence)

    @classmethod
    def _open(
        cls,
        policy: Policy,
        audit_path: str | Path,
        now: Callable[[], datetime] | None,
        licence: str | None,
    ) -> Paveo:
        """Build one, closing the log again if the key is refused: a log left
        open would hold its lock against the retry (/code-review, D62)."""
        audit = AuditLog(audit_path, now=now)
        try:
            return cls(policy, audit, now=now, licence=licence)
        except BaseException:
            audit.close()
            raise

    def _licensed(self) -> Policy:
        """The policy under today's plan, worked out again when the day changes,
        so a long-running process reverts when its key or trial ends and warns
        before (/code-review, D62). Two threads racing here both assign the same
        result: applying a plan is a pure function of the key and the day."""
        today = self._clock().date()
        if self._applied_on != today:
            plan = (
                DEVELOPER
                if self._licence is None
                else read_key(self._licence, today=today)
            )
            self._policy = apply(self._document, plan, today=today)
            self._applied_on = today
        return self._policy

    def session(self, *, agent_id: str, principal: str | None = None) -> Session:
        """Open a session for one agent. Use it as a context manager."""
        if not agent_id:
            raise ConfigError(
                "a session needs an agent_id.",
                remedy="pass the id of an agent the policy declares.",
            )
        policy = self._licensed()
        agent = policy.agents.get(agent_id)
        budget = agent.budget if agent is not None else None
        # §4.7: a session ceiling lasts as long as the session, so it gets a store
        # that dies with it. The shared one refuses a session ceiling outright
        # rather than let it outlive every session (D31).
        store: BudgetStore = (
            InMemoryBudgetStore(now=self._clock, one_session=True)
            if budget is not None and budget.period == "session"
            else self._store
        )
        return Session(
            gate=enforce.Checkpoint(
                policy=policy,
                audit=self._audit,
                table=self._table,
                store=store,
                now=self._clock,
            ),
            identity=Identity(agent_id=agent_id, principal=principal),
        )

    def close(self) -> None:
        self._audit.close()

    def __enter__(self) -> Paveo:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


class Session:
    """One agent's identity, and the checks made under it.

    Not safe to share across threads or tasks — see the module docstring.
    """

    def __init__(self, *, gate: enforce.Checkpoint, identity: Identity) -> None:
        self._gate = gate
        self._policy = gate.policy
        self._audit = gate.audit
        self._open_calls: list[LLMCall] = []
        # What this session's admitted tool calls left for `requires`, `rate`
        # and `repeat` (D58, D59): digests and times, never values. The same
        # class the guard keeps on disk, so the two count alike.
        self._memory = Memory()
        self._now = gate.now
        self._entered = False
        # Private, and not re-assignable from outside: rebinding it mid-session
        # would silently re-attribute every subsequent audit record. SPEC §3
        # names the entire public surface and this is not on it (Rule 12).
        self._identity = identity

    def __enter__(self) -> Session:
        self._entered = True
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        # Every exit path, an exception included, settles what this session
        # still holds (§3): a leaked reservation is a denial of service against
        # the customer's own agent.
        try:
            self._release()
        finally:
            self._entered = False
            # Memory for `requires`, `rate` and `repeat` belongs to one `with`
            # block: a session entered again starts with none (D58, D59).
            self._memory = Memory()

    def _release(self) -> None:
        """Charge every call still open at its worst case (§3, §4.2).

        Worst case rather than nothing: an open call may have reached the
        provider and been billed, and nothing came back to say otherwise. Every
        call is settled before any failure is raised, so one bad record cannot
        leave the rest holding the ceiling.
        """
        failures: list[Exception] = []
        for call in list(self._open_calls):
            try:
                call._close("unsettled_at_session_end")
            except Exception as e:  # re-raised below, once every call is settled
                failures.append(e)
        if failures:
            first = failures[0]
            if len(failures) > 1:
                first.add_note(
                    f"{len(failures) - 1} more open call(s) also failed to settle "
                    f"as this session closed."
                )
            raise first

    def _require_entered(self) -> None:
        if not self._entered:
            raise ConfigError(
                "this session was never entered.",
                remedy=(
                    "use it as a context manager: "
                    "`with pf.session(agent_id=...) as s:`. A session releases "
                    "what it holds on exit, and one that is never entered is "
                    "never cleaned up."
                ),
            )

    def check_tool(
        self, name: str, arguments: Mapping[str, object] | None = None
    ) -> None:
        """Check a tool call before making it. Raises ``PolicyDenied`` if refused.

        Returns nothing when the call is permitted — there is no result object to
        forget to check. The decision is written to the audit log first, and if
        it cannot be written the call is denied even if the policy allowed it.
        A call the rules admitted is remembered, as digests and a time, for the
        ``requires``, ``rate`` and ``repeat`` rules of later calls in this
        session (D58, D59).
        """
        self._require_entered()
        called = {} if arguments is None else arguments
        agent_id = self._identity.agent_id
        memory = self._memory
        recall = memory.recall(self._now().timestamp())
        admitting, admitted = memory.steps(self._policy, agent_id, name, called, recall)

        enforce.check_tool(
            policy=self._policy,
            audit=self._audit,
            identity=self._identity,
            tool=name,
            arguments=called,
            # `paveo stop` reaches the command's checks only (D49). *Trigger for
            # the library: a user asks for a stop that reaches a running process.*
            stopped=False,
            recall=recall,
            # Admitted by the rules, not merely let through by shadow mode; and
            # admitted is not succeeded: the call itself may still fail (D58).
            admitting=admitting,
            admitted=admitted,
        )

    def check_llm(self, request: Mapping[str, object], *, shape: str) -> LLMCall:
        """Check a model call before making it, and hold its worst case.

        ``request`` is exactly what you are about to send, the mapping you pass to
        ``messages.create``; ``shape`` names its format (``"anthropic"``). A model
        whose price the policy declares is read whatever its shape (D39). The
        request is measured and let go: nothing in it is kept or logged.

        Raises ``PolicyDenied``, ``PricingUnknown`` or ``BudgetExceeded`` if the
        call may not be made, each recorded first. Returns the call's handle:
        pass the response's usage to ``record`` when it returns, or call
        ``release`` if the request never reached the provider.
        """
        self._require_entered()
        admitted = enforce.check_llm(
            self._gate, identity=self._identity, request=request, shape=shape
        )
        call = LLMCall(self, admitted)
        self._open_calls.append(call)
        return call

    def wrap_anthropic(self, client: object) -> Wrapped:
        """An Anthropic client whose ``messages.create`` and ``messages.stream``
        are checked before they leave and recorded when they return (§2.1, S6).

        Only those two, and the same under ``beta``: anything else on the wrapped
        client raises rather than going out unchecked. Nothing here imports the
        Anthropic SDK.
        """
        self._require_entered()
        return Wrapped(self, client)

    def remaining(self) -> Decimal:
        """USD left under this agent's ceiling: ``limit - spent - reserved`` (D23 #6).

        What calls already in flight have claimed is subtracted too, so a call
        this says fits is one the ceiling will admit. Negative if a mispriced call
        has breached it, because that is the true answer.
        """
        self._require_entered()
        agent = self._policy.agents.get(self._identity.agent_id)
        if agent is None or agent.budget is None:
            raise ConfigError(
                f"agent {self._identity.agent_id!r} has no budget to report.",
                remedy="declare a budget for it in the policy.",
            )
        return self._gate.store.remaining(
            agent_id=self._identity.agent_id, budget=agent.budget
        )


class LLMCall:
    """One admitted model call, holding its worst case until it is settled.

    Settle it exactly once: ``record(usage)`` with what the response reported, or
    ``release()`` if the request never reached the provider. Use it as a context
    manager to have that guaranteed: a block that leaves without either charges
    the whole worst case, because the call may have been billed.

    Not safe to share across threads or tasks, like the session it belongs to.
    """

    def __init__(self, session: Session, admitted: enforce.Admitted) -> None:
        self._session = session
        self._admitted = admitted
        self._open = True

    @property
    def reservation_id(self) -> str:
        """The id that pairs this call's two audit records (D23 #2)."""
        return self._admitted.reservation.reservation_id

    def record(self, usage: Mapping[str, object]) -> Decimal:
        """Charge what the call cost, from the response's usage. Returns the charge.

        Pass ``response.usage.model_dump()`` for an adapter's shape, or
        ``{"input": n, "output": m}`` for a model priced in the policy. Usage that
        cannot be read is charged at the worst case and then raises.
        """
        self._settling()
        return self._settle(usage, "recorded")

    def release(self) -> None:
        """Give the hold back: the request never reached the provider."""
        self._settling()
        session = self._session
        enforce.release_llm(
            session._gate,
            identity=session._identity,
            admitted=self._admitted,
            settled=self._settled,
        )

    def __enter__(self) -> LLMCall:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self._open:
            self._close("unsettled_at_block_exit")

    def _close(self, reason: str) -> None:
        """Charge the worst case: settled by nobody, so possibly billed."""
        if self._open:
            self._settle(None, reason)

    def _settle(self, usage: Mapping[str, object] | None, reason: str) -> Decimal:
        session = self._session
        return enforce.settle_llm(
            session._gate,
            identity=session._identity,
            admitted=self._admitted,
            usage=usage,
            reason=reason,
            settled=self._settled,
        )

    def _settling(self) -> None:
        if not self._open:
            raise ConfigError(
                "this call was already recorded or released.",
                remedy=(
                    "settle each call exactly once: settling twice would charge the "
                    "ceiling twice for one call."
                ),
            )

    def _settled(self) -> None:
        """The ledger has taken this call's charge or its release: stop holding it.

        Called from inside the settlement, after the ledger and before the record,
        so a ledger that refused leaves the call open for the session to charge at
        exit, and a record that cannot be written still leaves it closed.
        """
        self._open = False
        if self in self._session._open_calls:
            self._session._open_calls.remove(self)
