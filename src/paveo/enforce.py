"""The decision point (``docs/SPEC_V1.md`` §6, §7).

Three steps, in this order, and the order is the whole point:

1. ask the policy
2. **write the record**
3. raise if the answer was no

Step 2 comes before step 3 because *an unlogged decision did not happen*. That
applies to a permitted call as much as a refused one: if the record cannot be
written, the call is denied even though the policy allowed it (§7).

**`fail_open` does not override a policy denial.** It covers the fail-*closed*
paths — the cases where we could not decide or could not record. A refusal is a
decision, not a failure to decide, and a mode that could turn refusals into
permissions would make the policy decorative. So `PolicyDenied` is raised whether
`fail_open` is on or off; only `PolicyUnavailable` is suppressed by it.

**Shadow mode is the one way a rule's refusal lets a call through**, and it is
chosen per agent in the policy, never by a failure. The refusal is still
recorded, as `would_deny`, and warned about on every call. It never reaches the
budget: a shadowed agent's ceiling refuses exactly as an enforced one's (D48).
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Protocol

from . import _anthropic, _gemini, _generic, _openai
from ._anthropic import RequestBound
from .audit import AuditLog, utc_now
from .budget import Reservation
from .errors import (
    BudgetExceeded,
    ConfigError,
    PolicyDenied,
    PolicyUnavailable,
    PricingUnknown,
)
from .policy import UNDECLARED_MODEL, UNDECLARED_TOOL, Denial, Policy, Recall
from .prices import Rates, _PriceTable, _printable
from .stores import BudgetStore

if TYPE_CHECKING:  # `session` imports this module, so the runtime import would cycle
    from .session import Identity


# How a request of each shape is measured, and how its response's usage is read.
# One entry per adapter (D39); S4b and S4c add "openai" and "gemini". A model the
# policy prices itself is read by the generic reader whatever its shape.
class _Reader(Protocol):
    def __call__(
        self, request: Mapping[str, object], *, assumed_max_output_tokens: int | None
    ) -> RequestBound: ...


_UsageReader = Callable[[Mapping[str, object]], dict[str, object]]
_ServedReader = Callable[[Mapping[str, object]], dict[str, str]]
_Readers = tuple[_Reader, _UsageReader, _ServedReader]
_SHAPES: Mapping[str, _Readers] = {
    "anthropic": (_anthropic.bound, _anthropic.usage_classes, _anthropic.served),
    "openai": (_openai.bound, _openai.usage_classes, _openai.served),
    "gemini": (_gemini.bound, _gemini.usage_classes, _gemini.served),
}
# The shape a caller names for a model the policy prices, in any provider's
# format: text bounded by its bytes, usage reported as {"input", "output"}.
_GENERIC = "generic"

# §4.5: the only token-count mode v1 has, written into every LLM record so a
# denial caused by it is explainable from the log alone.
_COUNT_MODE = "conservative"

# `logging` rather than `warnings`: the standard library dedupes warnings per
# call site by default, and §7 asks for a warning on *every* call. An unconfigured
# application still sees these, because Python's last-resort handler prints
# WARNING and above to stderr.
_logger = logging.getLogger("paveo")


_STOPPED = Denial(
    reason="stopped",
    rule="paveo.stop",
    remedy=(
        "someone ran `paveo stop`, so every call is refused until `paveo resume` "
        "is run by a person."
    ),
)


def check_tool(  # noqa: PLR0913 - keyword-only, each one a different fact about the call
    *,
    policy: Policy,
    audit: AuditLog,
    identity: Identity,
    tool: str,
    arguments: Mapping[str, object],
    stopped: bool,
    recall: Recall | None,
    admitting: Callable[[], object] | None = None,
    admitted: Callable[[], object] | None = None,
) -> None:
    """Decide a tool call, record it, and raise if it is not permitted.

    Returns ``None`` when the call may proceed. Never returns a value the caller
    could forget to check (§3).

    ``stopped`` is the panic button (``paveo stop``, D49): every call is refused
    before the policy is asked, and shadow mode is never offered it.

    ``recall`` is the caller's memory of this session, or ``None`` if it keeps
    none (D58, D59). When **the rules** admit the call, and never for one shadow
    mode let through, the caller remembers it in two steps (/code-review, D59):

    - ``admitting`` runs **before** the record is written, and counts the call
      for ``rate`` and ``repeat``. A failure after it over-counts, which refuses
      more. If it cannot keep the memory it raises ``ConfigError``, and the call
      is refused and recorded as ``memory_unavailable``.
    - ``admitted`` runs **after** the record is written, and leaves the call's
      footprints for ``requires``. Leaving them earlier would let a call whose
      record failed, and which was therefore refused, admit the one that needs it.
    """
    _warn_if_fail_open(policy)
    _warn_if_shadow(policy, identity.agent_id)

    denial = (
        _STOPPED
        if stopped
        else policy.evaluate_tool(identity.agent_id, tool, arguments, recall=recall)
    )

    # A tool the policy declares is one of a fixed set the operator wrote, so it
    # is safe to write down and is the most useful thing in the record. A name
    # that matches nothing came from the model, is unbounded, and would turn every
    # denied call into an exfiltration channel out of the deny path (D26).
    name = tool if policy.declares_tool(identity.agent_id, tool) else UNDECLARED_TOOL

    record: dict[str, object] = {
        "agent_id": identity.agent_id,
        "principal": identity.principal,
        "action": {"kind": "tool", "name": name},
        "decision": "allow",
        "reason": None,
        "rule": None,
        # §6: the cost fields are null for a tool decision. Written rather than
        # omitted so every record has the same shape for whatever reads them.
        "estimated_cost_usd": None,
        "actual_cost_usd": None,
        "rate_key": None,
        "usage_by_class": None,
        "fail_open": policy.fail_open,
        "policy_id": policy.policy_id,
        "policy_hash": policy.policy_hash,
        "plan": policy.plan,
    }

    remembered = denial is None
    unkept = False
    if remembered and admitting is not None:
        try:
            admitting()
        except ConfigError as e:
            # Memory that cannot be kept cannot count this call: refused, and
            # recorded like any refusal, never shadowed (/code-review, D59).
            # The remedy is the operator's; `for_model` never carries it (D48).
            remembered, unkept = False, True
            denial = Denial(
                reason="memory_unavailable", rule=f"{name}.memory", remedy=str(e)
            )
    if (
        denial is not None
        and not stopped
        and not unkept
        and policy.shadows(identity.agent_id)
    ):
        _shadow(audit, policy, record, denial)
        denial = None
    if denial is not None:
        record = {
            **record,
            "decision": "deny",
            "reason": denial.reason,
            "rule": denial.rule,
        }

    _append(audit, policy, record)

    if denial is not None:
        raise PolicyDenied(reason=denial.reason, rule=denial.rule, remedy=denial.remedy)
    if remembered and admitted is not None:
        try:
            admitted()
        except ConfigError as e:
            # The record already says allow, so the call goes on. A footprint
            # that could not be left only makes a later `requires` refuse, which
            # is the safe side (/code-review, D59).
            _logger.warning(
                "paveo could not keep a footprint for a call it allowed. "
                "agent_id=%s rule=%s.memory: %s",
                identity.agent_id,
                name,
                e,
            )


@dataclass(frozen=True, slots=True)
class Checkpoint:
    """Everything a model call is judged and charged against, held together.

    One per session: the policy, the log and the price table are the ``Paveo``
    object's and shared, and the store is shared too except for an agent whose
    ceiling lasts one session, which gets a store of its own (D31). Immutable;
    the objects it points at state their own thread safety.
    """

    policy: Policy
    audit: AuditLog
    table: _PriceTable
    store: BudgetStore
    # The injected clock (Rule 14), read by the tool rules that count time (D59).
    now: Callable[[], datetime] = utc_now


@dataclass(frozen=True, slots=True)
class Admitted:
    """A model call the decision point let through, and what settling it needs.

    ``model`` is the name as the audit log recorded it. Immutable, and holds no
    payload: the request was measured and let go.
    """

    reservation: Reservation
    rates: Rates
    read_usage: _UsageReader
    read_served: _ServedReader
    model: str
    # Reserved on the policy's assumed output, which caps nothing (§4.4): such a
    # call exceeding its reservation is the documented approximation, not a
    # broken bound, so it does not stop the table (/code-review, D42).
    assumed_output: bool


def check_llm(  # noqa: PLR0915 - one decision, refused and recorded at each step that can fail
    gate: Checkpoint,
    *,
    identity: Identity,
    request: Mapping[str, object],
    shape: str,
) -> Admitted:
    """Decide a model call, hold its worst case against the ceiling, record it.

    The order is the whole point, as for tools, with the budget in the middle:
    the policy first, so a model the operator never named is refused before its
    name reaches a price or a ceiling (§5.1); then the bound and the price, which
    refuse what they cannot measure (§4.5, §4.8); then the reservation, which
    refuses what does not fit (§4.1); and only then the record of the "yes".

    **Every refusal is recorded before it is raised** (§7), a request that is
    wired wrong included. **A "yes" that cannot be recorded, for any reason, gives
    its reservation back** and is refused, unless ``fail_open`` covers it.

    ``fail_open`` covers only a record that cannot be written. A policy
    refusal, a breached ceiling and a call that cannot be priced are all raised
    regardless (D18, D23 #4, D39).
    """
    policy, audit, table, store = gate.policy, gate.audit, gate.table, gate.store
    _warn_if_fail_open(policy)
    _warn_if_shadow(policy, identity.agent_id)
    agent_id = identity.agent_id
    model = request.get("model") if isinstance(request, Mapping) else None
    named = (
        model
        if isinstance(model, str) and policy.declares_model(agent_id, model)
        else UNDECLARED_MODEL
    )
    record = _record(gate, identity, named)

    try:
        model = _anthropic.model_of(request)
        denial = policy.evaluate_model(agent_id, model)
        shadowed = denial is not None and policy.shadows(agent_id)
        if denial is not None and shadowed:
            _shadow(audit, policy, record, denial)
            denial = None
        if denial is None:
            # After the rules and never offered to `shadows`, so shadow mode
            # cannot reach it (locked decision #7, D48).
            denial = policy.evaluate_budget(agent_id)
        if denial is not None:
            _append(
                audit,
                policy,
                {
                    **record,
                    "decision": "deny",
                    "reason": denial.reason,
                    "rule": denial.rule,
                },
            )
            raise PolicyDenied(
                reason=denial.reason, rule=denial.rule, remedy=denial.remedy
            )
        if shadowed:
            _refuse_unless_carried(table, model)

        reader, read_usage, read_served = _readers(model, policy, shape)
        bound = reader(
            request, assumed_max_output_tokens=policy.assumed_max_output_tokens
        )
        rates = table.resolve(model, bound.modifiers)
        worst_case = table.worst_case(
            rates, bound.input_upper_bound, bound.max_output_tokens
        )
    except PricingUnknown:
        _append(
            audit,
            policy,
            {**record, "decision": "deny", "reason": "pricing_unknown", "rule": None},
        )
        raise
    except ConfigError:
        # A request wired wrong is still a refused call, and an unlogged refusal
        # did not happen (§7).
        _append(
            audit,
            policy,
            {**record, "decision": "deny", "reason": "invalid_request", "rule": None},
        )
        raise

    budget = policy.agents[agent_id].budget
    if budget is None:  # evaluate_budget refuses this; a type checker cannot see that
        raise ConfigError(
            f"agent {agent_id!r} has no budget.",
            remedy="declare one; no ceiling is not an open one (§5.1).",
        )
    record["rate_key"] = rates.rate_key
    try:
        reservation = store.reserve(
            agent_id=agent_id, budget=budget, worst_case=worst_case
        )
    except BudgetExceeded as e:
        _append(
            audit,
            policy,
            {
                **record,
                "decision": "deny",
                "reason": "budget_exceeded",
                "rule": None,
                "estimated_cost_usd": str(e.requested),
            },
        )
        raise
    except Exception:
        # The reservation could not be made, whatever the cause: a store that
        # cannot answer, of ours or a customer's own, or a request too large to
        # account. Refused either way, and an unlogged refusal did not happen (§7).
        _append(
            audit,
            policy,
            {
                **record,
                "decision": "deny",
                "reason": "reserve_failed",
                "rule": None,
            },
        )
        raise

    try:
        _append(
            audit,
            policy,
            {
                **record,
                "decision": "allow",
                # §4.4: a ceiling enforced with an assumed output is approximate,
                # and every record that relied on it says so.
                "reason": "assumed_max_output_tokens" if bound.assumed_output else None,
                "rule": None,
                "estimated_cost_usd": str(reservation.worst_case),
                "reservation_id": reservation.reservation_id,
            },
        )
    except Exception:
        # Whatever stopped the record, nobody holds a handle to this reservation,
        # so nothing else would ever give it back (/code-review, D40).
        store.release(agent_id=agent_id, reservation_id=reservation.reservation_id)
        raise
    return Admitted(
        reservation=reservation,
        rates=rates,
        read_usage=read_usage,
        read_served=read_served,
        model=named,
        assumed_output=bound.assumed_output,
    )


def _counted(classes: dict[str, object]) -> dict[str, object]:
    """Usage without both counts is unreadable, not a call that cost nothing,
    whichever adapter read it (D44)."""
    if not {"input", "output"} <= classes.keys():
        raise ConfigError(
            "usage carries no input or output token count.",
            remedy="record the usage the response returned, unaltered.",
        )
    return classes


def settle_llm(  # noqa: PLR0913 - keyword-only, each one a different fact about the settle
    gate: Checkpoint,
    *,
    identity: Identity,
    admitted: Admitted,
    usage: Mapping[str, object] | None,
    reason: str,
    settled: Callable[[], None],
) -> Decimal:
    """Charge an admitted call and write its settlement record (§4.2, D23 #2).

    ``usage`` is what the response reported, read by the adapter the call was
    admitted through. ``None`` means nothing trustworthy came back, the call
    may still have been billed, and so the whole worst case is charged: the
    direction that cannot under-charge (Rule 6).

    ``settled`` is called the moment the ledger has taken the charge and before
    the record is written, so the caller stops holding the call exactly when the
    money side is done: a ledger that refused leaves the call open to be charged
    again at session end, and a record that cannot be written loses nothing.

    Returns what was charged. Usage that cannot be read is charged at the worst
    case and then raised as the wiring error it is.
    """
    policy, audit, table, store = gate.policy, gate.audit, gate.table, gate.store
    worst_case = admitted.reservation.worst_case
    classes: dict[str, object] = {}
    stale = False
    charged_at = admitted.rates.rate_key
    failure: Exception | None = None
    if usage is None:
        cost = worst_case
    else:
        try:
            classes = _counted(admitted.read_usage(usage))
            served = admitted.read_served(usage)
            charge = table.actual(admitted.rates, classes, served)
            cost, stale = charge.cost, charge.price_table_stale
            charged_at = charge.rate_key
            if cost > worst_case and not admitted.assumed_output:
                # The bound under this reservation was not a bound. Charged in
                # full all the same (D24 #5), and the table stops admitting
                # calls until it is fixed (D42).
                table.breached(charged_at)
                stale = True
        except (ConfigError, ValueError) as e:
            cost, reason, failure = worst_case, "usage_unreadable", e

    store.settle(
        agent_id=identity.agent_id,
        reservation_id=admitted.reservation.reservation_id,
        actual=cost,
    )
    settled()
    _append(
        audit,
        policy,
        {
            **_record(gate, identity, admitted.model),
            "decision": "settle",
            "reason": reason,
            "rule": None,
            "estimated_cost_usd": str(worst_case),
            "actual_cost_usd": str(cost),
            # The row actually charged: a long prompt or the tier the response
            # was served at can move it from the one reserved (D41).
            "rate_key": charged_at,
            "usage_by_class": _counts(classes),
            "price_table_stale": stale,
            "reservation_id": admitted.reservation.reservation_id,
        },
    )
    if failure is not None:
        raise ConfigError(
            "the usage passed to record() could not be read, so this call was "
            "charged its full worst case.",
            remedy=(
                "pass response.usage.model_dump() for an adapter's shape, or "
                '{"input": n, "output": m} for a model priced in the policy.'
            ),
        ) from failure
    return cost


def release_llm(
    gate: Checkpoint,
    *,
    identity: Identity,
    admitted: Admitted,
    settled: Callable[[], None],
) -> None:
    """Give back a hold whose request never reached the provider (§4.2).

    The ledger first, then ``settled``, then the record, for the reason
    ``settle_llm`` gives.
    """
    gate.store.release(
        agent_id=identity.agent_id, reservation_id=admitted.reservation.reservation_id
    )
    settled()
    _append(
        gate.audit,
        gate.policy,
        {
            **_record(gate, identity, admitted.model),
            "decision": "settle",
            "reason": "released",
            "rule": None,
            "estimated_cost_usd": str(admitted.reservation.worst_case),
            "actual_cost_usd": "0",
            "rate_key": admitted.rates.rate_key,
            "reservation_id": admitted.reservation.reservation_id,
        },
    )


def _record(gate: Checkpoint, identity: Identity, model: str) -> dict[str, object]:
    """The fields every model-call record carries, before its decision."""
    policy = gate.policy
    return {
        "agent_id": identity.agent_id,
        "principal": identity.principal,
        "action": {"kind": "llm", "model": model},
        "estimated_cost_usd": None,
        "actual_cost_usd": None,
        "rate_key": None,
        "usage_by_class": None,
        "price_table_stale": False,
        "token_count_mode": _COUNT_MODE,
        "fail_open": policy.fail_open,
        "policy_id": policy.policy_id,
        "policy_hash": policy.policy_hash,
        "plan": policy.plan,
        "prices_version": gate.table.version,
    }


def _readers(model: str, policy: Policy, shape: str) -> _Readers:
    """The reader for a request of this shape.

    **The shape decides how a request is read, and the policy only what it
    costs.** A request in a shape paveo has an adapter for is read by that
    adapter even when the policy prices the model itself, so its allowances and
    refusals still apply (/code-review, D40). Only a shape with no adapter falls
    back to the generic reader, and only for a model the policy prices.
    """
    readers = _SHAPES.get(shape)
    if readers is not None:
        return readers
    if shape == _GENERIC and model in policy.prices:
        return _generic.bound, _generic.usage_classes, _generic.served
    raise ConfigError(
        f"shape is {_printable(shape)!r}, and paveo reads "
        f"{', '.join(sorted(_SHAPES))}, or {_GENERIC!r} for a model the policy "
        f"prices.",
        remedy=(
            "pass the shape of the request you send. For a provider paveo has no "
            "adapter for, declare the model's price in the policy (D39)."
        ),
    )


def _counts(classes: Mapping[str, object]) -> dict[str, int]:
    """Usage for the audit record: printable class names and whole counts only."""
    return {
        _printable(name): count
        for name, count in classes.items()
        if isinstance(count, int) and not isinstance(count, bool)
    }


def _append(audit: AuditLog, policy: Policy, record: Mapping[str, object]) -> None:
    """Write a record; a failure is a refusal unless ``fail_open`` (§7, D18)."""
    try:
        audit.append(record)
    except PolicyUnavailable:
        if not policy.fail_open:
            raise
        _logger.warning(
            "paveo could not record a decision and fail_open is enabled, so the "
            "call proceeds unrecorded. agent_id=%s action=%s",
            record.get("agent_id"),
            # Names already reduced to what the policy declares (D26).
            record.get("action"),
        )


def _shadow(
    audit: AuditLog, policy: Policy, record: Mapping[str, object], denial: Denial
) -> None:
    """Record the refusal a shadowed rule would have made, then let the call on.

    Written where the ``deny`` would have been, and the call is then recorded as
    any other: ``allow``, or a ``deny`` from the ceiling. So ``allow`` still
    counts the calls that went out, and ``would_deny`` counts what enforcing
    would have stopped (D48).
    """
    _append(
        audit,
        policy,
        {
            **record,
            "decision": "would_deny",
            "reason": denial.reason,
            "rule": denial.rule,
        },
    )
    _logger.warning(
        "paveo shadow mode let through a call its policy would refuse. "
        "agent_id=%s rule=%s reason=%s",
        record.get("agent_id"),
        denial.rule,
        denial.reason,
    )


def _refuse_unless_carried(table: _PriceTable, model: str) -> None:
    """Refuse a model the table does not price, without repeating its name.

    Past the policy, a model's name is repeated in pricing errors, which is safe
    only for a name from a fixed set. A shadowed rule let through a name the
    model may have chosen, so one the table does not carry is refused under the
    name the log uses (D26; found by /code-review, D48).
    """
    if not table.carries(model):
        raise PricingUnknown(
            model=UNDECLARED_MODEL,
            detail="it is not in the price table",
            remedy=(
                "call a model the table prices, or declare its price in the "
                "policy's prices."
            ),
        )


def _warn_if_shadow(policy: Policy, agent_id: str) -> None:
    """On every call, like ``fail_open``: shadow mode is off by default and noisy
    when on, because rules that do not enforce must never be mistaken for rules
    that do (§7, D48)."""
    if policy.shadows(agent_id):
        _logger.warning(
            "paveo agent %s is in shadow mode: rule refusals are recorded as "
            "would_deny and not enforced. Its budget still is. policy_id=%s",
            agent_id,
            policy.policy_id,
        )


def _warn_if_fail_open(policy: Policy) -> None:
    if policy.fail_open:
        _logger.warning(
            "paveo is running with fail_open enabled: a decision that cannot be "
            "recorded will be allowed instead of denied. policy_id=%s",
            policy.policy_id,
        )
