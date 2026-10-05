"""The policy model: what an agent is allowed to do (``docs/SPEC_V1.md`` §5).

This module is the *semantics* — the shapes a policy takes and how a call is
judged against one. Turning a JSON document into one of these, and rejecting the
ways a document can be wrong, lives next door in ``_policy_document.py``.

**Deny by default, at every step.** An unknown agent, a tool with no allow rule,
an explicitly denied tool and an argument the rule does not name are all
refusals. Nothing here can return "allowed" by falling off the end.

**Arguments are a closed set.** If a tool rule declares a ``constraints`` block,
those are the only argument names the agent may pass, and a name with predicates
on it must actually be supplied — a constraint you can dodge by leaving the
argument out is not a constraint. Declare a name with ``{}`` to permit it without
constraining or requiring it. A tool rule with no ``constraints`` block at all
does not check arguments. See D14.

**Thread safety.** ``Policy`` and everything reachable from it are immutable and
safe to share across threads and tasks without synchronisation. Nothing is
lazily computed and nothing is cached, so there is no state to race over.

**Three rules remember** (B4, D58, D59): ``requires`` (a tool only after an
admitted call to another, with the same values for named arguments), ``rate``
(at most so many admitted calls in so many seconds) and ``repeat`` (not the same
call twice within so many seconds). What they remember of a call is a digest and
a time, never a value, and whoever holds the memory passes it in as a
``Recall``. The policy itself stays stateless.

No predicate here can run code. The comparisons are ``max``, ``min``, ``in``,
``equals``, an anchored ``matches`` and an unanchored ``not_matches`` —
deliberately dumb, because an expression
language in a policy file is a remote code execution vector (§5.1).
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping, Sequence, Set
from dataclasses import dataclass, field
from decimal import Decimal
from types import MappingProxyType

from ._canonical import canonical_json

_MAX_MATCH_LENGTH = 4096
# `not_matches` reads whole shell commands, which can be long heredocs. Over this,
# the predicate fails: a value too long to search is not a value known to be safe.
_MAX_SEARCH_LENGTH = 1 << 20

# Stands in for a tool name the policy does not declare. Such a name was chosen
# by the model rather than by the operator, so it is payload and is not written
# down — the same ruling D19 made for an argument name the policy does not
# define, applied to the field beside it (D26).
UNDECLARED_TOOL = "<undeclared>"
UNDECLARED_MODEL = "<undeclared>"

# A caller-supplied string is read as a number only in this exact shape. Decimal
# itself is far more permissive — it accepts surrounding whitespace, underscores,
# exponents and non-ASCII digits, so `" 2e2 "`, `"1_0"` and the Arabic-Indic
# `"\u0665\u0660"` all parse. At a money boundary that is type confusion waiting
# to happen: we would read one number and the tool being guarded would read
# another, or fail. D19.
_PLAIN_DECIMAL = re.compile(r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)")


@dataclass(frozen=True, slots=True)
class Denial:
    """Why a call was refused. ``enforce`` turns this into the raised error.

    The core returns a value and the caller raises it, for the same reason the
    budget core does (§4.9.1): a test wants to inspect thousands of outcomes
    without wrapping each in a ``try``, while a caller must not be able to ignore
    one. Converting value to exception happens in exactly one place.
    """

    reason: str
    rule: str
    remedy: str


@dataclass(frozen=True, slots=True)
class Constraint:
    """The dumb predicates permitted on one argument (§5.1).

    All present predicates must hold. None of them can run code.
    """

    maximum: Decimal | None = None
    minimum: Decimal | None = None
    permitted: tuple[str, ...] | None = None
    equals: str | bool | None = None
    pattern: re.Pattern[str] | None = None
    # `not_matches`: refused if any of these is found anywhere in the value.
    forbidden: tuple[re.Pattern[str], ...] | None = None

    @property
    def is_empty(self) -> bool:
        """True when this name is permitted but neither constrained nor required."""
        return all(
            predicate is None
            for predicate in (
                self.maximum,
                self.minimum,
                self.permitted,
                self.equals,
                self.pattern,
                self.forbidden,
            )
        )

    def failing_predicate(self, value: object) -> str | None:
        """Name the first predicate ``value`` fails, or ``None`` if it passes.

        The order below is fixed, so the same call always names the same rule. A
        denial whose reason shifts between runs is not explainable a year later
        (Rule 19).

        A value of the wrong shape fails rather than becoming something else. A
        string is read as a number only if it is a plain decimal — ``"250"``
        passes, ``"2e2"`` and ``"\u0665\u0660"`` do not — and anything unreadable
        fails the predicate rather than defaulting to zero.
        """
        if self.maximum is not None or self.minimum is not None:
            number = _comparable_number(value)
            if self.maximum is not None and (number is None or number > self.maximum):
                return "max"
            if self.minimum is not None and (number is None or number < self.minimum):
                return "min"
        if self.permitted is not None and not (
            isinstance(value, str) and value in self.permitted
        ):
            return "in"
        if self.equals is not None and not _same_scalar(value, self.equals):
            return "equals"
        return self._failing_pattern(value)

    def _failing_pattern(self, value: object) -> str | None:
        """The two regex predicates, in the same fixed order. A value that is not
        a string, or too long to test, fails either one."""
        if self.pattern is not None and not (
            isinstance(value, str)
            and len(value) <= _MAX_MATCH_LENGTH
            and self.pattern.fullmatch(value) is not None
        ):
            return "matches"
        if self.forbidden is not None and not (
            isinstance(value, str)
            and len(value) <= _MAX_SEARCH_LENGTH
            and not any(pattern.search(value) for pattern in self.forbidden)
        ):
            return "not_matches"
        return None


@dataclass(frozen=True, slots=True)
class Requirement:
    """``requires``: call this tool only after an admitted call to ``tool``, earlier
    in the same session, whose ``same`` arguments had the same values (D58).

    ``same`` may be empty: then any earlier call to ``tool`` meets it.
    """

    tool: str
    same: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Footprint:
    """What an admitted call leaves for a later ``requires`` to find.

    The tool, the argument names a rule compares, and a SHA-256 digest of their
    values, never the values themselves (locked decision #5). Hashable, so a
    session keeps a set of them.
    """

    tool: str
    same: tuple[str, ...]
    digest: str


@dataclass(frozen=True, slots=True)
class Rate:
    """``rate``: at most ``calls`` admitted calls to this tool in ``seconds`` (D59)."""

    calls: int
    seconds: int


@dataclass(frozen=True, slots=True)
class Repeat:
    """``repeat``: not the same call twice within ``seconds`` (D59). The same
    means equal in every argument, or with ``same`` in those it names: the
    arguments a model rewrites freely, such as a shell tool's ``description``,
    would otherwise make every retry look new (/code-review)."""

    seconds: int
    same: tuple[str, ...] | None


@dataclass(frozen=True, slots=True)
class Call:
    """One admitted call to a tool a ``rate`` or ``repeat`` rule watches: when, in
    seconds since the epoch, and a digest of all its arguments, never the
    arguments. ``digest`` is ``None`` when they could not be serialised."""

    tool: str
    at: float
    digest: str | None


@dataclass(frozen=True, slots=True)
class Recall:
    """What a checkpoint remembers of this session's calls the rules admitted.

    ``now`` is rule time, in seconds since the epoch, **never earlier than any
    time given before**: it moves on by how far the clock moved forward since its
    last reading, and not at all when it moved back (``_memory.Memory``).
    Otherwise a call forgotten as out of every window would fall back inside one
    and count for nothing (D59). ``salt`` is mixed
    into every digest: empty for a library session, whose memory never leaves the
    process, and random per file for the guard's memory on disk.
    """

    now: float
    footprints: Set[Footprint]
    calls: Sequence[Call]
    salt: bytes = b""


@dataclass(frozen=True, slots=True)
class ToolRule:
    """One permitted tool. ``constraints is None`` means arguments are unchecked."""

    name: str
    constraints: Mapping[str, Constraint] | None
    requires: Requirement | None = None
    rate: Rate | None = None
    repeat: Repeat | None = None


@dataclass(frozen=True, slots=True)
class BudgetPolicy:
    """The ceiling declared for an agent (§5). Consumed by ``budget`` in chunk 2."""

    period: str
    limit_usd: Decimal


@dataclass(frozen=True, slots=True)
class DeclaredPrice:
    """A price the operator wrote for a model no built-in table carries (§4.8.4).

    The door every provider without a verified table comes in by (D39): Mistral,
    DeepSeek, a local model, anything. USD per million tokens. The operator is the
    source of the number, so it is only as true as what they pay, and a policy may
    not declare a model a built-in table already prices: a typo there would
    under-price a model we know the real price of.
    """

    input_per_mtok: Decimal
    output_per_mtok: Decimal
    cached_input_per_mtok: Decimal | None


@dataclass(frozen=True, slots=True)
class AgentPolicy:
    """Everything one ``agent_id`` is permitted to do."""

    agent_id: str
    budget: BudgetPolicy | None
    models_allow: frozenset[str]
    models_deny: frozenset[str]
    tools: Mapping[str, ToolRule]
    tools_deny: frozenset[str]
    # `"mode": "shadow"`: a rule's refusal is recorded and the call goes on. Its
    # budget still enforces (D48).
    shadow: bool
    # For each tool some rule `requires`, the argument names those rules compare.
    # Built once at load, so an admitted call's footprints are one lookup, and
    # none at all for a policy that uses no `requires` (/code-review, D58). A
    # factory, because Python 3.11 refuses an unhashable default.
    compared_after: Mapping[str, tuple[tuple[str, ...], ...]] = field(
        default_factory=lambda: MappingProxyType({})
    )
    # Each tool a `rate` or `repeat` rule watches (D59).
    watched: frozenset[str] = frozenset()
    # The longest of those windows: calls older than this are forgotten.
    horizon: int = 0


@dataclass(frozen=True, slots=True)
class Policy:
    """A loaded, validated, immutable policy.

    Safe to share across threads and tasks: nothing here is mutable and nothing
    is lazily computed. Reloading produces a new object with a new
    ``policy_hash``; live mutation is not supported (§5.1).
    """

    policy_id: str
    fail_open: bool
    assumed_max_output_tokens: int | None
    agents: Mapping[str, AgentPolicy]
    policy_hash: str
    source: str
    prices: Mapping[str, DeclaredPrice]
    # The plan it runs under, and the agents past that plan's number, whose every
    # call is refused (``_licence.apply``, D62). A policy loaded but not yet
    # applied to a plan limits nobody: every entry point applies one.
    plan: str = "developer"
    over_plan: frozenset[str] = frozenset()

    def _over_plan(self, agent_id: str) -> Denial | None:
        if agent_id not in self.over_plan:
            return None
        return Denial(
            reason="plan_limit",
            rule=f"plan.{self.plan}",
            remedy=(
                f"the {self.plan} plan covers fewer agents than this policy "
                f"declares, and {agent_id!r} is past them in the file's order. "
                f"Remove agents, or add a licence key for a plan that covers "
                f"them."
            ),
        )

    def declares_model(self, agent_id: str, model: str) -> bool:
        """Whether this exact model name appears in the agent's rules.

        The model counterpart of ``declares_tool``, asked for the same reason: a
        name the operator wrote may be recorded, and one they did not may have
        been chosen by the model under injection (D26).
        """
        agent = self.agents.get(agent_id)
        if agent is None:
            return False
        return model in agent.models_allow or model in agent.models_deny

    def evaluate_model(self, agent_id: str, model: str) -> Denial | None:
        """Decide whether this agent may call this model. ``None`` means yes.

        Asked before anything prices the call, so a model the policy never names
        is refused before its name can reach a pricing error or the budget.
        Whether the agent has a ceiling at all is ``evaluate_budget``'s question.
        """
        agent = self.agents.get(agent_id)
        if agent is None:
            return Denial(
                reason="agent_unknown",
                rule=f"agent.{agent_id}",
                remedy=(
                    f"add an agent with id {agent_id!r} to the policy, or pass an "
                    f"agent_id that the policy declares."
                ),
            )
        over = self._over_plan(agent_id)
        if over is not None:
            return over
        if model in agent.models_deny:
            return Denial(
                reason="model_denied",
                rule=model,
                remedy=(
                    f"{model!r} is on agents[{agent_id!r}].models.deny, and an "
                    f"explicit deny always wins."
                ),
            )
        if model not in agent.models_allow:
            declared = ", ".join(sorted(agent.models_allow)) or "no models at all"
            return Denial(
                reason="model_not_allowed",
                rule=UNDECLARED_MODEL,
                remedy=(
                    f"agents[{agent_id!r}].models.allow declares {declared}, and the "
                    f"call named none of them. The name it used is not repeated "
                    f"here, for the reason a tool name is not (D26). Add the model "
                    f"to that list if it should be permitted."
                ),
            )
        return None

    def evaluate_budget(self, agent_id: str) -> Denial | None:
        """Refuse a model call by an agent that declares no ceiling (§5.1).

        Separate from ``evaluate_model`` so that it is never a rule: ``enforce``
        asks it after the rules and never asks ``shadows`` about it, so **a
        budget is never shadowed** by construction rather than by a check on a
        reason code (locked decision #7, D48).
        """
        agent = self.agents.get(agent_id)
        if agent is None or agent.budget is not None:
            return None
        return Denial(
            reason="no_budget",
            rule=f"agents.{agent_id}.budget",
            remedy=(
                f"agents[{agent_id!r}] declares no budget, so every model call "
                f"is refused: no ceiling is not an open one (§5.1). Add a budget "
                f"with a period and limit_usd."
            ),
        )

    def shadows(self, agent_id: str) -> bool:
        """Whether this agent's rule refusals are recorded as ``would_deny`` and
        let through.

        Asked only about the rules' refusals. The ceiling, the price table and
        ``evaluate_budget`` refuse where they are found and never reach here.
        """
        agent = self.agents.get(agent_id)
        # A plan's limit is not a rule, so it is never shadowed (D62).
        return agent is not None and agent.shadow and agent_id not in self.over_plan

    def declares_tool(self, agent_id: str, tool: str) -> bool:
        """Whether this exact name appears in the policy, allowed or denied.

        Asked by ``enforce`` before it writes a tool name down. A name that
        matches something the policy declares is one of a fixed set the operator
        wrote; a name that matches nothing came from the model, and the two cannot
        be recorded on the same terms (D26).
        """
        agent = self.agents.get(agent_id)
        if agent is None:
            return False
        return tool in agent.tools or tool in agent.tools_deny

    def evaluate_tool(
        self,
        agent_id: str,
        tool: str,
        arguments: Mapping[str, object],
        *,
        recall: Recall | None = None,
    ) -> Denial | None:
        """Decide a tool call. ``None`` means permitted.

        Deny-by-default at every step: an unknown agent, a tool with no allow
        rule, and an argument the rule does not name are all refusals.

        ``recall`` is what this session's admitted calls left behind
        (``remember``). ``None``, the default, says the caller keeps no memory at
        all, and then a tool with ``requires``, ``rate`` or ``repeat`` is refused,
        because the rule cannot be checked: a rule that cannot be checked is not
        a rule that passed. So the default is the safe one.
        """
        agent = self.agents.get(agent_id)
        if agent is None:
            return Denial(
                reason="agent_unknown",
                rule=f"agent.{agent_id}",
                remedy=(
                    f"add an agent with id {agent_id!r} to the policy, or pass an "
                    f"agent_id that the policy declares."
                ),
            )
        over = self._over_plan(agent_id)
        if over is not None:
            return over
        if tool in agent.tools_deny:
            return Denial(
                reason="tool_denied",
                rule=tool,
                remedy=(
                    f"{tool!r} is on agents[{agent_id!r}].tools.deny, and an "
                    f"explicit deny always wins. Remove it there if this is meant "
                    f"to be permitted."
                ),
            )
        rule = agent.tools.get(tool)
        if rule is None:
            declared = ", ".join(sorted(agent.tools)) or "no tools at all"
            return Denial(
                reason="tool_not_allowed",
                rule=UNDECLARED_TOOL,
                remedy=(
                    f"agents[{agent_id!r}].tools.allow declares {declared}, and the "
                    f"call named none of them. The name it did use is not repeated "
                    f"here: a tool name the policy does not define was chosen by the "
                    f"model, and a model under injection chooses it (D19, D26). Add "
                    f"the tool to that list if it should be permitted."
                ),
            )
        if rule.constraints is not None:
            denial = _check_arguments(agent_id, rule.name, rule.constraints, arguments)
            if denial is not None:
                return denial
        return _check_memory(rule, arguments, recall)

    def needs_memory(self, agent_id: str, tool: str) -> bool:
        """Whether judging or remembering a call to ``tool`` reads or writes
        memory: it has a remembering rule, or leaves a footprint for one."""
        agent = self.agents.get(agent_id)
        if agent is None:
            return False
        rule = agent.tools.get(tool)
        return (
            tool in agent.watched
            or tool in agent.compared_after
            or (rule is not None and rule.requires is not None)
        )

    def horizon(self, agent_id: str) -> int:
        """How many seconds back this agent's ``rate`` and ``repeat`` rules look:
        a remembered call older than this can be forgotten."""
        agent = self.agents.get(agent_id)
        return agent.horizon if agent is not None else 0

    def remember(
        self,
        agent_id: str,
        tool: str,
        arguments: Mapping[str, object],
        recall: Recall,
    ) -> tuple[frozenset[Footprint], Call | None]:
        """What a call the rules admitted leaves behind: a footprint for each
        ``requires`` that names ``tool``, and the call itself if a ``rate`` or
        ``repeat`` rule watches ``tool``. Digests carry ``recall.salt``.

        A call missing an argument a rule compares, or holding a value that is not
        a plain scalar, leaves no footprint for that rule, so a later call that
        needs it is refused rather than matched loosely.
        """
        agent = self.agents.get(agent_id)
        if agent is None:
            return frozenset(), None
        found = set()
        for same in agent.compared_after.get(tool, ()):
            digest = _digest(same, arguments, recall.salt)
            if digest is not None:
                found.add(Footprint(tool, same, digest))
        call = (
            Call(
                tool,
                recall.now,
                # Only `repeat` reads a digest: a rate-only tool keeps none.
                None
                if agent.tools[tool].repeat is None
                else _call_digest(agent.tools[tool], arguments, recall.salt),
            )
            if tool in agent.watched
            else None
        )
        return frozenset(found), call


def _check_arguments(
    agent_id: str,
    tool: str,
    constraints: Mapping[str, Constraint],
    arguments: Mapping[str, object],
) -> Denial | None:
    """Judge the arguments of a tool that declares a closed set of them (D14).

    Both loops iterate in sorted order so that a call failing two checks always
    reports the same one.

    **Nothing the caller supplied is echoed back.** A name that *matches* the
    policy is safe to name, because the policy is where it came from. A name that
    does **not** match is caller-supplied content — an agent under injection
    chooses it — so it is no more loggable than an argument value, and a denial
    describes it by the rule it broke instead (locked decision #5, D19).
    """
    where = f"agents[{agent_id!r}].tools.allow[{tool!r}].constraints"
    permitted = ", ".join(sorted(constraints)) or "no arguments at all"

    if any(not isinstance(name, str) for name in arguments):
        # A non-string key cannot match anything the policy declares, and sorting
        # a mixed-type key set raises TypeError — which would escape the error
        # taxonomy entirely rather than being the denial it plainly is.
        return Denial(
            reason="argument_not_permitted",
            rule=f"{tool}.<unpermitted>",
            remedy=(
                f"{tool!r} was called with an argument name that is not a string. "
                f"It accepts only these names: {permitted}."
            ),
        )

    for name in sorted(arguments):
        if name not in constraints:
            return Denial(
                reason="argument_not_permitted",
                rule=f"{tool}.<unpermitted>",
                remedy=(
                    f"{tool!r} declares constraints, so it accepts only these "
                    f"argument names: {permitted}. The call passed one that is "
                    f"not among them. It is not named here because a name the "
                    f"policy does not define is caller-supplied content, and "
                    f"audit records carry no payload. Either stop passing it, or "
                    f"add it to {where} as {{}} to permit it unchecked."
                ),
            )

    for name in sorted(constraints):
        constraint = constraints[name]
        if name not in arguments:
            if constraint.is_empty:
                continue
            return Denial(
                reason="argument_missing",
                rule=f"{tool}.{name}",
                remedy=(
                    f"{name!r} is constrained, so it must be supplied — a "
                    f"constraint that can be dodged by omitting the argument is "
                    f'not a constraint. Pass it, or relax it to "{name}": {{}} in '
                    f"{where}."
                ),
            )
        failed = constraint.failing_predicate(arguments[name])
        if failed is not None:
            return Denial(
                reason="constraint_violated",
                rule=f"{tool}.{name}.{failed}",
                remedy=(
                    f"the call must satisfy {where}[{name!r}][{failed!r}]. Change "
                    f"the call, or change the policy if the limit is wrong."
                ),
            )
    return None


def _check_memory(
    rule: ToolRule, arguments: Mapping[str, object], recall: Recall | None
) -> Denial | None:
    """The rules that remember, after every rule that does not (D58, D59)."""
    remembering = (
        rule.requires is not None or rule.rate is not None or rule.repeat is not None
    )
    if not remembering:
        return None
    if recall is None:
        return _unavailable(rule.name)
    if rule.requires is not None:
        denial = _check_requirement(rule.name, rule.requires, arguments, recall)
        if denial is not None:
            return denial
    return _check_frequency(rule, arguments, recall)


def _unavailable(tool: str) -> Denial:
    """A remembering rule, and nothing to remember with: refused, never skipped."""
    return Denial(
        reason="memory_unavailable",
        rule=f"{tool}.memory",
        remedy=(
            f"{tool!r} has a requires, rate or repeat rule, and what judged this "
            f"call had no memory of earlier ones: replay, init's check, a guard "
            f"whose agent sent no session id, or a guard whose memory file for "
            f"this session could not be read (delete it under .paveo/memory to "
            f"start that session afresh). A library Session and the `paveo` "
            f"guard remember; anywhere else such a tool is refused."
        ),
    )


def _check_requirement(
    tool: str,
    requirement: Requirement,
    arguments: Mapping[str, object],
    recall: Recall,
) -> Denial | None:
    """Judge ``requires`` (D58). Nothing the caller supplied is echoed: the rule
    and the argument names come from the policy, the values never appear."""
    rule = f"{tool}.requires.{requirement.tool}"
    same = f" with the same {', '.join(requirement.same)}" if requirement.same else ""
    for name in requirement.same:
        if name not in arguments:
            return Denial(
                reason="argument_missing",
                rule=f"{tool}.{name}",
                remedy=(
                    f"{name!r} is compared by {rule}, so it must be supplied: a "
                    f"check that can be dodged by omitting the argument is not a "
                    f"check."
                ),
            )
    digest = _digest(requirement.same, arguments, recall.salt)
    if digest is None or Footprint(requirement.tool, requirement.same, digest) not in (
        recall.footprints
    ):
        return Denial(
            reason="requires_unmet",
            rule=rule,
            remedy=(
                f"{tool!r} may be called only after {requirement.tool!r}{same}, "
                f"earlier in this session, and no such call was admitted. Make "
                f"that call first, or change the policy if the order is wrong."
            ),
        )
    return None


def _check_frequency(
    rule: ToolRule, arguments: Mapping[str, object], recall: Recall
) -> Denial | None:
    """Judge ``repeat``, then ``rate`` (D59). ``recall.now`` never runs backwards:
    whoever holds the memory keeps the latest time it has seen (``Recall``)."""
    mine = [call for call in recall.calls if call.tool == rule.name]
    if rule.repeat is not None:
        seconds = rule.repeat.seconds
        digest = _call_digest(rule, arguments, recall.salt)
        if digest is None:
            return Denial(
                reason="not_comparable",
                rule=f"{rule.name}.repeat",
                remedy=(
                    f"{rule.name!r} refuses a call repeated within "
                    f"{seconds} seconds, and this call's arguments could "
                    f"not be serialised to compare. Pass JSON values."
                ),
            )
        since = recall.now - seconds
        if any(call.digest == digest and call.at > since for call in mine):
            compared = (
                f"the same {', '.join(rule.repeat.same)}"
                if rule.repeat.same
                else "exactly these arguments"
            )
            return Denial(
                reason="repeated",
                rule=f"{rule.name}.repeat",
                remedy=(
                    f"{rule.name!r} was called with {compared} less than "
                    f"{seconds} seconds ago. Change the limit if repeating it is "
                    f"ordinary work."
                ),
            )
    if rule.rate is not None:
        since = recall.now - rule.rate.seconds
        if sum(call.at > since for call in mine) >= rule.rate.calls:
            return Denial(
                reason="rate_limited",
                rule=f"{rule.name}.rate",
                remedy=(
                    f"{rule.name!r} may be called at most {rule.rate.calls} times in "
                    f"{rule.rate.seconds} seconds, and that many were admitted. Raise "
                    f"the limit if this pace is ordinary work."
                ),
            )
    return None


def _call_digest(
    rule: ToolRule, arguments: Mapping[str, object], salt: bytes
) -> str | None:
    """SHA-256 of the arguments ``repeat`` compares: all of them, or the ones
    its ``same`` names, each with whether it was passed at all. Unlike
    ``_digest``, containers are fine: two different calls that serialise alike
    are refused as a repeat, a false refusal, and never let through."""
    same = None if rule.repeat is None else rule.repeat.same
    compared = (
        dict(arguments)
        if same is None
        else [[name, name in arguments, arguments.get(name)] for name in same]
    )
    try:
        encoded = canonical_json(compared)
    except (TypeError, ValueError, RecursionError):  # foreign type, NaN, too deep
        return None
    return "sha256:" + hashlib.sha256(salt + encoded).hexdigest()


def _digest(
    same: tuple[str, ...], arguments: Mapping[str, object], salt: bytes
) -> str | None:
    """SHA-256 of the canonical JSON of the ``same`` values, in the rule's order.

    ``None`` when one is missing or is not a string, integer, boolean or null:
    such a call can neither meet a requirement nor leave a footprint. Only those,
    because JSON writes each of them one way, keeping its type, so two values
    share a digest only if they are equal (``1``, ``"1"`` and ``True`` all
    differ). A container would not: ``{1: x}`` and ``{"1": x}``, or a tuple and a
    list, serialise alike (/code-review, D58). A false refusal, never a false
    match.
    """
    values = [arguments.get(name, _MISSING) for name in same]
    if not all(value is None or type(value) in _SCALARS for value in values):
        return None
    try:
        encoded = canonical_json(values)
    except ValueError:  # an integer past Python's digit limit for str()
        return None
    return "sha256:" + hashlib.sha256(salt + encoded).hexdigest()


_MISSING = object()
# Exact types, not subclasses: a str or int subclass could serialise as anything.
_SCALARS = (str, int, bool)


def _comparable_number(value: object) -> Decimal | None:
    """Coerce a caller-supplied argument to an exact ``Decimal``, or refuse.

    ``float`` is converted through ``str`` rather than directly: ``Decimal(0.1)``
    is 0.1000000000000000055511151231257827, while ``Decimal(str(0.1))`` is
    ``0.1``. ``str`` of a float round-trips, so this is the faithful decimal name
    for that value and never invents precision.

    ``bool`` is excluded deliberately. It is a subclass of ``int`` in Python, so
    without this ``True`` would silently compare as 1 against a numeric ceiling.

    A ``str`` must be a plain decimal (``_PLAIN_DECIMAL``) rather than anything
    ``Decimal`` happens to accept — see that constant for why.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return Decimal(value)
    if isinstance(value, Decimal):
        return value if value.is_finite() else None
    if isinstance(value, float):
        number = Decimal(str(value))
        return number if number.is_finite() else None
    if isinstance(value, str) and _PLAIN_DECIMAL.fullmatch(value):
        return Decimal(value)
    return None


def _same_scalar(value: object, expected: str | bool) -> bool:
    """Equality without type coercion, so ``True`` never equals ``"true"`` or 1."""
    if isinstance(expected, bool):
        return isinstance(value, bool) and value == expected
    return isinstance(value, str) and value == expected
