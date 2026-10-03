"""Turning a policy document into a ``Policy``, and refusing the ones that are wrong.

Loading is hostile on purpose. A policy file is the definition of *allowed*, so
every way it can be wrong has to be loud here rather than permissive at call
time: unknown keys are a hard error, a decimal number where money belongs is a
hard error, and a pattern that could blow up the regex engine is rejected before
it is ever compiled.

**No float is ever constructed while loading a policy.** ``_parse_json`` refuses
the literal and ``_reject_floats`` refuses the Python object, so both entry
points — a file and a dict handed to ``Paveo.from_policy`` — behave
identically. Money read from a binary float is the root of a whole class of
accounting bugs, and this is the cheapest place to make it impossible.

The only code this module compiles is a regular expression, under the limits in
``_compile_pattern``. There is no expression language, no callable, no ``eval``,
no ``pickle`` and no dynamic import.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from decimal import Decimal, InvalidOperation
from pathlib import Path
from types import MappingProxyType
from typing import NoReturn

from ._canonical import canonical_json
from .errors import ConfigError
from .policy import (
    AgentPolicy,
    BudgetPolicy,
    Constraint,
    DeclaredPrice,
    Policy,
    Rate,
    Repeat,
    Requirement,
    ToolRule,
)

_TOP_KEYS = frozenset({"version", "policy_id", "defaults", "agents", "prices"})
_PRICE_KEYS = frozenset({"input_per_mtok", "output_per_mtok", "cached_input_per_mtok"})
_DEFAULTS_KEYS = frozenset({"decision", "assumed_max_output_tokens", "fail_open"})
_AGENT_KEYS = frozenset({"id", "mode", "budget", "models", "tools"})
_MODES = frozenset({"enforce", "shadow"})
_BUDGET_KEYS = frozenset({"period", "limit_usd"})
_RULESET_KEYS = frozenset({"allow", "deny"})
_TOOL_RULE_KEYS = frozenset({"name", "constraints", "requires", "rate", "repeat"})
_REQUIRES_KEYS = frozenset({"tool", "same"})
_RATE_KEYS = frozenset({"calls", "seconds"})
_REPEAT_KEYS = frozenset({"seconds", "same"})
# A window longer than a day is a budget, not a rate: remembering a day of calls
# is already the most a session file should hold (D59).
_MAX_WINDOW_SECONDS = 86_400
_MAX_RATE_CALLS = 10_000
_PREDICATE_KEYS = frozenset({"max", "min", "in", "equals", "matches", "not_matches"})

_PERIODS = frozenset({"day", "hour", "session"})
_SUPPORTED_VERSION = 1

# A policy is a small document written by an operator. These caps exist so that a
# malformed or hostile one fails fast instead of consuming the process.
_MAX_FILE_BYTES = 1 << 20
_MAX_DOCUMENT_DEPTH = 32
_MAX_PATTERN_LENGTH = 200
_QUANTIFIERS = "*+?{"
_REPEATS = "*+{"  # the quantifiers that repeat; ? only makes optional


def load_file(path: str | Path) -> Policy:
    """Load and validate a policy from a JSON file.

    YAML is an optional extra in the spec (§5.1) and is not implemented yet;
    ``paveo.json`` is the canonical form.
    """
    location = Path(path)
    try:
        # Size first, then read. Checking afterwards would mean pulling an
        # arbitrarily large file into memory before deciding to refuse it.
        size = location.stat().st_size
        if size > _MAX_FILE_BYTES:
            raise ConfigError(
                f"the policy file at {location} is {size} bytes, over the "
                f"{_MAX_FILE_BYTES} byte limit.",
                remedy=(
                    "a policy is a small document; check you pointed at the right file."
                ),
            )
        raw = location.read_bytes()
    except OSError as e:
        raise ConfigError(
            f"could not read the policy file at {location}: {e.strerror}.",
            remedy="check the path and that the process can read it.",
        ) from e
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as e:
        raise ConfigError(
            f"the policy file at {location} is not valid UTF-8.",
            remedy="save it as UTF-8.",
        ) from e
    return load_document(_parse_json(text, source=str(location)), source=str(location))


def load_document(document: object, *, source: str = "<policy>") -> Policy:
    """Validate an already-parsed policy document and freeze it into a ``Policy``.

    The document is never retained or mutated: everything is copied into
    immutable structures, so a caller that edits their dict afterwards does not
    edit the policy.
    """
    _reject_floats(document, path="policy")
    top = _mapping(document, where="the policy")
    _reject_unknown_keys(top, _TOP_KEYS, where="the policy")

    version = top.get("version")
    if version != _SUPPORTED_VERSION:
        raise ConfigError(
            f"the policy declares version {version!r}, and this build "
            f"understands version {_SUPPORTED_VERSION}.",
            remedy=f'set "version": {_SUPPORTED_VERSION}.',
        )

    policy_id = _text(top.get("policy_id"), where="policy_id")
    fail_open, assumed_max_output_tokens = _read_defaults(top.get("defaults"))
    agents = _read_agents(top.get("agents"))
    prices = _read_prices(top.get("prices"))

    try:
        digest = hashlib.sha256(canonical_json(document)).hexdigest()
    except ValueError as e:
        raise ConfigError(
            "the policy contains a structure that refers to itself.",
            remedy="a policy is a plain tree; remove the self-reference.",
        ) from e
    return Policy(
        policy_id=policy_id,
        fail_open=fail_open,
        assumed_max_output_tokens=assumed_max_output_tokens,
        agents=MappingProxyType(agents),
        policy_hash=f"sha256:{digest}",
        source=source,
        prices=MappingProxyType(prices),
    )


def _read_defaults(raw: object) -> tuple[bool, int | None]:
    if raw is None:
        return False, None
    defaults = _mapping(raw, where="defaults")
    _reject_unknown_keys(defaults, _DEFAULTS_KEYS, where="defaults")

    decision = defaults.get("decision", "deny")
    if decision != "deny":
        raise ConfigError(
            f"defaults.decision is {decision!r}, and the only value this build "
            f'accepts is "deny".',
            remedy=(
                'remove the key or set it to "deny". Allow-by-default is how '
                "agents end up acting outside their intended scope, so a policy "
                "file is not permitted to switch it off (locked decision #6)."
            ),
        )

    fail_open = defaults.get("fail_open", False)
    if not isinstance(fail_open, bool):
        raise ConfigError(
            f"defaults.fail_open must be true or false, not "
            f"{type(fail_open).__name__}.",
            remedy='write "fail_open": false, or omit it — false is the default.',
        )

    assumed = defaults.get("assumed_max_output_tokens")
    if assumed is not None and (
        isinstance(assumed, bool) or not isinstance(assumed, int)
    ):
        raise ConfigError(
            "defaults.assumed_max_output_tokens must be a whole number of tokens "
            "or null.",
            remedy=(
                "omit it and set max_tokens on each call instead, which is the "
                "recommended option (§4.4)."
            ),
        )
    if isinstance(assumed, int) and not isinstance(assumed, bool) and assumed <= 0:
        raise ConfigError(
            f"defaults.assumed_max_output_tokens is {assumed}, which cannot bound "
            f"anything.",
            remedy="give a positive number of tokens, or omit the key.",
        )
    return fail_open, assumed


def _read_prices(raw: object) -> dict[str, DeclaredPrice]:
    """Prices the operator declares for models no built-in table carries (§4.8.4).

    Whether a declared model collides with a built-in one is checked where the
    table is built, not here: the loader does not know the table, and should not.
    """
    if raw is None:
        return {}
    declared = _mapping(raw, where="prices")
    prices: dict[str, DeclaredPrice] = {}
    for model, entry in declared.items():
        where = f"prices[{model!r}]"
        fields = _mapping(entry, where=where)
        _reject_unknown_keys(fields, _PRICE_KEYS, where=where)
        cached = fields.get("cached_input_per_mtok")
        prices[model] = DeclaredPrice(
            input_per_mtok=_price(
                fields.get("input_per_mtok"), where=where, key="input_per_mtok"
            ),
            output_per_mtok=_price(
                fields.get("output_per_mtok"), where=where, key="output_per_mtok"
            ),
            cached_input_per_mtok=(
                None
                if cached is None
                else _price(cached, where=where, key="cached_input_per_mtok")
            ),
        )
    return prices


def _price(value: object, *, where: str, key: str) -> Decimal:
    if value is None:
        raise ConfigError(
            f"{where} does not set {key}.",
            remedy=(
                f'write {where} as {{"input_per_mtok": "2.00", "output_per_mtok": '
                f'"6.00"}}: USD per million tokens, as your provider bills you.'
            ),
        )
    number = _money(value, where=f"{where}.{key}")
    if number < 0:
        raise ConfigError(
            f"{where}.{key} is negative.",
            remedy="a price is zero or more; zero is right for a model you host.",
        )
    return number


def _read_agents(raw: object) -> dict[str, AgentPolicy]:
    if not isinstance(raw, list) or not raw:
        raise ConfigError(
            "the policy declares no agents.",
            remedy=(
                'add at least one entry to "agents". A policy with no agents '
                "denies every call, which is safe but useless."
            ),
        )
    agents: dict[str, AgentPolicy] = {}
    for index, entry in enumerate(raw):
        agent = _read_agent(entry, index=index)
        if agent.agent_id in agents:
            raise ConfigError(
                f"two agents share the id {agent.agent_id!r}.",
                remedy="give each agent a unique id; the second would be ignored.",
            )
        agents[agent.agent_id] = agent
    return agents


def _read_agent(raw: object, *, index: int) -> AgentPolicy:
    where = f"agents[{index}]"
    entry = _mapping(raw, where=where)
    _reject_unknown_keys(entry, _AGENT_KEYS, where=where)
    agent_id = _text(entry.get("id"), where=f"{where}.id")

    models_allow, models_deny = _read_ruleset(
        entry.get("models"), where=f"agents[{agent_id!r}].models"
    )
    tools, tools_deny = _read_tools(
        entry.get("tools"), where=f"agents[{agent_id!r}].tools"
    )
    compared_after: dict[str, set[tuple[str, ...]]] = {}
    windows_by_tool: dict[str, int] = {}
    for rule in tools.values():
        if rule.requires is not None:
            compared_after.setdefault(rule.requires.tool, set()).add(rule.requires.same)
        windows = [
            window
            for window in (
                rule.rate.seconds if rule.rate is not None else None,
                rule.repeat.seconds if rule.repeat is not None else None,
            )
            if window is not None
        ]
        if windows:
            windows_by_tool[rule.name] = max(windows)
    return AgentPolicy(
        agent_id=agent_id,
        budget=_read_budget(entry.get("budget"), agent_id=agent_id),
        models_allow=models_allow,
        models_deny=models_deny,
        tools=MappingProxyType(tools),
        tools_deny=tools_deny,
        shadow=_read_mode(entry.get("mode", "enforce"), agent_id=agent_id),
        compared_after=MappingProxyType(
            {tool: tuple(sorted(sames)) for tool, sames in compared_after.items()}
        ),
        watched=frozenset(windows_by_tool),
        horizon=max(windows_by_tool.values(), default=0),
    )


def _read_mode(raw: object, *, agent_id: str) -> bool:
    """True for ``"shadow"``. Anything but the two words is refused, so a typo
    never quietly turns enforcement off, or quietly leaves it on (D48)."""
    if not isinstance(raw, str) or raw not in _MODES:
        raise ConfigError(
            f"agents[{agent_id!r}].mode is not one of {', '.join(sorted(_MODES))}.",
            remedy=(
                'omit it to enforce, or write "shadow" to record what the rules '
                "would refuse and let those calls through. The budget enforces "
                "either way."
            ),
        )
    return raw == "shadow"


def _read_budget(raw: object, *, agent_id: str) -> BudgetPolicy | None:
    if raw is None:
        return None
    where = f"agents[{agent_id!r}].budget"
    budget = _mapping(raw, where=where)
    _reject_unknown_keys(budget, _BUDGET_KEYS, where=where)

    period = _text(budget.get("period"), where=f"{where}.period")
    if period not in _PERIODS:
        raise ConfigError(
            f"{where}.period is {period!r}.",
            remedy=f"use one of: {', '.join(sorted(_PERIODS))}.",
        )
    limit = _money(budget.get("limit_usd"), where=f"{where}.limit_usd")
    if limit <= 0:
        raise ConfigError(
            f"{where}.limit_usd is {limit}, so no call could ever be afforded.",
            remedy="set a positive ceiling.",
        )
    return BudgetPolicy(period=period, limit_usd=limit)


def _read_ruleset(raw: object, *, where: str) -> tuple[frozenset[str], frozenset[str]]:
    if raw is None:
        return frozenset(), frozenset()
    ruleset = _mapping(raw, where=where)
    _reject_unknown_keys(ruleset, _RULESET_KEYS, where=where)
    return (
        frozenset(_text_list(ruleset.get("allow"), where=f"{where}.allow")),
        frozenset(_text_list(ruleset.get("deny"), where=f"{where}.deny")),
    )


def _read_tools(
    raw: object, *, where: str
) -> tuple[dict[str, ToolRule], frozenset[str]]:
    if raw is None:
        return {}, frozenset()
    tools = _mapping(raw, where=where)
    _reject_unknown_keys(tools, _RULESET_KEYS, where=where)

    allow = tools.get("allow")
    rules: dict[str, ToolRule] = {}
    if allow is not None:
        if not isinstance(allow, list):
            raise ConfigError(
                f"{where}.allow must be a list of tool rules.",
                remedy='write [{"name": "lookup_order"}].',
            )
        for index, entry in enumerate(allow):
            rule = _read_tool_rule(entry, where=f"{where}.allow[{index}]")
            if rule.name in rules:
                raise ConfigError(
                    f"{where}.allow lists {rule.name!r} twice.",
                    remedy="merge the two entries; the second would be ignored.",
                )
            rules[rule.name] = rule
    deny = frozenset(_text_list(tools.get("deny"), where=f"{where}.deny"))
    # A denied tool is never called, so its own `requires` is never judged: only
    # the tools that can be called are held to one (/code-review, D58).
    admissible = {name: rule for name, rule in rules.items() if name not in deny}
    for rule in admissible.values():
        _check_satisfiable(
            rule, admissible, where=f"{where}.allow[{rule.name!r}].requires"
        )
    return rules, deny


def _check_satisfiable(
    rule: ToolRule, rules: Mapping[str, ToolRule], *, where: str
) -> None:
    """Refuse a ``requires`` that could never be met (D58): one naming a tool the
    agent may not call, a chain that comes back to where it started (the tool
    itself, or A after B after A), or an argument one of the two tools could never
    be passed. Loud here, rather than a tool that is silently never callable."""
    requirement = rule.requires
    if requirement is None:
        return
    seen = [rule.name]
    step: ToolRule | None = rule
    while step is not None and step.requires is not None:
        if step.requires.tool in seen:
            chain = " after ".join(repr(name) for name in [*seen, step.requires.tool])
            raise ConfigError(
                f"{where} can never be met: {chain}, so none of them can be "
                f"called first.",
                remedy="break the chain: one of these tools must need nothing.",
            )
        seen.append(step.requires.tool)
        step = rules.get(step.requires.tool)
    earlier = rules.get(requirement.tool)
    if earlier is None:
        raise ConfigError(
            f"{where} names {requirement.tool!r}, which this agent is not allowed "
            f"to call, so {rule.name!r} could never be called.",
            remedy=(
                f"allow {requirement.tool!r} for the same agent, and keep it off "
                f"tools.deny."
            ),
        )
    for tool in (rule, earlier):
        undeclared = sorted(
            name
            for name in requirement.same
            if tool.constraints is not None and name not in tool.constraints
        )
        if undeclared:
            raise ConfigError(
                f"{where}.same compares {', '.join(map(repr, undeclared))}, which "
                f"{tool.name!r} declares constraints without, so it could never "
                f"be passed.",
                remedy=f"declare it in {tool.name!r}'s constraints, as {{}} if "
                f"it needs no limit.",
            )


def _read_tool_rule(raw: object, *, where: str) -> ToolRule:
    entry = _mapping(raw, where=where)
    _reject_unknown_keys(entry, _TOOL_RULE_KEYS, where=where)
    name = _text(entry.get("name"), where=f"{where}.name")

    raw_constraints = entry.get("constraints")
    constraints = None
    if raw_constraints is not None:
        declared = _mapping(raw_constraints, where=f"{where}.constraints")
        constraints = MappingProxyType(
            {
                argument: _read_constraint(
                    predicates, where=f"{where}.constraints[{argument!r}]"
                )
                for argument, predicates in declared.items()
            }
        )
    return ToolRule(
        name=name,
        constraints=constraints,
        requires=_read_requires(entry.get("requires"), where=f"{where}.requires"),
        rate=_read_rate(entry.get("rate"), where=f"{where}.rate"),
        repeat=_read_repeat(
            entry.get("repeat"), where=f"{where}.repeat", constraints=constraints
        ),
    )


def _read_rate(raw: object, *, where: str) -> Rate | None:
    """``{"calls": 20, "seconds": 60}``: whole numbers, both positive (D59)."""
    if raw is None:
        return None
    entry = _mapping(raw, where=where)
    _reject_unknown_keys(entry, _RATE_KEYS, where=where)
    return Rate(
        calls=_count(entry.get("calls"), where=f"{where}.calls", most=_MAX_RATE_CALLS),
        seconds=_count(
            entry.get("seconds"), where=f"{where}.seconds", most=_MAX_WINDOW_SECONDS
        ),
    )


def _read_repeat(
    raw: object, *, where: str, constraints: Mapping[str, Constraint] | None
) -> Repeat | None:
    """``{"seconds": 60}``, or with ``"same": ["command"]`` to compare only the
    arguments named (D59). A name the tool's closed constraint set does not
    declare could never be passed, so it does not load (D14)."""
    if raw is None:
        return None
    entry = _mapping(raw, where=where)
    _reject_unknown_keys(entry, _REPEAT_KEYS, where=where)
    seconds = _count(
        entry.get("seconds"), where=f"{where}.seconds", most=_MAX_WINDOW_SECONDS
    )
    if "same" not in entry:
        return Repeat(seconds=seconds, same=None)
    same = _text_list(entry.get("same"), where=f"{where}.same")
    if not same or len(set(same)) != len(same):
        raise ConfigError(
            f"{where}.same must name each argument once, and at least one.",
            remedy='leave "same" out to compare every argument.',
        )
    undeclared = sorted(
        name for name in same if constraints is not None and name not in constraints
    )
    if undeclared:
        raise ConfigError(
            f"{where}.same names {', '.join(map(repr, undeclared))}, which this "
            f"tool declares constraints without, so it could never be passed.",
            remedy="declare it in the tool's constraints, as {} if it needs no limit.",
        )
    return Repeat(seconds=seconds, same=same)


def _count(value: object, *, where: str, most: int) -> int:
    """A whole number from 1 to ``most``. A bool is not one, though Python says
    it is an int, and neither is a string: a limit is written as a number."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(
            f"{where} must be a whole number, not {type(value).__name__}.",
            remedy=f"write {where} as a number from 1 to {most}, without quotes.",
        )
    if not 1 <= value <= most:
        raise ConfigError(
            f"{where} is {value}, outside 1 to {most}.",
            remedy=f"write {where} as a number from 1 to {most}.",
        )
    return value


def _read_requires(raw: object, *, where: str) -> Requirement | None:
    if raw is None:
        return None
    entry = _mapping(raw, where=where)
    _reject_unknown_keys(entry, _REQUIRES_KEYS, where=where)
    same = _text_list(entry.get("same"), where=f"{where}.same")
    if len(set(same)) != len(same):
        raise ConfigError(
            f"{where}.same lists an argument twice.",
            remedy="list each argument once.",
        )
    return Requirement(tool=_text(entry.get("tool"), where=f"{where}.tool"), same=same)


def _read_constraint(raw: object, *, where: str) -> Constraint:
    predicates = _mapping(raw, where=where)
    _reject_unknown_keys(predicates, _PREDICATE_KEYS, where=where)

    explicit_nulls = sorted(k for k, v in predicates.items() if v is None)
    if explicit_nulls:
        # `{"max": null}` would otherwise read exactly like an absent `max`,
        # leaving the argument unconstrained *and* optional. That is the "a key
        # that silently does nothing" class this loader rejects everywhere else.
        raise ConfigError(
            f"{where} sets {', '.join(repr(k) for k in explicit_nulls)} to null.",
            remedy=(
                "remove the key rather than nulling it. As written it reads as "
                "absent, which silently makes the argument optional too."
            ),
        )

    equals = predicates.get("equals")
    if equals is not None and not isinstance(equals, str | bool):
        raise ConfigError(
            f"{where}.equals must be a string or a boolean, not "
            f"{type(equals).__name__}.",
            remedy="compare numbers with max and min, which are exact.",
        )

    pattern = predicates.get("matches")
    forbidden = predicates.get("not_matches")
    return Constraint(
        maximum=(
            None
            if predicates.get("max") is None
            else _money(predicates.get("max"), where=f"{where}.max")
        ),
        minimum=(
            None
            if predicates.get("min") is None
            else _money(predicates.get("min"), where=f"{where}.min")
        ),
        permitted=_read_membership(predicates.get("in"), where=where),
        equals=equals,
        pattern=(
            None
            if pattern is None
            else _compile_pattern(
                _text(pattern, where=f"{where}.matches"), where=f"{where}.matches"
            )
        ),
        # Searched, not fully matched, and ignoring case: a pattern that refuses
        # must find `rm -rf` after `echo ok;`, and `DROP` and `drop` are one
        # statement (D49).
        forbidden=(
            None if forbidden is None else _read_forbidden(forbidden, where=where)
        ),
    )


def _read_forbidden(raw: object, *, where: str) -> tuple[re.Pattern[str], ...]:
    """One pattern or a list of them, each under the limits one pattern has.

    A list rather than a longer cap: a refusal list for a shell tool is long, and
    the 200-character limit on one pattern is part of what keeps them safe (T5).
    """
    patterns = (
        [raw] if isinstance(raw, str) else _text_list(raw, where=f"{where}.not_matches")
    )
    if not patterns:
        raise ConfigError(
            f"{where}.not_matches is an empty list, so it refuses nothing.",
            remedy="list the patterns to refuse, or remove the predicate.",
        )
    return tuple(
        _compile_pattern(
            _text(pattern, where=f"{where}.not_matches"),
            where=f"{where}.not_matches",
            flags=re.IGNORECASE,
        )
        for pattern in patterns
    )


def _read_membership(raw: object, *, where: str) -> tuple[str, ...] | None:
    """Read an ``in`` list, refusing an empty one.

    An empty list would load happily and make the argument both mandatory (it
    carries a predicate, so D14 requires it) and impossible to satisfy (nothing
    is a member), so every call to that tool would be denied forever while the
    policy file looked perfectly reasonable.
    """
    if raw is None:
        return None
    values = _text_list(raw, where=f"{where}.in")
    if not values:
        raise ConfigError(
            f"{where}.in is an empty list, so no value could ever satisfy it.",
            remedy=(
                "list the permitted values, or remove the predicate. As written "
                "it would deny every call to this tool."
            ),
        )
    return values


def _compile_pattern(raw: str, *, where: str, flags: int = 0) -> re.Pattern[str]:
    """Compile a policy regex inside a deliberately narrow envelope (§9, T5).

    ``matches`` is applied with ``fullmatch``, so anchoring is structural rather
    than something the policy author has to remember; ``not_matches`` with
    ``search``, ignoring case, because a refusal must find its pattern anywhere.
    What is rejected here is the shape that makes backtracking blow up — a
    quantified group that itself contains a quantifier — plus backreferences
    and every group form except ``(?:``.

    This reduces ReDoS risk; it does not eliminate it. A repeated group holding an
    alternation, such as ``(a|aa)+``, is refused too, but a chain of optional ones,
    ``(?:a|aa)?(?:a|aa)?...``, and polynomial shapes like ``a*a*a*b`` are not. The
    matched value is length capped as well, regexes are compiled only here, only at
    load, only from a file an operator controls, and the guard refuses any call it
    cannot decide within its deadline; a library caller has no such deadline.
    """
    if len(raw) > _MAX_PATTERN_LENGTH:
        raise ConfigError(
            f"{where} is {len(raw)} characters, over the "
            f"{_MAX_PATTERN_LENGTH} character limit.",
            remedy="policy patterns are meant to be simple; use a shorter one.",
        )
    _reject_dangerous_constructs(raw, where=where)
    try:
        return re.compile(raw, flags)
    except re.error as e:
        raise ConfigError(
            f"{where} is not a valid regular expression ({e.msg}).",
            remedy="test the pattern before putting it in a policy file.",
        ) from e


def _reject_dangerous_constructs(pattern: str, *, where: str) -> None:
    """Refuse the pattern shapes that make backtracking blow up (§9, T5, D19).

    Four rules, and they are deliberately blunt:

    1. no backreferences,
    2. no group form but ``(?:`` — no lookarounds, named groups or inline flags,
    3. **no nested groups**, and no quantifier applied to a group that contains
       one,
    4. no ``*``, ``+`` or ``{`` applied to a group that contains ``|``: a
       repeated choice between overlapping alternatives, ``(?:a|aa)+``, is
       exponential without any nesting (D77).

    Rule 3 does the work of what used to be a small stack machine. That version
    had two state bugs at once: it rejected ``(ORD-[0-9]+)`` — because
    ``"" in "*+?"`` is ``True`` in Python, so a pattern ending in ``)`` looked
    quantified — and it *accepted* ``((a+))*``, because the inner group's
    quantifier was dropped when restoring the enclosing state. Wrong in both
    directions, in forty lines. Forbidding nesting outright removes the state,
    and with it the class of bug: a policy pattern with one level of grouping is
    all the ones in this repo's own examples need.

    This still reduces rather than eliminates ReDoS risk: a chain of optional
    alternation groups and polynomial shapes such as ``a*a*a*b`` pass, which is
    why the matched value is length-capped too, and why the docs say partially
    mitigated.
    """
    depth = 0
    group_has_quantifier = False
    group_has_alternation = False
    index = 0
    in_class = False

    while index < len(pattern):
        char = pattern[index]

        if char == "\\":
            if index + 1 < len(pattern) and pattern[index + 1].isdigit():
                raise ConfigError(
                    f"{where} uses a backreference.",
                    remedy=(
                        "backreferences make matching exponential in the worst "
                        "case; write the pattern without one."
                    ),
                )
            index += 2
            continue

        if in_class:
            in_class = char != "]"
            index += 1
            continue

        if char == "[":
            in_class = True
            index += 1
            continue

        if char == "(":
            if pattern.startswith("(?", index) and not pattern.startswith("(?:", index):
                raise ConfigError(
                    f"{where} uses a group form other than (?:.",
                    remedy=(
                        "lookarounds, named groups and inline flags are not "
                        "permitted in a policy pattern. Use a plain group or (?:."
                    ),
                )
            if depth:
                raise ConfigError(
                    f"{where} nests one group inside another.",
                    remedy=(
                        "policy patterns allow a single level of grouping. "
                        "Nesting is how a quantified group ends up containing "
                        "another quantifier, which is what makes backtracking "
                        "blow up. Flatten the pattern."
                    ),
                )
            depth = 1
            group_has_quantifier = False
            group_has_alternation = False
            index += 3 if pattern.startswith("(?:", index) else 1
            continue

        if char == ")":
            _check_group_close(
                pattern[index + 1 : index + 2],
                quantified=group_has_quantifier,
                alternated=group_has_alternation,
                where=where,
            )
            depth = 0
            index += 1
            continue

        if char == "|" and depth:
            group_has_alternation = True
        if char in _QUANTIFIERS and depth:
            group_has_quantifier = True
        index += 1


def _check_group_close(
    following: str, *, quantified: bool, alternated: bool, where: str
) -> None:
    """Rules 3 and 4 of ``_reject_dangerous_constructs``, at a group's ``)``."""
    if following and following in _QUANTIFIERS and quantified:
        raise ConfigError(
            f"{where} applies a quantifier to a group that already contains one.",
            remedy=(
                "this is the shape that makes backtracking blow up "
                "(for example (a+)+). Rewrite it without the nesting."
            ),
        )
    if following and following in _REPEATS and alternated:
        raise ConfigError(
            f"{where} repeats a group that contains an alternation.",
            remedy=(
                "a repeated choice such as (?:a|aa)+ can backtrack exponentially "
                "on a value the agent writes. Use a character class, [a-z-]+ "
                "rather than (?:[a-z]|-)+, or list the alternatives in separate "
                "patterns."
            ),
        )


def _parse_json(text: str, *, source: str) -> object:
    """Parse a policy, refusing floats and the non-standard JSON constants.

    ``parse_float`` rejects rather than converts, so **no float object is ever
    constructed** while loading a policy. Money read from a binary float is the
    root of a whole class of accounting bugs, and the cheapest place to make it
    impossible is here.
    """

    def reject_float(literal: str) -> NoReturn:
        raise ConfigError(
            f"{source} contains the decimal number {literal}.",
            remedy=(
                f'write it as a string — "{literal}" — so it is read exactly. '
                f"Binary floats cannot represent most decimal amounts."
            ),
        )

    def reject_constant(literal: str) -> NoReturn:
        raise ConfigError(
            f"{source} contains {literal}, which is not valid JSON.",
            remedy="remove it; a policy cannot be validated against a non-number.",
        )

    try:
        return json.loads(
            text, parse_float=reject_float, parse_constant=reject_constant
        )
    except json.JSONDecodeError as e:
        raise ConfigError(
            f"{source} is not valid JSON: {e.msg} at line {e.lineno}, "
            f"column {e.colno}.",
            remedy="check for a trailing comma, a missing brace or a stray quote.",
        ) from e


def _reject_floats(
    value: object, *, path: str, depth: int = 0, seen: set[int] | None = None
) -> None:
    """Refuse a float anywhere in a policy built in Python rather than parsed.

    ``_parse_json`` covers the file path; this covers ``Paveo.from_policy``,
    so both entry points behave identically.

    Each container is visited once. Without that, a document whose sub-objects
    are shared — or which refers to itself twice — is walked once per *path*
    rather than once per node, which is exponential in the depth cap rather than
    linear in the document. Visiting once is also sufficient: whether a subtree
    holds a float does not depend on how it was reached. A genuine cycle is
    caught when the document is hashed.
    """
    if seen is None:
        seen = set()
    if depth > _MAX_DOCUMENT_DEPTH:
        raise ConfigError(
            f"the policy nests more than {_MAX_DOCUMENT_DEPTH} levels deep at {path}.",
            remedy=(
                "a policy is a flat document; check for a structure that "
                "refers to itself."
            ),
        )
    if isinstance(value, float):
        raise ConfigError(
            f"{path} is the floating point number {value!r}.",
            remedy=(
                "pass it as a string or a Decimal so it is read exactly. Binary "
                "floats cannot represent most decimal amounts."
            ),
        )
    if not isinstance(value, Mapping | list | tuple):
        return
    if id(value) in seen:
        return
    seen.add(id(value))
    if isinstance(value, Mapping):
        for key, item in value.items():
            _reject_floats(item, path=f"{path}.{key}", depth=depth + 1, seen=seen)
    else:
        for index, item in enumerate(value):
            _reject_floats(item, path=f"{path}[{index}]", depth=depth + 1, seen=seen)


def _mapping(value: object, *, where: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ConfigError(
            f"{where} must be an object, not {type(value).__name__}.",
            remedy=f"see docs/SPEC_V1.md §5 for the shape of {where}.",
        )
    for key in value:
        if not isinstance(key, str):
            raise ConfigError(
                f"{where} has a non-string key {key!r}.",
                remedy="policy keys are always strings.",
            )
    return value


def _text(value: object, *, where: str) -> str:
    if not isinstance(value, str) or not value:
        raise ConfigError(
            f"{where} must be a non-empty string, not {value!r}.",
            remedy=f"set {where} to a string.",
        )
    return value


def _text_list(value: object, *, where: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ConfigError(
            f"{where} must be a list of strings, not {type(value).__name__}.",
            remedy=f'write {where} as ["one", "two"].',
        )
    return tuple(
        _text(item, where=f"{where}[{index}]") for index, item in enumerate(value)
    )


def _money(value: object, *, where: str) -> Decimal:
    """Read an exact decimal amount. ``float`` never gets this far (see above)."""
    if isinstance(value, Decimal):
        number = value
    elif isinstance(value, bool) or not isinstance(value, int | str):
        raise ConfigError(
            f"{where} must be a decimal written as a string, not "
            f"{type(value).__name__}.",
            remedy=f'write {where} as "500.00".',
        )
    else:
        try:
            number = Decimal(value)
        except InvalidOperation as e:
            raise ConfigError(
                f"{where} is {value!r}, which is not a decimal number.",
                remedy=f'write {where} as "500.00".',
            ) from e
    if not number.is_finite():
        raise ConfigError(
            f"{where} is {number}, which cannot bound anything.",
            remedy=f'write {where} as a finite decimal, for example "500.00".',
        )
    return number


def _reject_unknown_keys(
    mapping: Mapping[str, object], allowed: frozenset[str], *, where: str
) -> None:
    """A typo'd key that silently does nothing is a breach waiting to happen (§5.1)."""
    unknown = sorted(set(mapping) - allowed)
    if unknown:
        raise ConfigError(
            f"{where} has unknown key(s): {', '.join(repr(key) for key in unknown)}.",
            remedy=(
                f"check the spelling. {where} accepts: {', '.join(sorted(allowed))}."
            ),
        )
