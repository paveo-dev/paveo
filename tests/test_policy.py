"""Policy loading and tool evaluation.

The bar these tests set is not "a good policy loads". It is that **every way a
policy can be wrong is loud**, and that **nothing falls through to allowed**.
A policy loader that quietly ignores a typo'd `deny:` is how an agent ends up
acting outside its intended scope (§5.1).
"""

from __future__ import annotations

import copy
import json
from collections.abc import Callable
from decimal import Decimal
from pathlib import Path
from types import MappingProxyType
from typing import cast

import pytest

from paveo._policy_document import load_document, load_file
from paveo.errors import ConfigError

SENTINEL = "patient-name-jane-doe-payload"


def base_policy() -> dict[str, object]:
    """The spec's own §5 example, as JSON rather than the YAML shown there."""
    return {
        "version": 1,
        "policy_id": "prod-2026-09",
        "defaults": {"decision": "deny"},
        "agents": [
            {
                "id": "refund-bot",
                "budget": {"period": "day", "limit_usd": "50.00"},
                "models": {"allow": ["claude-sonnet-5"]},
                "tools": {
                    "allow": [
                        {"name": "lookup_order"},
                        {
                            "name": "refund",
                            "constraints": {
                                "amount_usd": {"max": "500.00"},
                                "currency": {"in": ["USD", "EUR"]},
                            },
                        },
                    ],
                    "deny": ["transfer_funds"],
                },
            }
        ],
    }


def with_tool(constraints: object) -> dict[str, object]:
    """A policy whose single tool carries the given ``constraints`` value."""
    document = base_policy()
    agents = cast("list[dict[str, object]]", document["agents"])
    agents[0]["tools"] = {"allow": [{"name": "refund", "constraints": constraints}]}
    return document


# --------------------------------------------------------------------------
# Loading: the happy path, and that loading changes nothing it was handed
# --------------------------------------------------------------------------


def test_the_spec_example_loads() -> None:
    policy = load_document(base_policy())
    assert policy.policy_id == "prod-2026-09"
    assert policy.fail_open is False
    assert set(policy.agents) == {"refund-bot"}

    agent = policy.agents["refund-bot"]
    assert agent.budget is not None
    assert agent.budget.limit_usd == Decimal("50.00")
    assert agent.budget.period == "day"
    assert agent.models_allow == frozenset({"claude-sonnet-5"})
    assert agent.tools_deny == frozenset({"transfer_funds"})
    assert set(agent.tools) == {"lookup_order", "refund"}
    assert agent.tools["lookup_order"].constraints is None


def test_a_loaded_policy_is_immutable_and_detached_from_its_document() -> None:
    document = base_policy()
    policy = load_document(document)

    agents = cast("list[dict[str, object]]", document["agents"])
    tools = cast("dict[str, object]", agents[0]["tools"])
    cast("list[object]", tools["deny"]).append("refund")
    agents.append({"id": "smuggled-in"})

    assert policy.agents["refund-bot"].tools_deny == frozenset({"transfer_funds"})
    assert set(policy.agents) == {"refund-bot"}

    tool_rules = policy.agents["refund-bot"].tools
    with pytest.raises(TypeError, match="does not support item assignment"):
        tool_rules["refund"] = tool_rules["lookup_order"]  # type: ignore[index]


def test_the_hash_identifies_the_policy_not_its_formatting() -> None:
    reordered: dict[str, object] = {}
    for key in reversed(list(base_policy())):
        reordered[key] = base_policy()[key]

    assert (
        load_document(base_policy()).policy_hash == load_document(reordered).policy_hash
    )
    assert load_document(base_policy()).policy_hash.startswith("sha256:")

    changed = base_policy()
    agents = cast("list[dict[str, object]]", changed["agents"])
    budget = cast("dict[str, object]", agents[0]["budget"])
    budget["limit_usd"] = "50.01"
    assert (
        load_document(changed).policy_hash != load_document(base_policy()).policy_hash
    )


def test_a_decimal_and_its_string_are_the_same_policy() -> None:
    """A policy built in Python may carry Decimals; it means the same thing."""
    document = base_policy()
    agents = cast("list[dict[str, object]]", document["agents"])
    budget = cast("dict[str, object]", agents[0]["budget"])
    budget["limit_usd"] = Decimal("50.00")

    policy = load_document(document)
    assert policy.agents["refund-bot"].budget is not None
    assert policy.agents["refund-bot"].budget.limit_usd == Decimal("50.00")
    assert policy.policy_hash == load_document(base_policy()).policy_hash


def test_a_models_deny_list_loads() -> None:
    """§5.1 says an explicit deny wins at every level, so models carry one too.

    Model enforcement itself arrives with the provider wrapper in chunk 3; what
    this asserts is that the schema is accepted rather than rejected as an
    unknown key, which would make the spec's own rule unwritable.
    """
    document = base_policy()
    agents = cast("list[dict[str, object]]", document["agents"])
    agents[0]["models"] = {"allow": ["claude-sonnet-5"], "deny": ["claude-opus-5"]}

    agent = load_document(document).agents["refund-bot"]
    assert agent.models_allow == frozenset({"claude-sonnet-5"})
    assert agent.models_deny == frozenset({"claude-opus-5"})


def test_loading_from_a_file(tmp_path: Path) -> None:
    location = tmp_path / "paveo.json"
    location.write_text(json.dumps(base_policy()), encoding="utf-8")
    policy = load_file(location)
    assert policy.policy_id == "prod-2026-09"
    assert policy.source == str(location)


# --------------------------------------------------------------------------
# Loading: every way it can be wrong is loud
# --------------------------------------------------------------------------


def test_a_missing_file_is_a_config_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="could not read"):
        load_file(tmp_path / "absent.json")


def test_malformed_json_is_a_config_error(tmp_path: Path) -> None:
    location = tmp_path / "paveo.json"
    location.write_text('{"version": 1,}', encoding="utf-8")
    with pytest.raises(ConfigError, match="not valid JSON"):
        load_file(location)


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(lambda d: d.update(nonsense=True), id="top-level"),
        pytest.param(
            lambda d: cast("dict[str, object]", d["defaults"]).update(deny_all=True),
            id="defaults",
        ),
    ],
)
def test_an_unknown_key_is_a_hard_error(
    mutate: Callable[[dict[str, object]], None],
) -> None:
    document = base_policy()
    mutate(document)
    with pytest.raises(ConfigError, match="unknown key"):
        load_document(document)


def test_an_unknown_key_inside_an_agent_is_a_hard_error() -> None:
    document = base_policy()
    agents = cast("list[dict[str, object]]", document["agents"])
    agents[0]["toolz"] = {}
    with pytest.raises(ConfigError, match="unknown key"):
        load_document(document)


def test_an_unknown_predicate_is_a_hard_error() -> None:
    with pytest.raises(ConfigError, match="unknown key"):
        load_document(with_tool({"amount_usd": {"maximum": "500.00"}}))


def test_a_policy_with_no_agents_is_refused() -> None:
    document = base_policy()
    document["agents"] = []
    with pytest.raises(ConfigError, match="no agents"):
        load_document(document)


def test_an_empty_document_is_refused() -> None:
    with pytest.raises(ConfigError):
        load_document({})


def test_a_wrong_version_is_refused() -> None:
    document = base_policy()
    document["version"] = 2
    with pytest.raises(ConfigError, match="version"):
        load_document(document)


def test_duplicate_agent_ids_are_refused() -> None:
    document = base_policy()
    agents = cast("list[dict[str, object]]", document["agents"])
    agents.append(copy.deepcopy(agents[0]))
    with pytest.raises(ConfigError, match="share the id"):
        load_document(document)


def test_a_policy_file_may_not_switch_off_deny_by_default() -> None:
    document = base_policy()
    document["defaults"] = {"decision": "allow"}
    with pytest.raises(ConfigError, match="locked decision #6"):
        load_document(document)


@pytest.mark.parametrize("period", ["week", "forever", ""])
def test_an_unsupported_budget_period_is_refused(period: str) -> None:
    document = base_policy()
    agents = cast("list[dict[str, object]]", document["agents"])
    cast("dict[str, object]", agents[0]["budget"])["period"] = period
    with pytest.raises(ConfigError):
        load_document(document)


@pytest.mark.parametrize("limit", ["0.00", "-1.00"])
def test_a_ceiling_that_affords_nothing_is_refused(limit: str) -> None:
    document = base_policy()
    agents = cast("list[dict[str, object]]", document["agents"])
    cast("dict[str, object]", agents[0]["budget"])["limit_usd"] = limit
    with pytest.raises(ConfigError, match="no call could ever be afforded"):
        load_document(document)


# --------------------------------------------------------------------------
# Money never arrives as a float, by either entry point
# --------------------------------------------------------------------------


def test_a_decimal_literal_in_a_file_is_refused(tmp_path: Path) -> None:
    location = tmp_path / "paveo.json"
    location.write_text(
        json.dumps(base_policy()).replace('"50.00"', "50.00"), encoding="utf-8"
    )
    with pytest.raises(ConfigError, match="write it as a string"):
        load_file(location)


@pytest.mark.parametrize("literal", ["NaN", "Infinity", "-Infinity"])
def test_the_non_standard_json_constants_are_refused(
    tmp_path: Path, literal: str
) -> None:
    location = tmp_path / "paveo.json"
    location.write_text(
        json.dumps(base_policy()).replace('"50.00"', literal), encoding="utf-8"
    )
    with pytest.raises(ConfigError, match="not valid JSON"):
        load_file(location)


def test_a_float_built_in_python_is_refused() -> None:
    document = base_policy()
    agents = cast("list[dict[str, object]]", document["agents"])
    cast("dict[str, object]", agents[0]["budget"])["limit_usd"] = 50.00
    with pytest.raises(ConfigError, match="floating point"):
        load_document(document)


def test_a_self_referential_document_does_not_hang() -> None:
    document = base_policy()
    document["defaults"] = document
    with pytest.raises(ConfigError):
        load_document(document)


def test_a_document_that_shares_subtrees_is_walked_once_not_once_per_path() -> None:
    """Visiting per path rather than per node is exponential in the depth cap.

    Two back-references over 32 levels is ~2^33 calls, which hangs before any
    other validation runs. The guard is a visited set, and the proof is that this
    returns at all.
    """
    document = base_policy()
    shared: dict[str, object] = {"a": {}, "b": {}}
    for _ in range(40):
        shared = {"a": shared, "b": shared}
    document["defaults"] = {"decision": "deny", "assumed_max_output_tokens": shared}
    with pytest.raises(ConfigError):
        load_document(document)


@pytest.mark.parametrize("predicate", ["max", "min", "in", "equals", "matches"])
def test_a_predicate_set_to_null_is_refused(predicate: str) -> None:
    """Otherwise it reads as absent, silently making the argument optional too."""
    with pytest.raises(ConfigError, match="to null"):
        load_document(with_tool({"x": {predicate: None}}))


def test_an_empty_membership_list_is_refused() -> None:
    """It would make the argument mandatory and impossible to satisfy at once."""
    with pytest.raises(ConfigError, match="no value could ever satisfy"):
        load_document(with_tool({"currency": {"in": []}}))


def test_a_policy_that_is_a_mapping_but_not_a_dict_loads() -> None:
    """`json` special-cases dict, so hashing one of these used to raise TypeError."""
    assert load_document(MappingProxyType(base_policy())).policy_id == "prod-2026-09"


# --------------------------------------------------------------------------
# Regex safety (§9, T5)
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("pattern", "expected"),
    [
        pytest.param("(a+)+", "quantifier to a group", id="nested-quantifier"),
        pytest.param("(a*)*", "quantifier to a group", id="nested-star"),
        pytest.param("([0-9]+)+", "quantifier to a group", id="nested-class"),
        pytest.param(r"(a)\1", "backreference", id="backreference"),
        pytest.param("(?=secret)x", "group form other than", id="lookahead"),
        pytest.param("(?i)abc", "group form other than", id="inline-flag"),
        pytest.param("(?P<name>a)", "group form other than", id="named-group"),
        pytest.param("a" * 201, "over the", id="too-long"),
        pytest.param("(unclosed", "not a valid regular expression", id="invalid"),
        pytest.param("(?:a|aa)+x", "contains an alternation", id="alternation-plus"),
        pytest.param("(ab|a)*", "contains an alternation", id="alternation-star"),
        pytest.param("(?:a|aa){20}", "contains an alternation", id="alternation-count"),
    ],
)
def test_a_dangerous_pattern_never_reaches_the_regex_engine(
    pattern: str, expected: str
) -> None:
    with pytest.raises(ConfigError, match=expected):
        load_document(with_tool({"order_id": {"matches": pattern}}))


@pytest.mark.parametrize(
    "pattern",
    [
        "[A-Z]{2}-[0-9]{4}",
        "(?:ab)+c",
        "order-[0-9]+",
        # The old scanner rejected every one of these, because `"" in "*+?"` is
        # True in Python so any pattern ending in `)` looked quantified (D19).
        "(ORD-[0-9]+)",
        "(a+)",
        "(?:[a-z]+)",
        # An optional choice cannot backtrack exponentially, and nor can a bare one.
        "(?:USD|EUR)?-[0-9]+",
        "USD|EUR|GBP",
        "[a|b]+",
    ],
)
def test_a_reasonable_pattern_is_accepted(pattern: str) -> None:
    load_document(with_tool({"order_id": {"matches": pattern}}))


@pytest.mark.parametrize("pattern", ["((a+))*", "(?:(a+))*", "((a)b)"])
def test_nesting_cannot_smuggle_a_quantified_group_past_the_guard(
    pattern: str,
) -> None:
    """One extra pair of parentheses used to defeat the whole check (D19)."""
    with pytest.raises(ConfigError, match=r"nests one group|quantifier to a group"):
        load_document(with_tool({"order_id": {"matches": pattern}}))


def test_matches_is_a_full_match_not_a_search() -> None:
    policy = load_document(with_tool({"order_id": {"matches": "[0-9]+"}}))
    assert policy.evaluate_tool("refund-bot", "refund", {"order_id": "123"}) is None

    denial = policy.evaluate_tool("refund-bot", "refund", {"order_id": "123 and more"})
    assert denial is not None
    assert denial.rule == "refund.order_id.matches"


# `not_matches` (D49): the pattern that refuses, for a shell command or a path.
DESTRUCTIVE = r"\brm\s+-[a-z]*r[a-z]*f|push\s+(?:--force|-f)\b|drop\s+table"


@pytest.mark.parametrize(
    "command",
    [
        "rm -rf build",
        "echo ok; rm -rf /",  # anywhere in the value, not only at the start
        "git push --force origin main",
        "psql -c 'DROP TABLE users'",  # case does not matter
        "RM -RF /",
    ],
)
def test_not_matches_refuses_its_pattern_anywhere_in_the_value(command: str) -> None:
    policy = load_document(with_tool({"command": {"not_matches": DESTRUCTIVE}}))
    denial = policy.evaluate_tool("refund-bot", "refund", {"command": command})

    assert denial is not None
    assert denial.rule == "refund.command.not_matches"
    assert command not in denial.remedy


@pytest.mark.parametrize("command", ["ls -la", "git push origin main", "rm notes.txt"])
def test_not_matches_permits_what_its_pattern_does_not_find(command: str) -> None:
    policy = load_document(with_tool({"command": {"not_matches": DESTRUCTIVE}}))
    assert policy.evaluate_tool("refund-bot", "refund", {"command": command}) is None


@pytest.mark.parametrize("value", [None, 7, ["rm -rf /"], "x" * ((1 << 20) + 1)])
def test_a_value_not_matches_cannot_search_fails_it(value: object) -> None:
    """Not a string, or too long to search, is not a value known to be safe."""
    policy = load_document(with_tool({"command": {"not_matches": "rm"}}))
    denial = policy.evaluate_tool("refund-bot", "refund", {"command": value})

    assert denial is not None
    assert denial.rule == "refund.command.not_matches"


def test_not_matches_is_a_constraint_so_the_argument_is_required() -> None:
    policy = load_document(with_tool({"command": {"not_matches": "rm"}}))
    denial = policy.evaluate_tool("refund-bot", "refund", {})

    assert denial is not None
    assert denial.reason == "argument_missing"


@pytest.mark.parametrize("pattern", ["(a+)+", "(?=x)", "\\1", "(", "x" * 201])
def test_not_matches_is_held_to_the_same_pattern_limits(pattern: str) -> None:
    with pytest.raises(ConfigError, match="not_matches"):
        load_document(with_tool({"command": {"not_matches": pattern}}))


def test_not_matches_takes_a_list_and_any_pattern_in_it_refuses() -> None:
    """A list rather than a longer pattern: each keeps the 200-character limit."""
    policy = load_document(
        with_tool({"command": {"not_matches": ["\\brm\\s+-rf", "drop\\s+table"]}})
    )
    for command in ("rm -rf /", "DROP TABLE users"):
        denial = policy.evaluate_tool("refund-bot", "refund", {"command": command})
        assert denial is not None
        assert denial.rule == "refund.command.not_matches"
    assert policy.evaluate_tool("refund-bot", "refund", {"command": "ls"}) is None


@pytest.mark.parametrize("patterns", [[], ["ok", "(a+)+"], ["ok", 7], ["ok", ""]])
def test_a_not_matches_list_that_is_empty_or_has_a_bad_pattern_does_not_load(
    patterns: list[object],
) -> None:
    with pytest.raises(ConfigError, match="not_matches"):
        load_document(with_tool({"command": {"not_matches": patterns}}))


# --------------------------------------------------------------------------
# Evaluation: deny by default, at every step
# --------------------------------------------------------------------------


def test_an_unknown_agent_is_denied() -> None:
    policy = load_document(base_policy())
    denial = policy.evaluate_tool("not-in-the-policy", "lookup_order", {})
    assert denial is not None
    assert denial.reason == "agent_unknown"


def test_an_unlisted_tool_is_denied() -> None:
    policy = load_document(base_policy())
    denial = policy.evaluate_tool("refund-bot", "delete_everything", {})
    assert denial is not None
    assert denial.reason == "tool_not_allowed"


def test_an_explicit_deny_beats_an_allow() -> None:
    document = base_policy()
    agents = cast("list[dict[str, object]]", document["agents"])
    tools = cast("dict[str, object]", agents[0]["tools"])
    cast("list[object]", tools["allow"]).append({"name": "transfer_funds"})

    policy = load_document(document)
    denial = policy.evaluate_tool("refund-bot", "transfer_funds", {})
    assert denial is not None
    assert denial.reason == "tool_denied"


def test_a_tool_without_constraints_does_not_check_arguments() -> None:
    policy = load_document(base_policy())
    assert (
        policy.evaluate_tool("refund-bot", "lookup_order", {"anything": SENTINEL})
        is None
    )


# --------------------------------------------------------------------------
# Evaluation: the closed argument set (D14)
# --------------------------------------------------------------------------


def test_a_permitted_call_is_permitted() -> None:
    policy = load_document(base_policy())
    call = {"amount_usd": 250, "currency": "USD"}
    assert policy.evaluate_tool("refund-bot", "refund", call) is None


def test_an_argument_the_rule_does_not_name_is_denied() -> None:
    policy = load_document(base_policy())
    denial = policy.evaluate_tool(
        "refund-bot",
        "refund",
        {"amount_usd": 250, "currency": "USD", "destination": SENTINEL},
    )
    assert denial is not None
    assert denial.reason == "argument_not_permitted"
    assert denial.rule == "refund.<unpermitted>"


def test_an_unpermitted_argument_name_is_never_echoed_back() -> None:
    """A name the policy does not define is caller-supplied content (D19).

    An agent under injection picks it, so it is no more loggable than a value.
    The denial names the permitted set instead, which comes from the policy.
    """
    policy = load_document(base_policy())
    denial = policy.evaluate_tool(
        "refund-bot", "refund", {"amount_usd": 1, "currency": "USD", SENTINEL: 1}
    )
    assert denial is not None
    assert SENTINEL not in denial.rule
    assert SENTINEL not in denial.remedy
    assert "amount_usd, currency" in denial.remedy


def test_a_non_string_argument_name_is_denied_not_an_exception() -> None:
    policy = load_document(base_policy())
    denial = policy.evaluate_tool("refund-bot", "refund", {1: "x", "amount_usd": 1})
    assert denial is not None
    assert denial.reason == "argument_not_permitted"


def test_a_constrained_argument_cannot_be_dodged_by_omitting_it() -> None:
    policy = load_document(base_policy())
    denial = policy.evaluate_tool("refund-bot", "refund", {"currency": "USD"})
    assert denial is not None
    assert denial.reason == "argument_missing"
    assert denial.rule == "refund.amount_usd"


def test_an_empty_constraint_permits_an_argument_without_requiring_it() -> None:
    policy = load_document(with_tool({"amount_usd": {"max": "500.00"}, "note": {}}))
    assert policy.evaluate_tool("refund-bot", "refund", {"amount_usd": 10}) is None
    assert (
        policy.evaluate_tool("refund-bot", "refund", {"amount_usd": 10, "note": "hi"})
        is None
    )


# --------------------------------------------------------------------------
# Evaluation: the predicates themselves
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "amount", [250, "250", "500.00", Decimal("499.999"), 250.5, 0, -5]
)
def test_values_at_or_under_the_ceiling_pass(amount: object) -> None:
    policy = load_document(with_tool({"amount_usd": {"max": "500.00"}}))
    assert policy.evaluate_tool("refund-bot", "refund", {"amount_usd": amount}) is None


@pytest.mark.parametrize(
    "amount",
    [
        pytest.param(600, id="int"),
        pytest.param("600", id="numeric-string"),
        pytest.param(500.01, id="float"),
        pytest.param(Decimal("500.000001"), id="decimal"),
        pytest.param("not a number", id="not-a-number"),
        pytest.param(None, id="null"),
        pytest.param(True, id="bool-is-not-one"),
        pytest.param([500], id="list"),
        pytest.param(float("nan"), id="nan"),
        pytest.param(float("inf"), id="infinity"),
        # Decimal itself accepts all of these; at a money boundary that is type
        # confusion, so a caller-supplied string must be a plain decimal (D19).
        pytest.param(" 250 ", id="padded"),
        pytest.param("2e2", id="exponent"),
        pytest.param("1_0", id="underscored"),
        pytest.param("\u0665\u0660", id="arabic-indic-digits"),
    ],
)
def test_anything_the_ceiling_cannot_bound_is_denied(amount: object) -> None:
    """A value of the wrong shape fails the predicate; it is never coerced."""
    policy = load_document(with_tool({"amount_usd": {"max": "500.00"}}))
    denial = policy.evaluate_tool("refund-bot", "refund", {"amount_usd": amount})
    assert denial is not None
    assert denial.rule == "refund.amount_usd.max"
    assert denial.reason == "constraint_violated"


def test_a_float_is_read_as_its_own_decimal_name() -> None:
    """0.1 + 0.2 is 0.30000000000000004, and the ceiling must see that."""
    policy = load_document(with_tool({"amount_usd": {"max": "0.3"}}))
    assert policy.evaluate_tool("refund-bot", "refund", {"amount_usd": 0.3}) is None
    denial = policy.evaluate_tool("refund-bot", "refund", {"amount_usd": 0.1 + 0.2})
    assert denial is not None
    assert denial.rule == "refund.amount_usd.max"


def test_min_and_max_together() -> None:
    policy = load_document(with_tool({"n": {"min": "1", "max": "10"}}))
    assert policy.evaluate_tool("refund-bot", "refund", {"n": 5}) is None
    for value, predicate in ((0, "min"), (11, "max")):
        denial = policy.evaluate_tool("refund-bot", "refund", {"n": value})
        assert denial is not None
        assert denial.rule == f"refund.n.{predicate}"


@pytest.mark.parametrize(
    ("value", "permitted"),
    [("GBP", False), ("USD", True), ("usd", False), (1, False), (None, False)],
)
def test_in_is_membership_without_coercion(value: object, permitted: bool) -> None:
    policy = load_document(with_tool({"currency": {"in": ["USD", "EUR"]}}))
    denial = policy.evaluate_tool("refund-bot", "refund", {"currency": value})
    assert (denial is None) is permitted
    if denial is not None:
        assert denial.rule == "refund.currency.in"


@pytest.mark.parametrize(
    ("expected", "value", "permitted"),
    [
        ("yes", "yes", True),
        ("yes", "no", False),
        (True, True, True),
        (True, "true", False),
        (True, 1, False),
        (False, False, True),
    ],
)
def test_equals_never_coerces_a_type(
    expected: object, value: object, permitted: bool
) -> None:
    policy = load_document(with_tool({"flag": {"equals": expected}}))
    denial = policy.evaluate_tool("refund-bot", "refund", {"flag": value})
    assert (denial is None) is permitted


def test_equals_must_be_a_string_or_a_boolean() -> None:
    with pytest.raises(ConfigError, match="string or a boolean"):
        load_document(with_tool({"flag": {"equals": 3}}))


def test_the_first_failing_predicate_is_always_the_same_one() -> None:
    """A denial that names a different rule between runs is unauditable."""
    policy = load_document(
        with_tool({"n": {"max": "10", "min": "5", "in": ["7"], "matches": "7"}})
    )
    rules = {policy.evaluate_tool("refund-bot", "refund", {"n": 99}) for _ in range(20)}
    assert len(rules) == 1
    denial = rules.pop()
    assert denial is not None
    assert denial.rule == "refund.n.max"


# --------------------------------------------------------------------------
# Locked decision #5: a denial explains itself without quoting the payload
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "call",
    [
        {"amount_usd": SENTINEL, "currency": "USD"},
        {"amount_usd": 600, "currency": SENTINEL},
        {"amount_usd": 600, "currency": "USD", "memo": SENTINEL},
        {"currency": SENTINEL},
    ],
)
def test_no_argument_value_ever_reaches_a_denial(call: dict[str, object]) -> None:
    policy = load_document(base_policy())
    denial = policy.evaluate_tool("refund-bot", "refund", call)
    assert denial is not None
    assert SENTINEL not in denial.rule
    assert SENTINEL not in denial.reason
    assert SENTINEL not in denial.remedy


def test_every_denial_says_how_to_fix_itself() -> None:
    policy = load_document(base_policy())
    calls: list[tuple[str, str, dict[str, object]]] = [
        ("nobody", "refund", {}),
        ("refund-bot", "transfer_funds", {}),
        ("refund-bot", "unknown_tool", {}),
        ("refund-bot", "refund", {"amount_usd": 600, "currency": "USD"}),
        ("refund-bot", "refund", {"currency": "USD"}),
        ("refund-bot", "refund", {"amount_usd": 1, "currency": "USD", "x": 1}),
    ]
    for agent_id, tool, arguments in calls:
        denial = policy.evaluate_tool(agent_id, tool, arguments)
        assert denial is not None, (agent_id, tool)
        assert denial.remedy.strip()
        assert denial.remedy.endswith(".")
