"""`check_tool`, end to end: policy in, decision out, record on disk.

This is the claim the README makes, tested as a whole rather than in parts — a
permitted call returns, a refused one raises, and either way there is a line in
the log naming the rule and not the value.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from conftest import SENTINEL, make_clock, records_in
from paveo import ConfigError, PolicyDenied, PolicyUnavailable, verify_chain
from paveo.session import Paveo


def policy_document(*, fail_open: bool = False) -> dict[str, object]:
    return {
        "version": 1,
        "policy_id": "prod-2026-09",
        "defaults": {"decision": "deny", "fail_open": fail_open},
        "agents": [
            {
                "id": "refund-bot",
                "budget": {"period": "day", "limit_usd": "50.00"},
                "tools": {
                    "allow": [
                        {"name": "lookup_order"},
                        {
                            "name": "refund",
                            "constraints": {"amount_usd": {"max": "500.00"}},
                        },
                    ],
                    "deny": ["transfer_funds"],
                },
            }
        ],
    }


def open_paveo(tmp_path: Path, *, fail_open: bool = False) -> Paveo:
    return Paveo.from_policy(
        policy_document(fail_open=fail_open),
        audit_path=tmp_path / "audit.jsonl",
        now=make_clock(),
    )


def records(tmp_path: Path) -> list[dict[str, object]]:
    return records_in(tmp_path / "audit.jsonl")


# --------------------------------------------------------------------------
# The README's own example
# --------------------------------------------------------------------------


def test_the_readme_example(tmp_path: Path) -> None:
    with open_paveo(tmp_path) as pf:
        with pf.session(agent_id="refund-bot", principal="user_123") as s:
            s.check_tool("refund", {"amount_usd": 250})

            with pytest.raises(PolicyDenied) as refused:
                s.check_tool("refund", {"amount_usd": 600})
            assert refused.value.rule == "refund.amount_usd.max"

            with pytest.raises(PolicyDenied) as forbidden:
                s.check_tool("transfer_funds")
            assert forbidden.value.reason == "tool_denied"

    assert verify_chain(tmp_path / "audit.jsonl").ok
    assert [r["decision"] for r in records(tmp_path)] == ["allow", "deny", "deny"]


def test_a_permitted_call_returns_nothing_to_check(tmp_path: Path) -> None:
    with open_paveo(tmp_path) as pf, pf.session(agent_id="refund-bot") as s:
        assert s.check_tool("lookup_order", {"order_id": "A-1"}) is None


def test_a_policy_on_disk_behaves_identically(tmp_path: Path) -> None:
    location = tmp_path / "paveo.json"
    location.write_text(json.dumps(policy_document()), encoding="utf-8")

    with (
        Paveo.from_file(location, audit_path=tmp_path / "audit.jsonl") as pf,
        pf.session(agent_id="refund-bot") as s,
    ):
        s.check_tool("refund", {"amount_usd": 10})
        with pytest.raises(PolicyDenied):
            s.check_tool("refund", {"amount_usd": 10_000})


# --------------------------------------------------------------------------
# Identity, and what reaches the record
# --------------------------------------------------------------------------


def test_a_record_carries_both_halves_of_the_identity(tmp_path: Path) -> None:
    with (
        open_paveo(tmp_path) as pf,
        pf.session(agent_id="refund-bot", principal="user_123") as s,
    ):
        s.check_tool("lookup_order")

    (record,) = records(tmp_path)
    assert record["agent_id"] == "refund-bot"
    assert record["principal"] == "user_123"
    assert record["action"] == {"kind": "tool", "name": "lookup_order"}
    assert record["policy_id"] == "prod-2026-09"
    assert str(record["policy_hash"]).startswith("sha256:")


def test_a_principal_is_optional(tmp_path: Path) -> None:
    with open_paveo(tmp_path) as pf, pf.session(agent_id="refund-bot") as s:
        s.check_tool("lookup_order")
    assert records(tmp_path)[0]["principal"] is None


def test_the_cost_fields_are_null_for_a_tool_decision(tmp_path: Path) -> None:
    """§6 names these four exactly. A uniform shape is easier to read a year on."""
    with open_paveo(tmp_path) as pf, pf.session(agent_id="refund-bot") as s:
        s.check_tool("lookup_order")

    (record,) = records(tmp_path)
    for field in (
        "estimated_cost_usd",
        "actual_cost_usd",
        "rate_key",
        "usage_by_class",
    ):
        assert field in record
        assert record[field] is None


def test_a_denial_records_the_rule_and_not_the_value(tmp_path: Path) -> None:
    with open_paveo(tmp_path) as pf, pf.session(agent_id="refund-bot") as s:
        with pytest.raises(PolicyDenied) as refused:
            s.check_tool("refund", {"amount_usd": 600, "memo": SENTINEL})

    assert SENTINEL not in str(refused.value)
    assert SENTINEL not in (tmp_path / "audit.jsonl").read_text(encoding="utf-8")
    (record,) = records(tmp_path)
    assert record["decision"] == "deny"
    assert record["rule"] == "refund.<unpermitted>"
    assert record["reason"] == "argument_not_permitted"


def test_an_injected_argument_name_never_reaches_the_log(tmp_path: Path) -> None:
    """The name is chosen by the model, so it is payload as much as a value is."""
    smuggled = f"note-{SENTINEL}"
    with open_paveo(tmp_path) as pf, pf.session(agent_id="refund-bot") as s:
        with pytest.raises(PolicyDenied):
            s.check_tool("refund", {"amount_usd": 1, smuggled: 1})

    assert SENTINEL not in (tmp_path / "audit.jsonl").read_text(encoding="utf-8")


def test_an_injected_tool_name_never_reaches_the_log(tmp_path: Path) -> None:
    """The sibling of the test above, and it was missing for a day.

    A tool name the policy does not declare matched nothing the operator wrote,
    so it came from the model — and a model under injection chooses it. Found by
    security review: it was landing verbatim in `action.name`, in `rule`, and in
    the raised error, on a *denied* call, with no length cap. The deny path is the
    one an attacker can always reach, which made it the cheapest exfiltration
    channel in the library (D26).
    """
    smuggled = f"exfil-{SENTINEL}"
    with open_paveo(tmp_path) as pf, pf.session(agent_id="refund-bot") as s:
        with pytest.raises(PolicyDenied) as refused:
            s.check_tool(smuggled, {})

    assert SENTINEL not in str(refused.value)
    assert SENTINEL not in (tmp_path / "audit.jsonl").read_text(encoding="utf-8")
    (record,) = records(tmp_path)
    assert record["action"] == {"kind": "tool", "name": "<undeclared>"}
    assert record["rule"] == "<undeclared>"
    assert record["reason"] == "tool_not_allowed"


def test_a_declared_tool_is_named_in_full_even_when_it_is_denied(
    tmp_path: Path,
) -> None:
    """The other half: redaction that hid everything would make the log useless.

    A name on the policy's own deny list is one of a fixed set the operator wrote
    down, so recording it tells an operator what happened and tells an attacker
    nothing they did not already supply.
    """
    with open_paveo(tmp_path) as pf, pf.session(agent_id="refund-bot") as s:
        with pytest.raises(PolicyDenied):
            s.check_tool("transfer_funds", {})

    (record,) = records(tmp_path)
    assert record["action"] == {"kind": "tool", "name": "transfer_funds"}
    assert record["reason"] == "tool_denied"


def test_an_unknown_agent_is_denied_and_recorded(tmp_path: Path) -> None:
    """A session opens for any id; the policy decides, and the refusal is logged."""
    with open_paveo(tmp_path) as pf, pf.session(agent_id="not-declared") as s:
        with pytest.raises(PolicyDenied) as refused:
            s.check_tool("lookup_order")

    assert refused.value.reason == "agent_unknown"
    assert records(tmp_path)[0]["agent_id"] == "not-declared"


def test_a_session_needs_an_agent_id(tmp_path: Path) -> None:
    with open_paveo(tmp_path) as pf, pytest.raises(ConfigError, match="agent_id"):
        pf.session(agent_id="")


# --------------------------------------------------------------------------
# An unlogged decision did not happen (§7)
# --------------------------------------------------------------------------


def test_an_unwritable_log_denies_a_call_the_policy_allowed(tmp_path: Path) -> None:
    pf = open_paveo(tmp_path)
    with pf.session(agent_id="refund-bot") as s:
        s.check_tool("lookup_order")
        pf.close()  # the log is now shut; the policy still says yes
        with pytest.raises(PolicyUnavailable):
            s.check_tool("lookup_order")


def test_fail_open_allows_an_unrecorded_call_but_says_so(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    pf = open_paveo(tmp_path, fail_open=True)
    with (
        caplog.at_level(logging.WARNING, logger="paveo"),
        pf.session(agent_id="refund-bot") as s,
    ):
        s.check_tool("lookup_order")
        pf.close()
        s.check_tool("lookup_order")  # allowed despite being unrecordable

    assert any("fail_open" in message for message in caplog.messages)
    assert any("unrecorded" in message for message in caplog.messages)


def test_fail_open_never_turns_a_refusal_into_a_permission(tmp_path: Path) -> None:
    """A refusal is a decision, not a failure to decide. fail_open covers the
    second, never the first — otherwise the policy is decorative."""
    with (
        open_paveo(tmp_path, fail_open=True) as pf,
        pf.session(agent_id="refund-bot") as s,
    ):
        with pytest.raises(PolicyDenied):
            s.check_tool("transfer_funds")
        with pytest.raises(PolicyDenied):
            s.check_tool("refund", {"amount_usd": 600})


def test_fail_open_warns_on_every_call(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """§7 says noisy by design, so nobody forgets the guard is half off."""
    with caplog.at_level(logging.WARNING, logger="paveo"):
        with (
            open_paveo(tmp_path, fail_open=True) as pf,
            pf.session(agent_id="refund-bot") as s,
        ):
            for _ in range(3):
                s.check_tool("lookup_order")

    assert sum("fail_open enabled" in m for m in caplog.messages) == 3
    assert all(r["fail_open"] is True for r in records(tmp_path))


# --------------------------------------------------------------------------
# The session contract
# --------------------------------------------------------------------------


def test_a_session_that_was_never_entered_refuses_to_work(tmp_path: Path) -> None:
    """Otherwise it works now and leaks a reservation once there are any."""
    with open_paveo(tmp_path) as pf:
        session = pf.session(agent_id="refund-bot")
        with pytest.raises(ConfigError, match="never entered"):
            session.check_tool("lookup_order")


def test_a_session_unwinds_even_when_the_body_raises(tmp_path: Path) -> None:
    with open_paveo(tmp_path) as pf:
        session = pf.session(agent_id="refund-bot")

        def use_then_fall_over() -> None:
            with session as s:
                s.check_tool("lookup_order")
                raise RuntimeError("the agent fell over")

        with pytest.raises(RuntimeError, match="fell over"):
            use_then_fall_over()

        # __exit__ ran despite the exception, so the session is closed again.
        with pytest.raises(ConfigError, match="never entered"):
            session.check_tool("lookup_order")


def test_sessions_share_one_log_and_one_chain(tmp_path: Path) -> None:
    with open_paveo(tmp_path) as pf:
        for agent in ("refund-bot", "refund-bot", "refund-bot"):
            with pf.session(agent_id=agent) as s:
                s.check_tool("lookup_order")

    status = verify_chain(tmp_path / "audit.jsonl")
    assert status.ok
    assert status.records == 3
