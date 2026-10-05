"""Plans and licence keys (D61, D62).

The claims:

1. With no key, the Developer plan covers two agents per policy: the first
   two in the file work, and every call from a third is refused as
   ``plan_limit``, in shadow mode too. The policy never fails to load for it.
2. A key we signed raises the limit; one we did not, or one changed by a byte,
   is refused loudly. An expired key reverts to Developer with a warning, and
   never stops the agents a plan still covers.
3. The free core never imports ``cryptography``; only reading a key does, and
   without ``paveo[team]`` that says how to install it.
4. The ``paveo`` guard reads ``.paveo/licence.key``, and ``init`` keeps it out
   of git.
"""

from __future__ import annotations

import io
import json
import logging
import subprocess
import sys
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import cast

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

from paveo import ConfigError, Paveo, PolicyDenied, _licence
from paveo._harnesses import CLAUDE_CODE
from paveo.cli import guard

TODAY = date(2026, 9, 27)
SIGNER = Ed25519PrivateKey.generate()


def at_noon() -> datetime:
    return datetime(TODAY.year, TODAY.month, TODAY.day, 12, tzinfo=UTC)


@pytest.fixture(autouse=True)
def our_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keys in these tests are signed by a key made here, never the real one."""
    public = SIGNER.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
    monkeypatch.setattr(_licence, "PUBLIC_KEY", public)


def key(
    plan: str = "team", *, days: int = 30, signer: Ed25519PrivateKey = SIGNER
) -> str:
    return _licence.encode(
        plan=plan,
        expires=TODAY + timedelta(days=days),
        reference="T-0001",
        sign=signer.sign,
    )


def document(agents: int) -> dict[str, object]:
    return {
        "version": 1,
        "policy_id": "plans",
        "agents": [
            {
                "id": f"agent-{n}",
                "budget": {"period": "day", "limit_usd": "1.00"},
                "models": {"allow": ["claude-sonnet-5"]},
                "tools": {"allow": [{"name": "lookup_order"}]},
            }
            for n in range(1, agents + 1)
        ],
    }


def outcome(pf: Paveo, agent: str) -> str:
    with pf.session(agent_id=agent) as s:
        try:
            s.check_tool("lookup_order", {})
        except PolicyDenied as e:
            return e.reason
    return "allow"


@pytest.fixture
def log(tmp_path: Path) -> Path:
    return tmp_path / "audit.jsonl"


def paveo(log: Path, agents: int, licence: str | None = None) -> Paveo:
    return Paveo.from_policy(
        document(agents), audit_path=log, now=at_noon, licence=licence
    )


# --------------------------------------------------------------------------
# 1. Developer: two agents per policy (D74)
# --------------------------------------------------------------------------


def test_two_agents_need_no_key(log: Path) -> None:
    with paveo(log, 2) as pf:
        assert [outcome(pf, f"agent-{n}") for n in (1, 2)] == ["allow"] * 2


def test_a_third_agent_is_refused_and_the_first_two_keep_working(
    log: Path,
) -> None:
    with paveo(log, 5) as pf:
        results = [outcome(pf, f"agent-{n}") for n in range(1, 6)]
    assert results == ["allow"] * 2 + ["plan_limit"] * 3


def test_a_model_call_past_the_plan_is_refused_too(log: Path) -> None:
    request = {
        "model": "claude-sonnet-5",
        "max_tokens": 10,
        "messages": [{"role": "user", "content": "hi"}],
    }
    with paveo(log, 4) as pf, pf.session(agent_id="agent-4") as s:
        with pytest.raises(PolicyDenied) as refused:
            s.check_llm(request, shape="anthropic")
    assert refused.value.reason == "plan_limit"


def test_shadow_mode_does_not_lift_the_plan(log: Path) -> None:
    policy = document(4)
    for agent in cast("list[dict[str, object]]", policy["agents"]):
        agent["mode"] = "shadow"
    with Paveo.from_policy(policy, audit_path=log, now=at_noon) as pf:
        assert outcome(pf, "agent-4") == "plan_limit"
        assert outcome(pf, "agent-2") == "allow"


def test_the_refusal_tells_the_model_nothing_about_the_plan(log: Path) -> None:
    with paveo(log, 4) as pf, pf.session(agent_id="agent-4") as s:
        with pytest.raises(PolicyDenied) as refused:
            s.check_tool("lookup_order", {})
    assert "(plan_limit)" in refused.value.for_model
    assert "licence key" not in refused.value.for_model
    assert "licence key" in str(refused.value)
    assert "paveo trial" not in str(refused.value)


# --------------------------------------------------------------------------
# 2. Keys
# --------------------------------------------------------------------------


def test_a_team_key_covers_ten_agents(log: Path) -> None:
    with paveo(log, 10, key("team")) as pf:
        assert outcome(pf, "agent-10") == "allow"
    with paveo(log, 11, key("team")) as pf:
        assert outcome(pf, "agent-11") == "plan_limit"


def test_an_enterprise_key_has_no_limit(log: Path) -> None:
    with paveo(log, 60, key("enterprise")) as pf:
        assert outcome(pf, "agent-60") == "allow"


@pytest.mark.parametrize(
    "broken",
    [
        lambda k: k.replace("paveo1.", "paveo2."),
        lambda k: k[:-4] + ("AAAA" if not k.endswith("AAAA") else "BBBB"),
        lambda k: k.split(".")[0] + "." + k.split(".")[2] + "." + k.split(".")[1],
        lambda _: "not a key",
        lambda k: k + "." + k,
        lambda _: "paveo1." + "A" * 2000 + ".AA",
    ],
)
def test_a_key_we_did_not_sign_or_that_was_changed_is_refused(
    log: Path, broken: object
) -> None:
    changed = broken(key())  # type: ignore[operator]  # a lambda from the list above
    with pytest.raises(ConfigError, match="licence key"):
        paveo(log, 1, changed)


def test_a_key_signed_by_someone_else_is_refused(log: Path) -> None:
    with pytest.raises(ConfigError, match="not one Paveo issued"):
        paveo(log, 1, key(signer=Ed25519PrivateKey.generate()))


def test_an_expired_key_reverts_to_developer_with_a_warning(
    log: Path, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING, logger="paveo"):
        pf = paveo(log, 4, key("team", days=-1))
    with pf:
        assert outcome(pf, "agent-2") == "allow"
        assert outcome(pf, "agent-4") == "plan_limit"
    assert "team licence ended" in caplog.text


def test_a_key_near_its_end_warns(log: Path, caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="paveo"):
        paveo(log, 1, key("team", days=10)).close()
    assert "team licence ends on" in caplog.text


def test_without_paveo_team_a_key_says_how_to_install_it(
    log: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "cryptography.exceptions", None)
    with pytest.raises(ConfigError, match=r"paveo\[team\]"):
        paveo(log, 1, key())


# --------------------------------------------------------------------------
# 3. The free core never loads cryptography
# --------------------------------------------------------------------------


def test_the_free_core_never_imports_cryptography(tmp_path: Path) -> None:
    audit = str(tmp_path / "a.jsonl")
    script = (
        "import sys, paveo\n"
        "from paveo import Paveo\n"
        f"doc = {document(2)!r}\n"
        f"with Paveo.from_policy(doc, audit_path={audit!r}) as pf:\n"
        "    with pf.session(agent_id='agent-1') as s:\n"
        "        s.check_tool('lookup_order', {})\n"
        "print(any(m.startswith('cryptography') for m in sys.modules))\n"
    )
    result = subprocess.run(  # noqa: S603 - our own interpreter and our own script
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=True,
        env={"PYTHONPATH": str(Path(__file__).parent.parent / "src")},
    )
    assert result.stdout.strip() == "False"


# --------------------------------------------------------------------------
# 4. The guard and init
# --------------------------------------------------------------------------


@pytest.fixture
def home(tmp_path: Path) -> Path:
    directory = tmp_path / ".paveo"
    directory.mkdir()
    policy = document(4)
    agents = cast("list[dict[str, object]]", policy["agents"])
    agents[3]["id"] = "claude-code"
    (directory / "policy.json").write_text(json.dumps(policy), encoding="utf-8")
    return directory


def through_guard(home: Path) -> tuple[int, str]:
    stderr = io.StringIO()
    payload = {"session_id": "s", "tool_name": "lookup_order", "tool_input": {}}
    code = guard(
        CLAUDE_CODE,
        directory=home,
        agent="claude-code",
        stdin=io.BytesIO(json.dumps(payload).encode()),
        stdout=io.StringIO(),
        stderr=stderr,
        stopped=lambda: False,
        now=at_noon,
    )
    return code, stderr.getvalue()


def test_the_guard_refuses_an_agent_past_the_free_plan(home: Path) -> None:
    code, said = through_guard(home)
    assert code == 2
    assert "(plan_limit)" in said


def test_the_guard_reads_a_key_beside_the_policy(home: Path) -> None:
    (home / "licence.key").write_text(key("team") + "\n", encoding="utf-8")
    assert through_guard(home) == (0, "")


def test_a_broken_key_beside_the_policy_refuses_every_call(home: Path) -> None:
    (home / "licence.key").write_text("not a key", encoding="utf-8")
    code, said = through_guard(home)
    assert code == 2
    assert "licence key" in said


# --------------------------------------------------------------------------
# 5. A trial written by 0.1.0 or 0.1.1 (D63): 30 days of Team, no signature.
#    No command starts one any more (D85); one already written runs to its end.
# --------------------------------------------------------------------------


def old_trial(started: date) -> str:
    """A trial as ``paveo trial`` wrote it in 0.1.0 and 0.1.1."""
    return f"paveo-trial.{started.isoformat()}"


def test_a_trial_grants_team_for_thirty_days(log: Path) -> None:
    started = old_trial(TODAY - timedelta(days=29))
    with paveo(log, 10, started) as pf:
        assert outcome(pf, "agent-10") == "allow"


def test_a_trial_after_thirty_days_reverts_with_a_warning(
    log: Path, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING, logger="paveo"):
        pf = paveo(log, 4, old_trial(TODAY - timedelta(days=31)))
    with pf:
        assert outcome(pf, "agent-4") == "plan_limit"
        assert outcome(pf, "agent-2") == "allow"
    assert "team trial licence ended" in caplog.text


@pytest.mark.parametrize("token", ["paveo-trial.2099-01-01", "paveo-trial.soon"])
def test_a_trial_paveo_trial_never_wrote_is_refused(log: Path, token: str) -> None:
    with pytest.raises(ConfigError):
        paveo(log, 1, token)


def test_a_trial_needs_no_paveo_team(
    log: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "cryptography.exceptions", None)
    with paveo(log, 10, old_trial(TODAY)) as pf:
        assert outcome(pf, "agent-10") == "allow"


def test_a_trial_written_before_upgrading_still_reaches_the_guard(
    home: Path,
) -> None:
    """Refusing the token would fail every call closed for someone who only
    upgraded (locked decision #4 turned against them); it runs to its end."""
    (home / "licence.key").write_text(old_trial(TODAY) + "\n", encoding="utf-8")
    assert through_guard(home) == (0, "")


def test_paveo_trial_is_no_longer_a_command(home: Path) -> None:
    from paveo.cli import main  # noqa: PLC0415

    with pytest.raises(SystemExit) as ended:
        main(["trial", "--dir", str(home)])
    assert ended.value.code == 2
    assert not (home / "licence.key").exists()


# --------------------------------------------------------------------------
# 6. What the review found (/code-review, D62)
# --------------------------------------------------------------------------


def test_the_trial_is_over_on_its_thirtieth_day(log: Path) -> None:
    with paveo(log, 4, old_trial(TODAY - timedelta(days=30))) as pf:
        assert outcome(pf, "agent-4") == "plan_limit"


def test_a_long_running_process_reverts_when_its_key_ends(log: Path) -> None:
    moment = [at_noon()]
    with Paveo.from_policy(
        document(4),
        audit_path=log,
        now=lambda: moment[0],
        licence=old_trial(TODAY - timedelta(days=29)),
    ) as pf:
        assert outcome(pf, "agent-4") == "allow"
        moment[0] += timedelta(days=1)  # the trial ends overnight
        assert outcome(pf, "agent-4") == "plan_limit"
        assert outcome(pf, "agent-2") == "allow"


def test_a_refused_key_leaves_the_log_free_for_the_retry(log: Path) -> None:
    with pytest.raises(ConfigError):
        paveo(log, 1, "not a key")
    with paveo(log, 1, key("team")) as pf:
        assert outcome(pf, "agent-1") == "allow"


def test_every_record_says_which_plan_was_in_force(log: Path) -> None:
    from conftest import records_in  # noqa: PLC0415

    with paveo(log, 4, key("team")) as pf:
        outcome(pf, "agent-4")
    with paveo(log, 4) as pf:
        outcome(pf, "agent-4")
    assert [(r["plan"], r["decision"]) for r in records_in(log)] == [
        ("team", "allow"),
        ("developer", "deny"),
    ]


def test_a_key_of_a_newer_format_is_refused(log: Path) -> None:
    payload = _licence.canonical_json(
        {"v": 2, "plan": "team", "expires": "2027-01-01", "ref": "T"}
    )
    newer = ".".join(
        ("paveo1", _licence._b64(payload), _licence._b64(SIGNER.sign(payload)))
    )
    with pytest.raises(ConfigError, match="version of Paveo"):
        paveo(log, 1, newer)


def test_replay_judges_under_the_plan_the_guard_applies(home: Path) -> None:
    from paveo._replay import from_history  # noqa: PLC0415

    # One response as Claude Code writes it (tests/test_replay.py `response`).
    line = {
        "type": "assistant",
        "timestamp": "2026-09-26T10:00:00Z",
        "sessionId": "s1",
        "cwd": str(home.parent),
        "message": {
            "id": "m1",
            "model": "claude-sonnet-5",
            "usage": {"input_tokens": 1, "output_tokens": 1},
            "content": [
                {"type": "tool_use", "id": "t1", "name": "lookup_order", "input": {}}
            ],
        },
    }
    session = home.parent / "session.jsonl"
    session.write_text(json.dumps(line) + "\n", encoding="utf-8")
    said = io.StringIO()
    from_history(
        "replay",
        paths=[session],
        within=None,
        directory=home,
        agent="claude-code",
        now=at_noon(),
        out=said,
    )
    assert "plan.developer" in said.getvalue()


def test_what_each_plan_switches_on() -> None:
    """Audit evidence is Team and up, a trial included; a lapsed key drops it
    with the rest of the plan (D66)."""
    assert _licence.read_key(key("team"), today=TODAY).features == {"evidence"}
    assert _licence.read_key(key("business"), today=TODAY).features == {"evidence"}
    assert _licence.read_key(old_trial(TODAY), today=TODAY).features == {"evidence"}
    assert _licence.DEVELOPER.features == frozenset()
    lapsed = _licence.read_key(key("team", days=-1), today=TODAY)
    assert lapsed.features == frozenset()
