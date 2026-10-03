"""Replay and learn, from the Claude Code sessions already on disk (B3, D53).

B3 is done when a month of real sessions replays to a one-screen summary with no
payload in it, and ``learn`` writes a policy the guard accepts. The real month was
replayed by hand (D53); these tests pin each thing that decided its numbers, on
records shaped exactly like Claude Code's, with invented values.
"""

from __future__ import annotations

import io
import json
import os
from datetime import UTC, datetime
from pathlib import Path

import pytest

from conftest import SENTINEL
from paveo import _replay
from paveo._policy_document import load_document
from paveo._replay import History, from_history, learn, read_history, replay
from paveo.cli import main

NOW = datetime(2026, 9, 26, 12, tzinfo=UTC)
PROJECT = Path("/work/app")
# 1M input and 100K output on claude-sonnet-5, at $2/$10, is $3.00; at the US
# rate, which is how a call that does not say where it ran is priced, $3.30.
MILLION = {"input_tokens": 1_000_000, "output_tokens": 100_000}

POLICY: dict[str, object] = {
    "version": 1,
    "policy_id": "replay",
    "agents": [
        {
            "id": "claude-code",
            "tools": {
                "allow": [
                    {
                        "name": "Bash",
                        "constraints": {
                            "command": {"not_matches": [r"\brm\s+-rf", r"\.claude\b"]},
                            "description": {},
                        },
                    },
                    {"name": "Read"},
                ],
                "deny": ["Nuke"],
            },
        }
    ],
}


def response(  # noqa: PLR0913 - keyword-only, each one a field of the record
    message_id: str,
    *,
    at: str = "2026-09-26T10:00:00Z",
    model: str = "claude-sonnet-5",
    usage: dict[str, object] | None = None,
    tools: tuple[tuple[str, str, dict[str, object]], ...] = (),
    session: str = "s1",
    cwd: str = str(PROJECT),
) -> dict[str, object]:
    """One line of a response as Claude Code writes it (2026-09-26)."""
    return {
        "type": "assistant",
        "timestamp": at,
        "sessionId": session,
        "cwd": cwd,
        "message": {
            "id": message_id,
            "model": model,
            "usage": MILLION if usage is None else usage,
            "content": [
                {"type": "tool_use", "id": tool_id, "name": name, "input": arguments}
                for tool_id, name, arguments in tools
            ],
        },
    }


def result(tool_id: str, *, error: bool = False) -> dict[str, object]:
    """The turn after a tool call. Its content is output, so it is the sentinel."""
    return {
        "type": "user",
        "cwd": str(PROJECT),
        "message": {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": tool_id,
                    "is_error": error,
                    "content": SENTINEL,
                }
            ],
        },
    }


def write(path: Path, *records: object, raw: tuple[str, ...] = ()) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(record) for record in records] + list(raw)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def history_of(tmp_path: Path, *records: object) -> History:
    return read_history([write(tmp_path / "s.jsonl", *records)], within=None)


def summary(history: History, policy: dict[str, object] | None = None) -> list[str]:
    loaded = load_document(POLICY if policy is None else policy)
    return replay(history, policy=loaded, agent="claude-code", now=NOW)


def with_budget(period: str, limit: str) -> dict[str, object]:
    document = json.loads(json.dumps(POLICY))
    document["agents"][0]["budget"] = {"period": period, "limit_usd": limit}
    return document  # type: ignore[no-any-return]  # json.loads returns Any


# -- reading ------------------------------------------------------------------


def test_a_response_written_across_lines_is_counted_once_at_its_final_output(
    tmp_path: Path,
) -> None:
    """Claude Code writes one line per content block, and the output count grows
    until the last one. Counting each line would triple the spend."""
    history = history_of(
        tmp_path,
        response("m", at="2026-09-26T10:00:05Z", usage={**MILLION, "output_tokens": 1}),
        response("m", at="2026-09-26T10:00:00Z", usage=MILLION),
        response("m", at="2026-09-26T10:00:09Z", usage={**MILLION, "output_tokens": 7}),
    )

    assert len(history.model_calls) == 1
    call = history.model_calls["m"]
    assert call.usage["output_tokens"] == 100_000
    assert call.at == datetime(2026, 9, 26, 10, tzinfo=UTC)
    assert "  API-equivalent spend at list prices: $3.30" in "\n".join(summary(history))


def test_a_resumed_session_copies_its_lines_and_each_is_counted_once(
    tmp_path: Path,
) -> None:
    line = response("m", tools=(("t1", "Bash", {"command": "ls"}),))
    write(tmp_path / "first.jsonl", line)
    write(tmp_path / "resumed.jsonl", line, response("m2"))

    history = read_history([tmp_path], within=None)

    assert history.files == 2
    assert sorted(history.model_calls) == ["m", "m2"]
    assert len(history.tool_calls) == 1


def test_a_subagents_files_are_read_with_the_session(tmp_path: Path) -> None:
    write(tmp_path / "s1.jsonl", response("m"))
    write(tmp_path / "s1" / "subagents" / "agent-a1.jsonl", response("sub"))

    history = read_history([tmp_path], within=None)

    assert sorted(history.model_calls) == ["m", "sub"]


def test_a_line_it_does_not_understand_is_counted_not_guessed(tmp_path: Path) -> None:
    broken = response("m3")
    del broken["timestamp"]
    history = read_history(
        [
            write(
                tmp_path / "s.jsonl",
                response("m"),
                {"type": "mode", "mode": "auto"},  # understood, and not a call
                broken,
                {"type": "assistant", "message": "not a message"},
                raw=("not json", "[1]", "[" * 100_000),
            )
        ],
        within=None,
    )

    assert history.not_understood == 5
    assert list(history.model_calls) == ["m"]
    assert "Lines not understood, skipped: 5" in summary(history)


def test_a_malformed_tool_call_does_not_lose_the_responses_cost(
    tmp_path: Path,
) -> None:
    """Its tool calls cannot be read, but its tokens were spent (/code-review)."""
    broken = response("m", tools=(("t1", "Bash", {}),))
    broken["message"]["content"][0]["input"] = "not a mapping"  # type: ignore[index]

    history = history_of(tmp_path, broken)

    assert list(history.model_calls) == ["m"]
    assert not history.tool_calls
    assert history.not_understood == 1


def test_a_file_that_cannot_be_read_is_counted_and_said(tmp_path: Path) -> None:
    """One root-owned file in another project must not stop every replay, and
    must not vanish from the numbers either (Rule 3)."""
    write(tmp_path / "good.jsonl", response("m"))
    (tmp_path / "folder.jsonl").mkdir()

    history = read_history([tmp_path], within=None)

    assert (history.files, history.unreadable) == (1, 1)
    assert "Files that could not be read, left out: 1" in summary(history)


def test_an_api_error_written_in_place_of_a_response_is_not_a_call(
    tmp_path: Path,
) -> None:
    error = {**response("m", model="<synthetic>"), "isApiErrorMessage": True}

    history = history_of(tmp_path, error)

    assert not history.model_calls
    assert history.not_understood == 0


def test_by_default_only_what_was_done_inside_the_project_is_read(
    tmp_path: Path,
) -> None:
    session = write(
        tmp_path / "s.jsonl",
        response("inside", cwd=str(PROJECT)),
        response("below", cwd=str(PROJECT / "src")),
        response("beside", cwd="/work/application"),
        response("elsewhere", cwd="/work/other"),
    )

    history = read_history([session], within=PROJECT)

    assert sorted(history.model_calls) == ["below", "inside"]


def test_a_path_that_is_not_there_is_refused_not_read_as_empty(
    tmp_path: Path,
) -> None:
    """A replay that skipped what it was given would report less as though it
    were all of it (Rule 3)."""
    out = io.StringIO()

    code = from_history(
        "replay",
        paths=[tmp_path / "missing"],
        within=None,
        directory=tmp_path,
        agent="claude-code",
        now=NOW,
        out=out,
    )

    assert code == 1
    assert "is not a file or folder" in out.getvalue()


def test_finding_no_session_files_is_said_and_is_a_failure(tmp_path: Path) -> None:
    out = io.StringIO()

    code = from_history(
        "replay",
        paths=[tmp_path],
        within=None,
        directory=tmp_path,
        agent="claude-code",
        now=NOW,
        out=out,
    )

    assert code == 1
    assert "found no Claude Code session files" in out.getvalue()


# -- the tool calls -------------------------------------------------------------


def test_each_refusal_is_named_with_the_pattern_that_fired(tmp_path: Path) -> None:
    history = history_of(
        tmp_path,
        response(
            "m",
            tools=(
                ("t1", "Bash", {"command": "ls", "description": "list"}),
                ("t2", "Bash", {"command": "rm -rf build"}),
                ("t3", "Bash", {"command": "rm -rf dist"}),
                ("t4", "Bash", {"command": "cat ~/.claude/notes.md"}),
                ("t5", "Nuke", {}),
                ("t6", "Read", {"file_path": "a.py"}),
            ),
        ),
    )

    lines = summary(history)

    assert "  judged by the policy: 6, allowed 2, refused 4" in lines
    assert "          2  Bash.command.not_matches  \\brm\\s+-rf" in lines
    assert "          1  Bash.command.not_matches  \\.claude\\b" in lines
    assert "          1  Nuke" in lines


def test_a_tool_the_policy_does_not_name_is_counted_and_not_named(
    tmp_path: Path,
) -> None:
    """Its name was chosen by the model (D26)."""
    history = history_of(
        tmp_path, response("m", tools=(("t1", SENTINEL, {}), ("t2", "Glob", {})))
    )

    lines = summary(history)

    assert any(
        line.startswith("  not judged: 2 calls to 2 tools the policy does not name")
        for line in lines
    )
    assert "Glob" not in "\n".join(lines)


def test_nothing_from_a_session_is_ever_printed(tmp_path: Path) -> None:
    """Every place a session file puts content: arguments and their names, tool
    and model names nobody declared, results, session ids, token classes."""
    session = write(
        tmp_path / "s.jsonl",
        response(
            "m",
            session=SENTINEL,
            tools=(
                ("t1", "Bash", {"command": f"rm -rf {SENTINEL}"}),
                ("t2", "Bash", {"command": "ls", SENTINEL: 1}),
                ("t3", "Read", {"file_path": SENTINEL}),
                ("t4", SENTINEL, {}),
            ),
        ),
        response("m2", model=SENTINEL),
        response("m3", usage={**MILLION, SENTINEL: 5}),
        result("t1"),
        raw=(SENTINEL,),
    )
    (tmp_path / "policy.json").write_text(json.dumps(POLICY), encoding="utf-8")
    out = io.StringIO()

    for command in ("replay", "learn"):
        code = from_history(
            command,
            paths=[session],
            within=None,
            directory=tmp_path,
            agent="claude-code",
            now=NOW,
            out=out,
        )
        assert code == 0

    assert SENTINEL not in out.getvalue()
    assert SENTINEL not in (tmp_path / "policy.learned.json").read_text()


# -- the model calls ------------------------------------------------------------


def test_a_class_the_table_cannot_price_never_stops_the_rest(
    tmp_path: Path,
) -> None:
    """With a malformed count beside it, `actual` raised after marking the table
    stale, and every later call went unpriced (/code-review)."""
    history = history_of(
        tmp_path,
        response(
            "a",
            at="2026-09-26T10:00:00Z",
            usage={**MILLION, "audio_tokens": 5, "cache_read_input_tokens": -1},
        ),
        response("b", at="2026-09-26T10:01:00Z"),
        response("c", at="2026-09-26T10:02:00Z"),
    )

    text = "\n".join(summary(history))

    assert "API-equivalent spend at list prices: $6.60" in text
    assert "1 calls to models or token classes" in text


def test_a_call_that_does_not_say_where_it_ran_is_priced_high_and_says_so(
    tmp_path: Path,
) -> None:
    """Claude Code records `inference_geo: not_available`. The library reserves
    such a call at the US rate, and replay prices it the same way, out loud."""
    unplaced = summary(
        history_of(tmp_path, response("m", usage={**MILLION, "inference_geo": "n/a"}))
    )
    placed = summary(
        history_of(
            tmp_path, response("m", usage={**MILLION, "inference_geo": "global"})
        )
    )

    assert any("$3.30" in line for line in unplaced)
    assert any("at the US rate: up to 10% high" in line for line in unplaced)
    assert any("$3.00" in line for line in placed)
    assert not any("US rate" in line for line in placed)


def test_what_cannot_be_priced_is_counted_and_the_rest_still_is(
    tmp_path: Path,
) -> None:
    """An unknown model, and a token class the table does not know. The table
    stops at the second, so the call after it must get a clean one."""
    history = history_of(
        tmp_path,
        response("a", at="2026-09-26T10:00:00Z", model="claude-unreleased-9"),
        response("b", at="2026-09-26T10:01:00Z", usage={**MILLION, "audio_tokens": 5}),
        response("c", at="2026-09-26T10:02:00Z"),
        response(
            "d",
            at="2026-09-26T10:03:00Z",
            usage={**MILLION, "server_tool_use": {"web_search_requests": 3}},
        ),
    )

    text = "\n".join(summary(history))

    assert "Model calls: 4" in text
    assert "API-equivalent spend at list prices: $6.60" in text
    assert "2 calls to models or token classes the table does not carry" in text
    assert "3 web searches and fetches" in text
    assert "claude-unreleased-9" not in text


def test_spend_is_split_by_model_day_and_session(tmp_path: Path) -> None:
    history = history_of(
        tmp_path,
        response("a", at="2026-09-24T10:00:00Z", session="one"),
        response("b", at="2026-09-25T10:00:00Z", session="one"),
        response("c", at="2026-09-25T11:00:00Z", session="two"),
        response("d", at="2026-09-25T12:00:00Z", model="claude-opus-5", session="two"),
    )

    lines = summary(history)

    assert any(
        line.split() == ["claude-sonnet-5", "$9.90", "3", "calls"] for line in lines
    )
    assert "  costliest day 25 Sep 2026, $" in "\n".join(lines)
    assert "API-equivalent spend at list prices: $" in "\n".join(lines)
    assert "24 Sep to 25 Sep 2026" in "\n".join(lines)


def test_without_a_budget_it_says_how_to_see_a_ceiling(tmp_path: Path) -> None:
    lines = summary(history_of(tmp_path, response("m")))

    assert (
        "has no budget: add one to see what a ceiling would have refused" in lines[-2]
    )


@pytest.mark.parametrize(
    ("period", "refused", "stopped"),
    [
        # $3.30 a call, a $5 ceiling. By day, the second call on the 25th
        # crosses it; the call on the 26th is a new day.
        ("day", 1, "$3.30"),
        # By hour, every call is in an hour of its own.
        ("hour", 0, "$0.00"),
        # By session, the three calls share one: the second and third cross it.
        ("session", 2, "$6.60"),
    ],
)
def test_a_ceiling_refuses_the_calls_that_would_have_crossed_it(
    tmp_path: Path, period: str, refused: int, stopped: str
) -> None:
    history = history_of(
        tmp_path,
        response("a", at="2026-09-25T10:00:00Z"),
        response("b", at="2026-09-25T11:00:00Z"),
        response("c", at="2026-09-26T10:00:00Z"),
    )

    lines = summary(history, with_budget(period, "5"))

    assert lines[-2] == (
        f"  a $5.00 per {period} ceiling, counting each call at what it cost, "
        f"would have refused {refused} calls, {stopped} of this spend"
    )


def test_a_ceiling_admits_a_call_that_reaches_it_exactly(tmp_path: Path) -> None:
    """As the ledger does: only spend *over* the limit is refused."""
    history = history_of(
        tmp_path,
        response("a", at="2026-09-25T10:00:00Z"),
        response("b", at="2026-09-25T11:00:00Z"),
    )

    lines = summary(history, with_budget("day", "6.60"))

    assert "would have refused 0 calls" in lines[-2]


def test_an_agent_the_policy_does_not_declare_is_refused(tmp_path: Path) -> None:
    (tmp_path / "policy.json").write_text(json.dumps(POLICY), encoding="utf-8")
    out = io.StringIO()

    code = from_history(
        "replay",
        paths=[write(tmp_path / "s.jsonl", response("m"))],
        within=None,
        directory=tmp_path,
        agent="someone-else",
        now=NOW,
        out=out,
    )

    assert code == 1
    assert "declares no agent 'someone-else'" in out.getvalue()


def test_an_unexpected_failure_is_reported_by_its_type_alone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(*_args: object, **_kwargs: object) -> list[str]:
        raise RuntimeError(SENTINEL)

    monkeypatch.setattr(_replay, "replay", fail)
    (tmp_path / "policy.json").write_text(json.dumps(POLICY), encoding="utf-8")
    out = io.StringIO()

    code = from_history(
        "replay",
        paths=[write(tmp_path / "s.jsonl", response("m"))],
        within=None,
        directory=tmp_path,
        agent="claude-code",
        now=NOW,
        out=out,
    )

    assert code == 1
    assert "replay failed (RuntimeError)" in out.getvalue()
    assert SENTINEL not in out.getvalue()


def test_the_project_is_the_one_dir_belongs_to_not_where_it_ran_from(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    project = tmp_path / "app"
    (project / ".paveo").mkdir(parents=True)
    (project / ".paveo" / "policy.json").write_text(
        json.dumps(POLICY), encoding="utf-8"
    )
    home = tmp_path / "home"
    write(
        home / ".claude" / "projects" / "-app" / "s.jsonl",
        response("here", cwd=str(project)),
        response("there", cwd=str(tmp_path / "other")),
    )
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(tmp_path)

    assert main(["replay", "claude-code", "--dir", str(project / ".paveo")]) == 0

    printed = capsys.readouterr().out
    assert f"keeping what was done in {project}" in printed
    assert "Model calls: 1" in printed


def test_the_command_reads_this_projects_sessions_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    project = tmp_path / "app"
    (project / ".paveo").mkdir(parents=True)
    (project / ".paveo" / "policy.json").write_text(
        json.dumps(POLICY), encoding="utf-8"
    )
    home = tmp_path / "home"
    write(
        home / ".claude" / "projects" / "-app" / "s.jsonl",
        response("here", cwd=str(project)),
        response("there", cwd=str(tmp_path / "other")),
    )
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(project)

    assert main(["replay", "claude-code"]) == 0

    printed = capsys.readouterr().out
    assert f"keeping what was done in {project}" in printed
    assert "Model calls: 1" in printed


# -- learn ------------------------------------------------------------------------


def test_learn_adds_the_tools_that_ran_and_the_arguments_they_ran_with(
    tmp_path: Path,
) -> None:
    history = history_of(
        tmp_path,
        response(
            "m",
            tools=(
                ("t1", "Glob", {"pattern": "*.py", "path": "src"}),
                ("t2", "Glob", {"pattern": "*.md", "head_limit": 5}),
                ("t3", "WebFetch", {"url": "u", "__unparsedToolInput": "x"}),
                ("t4", "Bash", {"command": "ls", "timeout": 5, "bad name": 1}),
                ("t9", "Grep", {"pattern": "x", "-n": True}),
                # A tool that errored, one with no result, and a name no harness
                # gives a tool: none of them is learned.
                ("t5", "Fabricated", {"secret": 1}),
                ("t6", "Unanswered", {}),
                ("t7", "name with spaces", {}),
                ("t8", "Nuke", {"now": True}),
            ),
        ),
        *(result(t) for t in ("t1", "t2", "t3", "t4", "t6x", "t7", "t8", "t9")),
        result("t5", error=True),
    )

    learned, changes = learn(history, base=POLICY, agent="claude-code")

    assert changes == [
        "permit Bash.timeout",
        "allow Glob(head_limit, path, pattern)",
        "allow Grep(-n, pattern)",
        "allow WebFetch(url)",
    ]
    tools = learned["agents"][0]["tools"]  # type: ignore[index]
    names = [entry["name"] for entry in tools["allow"]]
    assert names == ["Bash", "Read", "Glob", "Grep", "WebFetch"]
    assert tools["deny"] == ["Nuke"]


def test_learn_keeps_every_constraint_and_the_policy_still_refuses(
    tmp_path: Path,
) -> None:
    history = history_of(
        tmp_path,
        response("m", tools=(("t1", "Bash", {"command": "ls", "timeout": 5}),)),
        result("t1"),
    )

    learned, _ = learn(history, base=POLICY, agent="claude-code")
    policy = load_document(learned)

    assert POLICY["agents"][0]["tools"]["allow"][0]["constraints"] == {  # type: ignore[index]
        "command": {"not_matches": [r"\brm\s+-rf", r"\.claude\b"]},
        "description": {},
    }
    assert (
        policy.evaluate_tool("claude-code", "Bash", {"command": "ls", "timeout": 5})
        is None
    )
    refused = policy.evaluate_tool("claude-code", "Bash", {"command": "rm -rf /"})
    assert refused is not None
    assert refused.rule == "Bash.command.not_matches"


def test_learn_writes_beside_the_policy_and_never_over_it(tmp_path: Path) -> None:
    directory = tmp_path / ".paveo"
    directory.mkdir()
    policy = directory / "policy.json"
    policy.write_text(json.dumps(POLICY), encoding="utf-8")
    before = policy.read_bytes()
    session = write(
        tmp_path / "s.jsonl", response("m", tools=(("t1", "Glob", {}),)), result("t1")
    )
    out = io.StringIO()

    code = from_history(
        "learn",
        paths=[session],
        within=None,
        directory=directory,
        agent="claude-code",
        now=NOW,
        out=out,
    )

    assert code == 0
    assert policy.read_bytes() == before
    learned = json.loads((directory / "policy.learned.json").read_text())
    assert load_document(learned).declares_tool("claude-code", "Glob")
    assert "  allow Glob()" in out.getvalue()
    assert "sees only Bash, Write, Edit and NotebookEdit" in out.getvalue()


def test_learn_starts_from_the_starter_policy_when_there_is_none(
    tmp_path: Path,
) -> None:
    session = write(tmp_path / "s.jsonl", response("m"))

    code = from_history(
        "learn",
        paths=[session],
        within=None,
        directory=tmp_path,
        agent="claude-code",
        now=NOW,
        out=io.StringIO(),
    )

    assert code == 0
    learned = json.loads((tmp_path / "policy.learned.json").read_text())
    assert learned["policy_id"] == "starter-claude-code-production-repo"


def test_learn_does_not_write_through_a_link(tmp_path: Path) -> None:
    """A cloned repository can ship the file name as a link to somewhere else
    (/security-review, D50)."""
    victim = tmp_path / "victim.json"
    victim.write_text("keep", encoding="utf-8")
    directory = tmp_path / ".paveo"
    directory.mkdir()
    os.symlink(victim, directory / "policy.learned.json")
    out = io.StringIO()

    code = from_history(
        "learn",
        paths=[write(tmp_path / "s.jsonl", response("m"))],
        within=None,
        directory=directory,
        agent="claude-code",
        now=NOW,
        out=out,
    )

    assert code == 1
    assert victim.read_text() == "keep"


def test_learn_needs_a_folder_paveo_set_up(tmp_path: Path) -> None:
    out = io.StringIO()

    code = from_history(
        "learn",
        paths=[write(tmp_path / "s.jsonl", response("m"))],
        within=None,
        directory=tmp_path / "absent",
        agent="claude-code",
        now=NOW,
        out=out,
    )

    assert code == 1
    assert "paveo init claude-code" in out.getvalue()
