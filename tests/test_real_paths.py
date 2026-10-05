"""The guard judges the file a path really reaches, not only the name written (D79).

A starter rule refuses an edit to ``.claude/settings.json`` by its name. A link
in the project named ``notes.txt`` that points there has another name, so the
rule never saw it. The guard now resolves every file path a coding agent's file
tool names (links, ``..``, the working folder) and judges both the written path
and the real one: refused if either is. It can only refuse more (Rule 6).

What it still cannot stop is a link made *after* it said yes and before the
agent opened the file. ``check_then_open`` below is that race as a fixture an
executor can run with its own opener.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
from collections.abc import Callable
from pathlib import Path

import pytest

from conftest import make_clock, records_in
from paveo._harnesses import CLAUDE_CODE, CODEX, CURSOR, Harness
from paveo._setup import _policy_weakness
from paveo.cli import guard
from test_other_agents import codex, patch, pre_tool, seatbelt

STARTERS = Path(__file__).parent.parent / "src" / "paveo" / "starters"


def write_call(tool: str, tool_input: dict[str, object], **extra: object) -> bytes:
    """A Claude Code PreToolUse call, with ``cwd`` only when given."""
    return json.dumps(
        {
            "session_id": "abc-123",
            "hook_event_name": "PreToolUse",
            "tool_name": tool,
            "tool_input": tool_input,
            **extra,
        }
    ).encode()


def judge(home: Path, harness: Harness, stdin: bytes) -> tuple[int, str]:
    stderr = io.StringIO()
    code = guard(
        harness,
        directory=home,
        agent=harness.name,
        stdin=io.BytesIO(stdin),
        stdout=io.StringIO(),
        stderr=stderr,
        stopped=lambda: False,
        now=make_clock(),
    )
    return code, stderr.getvalue()


def project_with_guard(tmp_path: Path, harness: Harness = CLAUDE_CODE) -> Path:
    """A project folder holding the agent's starter policy; returns its .paveo."""
    project = tmp_path / "project"
    project.mkdir()
    return seatbelt(project, harness)


def global_settings(tmp_path: Path) -> Path:
    """The user's own Claude Code settings, outside the project."""
    settings = tmp_path / "home" / ".claude" / "settings.json"
    settings.parent.mkdir(parents=True)
    settings.write_text("{}", "utf-8")
    return settings


def log(home: Path) -> list[dict[str, object]]:
    return records_in(home / "audit.jsonl")


# --------------------------------------------------------------------------
# The hole: a name the rules allow, a file they do not
# --------------------------------------------------------------------------


def test_a_link_to_the_users_settings_is_refused_with_one_record(
    tmp_path: Path,
) -> None:
    home = project_with_guard(tmp_path)
    notes = home.parent / "notes.txt"
    notes.symlink_to(global_settings(tmp_path))

    code, stderr = judge(
        home,
        CLAUDE_CODE,
        write_call("Write", {"file_path": str(notes), "content": "x"}),
    )

    assert code == 2
    assert "Write.file_path" in stderr
    # Neither the written path nor the real one is handed back (§8).
    assert str(notes) not in stderr
    assert ".claude" not in stderr
    [record] = log(home)
    assert record["decision"] == "deny"
    assert record["reason"] == "constraint_violated"
    assert "settings" not in json.dumps(record)


def test_a_linked_folder_is_followed_to_the_guards_own_files(tmp_path: Path) -> None:
    home = project_with_guard(tmp_path)
    (home.parent / "cfg").symlink_to(home)

    code, _ = judge(
        home,
        CLAUDE_CODE,
        write_call(
            "Edit",
            {
                "file_path": str(home.parent / "cfg" / "policy.json"),
                "old_string": "a",
                "new_string": "b",
            },
        ),
    )

    assert code == 2


def test_a_notebook_through_a_link_is_refused(tmp_path: Path) -> None:
    home = project_with_guard(tmp_path)
    book = home.parent / "book.ipynb"
    book.symlink_to(home / "policy.json")

    code, _ = judge(
        home,
        CLAUDE_CODE,
        write_call("NotebookEdit", {"notebook_path": str(book), "new_source": "x"}),
    )

    assert code == 2


def test_a_relative_path_is_resolved_from_the_agents_working_folder(
    tmp_path: Path,
) -> None:
    home = project_with_guard(tmp_path)
    (home.parent / "cfg").symlink_to(home)

    code, _ = judge(
        home,
        CLAUDE_CODE,
        write_call(
            "Write",
            {"file_path": "cfg/policy.json", "content": "{}"},
            cwd=str(home.parent),
        ),
    )

    assert code == 2


def test_a_relative_path_with_no_working_folder_is_refused(tmp_path: Path) -> None:
    """It cannot be resolved, so it cannot be judged: refused (locked #4)."""
    home = project_with_guard(tmp_path)

    for cwd in ({}, {"cwd": None}, {"cwd": "relative/folder"}, {"cwd": 7}):
        code, stderr = judge(
            home,
            CLAUDE_CODE,
            write_call("Write", {"file_path": "notes.txt", "content": "x"}, **cwd),
        )
        assert code == 2, cwd
        assert "notes.txt" not in stderr


def test_a_path_that_cannot_be_resolved_is_refused(tmp_path: Path) -> None:
    home = project_with_guard(tmp_path)

    code, _ = judge(
        home,
        CLAUDE_CODE,
        write_call("Write", {"file_path": str(home.parent / "a\x00b"), "content": "x"}),
    )

    assert code == 2


# --------------------------------------------------------------------------
# Ordinary work: unchanged, counted once
# --------------------------------------------------------------------------


def test_ordinary_files_behind_a_linked_folder_are_allowed_once(tmp_path: Path) -> None:
    """macOS's /tmp is a link to /private/tmp; a project reached through any
    such link must not be refused for it, and must leave one record."""
    home = project_with_guard(tmp_path)
    (tmp_path / "alias").symlink_to(home.parent)
    alias = tmp_path / "alias"

    for stdin in (
        write_call(
            "Write", {"file_path": str(alias / "src" / "app.py"), "content": "x"}
        ),
        write_call(
            "Write",
            {"file_path": "src/app.py", "content": "x"},
            cwd=str(alias),
        ),
    ):
        assert judge(home, CLAUDE_CODE, stdin) == (0, "")

    assert [r["decision"] for r in log(home)] == ["allow", "allow"]


def test_a_call_judged_twice_is_counted_once(tmp_path: Path) -> None:
    """The real path is judged without the session's memory, so a rate rule
    counts the call once: two of two are allowed, and only the third refused."""
    home = project_with_guard(tmp_path)
    policy = {
        "version": 1,
        "policy_id": "rate",
        "agents": [
            {
                "id": "claude-code",
                "tools": {
                    "allow": [
                        {
                            "name": "Write",
                            "constraints": {
                                "file_path": {"not_matches": ["\\.paveo"]},
                                "content": {},
                            },
                            "rate": {"calls": 2, "seconds": 3600},
                        }
                    ]
                },
            }
        ],
    }
    (home / "policy.json").write_text(json.dumps(policy), "utf-8")
    (tmp_path / "alias").symlink_to(home.parent)
    stdin = write_call(
        "Write", {"file_path": str(tmp_path / "alias" / "app.py"), "content": "x"}
    )

    codes = [judge(home, CLAUDE_CODE, stdin)[0] for _ in range(3)]

    assert codes == [0, 0, 2]
    assert [r["reason"] for r in log(home)] == [None, None, "rate_limited"]


def test_a_path_that_is_not_text_is_judged_as_written(tmp_path: Path) -> None:
    """Nothing to resolve; the rule refuses it on its own (not_matches needs text)."""
    home = project_with_guard(tmp_path)

    code, _ = judge(
        home, CLAUDE_CODE, write_call("Write", {"file_path": 7, "content": "x"})
    )

    assert code == 2
    assert len(log(home)) == 1


# --------------------------------------------------------------------------
# Cursor and Codex
# --------------------------------------------------------------------------


def test_cursor_write_and_delete_through_a_link_are_refused(tmp_path: Path) -> None:
    home = project_with_guard(tmp_path, CURSOR)
    link = home.parent / "notes.txt"
    link.symlink_to(home / "policy.json")

    for tool, tool_input in (
        ("Write", {"file_path": str(link), "content": "x"}),
        ("Delete", {"file_path": str(link)}),
    ):
        call = pre_tool(tool, tool_input) | {"cwd": str(home.parent)}
        assert judge(home, CURSOR, json.dumps(call).encode())[0] == 2, tool


def test_codex_refuses_a_patch_with_one_linked_path_among_several(
    tmp_path: Path,
) -> None:
    home = project_with_guard(tmp_path, CODEX)
    (home.parent / "cfg").symlink_to(home)
    text = patch(
        "*** Update File: src/app.py",
        "*** Add File: cfg/policy.json",
        "*** Update File: README.md",
    )
    call = codex("apply_patch", {"command": text}) | {"cwd": str(home.parent)}

    assert judge(home, CODEX, json.dumps(call).encode())[0] == 2

    ordinary = patch("*** Update File: src/app.py", "*** Add File: README.md")
    call = codex("apply_patch", {"command": ordinary}) | {"cwd": str(home.parent)}
    assert judge(home, CODEX, json.dumps(call).encode()) == (0, "")


def test_a_relative_rule_refuses_a_relative_path_judged_in_full(
    tmp_path: Path,
) -> None:
    """The price of judging in full (D79): a rule listing the folders an agent may
    edit by relative name refuses the full path of a file inside them, until it
    accepts that form too. Refusing more is the side the guard errs on."""
    home = project_with_guard(tmp_path, CODEX)
    policy = json.loads((home / "policy.json").read_text("utf-8"))
    for rule in policy["agents"][0]["tools"]["allow"]:
        if rule["name"] == "apply_patch":
            rule["constraints"]["paths"]["matches"] = (
                "(?:src|tests)/.*|.*/(?:src|tests)/.*"
            )
    (home / "policy.json").write_text(json.dumps(policy), "utf-8")

    def run(header: str) -> int:
        call = codex("apply_patch", {"command": patch(header)})
        return judge(
            home, CODEX, json.dumps(call | {"cwd": str(home.parent)}).encode()
        )[0]

    assert run("*** Update File: src/app.py") == 0
    policy["agents"][0]["tools"]["allow"][1]["constraints"]["paths"]["matches"] = (
        "(?:src|tests)/.*"
    )
    (home / "policy.json").write_text(json.dumps(policy), "utf-8")
    assert run("*** Update File: src/app.py") == 2


def test_a_project_inside_a_guarded_folder_name_is_refused_for_it(
    tmp_path: Path,
) -> None:
    """The cost of judging in full (D79): a project under ~/.codex has the
    folder's name in every full path, so the rule guarding ``.codex`` refuses
    its edits. Pinned here so lifting it is a decision, not an accident."""
    project = tmp_path / ".codex" / "worktrees" / "project"
    project.mkdir(parents=True)
    home = seatbelt(project, CODEX)
    call = codex("apply_patch", {"command": patch("*** Update File: src/app.py")})

    code, _ = judge(home, CODEX, json.dumps(call | {"cwd": str(project)}).encode())

    assert code == 2


def test_codex_started_inside_its_own_folder_cannot_edit_its_config(
    tmp_path: Path,
) -> None:
    """Relative to ~/.codex, ``config.toml`` names no guarded folder; judged in
    full it does."""
    codex_home = tmp_path / ".codex"
    codex_home.mkdir()
    home = seatbelt(codex_home, CODEX)
    call = codex("apply_patch", {"command": patch("*** Update File: config.toml")})

    code, _ = judge(home, CODEX, json.dumps(call | {"cwd": str(codex_home)}).encode())

    assert code == 2


def test_a_working_folder_reached_through_a_link_is_judged_in_full(
    tmp_path: Path,
) -> None:
    """Relative to a folder that is a link into ~/.claude, ``settings.json``
    hides the name the rule looks for, so the path is judged in full."""
    home = project_with_guard(tmp_path)
    (home.parent / "cfg").symlink_to(global_settings(tmp_path).parent)

    code, _ = judge(
        home,
        CLAUDE_CODE,
        write_call(
            "Write",
            {"file_path": "settings.json", "content": "{}"},
            cwd=str(home.parent / "cfg"),
        ),
    )

    assert code == 2


def test_a_path_longer_than_4096_characters_is_refused(tmp_path: Path) -> None:
    home = project_with_guard(tmp_path)
    base = str(home.parent) + "/"

    def write(path: str) -> int:
        return judge(
            home, CLAUDE_CODE, write_call("Write", {"file_path": path, "content": "x"})
        )[0]

    assert write(base + "a" * (4096 - len(base))) == 0
    assert write(base + "a" * (4097 - len(base))) == 2
    relative = "a" * (4097 - len(base))
    stdin = write_call("Write", {"file_path": relative, "content": "x"}, cwd=base[:-1])
    assert judge(home, CLAUDE_CODE, stdin)[0] == 2


def test_an_opener_that_follows_no_link_takes_no_dot_dot_step(tmp_path: Path) -> None:
    (tmp_path / "root" / "src").mkdir(parents=True)

    with pytest.raises(PermissionError):
        open_beneath(tmp_path / "root", "src/../../escaped.txt", b"x")
    assert not (tmp_path / "escaped.txt").exists()


# --------------------------------------------------------------------------
# init asks the policy about the real path too
# --------------------------------------------------------------------------


def test_init_flags_a_policy_that_refuses_only_the_written_name(
    tmp_path: Path,
) -> None:
    """A rule anchored to the relative name refuses ``.paveo/policy.json`` and
    misses the absolute path the guard now also judges, so init says so."""
    home = project_with_guard(tmp_path)
    assert (
        _policy_weakness(CLAUDE_CODE, home / "policy.json", frozenset({"claude-code"}))
        is None
    )

    starter = json.loads((home / "policy.json").read_text("utf-8"))
    for rule in starter["agents"][0]["tools"]["allow"]:
        if rule["name"] == "Write":
            rule["constraints"]["file_path"]["not_matches"] = ["^\\.paveo"]
    (home / "policy.json").write_text(json.dumps(starter), "utf-8")

    weakness = _policy_weakness(
        CLAUDE_CODE, home / "policy.json", frozenset({"claude-code"})
    )
    assert weakness is not None
    assert "an edit to it" in weakness


# --------------------------------------------------------------------------
# What it does NOT stop: the check-then-open race (CWE-367)
# --------------------------------------------------------------------------

# Writes ``data`` to ``relative`` under ``root``. Raising OSError means it
# refused to follow the path, which is the passing verdict for an executor.
Opener = Callable[[Path, str, bytes], None]


def check_then_open(tmp_path: Path, opener: Opener) -> Path:
    """The guard approves a path, a folder on it is then swapped for a link to
    somewhere else, and only then is the file opened: a time-of-check to
    time-of-use race (TOCTOU, CWE-367). Returns the file outside the project
    that the write lands on if the opener followed the link.

    Reusable on purpose: an executor that claims to hold the boundary passes its
    own opener (for example one using ``openat2`` with ``RESOLVE_BENEATH``) and
    must leave that file absent. Same fixture, two verdicts.
    """
    home = project_with_guard(tmp_path)
    project = home.parent
    (project / "src").mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()

    approved = judge(
        home,
        CLAUDE_CODE,
        write_call(
            "Write", {"file_path": str(project / "src" / "app.py"), "content": "x"}
        ),
    )
    assert approved == (0, "")

    (project / "src").rename(project / "src-before")
    (project / "src").symlink_to(outside)

    with contextlib.suppress(OSError):
        opener(project, "src/app.py", b"x")
    return outside / "app.py"


def plain_open(root: Path, relative: str, data: bytes) -> None:
    (root / relative).write_bytes(data)


def open_beneath(root: Path, relative: str, data: bytes) -> None:
    """Walks the path one folder at a time with ``O_NOFOLLOW`` and takes no
    ``..`` step, so it never leaves ``root``: what ``openat2`` with
    ``RESOLVE_BENEATH`` does in one call on Linux. Only the code that opens the
    file can do this."""
    *folders, name = relative.split("/")
    if any(part in {"", ".", ".."} for part in (*folders, name)):
        raise PermissionError("only plain names, each beneath the last")
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    descriptor = os.open(root, flags)
    try:
        for folder in folders:
            inner = os.open(folder, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = inner
        file = os.open(
            name,
            os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW,
            0o600,
            dir_fd=descriptor,
        )
        with os.fdopen(file, "wb") as handle:
            handle.write(data)
    finally:
        os.close(descriptor)


def test_check_then_open_race_is_not_stopped(tmp_path: Path) -> None:
    """A limit, stated as a test so it cannot be quietly claimed otherwise. The
    guard runs before the agent's tool and does not open the file, so a link
    made between its yes and the open is followed. Only the opener can close
    this (THREAT_MODEL.md, D79)."""
    landed = check_then_open(tmp_path, plain_open)

    assert landed.exists()


def test_an_opener_that_follows_no_link_keeps_the_race_out(tmp_path: Path) -> None:
    landed = check_then_open(tmp_path, open_beneath)

    assert not landed.exists()
