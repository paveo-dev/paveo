"""The seatbelt for Codex and Cursor (B2c, D57).

The guard, the policy, the log and the panic button are the Claude Code ones
(``test_cli.py``). What is tested here is each agent's contract, as read from
its documentation and, for Codex, its source on 2026-09-26:

- **Codex** blocks on exit 2 only when stderr has text, and runs a hook only
  after the person trusts it, which ``--selftest`` must be able to see missing.
- **Cursor** blocks on exit 2 and reads a ``{"permission": "deny"}`` answer on
  stdout; a hook without ``failClosed`` lets a crash through.
"""

from __future__ import annotations

import io
import json
import subprocess
from collections.abc import Callable, Sequence
from datetime import date
from pathlib import Path

import pytest

from conftest import make_clock
from paveo._harnesses import CLAUDE_CODE, CODEX, CURSOR, Harness, _patch_paths
from paveo._licence import DEVELOPER
from paveo._setup import NOT_A_CALL, _policy_weakness, init, run_hook, selftest
from paveo.cli import guard
from test_cli import paveo_shim

STARTERS = Path(__file__).parent.parent / "src" / "paveo" / "starters"


def seatbelt(tmp_path: Path, harness: Harness) -> Path:
    """A ``.paveo`` folder holding the agent's starter policy."""
    home = tmp_path / ".paveo"
    home.mkdir()
    starter = STARTERS / f"{harness.name}.json"
    (home / "policy.json").write_text(starter.read_text("utf-8"), "utf-8")
    return home


def decide(
    home: Path, harness: Harness, call: dict[str, object]
) -> tuple[int, str, str]:
    stdout, stderr = io.StringIO(), io.StringIO()
    code = guard(
        harness,
        directory=home,
        agent=harness.name,
        stdin=io.BytesIO(json.dumps(call).encode()),
        stdout=stdout,
        stderr=stderr,
        stopped=lambda: False,
        now=make_clock(),
    )
    return code, stdout.getvalue(), stderr.getvalue()


def codex(tool: str, tool_input: dict[str, object]) -> dict[str, object]:
    """The fields Codex sends to a PreToolUse hook (its input schema)."""
    return {
        "session_id": "0199-thread",
        "turn_id": "t1",
        "transcript_path": None,
        "cwd": "/work",
        "hook_event_name": "PreToolUse",
        "model": "gpt-5.6",
        "permission_mode": "default",
        "tool_name": tool,
        "tool_use_id": "call_1",
        "tool_input": tool_input,
    }


def patch(*headers: str, body: str = "+x") -> str:
    return (
        "*** Begin Patch\n"
        + "\n".join(f"{h}\n{body}" for h in headers)
        + ("\n*** End Patch\n")
    )


def shell(command: str) -> dict[str, object]:
    """What Cursor sends to beforeShellExecution (its documentation)."""
    return {
        "conversation_id": "c1",
        "generation_id": "g1",
        "hook_event_name": "beforeShellExecution",
        "workspace_roots": ["/work"],
        "command": command,
        "cwd": "/work",
        "sandbox": False,
    }


def pre_tool(tool: str, tool_input: dict[str, object]) -> dict[str, object]:
    return {
        "conversation_id": "c1",
        "hook_event_name": "preToolUse",
        "tool_name": tool,
        "tool_input": tool_input,
        "tool_use_id": "u1",
        "cwd": "/work",
    }


# --------------------------------------------------------------------------
# Codex: exit 2 with text on stderr, silence otherwise
# --------------------------------------------------------------------------


def test_codex_refuses_rm_rf_on_exit_2_with_a_reason_on_stderr(tmp_path: Path) -> None:
    """Codex ignores an exit 2 whose stderr is empty (pre_tool_use.rs)."""
    code, stdout, stderr = decide(
        seatbelt(tmp_path, CODEX), CODEX, codex("Bash", {"command": "rm -rf src"})
    )
    assert code == 2
    assert stderr.strip()
    assert stdout == ""


def test_codex_lets_ordinary_work_through_saying_nothing(tmp_path: Path) -> None:
    """Exit 0 with no output leaves the call to Codex's own approval flow."""
    home = seatbelt(tmp_path, CODEX)
    for call in (
        codex("Bash", {"command": "git status"}),
        codex("apply_patch", {"command": patch("*** Update File: src/app.py")}),
        # A patch whose content mentions the guard is not an edit of the guard.
        codex(
            "apply_patch",
            {"command": patch("*** Update File: README.md", body="+see .paveo/")},
        ),
    ):
        assert decide(home, CODEX, call) == (0, "", "")


@pytest.mark.parametrize(
    "header",
    [
        "*** Update File: .paveo/policy.json",
        "*** Add File: /work/.PAVEO/stop",
        "*** Delete File: .codex/hooks.json",
        "*** Update File: src/app.py\n*** Move to: .codex/config.toml",
        "*** Add File: ../outside.py",
        "*** Add File: src/../../outside.py",
        "   *** Delete File: .paveo/audit.jsonl",
    ],
)
def test_codex_refuses_a_patch_that_names_the_guard_or_leaves_the_project(
    tmp_path: Path, header: str
) -> None:
    code, _, _ = decide(
        seatbelt(tmp_path, CODEX),
        CODEX,
        codex("apply_patch", {"command": patch("*** Update File: ok.py", header)}),
    )
    assert code == 2


def test_the_paths_are_read_from_headers_only_in_one_pass() -> None:
    text = (
        patch("*** Add File: a.py", "*** Update File: b.py", "*** Delete File: c.py")
        + "*** Move to: d.py\n+*** Add File: in-content.py\n"
    )
    assert _patch_paths(text) == "a.py\nb.py\nc.py\nd.py"


def test_a_crafted_patch_is_judged_in_well_under_a_second(tmp_path: Path) -> None:
    """A header pattern searched over the whole patch was quadratic here."""
    import time  # noqa: PLC0415

    home = seatbelt(tmp_path, CODEX)
    crafted = "*** Add File: " * ((1 << 20) // 14 - 8)
    started = time.perf_counter()
    decide(home, CODEX, codex("apply_patch", {"command": crafted}))
    assert time.perf_counter() - started < 1.0


# --------------------------------------------------------------------------
# Cursor: exit 2, and the same refusal as JSON on stdout
# --------------------------------------------------------------------------


def test_cursor_refuses_a_shell_command_in_its_own_format(tmp_path: Path) -> None:
    code, stdout, stderr = decide(
        seatbelt(tmp_path, CURSOR), CURSOR, shell("rm -rf node_modules")
    )
    assert code == 2
    answer = json.loads(stdout)
    assert answer["permission"] == "deny"
    assert answer["agent_message"] == answer["user_message"] == stderr.strip()


def test_cursor_lets_ordinary_work_through_saying_nothing(tmp_path: Path) -> None:
    """No "allow": it could skip Cursor's own approval (D57)."""
    home = seatbelt(tmp_path, CURSOR)
    assert decide(home, CURSOR, shell("npm test")) == (0, "", "")
    write = {"file_path": "/work/src/app.ts", "content": "x"}
    assert decide(home, CURSOR, pre_tool("Write", write)) == (0, "", "")


@pytest.mark.parametrize(
    "call",
    [
        pre_tool("Write", {"file_path": "/work/.cursor/hooks.json", "content": "{}"}),
        pre_tool("Delete", {"file_path": "/work/.paveo/policy.json"}),
        pre_tool("Write", {"file_path": "../x", "content": ""}),
        # A field the starter does not name: refused, never waved through.
        pre_tool("Write", {"file_path": "/work/a", "contents": "", "mode": "x"}),
        shell("cat ~/.cursor/hooks.json"),
        shell("paveo resume --dir .paveo"),
    ],
)
def test_cursor_refuses_edits_to_the_guard(
    tmp_path: Path, call: dict[str, object]
) -> None:
    assert decide(seatbelt(tmp_path, CURSOR), CURSOR, call)[0] == 2


@pytest.mark.parametrize(
    "call",
    [
        {"hook_event_name": "beforeReadFile", "file_path": "/work/a"},
        {"hook_event_name": "beforeShellExecution"},
        {"tool_name": "Write", "tool_input": {"file_path": "a", "content": ""}},
        {"paveo_selftest": True},
    ],
)
def test_cursor_refuses_what_it_cannot_read(
    tmp_path: Path, call: dict[str, object]
) -> None:
    code, stdout, stderr = decide(seatbelt(tmp_path, CURSOR), CURSOR, call)
    assert code == 2
    assert NOT_A_CALL in stderr
    assert json.loads(stdout)["permission"] == "deny"


# --------------------------------------------------------------------------
# init and --selftest
# --------------------------------------------------------------------------


def run_init(harness: Harness, project: Path, program: Path) -> tuple[int, str]:
    """Never the real ~/.codex or ~/.cursor: the project's file only."""
    out = io.StringIO()
    code = init(
        harness,
        project=project,
        program=program,
        hook_files=[project / harness.folder / harness.hook_file],
        run=run_hook,
        out=out,
    )
    return code, out.getvalue()


def test_init_codex_writes_a_hook_that_refuses_and_says_trust_is_left(
    tmp_path: Path,
) -> None:
    project = tmp_path / "my project"
    project.mkdir()
    shim = paveo_shim(tmp_path / "bin")

    code, out = run_init(CODEX, project, shim)

    assert code == 0, out
    assert "/hooks" in out
    assert "Done." not in out
    hooks = json.loads((project / ".codex" / "hooks.json").read_text("utf-8"))
    [group] = hooks["hooks"]["PreToolUse"]
    assert group["matcher"] == "^(?:Bash|apply_patch)$"
    assert "hooks.json" in (project / ".codex" / ".gitignore").read_text("utf-8")
    # The literal command, through a shell, from another folder: Codex runs a
    # hook from wherever the session started, so --dir must be absolute.
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    done = subprocess.run(  # noqa: S602 - our own shim, as the hook names it
        group["hooks"][0]["command"],
        shell=True,
        cwd=elsewhere,
        input=json.dumps(codex("Bash", {"command": "rm -rf src"})).encode(),
        capture_output=True,
        check=False,
    )
    assert done.returncode == 2, done.stderr
    assert done.stderr.strip()


def test_init_cursor_writes_both_hooks_failing_closed(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()

    code, out = run_init(CURSOR, project, paveo_shim(tmp_path / "bin"))

    assert code == 0, out
    hooks = json.loads((project / ".cursor" / "hooks.json").read_text("utf-8"))
    assert hooks["version"] == 1
    [shell_hook] = hooks["hooks"]["beforeShellExecution"]
    [tool_hook] = hooks["hooks"]["preToolUse"]
    assert shell_hook["failClosed"] is tool_hook["failClosed"] is True
    assert "matcher" not in shell_hook
    assert tool_hook["matcher"] == "Write|Delete"


def test_init_keeps_a_team_hook_file_and_does_not_ignore_it(tmp_path: Path) -> None:
    """A committed hooks.json is the team's: added to, never hidden from git."""
    project = tmp_path / "project"
    (project / ".codex").mkdir(parents=True)
    theirs = {"matcher": "Bash", "hooks": [{"type": "command", "command": "lint"}]}
    (project / ".codex" / "hooks.json").write_text(
        json.dumps({"hooks": {"PreToolUse": [theirs]}}), "utf-8"
    )

    code, out = run_init(CODEX, project, paveo_shim(tmp_path / "bin"))

    assert code == 0, out
    assert "keep it out of any commit" in out
    assert not (project / ".codex" / ".gitignore").exists()
    hooks = json.loads((project / ".codex" / "hooks.json").read_text("utf-8"))
    assert hooks["hooks"]["PreToolUse"][0] == theirs
    assert len(hooks["hooks"]["PreToolUse"]) == 2


def test_init_refuses_a_matcher_that_misses_apply_patch(tmp_path: Path) -> None:
    project = tmp_path / "project"
    (project / ".codex").mkdir(parents=True)
    shim = paveo_shim(tmp_path / "bin")
    mine = {"matcher": "^Bash$", "hooks": [{"command": f"{shim} guard codex"}]}
    (project / ".codex" / "hooks.json").write_text(
        json.dumps({"hooks": {"PreToolUse": [mine]}}), "utf-8"
    )

    code, out = run_init(CODEX, project, shim)

    assert code == 1
    assert "sees apply_patch" in out


# --------------------------------------------------------------------------
# One policy for every agent in a project (D88)
# --------------------------------------------------------------------------


def agent_ids(project: Path) -> list[str]:
    policy = json.loads((project / ".paveo" / "policy.json").read_text("utf-8"))
    return [agent["id"] for agent in policy["agents"]]


def test_init_for_a_second_agent_adds_it_to_the_policy_already_there(
    tmp_path: Path,
) -> None:
    """Before D88 the second init kept a policy naming only the first agent, and
    every call from the second was refused as unknown (/code-review, 11 Oct)."""
    project = tmp_path / "project"
    project.mkdir()
    shim = paveo_shim(tmp_path / "bin")
    assert run_init(CLAUDE_CODE, project, shim)[0] == 0

    code, out = run_init(CURSOR, project, shim)

    assert code == 0, out
    assert "added     'cursor'" in out
    assert agent_ids(project) == ["claude-code", "cursor"]
    home = project / ".paveo"
    assert decide(home, CURSOR, shell("rm -rf src"))[0] == 2
    assert decide(home, CURSOR, shell("git status"))[0] == 0
    claude = {"session_id": "s1", "tool_name": "Bash"}
    assert (
        decide(home, CLAUDE_CODE, {**claude, "tool_input": {"command": "rm -rf src"}})[
            0
        ]
        == 2
    )
    assert (
        decide(home, CLAUDE_CODE, {**claude, "tool_input": {"command": "ls"}})[0] == 0
    )


def test_init_for_the_same_agent_again_keeps_the_policy_as_it_is(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    shim = paveo_shim(tmp_path / "bin")
    assert run_init(CODEX, project, shim)[0] == 0
    before = (project / ".paveo" / "policy.json").read_text("utf-8")

    code, out = run_init(CODEX, project, shim)

    assert code == 0, out
    assert "added     'codex'" not in out
    assert (project / ".paveo" / "policy.json").read_text("utf-8") == before


def test_one_developer_runs_all_three_agents_on_the_free_plan(
    tmp_path: Path,
) -> None:
    """Until D94 the free plan covered two agents, so a third `init` was refused
    and one developer could not try Claude Code, Cursor and Codex together."""
    project = tmp_path / "project"
    project.mkdir()
    shim = paveo_shim(tmp_path / "bin")
    assert run_init(CLAUDE_CODE, project, shim)[0] == 0
    assert run_init(CURSOR, project, shim)[0] == 0

    code, out = run_init(CODEX, project, shim)

    assert code == 0, out
    agents = json.loads((project / ".paveo" / "policy.json").read_text("utf-8"))
    assert [a["id"] for a in agents["agents"]] == ["claude-code", "cursor", "codex"]


def test_init_refuses_to_add_an_agent_past_the_free_plan_and_changes_nothing(
    tmp_path: Path,
) -> None:
    """The append path: a policy already at the plan's number, and init for one
    more. Refused before anything is written (D94 kept this when it raised the
    number; /code-review found the first rewrite had dropped it)."""
    project, shim = claude_project(tmp_path)
    free = DEVELOPER.agents
    assert free is not None
    claude = json.loads((STARTERS / "claude-code.json").read_text("utf-8"))["agents"][0]
    fillers = [{**claude, "id": f"other-{n}"} for n in range(free - 1)]
    before = edit_policy(project, lambda d: d.update(agents=[*d["agents"], *fillers]))

    code, out = run_init(CODEX, project, shim)

    assert code == 1
    assert f"covers {free} per policy" in out
    assert "Nothing was changed" in out
    assert_nothing_changed(project, before)
    assert not (project / ".codex").exists()


def test_the_policy_check_fails_an_agent_the_policy_does_not_name(
    tmp_path: Path,
) -> None:
    """A kept hook can judge calls as an agent the policy lacks: every call is
    then refused as unknown, and init must not call that a working seatbelt."""
    home = seatbelt(tmp_path, CLAUDE_CODE)

    weakness = _policy_weakness(
        CURSOR,
        home / "policy.json",
        frozenset({"cursor"}),
        DEVELOPER,
        today=date(2026, 10, 12),
    )

    assert weakness is not None
    assert "names no agent 'cursor'" in weakness


def test_the_policy_check_fails_an_agent_past_the_plan(tmp_path: Path) -> None:
    home = tmp_path / ".paveo"
    home.mkdir()
    cursor = json.loads((STARTERS / "cursor.json").read_text("utf-8"))["agents"][0]
    free = DEVELOPER.agents
    assert free is not None
    others = [{**cursor, "id": f"other-{n}"} for n in range(free)]
    (home / "policy.json").write_text(
        json.dumps({"version": 1, "policy_id": "full", "agents": [*others, cursor]}),
        "utf-8",
    )

    weakness = _policy_weakness(
        CURSOR,
        home / "policy.json",
        frozenset({"cursor"}),
        DEVELOPER,
        today=date(2026, 10, 12),
    )

    assert weakness is not None
    assert f"past the developer plan's {free} agents" in weakness


def claude_project(tmp_path: Path) -> tuple[Path, Path]:
    """A project set up for Claude Code, and the paveo that set it up."""
    project = tmp_path / "project"
    project.mkdir()
    shim = paveo_shim(tmp_path / "bin")
    assert run_init(CLAUDE_CODE, project, shim)[0] == 0
    return project, shim


def edit_policy(project: Path, change: Callable[[dict[str, object]], None]) -> str:
    path = project / ".paveo" / "policy.json"
    document = json.loads(path.read_text("utf-8"))
    change(document)
    path.write_text(json.dumps(document), "utf-8")
    return path.read_text("utf-8")


def assert_nothing_changed(project: Path, before: str) -> None:
    assert (project / ".paveo" / "policy.json").read_text("utf-8") == before
    assert not (project / ".cursor").exists()


def test_init_refuses_to_merge_into_a_policy_that_does_not_load(tmp_path: Path) -> None:
    """Before the fix the merge rewrote the file and wrote the hook, then failed
    (/code-review, D88): a failing run must change nothing."""
    project, shim = claude_project(tmp_path)
    before = edit_policy(project, lambda d: d.update(version=99))

    code, out = run_init(CURSOR, project, shim)

    assert code == 1
    assert "does not load" in out
    assert_nothing_changed(project, before)


def test_init_refuses_to_add_an_agent_to_a_policy_that_fails_open(
    tmp_path: Path,
) -> None:
    project, shim = claude_project(tmp_path)
    before = edit_policy(project, lambda d: d.update(defaults={"fail_open": True}))

    code, out = run_init(CURSOR, project, shim)

    assert code == 1
    assert "fail_open" in out
    assert_nothing_changed(project, before)


def test_init_refuses_before_writing_when_the_agent_is_already_past_the_plan(
    tmp_path: Path,
) -> None:
    """A cloned policy listing the agent past the free plan's number: refused
    before any hook is written, not after (/code-review, D88)."""
    project, shim = claude_project(tmp_path)
    cursor = json.loads((STARTERS / "cursor.json").read_text("utf-8"))["agents"][0]
    codex_agent = json.loads((STARTERS / "codex.json").read_text("utf-8"))["agents"][0]
    free = DEVELOPER.agents
    assert free is not None
    # claude-code and codex, then others up to the plan's number, then cursor.
    fillers = [{**codex_agent, "id": f"other-{n}"} for n in range(free - 2)]
    before = edit_policy(
        project,
        lambda d: d.update(agents=[*d["agents"], codex_agent, *fillers, cursor]),
    )

    code, out = run_init(CURSOR, project, shim)

    assert code == 1
    assert f"covers {free} per policy" in out
    assert_nothing_changed(project, before)


def test_init_refuses_a_licence_key_that_is_a_link(tmp_path: Path) -> None:
    project, shim = claude_project(tmp_path)
    (tmp_path / "elsewhere").write_text("not a key", "utf-8")
    (project / ".paveo" / "licence.key").symlink_to(tmp_path / "elsewhere")

    code, out = run_init(CURSOR, project, shim)

    assert code == 1
    assert "symbolic link" in out
    assert not (project / ".cursor").exists()


def test_the_merge_keeps_characters_a_person_wrote(tmp_path: Path) -> None:
    project, shim = claude_project(tmp_path)

    def accent(document: dict[str, object]) -> None:
        agents = document["agents"]
        assert isinstance(agents, list)
        agents[0]["tools"]["allow"][0]["constraints"]["command"]["not_matches"].append(
            "supprimé"
        )

    edit_policy(project, accent)

    assert run_init(CURSOR, project, shim)[0] == 0
    assert "supprimé" in (project / ".paveo" / "policy.json").read_text("utf-8")


def codex_project(tmp_path: Path) -> tuple[Path, Path]:
    project = tmp_path / "project"
    project.mkdir()
    assert run_init(CODEX, project, paveo_shim(tmp_path / "bin"))[0] == 0
    return project, project / ".codex" / "hooks.json"


def check(
    harness: Harness, project: Path, codex_config: Path | None = None
) -> tuple[int, str]:
    out = io.StringIO()

    def run(argv: Sequence[str], where: Path) -> tuple[int, str]:
        return run_hook(argv, where)

    code = selftest(
        harness,
        hook_files=[project / harness.folder / harness.hook_file],
        project=project,
        run=run,
        confirm=lambda _: False,
        out=out,
        codex_config=codex_config,
    )
    return code, out.getvalue()


def test_codex_selftest_fails_a_hook_nobody_has_trusted(tmp_path: Path) -> None:
    """Codex skips an untrusted hook without a word: the missing record is the
    one thing here paveo can know for certain."""
    project, _ = codex_project(tmp_path)

    code, out = check(CODEX, project, tmp_path / "absent-config.toml")

    assert code == 1
    assert "no record of you trusting" in out
    assert f"trusts {project}" in out


def trust(config: Path, project: Path, hooks: Path, **state: object) -> None:
    lines = [
        f'[projects."{project}"]',
        'trust_level = "trusted"',
        f'[hooks.state."{hooks}:pre_tool_use:0:0"]',
        *(f"{key} = {json.dumps(value)}" for key, value in state.items()),
    ]
    config.write_text("\n".join(lines) + "\n", "utf-8")


def test_codex_selftest_passes_a_trusted_hook_but_says_what_it_cannot_check(
    tmp_path: Path,
) -> None:
    project, hooks = codex_project(tmp_path)
    config = tmp_path / "config.toml"
    trust(config, project, hooks, trusted_hash="sha256:abc")

    code, out = check(CODEX, project, config)

    assert code == 0, out
    assert "cannot check it is this exact hook" in out


@pytest.mark.parametrize(
    ("extra", "said"),
    [
        ({"trusted_hash": "sha256:abc", "enabled": False}, "switched off"),
        ({}, "no record of you trusting"),
    ],
)
def test_codex_selftest_fails_a_hook_switched_off_or_untrusted(
    tmp_path: Path, extra: dict[str, object], said: str
) -> None:
    project, hooks = codex_project(tmp_path)
    config = tmp_path / "config.toml"
    trust(config, project, hooks, **extra)

    code, out = check(CODEX, project, config)

    assert code == 1
    assert said in out


def test_codex_selftest_fails_when_hooks_are_switched_off(tmp_path: Path) -> None:
    project, hooks = codex_project(tmp_path)
    config = tmp_path / "config.toml"
    trust(config, project, hooks, trusted_hash="sha256:abc")
    config.write_text(
        "[features]\nhooks = false\n" + config.read_text("utf-8"), "utf-8"
    )

    code, out = check(CODEX, project, config)

    assert code == 1
    assert "features.hooks = false" in out


def test_codex_selftest_fails_a_config_it_cannot_read(tmp_path: Path) -> None:
    project, _ = codex_project(tmp_path)
    config = tmp_path / "config.toml"
    config.write_text("[unclosed", "utf-8")

    code, out = check(CODEX, project, config)

    assert code == 1
    assert "cannot tell whether Codex runs the hook" in out


def test_cursor_selftest_fails_a_hook_without_fail_closed(tmp_path: Path) -> None:
    """Without it, Cursor lets the call through when the guard crashes."""
    project = tmp_path / "project"
    project.mkdir()
    assert run_init(CURSOR, project, paveo_shim(tmp_path / "bin"))[0] == 0
    path = project / ".cursor" / "hooks.json"
    hooks = json.loads(path.read_text("utf-8"))
    del hooks["hooks"]["preToolUse"][0]["failClosed"]
    del hooks["hooks"]["beforeShellExecution"]
    path.write_text(json.dumps(hooks), "utf-8")

    code, out = check(CURSOR, project)

    assert code == 1
    assert 'no "failClosed": true' in out
    assert "no paveo hook for Cursor's beforeShellExecution" in out


def test_init_cursor_puts_back_a_hook_that_was_removed(tmp_path: Path) -> None:
    """--selftest says to run init again; init must then actually add it."""
    project = tmp_path / "project"
    project.mkdir()
    shim = paveo_shim(tmp_path / "bin")
    assert run_init(CURSOR, project, shim)[0] == 0
    path = project / ".cursor" / "hooks.json"
    hooks = json.loads(path.read_text("utf-8"))
    del hooks["hooks"]["beforeShellExecution"]
    path.write_text(json.dumps(hooks), "utf-8")

    code, out = run_init(CURSOR, project, shim)

    assert code == 0, out
    hooks = json.loads(path.read_text("utf-8"))
    assert len(hooks["hooks"]["beforeShellExecution"]) == 1
    assert len(hooks["hooks"]["preToolUse"]) == 1


# --------------------------------------------------------------------------
# What /code-review found (D57), each pinned
# --------------------------------------------------------------------------


@pytest.mark.parametrize("separator", ["\r", "\x0b", "\x0c", "\x1c", "\x85", "\u2028"])
def test_a_character_python_splits_on_and_codex_does_not_hides_no_path(
    tmp_path: Path, separator: str
) -> None:
    """One line to Codex, which writes the whole path; `splitlines` read `x`."""
    header = f"*** Update File: x{separator}/../.paveo/policy.json"
    code, _, _ = decide(
        seatbelt(tmp_path, CODEX),
        CODEX,
        codex("apply_patch", {"command": patch(header)}),
    )
    assert code == 2


@pytest.mark.parametrize(
    "tool_input",
    [
        {"command": ["apply_patch", "*** Update File: .paveo/policy.json"]},
        {"command": ["x"], "paths": "README.md"},
        {"paths": "README.md"},
    ],
)
def test_a_patch_without_text_is_refused_whatever_paths_it_brings(
    tmp_path: Path, tool_input: dict[str, object]
) -> None:
    assert (
        decide(seatbelt(tmp_path, CODEX), CODEX, codex("apply_patch", tool_input))[0]
        == 2
    )


def test_paths_a_call_brings_are_replaced_by_the_ones_read(tmp_path: Path) -> None:
    text = patch("*** Update File: .paveo/policy.json")
    call = codex("apply_patch", {"command": text, "paths": "README.md"})
    assert decide(seatbelt(tmp_path, CODEX), CODEX, call)[0] == 2


def test_a_global_hook_for_another_project_is_not_this_ones(tmp_path: Path) -> None:
    """~/.codex/hooks.json guarding project A must not make init skip project B,
    nor pass B's self-test: B's policy and `paveo stop` would never apply."""
    shim = paveo_shim(tmp_path / "bin")
    other = tmp_path / "a"
    other.mkdir()
    assert run_init(CODEX, other, shim)[0] == 0
    project = tmp_path / "b"
    project.mkdir()
    global_hooks = other / ".codex" / "hooks.json"  # stands in for ~/.codex

    out = io.StringIO()
    code = init(
        CODEX,
        project=project,
        program=shim,
        hook_files=[global_hooks, project / ".codex" / "hooks.json"],
        run=run_hook,
        out=out,
    )

    assert code == 0, out.getvalue()
    assert "guards another folder" in out.getvalue()
    mine = json.loads((project / ".codex" / "hooks.json").read_text("utf-8"))
    [group] = mine["hooks"]["PreToolUse"]
    assert str(project / ".paveo") in group["hooks"][0]["command"]
    checked = io.StringIO()
    assert (
        selftest(
            CODEX,
            hook_files=[global_hooks],
            project=project,
            run=run_hook,
            confirm=lambda _: False,
            out=checked,
        )
        == 1
    )
    assert f"reads {project / '.paveo'}" in checked.getvalue()


def test_codex_trust_is_found_at_the_git_root_as_codex_finds_it(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    (root / ".git").mkdir(parents=True)
    project = root / "service"
    project.mkdir()
    assert run_init(CODEX, project, paveo_shim(tmp_path / "bin"))[0] == 0
    hooks = project / ".codex" / "hooks.json"
    config = tmp_path / "config.toml"
    trust(config, root, hooks, trusted_hash="sha256:abc")

    assert check(CODEX, project, config)[0] == 0


def test_the_first_codex_project_entry_decides_even_without_a_level(
    tmp_path: Path,
) -> None:
    """Codex stops at the first key present: an entry for the folder with no
    trust level hides a trusted repository root (project_trust.rs)."""
    root = tmp_path / "repo"
    (root / ".git").mkdir(parents=True)
    project = root / "service"
    project.mkdir()
    assert run_init(CODEX, project, paveo_shim(tmp_path / "bin"))[0] == 0
    hooks = project / ".codex" / "hooks.json"
    config = tmp_path / "config.toml"
    trust(config, root, hooks, trusted_hash="sha256:abc")
    config.write_text(
        f'[projects."{project}"]\nmodel = "x"\n' + config.read_text("utf-8"), "utf-8"
    )

    code, out = check(CODEX, project, config)
    assert code == 1
    assert f"trusts {project}" in out


# --------------------------------------------------------------------------
# What /security-review found (D57), each pinned
# --------------------------------------------------------------------------


@pytest.mark.parametrize("harness", [CODEX, CURSOR])
def test_init_asks_the_policy_as_the_agent_a_shipped_hook_uses(
    tmp_path: Path, harness: Harness
) -> None:
    """A repository ships a strict agent under the default name and a lax one its
    own hook really uses: init must not say the seatbelt is on."""
    project = tmp_path / "project"
    home = project / ".paveo"
    home.mkdir(parents=True)
    starter = json.loads((STARTERS / f"{harness.name}.json").read_text("utf-8"))
    lax = {
        "id": "lax",
        "tools": {
            "allow": [
                {"name": n} for n in ("Bash", "Shell", "apply_patch", "Write", "Delete")
            ]
        },
    }
    starter["agents"].append(lax)
    (home / "policy.json").write_text(json.dumps(starter), "utf-8")
    shim = paveo_shim(tmp_path / "bin")
    command = f"{shim} guard {harness.name} --dir {home} --agent lax"
    folder = project / harness.folder
    folder.mkdir()
    if harness.flat:
        hooks = {
            event.name: [{"command": command, "failClosed": True}]
            for event in harness.events
        }
        document: dict[str, object] = {"version": 1, "hooks": hooks}
    else:
        group = {"matcher": "^(?:Bash|apply_patch)$", "hooks": [{"command": command}]}
        document = {"hooks": {"PreToolUse": [group]}}
    (folder / harness.hook_file).write_text(json.dumps(document), "utf-8")

    code, out = run_init(harness, project, shim)

    assert code == 1
    assert "for 'lax'" in out


@pytest.mark.parametrize(
    ("harness", "tool", "arguments"),
    [
        (
            CODEX,
            "apply_patch",
            {
                "command": patch(
                    "*** Update File: .venv/lib/python3.12/site-packages/paveo/cli.py"
                )
            },
        ),
        (
            CODEX,
            "apply_patch",
            {
                "command": patch(
                    "*** Add File: .venv/lib/python3.12/site-packages/x.pth"
                )
            },
        ),
        (
            CURSOR,
            "Write",
            {
                "file_path": "/w/.venv/lib/python3.12/site-packages/paveo/cli.py",
                "content": "",
            },
        ),
        (
            CURSOR,
            "Write",
            {"file_path": "/w/venv/lib/site-packages/evil.pth", "content": ""},
        ),
    ],
)
def test_the_guards_own_installed_code_is_refused(
    tmp_path: Path, harness: Harness, tool: str, arguments: dict[str, object]
) -> None:
    """A virtualenv inside the project holds paveo itself: editing it, or adding
    a .pth file Python runs at start-up, would switch the guard off."""
    call = codex(tool, arguments) if harness is CODEX else pre_tool(tool, arguments)
    assert decide(seatbelt(tmp_path, harness), harness, call)[0] == 2
