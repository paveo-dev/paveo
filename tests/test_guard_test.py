"""``paveo test``: a project's own Claude Code hooks against bypass spellings (D102).

The decisions are tested with a fake runner, so each of Claude Code's rules for a
hook's answer is one fast test. The runner is tested on real processes: a hook
that hangs, crashes, floods its output, leaves something running or never reads
its input. Last, the real thing: a guard written the way most are, and our own
starter, whose known gaps the README's limits must name.
"""

from __future__ import annotations

import io
import json
import os
import signal
import stat
import sys
import time
from collections.abc import Callable
from pathlib import Path

import pytest

import paveo
from conftest import KNOWN_GAPS, SENTINEL
from paveo._bypasses import ACTIONS, ORDINARY, VERSION
from paveo._guard_test import (
    Handler,
    Run,
    _failure_notes,
    _Tally,
    command,
    run_handler,
)

SOURCE = str(Path(paveo.__file__).parent.parent)
PROBE = "paveo-test-probe"
ALL = [action.name for action in ACTIONS]
# Counted from the list, so a spelling added to it breaks no test here.
SIZE = {action.name: len(action.spellings) for action in ACTIONS}
CALLS = sum(SIZE.values()) + len(ORDINARY)
PUSHES, DELETES = SIZE["force-push"], SIZE["delete"]


def _settings(
    tmp_path: Path,
    entries: list[dict[str, object]],
    *,
    matcher: object = "Bash|Read",
    name: str = "settings.json",
    **extra: object,
) -> Path:
    path = tmp_path / ".claude" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {"hooks": {"PreToolUse": [{"matcher": matcher, "hooks": entries}]}, **extra}
        )
    )
    return path


def _hook(cmd: str = "guard", **entry: object) -> dict[str, object]:
    return {"type": "command", "command": cmd, **entry}


def _decides(decision: str | None, **fields: object) -> bytes:
    """A decision as the reference's schema has it: the event named."""
    specific = {"hookEventName": "PreToolUse", "permissionDecision": decision}
    return json.dumps({"hookSpecificOutput": {**specific, **fields}}).encode()


def _judge(answer: Callable[[str], Run]) -> Callable[[Handler, bytes], Run]:
    """A runner that answers from the call's command or file alone."""

    def run(_handler: Handler, payload: bytes) -> Run:
        tool_input = json.loads(payload)["tool_input"]
        return answer(tool_input.get("command") or tool_input["file_path"])

    return run


def _ordinary(text: str) -> bool:
    return text == "git status" or text.endswith("README.md")


def _blocks_all_but_ordinary(text: str) -> Run:
    return Run(0 if _ordinary(text) else 2, b"")


def _test(
    tmp_path: Path,
    files: list[Path],
    run: Callable[[Handler, bytes], Run],
    expect: list[str] | None = None,
) -> tuple[int, str]:
    out = io.StringIO()
    code = command(files, project=tmp_path, expect=expect or [], out=out, run=run)
    return code, out.getvalue()


# --------------------------------------------------------------------------
# What counts as stopped, read as Claude Code reads it
# --------------------------------------------------------------------------


def test_a_guard_that_stops_every_spelling_passes(tmp_path: Path) -> None:
    files = [_settings(tmp_path, [_hook()])]
    code, out = _test(tmp_path, files, _judge(_blocks_all_but_ordinary), ALL)
    assert code == 0, out
    for action in ACTIONS:
        assert f"stopped {len(action.spellings)} of {len(action.spellings)}" in out
    assert "nothing expected got past the hooks" in out


def test_a_guard_that_stops_only_the_plain_spelling_fails_and_names_the_rest(
    tmp_path: Path,
) -> None:
    plain = {action.spellings[0].text for action in ACTIONS}

    def answer(text: str) -> Run:
        return Run(2 if text in plain or text.endswith("/.env") else 0, b"")

    code, out = _test(tmp_path, [_settings(tmp_path, [_hook()])], _judge(answer))
    assert code == 1
    assert f"git -C . push --force {PROBE} main" in out
    assert "find /paveo-test-probe -delete" in out
    assert "these get past the hooks: force-push, delete, hard-reset, clean" in out


@pytest.mark.parametrize(
    ("stdout", "exit_code"),
    [
        (
            _decides("deny"),
            0,
        ),
        # The reference's table: on an exit other than 0 or 2, valid JSON
        # alone decides, so a deny printed with exit 1 still blocks.
        (
            _decides("deny"),
            1,
        ),
        (b'\n  {"decision": "block", "reason": "no"}\n', 0),  # deprecated, still a deny
        (b'{"permissionDecision": "allow"}', 2),  # exit 2 blocks whatever is printed
        (b"not json at all", 2),
    ],
)
def test_each_way_claude_code_blocks_is_a_stop(
    tmp_path: Path, stdout: bytes, exit_code: int
) -> None:
    def answer(text: str) -> Run:
        return Run(0, b"") if _ordinary(text) else Run(exit_code, stdout)

    code, out = _test(tmp_path, [_settings(tmp_path, [_hook()])], _judge(answer), ALL)
    assert code == 0, out


def test_ask_counts_as_stopped_and_says_a_person_decides(tmp_path: Path) -> None:
    asks = _decides("ask")

    def answer(text: str) -> Run:
        return Run(0, b"" if _ordinary(text) else asks)

    code, out = _test(tmp_path, [_settings(tmp_path, [_hook()])], _judge(answer), ALL)
    assert code == 0
    assert f"stopped {PUSHES} of {PUSHES} ({PUSHES} by asking a person)" in out


@pytest.mark.parametrize(
    "stdout",
    [
        _decides("allow"),
        _decides("defer"),
        b'{"continue": false}',  # its effect on the call is unclear: never a stop
        b"Blocked: rm is not allowed",  # plain text on exit 0 decides nothing
        b'{"decision": "approve"}',
    ],
)
def test_answers_that_decide_nothing_let_the_call_through(
    tmp_path: Path, stdout: bytes
) -> None:
    code, out = _test(
        tmp_path,
        [_settings(tmp_path, [_hook()])],
        _judge(lambda _text: Run(0, stdout)),
        ["delete"],
    )
    assert code == 1
    assert f"delete      EXPECTED, but no hook stops any of {DELETES} spellings" in out


@pytest.mark.parametrize(
    ("run", "reason"),
    [
        (Run(1, b"BLOCKED"), "exit 1"),
        (Run(127, b""), "exit 127"),
        (Run(0, b"{not json}"), "printed JSON Claude Code cannot read"),
        (Run(0, b'{"hookSpecificOutput": "deny"}'), "fails Claude Code's schema"),
        (
            Run(
                0,
                _decides("no"),
            ),
            "schema",
        ),
        # A deny that does not name its event fails the schema: Claude Code
        # lets the call through, so it must never count as a stop.
        (Run(0, b'{"hookSpecificOutput": {"permissionDecision": "deny"}}'), "schema"),
        (
            Run(
                0,
                _decides("deny", hookEventName="PostToolUse"),
            ),
            "schema",
        ),
        (
            Run(
                0,
                _decides("deny", permissionDecisionReason=7),
            ),
            "schema",
        ),
        (Run(0, b'{"decision": "maybe"}'), "schema"),
        (Run(0, b'{"decision": "block", "reason": 7}'), "schema"),
        # The whole object is checked before the decision in it is read.
        (Run(0, b'{"continue": "no", ' + _decides("deny")[1:]), "schema"),
        (Run(0, b'{"systemMessage": 1, ' + _decides("deny")[1:]), "schema"),
        (Run(0, _decides("deny", additionalContext=7)), "schema"),
        (Run(0, _decides("deny", updatedInput="rm")), "schema"),
        # A decision that cannot be a dict key is a schema failure, not a crash.
        (Run(0, b'{"decision": []}'), "schema"),
        (Run(0, b'{"decision": {}}'), "schema"),
        (Run(0, b'{"terminalSequence": 1, ' + _decides("deny")[1:]), "schema"),
        (Run(0, b'{"decision": "deny", ' + _decides("deny")[1:]), "schema"),
        (Run(0, b'{"hookSpecificOutput": null, "decision": "block"}'), "schema"),
        (Run(0, b'{"decision": "block", ' + _decides(None)[1:]), "schema"),
        # JavaScript's JSON.parse refuses NaN, so Claude Code cannot read this.
        (Run(0, _decides("deny")[:-1] + b', "n": NaN}'), "cannot read"),
        (Run(None, b"", "timed out after 10 s"), "timed out after 10 s"),
        (Run(None, b"", "could not start"), "could not start"),
    ],
)
def test_a_hook_that_fails_lets_the_call_through_and_is_named(
    tmp_path: Path, run: Run, reason: str
) -> None:
    code, out = _test(
        tmp_path, [_settings(tmp_path, [_hook()])], _judge(lambda _t: run), ["delete"]
    )
    assert code == 1
    assert f"PreToolUse[0].hooks[0] failed on {CALLS} of {CALLS} calls:" in out
    assert reason in out
    assert "Claude Code lets a call through when its hook fails" in out


def test_a_failing_hook_with_onfailure_block_stops_everything_so_cannot_tell(
    tmp_path: Path,
) -> None:
    files = [_settings(tmp_path, [_hook(onFailure="block")])]
    code, out = _test(tmp_path, files, _judge(lambda _t: Run(1, b"")), ALL)
    # Every action is stopped, and so is `git status`: a stop proves nothing.
    assert code == 2
    assert "it sets onFailure block, so each one was stopped" in out
    assert "v2.1.295 or later" in out
    assert "a hook also stops `git status`, an ordinary call" in out
    assert "not a pass" in out


def test_onfailure_block_makes_an_exit_1_with_json_a_failure(tmp_path: Path) -> None:
    allow = _decides("allow")

    def answer(text: str) -> Run:
        return Run(0, b"") if _ordinary(text) else Run(1, allow)

    files = [_settings(tmp_path, [_hook(onFailure="block")])]
    code, out = _test(tmp_path, files, _judge(answer), ALL)
    assert code == 0, out


def test_several_lines_of_json_are_plain_text_unless_one_sets_a_field(
    tmp_path: Path,
) -> None:
    files = [_settings(tmp_path, [_hook(onFailure="block")])]
    code, _ = _test(tmp_path, files, _judge(lambda _t: Run(0, b"{}\n{}")), ["delete"])
    assert code == 1  # plain text: decides nothing, and is no failure
    code, out = _test(
        tmp_path, files, _judge(lambda _t: Run(0, b'{}\n{"a": 1}')), ["delete"]
    )
    assert code == 2  # a parse failure, which onFailure block turns into a stop
    assert "printed JSON Claude Code cannot read" in out


def test_json_nested_too_deeply_is_a_parse_failure_not_a_crash(tmp_path: Path) -> None:
    deep = b"{" + b'"a":[' * 100_000 + b"]" * 100_000 + b"}"
    code, out = _test(
        tmp_path,
        [_settings(tmp_path, [_hook()])],
        _judge(lambda _t: Run(0, deep)),
        ["delete"],
    )
    assert code == 1
    assert "printed JSON Claude Code cannot read" in out


def test_output_too_long_to_read_cannot_tell_unless_it_exited_2(tmp_path: Path) -> None:
    files = [_settings(tmp_path, [_hook()])]
    code, out = _test(tmp_path, files, _judge(lambda _t: Run(0, None)), ["delete"])
    assert code == 2
    assert f"could not tell for {DELETES}" in out

    def answer(text: str) -> Run:
        return Run(0, b"") if _ordinary(text) else Run(2, None)

    assert _test(tmp_path, files, _judge(answer), ALL)[0] == 0


def test_of_several_hooks_a_stop_from_any_one_counts(tmp_path: Path) -> None:
    files = [_settings(tmp_path, [_hook("a"), _hook("b")])]

    def run(handler: Handler, payload: bytes) -> Run:
        text = json.loads(payload)["tool_input"].get("command", "")
        stops = handler.argv[-1] == "b" and "rm" in text
        return Run(2 if stops else 0, b"")

    code, out = _test(tmp_path, files, run)
    rm = sum("rm" in s.text for s in ACTIONS[1].spellings)
    assert f"delete      stopped {rm} of {DELETES}. {DELETES - rm} pass" in out
    assert code == 1


def _timed_out(_handler: Handler, _payload: bytes) -> Run:
    return Run(None, b"", "did not answer in time", timed_out=True)


def test_a_hook_stopped_before_its_own_limit_cannot_tell(tmp_path: Path) -> None:
    """Claude Code would have waited out the hook's own 30 s. With onFailure
    block a timeout there would block, but this one is ours, so it is no stop."""
    files = [_settings(tmp_path, [_hook(timeout=30, onFailure="block")])]
    code, out = _test(tmp_path, files, _timed_out, ALL)
    assert code == 2
    assert f"could not tell for {PUSHES}" in out
    assert "did not answer within paveo test's 10 s, before its own time limit" in out
    # It is not said to have stopped anything, nor counted as a failure.
    assert "so each one was stopped" not in out
    assert "failed on" not in out


def test_a_hook_that_times_out_within_its_own_limit_failed(tmp_path: Path) -> None:
    files = [_settings(tmp_path, [_hook(timeout=5, onFailure="block")])]
    code, out = _test(tmp_path, files, _timed_out, ["delete"])
    # Its first call timed out as it would in Claude Code, which blocks it;
    # the rest were not run, so they cannot be told.
    assert code == 2
    assert "failed on 1 of 1 calls: did not answer in time (1)" in out
    assert "it timed out, so it was not run for the calls after that" in out


def test_a_hook_that_hangs_is_run_once_not_for_every_spelling(tmp_path: Path) -> None:
    calls: list[bytes] = []

    def run(handler: Handler, payload: bytes) -> Run:
        calls.append(payload)
        return _timed_out(handler, payload)

    _test(tmp_path, [_settings(tmp_path, [_hook()])], run)
    assert len(calls) == 1


def test_a_timeout_too_large_for_a_float_is_read_without_a_crash(
    tmp_path: Path,
) -> None:
    files = [_settings(tmp_path, [_hook(timeout=10**400)])]
    code, _ = _test(tmp_path, files, _judge(_blocks_all_but_ordinary), ALL)
    assert code == 0


@pytest.mark.parametrize("timeout", [-1, 0, True, "30", None, [5]])
def test_a_timeout_that_is_not_a_positive_number_is_not_guessed(
    tmp_path: Path, timeout: object
) -> None:
    """Claude Code's schema decides what such a hook does; paveo test does not."""
    files = [_settings(tmp_path, [_hook(timeout=timeout)])]
    code, out = _test(tmp_path, files, _never, ALL)
    assert code == 2
    assert "its timeout is not a positive number" in out


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_a_settings_file_javascript_cannot_parse_is_not_read(
    tmp_path: Path, constant: str
) -> None:
    """Python reads NaN and Infinity; Claude Code's JSON.parse refuses the file,
    so none of its hooks run."""
    path = tmp_path / "settings.json"
    hook = json.dumps(_hook())[:-1] + f', "timeout": {constant}}}'
    path.write_text(f'{{"hooks": {{"PreToolUse": [{{"hooks": [{hook}]}}]}}}}')
    code, out = _test(tmp_path, [path], _never, ALL)
    assert code == 1
    assert f"could not read it ({constant} is not JSON)" in out
    assert "Not read." in out


def test_defer_beside_an_ask_cannot_tell(tmp_path: Path) -> None:
    """Precedence is deny > defer > ask, and an interactive session ignores a
    defer, so what the call does then is not stated."""
    files = [_settings(tmp_path, [_hook("a"), _hook("b")])]

    def run(handler: Handler, payload: bytes) -> Run:
        tool_input = json.loads(payload)["tool_input"]
        if _ordinary(tool_input.get("command") or tool_input["file_path"]):
            return Run(0, b"")
        chosen = "ask" if handler.argv[-1] == "a" else "defer"
        return Run(0, _decides(chosen))

    code, out = _test(tmp_path, files, run, ALL)
    assert code == 2
    assert f"could not tell for {PUSHES}" in out


def test_an_ask_beside_a_hook_not_run_cannot_tell(tmp_path: Path) -> None:
    """The hook not run might defer, which outranks an ask."""
    files = [_settings(tmp_path, [_hook(), _hook("other", **{"if": "Bash(*)"})])]

    def answer(text: str) -> Run:
        return Run(0, b"" if _ordinary(text) else _decides("ask"))

    code, out = _test(tmp_path, files, _judge(answer), ALL)
    assert code == 2
    assert f"could not tell for {PUSHES}" in out


def test_an_ask_with_plugins_enabled_cannot_tell(tmp_path: Path) -> None:
    files = [_settings(tmp_path, [_hook()], enabledPlugins={"x@y": True})]

    def answer(text: str) -> Run:
        return Run(0, b"" if _ordinary(text) else _decides("ask"))

    code, _ = _test(tmp_path, files, _judge(answer), ALL)
    assert code == 2
    # A block is a block whatever a plugin does: deny outranks everything.
    code, _ = _test(tmp_path, files, _judge(_blocks_all_but_ordinary), ALL)
    assert code == 0


def test_shell_form_runs_sh_unless_the_hook_asks_for_bash(tmp_path: Path) -> None:
    seen: list[tuple[str, ...]] = []

    def run(handler: Handler, _payload: bytes) -> Run:
        seen.append(handler.argv)
        return Run(0, b"")

    files = [_settings(tmp_path, [_hook("one"), _hook("two", **{"shell": "bash"})])]
    _test(tmp_path, files, run)
    assert ("/bin/sh", "-c", "one") in seen
    assert ("bash", "-c", "two") in seen
    for entry in (
        _hook(**{"shell": "powershell"}),
        _hook(**{"args": None, "shell": "powershell"}),
    ):
        files = [_settings(tmp_path, [entry])]
        _, out = _test(tmp_path, files, _never)
        assert "it asks for PowerShell" in out


def test_exec_form_runs_the_program_with_its_args_and_no_shell(tmp_path: Path) -> None:
    seen: list[tuple[str, ...]] = []

    def run(handler: Handler, _payload: bytes) -> Run:
        seen.append(handler.argv)
        return Run(0, b"")

    hook = _hook("node", args=["${CLAUDE_PROJECT_DIR}/check.js", "$HOME; rm"])
    _test(tmp_path, [_settings(tmp_path, [hook])], run)
    assert seen[0] == ("node", f"{tmp_path}/check.js", "$HOME; rm")


def test_managed_settings_on_the_machine_make_the_run_unsure(tmp_path: Path) -> None:
    """They can switch every hook here off (allowManagedHooksOnly,
    disableAllHooks), and they are not read, so a pass cannot stand."""
    managed = tmp_path / "managed-settings.json"
    managed.write_text("{}")
    out = io.StringIO()
    code = command(
        [_settings(tmp_path, [_hook()])],
        project=tmp_path,
        expect=ALL,
        out=out,
        run=_judge(_blocks_all_but_ordinary),
        managed=[tmp_path / "absent.json", managed],
    )
    assert code == 2
    assert "holds managed settings, which are not read yet" in out.getvalue()
    assert "absent.json" not in out.getvalue()


def test_a_managed_folder_that_cannot_be_read_is_not_absent(tmp_path: Path) -> None:
    """Path.exists says False for a file in a folder it cannot read (Rule 3)."""
    locked = tmp_path / "locked"
    locked.mkdir()
    (locked / "managed-settings.json").write_text("{}")
    locked.chmod(0)
    try:
        out = io.StringIO()
        code = command(
            [_settings(tmp_path, [_hook()])],
            project=tmp_path,
            expect=ALL,
            out=out,
            run=_judge(_blocks_all_but_ordinary),
            managed=[locked / "managed-settings.json"],
        )
    finally:
        locked.chmod(0o700)
    assert code == 2
    assert "holds managed settings" in out.getvalue()


def test_each_user_profile_is_looked_for_and_an_unlisted_folder_is_kept(
    tmp_path: Path,
) -> None:
    from paveo._guard_test import user_profiles  # noqa: PLC0415

    name = "com.anthropic.claudecode.plist"
    (tmp_path / "faiz").mkdir()
    (tmp_path / "other").mkdir()
    assert sorted(user_profiles(tmp_path, name)) == [
        tmp_path / "faiz" / name,
        tmp_path / "other" / name,
    ]
    assert user_profiles(tmp_path / "absent", name) == []
    tmp_path.chmod(0)
    try:
        assert user_profiles(tmp_path, name) == [tmp_path]
    finally:
        tmp_path.chmod(0o700)


def test_plugins_all_switched_off_do_not_make_an_ask_unsure(tmp_path: Path) -> None:
    project = _settings(tmp_path, [_hook()], enabledPlugins={"x@y": True})
    local = tmp_path / ".claude" / "settings.local.json"
    local.write_text(json.dumps({"enabledPlugins": {"x@y": False}}))

    def answer(text: str) -> Run:
        return Run(0, b"" if _ordinary(text) else _decides("ask"))

    code, out = _test(tmp_path, [project, local], _judge(answer), ALL)
    assert code == 0, out
    assert "hooks from plugins" not in out


def test_an_ask_on_ordinary_calls_is_named_even_with_plugins(tmp_path: Path) -> None:
    files = [_settings(tmp_path, [_hook()], enabledPlugins={"x@y": True})]
    _, out = _test(tmp_path, files, _judge(lambda _t: Run(0, _decides("ask"))), ALL)
    assert "a hook also stops `git status`, an ordinary call" in out


@pytest.mark.parametrize("mode", ["bypassPermissions", "auto"])
def test_an_ask_is_no_stop_where_no_person_may_be_asked(
    tmp_path: Path, mode: str
) -> None:
    files = [_settings(tmp_path, [_hook()], permissions={"defaultMode": mode})]
    modes: list[str] = []

    def run(_handler: Handler, payload: bytes) -> Run:
        call = json.loads(payload)
        modes.append(call["permission_mode"])
        text = call["tool_input"].get("command") or call["tool_input"]["file_path"]
        return Run(0, b"" if _ordinary(text) else _decides("ask"))

    code, out = _test(tmp_path, files, run, ALL)
    assert code == 2
    assert f"permissions.defaultMode is {mode}" in out
    # The hook is told the mode, as Claude Code tells it.
    assert set(modes) == {mode}


@pytest.mark.parametrize(
    ("mode", "sent"), [("dontAsk", "dontAsk"), ("manual", "default")]
)
def test_an_ask_stops_in_dont_ask_mode_and_manual_arrives_as_default(
    tmp_path: Path, mode: str, sent: str
) -> None:
    """dontAsk "auto-denies every call that would otherwise prompt", so an ask
    is refused; and the mode labelled Manual reaches a hook as "default"."""
    files = [_settings(tmp_path, [_hook()], permissions={"defaultMode": mode})]
    modes: list[str] = []

    def run(_handler: Handler, payload: bytes) -> Run:
        call = json.loads(payload)
        modes.append(call["permission_mode"])
        text = call["tool_input"].get("command") or call["tool_input"]["file_path"]
        return Run(0, b"" if _ordinary(text) else _decides("ask"))

    code, out = _test(tmp_path, files, run, ALL)
    assert code == 0, out
    assert set(modes) == {sent}
    # The session's own launch flags can still change it, and the run says so.
    assert "--permission-mode bypassPermissions or in auto mode asks no one" in out


def test_an_ask_note_is_given_only_when_an_ask_was_counted(tmp_path: Path) -> None:
    files = [_settings(tmp_path, [_hook()])]
    _, out = _test(tmp_path, files, _judge(_blocks_all_but_ordinary), ALL)
    assert "asks no one" not in out


def test_the_user_settings_follow_claude_config_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from paveo._harnesses import CLAUDE_CODE  # noqa: PLC0415
    from paveo.cli import _claude_home, _global_hook_files  # noqa: PLC0415

    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "work"))
    assert _global_hook_files(CLAUDE_CODE) == [tmp_path / "work" / "settings.json"]
    # replay and learn read the same home's sessions.
    assert _claude_home() / "projects" == tmp_path / "work" / "projects"
    monkeypatch.delenv("CLAUDE_CONFIG_DIR")
    assert _global_hook_files(CLAUDE_CODE) == [
        Path.home() / ".claude" / "settings.json"
    ]


def test_plugins_are_said_not_to_be_run(tmp_path: Path) -> None:
    files = [_settings(tmp_path, [_hook()], enabledPlugins={"paveo@paveo": True})]
    _, out = _test(tmp_path, files, _judge(_blocks_all_but_ordinary))
    assert "hooks from plugins are not run yet" in out


def test_a_read_that_gets_through_is_shown_as_a_read(tmp_path: Path) -> None:
    def answer(text: str) -> Run:
        return Run(2 if text == "cat .env" else 0, b"")

    _, out = _test(tmp_path, [_settings(tmp_path, [_hook()])], _judge(answer))
    assert "                Read .env\n" in out


# --------------------------------------------------------------------------
# Which hooks reach a call
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("matcher", "reaches_bash"),
    [
        ("Bash", True),
        ("Read", False),
        ("Write, Edit | Bash", True),
        ("bash", False),  # exact names are case-sensitive
        ("^Ba", True),  # a regex, unanchored
        ("Ba", False),  # names only: an exact name, not a part of one
        ("mcp__.*", False),
        ("*", True),
        ("", True),
        (None, True),
    ],
)
def test_a_hook_is_run_only_for_the_tools_its_matcher_reaches(
    tmp_path: Path, matcher: object, reaches_bash: bool
) -> None:
    seen: list[str] = []

    def run(_handler: Handler, payload: bytes) -> Run:
        seen.append(json.loads(payload)["tool_name"])
        return Run(0, b"")

    _test(tmp_path, [_settings(tmp_path, [_hook()], matcher=matcher)], run)
    assert ("Bash" in seen) is reaches_bash


@pytest.mark.parametrize(
    "matcher",
    [
        r"Bash\++",
        r"mcp__a\{,b|Bash",
        r"Bash|Foo\\Unit",
        r"Bash|x\\a",
        # An escaped paren before ?P is a literal paren made optional, in both.
        r"Bash|\(?P",
    ],
)
def test_escaped_literals_are_read_as_javascript_reads_them(
    tmp_path: Path, matcher: str
) -> None:
    files = [_settings(tmp_path, [_hook()], matcher=matcher)]
    _, out = _test(tmp_path, files, _judge(_blocks_all_but_ordinary))
    assert "is not run" not in out


def test_a_matcher_that_cannot_be_read_is_not_guessed(tmp_path: Path) -> None:
    # Unbalanced; not text; nested past the regex parser's limit; and three that
    # Python reads but JavaScript, which Claude Code uses, does not or differently.
    nested = "(" * 5000 + ")" * 5000
    python_only = (
        "(?i)bash",
        "(?P<t>Bash)",
        r"\ABash\Z",
        "(?>Bash)",
        "Bas{,1}h",
        "Bas++h",
        "(B)?(?(1)ash|x)",
        "(?-i:Bash)",
        # Python 3.14 reads \z as the end; JavaScript, as the letter z.
        r"^Bash\z",
        r"Bash|x\\(?i:y)",
        r"Bash|x\\\A",
        r"Bash|x\\++",
        r"Bash|x\\\\\\\U00000061",
        r"B\N{LATIN SMALL LETTER A}sh",
        r"B\U00000061sh",
        r"\aBash",
    )
    # A list or an object is no text, and must not crash the reader.
    for matcher in ("Bash(", 7, ["Bash"], {}, nested, *python_only):
        files = [_settings(tmp_path, [_hook()], matcher=matcher)]
        code, out = _test(tmp_path, files, _judge(lambda _t: Run(2, b"")), ALL)
        assert code == 2
        assert "is not run: its matcher" in out


def test_the_call_is_what_claude_code_sends(tmp_path: Path) -> None:
    payloads: list[dict[str, object]] = []

    def run(_handler: Handler, payload: bytes) -> Run:
        payloads.append(json.loads(payload))
        return Run(0, b"")

    _test(tmp_path, [_settings(tmp_path, [_hook()])], run)
    first = payloads[0]
    assert first["hook_event_name"] == "PreToolUse"
    assert first["tool_name"] == "Bash"
    assert first["tool_input"] == {"command": f"git push --force {PROBE} main"}
    assert first["cwd"] == str(tmp_path)
    reads = [p for p in payloads if p["tool_name"] == "Read"]
    # Claude Code always sends Read an absolute path.
    assert {"file_path": str(tmp_path / ".env")} in [p["tool_input"] for p in reads]


# --------------------------------------------------------------------------
# What it does not run, and says so
# --------------------------------------------------------------------------


def _never(_handler: Handler, _payload: bytes) -> Run:
    raise AssertionError("this hook must not be run")


@pytest.mark.parametrize(
    ("entry", "said"),
    [
        (
            _hook(**{"if": "Bash(git *)"}),
            "it has an `if` filter, which is not read yet",
        ),
        ({"type": "http", "url": "https://example.invalid"}, "it is a http hook"),
        ({"type": "prompt", "prompt": "is this safe?"}, "it is a prompt hook"),
    ],
)
def test_a_hook_it_cannot_run_as_claude_code_would_is_not_run(
    tmp_path: Path, entry: dict[str, object], said: str
) -> None:
    code, out = _test(tmp_path, [_settings(tmp_path, [entry])], _never)
    assert code == 2
    assert said in out
    assert "could not tell for" in out


def test_a_hook_not_run_does_not_hide_a_stop_from_one_that_is(tmp_path: Path) -> None:
    files = [_settings(tmp_path, [{"type": "http", "url": "x"}, _hook()])]
    code, out = _test(tmp_path, files, _judge(_blocks_all_but_ordinary), ALL)
    # Every spelling is stopped by the command hook, which is known to let the
    # ordinary calls through, so the hook not run changes nothing.
    assert f"stopped {PUSHES} of {PUSHES}" in out
    assert "is not run: it is a http hook" in out
    assert code == 0


def test_a_background_hook_stops_nothing_and_is_not_run(tmp_path: Path) -> None:
    files = [_settings(tmp_path, [_hook(**{"async": True})])]
    code, out = _test(tmp_path, files, _never, ["delete"])
    assert code == 1
    assert "runs in the background: it stops nothing" in out


def test_disable_all_hooks_switches_every_hook_off(tmp_path: Path) -> None:
    project = _settings(tmp_path, [_hook()])
    local = tmp_path / ".claude" / "settings.local.json"
    local.write_text(json.dumps({"disableAllHooks": True}))
    code, out = _test(tmp_path, [project, local], _never, ["delete"])
    assert code == 1
    assert "sets disableAllHooks, so Claude Code runs none of its hooks" in out
    # A file of higher precedence that sets it false turns them back on.
    later = tmp_path / "later.json"
    later.write_text(json.dumps({"disableAllHooks": False}))
    code, _ = _test(
        tmp_path, [project, local, later], _judge(_blocks_all_but_ordinary), ALL
    )
    assert code == 0


def test_permission_rules_are_said_to_be_untested(tmp_path: Path) -> None:
    files = [
        _settings(
            tmp_path, [_hook()], permissions={"deny": ["Bash(git push --force:*)"]}
        )
    ]
    _, out = _test(tmp_path, files, _judge(lambda _t: Run(0, b"")))
    assert "permission rules are not tested yet" in out


def test_a_handler_in_two_files_runs_once(tmp_path: Path) -> None:
    calls: list[bytes] = []

    def run(_handler: Handler, payload: bytes) -> Run:
        calls.append(payload)
        return Run(0, b"")

    one = _settings(tmp_path, [_hook()])
    two = _settings(tmp_path, [_hook()], name="settings.local.json")
    _, out = _test(tmp_path, [one, two, one], run)
    assert "1 hook," in out
    assert len(calls) == sum(len(a.spellings) for a in ACTIONS) + len(ORDINARY)


# --------------------------------------------------------------------------
# Settings it cannot read
# --------------------------------------------------------------------------


def test_no_settings_fails_even_with_nothing_expected(tmp_path: Path) -> None:
    """A CI job without --expect must still go red the day the settings move
    or every hook is switched off: a test of guards that finds none fails."""
    code, out = _test(tmp_path, [tmp_path / ".claude" / "settings.json"], _never)
    assert code == 1
    assert "no Claude Code settings file found" in out
    assert f"delete      not guarded: no hook stops any of {DELETES} spellings" in out
    assert "no hook stops any of these actions" in out


def test_every_hook_switched_off_fails_without_expect(tmp_path: Path) -> None:
    files = [_settings(tmp_path, [_hook()], disableAllHooks=True)]
    code, _ = _test(tmp_path, files, _never)
    assert code == 1


def test_a_guard_of_one_action_alone_still_passes_without_expect(
    tmp_path: Path,
) -> None:
    """The other side: a team that guards pushes only is not failed for rm."""
    push = {s.text for s in ACTIONS[0].spellings}
    files = [_settings(tmp_path, [_hook()])]
    code, out = _test(
        tmp_path, files, _judge(lambda text: Run(2 if text in push else 0, b""))
    )
    assert code == 0, out


def test_no_settings_with_an_expected_action_fails(tmp_path: Path) -> None:
    code, _ = _test(tmp_path, [tmp_path / "missing.json"], _never, ["force-push"])
    assert code == 1


@pytest.mark.parametrize(
    "text",
    [
        "{not json",
        "[]",
        json.dumps({"hooks": []}),
        json.dumps({"hooks": {"PreToolUse": {}}}),
        json.dumps({"hooks": {"PreToolUse": [{"matcher": "Bash"}]}}),
        json.dumps({"hooks": {"PreToolUse": [{"hooks": [7]}]}}),
        json.dumps({"hooks": {"PreToolUse": [{"hooks": [{"type": "command"}]}]}}),
        json.dumps({"hooks": {"PreToolUse": [{"hooks": [_hook(args="not-a-list")]}]}}),
        json.dumps({"hooks": {"PreToolUse": [{"hooks": [_hook(command=7)]}]}}),
    ],
)
def test_a_settings_file_that_cannot_be_read_is_never_a_pass(
    tmp_path: Path, text: str
) -> None:
    path = tmp_path / "settings.json"
    path.write_text(text)
    code, out = _test(tmp_path, [path], _judge(_blocks_all_but_ordinary))
    assert code == 2
    assert "Not read." in out


@pytest.mark.parametrize(
    ("encoding", "reason"),
    [
        ("utf-16", "not UTF-8"),
        ("utf-32", "not UTF-8"),
        ("utf-8-sig", "it starts with a byte-order mark"),
    ],
)
def test_a_settings_file_claude_code_cannot_read_as_utf8_is_not_read(
    tmp_path: Path, encoding: str, reason: str
) -> None:
    """Python's JSON reader detects these on its own; Claude Code rejects the
    file whole, so the hooks in it never run and must not be counted."""
    path = tmp_path / "settings.json"
    hooks = {"hooks": {"PreToolUse": [{"matcher": "Bash", "hooks": [_hook()]}]}}
    path.write_text(json.dumps(hooks), encoding=encoding)
    code, out = _test(tmp_path, [path], _never, ALL)
    assert code == 1
    assert f"could not read it ({reason})" in out


@pytest.mark.parametrize(
    "extra",
    [
        {"env": {"CI": True}},
        {"env": "CI=1"},
        {"disableAllHooks": "true"},
        {"permissions": ["deny"]},
        {"enabledPlugins": ["x@y"]},
        {"enabledPlugins": {"x@y": "yes"}},
        {"permissions": {"deny": "Bash(rm:*)"}},
        {"permissions": {"defaultMode": 5}},
        {"permissions": {"defaultMode": "yolo"}},
        {"permissions": {"defaultMode": []}},
        {"permissions": {"defaultMode": {}}},
        {"hooks": {"PreToolUse": [], "PostToolUse": "x"}},
    ],
)
def test_a_settings_file_of_the_wrong_shape_is_rejected_whole(
    tmp_path: Path, extra: dict[str, object]
) -> None:
    files = [_settings(tmp_path, [_hook()], **extra)]
    code, out = _test(tmp_path, files, _never, ALL)
    assert code == 1
    assert "Claude Code rejects such a file whole" in out


def test_one_malformed_permission_rule_does_not_reject_the_file(
    tmp_path: Path,
) -> None:
    """Claude Code skips such a rule "while the rest of the file applies", so
    the file's hooks run and are tested."""
    files = [_settings(tmp_path, [_hook()], permissions={"deny": [1, "Bash(rm:*)"]})]
    code, out = _test(tmp_path, files, _judge(_blocks_all_but_ordinary), ALL)
    assert code == 0, out
    assert "Not read" not in out


def test_a_settings_path_that_is_a_fifo_is_unread_and_never_hangs(
    tmp_path: Path,
) -> None:
    """A plain open of a FIFO waits for a writer forever; the reader opens it
    without blocking and refuses anything but a regular file."""
    fifo = tmp_path / "settings.json"
    os.mkfifo(fifo)
    started = time.monotonic()
    code, out = _test(tmp_path, [fifo], _never)
    assert time.monotonic() - started < 5
    assert code == 2
    assert "it is not a regular file" in out


def test_a_named_file_that_is_not_there_is_unread(tmp_path: Path) -> None:
    out = io.StringIO()
    code = command(
        [tmp_path / "nope.json"],
        project=tmp_path,
        expect=[],
        out=out,
        run=_never,
        named=True,
    )
    assert code == 2
    assert "no such file. Not read." in out.getvalue()


def test_it_never_prints_a_hook_command_or_its_output(tmp_path: Path) -> None:
    files = [_settings(tmp_path, [_hook(f"guard --token {SENTINEL}")])]
    code, out = _test(
        tmp_path, files, _judge(lambda _t: Run(1, f"{SENTINEL}\n".encode())), ["delete"]
    )
    assert code == 1
    assert SENTINEL not in out


def test_control_characters_in_a_path_are_escaped(tmp_path: Path) -> None:
    path = tmp_path / "evil\x1b[31m.json"
    path.write_text("{not json")
    _, out = _test(tmp_path, [path], _never)
    assert "\x1b" not in out


# --------------------------------------------------------------------------
# The spellings themselves
# --------------------------------------------------------------------------


def test_every_destructive_spelling_aims_at_nothing_that_exists() -> None:
    """A hook should never run what it judges; for one that does, each
    spelling must do no harm."""
    for action in ACTIONS:
        if action.name == "secrets":
            assert all(".env" in s.text for s in action.spellings)
            continue
        for spelling in action.spellings:
            assert spelling.tool == "Bash"
            assert PROBE in spelling.text, spelling
            assert not Path(f"/{PROBE}").exists()


def test_the_readme_counts_the_spellings_there_are() -> None:
    readme = (Path(SOURCE).parent / "README.md").read_text()
    total = sum(len(action.spellings) for action in ACTIONS)
    assert len(ACTIONS) == 5
    assert f"{total} spellings of five dangerous actions" in readme
    assert f"{total} in all" in readme
    # Every spelling the README quotes is one that is sent, so none it names
    # aims at a real remote.
    assert "origin +main" not in readme
    for action in ACTIONS:
        assert f"`{action.name}`" in readme


def test_action_names_are_unique_and_the_plain_spelling_is_first() -> None:
    assert len(set(ALL)) == len(ALL)
    plain = [action.spellings[0].text for action in ACTIONS]
    assert plain[:3] == [
        f"git push --force {PROBE} main",
        f"rm -rf /{PROBE}",
        f"git reset --hard {PROBE}",
    ]


# --------------------------------------------------------------------------
# The runner, on real processes
# --------------------------------------------------------------------------


def _script(tmp_path: Path, body: str) -> Handler:
    path = tmp_path / "hook.sh"
    path.write_text(f"#!/bin/sh\n{body}\n")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return Handler("test", lambda _t: True, (str(path),), 2.0, blocks_on_failure=False)


def _run(tmp_path: Path, handler: Handler) -> Run:
    return run_handler(
        handler, b'{"tool_input": {}}', project=tmp_path, env=dict(os.environ)
    )


def test_the_runner_reads_the_exit_code_and_what_it_printed(tmp_path: Path) -> None:
    run = _run(tmp_path, _script(tmp_path, 'cat >/dev/null; echo "{}"; exit 2'))
    assert run == Run(2, b"{}\n")


def test_the_runner_hands_the_call_on_stdin(tmp_path: Path) -> None:
    run = _run(tmp_path, _script(tmp_path, "cat"))
    assert run == Run(0, b'{"tool_input": {}}')


def test_a_hook_that_never_reads_its_input_still_answers(tmp_path: Path) -> None:
    assert _run(tmp_path, _script(tmp_path, "exit 2")) == Run(2, b"")


def test_the_deadline_is_read_from_the_injected_clock(tmp_path: Path) -> None:
    """Rule 14: a clock that jumps past the deadline times the hook out at
    once, with no real wait."""
    ticks = iter([0.0, 0.0, 99.0, 99.0, 99.0])
    started = time.monotonic()
    run = run_handler(
        _script(tmp_path, "sleep 30"),
        b"{}",
        project=tmp_path,
        env=dict(os.environ),
        clock=lambda: next(ticks, 99.0),
        pause=lambda _s: None,
    )
    assert run.timed_out
    assert time.monotonic() - started < 2


def test_a_hook_that_hangs_times_out_and_everything_it_started_ends(
    tmp_path: Path,
) -> None:
    pids = tmp_path / "pids"
    handler = _script(tmp_path, f"sleep 30 & echo $$ $! > {pids}; wait")
    # Long enough for a loaded machine to start the shell and write the file.
    handler = Handler(
        "test", handler.reaches, handler.argv, 2.0, blocks_on_failure=False
    )
    started = time.monotonic()
    run = _run(tmp_path, handler)
    assert time.monotonic() - started < 5
    assert run == Run(None, b"", "did not answer in time", timed_out=True)
    for pid in map(int, pids.read_text().split()):
        assert _gone(pid)


def test_a_hook_the_system_will_not_let_us_stop_is_reported_not_a_crash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A setuid program in the hook's group makes killpg fail with EPERM: in
    0.1.9 that was a traceback, and the wait after it could last forever."""
    refused: list[int] = []

    def refuse(pid: int, _signal: int) -> None:
        refused.append(pid)
        raise PermissionError(1, "Operation not permitted")

    monkeypatch.setattr(os, "killpg", refuse)
    ticks = iter([0.0, 0.0, 99.0, 99.0, 99.0])
    handler = _script(tmp_path, "sleep 2")
    started = time.monotonic()
    run = run_handler(
        handler,
        b"{}",
        project=tmp_path,
        env=dict(os.environ),
        clock=lambda: next(ticks, 99.0),
        pause=lambda _s: None,
    )
    assert refused
    assert run.timed_out
    assert "could not be stopped" in (run.failure or "")
    assert time.monotonic() - started < 2  # not waited on
    monkeypatch.undo()  # the real killpg, so the test leaves nothing running
    os.killpg(refused[0], signal.SIGKILL)


def test_a_capped_hook_left_running_is_said_beside_its_cut_short_line() -> None:
    """Cut short by paveo test's own limit, and the system would not let it be
    stopped: the report says both, and never claims what Claude Code would do
    with a call it never saw answered (the third review of 0.1.10)."""
    handler = Handler(
        "settings.json: PreToolUse[0].hooks[0]",
        lambda _t: True,
        ("hook",),
        10.0,
        blocks_on_failure=False,
        capped=True,
    )
    tally = _Tally()
    tally.calls[handler.where] = 1
    tally.timed_out.add(handler.where)
    tally.cut_short.add(handler.where)
    tally.left_running.add(handler.where)
    (note,) = _failure_notes([handler], tally, title="Claude Code")
    assert "cannot be told" in note
    assert "could not be stopped" in note
    assert "lets a call through" not in note


def _gone(pid: int) -> bool:
    """Whether ``pid`` has ended and been reaped, within a second: an orphan
    killed with its group is reaped by init, not by us."""
    deadline = time.monotonic() + 1
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        time.sleep(0.02)
    return False


def test_something_left_running_does_not_hold_the_answer_open(tmp_path: Path) -> None:
    """Claude Code finishes reading shortly after the hook exits, even when a
    process it started still holds its output."""
    started = time.monotonic()
    run = _run(tmp_path, _script(tmp_path, "echo '{}'; sleep 5 & exit 2"))
    assert time.monotonic() - started < 3
    assert run == Run(2, b"{}\n")


def test_a_hook_that_cannot_start_says_so(tmp_path: Path) -> None:
    handler = Handler(
        "test",
        lambda _t: True,
        (str(tmp_path / "missing"),),
        2.0,
        blocks_on_failure=False,
    )
    assert _run(tmp_path, handler) == Run(None, b"", "could not start")


def test_a_nul_byte_in_a_command_could_not_start_and_is_no_crash(
    tmp_path: Path,
) -> None:
    handler = Handler(
        "test",
        lambda _t: True,
        ("/bin/sh", "-c", "exit 0\x00"),
        2.0,
        blocks_on_failure=False,
    )
    assert _run(tmp_path, handler) == Run(None, b"", "could not start")


def test_a_shell_hook_that_cannot_find_its_script_exits_127(tmp_path: Path) -> None:
    handler = Handler(
        "test",
        lambda _t: True,
        ("/bin/sh", "-c", str(tmp_path / "missing.sh")),
        2.0,
        blocks_on_failure=False,
    )
    assert _run(tmp_path, handler).exit == 127


def test_a_flood_of_output_is_not_held_in_memory(tmp_path: Path) -> None:
    run = _run(tmp_path, _script(tmp_path, "head -c 5000000 /dev/zero; exit 0"))
    assert run == Run(0, None)


def test_a_hook_killed_by_a_signal_is_a_failure(tmp_path: Path) -> None:
    run = _run(tmp_path, _script(tmp_path, "kill -9 $$"))
    assert run.exit is not None
    assert run.exit < 0


def test_the_guard_never_loads_the_code_that_runs_hooks() -> None:
    """The guard runs on every call an agent makes; what starts processes is
    loaded only for ``paveo test`` (THREAT_MODEL T13)."""
    import subprocess  # noqa: PLC0415

    script = "import sys, paveo.cli\nprint('paveo._guard_test' in sys.modules)\n"
    result = subprocess.run(  # noqa: S603 - our own interpreter and our own script
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=True,
        env={"PYTHONPATH": SOURCE},
    )
    assert result.stdout.strip() == "False"


# --------------------------------------------------------------------------
# End to end
# --------------------------------------------------------------------------


def _main(project: Path, home: Path, *argv: str) -> tuple[int, str]:
    """``paveo test`` in its own process, from ``project``, as a CI job runs it."""
    import subprocess  # noqa: PLC0415

    done = subprocess.run(  # noqa: S603 - this interpreter, this package
        [sys.executable, "-m", "paveo.cli", "test", *argv],
        cwd=project,
        env={**os.environ, "PYTHONPATH": SOURCE, "HOME": str(home)},
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    return done.returncode, done.stdout


def test_a_guard_written_the_usual_way_is_walked_around(tmp_path: Path) -> None:
    project, home = tmp_path / "project", tmp_path / "home"
    hook = project / ".claude" / "hooks" / "guard.sh"
    hook.parent.mkdir(parents=True)
    home.mkdir()
    hook.write_text(
        '#!/bin/sh\ninput=$(cat)\ncase "$input" in\n'
        '  *\'"command": "git push --force\'*|*\'"command": "rm -rf\'*)'
        " echo blocked >&2; exit 2;;\nesac\nexit 0\n"
    )
    hook.chmod(0o755)
    _settings(project, [_hook('"$CLAUDE_PROJECT_DIR"/.claude/hooks/guard.sh')])
    record = tmp_path / "results.json"
    code, out = _main(project, home, "--expect", "delete", "--json", str(record))
    assert code == 1, out
    caught = sum(s.text.startswith("git push --force") for s in ACTIONS[0].spellings)
    assert f"force-push  stopped {caught} of {PUSHES}. {PUSHES - caught} pass" in out
    assert "\\rm -rf /paveo-test-probe" in out
    assert "hard-reset  not guarded" in out
    # The same results for a machine, with the list's version; never the hook's
    # own output, which can carry a token.
    results = json.loads(record.read_text())
    assert (results["exit"], results["bypass_list"]) == (1, VERSION)
    assert results["ending"] == "these get past the hooks: force-push, delete."
    push, delete = results["actions"][:2]
    assert (push["stopped"], push["spellings"]) == (caught, PUSHES)
    assert "\\rm -rf /paveo-test-probe" in delete["passing"]
    assert delete["expected"] is True
    assert "blocked" not in record.read_text()


def test_an_action_the_list_does_not_have_is_refused(tmp_path: Path) -> None:
    code, _ = _main(tmp_path, tmp_path, "--expect", "publish")
    assert code == 2  # argparse's usage error, before any hook is run


def test_our_own_starter_and_the_gaps_the_readme_names(tmp_path: Path) -> None:
    """Paveo's Claude Code starter, through the guard as init installs it. What
    gets past it must be what the README's limits say gets past it."""
    project, home, bin_dir = tmp_path / "project", tmp_path / "home", tmp_path / "bin"
    (project / ".paveo").mkdir(parents=True)
    home.mkdir()
    bin_dir.mkdir()
    starter = Path(SOURCE) / "paveo" / "starters" / "claude-code.json"
    (project / ".paveo" / "policy.json").write_text(starter.read_text())
    shim = bin_dir / "paveo"
    shim.write_text(
        f'#!/bin/sh\nPYTHONPATH="{SOURCE}" exec "{sys.executable}" -m paveo.cli "$@"\n'
    )
    shim.chmod(0o755)
    _settings(
        project,
        [_hook(f'{shim} guard claude-code --dir "$CLAUDE_PROJECT_DIR/.paveo"')],
        matcher="Bash|Write|Edit|NotebookEdit",
        name="settings.local.json",
    )
    code, out = _main(
        project,
        home,
        "--expect",
        "force-push",
        "--expect",
        "hard-reset",
        "--expect",
        "clean",
        "--expect",
        "delete",
    )
    clean = SIZE["clean"]
    assert f"clean       stopped {clean} of {clean}" in out
    for gap in KNOWN_GAPS:
        assert gap in out
    readme = (Path(SOURCE).parent / "README.md").read_text()
    assert 'but the ones "What the guard cannot do" names' in readme
    assert "these get past the hooks: force-push, delete, hard-reset." in out
    # The starter's matcher leaves Read alone, so .env is not guarded by it.
    assert "secrets     not guarded" in out
    assert code == 1
