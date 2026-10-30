"""``paveo test cursor``: a project's Cursor hooks against bypass spellings (D116).

Each of Cursor's rules for a hook's answer is one fast test with a fake runner;
where a hook starts, the shell it runs in and the variables it gets are tested
on real processes.
"""

from __future__ import annotations

import io
import json
import os
import time
from collections.abc import Callable
from pathlib import Path

import pytest

from paveo import _cursor_hooks, _matcher_trial, cli
from paveo._bypasses import ACTIONS
from paveo._guard_test import Handler, Run, command, run_handler

ALL = [action.name for action in ACTIONS]
SHELL_SPELLINGS = {
    action.name: sum(s.tool == "Bash" for s in action.spellings) for action in ACTIONS
}
PUSHES, DELETES = SHELL_SPELLINGS["force-push"], SHELL_SPELLINGS["delete"]
LEFT_OUT = sum(len(a.spellings) for a in ACTIONS) - sum(SHELL_SPELLINGS.values())
CURSOR = _cursor_hooks.agent()
SHELL_EVENT, TOOL_EVENT = "beforeShellExecution", "preToolUse"


def _hooks_json(
    folder: Path,
    hooks: dict[str, list[dict[str, object]]],
    *,
    version: object = 1,
) -> Path:
    path = folder / ".cursor" / "hooks.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    document: dict[str, object] = {"hooks": hooks}
    if version is not None:
        document["version"] = version
    path.write_text(json.dumps(document))
    return path


def _shell_hook(**entry: object) -> dict[str, list[dict[str, object]]]:
    return {SHELL_EVENT: [{"command": "guard", **entry}]}


def _answer(permission: str | None, **fields: object) -> bytes:
    answer = dict(fields)
    if permission is not None:
        answer["permission"] = permission
    return json.dumps(answer).encode()


def _ordinary(text: str) -> bool:
    return text == "git status"


def _judge(
    answer: Callable[[str], Run], events: list[str] | None = None
) -> Callable[[Handler, bytes], Run]:
    """A fake runner: ``answer`` the command in each call, whichever event
    sent it, and record the events seen."""

    def run(handler: Handler, payload: bytes) -> Run:
        call = json.loads(payload)
        assert call["hook_event_name"] == handler.event
        if handler.event == SHELL_EVENT:
            text = call["command"]
        else:
            assert call["tool_name"] == "Shell"  # Cursor's name for its shell tool
            text = call["tool_input"]["command"]
        if events is not None:
            events.append(handler.event)
        return answer(text)

    return run


def _deny(text: str) -> Run:
    return Run(0, _answer("allow" if _ordinary(text) else "deny"))


def _test(
    tmp_path: Path,
    files: list[Path],
    run: Callable[[Handler, bytes], Run],
    expect: list[str] | None = None,
) -> tuple[int, str]:
    out = io.StringIO()
    code = command(
        files, project=tmp_path, expect=expect or [], out=out, run=run, agent=CURSOR
    )
    return code, out.getvalue()


# --------------------------------------------------------------------------
# What counts as stopped, read as Cursor reads it
# --------------------------------------------------------------------------


def test_a_cursor_guard_that_denies_every_spelling_passes(tmp_path: Path) -> None:
    code, out = _test(
        tmp_path, [_hooks_json(tmp_path, _shell_hook())], _judge(_deny), ALL
    )
    assert code == 0, out
    assert f"{sum(SHELL_SPELLINGS.values())} spellings of 5 dangerous actions" in out
    assert f"{LEFT_OUT} spellings that use Claude Code's Read tool are left out" in out
    assert "trusted workspace" in out
    assert "team in Cursor's dashboard" in out
    assert f"stopped {PUSHES} of {PUSHES}" in out


def test_exit_2_blocks_without_any_json(tmp_path: Path) -> None:
    def blocks(text: str) -> Run:
        return Run(0, _answer("allow")) if _ordinary(text) else Run(2, b"")

    code, out = _test(
        tmp_path, [_hooks_json(tmp_path, _shell_hook())], _judge(blocks), ALL
    )
    assert code == 0, out


def test_cursor_s_own_example_matcher_lets_other_spellings_of_rm_through(
    tmp_path: Path,
) -> None:
    """The hooks page's example: a beforeShellExecution hook with the matcher
    ``rm -rf``, tested against the command. Every delete spelled otherwise
    never reaches the hook."""
    files = [_hooks_json(tmp_path, _shell_hook(matcher="rm -rf"))]
    code, out = _test(tmp_path, files, _judge(lambda _text: Run(0, _answer("deny"))))
    assert code == 1, out
    # The README states this count; the list growing changes both.
    assert f"delete      stopped 9 of {DELETES}" in out
    assert DELETES == 19
    assert "rm -r -f" in out
    assert "force-push  not guarded" in out


def test_a_pre_tool_use_hook_gets_the_shell_tool_and_its_matcher_is_the_name(
    tmp_path: Path,
) -> None:
    events: list[str] = []
    hooks = {TOOL_EVENT: [{"command": "guard", "matcher": "Shell"}]}
    code, out = _test(
        tmp_path, [_hooks_json(tmp_path, hooks)], _judge(_deny, events), ALL
    )
    assert code == 0, out
    assert set(events) == {TOOL_EVENT}

    hooks = {TOOL_EVENT: [{"command": "guard", "matcher": "Read|Write"}]}
    code, out = _test(tmp_path, [_hooks_json(tmp_path, hooks)], _judge(_deny))
    assert code == 1
    assert "no hook stops any of these actions" in out


def test_both_events_run_and_one_deny_is_enough(tmp_path: Path) -> None:
    events: list[str] = []
    hooks = {
        SHELL_EVENT: [{"command": "allows"}],
        TOOL_EVENT: [{"command": "denies"}],
    }

    def run(handler: Handler, payload: bytes) -> Run:
        events.append(handler.event)
        if handler.event == SHELL_EVENT:
            return Run(0, _answer("allow"))
        text = json.loads(payload)["tool_input"]["command"]
        return _deny(text)

    code, out = _test(tmp_path, [_hooks_json(tmp_path, hooks)], run, ALL)
    assert code == 0, out
    assert set(events) == {SHELL_EVENT, TOOL_EVENT}
    assert "paveo test: 2 hooks," in out


def test_an_ask_is_never_counted_as_a_stop(tmp_path: Path) -> None:
    """From beforeShellExecution Cursor asks the person, but whether auto-run
    skips the question its page does not say, so the run cannot tell (the
    /code-review of D116); from preToolUse it is not enforced at all."""

    def asks(text: str) -> Run:
        return Run(0, _answer("allow" if _ordinary(text) else "ask"))

    code, out = _test(
        tmp_path, [_hooks_json(tmp_path, _shell_hook())], _judge(asks), ALL
    )
    assert code == 2, out
    assert f"stopped 0 of {PUSHES}, could not tell for {PUSHES}" in out
    assert "is not counted as a stop" in out
    assert "auto-run" in out

    hooks = {TOOL_EVENT: [{"command": "guard"}]}
    code, out = _test(tmp_path, [_hooks_json(tmp_path, hooks)], _judge(asks))
    assert code == 1
    assert "no hook stops any of these actions" in out


@pytest.mark.parametrize(
    ("answer", "why"),
    [
        (Run(1, b""), "exit 1"),
        (Run(0, b""), "printed nothing"),
        (Run(None, b"", "could not start"), "could not start"),
    ],
)
def test_a_failing_hook_lets_the_call_through_unless_it_fails_closed(
    tmp_path: Path, answer: Run, why: str
) -> None:
    # Its own limit under paveo test's, so a timeout is Cursor's too.
    files = [_hooks_json(tmp_path, _shell_hook(timeout=5))]
    code, out = _test(tmp_path, files, _judge(lambda _text: answer), ["delete"])
    assert code == 1, out
    assert why in out
    assert "Cursor lets a call through when its hook fails" in out

    files = [_hooks_json(tmp_path, _shell_hook(timeout=5, failClosed=True))]
    _code, out = _test(tmp_path, files, _judge(lambda _text: answer), ["delete"])
    assert f"delete      stopped {DELETES} of {DELETES}" in out
    assert "it sets failClosed, so each one was stopped" in out


@pytest.mark.parametrize(
    "printed",
    [
        b"not json at all",
        b'{"permission": "deny"',  # cut short
        b"[1, 2]",
        b'{"permission": "maybe"}',
        b'{"permission": true}',
        b'{"permission": "allow", "user_message": 3}',
    ],
)
def test_an_answer_cursor_cannot_read_blocks_even_without_fail_closed(
    tmp_path: Path, printed: bytes
) -> None:
    files = [_hooks_json(tmp_path, _shell_hook())]
    code, out = _test(tmp_path, files, _judge(lambda _text: Run(0, printed)), ALL)
    assert f"delete      stopped {DELETES} of {DELETES}" in out
    # It refuses ordinary calls the same way, which the run says.
    assert "an ordinary call" in out
    assert code == 2, out


def test_an_answer_with_no_permission_cannot_be_told(tmp_path: Path) -> None:
    files = [_hooks_json(tmp_path, _shell_hook())]
    answer = Run(0, _answer(None, **{"continue": True}))
    code, out = _test(tmp_path, files, _judge(lambda _text: answer))
    assert code == 2, out
    assert f"could not tell for {DELETES}" in out


def test_extra_fields_in_an_answer_are_allowed(tmp_path: Path) -> None:
    answer = Run(0, _answer("deny", **{"continue": True, "agent_message": "no"}))
    files = [_hooks_json(tmp_path, _shell_hook())]
    _code, out = _test(tmp_path, files, _judge(lambda _text: answer))
    assert f"delete      stopped {DELETES} of {DELETES}" in out


# --------------------------------------------------------------------------
# Reading the files, and what the run says it cannot see
# --------------------------------------------------------------------------


def test_no_cursor_file_says_where_it_looked(tmp_path: Path) -> None:
    code, out = _test(tmp_path, [tmp_path / ".cursor" / "hooks.json"], _judge(_deny))
    assert code == 1
    assert "no Cursor hooks file found. Looked for:" in out


@pytest.mark.parametrize(
    ("text", "why"),
    [
        ("{not json", "could not read it"),
        ("[]", "not a JSON object"),
        ('{"hooks": {}}', "requires version 1"),
        ('{"version": 0, "hooks": {}}', "requires version 1"),
        ('{"version": true, "hooks": {}}', "requires version 1"),
        ('{"version": 2, "hooks": {}}', "requires version 1"),
        ('{"version": 1.0, "hooks": {}}', "requires version 1"),
        ('{"version": 1, "hooks": []}', "not an object of events"),
        ('{"version": 1, "hooks": {"beforeShellExecution": {}}}', "is not a list"),
        ('{"version": 1, "hooks": {"preToolUse": [3]}}', "is not an object"),
        ('{"version": 1, "hooks": {"beforeShellExecution": [{}]}}', "has no command"),
    ],
)
def test_a_file_cursor_may_not_run_is_unread_and_the_run_unsure(
    tmp_path: Path, text: str, why: str
) -> None:
    path = tmp_path / ".cursor" / "hooks.json"
    path.parent.mkdir()
    path.write_text(text)
    code, out = _test(tmp_path, [path], _judge(_deny))
    assert why in out
    assert "Not read." in out
    assert code == 2, out


@pytest.mark.parametrize(
    ("entry", "why"),
    [
        ({"matcher": "(?<name>rm)"}, "matcher is not a pattern"),
        ({"matcher": "(?P<name>rm)"}, "matcher is not a pattern"),
        ({"matcher": "(?i)rm"}, "matcher is not a pattern"),
        ({"matcher": 3}, "matcher is not a pattern"),
        ({"timeout": 0}, "timeout is not a positive number"),
        ({"timeout": "5"}, "timeout is not a positive number"),
        ({"type": "prompt"}, "it is a prompt hook"),
        ({"type": "prompt", "command": None, "prompt": "deny rm"}, "a prompt hook"),
        ({"matcher": "rm.*+"}, "matcher is not a pattern"),
        ({"matcher": "(?>rm)"}, "matcher is not a pattern"),
        ({"matcher": "rm -r{,1}f"}, "matcher is not a pattern"),
        ({"matcher": "[^]x]"}, "matcher is not a pattern"),
    ],
)
def test_a_hook_that_cannot_be_run_as_cursor_would_makes_the_run_unsure(
    tmp_path: Path, entry: dict[str, object], why: str
) -> None:
    files = [_hooks_json(tmp_path, _shell_hook(**entry))]
    code, out = _test(tmp_path, files, _judge(_deny))
    assert why in out
    assert code == 2, out


def test_a_hook_whose_own_limit_is_past_ten_seconds_is_capped(tmp_path: Path) -> None:
    seen: list[Handler] = []

    def run(handler: Handler, _payload: bytes) -> Run:
        seen.append(handler)
        return Run(2, b"")

    _test(tmp_path, [_hooks_json(tmp_path, _shell_hook(timeout=60))], run)
    _test(tmp_path, [_hooks_json(tmp_path, _shell_hook(timeout=3))], run)
    assert (seen[0].timeout, seen[0].capped) == (10.0, True)
    assert (seen[-1].timeout, seen[-1].capped) == (3.0, False)


def test_every_file_runs_and_each_hook_starts_where_cursor_starts_it(
    tmp_path: Path,
) -> None:
    project, home = tmp_path / "repo", tmp_path / "home"
    organisation = tmp_path / "org" / "hooks.json"
    organisation.parent.mkdir()
    organisation.write_text(json.dumps({"version": 1, "hooks": _shell_hook()}))
    _hooks_json(home, _shell_hook())
    _hooks_json(project, _shell_hook())
    seen: dict[str, Path | None] = {}

    def run(handler: Handler, _payload: bytes) -> Run:
        seen[handler.where.split(":")[0]] = handler.cwd
        return Run(2, b"")

    files = _cursor_hooks.files(project, home, organisation)
    out = io.StringIO()
    command(files, project=project, expect=[], out=out, run=run, agent=CURSOR)
    assert "paveo test: 3 hooks," in out.getvalue()
    real = os.path.realpath
    assert seen == {
        real(organisation): Path(real(organisation.parent)),
        real(home / ".cursor" / "hooks.json"): Path(real(home / ".cursor")),
        real(project / ".cursor" / "hooks.json"): project,
    }


# --------------------------------------------------------------------------
# Real processes
# --------------------------------------------------------------------------


def _real(tmp_path: Path) -> Callable[[Handler, bytes], Run]:
    env = {
        "PATH": os.environ["PATH"],
        "HOME": str(tmp_path),
        "CURSOR_PROJECT_DIR": str(tmp_path),
    }

    def run(handler: Handler, payload: bytes) -> Run:
        return run_handler(handler, payload, project=tmp_path, env=env)

    return run


def test_a_real_hook_given_as_a_path_from_the_project_root_blocks(
    tmp_path: Path,
) -> None:
    script = tmp_path / ".cursor" / "hooks" / "guard.sh"
    script.parent.mkdir(parents=True)
    script.write_text(
        "#!/bin/sh\n"
        "input=$(cat)\n"
        'case "$input" in *"git status"*) exit 0;; esac\n'
        '[ "$CURSOR_PROJECT_DIR" = "$(pwd -P)" ] || exit 0\n'
        'printf \'%s\' \'{"permission": "deny", "user_message": "no"}\'\n'
    )
    script.chmod(0o755)
    files = [_hooks_json(tmp_path, _shell_hook(command=".cursor/hooks/guard.sh"))]
    code, out = _test(
        Path(os.path.realpath(tmp_path)),
        files,
        _real(Path(os.path.realpath(tmp_path))),
        ALL,
    )
    assert code == 0, out


def test_a_real_hook_reading_a_variable_cursor_never_sets_stops_nothing(
    tmp_path: Path,
) -> None:
    # Claude Code's old variable, which Cursor does not set either.
    hook = 'case "$CLAUDE_TOOL_INPUT" in *rm*|*push*) exit 2;; esac'
    files = [_hooks_json(tmp_path, _shell_hook(command=hook))]
    code, out = _test(tmp_path, files, _real(tmp_path))
    assert code == 1
    assert "no hook stops any of these actions" in out


def test_the_cli_tests_cursor_hooks_with_cursor_s_variables(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    hook = tmp_path / "hook.sh"
    hook.write_text(
        "input=$(cat)\n"
        'case "$input" in *\'"git status"\'*) exit 0;; esac\n'
        '[ -n "$CURSOR_PROJECT_DIR" ] && [ -n "$CLAUDE_PROJECT_DIR" ] && exit 2\n'
    )
    _hooks_json(tmp_path, _shell_hook(command=f"sh {hook}"))
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("CURSOR_PROJECT_DIR", raising=False)
    code = cli.main(
        ["test", "cursor", "--expect", "delete", "--settings", ".cursor/hooks.json"]
    )
    out = capsys.readouterr().out
    assert "delete      stopped" in out, out
    assert code == 0, out


def test_the_cli_reads_the_organisation_s_the_person_s_and_the_project_s_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[list[Path]] = []

    def capture(files: list[Path], **_kwargs: object) -> int:
        seen.append(files)
        return 0

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setattr("paveo._guard_test.command", capture)
    assert cli.main(["test", "cursor"]) == 0
    assert seen[0] == [
        cli._CURSOR_ORGANISATION,
        tmp_path / "home" / ".cursor" / "hooks.json",
        tmp_path / ".cursor" / "hooks.json",
    ]


def test_a_matcher_that_backtracks_for_minutes_makes_the_run_unsure_not_hang(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each optional repeat doubles the work, so this one takes minutes per
    command; it is tried in a child given a limit, never in paveo itself (the
    second /code-review of D116)."""
    monkeypatch.setattr(_matcher_trial, "TIMEOUT_S", 0.5)
    hooks = {
        SHELL_EVENT: [
            {"command": "slow", "matcher": ".?" * 40 + "Q"},
            {"command": "fine", "matcher": "rm"},
        ]
    }
    started = time.monotonic()
    code, out = _test(tmp_path, [_hooks_json(tmp_path, hooks)], _judge(_deny))
    assert time.monotonic() - started < 10
    assert "could not be tried in time" in out
    assert code == 2, out


def test_a_command_matcher_is_tried_against_the_command(tmp_path: Path) -> None:
    hooks = _shell_hook(matcher=r"git\s+push\s+.*--force")
    code, out = _test(tmp_path, [_hooks_json(tmp_path, hooks)], _judge(_deny))
    assert "force-push  stopped" in out
    assert "not guarded" in out  # every other action
    assert code == 1, out


def test_an_entry_of_a_type_cursor_does_not_know_is_unread(tmp_path: Path) -> None:
    for kind in (None, 5, "agent"):
        files = [_hooks_json(tmp_path, _shell_hook(type=kind))]
        code, out = _test(tmp_path, files, _judge(_deny))
        assert "has a type Cursor does not know" in out
        assert code == 2, out


def test_a_project_file_linked_to_the_person_s_runs_as_both(tmp_path: Path) -> None:
    """Cursor reads the two names as two sources, each started from its own
    folder, so a relative command may find a script in one and not the other."""
    project, home = tmp_path / "repo", tmp_path / "home"
    user = _hooks_json(home, _shell_hook())
    (project / ".cursor").mkdir(parents=True)
    (project / ".cursor" / "hooks.json").symlink_to(user)
    seen: set[Path | None] = set()

    def run(handler: Handler, _payload: bytes) -> Run:
        seen.add(handler.cwd)
        return Run(2, b"")

    files = _cursor_hooks.files(project, home, tmp_path / "none.json")
    out = io.StringIO()
    command(files, project=project, expect=[], out=out, run=run, agent=CURSOR)
    assert seen == {home / ".cursor", project}
    assert "trusted workspace" in out.getvalue()


def test_a_file_with_only_other_events_names_cursor_s_events(tmp_path: Path) -> None:
    hooks = {"afterFileEdit": [{"command": "format"}]}
    code, out = _test(tmp_path, [_hooks_json(tmp_path, hooks)], _judge(_deny))
    assert "no beforeShellExecution or preToolUse command hook found." in out
    assert "PreToolUse" not in out
    assert code == 1


def test_trust_is_said_only_when_the_project_has_hooks(tmp_path: Path) -> None:
    project, home = tmp_path / "repo", tmp_path / "home"
    project.mkdir()
    _hooks_json(home, _shell_hook())
    files = _cursor_hooks.files(project, home, tmp_path / "none.json")
    out = io.StringIO()
    command(files, project=project, expect=[], out=out, run=_judge(_deny), agent=CURSOR)
    assert "paveo test: 1 hook," in out.getvalue()
    assert "trusted workspace" not in out.getvalue()


def test_a_linked_user_file_starts_its_hooks_in_cursor_s_folder(tmp_path: Path) -> None:
    """Cursor starts a person's hooks in ~/.cursor, wherever the file links to
    (/security-review of D116), so a relative command finds the same script."""
    project, home, dotfiles = tmp_path / "repo", tmp_path / "home", tmp_path / "dots"
    project.mkdir()
    real = _hooks_json(dotfiles, _shell_hook())
    (home / ".cursor").mkdir(parents=True)
    (home / ".cursor" / "hooks.json").symlink_to(real)
    seen: list[Path | None] = []

    def run(handler: Handler, _payload: bytes) -> Run:
        seen.append(handler.cwd)
        return Run(2, b"")

    files = _cursor_hooks.files(project, home, tmp_path / "none.json")
    command(files, project=project, expect=[], out=io.StringIO(), run=run, agent=CURSOR)
    assert set(seen) == {home / ".cursor"}


@pytest.mark.parametrize("entry", [{"type": "prompt"}, {"timeout": -1}])
def test_a_hook_not_run_keeps_its_matcher_so_the_run_cannot_tell(
    tmp_path: Path, entry: dict[str, object]
) -> None:
    """Not run for a reason other than its matcher, it still reaches what its
    matcher reaches, so those spellings are could-not-tell, never "not
    guarded" (the security pass on the child-process matcher)."""
    files = [_hooks_json(tmp_path, _shell_hook(matcher="rm", **entry))]
    code, out = _test(tmp_path, files, _judge(_deny))
    assert "delete      stopped 0 of" in out
    assert "could not tell for" in out
    assert "delete      not guarded" not in out
    assert code == 2, out


def test_a_slow_tool_name_matcher_cannot_hang_the_run_either(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Measured at 3.6 s over the five letters of `Shell` with 80 repeats, and
    doubling: tried in a child too (the third /code-review of D116)."""
    monkeypatch.setattr(_matcher_trial, "TIMEOUT_S", 0.5)
    hooks = {TOOL_EVENT: [{"command": "guard", "matcher": "(?:.?){400}Q"}]}
    started = time.monotonic()
    code, out = _test(tmp_path, [_hooks_json(tmp_path, hooks)], _judge(_deny))
    assert time.monotonic() - started < 10
    assert "could not be tried in time" in out
    assert code == 2, out


def test_a_slow_project_matcher_leaves_the_organisation_s_tested(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(_matcher_trial, "TIMEOUT_S", 3.0)
    project = tmp_path / "repo"
    organisation = tmp_path / "org" / "hooks.json"
    organisation.parent.mkdir()
    organisation.write_text(
        json.dumps({"version": 1, "hooks": _shell_hook(matcher="push")})
    )
    _hooks_json(project, _shell_hook(matcher=".?" * 40 + "Q"))
    # Named project first: it is still tried last.
    files = [project / ".cursor" / "hooks.json", organisation]
    out = io.StringIO()
    command(files, project=project, expect=[], out=out, run=_judge(_deny), agent=CURSOR)
    text = out.getvalue()
    assert f"force-push  stopped {PUSHES} of {PUSHES}" in text
    assert "beforeShellExecution[0] is not run: its matcher could not be tried" in text


def test_a_matcher_never_tried_is_could_not_tell_never_not_guarded() -> None:
    """The default is fail closed: a hook whose matcher is not answered
    reaches every call and cannot be told (Rule 5)."""
    found = _cursor_hooks.Found()
    pending: dict[str, _cursor_hooks._Pending] = {}
    handler = Handler(
        "h", lambda _s: True, ("x",), 5.0, blocks_on_failure=False, event=SHELL_EVENT
    )
    found.handlers.append(_cursor_hooks._matched(handler, "rm", pending=pending))
    assert found.handlers[0].untestable == _cursor_hooks._NOT_TRIED

    def refuse(
        wanted: list[tuple[str, str]], _texts: dict[str, list[str]]
    ) -> list[str]:
        return [_matcher_trial.FAILED] * len(wanted)

    _cursor_hooks._try_matchers(found, pending, try_all=refuse)
    (after,) = found.handlers
    assert after.untestable == _matcher_trial.FAILED
    assert after.reaches(ACTIONS[0].spellings[0])


def test_the_child_answers_in_order_and_says_which_python_cannot_read() -> None:
    texts = {SHELL_EVENT: ["rm -rf x", "ls"], TOOL_EVENT: ["Shell"]}
    answers = _matcher_trial.try_all(
        [("rm", SHELL_EVENT), ("(?<name>x)", SHELL_EVENT), ("Sh", TOOL_EVENT)], texts
    )
    assert answers == [
        frozenset({"rm -rf x"}),
        _matcher_trial.UNREAD,
        frozenset({"Shell"}),
    ]


def test_an_ask_to_every_call_is_not_called_a_stop_of_an_ordinary_call(
    tmp_path: Path,
) -> None:
    files = [_hooks_json(tmp_path, _shell_hook())]
    code, out = _test(tmp_path, files, _judge(lambda _t: Run(0, _answer("ask"))))
    assert "an ordinary call" not in out
    assert "is not counted as a stop" in out
    assert code == 2, out


def test_the_ask_note_is_said_when_another_hook_already_made_it_unsure(
    tmp_path: Path,
) -> None:
    hooks = {
        SHELL_EVENT: [
            {"command": "asks"},
            {"type": "prompt", "prompt": "no"},
        ]
    }

    def asks(text: str) -> Run:
        return Run(0, _answer("allow" if _ordinary(text) else "ask"))

    code, out = _test(tmp_path, [_hooks_json(tmp_path, hooks)], _judge(asks))
    assert "is not counted as a stop" in out
    assert code == 2, out
