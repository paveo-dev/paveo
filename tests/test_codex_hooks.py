"""``paveo test codex``: a project's Codex hooks against bypass spellings (D111).

Each of Codex's rules for a hook's answer is one fast test with a fake runner;
the shell Codex runs a hook in, and the stderr it needs before exit 2 blocks,
are tested on real processes.
"""

from __future__ import annotations

import io
import json
import os
import sys
from collections.abc import Callable
from pathlib import Path

import pytest

from paveo import _codex_hooks, cli
from paveo._bypasses import ACTIONS
from paveo._guard_test import Handler, Run, command, run_handler

ALL = [action.name for action in ACTIONS]
SHELL_SPELLINGS = {
    action.name: sum(s.tool == "Bash" for s in action.spellings) for action in ACTIONS
}
PUSHES, DELETES = SHELL_SPELLINGS["force-push"], SHELL_SPELLINGS["delete"]
LEFT_OUT = sum(len(a.spellings) for a in ACTIONS) - sum(SHELL_SPELLINGS.values())
CODEX = _codex_hooks.agent("/bin/sh")


def _hooks_json(
    folder: Path, entries: list[dict[str, object]], *, matcher: object = "Bash"
) -> Path:
    path = folder / ".codex" / "hooks.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    group: dict[str, object] = {"hooks": entries}
    if matcher is not None:
        group["matcher"] = matcher
    path.write_text(json.dumps({"hooks": {"PreToolUse": [group]}}))
    return path


def _hook(cmd: str = "guard", **entry: object) -> dict[str, object]:
    return {"type": "command", "command": cmd, **entry}


def _deny(reason: str | None = "no", **fields: object) -> bytes:
    specific = {"hookEventName": "PreToolUse", "permissionDecision": "deny"}
    if reason is not None:
        specific["permissionDecisionReason"] = reason
    return json.dumps({"hookSpecificOutput": {**specific, **fields}}).encode()


def _judge(answer: Callable[[str], Run]) -> Callable[[Handler, bytes], Run]:
    def run(_handler: Handler, payload: bytes) -> Run:
        call = json.loads(payload)
        assert call["tool_name"] == "Bash"  # Codex has no Read tool
        return answer(call["tool_input"]["command"])

    return run


def _ordinary(text: str) -> bool:
    return text == "git status"


def _blocks(text: str) -> Run:
    """Codex's exit-2 block: the code, and a reason on stderr."""
    return Run(0, b"") if _ordinary(text) else Run(2, b"", wrote_stderr=True)


def _test(
    tmp_path: Path,
    files: list[Path],
    run: Callable[[Handler, bytes], Run],
    expect: list[str] | None = None,
) -> tuple[int, str]:
    out = io.StringIO()
    code = command(
        files, project=tmp_path, expect=expect or [], out=out, run=run, agent=CODEX
    )
    return code, out.getvalue()


# --------------------------------------------------------------------------
# What counts as stopped, read as Codex reads it
# --------------------------------------------------------------------------


def test_a_codex_guard_that_blocks_every_spelling_passes(tmp_path: Path) -> None:
    code, out = _test(
        tmp_path, [_hooks_json(tmp_path, [_hook()])], _judge(_blocks), ALL
    )
    assert code == 0, out
    assert f"{sum(SHELL_SPELLINGS.values())} spellings of 5 dangerous actions" in out
    assert f"{LEFT_OUT} spellings that use Claude Code's Read tool are left out" in out
    assert "once trusted" in out
    assert f"stopped {PUSHES} of {PUSHES}" in out


def test_a_hook_in_config_toml_is_read_like_one_in_hooks_json(tmp_path: Path) -> None:
    config = tmp_path / ".codex" / "config.toml"
    config.parent.mkdir()
    config.write_text(
        '[[hooks.PreToolUse]]\nmatcher = "Bash"\n'
        '[[hooks.PreToolUse.hooks]]\ntype = "command"\ncommand = "guard"\n'
    )
    code, out = _test(tmp_path, [config], _judge(_blocks), ALL)
    assert code == 0, out
    assert "paveo test: 1 hook," in out


def test_both_files_in_a_layer_load_and_neither_replaces_the_other(
    tmp_path: Path,
) -> None:
    hooks = _hooks_json(tmp_path, [_hook("one")])
    config = hooks.parent / "config.toml"
    config.write_text(
        '[[hooks.PreToolUse]]\n[[hooks.PreToolUse.hooks]]\ntype = "command"\n'
        'command = "two"\n'
    )
    _code, out = _test(tmp_path, [hooks, config], _judge(_blocks))
    assert "paveo test: 2 hooks," in out


@pytest.mark.parametrize(
    ("run", "reason"),
    [
        (Run(2, b""), "exit 2 with no reason on stderr"),
        (Run(1, b"", wrote_stderr=True), "exit 1"),
        (Run(0, _deny(None)), "denied with no reason"),
        (Run(0, _deny("  ")), "denied with no reason"),
        (Run(0, b'{"decision": "block"}'), "blocked with no reason"),
        (Run(0, _deny("no", hookEventName="Nope")), "fails Codex's schema"),
        (Run(0, b'{"unexpected": true, ' + _deny()[1:]), "fails Codex's schema"),
        (Run(0, b"{not json"), "printed JSON Codex cannot read"),
        (Run(0, b'["deny"]'), "printed JSON Codex cannot read"),
        (Run(0, b'{"continue": false, ' + _deny()[1:]), "does not support yet"),
        (
            Run(
                0,
                json.dumps(
                    {
                        "hookSpecificOutput": {
                            "hookEventName": "PreToolUse",
                            "permissionDecision": "ask",
                            "permissionDecisionReason": "sure?",
                        }
                    }
                ).encode(),
            ),
            "does not support yet",
        ),
    ],
)
def test_every_answer_codex_treats_as_a_failure_lets_the_call_through(
    tmp_path: Path, run: Run, reason: str
) -> None:
    code, out = _test(
        tmp_path, [_hooks_json(tmp_path, [_hook()])], _judge(lambda _t: run), ["delete"]
    )
    assert code == 1
    assert f"EXPECTED, but no hook stops any of {DELETES} spellings" in out
    assert reason in out
    assert "Codex lets a call through when its hook fails" in out


def test_a_timeout_inside_the_hook_s_own_limit_is_a_failure(
    tmp_path: Path,
) -> None:
    timed_out = Run(None, b"", "did not answer in time", timed_out=True)
    code, out = _test(
        tmp_path,
        [_hooks_json(tmp_path, [_hook(timeout=5)])],
        _judge(lambda _t: timed_out),
        ["delete"],
    )
    # The first call fails, so it lets the call on; a hook that hangs is not
    # run again, so the rest cannot be told, as for Claude Code.
    assert code == 2, out
    assert "did not answer in time (1)" in out
    assert "Codex lets a call through when its hook fails" in out
    assert "it was not run for the calls after that" in out


@pytest.mark.parametrize(
    "stdout",
    [
        _deny("no"),
        _deny("no", hookEventName="PostToolUse"),  # any event Codex knows
        b'{"decision": "block", "reason": "no"}',
        b'{"decision": "block", "reason": "no", "systemMessage": null}',
    ],
)
def test_each_way_codex_blocks_on_exit_0_is_a_stop(
    tmp_path: Path, stdout: bytes
) -> None:
    def answer(text: str) -> Run:
        return Run(0, b"" if _ordinary(text) else stdout)

    code, out = _test(tmp_path, [_hooks_json(tmp_path, [_hook()])], _judge(answer), ALL)
    assert code == 0, out


def test_plain_text_on_exit_0_decides_nothing_and_is_no_failure(tmp_path: Path) -> None:
    code, out = _test(
        tmp_path,
        [_hooks_json(tmp_path, [_hook()])],
        _judge(lambda _t: Run(0, b"Blocked: rm is not allowed")),
        ["delete"],
    )
    assert code == 1
    assert "failed on" not in out


# --------------------------------------------------------------------------
# What Codex runs, and what it does not
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("matcher", "reached"),
    [
        ("Bash", True),
        ("", True),
        ("*", True),
        (None, True),
        ("^Ba", True),  # a regex, matching anywhere
        ("Bash|Read", True),
        ("Ba", False),  # letters only: an exact name, not a prefix
        ("Bash, Read", False),  # a comma makes it a regex, which "Bash" fails
        ("apply_patch", False),
    ],
)
def test_matchers_reach_bash_as_codex_reads_them(
    tmp_path: Path, matcher: object, *, reached: bool
) -> None:
    files = [_hooks_json(tmp_path, [_hook()], matcher=matcher)]
    code, out = _test(tmp_path, files, _judge(_blocks), ["delete"])
    assert (code == 0) is reached, out


@pytest.mark.parametrize(
    "matcher",
    [
        "(?!Read)Bash",
        # Rust reads && inside a class as an intersection; Python as two
        # characters, so `[B&&x]ash` reaches Bash in Python, nothing in Rust.
        "[B&&x]ash",
    ],
)
def test_a_matcher_rust_would_refuse_is_not_run_and_the_run_is_unsure(
    tmp_path: Path, matcher: str
) -> None:
    files = [_hooks_json(tmp_path, [_hook()], matcher=matcher)]
    code, out = _test(tmp_path, files, _judge(_blocks))
    assert code == 2
    assert "its matcher is not a pattern paveo test can read" in out


def test_features_hooks_false_switches_every_hook_off(tmp_path: Path) -> None:
    hooks = _hooks_json(tmp_path, [_hook()])
    home = tmp_path / "home" / "config.toml"
    home.parent.mkdir()
    home.write_text("[features]\nhooks = false\n")
    code, out = _test(tmp_path, [home, hooks], _judge(_blocks), ["delete"])
    assert code == 1
    assert "sets features.hooks = false, so Codex runs no hook" in out


def test_a_hook_switched_off_in_slash_hooks_is_not_run(tmp_path: Path) -> None:
    hooks = _hooks_json(tmp_path, [_hook()])
    home = tmp_path / "home" / "config.toml"
    home.parent.mkdir()
    home.write_text(
        f'[hooks.state."{hooks}:pre_tool_use:0:0"]\nenabled = false\n'
        'trusted_hash = "x"\n'
    )
    code, out = _test(tmp_path, [home, hooks], _judge(_blocks), ["delete"])
    assert code == 1
    assert "is switched off in /hooks: not run" in out


@pytest.mark.parametrize(
    ("entry", "note"),
    [
        ({"type": "prompt"}, "a hook of type prompt, which Codex skips"),
        ({"type": "agent"}, "a hook of type agent, which Codex skips"),
        (_hook("   "), "empty command, which Codex skips"),
        (_hook(**{"async": True}), "runs in the background: it stops nothing"),
    ],
)
def test_hooks_codex_never_waits_for_stop_nothing(
    tmp_path: Path, entry: dict[str, object], note: str
) -> None:
    code, out = _test(
        tmp_path, [_hooks_json(tmp_path, [entry])], _judge(_blocks), ["delete"]
    )
    assert code == 1
    assert note in out


def test_an_mcp_tool_hook_is_not_run_and_leaves_the_run_unsure(tmp_path: Path) -> None:
    entry = {"type": "mcp_tool", "server": "s", "tool": "t"}
    code, out = _test(tmp_path, [_hooks_json(tmp_path, [entry])], _judge(_blocks))
    assert code == 2
    assert "MCP tool hook, which is not run" in out


def test_a_timeout_over_ten_seconds_is_capped_and_zero_is_one(tmp_path: Path) -> None:
    seen: list[float] = []

    def run(handler: Handler, _payload: bytes) -> Run:
        seen.append(handler.timeout)
        return Run(0, b"")

    _test(
        tmp_path,
        [_hooks_json(tmp_path, [_hook(timeout=0), _hook("b", timeout=300)])],
        run,
    )
    assert set(seen) == {1.0, 10.0}


# --------------------------------------------------------------------------
# Files Codex would skip whole
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("document", "why"),
    [
        ({"hooks": {}, "extra": 1}, "top-level 'extra'"),
        ({"hooks": {"PreToolUse": {}}}, "its PreToolUse is not a list"),
        ({"hooks": {"PreToolUse": [{"matcher": 3, "hooks": []}]}}, "not text"),
        (
            {"hooks": {"PreToolUse": [{"hooks": [{"type": "http", "url": "x"}]}]}},
            "a type Codex does not know",
        ),
        ({"hooks": {"PreToolUse": [{"hooks": [{"type": "command"}]}]}}, "no command"),
        (
            {"hooks": {"PreToolUse": [{"hooks": [_hook(timeout="10")]}]}},
            "timeout of the wrong type",
        ),
        (
            {"hooks": {"PreToolUse": [{"hooks": [_hook(timeout=-1)]}]}},
            "timeout out of range",
        ),
        (
            # Valid JSON, but past Codex's u64: Codex skips the file.
            {"hooks": {"PreToolUse": [{"hooks": [_hook(timeout=2**64)]}]}},
            "timeout out of range",
        ),
        (
            {"hooks": {"PreToolUse": [{"hooks": [_hook(timeout=True)]}]}},
            "timeout of the wrong type",
        ),
        # Another event's mistake also makes Codex skip the PreToolUse hooks.
        ({"hooks": {"PreToolUse": [], "Stop": [{"hooks": "x"}]}}, "not a list"),
    ],
)
def test_a_file_codex_would_skip_is_said_and_the_run_is_unsure(
    tmp_path: Path, document: dict[str, object], why: str
) -> None:
    path = tmp_path / ".codex" / "hooks.json"
    path.parent.mkdir()
    path.write_text(json.dumps(document))
    code, out = _test(tmp_path, [path], _judge(_blocks))
    assert code == 2, out
    assert why in out
    assert "Not read." in out


def test_broken_toml_is_unread_never_a_traceback(tmp_path: Path) -> None:
    config = tmp_path / "config.toml"
    config.write_text("[hooks\n")
    code, out = _test(tmp_path, [config], _judge(_blocks))
    assert code == 2
    assert "not valid TOML" in out


def test_a_config_toml_that_is_a_fifo_is_refused_never_waited_on(
    tmp_path: Path,
) -> None:
    fifo = tmp_path / "config.toml"
    os.mkfifo(fifo)
    code, out = _test(tmp_path, [fifo], _judge(_blocks))
    assert code == 2
    assert "not a regular file" in out


def test_toml_nested_past_the_parser_s_limit_is_unread(tmp_path: Path) -> None:
    config = tmp_path / "config.toml"
    config.write_text("x = " + "[" * 100_000 + "]" * 100_000 + "\n")
    code, out = _test(tmp_path, [config], _judge(_blocks))
    assert code == 2
    assert "Not read." in out


def test_no_codex_file_says_where_it_looked(tmp_path: Path) -> None:
    code, out = _test(tmp_path, [tmp_path / ".codex" / "hooks.json"], _judge(_blocks))
    assert code == 1
    assert "no Codex hooks file found. Looked for:" in out


def test_files_are_codex_s_layers_lowest_first(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    sub = root / "pkg"
    sub.mkdir(parents=True)
    (root / ".git").mkdir()
    home = tmp_path / "home"
    assert _codex_hooks.files(sub, home) == [
        home / "hooks.json",
        home / "config.toml",
        root / ".codex" / "hooks.json",
        root / ".codex" / "config.toml",
        sub / ".codex" / "hooks.json",
        sub / ".codex" / "config.toml",
    ]


# --------------------------------------------------------------------------
# Real processes: Codex's shell, and the stderr exit 2 needs
# --------------------------------------------------------------------------


def _real(tmp_path: Path, script: str) -> Callable[[Handler, bytes], Run]:
    hook = tmp_path / "hook.sh"
    hook.write_text(script)
    hook.chmod(0o755)
    env = {"PATH": os.environ["PATH"], "HOME": str(tmp_path)}

    def run(handler: Handler, payload: bytes) -> Run:
        return run_handler(
            handler, payload, project=tmp_path, env=env, read_stderr=True
        )

    return run


def test_a_real_codex_hook_with_a_reason_blocks_and_without_one_does_not(
    tmp_path: Path,
) -> None:
    script = (
        "#!/bin/sh\n"
        "input=$(cat)\n"
        'case "$input" in *"push"*) echo "no pushes" >&2; exit 2;; esac\n'
        'case "$input" in *"rm "*) exit 2;; esac\n'  # no reason: Codex lets it on
        "exit 0\n"
    )
    files = [_hooks_json(tmp_path, [_hook(str(tmp_path / "hook.sh"))])]
    code, out = _test(tmp_path, files, _real(tmp_path, script), ["delete"])
    assert code == 1
    assert "force-push  stopped" in out
    assert "exit 2 with no reason on stderr" in out


def test_a_hook_that_floods_stderr_is_read_and_nothing_of_it_printed(
    tmp_path: Path,
) -> None:
    script = (
        "#!/bin/sh\ncat >/dev/null\n"
        f"{sys.executable} -c \"import sys; sys.stderr.write('SECRET' * 400000)\"\n"
        "exit 2\n"
    )
    files = [_hooks_json(tmp_path, [_hook(str(tmp_path / "hook.sh"))])]
    _code, out = _test(tmp_path, files, _real(tmp_path, script))
    assert "SECRET" not in out
    assert "stopped" in out


def test_the_cli_runs_codex_hooks_in_the_shell_and_without_claude_s_variable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / ".git").mkdir()
    # Blocks only when Claude Code's variable is unset, as it is under Codex.
    hook = tmp_path / "hook.sh"
    hook.write_text(
        "input=$(cat)\n"
        'case "$input" in *\'"git status"\'*) exit 0;; esac\n'
        '[ -z "$CLAUDE_PROJECT_DIR" ] && echo no >&2 && exit 2\n'
    )
    _hooks_json(tmp_path, [_hook(f"sh {hook}")])
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "codex-home"))
    monkeypatch.setenv("SHELL", "/bin/sh")
    monkeypatch.setattr(cli, "_managed", lambda *, codex: [])  # noqa: ARG005
    # Set in the shell running the test, as inside a Claude Code session: Codex
    # would still not pass it on (/code-review).
    monkeypatch.setenv("CLAUDE_PROJECT_DIR", str(tmp_path))
    code = cli.main(
        ["test", "codex", "--expect", "delete", "--settings", ".codex/hooks.json"]
    )
    out = capsys.readouterr().out
    assert "delete      stopped" in out, out
    assert code == 0, out


def test_the_cli_reads_codex_s_system_folder_both_files_lowest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[list[Path]] = []

    def capture(files: list[Path], **_kwargs: object) -> int:
        seen.append(files)
        return 0

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "home"))
    monkeypatch.setattr("paveo._guard_test.command", capture)
    assert cli.main(["test", "codex"]) == 0
    assert seen[0][:2] == [
        Path("/etc/codex/hooks.json"),
        Path("/etc/codex/config.toml"),
    ]


@pytest.mark.parametrize("read_stderr", [True, False])
def test_stderr_is_read_only_when_asked(tmp_path: Path, *, read_stderr: bool) -> None:
    handler = Handler(
        "h", lambda _t: True, ("/bin/sh", "-c", "echo why >&2"), 5.0, False
    )
    done = run_handler(
        handler, b"{}", project=tmp_path, env=dict(os.environ), read_stderr=read_stderr
    )
    assert done.exit == 0
    assert done.wrote_stderr is read_stderr


def test_a_skipped_toml_file_switches_no_hook_off(tmp_path: Path) -> None:
    hooks = _hooks_json(tmp_path, [_hook()])
    config = tmp_path / "home" / "config.toml"
    config.parent.mkdir()
    # A state switching the hook off, in a [hooks] table Codex skips whole.
    config.write_text(
        'PreToolUse = 3\n[hooks]\nPreToolUse = "x"\n'
        f'[hooks.state."{hooks}:pre_tool_use:0:0"]\nenabled = false\n'
    )
    _code, out = _test(tmp_path, [config, hooks], _judge(_blocks), ["delete"])
    assert "switched off" not in out
    assert "delete      stopped" in out
