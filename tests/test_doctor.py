"""``paveo doctor``: the Claude Code hooks that cannot work as written (D86).

A false "broken" costs the trust of the people who wrote the guard, so half of
this file is hooks that work and must be left alone. The broken ones are the
shapes found in public repositories on 7 Oct 2026, each reported upstream.
"""

from __future__ import annotations

import io
import json
import os
from pathlib import Path

import pytest

from paveo._doctor import command
from paveo.cli import main

CHE = (
    'echo "$CLAUDE_TOOL_INPUT" | grep -qE '
    "'(rm\\s+-rf\\s+(/|\\./|\\*)|git\\s+reset\\s+--hard)'"
    " && echo 'BLOCK: destructive operation' && exit 1 || true"
)
NYLAS = (
    "if echo \"$TOOL_INPUT\" | grep -q 'git commit'; then git diff --cached --name-only"
    " | grep -E '\\.(env|pem|key|p12)$' && echo 'BLOCKED' && exit 2 || exit 0; fi"
)
GOFMT = 'test "${CLAUDE_FILE_PATH##*.}" = "go" && gofmt -w "$CLAUDE_FILE_PATH" || true'
STELLAR = "git-secrets --scan $CLAUDE_FILE_PATHS 2>/dev/null || exit 1"


def _settings(tmp_path: Path, hooks: dict[str, object], **extra: object) -> Path:
    path = tmp_path / ".claude" / "settings.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"hooks": hooks, **extra}))
    return path


def _one(
    event: str, cmd: str, matcher: str = "Bash", **entry: object
) -> dict[str, object]:
    return {
        event: [
            {
                "matcher": matcher,
                "hooks": [{"type": "command", "command": cmd, **entry}],
            }
        ]
    }


def _doctor(tmp_path: Path, files: list[Path]) -> tuple[int, str]:
    out = io.StringIO()
    return command(files, project=tmp_path, out=out), out.getvalue()


@pytest.mark.parametrize(
    ("event", "cmd", "expected"),
    [
        ("PreToolUse", CHE, ["reads $CLAUDE_TOOL_INPUT,", "exits 1 and never 2"]),
        ("PreToolUse", NYLAS, ["reads $TOOL_INPUT,"]),
        ("PostToolUse", GOFMT, ["reads $CLAUDE_FILE_PATH,"]),
        # git-secrets is not a known shell tool, so its exit is not judged.
        ("PreToolUse", STELLAR, ["reads $CLAUDE_FILE_PATHS,"]),
        (
            "PreToolUse",
            'CMD="$CLAUDE_TOOL_INPUT_COMMAND"; echo "$CMD"',
            ["$CLAUDE_TOOL_INPUT_COMMAND"],
        ),
        (
            "PreToolUse",
            'case "${CLAUDE_TOOL_INPUT_FILE_PATH:-}" in *.md) :;; esac',
            ["$CLAUDE_TOOL_INPUT_FILE_PATH"],
        ),
        (
            "PreToolUse",
            'powershell -Command "$p=$env:CLAUDE_TOOL_INPUT_file_path"',
            ["$CLAUDE_TOOL_INPUT_file_path"],
        ),
        (
            "PreToolUse",
            "python3 -c \"import os; os.environ.get('CLAUDE_TOOL_INPUT') or '{}'\"",
            ["reads $CLAUDE_TOOL_INPUT,"],
        ),
    ],
)
def test_the_broken_shapes_found_in_public_repos_are_named(
    tmp_path: Path, event: str, cmd: str, expected: list[str]
) -> None:
    code, out = _doctor(tmp_path, [_settings(tmp_path, _one(event, cmd))])
    assert code == 1
    for phrase in expected:
        assert phrase in out
    assert f"{event}[0].hooks[0]" in out
    assert out.count("\n") == len(expected) + 1  # one line each, then the summary
    assert "1 cannot work as written" in out


def test_a_hook_in_command_and_args_form_is_read(tmp_path: Path) -> None:
    script = "const t=JSON.parse(process.env.CLAUDE_TOOL_INPUT||'{}'); process.exit(2)"
    hooks = _one("PreToolUse", "node", args=["-e", script])
    code, out = _doctor(tmp_path, [_settings(tmp_path, hooks)])
    assert (code, "reads $CLAUDE_TOOL_INPUT," in out) == (1, True)


@pytest.mark.parametrize(
    "cmd",
    [
        "jq -r .tool_input.command | grep -q 'rm -rf' && { echo no >&2; exit 2; }",
        'INPUT=$(cat); echo "$INPUT" | jq -r .tool_input.command',
        'TOOL_INPUT=$(cat); echo "$TOOL_INPUT" | grep -q x && exit 2 || true',
        'read -r TOOL_INPUT; echo "$TOOL_INPUT"',
        # Tries the variable, then reads stdin when it is empty: it works.
        'if [ -n "$CLAUDE_TOOL_INPUT" ]; then i="$CLAUDE_TOOL_INPUT"; '
        "elif [ ! -t 0 ]; then i=$(cat); fi",
        '"$CLAUDE_PROJECT_DIR"/.paveo/bin/paveo guard claude-code',
        'grep -q x && echo \'{"permissionDecision":"deny"}\'; exit 1',
        "grep -q x && exit 2; [ -f y ] || exit 1",
        "python3 .claude/hooks/guard.py || exit 1",
        'grep -q x; code=$?; exit "$code"',
    ],
)
def test_hooks_that_work_are_left_alone(tmp_path: Path, cmd: str) -> None:
    code, out = _doctor(tmp_path, [_settings(tmp_path, _one("PreToolUse", cmd))])
    assert (code, out) == (
        0,
        "paveo: checked 1 command hook in 1 file: 0 cannot work as written.\n",
    )


def test_a_name_the_settings_set_in_env_is_not_reported(tmp_path: Path) -> None:
    path = _settings(
        tmp_path, _one("PreToolUse", 'echo "$TOOL_INPUT"'), env={"TOOL_INPUT": "{}"}
    )
    assert _doctor(tmp_path, [path])[0] == 0


def test_exit_1_matters_only_where_a_call_can_be_blocked(tmp_path: Path) -> None:
    path = _settings(tmp_path, _one("PostToolUse", "grep -q x && exit 1 || true"))
    assert _doctor(tmp_path, [path])[0] == 0


def test_a_script_the_hook_runs_is_read(tmp_path: Path) -> None:
    hook = tmp_path / ".claude" / "hooks" / "guard.sh"
    hook.parent.mkdir(parents=True)
    hook.write_text(
        '#!/bin/bash\necho "$CLAUDE_TOOL_INPUT" | grep -q rm && exit 1\nexit 0\n'
    )
    path = _settings(
        tmp_path, _one("PreToolUse", '"$CLAUDE_PROJECT_DIR"/.claude/hooks/guard.sh')
    )
    code, out = _doctor(tmp_path, [path])
    assert code == 1
    assert "reads $CLAUDE_TOOL_INPUT," in out
    assert "exits 1 and never 2" in out


def test_a_script_that_hands_off_is_not_judged_on_its_exit(tmp_path: Path) -> None:
    hook = tmp_path / "guard.sh"
    hook.write_text("python3 check.py || exit 1\n")
    path = _settings(tmp_path, _one("PreToolUse", "bash guard.sh"))
    assert _doctor(tmp_path, [path])[0] == 0


def test_a_hook_is_never_printed(tmp_path: Path) -> None:
    secret = "ghp_" + "x" * 36
    cmd = f'curl -H "Authorization: {secret}" -d "$CLAUDE_TOOL_INPUT" localhost'
    code, out = _doctor(tmp_path, [_settings(tmp_path, _one("PreToolUse", cmd))])
    assert code == 1
    assert secret not in out
    assert "curl" not in out


def test_a_file_that_cannot_be_read_is_never_passed_as_clean(tmp_path: Path) -> None:
    broken = tmp_path / "settings.json"
    broken.write_text('{"hooks": {},}')
    code, out = _doctor(tmp_path, [broken])
    assert code == 2
    assert "could not read it (not valid JSON). Not checked." in out


def test_a_shape_it_does_not_know_is_said_to_be_unchecked(tmp_path: Path) -> None:
    # Seen in a public repo: the command where the list of hooks should be.
    hooks = {
        "PreToolUse": [{"matcher": "Bash", "command": 'echo "$CLAUDE_TOOL_INPUT"'}]
    }
    code, out = _doctor(tmp_path, [_settings(tmp_path, hooks)])
    assert code == 2
    assert "PreToolUse[0] has no list of hooks. Not checked." in out


def test_an_unreadable_script_is_said_to_be_unread(tmp_path: Path) -> None:
    hook = tmp_path / "guard.sh"
    hook.write_text("exit 0\n")
    hook.chmod(0)
    if os.access(hook, os.R_OK):
        pytest.skip("running as a user who can read anything")
    try:
        path = _settings(tmp_path, _one("PreToolUse", "bash guard.sh"))
        code, out = _doctor(tmp_path, [path])
    finally:
        hook.chmod(0o600)
    assert code == 2
    assert "guard.sh: could not read it" in out


def test_no_settings_file_says_nothing_was_checked(tmp_path: Path) -> None:
    code, out = _doctor(tmp_path, [tmp_path / "absent.json"])
    assert code == 0
    assert out.startswith(
        "paveo: no Claude Code settings file found, so nothing was checked"
    )


def test_a_huge_settings_file_is_not_read(tmp_path: Path) -> None:
    big = tmp_path / "settings.json"
    big.write_text(" " * ((1 << 20) + 1))
    code, out = _doctor(tmp_path, [big])
    assert code == 2
    assert "Not checked." in out


def test_the_command_checks_the_files_it_is_given(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = _settings(tmp_path, _one("PreToolUse", CHE))
    assert main(["doctor", "--settings", str(path)]) == 1
    assert "reads $CLAUDE_TOOL_INPUT," in capsys.readouterr().out


def test_a_comment_naming_a_variable_is_not_a_read(tmp_path: Path) -> None:
    hook = tmp_path / "guard.sh"
    hook.write_text(
        "#!/bin/bash\n# Claude Code never sets $CLAUDE_TOOL_INPUT, so read stdin.\n"
        "jq -r .tool_input.command | grep -q rm && exit 2\nexit 0\n"
    )
    path = _settings(tmp_path, _one("PreToolUse", "bash guard.sh"))
    assert _doctor(tmp_path, [path])[0] == 0


def test_control_characters_from_the_file_are_escaped(tmp_path: Path) -> None:
    hooks = {
        "Pre\x1b[2JToolUse": [
            {"hooks": [{"type": "command", "command": 'echo "$TOOL_INPUT"'}]}
        ]
    }
    code, out = _doctor(tmp_path, [_settings(tmp_path, hooks)])
    assert code == 1
    assert "\x1b" not in out
    assert "Pre\\x1b[2JToolUse[0].hooks[0]" in out


# Each of these was found by /code-review on 7 Oct and is kept from coming back.


def test_deeply_nested_json_is_unread_not_a_crash(tmp_path: Path) -> None:
    deep = tmp_path / "settings.json"
    deep.write_text("[" * 200_000 + "]" * 200_000)
    code, out = _doctor(tmp_path, [deep])
    assert (code, "nested too deeply" in out) == (2, True)


def test_a_settings_file_asked_for_by_name_must_exist(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["doctor", "--settings", str(tmp_path / "setings.json")]) == 2
    assert "no such file. Not checked." in capsys.readouterr().out


@pytest.mark.parametrize(
    "cmd",
    [
        ".claude/hooks/guard || exit 1",
        "$HOME/.claude/hooks/guard.py || exit 1",
        "git-secrets --scan . || exit 1",
        'if grep -q rm; then echo "{\\"decision\\": \\"block\\"}"; exit 0; fi; '
        "ls || exit 1",
    ],
)
def test_exit_1_is_not_judged_when_something_else_may_decide(
    tmp_path: Path, cmd: str
) -> None:
    assert _doctor(tmp_path, [_settings(tmp_path, _one("PreToolUse", cmd))])[0] == 0


def test_a_javascript_template_literal_is_not_an_environment_read(
    tmp_path: Path,
) -> None:
    (tmp_path / "g.js").write_text(
        "const TOOL_INPUT = JSON.parse(require('fs').readFileSync(0));\n"
        "console.error(`bad ${TOOL_INPUT.command}`); process.exit(2);\n"
    )
    path = _settings(tmp_path, _one("PreToolUse", "node g.js"))
    assert _doctor(tmp_path, [path])[0] == 0


def test_a_name_set_through_jq_sh_and_eval_is_assigned(tmp_path: Path) -> None:
    cmd = (
        'eval "$(jq -r \'@sh "TOOL_NAME=\\(.tool_name)"\')"; '
        '[ "$TOOL_NAME" = Bash ] && exit 2'
    )
    assert _doctor(tmp_path, [_settings(tmp_path, _one("PreToolUse", cmd))])[0] == 0


@pytest.mark.parametrize(
    "cmd",
    [
        'read -r c <<< "$(echo "$CLAUDE_TOOL_INPUT" | jq -r .command)"; '
        'echo "$c" | grep -q rm && exit 2',
        'CLAUDE_TOOL_INPUT="${CLAUDE_TOOL_INPUT:-}"; '
        'echo "$CLAUDE_TOOL_INPUT" | grep -q rm && exit 2',
        "[ -z \"$CLAUDE_TOOL_INPUT\" ] && echo 'nothing to read' && exit 0; "
        'echo "$CLAUDE_TOOL_INPUT" | grep -q rm && exit 2',
    ],
)
def test_reading_the_empty_variable_is_not_hidden_by_lookalikes(
    tmp_path: Path, cmd: str
) -> None:
    code, out = _doctor(tmp_path, [_settings(tmp_path, _one("PreToolUse", cmd))])
    assert (code, "reads $CLAUDE_TOOL_INPUT," in out) == (1, True)


def test_an_apostrophe_does_not_stop_the_script_being_read(tmp_path: Path) -> None:
    (tmp_path / "g.sh").write_text('echo "$CLAUDE_TOOL_INPUT" | grep -q rm && exit 2\n')
    cmd = "echo don't; bash g.sh"
    code, out = _doctor(tmp_path, [_settings(tmp_path, _one("PreToolUse", cmd))])
    assert (code, "reads $CLAUDE_TOOL_INPUT," in out) == (1, True)


def test_a_file_not_checked_is_not_counted_as_checked(tmp_path: Path) -> None:
    listed = tmp_path / "settings.json"
    listed.write_text("[]")
    code, out = _doctor(tmp_path, [listed])
    assert code == 2
    assert "checked 0 command hooks in 0 files" in out


def test_a_shared_script_is_read_once_and_reported_once(tmp_path: Path) -> None:
    hook = tmp_path / "guard.sh"
    hook.write_text("exit 0\n")
    hook.chmod(0)
    if os.access(hook, os.R_OK):
        pytest.skip("running as a user who can read anything")
    hooks = {
        "PreToolUse": [
            {"matcher": m, "hooks": [{"type": "command", "command": "bash guard.sh"}]}
            for m in ("Bash", "Edit", "Write")
        ]
    }
    try:
        code, out = _doctor(tmp_path, [_settings(tmp_path, hooks)])
    finally:
        hook.chmod(0o600)
    assert code == 2
    assert out.count("guard.sh: could not read it") == 1


@pytest.mark.parametrize(
    ("cmd", "code", "asked"),
    [
        ("jq -r .tool_input.command", 0, True),
        ('echo "$CLAUDE_TOOL_INPUT"', 1, True),
        (None, 2, False),  # a file it could not read: nothing was tried
    ],
)
def test_doctor_asks_on_stderr_only_when_it_checked(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    cmd: str | None,
    code: int,
    asked: bool,
) -> None:
    if cmd is None:
        path = tmp_path / "settings.json"
        path.write_text("{")
    else:
        path = _settings(tmp_path, _one("PreToolUse", cmd))
    assert main(["doctor", "--settings", str(path)]) == code
    captured = capsys.readouterr()
    assert "paveo/discussions" not in captured.out
    assert ("paveo/discussions" in captured.err) is asked
