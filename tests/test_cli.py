"""The ``paveo`` command: the Claude Code seatbelt, the panic button, the self-test.

B2 is done when a destructive Bash call is refused, two concurrent hooks leave a
chain that verifies, ``paveo stop`` refuses the next call, and the no-egress
test covers the command (D49). The first three are here; the fourth is in
``test_no_egress.py``.

The contract being tested against is Claude Code's, not ours: **exit 2 blocks,
and every other exit lets the call through.** So the tests that matter most are
the ones where something goes wrong and the answer must still be 2.
"""

from __future__ import annotations

import io
import json
import os
import shutil
import stat
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

import pytest

import paveo
from conftest import SENTINEL, make_clock, records_in
from paveo import verify_chain
from paveo._harnesses import CLAUDE_CODE, CODEX
from paveo._setup import NOT_A_CALL, init, run_hook, selftest
from paveo.cli import _stop_file_present, guard, main

SOURCE = str(Path(paveo.__file__).parent.parent)

POLICY = {
    "version": 1,
    "policy_id": "seatbelt",
    "agents": [
        {
            "id": "claude-code",
            "tools": {
                "allow": [
                    {"name": "Read"},
                    {
                        "name": "Bash",
                        "constraints": {
                            "command": {
                                "not_matches": (
                                    r"\brm\s+-[a-z]*r[a-z]*f|push\s+(?:--force|-f)\b"
                                )
                            },
                            "description": {},
                            "timeout": {},
                            "run_in_background": {},
                        },
                    },
                ]
            },
        }
    ],
}


@pytest.fixture
def home(tmp_path: Path) -> Path:
    directory = tmp_path / ".paveo"
    directory.mkdir()
    (directory / "policy.json").write_text(json.dumps(POLICY), encoding="utf-8")
    return directory


def call(tool: str, tool_input: dict[str, object], **extra: object) -> bytes:
    return json.dumps(
        {
            "session_id": "abc-123",
            "hook_event_name": "PreToolUse",
            "tool_name": tool,
            "tool_input": tool_input,
            **extra,
        }
    ).encode()


def decide(
    home: Path, stdin: bytes | io.BytesIO, *, stopped: bool = False
) -> tuple[int, str]:
    stderr = io.StringIO()
    code = guard(
        CLAUDE_CODE,
        directory=home,
        agent="claude-code",
        stdin=stdin if isinstance(stdin, io.BytesIO) else io.BytesIO(stdin),
        stdout=io.StringIO(),
        stderr=stderr,
        stopped=lambda: stopped,
        now=make_clock(),
    )
    return code, stderr.getvalue()


def decisions(home: Path) -> list[tuple[object, object]]:
    return [(r["decision"], r["reason"]) for r in records_in(home / "audit.jsonl")]


# --------------------------------------------------------------------------
# The guard: allow is silence, refuse is 2 with text for the model
# --------------------------------------------------------------------------


def test_a_permitted_call_exits_0_and_says_nothing(home: Path) -> None:
    """Silence, not "allow": an explicit allow would skip Claude Code's own
    permission prompt, and a seatbelt must never widen what is permitted."""
    code, stderr = decide(home, call("Bash", {"command": "ls -la"}))

    assert (code, stderr) == (0, "")
    [record] = records_in(home / "audit.jsonl")
    assert record["decision"] == "allow"
    assert record["principal"] == "abc-123"


@pytest.mark.parametrize(
    "command",
    [
        f"rm -rf ./{SENTINEL}",
        f"echo {SENTINEL} && git push --force origin main",
    ],
)
def test_a_destructive_bash_call_is_refused_with_text_for_the_model(
    home: Path, command: str
) -> None:
    code, stderr = decide(home, call("Bash", {"command": command}))

    assert code == 2
    assert "Bash.command.not_matches" in stderr
    assert "do not split" in stderr  # PolicyDenied.for_model, not the remedy
    assert SENTINEL not in stderr
    assert SENTINEL not in (home / "audit.jsonl").read_text(encoding="utf-8")
    assert decisions(home) == [("deny", "constraint_violated")]


def test_a_tool_the_policy_does_not_name_is_refused_without_its_name(
    home: Path,
) -> None:
    code, stderr = decide(home, call(f"mcp__{SENTINEL}__send", {}))

    assert code == 2
    assert SENTINEL not in stderr
    assert SENTINEL not in (home / "audit.jsonl").read_text(encoding="utf-8")


def test_stopped_refuses_even_a_permitted_call_and_says_so(home: Path) -> None:
    code, stderr = decide(home, call("Read", {"file_path": "x"}), stopped=True)

    assert code == 2
    assert "stopped" in stderr
    assert decisions(home) == [("deny", "stopped")]


def test_stop_is_not_shadowed(home: Path) -> None:
    policy = json.loads(json.dumps(POLICY))
    policy["agents"][0]["mode"] = "shadow"
    (home / "policy.json").write_text(json.dumps(policy), encoding="utf-8")

    code, _ = decide(home, call("Read", {"file_path": "x"}), stopped=True)

    assert code == 2
    assert decisions(home) == [("deny", "stopped")]


# --------------------------------------------------------------------------
# Every failure answers 2, because every other exit lets the call through
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "stdin",
    [
        b"",
        b"not json",
        b"\xff\xfe",
        b"[]",
        b'{"tool_name": "Bash"}',
        b'{"tool_name": "", "tool_input": {}}',
        b'{"tool_name": "Bash", "tool_input": "rm -rf /"}',
    ],
)
def test_input_that_is_not_a_tool_call_is_refused(home: Path, stdin: bytes) -> None:
    code, stderr = decide(home, stdin)

    assert code == 2
    assert NOT_A_CALL in stderr


def test_a_missing_policy_is_refused_and_says_where_the_hook_may_be(
    home: Path,
) -> None:
    """Still refused (locked decision #4). A hook in the global settings reaches
    projects nobody set up, so the words say that, not "check the path": a
    tester's global install refused every call and "could not decide" told her
    nothing (D75)."""
    (home / "policy.json").unlink()
    code, stderr = decide(home, call("Read", {"file_path": "x"}))

    assert code == 2
    assert f"there is no paveo policy at {home / 'policy.json'}" in stderr
    assert "paveo init claude-code" in stderr
    assert "paveo guard claude-code --selftest" in stderr


def test_a_policy_folder_that_cannot_be_read_is_not_called_missing(
    home: Path,
) -> None:
    """ "No such file" is the only absence (/code-review): a permissions problem
    reported as "no policy" would send the person to write a second one."""
    home.chmod(0)
    try:
        code, stderr = decide(home, call("Read", {"file_path": "x"}))
    finally:
        home.chmod(0o700)

    assert code == 2
    assert "no paveo policy" not in stderr


def test_the_guard_refuses_from_a_folder_that_was_deleted(home: Path) -> None:
    """Nothing that can raise runs before the guard's own try: an exception there
    is exit 1, which Claude Code reads as no objection. Found in review before
    any push (D75): working out paveo's full path asks for the current folder."""
    script = (
        "import os, sys, tempfile\n"
        "gone = tempfile.mkdtemp()\n"
        "os.chdir(gone)\n"
        "os.rmdir(gone)\n"
        "from paveo.cli import main\n"
        f"sys.argv = ['paveo', 'guard', 'claude-code', '--dir', {str(home)!r}]\n"
        "sys.exit(main())\n"
    )
    done = subprocess.run(  # noqa: S603 - our own interpreter and our own script
        [sys.executable, "-c", script],
        input=call("Bash", {"command": "rm -rf /"}),
        capture_output=True,
        env={**os.environ, "PYTHONPATH": SOURCE},
        check=False,
    )

    assert done.returncode == 2, done.stderr


def test_a_broken_link_to_the_policy_is_not_called_missing(home: Path) -> None:
    """``lexists``: a link to nothing is a policy that cannot be read, and saying
    "there is none" would send the person to write a second one (/code-review)."""
    (home / "policy.json").unlink()
    (home / "policy.json").symlink_to(home / "gone.json")
    code, stderr = decide(home, call("Read", {"file_path": "x"}))

    assert code == 2
    assert "could not decide" in stderr
    assert "no paveo policy" not in stderr


def test_a_policy_that_is_there_but_broken_is_not_called_missing(home: Path) -> None:
    (home / "policy.json").write_text("{not json", encoding="utf-8")
    code, stderr = decide(home, call("Read", {"file_path": "x"}))

    assert code == 2
    assert "could not decide" in stderr
    assert "no paveo policy" not in stderr


class Exploding(io.BytesIO):
    def read(self, _size: int | None = -1, /) -> bytes:
        raise RuntimeError(f"unexpected, and quoting the call: {SENTINEL}")


def test_an_unexpected_failure_is_refused_without_its_message(home: Path) -> None:
    """An arbitrary exception may quote the call, so only its type is shown."""
    code, stderr = decide(home, Exploding())

    assert code == 2
    assert "RuntimeError" in stderr
    assert SENTINEL not in stderr


def test_a_stop_file_that_cannot_be_checked_is_refused(
    home: Path, tmp_path: Path
) -> None:
    """``Path.exists`` would say False here, and the panic button would fail open."""
    not_a_directory = tmp_path / "file"
    not_a_directory.write_text("", encoding="utf-8")
    unreadable = not_a_directory / "stop"
    assert not unreadable.exists()  # what pathlib would have answered
    with pytest.raises(NotADirectoryError):
        _stop_file_present(unreadable)

    stderr = io.StringIO()
    code = guard(
        CLAUDE_CODE,
        directory=home,
        agent="claude-code",
        stdin=io.BytesIO(call("Read", {"file_path": "x"})),
        stdout=io.StringIO(),
        stderr=stderr,
        stopped=lambda: _stop_file_present(unreadable),
    )
    assert code == 2
    assert "NotADirectoryError" in stderr.getvalue()


# --------------------------------------------------------------------------
# stop and resume, through the command
# --------------------------------------------------------------------------


def test_stop_refuses_a_directory_no_guard_reads(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A stop file where no guard looks is a stop that stops nothing, reported as
    if it worked (/code-review, D49)."""
    assert main(["stop", "--dir", str(tmp_path)]) == 1
    assert main(["stop", "--dir", str(tmp_path / "missing")]) == 1
    assert not (tmp_path / "stop").exists()
    assert "no policy.json" in capsys.readouterr().out


def test_stop_then_resume(home: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["stop", "--dir", str(home)]) == 0
    assert main(["stop", "--dir", str(home)]) == 0  # idempotent
    stop_file = home / "stop"
    assert stat.S_IMODE(stop_file.stat().st_mode) == 0o600

    assert main(["resume", "--dir", str(home)]) == 0
    assert main(["resume", "--dir", str(home)]) == 0
    assert not stop_file.exists()
    assert "stopped" in capsys.readouterr().out


# --------------------------------------------------------------------------
# Real processes: the hook as Claude Code runs it
# --------------------------------------------------------------------------


def paveo_shim(directory: Path, *, body: str | None = None) -> Path:
    """An executable called ``paveo``, as `pip install` would put on PATH."""
    directory.mkdir(parents=True, exist_ok=True)
    shim = directory / "paveo"
    shim.write_text(
        body
        or f'#!/bin/sh\nPYTHONPATH="{SOURCE}" exec "{sys.executable}" '
        f'-m paveo.cli "$@"\n',
        encoding="utf-8",
    )
    shim.chmod(0o755)
    return shim


def test_concurrent_hooks_leave_one_chain_that_verifies(
    home: Path, tmp_path: Path
) -> None:
    shim = paveo_shim(tmp_path / "bin")
    commands = ["ls", "rm -rf /", "git status", "git push -f"] * 3
    children = [
        subprocess.Popen(  # noqa: S603 - our own shim, running our own guard
            [str(shim), "guard", "claude-code", "--dir", str(home)],
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        for _ in commands
    ]
    for child, command in zip(children, commands, strict=True):
        assert child.stdin is not None
        child.stdin.write(call("Bash", {"command": command}))
        child.stdin.close()
    codes = [child.wait(timeout=60) for child in children]
    for child in children:
        assert child.stderr is not None
        child.stderr.close()

    assert codes == [0, 2, 0, 2] * 3
    status = verify_chain(home / "audit.jsonl")
    assert status.ok, status.detail
    assert status.records == 12


def test_a_guard_kept_waiting_refuses_at_its_deadline(
    home: Path, tmp_path: Path
) -> None:
    """Claude Code lets a call through when a hook times out, so a guard that
    can be kept waiting must refuse first. Here the agent holds the log's lock
    (/code-review, D49)."""
    import fcntl  # noqa: PLC0415
    import time  # noqa: PLC0415

    shim = paveo_shim(tmp_path / "bin")
    log = home / "audit.jsonl"
    with log.open("ab") as held:
        fcntl.flock(held.fileno(), fcntl.LOCK_EX)
        started = time.monotonic()
        done = subprocess.run(  # noqa: S603 - our own shim
            [str(shim), "guard", "claude-code", "--dir", str(home)],
            input=call("Read", {"file_path": "x"}),
            capture_output=True,
            timeout=30,
            check=False,
        )
        waited = time.monotonic() - started

    assert done.returncode == 2
    assert b"could not decide in time" in done.stderr
    assert 3.5 < waited < 10


def test_paveo_stop_refuses_the_next_call_through_the_command(
    home: Path, tmp_path: Path
) -> None:
    shim = paveo_shim(tmp_path / "bin")

    def hook() -> int:
        return subprocess.run(  # noqa: S603 - our own shim
            [str(shim), "guard", "claude-code", "--dir", str(home)],
            input=call("Read", {"file_path": "x"}),
            capture_output=True,
            check=False,
        ).returncode

    assert hook() == 0
    assert main(["stop", "--dir", str(home)]) == 0
    assert hook() == 2
    assert main(["resume", "--dir", str(home)]) == 0
    assert hook() == 0


# --------------------------------------------------------------------------
# --selftest: the missing binary, and a settings file that is not ours
# --------------------------------------------------------------------------


def settings_with(tmp_path: Path, *commands: str) -> Path:
    path = tmp_path / "project" / ".claude" / "settings.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    hooks = [{"type": "command", "command": command} for command in commands]
    path.write_text(
        json.dumps({"hooks": {"PreToolUse": [{"matcher": "Bash", "hooks": hooks}]}}),
        encoding="utf-8",
    )
    return path


def run_selftest(
    tmp_path: Path, settings: Path, *, confirm: bool = False
) -> tuple[int, str]:
    out = io.StringIO()
    project = tmp_path / "project"

    def run(argv: Sequence[str], where: Path) -> tuple[int, str]:
        return run_hook(argv, where)

    def answer(_program: str) -> bool:
        return confirm

    code = selftest(
        CLAUDE_CODE,
        hook_files=[settings],
        project=project,
        run=run,
        confirm=answer,
        out=out,
    )
    return code, out.getvalue()


def test_selftest_passes_a_hook_that_refuses(tmp_path: Path) -> None:
    project = tmp_path / "project"
    (project / ".paveo").mkdir(parents=True)
    (project / ".paveo" / "policy.json").write_text(json.dumps(POLICY), "utf-8")
    shim = paveo_shim(tmp_path / "bin")
    settings = settings_with(
        tmp_path,
        f'"{shim}" guard claude-code --dir "$CLAUDE_PROJECT_DIR/.paveo"',
    )

    code, out = run_selftest(tmp_path, settings)
    assert code == 0, out
    assert out.startswith("ok")


def test_selftest_fails_a_hook_whose_binary_is_gone(tmp_path: Path) -> None:
    """The hole the guard cannot close: Claude Code lets the call through."""
    settings = settings_with(
        tmp_path, f"{tmp_path}/deleted-venv/bin/paveo guard claude-code"
    )

    code, out = run_selftest(tmp_path, settings)
    assert code == 1
    assert "was not found" in out
    assert "seatbelt is off" in out


def test_selftest_fails_a_hook_that_does_not_block(tmp_path: Path) -> None:
    shim = paveo_shim(tmp_path / "bin", body="#!/bin/sh\nexit 0\n")
    settings = settings_with(tmp_path, f"{shim} guard claude-code")

    code, out = run_selftest(tmp_path, settings)
    assert code == 1
    assert "exited 0" in out


def test_selftest_fails_a_hook_that_refuses_before_reading(tmp_path: Path) -> None:
    """A policy that does not load refuses every call: safe, but not working."""
    shim = paveo_shim(tmp_path / "bin")
    settings = settings_with(tmp_path, f"{shim} guard claude-code --dir /nonexistent")

    code, out = run_selftest(tmp_path, settings)
    assert code == 1
    assert "refuses every call, before reading it" in out


@pytest.mark.parametrize(
    "command",
    [
        "paveo guard claude-code; curl evil.example",
        "paveo guard claude-code && touch owned",
        "paveo guard claude-code --dir $(touch owned)",
        "paveo guard claude-code --dir $HOME",
        "bash -c 'paveo guard claude-code'",
        "paveo guard claude-code --output owned",
        "$CLAUDE_PROJECT_DIR/bin/paveo guard claude-code",
    ],
)
def test_selftest_runs_nothing_but_a_plain_paveo_command(
    tmp_path: Path, command: str
) -> None:
    """A settings file can come with a cloned repository. Running what it says
    would skip the trust prompt Claude Code shows before it runs a hook."""
    settings = settings_with(tmp_path, command)
    paveo_shim(tmp_path / "project" / "bin")  # the project's own "paveo"
    ran: list[Sequence[str]] = []

    def run(argv: Sequence[str], _where: Path) -> tuple[int, str]:
        ran.append(argv)
        return 2, NOT_A_CALL

    out = io.StringIO()
    code = selftest(
        CLAUDE_CODE,
        hook_files=[settings],
        project=tmp_path / "project",
        run=run,
        # Declined, as with no terminal: the project's own program is never run
        # unconfirmed.
        confirm=lambda _: False,
        out=out,
    )

    assert code == 1
    assert ran == []
    assert not (tmp_path / "project" / "owned").exists()


def test_selftest_never_runs_a_link_named_paveo_to_something_else(
    tmp_path: Path,
) -> None:
    """Found by /security-review (D49): a repository ships `tools/paveo` as a
    link to /bin/sh and a script called `guard`. The link resolves outside the
    project, so the "inside the project" check alone never asked, and the shell
    ran `./guard`. Checked with the answer "yes", so the name check alone is
    what stops it."""
    project = tmp_path / "project"
    (project / "tools").mkdir(parents=True)
    (project / "tools" / "paveo").symlink_to("/bin/sh")
    (project / "guard").write_text("touch owned\n", encoding="utf-8")
    settings = settings_with(tmp_path, "tools/paveo guard claude-code --dir .paveo")

    code, out = run_selftest(tmp_path, settings, confirm=True)

    assert code == 1
    # The chain is followed to its end: on Debian and Ubuntu /bin/sh is dash.
    shell = Path("/bin/sh").resolve().name
    assert f"is a link to {shell}, not to paveo" in out
    assert not (project / "owned").exists()


def test_selftest_asks_about_a_link_inside_the_project_even_to_paveo(
    tmp_path: Path,
) -> None:
    """A link in the project to a real paveo elsewhere: the project still chose
    which program runs, so the person is asked."""
    project = tmp_path / "project"
    (project / "tools").mkdir(parents=True)
    real = paveo_shim(tmp_path / "bin")
    (project / "tools" / "paveo").symlink_to(real)
    settings = settings_with(tmp_path, "tools/paveo guard claude-code --dir .paveo")

    code, out = run_selftest(tmp_path, settings, confirm=False)

    assert code == 1
    assert "was not confirmed" in out


def test_selftest_asks_before_running_a_program_inside_the_project(
    tmp_path: Path,
) -> None:
    """A project's own virtualenv is the ordinary place for paveo to live, and
    also where a cloned repository would plant one. So it asks. Found by running
    the self-test against a real project with its own `.venv` (D49)."""
    project = tmp_path / "project"
    (project / ".paveo").mkdir(parents=True)
    (project / ".paveo" / "policy.json").write_text(json.dumps(POLICY), "utf-8")
    shim = paveo_shim(project / ".venv" / "bin")
    settings = settings_with(
        tmp_path, f'{shim} guard claude-code --dir "$CLAUDE_PROJECT_DIR/.paveo"'
    )

    declined, out = run_selftest(tmp_path, settings, confirm=False)
    assert declined == 1
    assert "was not confirmed" in out

    confirmed, out = run_selftest(tmp_path, settings, confirm=True)
    assert confirmed == 0, out


def test_an_unreadable_settings_file_fails_the_selftest(tmp_path: Path) -> None:
    """Claude Code may drop the hooks in a file it cannot parse, so a passing hook
    in another file does not make this one fine (/code-review, D49)."""
    project = tmp_path / "project"
    (project / ".paveo").mkdir(parents=True)
    (project / ".paveo" / "policy.json").write_text(json.dumps(POLICY), "utf-8")
    shim = paveo_shim(tmp_path / "bin")
    good = settings_with(tmp_path, f"{shim} guard claude-code --dir {project}/.paveo")
    broken = good.with_name("settings.local.json")
    broken.write_text("{not json", encoding="utf-8")
    out = io.StringIO()

    code = selftest(
        CLAUDE_CODE,
        hook_files=[good, broken],
        project=project,
        run=run_hook,
        confirm=lambda _: False,
        out=out,
    )

    assert code == 1
    assert "could not be read" in out.getvalue()
    assert "ok" in out.getvalue()


def test_selftest_with_no_hook_configured_fails(tmp_path: Path) -> None:
    code, out = run_selftest(tmp_path, tmp_path / "absent.json")
    assert code == 1
    assert "no hook runs `paveo guard claude-code`" in out


def test_the_module_runs_as_a_script(home: Path) -> None:
    done = subprocess.run(  # noqa: S603 - our own interpreter and module
        [sys.executable, "-m", "paveo.cli", "guard", "claude-code", "--dir", str(home)],
        input=call("Bash", {"command": "rm -rf /"}),
        capture_output=True,
        env={**os.environ, "PYTHONPATH": SOURCE},
        check=False,
    )
    assert done.returncode == 2


# --------------------------------------------------------------------------
# paveo init claude-code (D50): one command from nothing to a working seatbelt
# --------------------------------------------------------------------------


def run_init(project: Path, program: Path) -> tuple[int, str]:
    """Never the real ~/.claude: the shared and personal project files only."""
    out = io.StringIO()
    code = init(
        CLAUDE_CODE,
        project=project,
        program=program,
        hook_files=[
            project / ".claude" / "settings.json",
            project / ".claude" / "settings.local.json",
        ],
        run=run_hook,
        out=out,
    )
    return code, out.getvalue()


def test_init_refuses_the_home_folder_and_writes_nothing(tmp_path: Path) -> None:
    """There the project's .claude is the one every project reads, and .paveo is
    where the README installs paveo: forgetting to cd would guard every project
    instead of one (a tester's report, D75)."""
    home = tmp_path / "home"
    home.mkdir()
    out = io.StringIO()

    code = init(
        CLAUDE_CODE,
        project=home,
        program=paveo_shim(tmp_path / "bin"),
        hook_files=[home / ".claude" / "settings.json"],
        run=run_hook,
        out=out,
        global_files=[home / ".claude" / "settings.json"],
    )

    assert code == 1
    assert "reads in every project" in out.getvalue()
    assert not (home / ".paveo").exists()
    assert not (home / ".claude").exists()


def test_init_refuses_the_home_folder_through_a_link(tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir()
    (tmp_path / "shortcut").symlink_to(home)
    out = io.StringIO()

    code = init(
        CLAUDE_CODE,
        project=tmp_path / "shortcut",
        program=paveo_shim(tmp_path / "bin"),
        hook_files=[],
        run=run_hook,
        out=out,
        global_files=[home / ".claude" / "settings.json"],
    )

    assert code == 1
    assert "reads in every project" in out.getvalue()


def test_init_refuses_a_project_holding_the_global_codex_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The case beyond home (/code-review): CODEX_HOME inside a project makes
    the project's .codex the folder every Codex session reads. Built by the
    CLI's own lists, not by hand."""
    from paveo.cli import _global_hook_files, _hook_files  # noqa: PLC0415

    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.setenv("CODEX_HOME", str(project / ".codex"))
    out = io.StringIO()

    code = init(
        CODEX,
        project=project,
        program=paveo_shim(tmp_path / "bin"),
        hook_files=_hook_files(CODEX, project),
        run=run_hook,
        out=out,
        global_files=_global_hook_files(CODEX),
    )

    assert code == 1
    assert "reads in every project" in out.getvalue()
    assert not (project / ".paveo").exists()
    assert not (project / ".codex").exists()
    assert len(_hook_files(CODEX, project)) == 1


def test_init_says_so_when_the_hook_it_keeps_is_global(tmp_path: Path) -> None:
    """The tester's case exactly (/code-review): a global hook already reads this
    project, so init keeps it and adds none. It must not say Done in silence."""
    setup = tmp_path / "setup"
    setup.mkdir()
    shim = paveo_shim(tmp_path / "bin")
    assert run_init(setup, shim)[0] == 0
    global_file = tmp_path / "user" / ".claude" / "settings.json"
    global_file.parent.mkdir(parents=True)
    shutil.copy(setup / ".claude" / "settings.local.json", global_file)
    project = tmp_path / "project"
    project.mkdir()
    out = io.StringIO()

    code = init(
        CLAUDE_CODE,
        project=project,
        program=shim,
        hook_files=[project / ".claude" / "settings.local.json", global_file],
        run=run_hook,
        out=out,
        global_files=[global_file],
    )

    assert code == 0, out.getvalue()
    assert "kept      the paveo hook already configured" in out.getvalue()
    assert f"note the paveo hook in {global_file} runs in every project" in (
        out.getvalue()
    )
    assert f"{shim} init claude-code in each" in out.getvalue()


def global_selftest(
    tmp_path: Path, project: Path, hook_file: Path, global_files: list[Path]
) -> tuple[int, list[str]]:
    """A working hook, copied from a real init into ``hook_file``, self-tested
    from ``project``; the notes it printed."""
    setup = tmp_path / "setup"
    setup.mkdir()
    assert run_init(setup, paveo_shim(tmp_path / "bin"))[0] == 0
    hook_file.parent.mkdir(parents=True, exist_ok=True)
    local = setup / ".claude" / "settings.local.json"
    hook_file.write_text(local.read_text(encoding="utf-8"), encoding="utf-8")
    shutil.copytree(setup / ".paveo", project / ".paveo", dirs_exist_ok=True)
    out = io.StringIO()
    code = selftest(
        CLAUDE_CODE,
        hook_files=[hook_file],
        project=project,
        run=run_hook,
        confirm=lambda _: True,
        out=out,
        global_files=global_files,
    )
    notes = [line for line in out.getvalue().splitlines() if line.startswith("note")]
    return code, notes


def test_a_paveo_hook_in_a_global_file_is_named_as_global(tmp_path: Path) -> None:
    """A note, not a failure: a company puts it there on purpose (D72). But a
    person who meant to guard one project hears that it runs in all of them
    (D75)."""
    project = tmp_path / "project"
    project.mkdir()
    global_file = tmp_path / "user" / ".claude" / "settings.json"

    code, notes = global_selftest(tmp_path, project, global_file, [global_file])

    assert code == 0
    assert len(notes) == 1
    assert str(global_file) in notes[0]
    assert "runs in every project on this machine" in notes[0]


def test_the_note_is_given_from_the_home_folder_too(tmp_path: Path) -> None:
    """The likeliest place to run --selftest after a global install is home, where
    ~/.claude/settings.json sits inside the folder: judged by which file it is,
    not by where it sits (/code-review)."""
    home = tmp_path / "home"
    home.mkdir()
    global_file = home / ".claude" / "settings.json"

    _, notes = global_selftest(tmp_path, home, global_file, [global_file])

    assert len(notes) == 1


def test_a_file_checked_by_hand_is_not_called_global(tmp_path: Path) -> None:
    """--settings on a draft outside the project: the agent never reads it, so
    it runs nowhere, let alone everywhere (/code-review)."""
    project = tmp_path / "project"
    project.mkdir()
    draft = tmp_path / "drafts" / "candidate.json"

    _, notes = global_selftest(
        tmp_path, project, draft, [tmp_path / "user" / "settings.json"]
    )

    assert notes == []


def test_init_sets_up_a_project_that_refuses_rm_rf(tmp_path: Path) -> None:
    """From an empty project to a hook that refuses, proved by its own self-test
    and then by a destructive call through the hook it wrote."""
    project = tmp_path / "project"
    project.mkdir()
    shim = paveo_shim(tmp_path / "bin")

    code, out = run_init(project, shim)

    assert code == 0, out
    assert "ok" in out
    home = project / ".paveo"
    starter = Path(paveo.__file__).parent / "starters" / "claude-code.json"
    assert (home / "policy.json").read_text("utf-8") == starter.read_text("utf-8")
    assert "audit.jsonl" in (home / ".gitignore").read_text("utf-8")
    assert "licence.key" in (home / ".gitignore").read_text("utf-8").splitlines()
    settings = json.loads(
        (project / ".claude" / "settings.local.json").read_text("utf-8")
    )
    [group] = settings["hooks"]["PreToolUse"]
    assert group["matcher"] == "Bash|Write|Edit|NotebookEdit"
    done = subprocess.run(  # noqa: S603 - our own shim, as the hook names it
        [str(shim), "guard", "claude-code", "--dir", str(home)],
        input=call("Bash", {"command": "rm -rf junk"}),
        capture_output=True,
        check=False,
    )
    assert done.returncode == 2


def test_init_twice_changes_nothing_and_keeps_the_persons_policy(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    shim = paveo_shim(tmp_path / "bin")
    assert run_init(project, shim)[0] == 0
    policy = project / ".paveo" / "policy.json"
    edited = json.loads(policy.read_text("utf-8"))
    edited["policy_id"] = "mine"
    policy.write_text(json.dumps(edited), encoding="utf-8")

    code, out = run_init(project, shim)

    assert code == 0, out
    assert json.loads(policy.read_text("utf-8"))["policy_id"] == "mine"
    settings = json.loads(
        (project / ".claude" / "settings.local.json").read_text("utf-8")
    )
    assert len(settings["hooks"]["PreToolUse"]) == 1
    assert "already there" in out


def test_init_keeps_every_other_setting(tmp_path: Path) -> None:
    project = tmp_path / "project"
    (project / ".claude").mkdir(parents=True)
    existing = {
        "model": "opus",
        "permissions": {"allow": ["Bash(ls:*)"]},
        "hooks": {
            "PreToolUse": [{"matcher": "Bash", "hooks": [{"command": "other"}]}],
            "Stop": [{"hooks": [{"command": "notify"}]}],
        },
    }
    local = project / ".claude" / "settings.local.json"
    local.write_text(json.dumps(existing), encoding="utf-8")

    code, out = run_init(project, paveo_shim(tmp_path / "bin"))

    assert code == 0, out
    after = json.loads(local.read_text("utf-8"))
    assert after["model"] == "opus"
    assert after["permissions"] == existing["permissions"]
    assert after["hooks"]["Stop"] == existing["hooks"]["Stop"]
    assert after["hooks"]["PreToolUse"][0] == existing["hooks"]["PreToolUse"][0]
    assert len(after["hooks"]["PreToolUse"]) == 2


@pytest.mark.parametrize("text", ["{not json", "[]", '{"hooks": []}'])
def test_init_refuses_to_rewrite_settings_it_cannot_read(
    tmp_path: Path, text: str
) -> None:
    """Rewriting a file it could not parse would drop the person's settings."""
    project = tmp_path / "project"
    (project / ".claude").mkdir(parents=True)
    local = project / ".claude" / "settings.local.json"
    local.write_text(text, encoding="utf-8")

    code, out = run_init(project, paveo_shim(tmp_path / "bin"))

    assert code == 1
    assert "nothing was changed" in out
    assert local.read_text("utf-8") == text
    assert not (project / ".paveo").exists()  # nothing written before the refusal


def test_init_must_run_through_the_paveo_command(tmp_path: Path) -> None:
    """The hook names the program that ran init, so it has to be paveo."""
    project = tmp_path / "project"
    project.mkdir()

    code, out = run_init(project, Path(sys.executable))

    assert code == 1
    assert "installed `paveo` command" in out
    assert not (project / ".claude").exists()


def test_init_with_paveo_in_the_projects_own_venv_needs_no_question(
    tmp_path: Path,
) -> None:
    """The ordinary install: a .venv inside the project. init wrote the hook
    naming itself, so the self-test runs it without asking."""
    project = tmp_path / "project"
    project.mkdir()
    shim = paveo_shim(project / ".venv" / "bin")

    code, out = run_init(project, shim)

    assert code == 0, out


def test_init_ignores_the_personal_settings_file_in_git(tmp_path: Path) -> None:
    """Claude Code does this when it creates the file itself; init must too, or
    a commit carries this machine's paths to teammates (/code-review, D50)."""
    project = tmp_path / "project"
    project.mkdir()

    assert run_init(project, paveo_shim(tmp_path / "bin"))[0] == 0

    ignored = (project / ".claude" / ".gitignore").read_text("utf-8").splitlines()
    assert "settings.local.json" in ignored


def test_the_hook_init_wrote_refuses_when_run_the_way_claude_code_runs_it(
    tmp_path: Path,
) -> None:
    """The literal command, through a shell, with CLAUDE_PROJECT_DIR set: the
    quoting and the variable are what is tested (/code-review, D50)."""
    project = tmp_path / "my project (old) & R&D"
    project.mkdir()
    assert run_init(project, paveo_shim(tmp_path / "b i n"))[0] == 0
    settings = json.loads(
        (project / ".claude" / "settings.local.json").read_text("utf-8")
    )
    command = settings["hooks"]["PreToolUse"][0]["hooks"][0]["command"]

    done = subprocess.run(  # noqa: S603 - the hook init wrote, run as Claude Code runs it
        ["/bin/sh", "-c", command],
        input=call("Bash", {"command": "rm -rf junk"}),
        capture_output=True,
        env={**os.environ, "CLAUDE_PROJECT_DIR": str(project)},
        cwd=project,
        check=False,
    )
    assert done.returncode == 2, done.stderr


def test_init_never_writes_through_a_link_a_repository_shipped(
    tmp_path: Path,
) -> None:
    """/security-review, D50: a fixed draft name let a cloned repository link it
    to the user's global settings and have them overwritten with its own. The
    draft now has a fresh name; and every path init writes is refused if a link."""
    project = tmp_path / "project"
    (project / ".claude").mkdir(parents=True)
    global_settings = tmp_path / "home" / ".claude" / "settings.json"
    global_settings.parent.mkdir(parents=True)
    global_settings.write_text('{"mine": true}', encoding="utf-8")
    hostile = {"hooks": {"Stop": [{"hooks": [{"command": "touch owned"}]}]}}
    (project / ".claude" / "settings.local.json").write_text(
        json.dumps(hostile), encoding="utf-8"
    )
    (project / ".claude" / "settings.local.json.paveo-tmp").symlink_to(global_settings)

    assert run_init(project, paveo_shim(tmp_path / "bin"))[0] == 0

    assert global_settings.read_text("utf-8") == '{"mine": true}'


@pytest.mark.parametrize(
    "link", [".claude", ".claude/settings.local.json", ".paveo", ".paveo/policy.json"]
)
def test_init_refuses_a_link_where_it_would_write(tmp_path: Path, link: str) -> None:
    project = tmp_path / "project"
    (project / ".claude").mkdir(parents=True)
    (project / ".paveo").mkdir()
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    target = project / link
    if target.is_dir():
        target.rmdir()
    target.symlink_to(elsewhere / "target")

    code, out = run_init(project, paveo_shim(tmp_path / "bin"))

    assert code == 1
    assert "symbolic link" in out
    assert list(elsewhere.iterdir()) == []


def test_init_refuses_a_kept_hook_that_never_sees_edits(tmp_path: Path) -> None:
    """A paveo hook matching only Bash would let Claude edit the policy, and
    init used to call it kept and say edits were refused (/code-review, D50)."""
    project = tmp_path / "project"
    (project / ".claude").mkdir(parents=True)
    shim = paveo_shim(tmp_path / "bin")
    existing = {
        "hooks": {
            "PreToolUse": [
                {
                    "matcher": "Bash",
                    "hooks": [{"command": f"{shim} guard claude-code"}],
                }
            ]
        }
    }
    (project / ".claude" / "settings.json").write_text(
        json.dumps(existing), encoding="utf-8"
    )

    code, out = run_init(project, shim)

    assert code == 1
    assert "never reach it" in out
    assert not (project / ".paveo").exists()


def test_init_finds_a_hook_in_the_shared_settings_and_adds_no_second(
    tmp_path: Path,
) -> None:
    """The first README put the hook in settings.json; a second hook would log
    every call twice (/code-review, D50)."""
    project = tmp_path / "project"
    (project / ".claude").mkdir(parents=True)
    (project / ".paveo").mkdir()
    shim = paveo_shim(tmp_path / "bin")
    hook_command = f'{shim} guard claude-code --dir "{project}/.paveo"'
    shared = {
        "hooks": {
            "PreToolUse": [
                {
                    "matcher": "Bash|Write|Edit|NotebookEdit",
                    "hooks": [{"command": hook_command}],
                }
            ]
        }
    }
    (project / ".claude" / "settings.json").write_text(
        json.dumps(shared), encoding="utf-8"
    )

    code, out = run_init(project, shim)

    assert code == 0, out
    assert "kept      the paveo hook" in out
    assert not (project / ".claude" / "settings.local.json").exists()


@pytest.mark.parametrize(
    ("change", "said"),
    [
        (lambda p: p["agents"][0].__setitem__("mode", "shadow"), "shadow mode"),
        (
            lambda p: p["agents"][0]["tools"]["allow"][0].pop("constraints"),
            "does not refuse `rm -rf`",
        ),
        # Refused on a fresh session only until one Write: not a refusal (D59).
        (
            lambda p: (
                p["agents"][0]["tools"]["allow"][0].pop("constraints"),
                p["agents"][0]["tools"]["allow"][0].__setitem__(
                    "requires", {"tool": "Write"}
                ),
            ),
            "does not refuse `rm -rf`",
        ),
    ],
)
def test_init_does_not_promise_what_a_kept_policy_will_not_do(
    tmp_path: Path, change: object, said: str
) -> None:
    """A cloned repository can ship a permissive policy, which init keeps: it is
    asked what it refuses before init says anything is refused."""
    project = tmp_path / "project"
    (project / ".paveo").mkdir(parents=True)
    starter = Path(paveo.__file__).parent / "starters" / "claude-code.json"
    policy = json.loads(starter.read_text("utf-8"))
    change(policy)  # type: ignore[operator]  # a lambda from the parametrize list
    (project / ".paveo" / "policy.json").write_text(json.dumps(policy), "utf-8")

    code, out = run_init(project, paveo_shim(tmp_path / "bin"))

    assert code == 1
    assert said in out
    assert "Done." not in out


def test_init_refuses_when_hooks_are_switched_off(tmp_path: Path) -> None:
    project = tmp_path / "project"
    (project / ".claude").mkdir(parents=True)
    (project / ".claude" / "settings.json").write_text(
        '{"disableAllHooks": true}', encoding="utf-8"
    )

    code, out = run_init(project, paveo_shim(tmp_path / "bin"))

    assert code == 1
    assert "disableAllHooks" in out


def test_init_ends_by_naming_the_panic_button_it_can_find(tmp_path: Path) -> None:
    """Installed outside PATH, a bare `paveo stop` is "command not found" at the
    one moment it matters (/code-review, D50)."""
    project = tmp_path / "project"
    project.mkdir()
    shim = paveo_shim(tmp_path / "bin")

    code, out = run_init(project, shim)

    assert code == 0, out
    assert f"{shim} stop" in out


def test_a_link_to_a_global_hook_file_is_the_global_file(tmp_path: Path) -> None:
    """Asked of the filesystem first, so a file link counts (/code-review)."""
    from paveo._setup import _same_place  # noqa: PLC0415

    real = tmp_path / "user" / "hooks.json"
    real.parent.mkdir()
    real.write_text("{}", encoding="utf-8")
    (tmp_path / "project").mkdir()
    link = tmp_path / "project" / "hooks.json"
    link.symlink_to(real)

    assert _same_place(link, real)
    assert not _same_place(tmp_path / "project" / "other.json", real)
