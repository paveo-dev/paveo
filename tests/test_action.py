"""The GitHub Action (``action.yml``, ``action/test.sh``, D106): a pull request
fails when a spelling gets past the hooks and passes when none does, the report
lands in the job summary, and nothing a workflow passes in runs as a command."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import paveo

SOURCE = Path(paveo.__file__).parent.parent
ROOT = SOURCE.parent
PROBE = "paveo-test-probe"


def test_inputs_reach_the_shell_only_as_environment_variables() -> None:
    """`${{ }}` inside `run:` is pasted into the script before bash reads it,
    the template injection zizmor flags; in `env:` it is only a value."""
    lines = (ROOT / "action.yml").read_text("utf-8").splitlines()
    expressions = [line for line in lines if "${{" in line]
    assert expressions
    for line in expressions:
        assert re.fullmatch(r"\s+PAVEO_[A-Z]+: \$\{\{ inputs\.[a-z]+ \}\}", line), line


def _job(tmp_path: Path, guard: str, expect: str) -> tuple[int, str, str]:
    """``action/test.sh`` as the Action's second step runs it, against a
    project whose one hook is ``guard``, a shell script."""
    project, home, bin_dir = (tmp_path / d for d in ("project", "home", "bin"))
    for folder in (project / ".claude", home, bin_dir):
        folder.mkdir(parents=True)
    hook = project / ".claude" / "guard.sh"
    hook.write_text(f"#!/bin/sh\ninput=$(cat)\n{guard}\nexit 0\n")
    hook.chmod(0o755)
    entry = {"type": "command", "command": str(hook)}
    settings = {"hooks": {"PreToolUse": [{"matcher": "Bash|Read", "hooks": [entry]}]}}
    (project / ".claude" / "settings.json").write_text(json.dumps(settings))
    paveo_command = bin_dir / "paveo"
    paveo_command.write_text(
        f'#!/bin/sh\nPYTHONPATH="{SOURCE}" exec "{sys.executable}" -m paveo.cli "$@"\n'
    )
    paveo_command.chmod(0o755)
    summary = tmp_path / "summary.md"
    done = subprocess.run(  # noqa: S603 - our own script, this interpreter
        ["bash", str(ROOT / "action" / "test.sh")],  # noqa: S607 - bash on PATH
        cwd=project,
        env={
            "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
            "HOME": str(home),
            "PAVEO_EXPECT": expect,
            "PAVEO_JSON": str(tmp_path / "results.json"),
            "GITHUB_STEP_SUMMARY": str(summary),
        },
        capture_output=True,
        text=True,
        check=False,
        timeout=180,
    )
    return done.returncode, done.stdout, summary.read_text() if summary.exists() else ""


def test_a_guard_that_leaks_fails_the_job(tmp_path: Path) -> None:
    guard = 'case "$input" in *\'"command": "rm -rf\'*) exit 2;; esac'
    code, out, summary = _job(tmp_path, guard, "delete")
    assert code == 1, out
    assert f"rm -r -f /{PROBE}" in out
    assert summary.startswith("### paveo test")
    assert f"rm -r -f /{PROBE}" in summary
    assert json.loads((tmp_path / "results.json").read_text())["exit"] == 1


def test_a_guard_that_stops_everything_passes_the_job(tmp_path: Path) -> None:
    guard = f'case "$input" in *{PROBE}*|*.env*) exit 2;; esac'
    code, out, _ = _job(tmp_path, guard, "force-push delete hard-reset clean secrets")
    assert code == 0, out
    assert "nothing expected got past the hooks." in out


def test_an_input_that_is_not_an_action_fails_and_runs_nothing(tmp_path: Path) -> None:
    marker = tmp_path / "ran"
    code, _, _ = _job(tmp_path, "exit 0", f"* $(touch {marker}) ;touch {marker}")
    assert code == 2  # argparse refused the action names
    assert not marker.exists()


def test_a_wildcard_in_expect_never_becomes_a_file_name(tmp_path: Path) -> None:
    """A file named `delete` in the repository must not turn `d*` into an
    expected action (/code-review of D106)."""
    (tmp_path / "project").mkdir()
    (tmp_path / "project" / "delete").write_text("")
    guard = f'case "$input" in *{PROBE}*|*.env*) exit 2;; esac'
    code, _, _ = _job(tmp_path, guard, "d*")
    assert code == 2  # d* is no action; globbed to `delete`, this guard passes: 0
