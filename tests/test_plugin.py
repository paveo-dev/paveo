"""The company rollout: the managed hook and the Claude Code plugin (D89).

Both run one script, ``rollout/guard``, that runs the guard and refuses every
call it cannot judge. These tests hold the shipped files to the code they stand
for, and run the script the way the agents do: as a shell command, with the call
on stdin and only the environment the agent gives it.
"""

from __future__ import annotations

import io
import json
import os
import shutil
import subprocess
import sys
import tomllib
from datetime import date
from pathlib import Path

import pytest

from conftest import make_clock
from paveo._harnesses import CLAUDE_CODE, CODEX, Harness
from paveo._licence import plan_beside
from paveo.cli import guard, stop
from paveo.errors import ConfigError
from test_cli import SOURCE

REPO = Path(__file__).parent.parent
PLUGIN = REPO / "plugins" / "paveo"
SCRIPT = PLUGIN / "scripts" / "guard"
STARTERS = REPO / "src" / "paveo" / "starters"
STARTER = STARTERS / "claude-code.json"
ROLLOUT = REPO / "rollout"
# The plugin hook's command as Claude Code runs it, through a shell.
HOOK = json.loads((PLUGIN / "hooks" / "hooks.json").read_text("utf-8"))["hooks"][
    "PreToolUse"
][0]["hooks"][0]["command"]
MANAGED = "/bin/sh /opt/paveo/guard"


def python_shim(where: Path, *, body: str | None = None) -> Path:
    """A ``python`` that runs this repository's paveo, and only when asked with
    ``-I``, as the script must ask: a test that passes proves it isolated."""
    where.mkdir(parents=True, exist_ok=True)
    shim = where / "python"
    shim.write_text(
        body
        or (
            '#!/bin/sh\n[ "$1" = "-I" ] || { echo "not isolated" >&2; exit 3; }\n'
            f'shift\nPYTHONPATH="{SOURCE}" exec "{sys.executable}" "$@"\n'
        ),
        encoding="utf-8",
    )
    shim.chmod(0o755)
    return shim


def bash(command: str) -> dict[str, object]:
    return {"session_id": "s1", "tool_name": "Bash", "tool_input": {"command": command}}


def run(
    argv: list[str], call: dict[str, object], env: dict[str, str], cwd: Path
) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(  # noqa: S603 - the rollout's own script, as an agent runs it
        argv,
        input=json.dumps(call).encode(),
        capture_output=True,
        env={"PATH": "/usr/bin:/bin", **env},
        cwd=cwd,
        timeout=60,
        check=False,
    )


# --------------------------------------------------------------------------
# The shipped files agree with the library
# --------------------------------------------------------------------------


def test_the_plugin_and_the_managed_hook_run_the_same_script() -> None:
    assert SCRIPT.read_bytes() == (ROLLOUT / "guard").read_bytes()
    assert os.access(SCRIPT, os.X_OK)
    assert os.access(ROLLOUT / "guard", os.X_OK)


def test_the_plugin_ships_the_starter_policy_unchanged() -> None:
    """A starter fix that missed the plugin would leave its users on old rules."""
    assert (PLUGIN / "policy.json").read_bytes() == STARTER.read_bytes()


def test_the_team_policy_is_the_two_starters_together() -> None:
    """Two agents, which every plan covers, so nothing is refused as plan_limit."""
    policy = json.loads((ROLLOUT / "policy.json").read_text("utf-8"))
    starters = [
        json.loads((STARTERS / f"{name}.json").read_text("utf-8"))["agents"][0]
        for name in ("claude-code", "codex")
    ]
    assert policy["agents"] == starters


def matches_harness(harness: Harness, event: str, matcher: object) -> None:
    [expected] = [each for each in harness.events if each.name == event]
    assert matcher == expected.matcher


def test_the_plugin_hook_reaches_every_governed_tool_through_sh() -> None:
    hooks = json.loads((PLUGIN / "hooks" / "hooks.json").read_text("utf-8"))
    [group] = hooks["hooks"]["PreToolUse"]
    matches_harness(CLAUDE_CODE, "PreToolUse", group["matcher"])
    [handler] = group["hooks"]
    # Through sh, so a lost executable bit cannot become a hook that fails to
    # start, which Claude Code treats as no objection.
    assert HOOK == '/bin/sh "${CLAUDE_PLUGIN_ROOT}/scripts/guard" claude-code --plugin'
    assert handler["timeout"] > 4  # above the guard's own deadline


def test_the_claude_code_managed_settings_reach_every_governed_tool() -> None:
    settings = json.loads(
        (ROLLOUT / "claude-code-managed-settings.json").read_text("utf-8")
    )
    [group] = settings["hooks"]["PreToolUse"]
    matches_harness(CLAUDE_CODE, "PreToolUse", group["matcher"])
    assert group["hooks"][0]["command"] == f"{MANAGED} claude-code"


def test_the_codex_requirements_reach_every_governed_tool() -> None:
    requirements = tomllib.loads(
        (ROLLOUT / "codex-requirements.toml").read_text("utf-8")
    )
    assert requirements["features"]["hooks"] is True
    [group] = requirements["hooks"]["PreToolUse"]
    matches_harness(CODEX, "PreToolUse", group["matcher"])
    assert group["hooks"][0]["command"] == f"{MANAGED} codex"


def test_the_marketplace_lists_the_plugin_by_a_path_inside_it() -> None:
    marketplace = json.loads(
        (REPO / ".claude-plugin" / "marketplace.json").read_text("utf-8")
    )
    assert marketplace["name"] == "paveo"
    assert marketplace["owner"]["name"]
    [entry] = marketplace["plugins"]
    assert entry["name"] == "paveo"
    assert (REPO / entry["source"]).resolve() == PLUGIN.resolve()
    manifest = json.loads(
        (PLUGIN / ".claude-plugin" / "plugin.json").read_text("utf-8")
    )
    assert manifest["name"] == "paveo"


# --------------------------------------------------------------------------
# The managed hook: everything it trusts inside one root-owned /opt/paveo
# --------------------------------------------------------------------------

# The test's files are the test runner's, not root's, so the copy installed here
# also accepts that owner; the folders above tmp_path are the system's. As
# shipped, only root is accepted.
OWNER = str(os.getuid())  # find -user takes a number: no passwd entry needed
CFG = "home = {home}\nversion = 3.12.4\ninclude-system-site-packages = false\n"


def managed_install(tmp_path: Path, *, owner: str = OWNER) -> Path:
    """/opt/paveo as docs/ROLLOUT.md lays it out: the script, the policy, a
    standalone Python, and a venv made from it."""
    opt = tmp_path.resolve() / "opt" / "paveo"
    opt.mkdir(parents=True)
    script = (ROLLOUT / "guard").read_text("utf-8")
    assert script.count("! -user root") == 1
    accepted = "-user root" if owner == "root" else f"( -user root -o -user {owner} )"
    script = script.replace("-user root", accepted)
    # The walk above /opt/paveo stops at tmp_path, the test's own root: above
    # it, Linux keeps /tmp world-writable by design, which the shipped script
    # rightly refuses for a real install.
    walk = 'while [ "$folder" != / ]; do'
    assert script.count(walk) == 1
    script = script.replace(walk, f'while [ "$folder" != "{opt.parent.parent}" ]; do')
    (opt / "guard").write_text(script, "utf-8")
    shutil.copy2(ROLLOUT / "policy.json", opt / "policy.json")
    python = opt / "python"
    (python / "lib" / "python3.12").mkdir(parents=True)
    shim = python_shim(python / "bin")
    shim.rename(python / "bin" / "python3.12")
    venv = opt / "venv"
    (venv / "bin").mkdir(parents=True)
    (venv / "bin" / "python").symlink_to("../../python/bin/python3.12")
    (venv / "lib" / "python3.12" / "site-packages").mkdir(parents=True)
    (venv / "pyvenv.cfg").write_text(CFG.format(home=python / "bin"), "utf-8")
    # Whatever the umask, as root would install it: writable by its owner only.
    for path in (opt.parent, opt, *opt.rglob("*")):
        if not path.is_symlink():
            runnable = path.is_dir() or path.name in {"guard", "python3.12"}
            path.chmod(0o755 if runnable else 0o644)
    return opt


def run_managed(
    tmp_path: Path, opt: Path, agent: str, call: dict[str, object], **env: str
) -> subprocess.CompletedProcess[bytes]:
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    return run(
        ["/bin/sh", str(opt / "guard"), agent],
        call,
        {"HOME": str(home), **env},
        tmp_path,
    )


def refused(tmp_path: Path, opt: Path) -> bytes:
    done = run_managed(tmp_path, opt, "claude-code", bash("ls"))
    assert done.returncode == 2, done.stderr
    return done.stderr


def test_the_managed_hook_judges_both_agents(tmp_path: Path) -> None:
    opt = managed_install(tmp_path)
    codex = {**bash("rm -rf src"), "hook_event_name": "PreToolUse"}

    codex_refused = run_managed(tmp_path, opt, "codex", codex)
    claude_refused = run_managed(tmp_path, opt, "claude-code", bash("rm -rf src"))
    allowed = run_managed(tmp_path, opt, "claude-code", bash("git status"))

    assert codex_refused.returncode == 2, codex_refused.stderr
    assert b"Refused by policy" in codex_refused.stderr
    assert claude_refused.returncode == 2, claude_refused.stderr
    assert allowed.returncode == 0, allowed.stderr
    log = tmp_path / "home" / ".paveo" / "managed" / "audit.jsonl"
    lines = log.read_text("utf-8").splitlines()
    assert [json.loads(line)["decision"] for line in lines] == ["deny", "deny", "allow"]


def test_the_managed_hook_never_runs_a_paveo_from_home_or_path(tmp_path: Path) -> None:
    """The person and their agent can replace ~/.paveo or anything on PATH; the
    managed hook runs only the paveo in /opt/paveo (security review, D89)."""
    opt = managed_install(tmp_path)
    (opt / "venv" / "bin" / "python").unlink()
    planted = tmp_path / "planted"
    python_shim(planted, body="#!/bin/sh\nexit 0\n")
    python_shim(tmp_path / "home" / ".paveo" / "bin", body="#!/bin/sh\nexit 0\n")

    done = run_managed(
        tmp_path,
        opt,
        "claude-code",
        bash("rm -rf src"),
        PATH=f"{planted}:/usr/bin:/bin",
    )

    assert done.returncode == 2
    assert b"not installed" in done.stderr


def test_the_managed_hook_turns_any_other_exit_into_a_refusal(tmp_path: Path) -> None:
    """A paveo that crashes with 1 would let the call through; the script says 2."""
    opt = managed_install(tmp_path)
    (opt / "python" / "bin" / "python3.12").write_text("#!/bin/sh\nexit 1\n", "utf-8")

    refused(tmp_path, opt)


def test_the_managed_hook_refuses_an_agent_it_does_not_guard(tmp_path: Path) -> None:
    """Cursor is not in the rollout: its guard has not run in a real session."""
    opt = managed_install(tmp_path)
    for agent in ("", "cursor"):
        done = run_managed(tmp_path, opt, agent, {})
        assert done.returncode == 2
        assert b"names no agent" in done.stderr


@pytest.mark.skipif(os.getuid() == 0, reason="as root the files really are root's")
def test_the_managed_hook_refuses_files_the_person_owns(tmp_path: Path) -> None:
    """As shipped, the script wants root's files only."""
    opt = managed_install(tmp_path, owner="root")

    assert b"is not owned by root" in refused(tmp_path, opt)


def test_the_managed_hook_refuses_any_file_in_it_others_can_write(
    tmp_path: Path,
) -> None:
    """python.org leaves os.py writable by the admin group; inside /opt/paveo
    every file is checked, the standard library included (D89)."""
    opt = managed_install(tmp_path)
    module = opt / "python" / "lib" / "python3.12" / "os.py"
    module.write_text("", "utf-8")
    module.chmod(0o664)

    stderr = refused(tmp_path, opt)

    assert b"os.py is not owned by root, or others can write" in stderr
    assert b"chown -R root:wheel" in stderr


def test_the_managed_hook_refuses_a_folder_above_it_others_can_write(
    tmp_path: Path,
) -> None:
    """Whoever can write a folder above /opt/paveo can swap it (D89)."""
    opt = managed_install(tmp_path)
    opt.parent.chmod(0o775)

    assert b"/opt is not owned by root" in refused(tmp_path, opt)


def test_the_managed_hook_refuses_a_python_outside_it(tmp_path: Path) -> None:
    """A Python anywhere else is one this hook cannot vouch for (D89)."""
    opt = managed_install(tmp_path)
    elsewhere = tmp_path.resolve() / "elsewhere" / "bin"
    elsewhere.mkdir(parents=True)
    (opt / "venv" / "pyvenv.cfg").write_text(CFG.format(home=elsewhere), "utf-8")

    assert b"Unpack a standalone Python" in refused(tmp_path, opt)


def test_the_managed_hook_refuses_a_home_elsewhere_inside_it(tmp_path: Path) -> None:
    """Python climbs from home for its standard library: home must be the
    standalone build's bin, or it could find one outside (/code-review, D89)."""
    opt = managed_install(tmp_path)
    (opt / "venv" / "pyvenv.cfg").write_text(
        CFG.format(home=opt / "venv" / "bin"), "utf-8"
    )

    assert b"Unpack a standalone Python" in refused(tmp_path, opt)


def test_the_managed_hook_refuses_a_link_leading_outside_it(tmp_path: Path) -> None:
    opt = managed_install(tmp_path)
    outside = tmp_path.resolve() / "outside"
    python_shim(outside, body="#!/bin/sh\nexit 0\n")
    (opt / "venv" / "bin" / "python").unlink()
    (opt / "venv" / "bin" / "python").symlink_to(outside / "python")

    assert b"outside" in refused(tmp_path, opt)


def test_the_managed_hook_refuses_a_link_climbing_out_with_dot_dot(
    tmp_path: Path,
) -> None:
    """A relative link with .. is resolved by its real folder, not as text."""
    opt = managed_install(tmp_path)
    outside = tmp_path.resolve() / "opt" / "evil"
    python_shim(outside, body="#!/bin/sh\nexit 0\n")
    (opt / "venv" / "bin" / "python").unlink()
    (opt / "venv" / "bin" / "python").symlink_to("../../../evil/python")

    assert b"outside" in refused(tmp_path, opt)


def test_the_managed_hook_refuses_a_home_reached_through_a_link(
    tmp_path: Path,
) -> None:
    """A link in home could be swapped between the check and Python starting."""
    opt = managed_install(tmp_path)
    (opt / "linked").symlink_to(opt / "python")
    (opt / "venv" / "pyvenv.cfg").write_text(
        CFG.format(home=opt / "linked" / "bin"), "utf-8"
    )

    assert b"Unpack a standalone Python" in refused(tmp_path, opt)


@pytest.mark.parametrize(
    "config",
    [
        "home = {home}\nversion = 3.12.4\n",
        "home = {home}\ninclude-system-site-packages = true\n",
        "home = {home}\n Include-System-Site-Packages = TRUE\r\n",
        "home = {home}\nhome = {home}\ninclude-system-site-packages = false\n",
        "version = 3.12.4\ninclude-system-site-packages = false\n",
    ],
)
def test_the_managed_hook_refuses_a_venv_it_cannot_vouch_for(
    tmp_path: Path, config: str
) -> None:
    """The system's site-packages unless told "false" (Python's default), a key
    named twice, or no home: each refuses rather than passing unchecked."""
    opt = managed_install(tmp_path)
    (opt / "venv" / "pyvenv.cfg").write_text(
        config.format(home=opt / "python" / "bin"), "utf-8"
    )

    refused(tmp_path, opt)


def test_the_managed_hook_reads_a_carriage_return_as_python_does(
    tmp_path: Path,
) -> None:
    opt = managed_install(tmp_path)
    (opt / "venv" / "pyvenv.cfg").write_text(
        CFG.format(home=opt / "python" / "bin").replace("\n", "\r\n"), "utf-8"
    )

    done = run_managed(tmp_path, opt, "claude-code", bash("ls"))

    assert done.returncode == 0, done.stderr


def test_the_managed_hook_refuses_a_second_pyvenv_cfg(tmp_path: Path) -> None:
    """Python reads venv/bin/pyvenv.cfg first; only the checked one may exist."""
    opt = managed_install(tmp_path)
    (opt / "venv" / "bin" / "pyvenv.cfg").write_text("", "utf-8")

    assert b"bin/pyvenv.cfg exists" in refused(tmp_path, opt)


def test_the_managed_hook_refuses_any_link_in_it_leading_outside(
    tmp_path: Path,
) -> None:
    """A root-owned link in the standard library to a folder the person owns
    would let them add code there (/code-review, D89)."""
    opt = managed_install(tmp_path)
    theirs = tmp_path.resolve() / "theirs"
    theirs.mkdir()
    (opt / "python" / "lib" / "python3.12" / "extra").symlink_to(theirs)

    assert b"outside" in refused(tmp_path, opt)


def test_the_managed_hook_refuses_an_absolute_link_through_a_folder_outside(
    tmp_path: Path,
) -> None:
    """A target written through a link outside could be swapped after the
    check, even if it leads inside now (/code-review, D89)."""
    opt = managed_install(tmp_path)
    bridge = tmp_path.resolve() / "bridge"
    bridge.symlink_to(opt / "python" / "bin")
    (opt / "venv" / "bin" / "python").unlink()
    (opt / "venv" / "bin" / "python").symlink_to(bridge / "python3.12")

    assert b"points outside" in refused(tmp_path, opt)


def test_the_managed_hook_starts_python_with_a_cleared_environment(
    tmp_path: Path,
) -> None:
    """LD_PRELOAD and its kind would load code from outside /opt/paveo into
    Python, and -I does not stop them (/code-review, D89)."""
    opt = managed_install(tmp_path)
    shim = opt / "python" / "bin" / "python3.12"
    body = shim.read_text("utf-8").split("\n", 1)[1]
    shim.write_text(
        '#!/bin/sh\n[ -z "${LEAKED:-}" ] || '
        '{ echo "environment leaked" >&2; exit 0; }\n' + body,
        "utf-8",
    )

    done = run_managed(tmp_path, opt, "claude-code", bash("rm -rf src"), LEAKED="1")

    assert done.returncode == 2, done.stderr
    assert b"environment leaked" not in done.stderr


@pytest.mark.parametrize("target", ["..", "x/../..", "../../.."])
def test_the_managed_hook_refuses_a_link_climbing_out_through_dot_dot(
    tmp_path: Path, target: str
) -> None:
    """A target ending in .. points above the folder it is written in."""
    opt = managed_install(tmp_path)
    (opt / "x").mkdir(mode=0o755)
    (opt / "up").symlink_to(target)

    assert b"outside" in refused(tmp_path, opt)


@pytest.mark.parametrize("command", ["mkdir", "find", "awk", "dirname"])
def test_the_managed_hook_ignores_shell_functions_from_the_environment(
    tmp_path: Path, command: str
) -> None:
    """macOS /bin/sh is bash, which loads functions from BASH_FUNC_<name>%%: a
    function standing in for a command must not turn a refusal into exit 0
    (/code-review, D89)."""
    opt = managed_install(tmp_path)

    done = run_managed(
        tmp_path,
        opt,
        "claude-code",
        bash("rm -rf src"),
        **{f"BASH_FUNC_{command}%%": "() { exit 0; }"},
    )

    assert done.returncode == 2, done.stderr


def test_a_function_named_echo_cannot_silence_a_refusal(tmp_path: Path) -> None:
    """refuse() prints with echo: on a path that refuses before Python, a
    function standing in for echo must not turn the refusal into exit 0."""
    opt = managed_install(tmp_path)
    (opt / "policy.json").chmod(0o666)

    done = run_managed(
        tmp_path,
        opt,
        "claude-code",
        bash("ls"),
        **{"BASH_FUNC_echo%%": "() { exit 0; }"},
    )

    assert done.returncode == 2, done.stderr
    assert b"policy.json is not owned by root" in done.stderr


def test_the_managed_hook_refuses_when_its_clean_rerun_cannot_start(
    tmp_path: Path,
) -> None:
    """The re-run is a child, so any failure to start it, here a script path
    that cannot be opened, refuses rather than ending in 126 or 127, which
    every agent lets through (/code-review, D89)."""
    opt = managed_install(tmp_path)
    script = (opt / "guard").read_text("utf-8")

    done = run(
        ["/bin/sh", "-c", script, str(tmp_path / "gone" / "guard"), "claude-code"],
        bash("ls"),
        {"HOME": str(tmp_path / "home")},
        tmp_path,
    )

    assert done.returncode == 2, done.stderr


@pytest.mark.skipif(not Path("/bin/dash").exists(), reason="no dash here")
def test_the_managed_hook_starts_under_dash(tmp_path: Path) -> None:
    """Debian's and Ubuntu's /bin/sh starts the hook. The hook then runs itself
    again through /bin/sh, so on macOS the rest runs under bash; on Linux CI,
    where /bin/sh is dash, every test here runs it all under dash."""
    opt = managed_install(tmp_path)
    home = tmp_path / "home"
    home.mkdir()

    def under_dash(command: str) -> int:
        return run(
            ["/bin/dash", str(opt / "guard"), "claude-code"],
            bash(command),
            {"HOME": str(home)},
            tmp_path,
        ).returncode

    assert under_dash("rm -rf src") == 2
    assert under_dash("git status") == 0


def test_a_symbolic_link_is_held_to_its_owner_not_its_mode(tmp_path: Path) -> None:
    """Linux makes every link 0777: a venv's links must not be refused for it."""
    opt = managed_install(tmp_path)
    (opt / "venv" / "bin" / "python3").symlink_to("python")

    assert run_managed(tmp_path, opt, "claude-code", bash("ls")).returncode == 0


def test_the_check_the_rollout_guide_gives_answers_as_it_says(tmp_path: Path) -> None:
    """docs/ROLLOUT.md's check: the test call is refused, naming the input."""
    opt = managed_install(tmp_path)

    done = run_managed(tmp_path, opt, "claude-code", {"paveo_selftest": True})

    assert done.returncode == 2
    assert b"is not a tool call this guard reads" in done.stderr
    guide = (REPO / "docs" / "ROLLOUT.md").read_text("utf-8")
    assert "is not a tool call this guard reads" in guide


def test_the_panic_button_the_guide_gives_stops_the_managed_hook(
    tmp_path: Path,
) -> None:
    """`paveo stop --dir ~/.paveo/managed --policy /opt/paveo/policy.json`: the
    first version refused, the policy not being in --dir (/code-review, D89)."""
    opt = managed_install(tmp_path)
    data = tmp_path / "home" / ".paveo" / "managed"
    data.mkdir(parents=True)

    assert stop(data, out=io.StringIO(), policy=opt / "policy.json") == 0
    done = run_managed(tmp_path, opt, "claude-code", bash("ls"))

    assert done.returncode == 2
    assert b"stop" in done.stderr.lower()


# --------------------------------------------------------------------------
# The plugin: the policy it ships, paveo at ~/.paveo
# --------------------------------------------------------------------------


def run_plugin(
    tmp_path: Path, call: dict[str, object], **env: str
) -> subprocess.CompletedProcess[bytes]:
    return run(
        ["/bin/sh", "-c", HOOK],
        call,
        {
            "HOME": str(tmp_path / "home"),
            "CLAUDE_PLUGIN_ROOT": str(PLUGIN),
            "CLAUDE_PLUGIN_DATA": str(tmp_path / "data"),
            "CLAUDE_PROJECT_DIR": str(tmp_path),
            **env,
        },
        tmp_path,
    )


def test_the_plugin_refuses_rm_rf_and_lets_ordinary_work_through(
    tmp_path: Path,
) -> None:
    python_shim(tmp_path / "home" / ".paveo" / "bin")

    refused = run_plugin(tmp_path, bash("rm -rf src"))
    allowed = run_plugin(tmp_path, bash("git status"))

    assert refused.returncode == 2, refused.stderr
    assert allowed.returncode == 0, allowed.stderr
    log = (tmp_path / "data" / "audit.jsonl").read_text("utf-8").splitlines()
    assert [json.loads(line)["decision"] for line in log] == ["deny", "allow"]


def test_the_plugin_keeps_its_environment(tmp_path: Path) -> None:
    """A seatbelt against mistakes, not a sealed boundary: a ~/.paveo Python
    that needs a variable such as LD_LIBRARY_PATH to start still gets it."""
    python_shim(
        tmp_path / "home" / ".paveo" / "bin",
        body=f'#!/bin/sh\n[ "${{NEEDED:-}}" = yes ] || exit 3\nshift\n'
        f'PYTHONPATH="{SOURCE}" exec "{sys.executable}" "$@"\n',
    )

    done = run_plugin(tmp_path, bash("git status"), NEEDED="yes")

    assert done.returncode == 0, done.stderr


def test_the_plugin_refuses_every_call_when_paveo_is_not_installed(
    tmp_path: Path,
) -> None:
    """Even with a paveo on PATH: only ~/.paveo is run, never PATH."""
    (tmp_path / "home").mkdir()
    planted = tmp_path / "planted"
    python_shim(planted, body="#!/bin/sh\nexit 0\n")

    done = run_plugin(tmp_path, bash("git status"), PATH=f"{planted}:/usr/bin:/bin")

    assert done.returncode == 2
    assert b"pip install paveo" in done.stderr


def test_the_plugin_refuses_without_its_paths_or_home(tmp_path: Path) -> None:
    python_shim(tmp_path / "home" / ".paveo" / "bin")
    for missing in ("CLAUDE_PLUGIN_DATA", "HOME"):
        done = run_plugin(tmp_path, bash("git status"), **{missing: ""})
        assert done.returncode == 2, missing


# --------------------------------------------------------------------------
# guard --policy, and the licence key it reads
# --------------------------------------------------------------------------


def decide(directory: Path, policy: Path, command: str) -> tuple[int, str]:
    stderr = io.StringIO()
    code = guard(
        CLAUDE_CODE,
        directory=directory,
        agent="claude-code",
        stdin=io.BytesIO(json.dumps(bash(command)).encode()),
        stdout=io.StringIO(),
        stderr=stderr,
        stopped=lambda: False,
        now=make_clock(),
        policy=policy,
    )
    return code, stderr.getvalue()


def two_folders(tmp_path: Path) -> tuple[Path, Path]:
    root, data = tmp_path / "root", tmp_path / "data"
    root.mkdir()
    data.mkdir()
    (root / "policy.json").write_bytes(STARTER.read_bytes())
    return root, data


def test_guard_reads_the_named_policy_and_logs_in_its_own_folder(
    tmp_path: Path,
) -> None:
    root, data = two_folders(tmp_path)

    assert decide(data, root / "policy.json", "rm -rf src")[0] == 2
    assert decide(data, root / "policy.json", "ls")[0] == 0
    assert len((data / "audit.jsonl").read_text("utf-8").splitlines()) == 2
    assert not (root / "audit.jsonl").exists()


def test_guard_reads_a_key_beside_the_policy_first(tmp_path: Path) -> None:
    """A key a company ships with its policy counts; one that cannot be read
    refuses every call rather than guessing a plan."""
    root, data = two_folders(tmp_path)
    (root / "licence.key").write_text("not a key", "utf-8")

    code, stderr = decide(data, root / "policy.json", "ls")

    assert code == 2
    assert "could not decide" in stderr


def test_guard_falls_back_to_the_key_in_its_folder(tmp_path: Path) -> None:
    """With no key beside the policy, the one in --dir counts, as before
    --policy: a key only raises how many agents a policy covers (D89)."""
    root, data = two_folders(tmp_path)
    (data / "licence.key").write_text("not a key", "utf-8")

    assert decide(data, root / "policy.json", "ls")[0] == 2


def test_evidence_still_falls_back_to_the_key_in_its_folder(tmp_path: Path) -> None:
    """`paveo evidence --policy archive/...` is a person's own command: the key
    in --dir counts when none is beside the policy, as before --policy."""
    root, data = two_folders(tmp_path)
    (data / "licence.key").write_text("not a key", "utf-8")

    with pytest.raises(ConfigError):
        plan_beside(root, data, today=date(2026, 10, 12))


def test_the_panic_button_refuses_a_folder_that_does_not_exist(
    tmp_path: Path,
) -> None:
    """A mistyped --dir must not get a stop file no guard reads (D49)."""
    opt = managed_install(tmp_path)
    out = io.StringIO()
    typo = tmp_path / "home" / ".paveo" / "manged"

    assert stop(typo, out=out, policy=opt / "policy.json") == 1
    assert "does not exist" in out.getvalue()
    assert not typo.exists()


def test_a_trial_key_in_the_persons_folder_does_not_count(tmp_path: Path) -> None:
    """An unsigned trial written to ~/.paveo/managed, and restarted by its date,
    must not lift plan_limit (/code-review, D89)."""
    root, data = two_folders(tmp_path)
    (data / "licence.key").write_text("paveo-trial.2026-10-12", "utf-8")

    assert plan_beside(root, data, today=date(2026, 10, 12)).name == "developer"


@pytest.mark.parametrize("named", [False, True])
def test_the_panic_button_says_when_dir_is_a_file(tmp_path: Path, named: bool) -> None:
    root, _ = two_folders(tmp_path)
    notes = tmp_path / "notes.txt"
    notes.write_text("", "utf-8")
    out = io.StringIO()

    policy = root / "policy.json" if named else None
    assert stop(notes, out=out, policy=policy) == 1
    assert "is not a folder" in out.getvalue()


def test_paveo_runs_as_a_module_the_way_the_hook_starts_it(tmp_path: Path) -> None:
    """The hook runs `python -I -m paveo.cli`, never the console script, so the
    module must run the command line when run as __main__."""
    root, data = two_folders(tmp_path)

    done = subprocess.run(  # noqa: S603 - this repository's own module
        [
            sys.executable,
            "-m",
            "paveo.cli",
            "guard",
            "claude-code",
            "--dir",
            str(data),
            "--policy",
            str(root / "policy.json"),
        ],
        input=json.dumps(bash("rm -rf src")).encode(),
        capture_output=True,
        env={"PYTHONPATH": SOURCE, "PATH": "/usr/bin:/bin"},
        timeout=60,
        check=False,
    )

    assert done.returncode == 2, done.stderr
    assert b"Refused by policy" in done.stderr
