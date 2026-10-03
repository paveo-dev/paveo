"""The starter policies in ``src/paveo/starters/`` (D49, D50): each loads,
refuses what it says it refuses, and permits ordinary work.

A starter policy that refuses ordinary work gets deleted on the first day, so
the "permits" lists here matter as much as the "refuses" ones.
"""

from __future__ import annotations

import itertools
import json
import re
import string
from pathlib import Path

import pytest

from paveo._policy_document import load_file
from paveo.policy import Footprint, Policy, Recall

POLICIES = Path(__file__).parent.parent / "src" / "paveo" / "starters"


def starter(name: str) -> Policy:
    return load_file(POLICIES / f"{name}.json")


def test_every_starter_policy_loads() -> None:
    names = sorted(path.stem for path in POLICIES.glob("*.json"))
    assert names == ["claude-code", "codex", "cursor", "finance-ops", "support-agent"]
    for name in names:
        starter(name)


def bash(command: str) -> dict[str, object]:
    """Every field Claude Code's Bash tool sends, as of 2026-09-25 (D49)."""
    return {
        "command": command,
        "description": "run it",
        "timeout": 120000,
        "run_in_background": False,
        "dangerouslyDisableSandbox": False,
    }


@pytest.mark.parametrize(
    "command",
    [
        "rm -rf node_modules",
        "rm -fr /",
        "rm -Rf ./build",
        "cd /tmp && rm --recursive x",
        "git push --force origin main",
        "git push -f",
        "git reset --hard HEAD~3",
        "git clean -fdx",
        "psql -c 'DROP TABLE users'",
        "truncate table orders",
        "terraform destroy -auto-approve",
        "kubectl delete ns prod",
        "cat .paveo/audit.jsonl",
        "echo '{}' > .claude/settings.json",
        "paveo resume --dir .paveo",
        # The obvious ways around the first version (/code-review, D49).
        "cat > .pave?/policy.json <<EOF",
        "cd .claude && echo '{}' > settings.local.json",
        "cp /tmp/x ~/.claude/settings.json",
        # A literal `.claude` stays refused whatever follows it, except
        # `.claude.c...` and `.claude.a...` (claude.com, claude.ai addresses,
        # D54). The spellings below each got past a narrower draft of D54
        # (/code-review), so they are pinned.
        "echo x >> .claude/hooks/pre.sh",
        "echo '{}' > ~/.claude//settings.json",
        "cd ~/.claude && echo '{}' > s.json",
        "mv .claude .claude.bak",
        "mv '.claude' /tmp",
        "rm .claude/*",
        "cp /tmp/x .claude/sett*",
        "cp /tmp/x .claude/{a,b}",
        "tar cf - x | (cd .claude; tar xf -)",
        # Quoting or a variable inside the name hides `settings` from the first
        # pattern; the old blanket `\.claude\b` refused these, so these must too.
        "echo '{}' > .claude/'set'tings.json",
        'echo x > .claude/"hooks"/pre.sh',
        "cp /tmp/x .claude/$F",
        "cp /tmp/x .claude/se\\ttings.json",
        # Path spellings that got past the first draft (/code-review, D54).
        "echo x >> .claude/./hooks/pre.sh",
        "echo x >> ~/.claude/projects/../hooks/pre.sh",
        "cd ~/.claude/projects/.. && rm -r hooks",
        "cd ~/.claude/. && rm -r hooks",
        "cd ~/.claude/projects && rm -r ../hooks",
        "echo x >> .claude\\/hooks/pre.sh",
        "echo x >> .claude${X}/hooks/pre.sh",
        "echo x >> .claude$X/hooks/pre.sh",
        "cp /tmp/x ~/.claude.json",
        "echo '{}' > .mcp.json",
        "mkdir -p .claude/projects/a && cd .claude/projects/a",
        "cd ~/.claude/projects/-app/memory && ls",
        "find ~/.claude/projects -maxdepth 0 -execdir rm -r hooks \\;",
        "ls ~/.claude/",
        "cat ~/.claude/proj*/x",
        # Reading Claude Code's own sessions is refused too, for now: D54's
        # trigger to exempt it is a tester's replay showing it.
        "cat ~/.claude/projects/-app/memory/notes.md",
    ],
)
def test_claude_code_refuses_destructive_and_self_disabling_commands(
    command: str,
) -> None:
    denial = starter("claude-code").evaluate_tool("claude-code", "Bash", bash(command))
    assert denial is not None
    assert denial.rule == "Bash.command.not_matches"


@pytest.mark.parametrize(
    "command",
    [
        "ls -la",
        "rm notes.txt",
        "git push origin feature/x",
        "git status && git diff",
        "npm test",
        "python -m pytest -q",
        "paveo stop",
        # A web address is not the folder (D54).
        "curl -s https://code.claude.com/docs/en/hooks",
        "curl -s https://support.claude.ai/articles/1",
    ],
)
def test_claude_code_permits_ordinary_commands(command: str) -> None:
    policy = starter("claude-code")
    assert policy.evaluate_tool("claude-code", "Bash", bash(command)) is None


def test_claude_code_permits_a_bash_call_with_only_the_command() -> None:
    policy = starter("claude-code")
    assert policy.evaluate_tool("claude-code", "Bash", {"command": "ls"}) is None


# What `\.claude\b` refused before D54, as a regular expression rather than a
# copy of the old file: the blanket rule every change is measured against.
_BLANKET = re.compile(r"\.claude\b", re.IGNORECASE)


def test_the_only_thing_bash_no_longer_refuses_is_a_web_address() -> None:
    """Every pair of printable characters after `.claude`: anything the blanket
    rule refused and the starter lets through must be `.claude.c...` or
    `.claude.a...`, a web address and not the folder (D54). A narrowing nobody
    meant fails here."""
    policy = starter("claude-code")
    printable = [c for c in string.printable if c not in "\x0b\x0c"]
    loosened = {
        pair
        for pair in ("".join(p) for p in itertools.product(printable, repeat=2))
        for prefix in ("cat ", "cd ~/")
        if _BLANKET.search(f"{prefix}.claude{pair}")
        and policy.evaluate_tool("claude-code", "Bash", bash(f"{prefix}.claude{pair}"))
        is None
    }

    assert loosened == {".c", ".C", ".a", ".A"}
    # And at the very end of a command, where a pair does not reach.
    for ending in ("cat ~/.claude", "cat ~/.claude.", "cat ~/.claude/"):
        assert policy.evaluate_tool("claude-code", "Bash", bash(ending)) is not None


def path_call(tool: str, path: str) -> dict[str, object]:
    """A call to one of the three tools that write a file, at ``path``."""
    if tool == "NotebookEdit":
        # Its fields as its schema gave them on 2026-09-25 (D49).
        return {
            "notebook_path": path,
            "new_source": "x",
            "cell_id": "a",
            "cell_type": "code",
            "edit_mode": "replace",
        }
    if tool == "Edit":
        return {"file_path": path, "old_string": "a", "new_string": "b"}
    return {"file_path": path, "content": "{}"}


PATH_TOOLS = ("Write", "Edit", "NotebookEdit")


@pytest.mark.parametrize("tool", PATH_TOOLS)
@pytest.mark.parametrize(
    "path",
    [
        ".paveo/policy.json",
        "/home/me/project/.paveo/stop",
        ".claude/settings.json",
        "/Users/me/.claude/settings.local.json",
        ".claude/hooks/guard.sh",
        "/Users/me/.claude.json",
        # Everything under `.claude` runs or configures something (D54, round 4):
        # a plugin's hooks, an agent or a skill with hooks in its frontmatter.
        "/Users/me/.claude/plugins/x/hooks/hooks.json",
        "/Users/me/.claude/agents/x.md",
        "/p/.claude/skills/x/SKILL.md",
        "/p/.claude/commands/deploy.md",
        # An MCP server added here runs tools the hook's matcher never sees.
        "/p/.mcp.json",
        "/Users/me/.Claude/Hooks/pre.sh",
        # Spellings that got past earlier drafts (/code-review, D54).
        "/Users/me/.claude//hooks/pre.sh",
        "/p/.claude/./hooks/pre.sh",
        "/Users/me/.claude/plans/../hooks/pre.sh",
        "/home/u/.claude/plans" + "/" * 1100 + "../settings.json",
        "/p/.claude/plans/" + "./" * 520 + "../hooks/x",
        "/Users/me/.claude/plans/x\n/../../hooks/pre.sh",
        "../hooks/x",
        "..",
    ],
)
def test_claude_code_refuses_writing_its_own_guard(tool: str, path: str) -> None:
    policy = starter("claude-code")
    assert policy.evaluate_tool("claude-code", tool, path_call(tool, path)) is not None


@pytest.mark.parametrize("tool", PATH_TOOLS)
@pytest.mark.parametrize(
    "path",
    [
        "/p/src/app.py",
        "/p/analysis.ipynb",
        "/p/docs/v1..v2.md",
        # Claude Code's plan mode saves plans here through Write (21 of 687 real
        # writes, D54). A plan runs nothing, and a `..` cannot leave it.
        "/Users/me/.claude/plans/fix-the-importer.md",
    ],
)
def test_claude_code_permits_writing_the_project_and_plans(
    tool: str, path: str
) -> None:
    policy = starter("claude-code")
    assert policy.evaluate_tool("claude-code", tool, path_call(tool, path)) is None


def test_the_three_file_tools_share_one_list_of_refusals() -> None:
    """Three copies of one list drift when a fix reaches two of them (/code-review)."""
    tools = starter("claude-code").agents["claude-code"].tools
    lists = [
        [pattern.pattern for pattern in constraint.forbidden or ()]
        for tool in PATH_TOOLS
        for constraint in (tools[tool].constraints or {}).values()
        if constraint.forbidden
    ]
    assert len(lists) == 3
    assert lists[0] == lists[1] == lists[2]


# What the file tools refused before D54.
_OLD_PATH_RULE = re.compile(r"\.paveo|\.claude/(?:settings|hooks)", re.IGNORECASE)


def test_the_file_tools_refuse_everything_they_refused_before() -> None:
    """Every pair of printable characters around the spellings the old rule
    matched: nothing it refused may be let through now (D54)."""
    policy = starter("claude-code")
    printable = [c for c in string.printable if c not in "\x0b\x0c"]
    templates = (
        "/u/{a}.claude/settings{b}",
        "/u/.claude/hooks{a}{b}",
        "{a}{b}/.claude/hooks/x",
        "/u/.claude/plans/{a}{b}/.claude/settings.json",
        "/u/{a}{b}.paveo/x",
    )
    matched = dict.fromkeys(templates, 0)
    for a, b in itertools.product(printable, repeat=2):
        for template in templates:
            path = template.format(a=a, b=b)
            if _OLD_PATH_RULE.search(path):
                matched[template] += 1
                for tool in PATH_TOOLS:
                    call = path_call(tool, path)
                    assert policy.evaluate_tool("claude-code", tool, call), (tool, path)
    # A template the old rule never matched would check nothing (/code-review).
    assert all(matched.values()), matched


def test_no_starter_pattern_is_slow_on_a_crafted_megabyte() -> None:
    """Claude Code lets a call through when the hook times out, so a pattern made
    slow is a pattern switched off. The first version's `.*` took over a minute
    here (/code-review, D49); the guard also refuses at its own deadline."""
    import time  # noqa: PLC0415

    policy = starter("claude-code")
    for filler in (
        "rm ",
        "git push ",
        "a",
        ".claude/projects/",
        "cd ",
        "..",
        ".claude/projects/x ",
        "/",
        "/.",
        ".claude/plans/",
    ):
        crafted = "ls; " + filler * ((1 << 20) // len(filler) - 2)
        started = time.perf_counter()
        policy.evaluate_tool("claude-code", "Bash", bash(crafted[: 1 << 20]))
        assert time.perf_counter() - started < 1.0, filler
        # The same megabyte as a path, through each tool whose path is checked.
        for tool in PATH_TOOLS:
            started = time.perf_counter()
            policy.evaluate_tool(
                "claude-code", tool, path_call(tool, crafted[: 1 << 20])
            )
            assert time.perf_counter() - started < 1.0, (tool, filler)


def test_claude_code_permits_writing_the_project() -> None:
    policy = starter("claude-code")
    write = {"file_path": "src/app.py", "content": "print(1)"}
    edit = {
        "file_path": "README.md",
        "old_string": "a",
        "new_string": "b",
        "replace_all": True,
    }
    assert policy.evaluate_tool("claude-code", "Write", write) is None
    assert policy.evaluate_tool("claude-code", "Edit", edit) is None


def test_claude_code_refuses_every_other_tool() -> None:
    """Deny by default: the hook's matcher decides which tools it sees at all."""
    policy = starter("claude-code")
    denial = policy.evaluate_tool("claude-code", "WebFetch", {"url": "x"})
    assert denial is not None
    assert denial.reason == "tool_not_allowed"


@pytest.mark.parametrize(
    ("agent", "tool", "arguments", "rule"),
    [
        (
            "support-agent",
            "refund",
            {"order_id": "ORD-123456", "amount_usd": "250.00", "currency": "USD"},
            "refund.amount_usd.max",
        ),
        ("support-agent", "delete_account", {}, "delete_account"),
        (
            "finance-ops",
            "issue_payment",
            {"payee_id": "V-123456", "amount_usd": "10", "currency": "EUR"},
            "issue_payment.currency.equals",
        ),
        ("finance-ops", "change_bank_details", {}, "change_bank_details"),
    ],
)
def test_the_library_starters_refuse_what_they_say(
    agent: str, tool: str, arguments: dict[str, object], rule: str
) -> None:
    denial = starter(agent).evaluate_tool(agent, tool, arguments)
    assert denial is not None
    assert denial.rule == rule


def test_the_library_starters_permit_ordinary_work() -> None:
    support = starter("support-agent")
    refund = {"order_id": "ORD-123456", "amount_usd": "42.00", "currency": "EUR"}
    fresh = Recall(now=0.0, footprints=frozenset(), calls=())
    looked_up, _ = support.remember(
        "support-agent", "lookup_order", {"order_id": "ORD-123456"}, fresh
    )
    after = Recall(now=0.0, footprints=looked_up, calls=())
    assert (
        support.evaluate_tool("support-agent", "refund", refund, recall=after) is None
    )
    # An order it never looked up is refused (D58): Salus's headline case.
    unmet = support.evaluate_tool("support-agent", "refund", refund, recall=fresh)
    assert unmet is not None
    assert unmet.reason == "requires_unmet"
    finance = starter("finance-ops")
    assert (
        finance.evaluate_tool(
            "finance-ops",
            "issue_payment",
            {"payee_id": "V-123456", "amount_usd": "999.99", "currency": "USD"},
            recall=fresh,
        )
        is None
    )


@pytest.mark.parametrize(
    ("agent", "tool", "arguments", "rewritten"),
    [
        (
            "support-agent",
            "refund",
            {"order_id": "ORD-123456", "amount_usd": "42.00", "currency": "EUR"},
            {"amount_usd": "42.0", "currency": "USD", "reason": "asked again"},
        ),
        (
            "finance-ops",
            "issue_payment",
            {"payee_id": "V-123456", "amount_usd": "999.99", "currency": "USD"},
            {"amount_usd": "999.990", "memo": "retry"},
        ),
    ],
)
def test_the_library_starters_refuse_the_same_money_twice_in_an_hour(
    agent: str,
    tool: str,
    arguments: dict[str, object],
    rewritten: dict[str, object],
) -> None:
    """A loop that retries a refund or a payment pays it again unless something
    stops it. Only who is paid is compared: values are compared as sent, so an
    amount spelt another way, another currency or a reworded reason must not make
    the same money look new. The refusal says what it compared, never that this
    exact call was made, since it was not (D77)."""
    policy = starter(agent)
    footprints: frozenset[Footprint] = frozenset()
    if tool == "refund":
        footprints, _ = policy.remember(
            agent,
            "lookup_order",
            {"order_id": "ORD-123456"},
            Recall(now=0.0, footprints=frozenset(), calls=()),
        )
    first = Recall(now=0.0, footprints=footprints, calls=())
    assert policy.evaluate_tool(agent, tool, arguments, recall=first) is None
    _, call = policy.remember(agent, tool, arguments, first)
    assert call is not None
    again = {**arguments, **rewritten}
    soon = Recall(now=60.0, footprints=footprints, calls=(call,))
    denial = policy.evaluate_tool(agent, tool, again, recall=soon)
    assert denial is not None
    assert denial.reason == "repeated"
    assert f"the same {'order_id' if tool == 'refund' else 'payee_id'}" in denial.remedy
    later = Recall(now=3601.0, footprints=footprints, calls=(call,))
    assert policy.evaluate_tool(agent, tool, again, recall=later) is None


# --------------------------------------------------------------------------
# Codex and Cursor (B2c, D57): the same list of destructive commands
# --------------------------------------------------------------------------

SHELLS = (("claude-code", "Bash"), ("codex", "Bash"), ("cursor", "Shell"))


def shell_refusals(name: str, tool: str) -> list[str]:
    document = json.loads((POLICIES / f"{name}.json").read_text("utf-8"))
    [rule] = [r for r in document["agents"][0]["tools"]["allow"] if r["name"] == tool]
    refusals: list[str] = rule["constraints"]["command"]["not_matches"]
    return refusals


def test_every_agent_refuses_the_same_destructive_commands() -> None:
    """Three copies of one list (a JSON file cannot share one): an edit to one
    that misses the others would leave an agent less guarded (Rule 2)."""
    destructive = shell_refusals("claude-code", "Bash")[:9]
    assert destructive[-1] == "\\bmkfs"
    for name, tool in SHELLS:
        assert shell_refusals(name, tool)[:9] == destructive, name
        assert "\\bpaveo\\s+resume" in shell_refusals(name, tool), name


@pytest.mark.parametrize(
    "command",
    [
        "rm -rf build",
        "git " + "push --force origin main",
        "git reset --hard HEAD~3",
        "cat .paveo/audit.jsonl",
        "paveo resume",
    ],
)
@pytest.mark.parametrize(("name", "tool"), SHELLS[1:])
def test_codex_and_cursor_refuse_what_claude_code_refuses(
    name: str, tool: str, command: str
) -> None:
    assert starter(name).evaluate_tool(name, tool, {"command": command}) is not None


@pytest.mark.parametrize(
    ("name", "tool", "command"),
    [
        ("codex", "Bash", "echo '{}' > .codex/hooks.json"),
        ("codex", "Bash", "cat ~/.codex/config.toml"),
        ("codex", "Bash", "cd ~/.codex && ls"),
        ("cursor", "Shell", "echo '{}' > .cursor/hooks.json"),
        ("cursor", "Shell", "cd ~/.cursor"),
        # Past the folder pattern by quoting, caught by the file name.
        ("codex", "Bash", "cd .cod''ex && : > hooks.json"),
        ("cursor", "Shell", "cd .cur''sor && : > hooks.json"),
    ],
)
def test_each_agent_refuses_its_own_hook_folder(
    name: str, tool: str, command: str
) -> None:
    assert starter(name).evaluate_tool(name, tool, {"command": command}) is not None


@pytest.mark.parametrize(
    "command",
    ["git status", "npm test", "pytest -q", "ls -la src", "git diff HEAD~1"],
)
@pytest.mark.parametrize(("name", "tool"), SHELLS[1:])
def test_codex_and_cursor_permit_ordinary_commands(
    name: str, tool: str, command: str
) -> None:
    assert starter(name).evaluate_tool(name, tool, {"command": command}) is None


@pytest.mark.parametrize(("name", "tool"), SHELLS[1:])
def test_no_codex_or_cursor_pattern_is_slow_on_a_crafted_megabyte(
    name: str, tool: str
) -> None:
    import time  # noqa: PLC0415

    policy = starter(name)
    for filler in ("rm ", "git " + "push ", ".codex", ".cursor", "/", "..", "a"):
        crafted = ("ls; " + filler * ((1 << 20) // len(filler)))[: 1 << 20]
        started = time.perf_counter()
        policy.evaluate_tool(name, tool, {"command": crafted})
        assert time.perf_counter() - started < 1.0, filler
    other, field, extra = {
        "codex": ("apply_patch", "paths", {"command": ""}),
        "cursor": ("Write", "file_path", {"content": ""}),
    }[name]
    for filler in ("/", "..", "../", "\n", ".paveo"):
        crafted = (filler * ((1 << 20) // len(filler)))[: 1 << 20]
        started = time.perf_counter()
        policy.evaluate_tool(name, other, {field: crafted, **extra})
        assert time.perf_counter() - started < 1.0, filler


@pytest.mark.parametrize("tool", PATH_TOOLS)
@pytest.mark.parametrize(
    "path",
    [".venv/lib/python3.12/site-packages/paveo/cli.py", "venv/lib/site-packages/a.pth"],
)
def test_claude_code_refuses_writing_the_guards_installed_code(
    tool: str, path: str
) -> None:
    """/security-review, D57: a project virtualenv holds paveo itself."""
    policy = starter("claude-code")
    assert policy.evaluate_tool("claude-code", tool, path_call(tool, path)) is not None
