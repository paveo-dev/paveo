"""Replay and learn: the Claude Code sessions already on disk, judged by a policy.

::

    paveo replay claude-code [PATH ...] [--dir .paveo] [--agent claude-code]
    paveo learn claude-code --from-history [PATH ...] [--dir .paveo] [--agent ...]

Claude Code writes every session to ``~/.claude/projects/*/<id>.jsonl``: each
tool call with its input, and each model response with its model and token
usage. **The format is undocumented.** It was read from the files themselves on
2026-09-26, so a line this does not understand is skipped and counted, never
guessed at. Three facts about it decide the numbers (D53):

- One response is written as several lines, one per content block, sharing a
  ``message.id``. The usage repeats on each and the output count grows until the
  last, so a response is counted once, at its largest output count.
- A resumed session copies earlier lines into a new file, so responses are
  counted once per ``message.id`` and tool calls once per ``id``, across files.
- Subagents write their own files under ``<session>/subagents/``. They are read
  too: their calls were made and paid for all the same.

**A tool name in a session file was chosen by the model**, and a model under
injection can name a tool that does not exist, carrying anything in its name.
Claude Code refuses to run it, but the name is on disk. So ``learn`` takes names
only from calls whose result came back without ``is_error``: a tool that ran.
That proves the tool is real, not that every argument name was declared: an MCP
server may ignore an extra one, so ``learn`` prints each name it adds (D53).

**Nothing is stored and no payload is printed.** The summary carries counts,
sums, and names the policy or the price table already holds. A tool or model
name neither holds is counted, not printed (D26). ``learn`` writes tool and
argument names only, never a value, and only names shaped like identifiers.

**A file path is judged as written only.** The guard also judges the file a path
really reaches (D79), but that is a fact about the disk at the moment of the
call, and the disk replay reads is the disk of today: a link made or removed
since would change the verdict on a call that has already happened.

Opens no socket: it reads files and writes one.
"""

from __future__ import annotations

import copy
import json
import re
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from decimal import Decimal, localcontext
from importlib import resources
from pathlib import Path, PurePath
from typing import TextIO

from ._anthropic import served, usage_classes
from ._harnesses import CLAUDE_CODE
from ._licence import apply, plan_in
from ._policy_document import load_document, load_file
from ._setup import POLICY, _replace
from .budget import _ARITHMETIC, _WINDOWS
from .errors import ConfigError, PaveoError, PricingUnknown
from .policy import BudgetPolicy, Policy
from .prices import _LISTINGS, _PriceTable

# Claude Code's own tool names and MCP names (`mcp__server__tool`) fit; a name
# that does not was not chosen by the harness, and is not written into a policy.
_TOOL_NAME = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}")
# Grep takes `-n` and `-i`. A leading `__` is Claude Code's own marker
# (`__unparsedToolInput`), not an argument.
_ARGUMENT_NAME = re.compile(r"(?!__)[A-Za-z0-9_-]{1,64}")
_LEARNED = "policy.learned.json"


@dataclass(frozen=True, slots=True)
class _ModelCall:
    model: str
    usage: Mapping[str, object]
    at: datetime
    session: str


@dataclass(slots=True)
class History:
    """What the session files hold, deduplicated. Read by one thread, then dropped."""

    files: int = 0
    unreadable: int = 0
    not_understood: int = 0
    tool_calls: list[tuple[str, str, Mapping[str, object]]] = field(
        default_factory=list
    )
    model_calls: dict[str, _ModelCall] = field(default_factory=dict)
    # Tool calls whose result came back without an error: they really ran.
    ran: set[str] = field(default_factory=set)
    _tool_ids: set[str] = field(default_factory=set)


def read_history(paths: Sequence[Path], *, within: Path | None) -> History:
    """Every tool call and model response in ``paths`` (files, or folders searched
    for ``*.jsonl``). With ``within``, only what was done inside that folder.

    A file that cannot be read is counted, and the summary says so: one owned by
    root in another project must not stop every replay, and a replay that skipped
    it silently would report less as though it were all (Rule 3).
    """
    history = History()
    for path in _session_files(paths):
        try:
            with path.open("rb") as lines:
                for line in lines:
                    _read_line(line, history, within)
        except OSError:
            history.unreadable += 1
            continue
        history.files += 1
    return history


def _session_files(paths: Sequence[Path]) -> list[Path]:
    found: list[Path] = []
    for path in paths:
        if path.is_dir():
            found.extend(sorted(path.rglob("*.jsonl")))
        elif path.is_file():
            found.append(path)
        else:
            raise ConfigError(
                f"{path} is not a file or folder that can be read.",
                remedy="pass Claude Code's session files, or ~/.claude/projects.",
            )
    return found


def _read_line(line: bytes, history: History, within: Path | None) -> None:
    try:
        record = json.loads(line)
    except (ValueError, RecursionError):
        history.not_understood += 1
        return
    if not isinstance(record, dict):
        history.not_understood += 1
        return
    # Responses carry tool calls and usage; the turns after them, the results.
    # An API error Claude Code wrote in place of a response cost nothing.
    kind = record.get("type")
    if kind not in {"assistant", "user"} or record.get("isApiErrorMessage") is True:
        return
    if within is not None and not _inside(record.get("cwd"), within):
        return
    message = record.get("message")
    if kind == "user":
        history.ran.update(_results(message))
        return
    found = _model_call(record, message)
    if found is None:
        history.not_understood += 1
        return
    tools = _tool_calls(message)
    if tools is None:
        # Its tool calls are not read, but its tokens were still spent.
        history.not_understood += 1
        tools = []
    key, call = found
    earlier = history.model_calls.get(key)
    if earlier is None or _output(call) > _output(earlier):
        # Admitted when it started: the first line's time, whichever line counts.
        at = call.at if earlier is None else min(call.at, earlier.at)
        history.model_calls[key] = replace(call, at=at)
    for tool_id, name, arguments in tools:
        if tool_id not in history._tool_ids:
            history._tool_ids.add(tool_id)
            history.tool_calls.append((tool_id, name, arguments))


def _results(message: object) -> list[str]:
    """The ids of the tool calls this turn reports as having run without error."""
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, list):
        return []
    return [
        block["tool_use_id"]
        for block in content
        if isinstance(block, dict)
        and block.get("type") == "tool_result"
        and isinstance(block.get("tool_use_id"), str)
        and block.get("is_error") is not True
    ]


def _inside(cwd: object, folder: Path) -> bool:
    if not isinstance(cwd, str):
        return False
    where = PurePath(cwd)
    return where == folder or folder in where.parents


def _model_call(
    record: dict[str, object], message: object
) -> tuple[str, _ModelCall] | None:
    if not isinstance(message, dict):
        return None
    key, model, usage = message.get("id"), message.get("model"), message.get("usage")
    stamp, session = record.get("timestamp"), record.get("sessionId")
    if not (
        isinstance(key, str)
        and isinstance(model, str)
        and isinstance(usage, dict)
        and isinstance(stamp, str)
        and isinstance(session, str)
    ):
        return None
    try:
        at = datetime.fromisoformat(stamp)
    except ValueError:
        return None
    if at.tzinfo is None:
        return None
    return key, _ModelCall(model=model, usage=usage, at=at, session=session)


def _tool_calls(message: object) -> list[tuple[str, str, Mapping[str, object]]] | None:
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, list):
        return None
    calls: list[tuple[str, str, Mapping[str, object]]] = []
    for block in content:
        if not isinstance(block, dict) or block.get("type") != "tool_use":
            continue
        tool_id, name, arguments = (
            block.get("id"),
            block.get("name"),
            block.get("input"),
        )
        if not (
            isinstance(tool_id, str)
            and isinstance(name, str)
            and isinstance(arguments, dict)
        ):
            return None
        calls.append((tool_id, name, arguments))
    return calls


def _output(call: _ModelCall) -> int:
    count = call.usage.get("output_tokens")
    return count if isinstance(count, int) else -1


def from_history(  # noqa: PLR0913 - keyword-only; the clock and stream are injected (Rule 14)
    command: str,
    *,
    paths: Sequence[Path],
    within: Path | None,
    directory: Path,
    agent: str,
    now: datetime,
    out: TextIO,
) -> int:
    """``replay`` or ``learn``. Returns the exit code: 0, or 1 if it could not.

    A failure prints its type, and for Paveo's own errors their message, which
    never holds payload (§8): an arbitrary exception's message may quote a line.
    """
    try:
        history = read_history(paths, within=within)
        if not history.files:
            out.write(
                f"paveo: found no Claude Code session files there that could be "
                f"read ({history.unreadable:,} could not).\n"
            )
            return 1
        out.write(
            f"paveo: read {history.files:,} session files"
            f"{f', keeping what was done in {within}' if within else ''}.\n"
        )
        if command == "replay":
            # Under the plan the guard would apply, so the two agree (D62).
            today = now.date()
            policy = apply(
                load_file(directory / POLICY),
                plan_in(directory, today=today),
                today=today,
            )
            summary = replay(history, policy=policy, agent=agent, now=now)
            out.write("\n".join(summary) + "\n")
            return 0
        return _learn(history, directory=directory, agent=agent, out=out)
    except PaveoError as e:
        out.write(f"paveo: {e}\n")
    except Exception as e:  # reported by type only: its message may quote a line
        out.write(f"paveo: {command} failed ({type(e).__name__}).\n")
    return 1


def _learn(history: History, *, directory: Path, agent: str, out: TextIO) -> int:
    """Write ``policy.learned.json`` beside the policy, never over it."""
    if directory.is_symlink() or not directory.is_dir():
        out.write(
            f"paveo: {directory} is not a folder paveo set up. Run "
            f"`paveo init claude-code` first, or pass --dir.\n"
        )
        return 1
    current = directory / POLICY
    if current.is_file():
        load_file(current)  # the checks, and the size limit, before it is used
        base = json.loads(current.read_text(encoding="utf-8"))
    else:
        base = json.loads(
            resources.files("paveo")
            .joinpath("starters", f"{CLAUDE_CODE.name}.json")
            .read_text(encoding="utf-8")
        )
    learned, changes = learn(history, base=base, agent=agent)
    target = directory / _LEARNED
    if target.is_symlink():
        out.write(f"paveo: {target} is a link, so it is not written through.\n")
        return 1
    _replace(target, json.dumps(learned, indent=2) + "\n")
    out.writelines(f"  {change}\n" for change in changes)
    out.write(
        f"paveo: wrote {target} ({len(changes)} additions"
        f"{', none needed' if not changes else ''}). Every name added is permitted "
        f"unchecked. Read it, then move it over {current} to use it. The hook "
        f"`paveo init` installs sees only Bash, Write, Edit and NotebookEdit: a "
        f"rule for any other tool takes effect only if you widen its matcher.\n"
    )
    return 0


# -- replay ------------------------------------------------------------------


def replay(history: History, *, policy: Policy, agent: str, now: datetime) -> list[str]:
    """The one-screen summary: what the policy would have refused, and what the
    model calls cost at list prices. Every line is safe to print (module doc)."""
    if agent not in policy.agents:
        raise ConfigError(
            f"the policy declares no agent {agent!r}.",
            remedy="pass the --agent your hook uses.",
        )
    lines = _tool_lines(history, policy, agent)
    lines += _model_lines(history, policy, agent, now)
    if history.not_understood:
        lines.append(f"Lines not understood, skipped: {history.not_understood:,}")
    if history.unreadable:
        lines.append(f"Files that could not be read, left out: {history.unreadable:,}")
    lines.append("Nothing was stored, and no prompt, command or argument is shown.")
    return lines


def _tool_lines(history: History, policy: Policy, agent: str) -> list[str]:
    allowed, refused, unnamed = 0, Counter[str](), Counter[str]()
    unremembered = 0
    for _, name, arguments in history.tool_calls:
        if not policy.declares_tool(agent, name):
            unnamed[name] += 1
            continue
        # With no memory: the session files do not say which calls ran together
        # or when, so a requires, rate or repeat rule cannot be replayed. Such
        # calls are counted apart rather than as refusals (D59).
        denial = policy.evaluate_tool(agent, name, arguments, recall=None)
        if denial is None:
            allowed += 1
        elif denial.reason == "memory_unavailable":
            unremembered += 1
        else:
            refused[_which(denial.rule, policy, agent, name, arguments)] += 1
    checked = allowed + sum(refused.values())
    lines = [
        f"Tool calls: {len(history.tool_calls):,}",
        f"  judged by the policy: {checked:,}, allowed {allowed:,}, "
        f"refused {sum(refused.values()):,}",
    ]
    lines += [f"    {count:>7,}  {rule}" for rule, count in refused.most_common()]
    if unnamed:
        lines.append(
            f"  not judged: {sum(unnamed.values()):,} calls to {len(unnamed)} tools "
            f"the policy does not name (paveo learn lists them)"
        )
    if unremembered:
        lines.append(
            f"  not judged: {unremembered:,} calls to tools with a requires, rate "
            f"or repeat rule, which need to know what ran before them"
        )
    return lines


def _which(
    rule: str, policy: Policy, agent: str, tool: str, arguments: Mapping[str, object]
) -> str:
    """A ``not_matches`` refusal, with the pattern that fired: tuning a policy is
    knowing which one refuses ordinary work. The pattern is the policy's own text,
    so it is safe to print; the value it found is not printed."""
    suffix = ".not_matches"
    if not (rule.startswith(f"{tool}.") and rule.endswith(suffix)):
        return rule
    argument = rule[len(tool) + 1 : -len(suffix)]
    constraints = policy.agents[agent].tools[tool].constraints or {}
    constraint = constraints.get(argument)
    value = arguments.get(argument)
    for pattern in (constraint.forbidden or ()) if constraint else ():
        if isinstance(value, str) and pattern.search(value):
            return f"{rule}  {pattern.pattern}"
    return rule


def _model_lines(
    history: History, policy: Policy, agent: str, now: datetime
) -> list[str]:
    priced, unpriced, searches, unplaced = _price(history, policy, now)
    lines = [f"Model calls: {len(history.model_calls):,}"]
    if not priced:
        return [*lines, f"  none priced ({unpriced:,} unpriced)"]
    with localcontext(_ARITHMETIC):
        total = sum((cost for _, cost in priced), Decimal(0))
        by_model: dict[str, list[Decimal]] = defaultdict(list)
        by_day: dict[str, Decimal] = defaultdict(Decimal)
        by_session: dict[str, Decimal] = defaultdict(Decimal)
        for call, cost in priced:
            by_model[call.model].append(cost)
            by_day[call.at.astimezone(UTC).strftime("%d %b %Y")] += cost
            by_session[call.session] += cost
        models = sorted(
            (
                (sum(costs, Decimal(0)), model, len(costs))
                for model, costs in by_model.items()
            ),
            reverse=True,
        )
    first, last = priced[0][0].at, priced[-1][0].at
    day, day_cost = max(by_day.items(), key=lambda item: item[1])
    lines += [
        f"  API-equivalent spend at list prices: {_usd(total)}, "
        f"{first:%d %b} to {last:%d %b %Y}",
        *(
            f"    {model:<30} {_usd(cost):>12} {count:>7,} calls"
            for cost, model, count in models
        ),
        f"  costliest day {day}, {_usd(day_cost)}; "
        f"costliest session {_usd(max(by_session.values()))}",
    ]
    if unplaced:
        lines.append(
            f"  {unplaced:,} calls do not say where they ran, so they are priced as "
            f"paveo reserves them, at the US rate: up to 10% high"
        )
    if unpriced or searches:
        lines.append(
            f"  left out of the spend, not priced: {unpriced:,} calls to models or "
            f"token classes the table does not carry, {searches:,} web searches "
            f"and fetches"
        )
    lines.append(_ceiling_line(priced, policy.agents[agent].budget, agent))
    return lines


def _price(
    history: History, policy: Policy, now: datetime
) -> tuple[list[tuple[_ModelCall, Decimal]], int, int, int]:
    """Each call's cost at today's list price, oldest first.

    Anthropic's table only: Claude Code's usage is Anthropic's shape. An unknown
    model, speed or token class is counted as unpriced, never guessed. Where the
    record does not say where a call ran, it is priced as the library reserves it,
    at the dearer rate (D53).
    """

    table = _PriceTable(_LISTINGS, declared=policy.prices, now=lambda: now)
    unpriced, searches, unplaced = 0, 0, 0
    priced: list[tuple[_ModelCall, Decimal]] = []
    for call in sorted(history.model_calls.values(), key=lambda c: c.at):
        classes = usage_classes(call.usage)
        used = classes.pop("server_tool_use", 0)
        searches += used if isinstance(used, int) else 0
        speed = call.usage.get("speed")
        try:
            rates = table.resolve(call.model, {"speed": speed} if speed else {})
            # A class the row does not price would be charged at a guess and
            # stop the table for every call after it (§4.8.3). Asked first, so it
            # is counted instead, and the table never stops (/code-review, D53).
            if any(name not in rates.per_token for name in classes):
                unpriced += 1
                continue
            charge = table.actual(rates, classes, served(call.usage))
        except (PricingUnknown, ValueError):
            unpriced += 1
            continue
        unplaced += "inference_geo=unset" in charge.rate_key
        priced.append((call, charge.cost))
    return priced, unpriced, searches, unplaced


def _ceiling_line(
    priced: list[tuple[_ModelCall, Decimal]], budget: BudgetPolicy | None, agent: str
) -> str:
    if budget is None:
        return (
            f"  agents[{agent!r}] has no budget: add one to see what a ceiling "
            f"would have refused."
        )
    spent: dict[object, Decimal] = defaultdict(Decimal)
    refused, stopped = 0, Decimal(0)
    with localcontext(_ARITHMETIC):
        for call, cost in priced:
            period = _period(budget.period, call)
            if spent[period] + cost > budget.limit_usd:
                refused += 1
                stopped += cost
            else:
                spent[period] += cost
    # Not a floor, and not the ledger's answer: the ledger reserves each call's
    # worst case, which can refuse a call this admits and admit ones this refuses
    # (/code-review, D53). Nothing on disk records max_tokens, so what each call
    # came to is what there is to replay.
    return (
        f"  a {_usd(budget.limit_usd)} per {budget.period} ceiling, counting each "
        f"call at what it cost, would have refused {refused:,} calls, "
        f"{_usd(stopped)} of this spend"
    )


def _period(period: str, call: _ModelCall) -> object:
    """The ledger's own window for this call, or its session (§4.7)."""
    truncate = _WINDOWS[period]
    return call.session if truncate is None else truncate(call.at.astimezone(UTC))


def _usd(amount: Decimal) -> str:
    return f"${amount:,.2f}"


# -- learn -------------------------------------------------------------------


def learn(
    history: History, *, base: Mapping[str, object], agent: str
) -> tuple[dict[str, object], list[str]]:
    """``base`` with the name of every tool that ran, and of every argument it
    ran with, added; and a line for each addition.

    Only added, never removed or loosened: a constraint already written stays, a
    denied tool stays denied, and a tool already allowed unchecked is left so.
    An added name is permitted **unchecked**, because a name is all it knows.
    The result is validated as a policy before it is returned.
    """
    document = copy.deepcopy(dict(base))
    allow, denied = _allow_list(document, agent)
    seen: dict[str, set[str]] = defaultdict(set)
    for tool_id, name, arguments in history.tool_calls:
        if tool_id in history.ran and _TOOL_NAME.fullmatch(name):
            seen[name].update(a for a in arguments if _ARGUMENT_NAME.fullmatch(a))
    changes = []
    for name in sorted(seen.keys() - denied):
        entry = next((e for e in allow if _mapping(e).get("name") == name), None)
        if entry is None:
            allow.append(
                {"name": name, "constraints": {a: {} for a in sorted(seen[name])}}
            )
            changes.append(f"allow {name}({', '.join(sorted(seen[name]))})")
            continue
        constraints = _mapping(entry).get("constraints")
        if not isinstance(constraints, dict):
            continue
        for argument in sorted(seen[name] - constraints.keys()):
            constraints[argument] = {}
            changes.append(f"permit {name}.{argument}")
    load_document(document, source="the learned policy")
    return document, changes


def _allow_list(
    document: dict[str, object], agent: str
) -> tuple[list[object], set[str]]:
    """The agent's ``tools.allow`` list, created if absent, and its denied names."""
    agents = document.get("agents")
    entries = agents if isinstance(agents, list) else []
    entry = next(
        (a for a in entries if isinstance(a, dict) and a.get("id") == agent), None
    )
    if entry is None:
        raise ConfigError(
            f"the policy declares no agent {agent!r}.",
            remedy="pass the --agent your hook uses.",
        )
    tools = entry.setdefault("tools", {})
    allow = tools.setdefault("allow", []) if isinstance(tools, dict) else None
    if not isinstance(allow, list):
        raise ConfigError(
            f"agents[{agent!r}].tools is not in a shape learn can add to.",
            remedy='write "tools": {"allow": [...]}.',
        )
    denied = tools.get("deny")
    names = denied if isinstance(denied, list) else []
    return allow, {name for name in names if isinstance(name, str)}


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, dict) else {}
