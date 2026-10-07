"""``paveo doctor``: the Claude Code hooks that cannot work as written (D86).

::

    paveo doctor [--settings FILE ...]

A hook a team wrote to stop ``rm -rf`` or a secret in a commit fails silently
when it is wrong: Claude Code runs it, it lets the call through, and nothing
says so. ``doctor`` reads the settings files Claude Code reads hooks from, and
any script a hook names, and names two faults it can prove from the text alone:

- **A variable Claude Code never sets.** A hook gets the call as JSON on stdin.
  ``$CLAUDE_TOOL_INPUT``, ``$TOOL_INPUT``, ``$CLAUDE_FILE_PATH`` and their
  relatives are not set (the hooks reference lists what is, and a hook run by
  Claude Code 2.1.292 saw all of them empty, 7 Oct 2026), so a hook that reads
  one always reads nothing. A name the hook assigns itself, or the settings'
  own ``env`` sets, is not reported.
- **``exit 1`` in a guard that never exits 2.** Only exit 2, or a printed
  decision, blocks a PreToolUse call; exit 1 is a non-blocking error and the
  call runs. Reported only when every command the hook runs is a shell tool
  known not to decide (``grep``, ``jq``, ``echo``...) or a shell script this
  read: a hook that runs Python, Node or any program it does not know may
  block from there, so it is left alone.

**Precision over recall.** A false "broken" costs the trust of exactly the
people who wrote a guard, so each check reports only what the text proves, and
anything else, a missing ``jq`` among them, goes unreported rather than guessed.

**It never runs a hook**, so it has no side effect, and **never prints a
command or a script's text**: a command can carry a token. It names where (file,
event, index, or the path of a script it could not read) and what to fix.
A file it cannot read, or a ``--settings`` file that is not there, is said to
be unread, never passed as clean, and the exit is 2; a fault found is 1;
nothing found is 0.

Opens no socket: it reads files.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import stat
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TextIO

# Claude Code sets no variable with these prefixes: the hooks reference lists the
# ones it does set (CLAUDE_PROJECT_DIR, CLAUDE_ENV_FILE, CLAUDE_PLUGIN_*, ...),
# and a hook it ran on 7 Oct 2026 saw no CLAUDE_TOOL_*, TOOL_* or CLAUDE_FILE_*.
# The prefixes, not a list of names, because the wrong names vary:
# CLAUDE_TOOL_INPUT_FILE_PATH, CLAUDE_TOOL_INPUT_command, TOOL_INPUT_FILE, ...
_NEVER_SET = r"(?:CLAUDE_TOOL_|CLAUDE_FILE_PATH|TOOL_INPUT|TOOL_NAME|TOOL_OUTPUT)\w*"
_SHELL_READ = re.compile(rf"\$(?:env:)?\{{?(?P<name>{_NEVER_SET})")
_PYTHON_READ = re.compile(
    rf"(?:environ(?:\.get)?\s*[\[(]|getenv\s*\()\s*[\"'](?P<name>{_NEVER_SET})[\"']"
)
_NODE_READ = re.compile(rf"process\.env(?:\.|\[\s*[\"'])(?P<name>{_NEVER_SET})")
_RUBY_READ = re.compile(rf"ENV(?:\[|\.fetch\()\s*[\"'](?P<name>{_NEVER_SET})[\"']")
_SHELL_SUFFIXES = {".sh", ".bash", ".zsh"}
# Each language read only for its own way of reading a variable: ``${X}`` in a
# JavaScript template literal is not an environment read. A shell text gets every
# way, since ``python3 -c`` and ``node -e`` put the others inside it.
_READS_BY_SUFFIX = {
    ".py": (_PYTHON_READ,),
    ".js": (_NODE_READ,),
    ".mjs": (_NODE_READ,),
    ".cjs": (_NODE_READ,),
    ".ts": (_NODE_READ,),
    ".rb": (_RUBY_READ,),
}
_SHELL_READS = (_SHELL_READ, _PYTHON_READ, _NODE_READ, _RUBY_READ)
_PROJECT, _HOME = "@@PROJECT@@", "@@HOME@@"
_SCRIPT = re.compile(
    r"[^\s\"'`|;&()<>=]+\.(?:sh|bash|zsh|py|js|mjs|cjs|ts|rb)(?![\w.])"
)
# Shell tools that cannot block a call themselves: a hook made only of these and
# of scripts this read decides by its own exits and output, all of them visible.
_KNOWN = frozenset(
    [
        "echo",
        "printf",
        "test",
        "[",
        "[[",
        "true",
        "false",
        ":",
        "exit",
        "return",
        "grep",
        "egrep",
        "fgrep",
        "jq",
        "sed",
        "awk",
        "tr",
        "cut",
        "head",
        "tail",
        "cat",
        "wc",
        "sort",
        "uniq",
        "tee",
        "basename",
        "dirname",
        "read",
        "local",
        "export",
        "set",
        "unset",
        "shift",
        "cd",
        "pwd",
        "git",
        "sleep",
        "date",
        "mkdir",
        "touch",
        "ls",
        "command",
        "type",
        "which",
        "bash",
        "sh",
        "zsh",
    ]
)
_KEYWORDS = frozenset(
    [
        "if",
        "then",
        "elif",
        "else",
        "fi",
        "do",
        "done",
        "while",
        "until",
        "case",
        "esac",
        "in",
        "!",
        "time",
    ]
)
_SEPARATOR = re.compile(r"\$\(|[;&|\n(){}`]")
_ASSIGNMENT = re.compile(r"\w+=\S*")
_EXIT_1 = re.compile(r"\bexit\s+1\b")
_BLOCKS = re.compile(
    r"\b(?:exit|return)\s+2\b|permissionDecision|\\?[\"']decision\\?[\"']"
)
_EXIT_UNKNOWN = re.compile(r"\bexit\s+[\"'$]")
_AT_COMMAND = r"(?:^|[;&|{(\n]|\b(?:then|do|else|while)\b)\s*"
_STDIN = re.compile(
    _AT_COMMAND + r"(?:IFS=\S*\s+)?read\s+(?:-[a-zA-Z]+\s+)*[A-Za-z_]"
    r"|\$\(\s*cat\s*\)|/dev/stdin|-t\s+0\b|sys\.stdin|process\.stdin"
    r"|readFileSync\(\s*0",
    re.MULTILINE,
)
_COMMENT = re.compile(r"\s*(?:#|//)")
_MAX_BYTES = 1 << 20


@dataclass(frozen=True)
class _Hook:
    where: str
    event: str
    command: str


@dataclass
class _Scripts:
    """Each script read once a run, and each one it could not read said once."""

    project: Path
    unread: list[str]
    texts: dict[Path, str | None] = field(default_factory=dict)

    def named_in(self, command: str) -> dict[str, tuple[Path, str | None]]:
        """Every script path ``command`` names, as written, with its text, or
        None when it is not a file this read."""
        found: dict[str, tuple[Path, str | None]] = {}
        for match in _SCRIPT.finditer(_marked(command)):
            word = match.group()
            path = Path(
                os.path.expanduser(
                    word.replace(_PROJECT, str(self.project)).replace(
                        _HOME, str(Path.home())
                    )
                )
            )
            if not path.is_absolute():
                path = self.project / path
            found[word] = (path, self._text(path))
        return found

    def _text(self, path: Path) -> str | None:
        if path not in self.texts:
            self.texts[path] = None
            if path.is_file():
                try:
                    self.texts[path] = _read(path).decode("utf-8", "replace")
                except OSError as e:
                    self.unread.append(f"{path}: could not read it ({_reason(e)})")
        return self.texts[path]


def command(
    files: Sequence[Path], *, project: Path, out: TextIO, named: bool = False
) -> int:
    """Report each hook in ``files`` that cannot work as written; see the module.

    ``named``: the files were asked for by name, so one that is not there is
    reported as unread rather than passed over."""
    hooks: list[_Hook] = []
    env: set[str] = set()
    unread: list[str] = []
    read = 0
    for path in files:
        settings = _settings(path, named=named, unread=unread)
        if settings is None:
            continue
        read += 1
        declared = settings.get("env")
        if isinstance(declared, dict):
            env.update(map(str, declared))
        found, odd = _hooks(path, settings.get("hooks") or {})
        hooks.extend(found)
        unread.extend(odd)
    scripts = _Scripts(project, unread)
    faults = 0
    for hook in hooks:
        problems = list(_problems(hook, env=env, scripts=scripts))
        for problem in problems:
            out.write(f"{_shown(hook.where)}: {problem}\n")
        faults += bool(problems)
    for line in unread:
        out.write(f"paveo: {_shown(line)}. Not checked.\n")
    if read == 0 and not unread:
        out.write(
            "paveo: no Claude Code settings file found, so nothing was checked:\n"
        )
        for path in files:
            out.write(f"  {_shown(str(path))}\n")
        return 0
    out.write(
        f"paveo: checked {len(hooks)} command hook{'s' * (len(hooks) != 1)} in "
        f"{read} file{'s' * (read != 1)}: {faults} cannot work as written.\n"
    )
    if unread:
        return 2
    return 1 if faults else 0


def _settings(
    path: Path, *, named: bool, unread: list[str]
) -> dict[str, object] | None:
    """The settings in ``path``, or None, with the reason added to ``unread``
    unless the file is simply absent and was not asked for by name."""
    try:
        if not stat.S_ISREG(path.stat().st_mode):
            unread.append(f"{path}: not a file")
            return None
        settings = json.loads(_read(path))
    except FileNotFoundError:
        if named:
            unread.append(f"{path}: no such file")
        return None
    except (OSError, ValueError, RecursionError) as e:
        unread.append(f"{path}: could not read it ({_reason(e)})")
        return None
    if not isinstance(settings, dict):
        unread.append(f"{path}: not a JSON object")
        return None
    return settings


def _hooks(path: Path, hooks: object) -> tuple[list[_Hook], list[str]]:
    found: list[_Hook] = []
    odd: list[str] = []
    if not isinstance(hooks, dict):
        return found, [f"{path}: hooks is not an object"]
    for event, groups in hooks.items():
        if not isinstance(groups, list):
            odd.append(f"{path}: {event} is not a list")
            continue
        for g, group in enumerate(groups):
            entries = group.get("hooks") if isinstance(group, dict) else None
            if not isinstance(entries, list):
                odd.append(f"{path}: {event}[{g}] has no list of hooks")
                continue
            for h, entry in enumerate(entries):
                where = f"{path}: {event}[{g}].hooks[{h}]"
                if not isinstance(entry, dict):
                    odd.append(f"{where} is not an object")
                elif entry.get("type") == "command":
                    command, args = entry.get("command"), entry.get("args", [])
                    if isinstance(command, str) and isinstance(args, list):
                        words = [command, *map(shlex.quote, map(str, args))]
                        found.append(_Hook(where, str(event), " ".join(words)))
                    else:
                        odd.append(f"{where} has no command")
    return found, odd


def _problems(hook: _Hook, *, env: set[str], scripts: _Scripts) -> Iterator[str]:
    named = scripts.named_in(hook.command)
    texts = [(_SHELL_READS, _uncommented(hook.command))] + [
        (_READS_BY_SUFFIX.get(path.suffix, _SHELL_READS), _uncommented(text))
        for path, text in named.values()
        if text is not None
    ]
    names = {
        match.group("name")
        for reads, text in texts
        for pattern in reads
        for match in pattern.finditer(text)
    }
    for name in sorted(names):
        if name in env or any(
            _assigns(name, text) or _falls_back(name, text) for _, text in texts
        ):
            continue
        yield (
            f"reads ${name}, which Claude Code never sets, so it always reads an "
            "empty value. The call arrives as JSON on stdin: read it from there, "
            f"e.g. jq -r .tool_input, instead of ${name}."
        )
    if hook.event == "PreToolUse" and _only_known(named, [t for _, t in texts]):
        text = "\n".join(t for _, t in texts)
        if (
            _EXIT_1.search(text)
            and not _BLOCKS.search(text)
            and not _EXIT_UNKNOWN.search(text)
        ):
            yield (
                "exits 1 and never 2, and prints no decision, so it cannot block "
                "a call: Claude Code treats exit 1 as a non-blocking error and "
                "runs the call. Exit 2 to block (stderr becomes the reason)."
            )


def _only_known(named: dict[str, tuple[Path, str | None]], texts: list[str]) -> bool:
    """Whether every command in ``texts`` is a known shell tool or a shell
    script this read, so nothing out of sight can exit 2 or print a decision."""
    shell = {
        word
        for word, (path, text) in named.items()
        if text is not None and path.suffix in _SHELL_SUFFIXES
    }
    if len(shell) != len(named):
        return False
    for text in texts:
        bare = re.sub(r"'[^']*'", "''", _marked(text))
        bare = re.sub(r"\"[^\"$`]*\"", '""', bare)
        for segment in _SEPARATOR.split(bare):
            words = [w.strip("\"'") for w in segment.split()]
            while words and (words[0] in _KEYWORDS or _ASSIGNMENT.fullmatch(words[0])):
                words.pop(0)
            if not words:
                continue
            first = words[0]
            if first not in _KNOWN and first not in shell:
                return False
    return True


def _marked(text: str) -> str:
    """``text`` with the project and home variables as markers, so a script path
    written ``"$CLAUDE_PROJECT_DIR"/x.sh`` is one word that can be resolved."""
    text = re.sub(r"\"?\$\{?CLAUDE_PROJECT_DIR\}?\"?", _PROJECT, text)
    return re.sub(r"\"?\$\{?HOME\}?\"?", _HOME, text)


def _assigns(name: str, text: str) -> bool:
    """Whether ``text`` gives ``name`` a value of its own: ``X=``, ``read X``,
    ``os.environ["X"] =``. ``X="${X:-}"`` reads the empty variable, so not."""
    for match in re.finditer(
        rf"(?:^|[\s;&|(\"'])(?:export\s+|local\s+|declare\s+(?:-\w+\s+)*)?"
        rf"{name}=(?P<value>[^\n;&|]*)",
        text,
    ):
        if not re.search(rf"\$\{{?{name}\b", match.group("value")):
            return True
    return bool(
        re.search(
            _AT_COMMAND + rf"(?:IFS=\S*\s+)?read\s+(?:-[a-zA-Z]+\s+)*"
            rf"(?:[A-Za-z_]\w*\s+)*{name}\b",
            text,
            re.MULTILINE,
        )
        or re.search(rf"environ\[\s*[\"']{name}[\"']\s*\]\s*=(?!=)", text)
    )


def _falls_back(name: str, text: str) -> bool:
    """Whether ``text`` tests ``name`` for empty and reads stdin: a hook that
    tries the variable first and the real input second works, so it is left
    alone. Found on a public repo by running this over 153 of them, 7 Oct."""
    tested = re.search(rf"-[nz]\s+\"?\$\{{?{name}\b|\$\{{{name}:[-=][^}}]", text)
    return bool(tested and _STDIN.search(text))


def _uncommented(text: str) -> str:
    """``text`` without its whole-line comments, which may name a variable to
    say it is not used. A comment after code stays: cutting at ``#`` would cut
    ``${x#y}`` and ``"#"`` too."""
    return "\n".join(line for line in text.splitlines() if not _COMMENT.match(line))


def _read(path: Path) -> bytes:
    # A bounded read, not a size check and then a read: the file can grow between.
    with path.open("rb") as file:
        data = file.read(_MAX_BYTES + 1)
    if len(data) > _MAX_BYTES:
        raise OSError(f"over {_MAX_BYTES} bytes")
    return data


def _reason(error: Exception) -> str:
    if isinstance(error, OSError):
        return error.strerror or str(error)
    if isinstance(error, RecursionError):
        return "nested too deeply"
    return "not valid JSON"


def _shown(text: str) -> str:
    """``text`` with control characters escaped: a settings file from a cloned
    repository chooses its event names and paths, and the terminal prints them."""
    return "".join(
        char if char.isprintable() else char.encode("unicode_escape").decode()
        for char in text
    )
