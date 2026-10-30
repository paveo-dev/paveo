"""``paveo doctor``: the Claude Code hooks that cannot work as written (D86).

::

    paveo doctor [--settings FILE ...]

A hook a team wrote to stop ``rm -rf`` or a secret in a commit fails silently
when it is wrong: Claude Code runs it, it lets the call through, and nothing
says so. ``doctor`` reads the settings files Claude Code reads hooks from, and
any script a hook names, and names two faults it can prove from the text alone:

- **A variable Claude Code does not set.** A hook gets the call as JSON on stdin.
  ``$CLAUDE_TOOL_INPUT``, ``$TOOL_INPUT``, ``$CLAUDE_FILE_PATH`` and their
  relatives are not set: the hooks reference has never listed them (archived
  copies from launch week, July 2025, on), and a hook run by Claude Code
  2.1.292 saw all of them empty, 7 Oct 2026. A hook that reads one always reads
  nothing. A name the hook assigns itself, or the settings'
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
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TextIO

from . import _matcher_trial

# Claude Code sets no variable with these prefixes: the hooks reference lists the
# ones it does set (CLAUDE_PROJECT_DIR, CLAUDE_ENV_FILE, CLAUDE_PLUGIN_*, ...),
# and a hook it ran on 7 Oct 2026 saw no CLAUDE_TOOL_*, TOOL_* or CLAUDE_FILE_*.
# The prefixes, not a list of names, because the wrong names vary:
# CLAUDE_TOOL_INPUT_FILE_PATH, CLAUDE_TOOL_INPUT_command, TOOL_INPUT_FILE, ...
# CLAUDE_BASH_COMMAND by name, not CLAUDE_BASH_ as a prefix: the environment
# variables page documents CLAUDE_BASH_MAINTAIN_PROJECT_WORKING_DIR (read 15 Oct
# 2026), and a real variable reported as unset is the false alarm this avoids.
_NEVER_SET = (
    r"(?:CLAUDE_TOOL_|CLAUDE_FILE_PATH|CLAUDE_BASH_COMMAND|TOOL_INPUT|TOOL_NAME"
    r"|TOOL_OUTPUT)\w*"
)
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
_PROJECT, _TOP, _HOME = "@@PROJECT@@", "@@TOP@@", "@@HOME@@"
# The ways a hook names the project, the git top level and home. ``$(git
# rev-parse --show-toplevel)`` passed a public repo's broken hook as clean, 11 Oct.
_GIT_TOP = (
    r"\$\(\s*git\s+rev-parse\s+--show-toplevel(?:[^()]|\$\([^()]*\))*\)"
    r"|`\s*git\s+rev-parse\s+--show-toplevel[^`]*`"
)
_MARKS = (
    (r"\$\{CLAUDE_PROJECT_DIR(?::-[^}]*)?\}|\$CLAUDE_PROJECT_DIR\b", _PROJECT),
    (_GIT_TOP, _TOP),
    (r"\$\{HOME\}|\$HOME\b", _HOME),
)
# A quoted string, scanned left to right so each closing quote ends its own
# string: ``"[^"$]*"`` alone paired the end of ``"in $(pwd)"`` with the start of
# the next string and hid the command between them.
_QUOTED = re.compile(r"'[^']*'|\"(?:[^\"\\]|\\.)*\"")
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
    r"|\$\(\s*cat(?:\s+-)?\s*(?:\)|2>|\|\|)|/dev/stdin|-t\s+0\b|sys\.stdin"
    r"|process\.stdin"
    r"|readFileSync\(\s*0",
    re.MULTILINE,
)
_COMMENT = re.compile(r"\s*(?:#|//)")
# Paveo sends nothing home (locked decision #2), so the only way to hear from a
# person who tried it is to ask; only after doctor actually checked a hook (D86).
_TELL_US = (
    "paveo: tried it? Tell us what worked and what didn't: "
    "https://github.com/paveo-dev/paveo/discussions\n"
)
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
    failed: set[Path] = field(default_factory=set)

    def named_in(self, command: str) -> dict[str, tuple[Path, str | None]]:
        """Every script path ``command`` names, as written, with its text, or
        None when it is not a file this read."""
        found: dict[str, tuple[Path, str | None]] = {}
        for match in _SCRIPT.finditer(_marked(command)):
            word = match.group()
            resolved = word.replace(_PROJECT, str(self.project))
            if _TOP in resolved:
                resolved = resolved.replace(_TOP, str(self._top()))
            path = Path(os.path.expanduser(resolved.replace(_HOME, str(Path.home()))))
            if not path.is_absolute():
                path = self.project / path
            found[word] = (path, self._text(path))
        return found

    def _top(self) -> Path:
        """What ``git rev-parse --show-toplevel`` prints for the project: the
        nearest directory holding ``.git``, or the project when none does."""
        for directory in (self.project, *self.project.parents):
            if (directory / ".git").exists():
                return directory
        return self.project

    def _text(self, path: Path) -> str | None:
        if path not in self.texts:
            self.texts[path] = None
            if path.is_file():
                try:
                    self.texts[path] = _read(path).decode("utf-8", "replace")
                except OSError as e:
                    self.failed.add(path)
                    self.unread.append(f"{path}: could not read it ({_reason(e)})")
        return self.texts[path]


def command(
    files: Sequence[Path],
    *,
    project: Path,
    out: TextIO,
    named: bool = False,
    feedback: TextIO | None = None,
) -> int:
    """Report each hook in ``files`` that cannot work as written; see the module.

    ``named``: the files were asked for by name, so one that is not there is
    reported as unread rather than passed over. ``feedback``: where to ask how
    it went, once at least one hook was fully checked (every script it names
    read), and only if it is a terminal: never with the findings on ``out``,
    which scripts read, and never into a CI log. The checker decides because
    only it knows what it read (D86)."""
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
    faults = checked = 0
    for hook in hooks:
        problems = list(_problems(hook, env=env, scripts=scripts))
        for problem in problems:
            out.write(f"{_shown(hook.where)}: {problem}\n")
        faults += bool(problems)
        its_scripts = scripts.named_in(hook.command).values()
        checked += not any(path in scripts.failed for path, _ in its_scripts)
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
    if feedback is not None and checked and feedback.isatty():
        feedback.write(_TELL_US)
    if unread:
        return 2
    return 1 if faults else 0


def _settings(
    path: Path, *, named: bool, unread: list[str]
) -> dict[str, object] | None:
    """The settings in ``path``, or None, with the reason added to ``unread``
    unless the file is simply absent and was not asked for by name."""
    try:
        settings = parse_settings(path)
    except FileNotFoundError:
        if named:
            unread.append(f"{path}: no such file")
        return None
    except (OSError, ValueError) as e:
        unread.append(f"{path}: could not read it ({_reason(e)})")
        return None
    if not isinstance(settings, dict):
        unread.append(f"{path}: not a JSON object")
        return None
    wrong = misshapen(settings)
    if wrong is not None:
        unread.append(
            f"{path}: its {wrong} has the wrong type, and Claude Code rejects such "
            "a file whole, so its hooks would not run"
        )
        return None
    return settings


# The settings keys Paveo reads, and their types. Claude Code rejects a user,
# project or local settings file "whose JSON or top-level shape fails
# validation" as a whole (managed-settings page, read 20 Oct 2026), so a reader
# that loaded it would check hooks that never run.
_SHAPE: dict[str, type] = {
    "hooks": dict,
    "disableAllHooks": bool,
    "permissions": dict,
    "enabledPlugins": dict,
    "env": dict,
}
# The seven permission modes the settings reference lists, and whether a
# hook's ask stops the call in each: it goes to a person, or in dontAsk is
# denied ("auto-denies every call that would otherwise prompt"); in auto and
# bypassPermissions no one is asked. "manual" is "default" by another name.
PERMISSION_MODES = {
    "default": True,
    "manual": True,
    "plan": True,
    "acceptEdits": True,
    "dontAsk": True,
    "auto": False,
    "bypassPermissions": False,
}


# Matchers, as Claude Code reads them (``paveo test`` and ``--selftest``): an
# exact matcher is a list of names; anything else is an unanchored regex.
_NAMES_ONLY = re.compile(r"[A-Za-z0-9_\- ,|]*")
# Claude Code tests a matcher as a JavaScript regex. Python reads these, and
# JavaScript refuses them or reads them as something else: inline flags, named
# groups written Python's way, comments, atomic groups, conditionals, \A or \Z
# for the ends (and 3.14's \z), \a, \N{...} and \U escapes, possessive
# quantifiers and ``{,n}``.
# Each needs an even run of backslashes before it, none included, or what
# follows a backslash is escaped and means the same in both.
_PYTHON_ONLY = re.compile(
    r"(?<!\\)(?:\\\\)*(?:"
    r"\(\?(?:P|#|>|\(|[aiLmsux]*-[aiLmsux]*:|[aiLmsux]+[:)])"
    r"|\\[AZaNUz]|[*+?}]\+|\{,)"
)


def reaches_tools(  # noqa: PLR0911 - one return per rule
    matcher: object, names: Sequence[str]
) -> tuple[Callable[[str], bool], str | None]:
    """Which of ``names`` a group's matcher reaches, as the reference's matcher
    table says; a matcher that cannot be read reaches everything, untested. A
    regular expression is the repository's own, so it is tried in an isolated
    child with a time limit, never here (``_matcher_trial``, D118)."""
    if matcher is None or matcher in (
        "",
        "*",
    ):  # a tuple: a list matcher is not hashable
        return (lambda _tool: True), None
    if not isinstance(matcher, str):
        return (lambda _tool: True), "its matcher is not text"
    if len(matcher) > _matcher_trial.MAX_LENGTH:
        return (lambda _tool: True), _matcher_trial.UNREAD
    if _NAMES_ONLY.fullmatch(matcher):
        names_listed = {name.strip() for name in re.split(r"[|,]", matcher)}
        return names_listed.__contains__, None
    if _PYTHON_ONLY.search(matcher):
        return (lambda _tool: True), _matcher_trial.UNREAD
    hits = _matcher_trial.reach(matcher, names)
    if isinstance(hits, str):
        return (lambda _tool: True), hits
    return hits.__contains__, None


def runs_in_background(entry: dict[str, object]) -> bool:
    """Whether Claude Code runs this hook entry without waiting for it, so it
    can stop nothing. One rule for ``paveo test`` and ``--selftest``."""
    return entry.get("async") is True or entry.get("asyncRewake") is True


def misshapen(settings: dict[str, object]) -> str | None:
    """The first key Paveo reads whose type the settings reference rules out.
    Keys Paveo does not read are not checked, and the README says so."""
    for name, kind in _SHAPE.items():
        if name in settings and not isinstance(settings[name], kind):
            return name
    hooks = settings.get("hooks", {})
    env = settings.get("env", {})
    plugins = settings.get("enabledPlugins", {})
    permissions = settings.get("permissions", {})
    # Each is a dict by now; the checks below are of what they hold.
    wrong = [
        (
            "hooks",
            isinstance(hooks, dict)
            and any(not isinstance(groups, list) for groups in hooks.values()),
        ),
        (
            "env",
            isinstance(env, dict)
            and any(not isinstance(value, str) for value in env.values()),
        ),
        (
            "enabledPlugins",
            isinstance(plugins, dict)
            and any(not isinstance(value, bool) for value in plugins.values()),
        ),
    ]
    if isinstance(permissions, dict):
        # A malformed rule inside a list is skipped on its own, "while the rest
        # of the file applies" (managed-settings page), so only the list counts.
        for rules in ("allow", "ask", "deny"):
            listed = permissions.get(rules, [])
            wrong.append((f"permissions.{rules}", not isinstance(listed, list)))
        mode = permissions.get("defaultMode", "default")
        known = isinstance(mode, str) and mode in PERMISSION_MODES
        wrong.append(("permissions.defaultMode", not known))
    return next((name for name, bad in wrong if bad), None)


class _RefusedError(ValueError):
    """A file Claude Code would refuse, with the reason to print."""


def parse_settings(path: Path) -> object:
    """``path`` parsed as Claude Code parses a settings file: at most 1 MiB,
    strict UTF-8, no byte-order mark, no ``NaN`` or ``Infinity``, a regular
    file. Raises ``FileNotFoundError``, ``OSError`` or ``ValueError``, and
    nothing else; init, the self-test, doctor and test all read through here,
    so none trusts a file the others would refuse."""
    try:
        return json.loads(_text(_read(path)), parse_constant=_not_json)
    except RecursionError as e:
        raise _RefusedError("it is nested too deeply") from e


def _text(data: bytes) -> str:
    """A settings file as Claude Code reads it: UTF-8, strictly. Python's JSON
    reader would detect UTF-16 and skip a byte-order mark on its own, and read
    files Claude Code rejects whole, whose hooks never run."""
    if data.startswith(b"\xef\xbb\xbf"):
        raise _RefusedError("it starts with a byte-order mark")
    return data.decode("utf-8")


def hook_groups(hooks: object) -> dict[str, list[object]]:
    """``hooks`` as ``misshapen`` has passed it: an object of lists. Raises
    rather than skip anything else, since a reader that skipped would report
    clean on hooks it never read (Rule 3); ``misshapen`` makes it unreachable,
    and this makes that mechanical (Rule 5)."""
    if not isinstance(hooks, dict) or not all(
        isinstance(groups, list) for groups in hooks.values()
    ):
        raise TypeError("hooks were read without the shape check")
    return hooks


def _not_json(constant: str) -> object:
    """Python reads ``NaN`` and ``Infinity``; Claude Code's JSON parser, like
    every JavaScript one, refuses the file that holds them."""
    raise _RefusedError(f"{constant} is not JSON")


def _hooks(path: Path, hooks: object) -> tuple[list[_Hook], list[str]]:
    found: list[_Hook] = []
    odd: list[str] = []
    for event, groups in hook_groups(hooks).items():
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
            f"reads ${name}, which Claude Code does not set, so it always reads an "
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
        bare = _QUOTED.sub(_emptied, _marked(text))
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


def _emptied(quoted: re.Match[str]) -> str:
    """A quoted string with nothing that could run in it, emptied; one that can
    expand (``$``, a backtick) is kept, so the commands inside it are judged."""
    text = quoted.group()
    return text if text[0] == '"' and re.search(r"[$`]", text) else text[0] * 2


def _marked(text: str) -> str:
    """``text`` with the project, git top level and home as markers, so a script
    path written ``"$CLAUDE_PROJECT_DIR"/x.sh`` is one word that can be resolved.
    A quote is taken only with its partner, so the quoting the rest of the
    command relies on is left as it was."""
    for form, marker in _MARKS:
        text = re.sub(rf'"(?:{form})"|(?:{form})', marker, text)
    return text


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
    alone. Found on a public repo by running this over 153 of them, 7 Oct. The
    test may be on a copy, ``cmd="${X:-}"`` then ``-z "$cmd"``, which a public
    repository's hooks did (D93); a copy counts only when it is made from
    ``name`` alone."""
    # The value is the variable and nothing else: matching quotes or none, and
    # only an empty default (`${X:-}`), so the copy is empty exactly when X is.
    copies = [
        match.group(1)
        for match in re.finditer(
            rf"\b(\w+)=([\"']?)\$(?:\{{{name}(?::-)?\}}|{name}\b)\2"
            r"(?=[ \t]*(?:;|&&|\|\||\)|\n|$)|[ \t]+#)",
            text,
        )
    ]
    tested = any(
        re.search(rf"-[nz]\s+\"?\$\{{?{each}\b|\$\{{{each}:[-=][^}}]", text)
        for each in (name, *copies)
    )
    return bool(tested and _STDIN.search(text))


def _uncommented(text: str) -> str:
    """``text`` without its whole-line comments, which may name a variable to
    say it is not used. A comment after code stays: cutting at ``#`` would cut
    ``${x#y}`` and ``"#"`` too."""
    return "\n".join(line for line in text.splitlines() if not _COMMENT.match(line))


def _read(path: Path) -> bytes:
    """A bounded read, not a size check and then a read: the file can grow
    between. Opened without blocking and checked to be a regular file before a
    byte is read: a settings path a cloned repository links to a FIFO or a
    terminal would otherwise hang the command forever."""
    fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as file:
        if not stat.S_ISREG(os.fstat(file.fileno()).st_mode):
            raise _RefusedError("it is not a regular file")
        data = file.read(_MAX_BYTES + 1)
    if len(data) > _MAX_BYTES:
        raise OSError(f"over {_MAX_BYTES} bytes")
    return data


def _reason(error: Exception) -> str:
    if isinstance(error, OSError):
        return error.strerror or str(error)
    if isinstance(error, UnicodeDecodeError):
        return "not UTF-8"
    if isinstance(error, _RefusedError):
        return str(error)
    return "not valid JSON"


def _shown(text: str) -> str:
    """``text`` with control characters escaped: a settings file from a cloned
    repository chooses its event names and paths, and the terminal prints them."""
    return "".join(
        char if char.isprintable() else char.encode("unicode_escape").decode()
        for char in text
    )
