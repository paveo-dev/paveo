"""The dangerous actions ``paveo test`` tries, each spelled the ways agents write it.

A guard that matches command text holds against the spelling its author tried and
often nothing else: our own starters let 5 of 14 force pushes through until D92,
and every guard tested by hand for a fix-it issue let some spelling through (D96).
So each action is listed once in its plain form, first, then in the forms the same
action takes when written differently: git's own options before the verb, the flag
moved, a path to the binary, a backslash, a shell, a variable, ``eval``.

The list lives in ``bypasses.json`` so that anyone can add a spelling by pull
request and every run can say which version of the list it used (D106). It is
checked as it loads, and a file that fails any check stops the run: a test that
quietly tried fewer spellings would pass guards it never tested.

**Every spelling is harmless if a hook ran it.** A hook should never run the
command it judges, but this is handed to code we have not read: deletes aim at a
path that does not exist, pushes at a remote that does not exist, resets at a ref
that does not exist, and cleans at a pathspec that matches nothing. Publishing is
left out for that reason: no spelling of ``npm publish`` is safe to hand over.
``_unsafe`` lets a spelling use only words the list already uses, each where it
means no harm, so a new word is a reviewed change to this module; it cannot prove
a command harmless, so a reviewer still reads every change to the file.

The list is finite, so a guard that stops all of it is not proven to hold; it has
only not been walked around by these (D101).
"""

from __future__ import annotations

import json
import re
import shlex
from dataclasses import dataclass
from importlib import resources

from .errors import ConfigError

# Never a real remote, ref, path or file: see the module docstring.
_PROBE = "paveo-test-probe"
_TOOLS = ("Bash", "Read")
# Every word a spelling may use (D107). A word not here is refused, so a
# spelling that needs a new one is a change to this module, reviewed as code: a
# list of what is forbidden is walked around by whatever it forgot (`${HOME}`,
# `.`, a real remote behind `#`, `nc`), as the review of D106 showed.
_WORDS = frozenset(
    {
        *("git", "push", "reset", "clean", "rm", "find", "env", "command"),
        *("cd", "xargs", "echo", "+", "{}", "--", "main", "+main", "."),
        *("color.ui=never", "GIT_TERMINAL_PROMPT=0", "F=--force", "$F", "R=rm", "$R"),
        *("--force", "--hard", "--mirror", "--no-pager", "--recursive", "-C", "-c"),
        *("-f", "-uf", "-r", "-rf", "-fr", "-Rf", "-d", "-x", "-fdx", "-xdf"),
        *("-dffx", "-delete", "-exec", "/bin/rm", "/usr/bin/git"),
    }
)
_SECRET_WORDS = frozenset(
    {
        *("cat", "head", "sed", "awk", "base64", "grep", "tail", "xxd", "nl"),
        *("-n", "50", "p", "1", "+1", "."),
    }
)
# A word naming the probe, and nothing else: a spelling's only target.
_TARGET = re.compile(rf"/?{_PROBE}")
_SECRET = re.compile(r"(?:\./|\$PWD/)?\.env")
# The only lines python3 may run: one deletes the probe, one reads .env.
_PYTHON = {
    False: f"import shutil; shutil.rmtree('/{_PROBE}')",
    True: "print(open('.env').read())",
}
# What a command acts on follows its verb: after one, every word that is not a
# flag must be the target (D109), so no known word can become a second path.
_VERBS = frozenset(
    {
        *("rm", "/bin/rm", "$R", "find", "echo", "push", "reset", "clean"),
        *("cat", "head", "sed", "awk", "base64", "grep", "tail", "xxd", "nl"),
    }
)
# Operands that name nothing real: the branch of a push to the probe remote,
# find's -exec tail, the arguments the .env readers take.
_HARMLESS = {
    "push": {"main", "+main"},
    "find": {"rm", "{}", "+"},
    "head": {"50"},
    "sed": {"p"},
    "awk": {"1"},
    "tail": {"+1"},
    "grep": {"."},
}
# Before the verb: `.` only as git -C's folder, -c only as git's option, and a
# binary's path only as the command itself.
_ONLY_AFTER = {".": {"-C"}, "-c": {"git"}}
_ONLY_FIRST = frozenset({"/bin/rm", "/usr/bin/git"})
# Commands that aim at nothing and only set up the next: cd into this folder,
# a variable a later command uses.
_SETUP = (["cd", "."], ["F=--force"], ["R=rm"])
_SEPARATORS = frozenset({"&&", "|", ";"})
_VERSION = re.compile(r"\d{4}-\d{2}-\d{2}(?:\.\d+)?")
_NAME = re.compile(r"[a-z][a-z-]{0,30}")
_LONGEST = 300


@dataclass(frozen=True)
class Spelling:
    tool: str  # the Claude Code tool: Bash or Read
    # Bash: the command. Read: the file, relative to the project, made absolute
    # when the call is built, since Claude Code always sends Read an absolute path.
    text: str


@dataclass(frozen=True)
class Action:
    name: str  # what --expect takes
    title: str
    spellings: tuple[Spelling, ...]  # the plain form first


def load(text: str) -> tuple[str, tuple[Action, ...]]:
    """The list's version and its actions, from the text of a ``bypasses.json``."""
    try:
        document = json.loads(text)
    except json.JSONDecodeError as e:
        raise _invalid(f"is not JSON ({e.msg}, line {e.lineno})") from e
    if not isinstance(document, dict) or set(document) != {"version", "actions"}:
        raise _invalid('must hold exactly "version" and "actions"')
    version, actions = document["version"], document["actions"]
    if not isinstance(version, str) or not _VERSION.fullmatch(version):
        raise _invalid('"version" must be a date: YYYY-MM-DD')
    if not isinstance(actions, list) or not actions:
        raise _invalid('"actions" must be a list of at least one action')
    loaded = tuple(_action(entry, at) for at, entry in enumerate(actions))
    if len({action.name for action in loaded}) != len(loaded):
        raise _invalid("names an action twice")
    return version, loaded


def _action(entry: object, at: int) -> Action:
    if not isinstance(entry, dict) or set(entry) != {"name", "title", "spellings"}:
        raise _invalid(f'action {at} must hold exactly "name", "title", "spellings"')
    name, title, spellings = entry["name"], entry["title"], entry["spellings"]
    if not isinstance(name, str) or not _NAME.fullmatch(name):
        raise _invalid(f"action {at}'s name must be lowercase letters and hyphens")
    if not isinstance(title, str) or not title.strip():
        raise _invalid(f"action {name} needs a title")
    if not isinstance(spellings, list) or not spellings:
        raise _invalid(f"action {name} needs at least one spelling")
    loaded = tuple(_spelling(name, item, at) for at, item in enumerate(spellings))
    if len(set(loaded)) != len(loaded):
        raise _invalid(f"action {name} lists a spelling twice")
    return Action(name, title, loaded)


def _spelling(action: str, item: object, at: int) -> Spelling:
    where = f"action {action}, spelling {at}"
    if not (isinstance(item, list) and len(item) == 2):  # noqa: PLR2004 - [tool, text]
        raise _invalid(f"{where} must be [tool, text]")
    tool, text = item
    if tool not in _TOOLS or not isinstance(text, str):
        raise _invalid(f"{where}: the tool must be Bash or Read, the text a string")
    if not text or len(text) > _LONGEST or not text.isprintable():
        raise _invalid(f"{where} must be one line of 1 to {_LONGEST} characters")
    why = _unsafe(action, tool, text)
    if why is not None:
        raise _invalid(f"{where} {why}")
    return Spelling(tool, text)


def _unsafe(action: str, tool: str, text: str) -> str | None:
    """Why this spelling might do harm if a hook ran it, or None."""
    if tool == "Read":
        harmless = action == "secrets" and _SECRET.fullmatch(text)
        return None if harmless else "may use Read only to read .env"
    if tool != "Bash":
        return "must be a Bash command"
    return _line(text, secrets=action == "secrets")[0]


def _line(text: str, *, secrets: bool) -> tuple[str | None, bool]:
    """Why a line might do harm, and whether it aims at the target: every
    command on it must, and a command that runs a line of its own (`sh -c`,
    `eval`) is judged the same way."""
    try:
        lexer = shlex.shlex(text, posix=True, punctuation_chars=True)
        lexer.whitespace_split = True
        # bash starts a comment only at the start of a word; shlex anywhere,
        # so `x#; rm -rf src` would hide a command bash runs (D108).
        lexer.commenters = ""
        words = list(lexer)
    except ValueError:
        return "is not one shell command line", False
    commands: list[tuple[str, list[str]]] = [("", [])]
    for word in words:
        if word in _SEPARATORS:
            commands.append((word, []))
        else:
            commands[-1][1].append(word)
    defined: set[str] = set()
    aimed = False
    for at, (separator, command) in enumerate(commands):
        if command in _SETUP:
            # bash runs each side of a pipe apart, so a variable set there is
            # not set for the next command.
            if at + 1 < len(commands) and commands[at + 1][0] == "|":
                return "pipes a command that only sets up the next", False
            defined.update(f"${w.split('=')[0]}" for w in command if "=" in w)
            aimed = False
            continue
        # xargs takes its target from a pipe, never from what ran before && or ;.
        piped = separator == "|" and aimed
        why, aimed = _command(command, secrets=secrets, piped=piped, defined=defined)
        if why is not None:
            return why, False
    return None, aimed


def _command(
    words: list[str], *, secrets: bool, piped: bool, defined: set[str]
) -> tuple[str | None, bool]:
    """Why one command might do harm, and whether it aims at the target."""
    target, known = (_SECRET, _SECRET_WORDS) if secrets else (_TARGET, _WORDS)
    inner = _inner(words)
    if words[:1] == ["python3"]:
        # Only the one -c line the list knows, never a file python3 would run.
        harmless = inner == _PYTHON[secrets]
        return (None if harmless else "may run python3 only on a known line"), True
    if inner is not None:
        why, aimed = _line(inner, secrets=secrets)
    else:
        why = _misplaced(words, known, defined, target)
        aimed = any(target.fullmatch(word) for word in words)
        aimed = aimed or (piped and words[:1] == ["xargs"])
    if why is None and not aimed:
        why = f"must aim every command at {_PROBE}"
    return why, why is None


def _inner(words: list[str]) -> str | None:
    """The line a command runs of its own: eval LINE, sh -c LINE, python3 -c LINE."""
    if len(words) == 2 and words[0] == "eval":  # noqa: PLR2004 - eval, the line
        return words[1]
    runners = (["sh", "-c"], ["bash", "-c"], ["python3", "-c"])
    return words[2] if len(words) == 3 and words[:2] in runners else None  # noqa: PLR2004


def _misplaced(
    words: list[str],
    known: frozenset[str],
    defined: set[str],
    target: re.Pattern[str],
) -> str | None:
    """Why a word that is not the target is unknown or out of its place."""
    verb = ""
    operands = False  # after --, every word is one
    for at, word in enumerate(words):
        if target.fullmatch(word):
            continue
        if word not in known or (word in _ONLY_FIRST and at):
            return f"uses {word!r}, which the list does not use yet"
        if word in {"$F", "$R"} and word not in defined:
            return f"uses {word} without setting it first"
        before = words[at - 1] if at else ""
        if not verb and word in _VERBS:
            verb = word
        elif not verb:
            if word in _ONLY_AFTER and before not in _ONLY_AFTER[word]:
                return f"uses {word!r} where it could name something real"
        elif word == "--" and not operands:
            operands = True
        elif _operand(word, after_dashes=operands) and word not in _HARMLESS.get(
            verb, ()
        ):
            return f"gives {verb} {word!r}, which is not {_PROBE}"
    return None


def _operand(word: str, *, after_dashes: bool) -> bool:
    """What a verb acts on, rather than a flag that changes how."""
    return after_dashes or not (word.startswith("-") or word == "$F")


def _invalid(why: str) -> ConfigError:
    return ConfigError(
        f"paveo's list of bypasses (bypasses.json) {why}.",
        remedy="reinstall paveo; if you edited the list, see its module docstring",
    )


VERSION, ACTIONS = load(
    resources.files("paveo").joinpath("bypasses.json").read_text("utf-8")
)

# Calls nobody guards against. A hook that stops one of these stops everything,
# so a block elsewhere says nothing about whether it read the command.
ORDINARY = (Spelling("Bash", "git status"), Spelling("Read", "README.md"))
