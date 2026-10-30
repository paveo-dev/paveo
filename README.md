# Paveo

**Paveo finds the dangerous commands your AI coding agent's guard lets through, and checks it again on every change.**

Coding agents write the same dangerous command many ways: `rm -r -f`, `find
-delete`, `sh -c '...'`. Many guards match text and catch only the ways their
author tried. `uvx paveo test` sends your Claude Code guard
68 spellings of five dangerous actions, the way Claude Code sends a call, and
names each one that gets past it; `uvx paveo test codex` does the same for Codex, by Codex's rules (66
spellings, since Codex has no Read tool, and as if each hook were trusted in
Codex's `/hooks`, without which Codex runs none); `uvx paveo test cursor` does the
same for Cursor, by the rules on Cursor's hooks page (not yet checked in a real
Cursor session). In CI,
`--expect` fails the build the day a guard stops stopping one
([details](#test-the-hooks-you-already-have), [Codex](#codex-hooks),
[Cursor](#cursor-hooks)). (`uvx`
comes with [uv](https://docs.astral.sh/uv/) and fetches a suitable Python
itself.) For Claude Code, `uvx paveo doctor` names the hooks that cannot work as
written, such as guards that can never block.

Paveo also ships a guard of its own: Claude Code, Codex and
Cursor ask it before they run a shell command or write a file. `rm -rf`, `git push
--force`, `DROP TABLE` and the obvious ways of editing the guard itself are
refused, the agent is told why, and every decision goes into a hash-chained record
your security team can verify. It matches patterns, so it is a seatbelt, not a
sandbox. The Cursor guard has not been tried in a real session yet, and by
Cursor's documentation it may refuse every call
([details](#coding-agents-claude-code-codex-and-cursor)).

**Nothing leaves your machine.** Paveo runs on each engineer's machine, opens no
network sockets (a test fails the build if it does), and installs no other package.
It is open source under Apache-2.0, except the paid audit evidence export
([licence](#licence)), and free for up to three agents per policy.

```
pip install paveo            # Python 3.11+ (python3 --version), macOS or Linux
```

**For a whole team:** the hook, one policy, a standalone Python and paveo, all
inside a root-owned `/opt/paveo`, for Claude Code and Codex on every engineer's
machine and set in each agent's managed settings (in the one managed source you
already deploy: a second source can make Claude Code skip it, see the guide); on
every call the hook checks
that everything in `/opt/paveo` is root's, file by file, so the agent cannot
switch it off through any file the hook runs (it can still edit what the
engineer's shell starts with, such as `~/.zshrc`: the guide says what is not
covered); or, without device management, the Claude Code plugin (`/plugin marketplace
add paveo-dev/paveo`, then `/plugin install paveo@paveo`).
[docs/ROLLOUT.md](docs/ROLLOUT.md) has both and what each does not cover; we set
it up with you if you like: [talk to us](https://paveo.dev/#contact).

Paveo also guards agents you build yourself (every tool call and model call
checked against a policy, with a budget) and
[any MCP server](#any-mcp-client-any-language); see
[guard your own agent](#five-minutes-guard-your-own-agent).

**Tried it?** Paveo sends nothing home, so we only learn what works from you:
[say hi in Discussions](https://github.com/paveo-dev/paveo/discussions).

## Two minutes: a seatbelt for your coding agent

For Claude Code, Codex or Cursor, with no code to write. It needs Python 3.11 or
newer, so run `python3 --version` first. The one that comes with Apple's command
line tools can be 3.9, where pip says only `No matching distribution found for
paveo`, without mentioning Python. If yours is older, put a newer one, such as
Homebrew's `python3.12`, in the first line, and if you already ran it with the
old one, `rm -rf ~/.paveo` first so nothing of it is left behind.

```
python3 -m venv ~/.paveo && ~/.paveo/bin/pip install paveo
cd /your/project
~/.paveo/bin/paveo init claude-code      # or: init codex, init cursor
```

From then on, before the agent runs a shell command or writes a file, Paveo
checks it. `rm -rf`, `git push --force`, `git reset --hard` and `DROP TABLE` are
refused, and so are the obvious ways of editing the guard's own policy, log and
hook settings. The refusal goes back to the agent, which tries something else.
`init` tests the hook it wrote before it says `Done`. Details, and what it
cannot do, are [below](#coding-agents-claude-code-codex-and-cursor).

**Before you trust it, replay what already happened:**
`~/.paveo/bin/paveo replay claude-code` puts every tool call from your past
Claude Code sessions in this project through the policy, and prints how many it
would have refused and why. Nothing is stored and nothing from the sessions is
printed.

## Five minutes: guard your own agent

A policy names each agent and what it may do:

```json
{"version": 1, "policy_id": "support",
 "agents": [{"id": "refund-bot",
             "tools": {"allow": [
               {"name": "lookup_order", "constraints": {"order_id": {}}},
               {"name": "refund",
                "constraints": {"order_id": {}, "amount_usd": {"max": "500.00"}},
                "requires": {"tool": "lookup_order", "same": ["order_id"]}}]},
             "budget": {"period": "day", "limit_usd": "5.00"},
             "models": {"allow": ["claude-sonnet-5"]}}]}
```

Then every tool call goes through `check_tool`, and a refusal goes back to the
model as the tool's result, so it tries something else instead of your loop
crashing:

```python
from paveo import Paveo, PolicyDenied

# Both `with` blocks matter: the outer one closes the audit log and its anchor.
with Paveo.from_file("paveo.json", audit_path="audit.jsonl") as pf:
    with pf.session(agent_id="refund-bot", principal="user_123") as s:
        try:
            s.check_tool(name, arguments)
        except PolicyDenied as refused:
            result = refused.for_model   # e.g. "Refused by policy: refund.amount_usd.max
                                         # (constraint_violated). An argument is outside..."
        else:
            result = run_tool(name, arguments)
```

Here a refund over $500, a refund for an order the agent never looked up, and
any tool not listed are all refused, and recorded. `for_model` names the rule and
says what the model can do about it. It deliberately leaves out what
`str(refused)` includes, which is how to change the policy: that advice is for
you, not for an agent that might act on it.

To see a ceiling refuse with no API key and no money, clone this repository and
run `python examples/runaway_loop.py`: it spends a $1.00 ceiling down on recorded
responses and is refused before the call that would breach it.

## What a policy can say

**Which tools, and with which arguments.** When a tool rule lists constraints,
**those are the only arguments that tool may be called with**, and an argument
carrying a constraint has to actually be supplied:

```jsonc
"refund": { "amount_usd": { "max": "500.00" }, "note": {} }

{"amount_usd": 250, "note": "duplicate charge"}   // allowed
{"amount_usd": 600}                               // denied — over the max
{"amount_usd": 250, "destination": "..."}         // denied — not a named argument
{"note": "..."}                                   // denied — amount_usd is constrained,
                                                  //   so it cannot be left out
```

A constraint you can dodge by omitting the argument is not a constraint. `{}`
permits an argument without constraining or requiring it. `not_matches` refuses
an argument that matches a pattern, which is how the coding-agent guard refuses
`rm -rf`. An agent with no rule for a tool may not call it.

**A tool only after another.** `requires` permits a tool only once an earlier
call in the same session, to a tool you name, had the same values for the
arguments you name:

```jsonc
{ "name": "refund",
  "constraints": { "order_id": {}, "amount_usd": { "max": "100.00" } },
  "requires": { "tool": "lookup_order", "same": ["order_id"] } }

// refund A-1 with no lookup                  denied (requires_unmet)
// lookup_order A-1, then refund A-1          allowed
// lookup_order A-1, then refund B-2          denied
```

"Earlier" means a call the policy admitted, which is not one that succeeded: if
the lookup itself failed, the refund is still allowed. Only strings, integers,
booleans and null can be compared; any other value meets nothing. The session
remembers a SHA-256 digest of the compared values, never the values, and forgets
it when its `with` block ends. A policy whose `requires` could never be met does
not load.

**How often.** `"rate": {"calls": 20, "seconds": 60}` refuses the 21st call to a
tool inside any 60 seconds, and `"repeat": {"seconds": 30}` refuses the same
call, every argument equal, twice inside 30 seconds: the loop that retries one
failing request forever. Add `"same": ["command"]` to compare only the arguments
named, since a model rewrites some freely, like Claude Code's `description` on
Bash. Values are compared exactly as sent, so `42` and `"42.00"` are different
calls: name in `same` the arguments a rule pins, such as an id or a currency, not
an amount the agent can spell another way. Both count only calls the rules
admitted, per session, so an agent's calls
in two sessions are not added together. Time never runs backwards for them: they
move on by how far the clock moves forward, and not at all when it steps back.
**They are brakes on an agent stuck in a loop, not a quota against one trying to
get round them**: an agent that can edit files can delete its memory. In the
guard, a call counts when the hook admits it, before Claude Code asks you: one
you decline still counts, so an identical retry you then approve can be refused
as `repeated` until its window passes.

**Which models, and how much.** `models.allow` names the models an agent may
call, and `budget` puts a ceiling on what it may spend in a period. An agent with
no budget makes no model calls. See [Model calls](#model-calls-and-ceilings).

**Try it before it enforces.** Give an agent `"mode": "shadow"` and each call its
rules would refuse is recorded as `would_deny` and allowed through, so you can
run your agent for a day and read what the policy would have broken. Every call
from that agent logs a warning while this is on. **Its budget is never
shadowed**: the ceiling refuses as it always does.

## Model calls and ceilings

A model call is checked before it is sent, and its worst case is held against
the agent's ceiling until you say what it really cost. Alerts tell you after the
money is gone; this refuses the call that would breach the ceiling.

```python
request = {"model": "claude-sonnet-5", "max_tokens": 1024, "messages": [...]}

with pf.session(agent_id="refund-bot", principal="user_123") as s:
    # refused before a byte leaves if the model is not allowed, the call cannot
    # be priced, or its worst case would breach the ceiling
    with s.check_llm(request, shape="anthropic") as call:
        response = client.messages.create(**request)
        call.record(response.usage.model_dump())

    s.remaining()   # USD left, counting calls still in flight
```

For Anthropic, `wrap_anthropic` does those three steps for you:

```python
import anthropic

with pf.session(agent_id="refund-bot", principal="user_123") as s:
    client = s.wrap_anthropic(anthropic.Anthropic())
    response = client.messages.create(**request)   # checked, sent, recorded
```

It offers `messages.create` and `messages.stream`, and the same under `beta`,
**and nothing else**: any other attribute raises rather than letting a call out
unchecked, so use the unwrapped client for those, knowingly. A call that raises
is charged its worst case (Anthropic does not say which failed requests go
unbilled). A stream read to its end, or closed after its usage arrived, is
charged what it reported; one abandoned before that, or still open when the
session ends, is charged its worst case. Leaving a stream early never reads the
rest of it. It takes a sync client for now: `AsyncAnthropic` is refused
until `async_session` exists.

OpenAI Chat Completions (also what LiteLLM sends for any provider) and Gemini
`generate_content` go through `check_llm` the same way:

```python
with s.check_llm(request, shape="openai") as call:
    response = client.chat.completions.create(**request)
    call.record(response.model_dump())

with s.check_llm(request, shape="gemini") as call:   # {"model", "contents", "config"}
    response = client.models.generate_content(**request)
    call.record(response.usage_metadata.model_dump())
```

If the request never reached the provider, call `call.release()` instead. A call
settled neither way is charged its whole worst case, because it may have been
billed. For any provider without a built-in table, declare the model's price in
the policy (`"prices": {"mistral-large-3": {"input_per_mtok": "2.00",
"output_per_mtok": "6.00"}}`). Then either send it in OpenAI's shape
(`shape="openai"`) and record the response as OpenAI returns it, or use
`shape="generic"` for any other shape and record `{"input": n, "output": m}`.

For OpenAI, an unset `service_tier` is reserved at Fast, since a project setting
can make it Fast, and the whole response tells us the tier really used, so it is
charged at that.

**What it will refuse, and why.** Anything whose cost it cannot bound from the
request: images, audio, video and files; content sent by URL or file id; PDFs;
server-run tools such as web search or code execution; any request parameter it
has not verified; and any call without an output cap (`max_tokens`,
`max_completion_tokens` or `max_output_tokens`), unless the policy sets
`defaults.assumed_max_output_tokens`. Each refusal names the part of the request
and says what to do. If one of these is what your agent does, open an issue: it
is the most useful thing you can tell us.

**If a call ever costs more than was reserved for it, Paveo stops admitting
calls** until the process restarts: it means an estimate under the ceiling was
wrong, and a guard that knows it is wrong should not keep guessing. (Not for a
call reserved on your policy's `assumed_max_output_tokens`, which caps nothing
and is approximate by design.) The checks are plain functions that never wait,
so they work inside async code as they are.

**What is recorded:** identities, decisions, reasons, costs and token counts,
hash-chained, on your disk. Never a prompt, a completion or a tool argument.

**Built:** `Paveo.from_file`, `from_policy`, `session`, `check_tool`,
`check_llm`, `wrap_anthropic`, `remaining`, `PolicyDenied.for_model`, the policy
file with shadow mode, `not_matches`, `requires`, `rate` and `repeat`,
`verify_chain`, the `paveo` command (`init`, the guard for Claude Code, Codex and
Cursor, `stop`, `resume`, `replay`, `learn`, `evidence`, `mcp`, `doctor`, `test`), starter
policies, and readers for Anthropic, OpenAI Chat Completions and Gemini
`generate_content` requests. **Not built yet:** `async_session`, and a budget
shared across processes.

## Coding agents: Claude Code, Codex and Cursor

`paveo guard <agent>` is a pre-tool hook. Before the agent runs a shell command
or writes a file, the hook checks the call against a policy, records it, and
refuses it if the policy says no.

**Claude Code.** `init claude-code` writes the starter policy to
`.paveo/policy.json` and a hook to `.claude/settings.local.json` (your personal
settings, kept out of git, since the hook names this machine's `paveo`), keeping
every other setting. It then **tests the hook it wrote, and asks the policy
whether it refuses `rm -rf`**, and says `Done` only if both pass. Start Claude
Code in that folder and it is on. Running `init` again changes nothing. It
refuses, and changes nothing, if a settings file cannot be read, if an existing
paveo hook would miss Write or Edit, if any file it would write is a symbolic
link, or if the project's `.claude` folder is the one Claude Code reads in every
project, as it is in your home folder: a guard meant for one project would guard
them all, so `cd` into the project first.

**One policy for every agent in a project.** Run `init` for a second agent in
the same project (`init cursor` after `init claude-code`, say) and its starter
rules are added to the `.paveo/policy.json` already there, so one file covers
all of them. The free plan covers three agents per policy, so Claude Code, Codex
and Cursor fit together; `init` for an agent past three says so and changes
nothing.

If you move or delete that virtualenv, the hook stops working **silently**:
Claude Code treats a hook it cannot run as one with no objection. So after any
change, run `~/.paveo/bin/paveo guard --selftest` in the project. It checks
every configured hook really refuses, fails one Claude Code would skip for some
calls (a matcher that misses file edits, a `type` other than `"command"`,
`"args"`, `"shell": "powershell"`, `"async": true`, an `"if"` filter, or a
`timeout` under 10 seconds),
and says so if a paveo hook sits in your
global settings (`~/.claude/settings.json`). There it runs in every project, and
the hook `init` writes refuses every call in a project with no
`.paveo/policy.json`.

- **Panic button:** `~/.paveo/bin/paveo stop`, run in the project, refuses
  every call until `~/.paveo/bin/paveo resume`. `init` prints the exact line for
  your install. The starter policy refuses `paveo resume` from the agent itself.
- **Allowing is silent.** When the policy has no objection, the agent's own
  permission prompts still apply. The hook only ever refuses.
- **Anything unexpected inside the hook refuses the call, and so does taking
  longer than 4 seconds.** Claude Code lets a call through when a hook fails or
  times out, so the hook never does either. A normal check takes about 60 ms
  ([Overhead](#overhead)). Leave the hook's `timeout` at 10 seconds or more.
- **Every decision goes in `.paveo/audit.jsonl`**, hash-chained, and many hooks
  running at once share it safely.
- Start in shadow mode (`"mode": "shadow"` in the policy) to see what it would
  refuse in your own work before it refuses anything.

**Replay and learn.** `~/.paveo/bin/paveo replay claude-code`, run in the
project, puts each tool call from your past Claude Code sessions through
`.paveo/policy.json`. It prints one screen: how many calls it would have
refused, the pattern that refused each, and what the model calls would have cost
at API list prices (a subscription is billed differently). Add a `budget` to the
policy and it also says what that ceiling would have refused, counting each call
at what it cost. Pass `~/.claude/projects` to replay every project instead of
this one. **Nothing is stored, and nothing from the sessions is printed**:
counts, sums, and names your policy already holds.

`~/.paveo/bin/paveo learn claude-code --from-history` writes
`.paveo/policy.learned.json`: your policy, plus the name of every tool that ran
in those sessions and every argument it ran with, each permitted **unchecked**.
Names only, never a value, and only from calls that ran without error. It never
writes over `policy.json`: read it, then move it over yourself. The hook `init`
installs sees Bash, Write, Edit and NotebookEdit, so a rule for any other tool
matters only if you widen its matcher. `replay` and `learn` read Claude Code's
sessions only.

**Check the hooks you already have.** `paveo doctor`, run in a project, reads
`.claude/settings.json`, `.claude/settings.local.json`, `~/.claude/settings.json`
and any script a hook in them names, and names each hook that cannot work as
written, with the fix. It looks for two faults it can prove from the text: a hook
that reads `$CLAUDE_TOOL_INPUT`, `$TOOL_INPUT`, `$CLAUDE_FILE_PATH` or a relative,
which Claude Code does not set and its docs have never listed (a hook gets the
call as JSON on stdin), so it always reads an empty value; and a PreToolUse guard that exits 1 and never 2,
which Claude Code treats as a non-blocking error, so it never blocks. That is
judged only when every command the hook runs is a shell tool such as `grep`,
`jq` or `echo`, or a shell script it read: anything else may decide from out of
sight, so it is left alone. Other faults, a missing `jq` among them, are not looked for. It never
runs a hook and never prints a command or a script's text, only where each is. Its
report is on stdout; once it has checked at least one hook, one line on stderr asks
how it went, with a link to Discussions. Exit 0: nothing found; 1: a fault found; 2: a
file it could not read, one Claude Code would reject whole (not strict UTF-8, a
byte-order mark, `NaN` or `Infinity`, or a key Paveo reads of the wrong type: a hook event that is not a list, an `env` or
`enabledPlugins` value, a permission rule list, `permissions.defaultMode`), or a
`--settings` file that is not there, which it says it did not check. It does not need a
policy, and works whether or not you use Paveo's guard. Pass `--settings FILE`
to check another file.

### Test the hooks you already have

`paveo test`, run in a project, **runs** the PreToolUse hooks in the same three
settings files (like `make test`, run it only on a project whose code you would
run anyway). It sends each one, as the JSON Claude Code sends, every spelling of
five dangerous actions: a force push, a recursive delete, a hard reset, `git
clean`, and reading `.env`, 68 in all, such as `git -C . push --force`, `git push
paveo-test-probe +main`, `\rm -rf`, `find -delete`, `sh -c '...'`, `eval`, a flag in a
variable and the Read tool. The list is open: `src/paveo/bypasses.json`, dated, so a
report says which version it used, and anyone can add a spelling by pull request. It
reads each answer as Claude Code's hooks reference does: exit 2, a
`permissionDecision` of `deny` (in a `hookSpecificOutput` that names
`PreToolUse`, as the schema requires) or the older `decision: block` stops the
call, and `ask` stops it for a person to decide (in a session that starts in
default, plan or acceptEdits mode, or dontAsk, which refuses it; the report says
so whenever it counts one); any other exit, JSON that fails
the schema, a hook that will not start, and one past its own time limit let it
through, unless the hook sets `onFailure: "block"`. Shell-form hooks run under
`sh -c`, or bash when the hook says `"shell": "bash"`. No hook is given more than
10 seconds; one stopped then, before its own limit, cannot be told, and a hook
that times out is not run again for the other spellings. Every spelling is harmless if
a hook ran it: deletes and resets aim at a path or ref that does not exist,
pushes at a remote that does not exist.

```
paveo test --expect force-push --expect delete   # in CI: fails on any leak
```

Or as a GitHub Action, on every pull request (`--json` writes the same results
for a machine; the Action puts the report in the job summary):

```yaml
- uses: actions/checkout@v7
- uses: paveo-dev/paveo@v0.1.15
  with:
    expect: force-push delete hard-reset clean
```

It prints, per action, how many spellings were stopped and each one that passed
every hook. **Exit 1** when an action is stopped in some spellings and not others,
when no hook stops any action at all (the settings moved, or every hook was
switched off), or when an `--expect` action (`force-push`, `delete`, `hard-reset`, `clean`,
`secrets`) passes in any spelling; an action no hook stops is reported, and fails
only when expected. **Exit 2** when it could not tell: a settings file it could not
read, a hook it does not run (one with an `if` filter, a timeout that is not a
positive number, a matcher written in regex syntax JavaScript reads differently,
PowerShell, or an `http`, `prompt`, `agent` or `mcp_tool` hook) where no other
hook stops the call, or only an ask does (the hook not run might defer, which
outranks it), output over 1 MiB,
a hook that also stops `git status`, so its stops may not mean it read the
command, or managed settings on the machine
(`/Library/Application Support/ClaudeCode/`, `/etc/claude-code/` or a macOS
profile), which are not read yet and can switch every hook off; server-managed
settings leave nothing on disk, so it cannot see them. Otherwise 0. **A pass means these spellings were stopped, not that the
guard holds**: a list cannot be complete. Claude Code's own permission rules are not
tested yet, and it says so when a file has them: they can only stop more. A
settings file Claude Code would reject whole (not UTF-8, a byte-order mark, `NaN`
or `Infinity`, a key Paveo reads of the wrong type, as for `doctor`) is reported as unread; `init`
refuses to write a guard into one, and `paveo guard --selftest` counts no hook
in one, so a guard found only there fails as missing; a wrong type in a key it does
not read is not detected, and Claude Code then runs none of that file's hooks.
A stop that relies on `onFailure: "block"` holds on Claude Code v2.1.295 or later. Hooks
from plugins are not run yet either; they can stop more too, but one that defers
would outrank an ask, so with plugins enabled an ask is not counted as a stop. It
never prints a hook's command or
output, only where the hook is. A matcher that is a regular expression is the
repository's own, so for every agent paveo compiles and tries it only in a
separate, isolated Python, with 5 seconds for all of a run's matchers together:
one not answered in time makes the run say it could not tell, and makes `paveo
guard --selftest` count its hook as not seeing the call. A matcher longer than
1,000 characters is not read at all. Run against Paveo's own starter it stops every spelling of the four actions the
starter refuses but the ones "What the guard cannot do" names (a quoted
verb, a flag or `rm` in a variable, `rm -r` without `-f`), and reports `.env` not
guarded, since `init`'s hook does not watch Read.

#### Codex hooks

`paveo test codex` runs the same test on a project's Codex hooks, read as Codex
reads them (its hooks page and its source, `codex-rs/hooks`, read 25 Oct 2026).
It reads every place Codex loads hooks from, and all of them count, since in
Codex a higher layer adds hooks rather than replacing them: `hooks.json` and the
`[hooks]` table of `config.toml` in `/etc/codex`, then in `$CODEX_HOME`
(`~/.codex`), then in the `.codex` folder of each folder from the repository's
root down to where you run it. Each hook runs as Codex runs it, in your own
login shell (`$SHELL -lc`), without Claude Code's `CLAUDE_PROJECT_DIR`, even when
the shell you run it from has one. The
answers differ from Claude Code's, and the test follows Codex: **exit 2 stops a
call only when the hook also writes a reason to stderr**; on exit 0, a
`permissionDecision` of `deny` or the older `decision: block` stops it only with
a non-empty reason; **`ask` is not a stop**, since Codex reads it but does not
support it yet and lets the call on; and every other answer, an unknown field in
the JSON among them, is the hook failing, which lets the call through, since
Codex has no way for a hook to fail closed. A matcher of only letters, digits,
`_` and `|` names tools exactly; any other is a regex that may match anywhere in
the name, and one written in syntax Codex's regex engine refuses is not run.
Codex sends every shell command as `Bash`, and has no Read tool, so the 2
spellings that read `.env` with Claude Code's Read tool are left out (66 in
all); its shell spellings of reading `.env` stay in. Not run, and said: `mcp_tool`
hooks, hooks switched off in `/hooks`, and `prompt` and `agent` hooks, which Codex
skips too. A file Codex would skip whole (a `hooks.json` with an unknown
top-level key, a hook of a type Codex does not know, a timeout that is not a
whole number) is reported as unread, and the run cannot tell; so can
`/etc/codex/requirements.toml`, `/etc/codex/managed_config.toml` or a macOS
profile for `com.openai.codex`, which can add hooks of their own and are not
read. **Codex runs a hook only once someone has trusted it in `/hooks`, and a
project's only in a trusted project**; this tests what each hook does once
trusted, and says so. `paveo guard codex --selftest` checks the trust records on
your own machine. In the GitHub Action, add `agent: codex` under `with:`.

#### Cursor hooks

`paveo test cursor` runs the same test on a project's Cursor hooks, read as
Cursor's hooks page describes them (cursor.com/docs/agent/hooks, read 30 Oct
2026). **It has not been checked against a real Cursor session yet.** It reads
the organisation's `hooks.json` (`/Library/Application Support/Cursor/hooks.json`
on macOS, `/etc/cursor/hooks.json` on Linux), `~/.cursor/hooks.json` and the
project's `.cursor/hooks.json`, and every hook in them counts, since Cursor runs
every matching hook and the strictest answer wins: deny over ask, ask over
allow. It tests both hooks Cursor sends a shell command to:
`beforeShellExecution`, whose matcher is a regex tested against the whole
command, and `preToolUse`, whose matcher is tested against the tool's name,
`Shell`. Each hook runs in `/bin/sh` (the page names no shell for macOS and
Linux), a project's from the project root, a person's from `~/.cursor`, with
`CURSOR_PROJECT_DIR` and `CLAUDE_PROJECT_DIR` set as Cursor sets them. The
answers follow Cursor: exit 2, or `permission: "deny"` on exit 0, stops the
command; `ask` is never counted as a stop: from `beforeShellExecution` Cursor
asks the person, but its page does not say whether auto-run skips the question,
so the run cannot tell, and from `preToolUse` Cursor does not enforce it yet;
**JSON
Cursor cannot read, or an answer outside its schema, stops the command too**,
even without `failClosed`; any other failure (another exit, a crash, a timeout,
nothing printed) lets it through unless the hook sets `failClosed: true`. An
answer with no `permission` cannot be told, since the page does not say what
Cursor does with one. A file whose `version` is not 1, a matcher in syntax that
JavaScript and Python read differently, a matcher not tried within 5 seconds
(all are compiled and tried in one separate, isolated Python, one at a time,
the organisation's and your own before the project's, so a pattern that would
take minutes cannot hang the run), a hook of a type Cursor
does not know, a timeout that is not a positive number and `prompt` hooks are
not run, and the run cannot tell. Hooks set in Cursor's
team dashboard and plugins' hooks are not on disk, so they are not run; they can
only stop more. **Cursor runs a project's hooks only in a trusted workspace**;
this tests what each hook does once trusted, and says so when the project has
hooks of its own. The 2 spellings that
use Claude Code's Read tool are left out (66 in all). The example guard on
Cursor's hooks page, a `beforeShellExecution` hook matching `rm -rf`, stops 9
of the 19 shell spellings of a delete. In the GitHub Action, add
`agent: cursor` under `with:`.

**Codex.** Built against Codex's documented hook contract and its source, and
**it has refused `rm -rf` in a real session** (macOS, 26 Sep 2026). It checks
every shell command and every file edit (`apply_patch`, judged by the files the
patch names). Codex runs a hook only after you trust it, so `init codex` ends by
telling you to start Codex, trust the folder, type `/hooks` and trust the paveo
hook, and then run `~/.paveo/bin/paveo guard codex --selftest`. That self-test
fails if Codex has no record of you trusting it. Codex records trust against a
hash paveo does not reproduce, so if you change the hook afterwards, look in
`/hooks` too. Text typed into a shell Codex has already started is not checked.

**Cursor.** Built against Cursor's documented hook contract; **it has not been
tried in a real session yet**, because its hooks need a paid Cursor plan. Tell
us what happens if you try it. **It may refuse every call:** Cursor's
documentation says a permission hook that answers with no JSON blocks the action,
and the guard answers nothing when it allows, so as never to skip Cursor's own
approval prompt (D57). Until a real session settles which one Cursor does, treat
the Cursor guard as unproven. It checks every shell command, and `Write` and
`Delete`, the file tools Cursor's documentation names; a file tool under any
other name is not seen. The hooks are written with `failClosed: true`, so Cursor
refuses the call if the guard crashes or times out; `--selftest cursor` fails a
hook without it. Cursor does not document what its file tools send, so the
starter policy's field names for them are a guess: if the guess is wrong, every
file write is refused, never let through. Cursor runs a project's hooks only in
a workspace you have trusted.

Codex and Cursor hook files carry this machine's paths. If `init` created the
file it is kept out of git; if your team already had one, `init` adds to it and
says so.

**What the guard cannot do.** It matches patterns in a command. It does not
understand the shell, so `rm -r` without `-f`, a command hidden in a variable or an alias,
a script that does the deleting or pushing, a quoted verb (`git 'push'`), an
abbreviated flag (`git reset --ha`), a line continuation, or `git` given four or
more words of its own options before `push`, `reset` or `clean`, or a flag more
than 100 characters after the verb will get past it. The other way round,
within those 100 characters a force flag (`--force`, or a short cluster ending
in `f`, read without regard to case), a `+` after `push` or `--hard` is refused
even when it belongs to the next command (`git push && date +%s`, `git -C repo
push && docker compose -f x.yml up`): the guard reads text, not shell words, and
leans to refusing. So is `rm` followed within 100 characters by a recursive
and a force flag, even another command's (`rm -f x && grep -rn TODO src`,
`git rm --cached x && pytest -rf`), and `rmtree` wherever it appears; `find ... -delete` and `find ... -exec rm` are refused whatever they
match, `*.pyc` files included. A file the agent writes is judged by
its path as written and by the file it really reaches through symbolic links
and `..` (a hard link is judged by its own name),
but a link made after the guard said yes and before the agent opened the file
is followed: only the code that opens a file can close that race
([THREAT_MODEL.md](docs/THREAT_MODEL.md), T10). Because that real path is judged in full,
a policy rule that lists permitted folders by relative name (`src/.*`) must also
accept the full form, and a project kept inside `~/.codex` has its edits refused. It sees tool calls, not model calls, so
it caps no spend. For `requires`, `rate` and `repeat` it remembers each agent
session, in Claude Code, Codex or Cursor, in `.paveo/memory/`: one file per
session, private to you, holding salted digests and times, never a command, and
deleted after a week untouched. A damaged memory file refuses every call from
that session until you delete it, and a call with no session id is refused by
any tool that has such a rule. And an agent that finds a way to edit the hook
settings can switch it off. The starter policy refuses the obvious ways, which
is not all of them. It is a seatbelt, not a sandbox.

## Any MCP client, any language

`paveo mcp` sits between an MCP client and a local MCP server the client starts
over stdio, as Claude Desktop, Cursor and Windsurf do, and checks every tool call against your policy before the server sees it. A refused
call never reaches the server: the client gets a tool result marked as an error,
naming the rule, and the model reads it. There is no code to write, in whatever
language your agent is built.

In the client's MCP settings, put `paveo mcp` in front of the server's own
command, with full paths:

```json
{
  "mcpServers": {
    "filesystem": {
      "command": "/Users/you/.paveo/bin/paveo",
      "args": ["mcp", "--agent", "filesystem", "--dir", "/Users/you/project/.paveo",
               "--", "npx", "-y", "@modelcontextprotocol/server-filesystem",
               "/Users/you/project"]
    }
  }
}
```

The policy declares an agent with the id after `--agent`, and the server's tools
it may call, by the names the server lists them under. As everywhere in Paveo, a
tool the policy does not allow is refused, and every rule works here: argument
limits, order, rates and repeats, for as long as the client keeps the server
running. `paveo stop` refuses every call of a server already running, and each
decision is one record in the audit log. Paveo will not start the server if the
policy does not load, so the client shows a server that failed to start, and
nothing runs.

**What it does not do.** It judges tool calls only: resources, prompts and
sampling pass through as they are. It guards local servers over stdio, not
remote ones over HTTP. A path in a tool's arguments is judged as written: unlike
the coding-agent hooks, it does not follow symbolic links, so keep the server's
own allowed folders tight. The server is your own program, and what it does with a
call it was allowed is its own. A client set up to start the server directly,
without `paveo mcp`, is not guarded, and nothing in Paveo can see that. A change
to the policy applies when the client next starts the server. A message it
cannot read safely (not JSON, over 64 MiB, or naming the same key twice) is
dropped, and a batch that holds a tool call is refused.

## Plans

**Developer is free, forever: up to three agents per policy**, every rule, the
seatbelt, budgets and the audit log. A policy that declares more still loads: the
first three agents in the file keep working, and every call from the others is
refused as `plan_limit`, never let through. (0.1.7 covered five: after upgrading,
a fourth and fifth agent are refused; `paveo mcp` agents count too.) A paid plan raises the limit with a
licence key, `Paveo.from_file(..., licence="paveo1....")` in the library or the
key in `.paveo/licence.key` for the guard (which `init` keeps out of git). The key
is checked on your machine, with no call to anyone; reading one needs
`pip install 'paveo[team]'`, which adds the one dependency the free core never
has, `cryptography`, to verify its signature. A key that runs out reverts to the
free plan, with warnings from 14 days before, and never stops the agents the free
plan covers.

**Coding agents across a company are priced per developer**, since each
engineer's Claude Code, Codex or Cursor guard is one agent and would otherwise
never leave the free plan: $15 per developer a month, no minimum, up to 25
developers; a company with more than 25 is on Custom, an annual contract priced
with it. One developer
guarding their own machine stays on Developer, free. The company's key adds audit
evidence from each developer's log, email support, and an hour with us writing the
policy you roll out to everyone.

**Try it on the free plan**: Developer needs no key, no card and no account.
Paid plans and the design-partner offer are on the
[pricing page](https://paveo.dev/#pricing).

**Audit evidence (Team and up):** `paveo evidence --since 2026-09-01 --until
2026-09-30` writes a folder for an auditor: `report.html` (who was allowed and
refused what, by which rule, under which policy, and what the hash chain proves),
`records.csv` (every record in the period), `audit.jsonl` (the same records, byte
for byte, so the chain can be recomputed without trusting the report) and the
policy file when the period ran under it. It exports only a log whose chain
verifies from record 1, and says so; where it breaks, nothing is written. The
period is one unbroken run of the log, so if the clock went back while it was
written and a record dated inside the period follows one dated after it, it
refuses and says how to widen `--until`, rather than leave that record out. It is
still tamper-evident, not tamper-proof: what it adds is a copy of the chain's
hash in someone else's hands, so a later rewrite of the records it covers shows.
`--log` and `--policy` point it at a library's log and policy; the plan is read
from the `licence.key` beside the policy, else the one in `--dir`.

## Why

A prompt can talk your agent into anything. **It cannot talk its way past a
numeric constraint checked at the call site.** Policy is evaluated on the call —
the tool name and its arguments — not on the prompt.

## Overhead

What Paveo adds to a call, not counting the call itself. Measured on an Apple M4
laptop, Python 3.14, as medians; `make check` measures again and fails the build
over any budget (`tests/overhead.py`).

| | Median | Budget |
|---|---|---|
| `check_tool`, recorded | 50 µs | 250 µs |
| `check_llm` then `record`, both recorded | 130 µs | 600 µs |
| The coding-agent hook, a whole process, session memory included | 60 ms | 200 ms after Python starts |
| An MCP tool call, parsed, judged and recorded | 60 µs | 250 µs |

The hook is a new process for each tool call: Python starting takes about
12 ms of its 60, loading Paveo most of the rest, and the decision a few.

## What it does not do

Stated plainly, because you will find out anyway:

- **It is a guard rail, not a sandbox.** Code that calls your provider SDK
  directly bypasses it entirely.
- **The audit log is tamper-evident, not tamper-proof.** Anyone with write
  access can rewrite the chain; editing or reordering a record, or deleting one
  from the middle, invalidates every record after it and `verify_chain` names the
  sequence number where it broke.
- Deleting records from the *end* of the log is caught by a small **anchor file**
  written beside it, recording how far the log reached. **If that anchor is
  deleted too, truncation becomes undetectable again** — losing it is a warning,
  not a failure, because an anchor that can stop your agent when a sidecar goes
  missing is a worse problem than the one it solves. Back the two up together.
- It does not inspect model output, filter content, or detect jailbreaks.
- It does not manage your API keys.
- Tool checks only cover calls you route through `check_tool`. If your framework
  dispatches a tool without asking us, we never see it.
- **An agent in shadow mode is not protected by its rules**, its deny lists
  included, only by its budget.
  That includes the constraint above that a prompt cannot talk its way past:
  in shadow mode it is recorded, not enforced.
- The audit log is written straight through but not `fsync`ed per record, so a
  power loss can cost you the tail of it. Durability per record would cost
  milliseconds a call, which is the whole latency budget.
- It runs on macOS and Linux. The audit log needs POSIX file locking, so not
  Windows.

And for budgets:

- Budgets are per-process, and a restart clears the spend record.
- Threads and asyncio are both supported, **including against the same budget
  at once.** One lock serves both, held only around a few microseconds of
  arithmetic and never across an `await`. The cost: a coroutine can wait while a
  thread holds it. Microseconds normally; it grows with the number of threads
  contending for one ceiling.
- Input is estimated conservatively from the request's bytes, so it will deny
  too eagerly on prompt-heavy workloads. On top of the bytes we add what the
  provider adds: its tool instructions, the tools it defines, and the most an
  image can cost. **What we cannot bound, we refuse**: content sent by URL or
  `file_id`, PDFs, server-run tools like web search, and any request parameter we
  have not verified. Each is added once someone needs it.
- **A model priced in your policy is only as bounded as your price.** Its text is
  bounded by its bytes, which holds for any tokenizer, and anything that is not
  text is refused, and so is a cache setting, which bills above the input price. What that provider adds on its own side, we cannot see.
- **There is a second reason it denies early.** Your prompt's split between fresh
  tokens, cache reads and cache writes is only reported *after* the call, so we
  reserve at the most expensive of those rates. On a cache-heavy workload the
  true rate can be 0.1×, 0.05× on Opus 5.5 or 0.025× on Fable 5.1, and we
  reserve at 2.0×. Safe, never silent — the rate used is in every audit line.
- **Set `inference_geo` explicitly if you can.** Left unset, a workspace default
  we cannot see decides it, and US-only inference costs 1.1× on the models that
  offer it, so on those we reserve and charge at 1.1×. On a global workspace that
  is 10% too high, in the safe direction, and the audit line says `unset`.
- **On OpenAI, two things are charged high because we cannot see them**: the
  10% regional-processing uplift on the ten models eligible for it (it depends
  on your endpoint or project region), and 1,000 tokens whenever you pass tools,
  for the formatting OpenAI adds and does not publish.
- **OpenAI cache writes are charged at the 30-minute rate** (1.25× input on
  GPT-5.6 and later), the only cache lifetime OpenAI's SDK lists as supported
  (its generated types as updated on 28 Oct 2026); a request asking for another
  lifetime, or a `prompt_cache_retention` OpenAI does not list, is refused. If a response reports cache writes some way Paveo does
  not read, Paveo charges them at the highest rate and stops allowing calls until
  it is upgraded, rather than under-charge. On an OpenAI model, a response with no
  write count at all (a proxy can drop it) is charged as if every uncached token
  were a write; on a model whose price your policy declares, as plain input, the
  price you declared. The
  one case no reader can tell apart: a proxy that writes a false `0` there makes
  writes look like plain input, 0.25× under.
- **Prices are the Claude API's own.** On Amazon Bedrock and Google Cloud the
  cloud provider sets the price, and a Google Cloud model id would be priced at
  Claude API rates, which can be lower than what you pay there.
- **A price change on the provider's side is not detected**, only dated. If they
  introduce a token class we don't know, Paveo stops allowing calls rather
  than guess at a ceiling it can no longer enforce. That is deliberate, it is an
  outage, and in v1 **`fail_open` does not override it**: recovery is an upgrade.
  Know this before you deploy.

## Your data

**Paveo opens no sockets.** Your prompts, completions and tool arguments
never leave your process, and are never written to the audit log — only
identities, decisions, reasons and costs.

Don't take our word for it. `tests/test_no_egress.py` fails the build if a
socket is constructed during any library operation, and you can verify a release
yourself with a firewall rule or `tcpdump`. **Please do.**

## Install

```
pip install paveo            # zero required dependencies
pip install 'paveo[team]'    # only to read a paid licence key: adds cryptography
```

Python 3.11 or newer, on macOS or Linux. Releases are published only by this
repository's release workflow, through PyPI's trusted publishing; see
[`SECURITY.md`](https://github.com/paveo-dev/paveo/blob/main/SECURITY.md).

## Licence

**Open source: [Apache License 2.0](https://github.com/paveo-dev/paveo/blob/main/LICENSE)**,
except one file. The guard, the starter policies, `doctor`, `test`, the MCP guard,
replay, budgets, the audit log and the library are all Apache-2.0: use them,
change them, ship them inside your own product, commercial or not.

The exception is the paid audit evidence export, `src/paveo/_evidence.py`, under
the [Elastic License 2.0](https://github.com/paveo-dev/paveo/blob/main/LICENSE-ELASTIC):
you can read it, and its licence says you may not provide it to others as a hosted
or managed service, or "move, change, disable, or circumvent the license key
functionality" in it. That is a licence term, not a lock: the key check itself is
in the Apache-licensed core, where anyone can read it. **Nothing free ever loads
the export**: only `paveo evidence` imports it, and a test fails if the guard or
any free command does.

Being able to *read* the code is half of why a security-conscious team would
accept it in their call path; the other half is that nothing leaves their
machine. Neither substitutes for the other.

## Status

Early: the first release is 0.1. Tool rules, model-call ceilings, the coding-agent guard, the
audit log and audit evidence all run, for Anthropic, OpenAI, Gemini and any model
priced in the policy.

Built in the open by one person. Issues are answered, occasionally slowly, and
there will be quiet stretches. Nothing is abandoned; it just pauses. Security
reports go through GitHub's private reporting, never an issue: see
[`SECURITY.md`](https://github.com/paveo-dev/paveo/blob/main/SECURITY.md).

## From source

```
git clone https://github.com/paveo-dev/paveo.git && cd paveo
make dev          # build .venv and install
make example      # a refund bot's tool calls, then a runaway loop refused at $1
make check        # lint, types, tests, the no-egress check, and the overhead
```

## Docs

Absolute links, because this file is the PyPI long description and PyPI does not
resolve relative ones.

- [`docs/SPEC_V1.md`](https://github.com/paveo-dev/paveo/blob/main/docs/SPEC_V1.md) — what it does and how
- [`docs/THREAT_MODEL.md`](https://github.com/paveo-dev/paveo/blob/main/docs/THREAT_MODEL.md) — what it defends against, and what it doesn't
- [`docs/SECURITY_REVIEW.md`](https://github.com/paveo-dev/paveo/blob/main/docs/SECURITY_REVIEW.md) — answers for your security review, each with where to check it
- [`SECURITY.md`](https://github.com/paveo-dev/paveo/blob/main/SECURITY.md) — how to report a vulnerability, and how releases are published

Comments in the code cite design decisions as D-numbers (`D44`) and working rules
as `Rule N`. That record is not published yet; ask in an issue about any one.
