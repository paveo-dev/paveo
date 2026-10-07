# Paveo

**Stops the call your agent shouldn't make — before it makes it.**

Every tool in this space tells you what your agent *already did*. Paveo sits
between your agent and everything it calls (a tool, a shell command, a model)
and checks each call against a policy you write: which tools, with which
arguments, in what order, how often, at what cost. A call the policy does not
allow is refused before it leaves, and every decision goes into a hash-chained
audit log on your disk. That is **admission control**, in the Kubernetes sense:
a gate that inspects the request and rejects it before it takes effect.

**Nothing leaves your machine.** Paveo runs in your own process, opens no network
sockets (a test fails the build if it does), and installs no other package.

**Design partners.** With a small group of teams putting agents into production,
we are building what comes next to the rules and the record below: judgment for
grey-zone calls the rules allow, which can only make a decision stricter, and
approval by a person when a call is unclear. Neither is in the library yet.
[Want in?](https://paveo.dev/#partners)

```
pip install paveo            # Python 3.11+ (python3 --version), macOS or Linux
```

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
Cursor, `stop`, `resume`, `replay`, `learn`, `evidence`, `mcp`, `doctor`), starter
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

If you move or delete that virtualenv, the hook stops working **silently**:
Claude Code treats a hook it cannot run as one with no objection. So after any
change, run `~/.paveo/bin/paveo guard --selftest` in the project. It checks
every configured hook really refuses, and says so if a paveo hook sits in your
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
  ([Overhead](#overhead)). Leave the hook's `timeout` above 4 seconds.
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
which Claude Code never sets (a hook gets the call as JSON on stdin), so it
always reads an empty value; and a PreToolUse guard that exits 1 and never 2,
which Claude Code treats as a non-blocking error, so it never blocks. That is
judged only when every command the hook runs is a shell tool such as `grep`,
`jq` or `echo`, or a shell script it read: anything else may decide from out of
sight, so it is left alone. Other faults, a missing `jq` among them, are not looked for. It never
runs a hook and never prints a command or a script's text, only where each is. Exit 0: nothing found; 1: a fault found; 2: a
file it could not read, or a `--settings` file that is not there, which it says
it did not check. It does not need a
policy, and works whether or not you use Paveo's guard. Pass `--settings FILE`
to check another file.

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
us what happens if you try it. It checks every shell command, and `Write` and
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
understand the shell, so `rm -r -f`, a command hidden in a variable, or a script
that does the deleting will get past it. A file the agent writes is judged by
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

**Developer is free, forever: up to two agents per policy**, every rule, the
seatbelt, budgets and the audit log. A policy that declares more still loads: the
first two agents in the file keep working, and every call from the others is
refused as `plan_limit`, never let through. A paid plan raises the limit with a
licence key, `Paveo.from_file(..., licence="paveo1....")` in the library or the
key in `.paveo/licence.key` for the guard (which `init` keeps out of git). The key
is checked on your machine, with no call to anyone; reading one needs
`pip install 'paveo[team]'`, which adds the one dependency the free core never
has, `cryptography`, to verify its signature. A key that runs out reverts to the
free plan, with warnings from 14 days before, and never stops the agents the free
plan covers.

**Coding agents across a company are priced per developer**, since each
engineer's Claude Code, Codex or Cursor guard is one agent and would otherwise
never leave the free plan: $15 per developer a month, minimum 5. One developer
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
from `--dir`'s `licence.key`.

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

[Elastic License 2.0](https://github.com/paveo-dev/paveo/blob/main/LICENSE)
— **source-available, not open source.**

Read every line, run it, change it, embed it in your own product, commercial or
not, and pay nothing. What it forbids is narrow and worth reading in full, but in
short: you may not offer Paveo itself to third parties as a hosted or managed
service; you may not circumvent licence-key functionality; and you may not remove
the licensor's notices. If you pass a copy on, the terms go with it, and a
modified copy has to say prominently that it was modified.

Being able to *read* the code is half of why a security-conscious team would
accept it in their call path; the other half is that nothing leaves their
machine. Neither substitutes for the other, which is why the licence is
source-available rather than closed.

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
