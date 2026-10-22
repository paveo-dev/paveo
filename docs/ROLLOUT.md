# Rolling Paveo out to a team

Two ways to put the same rules on every engineer's coding agent, without each
person setting it up by hand:

- **A. An administrator sets it for every machine**, for Claude Code and Codex:
  the hook, one policy, a standalone Python and paveo, all inside `/opt/paveo`,
  and each agent's managed settings pointing at it. On every call the hook
  checks that everything in `/opt/paveo`, file by file, and every folder above
  it is owned by root and writable by no one else, so the agent cannot change
  or switch it off through any file the hook runs. What it does not cover, the
  environment the agent starts with among it, is at the end.
- **B. Claude Code only, no device management**: Paveo's Claude Code plugin. It
  is a seatbelt against mistakes, not against an agent set on removing it: every
  file it uses belongs to the person.

Both refuse every call when they cannot judge it: a hook that cannot find
paveo, read its policy or write its log answers with a refusal, because these
agents let a call through when a hook fails any other way.

**Cursor is not in the rollout yet.** Its guard has not been tried in a real
session, and by Cursor's documentation it may refuse every call (README,
"Cursor").

## A. Every machine, set by an administrator

### 1. Install Python, paveo, the hook and the policy under `/opt/paveo`

Everything the hook trusts lives in this one folder, owned by root: the hook,
the policy, a Python and the venv paveo runs in. A venv reads the standard
library of the Python it was made from, so that Python must be inside
`/opt/paveo` too, where only root can change it: a standalone Python 3.11 or
newer, such as a python-build-standalone "install_only" archive, unpacked at
`/opt/paveo/python`. A system, python.org or Homebrew Python is refused,
because the person, and so their agent, may be able to write to it.

```
sudo mkdir -p /opt/paveo
sudo tar -xzf cpython-3.12.*-install_only.tar.gz -C /opt/paveo      # makes /opt/paveo/python
sudo /opt/paveo/python/bin/python3 -m venv /opt/paveo/venv
sudo /opt/paveo/venv/bin/pip install paveo          # with a licence key: 'paveo[team]'
sudo cp rollout/guard rollout/policy.json /opt/paveo/
sudo chown -R root:wheel /opt/paveo && sudo chmod -R go-w /opt/paveo
```

(On Linux, the group is `root` rather than `wheel`.) The files are in this
repository's `rollout/` folder.

On every call, before Python starts, the hook checks that the venv runs on
`/opt/paveo/python` and does not read the system's site-packages; that every
file and folder in `/opt/paveo`, and every folder above it up to `/`, is owned
by root and writable by no one else; and that every link in `/opt/paveo` leads
somewhere in `/opt/paveo`, written as a path there. It refuses otherwise, naming
the first thing that fails. Its first line runs it again, and then Python, in a
cleared environment that keeps only `HOME` (where the person's log goes) and
sets `LC_ALL=C`, so nothing else passed in the environment reaches the tools it
runs or Python. It needs `/usr/bin/env` and `/bin/sh`, and refuses every call
without them. **The environment the agent itself is started with is
trusted**: the shells that start the hook read some variables before any line
of it runs (below). The check takes about 70 ms
for a Python of some 11,000 files (130 ms the first time), measured on a Mac.
Until paveo is installed, the hook refuses every call and says why.

If `/opt` is itself a link on your system (Fedora Silverblue, for one), use its
real path, such as `/var/opt/paveo`, in every command and config here: the hook
compares real paths and refuses the linked one.

`rollout/policy.json` is the Claude Code and Codex starter policies in one file.
Edit it to fit. A licence key goes at `/opt/paveo/licence.key`, which needs
`paveo[team]` installed in the venv. When none is there, a signed key in the
person's `~/.paveo/managed` counts; a key only raises how many agents the policy
covers, and a signed one cannot be made up.

### 2. Point each agent at it

**Claude Code.** Merge `rollout/claude-code-managed-settings.json` into the
**one managed source you already deploy**: the claude.ai admin console, an MDM
profile, or the file `/Library/Application Support/ClaudeCode/managed-settings.json`
on macOS and `/etc/claude-code/managed-settings.json` on Linux. They are not
interchangeable. By default Claude Code reads only the highest-ranked source
that delivers any policy, console first, then MDM, then the file, and skips the
rest without a warning: a hook placed in the file on a machine that also gets a
console or MDM policy never loads. To apply every source instead, set
`managedSourcesBehavior` to `"merge"` (Claude Code 2.1.242 or later). Source:
Anthropic's managed settings documentation, "How Claude Code combines managed
sources", read 13 Oct 2026 (https://code.claude.com/docs/en/managed-settings).
Managed hooks run in every project, and only managed settings can turn them off.
To stop people adding hooks of their own as well, add
`"allowManagedHooksOnly": true`.

**Codex.** Add `rollout/codex-requirements.toml` to Codex's managed
`requirements.toml` (see OpenAI's managed configuration documentation for where
it lives on each platform). Codex trusts a managed hook by policy, so nobody has
to approve it in `/hooks`, and `[features] hooks = true` keeps hooks on for a
person who turned them off.

### 3. Check it on a machine

```
echo '{"paveo_selftest": true}' | /bin/sh /opt/paveo/guard claude-code; echo "exit $?"
```

A working hook refuses this test call with `exit 2` and a message saying the
input "is not a tool call this guard reads": it started, loaded the policy and
opened the log. Any other result names what is wrong. Repeat with `codex`.

That proves the hook works, not that Claude Code runs it. Then start a Claude
Code session on the same machine and run `/status`: it names the managed source
Claude Code used and the ones it skipped, and the source holding the hook must
be the one used. Last, ask the agent to run `rm -rf ./paveo-check` and see it
refused.

### Each person's log, the panic button and evidence

- Every decision goes into `~/.paveo/managed/audit.jsonl`, hash-chained, on that
  person's machine. Nothing is sent anywhere.
- To stop every call on one machine:
  `/opt/paveo/venv/bin/paveo stop --dir ~/.paveo/managed --policy /opt/paveo/policy.json`,
  and `resume` with the same arguments to lift it. On a machine where the hook
  has never run, `mkdir -p ~/.paveo/managed` first: `stop` refuses a folder that
  does not exist, so a mistyped one cannot look stopped.
- On Team and up, `/opt/paveo/venv/bin/paveo evidence --dir ~/.paveo/managed
  --policy /opt/paveo/policy.json` turns a period of one person's log into a
  report for your security team.

## B. Claude Code only: the plugin

```
/plugin marketplace add paveo-dev/paveo
/plugin install paveo@paveo
```

Each person runs these once in Claude Code, after installing paveo for
themselves (`python3 -m venv ~/.paveo && ~/.paveo/bin/pip install paveo`). To
install the plugin for everyone and stop anyone disabling it, put this in your
managed settings instead:

```json
{
  "extraKnownMarketplaces": {
    "paveo": { "source": { "source": "github", "repo": "paveo-dev/paveo" } }
  },
  "enabledPlugins": { "paveo@paveo": true }
}
```

Claude Code installs it at the start of each person's next session. It uses the
Claude Code starter policy it ships and the paveo at `~/.paveo`, and keeps each
person's log in its data folder, `~/.claude/plugins/data/paveo-paveo/`. Panic
button: `~/.paveo/bin/paveo stop --dir ~/.claude/plugins/data/paveo-paveo
--policy <the plugin's policy.json>`, where that policy is the `policy.json` in
the plugin's folder under `~/.claude/plugins/cache/`.

For your own policy, or for protection against an agent trying to remove it,
use A.

## What it does not cover

- **It matches patterns.** It does not understand the shell, so a disguised
  command (`rm -r` without `-f`, a script that does the deleting, a path written with a
  wildcard) can get past it. It is a seatbelt, not a sandbox.
- **The log, the stop file and the session memory are the person's**, in their
  home folder. An agent that gets a command past the patterns can delete them.
  The hash chain makes an edited or truncated log show; it cannot stop one being
  deleted.
- **In the plugin (B), every file is the person's**: the paveo at `~/.paveo` and
  the plugin's own policy under `~/.claude/plugins`. A disguised command can
  replace either and switch the guard off. A protects against that; B does not.
- **A person with administrator rights**, or one who removes the managed
  settings themselves, can switch A off.
- **The environment the agent is started with.** The agent runs a hook
  through a shell, and shells act on some variables before the hook's first
  line: a loader variable such as `LD_PRELOAD` or `DYLD_INSERT_LIBRARIES`, and
  for bash (macOS's `/bin/sh`) `SHELLOPTS` with `PS4`, which runs a command.
  The hook clears the environment for everything it starts, but not for the
  shells that start it. So a person who starts their agent with such a variable
  set can switch the hook off for that agent.
- **What sets that environment.** The starter policy does not stop the agent
  writing the engineer's shell startup files (`~/.zshrc`, `~/.zshenv`,
  `~/.bashrc`, `~/.profile`) or running `launchctl setenv` on macOS. Either can
  set such a variable for agents started later; `~/.zshenv` is read by every zsh,
  so if the agent starts hooks through zsh it could act on the next call. Add
  rules for those files and commands to the policy if that matters to you.
- **In the plugin (B), the environment too**: it keeps the environment it is
  started with, so a variable or an exported shell function there can switch
  it off as well.
- **A timeout the agent enforces** lets the call through in Claude Code and
  Codex. The guard refuses after 4 seconds on its own, inside the hook's
  10-second timeout, but Python starting slowly on a cold first run is not
  covered by that deadline.
- **Each agent sees only the tools its hook covers**: for Claude Code, Bash,
  Write, Edit and NotebookEdit; for Codex, Bash and apply_patch.
- **Logs stay on each machine.** Gathering every person's log into one place is
  not built.
