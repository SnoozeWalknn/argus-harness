# argus-harness

A headless coding agent that works with any model: a local model on
llama-server, Ollama, vLLM or LM Studio, or Claude, GPT and Gemini through
their own APIs. It borrows the best ideas from Claude Code (subagents, hooks,
plan mode, todos, progressive skills), Codex (sandboxed commands and approval
policies), OpenCode (switching models mid-session, LSP diagnostics, session
resume) and Omarchy (defaults that just work, a one-command installer, a
keyboard-driven TUI with themes).

It is also a measurement instrument: every run is logged to SQLite in enough
detail to say why it failed and whether config B did better than config A.

Design: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) (the headless core) and
[docs/MODEL-AGNOSTIC.md](docs/MODEL-AGNOSTIC.md) (providers, profiles, and
everything on top). A 60-second promo video and its Remotion source live in
[promo/](promo/).

## Install

```sh
curl -fsSL https://raw.githubusercontent.com/SnoozeWalknn/argus-harness/main/install.sh | sh
```

The installer needs no sudo and is safe to re-run (that is how you upgrade).
It installs [uv](https://docs.astral.sh/uv/) if it is missing (or uses pipx
with `ARGUS_NO_UV=1`), installs argus with the TUI as a uv tool, writes
`~/.config/argus/config.toml` unless one exists, and runs `argus doctor`,
which reports what works on this machine and what to install for the rest
(bubblewrap, ripgrep, a language server). `ARGUS_SOURCE` installs from a
local checkout or another git URL.

From a checkout: `uv venv && uv pip install -e '.[dev]'` (Python ≥ 3.11).

## Quick start

```sh
cd ~/src/proj
argus                                        # the TUI
argus run "test_add fails; fix calc.py" -v   # headless; -v streams reasoning and output
argus show last                              # transcript, tokens, timings, cost
```

With no model configured, argus picks one: a local server on its usual port
(llama-server :8080, Ollama :11434, LM Studio :1234, vLLM :8000), else the
first cloud API whose key is set (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`,
`GEMINI_API_KEY`/`GOOGLE_API_KEY`, `OPENROUTER_API_KEY`). It says which on
stderr. To choose, pass `-m`:

```sh
argus run -m anthropic/claude-opus-5 "..."
argus run -m openai/gpt-5 "..."
argus run -m gemini/gemini-2.5-pro "..."
argus run -m ollama/qwen3-coder:30b "..."
argus run -m openrouter/qwen/qwen3-coder "..."
argus run -m llama/qwen3-coder-30b "..."     # llama-server at model.base_url

# a local model with tool calling (needs --jinja), and optionally a small model for compaction
llama-server -m Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf --jinja -c 65536 -ngl 99 --port 8080
llama-server -m Qwen3-1.7B-Q8_0.gguf --jinja -c 16384 --port 8081
```

Configuration is TOML; every key is optional and documented in
[examples/argus.toml](examples/argus.toml). Files are layered, lowest
precedence first: `~/.config/argus/config.toml` (global, written by
`argus init`), the workspace's `.argus/config.toml`, `-c FILE`, then `-m`
and `-o` overrides such as `-o agent.protocol='"grammar"' -o model.temperature=0.2`.

## Commands

| command | what it does |
|---|---|
| `argus` / `argus tui` | the terminal UI (when Textual is installed and stdout is a terminal) |
| `argus run TASK` | run one task (`-f FILE`, `-` for stdin, `--json` for a machine-readable result, `-m SPEC`, `--plan`, `--approval P`, `-y`, `--continue`, `--session S`, `--record DIR`) |
| `argus sessions` / `argus resume [SESSION] TASK` | list sessions / continue one (possibly on another model) |
| `argus runs` / `argus show RUN` | list runs / full transcript with reasoning, tool calls, tokens, timings, cost |
| `argus failures [--batch B] [--tag T]` | failure-tag summary and recent examples |
| `argus overhead [--all]` | token cost of system prompt, AGENTS.md, skills, each tool, template |
| `argus models [SPEC] [--detect]` | the model catalog and your profiles; with a spec, what applies to that model |
| `argus tune SPEC` | try each protocol (and thinking on/off with `--thinking`) on a suite and save the best to the model's profile |
| `argus suite run SUITE` | replay a saved task suite, each task in a fresh workspace, with checks |
| `argus suite add SUITE --run RUN` | save a logged run's task into a suite |
| `argus ab A.toml B.toml --suite SUITE -n 3` | interleaved A/B over a suite, with a comparison report (`--ma`/`--mb` for model specs) |
| `argus report [BATCH]` | report for a suite/A-B batch (`--json`) |
| `argus checkpoints / diff / restore RUN` | inspect and roll back workspace checkpoints |
| `argus init` / `argus doctor` | write the global config / report what works here (`--json`) |
| `argus mock-server --script S.json` | scripted stand-in for llama-server and the cloud APIs |

Run, batch and session ids accept a prefix or `last`.

## Providers

| provider | API | reasoning carried across turns |
|---|---|---|
| `llama_server`, `ollama`, `vllm`, `lmstudio`, `openrouter`, `openai_compat` | OpenAI chat completions (`/v1/chat/completions`) | `reasoning_content` / `reasoning` (not replayed) |
| `anthropic` | Messages (`/v1/messages`) | thinking blocks with signatures, replayed exactly; prompt caching |
| `openai` | Responses (`/v1/responses`, stateless) | encrypted reasoning items, replayed to the same model |
| `gemini` | `streamGenerateContent` | thought signatures, replayed exactly |

`provider = "auto"` (the default) decides by the base URL's host and, for
OpenAI-compatible servers, probes which server it is (`/props`, `/api/show`,
`/api/v0/models`, `/v1/models`) for the context window, the model name, tool
and thinking support, and OpenRouter's pricing. API keys come only from the
environment: the provider's usual variable, or the one named by
`model.api_key_env`. Thinking is one knob across providers
(`model.thinking`, `model.effort`, `model.thinking_budget`), translated to
each API's format.

## Model profiles and tuning

A profile says how to drive a model: context size, tool-call protocol
(native for frontier models, `grammar`/`json_schema` for local ones), thinking
format, chat-template quirks and pricing. Built-in profiles live in
[argus/data/models.toml](argus/data/models.toml); yours go in
`~/.config/argus/models.toml` (or `$ARGUS_MODELS`) and can `extends` a
built-in one. The best-matching profile fills only the keys you did not set,
and what the server reports (e.g. llama-server's `n_ctx`) beats the catalog.
`argus models qwen3-coder-30b` shows what applies and why.

`argus tune ollama/qwen3-coder:30b` runs a small built-in suite (or
`--suite S`) under every protocol the provider supports, ranks them by the
lower bound of the pass rate's Wilson interval (then tokens and time), and
writes the winner into a profile for that exact model. Costs are computed
per turn from the profile's pricing, including cache reads and writes, and
shown in `argus show` and the TUI.

## Approvals and the sandbox

`agent.approval` (or `--approval`):

| policy | `edit` | `bash` |
|---|---|---|
| `read-only` | denied | read-only sandbox, no network |
| `ask` | asks every time | workspace-write sandbox; asks except for known-safe read-only commands |
| `auto` (default) | allowed in the workspace, asks outside it | workspace-write sandbox; a sandbox denial asks to re-run unsandboxed |
| `full` | allowed | no sandbox |

Commands run under [bubblewrap](https://github.com/containers/bubblewrap)
(the filesystem read-only except the workspace and `sandbox.writable`,
a private `/tmp`, optionally no network) or, where bubblewrap cannot create
namespaces, Landlock (writes confined to the same paths). `sandbox.required`
refuses to run commands without one. "Ask" goes to the terminal, the TUI's
modal, or `agent.headless_approval` (deny by default) when nobody can answer;
`-y` approves everything. Every decision is logged in `approvals`.

## Plan mode, todos, subagents, hooks, skills

- **Plan mode** (`--plan`, or tab in the TUI): read-only tools and the
  read-only policy; the answer is a plan. Continuing the session in build
  mode carries it out.
- **Todos**: a `todo` tool that replaces a short checklist, shown in the TUI
  sidebar. `tools.todo = "auto"` turns it on for frontier models only, since
  every tool costs a small local model prompt tokens (likewise `agent.subagents`).
- **Subagents**: a `task` tool starts an agent with its own fresh context and
  returns only its answer. Built in: `explore` (read-only) and `general`;
  define more as Markdown files with front matter (`name`, `description`,
  `tools`, `model`) in `.argus/agents`, `.claude/agents` or
  `~/.config/argus/agents`. A subagent may run on another model; its run is
  logged with the parent run and turn.
- **Hooks** (`[[hooks]]`): commands run at `session_start`, `user_prompt`,
  `pre_tool`, `post_tool` and `stop`, given the event as JSON on stdin. Exit 2
  blocks the prompt or tool call, or makes the agent keep going at `stop`
  with the hook's stderr as feedback; a `pre_tool` hook can also print
  `{"decision": "allow" | "deny", "reason": ...}` to answer an approval.
- **Skills** load progressively: the prompt lists names and descriptions,
  `skill(name)` loads a skill's instructions, `skill(name, file)` one of its
  files.

## Sessions, switching models, LSP

Interactive runs belong to a session; `argus run --continue`, `--session S`
and `argus resume` continue one with its history. The next run may use
another model, provider or protocol (`-m`, ctrl+o in the TUI): the history
is converted between native tool calls and constrained JSON actions, and
provider-specific reasoning state is kept only where the new model can accept
it.

After each edit, a language server for the file (pyright/basedpyright, pylsp,
typescript-language-server, gopls, rust-analyzer, clangd; whichever is
installed, or `lsp.servers`) is asked for diagnostics, and errors in the
edited file are appended to the edit's result. Works over SSH too.

## The TUI

A thin client of the same headless agent (the agent runs in a worker thread;
the UI shows what the reporter callbacks say). Transcript with reasoning,
tool calls, results and diffs; a sidebar with todos, changed files and the
session; a status line with model, provider, mode, approval policy, tokens,
cost and context use.

| key | |
|---|---|
| enter | send (a follow-up continues the session) |
| tab | plan / build mode |
| ctrl+o | switch model (mid-session) |
| ctrl+s / ctrl+n | open a session / start a new one |
| ctrl+d | diff of the last run |
| ctrl+t | next theme |
| esc | interrupt the run |
| y / a / n | approve once / always / deny, in an approval |
| ctrl+p / ctrl+q | command palette / quit |

Themes: Textual's built-ins plus matte-black, everforest, kanagawa,
osaka-jade and ristretto. With `tui.theme = "auto"` the TUI uses your last
ctrl+t choice, else Omarchy's current theme (`~/.config/omarchy/current/theme`),
else tokyo-night.

## Tool-call protocols

Selected with `agent.protocol`; A/B them with `argus ab`.

| protocol | how tools are offered | how calls are constrained |
|---|---|---|
| `native` | OpenAI `tools` field, rendered by the chat template | llama-server's lazy tool-call grammar |
| `json_schema` | one-line signatures in the system prompt | `response_format` JSON schema over `{"tool", "args"}` with one branch per tool |
| `grammar` | one-line signatures in the system prompt | GBNF generated by argus: no free whitespace, fixed key order, optional `<think>` prefix |

Because the sampler enforces argument structure under the constrained
protocols, the prompt only needs signatures such as
`read(path, offset?, limit?) - show a file with line numbers`, which is where
most of the token saving comes from. Arguments are validated harness-side
under every protocol, so a bad call always becomes an error result the model
can recover from (plus a `malformed_call` tag).

`argus overhead --all` measures this exactly with llama-server's
`/apply-template` and `/tokenize` (falling back to an estimate offline).
Every run also stores its overhead (`runs.overhead_tokens`, `overhead_json`).

## Logging

One SQLite file (`$ARGUS_DB`, default `~/.local/share/argus/argus.db`),
committed every turn:

- `runs`: config (full JSON + hash), protocol, status, final answer, token totals, overhead, check result
- `turns`: prompt/cached/completion/reasoning tokens, full reasoning and content, finish reason,
  TTFT, prompt and generation ms, tokens/s, abort reason, raw response, and the ids of the
  messages in that request (`context_ids`), so each turn's exact prompt is reconstructable
- `messages` (append-only), `tool_calls` (args, result, duration, fs changes, checkpoint)
- `failures`, `checkpoints`, `skill_invocations`, `compactions`, `events`, `batches`

```sql
-- which failure tags does each config produce?
SELECT r.config_name, f.tag, COUNT(DISTINCT r.id) FROM failures f JOIN runs r ON r.id = f.run_id GROUP BY 1, 2;
-- per-turn token profile of a run
SELECT idx, prompt_tokens, cached_tokens, completion_tokens, reasoning_tokens, gen_tps FROM turns WHERE run_id = ?;
```

### Failure tags

| tag | when |
|---|---|
| `loop` | the same call returns the same result repeatedly, a short cycle repeats, or a generation degenerates into repeating text (the model is nudged, then the run aborted) |
| `overrun` | output continues past the final answer (role markers, repeated summary), tool turns continue after the model declared completion, or (suite `--oracle`) the task check already passed but the agent kept going |
| `malformed_call` | unparseable or schema-invalid arguments, unknown tool, unparsed tool-call markup, empty replies |
| `token_cap` | `max_tokens` cut-off, runaway reasoning, reasoning budget, context overflow, run token budget |

Also `max_turns`, `timeout`, `server_error`, `fs_violation`, `interrupted`,
`refusal` (the model or API declined) and `blocked` (a hook blocked the prompt).
With streaming on, generation is aborted mid-stream when reasoning exceeds
`agent.max_reasoning_tokens` or degenerates into repetition; the turn is
retried once (with thinking disabled if the reasoning ran away).

## Tools

`read`, `edit`, `bash`, `glob`, `grep` (plus `skill`, `todo` and `task` when enabled). All go
through an executor, so they behave identically locally and over SSH
(`executor.kind = "ssh"`, one `ssh` call per operation over a shared control
socket, with remote-side timeouts).

- `edit` is search/replace. When `old` is not exact it tries, in order: copied
  line-number prefixes, trailing whitespace, indentation (re-indenting `new`),
  unicode punctuation, in-line spacing, then similarity above
  `tools.fuzzy_threshold` if one candidate clearly wins. Failures show the
  closest text with line numbers and a diff.
- `bash` output is cleaned (ANSI, `\r` progress bars, repeated lines) and cut to
  head + tail, keeping error-looking lines from the middle.

## Checkpoints and filesystem checks

A shadow git repository per workspace (its own git dir and index, never your
repo's) takes a checkpoint before every `edit`/`bash` call. After each call
the change set is diffed: an edit must change exactly its file, bash results
list the files they changed, and surprises are tagged `fs_violation`.
`argus diff RUN` shows what a run changed; `argus restore RUN --to baseline --yes`
rolls it back (saving the current state first).

## AGENTS.md, skills, compaction

- `AGENTS.md` from `~/.config/argus/` and the workspace root is appended to the system prompt.
- Skills (`*/SKILL.md` with `name`/`description` front matter) in `.agents/skills`,
  `.claude/skills` and `~/.config/argus/skills`: only the index goes in the prompt,
  bodies load through the `skill` tool, and every load is logged.
- Compaction starts at `compaction.threshold` of the context window: old tool
  output is masked first, then the middle of the history is summarised by the
  small model at `compaction.base_url` (dropped if that server is down).

## Suites and A/B

```sh
argus suite run examples/suite/suite.toml -c examples/argus.toml -n 3 --oracle
argus ab examples/argus.toml examples/grammar.toml --suite examples/suite/suite.toml -n 3
argus ab base.toml base.toml --suite s.toml --oa agent.json_thought=false --ob agent.json_thought=true
```

Each task runs in a fresh copy of its fixture (locally or on the SSH host),
then its `check` command decides pass/fail. Variants are interleaved in a
seeded random order per task. The report gives pass rates with Wilson
intervals, turns, tokens, context, wall time, overhead, failure tags, a
per-task table, and an exact McNemar test on paired runs.

## Tests and the mock server

```sh
pytest            # no model, API key or network needed
ruff check . && ruff format --check .
```

`argus.mock.MockServer` implements the llama-server endpoints argus uses, the
discovery endpoints of the other OpenAI-compatible servers, and Anthropic's
Messages, OpenAI's Responses and Gemini's `generateContent` APIs. It replays
scripted steps: tool calls, malformed calls, loops, runaway reasoning/content
(honouring `max_tokens` and `n_ctx`), HTTP errors, request expectations, and
Python callables for dynamic "policies". Steps are rendered for whichever
protocol and API the request uses, constrained output is checked against the
request's schema or grammar, requests are validated the way the APIs do
(tool-result pairing, role alternation), and replayed thinking signatures,
encrypted reasoning and thought signatures are checked against the
conversation they came from. SSH is tested through a fake `ssh` shim and,
where `sshd` is installed, a throwaway real sshd; the sandbox against real
bubblewrap and Landlock; LSP against a fake server and, when installed, pylsp;
the TUI with Textual's Pilot; the installer with stand-ins for uv and pipx
(`ARGUS_TEST_INSTALL=1` installs for real).

Each cloud API also has recorded fixtures in
[tests/fixtures/providers/](tests/fixtures/providers/): a canonical request,
the exact wire request argus must send, the streamed or JSON response, and
the parsed result. **These were written from each provider's documented
format, not captured from the live APIs** (no keys were available). To
replace them with real captures, run with `--record DIR` (or
`ARGUS_RECORD=DIR`) and point `ARGUS_FIXTURES` at the directory; API keys
are never written to a recording.

Environment variables: `ARGUS_DB` (log database), `ARGUS_CONFIG_DIR`
(instead of `~/.config/argus`), `ARGUS_MODELS` (profiles file),
`ARGUS_RECORD`, `ARGUS_FIXTURES`, `ARGUS_LOCAL_SERVERS` (`flavor=url,...`
to probe instead of the usual ports; empty for none), `ARGUS_LSP_AUTODETECT=0`
(no language servers from `PATH`).

## Verify against the real servers and APIs

The mock and the fixtures encode my reading of each API. Check these first:

- whether `response_format` / `grammar` constrain Qwen3's `<think>` block
  (`json_schema` disables thinking by default; `grammar` allows a `<think>` prefix);
- the shape of streamed `tool_calls` deltas and of `usage`/`timings` in llama-server's final chunk;
- `/apply-template` accepting `tools` (used for exact overhead numbers);
- the cloud fixtures, by recording real exchanges with `--record`;
- Gemini's rules for thought signatures on function calls made by another model;
- bubblewrap on distributions that restrict unprivileged user namespaces
  (argus falls back to Landlock; `argus doctor` says which it uses).
