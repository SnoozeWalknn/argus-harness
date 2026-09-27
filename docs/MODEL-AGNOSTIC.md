# argus, model-agnostic: architecture proposal

This extends argus from a llama-server harness into a model-agnostic coding
agent that borrows the best ideas from Claude Code, Codex, OpenCode and
Omarchy. The headless core stays as it is: the tool loop, the SQLite log, the
failure tags, suites and A/B. Everything new either plugs in behind a seam
that already exists (the LLM client, the executor, the tool registry, the
`Reporter` callbacks) or sits on top of the core (sessions, the TUI).

The original design is in [ARCHITECTURE.md](ARCHITECTURE.md).

## Principles

1. **Headless core first.** The CLI and the TUI are both clients of the same
   `Agent` / `Session` API and the same `Reporter` callbacks. Nothing the TUI
   can do is out of reach of a script.
2. **One conversation format.** argus already stores OpenAI chat-completions
   messages. That stays the canonical format, extended by one opaque field for
   provider state that must be replayed verbatim. Adapters translate at the
   wire boundary and nowhere else.
3. **Everything stays measured.** Every provider reports tokens (including
   cache reads and writes), timings and cost into the same `turns` columns.
   Failure tags mean the same thing on every provider (see the mapping below),
   so A/B across providers is meaningful.
4. **Opinionated defaults, all overridable.** Zero new runtime dependencies
   for the core (still only `httpx`). Textual is an optional extra
   (`argus-harness[tui]`); LSP servers and bubblewrap are external programs
   that are used when present.
5. **Everything testable offline.** The mock server speaks every provider's
   wire format, and fixtures in each provider's format pin the adapters.

## 1. Providers

### Interface

The current `LLMClient` becomes one implementation of a small interface:

```python
class Provider:
    kind: str                                  # openai_compat | anthropic | openai | gemini
    def chat(body, *, stream=True, monitors=()) -> Completion
    def count_prompt(body) -> int | None       # exact request token count, where the API has one
    def tokenize(text) -> list[int]            # llama-server only (NotSupported elsewhere)
    def apply_template(body) -> str            # llama-server only
    def detect() -> Detected                   # context window, output cap, capabilities, pricing
    def health() -> bool
    def close() -> None
```

`body` is the request argus builds today (`messages`, `tools`, `max_tokens`,
sampling, protocol fields, `extra_body`) plus two neutral knobs, `thinking`
(`{"effort": ..., "budget": ...}` or off) and `cache` (bool). Each adapter maps
what it supports. What it cannot support is rejected when the agent is built
(for example `grammar` on Anthropic), never dropped silently at runtime.

Stream monitors keep working unchanged: every adapter emits the same `Delta`
events (reasoning / content / tool), so the reasoning budget and repetition
monitors can abort a Claude or Gemini stream exactly as they abort
llama-server.

`Completion` gains `cache_read_tokens`, `cache_write_tokens`, `stop_raw` (the
provider's own stop reason), `refusal` and `replay` (provider state to attach
to the assistant message). `finish_reason` is normalised to
`stop | tool_calls | length | refusal`.

The adapters speak HTTP directly through `httpx` rather than through the
vendor SDKs: one dependency, one streaming and abort path for every provider,
and wire-level recording and fixtures.

### Canonical messages and replay state

Messages keep the OpenAI shape (`system`, `user`, `assistant` with
`tool_calls`, `tool`). Assistant messages may carry
`replay = {"provider", "model", "items"}`: the parts of a response that must be
sent back unchanged on later turns.

| provider | replay items | rule |
|---|---|---|
| Anthropic | `thinking` / `redacted_thinking` blocks with signatures, in their original order among text and `tool_use` blocks | replayed to Anthropic only; not replayed for messages older than the last history edit (compaction, model switch) |
| OpenAI (Responses API) | `reasoning` items with `encrypted_content` | replayed to the same model only |
| Gemini | `thoughtSignature` of each part | replayed to Gemini; function calls without one (history from another model) get the documented placeholder on models that require signatures |

Replay state is stored in a new `messages.replay_json` column, so a resumed
session replays exactly what the provider sent.

Why the "last history edit" rule: argus compaction rewrites history (masking
old tool output, summarising the middle). Anthropic binds thinking blocks to
the conversation prefix that produced them, and newer models reject blocks
whose prefix changed. So after a compaction or switch, older blocks are
dropped; if the API still rejects a block, the adapter strips all thinking
blocks and retries once (logged as an event).

### Adapters

| adapter | wire format | notes |
|---|---|---|
| `openai_compat` | `POST /v1/chat/completions`, SSE | Flavours, detected by probing: **llama-server** (`/props`, `/tokenize`, `/apply-template`, GBNF, `timings`), **Ollama** (`/api/show` for context length and capabilities), **vLLM** (`max_model_len` in `/v1/models`, GBNF via `guided_grammar`), **LM Studio** (`/api/v0/models`), **OpenRouter** (`/api/v1/models` for context length and pricing; `reasoning` field), generic. Reasoning is read from `reasoning_content` or `reasoning`. |
| `anthropic` | `POST /v1/messages`, SSE (`message_start`, `content_block_start/delta/stop`, `message_delta`, `message_stop`, `ping`, `error`) | System prompt split out; tool results become `tool_result` blocks, all results of a turn in one user message; adaptive thinking with `output_config.effort` and `display: "summarized"` (so reasoning can be logged), `budget_tokens` for older models; sampling parameters dropped for models that reject them; `cache_control` breakpoints on the system prompt and the newest message; usage with cache reads/writes; `429` honours `retry-after`, `529 overloaded` retried; "prompt is too long" becomes `ContextOverflow`; `model_context_window_exceeded` becomes `length`; `refusal` surfaced; `/v1/messages/count_tokens` for overhead; `/v1/models/{id}` for the context window. |
| `openai` | `POST /v1/responses`, SSE (`response.output_text.delta`, `response.function_call_arguments.delta`, `response.reasoning_summary_text.delta`, `response.output_item.done`, `response.completed`, …) | Stateless (`store: false`) with `include: ["reasoning.encrypted_content"]` so reasoning is carried across tool turns; `function_call` / `function_call_output` items; `reasoning.effort`; `max_output_tokens`; usage with cached and reasoning tokens; `incomplete` with `max_output_tokens` becomes `length`; `context_length_exceeded` becomes `ContextOverflow`. (Chat Completions against OpenAI still works through `openai_compat` with the `max_completion_tokens` quirk.) |
| `gemini` | `POST /v1beta/models/{m}:streamGenerateContent?alt=sse` (and `:generateContent`) | `contents` with `user`/`model` roles, `systemInstruction`, `functionDeclarations` (tool schemas reduced to the supported subset), `functionCall` / `functionResponse` parts, `thinkingConfig` (`thinkingBudget` or `thinkingLevel`, `includeThoughts`), thought parts become reasoning, `thoughtSignature` replay; `usageMetadata` (prompt, candidates, thoughts, cached); `MAX_TOKENS` becomes `length`, `MALFORMED_FUNCTION_CALL` a `malformed_call`, `SAFETY` a refusal; `countTokens`; `models.get` for `inputTokenLimit`. |

API keys come only from environment variables: `ANTHROPIC_API_KEY`,
`OPENAI_API_KEY`, `GEMINI_API_KEY` (or `GOOGLE_API_KEY`),
`OPENROUTER_API_KEY`; `model.api_key_env` names another variable. Keys are
never written to the database (the stored config is redacted), to fixtures or
to logs. Cloud adapters use the environment's proxy settings and TLS
verification; loopback servers bypass proxies as they do today.

### Failure tags across providers

| tag | local (unchanged) | cloud adapters |
|---|---|---|
| `malformed_call` | unparseable / schema-invalid arguments, unparsed markup | the same harness-side validation, plus Gemini `MALFORMED_FUNCTION_CALL` and tool input that fails to parse after a truncated stream |
| `token_cap` | `finish_reason=length`, reasoning budget, overflow | `max_tokens` / `max_output_tokens` / `MAX_TOKENS`, `model_context_window_exceeded`, prompt-too-long errors |
| `loop`, `overrun` | detectors on calls and text | unchanged (they only look at parsed calls and text) |
| `server_error` | HTTP errors | plus `refusal` (with the provider's category) and exhausted overload retries |

### Protocols per provider

| | native | json_schema | grammar |
|---|---|---|---|
| llama-server | ✓ | ✓ | ✓ |
| vLLM | ✓ | ✓ | ✓ (GBNF via `guided_grammar`) |
| Ollama, LM Studio, OpenRouter | ✓ | ✓ | – |
| Anthropic, OpenAI, Gemini | ✓ | – | – |

This matches the intended split: native tool calling for frontier models,
grammar or JSON schema for local ones. The matrix is checked when the agent is
built.

### Mock server, fixtures and recording

- `argus.mock` learns every wire format: `/v1/messages`, `/v1/responses`,
  `/v1beta/models/*:generateContent|streamGenerateContent`, and the flavour
  endpoints (`/api/show`, `/api/v0/models`, `/api/v1/models`). One script step
  (reasoning, text, tool calls, malformed arguments, runaway output, errors)
  renders into whichever format the request arrived in, so every agent-level
  scenario can run against every provider.
- `tests/fixtures/providers/<provider>/<case>.json` holds a request and the
  exact response in that provider's format (a JSON body or the list of SSE
  events). Fixture tests serve the response from the mock, check the parsed
  `Completion`, and check the adapter's outgoing request against the
  fixture's request.
- **Where the fixtures come from.** This sandbox has no API keys and cannot
  reach the cloud APIs, so the fixtures are written from each provider's
  documented wire format, not captured. `--record DIR` (or `ARGUS_RECORD`)
  captures real exchanges in the same fixture format with keys and auth
  headers removed, and the fixture tests pick up any case dropped into the
  directory, so real recordings can replace or extend them.

## 2. Model profiles

A profile is what argus needs to know about a model beyond its id:

```toml
[profiles."qwen3-coder"]
match = ["qwen3-coder*", "*Qwen3-Coder*"]  # model ids this profile applies to
provider = "llama_server"
base_url = "http://127.0.0.1:8080/v1"
model = "qwen3-coder"
context_window = 65536
max_output = 8192
protocol = "grammar"            # native | json_schema | grammar
thinking = "qwen"               # none | qwen | anthropic_adaptive | anthropic_budget |
                                # openai_effort | gemini_budget | gemini_level | reasoning_field
effort = "high"
sampling = { temperature = 0.7, top_p = 0.8, top_k = 20 }
chat_template_kwargs = {}
quirks = []                     # named, tested request/response adjustments
pricing = { input = 0, output = 0, cache_read = 0, cache_write = 0 }  # USD per 1M tokens

[profiles."qwen3-coder".tuned]  # written by `argus tune`
```

Quirks are small named adjustments, each with a test: `no_sampling`,
`max_completion_tokens`, `no_system_role`, `merge_same_role`,
`no_parallel_tools`, `strip_reasoning_history`, `thought_signatures`, and
local chat-template settings (`chat_template_kwargs`, stop strings).

**Resolution**, lowest precedence first:

1. the built-in catalog (`argus/data/models.toml`): Claude, GPT, Gemini, Qwen,
   Llama, DeepSeek, gpt-oss and others, with protocol, thinking format, quirks
   and dated pricing (a convenience; override it);
2. auto-detection (cached per server and model);
3. user profiles in `~/.config/argus/models.toml`, which `argus tune` writes;
4. `[model]` / `[agent]` keys that a config file actually sets;
5. `-m` and `-o` on the command line.

Only keys that were explicitly set override a profile: the loader records which
keys each TOML file and override set, so dataclass defaults never mask a
profile.

**Model specs**: `-m anthropic/claude-opus-5`, `-m openai/gpt-5`,
`-m gemini/gemini-2.5-pro`, `-m ollama/qwen3-coder:30b`,
`-m openrouter/qwen/qwen3-coder`, `-m llama` (the local llama-server), or a
profile name.

**Auto-detection**

| source | yields |
|---|---|
| llama-server `/props` | `n_ctx`, model alias, chat template (tool support, thinking) |
| Ollama `/api/show` | context length, `num_ctx`, capabilities (tools, thinking) |
| vLLM `/v1/models` | `max_model_len` |
| LM Studio `/api/v0/models` | max and loaded context length |
| OpenRouter `/api/v1/models` | context length, pricing, supported parameters |
| Anthropic `/v1/models/{id}` | `max_input_tokens`, `max_tokens` |
| Gemini `models/{id}` | `inputTokenLimit`, `outputTokenLimit`, thinking |
| OpenAI | nothing useful; catalog only |

**Cost**: each turn's usage times the profile's pricing (cache reads and writes
priced separately) goes to `turns.cost_usd` and `runs.cost_usd`; reports gain
a cost row.

### `argus tune MODEL`

Runs a suite (a small built-in tuning suite, or `--suite`) once per protocol the
provider supports (optionally × thinking on/off), `-n` repetitions each, as one
batch. Variants are ranked by the Wilson lower bound of the pass rate, then
failure count, then tokens and wall time. The winner is written to the model's
user profile with a `[tuned]` record (batch id, date, per-variant results), and
the usual report is printed. `--dry-run` reports without saving.

## 3. From Claude Code

- **Subagents with isolated context.** A `task` tool starts a child run with a
  fresh context: its own system prompt, a tool subset and optionally another
  model. Only its final answer (plus a one-line stats summary) returns to the
  parent. Built-in agents: `explore` (read-only: read, glob, grep, bash in the
  read-only sandbox) and `general`. User agents come from `.argus/agents/*.md`,
  `.claude/agents/*.md` and `~/.config/argus/agents/*.md` (front matter:
  `name`, `description`, `tools`, `model`). Child runs are ordinary runs in
  SQLite (`parent_run_id`, `parent_turn`), so failure tags and reports cover
  them. Depth is limited to one.
- **Hooks.** `[[hooks]]` with `event` (`session_start`, `user_prompt`,
  `pre_tool`, `post_tool`, `stop`), a `matcher` regex on the tool name, a
  `command` and a `timeout`. The event is JSON on stdin. Exit 0 continues
  (stdout of `user_prompt` / `post_tool` hooks is added to the context); exit 2
  blocks: `pre_tool` turns the call into an error result carrying stderr, and
  `stop` forces another turn with stderr as the message. Other exit codes are
  logged and ignored. Every hook run is logged.
- **Plan mode** (`--plan`; Tab in the TUI). The run uses the read-only policy
  (no `edit`; `bash` only inside the read-only sandbox, removed if there is
  none), the prompt asks for a plan, and the final answer is the plan
  (`runs.mode = 'plan'`). Approving it continues the same session in build
  mode.
- **Todo list.** `todo(items=[{content, status}])` replaces the list and
  echoes it compactly. Stored in a `todos` table, shown in the TUI sidebar and
  in `argus show`. On by default for models with at least 64k context, off
  for small local models where every tool costs prompt tokens.
- **Progressive-disclosure skills.** Level 1 (name and description in the
  prompt) and level 2 (`skill(name)` loads SKILL.md) exist; level 2 now also
  lists the skill's other files, and level 3 `skill(name, file)` loads one of
  them. Each level is logged.

## 4. From Codex: approval policies and sandboxing

`agent.approval` chooses a policy; the sandbox enforces it for commands.

| policy | `edit` | `bash` | escalation |
|---|---|---|---|
| `read-only` | denied | read-only sandbox, no network | none |
| `ask` | asks every time | workspace-write sandbox; asks every time except known-safe read-only commands | – |
| `auto` (default) | allowed inside the workspace, asks outside it | workspace-write sandbox | a failure that looks like a sandbox denial asks to re-run unsandboxed (headless: reported to the model) |
| `full` | allowed | no sandbox | – |

Asking goes to an `Approver`: the TUI's modal, a TTY prompt (yes / no / always
/ never), or the headless approver (`agent.headless_approval = "deny"` by
default, with the reason passed to the model). "Always" decisions persist for
the session (per tool, or per command prefix). Every decision is logged in an
`approvals` table.

Sandbox backends (`sandbox.backend = auto | bwrap | landlock | none`):

- **bubblewrap**: `--ro-bind / /`, the workspace bound read-write (for
  workspace-write), `--tmpfs /tmp`, `--dev /dev`, `--proc /proc`,
  `--unshare-net` when network is off, `--die-with-parent`, `--new-session`.
- **Landlock**: applied in the child just before `exec` (raw syscalls via
  `ctypes`, no helper binary): read and execute everywhere, write only to the
  workspace, `/tmp` and `/dev`; with Landlock ABI ≥ 4, TCP connect and bind are
  denied when network is off.
- `auto` uses bubblewrap if it can actually create a sandbox (unprivileged user
  namespaces are restricted on some distributions), else Landlock, else none
  with a warning (or an error with `sandbox.required = true`).

The sandbox wraps `bash` on the local executor. The file tools enforce the same
policy in-process with path checks. Over SSH the remote machine is the
boundary; the sandbox is off there.

## 5. From OpenCode

- **Sessions and resume.** A session is a sequence of runs sharing one
  conversation. `argus run --continue` (latest session), `--session ID` and
  `argus resume` continue a session: the next run starts from the previous
  run's final context plus the new user message. Each run remains a complete,
  separately analysable row; a `sessions` table ties them together.
- **Switching providers mid-session.** The next run of a session may use
  another model, provider or protocol (`-m`, `/model` in the TUI). When the
  protocol family changes, history is converted (constrained JSON actions ⇄
  native tool calls, using the logged tool calls), the system prompt is
  rebuilt, and replay state is filtered by provider. Logged as a `switch`
  event.
- **LSP diagnostics after edits.** A minimal JSON-RPC client over stdio
  (`argus/lsp.py`) starts language servers lazily per language (pyright or
  basedpyright, pylsp, typescript-language-server, gopls, rust-analyzer,
  clangd; whichever is installed, configurable), sends `didOpen`/`didChange`
  after each successful edit, waits up to `lsp.wait_ms` for
  `publishDiagnostics`, appends errors for the edited file to the edit result,
  and logs them. Over SSH the server is started through the same ssh command.
  Tests use a small fake language server; a real `pylsp` test runs when it is
  installed.

## 6. From Omarchy

- **Defaults that work.** `argus` with no arguments opens the TUI in the
  current directory. Without a configured model, argus probes local servers
  (llama-server :8080, Ollama :11434, LM Studio :1234, vLLM :8000), then
  uses the first cloud key present (Anthropic, OpenAI, Gemini, OpenRouter).
  Global config `~/.config/argus/config.toml`, project config
  `.argus/config.toml`. Sandbox auto, approval auto, compaction and
  checkpoints on.
- **One-command installer.** `install.sh`:
  `curl -fsSL https://raw.githubusercontent.com/SnoozeWalknn/argus-harness/main/install.sh | sh`.
  Installs `uv` if missing, installs argus with the TUI extra as a uv tool,
  writes the default config if there is none, prints distro-specific hints for
  bubblewrap and language servers, and runs `argus doctor`. Idempotent, no
  sudo. Tested from a local checkout in a temporary `HOME`.
- **`argus doctor`**: Python, sandbox backends, language servers, local model
  servers, which API key variables are set (names only), database path.
- **TUI** (Textual). A thin client: the agent runs in a worker thread,
  `Reporter` callbacks are posted to the UI as messages, and approvals are a
  modal that the worker waits on. Header (model, provider, mode, policy,
  context %, tokens, cost), transcript (reasoning collapsed, tool calls and
  results, diffs), sidebar (todos, changed files, subagents), prompt. Keys:
  enter send, tab plan/build, ctrl+o model, ctrl+t theme, ctrl+s sessions,
  ctrl+n new session, esc interrupt, y/n/a in approvals, ctrl+p command
  palette, ctrl+q quit. Themes: Textual's built-ins plus Omarchy-style ones
  (tokyo-night, catppuccin, gruvbox, nord, everforest, kanagawa, rose-pine,
  matte-black), following `~/.config/omarchy/current/theme` when it exists.
  Tested with Textual's `Pilot` against the mock server.

## Storage: schema v2, migrated in place

- `runs` + `session_id`, `parent_run_id`, `parent_turn`, `provider`, `mode`,
  `cost_usd`, `cache_read_tokens`, `cache_write_tokens`
- `turns` + `cache_write_tokens`, `cost_usd`, `provider`, `model`
- `messages` + `replay_json`
- new tables: `sessions`, `approvals`, `todos`, `diagnostics`

Hooks, switches, sandbox decisions and retries go to `events`.

## Config additions

```
[model]    provider, profile, api_key_env, effort, thinking_budget, record_dir
[agent]    mode (build|plan), approval (read-only|ask|auto|full), headless_approval,
           todo, subagents
[sandbox]  backend, network, writable, required
[lsp]      enabled, servers, wait_ms, warnings
[[hooks]]  event, matcher, command, timeout
[tui]      theme, show_reasoning
```

## CLI additions

```
argus                                   TUI if installed, else help
argus run TASK -m SPEC [--plan] [--approval P] [--continue | --session S] [--record DIR]
argus sessions | resume [SESSION]
argus models [--detect]                 catalog, user profiles, detected servers
argus tune MODEL [--suite S] [-n N] [--thinking] [--dry-run]
argus doctor | init | tui
```

## Module layout (new files)

```
argus/providers/   __init__.py (make_provider, model specs), base.py, sse.py, record.py,
                   openai_compat.py, anthropic.py, openai.py, gemini.py
argus/profiles.py  catalog, resolution, detection cache, pricing
argus/data/        models.toml, tune/ (built-in tuning suite)
argus/tune.py
argus/sandbox.py   bwrap and Landlock backends
argus/approval.py  policies, approvers, known-safe commands
argus/hooks.py
argus/subagents.py task tool, agent definitions
argus/tools/todo.py
argus/session.py   sessions, continuation, history conversion
argus/lsp.py
argus/tui/         app, widgets, themes, stylesheet
argus/mock/        + anthropic, openai_responses, gemini renderers, fixture replay
install.sh
```

## Milestones (one commit each)

| # | milestone | done when |
|---|---|---|
| 9 | Provider layer, OpenAI-compatible flavours, recorder, fixture replay, schema v2 | all existing tests pass through the provider layer; flavour fixtures parse; record → replay round-trip |
| 10 | Anthropic adapter | fixture tests; a full scripted task through the mock's `/v1/messages`, including thinking replay, cache usage, overflow and overload retry |
| 11 | OpenAI (Responses) and Gemini adapters | the same for both |
| 12 | Profiles, auto-detection, cost, `argus models`, `argus tune` | `tune` picks the protocol the mock makes succeed and writes it to the profile |
| 13 | Sandbox and approval policies | writes outside the workspace blocked under bubblewrap and under Landlock; every policy tested; approvals logged |
| 14 | Todo, plan mode, subagents, hooks, progressive skills | a scripted scenario for each |
| 15 | Sessions, resume, mid-session switching, LSP diagnostics | resume across processes; llama-server → Anthropic switch mid-session; diagnostics appended after a bad edit |
| 16 | Defaults, installer, doctor, TUI and themes | Pilot tests drive a full session; the installer works in a temporary `HOME` |

Providers come first because profiles, tuning, subagents on other models and
switching all need them. The sandbox comes before plan mode, which is the
read-only policy. Sessions come before the TUI, which is a session client. The
TUI is last because it is the thinnest layer.

## Things to verify against the real APIs

- The fixtures follow the providers' documentation; replace them with
  `--record` captures when keys are available.
- Gemini's rules for thought signatures on function calls that another model
  made, and the placeholder value, are the part most likely to change.
- bubblewrap needs unprivileged user namespaces, which some distributions
  restrict (Landlock is the fallback there).
- Model catalogs and prices change quickly; the catalog is data, and detection
  plus `argus tune` exist so that it does not have to be perfect.
