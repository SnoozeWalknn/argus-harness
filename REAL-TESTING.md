# Real testing checklist

Everything in argus so far has been tested against its own mock server. This
checklist runs it against your real models: Qwen and Bonsai on llama-server at
`127.0.0.1:8080`, plus one cloud model. New features are frozen until this
passes; what it finds gets fixed first.

**How to use it.** Work top to bottom. Every step saves its output under
`~/argus-logs/`. Each step says what a pass looks like and what to paste back
if it fails. You can also paste the files of steps that passed; the numbers
help (real token counts, tok/s, which protocol your models handle best).
argus never writes API keys into its logs, but look over anything before you
paste it.

```sh
mkdir -p ~/argus-logs
```

---

## 0 · Before you start

- Python 3.11 or newer, git, curl.
- llama-server running on `127.0.0.1:8080` with **`--jinja`** (the `native`
  protocol needs the model's chat template to handle tools). For example:

  ```sh
  llama-server -m /path/to/Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf --jinja -c 32768 --host 127.0.0.1 --port 8080
  ```

  If one llama-server serves both models (router mode, or llama-swap in front
  of it), argus picks by model name. If it serves one model at a time, restart
  it with the other model where a step says **Bonsai**.
- For step 7, one API key: `ANTHROPIC_API_KEY` (used below), or
  `OPENAI_API_KEY` or `GEMINI_API_KEY` (swap `-m sonnet` for `-m gpt` or `-m flash`).
  The step costs a few cents.

---

## 1 · Install from this branch

```sh
git clone -b claude/headless-coding-agent-harness-xd4bec https://github.com/SnoozeWalknn/argus-harness.git ~/src/argus-harness
cd ~/src/argus-harness
git log --oneline -1 | tee ~/argus-logs/01-commit.txt
ARGUS_SOURCE=. sh install.sh 2>&1 | tee ~/argus-logs/01-install.log
argus --version
```

The installer installs `uv` if it's missing, installs argus (with the TUI) as a
uv tool from this checkout, writes `~/.config/argus/config.toml` and runs
`argus doctor`. No sudo.

**Pass**
- the log ends with `done. Run argus in a project directory, or argus run "your task".`
- `argus --version` prints `argus 0.2.0`

If it prints `… is not on your PATH`, run the `export PATH=…` line it shows (and
add it to your shell profile), then `argus --version` again. That isn't a failure.

**If it fails, paste**
- `~/argus-logs/01-install.log`
- `~/argus-logs/01-commit.txt`
- the output of `python3 --version; uv --version; head -3 /etc/os-release`

---

## 2 · Point argus at your two models

Find the id llama-server reports for each model (with one model loaded at a
time, do this once per model):

```sh
curl -s http://127.0.0.1:8080/v1/models | tee -a ~/argus-logs/02-models.json
```

Write `~/.config/argus/models.toml`, replacing `QWEN_ID` and `BONSAI_ID` with
the `"id"` values from that output:

```toml
[aliases]
qwen = "llama_server/QWEN_ID@http://127.0.0.1:8080/v1"
bonsai = "llama_server/BONSAI_ID@http://127.0.0.1:8080/v1"

# Bonsai has no built-in profile; this marks it as a local model.
[profiles.bonsai]
match = ["*Bonsai*", "*bonsai*"]
tier = "local"
```

Then make Qwen the default: in `~/.config/argus/config.toml`, edit the
existing `[model]` section so it reads as follows. Don't add a second
`[model]`: TOML rejects duplicate tables.

```toml
[model]
provider = "llama_server"
base_url = "http://127.0.0.1:8080/v1"
model = "QWEN_ID"
```

Check what argus makes of each:

```sh
argus models qwen --detect 2>&1 | tee ~/argus-logs/02-qwen.txt
argus models bonsai --detect 2>&1 | tee ~/argus-logs/02-bonsai.txt    # Bonsai loaded
```

**Pass**
- `02-qwen.txt`: `provider:  llama_server`, `profile:   qwen3-coder` (or `qwen3`),
  `protocol:  grammar`, and under `detected:` a context window equal to your
  `-c` value, with `protocols: native, json_schema, grammar`
- `02-bonsai.txt`: `model:     BONSAI_ID`, `profile:   bonsai` with
  `model.tier = "local"`, and the same `detected:` block

**If it fails, paste**
- `02-models.json`, `02-qwen.txt`, `02-bonsai.txt`
- both TOML files (they contain no secrets)

---

## 3 · argus doctor

```sh
argus doctor 2>&1 | tee ~/argus-logs/03-doctor.txt
```

**Pass**
- `model server:   ✓ llama_server http://127.0.0.1:8080/v1 ctx <your -c> <model file>`
- `sandbox:        ✓ bwrap` or `✓ landlock` (commands run sandboxed)
- `tui:            ✓ textual …`

`✗` on `rg`, the `lsp …` lines or `api keys: none set` is fine. doctor prints
what to install under `suggestions:`. A Python language server (pyright or
pylsp) makes step 5 more interesting: edits get diagnostics.

**If it fails, paste**
- `03-doctor.txt` and `argus doctor --json`
- `curl -s http://127.0.0.1:8080/props | head -c 2000`
- the exact llama-server command line you started

---

## 4 · Make a trial workspace, then argus overhead --all

A tiny repository with one bug, used by the next steps:

```sh
mkdir -p ~/argus-trial && cd ~/argus-trial && git init -q
cat > calc.py <<'EOF'
def add(a, b):
    return a - b


def mean(xs):
    return sum(xs) / len(xs)
EOF
cat > test_calc.py <<'EOF'
from calc import add, mean


def test_add():
    assert add(2, 3) == 5


def test_mean():
    assert mean([1, 2, 3]) == 2


if __name__ == "__main__":
    test_add()
    test_mean()
    print("all tests pass")
EOF
git add -A && git commit -qm "trial workspace"
```

Measure what the harness adds to every prompt, per protocol:

```sh
cd ~/argus-trial
argus overhead --all -m qwen 2>&1 | tee ~/argus-logs/04-overhead-qwen.txt
argus overhead --all -m bonsai 2>&1 | tee ~/argus-logs/04-overhead-bonsai.txt   # Bonsai loaded
```

**Pass**
- the `method` row reads `server server server`: exact counts, from
  llama-server's `/apply-template` and `/tokenize`
- `TOTAL overhead` for `grammar` and `json_schema` is a fraction of `native`
  (about a fifth on the mock server; your tokenizer will give other numbers)
- `share of context window` is well under 10% for `native`, and a percent or
  two for the other two (on the mock at 32k context: 6.5%, 1.4%, 1.4%)

**If it fails, paste**
- both files
- `llama-server --version`

`estimate` instead of `server` means llama-server didn't answer
`/apply-template` or `/tokenize` (an older build). An HTTP error means argus
couldn't reach it.

---

## 5 · One task per protocol, with -v

Each run gets the same bug to fix under a different tool-call protocol. `-v`
streams the model's reasoning and replies.

```sh
cd ~/argus-trial
for proto in native json_schema grammar; do
  git checkout -q -- calc.py
  argus run -v -m qwen -o "agent.protocol=\"$proto\"" \
    "test_add fails. Find the bug in calc.py, fix it, then run: python3 test_calc.py" \
    2>&1 | tee ~/argus-logs/05-qwen-$proto.log
  argus show last > ~/argus-logs/05-qwen-$proto-show.txt
  python3 test_calc.py 2>&1 | tee -a ~/argus-logs/05-qwen-$proto.log
done
git checkout -q -- calc.py
```

Then the same for **Bonsai** (restart llama-server with Bonsai first if it
serves one model at a time): run the loop again with `-m bonsai` and `bonsai`
in the file names.

**Pass, for each protocol**
- a line `completed · N turns · N calls · N gen tok · max ctx N · Ns · run <id>`,
  with nothing in `[brackets]` after `completed`, and a handful of turns
  (3–6 is typical); the model's summary is printed after it
- the tool lines show `read(path=calc.py)`, then an `edit(…)` changing
  `return a - b` to `return a + b` marked `✓`, then a `bash(…)` marked `✓`
- the last line of the log is `all tests pass`

Under `json_schema` and `grammar`, `-v` also prints the raw JSON action the
model writes each turn (`{"tool":"read","args":{"path":"calc.py"}}`), and the
last one is `{"tool":"done",…}`. That's expected: under those protocols the
JSON is the model's output.

A small model failing under `native` but passing under `grammar` is a finding,
not a broken checklist. Qwen3-Coder doesn't think out loud, so no reasoning text
in its logs is normal.

**If a protocol fails, paste**
- `05-<model>-<proto>.log` and `05-<model>-<proto>-show.txt` for that run
- `argus failures --tag <tag>` for each tag the run line showed (for example
  `argus failures --tag malformed_call`)
- if llama-server printed an error at that moment, its last 30 lines

---

## 6 · argus show last

`argus show last` prints the last run's full transcript from the log: every
turn's prompt, reasoning, tool calls and results, plus timings and token
counts. Step 5 saved one per run; look at `05-qwen-grammar-show.txt`.

**Pass**
- the header reads `run <id>  completed` and has `protocol=grammar`,
  `provider:  llama_server http://127.0.0.1:8080/v1`, and
  `overhead` matching the grammar total from step 4
- each `── turn N ·` line has a real ttft and tok/s for your hardware (not the
  `1000.0 tok/s` the mock reports) and `finish=stop`
- the transcript ends with `final answer:` and the model's summary

**If it fails, paste**
- the `-show.txt` file
- the matching `.log` from step 5

---

## 7 · Switch to a cloud model mid-session

Qwen looks first, then a cloud model continues the same session. It gets the
history, including the file Qwen already read. The first line fixes `add` by
hand, so the test run at the end fails only if `mean` is wrong.

```sh
cd ~/argus-trial && git checkout -q -- calc.py && sed -i 's/return a - b/return a + b/' calc.py
argus run -v -m qwen "Read calc.py and tell me what mean([]) does. Don't change any files." \
  2>&1 | tee ~/argus-logs/07a-qwen.log

export ANTHROPIC_API_KEY=sk-ant-...        # or OPENAI_API_KEY (-m gpt), GEMINI_API_KEY (-m flash)
argus run -v --continue -m sonnet \
  "Now make mean([]) return 0.0 instead of raising, then run python3 test_calc.py." \
  2>&1 | tee ~/argus-logs/07b-cloud.log

python3 - <<'PY' | tee ~/argus-logs/07c-session.json
import json, os, sqlite3
db = sqlite3.connect(os.path.expanduser("~/.local/share/argus/argus.db"))
row = db.execute("SELECT data_json FROM events WHERE kind = 'session' ORDER BY id DESC LIMIT 1").fetchone()
print(json.dumps(json.loads(row[0]), indent=2))
PY
argus sessions | head -3 | tee -a ~/argus-logs/07c-session.json
git diff | tee -a ~/argus-logs/07b-cloud.log
```

(If `argus doctor` showed a different `log:` path, use it in the Python snippet.)

**Pass**
- `07a` completes with an answer saying `mean([])` divides by zero
- `07b` completes with a `$` cost on its run line, and no
  `read calc.py before editing it` error
- `07c-session.json` has `"history"` above 0, `"converted": "grammar -> native"`,
  and `"switched"` from `llama_server …, QWEN_ID` to
  `anthropic https://api.anthropic.com/v1, claude-sonnet-5`
- the first session under the `argus sessions` header has `2` runs and model
  `claude-sonnet-5`
- `07b`'s `bash(cmd=python3 test_calc.py)` result is `✓ all tests pass`
- `git diff` shows your `add` fix and `mean` returning `0.0` for an empty list

**If it fails, paste**
- `07a-qwen.log`, `07b-cloud.log`, `07c-session.json`
- `argus show last`

`anthropic needs an API key` means the key isn't exported in that shell. A
`400` or `401` from the API: paste the log.

The same switch in the TUI: run `argus` in `~/argus-trial`, ask the first
question, press ctrl+o, pick `sonnet`, ask the second.

---

## 8 · A background job

The model starts a server in the background, checks it, then stops it.

```sh
cd ~/argus-trial
argus run -v -m qwen "Start 'python3 -m http.server 8765 --bind 127.0.0.1' as a background job (bash with background=true). Then fetch http://127.0.0.1:8765/ to check it answers. Then stop the job with the job tool." \
  2>&1 | tee ~/argus-logs/08-job.log
argus show last > ~/argus-logs/08-job-show.txt
grep -E "started job|\[job 1" ~/argus-logs/08-job-show.txt
curl -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1:8765/ | tee -a ~/argus-logs/08-job.log
```

**Pass**
- `08-job.log` shows `bash(cmd=python3 -m http.server 8765 --bind 127.0…, background=true)`
  with `✓ started job 1: python3 -m http.server 8765 --bind 127.0.0.1`
- then `fetch(url=http://127.0.0.1:8765/)` with
  `✓ http://127.0.0.1:8765/  [200 text/html]  Directory listing for /`
- the `grep` finds `[job 1 killed]` (or `[job 1 exited …]`)
- the final `curl` prints `000`: the server is gone (`argus run` also stops any
  job still running when the run ends)
- the run line reads `completed`

If you see `[timed out after 120s]`, the model ran the server in the foreground
instead of with `background=true`. That's the model, not argus: paste the log,
then try `-m sonnet` to tell the two apart.

**If it fails, paste**
- `08-job.log` and `08-job-show.txt`
- `argus doctor | grep sandbox`

---

## 9 · Optional: the TUI

```sh
cd ~/argus-trial && git checkout -q -- calc.py && argus
```

Ask `test_add fails, fix it`, then watch the transcript and the **tokens** box
top right while it works. Try ctrl+o (models), `/help`, ctrl+t (themes) and
ctrl+q (quit).

**Pass**
- reasoning (if the model gives any) and tool calls appear while it works
- the token box shows `↓ N tok · N/s` updating, then `○ idle` with the last turn
- after ctrl+o the list marks your local model and the cloud models with a key
  as ● ready

**If something looks wrong,** paste a screenshot or describe it. Add the
output of `echo $TERM; tput colors` if colours or characters look broken.

---

## Reporting back

For a failure, paste the files named in that step. When everything passes, the
most useful thing to paste is one line per run and the switch record:

```sh
grep -h "· run " ~/argus-logs/*.log
cat ~/argus-logs/07c-session.json ~/argus-logs/04-overhead-qwen.txt
```
