# argus-harness

A personal, headless coding-agent harness for a local Qwen model served by
`llama-server`, with every run logged to SQLite. See
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the design and milestones.

```sh
uv venv && uv pip install -e '.[dev]'
argus run "Fix the failing test in calc.py" -w ~/src/proj      # llama-server on :8080
argus show last                                               # transcript, tokens, timings
argus mock-server --script examples/mock/fix-calc.json        # scripted stand-in for llama-server
pytest                                                        # everything runs against the mock
```
