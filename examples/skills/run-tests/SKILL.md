---
name: run-tests
description: How to run and read this project's tests (pytest), including running a single test.
---

# Running tests

- All tests: `python3 -m pytest -q`
- One test: `python3 -m pytest -q path/to/test_file.py::test_name`
- Stop at the first failure and show locals: `python3 -m pytest -x -l`

Read the assertion message and the last frame of the traceback first; fix the
code, not the test, unless the task says the test is wrong.
