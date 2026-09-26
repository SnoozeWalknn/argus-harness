#!/usr/bin/env python3
"""Stand-in for the ssh client: drops options and the host, runs the command locally.

Like real ssh it hands the remote command string to ``sh -c``, so the quoting
argus does for the remote side is exercised for real. Host ``unreachable``
simulates a connection failure (exit 255).
"""

import os
import sys

WITH_VALUE = set("oOpilFJSEcmbLRDWw")

args = sys.argv[1:]
i = 0
while i < len(args) and args[i].startswith("-"):
    flag = args[i]
    i += 2 if len(flag) == 2 and flag[1] in WITH_VALUE else 1
if i >= len(args):
    sys.exit(0)
host, command = args[i], " ".join(args[i + 1 :])
log = os.environ.get("FAKE_SSH_LOG")
if log:
    with open(log, "a") as f:
        f.write(f"{host}\t{command}\n")
if host == "unreachable":
    sys.stderr.write("ssh: connect to host unreachable port 22: Connection refused\n")
    sys.exit(255)
if not command:
    sys.exit(0)
os.execvp("sh", ["sh", "-c", command])
