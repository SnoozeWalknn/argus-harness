#!/bin/sh
# argus installer.
#
#   curl -fsSL https://raw.githubusercontent.com/SnoozeWalknn/argus-harness/main/install.sh | sh
#
# Installs argus (with the terminal UI) as a uv tool, writes a default config
# if there is none, and runs `argus doctor`. No sudo; run it again to upgrade.
#
# Environment:
#   ARGUS_SOURCE   what to install (default: git+https://github.com/SnoozeWalknn/argus-harness@main;
#                  a local checkout path works too)
#   ARGUS_NO_UV    set to 1 to use pipx instead of installing uv
#   NO_COLOR       plain output
set -eu

REPO="https://github.com/SnoozeWalknn/argus-harness"
SOURCE="${ARGUS_SOURCE:-git+$REPO@${ARGUS_REF:-main}}"

if [ -t 1 ] && [ -z "${NO_COLOR:-}" ]; then
  B="$(printf '\033[1m')"; D="$(printf '\033[2m')"; R="$(printf '\033[0m')"
else
  B=""; D=""; R=""
fi
say() { printf '%s==>%s %s\n' "$B" "$R" "$*"; }
note() { printf '    %s%s%s\n' "$D" "$*" "$R"; }
die() { printf 'argus install: %s\n' "$*" >&2; exit 1; }

# -- Python ≥ 3.11 ---------------------------------------------------------------------------
PY=""
for candidate in python3.13 python3.12 python3.11 python3; do
  if command -v "$candidate" >/dev/null 2>&1 &&
     "$candidate" -c 'import sys; raise SystemExit(sys.version_info < (3, 11))' 2>/dev/null; then
    PY="$candidate"; break
  fi
done

# -- installer: uv (fetched if missing) or pipx ---------------------------------------------------
case "$SOURCE" in
  git+*|http*) SPEC="argus-harness[tui] @ $SOURCE" ;;
  *) [ -e "$SOURCE" ] || die "ARGUS_SOURCE=$SOURCE does not exist"
     SPEC="argus-harness[tui] @ file://$(cd "$SOURCE" && pwd)" ;;
esac

if [ "${ARGUS_NO_UV:-}" != "1" ] && ! command -v uv >/dev/null 2>&1; then
  say "installing uv (a fast Python package manager) into ~/.local/bin"
  command -v curl >/dev/null 2>&1 || die "curl is needed to fetch uv (or set ARGUS_NO_UV=1 and install pipx)"
  curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null
  PATH="$HOME/.local/bin:$PATH"; export PATH
fi

if [ "${ARGUS_NO_UV:-}" != "1" ] && command -v uv >/dev/null 2>&1; then
  say "installing argus with uv"
  note "$SPEC"
  if [ -n "$PY" ]; then
    uv tool install --force --python "$PY" "$SPEC"
  else
    uv tool install --force --python 3.12 "$SPEC"   # uv fetches a Python if needed
  fi
  BIN="$(uv tool dir --bin 2>/dev/null || echo "$HOME/.local/bin")"
elif command -v pipx >/dev/null 2>&1; then
  [ -n "$PY" ] || die "Python 3.11 or newer is needed"
  say "installing argus with pipx"
  pipx install --force --python "$PY" "$SPEC"
  BIN="${PIPX_BIN_DIR:-$HOME/.local/bin}"
else
  die "neither uv nor pipx is available"
fi

ARGUS="$BIN/argus"
[ -x "$ARGUS" ] || die "argus was not installed where expected ($ARGUS)"

# -- configuration --------------------------------------------------------------------------------
say "setting up ~/.config/argus"
"$ARGUS" init | while IFS= read -r line; do note "$line"; done

# -- what works here --------------------------------------------------------------------------------
say "checking this machine"
"$ARGUS" doctor

case ":$PATH:" in
  *":$BIN:"*) ;;
  *) printf '\n%s is not on your PATH; add it, e.g.:\n    export PATH="%s:$PATH"\n' "$BIN" "$BIN" ;;
esac
printf '\n%sdone.%s Run %sargus%s in a project directory, or %sargus run "your task"%s.\n' "$B" "$R" "$B" "$R" "$B" "$R"
