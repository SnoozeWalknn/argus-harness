"""AGENTS.md and SKILL.md loading.

Paths in ``context.skill_dirs`` / ``context.global_dir``:

* relative paths are inside the workspace and read through the executor
  (so they work over SSH);
* paths starting with ``~`` or ``local:`` are on the machine running argus.

Skills use progressive disclosure: only ``name: description`` lines go into
the system prompt; the body is loaded when the model calls the ``skill``
tool (or reads the SKILL.md file). Every load is logged.
"""

from __future__ import annotations

import glob
import os
import posixpath
import shlex
from dataclasses import dataclass
from typing import Any

from argus.config import ContextConfig
from argus.executors import Executor
from argus.tools.base import Tool, ToolContext, ToolError, ToolResult, obj

MAX_SKILL_FILES = 20


@dataclass
class Skill:
    name: str
    description: str
    path: str  # SKILL.md
    local: bool  # True: on the argus host; False: in the workspace (via the executor)

    @property
    def dir(self) -> str:
        return posixpath.dirname(self.path)


def parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    """Minimal YAML front matter: ``key: value``, quoted values and ``>``/``|`` blocks."""
    if not text.startswith("---"):
        return {}, text
    lines = text.split("\n")
    end = next((i for i in range(1, len(lines)) if lines[i].strip() == "---"), None)
    if end is None:
        return {}, text
    meta: dict[str, str] = {}
    i = 1
    while i < end:
        line = lines[i]
        if not line.strip() or line.lstrip().startswith("#") or line[0] in " \t" or ":" not in line:
            i += 1
            continue
        key, value = line.split(":", 1)
        value = value.strip()
        if value in (">", "|", ">-", "|-", ">+", "|+"):
            block = []
            i += 1
            while i < end and (not lines[i].strip() or lines[i][0] in " \t"):
                block.append(lines[i].strip())
                i += 1
            sep = " " if value.startswith(">") else "\n"
            meta[key.strip()] = sep.join(b for b in block).strip()
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        meta[key.strip()] = value
        i += 1
    return meta, "\n".join(lines[end + 1 :]).lstrip("\n")


def _is_local(path: str) -> bool:
    return path.startswith(("~", "local:"))


def _local_path(path: str) -> str:
    return os.path.expanduser(path.removeprefix("local:"))


def read_file(executor: Executor, path: str, local: bool) -> str:
    if local:
        with open(path, encoding="utf-8", errors="replace") as f:
            return f.read()
    return executor.read_bytes(path).decode("utf-8", "replace")


def load_agents_md(executor: Executor, cfg: ContextConfig) -> list[tuple[str, str]]:
    """(source, text) for the global and workspace AGENTS.md files that exist."""
    found: list[tuple[str, str]] = []
    for name in cfg.agents_md_names:
        path = os.path.join(_local_path(cfg.global_dir), name)
        if os.path.isfile(path):
            found.append((path, read_file(executor, path, True)))
    for name in cfg.agents_md_names:
        path = executor.resolve(name)
        try:
            found.append((executor.rel(path), read_file(executor, path, False)))
        except (FileNotFoundError, IsADirectoryError, PermissionError):
            pass
    return found


def render_agents_md(files: list[tuple[str, str]], max_chars: int) -> str:
    if not files:
        return ""
    parts = []
    for source, text in files:
        text = text.strip()
        if len(text) > max_chars:
            text = text[:max_chars] + f"\n[... truncated; read {source} for the rest]"
        parts.append(f"Instructions from {source}:\n{text}")
    return "\n\n".join(parts)


def discover_skills(executor: Executor, cfg: ContextConfig) -> list[Skill]:
    skills: dict[str, Skill] = {}
    for d in cfg.skill_dirs:
        if _is_local(d):
            paths = sorted(glob.glob(os.path.join(_local_path(d), "*", "SKILL.md")))
            local = True
        else:
            root = executor.resolve(d)
            q = shlex.quote(root)
            r = executor.run(
                f'for f in {q}/*/SKILL.md; do [ -f "$f" ] && printf "%s\\0" "$f"; done; true',
                timeout=30,
                merge_stderr=False,
            )
            paths = sorted(p for p in r.output.split("\0") if p)
            local = False
        for path in paths:
            try:
                meta, body = parse_frontmatter(read_file(executor, path, local))
            except OSError:
                continue
            name = (meta.get("name") or posixpath.basename(posixpath.dirname(path))).strip()
            desc = meta.get("description") or next(
                (ln.strip("# ").strip() for ln in body.splitlines() if ln.strip()), ""
            )
            if name and name not in skills:  # earlier directories win
                skills[name] = Skill(name, " ".join(desc.split()), path, local)
    return list(skills.values())


def render_skill_index(skills: list[Skill]) -> str:
    if not skills:
        return ""
    lines = ["Skills (call skill with the name to load one before doing a task it covers):"]
    lines += [f"- {s.name}: {s.description}" for s in skills]
    return "\n".join(lines)


class SkillTool(Tool):
    """Progressive disclosure: the prompt lists names and descriptions (level 1); ``skill(name)``
    loads SKILL.md and lists the skill's other files (level 2); ``skill(name, file)`` loads
    one of those files (level 3). Each load is logged."""

    name = "skill"
    description = (
        "Load the instructions of a skill listed under Skills, or with file, one of the "
        "skill's other files."
    )
    summary = "load a skill's instructions (file?: one of its files)"

    def __init__(self, skills: list[Skill], executor: Executor):
        self.skills = {s.name: s for s in skills}
        self.executor = executor
        self.parameters = obj(
            {"name": {"type": "string", "enum": list(self.skills)}, "file": {"type": "string"}},
            ["name"],
        )

    def run(self, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
        skill = self.skills.get(args["name"])
        if skill is None:
            raise ToolError(f"no skill {args['name']!r}; available: {', '.join(self.skills)}")
        if args.get("file"):
            return self._file(ctx, skill, str(args["file"]))
        text = read_file(self.executor, skill.path, skill.local)
        _, body = parse_frontmatter(text)
        files = self._files(skill)
        where = (
            skill.dir
            if not skill.local
            else f"{skill.dir} (on the argus host, not in the workspace)"
        )
        footer = f"[skill directory: {where}"
        if files:
            footer += "; files: " + ", ".join(files)
        ctx.events.append({"kind": "skill", "skill": skill.name, "path": skill.path, "via": "tool"})
        return ToolResult(f"{body.strip()}\n{footer}]", meta={"skill": skill.name})

    def _file(self, ctx: ToolContext, skill: Skill, rel: str) -> ToolResult:
        path = posixpath.normpath(posixpath.join(skill.dir, rel))
        if not path.startswith(skill.dir.rstrip("/") + "/"):
            raise ToolError(f"{rel!r} is outside the skill directory")
        try:
            text = read_file(self.executor, path, skill.local)
        except (FileNotFoundError, IsADirectoryError):
            files = ", ".join(self._files(skill)) or "none"
            raise ToolError(f"skill {skill.name} has no file {rel!r}; files: {files}") from None
        limit = 20_000
        if len(text) > limit:
            text = text[:limit] + f"\n[... {len(text) - limit} more chars]"
        ctx.events.append({"kind": "skill", "skill": skill.name, "path": path, "via": "file"})
        return ToolResult(text, meta={"skill": skill.name, "file": rel})

    def _files(self, skill: Skill) -> list[str]:
        if skill.local:
            out = []
            for root, _, names in os.walk(skill.dir):
                for n in names:
                    out.append(os.path.relpath(os.path.join(root, n), skill.dir))
        else:
            r = self.executor.run(
                f"cd {shlex.quote(skill.dir)} && find . -type f | head -n {MAX_SKILL_FILES + 1}",
                timeout=30,
                merge_stderr=False,
            )
            out = [p.removeprefix("./") for p in r.output.splitlines() if p.strip()]
        out = sorted(p for p in out if p != "SKILL.md")
        if len(out) > MAX_SKILL_FILES:
            out = out[:MAX_SKILL_FILES] + ["…"]
        return out


def skill_for_path(skills: list[Skill], abs_path: str) -> Skill | None:
    for s in skills:
        if not s.local and s.path == abs_path:
            return s
    return None
