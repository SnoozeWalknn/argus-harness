from __future__ import annotations

from argus.config import ToolsConfig
from argus.tools.base import FileTracker, Tool, ToolContext, ToolError, ToolResult
from argus.tools.bash import BashTool
from argus.tools.edit import EditTool
from argus.tools.read import ReadTool
from argus.tools.search import GlobTool, GrepTool

BUILTIN: dict[str, type[Tool]] = {
    "read": ReadTool,
    "edit": EditTool,
    "bash": BashTool,
    "glob": GlobTool,
    "grep": GrepTool,
}


def build_tools(cfg: ToolsConfig) -> dict[str, Tool]:
    tools: dict[str, Tool] = {}
    for name in cfg.enabled:
        if name == "skill":  # added by the agent when skills exist
            continue
        if name not in BUILTIN:
            raise ValueError(f"unknown tool {name!r}; available: {', '.join(BUILTIN)}")
        tool = BUILTIN[name]()
        if name in cfg.descriptions:
            tool.description = tool.summary = cfg.descriptions[name]
        tools[name] = tool
    return tools


__all__ = [
    "BUILTIN",
    "FileTracker",
    "Tool",
    "ToolContext",
    "ToolError",
    "ToolResult",
    "build_tools",
]
