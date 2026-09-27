from __future__ import annotations

from argus.config import ToolsConfig
from argus.tools.base import FileTracker, Tool, ToolContext, ToolError, ToolResult
from argus.tools.bash import BashTool
from argus.tools.edit import EditTool
from argus.tools.files import LsTool, WriteTool
from argus.tools.jobs import JobTool
from argus.tools.read import ReadTool
from argus.tools.search import GlobTool, GrepTool
from argus.tools.web import FetchTool, WebSearchTool, search_backend

BUILTIN: dict[str, type[Tool]] = {
    "read": ReadTool,
    "edit": EditTool,
    "write": WriteTool,
    "bash": BashTool,
    "job": JobTool,
    "glob": GlobTool,
    "grep": GrepTool,
    "ls": LsTool,
    "fetch": FetchTool,
    "web_search": WebSearchTool,
}
FILE_WRITERS = ("edit", "write")  # tools that write one file the model names
NETWORK_TOOLS = ("fetch", "web_search")  # tools that reach the network from argus itself


def build_tools(cfg: ToolsConfig) -> dict[str, Tool]:
    tools: dict[str, Tool] = {}
    for name in cfg.enabled:
        if name == "skill":  # added by the agent when skills exist
            continue
        if name not in BUILTIN:
            raise ValueError(f"unknown tool {name!r}; available: {', '.join(BUILTIN)}")
        if name == "web_search":
            backend = search_backend(cfg)
            if backend is None:  # needs BRAVE_API_KEY or a SearXNG instance
                continue
            tool: Tool = WebSearchTool(*backend)
        else:
            tool = BUILTIN[name]()
        if name in cfg.descriptions:
            tool.description = tool.summary = cfg.descriptions[name]
        tools[name] = tool
    return tools


__all__ = [
    "BUILTIN",
    "FILE_WRITERS",
    "NETWORK_TOOLS",
    "FileTracker",
    "Tool",
    "ToolContext",
    "ToolError",
    "ToolResult",
    "build_tools",
]
