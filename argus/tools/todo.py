"""The ``todo`` tool: a task list the model keeps for multi-step work.

The model passes the whole list every time (replace, not patch), which keeps the
tool trivial for small models and makes every state change a logged snapshot.
"""

from __future__ import annotations

from typing import Any

from argus.tools.base import Tool, ToolContext, ToolError, ToolResult, obj

STATUSES = ("pending", "in_progress", "done")
MARKS = {"pending": "[ ]", "in_progress": "[>]", "done": "[x]"}
MAX_ITEMS = 30


def render(items: list[dict[str, Any]]) -> str:
    return "\n".join(f"{MARKS[i['status']]} {i['content']}" for i in items)


class TodoTool(Tool):
    name = "todo"
    description = (
        "Keep a task list for multi-step work. Pass the whole list each time; "
        "mark one item in_progress while you work on it and done when finished."
    )
    summary = "replace your task list (items: content, status)"
    parameters = obj(
        {
            "items": {
                "type": "array",
                "items": obj(
                    {
                        "content": {"type": "string"},
                        "status": {"type": "string", "enum": list(STATUSES)},
                    },
                    ["content", "status"],
                ),
            }
        },
        ["items"],
    )
    mutating = False

    def run(self, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
        items = [
            {"content": " ".join(str(i["content"]).split()), "status": i["status"]}
            for i in args["items"]
            if str(i.get("content", "")).strip()
        ]
        if len(items) > MAX_ITEMS:
            raise ToolError(f"keep the list under {MAX_ITEMS} items; merge small steps")
        ctx.todos = items
        ctx.events.append({"kind": "todo", "items": items})
        if not items:
            return ToolResult("task list cleared", meta={"items": 0})
        done = sum(i["status"] == "done" for i in items)
        active = sum(i["status"] == "in_progress" for i in items)
        note = ""
        if active > 1:
            note = "\n(note: more than one item is in progress; finish one at a time)"
        return ToolResult(
            f"{render(items)}\n({done}/{len(items)} done){note}",
            meta={"items": len(items), "done": done, "in_progress": active},
        )
