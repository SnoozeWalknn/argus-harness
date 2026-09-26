"""System prompt assembly. Kept deliberately short: every token here is paid on every turn."""

from __future__ import annotations

from dataclasses import dataclass, field

BASE_PROMPT = (
    "You are a coding agent working in {workdir}. Use the tools to inspect, edit and test "
    "code; read a file before editing it. {finish}"
)


@dataclass
class PromptParts:
    """The pieces of the system prompt, kept separate so overhead can be attributed."""

    base: str
    protocol: str = ""
    agents_md: str = ""
    skills: str = ""
    extra: list[str] = field(default_factory=list)

    def render(self) -> str:
        parts = [self.base, self.protocol, self.agents_md, self.skills, *self.extra]
        return "\n\n".join(p.strip() for p in parts if p and p.strip())


def base_prompt(template: str, workdir: str, finish_hint: str) -> str:
    template = template or BASE_PROMPT
    return template.replace("{workdir}", workdir).replace("{finish}", finish_hint).strip()
