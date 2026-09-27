"""Backwards-compatible names for the model client.

The client moved to :mod:`argus.providers`; ``LLMClient`` is the
OpenAI-compatible adapter (llama-server and friends).
"""

from __future__ import annotations

from argus.providers.base import (
    Completion,
    ContextOverflow,
    Delta,
    LLMError,
    Monitor,
    NotSupported,
    Provider,
    RawToolCall,
    StopGeneration,
    root_url,
)
from argus.providers.openai_compat import OpenAICompatProvider as LLMClient

__all__ = [
    "Completion",
    "ContextOverflow",
    "Delta",
    "LLMClient",
    "LLMError",
    "Monitor",
    "NotSupported",
    "Provider",
    "RawToolCall",
    "StopGeneration",
    "root_url",
]
