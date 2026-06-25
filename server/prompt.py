"""Flatten an OpenAI ``messages`` array into a single Copilot prompt.

Copilot's protocol has no role/system channel — it takes one prompt string per
turn — so we collapse the whole conversation into one piece of text.
"""

import json

from typing import Any, List, Optional, Union

from .schemas import ChatMessage


def content_text(content: Optional[Union[str, List[Any]]]) -> str:
    """Extract plain text from a message's content (string or content-parts)."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts = []
    for part in content:
        if isinstance(part, dict):
            if part.get("type") == "text":
                parts.append(part.get("text", ""))
        else:
            parts.append(str(part))
    return "\n".join(p for p in parts if p)


def messages_to_prompt(messages: List[ChatMessage]) -> str:
    """Flatten an OpenAI ``messages`` array into a single Copilot prompt."""
    system = "\n\n".join(
        content_text(m.content) for m in messages if m.role == "system" and m.content
    )
    convo = [m for m in messages if m.role != "system"]

    if len(convo) == 1 and convo[0].role == "user":
        body = content_text(convo[0].content)  # simple single-turn request
    else:
        lines = []
        for m in convo:
            if m.role == "user":
                lines.append(f"User: {content_text(m.content)}")
            elif m.role == "tool":
                tool_id = f" {m.tool_call_id}" if m.tool_call_id else ""
                lines.append(f"Tool result{tool_id}: {content_text(m.content)}")
            elif m.tool_calls:
                lines.append(
                    "Assistant tool calls: "
                    + json.dumps(m.tool_calls, ensure_ascii=False, separators=(",", ":"))
                )
            else:
                lines.append(f"Assistant: {content_text(m.content)}")
        lines.append("Assistant:")  # cue Copilot to continue
        body = "\n".join(lines)

    if system and body:
        return f"{system}\n\n{body}"
    return system or body


def compact_messages_to_prompt(messages: List[ChatMessage], max_chars: int) -> str:
    """Flatten messages, keeping recent user-visible context within a char budget."""
    prompt = messages_to_prompt(messages)
    if len(prompt) <= max_chars:
        return prompt

    system = "\n\n".join(
        content_text(m.content) for m in messages if m.role == "system" and m.content
    )
    convo = [m for m in messages if m.role != "system"]

    lines = []
    for m in reversed(convo[-12:]):
        rendered = _render_message(m)
        if rendered:
            lines.append(rendered)
    recent = "\n".join(reversed(lines))
    if recent and not recent.rstrip().endswith("Assistant:"):
        recent = f"{recent}\nAssistant:"

    notice = (
        "[Compatibility note: older system/developer context was omitted before "
        "sending this request to Copilot because the upstream consumer endpoint "
        "has a much smaller prompt limit than Codex.]"
    )
    system_budget = max(0, max_chars - len(recent) - len(notice) - 6)
    compact_system = _head_tail(system, min(system_budget, 6000)) if system_budget else ""

    parts = [p for p in (compact_system, notice, recent) if p]
    compact = "\n\n".join(parts)
    if len(compact) <= max_chars:
        return compact
    return _head_tail(compact, max_chars)


def _render_message(message: ChatMessage) -> str:
    if message.role == "user":
        return f"User: {content_text(message.content)}"
    if message.role == "tool":
        tool_id = f" {message.tool_call_id}" if message.tool_call_id else ""
        return f"Tool result{tool_id}: {content_text(message.content)}"
    if message.tool_calls:
        return (
            "Assistant tool calls: "
            + json.dumps(message.tool_calls, ensure_ascii=False, separators=(",", ":"))
        )
    if message.role == "assistant":
        return f"Assistant: {content_text(message.content)}"
    return f"{message.role}: {content_text(message.content)}"


def _head_tail(text: str, max_chars: int) -> str:
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    if max_chars < 200:
        return text[-max_chars:]
    head = max_chars // 3
    tail = max_chars - head - 44
    return f"{text[:head]}\n\n[... omitted for Copilot prompt budget ...]\n\n{text[-tail:]}"
