"""OpenAI tool-calling compatibility helpers.

Copilot's upstream web protocol accepts plain prompts and returns plain text.
This module keeps tool support as a small compatibility layer around that
constraint: describe the available tools in the prompt, then parse a strict JSON
tool-call response back into OpenAI's wire shape.
"""

import json
import re
import uuid
from typing import Any, Iterable, List, Optional


_FENCED_JSON_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)


def has_tools(tools: Optional[List[Any]]) -> bool:
    return bool(tools)


def build_tool_prompt(
    prompt: str,
    tools: List[Any],
    tool_choice: Optional[Any] = None,
    parallel_tool_calls: Optional[bool] = None,
) -> str:
    """Append deterministic tool-use instructions to a flattened prompt."""
    mode_lines = [
        "You have access to the following tools in OpenAI Chat Completions format.",
        "When a tool is needed, reply with only valid JSON and no prose.",
        'Use exactly this shape: {"tool_calls":[{"name":"tool_name","arguments":{}}]}',
        "When no tool is needed, answer normally in plain text.",
    ]

    if tool_choice == "none":
        mode_lines.append("Tool choice is none: do not call tools.")
    elif tool_choice == "required":
        mode_lines.append("Tool choice is required: call one of the tools.")
    elif isinstance(tool_choice, dict):
        function = tool_choice.get("function") or {}
        name = function.get("name")
        if name:
            mode_lines.append(f"Tool choice requires calling only this tool: {name}.")

    if parallel_tool_calls is False:
        mode_lines.append("Call at most one tool.")

    return "\n\n".join(
        [
            "Tool calling instructions:\n" + "\n".join(f"- {line}" for line in mode_lines),
            "Available tools JSON:\n" + json.dumps(tools, ensure_ascii=False, separators=(",", ":")),
            "Conversation:\n" + prompt,
        ]
    )


def parse_tool_calls(text: str) -> Optional[List[dict]]:
    """Parse a Copilot text reply into OpenAI ``tool_calls`` if possible."""
    payload = _load_json_object(text)
    if not isinstance(payload, dict):
        return None

    raw_calls = payload.get("tool_calls")
    if raw_calls is None and "name" in payload:
        raw_calls = [payload]
    if not isinstance(raw_calls, list) or not raw_calls:
        return None

    calls = []
    for raw in raw_calls:
        call = _normalize_call(raw)
        if call is None:
            return None
        calls.append(call)
    return calls


def _load_json_object(text: str) -> Optional[Any]:
    candidate = text.strip()
    fenced = _FENCED_JSON_RE.search(candidate)
    if fenced:
        candidate = fenced.group(1).strip()

    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return None


def _normalize_call(raw: Any) -> Optional[dict]:
    if not isinstance(raw, dict):
        return None

    function = raw.get("function") if isinstance(raw.get("function"), dict) else {}
    name = raw.get("name") or function.get("name")
    if not isinstance(name, str) or not name:
        return None

    arguments = raw.get("arguments", function.get("arguments", {}))
    if isinstance(arguments, str):
        try:
            parsed_args = json.loads(arguments) if arguments.strip() else {}
        except json.JSONDecodeError:
            parsed_args = arguments
    else:
        parsed_args = arguments

    if parsed_args is None:
        parsed_args = {}

    return {
        "id": raw.get("id") or f"call_{uuid.uuid4().hex[:24]}",
        "type": "function",
        "function": {
            "name": name,
            "arguments": _arguments_json(parsed_args),
        },
    }


def _arguments_json(arguments: Any) -> str:
    if isinstance(arguments, str):
        return arguments
    return json.dumps(arguments, ensure_ascii=False, separators=(",", ":"))


def tool_calls_delta(tool_calls: Iterable[dict]) -> List[dict]:
    """Build a compact streaming delta for parsed tool calls."""
    deltas = []
    for index, call in enumerate(tool_calls):
        deltas.append(
            {
                "index": index,
                "id": call["id"],
                "type": "function",
                "function": {
                    "name": call["function"]["name"],
                    "arguments": call["function"]["arguments"],
                },
            }
        )
    return deltas
