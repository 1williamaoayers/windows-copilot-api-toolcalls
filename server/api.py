"""FastAPI app wiring Copilot onto the OpenAI Chat Completions API."""

import os
import threading
import time

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from copilot import CopilotClient
from copilot.driver import ClearanceRequired

from .config import MODEL_NAME, RATE_LIMIT_BURST, RATE_LIMIT_RPM
from .openai_format import (
    completion_response,
    new_id,
    sse_event,
    stream_chunk,
    tool_calls_response,
)
from .prompt import compact_messages_to_prompt
from .ratelimit import TokenBucket
from .schemas import ChatCompletionRequest
from .tool_calling import build_tool_prompt, has_tools, parse_tool_calls, tool_calls_delta

app = FastAPI(title="Copilot OpenAI-compatible API", version="1.0.0")
# Server runs headless and must never pop a visible browser mid-request. With
# both recovery passes disabled, an expired clearance surfaces immediately as a
# 503 (see ClearanceRequired handling below) so an operator can re-clear out of
# band (`python -m copilot login`). Headless auto-solve is intentionally off:
# it's unreliable on low-trust egress and a failed pass can wedge the session.
client = CopilotClient(interactive_clear=False, headless_clear=False)

_CLEARANCE_HELP = (
    "Cloudflare clearance expired and could not be refreshed headlessly. "
    "Re-clear in a browser: run `python -m copilot login` (or `python tests/diagnostic.py`) "
    "and pass the 'verify you're human' check, then retry."
)
_MAX_COPILOT_PROMPT_CHARS = int(os.environ.get("MAX_COPILOT_PROMPT_CHARS", "18000"))

# Self-imposed rate limit on top of the concurrency lock below: this caps
# requests-per-minute, the lock caps requests-in-flight. See server/ratelimit.py.
_rate_limiter = TokenBucket(RATE_LIMIT_RPM, RATE_LIMIT_BURST)


def _rate_limited_response():
    """Spend a token; return an OpenAI-shaped 429 if none left, else ``None``."""
    allowed, wait = _rate_limiter.try_acquire()
    if allowed:
        return None
    secs = max(1, round(wait))
    return JSONResponse(
        status_code=429,
        headers={"Retry-After": str(secs)},
        content={"error": {
            "message": (
                f"Rate limit exceeded (>{RATE_LIMIT_RPM:g} req/min). "
                f"Retry in {secs}s."
            ),
            "type": "rate_limit_error",
            "code": "rate_limit_exceeded",
        }},
    )

# Copilot's per-account chat socket doesn't tolerate concurrent conversations
# from one process (parallel requests error out or hang). This server bridges a
# single signed-in account, so we serialize upstream calls: concurrent HTTP
# requests queue here and run one at a time. Predictable, at the cost of
# parallelism — fine for a personal bridge.
_upstream_lock = threading.Lock()


def _stream(prompt: str, model: str, conversation_id=None, req: ChatCompletionRequest = None):
    """Yield OpenAI ``chat.completion.chunk`` SSE events for ``prompt``.

    ``conversation_id`` continues an existing Copilot thread; ``None`` starts a
    fresh one (its id is emitted on the final chunk).
    """
    cid = new_id()
    created = int(time.time())
    try:
        with _upstream_lock:  # one upstream chat at a time (released on disconnect)
            yield sse_event(stream_chunk(cid, created, model, {"role": "assistant"}))
            if req is not None and has_tools(req.tools):
                tool_prompt = build_tool_prompt(
                    prompt, req.tools, req.tool_choice, req.parallel_tool_calls,
                    _MAX_COPILOT_PROMPT_CHARS,
                )
                reply = client.chat(tool_prompt, conversation_id=conversation_id)
                tool_calls = parse_tool_calls(reply.text)
                if tool_calls:
                    yield sse_event(
                        stream_chunk(cid, created, model, {"tool_calls": tool_calls_delta(tool_calls)})
                    )
                    finish = "tool_calls"
                else:
                    yield sse_event(stream_chunk(cid, created, model, {"content": reply.text}))
                    finish = "stop"
                yield sse_event(
                    stream_chunk(cid, created, model, {}, finish=finish, conversation_id=reply.conversation_id)
                )
            else:
                stream = client.stream(prompt, conversation_id=conversation_id)
                for piece in stream:
                    if isinstance(piece, str) and piece:
                        yield sse_event(stream_chunk(cid, created, model, {"content": piece}))
                # Copilot's conversation id is known once the stream has run; emit it
                # on the final chunk so callers can track the upstream thread.
                yield sse_event(
                    stream_chunk(
                        cid, created, model, {}, finish="stop",
                        conversation_id=stream.conversation_id,
                    )
                )
    except ClearanceRequired:
        yield sse_event(
            stream_chunk(cid, created, model, {"content": f"\n[error: {_CLEARANCE_HELP}]"}, finish="error")
        )
    except Exception as exc:  # surface errors to the client instead of hanging
        yield sse_event(
            stream_chunk(cid, created, model, {"content": f"\n[error: {exc}]"}, finish="error")
        )
    yield "data: [DONE]\n\n"


@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [
            {"id": MODEL_NAME, "object": "model", "created": 0, "owned_by": "microsoft"}
        ],
    }


@app.post("/v1/chat/completions")
def chat_completions(req: ChatCompletionRequest):
    prompt = compact_messages_to_prompt(req.messages, _MAX_COPILOT_PROMPT_CHARS)
    if not prompt.strip():
        return JSONResponse(
            status_code=400,
            content={"error": {"message": "no text content in messages", "type": "invalid_request_error"}},
        )
    model = req.model or MODEL_NAME

    # Enforce the per-minute ceiling before touching the upstream lock, so excess
    # callers get a fast 429 instead of piling up behind the serialized queue.
    limited = _rate_limited_response()
    if limited is not None:
        return limited

    if req.stream:
        return StreamingResponse(
            _stream(prompt, model, req.conversation_id, req), media_type="text/event-stream"
        )

    try:
        with _upstream_lock:  # serialize: one upstream chat at a time
            upstream_prompt = (
                build_tool_prompt(
                    prompt, req.tools, req.tool_choice, req.parallel_tool_calls,
                    _MAX_COPILOT_PROMPT_CHARS,
                )
                if has_tools(req.tools)
                else prompt
            )
            reply = client.chat(upstream_prompt, conversation_id=req.conversation_id)
    except ClearanceRequired:
        return JSONResponse(
            status_code=503,
            content={"error": {"message": _CLEARANCE_HELP, "type": "clearance_required"}},
        )
    except Exception as exc:
        return JSONResponse(
            status_code=502,
            content={"error": {"message": str(exc), "type": "upstream_error"}},
        )
    if has_tools(req.tools):
        tool_calls = parse_tool_calls(reply.text)
        if tool_calls:
            return tool_calls_response(tool_calls, model, reply.conversation_id)
    return completion_response(reply.text, model, reply.conversation_id)


@app.post("/v1/responses")
async def responses(req: Request):
    """Minimal OpenAI Responses API shim for Codex-style clients."""
    payload = await req.json()
    chat_req = ChatCompletionRequest(
        messages=_responses_input_to_messages(payload.get("input")),
        model=payload.get("model") or MODEL_NAME,
        stream=False,
        tools=payload.get("tools"),
        tool_choice=payload.get("tool_choice"),
        parallel_tool_calls=payload.get("parallel_tool_calls"),
    )
    result = chat_completions(chat_req)
    if isinstance(result, JSONResponse):
        return result

    choice = result["choices"][0]
    message = choice["message"]
    output = []
    if message.get("tool_calls"):
        for call in message["tool_calls"]:
            output.append(
                {
                    "id": call["id"],
                    "type": "function_call",
                    "name": call["function"]["name"],
                    "arguments": call["function"]["arguments"],
                    "call_id": call["id"],
                }
            )
    else:
        output.append(
            {
                "id": f"msg_{new_id()[9:]}",
                "type": "message",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": message.get("content") or "",
                    }
                ],
            }
        )

    return {
        "id": f"resp_{new_id()[9:]}",
        "object": "response",
        "created_at": result["created"],
        "model": result["model"],
        "output": output,
        "conversation_id": result.get("conversation_id"),
        "usage": result.get("usage", {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}),
    }


def _responses_input_to_messages(input_value):
    if isinstance(input_value, str):
        return [{"role": "user", "content": input_value}]
    if isinstance(input_value, list):
        messages = []
        for item in input_value:
            if not isinstance(item, dict):
                messages.append({"role": "user", "content": str(item)})
                continue
            if item.get("type") == "message":
                content = item.get("content")
                if isinstance(content, list):
                    text = "\n".join(
                        part.get("text", "")
                        for part in content
                        if isinstance(part, dict) and part.get("type") in {"input_text", "output_text"}
                    )
                else:
                    text = content
                messages.append({"role": item.get("role", "user"), "content": text})
            elif "role" in item:
                messages.append(item)
            elif item.get("type") == "function_call_output":
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": item.get("call_id"),
                        "content": item.get("output", ""),
                    }
                )
        if messages:
            return messages
    return [{"role": "user", "content": ""}]


@app.get("/")
def root():
    return {"service": "Copilot OpenAI-compatible API", "endpoints": ["/v1/models", "/v1/chat/completions", "/v1/responses"]}
