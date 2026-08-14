"""FastAPI app exposing POST /chat, as required by the challenge brief.
Also serves a simple browser UI (bonus point: "interface utilisateur simple") at "/".
"""

import json
import os
import uuid
from pathlib import Path

import openai
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .router import build_client, chat as router_chat, chat_stream as router_chat_stream

load_dotenv()  # picks up OPENROUTER_API_KEY from a local .env file, if present

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="SafarIA Assistant")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

_client = None

# In-memory conversation store: session_id -> the "messages" list from that session's last
# router.chat() call. Ephemeral (lost on restart) and unbounded (no eviction) - fine for a
# demo/single-process deployment, not for production (would need Redis/a DB with TTL there).
_sessions: dict[str, list[dict]] = {}


def get_client():
    global _client
    if _client is None:
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise HTTPException(
                status_code=503,
                detail="OPENROUTER_API_KEY environment variable is not set",
            )
        _client = build_client(api_key)
    return _client


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None  # omit to start a new conversation


class ChatResponse(BaseModel):
    answer: str
    source: str
    session_id: str


@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest) -> ChatResponse:
    client = get_client()
    session_id = request.session_id or str(uuid.uuid4())
    history = _sessions.get(session_id)
    try:
        result = router_chat(client, request.message, messages=history)
    except openai.RateLimitError:
        raise HTTPException(status_code=429, detail="Rate limited by the LLM provider - please try again shortly.")
    except openai.APIStatusError as e:
        if e.status_code == 402:
            detail = "The LLM provider account has run out of credits."
        else:
            detail = f"The LLM provider returned an error ({e.status_code})."
        raise HTTPException(status_code=502, detail=detail)
    _sessions[session_id] = result["messages"]
    return ChatResponse(answer=result["answer"], source=result["source"], session_id=session_id)


@app.post("/chat/stream")
def chat_stream_endpoint(request: ChatRequest) -> StreamingResponse:
    """Server-sent events version of /chat (challenge bonus: streaming).

    /chat is kept as-is rather than reimplemented on top of this. It is what the tests and
    eval scripts drive, and its single JSON response is the more useful contract for a
    programmatic caller; streaming only helps a human watching text appear.

    Errors are a genuine limitation of streaming: the HTTP status is committed the moment
    the first byte is sent, so a provider failure mid-answer cannot become a 502. Those are
    emitted as an "error" event instead and rendered in the bubble by the UI. Failures
    raised before the first token still surface as normal HTTP errors.
    """
    client = get_client()
    session_id = request.session_id or str(uuid.uuid4())
    history = _sessions.get(session_id)

    def event_stream():
        try:
            for event in router_chat_stream(client, request.message, messages=history):
                if event["type"] == "done":
                    _sessions[session_id] = event.pop("messages")
                    event["session_id"] = session_id
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        except openai.RateLimitError:
            yield f"data: {json.dumps({'type': 'error', 'detail': 'Rate limited by the LLM provider - please try again shortly.'})}\n\n"
        except openai.APIStatusError as e:
            detail = ("The LLM provider account has run out of credits."
                      if e.status_code == 402
                      else f"The LLM provider returned an error ({e.status_code}).")
            yield f"data: {json.dumps({'type': 'error', 'detail': detail})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        # Without this an intermediate proxy will happily buffer the whole response and
        # deliver it in one piece, which looks exactly like streaming not working.
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/")
def ui() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")
