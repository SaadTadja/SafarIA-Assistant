"""FastAPI app exposing POST /chat, as required by the challenge brief.
Also serves a simple browser UI (bonus point: "interface utilisateur simple") at "/".
"""

import os
from pathlib import Path

import openai
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .router import build_client, chat as router_chat

load_dotenv()  # picks up OPENROUTER_API_KEY from a local .env file, if present

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="SafarIA Assistant")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

_client = None


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


class ChatResponse(BaseModel):
    answer: str
    source: str


@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest) -> ChatResponse:
    client = get_client()
    try:
        result = router_chat(client, request.message)
    except openai.RateLimitError:
        raise HTTPException(status_code=429, detail="Rate limited by the LLM provider - please try again shortly.")
    except openai.APIStatusError as e:
        if e.status_code == 402:
            detail = "The LLM provider account has run out of credits."
        else:
            detail = f"The LLM provider returned an error ({e.status_code})."
        raise HTTPException(status_code=502, detail=detail)
    return ChatResponse(answer=result["answer"], source=result["source"])


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/")
def ui() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")
