"""API-shape tests for the FastAPI app: doesn't require a live LLM call, just checks
the endpoint contract (request/response shape, validation) matches what the brief specifies.
"""

import json

import httpx
from openai import APIStatusError, RateLimitError
from fastapi.testclient import TestClient

import app.main as main_module
from app.main import app

client = TestClient(app)


def _fake_response(status_code: int) -> httpx.Response:
    return httpx.Response(status_code, request=httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions"))


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_chat_endpoint_requires_message_field():
    response = client.post("/chat", json={})
    assert response.status_code == 422  # pydantic validation error, not a 500


def test_chat_endpoint_rejects_wrong_type():
    response = client.post("/chat", json={"message": 12345})
    assert response.status_code == 422


def test_ui_served_at_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "SafarIA Assistant" in response.text


def test_chat_endpoint_handles_insufficient_credits_gracefully(monkeypatch):
    """Real bug caught via manual UI testing against OpenRouter: an unhandled 402 used to
    propagate as an unhandled exception (blank/broken response in the UI) instead of a
    clean error the frontend can render. This is the regression test for the fix."""
    def raise_402(*args, **kwargs):
        raise APIStatusError("Insufficient credits", response=_fake_response(402), body=None)

    monkeypatch.setattr(main_module, "router_chat", raise_402)
    response = client.post("/chat", json={"message": "What are the baggage rules?"})
    assert response.status_code == 502
    assert "credits" in response.json()["detail"].lower()


def test_chat_endpoint_handles_rate_limit_gracefully(monkeypatch):
    def raise_rate_limit(*args, **kwargs):
        raise RateLimitError("Rate limited", response=_fake_response(429), body=None)

    monkeypatch.setattr(main_module, "router_chat", raise_rate_limit)
    response = client.post("/chat", json={"message": "What are the baggage rules?"})
    assert response.status_code == 429


def test_chat_endpoint_generates_session_id_when_none_given(monkeypatch):
    """No live LLM call needed here - this just checks the session plumbing in main.py,
    not whether the model actually uses the history correctly (that's covered live in
    test_router.py::test_conversational_memory_uses_prior_context)."""
    def fake_chat(client, message, messages=None, **kwargs):
        history = messages or []
        return {"answer": "ok", "source": "llm", "tool_calls": [], "messages": history + [{"role": "user", "content": message}]}

    monkeypatch.setattr(main_module, "router_chat", fake_chat)
    response = client.post("/chat", json={"message": "Hello"})
    assert response.status_code == 200
    session_id = response.json()["session_id"]
    assert session_id  # non-empty, server-generated


def test_chat_endpoint_reuses_and_extends_session_history(monkeypatch):
    received_histories = []

    def fake_chat(client, message, messages=None, **kwargs):
        received_histories.append(messages)
        history = messages or []
        new_history = history + [{"role": "user", "content": message}, {"role": "assistant", "content": "ok"}]
        return {"answer": "ok", "source": "llm", "tool_calls": [], "messages": new_history}

    monkeypatch.setattr(main_module, "router_chat", fake_chat)

    first = client.post("/chat", json={"message": "My flight is AH1235"})
    session_id = first.json()["session_id"]

    second = client.post("/chat", json={"message": "Is it delayed?", "session_id": session_id})
    assert second.json()["session_id"] == session_id

    # first call had no prior history; second call received exactly what the first call returned
    assert received_histories[0] is None
    assert received_histories[1] == [
        {"role": "user", "content": "My flight is AH1235"},
        {"role": "assistant", "content": "ok"},
    ]


def test_stream_endpoint_emits_sse_events_and_persists_session(monkeypatch):
    """The streaming transport must produce the same routing decision and session
    behaviour as /chat - only the delivery differs. Driven with a fake generator so this
    stays a non-live test."""
    def fake_stream(client, message, messages=None, **kwargs):
        yield {"type": "tool", "name": "search_knowledge_base"}
        yield {"type": "token", "text": "Cabin "}
        yield {"type": "token", "text": "baggage is 8kg."}
        yield {"type": "done", "answer": "Cabin baggage is 8kg.", "source": "rag",
               "messages": [{"role": "assistant", "content": "Cabin baggage is 8kg."}]}

    monkeypatch.setattr(main_module, "router_chat_stream", fake_stream)

    with client.stream("POST", "/chat/stream", json={"message": "baggage?"}) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        events = [json.loads(line[6:]) for line in response.iter_lines()
                  if line.startswith("data: ")]

    assert [e["type"] for e in events] == ["tool", "token", "token", "done"]
    # Tokens must reassemble into exactly the final answer, or the UI shows something
    # different from what the badge claims was produced.
    assert "".join(e["text"] for e in events if e["type"] == "token") == events[-1]["answer"]
    assert events[-1]["source"] == "rag"
    assert events[-1]["session_id"]
    # "messages" is server state for the next turn, not something a browser should receive.
    assert "messages" not in events[-1]


def test_stream_endpoint_reports_provider_errors_as_events(monkeypatch):
    """A streamed response has already committed HTTP 200 by the time a provider fails,
    so the failure has to arrive as an event rather than a status code."""
    def raise_402(client, message, messages=None, **kwargs):
        raise APIStatusError("Insufficient credits", response=_fake_response(402), body=None)
        yield  # pragma: no cover - makes this a generator

    monkeypatch.setattr(main_module, "router_chat_stream", raise_402)

    with client.stream("POST", "/chat/stream", json={"message": "hi"}) as response:
        assert response.status_code == 200
        events = [json.loads(line[6:]) for line in response.iter_lines()
                  if line.startswith("data: ")]

    assert events[-1]["type"] == "error"
    assert "credits" in events[-1]["detail"]
