"""API-shape tests for the FastAPI app: doesn't require a live LLM call, just checks
the endpoint contract (request/response shape, validation) matches what the brief specifies.
"""

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
