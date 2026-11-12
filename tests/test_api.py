"""
API tests — exercise the app through HTTP, the same way a real client would.

These turn the manual curl checks into automated ones: health 200,
chat 200 with the right shape, bad input 422, provider failure 502.
"""

from fastapi.testclient import TestClient

from app.services.mock_llm import MockLLMProvider

CHAT_URL = "/v1/chat/completions"
VALID_BODY = {
    "model": "mock-gpt-4",
    "messages": [{"role": "user", "content": "Explain rate limiting"}],
}


def test_health_returns_ok(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_chat_completion_returns_openai_shape(client: TestClient, api_headers: dict[str, str]) -> None:
    response = client.post(CHAT_URL, json=VALID_BODY, headers=api_headers)

    assert response.status_code == 200
    data = response.json()
    # Check the CONTRACT (field names/types an OpenAI client relies on),
    # not the exact mock text — that would make the test brittle.
    assert data["object"] == "chat.completion"
    assert data["id"].startswith("chatcmpl-")
    assert isinstance(data["created"], int)
    assert len(data["choices"]) == 1
    assert data["choices"][0]["message"]["role"] == "assistant"
    assert data["choices"][0]["finish_reason"] == "stop"


def test_usage_totals_add_up(client: TestClient, api_headers: dict[str, str]) -> None:
    usage = client.post(CHAT_URL, json=VALID_BODY, headers=api_headers).json()["usage"]

    assert usage["prompt_tokens"] > 0
    assert usage["completion_tokens"] > 0
    assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]


def test_each_response_gets_a_unique_id(client: TestClient, api_headers: dict[str, str]) -> None:
    # Guards the default_factory decision in schemas.py: a plain default
    # would give every response the SAME id.
    first = client.post(CHAT_URL, json=VALID_BODY, headers=api_headers).json()["id"]
    second = client.post(CHAT_URL, json=VALID_BODY, headers=api_headers).json()["id"]

    assert first != second


def test_temperature_out_of_range_is_rejected(client: TestClient, api_headers: dict[str, str]) -> None:
    body = {**VALID_BODY, "temperature": 5}

    response = client.post(CHAT_URL, json=body, headers=api_headers)

    assert response.status_code == 422
    # The error points at the exact offending field.
    assert response.json()["detail"][0]["loc"] == ["body", "temperature"]


def test_missing_messages_is_rejected(client: TestClient, api_headers: dict[str, str]) -> None:
    response = client.post(CHAT_URL, json={"model": "mock-gpt-4"}, headers=api_headers)

    assert response.status_code == 422


def test_provider_failure_maps_to_502(client: TestClient, api_headers: dict[str, str], use_provider) -> None:
    use_provider(MockLLMProvider(latency_min=0.0, latency_max=0.0, failure_rate=1.0))

    response = client.post(CHAT_URL, json=VALID_BODY, headers=api_headers)

    assert response.status_code == 502
    assert "Upstream provider error" in response.json()["detail"]
