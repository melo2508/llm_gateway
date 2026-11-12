"""
Tests for tenant authentication on /v1/chat/completions.
"""

from fastapi.testclient import TestClient

from app.api.deps import get_redis
from app.core.redis import create_redis_client
from app.main import app
from tests.conftest import UNREACHABLE_REDIS_URL

CHAT_URL = "/v1/chat/completions"
VALID_BODY = {"model": "mock-gpt-4", "messages": [{"role": "user", "content": "hi"}]}


# ─── Accepted ways to send the key ──────────────────────────────────────────


def test_x_api_key_header_is_accepted(client: TestClient, api_key: str) -> None:
    response = client.post(CHAT_URL, json=VALID_BODY, headers={"X-API-Key": api_key})

    assert response.status_code == 200


def test_bearer_token_is_accepted(client: TestClient, api_key: str) -> None:
    # Exactly what the official OpenAI SDK sends.
    response = client.post(
        CHAT_URL, json=VALID_BODY, headers={"Authorization": f"Bearer {api_key}"}
    )

    assert response.status_code == 200


def test_bearer_scheme_is_case_insensitive(client: TestClient, api_key: str) -> None:
    response = client.post(
        CHAT_URL, json=VALID_BODY, headers={"Authorization": f"bearer {api_key}"}
    )

    assert response.status_code == 200


def test_bearer_wins_when_both_headers_sent(client: TestClient, api_key: str) -> None:
    headers = {"Authorization": f"Bearer {api_key}", "X-API-Key": "gw_wrong"}

    response = client.post(CHAT_URL, json=VALID_BODY, headers=headers)

    assert response.status_code == 200


# ─── Rejected requests ──────────────────────────────────────────────────────


def test_missing_key_is_401_with_www_authenticate(client: TestClient) -> None:
    response = client.post(CHAT_URL, json=VALID_BODY)

    assert response.status_code == 401
    # HTTP spec: a 401 must say how to authenticate.
    assert response.headers["WWW-Authenticate"] == "Bearer"


def test_unknown_key_is_401(client: TestClient) -> None:
    response = client.post(CHAT_URL, json=VALID_BODY, headers={"X-API-Key": "gw_made-up"})

    assert response.status_code == 401
    assert response.json() == {"detail": "Invalid API key"}


def test_non_bearer_authorization_is_401(client: TestClient, api_key: str) -> None:
    # A real key, but under the wrong scheme — not accepted.
    response = client.post(
        CHAT_URL, json=VALID_BODY, headers={"Authorization": f"Basic {api_key}"}
    )

    assert response.status_code == 401


def test_auth_runs_before_body_validation(client: TestClient) -> None:
    # Bad body AND no key → 401, not 422. Unauthenticated callers learn
    # nothing about what our API accepts.
    response = client.post(CHAT_URL, json={"garbage": True})

    assert response.status_code == 401


def test_redis_down_during_auth_is_503_not_500(client: TestClient, api_key: str) -> None:
    app.dependency_overrides[get_redis] = lambda: create_redis_client(UNREACHABLE_REDIS_URL)

    response = client.post(CHAT_URL, json=VALID_BODY, headers={"X-API-Key": api_key})

    assert response.status_code == 503


# ─── Unprotected routes stay unprotected ────────────────────────────────────


def test_health_and_ready_need_no_key(client: TestClient) -> None:
    # Load balancers and orchestrators call these without credentials.
    assert client.get("/health").status_code == 200
    assert client.get("/ready").status_code == 200
