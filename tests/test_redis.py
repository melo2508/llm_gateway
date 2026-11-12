"""
Tests for the Redis connection: fail-fast startup, /ready, and clean errors.
"""

import pytest
from fastapi.testclient import TestClient

from app.api.deps import get_redis
from app.core.config import settings
from app.core.redis import connect_redis, create_redis_client
from app.main import app
from tests.conftest import UNREACHABLE_REDIS_URL

# ─── Unit tests: app/core/redis.py ──────────────────────────────────────────


@pytest.mark.anyio
async def test_connect_redis_returns_working_client(test_redis_url: str) -> None:
    client = await connect_redis(test_redis_url)
    try:
        assert await client.ping() is True
        # decode_responses=True → we get str back, not bytes.
        await client.set("greeting", "hello")
        assert await client.get("greeting") == "hello"
    finally:
        await client.aclose()


@pytest.mark.anyio
async def test_connect_redis_fails_fast_when_unreachable() -> None:
    with pytest.raises(RuntimeError, match="Cannot reach Redis"):
        await connect_redis(UNREACHABLE_REDIS_URL)


@pytest.mark.anyio
async def test_connect_error_does_not_leak_password() -> None:
    # Locks in the security decision: error messages end up in logs, so the
    # password embedded in a Redis URL must never appear in them.
    with pytest.raises(RuntimeError) as exc_info:
        await connect_redis("redis://:supersecret@localhost:1/0")

    assert "supersecret" not in str(exc_info.value)


# ─── App-level tests: lifespan and /ready ───────────────────────────────────


def test_app_refuses_to_start_without_redis(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "redis_url", UNREACHABLE_REDIS_URL)

    # Entering `with TestClient(app)` runs lifespan startup; connect_redis
    # raises, so the app never becomes ready to serve.
    with pytest.raises(RuntimeError, match="Cannot reach Redis"):
        with TestClient(app):
            pass


def test_ready_ok_when_redis_reachable(client: TestClient) -> None:
    response = client.get("/ready")

    assert response.status_code == 200
    assert response.json() == {"status": "ready"}


def test_ready_503_when_redis_goes_down(client: TestClient) -> None:
    # The app started fine (Redis was up). Now simulate Redis dying mid-run
    # by swapping in a client that points at a dead port.
    app.dependency_overrides[get_redis] = lambda: create_redis_client(UNREACHABLE_REDIS_URL)

    response = client.get("/ready")

    assert response.status_code == 503
    assert response.json() == {"detail": "Redis unavailable"}


def test_health_stays_ok_when_redis_goes_down(client: TestClient) -> None:
    # Liveness must NOT depend on Redis — otherwise a Redis outage would make
    # the orchestrator restart perfectly healthy gateway processes.
    app.dependency_overrides[get_redis] = lambda: create_redis_client(UNREACHABLE_REDIS_URL)

    assert client.get("/health").status_code == 200
