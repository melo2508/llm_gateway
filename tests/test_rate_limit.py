"""
Tests for per-tenant rate limiting (token bucket in Redis).
"""

import asyncio

import pytest
from fastapi.testclient import TestClient
from redis.asyncio import Redis

from app.services.rate_limiter import RateLimiter

CHAT_URL = "/v1/chat/completions"
VALID_BODY = {"model": "mock-gpt-4", "messages": [{"role": "user", "content": "hi"}]}


# ─── Unit tests: the token bucket itself ────────────────────────────────────


@pytest.mark.anyio
async def test_bucket_allows_up_to_limit_then_rejects(redis_client: Redis) -> None:
    limiter = RateLimiter(redis_client)

    results = [await limiter.hit("bucket", limit=3) for _ in range(4)]

    assert [r.allowed for r in results] == [True, True, True, False]
    assert [r.remaining for r in results] == [2, 1, 0, 0]
    # Rejected → told how long to wait: 1 token at 3/min = 20s.
    assert results[3].retry_after == pytest.approx(20, abs=0.5)


@pytest.mark.anyio
async def test_bucket_refills_over_time(redis_client: Redis) -> None:
    limiter = RateLimiter(redis_client)
    # A tiny period so the test doesn't wait a minute: 2 tokens per 0.2s
    # = refills one token every 0.1s.
    await limiter.hit("bucket", limit=2, period_seconds=0.2)
    await limiter.hit("bucket", limit=2, period_seconds=0.2)
    assert not (await limiter.hit("bucket", limit=2, period_seconds=0.2)).allowed

    await asyncio.sleep(0.15)  # > 0.1s → at least one token is back

    assert (await limiter.hit("bucket", limit=2, period_seconds=0.2)).allowed


@pytest.mark.anyio
async def test_buckets_are_independent_per_key(redis_client: Redis) -> None:
    limiter = RateLimiter(redis_client)
    await limiter.hit("tenant-a", limit=1)

    # tenant-a is now empty; tenant-b is untouched.
    assert not (await limiter.hit("tenant-a", limit=1)).allowed
    assert (await limiter.hit("tenant-b", limit=1)).allowed


@pytest.mark.anyio
async def test_no_extra_requests_slip_through_under_concurrency(redis_client: Redis) -> None:
    """THE race-condition test.

    50 hits fired at the same instant at a bucket of 10. If read-check-write
    were separate commands, several hits would read the same token count
    and more than 10 would get through. The Lua script makes each hit one
    indivisible step, so EXACTLY 10 succeed.
    """
    limiter = RateLimiter(redis_client)

    # asyncio.gather starts all 50 coroutines concurrently; they go out over
    # the connection pool in parallel.
    results = await asyncio.gather(*(limiter.hit("hot", limit=10) for _ in range(50)))

    assert sum(r.allowed for r in results) == 10


@pytest.mark.anyio
async def test_idle_bucket_expires(redis_client: Redis) -> None:
    limiter = RateLimiter(redis_client)
    await limiter.hit("bucket", limit=60)

    # 1 token used at 1 token/s → full again in ~1s → key expires ~2s.
    ttl = await redis_client.ttl("bucket")
    assert 0 < ttl <= 2


# ─── API tests: the limit enforced on /v1/chat/completions ──────────────────


def make_tenant_headers(
    client: TestClient, admin_headers: dict[str, str], requests_per_minute: int
) -> dict[str, str]:
    """Create a tenant with a specific limit; return its auth headers."""
    created = client.post(
        "/admin/tenants",
        json={"name": "limited", "requests_per_minute": requests_per_minute},
        headers=admin_headers,
    ).json()
    return {"X-API-Key": created["api_key"]}


def test_success_includes_rate_limit_headers(
    client: TestClient, admin_headers: dict[str, str]
) -> None:
    headers = make_tenant_headers(client, admin_headers, requests_per_minute=5)

    response = client.post(CHAT_URL, json=VALID_BODY, headers=headers)

    assert response.status_code == 200
    assert response.headers["X-RateLimit-Limit-Requests"] == "5"
    assert response.headers["X-RateLimit-Remaining-Requests"] == "4"
    assert int(response.headers["X-RateLimit-Reset-Requests"]) > 0


def test_exceeding_limit_returns_429_with_retry_after(
    client: TestClient, admin_headers: dict[str, str]
) -> None:
    headers = make_tenant_headers(client, admin_headers, requests_per_minute=2)
    client.post(CHAT_URL, json=VALID_BODY, headers=headers)
    client.post(CHAT_URL, json=VALID_BODY, headers=headers)

    response = client.post(CHAT_URL, json=VALID_BODY, headers=headers)

    assert response.status_code == 429
    assert response.headers["X-RateLimit-Remaining-Requests"] == "0"
    # 1 token at 2/min = 30s; Retry-After must be whole seconds.
    assert response.headers["Retry-After"] == "30"


def test_limits_are_per_tenant(client: TestClient, admin_headers: dict[str, str]) -> None:
    noisy = make_tenant_headers(client, admin_headers, requests_per_minute=1)
    quiet = make_tenant_headers(client, admin_headers, requests_per_minute=1)
    client.post(CHAT_URL, json=VALID_BODY, headers=noisy)

    # The noisy tenant is blocked; the quiet one is unaffected by it.
    assert client.post(CHAT_URL, json=VALID_BODY, headers=noisy).status_code == 429
    assert client.post(CHAT_URL, json=VALID_BODY, headers=quiet).status_code == 200


def test_unauthenticated_requests_do_not_consume_quota(
    client: TestClient, admin_headers: dict[str, str]
) -> None:
    headers = make_tenant_headers(client, admin_headers, requests_per_minute=1)
    for _ in range(5):
        client.post(CHAT_URL, json=VALID_BODY)  # no key → 401, before the limiter

    # The tenant's single request is still available.
    assert client.post(CHAT_URL, json=VALID_BODY, headers=headers).status_code == 200
