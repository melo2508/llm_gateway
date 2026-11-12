"""
FastAPI dependencies — small functions that hand route handlers what they need.

WHY DEPENDENCIES INSTEAD OF IMPORTING A GLOBAL?
A route could just do `from somewhere import provider` and call it. That works,
but it hard-wires the route to one specific object. With `Depends(get_provider)`:

  1. The route declares WHAT it needs, not WHERE it comes from.
  2. Tests can swap the real thing for a fake in one line:
         app.dependency_overrides[get_provider] = lambda: FakeProvider()
     No monkeypatching of module globals.
  3. This file is the single place for "per-request plumbing": shared
     resources (provider, Redis) and authentication (admin, tenant).
"""

import secrets

from fastapi import Depends, Header, HTTPException, Request, status
from redis.asyncio import Redis
from redis.exceptions import RedisError

from app.core.config import settings
from app.schemas import Tenant
from app.services.mock_llm import MockLLMProvider
from app.services.tenants import get_tenant_by_api_key


def get_provider(request: Request) -> MockLLMProvider:
    """Return the LLM provider created at startup (see `lifespan` in main.py).

    The provider lives on `app.state`, which is FastAPI's place for objects
    that exist once per application (not once per request). Every request
    reuses the same provider, so the tiktoken encoder is loaded only once.
    """
    return request.app.state.provider


def get_redis(request: Request) -> Redis:
    """Return the shared Redis client created at startup.

    Same pattern as get_provider: one client (with its connection pool) per
    application, borrowed by every request — and swappable in tests.
    """
    return request.app.state.redis


def require_admin(x_admin_key: str | None = Header(default=None)) -> None:
    """Allow the request only if X-Admin-Key matches settings.admin_api_key.

    `Header()` tells FastAPI to read this from a request header. It converts
    the parameter name automatically: x_admin_key → "X-Admin-Key"
    (underscores → hyphens, case-insensitive, as HTTP headers are).

    Returns nothing: it's a GATE. Either it raises (request stops here) or
    it returns and the route runs.
    """
    expected = settings.admin_api_key
    if expected is None:
        # Fail closed: no admin secret configured → admin API is OFF,
        # not open to everyone.
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin API disabled: ADMIN_API_KEY is not configured",
        )

    # compare_digest instead of ==: `==` stops at the FIRST differing
    # character, so a wrong key that starts correctly takes a few nanoseconds
    # longer to reject. Measured over many requests, that timing difference
    # can leak the secret one character at a time (a "timing attack").
    # compare_digest always takes the same time regardless of where they differ.
    if x_admin_key is None or not secrets.compare_digest(
        x_admin_key.encode(), expected.get_secret_value().encode()
    ):
        # 401 = "we don't know who you are" (missing/invalid credentials).
        # 403 (above) = "we know, but this is not allowed".
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing X-Admin-Key",
        )


def _extract_api_key(authorization: str | None, x_api_key: str | None) -> str | None:
    """Pull the tenant key from either supported header.

    1. `Authorization: Bearer <key>` — what the official OpenAI SDKs send, so
       existing OpenAI clients work by just changing base_url and api_key.
    2. `X-API-Key: <key>` — simpler for curl and scripts.
    Bearer wins if both are present. The scheme is case-insensitive
    ("bearer" == "Bearer"), per the HTTP spec.
    """
    if authorization:
        scheme, _, token = authorization.partition(" ")
        if scheme.lower() == "bearer" and token.strip():
            return token.strip()
    if x_api_key:
        return x_api_key.strip()
    return None


# Per the HTTP spec, a 401 must tell the client HOW to authenticate.
_UNAUTHORIZED_HEADERS = {"WWW-Authenticate": "Bearer"}


async def get_current_tenant(
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None),
    redis: Redis = Depends(get_redis),
) -> Tenant:
    """Authenticate the request and return the calling tenant.

    A dependency with its OWN dependency (get_redis): FastAPI resolves the
    whole chain — get_redis first, then this — before the route runs.

    Returning the Tenant (instead of just allowing/denying like
    require_admin) is the point: the route — and next, the rate limiter —
    needs to know WHO is calling to apply THEIR limits.

    NO TIMING ATTACK HERE (unlike the admin key):
    We never compare the raw key. We hash it and look the hash up in Redis.
    Even if lookup timing leaked something about the stored hash, an
    attacker can't work backwards from a SHA-256 hash to a valid key.
    """
    api_key = _extract_api_key(authorization, x_api_key)
    if api_key is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API key: send 'Authorization: Bearer <key>' or 'X-API-Key: <key>'",
            headers=_UNAUTHORIZED_HEADERS,
        )

    try:
        tenant = await get_tenant_by_api_key(redis, api_key)
    except RedisError as exc:
        # Redis down mid-run. Without this, the error escapes as a generic
        # 500 ("bug in our code"). It's really "temporarily can't verify
        # you" → 503, the same signal /ready gives.
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication temporarily unavailable",
        ) from exc

    if tenant is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers=_UNAUTHORIZED_HEADERS,
        )
    return tenant
