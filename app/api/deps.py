"""
FastAPI dependencies — small functions that hand route handlers what they need.

WHY DEPENDENCIES INSTEAD OF IMPORTING A GLOBAL?
A route could just do `from somewhere import provider` and call it. That works,
but it hard-wires the route to one specific object. With `Depends(get_provider)`:

  1. The route declares WHAT it needs, not WHERE it comes from.
  2. Tests can swap the real thing for a fake in one line:
         app.dependency_overrides[get_provider] = lambda: FakeProvider()
     No monkeypatching of module globals.
  3. This file becomes the single place for "per-request plumbing". Next up,
     API-key authentication (X-API-Key header) will live here too.
"""

import secrets

from fastapi import Header, HTTPException, Request, status
from redis.asyncio import Redis

from app.core.config import settings
from app.services.mock_llm import MockLLMProvider


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
