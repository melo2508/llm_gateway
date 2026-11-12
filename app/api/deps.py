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

from fastapi import Request
from redis.asyncio import Redis

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
