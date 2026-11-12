"""
Application entry point — builds the FastAPI app and wires everything together.

Run it with:
    uv run uvicorn app.main:app --reload

`app.main:app` means "import module app.main, use the variable named `app`".
uvicorn is the ASGI server: it owns the socket and the event loop, receives
raw HTTP bytes, and hands each request to FastAPI.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes import router
from app.core.config import settings
from app.core.redis import connect_redis
from app.services.mock_llm import MockLLMProvider


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Startup and shutdown logic, in one function.

    Everything BEFORE `yield` runs once when the server starts (before it
    accepts any request). Everything AFTER `yield` runs once on shutdown.

    WHY CREATE THE PROVIDER HERE, not at module import time?
      - Expensive/stateful resources (tokenizers, Redis connection pools,
        HTTP clients) get a clear, explicit birth and death.
      - Importing app.main (e.g. in a test) doesn't trigger side effects.
      - When we add Redis, its connection pool opens here and is closed
        after `yield` — otherwise connections leak on every restart.
    """
    # Connect first: if Redis is unreachable this raises, uvicorn reports
    # "Application startup failed" and exits — the server never accepts
    # traffic it can't serve (fail fast).
    app.state.redis = await connect_redis()
    app.state.provider = MockLLMProvider()
    yield
    # Shutdown: close pooled connections cleanly instead of leaving the OS
    # (and Redis, which tracks clients) to discover dead sockets later.
    await app.state.redis.aclose()


app = FastAPI(title=settings.app_name, debug=settings.debug, lifespan=lifespan)
app.include_router(router)


if __name__ == "__main__":
    # Convenience: `uv run python -m app.main` also works.
    import uvicorn

    uvicorn.run("app.main:app", host=settings.host, port=settings.port, reload=True)
