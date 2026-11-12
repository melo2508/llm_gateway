"""
HTTP routes — the public surface of the gateway.

A route handler should be THIN: parse input, call a service, translate the
result (or error) into HTTP. Business logic (token counting, rate limiting,
failover) lives in `app/services/`, not here. That keeps the logic testable
without spinning up HTTP at all.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from redis.asyncio import Redis
from redis.exceptions import RedisError

from app.api.deps import get_current_tenant, get_provider, get_redis
from app.schemas import ChatCompletionRequest, ChatCompletionResponse, Tenant
from app.services.mock_llm import MockLLMProvider, MockProviderError

# APIRouter is a "mini app": a group of routes that main.py plugs into the
# real FastAPI app. As the project grows (admin routes, etc.) each group gets
# its own router and file instead of one giant main.py.
router = APIRouter()


@router.get("/health", tags=["ops"])
async def health() -> dict[str, str]:
    """LIVENESS: "is this process alive and able to answer HTTP?"

    Orchestrators (e.g. Kubernetes) RESTART the process when this fails.
    So it must not check dependencies like Redis: if Redis goes down,
    restarting healthy gateway processes can't fix Redis — it only causes a
    restart storm. Dependency checks belong in /ready.
    """
    return {"status": "ok"}


@router.get("/ready", tags=["ops"])
async def ready(redis: Redis = Depends(get_redis)) -> dict[str, str]:
    """READINESS: "can this process serve real traffic right now?"

    Orchestrators STOP SENDING TRAFFIC here when this fails (no restart).
    When Redis recovers, /ready passes again and traffic resumes on its own.
    """
    try:
        await redis.ping()
    except RedisError as exc:
        # 503 Service Unavailable: "temporarily can't serve, try later".
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Redis unavailable",
        ) from exc
    return {"status": "ready"}


@router.post(
    "/v1/chat/completions",
    response_model=ChatCompletionResponse,
    tags=["chat"],
)
async def chat_completions(
    body: ChatCompletionRequest,
    provider: MockLLMProvider = Depends(get_provider),
    tenant: Tenant = Depends(get_current_tenant),
) -> ChatCompletionResponse:
    """OpenAI-compatible chat completion endpoint.

    By the time this function runs, FastAPI has already:
      - authenticated the caller (no/bad key → 401, we never run)
      - parsed the JSON body
      - validated it against ChatCompletionRequest (bad input → 422, we never run)
      - called get_provider() and passed us the result

    `tenant` is a parameter (not a router-level `dependencies=[...]` like the
    admin gate) because we need its VALUE: the rate limiter, next, will
    enforce this tenant's own limits.
    """
    try:
        return await provider.generate(body)
    except MockProviderError as exc:
        # 502 Bad Gateway, not 500: the error isn't a bug in OUR code — the
        # upstream provider we proxy to failed. 500 would send on-call engineers
        # hunting through gateway code for a problem that isn't there.
        # (503 we'll reserve for "we refused to even try", e.g. circuit open.)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Upstream provider error: {exc.message}",
        ) from exc
