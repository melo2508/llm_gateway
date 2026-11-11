"""
HTTP routes — the public surface of the gateway.

A route handler should be THIN: parse input, call a service, translate the
result (or error) into HTTP. Business logic (token counting, rate limiting,
failover) lives in `app/services/`, not here. That keeps the logic testable
without spinning up HTTP at all.
"""

from fastapi import APIRouter, Depends, HTTPException, status

from app.api.deps import get_provider
from app.schemas import ChatCompletionRequest, ChatCompletionResponse
from app.services.mock_llm import MockLLMProvider, MockProviderError

# APIRouter is a "mini app": a group of routes that main.py plugs into the
# real FastAPI app. As the project grows (admin routes, etc.) each group gets
# its own router and file instead of one giant main.py.
router = APIRouter()


@router.get("/health", tags=["ops"])
async def health() -> dict[str, str]:
    """Liveness check: "is the process up and able to answer HTTP?"

    Load balancers and Kubernetes hit this every few seconds. It must be
    cheap and must NOT call the LLM. (Once Redis exists we'll discuss whether
    to also check dependencies here — liveness vs. readiness.)
    """
    return {"status": "ok"}


@router.post(
    "/v1/chat/completions",
    response_model=ChatCompletionResponse,
    tags=["chat"],
)
async def chat_completions(
    body: ChatCompletionRequest,
    provider: MockLLMProvider = Depends(get_provider),
) -> ChatCompletionResponse:
    """OpenAI-compatible chat completion endpoint.

    By the time this function runs, FastAPI has already:
      - parsed the JSON body
      - validated it against ChatCompletionRequest (bad input → 422, we never run)
      - called get_provider() and passed us the result
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
