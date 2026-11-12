"""
Unit tests for MockLLMProvider — called directly, no HTTP, no FastAPI.

UNIT vs API TESTS:
  - API tests (test_api.py) prove the pieces are wired together correctly.
  - Unit tests prove one piece's logic is right, in isolation. They're faster
    and, when they fail, point at exactly one file.
"""

import pytest

from app.schemas import ChatCompletionRequest, ChatMessage
from app.services.mock_llm import MockLLMProvider, MockProviderError


def make_request(*contents: str) -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model="mock-gpt-4",
        messages=[ChatMessage(role="user", content=c) for c in contents],
    )


# `generate()` is async, so its tests must run inside an event loop.
# The anyio pytest plugin (installed alongside FastAPI) provides that:
# mark the test with @pytest.mark.anyio and pin the backend to asyncio.
@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def test_count_tokens_uses_real_tokenizer(fast_provider: MockLLMProvider) -> None:
    # tiktoken splits "Explain rate limiting" into ["Ex", "plain", " rate",
    # " limiting"]. We pin the exact number so a tokenizer change is noticed.
    assert fast_provider._count_tokens("Explain rate limiting") == 4


def test_prompt_tokens_add_four_overhead_per_message(fast_provider: MockLLMProvider) -> None:
    one = fast_provider._count_prompt_tokens(make_request("hi"))
    two = fast_provider._count_prompt_tokens(make_request("hi", "hi"))

    # Each extra identical message adds its content tokens + 4 overhead.
    assert two - one == fast_provider._count_tokens("hi") + 4


def test_response_text_is_deterministic(fast_provider: MockLLMProvider) -> None:
    request = make_request("same input")

    assert fast_provider._generate_response_text(request) == (
        fast_provider._generate_response_text(request)
    )


@pytest.mark.anyio
async def test_generate_returns_consistent_usage(fast_provider: MockLLMProvider) -> None:
    response = await fast_provider.generate(make_request("Explain rate limiting"))

    assert response.usage.prompt_tokens == 4 + 4  # content + per-message overhead
    assert response.usage.total_tokens == (
        response.usage.prompt_tokens + response.usage.completion_tokens
    )


@pytest.mark.anyio
async def test_generate_raises_when_failure_rate_is_one() -> None:
    provider = MockLLMProvider(latency_min=0.0, latency_max=0.0, failure_rate=1.0)

    # pytest.raises asserts the block raises THIS exception type;
    # the test fails if nothing (or something else) is raised.
    with pytest.raises(MockProviderError):
        await provider.generate(make_request("hi"))
