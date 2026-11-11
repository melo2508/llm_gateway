"""
Mock LLM Provider — a controllable fake that simulates real LLM behavior.

WHAT THIS MODULE DOES:
  Accepts a chat completion request (same format as OpenAI), pretends to
  "think" for a realistic amount of time, and returns a fake but correctly
  structured response — complete with accurate token counts.

WHY EVERY DESIGN CHOICE WAS MADE:

1. ASYNC SLEEP, NOT SYNC SLEEP
   We use `await asyncio.sleep()` to simulate latency. This is critical.
   - `time.sleep(0.2)` would BLOCK the thread. During those 200ms, the
     entire server is frozen — no other request can be processed.
   - `await asyncio.sleep(0.2)` YIELDS control to the event loop. The
     server continues handling other requests during the wait, then comes
     back to this request when the timer fires.
   This is the difference between a restaurant where the waiter stands
   frozen in the kitchen vs. one where the waiter serves other tables.

2. TIKTOKEN FOR TOKEN COUNTING
   We use OpenAI's official `tiktoken` library to count tokens accurately.
   Why not just divide character count by 4?
   - "Hello" = 1 token (5 chars → naive estimate = 1.25, rounds to 1 ✓)
   - "unconstitutional" = 4 tokens (18 chars → naive = 4.5, close)
   - "🎉" = 1 token (4 bytes → naive says 1, but encoding varies)
   - JSON/code has very different tokenization than prose.
   Accurate token counts matter because our rate limiter will enforce
   tokens-per-minute limits, and our cost tracker will bill per token.

3. DETERMINISTIC RESPONSE GENERATION
   Given the same input, the mock always returns the same response text.
   This makes our tests reliable and repeatable. Real LLMs are stochastic
   (random), which makes debugging hard.

4. INJECTABLE FAILURE MODES
   The mock can be configured to fail X% of the time. This isn't useful
   now, but when we build the Circuit Breaker in Week 2, we'll crank up
   the failure rate and verify that the gateway gracefully falls back to
   a secondary provider instead of crashing.
"""

import asyncio
import random

import tiktoken

from app.core.config import settings
from app.schemas import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    Choice,
    ChoiceMessage,
    Usage,
)


class MockProviderError(Exception):
    """Raised when the mock provider simulates a failure.

    In a real system, this would be an HTTP 503 (Service Unavailable) or
    a timeout from the upstream LLM provider. Our gateway will catch this
    and trigger failover / circuit breaker logic.
    """

    def __init__(self, message: str = "Mock provider simulated failure"):
        self.message = message
        super().__init__(self.message)


class MockLLMProvider:
    """A fake LLM provider with controllable latency and failure behavior.

    DESIGN PATTERN: This is a "Port/Adapter" or "Strategy" pattern.
    Our gateway code will call `provider.generate(request)` without knowing
    or caring whether `provider` is a MockLLMProvider, an OpenAIProvider,
    or an AnthropicProvider. This lets us swap providers without changing
    gateway logic — which is exactly what failover does.
    """

    def __init__(
        self,
        latency_min: float = settings.mock_latency_min,
        latency_max: float = settings.mock_latency_max,
        failure_rate: float = settings.mock_failure_rate,
        model_name: str = "mock-gpt-4",
    ):
        self.latency_min = latency_min
        self.latency_max = latency_max
        self.failure_rate = failure_rate
        self.model_name = model_name

        # Load the tokenizer once during initialization, not on every request.
        # tiktoken.get_encoding("cl100k_base") returns the tokenizer used by
        # GPT-4 and GPT-3.5-turbo. It splits text into tokens using Byte Pair
        # Encoding (BPE), the same algorithm the real model uses.
        #
        # WHY LOAD ONCE?
        # Loading a tokenizer involves reading a ~1.7MB vocabulary file from
        # disk. If we did this on every request, we'd add ~10ms of latency
        # and unnecessary disk I/O. Loading once and reusing is a classic
        # optimization pattern called "eager initialization."
        self._encoding = tiktoken.get_encoding("cl100k_base")

    def _count_tokens(self, text: str) -> int:
        """Count the exact number of tokens in a text string.

        Uses tiktoken's Byte Pair Encoding (BPE) tokenizer, the same one
        GPT-4 uses internally. This gives us production-accurate counts.

        Example:
            "What is a rate limiter?" → 7 tokens
            ["What", " is", " a", " rate", " lim", "iter", "?"]
        """
        return len(self._encoding.encode(text))

    def _count_prompt_tokens(self, request: ChatCompletionRequest) -> int:
        """Count total tokens across all messages in a request.

        In the real OpenAI API, token counting includes:
        - The content of each message
        - A few overhead tokens per message for role markers and formatting
          (approximately 4 tokens per message)

        We add 4 tokens per message to approximate this overhead.
        """
        total = 0
        for message in request.messages:
            # Tokens in the actual text content
            total += self._count_tokens(message.content)
            # Overhead per message: role name, formatting tokens.
            # The real API uses ~4 tokens per message for metadata.
            total += 4
        return total

    def _generate_response_text(self, request: ChatCompletionRequest) -> str:
        """Generate a deterministic fake response based on the input.

        WHY DETERMINISTIC?
        If the mock returned random text each time, our tests would be
        flaky — sometimes passing, sometimes failing, depending on the
        random content. Deterministic output means:
          Same input → Same output → Reliable, repeatable tests.

        We include the model name and message count to make responses
        distinguishable when debugging multi-provider scenarios.
        """
        last_message = request.messages[-1].content
        return (
            f"This is a mock response from {self.model_name}. "
            f"You sent {len(request.messages)} message(s). "
            f"Your last message was: '{last_message[:100]}'"
        )

    async def generate(
        self, request: ChatCompletionRequest
    ) -> ChatCompletionResponse:
        """Process a chat completion request and return a response.

        This is the main entry point. The flow is:
        1. Check if we should simulate a failure (based on failure_rate)
        2. Simulate network latency (async sleep — non-blocking!)
        3. Count tokens in the prompt
        4. Generate a deterministic response
        5. Count tokens in the response
        6. Package everything into the OpenAI response format

        This method is `async` because it uses `await asyncio.sleep()`.
        The `async` keyword tells Python: "This function contains pause
        points (await). Don't run it directly — schedule it on the event
        loop so other tasks can run during the pauses."
        """

        # ── Step 1: Simulate failures ────────────────────────────────────
        # random.random() returns a float between 0.0 and 1.0.
        # If failure_rate is 0.3, there's a 30% chance this condition is True.
        if random.random() < self.failure_rate:
            # Simulate the delay even for failures — real servers don't
            # fail instantly. They might hang for a bit before returning 503.
            latency = random.uniform(self.latency_min, self.latency_max)
            await asyncio.sleep(latency)
            raise MockProviderError(
                f"Simulated failure from {self.model_name} after {latency:.3f}s"
            )

        # ── Step 2: Simulate network latency ─────────────────────────────
        # Real LLM API calls take 500ms to 5000ms. We simulate a smaller
        # range (50ms to 300ms by default) so tests run fast, but this is
        # configurable via settings for realistic load testing.
        #
        # random.uniform(a, b) picks a random float between a and b.
        # This creates realistic jitter — not every request takes the same
        # time, just like real network calls.
        latency = random.uniform(self.latency_min, self.latency_max)
        await asyncio.sleep(latency)

        # ── Step 3: Count prompt tokens ──────────────────────────────────
        prompt_tokens = self._count_prompt_tokens(request)

        # ── Step 4: Generate response text ───────────────────────────────
        response_text = self._generate_response_text(request)

        # ── Step 5: Count completion tokens ──────────────────────────────
        completion_tokens = self._count_tokens(response_text)

        # ── Step 6: Build the OpenAI-format response ─────────────────────
        return ChatCompletionResponse(
            model=self.model_name,
            choices=[
                Choice(
                    message=ChoiceMessage(content=response_text),
                )
            ],
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
        )
