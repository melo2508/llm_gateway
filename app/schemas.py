"""
Pydantic models that define the exact shape of data flowing through the gateway.

These schemas serve two purposes:
1. VALIDATION — When a client sends a request, Pydantic automatically checks
   that every required field is present and has the correct type. If someone
   sends {"model": 123} instead of {"model": "gpt-4"}, Pydantic rejects it
   instantly with a clear error, before our code even runs.
2. SERIALIZATION — When we send a response back, Pydantic converts our Python
   objects into clean JSON matching the OpenAI format.

We follow the OpenAI API format because it's the industry standard. If a client
already works with OpenAI, it can point at our gateway with zero code changes.
"""

from pydantic import BaseModel, Field
import time
import uuid


# ─── REQUEST SCHEMAS ─────────────────────────────────────────────────────────


class ChatMessage(BaseModel):
    """A single message in a conversation.

    LLMs are stateless — they don't "remember" previous messages. The client
    must send the ENTIRE conversation history every time. Each message has a
    role:
      - "system": Instructions to the LLM (e.g., "You are a code reviewer")
      - "user": The human's message
      - "assistant": The LLM's previous reply (for multi-turn conversations)
    """

    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    """The request body a client sends to POST /v1/chat/completions.

    Example:
    {
        "model": "mock-gpt-4",
        "messages": [
            {"role": "user", "content": "Explain rate limiting"}
        ],
        "temperature": 0.7
    }
    """

    model: str
    messages: list[ChatMessage]

    # temperature controls randomness. 0.0 = deterministic, 1.0 = creative.
    # Default 0.7 matches OpenAI's default.
    # Our mock ignores this, but we accept it so clients don't get errors.
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)

    # max_tokens caps how many tokens the LLM can generate in its response.
    # None means "no limit" (let the model decide when to stop).
    max_tokens: int | None = None


# ─── RESPONSE SCHEMAS ────────────────────────────────────────────────────────


class Usage(BaseModel):
    """Token usage statistics for a single request.

    This is how LLM providers bill you:
      - prompt_tokens: tokens in your input (you pay ~$0.03 per 1K for GPT-4)
      - completion_tokens: tokens the LLM generated (~$0.06 per 1K for GPT-4)
      - total_tokens: the sum

    Our gateway will use these numbers for:
      1. Per-tenant cost tracking (who spent how much)
      2. Token-based rate limiting (cap tokens/minute, not just requests/minute)
    """

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChoiceMessage(BaseModel):
    """The LLM's response message inside a Choice."""

    role: str = "assistant"
    content: str


class Choice(BaseModel):
    """One possible response from the LLM.

    Why is this a list called "choices"?
    OpenAI's API supports generating multiple alternative responses to the same
    prompt (the `n` parameter). In practice, almost everyone uses n=1, so
    choices always has exactly one item. But the list structure is part of the
    API contract, so we follow it.

    finish_reason tells the client WHY the LLM stopped generating:
      - "stop": The LLM naturally finished its response.
      - "length": The response was cut off because it hit max_tokens.
    """

    index: int = 0
    message: ChoiceMessage
    finish_reason: str = "stop"


class ChatCompletionResponse(BaseModel):
    """The complete response object returned to the client.

    We generate a unique ID and timestamp for every response. These are
    essential for:
      - Debugging: "Which exact request caused this issue?"
      - Logging: Correlating gateway logs with client logs
      - Caching: Identifying duplicate responses
    """

    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:12]}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[Choice]
    usage: Usage


# ─── TENANT SCHEMAS (our own admin API, not OpenAI format) ──────────────────
#
# Three models for three "views" of a tenant — a common pattern:
#   TenantCreate  → what the admin SENDS (no id: the server assigns it)
#   Tenant        → what we STORE and SHOW (never contains the API key)
#   TenantCreated → Tenant + the raw api_key, returned exactly ONCE
# Separate models make it structurally impossible to, say, accidentally
# return the API key from a "get tenant" endpoint.


class TenantCreate(BaseModel):
    """Request body for POST /admin/tenants."""

    name: str = Field(min_length=1, max_length=100)
    # ge=1: a limit of 0 or below would make the tenant unusable — reject it
    # at the edge rather than discover it in the rate limiter.
    requests_per_minute: int = Field(default=60, ge=1)
    tokens_per_minute: int = Field(default=10_000, ge=1)


class Tenant(BaseModel):
    """A tenant as stored in Redis and returned by the admin API."""

    id: str
    name: str
    requests_per_minute: int
    tokens_per_minute: int
    created_at: int


class TenantCreated(Tenant):
    """Response for POST /admin/tenants — the ONLY time the key is shown.

    We store just its hash, so we can't show it again later. Same model as
    GitHub/Stripe: "copy this key now, you won't see it again".
    """

    api_key: str
