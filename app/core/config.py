"""
Application configuration using Pydantic Settings.

WHY NOT HARDCODE VALUES?
In production, you never hardcode things like Redis addresses, port numbers,
or latency parameters directly in your code. Why?

1. Different environments need different values:
   - Local development: Redis at localhost:6379
   - Production: Redis at redis-cluster.internal:6380
   If values are hardcoded, you'd need to edit code for each environment.

2. Secrets (API keys, passwords) should NEVER be in source code.
   They come from environment variables or secret managers.

HOW PYDANTIC SETTINGS WORKS:
It reads values from environment variables automatically.
If you set REDIS_URL=redis://prod-server:6380 in your environment,
Pydantic Settings picks it up. If not set, it uses the default.

This is the "12-Factor App" methodology (https://12factor.net/config),
which every Big Tech company follows: configuration lives in the
environment, not in code.
"""

from pydantic import SecretStr
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """All configuration for the LLM Gateway.

    Every field here can be overridden by setting an environment variable
    with the same name (case-insensitive). For example:
        export MOCK_LATENCY_MIN=0.1
        export REDIS_URL=redis://prod:6380
    """

    # ── Application ──────────────────────────────────────────────────────
    app_name: str = "LLM Gateway"
    debug: bool = True

    # ── Mock LLM Provider ────────────────────────────────────────────────
    # Simulated latency range in seconds.
    # Real LLMs take 0.5s to 5s. We use smaller values for fast testing
    # but make them configurable so we can simulate slow providers too.
    mock_latency_min: float = 0.05  # 50 milliseconds
    mock_latency_max: float = 0.3   # 300 milliseconds

    # Probability of simulated failure (0.0 = never fail, 1.0 = always fail).
    # We start at 0.0 (reliable mock) and crank this up when testing our
    # circuit breaker and failover logic in later milestones.
    mock_failure_rate: float = 0.0

    # ── Redis ────────────────────────────────────────────────────────────
    redis_url: str = "redis://localhost:6379/0"

    # Max seconds to wait when connecting to Redis or for a reply.
    # Without a timeout, an unreachable Redis can make calls HANG instead of
    # failing — freezing startup or the /ready check. Redis normally answers
    # in well under 1ms, so 2s only trips when something is genuinely wrong.
    redis_timeout: float = 2.0

    # ── Admin ────────────────────────────────────────────────────────────
    # Secret required (in the X-Admin-Key header) to call /admin endpoints.
    # No default on purpose: if it's not set, admin endpoints are DISABLED
    # ("fail closed") rather than open to anyone.
    #
    # SecretStr instead of str: printing/logging it shows '**********', so it
    # can't leak by accident. Reading it requires .get_secret_value().
    admin_api_key: SecretStr | None = None

    # ── Server ───────────────────────────────────────────────────────────
    host: str = "0.0.0.0"
    port: int = 8000


# Single global instance. Import this from anywhere in the app.
# Created once when the module is first imported, reused everywhere.
settings = Settings()
