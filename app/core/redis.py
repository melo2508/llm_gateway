"""
Redis connection management.

WHY REDIS?
The gateway needs SHARED, FAST state: tenants, API keys, rate-limit counters.
A Python dict lives inside one process — run 3 gateway copies behind a load
balancer and each has its own dict, so a "100 req/min" limit becomes 300.
Redis is one store every copy talks to, it's in-memory (sub-millisecond),
and it has atomic operations (INCR etc.) that rate limiting depends on.

WHY ONE CLIENT WITH A CONNECTION POOL?
Opening a TCP connection costs a network round trip (plus a TLS handshake in
production). The client keeps a POOL of already-open connections: each
command borrows one, uses it, and returns it. We create the client once at
startup (lifespan) and share it — never one connection per request.
"""

from redis.asyncio import Redis
from redis.exceptions import RedisError

from app.core.config import settings


def create_redis_client(url: str | None = None) -> Redis:
    """Build an async Redis client. Does NOT connect yet — the pool opens
    connections lazily, on the first command.

    `url=None` then falling back to settings (instead of
    `url: str = settings.redis_url`) avoids the default-argument trap: a
    default is evaluated ONCE at import time, so later changes to settings
    (e.g. tests pointing at a test database) would be ignored.
    """
    return Redis.from_url(
        url or settings.redis_url,
        # Return str instead of bytes. Redis stores raw bytes; without this,
        # every value comes back as b"..." and we'd .decode() everywhere.
        decode_responses=True,
        # Fail fast instead of hanging if Redis is unreachable or stalls.
        socket_connect_timeout=settings.redis_timeout,
        socket_timeout=settings.redis_timeout,
    )


async def connect_redis(url: str | None = None) -> Redis:
    """Create a client and PROVE Redis is reachable, or raise.

    This is the "fail fast" decision: because the client connects lazily, a
    wrong REDIS_URL would otherwise go unnoticed until the first real request.
    PING at startup surfaces it immediately, as a crash with a clear message,
    before the server accepts any traffic.
    """
    client = create_redis_client(url)
    try:
        await client.ping()
    except RedisError as exc:
        # Don't leak the half-built pool on the way out.
        await client.aclose()
        # Deliberately NOT including the URL: in production it often embeds a
        # password (redis://:secret@host), and error messages end up in logs.
        # redis-py's own message already names the host and port.
        raise RuntimeError(f"Cannot reach Redis: {exc}") from exc
    return client
