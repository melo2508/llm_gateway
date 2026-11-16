"""
Per-tenant rate limiting with a TOKEN BUCKET, run atomically inside Redis.

THE ALGORITHM
Each tenant has a bucket holding up to `limit` tokens. Tokens refill
CONTINUOUSLY at `limit / period` per second (60/min → one every second).
Each request removes `cost` tokens (1 for request limits; later, the LLM
token count for token limits). Not enough tokens → rejected (HTTP 429).

  - An idle tenant's bucket fills up, so it can burst up to `limit` at once,
    but never more — unlike a fixed window, which allows ~2x the limit around
    a window boundary.
  - Only two numbers per bucket: tokens left and last update time. Refill is
    computed lazily ("how much time passed since I last looked?") — no
    background job ticking every bucket.

WHY A LUA SCRIPT? (atomicity)
The logic is read → compute → write. Done as separate commands from Python,
two concurrent requests can both READ "1 token left", both decide "allowed",
and both WRITE — letting extra requests through. With several gateway
processes this isn't rare, it's guaranteed under load.

Redis executes commands one at a time, and a Lua script runs as ONE command:
nothing else can run in the middle of it. So read-compute-write becomes
indivisible, with no locks needed.

WHY REDIS'S CLOCK (TIME) AND NOT time.time() IN PYTHON?
Several gateway processes on different machines have slightly different
clocks. If each passed its own "now", a bucket's timeline would jump back
and forth. Redis's clock is the single source of truth for every caller.
"""

from dataclasses import dataclass

from redis.asyncio import Redis

# KEYS[1] = bucket key
# ARGV[1] = limit (bucket capacity), ARGV[2] = refill rate (tokens/second),
# ARGV[3] = cost of this request
TOKEN_BUCKET_LUA = """
local key = KEYS[1]
local limit = tonumber(ARGV[1])
local rate = tonumber(ARGV[2])
local cost = tonumber(ARGV[3])

-- Redis server time: {seconds, microseconds}
local t = redis.call('TIME')
local now = tonumber(t[1]) + tonumber(t[2]) / 1000000

-- Load bucket state. A missing bucket (new or expired tenant) starts full.
local state = redis.call('HMGET', key, 'tokens', 'ts')
local tokens = tonumber(state[1])
local ts = tonumber(state[2])
if tokens == nil then
  tokens = limit
  ts = now
end

-- Lazy refill: add what accumulated since the last update, capped at limit.
tokens = math.min(limit, tokens + math.max(0, now - ts) * rate)

local allowed = 0
local retry_after = 0
if tokens >= cost then
  tokens = tokens - cost
  allowed = 1
else
  -- How long until enough tokens have refilled to cover this cost.
  retry_after = (cost - tokens) / rate
end

redis.call('HSET', key, 'tokens', tokens, 'ts', now)

-- Seconds until the bucket is completely full again.
local reset_after = (limit - tokens) / rate

-- Housekeeping: once a bucket would be full again, its state is identical to
-- "no key at all" (a missing bucket starts full). So let Redis delete it then;
-- idle tenants cost no memory.
redis.call('EXPIRE', key, math.ceil(reset_after) + 1)

-- Lua numbers returned to Redis are TRUNCATED to integers (0.6 → 0).
-- Return fractional values as strings to keep their precision.
return {allowed, tostring(tokens), tostring(retry_after), tostring(reset_after)}
"""


@dataclass(frozen=True)
class RateLimitResult:
    """Outcome of one rate-limit check — everything the HTTP layer needs."""

    allowed: bool
    limit: int
    remaining: int  # whole tokens left after this request
    reset_after: float  # seconds until the bucket is full again
    retry_after: float  # seconds until this request would be allowed (0 if allowed)


class RateLimiter:
    """Token-bucket rate limiter backed by a Redis Lua script."""

    def __init__(self, redis: Redis):
        # register_script computes the script's SHA1 once. Calls then send
        # EVALSHA <sha> (40 bytes) instead of the whole script every time;
        # redis-py falls back to EVAL automatically if Redis hasn't seen the
        # script yet (e.g. after a Redis restart).
        self._script = redis.register_script(TOKEN_BUCKET_LUA)

    async def hit(
        self, key: str, limit: int, period_seconds: float = 60.0, cost: int = 1
    ) -> RateLimitResult:
        """Try to take `cost` tokens from the bucket at `key`.

        `limit` per `period_seconds` defines both the bucket size and the
        refill speed: 60 per 60s → capacity 60, refilling 1 token/second.
        """
        rate = limit / period_seconds
        allowed, tokens, retry_after, reset_after = await self._script(
            keys=[key], args=[limit, rate, cost]
        )
        return RateLimitResult(
            allowed=allowed == 1,
            limit=limit,
            remaining=int(float(tokens)),  # floor: 0.9 tokens can't serve a request
            reset_after=float(reset_after),
            retry_after=float(retry_after),
        )
