"""
Tenant storage — creating tenants, minting API keys, looking them up.

REDIS LAYOUT (two keys per tenant):

    tenant:{tenant_id}   HASH   name, requests_per_minute, tokens_per_minute, created_at
    apikey:{sha256(key)} STRING tenant_id

  - A Redis HASH is a mini dict stored under one key: we can read/update
    single fields (e.g. just the limits) without rewriting the whole record.
  - The apikey → tenant_id pointer means a tenant could have several keys,
    and a key can be rotated without touching tenant data.
  - "tenant:" / "apikey:" prefixes namespace the keys. Redis has one flat
    keyspace; prefixes keep different kinds of data from colliding and make
    them easy to find (e.g. SCAN MATCH tenant:*).

WHY STORE A HASH OF THE KEY, NOT THE KEY?
If Redis (or a backup, or a debug dump) leaks, attackers get only hashes,
which can't be turned back into keys. To authenticate, we hash the key the
client sends and look THAT up — we never need the original.

WHY SHA-256 AND NOT bcrypt (used for passwords)?
bcrypt is deliberately SLOW to stop attackers guessing human passwords like
"summer2024" by brute force. Our keys are 256 bits of cryptographic
randomness — there's nothing to guess, so slowness buys no security. And auth
runs on EVERY request: SHA-256 costs ~1µs, bcrypt ~100ms.
"""

import hashlib
import secrets
import time
import uuid

from redis.asyncio import Redis

from app.schemas import Tenant, TenantCreate, TenantCreated

API_KEY_PREFIX = "gw_"


def generate_api_key() -> str:
    """Mint a new API key like 'gw_Xk3...'.

    `secrets` (not `random`): it uses the OS's cryptographically secure
    random source. `random` is predictable — given enough outputs, an
    attacker can reconstruct its state and predict future keys.

    token_urlsafe(32) = 32 random bytes (256 bits), base64-encoded so it's
    safe in headers and URLs.
    """
    return API_KEY_PREFIX + secrets.token_urlsafe(32)


def hash_api_key(api_key: str) -> str:
    """One-way hash of a key: same input → same output, but irreversible."""
    return hashlib.sha256(api_key.encode()).hexdigest()


def _tenant_key(tenant_id: str) -> str:
    return f"tenant:{tenant_id}"


def _apikey_key(key_hash: str) -> str:
    return f"apikey:{key_hash}"


async def create_tenant(redis: Redis, data: TenantCreate) -> TenantCreated:
    """Create a tenant and its first API key. Returns the raw key ONCE."""
    tenant = Tenant(
        id=f"tn_{uuid.uuid4().hex}",
        name=data.name,
        requests_per_minute=data.requests_per_minute,
        tokens_per_minute=data.tokens_per_minute,
        created_at=int(time.time()),
    )
    api_key = generate_api_key()

    # transaction=True wraps both writes in MULTI/EXEC: Redis applies them
    # together, as one step. Without it, a crash between the two writes
    # could leave a tenant with no key, or a key pointing at no tenant.
    # The pipeline also sends both commands in ONE network round trip.
    async with redis.pipeline(transaction=True) as pipe:
        pipe.hset(_tenant_key(tenant.id), mapping=tenant.model_dump(exclude={"id"}))
        pipe.set(_apikey_key(hash_api_key(api_key)), tenant.id)
        await pipe.execute()

    return TenantCreated(**tenant.model_dump(), api_key=api_key)


async def get_tenant(redis: Redis, tenant_id: str) -> Tenant | None:
    """Load a tenant by id, or None if it doesn't exist."""
    fields = await redis.hgetall(_tenant_key(tenant_id))
    if not fields:  # Redis returns {} for a missing hash, not an error
        return None
    # Redis stores everything as strings ("60"). Pydantic converts them back
    # to the declared types (int 60) as part of validation.
    return Tenant(id=tenant_id, **fields)


async def get_tenant_by_api_key(redis: Redis, api_key: str) -> Tenant | None:
    """Resolve a raw API key to its tenant, or None if the key is unknown.

    This is what authentication (next step) will call on every request:
    hash the presented key → look up the pointer → load the tenant.
    """
    tenant_id = await redis.get(_apikey_key(hash_api_key(api_key)))
    if tenant_id is None:
        return None
    return await get_tenant(redis, tenant_id)
