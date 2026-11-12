"""
Tests for tenants: the storage service and the admin API.
"""

import pytest
from fastapi.testclient import TestClient
from redis.asyncio import Redis

from app.core.config import settings
from app.schemas import TenantCreate
from app.services import tenants

TENANTS_URL = "/admin/tenants"


# ─── Unit tests: app/services/tenants.py ────────────────────────────────────


def test_generated_keys_are_prefixed_and_unique() -> None:
    keys = {tenants.generate_api_key() for _ in range(100)}

    assert len(keys) == 100  # no duplicates
    assert all(k.startswith("gw_") for k in keys)


@pytest.mark.anyio
async def test_raw_api_key_is_never_stored(redis_client: Redis) -> None:
    created = await tenants.create_tenant(redis_client, TenantCreate(name="acme"))

    # Read EVERYTHING in the test DB and make sure the raw key appears nowhere.
    stored: list[str] = []
    async for key in redis_client.scan_iter():
        stored.append(key)
        if await redis_client.type(key) == "hash":
            stored.extend((await redis_client.hgetall(key)).values())
        else:
            stored.append(await redis_client.get(key))

    assert all(created.api_key not in item for item in stored)
    # ...but its hash IS there, as the lookup pointer.
    assert f"apikey:{tenants.hash_api_key(created.api_key)}" in stored


@pytest.mark.anyio
async def test_api_key_resolves_to_its_tenant(redis_client: Redis) -> None:
    created = await tenants.create_tenant(
        redis_client, TenantCreate(name="acme", requests_per_minute=5)
    )

    tenant = await tenants.get_tenant_by_api_key(redis_client, created.api_key)

    assert tenant is not None
    assert tenant.id == created.id
    # Stored as the string "5" in Redis; Pydantic turned it back into int 5.
    assert tenant.requests_per_minute == 5


@pytest.mark.anyio
async def test_unknown_api_key_resolves_to_none(redis_client: Redis) -> None:
    assert await tenants.get_tenant_by_api_key(redis_client, "gw_not-a-real-key") is None


# ─── API tests: /admin/tenants ──────────────────────────────────────────────


def test_create_tenant_returns_201_and_key_once(
    client: TestClient, admin_headers: dict[str, str]
) -> None:
    response = client.post(TENANTS_URL, json={"name": "acme"}, headers=admin_headers)

    assert response.status_code == 201
    data = response.json()
    assert data["api_key"].startswith("gw_")
    assert data["id"].startswith("tn_")
    # Defaults applied for limits we didn't send.
    assert data["requests_per_minute"] == 60
    assert data["tokens_per_minute"] == 10_000


def test_get_tenant_never_returns_api_key(
    client: TestClient, admin_headers: dict[str, str]
) -> None:
    created = client.post(TENANTS_URL, json={"name": "acme"}, headers=admin_headers).json()

    response = client.get(f"{TENANTS_URL}/{created['id']}", headers=admin_headers)

    assert response.status_code == 200
    assert response.json()["name"] == "acme"
    assert "api_key" not in response.json()


def test_get_unknown_tenant_returns_404(
    client: TestClient, admin_headers: dict[str, str]
) -> None:
    response = client.get(f"{TENANTS_URL}/tn_missing", headers=admin_headers)

    assert response.status_code == 404


def test_invalid_limits_are_rejected(
    client: TestClient, admin_headers: dict[str, str]
) -> None:
    body = {"name": "acme", "requests_per_minute": 0}

    response = client.post(TENANTS_URL, json=body, headers=admin_headers)

    assert response.status_code == 422


# ─── Admin auth ─────────────────────────────────────────────────────────────


def test_missing_admin_key_is_401(client: TestClient, admin_headers: dict[str, str]) -> None:
    # admin_headers fixture is requested so an admin secret IS configured;
    # we just don't send it.
    response = client.post(TENANTS_URL, json={"name": "acme"})

    assert response.status_code == 401


def test_wrong_admin_key_is_401(client: TestClient, admin_headers: dict[str, str]) -> None:
    response = client.post(
        TENANTS_URL, json={"name": "acme"}, headers={"X-Admin-Key": "wrong"}
    )

    assert response.status_code == 401


def test_admin_api_disabled_when_no_secret_configured(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "admin_api_key", None)

    response = client.post(
        TENANTS_URL, json={"name": "acme"}, headers={"X-Admin-Key": "anything"}
    )

    # Fail closed: no secret configured → nobody gets in, not everybody.
    assert response.status_code == 403
