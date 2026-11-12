"""
Admin API — managing tenants. Not part of the OpenAI-compatible surface.

Every route here is protected by require_admin, applied ONCE at the router
level (`dependencies=[...]`) rather than repeated on each route. A new admin
route can't accidentally be left unprotected.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from redis.asyncio import Redis

from app.api.deps import get_redis, require_admin
from app.schemas import Tenant, TenantCreate, TenantCreated
from app.services import tenants

router = APIRouter(
    prefix="/admin",  # every path below starts with /admin
    tags=["admin"],
    dependencies=[Depends(require_admin)],
)


@router.post(
    "/tenants",
    response_model=TenantCreated,
    # 201 Created (not 200 OK): the standard status for "a new resource now exists".
    status_code=status.HTTP_201_CREATED,
)
async def create_tenant(
    body: TenantCreate, redis: Redis = Depends(get_redis)
) -> TenantCreated:
    """Create a tenant. The response contains the API key — shown only once."""
    return await tenants.create_tenant(redis, body)


@router.get("/tenants/{tenant_id}", response_model=Tenant)
async def get_tenant(tenant_id: str, redis: Redis = Depends(get_redis)) -> Tenant:
    """Fetch a tenant. Note response_model=Tenant: no api_key field exists here."""
    tenant = await tenants.get_tenant(redis, tenant_id)
    if tenant is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Tenant not found")
    return tenant
