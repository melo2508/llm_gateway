"""
Shared pytest fixtures. pytest auto-discovers this file; any test can request
a fixture just by naming it as a function parameter.

WHAT IS A FIXTURE?
A function that builds something a test needs (setup), hands it over with
`yield`, then cleans up after the test finishes (teardown). Each test gets a
FRESH copy, so tests can't leak state into each other.
"""

from collections.abc import AsyncIterator, Callable, Iterator

import pytest
import redis
from fastapi.testclient import TestClient
from pydantic import SecretStr
from redis.asyncio import Redis

from app.api.deps import get_provider
from app.core.config import settings
from app.core.redis import connect_redis
from app.main import app
from app.services.mock_llm import MockLLMProvider

# Redis ships with 16 numbered databases (0–15) on one server. The app uses
# DB 0; tests use DB 15, so wiping test data can never touch dev data.
TEST_REDIS_URL = "redis://localhost:6379/15"

# A port nothing listens on — connecting fails instantly ("connection refused").
UNREACHABLE_REDIS_URL = "redis://localhost:1/0"


@pytest.fixture
def anyio_backend() -> str:
    """Async tests (marked @pytest.mark.anyio) need an event loop to run in.
    The anyio pytest plugin (installed alongside FastAPI) provides one; this
    pins it to asyncio — the same loop uvicorn uses — instead of also
    running every async test a second time under trio."""
    return "asyncio"


@pytest.fixture
def test_redis_url(monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    """Point the app at the test database, starting from an empty state.

    - Skips the test (rather than failing) if Redis isn't running, so a
      missing local service reads as "not run", not "code is broken".
    - monkeypatch changes settings.redis_url for this test only and restores
      it automatically afterwards.
    - flushdb() wipes DB 15 before each test: every test starts clean, so
      no test depends on leftovers from another.
    """
    # A plain synchronous client is fine here: this is test setup, not app code.
    admin = redis.Redis.from_url(TEST_REDIS_URL, socket_connect_timeout=1)
    try:
        admin.ping()
    except redis.exceptions.ConnectionError:
        pytest.skip("Redis is not running on localhost:6379")
    admin.flushdb()
    monkeypatch.setattr(settings, "redis_url", TEST_REDIS_URL)
    yield TEST_REDIS_URL
    admin.flushdb()
    admin.close()


@pytest.fixture
async def redis_client(test_redis_url: str, anyio_backend: str) -> AsyncIterator[Redis]:
    """An async Redis client on the (clean) test DB, for service unit tests.

    An `async def` fixture: setup/teardown can `await`. The anyio plugin runs
    it in the same event loop as the async test that requests it.
    """
    client = await connect_redis(test_redis_url)
    yield client
    await client.aclose()


TEST_ADMIN_KEY = "test-admin-secret"


@pytest.fixture
def admin_headers(monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    """Configure an admin secret for this test and return matching headers."""
    monkeypatch.setattr(settings, "admin_api_key", SecretStr(TEST_ADMIN_KEY))
    return {"X-Admin-Key": TEST_ADMIN_KEY}


@pytest.fixture
def fast_provider() -> MockLLMProvider:
    """A provider with zero latency and zero failures.

    The real defaults sleep 50–300ms per call. Across hundreds of tests that
    adds up to a slow suite people stop running. Tests about LOGIC shouldn't
    pay for simulated network time.
    """
    return MockLLMProvider(latency_min=0.0, latency_max=0.0, failure_rate=0.0)


@pytest.fixture
def client(fast_provider: MockLLMProvider, test_redis_url: str) -> Iterator[TestClient]:
    """An HTTP client wired to our app, with the fast provider injected.

    WHY `with TestClient(app)`?
    The `with` block runs the app's lifespan (startup before, shutdown after),
    exactly like uvicorn would. Without it, startup never runs and
    `app.state.provider` wouldn't exist.

    WHY dependency_overrides?
    This is the payoff of `Depends(get_provider)`: we swap the provider for
    the whole app in one line — no env vars, no monkeypatching.
    """
    app.dependency_overrides[get_provider] = lambda: fast_provider
    with TestClient(app) as test_client:
        yield test_client
    # Teardown: `app` is a module-level global shared by every test, so an
    # override left behind would silently affect the NEXT test.
    app.dependency_overrides.clear()


@pytest.fixture
def use_provider() -> Callable[[MockLLMProvider], None]:
    """Let a test swap in its own provider (e.g. one that always fails).

    Usage inside a test:
        use_provider(MockLLMProvider(failure_rate=1.0, ...))
    """

    def _use(provider: MockLLMProvider) -> None:
        app.dependency_overrides[get_provider] = lambda: provider

    return _use
