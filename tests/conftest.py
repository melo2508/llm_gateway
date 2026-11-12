"""
Shared pytest fixtures. pytest auto-discovers this file; any test can request
a fixture just by naming it as a function parameter.

WHAT IS A FIXTURE?
A function that builds something a test needs (setup), hands it over with
`yield`, then cleans up after the test finishes (teardown). Each test gets a
FRESH copy, so tests can't leak state into each other.
"""

from collections.abc import Callable, Iterator

import pytest
from fastapi.testclient import TestClient

from app.api.deps import get_provider
from app.main import app
from app.services.mock_llm import MockLLMProvider


@pytest.fixture
def fast_provider() -> MockLLMProvider:
    """A provider with zero latency and zero failures.

    The real defaults sleep 50–300ms per call. Across hundreds of tests that
    adds up to a slow suite people stop running. Tests about LOGIC shouldn't
    pay for simulated network time.
    """
    return MockLLMProvider(latency_min=0.0, latency_max=0.0, failure_rate=0.0)


@pytest.fixture
def client(fast_provider: MockLLMProvider) -> Iterator[TestClient]:
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
