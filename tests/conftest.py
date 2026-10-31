import pytest

from jobagent import config


@pytest.fixture(autouse=True)
def no_waiting(monkeypatch):
    """Retries back off for 0 s in tests."""
    monkeypatch.setattr(config, "HTTP_BACKOFF", 0.0)
