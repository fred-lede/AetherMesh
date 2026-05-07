"""
Shared HTTP client with connection pooling.
Provides a singleton session with HTTPAdapter for connection reuse.
"""

from __future__ import annotations

import logging
import os
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from typing import Any

from config.settings import settings

LOGGER = logging.getLogger("aiih.http")


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, str(default)).strip()
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


# Retry configuration
POOL_CONNECTIONS = _env_int("AIIH_HTTP_POOL_CONNECTIONS", 20)
POOL_MAXSIZE = _env_int("AIIH_HTTP_POOL_MAXSIZE", 10)
MAX_RETRIES = _env_int("AIIH_HTTP_MAX_RETRIES", 3)
RETRY_BACKOFF_FACTOR = float(os.getenv("AIIH_HTTP_RETRY_BACKOFF", "0.5"))


def _create_session() -> requests.Session:
    """Create a session with connection pooling and retry logic."""
    session = requests.Session()

    # Configure retry strategy
    retry_strategy = Retry(
        total=MAX_RETRIES,
        connect=MAX_RETRIES,
        read=MAX_RETRIES,
        redirect=2,
        status_forcelist={429, 500, 502, 503, 504},
        backoff_factor=RETRY_BACKOFF_FACTOR,
    )

    # Mount adapters for both http and https
    adapter = HTTPAdapter(
        pool_connections=POOL_CONNECTIONS,
        pool_maxsize=POOL_MAXSIZE,
        max_retries=retry_strategy,
    )
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    # Set default timeout if configured
    default_timeout = settings.request_timeout_s
    if default_timeout > 0:
        # Note: timeout must be set per-request, not global
        pass

    return session


# Singleton session instance
_session: requests.Session | None = None


def get_session() -> requests.Session:
    """Get or create the shared session singleton."""
    global _session
    if _session is None:
        LOGGER.info(
            f"Creating HTTP session: pool_connections={POOL_CONNECTIONS}, "
            f"pool_maxsize={POOL_MAXSIZE}, max_retries={MAX_RETRIES}"
        )
        _session = _create_session()
    return _session


def close_session() -> None:
    """Close the session and release connections."""
    global _session
    if _session is not None:
        LOGGER.info("Closing HTTP session")
        _session.close()
        _session = None


# Convenience function for common requests
def get(url: str, **kwargs: Any) -> requests.Response:
    """Send GET request using the pooled session."""
    return get_session().get(url, **kwargs)


def post(url: str, **kwargs: Any) -> requests.Response:
    """Send POST request using the pooled session."""
    return get_session().post(url, **kwargs)


def put(url: str, **kwargs: Any) -> requests.Response:
    """Send PUT request using the pooled session."""
    return get_session().put(url, **kwargs)


def delete(url: str, **kwargs: Any) -> requests.Response:
    """Send DELETE request using the pooled session."""
    return get_session().delete(url, **kwargs)