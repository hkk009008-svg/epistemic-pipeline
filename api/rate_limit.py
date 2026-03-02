"""Simple in-memory per-IP rate limiter using a sliding window."""
from __future__ import annotations

import threading
import time
from collections import defaultdict

from fastapi import HTTPException, Request

import config

# Global state: {ip_address: [timestamp, timestamp, ...]}
_requests: dict[str, list[float]] = defaultdict(list)
_lock = threading.Lock()

# How often (in seconds) to purge expired entries from the dict.
_CLEANUP_INTERVAL = 60.0
_last_cleanup: float = 0.0


def _cleanup_expired(now: float, window: float) -> None:
    """Remove IPs whose timestamps are all older than the window."""
    global _last_cleanup
    if now - _last_cleanup < _CLEANUP_INTERVAL:
        return
    _last_cleanup = now
    expired_ips = [
        ip for ip, timestamps in _requests.items()
        if not timestamps or timestamps[-1] < now - window
    ]
    for ip in expired_ips:
        del _requests[ip]


def rate_limit_dependency(request: Request) -> None:
    """FastAPI dependency that enforces the per-IP rate limit.

    Raises HTTP 429 when the caller exceeds RATE_LIMIT_PER_MINUTE requests
    within the last 60-second sliding window.
    """
    limit = config.RATE_LIMIT_PER_MINUTE
    window = 60.0  # seconds
    now = time.time()
    cutoff = now - window

    # Resolve client IP (respect X-Forwarded-For behind a reverse proxy).
    forwarded = request.headers.get("x-forwarded-for", "").split(",")[0].strip()
    ip = forwarded or (request.client.host if request.client else "unknown")

    with _lock:
        # Prune timestamps outside the current window for this IP.
        timestamps = _requests[ip]
        _requests[ip] = [t for t in timestamps if t > cutoff]

        if len(_requests[ip]) >= limit:
            raise HTTPException(
                status_code=429,
                detail=(
                    f"Rate limit exceeded: {limit} requests per minute. "
                    "Please wait before making another request."
                ),
            )

        _requests[ip].append(now)

        # Periodically clean up IPs with no recent activity.
        _cleanup_expired(now, window)
