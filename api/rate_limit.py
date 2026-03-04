"""Simple in-memory per-IP rate limiter using a sliding window."""
from __future__ import annotations

import os
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

# Number of trusted reverse proxies between the client and this server.
# With depth=1 (default: single proxy like Railway), we take the last
# X-Forwarded-For entry. With depth=2 (e.g. Cloudflare → Railway),
# we take the second-to-last entry.
_TRUSTED_PROXY_DEPTH = int(os.getenv("TRUSTED_PROXY_DEPTH", "1"))


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


def _extract_client_ip(request: Request) -> str:
    """Safely extract the real client IP from the request.

    PaaS load balancers (Railway, Heroku, etc.) append the real client IP
    to the *end* of the X-Forwarded-For chain.  Taking the first entry
    (index 0) trusts user-supplied headers and allows trivial IP spoofing.

    TRUSTED_PROXY_DEPTH controls how many proxy-appended entries to skip
    from the right:
    - depth=1 (default, single proxy): take [-1] (the proxy-appended IP)
    - depth=2 (CDN + proxy): take [-2] (CDN-appended IP, skipping proxy)
    """
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        parts = [p.strip() for p in forwarded.split(",")]
        # Pick the entry at position -(depth), clamped to the first entry
        idx = max(0, len(parts) - _TRUSTED_PROXY_DEPTH)
        return parts[idx]
    return request.client.host if request.client else "unknown"


def get_rate_limit_info(request: Request) -> dict:
    """Return current rate limit usage for the request IP without consuming a slot."""
    limit = config.RATE_LIMIT_PER_MINUTE
    window = 60.0
    now = time.time()
    cutoff = now - window

    ip = _extract_client_ip(request)

    with _lock:
        timestamps = _requests.get(ip, [])
        current = [t for t in timestamps if t > cutoff]
        return {"limit": limit, "remaining": max(0, limit - len(current)), "used": len(current)}


def rate_limit_dependency(request: Request) -> None:
    """FastAPI dependency that enforces the per-IP rate limit.

    Raises HTTP 429 when the caller exceeds RATE_LIMIT_PER_MINUTE requests
    within the last 60-second sliding window.
    """
    limit = config.RATE_LIMIT_PER_MINUTE
    window = 60.0  # seconds
    now = time.time()
    cutoff = now - window

    # Resolve client IP — use rightmost X-Forwarded-For entry (set by ingress proxy).
    ip = _extract_client_ip(request)

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
