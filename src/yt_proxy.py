"""YouTube residential proxy: Scrapfly or ScrapingDog.

Select with PROXY_PROVIDER=scrapfly|scrapingdog|auto|none (UI Settings).
`auto` uses the first provider that has credentials configured.

Scrapfly rate limits: honor Retry-After — no new proxied jobs until it expires.
"""

from __future__ import annotations

import os
import re
import threading
import time
from typing import Callable, Literal
from urllib.parse import quote

Provider = Literal["scrapfly", "scrapingdog", "auto", "none"]

_AUTO_ORDER = ("scrapfly", "scrapingdog")

_DEFAULT_THROTTLE_SEC = 60.0
_MAX_THROTTLE_SEC = 300.0
# Scrapfly counts each proxied HTTP call — keep one Scrapfly job in flight from this PC.
_SCRAPFLY_MAX_INFLIGHT = 1

_gate = threading.Condition()
_cooldown_until = 0.0  # time.monotonic()
_inflight = 0


def _load_env() -> None:
    try:
        from config import load_env

        load_env()
    except Exception:
        pass


def configured_provider() -> Provider:
    """Resolved provider actually used for the next download (never 'auto')."""
    _load_env()
    raw = (os.environ.get("PROXY_PROVIDER") or "auto").strip().lower()
    # Legacy IPRoyal setting — treat as auto.
    if raw == "iproyal":
        raw = "auto"
    if raw in ("", "none", "off", "false", "0"):
        return "none"
    if raw in ("scrapfly", "scrapingdog"):
        if _provider_url(raw):  # type: ignore[arg-type]
            return raw  # type: ignore[return-value]
        return "none"
    # auto
    for name in _AUTO_ORDER:
        if _provider_url(name):  # type: ignore[arg-type]
            return name  # type: ignore[return-value]
    return "none"


def proxy_provider_name() -> str:
    """Short label for status lines (scrapfly / scrapingdog / none)."""
    return configured_provider()


def proxy_configured() -> bool:
    return configured_provider() != "none"


def residential_proxy_url() -> str | None:
    """HTTP proxy URL for yt-dlp. Never log the return value (contains secrets)."""
    prov = configured_provider()
    if prov == "none":
        return None
    return _provider_url(prov)


def provider_proxy_url(name: str) -> str | None:
    """URL for a specific provider (even if it is not the active auto pick)."""
    _load_env()
    key = (name or "").strip().lower()
    if key not in ("scrapfly", "scrapingdog"):
        return None
    return _provider_url(key)


def proxy_needs_insecure_ssl(provider: str | None = None) -> bool:
    """Scrapfly proxy mode terminates TLS; yt-dlp needs --no-check-certificates."""
    name = (provider or configured_provider()).strip().lower()
    return name == "scrapfly"


def fallback_proxy_provider(current: str) -> str | None:
    """Next provider after a failure — Scrapfly → ScrapingDog when credentials exist."""
    _load_env()
    cur = (current or "").strip().lower()
    if cur == "scrapfly" and _provider_url("scrapingdog"):
        return "scrapingdog"
    return None


def is_google_block_error(msg: str) -> bool:
    low = (msg or "").lower()
    markers = (
        "not a bot",
        "sign in to confirm",
        "confirm you're not a bot",
        "confirm you are not a bot",
        "login_required",
        "unsupported client",
        "could not complete the youtube request",
        "page needs to be reloaded",
    )
    return any(m in low for m in markers)


def is_proxy_throttle_error(msg: str) -> bool:
    """True for Scrapfly / HTTP rate-limit responses (Retry-After)."""
    low = (msg or "").lower()
    markers = (
        "retry-after",
        "retry after",
        "retry_after",
        "scrapfly_throttled",
        "max_request_rate",
        "max request rate",
        "err::throttle",
        "too many requests",
        "http error 429",
        "status code 429",
        "429 too many",
        "rate exceeded",
        "rate limit",
        "x-scrapfly-reject",
    )
    return any(m in low for m in markers)


def is_scrapfly_hard_fail(msg: str) -> bool:
    """Errors that warrant leaving Scrapfly for ScrapingDog."""
    low = (msg or "").lower()
    if is_proxy_throttle_error(low):
        return True
    markers = (
        "scrapfly",
        "certificate verify failed",
        "self-signed certificate",
        "x-scrapfly-reject",
        "err::throttle",
        "max_request_rate",
        "proxy connect aborted",
        "tunnel connection failed",
        "407 proxy",
        "proxy authentication",
    )
    return any(m in low for m in markers)


def parse_retry_after_seconds(msg: str | None) -> float | None:
    """Extract Retry-After delay (seconds) from headers/error text."""
    if not msg:
        return None
    patterns = (
        r"(?i)retry-after['\"\s:=]+(\d+(?:\.\d+)?)",
        r"(?i)retry after['\"\s:=]+(\d+(?:\.\d+)?)",
        r"(?i)retry_after[=:\s]+(\d+(?:\.\d+)?)",
        r"(?i)retry[_-]delay[=:\s]+(\d+(?:\.\d+)?)",
    )
    for pat in patterns:
        m = re.search(pat, msg)
        if not m:
            continue
        try:
            sec = float(m.group(1))
        except ValueError:
            continue
        if sec > 0:
            return min(sec, _MAX_THROTTLE_SEC)
    return None


def note_proxy_throttle(msg: str | None = None, *, retry_after: float | None = None) -> float:
    """Extend global cooldown from Retry-After (or default 60s). Returns wait seconds."""
    global _cooldown_until
    if retry_after is not None:
        sec = float(retry_after)
    else:
        parsed = parse_retry_after_seconds(msg)
        sec = float(parsed) if parsed is not None else _DEFAULT_THROTTLE_SEC
    if sec <= 0:
        return 0.0
    sec = max(1.0, min(sec, _MAX_THROTTLE_SEC))
    with _gate:
        _cooldown_until = max(_cooldown_until, time.monotonic() + sec)
        _gate.notify_all()
    return sec


def proxy_cooldown_remaining() -> float:
    with _gate:
        return max(0.0, _cooldown_until - time.monotonic())


def wait_proxy_ready(
    *,
    provider: str | None = None,
    on_wait: Callable[[float], None] | None = None,
) -> None:
    """Block until Scrapfly Retry-After cooldown has expired. No-op for other providers."""
    prov = (provider or configured_provider()).strip().lower()
    if prov != "scrapfly":
        return
    while True:
        left = proxy_cooldown_remaining()
        if left <= 0:
            return
        if on_wait:
            try:
                on_wait(left)
            except Exception:
                pass
        time.sleep(min(left, 5.0))


def try_acquire_scrapfly_slot() -> bool:
    """Non-blocking Scrapfly slot. False if throttled or another job holds it."""
    global _inflight
    with _gate:
        if _cooldown_until > time.monotonic():
            return False
        if _inflight >= _SCRAPFLY_MAX_INFLIGHT:
            return False
        _inflight += 1
        return True


def acquire_scrapfly_slot(
    *,
    on_wait: Callable[[float], None] | None = None,
    block: bool = True,
) -> bool:
    """Take the single Scrapfly in-flight slot.

    When ``block=False``, returns False immediately if throttled/busy (caller
    should fall back to ScrapingDog). Blocking mode waits for Retry-After.
    """
    global _inflight
    if not block:
        return try_acquire_scrapfly_slot()
    while True:
        with _gate:
            now = time.monotonic()
            left = _cooldown_until - now
            if left > 0:
                wait_s = left
            elif _inflight >= _SCRAPFLY_MAX_INFLIGHT:
                wait_s = 2.0
            else:
                _inflight += 1
                return True
        if on_wait:
            try:
                on_wait(max(0.0, wait_s))
            except Exception:
                pass
        time.sleep(min(max(wait_s, 0.2), 5.0))


def release_scrapfly_slot() -> None:
    global _inflight
    with _gate:
        _inflight = max(0, _inflight - 1)
        _gate.notify_all()


def _provider_url(name: str) -> str | None:
    if name == "scrapfly":
        return _scrapfly_url()
    if name == "scrapingdog":
        return _scrapingdog_url()
    return None


def _scrapfly_url() -> str | None:
    full = (os.environ.get("SCRAPFLY_PROXY_URL") or "").strip()
    if full:
        return full if "://" in full else f"http://{full}"
    key = (os.environ.get("SCRAPFLY_API_KEY") or os.environ.get("SCRAPFLY_KEY") or "").strip()
    if not key:
        return None
    # Proxy mode: options in username, API key as password.
    # renderJs MUST be false for yt-dlp media — default proxy-mode renderJs=true burns rate limits.
    country = (os.environ.get("SCRAPFLY_COUNTRY") or "us").strip().lower() or "us"
    opts = (
        os.environ.get("SCRAPFLY_PROXY_OPTS")
        or f"country-{country}-asp-true-renderJs-false-proxyPool-public_residential_pool"
    ).strip()
    host = (os.environ.get("SCRAPFLY_PROXY_HOST") or "proxy.scrapfly.io").strip()
    port = (os.environ.get("SCRAPFLY_PROXY_PORT") or "7777").strip()
    return f"http://{quote(opts, safe='-._')}:{quote(key, safe='')}@{host}:{port}"


def _scrapingdog_url() -> str | None:
    full = (os.environ.get("SCRAPINGDOG_PROXY_URL") or "").strip()
    if full:
        return full if "://" in full else f"http://{full}"
    key = (
        os.environ.get("SCRAPINGDOG_API_KEY")
        or os.environ.get("SCRAPING_DOG_API_KEY")
        or os.environ.get("SCRAPINGDOG_KEY")
        or ""
    ).strip()
    if not key:
        return None
    user = (os.environ.get("SCRAPINGDOG_PROXY_USER") or "scrapingdog").strip() or "scrapingdog"
    host = (os.environ.get("SCRAPINGDOG_PROXY_HOST") or "proxy.scrapingdog.com").strip()
    port = (os.environ.get("SCRAPINGDOG_PROXY_PORT") or "8081").strip()
    return f"http://{quote(user, safe='')}:{quote(key, safe='')}@{host}:{port}"
