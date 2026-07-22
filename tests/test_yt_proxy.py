"""Unit tests for Scrapfly / ScrapingDog YouTube proxy selection."""

import pytest

from yt_proxy import (
    configured_provider,
    fallback_proxy_provider,
    is_proxy_throttle_error,
    is_scrapfly_hard_fail,
    note_proxy_throttle,
    parse_retry_after_seconds,
    provider_proxy_url,
    proxy_configured,
    proxy_cooldown_remaining,
    proxy_needs_insecure_ssl,
    residential_proxy_url,
)


@pytest.fixture(autouse=True)
def _no_dotenv(monkeypatch):
    """Do not reload .env / UI settings over monkeypatched env vars."""
    monkeypatch.setattr("yt_proxy._load_env", lambda: None)


def _clear_proxy_env(monkeypatch) -> None:
    for k in (
        "PROXY_PROVIDER",
        "SCRAPFLY_API_KEY",
        "SCRAPFLY_KEY",
        "SCRAPFLY_PROXY_URL",
        "SCRAPINGDOG_API_KEY",
        "SCRAPING_DOG_API_KEY",
        "SCRAPINGDOG_KEY",
        "SCRAPINGDOG_PROXY_URL",
        "YT_PROXY_URL",
    ):
        monkeypatch.delenv(k, raising=False)


def test_none_when_empty(monkeypatch):
    _clear_proxy_env(monkeypatch)
    monkeypatch.setenv("PROXY_PROVIDER", "auto")
    assert configured_provider() == "none"
    assert proxy_configured() is False
    assert residential_proxy_url() is None


def test_scrapfly_preferred_in_auto(monkeypatch):
    _clear_proxy_env(monkeypatch)
    monkeypatch.setenv("PROXY_PROVIDER", "auto")
    monkeypatch.setenv("SCRAPFLY_API_KEY", "sf_test_key")
    monkeypatch.setenv("SCRAPINGDOG_API_KEY", "dog_key")
    assert configured_provider() == "scrapfly"
    url = residential_proxy_url()
    assert url is not None
    assert "proxy.scrapfly.io" in url
    assert "sf_test_key" in url
    assert proxy_needs_insecure_ssl() is True


def test_scrapingdog_when_selected(monkeypatch):
    _clear_proxy_env(monkeypatch)
    monkeypatch.setenv("PROXY_PROVIDER", "scrapingdog")
    monkeypatch.setenv("SCRAPINGDOG_API_KEY", "dog_key")
    monkeypatch.setenv("SCRAPFLY_API_KEY", "sf_ignored")
    url = residential_proxy_url()
    assert configured_provider() == "scrapingdog"
    assert url is not None
    assert "proxy.scrapingdog.com" in url
    assert proxy_needs_insecure_ssl() is False


def test_scrapingdog_auto_when_no_scrapfly(monkeypatch):
    _clear_proxy_env(monkeypatch)
    monkeypatch.setenv("PROXY_PROVIDER", "auto")
    monkeypatch.setenv("SCRAPINGDOG_API_KEY", "dog_key")
    assert configured_provider() == "scrapingdog"


def test_legacy_iproyal_falls_to_auto(monkeypatch):
    _clear_proxy_env(monkeypatch)
    monkeypatch.setenv("PROXY_PROVIDER", "iproyal")
    monkeypatch.setenv("SCRAPFLY_API_KEY", "sf_key")
    assert configured_provider() == "scrapfly"


def test_forced_provider_missing_falls_none(monkeypatch):
    _clear_proxy_env(monkeypatch)
    monkeypatch.setenv("PROXY_PROVIDER", "scrapfly")
    monkeypatch.setenv("SCRAPINGDOG_API_KEY", "dog_key")
    assert configured_provider() == "none"
    assert residential_proxy_url() is None


def test_parse_retry_after_seconds():
    assert parse_retry_after_seconds("HTTP Error 429; Retry-After: 42") == 42.0
    assert parse_retry_after_seconds("retry_after=15") == 15.0
    assert parse_retry_after_seconds("nope") is None


def test_note_proxy_throttle_sets_cooldown(monkeypatch):
    import yt_proxy as yp

    monkeypatch.setattr(yp, "_cooldown_until", 0.0)
    assert is_proxy_throttle_error("ERR::THROTTLE::MAX_REQUEST_RATE_EXCEEDED 429")
    waited = note_proxy_throttle("HTTP Error 429 Retry-After: 12")
    assert waited == 12.0
    assert proxy_cooldown_remaining() > 10.0


def test_fallback_scrapfly_to_scrapingdog(monkeypatch):
    _clear_proxy_env(monkeypatch)
    monkeypatch.setenv("SCRAPFLY_API_KEY", "sf")
    monkeypatch.setenv("SCRAPINGDOG_API_KEY", "dog")
    assert fallback_proxy_provider("scrapfly") == "scrapingdog"
    assert "proxy.scrapingdog.com" in (provider_proxy_url("scrapingdog") or "")
    assert is_scrapfly_hard_fail("HTTP Error 429 Retry-After: 30")
    assert is_scrapfly_hard_fail("CERTIFICATE_VERIFY_FAILED self-signed certificate")


def test_fallback_none_without_scrapingdog(monkeypatch):
    _clear_proxy_env(monkeypatch)
    monkeypatch.setenv("SCRAPFLY_API_KEY", "sf")
    assert fallback_proxy_provider("scrapfly") is None
