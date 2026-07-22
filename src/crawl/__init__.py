"""Expand multi-video hub URLs into individual downloadable queue entries.

Supports:
  - YouTube playlist / channel / @handle / results (via yt-dlp flat listing)
  - Internet Archive search pages and collection items
  - Wikimedia Commons MediaSearch / category pages
"""

from __future__ import annotations

from urllib.parse import urlparse

from .archive_org import _crawl_archive_org
from .common import DEFAULT_MAX, is_crawlable
from .commons import _crawl_commons
from .youtube import _crawl_ytdlp, _normalize_youtube_hub

__all__ = ["DEFAULT_MAX", "crawl_url", "is_crawlable"]


def crawl_url(url: str, max_items: int = DEFAULT_MAX, on_status=None) -> dict:
    """
    Expand a hub URL into discrete video entries.

    on_status: optional callable(str) for live progress (UI + console).

    Returns:
      {
        ok, kind, source_url, entries: [{url, title, year, identifier, source, downloadable, notes}],
        truncated, error?
      }
    """
    url = (url or "").strip()
    # Safety ceiling; discover may pass up to 1e6
    max_items = max(1, min(int(max_items or DEFAULT_MAX), 1_000_000))
    if not url:
        return {"ok": False, "error": "url_required", "entries": []}

    def note(msg: str) -> None:
        if on_status:
            try:
                on_status(msg)
            except Exception:
                pass

    try:
        note(f"Detecting hub type for {urlparse(url).netloc}…")
        if "youtube.com" in url.lower() or "youtu.be" in url.lower():
            return _crawl_ytdlp(
                _normalize_youtube_hub(url), max_items, kind="youtube", on_status=on_status
            )
        if "archive.org" in url.lower():
            return _crawl_archive_org(url, max_items, on_status=on_status)
        if "commons.wikimedia.org" in url.lower() or "wikimedia.org" in url.lower():
            return _crawl_commons(url, max_items, on_status=on_status)
        return _crawl_ytdlp(url, max_items, kind="generic", on_status=on_status)
    except Exception as e:
        note(f"Crawl failed: {e}")
        return {"ok": False, "error": str(e), "entries": [], "source_url": url}
