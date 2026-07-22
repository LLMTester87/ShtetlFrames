from __future__ import annotations

from urllib.parse import urlparse

USER_AGENT = "ShtetlFramesCrawl/1.0 (research; respectful archival use)"
DEFAULT_MAX = 25

VIDEO_EXTS = {".mp4", ".webm", ".ogv", ".avi", ".mkv", ".mpg", ".mpeg", ".mov"}


def is_crawlable(url: str) -> bool:
    """True when a URL is a hub that should be expanded into many items."""
    u = (url or "").lower()
    if not u.startswith("http"):
        return False
    # Single watch / direct file — do not crawl
    if "youtu.be/" in u:
        return False
    if "youtube.com/watch" in u and "list=" not in u:
        return False
    if "upload.wikimedia.org/" in u:
        return False
    if "britishpathe.com/asset/" in u:
        return False
    path = urlparse(url).path.lower()
    if any(path.endswith(ext) for ext in VIDEO_EXTS):
        return False
    if path.endswith(".m3u8"):
        return False
    if "commons.wikimedia.org/wiki/file:" in u:
        return False

    # YouTube hubs
    if "youtube.com" in u:
        if any(
            x in u
            for x in (
                "/playlist",
                "list=",
                "/channel/",
                "/@",
                "/c/",
                "/user/",
                "/results",
            )
        ):
            return True

    # Internet Archive search / collections
    if "archive.org" in u:
        if "/search" in u or "query=" in u:
            return True
        if "/details/" in u:
            return True  # may be collection or multi-file; expand if needed

    # Commons search / category
    if "commons.wikimedia.org" in u:
        if "special:mediasearch" in u or "search=" in u:
            return True
        if "/wiki/category:" in u:
            return True

    return False


def _year_from(val) -> str:
    if not val:
        return ""
    s = str(val)
    if len(s) >= 4 and s[:4].isdigit():
        return s[:4]
    return ""
