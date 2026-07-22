from __future__ import annotations

import re
from urllib.parse import parse_qs, unquote, urlparse

import requests

from .common import USER_AGENT


def _crawl_commons(url: str, max_items: int, on_status=None) -> dict:
    def note(msg: str) -> None:
        if on_status:
            try:
                on_status(msg)
            except Exception:
                pass

    parsed = urlparse(url)
    path = unquote(parsed.path or "")
    qs = parse_qs(parsed.query)
    api = "https://commons.wikimedia.org/w/api.php"

    # Category members
    if re.search(r"/wiki/Category:", path, re.I):
        cat = path.split("/wiki/", 1)[-1]
        note(f"Commons category: {cat}")
        entries = _commons_category(api, cat, max_items)
        note(f"Commons category returned {len(entries):,}")
        return {
            "ok": True,
            "kind": "commons_category",
            "source_url": url,
            "entries": entries,
            "truncated": len(entries) >= max_items,
            "n_found": len(entries),
        }

    # MediaSearch / search=
    search = (qs.get("search") or qs.get("searchToken") or [""])[0]
    if not search and "search" in path.lower():
        search = (qs.get("search") or [""])[0]
    if not search:
        return {"ok": False, "error": "commons_missing_search", "entries": [], "source_url": url}

    q = search if "filetype" in search.lower() else f"filetype:video {search}"
    note(f"Commons search: {q[:80]}")
    entries = _commons_search(api, q, max_items)
    note(f"Commons search returned {len(entries):,}")
    return {
        "ok": True,
        "kind": "commons_search",
        "source_url": url,
        "entries": entries,
        "truncated": len(entries) >= max_items,
        "n_found": len(entries),
    }


def _commons_search(api: str, query: str, max_items: int) -> list[dict]:
    r = requests.get(
        api,
        params={
            "action": "query",
            "list": "search",
            "srsearch": query,
            "srnamespace": "6",  # File
            "srlimit": min(max_items, 50),
            "format": "json",
        },
        timeout=60,
        headers={"User-Agent": USER_AGENT},
    )
    r.raise_for_status()
    hits = r.json().get("query", {}).get("search", [])
    entries = []
    for h in hits:
        title = h.get("title") or ""
        if not title.lower().startswith("file:"):
            title = f"File:{title}"
        # Keep videos only by extension when possible
        low = title.lower()
        if not any(low.endswith(ext) for ext in (".webm", ".ogv", ".oggy", ".mp4", ".gif")):
            # Still include; resolve at download time
            if "video" not in low and not any(x in low for x in (".webm", ".ogv", ".mp4")):
                continue
        page = f"https://commons.wikimedia.org/wiki/{title.replace(' ', '_')}"
        entries.append(
            {
                "url": page,
                "title": title.replace("File:", "", 1)[:200],
                "year": "",
                "identifier": title,
                "source": "Wikimedia Commons (crawl)",
                "downloadable": "yes",
                "notes": "Crawled from Commons search",
            }
        )
        if len(entries) >= max_items:
            break
    return entries


def _commons_category(api: str, category: str, max_items: int) -> list[dict]:
    r = requests.get(
        api,
        params={
            "action": "query",
            "list": "categorymembers",
            "cmtitle": category.replace("_", " "),
            "cmnamespace": "6",
            "cmlimit": min(max_items, 50),
            "format": "json",
        },
        timeout=60,
        headers={"User-Agent": USER_AGENT},
    )
    r.raise_for_status()
    members = r.json().get("query", {}).get("categorymembers", [])
    entries = []
    for m in members:
        title = m.get("title") or ""
        low = title.lower()
        if not any(low.endswith(ext) for ext in (".webm", ".ogv", ".mp4", ".gif", ".avi")):
            continue
        page = f"https://commons.wikimedia.org/wiki/{title.replace(' ', '_')}"
        entries.append(
            {
                "url": page,
                "title": title.replace("File:", "", 1)[:200],
                "year": "",
                "identifier": title,
                "source": "Wikimedia Commons (crawl)",
                "downloadable": "yes",
                "notes": f"Crawled from {category}",
            }
        )
        if len(entries) >= max_items:
            break
    return entries
