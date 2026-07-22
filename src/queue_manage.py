"""Read/write bulk_queue.csv for user-added sources."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from urllib.parse import urlparse

from config import BULK_QUEUE_CSV, OUTPUT_DIR
from crawl import DEFAULT_MAX, crawl_url, is_crawlable

QUEUE_FIELDS = [
    "source",
    "url",
    "title",
    "year",
    "tier",
    "relevance",
    "downloadable",
    "notes",
    "identifier",
    "licenseurl",
    "blocked",
]


def load_queue() -> list[dict]:
    if not BULK_QUEUE_CSV.exists():
        return []
    return list(csv.DictReader(BULK_QUEUE_CSV.open(encoding="utf-8")))


def save_queue(rows: list[dict]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with BULK_QUEUE_CSV.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=QUEUE_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in QUEUE_FIELDS})
    # Refresh summary JSON used by landing
    summary = {
        "n_queue": len(rows),
        "n_downloadable": sum(1 for r in rows if r.get("downloadable") == "yes"),
        "n_stream": sum(1 for r in rows if r.get("downloadable") == "stream"),
        "n_seed": sum(1 for r in rows if str(r.get("tier", "")).startswith("A-seed")),
        "n_user": sum(1 for r in rows if r.get("tier") == "A-user"),
        "top_10": sorted(rows, key=lambda x: -int(float(x.get("relevance") or 0)))[:10],
        "items": sorted(rows, key=lambda x: -int(float(x.get("relevance") or 0))),
    }
    (OUTPUT_DIR / "bulk_queue_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (OUTPUT_DIR / "bulk_queue.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")


def guess_downloadable(url: str) -> str:
    u = url.lower()
    if is_crawlable(url):
        # Hub URLs are expanded on add; if left as-is mark as search/stream
        if "youtube.com/results" in u or "/search" in u or "list=" in u:
            return "search"
        return "stream"
    if any(
        x in u
        for x in (
            "youtube.com/watch",
            "youtu.be/",
            "archive.org/details/",
            "archive.org/download/",
            "upload.wikimedia.org/",
            "britishpathe.com/asset/",
        )
    ):
        return "yes"
    if "commons.wikimedia.org/wiki/file:" in u:
        return "yes"
    path = urlparse(url).path.lower()
    if any(
        path.endswith(ext)
        for ext in (".mp4", ".webm", ".ogv", ".mkv", ".avi", ".mov", ".m3u8")
    ):
        return "yes"
    if "youtube.com/results" in u or "/search" in u:
        return "search"
    return "stream"


def guess_source(url: str) -> str:
    u = url.lower()
    if "britishpathe.com" in u:
        return "British Pathé (user)"
    if "youtube.com" in u or "youtu.be" in u:
        return "YouTube (user)"
    if "archive.org" in u:
        return "Internet Archive (user)"
    if "wikimedia.org" in u or "commons.wikimedia" in u:
        return "Wikimedia Commons (user)"
    return "User link"


def _make_item(
    *,
    url: str,
    title: str,
    source: str,
    downloadable: str,
    year: str = "",
    identifier: str = "",
    notes: str = "Added from ShtetlFrames UI",
    relevance: str = "99",
) -> dict:
    return {
        "source": source,
        "url": url,
        "title": (title or "Untitled")[:200],
        "year": year or "",
        "tier": "A-user",
        "relevance": relevance,
        "downloadable": downloadable,
        "notes": notes,
        "identifier": identifier or "",
        "licenseurl": "",
        "blocked": "false",
    }


def add_source(url: str, title: str = "", crawl: bool = True, max_items: int | None = None) -> dict:
    url = (url or "").strip()
    if not url:
        return {"ok": False, "error": "url_required"}
    if not url.startswith("http://") and not url.startswith("https://"):
        return {"ok": False, "error": "url_must_be_http"}

    rows = load_queue()
    existing_urls = {(r.get("url") or "").strip() for r in rows}
    max_items = max_items if max_items is not None else DEFAULT_MAX

    # Multi-video hub → expand into downloadable rows
    if crawl and is_crawlable(url):
        result = crawl_url(url, max_items=max_items)
        if not result.get("ok"):
            return {"ok": False, "error": result.get("error") or "crawl_failed", "crawl": result}
        if result.get("skip_hub"):
            # Single IA details page — fall through to normal add
            pass
        elif result.get("entries"):
            added = []
            skipped = 0
            for e in result["entries"]:
                eu = (e.get("url") or "").strip()
                if not eu or eu in existing_urls:
                    skipped += 1
                    continue
                item = _make_item(
                    url=eu,
                    title=e.get("title") or eu,
                    source=e.get("source") or guess_source(eu),
                    downloadable=e.get("downloadable") or "yes",
                    year=e.get("year") or "",
                    identifier=e.get("identifier") or "",
                    notes=e.get("notes") or f"Crawled from {url}",
                    relevance="95",
                )
                rows.insert(0, item)
                existing_urls.add(eu)
                added.append(item)
            if not added and skipped:
                return {
                    "ok": False,
                    "error": "already_exists",
                    "crawled": True,
                    "n_added": 0,
                    "n_skipped": skipped,
                }
            if added:
                save_queue(rows)
                return {
                    "ok": True,
                    "crawled": True,
                    "kind": result.get("kind"),
                    "n_added": len(added),
                    "n_skipped": skipped,
                    "truncated": bool(result.get("truncated")),
                    "items": added[:20],
                    "n_queue": len(rows),
                    "source_url": url,
                }
            return {"ok": False, "error": "crawl_empty", "crawl": result}
        else:
            return {"ok": False, "error": "crawl_empty", "crawl": result}

    for r in rows:
        if (r.get("url") or "").strip() == url:
            return {"ok": False, "error": "already_exists", "item": r}

    title = (title or "").strip() or urlparse(url).path.rsplit("/", 1)[-1] or "User source"
    item = _make_item(
        url=url,
        title=title,
        source=guess_source(url),
        downloadable=guess_downloadable(url),
    )
    rows.insert(0, item)
    save_queue(rows)
    return {"ok": True, "crawled": False, "item": item, "n_queue": len(rows)}


def delete_source(url: str) -> dict:
    url = (url or "").strip()
    if not url:
        return {"ok": False, "error": "url_required"}
    rows = load_queue()
    kept = [r for r in rows if (r.get("url") or "").strip() != url]
    if len(kept) == len(rows):
        return {"ok": False, "error": "not_found"}
    save_queue(kept)
    return {"ok": True, "n_queue": len(kept)}


def merge_entries_into_queue(
    entries: list[dict],
    *,
    source: str | None = None,
    notes: str | None = None,
    tier: str = "B-discover",
    relevance: str = "80",
) -> dict:
    """Append crawl/discover entries; skip duplicate URLs. Returns counts + saved rows."""
    rows = load_queue()
    existing = {(r.get("url") or "").strip() for r in rows}
    added: list[dict] = []
    skipped = 0
    for e in entries:
        eu = (e.get("url") or "").strip()
        if not eu or eu in existing:
            skipped += 1
            continue
        item = _make_item(
            url=eu,
            title=e.get("title") or eu,
            source=source or e.get("source") or guess_source(eu),
            downloadable=e.get("downloadable") or "yes",
            year=e.get("year") or "",
            identifier=e.get("identifier") or "",
            notes=notes or e.get("notes") or "Discovered from archive Run",
            relevance=str(e.get("relevance") or relevance),
        )
        item["tier"] = tier
        rows.insert(0, item)
        existing.add(eu)
        added.append(item)
    if added:
        save_queue(rows)
    return {
        "ok": True,
        "n_added": len(added),
        "n_skipped": skipped,
        "n_queue": len(rows),
        "items": added,
    }
