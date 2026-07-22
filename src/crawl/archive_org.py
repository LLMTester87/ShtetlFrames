from __future__ import annotations

from urllib.parse import parse_qs, urlparse

import requests

from .common import USER_AGENT, VIDEO_EXTS, _year_from


def _crawl_archive_org(url: str, max_items: int, on_status=None) -> dict:
    def note(msg: str) -> None:
        if on_status:
            try:
                on_status(msg)
            except Exception:
                pass

    parsed = urlparse(url)
    path = parsed.path or ""
    qs = parse_qs(parsed.query)

    # Search page → advancedsearch
    if "/search" in path or qs.get("query"):
        query = (qs.get("query") or [""])[0]
        if not query:
            return {"ok": False, "error": "ia_search_missing_query", "entries": [], "source_url": url}
        note(f"Internet Archive search: {query[:80]}")
        docs = _ia_search_all(query, max_items, on_status=on_status)
        entries = [_ia_doc_to_entry(d, url) for d in docs]
        entries = [e for e in entries if e]
        note(f"IA search returned {len(entries):,} video item(s)")
        return {
            "ok": True,
            "kind": "ia_search",
            "source_url": url,
            "entries": entries[:max_items],
            "truncated": len(docs) >= max_items,
            "n_found": len(entries),
        }

    # details/ITEM — collection children or multi-video files
    if "/details/" in path:
        identifier = path.rstrip("/").split("/")[-1].split("?")[0]
        if not identifier:
            return {"ok": False, "error": "ia_missing_identifier", "entries": [], "source_url": url}
        note(f"Fetching IA metadata for {identifier}…")
        meta = _ia_metadata(identifier)
        mediatype = (meta.get("metadata") or {}).get("mediatype") or ""
        if mediatype == "collection" or _ia_has_children(identifier):
            note(f"Listing collection:{identifier}…")
            docs = _ia_search_all(f"collection:{identifier}", max_items, on_status=on_status)
            entries = [_ia_doc_to_entry(d, url) for d in docs]
            entries = [e for e in entries if e]
            if entries:
                note(f"Collection yielded {len(entries):,} items")
                return {
                    "ok": True,
                    "kind": "ia_collection",
                    "source_url": url,
                    "entries": entries[:max_items],
                    "truncated": len(docs) >= max_items,
                    "n_found": len(entries),
                }
        # Multi-file item: one queue row per video file (direct download URL)
        files = meta.get("files") or []
        video_files = []
        for f in files:
            name = f.get("name") or ""
            ext = ("." + name.rsplit(".", 1)[-1].lower()) if "." in name else ""
            if ext in VIDEO_EXTS and not name.endswith(".thumbs") and "/." not in name:
                size = int(f.get("size") or 0)
                video_files.append((size, name, f.get("title") or name))
        if len(video_files) > 1:
            video_files.sort(reverse=True)
            entries = []
            for _, name, title in video_files[:max_items]:
                entries.append(
                    {
                        "url": f"https://archive.org/download/{identifier}/{name}",
                        "title": str(title)[:200],
                        "year": _year_from((meta.get("metadata") or {}).get("year") or ""),
                        "identifier": f"{identifier}/{name}",
                        "source": "Internet Archive (crawl)",
                        "downloadable": "yes",
                        "notes": f"File from IA item {identifier}",
                    }
                )
            note(f"Multi-file item — {len(entries):,} video files")
            return {
                "ok": True,
                "kind": "ia_multifile",
                "source_url": url,
                "entries": entries,
                "truncated": len(video_files) > max_items,
                "n_found": len(entries),
            }
        note("IA item is a single video — skipping hub expand")
        return {
            "ok": True,
            "kind": "ia_single",
            "source_url": url,
            "entries": [],
            "truncated": False,
            "n_found": 0,
            "skip_hub": True,
        }

    return {"ok": False, "error": "ia_url_not_supported", "entries": [], "source_url": url}


def _ia_metadata(identifier: str) -> dict:
    r = requests.get(
        f"https://archive.org/metadata/{identifier}",
        timeout=60,
        headers={"User-Agent": USER_AGENT},
    )
    r.raise_for_status()
    return r.json()


def _ia_has_children(identifier: str) -> bool:
    try:
        docs = _ia_search(f"collection:{identifier}", rows=1)
        return bool(docs)
    except Exception:
        return False


def _ia_search(query: str, rows: int = 25, page: int = 1) -> list[dict]:
    r = requests.get(
        "https://archive.org/advancedsearch.php",
        params=[
            ("q", query),
            ("fl[]", "identifier"),
            ("fl[]", "title"),
            ("fl[]", "year"),
            ("fl[]", "mediatype"),
            ("rows", str(min(rows, 100))),
            ("page", str(page)),
            ("output", "json"),
            ("sort[]", "downloads desc"),
        ],
        timeout=90,
        headers={"User-Agent": USER_AGENT},
    )
    r.raise_for_status()
    return r.json().get("response", {}).get("docs", [])


def _ia_search_all(query: str, max_items: int, on_status=None) -> list[dict]:
    """Paginate IA advancedsearch up to max_items."""
    out: list[dict] = []
    page = 1
    page_size = min(100, max_items)
    while len(out) < max_items:
        if on_status:
            try:
                on_status(f"IA page {page}… {len(out):,}/{max_items:,} so far")
            except Exception:
                pass
        docs = _ia_search(query, rows=page_size, page=page)
        if not docs:
            break
        out.extend(docs)
        if len(docs) < page_size:
            break
        page += 1
        if page > 50:
            break
    return out[:max_items]


def _ia_doc_to_entry(d: dict, source_url: str) -> dict | None:
    ident = d.get("identifier")
    if not ident:
        return None
    mt = d.get("mediatype") or ""
    if isinstance(mt, list):
        mt = mt[0] if mt else ""
    if mt and mt not in ("movies", "movie"):
        return None
    title = d.get("title") or ident
    if isinstance(title, list):
        title = title[0]
    year = d.get("year") or ""
    if isinstance(year, list):
        year = year[0]
    return {
        "url": f"https://archive.org/details/{ident}",
        "title": str(title)[:200],
        "year": str(year)[:8],
        "identifier": ident,
        "source": "Internet Archive (crawl)",
        "downloadable": "yes",
        "notes": f"Crawled from IA hub",
    }
