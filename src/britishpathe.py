"""British Pathé — catalog discovery + free preview HLS resolve.

Dedicated Pathé UI uses this module. Main YouTube scrape is unchanged.

Asset HTML is Cloudflare-blocked from datacenters; Scrapfly ASP unlocks pages.
Once a playlist URL is known, yt-dlp downloads segments with a Referer (no YT proxy).
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable

from config import DATA_DIR, load_env

BP_ORIGIN = "https://www.britishpathe.com"
# Discover + many scrape workers share Scrapfly — cap concurrent asset HTML fetches.
_RESOLVE_SLOTS = threading.Semaphore(4)
_RESOLVE_MAX_ATTEMPTS = 3
ASSET_RE = re.compile(r"britishpathe\.com/asset/(\d+)", re.I)
M3U8_RE = re.compile(r"https://[^\s\"'<>]+\.m3u8", re.I)
ASSET_HREF_RE = re.compile(r"/asset/(\d+)/?", re.I)
ASSET_JSON_RE = re.compile(
    r'["\'](?:asset[_-]?id|id|post_id|mediaId)["\']\s*:\s*["\']?(\d{3,})',
    re.I,
)
TITLE_NEAR_ASSET_RE = re.compile(
    r'href=["\']/asset/(\d+)/?["\'][^>]*>\s*([^<]{2,160})',
    re.I,
)
# Listing cards: <a href="…/asset/36553" … aria-label="TUG MASTER">
CARD_ARIA_TITLE_RE = re.compile(
    r'href=["\'][^"\']*?/asset/(\d+)/?["\'][^>]*?aria-label=["\']([^"\']{2,160})["\']',
    re.I | re.S,
)
CARD_ARIA_TITLE_RE_ALT = re.compile(
    r'aria-label=["\']([^"\']{2,160})["\'][^>]*?href=["\'][^"\']*?/asset/(\d+)/?',
    re.I | re.S,
)
OG_TITLE_RE = re.compile(
    r'property=["\']og:title["\']\s+content=["\']([^"\']+)["\']',
    re.I,
)
OG_TITLE_RE_ALT = re.compile(
    r'content=["\']([^"\']+)["\']\s+property=["\']og:title["\']',
    re.I,
)
BP_TITLE_MARKERS = (
    "| british pathe",
    "| british pathé",
    "british pathe",
    "british pathé",
)

# Full archive span (newsreel era).
CATALOG_YEAR_START = 1896
CATALOG_YEAR_END = 1978
DEFAULT_PAGE_LIMIT = 48

_CACHE_PATH = DATA_DIR / "pathe_resolve_cache.json"
_CATALOG_PATH = DATA_DIR / "pathe_catalog.jsonl"
_cache_lock = threading.Lock()
_cache: dict[str, Any] | None = None

OnStatus = Callable[[str], None]


def is_britishpathe_url(url: str) -> bool:
    return "britishpathe.com" in (url or "").lower()


def is_britishpathe_asset_url(url: str) -> bool:
    return bool(ASSET_RE.search(url or ""))


def is_britishpathe_title(title: str) -> bool:
    low = (title or "").lower()
    return any(m in low for m in BP_TITLE_MARKERS)


def asset_id_from_url(url: str) -> str | None:
    m = ASSET_RE.search(url or "")
    return m.group(1) if m else None


def asset_page_url(asset_id: str | int) -> str:
    return f"{BP_ORIGIN}/asset/{int(asset_id)}/"


def normalize_asset_url(url: str) -> str:
    """Canonical asset URL so trailing-slash / host variants don't duplicate."""
    raw = (url or "").strip()
    if not raw:
        return ""
    aid = asset_id_from_url(raw)
    if aid:
        return asset_page_url(aid)
    return raw


def extract_m3u8(html: str) -> str | None:
    if not html:
        return None
    preferred = re.findall(
        r"https://www\.britishpathe\.com/fe-cdn/[^\s\"'<>]+playlist\.m3u8",
        html,
        re.I,
    )
    if preferred:
        return preferred[0]
    m = M3U8_RE.search(html)
    return m.group(0) if m else None


def extract_og_title(html: str) -> str:
    m = OG_TITLE_RE.search(html or "") or OG_TITLE_RE_ALT.search(html or "")
    return (m.group(1).strip() if m else "") or ""


def _normalize_search_query(title: str) -> str:
    t = (title or "").strip()
    for sep in ("|", "–", "—"):
        if sep in t:
            t = t.split(sep)[0].strip()
    t = re.sub(r"\s*\(\d{4}\)\s*$", "", t).strip()
    t = re.sub(r"\s+", " ", t)
    return t[:120]


def _load_cache() -> dict[str, Any]:
    global _cache
    with _cache_lock:
        if _cache is not None:
            return _cache
        if _CACHE_PATH.is_file():
            try:
                _cache = json.loads(_CACHE_PATH.read_text(encoding="utf-8"))
                if not isinstance(_cache, dict):
                    _cache = {}
            except Exception:
                _cache = {}
        else:
            _cache = {}
        return _cache


def _save_cache() -> None:
    with _cache_lock:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        tmp = _CACHE_PATH.with_suffix(".tmp")
        tmp.write_text(json.dumps(_cache or {}, indent=2), encoding="utf-8")
        tmp.replace(_CACHE_PATH)


def _scrapfly_api_key() -> str:
    load_env()
    return (
        os.environ.get("SCRAPFLY_API_KEY") or os.environ.get("SCRAPFLY_KEY") or ""
    ).strip()


def scrapfly_fetch_html(
    url: str,
    *,
    render_js: bool = True,
    rendering_wait: int | None = None,
    auto_scroll: bool = False,
) -> str:
    """Fetch HTML through Scrapfly ASP (Cloudflare bypass). Raises on failure."""
    key = _scrapfly_api_key()
    if not key:
        raise RuntimeError("SCRAPFLY_API_KEY required for British Pathé pages")
    params: dict[str, str] = {
        "key": key,
        "url": url,
        "asp": "true",
        "country": (os.environ.get("SCRAPFLY_COUNTRY") or "us").strip() or "us",
        "render_js": "true" if render_js else "false",
    }
    if render_js:
        wait = 3500 if rendering_wait is None else max(0, int(rendering_wait))
        params["rendering_wait"] = str(wait)
    if auto_scroll and render_js:
        params["auto_scroll"] = "true"
    api = "https://api.scrapfly.io/scrape?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(
        api,
        headers={"User-Agent": "ShtetlFrames/1.0 (britishpathe)"},
    )
    # Search pages need long JS wait — allow up to 3 minutes.
    timeout = 180 if (render_js and (rendering_wait or 0) >= 8000) else 120
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8", "replace"))
    result = data.get("result") or {}
    if not result.get("success"):
        reason = (
            result.get("error")
            or result.get("reason")
            or data.get("message")
            or "scrape_failed"
        )
        raise RuntimeError(f"scrapfly_pathe_page: {reason}")
    status = int(result.get("status_code") or 0)
    html = result.get("content") or ""
    if status >= 400 or not html:
        raise RuntimeError(f"scrapfly_pathe_page_http_{status}")
    return html


def resolve_asset(
    url_or_id: str,
    *,
    force: bool = False,
) -> dict[str, Any]:
    """Resolve asset id/URL → {asset_id, asset_url, m3u8_url, title}."""
    raw = (url_or_id or "").strip()
    if not raw:
        raise ValueError("empty_asset")
    aid = asset_id_from_url(raw) if "://" in raw else raw
    if not aid or not str(aid).isdigit():
        raise ValueError(f"not_a_pathe_asset: {raw[:80]}")
    aid = str(int(aid))
    asset_url = asset_page_url(aid)
    cache = _load_cache()
    cached = cache.get(aid)
    if (
        not force
        and isinstance(cached, dict)
        and cached.get("m3u8_url")
        and str(cached.get("m3u8_url")).endswith(".m3u8")
    ):
        return {
            "asset_id": aid,
            "asset_url": asset_url,
            "m3u8_url": cached["m3u8_url"],
            "title": cached.get("title") or "",
            "cached": True,
        }

    last_err: Exception | None = None
    for attempt in range(1, _RESOLVE_MAX_ATTEMPTS + 1):
        with _RESOLVE_SLOTS:
            try:
                # Playlist URL is usually in static HTML — skip JS render (faster).
                html = scrapfly_fetch_html(asset_url, render_js=False)
                m3u8 = extract_m3u8(html)
                if not m3u8:
                    html = scrapfly_fetch_html(
                        asset_url, render_js=True, rendering_wait=2000
                    )
                    m3u8 = extract_m3u8(html)
                if not m3u8:
                    raise RuntimeError(f"pathe_no_m3u8 asset={aid}")
                title = extract_og_title(html)
                entry = {
                    "asset_id": aid,
                    "asset_url": asset_url,
                    "m3u8_url": m3u8,
                    "title": title,
                }
                with _cache_lock:
                    cache[aid] = {
                        "m3u8_url": m3u8,
                        "title": title,
                        "asset_url": asset_url,
                    }
                _save_cache()
                entry["cached"] = False
                return entry
            except Exception as e:
                last_err = e
                if attempt < _RESOLVE_MAX_ATTEMPTS:
                    time.sleep(min(12.0, 1.5 * attempt))
    raise RuntimeError(
        f"pathe_resolve_failed asset={aid}: {last_err}"
    ) from last_err


def _clean_title(raw: str) -> str:
    t = re.sub(r"\s+", " ", (raw or "").strip())
    # Skip template leftovers / chrome labels.
    if not t or "{{" in t or t.lower() in ("british pathé", "british pathe"):
        return ""
    return t[:200]


def parse_assets_from_html(html: str, *, year: str = "") -> list[dict[str, str]]:
    """Extract discoverable asset entries from any Pathé HTML page."""
    if not html:
        return []
    titles: dict[str, str] = {}
    for aid, title in CARD_ARIA_TITLE_RE.findall(html):
        t = _clean_title(title)
        if t and aid not in titles:
            titles[aid] = t
    for title, aid in CARD_ARIA_TITLE_RE_ALT.findall(html):
        t = _clean_title(title)
        if t and aid not in titles:
            titles[aid] = t
    for aid, title in TITLE_NEAR_ASSET_RE.findall(html):
        t = _clean_title(title)
        if t and aid not in titles:
            titles[aid] = t

    ids: list[str] = []
    seen: set[str] = set()
    for aid in ASSET_HREF_RE.findall(html):
        if aid not in seen:
            seen.add(aid)
            ids.append(aid)
    for aid in ASSET_JSON_RE.findall(html):
        if aid not in seen and len(aid) >= 3:
            seen.add(aid)
            ids.append(aid)

    out: list[dict[str, str]] = []
    for aid in ids:
        out.append(
            {
                "url": asset_page_url(aid),  # canonical — unique per asset id
                "title": titles.get(aid) or f"Asset {aid}",
                "year": year or "",
                "identifier": aid,
                "source": "British Pathé",
                "downloadable": "yes",
            }
        )
    return out


def search_url(
    query: str = "",
    *,
    page: int | None = 1,
    year: int | str | None = None,
    start: str | int = "+",
    end: str | int = "+",
    limit: int = DEFAULT_PAGE_LIMIT,
) -> str:
    """Pathé browse/search URL that returns real /asset/ links after JS render.

    Matches the site's own listing:
    ``/search/?searchQuery=&page=null&refined[]=&selection=``
    """
    del limit, start, end  # unused; kept for call-site compat
    q = (query or "").strip()
    if year is not None and str(year).isdigit():
        # Optional year hint in the query box (site may treat as text search).
        y = str(int(year))
        q = f"{q} {y}".strip() if q else y
    # page=null is what the site uses for the first listing page.
    page_val: str
    if page is None or page == 0 or page == 1:
        page_val = "null"
    else:
        page_val = str(max(1, int(page)))
    # Build manually so refined[] stays as refined[]= (empty array param).
    qs = urllib.parse.urlencode(
        {"searchQuery": q, "page": page_val, "selection": ""},
        doseq=True,
    )
    return f"{BP_ORIGIN}/search/?{qs}&refined[]="


def fetch_search_html(url: str, *, rendering_wait: int | None = None) -> str:
    """Search listing needs JS; links usually appear without scroll."""
    if rendering_wait is None:
        rendering_wait = int(os.environ.get("PATHE_SEARCH_RENDER_WAIT_MS") or "4000")
    return scrapfly_fetch_html(
        url,
        render_js=True,
        rendering_wait=max(500, int(rendering_wait)),
        auto_scroll=False,
    )


def fetch_search_assets(
    url: str,
    *,
    remote_base: str | None = None,
) -> list[dict[str, str]]:
    """Fetch a listing page and parse assets, retrying when JS/ASP returns empty.

    When ``remote_base`` is set (dedicated discover pod), Scrapfly runs on that
    pod via ``POST /pathe_list`` so local scrape resolve is not starved.
    """
    attempts = (
        (4000, False),
        (8000, False),
        (10000, True),
    )
    last_err: Exception | None = None
    use_remote = bool(remote_base)
    for wait, scroll in attempts:
        try:
            if use_remote:
                from runpod_client import fetch_pathe_list_html_remote

                try:
                    html = fetch_pathe_list_html_remote(
                        url,
                        base=remote_base,
                        rendering_wait=wait,
                        auto_scroll=scroll,
                    )
                except Exception as e:
                    # Pod missing /pathe_list or Scrapfly 502 — fall back to local.
                    err = str(e).lower()
                    if (
                        "pathe_list_not_on_pod" in err
                        or "http_404" in err
                        or "http_502" in err
                        or "http_503" in err
                        or "pathe_list_bad_json" in err
                        or "pathe_list_empty" in err
                    ):
                        use_remote = False
                        html = scrapfly_fetch_html(
                            url,
                            render_js=True,
                            rendering_wait=wait,
                            auto_scroll=scroll,
                        )
                    else:
                        raise
            else:
                html = scrapfly_fetch_html(
                    url,
                    render_js=True,
                    rendering_wait=wait,
                    auto_scroll=scroll,
                )
        except Exception as e:
            last_err = e
            continue
        batch = parse_assets_from_html(html)
        if batch:
            return batch
    if last_err is not None:
        raise last_err
    return []


def _discover_concurrency() -> int:
    # Keep low — Scrapfly ASP + Pathé JS choke when many pages hit at once.
    try:
        n = int(os.environ.get("PATHE_DISCOVER_CONCURRENCY") or "4")
    except (TypeError, ValueError):
        n = 2
    return max(1, min(n, 6))


def search_first_asset(query: str, *, limit: int = 8) -> str | None:
    del limit
    q = _normalize_search_query(query)
    if len(q) < 3:
        return None
    entries = fetch_search_assets(search_url(q, page=1))
    return entries[0]["url"] if entries else None


def map_youtube_title_to_asset(title: str) -> str | None:
    """Map a YouTube title to a Pathé /asset/ URL via site search."""
    t = (title or "").strip()
    if not t:
        return None
    # Prefer Pathé-marked uploads, but allow plain @britishpathe titles too.
    try:
        return search_first_asset(t)
    except Exception:
        return None


def load_local_catalog(*, max_items: int = 5000) -> list[dict[str, str]]:
    """Load unique asset entries previously saved to ``pathe_catalog.jsonl``."""
    max_items = max(1, min(int(max_items), 1_000_000))
    if not _CATALOG_PATH.is_file():
        return []
    found: dict[str, dict[str, str]] = {}
    try:
        with _CATALOG_PATH.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                if not isinstance(row, dict):
                    continue
                aid = str(row.get("identifier") or asset_id_from_url(row.get("url") or "") or "")
                if not aid or not aid.isdigit() or aid in found:
                    continue
                url = normalize_asset_url(row.get("url") or "") or asset_page_url(aid)
                title = _clean_title(str(row.get("title") or "")) or f"British Pathé asset {aid}"
                found[aid] = {
                    "url": url,
                    "title": title,
                    "year": str(row.get("year") or ""),
                    "identifier": aid,
                    "source": "British Pathé",
                    "downloadable": "yes",
                }
                if len(found) >= max_items:
                    break
    except Exception:
        return list(found.values())
    return list(found.values())


def discover_from_youtube_titles(
    titles: list[str],
    *,
    max_items: int = 500,
    on_status: OnStatus | None = None,
    on_batch: Callable[[list[dict[str, str]]], None] | None = None,
    concurrency: int | None = None,
) -> dict[str, Any]:
    """Resolve Pathé /asset/ URLs by searching each YouTube title by name.

    Used when catalog listing pages return blank (Scrapfly/ASP). Each title is
    searched on britishpathe.com; first hit wins.
    """
    max_items = max(1, min(int(max_items), 50_000))
    try:
        workers = int(concurrency if concurrency is not None else (_discover_concurrency()))
    except (TypeError, ValueError):
        workers = 2
    workers = max(1, min(workers, 4))

    # Dedupe / normalize queries; keep original title for the queue row.
    queries: list[tuple[str, str]] = []
    seen_q: set[str] = set()
    for raw in titles:
        title = (raw or "").strip()
        q = _normalize_search_query(title)
        if len(q) < 3:
            continue
        key = q.casefold()
        if key in seen_q:
            continue
        seen_q.add(key)
        queries.append((q, title))
        if len(queries) >= max_items * 3:
            # Cap probe set; we stop once max_items assets are found.
            break

    def note(msg: str) -> None:
        if on_status:
            try:
                on_status(msg)
            except Exception:
                pass

    found: dict[str, dict[str, str]] = {}
    errors: list[str] = []
    probed = 0

    def _one(item: tuple[str, str]) -> tuple[str, dict[str, str] | None, str | None]:
        q, title = item
        try:
            entries = fetch_search_assets(search_url(q, page=1))
        except Exception as e:
            return q, None, str(e)[:160]
        if not entries:
            return q, None, None
        e0 = dict(entries[0])
        # Prefer the YouTube title when the listing only has a placeholder.
        t = _clean_title(e0.get("title") or "") or _clean_title(title) or e0.get("title") or ""
        e0["title"] = t
        e0["source"] = "British Pathé"
        e0["downloadable"] = "yes"
        return q, e0, None

    note(f"Pathé discover via YouTube titles · {len(queries):,} names · {workers}×…")
    # Chunk so we can stop early without queuing tens of thousands of Scrapfly jobs.
    chunk = max(workers * 8, 16)
    for i in range(0, len(queries), chunk):
        if len(found) >= max_items:
            break
        batch_q = queries[i : i + chunk]
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = [pool.submit(_one, item) for item in batch_q]
            for fut in as_completed(futs):
                if len(found) >= max_items:
                    for pending in futs:
                        pending.cancel()
                    break
                probed += 1
                q, entry, err = fut.result()
                if err:
                    errors.append(f"{q[:40]}: {err}")
                    continue
                if not entry:
                    continue
                aid = str(
                    entry.get("identifier")
                    or asset_id_from_url(entry.get("url") or "")
                    or ""
                )
                if not aid or aid in found:
                    continue
                entry["identifier"] = aid
                entry["url"] = (
                    normalize_asset_url(entry.get("url") or "") or asset_page_url(aid)
                )
                found[aid] = entry
                emit = [entry]
                _append_catalog(emit)
                if on_batch:
                    try:
                        on_batch(emit)
                    except Exception:
                        pass
                if probed % 10 == 0 or len(found) >= max_items:
                    note(
                        f"Pathé via YT titles · {len(found):,}/{max_items:,} urls · "
                        f"probed {probed:,}/{len(queries):,}…"
                    )

    entries = list(found.values())[:max_items]
    note(f"Pathé YouTube-title discover done · {len(entries):,} asset URLs")
    return {
        "ok": True,
        "entries": entries,
        "n": len(entries),
        "pages_fetched": probed,
        "years": "youtube_titles",
        "query": "youtube_titles",
        "errors": errors[:12],
        "truncated": len(entries) >= max_items,
        "concurrency": workers,
        "source": "youtube_titles",
    }


def prepare_pathe_job(
    url: str,
    title: str = "",
    *,
    on_status: OnStatus | None = None,
) -> dict[str, Any] | None:
    """Resolve a britishpathe.com/asset URL to HLS download fields (asset pages only)."""
    url = (url or "").strip()
    title = (title or "").strip()

    def _note(msg: str) -> None:
        if on_status:
            try:
                on_status(msg)
            except Exception:
                pass

    if not is_britishpathe_asset_url(url) and not (
        is_britishpathe_url(url) and asset_id_from_url(url)
    ):
        return None

    aid = asset_id_from_url(url)
    asset_url = asset_page_url(aid or "0")
    _note("Resolving British Pathé preview HLS…")
    try:
        resolved = resolve_asset(asset_url)
    except Exception as e:
        _note(f"Pathé resolve failed: {e}"[:160])
        raise RuntimeError(f"britishpathe_resolve_failed: {e}") from e

    return {
        "source": "britishpathe",
        "asset_id": resolved["asset_id"],
        "asset_url": resolved["asset_url"],
        "m3u8_url": resolved["m3u8_url"],
        "referer": resolved["asset_url"],
        "title": resolved.get("title") or title,
        "download_url": resolved["m3u8_url"],
        "cached": bool(resolved.get("cached")),
    }


def _append_catalog(entries: list[dict[str, str]]) -> None:
    if not entries:
        return
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with _CATALOG_PATH.open("a", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")


def discover_catalog(
    *,
    query: str = "",
    year_start: int | None = None,
    year_end: int | None = None,
    max_items: int = 5000,
    on_status: OnStatus | None = None,
    on_batch: Callable[[list[dict[str, str]]], None] | None = None,
    on_window: Callable[[dict[str, Any]], bool | None] | None = None,
    remote_base: str | None = None,
) -> dict[str, Any]:
    """Discover Pathé assets by paginating the site search listing.

    Uses the same URL shape as the website browse-all view::

        /search/?searchQuery=&page=null&refined[]=&selection=

    Empty ``query`` = full catalog. Keyword filters via ``searchQuery``.
    Fetches several pages in parallel (``PATHE_DISCOVER_CONCURRENCY``, default 4).
    ``on_batch`` is called with newly found entries as pages complete.
    ``on_window`` may return ``False`` to stop early (e.g. all-dupe windows).
    When ``remote_base`` is set, listing pages are fetched on that RunPod.
    """
    del year_start, year_end  # full listing URL does not use year facets
    max_items = max(1, min(int(max_items), 1_000_000))
    q = (query or "").strip()
    concurrency = _discover_concurrency()
    remote = (remote_base or "").rstrip("/") or None

    def note(msg: str) -> None:
        if on_status:
            try:
                on_status(msg)
            except Exception:
                pass

    def emit_batch(rows: list[dict[str, str]]) -> None:
        if not rows:
            return
        _append_catalog(rows)
        if on_batch:
            try:
                on_batch(rows)
            except Exception:
                pass

    found: dict[str, dict[str, str]] = {}
    pages_fetched = 0
    errors: list[str] = []
    empty_windows = 0
    page = 1
    stopped_early = False
    label = f"q:{q[:40]}" if q else "all"
    where = "pod" if remote else "local"
    note(f"Pathé discover via {where}" + (f" · {remote.split('//')[-1][:32]}" if remote else ""))

    def _fetch_page(p: int) -> tuple[int, list[dict[str, str]] | None, str | None]:
        url = search_url(q, page=p)
        try:
            return p, fetch_search_assets(url, remote_base=remote), None
        except Exception as e:
            return p, None, str(e)[:160]

    while len(found) < max_items and empty_windows < 3 and page <= 50_000:
        window = list(range(page, min(page + concurrency, 50_001)))
        note(
            f"Pathé discover · {label} · pages {window[0]}–{window[-1]} "
            f"×{concurrency} · {len(found):,}/{max_items:,} urls…"
        )
        results: list[tuple[int, list[dict[str, str]] | None, str | None]] = []
        with ThreadPoolExecutor(max_workers=len(window)) as pool:
            futs = [pool.submit(_fetch_page, p) for p in window]
            for fut in as_completed(futs):
                results.append(fut.result())
        results.sort(key=lambda r: r[0])

        new_in_window = 0
        for p, batch, err in results:
            if len(found) >= max_items:
                break
            if err is not None:
                errors.append(f"p{p}: {err}")
                continue
            pages_fetched += 1
            assert batch is not None
            fresh: list[dict[str, str]] = []
            for e in batch:
                aid = e.get("identifier") or ""
                if not aid or aid in found:
                    continue
                found[aid] = e
                fresh.append(e)
                if len(found) >= max_items:
                    break
            if fresh:
                new_in_window += len(fresh)
                emit_batch(fresh)

        if new_in_window == 0:
            empty_windows += 1
            # Soft backoff when Scrapfly/Pathé return blank under load.
            note(
                f"Pathé discover · empty window ({empty_windows}/3) · "
                f"retrying slower…"
            )
            time.sleep(1.5 * empty_windows)
        else:
            empty_windows = 0

        if on_window is not None:
            try:
                cont = on_window(
                    {
                        "page": window[0],
                        "pages": window,
                        "new_in_window": new_in_window,
                        "found": len(found),
                    }
                )
            except Exception:
                cont = True
            if cont is False:
                stopped_early = True
                note(
                    "Pathé discover paused early — "
                    "listing windows are all duplicates / no new queue rows"
                )
                break

        page += len(window)

    entries = list(found.values())[:max_items]
    err_note = f" · {len(errors)} page error(s)" if errors else ""
    if stopped_early:
        err_note += " · paused (dupe windows)"
    note(f"Pathé discover done · {len(entries):,} unique asset URLs{err_note}")
    return {
        "ok": True,
        "entries": entries,
        "n": len(entries),
        "pages_fetched": pages_fetched,
        "years": "all",
        "query": q,
        "errors": errors[:12],
        "truncated": len(entries) >= max_items,
        "concurrency": concurrency,
        "stopped_early": stopped_early,
    }
