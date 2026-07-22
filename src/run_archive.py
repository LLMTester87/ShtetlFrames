"""Download + scan for one archive from the UI (timestamps-only).

Downloads matching queue items, scans without contact sheets, appends hits
with source_url timestamps, then deletes local video files.
"""

from __future__ import annotations

import csv
import json
import threading
import time
import traceback
from dataclasses import asdict
from pathlib import Path
import shutil
from typing import Callable

import requests
from ultralytics import YOLO

from config import (
    BULK_QUEUE_CSV,
    CANDIDATES_PATH,
    CONTACT_DIR,
    CROPS_DIR,
    DEFAULT_FPS,
    DEFAULT_SCORE_THRESHOLD,
    OUTPUT_DIR,
    VIDEOS_DIR,
)
from detect import CueScorer, aggregate_segments, scan_video, segments_to_jsonl
from download import download_archive_org, download_entry, download_http, slugify

USER_AGENT = "ShtetlFramesRun/1.0"
JOBS_PATH = OUTPUT_DIR / "jobs.json"
_lock = threading.Lock()
_active: dict[str, dict] = {}


# Maps UI archive id -> how to select bulk_queue rows
ARCHIVE_RUNNERS: dict[str, dict] = {
    "spielberg": {
        "title": "Spielberg Jewish Film Archive / JFC",
        "match": lambda r: (
            "spielberg" in (r.get("source") or "").lower()
            or "spielberg" in (r.get("title") or "").lower()
            or "five cities" in (r.get("title") or "").lower()
            or (
                "krakow" in (r.get("title") or "").lower()
                and "jewish life" in (r.get("title") or "").lower()
            )
        ),
        "runnable": True,
        "max_downloads": 3,
    },
    "ushmm": {
        "title": "USHMM Film & Video Archive",
        "match": lambda r: "ushmm" in (r.get("source") or "").lower()
        or "munkatch" in (r.get("title") or "").lower()
        or "munkacs" in (r.get("title") or "").lower(),
        "runnable": True,
        "max_downloads": 2,
        "notes": "Catalog is stream-first; runs known public mirrors (e.g. Munkács).",
    },
    "mirc": {
        "title": "USC MIRC Fox Movietone",
        "match": lambda r: (
            "mirc" in (r.get("source") or "").lower()
            or "agudah" in (r.get("title") or "").lower()
            or "agudas" in (r.get("title") or "").lower()
            or "chofetz" in (r.get("title") or "").lower()
            or "chofetz" in (r.get("url") or "").lower()
        ),
        "runnable": True,
        "max_downloads": 2,
    },
    "internet_archive": {
        "title": "Internet Archive (relevance-gated)",
        "match": lambda r: "internet archive" in (r.get("source") or "").lower(),
        "runnable": True,
        "max_downloads": 4,
    },
    "commons": {
        "title": "Wikimedia Commons video",
        "match": lambda r: "commons" in (r.get("source") or "").lower()
        and "ghetto" not in (r.get("title") or "").lower(),
        "runnable": True,
        "max_downloads": 3,
    },
    "yivo": {
        "title": "YIVO Film / polishjews.yivo.org",
        "match": lambda r: "yivo" in (r.get("source") or "").lower(),
        "runnable": False,
        "max_downloads": 0,
        "notes": "Mostly stream/institutional — no bulk download from this UI yet.",
    },
    "ncjf": {
        "title": "National Center for Jewish Film",
        "match": lambda r: "five cities" in (r.get("title") or "").lower()
        or "ncjf" in (r.get("source") or "").lower()
        or ("spielberg" in (r.get("title") or "").lower() and "jewish life" in (r.get("title") or "").lower()),
        "runnable": True,
        "max_downloads": 3,
    },
    "nara": {
        "title": "NARA / Universal Newsreels",
        "match": lambda r: "munkatch" in (r.get("title") or "").lower()
        or "munkacs" in (r.get("title") or "").lower()
        or "nara" in (r.get("source") or "").lower(),
        "runnable": True,
        "max_downloads": 2,
    },
    "periscope": {
        "title": "Periscope Film on Internet Archive",
        "match": lambda r: (
            "periscope" in (r.get("notes") or "").lower()
            or "periscope" in (r.get("title") or "").lower()
            or "periscope" in (r.get("source") or "").lower()
            or "periscopefilm" in (r.get("identifier") or "").lower()
            or (r.get("tier") or "") == "B-periscope"
        ),
        "runnable": True,
        "max_downloads": 3,
        # Auto-fill bulk_queue from IA when Run finds nothing tagged Periscope
        "discover": {
            # Jewish/region keywords first (works on IA); year filtered client-side
            "ia_query": (
                "creator:(PeriscopeFilm) AND "
                "(Jewish OR Jews OR Israel OR Palestine OR Hebrew OR synagogue OR Yiddish) AND "
                "mediatype:movies"
            ),
            "crawl_url": (
                "https://archive.org/search?query="
                "creator%3A(PeriscopeFilm)+AND+"
                "(Jewish+OR+Jews+OR+Israel+OR+Palestine+OR+Hebrew+OR+synagogue+OR+Yiddish)+AND+"
                "mediatype%3Amovies"
            ),
            "year_max": 1950,
            "require_any": [
                "jewish",
                "jews",
                "israel",
                "palestine",
                "hebrew",
                "synagogue",
                "yiddish",
                "jerusalem",
                "tel aviv",
                "hasid",
                "orthodox",
            ],
            "block_any": [
                "hitler",
                "nazi germany",
                "holocaust trial",
                "nuremberg",
                "nürnberg",
                "concentration camp",
                "babi yar",
                "expo",
                "palm springs",
                "steel mill",
                "soundies",
                "six day war",
                "1967",
            ],
            "max_items": 15,
            "source": "Periscope Film (IA)",
            "notes": "Periscope Film — year<=1950 Jewish/IA keywords",
            "tier": "B-periscope",
            "relevance": "75",
        },
    },
    "custom": {
        "title": "Your scraped links",
        "match": lambda r: (r.get("tier") or "") == "A-user"
        or "(user)" in (r.get("source") or "").lower()
        or "(crawl)" in (r.get("source") or "").lower(),
        "runnable": True,
        "max_downloads": 8,
    },
}


def _load_jobs() -> dict:
    if JOBS_PATH.exists():
        return json.loads(JOBS_PATH.read_text(encoding="utf-8"))
    return {}


def _save_jobs(jobs: dict) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    JOBS_PATH.write_text(json.dumps(jobs, indent=2), encoding="utf-8")


def get_job(archive_id: str) -> dict | None:
    with _lock:
        if archive_id in _active:
            return dict(_active[archive_id])
        jobs = _load_jobs()
        return jobs.get(archive_id)


def list_jobs() -> dict:
    with _lock:
        jobs = _load_jobs()
        jobs.update({k: dict(v) for k, v in _active.items()})
        return jobs


def _set_job(archive_id: str, **kwargs) -> None:
    with _lock:
        cur = _active.get(archive_id) or _load_jobs().get(archive_id) or {}
        cur.update(kwargs)
        cur["archive_id"] = archive_id
        cur["updated_at"] = time.time()
        _active[archive_id] = cur
        jobs = _load_jobs()
        jobs[archive_id] = dict(cur)
        _save_jobs(jobs)


def queue_rows_for_archive(archive_id: str) -> list[dict]:
    cfg = ARCHIVE_RUNNERS.get(archive_id)
    if not cfg or not BULK_QUEUE_CSV.exists():
        return []
    rows = list(csv.DictReader(BULK_QUEUE_CSV.open(encoding="utf-8")))
    matched = [r for r in rows if cfg["match"](r)]
    # Prefer direct downloadables
    matched.sort(
        key=lambda r: (
            0 if r.get("downloadable") == "yes" else 1,
            -int(float(r.get("relevance") or 0)),
        )
    )
    return matched


def discover_into_queue(archive_id: str, discover: dict) -> dict:
    """Crawl catalog hub into bulk_queue with archive-specific tags."""
    import re

    from crawl import crawl_url
    from crawl.archive_org import _ia_doc_to_entry, _ia_search
    from queue_manage import merge_entries_into_queue

    max_items = int(discover.get("max_items") or 15)
    year_max = discover.get("year_max")  # e.g. 1950
    require_terms = [t.lower() for t in (discover.get("require_any") or [])]
    block_terms = [t.lower() for t in (discover.get("block_any") or [])]

    def inferred_year(e: dict) -> int | None:
        y = (e.get("year") or "").strip()
        if y.isdigit():
            return int(y[:4])
        title = e.get("title") or ""
        # Prefer explicit 4-digit years in title
        years = [int(m) for m in re.findall(r"\b(19\d{2}|20\d{2})\b", title)]
        if years:
            return min(years)
        # Decade forms: 1940s → treat as end of decade for year_max checks
        m = re.search(r"\b(19\d0)s\b", title, re.I)
        if m:
            return int(m.group(1)) + 9
        return None

    def keep_entry(e: dict) -> bool:
        title = (e.get("title") or "").lower()
        blob = f"{title} {(e.get('notes') or '')} {(e.get('identifier') or '')}".lower()
        if require_terms and not any(t in blob for t in require_terms):
            return False
        if block_terms and any(t in blob for t in block_terms):
            return False
        if year_max is not None:
            iy = inferred_year(e)
            if iy is not None and iy > int(year_max):
                return False
            # Prefer dated items; undated OK if keywords match
        return True

    entries: list[dict] = []
    ia_query = (discover.get("ia_query") or "").strip()
    if ia_query:
        fetch_n = max_items * 4 if (year_max is not None or require_terms) else max_items
        docs = _ia_search(ia_query, rows=min(fetch_n, 100))
        for d in docs:
            e = _ia_doc_to_entry(d, ia_query)
            if not e or not keep_entry(e):
                continue
            entries.append(e)
            if len(entries) >= max_items:
                break
        if not entries:
            return {"ok": False, "error": "ia_query_empty", "n_added": 0}
        kind = "ia_query"
        truncated = len(docs) >= fetch_n
    else:
        url = (discover.get("crawl_url") or "").strip()
        if not url:
            return {"ok": False, "error": "no_crawl_url", "n_added": 0}
        result = crawl_url(url, max_items=max_items * 3 if year_max else max_items)
        if not result.get("ok"):
            return {
                "ok": False,
                "error": result.get("error") or "crawl_failed",
                "n_added": 0,
            }
        for e in result.get("entries") or []:
            if not keep_entry(e):
                continue
            entries.append(e)
            if len(entries) >= max_items:
                break
        if not entries:
            return {"ok": False, "error": "crawl_empty", "n_added": 0}
        kind = result.get("kind")
        truncated = bool(result.get("truncated"))

    merged = merge_entries_into_queue(
        entries,
        source=discover.get("source"),
        notes=discover.get("notes"),
        tier=discover.get("tier") or "B-discover",
        relevance=discover.get("relevance") or "80",
    )
    merged["kind"] = kind
    merged["truncated"] = truncated
    merged["archive_id"] = archive_id
    return merged


def resolve_commons_file_url(wiki_url: str) -> str | None:
    """Turn a commons wiki File: page into a direct upload URL."""
    if "upload.wikimedia.org" in wiki_url:
        return wiki_url
    if "commons.wikimedia.org" not in wiki_url:
        return None
    # Extract File:Name
    name = wiki_url.split("/wiki/")[-1]
    name = requests.utils.unquote(name)
    api = "https://commons.wikimedia.org/w/api.php"
    r = requests.get(
        api,
        params={
            "action": "query",
            "titles": name.replace("_", " "),
            "prop": "imageinfo",
            "iiprop": "url",
            "format": "json",
        },
        timeout=60,
        headers={"User-Agent": USER_AGENT},
    )
    r.raise_for_status()
    pages = r.json().get("query", {}).get("pages", {})
    for page in pages.values():
        info = page.get("imageinfo") or []
        if info and info[0].get("url"):
            return info[0]["url"]
    return None


def download_queue_item(row: dict) -> dict:
    """Download one bulk_queue row; returns download_entry-like dict."""
    url = (row.get("url") or "").strip()
    title = row.get("title") or "video"
    if row.get("downloadable") not in ("yes",):
        return {"error": f"not_downloadable:{row.get('downloadable')}", "url": url, "title": title}
    if "youtube.com/results" in url:
        return {"error": "search_url_not_direct", "url": url, "title": title}

    # Commons wiki page → direct file
    if "commons.wikimedia.org/wiki/File:" in url:
        direct = resolve_commons_file_url(url)
        if not direct:
            return {"error": "commons_resolve_failed", "url": url, "title": title}
        url = direct

    vid = slugify(title)
    if "Chofetz" in url or "Agudas" in title or "Agudah" in title:
        vid = "agudah_1923_commons"
    if "rp1OeIf0D0w" in url or "Munkatch" in title:
        vid = "munkacs_1933_yt"
    if "hdf6-qnr11s" in url or "Krakow" in title:
        vid = "five_cities_krakow"

    info = download_entry(url, title, video_id=vid)
    info["source_url"] = row.get("url") or url
    info["archive_title"] = title
    return info


def append_hits_export(segments: list, source_url: str, archive_id: str) -> int:
    """Append segments to candidates.jsonl + rebuild review_queue (keeps hit stills)."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    annotated = []
    for s in segments:
        d = asdict(s) if hasattr(s, "__dataclass_fields__") else dict(s)
        d["source_url"] = source_url
        d["archive_id"] = archive_id
        # Preserve contact_sheet from segment (hit-frame stills)
        if not d.get("contact_sheet"):
            d["contact_sheet"] = None
        annotated.append(d)
        with CANDIDATES_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    rebuild_review_queue_from_jsonl()
    return len(annotated)


def rebuild_review_queue_from_jsonl() -> None:
    if not CANDIDATES_PATH.exists():
        return
    rows = []
    for line in CANDIDATES_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))
    rows.sort(key=lambda r: r.get("rank_score", 0), reverse=True)
    path = OUTPUT_DIR / "review_queue.csv"
    fields = [
        "rank",
        "video_id",
        "start_sec",
        "end_sec",
        "peak_score",
        "mean_score",
        "rank_score",
        "hit_count",
        "best_cue",
        "source_path",
        "source_url",
        "archive_id",
        "contact_sheet",
        "label",
        "human_accept_reject",
        "reviewer_notes",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for i, r in enumerate(rows, 1):
            sheet = r.get("contact_sheet") or ""
            w.writerow(
                {
                    "rank": i,
                    "video_id": r.get("video_id"),
                    "start_sec": r.get("start_sec"),
                    "end_sec": r.get("end_sec"),
                    "peak_score": r.get("peak_score"),
                    "mean_score": r.get("mean_score"),
                    "rank_score": r.get("rank_score"),
                    "hit_count": r.get("hit_count"),
                    "best_cue": r.get("best_cue"),
                    "source_path": "",  # cleared — video deleted after scan
                    "source_url": r.get("source_url", ""),
                    "archive_id": r.get("archive_id", ""),
                    "contact_sheet": sheet,
                    "label": r.get("label", "visual_candidate_not_identity"),
                    "human_accept_reject": "",
                    "reviewer_notes": "",
                }
            )


def delete_local_video(path: Path | None) -> None:
    if not path:
        return
    p = Path(path)
    if p.exists() and p.is_file():
        try:
            p.unlink()
        except OSError:
            pass
    meta = p.with_suffix(p.suffix + ".meta.json")
    # also try video_id.meta.json
    for m in VIDEOS_DIR.glob(f"{p.stem}*.meta.json"):
        try:
            m.unlink()
        except OSError:
            pass
    if meta.exists():
        try:
            meta.unlink()
        except OSError:
            pass


def run_archive_job(archive_id: str, max_downloads: int | None = None) -> None:
    cfg = ARCHIVE_RUNNERS.get(archive_id)
    if not cfg:
        _set_job(archive_id, status="error", message="Unknown archive", progress=100)
        return
    if not cfg.get("runnable"):
        _set_job(
            archive_id,
            status="error",
            message=cfg.get("notes") or "This archive cannot be bulk-run from the UI",
            progress=100,
        )
        return

    limit = int(max_downloads) if max_downloads is not None else int(cfg.get("max_downloads") or 3)
    limit = max(1, min(limit, 50))

    try:
        _set_job(
            archive_id,
            status="running",
            message="Selecting queue items…",
            progress=5,
            hits=0,
            downloaded=0,
            max_downloads=limit,
        )
        items = [
            r
            for r in queue_rows_for_archive(archive_id)
            if r.get("downloadable") == "yes"
        ][:limit]

        # Auto-discover from catalog when queue has nothing for this archive
        if not items and cfg.get("discover"):
            disc = dict(cfg["discover"])
            # Pull at least as many as this run needs
            disc["max_items"] = max(int(disc.get("max_items") or 15), limit)
            _set_job(
                archive_id,
                message="Queue empty — discovering from catalog…",
                progress=8,
            )
            found = discover_into_queue(archive_id, disc)
            if found.get("n_added"):
                _set_job(
                    archive_id,
                    message=f"Discovered {found['n_added']} item(s); selecting downloads…",
                    progress=12,
                    discovered=found["n_added"],
                )
                items = [
                    r
                    for r in queue_rows_for_archive(archive_id)
                    if r.get("downloadable") == "yes"
                ][:limit]
            else:
                _set_job(
                    archive_id,
                    status="error",
                    message=f"Catalog discover found nothing ({found.get('error') or 'empty'}).",
                    progress=100,
                )
                return

        if not items:
            _set_job(
                archive_id,
                status="error",
                message="No downloadable items for this archive in the queue.",
                progress=100,
            )
            return

        _set_job(archive_id, message=f"Downloading {len(items)} video(s)…", progress=10)
        downloaded = []
        for i, row in enumerate(items):
            _set_job(
                archive_id,
                message=f"Downloading {i+1}/{len(items)}: {(row.get('title') or '')[:50]}",
                progress=10 + int(25 * i / max(len(items), 1)),
            )
            info = download_queue_item(row)
            if info.get("path") and not info.get("error"):
                downloaded.append(info)
            else:
                _set_job(archive_id, last_error=info.get("error") or "download_failed")

        if not downloaded:
            _set_job(archive_id, status="error", message="All downloads failed", progress=100)
            return

        _set_job(
            archive_id,
            message="Loading models + scanning (timestamps only)…",
            progress=40,
            downloaded=len(downloaded),
        )
        from config import YOLO_WEIGHTS

        yolo = YOLO(YOLO_WEIGHTS)
        scorer = CueScorer()
        total_hits = 0
        CONTACT_DIR.mkdir(parents=True, exist_ok=True)
        for i, info in enumerate(downloaded):
            path = Path(info["path"])
            video_id = info.get("video_id") or path.stem
            _set_job(
                archive_id,
                message=f"Scanning {i+1}/{len(downloaded)}: {video_id}",
                progress=40 + int(45 * i / max(len(downloaded), 1)),
            )
            crop_dir = CROPS_DIR / video_id
            hits = scan_video(
                path,
                video_id=video_id,
                scorer=scorer,
                yolo=yolo,
                sample_fps=DEFAULT_FPS,
                score_threshold=DEFAULT_SCORE_THRESHOLD,
                save_crops_dir=crop_dir,
            )
            segs = aggregate_segments(
                hits,
                source_path=str(path),
                sheet_dir=CONTACT_DIR,
            )
            n = append_hits_export(segs, source_url=info.get("source_url") or "", archive_id=archive_id)
            total_hits += n
            # Drop raw crop files + video; keep contact sheet stills in CONTACT_DIR
            shutil.rmtree(crop_dir, ignore_errors=True)
            delete_local_video(path)
            _set_job(archive_id, hits=total_hits)

        # Update scan summary lightly
        summary_path = OUTPUT_DIR / "scan_summary.json"
        prev = {}
        if summary_path.exists():
            prev = json.loads(summary_path.read_text(encoding="utf-8"))
        prev.update(
            {
                "last_archive_id": archive_id,
                "last_hits_added": total_hits,
                "timestamps_only": True,
                "candidates_path": str(CANDIDATES_PATH),
            }
        )
        # recount
        if CANDIDATES_PATH.exists():
            nseg = sum(1 for line in CANDIDATES_PATH.read_text(encoding="utf-8").splitlines() if line.strip())
            prev["segments"] = nseg
        summary_path.write_text(json.dumps(prev, indent=2), encoding="utf-8")

        _set_job(
            archive_id,
            status="done",
            message=f"Done — {total_hits} hit segments saved (timestamps only; videos deleted)",
            progress=100,
            hits=total_hits,
            downloaded=len(downloaded),
        )
    except Exception as e:
        _set_job(
            archive_id,
            status="error",
            message=str(e),
            progress=100,
            traceback=traceback.format_exc()[-1500:],
        )
    finally:
        with _lock:
            # keep in jobs.json; drop from active
            if archive_id in _active:
                jobs = _load_jobs()
                jobs[archive_id] = dict(_active[archive_id])
                _save_jobs(jobs)
                # leave in _active until done so polling works — clear when done
                if _active[archive_id].get("status") in ("done", "error"):
                    pass


def start_archive_run(archive_id: str, max_downloads: int | None = None) -> dict:
    if archive_id not in ARCHIVE_RUNNERS:
        return {"ok": False, "error": "unknown_archive"}
    with _lock:
        cur = _active.get(archive_id) or {}
        if cur.get("status") == "running":
            return {"ok": False, "error": "already_running", "job": cur}

    limit = None
    if max_downloads is not None:
        try:
            limit = max(1, min(int(max_downloads), 50))
        except (TypeError, ValueError):
            limit = None

    _set_job(
        archive_id,
        status="running",
        message="Starting…",
        progress=0,
        hits=0,
        downloaded=0,
        max_downloads=limit,
        started_at=time.time(),
    )
    t = threading.Thread(
        target=run_archive_job,
        args=(archive_id,),
        kwargs={"max_downloads": limit},
        daemon=True,
    )
    t.start()
    return {"ok": True, "job": get_job(archive_id)}
