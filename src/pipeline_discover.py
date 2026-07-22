"""Discover hubs into SQLite queue."""

from __future__ import annotations

import threading
import traceback

from config import DEFAULT_DISCOVER_MAX, DISCOVER_HARD_CAP
from crawl import crawl_url, is_crawlable
from db import get_job, init_db, insert_queue_items, set_job
from pipeline_state import _active, _lock


def start_discover(hub_url: str, max_items: int | None = None) -> dict:
    init_db()
    hub_url = (hub_url or "").strip()
    if not hub_url.startswith("http"):
        return {"ok": False, "error": "url_must_be_http"}

    try:
        cap = int(max_items) if max_items is not None else DEFAULT_DISCOVER_MAX
    except (TypeError, ValueError):
        cap = DEFAULT_DISCOVER_MAX
    cap = max(1, min(cap, DISCOVER_HARD_CAP))

    with _lock:
        if _active["discover"] or _active["scrape"]:
            return {"ok": False, "error": "busy", "job": get_job("discover")}
        _active["discover"] = True
    set_job(
        "discover",
        status="running",
        phase="discovering",
        message=f"Discovering up to {cap} video(s)…",
        progress=5,
        discovered=0,
        total=0,
        hub_url=hub_url,
        error="",
        max_videos=str(cap),
    )
    t = threading.Thread(target=_discover_job, args=(hub_url, cap), daemon=True)
    t.start()
    return {"ok": True, "job": get_job("discover")}


def _discover_job(hub_url: str, cap: int) -> None:
    from logutil import log_event, status

    result: dict = {}
    last_persist = [0.0]

    def report(msg: str, *, progress: float | None = None, persist: bool = False, **job_kw) -> None:
        import time as _t

        now = _t.time()
        # Persist milestone messages; throttle noisy tick updates to the DB log
        do_persist = persist or (now - last_persist[0] >= 15)
        if do_persist:
            last_persist[0] = now
        status(msg, job="discover", url=hub_url, persist=do_persist)
        fields = {"message": msg}
        if progress is not None:
            fields["progress"] = progress
        fields.update(job_kw)
        set_job("discover", **fields)

    try:
        report(
            f"Discover started — max {cap:,} · {hub_url}",
            progress=8,
            persist=True,
        )
        entries: list[dict] = []
        truncated = False
        if is_crawlable(hub_url):
            report(f"Crawling hub (max {cap:,})…", progress=12, persist=True)

            listed = [0]

            def on_status(msg: str) -> None:
                import re as _re

                m = _re.search(r"(?:Listed|Listing…)\s*([\d,]+)", msg)
                if m:
                    listed[0] = int(m.group(1).replace(",", ""))
                pct = 12 + min(53, int(53 * listed[0] / max(cap, 1)))
                report(msg, progress=pct, discovered=listed[0])

            result = crawl_url(hub_url, max_items=cap, on_status=on_status)
            if not result.get("ok"):
                err = result.get("error") or "discover_failed"
                log_event(err, job="discover", url=hub_url, detail=str(result)[:2000])
                set_job(
                    "discover",
                    status="error",
                    phase="error",
                    message=err,
                    error=err,
                    progress=100,
                )
                return
            if result.get("skip_hub"):
                entries = [
                    {
                        "url": hub_url,
                        "title": hub_url.rstrip("/").split("/")[-1],
                        "source": "Direct link",
                        "downloadable": "yes",
                    }
                ]
            else:
                entries = result.get("entries") or []
            truncated = bool(result.get("truncated"))
            report(
                f"Found {len(entries):,} video(s) — saving to queue…",
                progress=70,
                persist=True,
                discovered=len(entries),
            )
        else:
            entries = [
                {
                    "url": hub_url,
                    "title": hub_url.rstrip("/").split("/")[-1] or "Video",
                    "source": "Direct link",
                    "downloadable": "yes",
                }
            ]
            report("Single video link — saving…", progress=70, persist=True, discovered=1)

        chunk = 2000
        total_added = 0
        total_skipped = 0
        for i in range(0, len(entries), chunk):
            part = entries[i : i + chunk]
            merged = insert_queue_items(part, hub_url=hub_url)
            total_added += merged["n_added"]
            total_skipped += merged["n_skipped"]
            pct = 70 + int(25 * min(1.0, (i + len(part)) / max(len(entries), 1)))
            report(
                f"Saving {i + len(part):,}/{len(entries):,}… (+{total_added:,} new)",
                progress=pct,
                discovered=total_added,
            )

        trunc = " (listing truncated at max)" if truncated else ""
        done_msg = (
            f"Discovered {total_added:,} new video(s)"
            + (f" (skipped {total_skipped:,} dupes)" if total_skipped else "")
            + trunc
            + ". Choose All or a count, then Start scrape."
        )
        report(done_msg, progress=100, persist=True, discovered=total_added, total=total_added)
        set_job(
            "discover",
            status="done",
            phase="ready",
            message=done_msg,
            progress=100,
            discovered=total_added,
            total=total_added,
        )
        log_event(
            f"discover_ok added={total_added} skipped={total_skipped} cap={cap}",
            level="info",
            job="discover",
            url=hub_url,
        )
    except Exception as e:
        log_event(
            "discover_exception",
            job="discover",
            url=hub_url,
            exc=e,
            fatal_dashboard=True,
        )
        set_job(
            "discover",
            status="error",
            phase="error",
            message=str(e),
            error=traceback.format_exc()[-800:],
            progress=100,
        )
    finally:
        with _lock:
            _active["discover"] = False
