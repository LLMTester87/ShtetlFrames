"""Parallel scrape/scan with cloud stills only."""

from __future__ import annotations

import json
import shutil
import tempfile
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

from ultralytics import YOLO

from cloud_images import upload_image
from config import (
    CROPS_DIR,
    DEFAULT_FPS,
    DEFAULT_SCORE_THRESHOLD,
    DEFAULT_WORKERS,
    MAX_WORKERS,
    ROOT,
    VIDEOS_DIR,
    effective_scan_backend,
    load_env,
)
import config as app_config
from db import get_job, init_db, insert_candidates, set_job, set_queue_status, take_pending


def _safe_queue_status(item_id: int, status: str, error: str = "", detail: str = "") -> None:
    """Retry queue status writes — silent SQLite failures left rows stuck scanning."""
    last: Exception | None = None
    for attempt in range(1, 4):
        try:
            set_queue_status(item_id, status, error=error, detail=detail)
            return
        except Exception as e:
            last = e
            time.sleep(0.2 * attempt)
    if last is not None:
        try:
            set_job("scrape", error=f"queue_status_failed #{item_id}: {last}"[:400])
        except Exception:
            pass
from detect import CueScorer, aggregate_segments, scan_video, write_sheet_from_crops
from download import slugify
from pipeline_state import _active, _lock
from run_archive import delete_local_video, download_queue_item

# Live scrape workers: queue_id -> {title, phase, detail, started}
_scrape_live: dict[int, dict] = {}
_scrape_counts = {"done": 0, "hits": 0, "errors": 0, "total": 0}
_scrape_counts_lock = threading.Lock()
_scrape_last_publish = 0.0
# Jump the line during a running scrape (next free worker slot).
_scrape_priority: list[dict] = []
_scrape_priority_lock = threading.Lock()

# Per-thread models so workers can scan in parallel safely
_tls = threading.local()

# #region agent log
_DBG_LOG = Path(ROOT) / "debug-30525a.log"
_DBG_LOCK = threading.Lock()


def _dbg(hypothesis_id: str, location: str, message: str, **data: object) -> None:
    try:
        payload = {
            "sessionId": "30525a",
            "hypothesisId": hypothesis_id,
            "location": location,
            "message": message,
            "data": data,
            "timestamp": int(time.time() * 1000),
            "tid": threading.get_ident(),
        }
        line = json.dumps(payload, default=str) + "\n"
        with _DBG_LOCK:
            with _DBG_LOG.open("a", encoding="utf-8") as f:
                f.write(line)
    except Exception:
        pass


# #endregion


def _short_title(title: str, n: int = 52) -> str:
    t = (title or "video").strip().replace("\n", " ")
    return t if len(t) <= n else t[: n - 1] + "…"


def _publish_scrape_status(*, force_persist: bool = False, force: bool = False) -> None:
    """Build UI + console dashboard from live workers and counters.

    Safe to call from the scrape coordinator thread. Workers must not call this —
    they only update `_scrape_live` in memory (SQLite set_job was freezing completions).
    """
    global _scrape_last_publish
    from logutil import status

    now = time.time()
    if not force and not force_persist and (now - _scrape_last_publish) < 1.25:
        return
    _scrape_last_publish = now

    with _scrape_counts_lock:
        done = _scrape_counts["done"]
        hits = _scrape_counts["hits"]
        errors = _scrape_counts["errors"]
        total = max(_scrape_counts["total"], 1)
    with _lock:
        live = list(_scrape_live.items())

    lines = [f"{done}/{total} done · {hits} hits · {errors} err · {len(live)} active"]
    for qid, info in live[:8]:
        phase = info.get("phase") or "?"
        title = _short_title(info.get("title") or "")
        detail = (info.get("detail") or "").strip()
        bit = f"#{qid} {phase}: {title}"
        if detail:
            bit += f" ({detail})"
        lines.append(bit)

    ui_msg = "\n".join(lines)[:1200]
    pct = 5 + int(90 * done / total)
    try:
        set_job(
            "scrape",
            completed=done,
            hits=hits,
            progress=min(99, pct),
            message=ui_msg,
            phase="scraping",
        )
    except Exception:
        pass
    try:
        from console_dash import is_enabled, set_scrape

        if is_enabled():
            set_scrape(
                done=done,
                total=total,
                hits=hits,
                errors=errors,
                live=[
                    {
                        "title": (info.get("title") or ""),
                        "phase": (info.get("phase") or ""),
                        "detail": (info.get("detail") or ""),
                    }
                    for _, info in live[:8]
                ],
            )
            if force_persist:
                status(lines[0], job="scrape", persist=False, console=False)
            return
    except Exception:
        pass
    status(lines[0], job="scrape", persist=False)
    for line in lines[1:]:
        status(f"  → {line}", job="scrape", persist=False)


def _set_worker_phase(row: dict, phase: str, detail: str = "") -> None:
    """Memory-only phase update from worker threads (no SQLite)."""
    qid = int(row["id"])
    title = row.get("title") or row.get("url") or "video"
    with _lock:
        _scrape_live[qid] = {
            "title": title,
            "phase": phase,
            "detail": detail,
            "started": time.time(),
        }


def _clear_worker(qid: int) -> None:
    with _lock:
        _scrape_live.pop(int(qid), None)


def _start_status_ticker(stop: threading.Event) -> threading.Thread:
    """Refresh dashboard from memory while workers run (coordinator-owned SQLite writes)."""

    def _loop() -> None:
        while not stop.wait(1.0):
            try:
                # #region agent log
                now = time.time()
                with _lock:
                    stuck = [
                        {
                            "qid": qid,
                            "phase": info.get("phase"),
                            "detail": (info.get("detail") or "")[:60],
                            "age_s": int(now - float(info.get("started") or now)),
                        }
                        for qid, info in _scrape_live.items()
                        if "no hit" in (info.get("detail") or "").lower()
                    ]
                if stuck:
                    _dbg(
                        "C",
                        "pipeline_scrape.py:ticker",
                        "live no-hit still active",
                        stuck=stuck,
                        live_n=len(_scrape_live),
                    )
                # #endregion
                _publish_scrape_status()
            except Exception:
                pass

    t = threading.Thread(target=_loop, name="scrape-status-ticker", daemon=True)
    t.start()
    return t


def _models():
    """Each worker thread gets its own YOLO + CLIP (safe parallel scans)."""
    if not getattr(_tls, "ready", False):
        from logutil import status

        tid = threading.get_ident()
        status(f"Loading scan models on worker thread {tid}…", job="scrape")
        from config import YOLO_WEIGHTS

        _tls.yolo = YOLO(YOLO_WEIGHTS)
        _tls.scorer = CueScorer()
        _tls.ready = True
        status(f"Models ready on worker thread {tid}", job="scrape")
    return _tls.yolo, _tls.scorer


def _warm_worker_models(n: int) -> None:
    """Preload one model set per scrape worker so first videos aren't delayed."""
    n = max(1, int(n))

    def _load(_: int) -> None:
        _models()

    with ThreadPoolExecutor(max_workers=n) as ex:
        list(ex.map(_load, range(n)))


def is_scrape_running() -> bool:
    return bool(_active.get("scrape"))


def prioritize_queue_url(url: str, *, title: str = "") -> dict:
    """Put a queue URL first for the next claim, and jump the line if scrape is running."""
    from db import db, insert_queue_items

    url = (url or "").strip()
    if not url.startswith("http"):
        return {"ok": False, "error": "url_required"}
    init_db()
    with db() as conn:
        row = conn.execute("SELECT * FROM queue_items WHERE url=?", (url,)).fetchone()
    if row is None:
        insert_queue_items(
            [
                {
                    "url": url,
                    "title": (title or url)[:300],
                    "source": "priority",
                    "downloadable": "yes",
                }
            ],
            hub_url="manual:priority",
        )
    with db(write=True) as conn:
        row = conn.execute("SELECT * FROM queue_items WHERE url=?", (url,)).fetchone()
        if row is None:
            return {"ok": False, "error": "insert_failed"}
        mid = int(row["id"])
        conn.execute(
            "UPDATE queue_items SET status='pending', error='', detail='next' WHERE id=?",
            (mid,),
        )
        # Become the first claimable row (pending ranks above queued).
        first = conn.execute(
            "SELECT id, url FROM queue_items WHERE "
            "status IN ('pending','queued','scanning','downloading','uploading','error') "
            "AND downloadable='yes' ORDER BY "
            "CASE status WHEN 'pending' THEN 0 WHEN 'queued' THEN 1 "
            "WHEN 'error' THEN 2 ELSE 3 END, id LIMIT 1"
        ).fetchone()
        if first and first["url"] != url:
            fid = int(first["id"])
            conn.execute("UPDATE queue_items SET id=? WHERE id=?", (-999999, mid))
            conn.execute("UPDATE queue_items SET id=? WHERE id=?", (mid, fid))
            conn.execute("UPDATE queue_items SET id=? WHERE id=?", (fid, -999999))
            mid = fid
        row = conn.execute("SELECT * FROM queue_items WHERE url=?", (url,)).fetchone()

    row_dict = dict(row)
    injected = False
    with _lock:
        running = bool(_active.get("scrape"))
    if running:
        with _scrape_priority_lock:
            # Avoid duplicate priority entries for same id.
            _scrape_priority[:] = [r for r in _scrape_priority if int(r.get("id") or 0) != int(row_dict["id"])]
            _scrape_priority.insert(0, row_dict)
            injected = True
    return {
        "ok": True,
        "id": int(row_dict["id"]),
        "status": row_dict.get("status"),
        "injected": injected,
        "scrape_running": running,
        "title": row_dict.get("title"),
    }


def start_scrape(max_videos: str | int = "all", workers: int = DEFAULT_WORKERS) -> dict:
    init_db()
    load_env()
    backend = effective_scan_backend()
    with _lock:
        if _active["discover"] or _active["scrape"]:
            return {"ok": False, "error": "busy", "job": get_job("scrape")}
        _active["scrape"] = True

    # RunPod: fan out many in-flight jobs; local: thread pool size
    if backend == "runpod":
        max_inflight = max(1, int(app_config.RUNPOD_MAX_INFLIGHT or 8))
        workers = max(1, min(int(workers or max_inflight), max_inflight, MAX_WORKERS * 2))
    else:
        workers = max(1, min(int(workers or DEFAULT_WORKERS), MAX_WORKERS))
    from config import DISCOVER_HARD_CAP

    if isinstance(max_videos, str) and max_videos.lower() == "all":
        limit = None
        max_label = "all"
    else:
        try:
            limit = max(1, min(int(max_videos), DISCOVER_HARD_CAP))
            max_label = str(limit)
        except (TypeError, ValueError):
            limit = None
            max_label = "all"
    items = take_pending(limit)
    if not items:
        with _lock:
            _active["scrape"] = False
        set_job(
            "scrape",
            status="error",
            phase="error",
            message="No pending or failed videos to retry. Discover a URL first.",
            progress=100,
        )
        return {"ok": False, "error": "queue_empty", "job": get_job("scrape")}

    # Manual "next" / priority rows first, then normal claim order.
    items = sorted(
        items,
        key=lambda r: (
            0 if (r.get("detail") or "") == "next" else 1,
            int(r.get("id") or 0),
        ),
    )

    n_retry = sum(1 for r in items if (r.get("status") or "") == "error")
    start_msg = f"Starting scrape of {len(items)} video(s) · backend={backend} · workers={workers}"
    if n_retry:
        start_msg += f" · retrying {n_retry} earlier error(s)"

    set_job(
        "scrape",
        status="running",
        phase="scraping",
        message=start_msg,
        progress=2,
        total=len(items),
        completed=0,
        hits=0,
        error="",
        max_videos=max_label,
        workers=workers,
    )
    t = threading.Thread(target=_scrape_job, args=(items, workers, backend), daemon=True)
    t.start()
    return {"ok": True, "job": get_job("scrape"), "backend": backend}


def _scrape_job(items: list[dict], workers: int, backend: str = "local") -> None:
    from logutil import log_event, status

    with _lock:
        _scrape_live.clear()
    with _scrape_counts_lock:
        _scrape_counts.update({"done": 0, "hits": 0, "errors": 0, "total": len(items)})

    try:
        if backend == "runpod":
            status(
                f"RunPod scrape — {len(items)} video(s), up to {workers} on auto GPU Pod",
                job="scrape",
                persist=True,
            )
            set_job("scrape", message="Spinning up RunPod GPU Pod…", progress=3)

            from runpod_client import set_pod_pool
            from runpod_provision import MAX_PARALLEL_PODS, ensure_pods, stop_pod

            def pod_status(msg: str) -> None:
                status(msg, job="scrape")
                set_job("scrape", message=msg)

            n_pods = max(1, min(int(app_config.RUNPOD_MAX_INFLIGHT or workers or 2), MAX_PARALLEL_PODS))
            # Start scrape as soon as the first pod is healthy — don't wait for the full pool.
            bases = ensure_pods(
                count=n_pods,
                on_status=pod_status,
                min_ready=1,
                extra_fill_sec=0,
            )
            set_pod_pool(bases)
            status(
                f"{len(bases)}/{n_pods} GPU pod(s) ready — starting work",
                job="scrape",
                persist=True,
            )
            if len(bases) < n_pods:

                def _fill_remaining_pods() -> None:
                    try:
                        more = ensure_pods(
                            count=n_pods,
                            on_status=pod_status,
                            min_ready=1,
                            extra_fill_sec=900,
                        )
                        set_pod_pool(more)
                        status(
                            f"Pod pool expanded to {len(more)}/{n_pods}",
                            job="scrape",
                            persist=True,
                        )
                    except Exception as e:
                        status(f"Background pod fill: {e}", job="scrape", persist=True)

                threading.Thread(
                    target=_fill_remaining_pods,
                    daemon=True,
                    name="runpod-fill",
                ).start()
            # Export browser YouTube cookies so pods can bypass “not a bot”.
            try:
                from yt_cookies import ensure_cookies_for_scrape

                ck = ensure_cookies_for_scrape()
                if ck.get("ok"):
                    status(
                        f"YouTube cookies ready ({ck.get('bytes', 0)} bytes)",
                        job="scrape",
                        persist=True,
                    )
                else:
                    from yt_proxy import proxy_configured, proxy_provider_name

                    if proxy_configured():
                        status(
                            f"Browser cookies locked (Edge open) — {proxy_provider_name()} "
                            "will handle Google blocks",
                            job="scrape",
                            persist=True,
                        )
                    else:
                        status(
                            "YouTube cookies unavailable — add Scrapfly or ScrapingDog in "
                            "Settings, or quit Edge and refresh cookies",
                            job="scrape",
                            persist=True,
                        )
            except Exception as e:
                status(f"YouTube cookie export skipped: {e}", job="scrape", persist=True)
            set_job(
                "scrape",
                message=f"{len(bases)}/{n_pods} GPU pod(s) — scanning {len(items)}…",
                progress=5,
            )
            # Modest fan-out — too many PC threads stampede a dead pool into mass failures.
            workers = max(workers, n_pods)
            workers = min(max(workers, n_pods * 2), MAX_PARALLEL_PODS * 2)
        else:
            status(f"Loading models for {len(items)} video(s), {workers} worker(s)…", job="scrape", persist=True)
            set_job("scrape", message=f"Loading {workers} worker model set(s)…", progress=5)
            _warm_worker_models(workers)
            CROPS_DIR.mkdir(parents=True, exist_ok=True)
            VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
            status("Models ready — workers starting (parallel scans)", job="scrape", persist=True)
        _publish_scrape_status(force_persist=True)
        ticker_stop = threading.Event()
        _start_status_ticker(ticker_stop)

        # Sliding window — do not enqueue all 80k futures at once (starves + orphans).
        infra_streak = {"n": 0}
        pause_until = {"t": 0.0}
        watchdog_stop = threading.Event()

        def _pool_watchdog() -> None:
            if backend != "runpod":
                return
            from runpod_client import maintain_pod_pool, pool_size

            while not watchdog_stop.wait(45.0):
                try:
                    if pool_size() < 1:
                        status("Watchdog: GPU pool empty — refreshing…", job="scrape")
                    maintain_pod_pool(
                        target=max(
                            1,
                            min(
                                int(app_config.RUNPOD_MAX_INFLIGHT or 1),
                                MAX_PARALLEL_PODS,
                            ),
                        ),
                        on_status=lambda m: status(m, job="scrape"),
                    )
                except Exception as e:
                    status(f"Watchdog pod maintain: {e}"[:160], job="scrape")

        if backend == "runpod":
            threading.Thread(
                target=_pool_watchdog, daemon=True, name="runpod-watchdog"
            ).start()

        try:
            with ThreadPoolExecutor(max_workers=workers) as ex:
                item_iter = iter(items)
                inflight: dict = {}
                more_items = {"v": True}

                def _submit_more() -> None:
                    # Circuit breaker: after a streak of infra failures, cool down + refresh.
                    if time.time() < pause_until["t"]:
                        return
                    while len(inflight) < workers:
                        row = None
                        with _scrape_priority_lock:
                            if _scrape_priority:
                                row = _scrape_priority.pop(0)
                        if row is None:
                            try:
                                row = next(item_iter)
                            except StopIteration:
                                more_items["v"] = False
                                return
                        qid = int(row["id"])
                        _set_worker_phase(row, "scanning", "starting…")
                        try:
                            _safe_queue_status(qid, "scanning", detail="starting…")
                        except Exception:
                            pass
                        fut = ex.submit(_process_one, row, backend)
                        inflight[fut] = row

                _submit_more()
                while inflight or more_items["v"]:
                    if not inflight:
                        # Wait out circuit-breaker pause, then refill — don't end the batch.
                        if time.time() < pause_until["t"]:
                            _publish_scrape_status()
                            time.sleep(1.0)
                            continue
                        _submit_more()
                        if not inflight:
                            break
                    try:
                        fut = next(as_completed(inflight, timeout=1.0))
                    except TimeoutError:
                        # #region agent log
                        done_waiting = [inflight[f]["id"] for f in list(inflight) if f.done()]
                        if done_waiting:
                            _dbg(
                                "B",
                                "pipeline_scrape.py:timeout_with_done",
                                "as_completed timeout but futures already done",
                                done_qids=done_waiting,
                                inflight=len(inflight),
                                runId="post-fix",
                            )
                        # #endregion
                        _publish_scrape_status()
                        _submit_more()
                        continue
                    row = inflight.pop(fut)
                    qid = row["id"]
                    title = _short_title(row.get("title") or "")
                    # #region agent log
                    with _lock:
                        live_before = dict(_scrape_live.get(qid) or {})
                        live_n = len(_scrape_live)
                    _dbg(
                        "B",
                        "pipeline_scrape.py:as_completed",
                        "coordinator got completed future",
                        qid=qid,
                        inflight=len(inflight),
                        live_n=live_n,
                        live_detail=(live_before.get("detail") or "")[:80],
                        live_phase=live_before.get("phase"),
                        fut_done=fut.done(),
                        runId="post-fix",
                    )
                    # #endregion
                    # Drop from live UI immediately — never wait on SQLite while still "active".
                    _clear_worker(qid)
                    try:
                        # #region agent log
                        t_res = time.time()
                        # #endregion
                        n = fut.result()
                        # #region agent log
                        _dbg(
                            "B",
                            "pipeline_scrape.py:fut.result",
                            "fut.result returned",
                            qid=qid,
                            n=n,
                            result_ms=int((time.time() - t_res) * 1000),
                            runId="post-fix",
                        )
                        # #endregion
                        with _scrape_counts_lock:
                            _scrape_counts["hits"] += n
                            _scrape_counts["done"] += 1
                        # Worker already marks done; keep this as a fast idempotent write.
                        # #region agent log
                        t_db = time.time()
                        # #endregion
                        _safe_queue_status(qid, "done", error="", detail=f"{n} hit segment(s)")
                        # #region agent log
                        _dbg(
                            "D",
                            "pipeline_scrape.py:coord_set_done",
                            "coordinator set_queue_status done",
                            qid=qid,
                            db_ms=int((time.time() - t_db) * 1000),
                            runId="post-fix",
                        )
                        # #endregion
                        status(f"DONE #{qid} {title} → {n} hits", job="scrape", persist=False)
                    except Exception as e:
                        err_txt = str(e)[:800]
                        from runpod_client import (
                            is_infra_error,
                            is_permanent_youtube_skip,
                            refresh_pod_pool,
                        )

                        skip = is_permanent_youtube_skip(err_txt)
                        infra = (not skip) and is_infra_error(err_txt)
                        with _scrape_counts_lock:
                            if skip:
                                _scrape_counts["done"] += 1
                            elif infra:
                                # Do not count as a permanent failure — will retry later.
                                pass
                            else:
                                _scrape_counts["errors"] += 1
                                _scrape_counts["done"] += 1
                        try:
                            if skip:
                                # Members/private/removed — will never succeed; don't poison the queue.
                                _safe_queue_status(
                                    qid,
                                    "done",
                                    error="",
                                    detail=f"skipped: {err_txt[:160]}",
                                )
                                infra_streak["n"] = 0
                            elif infra:
                                # Put back on the queue — GPU/proxy blips must not burn videos.
                                _safe_queue_status(
                                    qid,
                                    "pending",
                                    error="",
                                    detail=f"retry later: {err_txt[:140]}",
                                )
                                infra_streak["n"] += 1
                                if infra_streak["n"] >= 3:
                                    pause_until["t"] = time.time() + 45.0
                                    status(
                                        f"Infra streak ({infra_streak['n']}) — "
                                        "pausing new submits 45s and refreshing pods…",
                                        job="scrape",
                                        persist=True,
                                    )
                                    try:
                                        refresh_pod_pool(
                                            count=max(
                                                1,
                                                min(
                                                    int(app_config.RUNPOD_MAX_INFLIGHT or 1),
                                                    4,
                                                ),
                                            ),
                                            on_status=lambda m: status(m, job="scrape"),
                                            force=True,
                                        )
                                    except Exception as refresh_err:
                                        status(
                                            f"Circuit refresh failed: {refresh_err}"[:160],
                                            job="scrape",
                                            persist=True,
                                        )
                                    infra_streak["n"] = 0
                            else:
                                _safe_queue_status(qid, "error", error=err_txt, detail="")
                                set_job("scrape", error=err_txt)
                                infra_streak["n"] = 0
                        except Exception:
                            pass
                        try:
                            log_event(
                                err_txt,
                                job="scrape",
                                queue_id=qid,
                                url=row.get("url") or "",
                                exc=e,
                                fatal_dashboard=False,
                            )
                        except Exception as log_err:
                            # #region agent log
                            _dbg(
                                "H6",
                                "pipeline_scrape.py:log_event_failed",
                                "per-item log_event crashed",
                                qid=qid,
                                err=str(log_err)[:200],
                                exc_type=type(e).__name__,
                                runId="traceback-fix",
                            )
                            # #endregion
                        if skip:
                            status(
                                f"SKIP #{qid} {title}: {err_txt[:140]}",
                                job="scrape",
                                persist=False,
                            )
                        elif infra:
                            status(
                                f"REQUEUE #{qid} {title}: {err_txt[:140]}",
                                job="scrape",
                                persist=False,
                            )
                        else:
                            status(
                                f"ERROR #{qid} {title}: {err_txt[:160]}",
                                job="scrape",
                                persist=False,
                            )
                    else:
                        infra_streak["n"] = 0
                    _publish_scrape_status(force=True)
                    _submit_more()
        finally:
            watchdog_stop.set()
            ticker_stop.set()

        with _scrape_counts_lock:
            done = _scrape_counts["done"]
            hits_total = _scrape_counts["hits"]
            errors = _scrape_counts["errors"]
        done_msg = (
            f"Done — {done}/{len(items)} videos · {hits_total} hit segments · {errors} errors "
            f"(backend={backend})."
        )
        try:
            from console_dash import is_enabled, set_done

            if is_enabled():
                set_done(hits=hits_total, errors=errors, done=done)
        except Exception:
            pass
        status(done_msg, job="scrape", persist=True)
        set_job(
            "scrape",
            status="done",
            phase="done",
            progress=100,
            completed=done,
            hits=hits_total,
            message=done_msg,
        )
        if backend == "runpod" and app_config.RUNPOD_STOP_WHEN_DONE:
            try:
                from runpod_provision import stop_pod

                status("Stopping GPU Pod to save cost…", job="scrape", persist=True)
                stop_pod()
            except Exception as e:
                status(f"pod_stop_warning: {e}", job="scrape", persist=True)
    except Exception as e:
        try:
            from console_dash import is_enabled, set_error

            if is_enabled():
                set_error(str(e)[:140])
        except Exception:
            pass
        try:
            log_event(
                "scrape_job_exception",
                job="scrape",
                exc=e,
                fatal_dashboard=True,
            )
        except Exception:
            pass
        set_job(
            "scrape",
            status="error",
            phase="error",
            message=str(e),
            error=traceback.format_exc()[-1200:],
            progress=100,
        )
    finally:
        with _lock:
            _scrape_live.clear()
            _active["scrape"] = False


def _process_one_runpod(row: dict) -> int:
    """Send one video to the auto-provisioned GPU Pod; insert candidate segments.

    Always downloads and scans the full video (no sparse/dense section slicing).
    """
    from runpod_client import process_video_remote, segments_to_candidate_rows

    qid = row["id"]
    title = row.get("title") or "video"
    url = (row.get("url") or "").strip()
    source_url = url
    t0 = time.time()
    _set_worker_phase(row, "scanning", "pod job starting…")

    def on_status(msg: str, *, phase: str | None = None) -> None:
        """Memory-only — never touch SQLite from worker/poller threads."""
        use_phase = phase
        with _lock:
            live = _scrape_live.get(qid) or {}
            if use_phase is None:
                use_phase = live.get("phase") or "scanning"
            _scrape_live[qid] = {
                "title": live.get("title") or title,
                "phase": use_phase,
                "detail": msg,
                "started": live.get("started") or time.time(),
            }

    load_env()
    thr = float(getattr(app_config, "SCORE_THRESHOLD", None) or DEFAULT_SCORE_THRESHOLD)

    on_status("full video download + scan…")
    out = process_video_remote(
        url=url,
        title=title,
        queue_id=qid,
        sample_fps=DEFAULT_FPS,
        score_threshold=thr,
        source_url=source_url,
        on_status=on_status,
        max_attempts=3,
    )

    rows = segments_to_candidate_rows(out, source_url=source_url)
    if rows:
        from openai_verify import filter_candidates_openai, openai_verify_enabled

        def on_openai_status(msg: str) -> None:
            on_status(msg, phase="uploading")

        if openai_verify_enabled():
            _set_worker_phase(row, "uploading", f"OpenAI verify · {len(rows)} stills")
            before = len(rows)
            rows = filter_candidates_openai(rows, on_status=on_openai_status)
            if before and len(rows) < before:
                on_openai_status(f"OpenAI kept {len(rows)}/{before}")
        if rows:
            _set_worker_phase(row, "uploading", f"writing {len(rows)} hit(s) to SQLite")
            # still_b64 / _local_still are consumed by insert_candidates into contact_sheets/
            insert_candidates(rows)
    else:
        _set_worker_phase(row, "uploading", "no hit segments")
        # #region agent log
        _dbg(
            "A",
            "pipeline_scrape.py:no_hit_path",
            "set no hit segments in memory",
            qid=qid,
            n_seg=len(out.get("segments") or []),
            runId="post-fix",
        )
        # #endregion
    n = len(rows)
    # Critical: leave Active UI immediately. Coordinator may stall for tens of seconds
    # (log evidence: worker returned but live "no hit" aged 50s+ with no as_completed).
    _clear_worker(qid)
    # #region agent log
    _dbg(
        "A",
        "pipeline_scrape.py:cleared_live_before_db",
        "cleared live before set_queue_status",
        qid=qid,
        n=n,
        runId="post-fix",
    )
    t_done = time.time()
    # #endregion
    try:
        _safe_queue_status(qid, "done", error="", detail=f"{n} hit segment(s)")
        # #region agent log
        _dbg(
            "A",
            "pipeline_scrape.py:after_worker_done",
            "worker set_queue_status returned",
            qid=qid,
            n=n,
            db_ms=int((time.time() - t_done) * 1000),
            runId="post-fix",
        )
        # #endregion
    except Exception as e:
        # #region agent log
        _dbg(
            "A",
            "pipeline_scrape.py:worker_done_err",
            "worker set_queue_status failed",
            qid=qid,
            err=str(e)[:200],
            db_ms=int((time.time() - t_done) * 1000),
            runId="post-fix",
        )
        # #endregion
    # #region agent log
    _dbg(
        "A",
        "pipeline_scrape.py:worker_return",
        "worker returning",
        qid=qid,
        n=n,
        runId="post-fix",
    )
    # #endregion
    return n


def _process_one(row: dict, backend: str = "local") -> int:
    """Download → scan → cloud stills → SQLite candidates → wipe local video/crops.

    When backend=runpod, offload download+scan to Serverless GPU workers.
    """
    if backend == "runpod":
        return _process_one_runpod(row)

    qid = row["id"]
    title = row.get("title") or "video"
    t0 = time.time()
    _set_worker_phase(row, "downloading", "yt-dlp starting")
    info = download_queue_item(row)
    if not info.get("path") or info.get("error"):
        err = info.get("error") or "download_failed"
        vid = info.get("video_id") or slugify(row.get("title") or "video")
        side = VIDEOS_DIR / f"{vid}.ytdlp_error.txt"
        if side.exists():
            err = f"{err}: {side.read_text(encoding='utf-8', errors='ignore')[:400]}"
        raise RuntimeError(err)

    path = Path(info["path"])
    video_id = info.get("video_id") or path.stem
    source_url = info.get("source_url") or row.get("url") or ""
    size_mb = path.stat().st_size / (1024 * 1024) if path.exists() else 0
    _set_worker_phase(row, "scanning", f"downloaded {size_mb:.1f} MB · starting scan")
    crop_dir = CROPS_DIR / video_id
    try:
        yolo, scorer = _models()

        def on_scan_progress(t_sec: float, duration: float, n_hits: int) -> None:
            if duration > 0:
                pct = min(100, int(100 * t_sec / duration))
                detail = f"{t_sec:.0f}s / {duration:.0f}s ({pct}%) · {n_hits} frame hits"
            else:
                detail = f"{t_sec:.0f}s · {n_hits} frame hits"
            with _lock:
                if qid in _scrape_live:
                    _scrape_live[qid]["detail"] = detail
                    _scrape_live[qid]["phase"] = "scanning"

        load_env()
        thr = float(getattr(app_config, "SCORE_THRESHOLD", None) or DEFAULT_SCORE_THRESHOLD)
        hits = scan_video(
            path,
            video_id=video_id,
            scorer=scorer,
            yolo=yolo,
            sample_fps=DEFAULT_FPS,
            score_threshold=thr,
            save_crops_dir=crop_dir,
            on_progress=on_scan_progress,
        )
        segs = aggregate_segments(hits, source_path="", sheet_dir=None)

        _set_worker_phase(row, "uploading", f"{len(segs)} segments · uploading stills")
        from openai_verify import (
            format_verdict_notes,
            openai_verify_enabled,
            verdict_is_keep,
            verify_still,
        )

        use_openai = openai_verify_enabled()
        cand_rows = []
        dropped_openai = 0
        for i, s in enumerate(segs, 1):
            d = asdict(s) if hasattr(s, "__dataclass_fields__") else dict(s)
            image_url = None
            notes = ""
            seg_hits = [
                h
                for h in hits
                if h.time_sec >= s.start_sec - 0.5 and h.time_sec <= s.end_sec + 0.5 and h.crop_path
            ]
            if not seg_hits:
                seg_hits = [max(hits, key=lambda h: h.score)] if hits else []
            if seg_hits:
                with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                    tmp_path = Path(tmp.name)
                wrote = write_sheet_from_crops(seg_hits, tmp_path)
                local_still = None
                if wrote:
                    if use_openai:
                        if i == 1 or i % 2 == 0 or i == len(segs):
                            _set_worker_phase(
                                row, "uploading", f"OpenAI verify {i}/{len(segs)}"
                            )
                        # Native crops (not contact sheet). Try top CLIP hits —
                        # peak score alone can be an OpenAI false drop.
                        from openai_verify import verify_stills_any

                        crop_paths = [
                            h.crop_path
                            for h in sorted(seg_hits, key=lambda x: -x.score)
                            if getattr(h, "crop_path", None)
                        ]
                        verdict = (
                            verify_stills_any(crop_paths, max_attempts=3)
                            if crop_paths
                            else verify_still(image_path=wrote)
                        )
                        notes = format_verdict_notes(verdict)
                        if not verdict_is_keep(verdict):
                            dropped_openai += 1
                    image_url = upload_image(wrote)
                    # Keep bytes until SQLite insert copies into contact_sheets/
                    local_still = wrote
                elif use_openai:
                    dropped_openai += 1
                    try:
                        tmp_path.unlink(missing_ok=True)
                    except OSError:
                        pass
                    continue
                else:
                    try:
                        tmp_path.unlink(missing_ok=True)
                    except OSError:
                        pass
            elif use_openai:
                # No still to verify — do not send to Review.
                dropped_openai += 1
                continue
            else:
                local_still = None
            row_out = {
                "video_id": video_id,
                "start_sec": d["start_sec"],
                "end_sec": d["end_sec"],
                "peak_score": d["peak_score"],
                "mean_score": d["mean_score"],
                "rank_score": d["rank_score"],
                "hit_count": d["hit_count"],
                "best_cue": d["best_cue"],
                "source_url": source_url,
                "image_url": image_url,
                "notes": notes,
            }
            if local_still:
                row_out["_local_still"] = str(local_still)
            cand_rows.append(row_out)
            if i == 1 or i % 3 == 0 or i == len(segs):
                _set_worker_phase(row, "uploading", f"stills {i}/{len(segs)}")

        if dropped_openai:
            _set_worker_phase(
                row,
                "uploading",
                f"OpenAI dropped {dropped_openai} · saved {len(cand_rows)} for Review",
            )
        if cand_rows:
            insert_candidates(cand_rows)
            for r in cand_rows:
                p = r.get("_local_still")
                if p:
                    try:
                        Path(p).unlink(missing_ok=True)
                    except OSError:
                        pass
        elapsed = time.time() - t0
        from logutil import status

        status(
            f"#{qid} {_short_title(title)} finished in {elapsed:.0f}s → {len(cand_rows)} segments",
            job="scrape",
        )
        return len(cand_rows)
    finally:
        shutil.rmtree(crop_dir, ignore_errors=True)
        delete_local_video(path)
        for p in VIDEOS_DIR.glob(f"{video_id}*"):
            if p.is_file():
                try:
                    p.unlink()
                except OSError:
                    pass
