"""British Pathé dedicated discover + scrape (separate from YouTube pipeline).

Discover and scrape can run at the same time: scrape continuously claims new
pending rows as discover inserts them.
"""

from __future__ import annotations

import json
import threading
import time
import traceback
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

import config as app_config

# #region agent log
_DBG_LOG = Path(__file__).resolve().parents[1] / "debug-30525a.log"
_DBG_LOCK = threading.Lock()


def _dbg(hypothesis_id: str, location: str, message: str, **data: object) -> None:
    try:
        payload = {
            "sessionId": "30525a",
            "runId": "scrape-stuck",
            "hypothesisId": hypothesis_id,
            "location": location,
            "message": message,
            "data": data,
            "timestamp": int(time.time() * 1000),
            "tid": threading.get_ident(),
        }
        with _DBG_LOCK:
            with _DBG_LOG.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(payload, default=str) + "\n")
    except Exception:
        pass


# #endregion
from config import (
    DEFAULT_FPS,
    DEFAULT_SCORE_THRESHOLD,
    DEFAULT_WORKERS,
    DISCOVER_HARD_CAP,
    MAX_WORKERS,
    effective_scan_backend,
    load_env,
)
from db import (
    get_job,
    init_db,
    insert_candidates,
    insert_queue_items,
    queue_stats_pathe,
    requeue_pathe_stuck,
    set_job,
    set_queue_status,
    take_pending_pathe,
)
from pipeline_state import _active, _lock

_pathe_live: dict[int, dict] = {}
_scrape_stop = threading.Event()


def stop_pathe_scrape(*, message: str = "Pathé scrape stopped") -> dict:
    """Signal the running Pathé scrape loop to exit and mark the job idle."""
    _scrape_stop.set()
    with _lock:
        was = bool(_active.get("pathe_scrape"))
        _active["pathe_scrape"] = False
    set_job(
        "pathe_scrape",
        status="idle",
        phase="",
        message=message[:240],
        progress=0,
        error="",
    )
    return {"ok": True, "was_running": was, "job": get_job("pathe_scrape")}


def _pathe_stack_ceiling() -> int:
    try:
        from runpod_client import pathe_stack_max

        return max(1, min(6, int(pathe_stack_max())))
    except Exception:
        try:
            return max(1, min(6, int(getattr(app_config, "PATHE_STACK_MAX", 3) or 3)))
        except Exception:
            return 3


def _pathe_client_slots(n_pods: int, stack_n: int, *, cap: int) -> int:
    """Client threads = healthy GPUs × Pathé stack (so stacking can fill idle GPUs)."""
    pods = max(1, int(n_pods or 1))
    ceiling = _pathe_stack_ceiling()
    stack = max(1, min(ceiling, int(stack_n or 1)))
    return max(1, min(pods * stack, int(cap)))


def start_pathe_discover(
    *,
    query: str = "",
    year_start: int | None = None,
    year_end: int | None = None,
    max_items: int | None = None,
    auto_scrape: bool = False,
    workers: int | None = None,
    resume: bool = True,
) -> dict:
    init_db()
    try:
        cap = int(max_items) if max_items is not None else 5000
    except (TypeError, ValueError):
        cap = 5000
    cap = max(1, min(cap, DISCOVER_HARD_CAP))

    with _lock:
        if _active.get("pathe_discover"):
            return {"ok": False, "error": "busy", "job": get_job("pathe_discover")}
        _active["pathe_discover"] = True

    # Capture page progress before we overwrite the job row (crash recovery).
    if resume:
        try:
            from britishpathe import (
                cursor_from_discover_message,
                load_discover_cursor,
                save_discover_cursor,
            )

            q = (query or "").strip()
            if not load_discover_cursor(q):
                prev = get_job("pathe_discover") or {}
                parsed = cursor_from_discover_message(
                    str(prev.get("message") or ""), query=q
                )
                if parsed and int(parsed.get("next_page") or 1) > 1:
                    save_discover_cursor(
                        query=q,
                        next_page=int(parsed["next_page"]),
                        n_found=int(parsed["n_found"]),
                    )
        except Exception:
            pass
    else:
        try:
            from britishpathe import clear_discover_cursor

            clear_discover_cursor((query or "").strip())
        except Exception:
            pass

    set_job(
        "pathe_discover",
        status="running",
        phase="discovering",
        message=f"Discovering British Pathé assets (max {cap:,})…",
        progress=5,
        discovered=0,
        total=0,
        hub_url="britishpathe.com",
        error="",
        max_videos=str(cap),
    )
    scrape_workers = workers if workers is not None else DEFAULT_WORKERS
    t = threading.Thread(
        target=_pathe_discover_job,
        args=(query, year_start, year_end, cap, auto_scrape, scrape_workers, resume),
        daemon=True,
    )
    t.start()
    return {"ok": True, "job": get_job("pathe_discover"), "auto_scrape": auto_scrape}


def _pathe_discover_job(
    query: str,
    year_start: int | None,
    year_end: int | None,
    cap: int,
    auto_scrape: bool = True,
    scrape_workers: int = DEFAULT_WORKERS,
    resume: bool = True,
) -> None:
    from britishpathe import (
        clear_discover_cursor,
        discover_catalog,
        load_discover_cursor,
    )
    from logutil import status

    discover_pod: str | None = None
    try:
        # Discover may reuse one GPU for listing — never grow a scrape fleet.
        try:
            from runpod_provision import set_pod_create_ceiling

            set_pod_create_ceiling(1)
        except Exception:
            pass

        def report(msg: str, **kw) -> None:
            status(msg, job="pathe_discover", persist=False)
            set_job("pathe_discover", message=msg, **kw)

        start_page = None
        start_found = None
        if resume:
            cur = load_discover_cursor((query or "").strip())
            if cur:
                start_page = int(cur["next_page"])
                start_found = int(cur["n_found"])
            if start_page and start_page > 1:
                report(
                    f"Resuming Pathé discover at page {start_page:,} "
                    f"({start_found or 0:,} urls already seen)…",
                    progress=8,
                )
            else:
                report("Pathé catalog discover started…", progress=8)
        else:
            clear_discover_cursor((query or "").strip())
            report("Pathé catalog discover started (from page 1)…", progress=8)
        counters = {"added": 0, "skipped": 0}
        scrape_started = {"v": False}

        # Prefer a ready RunPod for listing Scrapfly so scrape resolve on the
        # PC is not starved. Never block discover on cold pod boots — fall back
        # to local Scrapfly within a few seconds.
        if effective_scan_backend() == "runpod":
            try:
                from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutTimeout
                from runpod_client import reserve_pathe_discover_pod

                report("Finding ready Pathé discover pod (max 1)…", progress=10)

                def _reserve() -> str:
                    return reserve_pathe_discover_pod(
                        on_status=lambda m: report(m, progress=10),
                    )

                with ThreadPoolExecutor(max_workers=1) as pool:
                    fut = pool.submit(_reserve)
                    try:
                        discover_pod = fut.result(timeout=15.0)
                    except FutTimeout:
                        report(
                            "Discover pod slow — using local Scrapfly…",
                            progress=10,
                        )
                        discover_pod = None
            except Exception as e:
                report(
                    f"Discover pod unavailable ({e}) — using local Scrapfly…",
                    progress=10,
                )
                discover_pod = None

        dupe_windows = {"n": 0}
        added_at_window = {"v": 0}

        def on_batch(rows: list) -> None:
            merged = insert_queue_items(rows, hub_url="britishpathe.com")
            counters["added"] += int(merged.get("n_added") or 0)
            counters["skipped"] += int(merged.get("n_skipped") or 0)
            # Start scrape only after first URLs land — avoids Scrapfly contention
            # that was returning blank listing pages.
            # Auto-scrape spins RunPod GPUs via ensure_pods — off unless explicitly enabled.
            if auto_scrape and not scrape_started["v"] and counters["added"] > 0:
                scrape_started["v"] = True
                start_pathe_scrape(
                    max_videos="all",
                    workers=scrape_workers,
                    allow_empty=True,
                    retry_errors=False,
                )

        def on_window(_stats: dict) -> bool:
            """Stop discover when scrape is running and listing is 100% dupes."""
            gained = counters["added"] - added_at_window["v"]
            added_at_window["v"] = counters["added"]
            if gained > 0:
                dupe_windows["n"] = 0
                return True
            # Window produced no new queue rows (all already queued or blank).
            dupe_windows["n"] += 1
            scraping = bool(_active.get("pathe_scrape"))
            if scraping and dupe_windows["n"] >= 3:
                report(
                    "Pausing discover — 3 windows with +0 new "
                    "(freeing Scrapfly for Pathé resolve)…",
                    progress=min(90, 10 + (counters["added"] + counters["skipped"]) % 80),
                    discovered=counters["added"],
                )
                return False
            return True

        def on_status(msg: str) -> None:
            # Keep page progress, but always show real queue inserts (else UI
            # looks "stuck" when listing pages are mostly duplicates).
            seen = counters["added"] + counters["skipped"]
            suffix = (
                f" · +{counters['added']:,} new"
                f" · {counters['skipped']:,} already queued"
                if seen
                else ""
            )
            report(
                f"{msg}{suffix}",
                progress=min(90, 10 + max(seen, 1) % 80),
                discovered=counters["added"],
                total=max(seen, counters["added"]),
            )

        result = discover_catalog(
            query=query or "",
            year_start=year_start,
            year_end=year_end,
            max_items=cap,
            on_status=on_status,
            on_batch=on_batch,
            on_window=on_window,
            remote_base=discover_pod,
            start_page=start_page,
            start_found=start_found,
            resume=resume,
        )
        entries = result.get("entries") or []
        source_note = ""
        # Listing often blanks under Scrapfly load — fall back to local catalog,
        # then map @britishpathe YouTube titles → /asset/ URLs by name search.
        if not entries:
            from britishpathe import discover_from_youtube_titles, load_local_catalog
            from db import list_youtube_pathe_titles

            report(
                "Listing empty — seeding from local Pathé catalog…",
                progress=40,
            )
            local = load_local_catalog(max_items=cap)
            if local:
                source_note = " · from local catalog"
                on_batch(local)
                entries = local
                result = {
                    **result,
                    "entries": local,
                    "n": len(local),
                    "source": "local_catalog",
                    "concurrency": result.get("concurrency") or 1,
                }
            else:
                report(
                    "No local catalog — searching Pathé by YouTube titles…",
                    progress=45,
                )
                yt_titles = list_youtube_pathe_titles(limit=max(cap * 3, 500))
                yt_result = discover_from_youtube_titles(
                    yt_titles,
                    max_items=cap,
                    on_status=on_status,
                    on_batch=on_batch,
                )
                entries = yt_result.get("entries") or []
                source_note = " · from YouTube titles"
                result = {**result, **yt_result}
        added = counters["added"]
        skipped = counters["skipped"]
        if entries and added == 0 and skipped == 0:
            merged = insert_queue_items(entries, hub_url="britishpathe.com")
            added = int(merged.get("n_added") or 0)
            skipped = int(merged.get("n_skipped") or 0)
        err_note = ""
        errs = result.get("errors") or []
        if errs:
            err_note = f" · {len(errs)} page error(s): {errs[0][:80]}"
        conc = result.get("concurrency") or 1
        if len(entries) == 0 and not err_note:
            err_note = " · listing returned no /asset/ links (Scrapfly/ASP blank?)"
        final_msg = (
            f"Found {len(entries):,} · added {added:,} new · "
            f"skipped {skipped:,} dupes · {conc}× parallel"
            f"{source_note}{err_note}"
        )
        set_job(
            "pathe_discover",
            status="done" if entries else "error",
            phase="done" if entries else "error",
            message=final_msg,
            progress=100,
            discovered=added,
            total=len(entries),
            error=(errs[0] if errs else "")[:500],
        )
        status(final_msg, job="pathe_discover", persist=False)
        if not entries:
            try:
                from console_dash import is_enabled, set_error

                if is_enabled():
                    set_error(final_msg)
            except Exception:
                pass
    except Exception as e:
        set_job(
            "pathe_discover",
            status="error",
            phase="error",
            message=str(e)[:200],
            error=traceback.format_exc()[-1200:],
            progress=100,
        )
        try:
            from console_dash import is_enabled, set_error

            if is_enabled():
                set_error(str(e)[:200])
        except Exception:
            pass
    finally:
        if discover_pod:
            try:
                from runpod_client import release_pathe_discover_pod

                release_pathe_discover_pod()
            except Exception:
                pass
        with _lock:
            _active["pathe_discover"] = False
            scraping = bool(_active.get("pathe_scrape"))
        # Restore normal pod cap unless scrape already owns the fleet.
        if not scraping:
            try:
                from runpod_provision import set_pod_create_ceiling

                set_pod_create_ceiling(None)
            except Exception:
                pass


def start_pathe_scrape(
    max_videos: str | int = "all",
    workers: int = DEFAULT_WORKERS,
    *,
    allow_empty: bool = False,
    retry_errors: bool = True,
) -> dict:
    """Start (or no-op if already running) continuous Pathé scrape.

    Pulls new pending rows as discover inserts them until discover finishes
    and the pending queue is empty.
    """
    init_db()
    load_env()
    backend = effective_scan_backend()
    with _lock:
        if _active.get("pathe_scrape"):
            return {
                "ok": True,
                "already_running": True,
                "job": get_job("pathe_scrape"),
                "backend": backend,
            }
        _active["pathe_scrape"] = True
        _scrape_stop.clear()

    try:
        if backend == "runpod":
            from runpod_provision import (
                MAX_PARALLEL_PODS,
                set_pod_create_ceiling,
                set_pod_creates_blocked,
            )

            # Scrape must be allowed to grow the GPU fleet to match workers.
            # Discover may set POD_CREATES_BLOCKED / ceiling=1 — clear for scrape.
            set_pod_creates_blocked(False)
            set_pod_create_ceiling(None)

            requested = max(1, min(int(workers or DEFAULT_WORKERS), MAX_WORKERS))
            # Raise stored MAX_INFLIGHT when UI workers ask for a bigger fleet.
            try:
                from settings_store import set_settings

                cur = int(app_config.RUNPOD_MAX_INFLIGHT or 1)
                if requested > cur:
                    set_settings(
                        {"RUNPOD_MAX_INFLIGHT": str(min(requested, MAX_PARALLEL_PODS))}
                    )
                    load_env()
            except Exception:
                pass

            max_inflight = max(
                1,
                min(
                    max(int(app_config.RUNPOD_MAX_INFLIGHT or requested), requested),
                    MAX_PARALLEL_PODS,
                ),
            )
            # Client slots = pods × stack. Cap at pod count only starves stacking.
            stack_ceil = _pathe_stack_ceiling()
            try:
                from runpod_client import pathe_stack_limit

                stack_hint = max(1, min(stack_ceil, pathe_stack_limit()))
            except Exception:
                stack_hint = stack_ceil
            workers_cap_hint = max(
                1, min(max_inflight * stack_ceil, MAX_PARALLEL_PODS * stack_ceil)
            )
            workers = _pathe_client_slots(
                max_inflight, stack_hint, cap=workers_cap_hint
            )
        else:
            workers = max(1, min(int(workers or DEFAULT_WORKERS), MAX_WORKERS))

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

        # Reclaim stuck in-flight / error rows left by a dead pod or crashed scrape.
        requeue_pathe_stuck()

        pending = int(queue_stats_pathe().get("n_pending") or 0)
        discovering = bool(_active.get("pathe_discover"))
        if pending <= 0 and not discovering and not allow_empty:
            with _lock:
                _active["pathe_scrape"] = False
            set_job(
                "pathe_scrape",
                status="error",
                phase="error",
                message="No pending Pathé assets. Run Discover on this page first.",
                progress=100,
            )
            return {"ok": False, "error": "queue_empty", "job": get_job("pathe_scrape")}

        from db import queue_claim_from_end

        order_label = "end→newest" if queue_claim_from_end() else "start→oldest"
        set_job(
            "pathe_scrape",
            status="running",
            phase="scraping",
            message=(
                f"Pathé scrape · continuous · backend={backend} · workers={workers} · "
                f"queue={order_label}"
                + (
                    " · waiting for discover…"
                    if pending <= 0
                    else f" · {pending} pending"
                )
            ),
            progress=2,
            total=pending,
            completed=0,
            hits=0,
            error="",
            max_videos=max_label,
            workers=workers,
        )
        t = threading.Thread(
            target=_pathe_scrape_job,
            args=(workers, backend, limit),
            daemon=True,
        )
        t.start()
        return {"ok": True, "job": get_job("pathe_scrape"), "backend": backend}
    except Exception:
        with _lock:
            _active["pathe_scrape"] = False
        raise


_last_live_touch = 0.0


def _pathe_live_rows() -> list[dict]:
    return [
        {
            "title": (info.get("title") or ""),
            "phase": (info.get("phase") or ""),
            "detail": (info.get("detail") or ""),
        }
        for info in list(_pathe_live.values())[:8]
    ]


def _touch_pathe_live_dash() -> None:
    """Push in-flight worker detail to the console ~2×/sec during long scans."""
    global _last_live_touch
    now = time.monotonic()
    if now - _last_live_touch < 0.5:
        return
    _last_live_touch = now
    try:
        from console_dash import is_enabled, touch_live

        if not is_enabled():
            return
        with _lock:
            live = _pathe_live_rows()
        touch_live(live)
    except Exception:
        pass


def _publish_pathe_dash(
    *,
    completed: int,
    hits: int,
    errors: int,
    total: int,
    pending: int = 0,
    reset_session: bool = False,
) -> None:
    try:
        from console_dash import is_enabled, set_scrape

        if not is_enabled():
            return
        with _lock:
            live = _pathe_live_rows()
            discovering = bool(_active.get("pathe_discover"))
        set_scrape(
            done=completed,
            total=max(total, 1),
            hits=hits,
            errors=errors,
            live=live,
            headline=(
                "Discover + scrape running together…"
                if discovering
                else "Looking through British Pathé…"
            ),
            sub=(
                f"{pending:,} more waiting from discover…"
                if pending > 0
                else (f"{hits} clip(s) found so far" if hits else "")
            ),
            reset_session=reset_session,
        )
    except Exception:
        pass


def _pathe_scrape_job(workers: int, backend: str, limit: int | None) -> None:
    from logutil import status

    completed = 0
    hits = 0
    errors = 0
    claimed = 0
    try:
        if backend == "runpod":
            from runpod_client import set_pod_pool
            from runpod_provision import MAX_PARALLEL_PODS, ensure_pods

            # GPU count = Settings Parallel GPU pods (synced from UI workers on start).
            # ``workers`` arg may be pods×stack client slots — do not use it as pod count.
            n_pods = max(
                1,
                min(
                    int(app_config.RUNPOD_MAX_INFLIGHT or 4),
                    MAX_PARALLEL_PODS,
                ),
            )

            def pod_status(msg: str) -> None:
                status(msg, job="pathe_scrape")
                set_job("pathe_scrape", message=msg)

            set_job(
                "pathe_scrape",
                message=f"Spinning up {n_pods} RunPod GPU pod(s)…",
                progress=3,
            )
            # #region agent log
            _dbg(
                "A",
                "pipeline_pathe.py:_pathe_scrape_job",
                "ensure_pods_enter",
                n_pods=n_pods,
                min_ready=1,
                extra_fill_sec=0,
            )
            # #endregion
            bases = ensure_pods(
                count=n_pods,
                on_status=pod_status,
                min_ready=1,
                # ensure_pods fills remaining pods in a background thread
                extra_fill_sec=0,
            )
            # #region agent log
            _dbg(
                "B",
                "pipeline_pathe.py:_pathe_scrape_job",
                "ensure_pods_exit",
                n_bases=len(bases or []),
                n_pods=n_pods,
            )
            # #endregion
            set_pod_pool(bases)
            status(
                f"{len(bases)}/{n_pods} GPU pod(s) ready — Pathé scrape starting",
                job="pathe_scrape",
                persist=True,
            )
            pending0 = int(queue_stats_pathe().get("n_pending") or 0)
            _publish_pathe_dash(
                completed=0,
                hits=0,
                errors=0,
                total=max(pending0, 1),
                pending=pending0,
                reset_session=True,
            )

            def _fill_remaining_pods() -> None:
                """Block until pool is full (or capacity runs out); expand workers via pool."""

                def _fill_status(msg: str) -> None:
                    # Console/log only — do not overwrite the live scrape job message.
                    status(msg, job="pathe_scrape", persist=False)

                try:
                    # Wait for a full pool (min_ready=n) — do not return after the first pod.
                    more = ensure_pods(
                        count=n_pods,
                        on_status=_fill_status,
                        min_ready=max(1, n_pods),
                        extra_fill_sec=1200,
                    )
                    set_pod_pool(more)
                    status(
                        f"Pod pool expanded to {len(more)}/{n_pods}",
                        job="pathe_scrape",
                        persist=True,
                    )
                except Exception as e:
                    status(
                        f"Background pod fill: {e}",
                        job="pathe_scrape",
                        persist=True,
                    )

            if len(bases) < n_pods:
                threading.Thread(
                    target=_fill_remaining_pods,
                    daemon=True,
                    name="pathe-runpod-fill",
                ).start()
            # Adaptive stack — client threads must track healthy×stack or idle
            # GPUs stay empty while others double-up.
            stack_ceil = _pathe_stack_ceiling()
            try:
                from runpod_client import pathe_stack_limit

                stack_n = max(1, min(stack_ceil, pathe_stack_limit()))
            except Exception:
                stack_n = stack_ceil
            workers_cap = max(
                1, min(n_pods * stack_ceil, MAX_PARALLEL_PODS * stack_ceil)
            )
            workers = _pathe_client_slots(len(bases) or 1, stack_n, cap=workers_cap)
            set_job(
                "pathe_scrape",
                message=(
                    f"Pathé scrape · {len(bases)}/{n_pods} GPU pod(s) · "
                    f"workers={workers} (stack≤{stack_n})"
                ),
                workers=workers,
                progress=5,
            )
        else:
            workers_cap = max(1, workers)

        # Size pool at hard ceiling so raising PATHE_STACK_MAX mid-run can grow
        # logical ``workers`` without recreating the executor.
        executor_max = max(
            workers_cap,
            MAX_PARALLEL_PODS * 6 if backend == "runpod" else workers_cap,
        )

        with ThreadPoolExecutor(max_workers=executor_max) as pool:
            futs: dict = {}
            last_pool_sync = 0.0
            while True:
                if _scrape_stop.is_set():
                    status("Pathé scrape stop requested", job="pathe_scrape")
                    break
                # Re-read stack ceiling every loop so Settings raises apply live.
                if backend == "runpod":
                    try:
                        from runpod_client import (
                            get_pod_pool,
                            maintain_pod_pool,
                            pathe_stack_limit,
                            pathe_stack_max,
                        )

                        load_env()
                        # Touch ceiling so AIMD limit jumps when UI raises it.
                        stack_ceil = max(1, min(6, int(pathe_stack_max())))
                        n_pods = max(
                            1,
                            min(
                                int(app_config.RUNPOD_MAX_INFLIGHT or 4),
                                MAX_PARALLEL_PODS,
                            ),
                        )
                        workers_cap = max(
                            1,
                            min(n_pods * stack_ceil, MAX_PARALLEL_PODS * stack_ceil),
                        )
                        # Self-heal dead/broken GPUs every few seconds.
                        if time.monotonic() - last_pool_sync >= 3.0:
                            last_pool_sync = time.monotonic()

                            def _heal_status(msg: str) -> None:
                                if "self-heal" in (msg or "").lower():
                                    status(msg, job="pathe_scrape", persist=True)

                            more = (
                                maintain_pod_pool(
                                    target=n_pods, on_status=_heal_status
                                )
                                or []
                            )
                            if not more:
                                more = get_pod_pool()
                        else:
                            more = get_pod_pool()
                        stack_n = max(1, min(stack_ceil, pathe_stack_limit()))
                        healthy = max(1, len(more) or len(get_pod_pool()) or 1)
                        new_workers = _pathe_client_slots(
                            healthy, stack_n, cap=workers_cap
                        )
                        if new_workers != workers:
                            workers = new_workers
                            set_job(
                                "pathe_scrape",
                                workers=workers,
                                message=(
                                    f"Pathé scrape · {healthy}/{n_pods} GPU pod(s) · "
                                    f"workers={workers} (stack≤{stack_n}/{stack_ceil})"
                                ),
                            )
                            status(
                                f"Using {workers} client slots on {healthy} GPUs "
                                f"(stack≤{stack_n}/{stack_ceil})",
                                job="pathe_scrape",
                                persist=True,
                            )
                    except Exception:
                        pass

                slots = max(0, min(workers, executor_max) - len(futs))
                if slots > 0 and (limit is None or claimed < limit):
                    take_n = slots if limit is None else min(slots, limit - claimed)
                    batch = take_pending_pathe(take_n, only_pending=True)
                    # #region agent log
                    if claimed == 0 or batch:
                        _dbg(
                            "B",
                            "pipeline_pathe.py:_pathe_scrape_job",
                            "claim_batch",
                            slots=slots,
                            take_n=take_n,
                            batch=len(batch or []),
                            claimed=claimed,
                            workers=workers,
                            futs=len(futs),
                        )
                    # #endregion
                    for row in batch:
                        futs[pool.submit(_process_one_pathe, row, backend)] = row
                        claimed += 1

                if futs:
                    done, _ = wait(futs.keys(), return_when=FIRST_COMPLETED, timeout=1.0)
                    for fut in done:
                        row = futs.pop(fut)
                        try:
                            n = fut.result()
                            hits += int(n or 0)
                            completed += 1
                        except Exception as e:
                            err_s = str(e)
                            from runpod_client import is_infra_error

                            soft = is_infra_error(err_s) or any(
                                x in err_s.lower()
                                for x in (
                                    "britishpathe_resolve",
                                    "pathe_resolve",
                                    "scrapfly",
                                    "pathe_no_m3u8",
                                )
                            )
                            if soft:
                                # Pod blip or Scrapfly resolve contention — retry later.
                                set_queue_status(
                                    row["id"],
                                    "pending",
                                    error="",
                                    detail=f"retry_soft: {err_s[:80]}",
                                )
                                status(
                                    f"Pathé retry later ({err_s[:60]})…",
                                    job="pathe_scrape",
                                    persist=False,
                                )
                                claimed = max(0, claimed - 1)
                            else:
                                errors += 1
                                completed += 1
                                set_queue_status(
                                    row["id"], "error", error=err_s[:500]
                                )
                                status(
                                    f"Pathé error: {e}"[:160],
                                    job="pathe_scrape",
                                    persist=True,
                                )
                        pending_left = int(queue_stats_pathe().get("n_pending") or 0)
                        total_est = max(claimed + pending_left, completed)
                        set_job(
                            "pathe_scrape",
                            completed=completed,
                            hits=hits,
                            total=total_est,
                            progress=min(
                                99,
                                5 + 90 * completed / max(1, total_est),
                            ),
                            message=(
                                f"Pathé scrape {completed}/{total_est} · "
                                f"{hits} hits · {pending_left} pending"
                            ),
                        )
                        _publish_pathe_dash(
                            completed=completed,
                            hits=hits,
                            errors=errors,
                            total=total_est,
                            pending=pending_left,
                        )
                    continue

                # No in-flight work — wait for discover to enqueue more, or finish.
                discovering = bool(_active.get("pathe_discover"))
                pending_left = int(queue_stats_pathe().get("n_pending") or 0)
                if discovering or pending_left > 0:
                    if limit is not None and claimed >= limit:
                        break
                    set_job(
                        "pathe_scrape",
                        message=(
                            f"Pathé scrape idle · waiting for discover "
                            f"({pending_left} pending)…"
                            if discovering
                            else f"Pathé scrape · claiming {pending_left} pending…"
                        ),
                        total=max(claimed + pending_left, completed),
                    )
                    _publish_pathe_dash(
                        completed=completed,
                        hits=hits,
                        errors=errors,
                        total=max(claimed + pending_left, completed, 1),
                        pending=pending_left,
                    )
                    time.sleep(0.25)
                    continue
                break

        set_job(
            "pathe_scrape",
            status="done",
            phase="done",
            message=f"Pathé scrape done · {completed} videos · {hits} hits",
            progress=100,
            completed=completed,
            hits=hits,
        )
        try:
            from console_dash import is_enabled, set_done

            if is_enabled():
                set_done(hits=hits, errors=errors, done=completed)
        except Exception:
            pass
    except Exception as e:
        set_job(
            "pathe_scrape",
            status="error",
            phase="error",
            message=str(e)[:200],
            error=traceback.format_exc()[-1200:],
            progress=100,
        )
        try:
            from console_dash import is_enabled, set_error

            if is_enabled():
                set_error(str(e)[:200])
        except Exception:
            pass
    finally:
        with _lock:
            _pathe_live.clear()
            _active["pathe_scrape"] = False


def _process_one_pathe(row: dict, backend: str) -> int:
    from runpod_client import process_pathe_remote, segments_to_candidate_rows
    from shtetl_core.textutil import slugify

    qid = row["id"]
    title = row.get("title") or "video"
    url = (row.get("url") or "").strip()
    thr = float(getattr(app_config, "SCORE_THRESHOLD", None) or DEFAULT_SCORE_THRESHOLD)

    def on_status(msg: str, *, phase: str | None = None) -> None:
        with _lock:
            live = _pathe_live.get(qid) or {}
            _pathe_live[qid] = {
                "title": live.get("title") or title,
                "phase": phase or live.get("phase") or "scanning",
                "detail": msg,
                "started": live.get("started") or time.time(),
            }
        _touch_pathe_live_dash()

    set_queue_status(qid, "scanning", detail="pathe")
    on_status("Pathé HLS download + scan…")

    if backend == "runpod":
        out = process_pathe_remote(
            url=url,
            title=title,
            queue_id=qid,
            sample_fps=DEFAULT_FPS,
            score_threshold=thr,
            source_url=url,
            on_status=on_status,
            max_attempts=8,
        )
        rows = segments_to_candidate_rows(out, source_url=url)
    else:
        from ultralytics import YOLO

        from config import YOLO_WEIGHTS
        from detect import (
            CueScorer,
            aggregate_segments,
            scan_video,
            write_sheet_from_crops,
        )
        from download import download_britishpathe
        from still_ensure import extract_frame

        vid = slugify(title)
        path = download_britishpathe(url, app_config.VIDEOS_DIR, vid, title=title)
        if not path:
            raise RuntimeError("britishpathe_local_download_failed")
        on_status("scanning locally…")
        import shutil
        import tempfile
        from pathlib import Path

        crop_dir = Path(tempfile.mkdtemp(prefix=f"pathe_crops_{vid}_"))
        yolo = YOLO(YOLO_WEIGHTS)
        scorer = CueScorer()
        rows = []
        try:
            hits = scan_video(
                path,
                video_id=vid,
                scorer=scorer,
                yolo=yolo,
                sample_fps=DEFAULT_FPS,
                score_threshold=thr,
                save_crops_dir=crop_dir,
            )
            segs = aggregate_segments(hits, source_path="", sheet_dir=None)
            rows = []
            for s in segs:
                d = s.__dict__ if hasattr(s, "__dict__") else dict(s)
                row_out = {
                    "video_id": vid,
                    "start_sec": d.get("start_sec"),
                    "end_sec": d.get("end_sec"),
                    "peak_score": d.get("peak_score"),
                    "mean_score": d.get("mean_score"),
                    "rank_score": d.get("rank_score"),
                    "hit_count": d.get("hit_count"),
                    "best_cue": d.get("best_cue"),
                    "source_url": url,
                }
                # Durable local still (same as YouTube local scrape path).
                try:
                    t0 = float(d.get("start_sec") or 0)
                    t1 = float(d.get("end_sec") or t0)
                    seg_hits = [
                        h
                        for h in hits
                        if h.time_sec >= t0 - 0.5
                        and h.time_sec <= t1 + 0.5
                        and getattr(h, "crop_path", None)
                    ]
                    if not seg_hits and hits:
                        best = max(hits, key=lambda h: h.score)
                        if getattr(best, "crop_path", None):
                            seg_hits = [best]
                    wrote = None
                    if seg_hits:
                        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                            tmp_path = Path(tmp.name)
                        wrote = write_sheet_from_crops(seg_hits, tmp_path)
                    if not wrote:
                        mid = t0 if t1 <= t0 else (t0 + t1) / 2.0
                        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                            tmp_path = Path(tmp.name)
                        if extract_frame(Path(path), mid, tmp_path):
                            wrote = tmp_path
                    if wrote:
                        row_out["_local_still"] = str(wrote)
                except Exception:
                    pass
                rows.append(row_out)
        finally:
            shutil.rmtree(crop_dir, ignore_errors=True)

    if rows:
        # still_b64 / _local_still (if present) are saved by insert_candidates
        insert_candidates(rows)
    set_queue_status(qid, "done", detail=f"hits={len(rows)}")
    with _lock:
        _pathe_live.pop(qid, None)
    return len(rows)


def pathe_live_snapshot() -> list[dict]:
    with _lock:
        return list(_pathe_live.values())


def pathe_summary() -> dict:
    init_db()
    return {
        "queue": queue_stats_pathe(),
        "discover": get_job("pathe_discover"),
        "scrape": get_job("pathe_scrape"),
        "live": pathe_live_snapshot(),
    }
