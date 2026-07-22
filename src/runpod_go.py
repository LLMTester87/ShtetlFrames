"""One-shot: save API key → start GPU pod (public image bootstrap) → start scrape.

No local Docker / Hub / image build — RunPod API key only.
"""

from __future__ import annotations

import threading
import time
from typing import Any

from config import load_env, runpod_configured
import config as app_config

_lock = threading.Lock()
_job: dict[str, Any] = {
    "status": "idle",
    "phase": "",
    "message": "",
    "log": "",
    "started_at": 0.0,
    "finished_at": 0.0,
}


def go_status() -> dict[str, Any]:
    with _lock:
        return dict(_job)


def _append(line: str) -> None:
    with _lock:
        _job["log"] = (_job["log"] + line + "\n")[-16000:]
        _job["message"] = line[:400]
    try:
        from console_dash import is_enabled, set_setup

        if is_enabled():
            set_setup(line)
    except Exception:
        pass


def _set(**kwargs: Any) -> None:
    with _lock:
        _job.update(kwargs)


def start_runpod_scrape(
    *,
    settings: dict[str, Any] | None = None,
    max_videos: Any = "all",
    workers: int = 2,
) -> dict[str, Any]:
    with _lock:
        if _job["status"] == "running":
            return {"ok": False, "error": "RunPod setup already running", "job": dict(_job)}
        from pipeline import is_scrape_running

        if is_scrape_running():
            return {"ok": False, "error": "Scrape already running", "job": dict(_job)}
        _job.update(
            {
                "status": "running",
                "phase": "starting",
                "message": "Starting RunPod…",
                "log": "",
                "started_at": time.time(),
                "finished_at": 0.0,
            }
        )

    t = threading.Thread(
        target=_run,
        kwargs={
            "settings": settings or {},
            "max_videos": max_videos,
            "workers": int(workers or 2),
        },
        daemon=True,
    )
    t.start()
    return {"ok": True, "job": go_status()}


def _run(*, settings: dict[str, Any], max_videos: Any, workers: int) -> None:
    try:
        _phase_settings(settings)
        bases = _phase_pod()
        _phase_scrape(max_videos=max_videos, workers=workers, base_urls=bases)
        _set(
            status="done",
            phase="scraping",
            message="GPU scrape started — watch progress below",
            finished_at=time.time(),
        )
    except Exception as e:
        _set(
            status="error",
            phase="error",
            message=str(e)[:600],
            finished_at=time.time(),
        )
        _append(f"ERROR: {e}")


def _phase_settings(settings: dict[str, Any]) -> None:
    _set(phase="settings", message="Saving settings…")
    from settings_store import set_settings

    payload = dict(settings or {})
    payload["SCAN_BACKEND"] = "runpod"
    set_settings(payload)
    load_env()
    if not app_config.RUNPOD_API_KEY:
        raise RuntimeError("Paste your RunPod API key in Settings first")
    _append("Settings saved — cloud GPU (no Docker on this PC)")


def _phase_pod() -> list[str]:
    load_env()
    if not runpod_configured():
        raise RuntimeError("RunPod API key required")

    from runpod_provision import MAX_PARALLEL_PODS

    n = max(1, min(int(app_config.RUNPOD_MAX_INFLIGHT or 2), MAX_PARALLEL_PODS))
    _set(phase="pod", message=f"Starting {n} GPU pod(s)…")
    _append(f"Ensuring {n} RunPod GPU pod(s)…")

    from runpod_client import set_pod_pool
    from runpod_provision import ensure_pods

    def on_status(m: str) -> None:
        _append(m)
        _set(message=m[:400])

    bases = ensure_pods(count=n, on_status=on_status, recreate=True)
    set_pod_pool(bases)
    _append(f"Pods ready ({len(bases)}): " + ", ".join(bases))
    _set(message=f"{len(bases)} GPU pod(s) ready")
    return bases


def _phase_scrape(*, max_videos: Any, workers: int, base_urls: list[str]) -> None:
    _set(phase="scrape", message="Starting scrape on GPU…")
    from pipeline import start_scrape
    from runpod_provision import MAX_PARALLEL_PODS

    n_pods = max(1, min(len(base_urls) or 1, MAX_PARALLEL_PODS))
    # ~2 HTTP jobs per pod: one can download while another holds the GPU.
    w = max(1, min(int(workers or n_pods) * 1, n_pods * 2, 8))
    if w < n_pods:
        w = n_pods
    result = start_scrape(max_videos=max_videos, workers=w)
    if not result.get("ok"):
        raise RuntimeError(result.get("error") or "Could not start scrape")
    _append(f"Scrape started (workers={w}, pods={n_pods})")
