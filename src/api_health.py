"""GET /api/health — live ops snapshot: pods, jobs, queue, alerts."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from http.server import BaseHTTPRequestHandler
from typing import Any
from urllib.parse import ParseResult

import requests

from api_http import json_response
from config import load_env
import config as app_config


def probe_pod(pod: dict) -> dict[str, Any]:
    """Probe one RunPod worker /health + /progress."""
    from runpod_provision import pod_proxy_url

    pid = pod.get("id") or ""
    name = pod.get("name") or (pid[:12] if pid else "?")
    base = pod_proxy_url(pid) if pid else ""
    entry: dict[str, Any] = {
        "id": pid,
        "name": name,
        "desiredStatus": pod.get("desiredStatus"),
        "imageName": pod.get("imageName"),
        "base_url": base,
        "healthy": False,
        "models_ready": False,
        "ok": False,
        "inflight": None,
        "inflight_limit_pathe": None,
        "inflight_limit_yt": None,
        "phase": None,
        "message": "",
        "title": "",
        "queue_id": None,
        "pct": None,
        "busy": False,
        "error": None,
        "sync_error": None,
        "ollama_ready": None,
        "ollama_model_ready": None,
        "ollama_model": None,
        "ollama_pulling": None,
    }
    if not base:
        entry["error"] = "no_proxy_url"
        return entry

    try:
        hr = requests.get(f"{base.rstrip('/')}/health", timeout=8)
        if hr.status_code == 200 and hr.content:
            data = hr.json() if hr.content else {}
            if not isinstance(data, dict):
                data = {}
            entry["ok"] = bool(data.get("ok"))
            entry["models_ready"] = bool(data.get("models_ready"))
            entry["healthy"] = entry["ok"] and entry["models_ready"]
            entry["inflight"] = data.get("inflight")
            entry["inflight_limit_pathe"] = data.get("inflight_limit_pathe")
            entry["inflight_limit_yt"] = data.get("inflight_limit_yt")
            ollama = data.get("ollama") if isinstance(data.get("ollama"), dict) else {}
            if ollama:
                entry["ollama_ready"] = bool(ollama.get("ready"))
                entry["ollama_model_ready"] = bool(ollama.get("model_ready"))
                entry["ollama_model"] = (ollama.get("model") or "")[:64] or None
                entry["ollama_pulling"] = bool(ollama.get("pulling"))
            if data.get("warm_error"):
                entry["error"] = str(data.get("warm_error"))[:200]
            sync = data.get("github_sync") if isinstance(data.get("github_sync"), dict) else {}
            if sync.get("last_error"):
                entry["sync_error"] = str(sync.get("last_error"))[:160]
            prog = data.get("progress") if isinstance(data.get("progress"), dict) else {}
            if prog:
                entry["phase"] = prog.get("phase") or entry["phase"]
                entry["message"] = (prog.get("message") or "")[:80]
                entry["title"] = (prog.get("title") or "")[:80]
                entry["queue_id"] = prog.get("queue_id")
                entry["pct"] = prog.get("pct")
        else:
            entry["error"] = f"http_{hr.status_code}"
    except Exception as e:
        entry["error"] = str(e)[:200]

    try:
        pr = requests.get(f"{base.rstrip('/')}/progress", timeout=8)
        if pr.status_code == 200 and pr.content:
            pj = pr.json() if pr.content else {}
            if isinstance(pj, dict):
                entry["phase"] = pj.get("phase") or entry["phase"] or "idle"
                entry["message"] = (pj.get("message") or entry["message"] or "")[:80]
                entry["title"] = (pj.get("title") or entry["title"] or "")[:80]
                entry["queue_id"] = pj.get("queue_id")
                entry["pct"] = pj.get("pct")
    except Exception:
        if entry["phase"] is None and not entry["error"]:
            entry["error"] = "progress_unreachable"

    phase = (entry.get("phase") or "idle").strip().lower()
    inf = entry.get("inflight")
    entry["busy"] = phase not in ("", "idle", "done") or (
        isinstance(inf, int) and inf > 0
    )
    if entry["phase"] is None:
        entry["phase"] = "idle" if entry["healthy"] else "unknown"
    return entry


def build_health_snapshot() -> dict[str, Any]:
    from db import init_db, list_jobs, queue_stats_pathe
    from runpod_client import (
        get_pod_pool,
        pathe_stack_limit,
        pathe_stack_max,
        pool_size,
    )
    from runpod_provision import MAX_PARALLEL_PODS, find_shtetl_pods

    load_env()
    init_db()
    t0 = time.time()

    pods_raw = find_shtetl_pods()
    probes: list[dict] = []
    with ThreadPoolExecutor(max_workers=max(1, min(12, len(pods_raw) or 1))) as ex:
        futs = [ex.submit(probe_pod, p) for p in pods_raw if p.get("id")]
        for fut in as_completed(futs):
            try:
                probes.append(fut.result())
            except Exception as e:
                probes.append(
                    {
                        "name": "?",
                        "healthy": False,
                        "busy": False,
                        "error": str(e)[:160],
                        "phase": "unknown",
                    }
                )
    probes.sort(key=lambda r: (r.get("name") or ""))

    jobs = list_jobs()
    pathe_q = queue_stats_pathe()
    try:
        scrape_pool = get_pod_pool()
        pool_n = pool_size()
    except Exception:
        scrape_pool = []
        pool_n = 0

    stack = pathe_stack_limit()
    stack_max = pathe_stack_max()
    max_inflight = int(getattr(app_config, "RUNPOD_MAX_INFLIGHT", None) or 8)
    healthy_n = sum(1 for p in probes if p.get("healthy"))
    busy_n = sum(1 for p in probes if p.get("busy"))
    ollama_ready_n = sum(1 for p in probes if p.get("ollama_model_ready"))
    idle_healthy = [
        p for p in probes if p.get("healthy") and not p.get("busy")
    ]
    dead = [p for p in probes if not p.get("healthy")]

    pathe_scrape = jobs.get("pathe_scrape") or {}
    scrape_running = (pathe_scrape.get("status") or "") == "running"
    pending = int(pathe_q.get("n_pending") or 0)

    alerts: list[dict[str, str]] = []

    def alert(level: str, code: str, msg: str) -> None:
        alerts.append({"level": level, "code": code, "message": msg})

    for p in dead:
        nm = p.get("name") or "?"
        err = p.get("error") or "unhealthy"
        alert("red", "pod_dead", f"{nm}: {err}")

    for p in probes:
        if p.get("healthy") and not p.get("models_ready"):
            alert("amber", "models_not_ready", f"{p.get('name')}: models not ready")
        if p.get("healthy") and p.get("ollama_model_ready") is False:
            pulling = " (pulling…)" if p.get("ollama_pulling") else ""
            alert(
                "amber",
                "ollama_not_ready",
                f"{p.get('name')}: Ollama model not ready{pulling}",
            )
        if p.get("sync_error"):
            alert(
                "amber",
                "sync_error",
                f"{p.get('name')}: sync {p.get('sync_error')}",
            )

    if pending > 0 and not scrape_running:
        alert(
            "amber",
            "scrape_idle",
            f"Pathé scrape not running with {pending} pending",
        )

    if scrape_running and pending > 0 and idle_healthy:
        names = ", ".join((p.get("name") or "?") for p in idle_healthy[:4])
        extra = f" (+{len(idle_healthy) - 4})" if len(idle_healthy) > 4 else ""
        alert(
            "amber",
            "idle_gpus",
            f"{len(idle_healthy)} healthy GPU(s) idle while {pending} pending: {names}{extra}",
        )

    if stack <= 1 and scrape_running and pending > 20:
        alert(
            "amber",
            "stack_scaled_down",
            "Pathé client stack scaled to 1 (recovering from overload)",
        )

    pod_lims = [
        int(p["inflight_limit_pathe"])
        for p in probes
        if isinstance(p.get("inflight_limit_pathe"), int)
    ]
    if pod_lims and stack_max > min(pod_lims) and scrape_running:
        alert(
            "amber",
            "worker_stack_lag",
            f"Client stack max {stack_max} but pods report Pathé inflight≤{min(pod_lims)} — sync worker entry.py",
        )

    live_n = len(probes)
    if live_n > 0 and pool_n < live_n and scrape_running:
        alert(
            "amber",
            "pool_short",
            f"Scrape pool has {pool_n} URL(s) but {live_n} live pod(s)",
        )

    if live_n < max_inflight and scrape_running:
        alert(
            "amber",
            "under_provisioned",
            f"{live_n}/{max_inflight} pods live (want {max_inflight})",
        )

    level_rank = {"red": 0, "amber": 1}
    alerts.sort(key=lambda a: (level_rank.get(a["level"], 9), a["code"]))

    return {
        "ok": True,
        "ts": time.time(),
        "probe_ms": int((time.time() - t0) * 1000),
        "alerts": alerts,
        "summary": {
            "pod_count": live_n,
            "healthy_count": healthy_n,
            "busy_count": busy_n,
            "idle_healthy_count": len(idle_healthy),
            "ollama_ready_count": ollama_ready_n,
            "max_inflight": max_inflight,
            "max_parallel_pods": MAX_PARALLEL_PODS,
            "scrape_pool_size": pool_n,
            "pathe_stack": stack,
            "pathe_stack_max": stack_max,
        },
        "pool": {
            "urls": [u.split("//")[-1][:40] for u in scrape_pool],
            "size": pool_n,
        },
        "jobs": {
            "pathe_scrape": _job_brief(jobs.get("pathe_scrape")),
            "pathe_discover": _job_brief(jobs.get("pathe_discover")),
            "scrape": _job_brief(jobs.get("scrape")),
            "discover": _job_brief(jobs.get("discover")),
        },
        "queue": {"pathe": pathe_q},
        "pods": probes,
    }


def _job_brief(job: dict | None) -> dict[str, Any]:
    if not isinstance(job, dict):
        return {"status": "none"}
    return {
        "status": job.get("status") or "",
        "phase": job.get("phase") or "",
        "message": (job.get("message") or "")[:160],
        "workers": job.get("workers"),
        "completed": job.get("completed"),
        "total": job.get("total"),
        "hits": job.get("hits"),
        "progress": job.get("progress"),
        "error": (job.get("error") or "")[:160],
    }


def handle_get_health(handler: BaseHTTPRequestHandler, parsed: ParseResult) -> None:
    try:
        snap = build_health_snapshot()
        json_response(handler, 200, snap)
    except Exception as e:
        json_response(
            handler,
            500,
            {"ok": False, "error": str(e)[:400], "alerts": [], "pods": []},
        )
