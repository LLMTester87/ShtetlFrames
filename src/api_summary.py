"""GET /api/summary."""

from __future__ import annotations

from http.server import BaseHTTPRequestHandler
from urllib.parse import ParseResult

import config as app_config
from api_http import json_response
from config import (
    DEFAULT_DISCOVER_MAX,
    DISCOVER_HARD_CAP,
    QUEUE_PAGE_SIZE,
    effective_scan_backend,
    load_env,
    runpod_configured,
)
from db import candidate_stats, get_job, init_db, queue_stats


def load_summary() -> dict:
    init_db()
    load_env()
    backend = effective_scan_backend()
    summary = dict(candidate_stats())
    qstats = queue_stats()
    summary["queue"] = qstats
    summary["discover"] = get_job("discover")
    summary["scrape"] = get_job("scrape")
    # "Videos scanned" = finished queue rows (0-hit videos still count), not only hits.
    scrape_job = summary["scrape"] or {}
    try:
        summary["videos_scanned"] = int(
            scrape_job.get("completed")
            if scrape_job.get("status") == "running" and scrape_job.get("completed") is not None
            else qstats.get("n_done")
            or summary.get("videos_scanned")
            or 0
        )
    except (TypeError, ValueError):
        summary["videos_scanned"] = int(qstats.get("n_done") or 0)
    summary["discover_defaults"] = {
        "default_max": DEFAULT_DISCOVER_MAX,
        "hard_cap": DISCOVER_HARD_CAP,
        "page_size": QUEUE_PAGE_SIZE,
    }
    summary["scan"] = {
        "backend": backend,
        "requested": app_config.SCAN_BACKEND,
        "runpod_configured": runpod_configured(),
        "image_set": bool(app_config.RUNPOD_DOCKER_IMAGE),
        "pod_id": (app_config.RUNPOD_POD_ID or "")[:12],
        "gpu_type": app_config.RUNPOD_GPU_TYPE,
        "max_inflight": app_config.RUNPOD_MAX_INFLIGHT,
        "stop_when_done": app_config.RUNPOD_STOP_WHEN_DONE,
    }
    summary["archives"] = []
    return summary


def handle_get_summary(handler: BaseHTTPRequestHandler, parsed: ParseResult) -> None:
    json_response(handler, 200, load_summary())
