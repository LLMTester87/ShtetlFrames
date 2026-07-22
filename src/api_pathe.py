"""British Pathé dedicated page APIs."""

from __future__ import annotations

from http.server import BaseHTTPRequestHandler
from urllib.parse import ParseResult, parse_qs

from api_http import json_response
from config import DEFAULT_DISCOVER_MAX, QUEUE_PAGE_SIZE
from db import clear_queue_pathe, init_db, list_queue_page_pathe, queue_stats_pathe


def handle_get_summary(handler: BaseHTTPRequestHandler, parsed: ParseResult) -> None:
    from pipeline_pathe import pathe_summary

    json_response(handler, 200, pathe_summary())


def handle_get_queue(handler: BaseHTTPRequestHandler, parsed: ParseResult) -> None:
    init_db()
    qs = parse_qs(parsed.query)
    offset = int((qs.get("offset") or ["0"])[0] or 0)
    limit = int((qs.get("limit") or [str(QUEUE_PAGE_SIZE)])[0] or QUEUE_PAGE_SIZE)
    status = (qs.get("status") or [""])[0].strip()
    q = (qs.get("q") or [""])[0].strip()
    page = list_queue_page_pathe(offset=offset, limit=limit, status=status, q=q)
    json_response(handler, 200, {**queue_stats_pathe(), **page})


def handle_post_discover(handler: BaseHTTPRequestHandler, body: dict) -> None:
    from pipeline_pathe import start_pathe_discover

    body = body if isinstance(body, dict) else {}
    max_items = body.get("max_items", DEFAULT_DISCOVER_MAX)
    query = str(body.get("query") or body.get("q") or "").strip()
    year_start = body.get("year_start")
    year_end = body.get("year_end")
    try:
        ys = int(year_start) if year_start not in (None, "", "+") else None
    except (TypeError, ValueError):
        ys = None
    try:
        ye = int(year_end) if year_end not in (None, "", "+") else None
    except (TypeError, ValueError):
        ye = None
    # "Discover all" button: empty query + full year range (defaults inside discover_catalog).
    if body.get("all"):
        query = ""
        ys = None
        ye = None
    auto_scrape = body.get("auto_scrape", True)
    if isinstance(auto_scrape, str):
        auto_scrape = auto_scrape.strip().lower() not in ("0", "false", "no")
    result = start_pathe_discover(
        query=query,
        year_start=ys,
        year_end=ye,
        max_items=max_items,
        auto_scrape=bool(auto_scrape),
        workers=body.get("workers"),
    )
    code = 200 if result.get("ok") else 409
    json_response(handler, code, result)


def handle_post_scrape(handler: BaseHTTPRequestHandler, body: dict) -> None:
    from pipeline_pathe import start_pathe_scrape

    body = body if isinstance(body, dict) else {}
    result = start_pathe_scrape(
        max_videos=body.get("max_videos", "all"),
        workers=body.get("workers", 2),
    )
    code = 200 if result.get("ok") else 409
    json_response(handler, code, result)


def handle_post_queue_clear(handler: BaseHTTPRequestHandler, body: dict) -> None:
    init_db()
    n = clear_queue_pathe()
    json_response(handler, 200, {"ok": True, "deleted": n, **queue_stats_pathe()})
