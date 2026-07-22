"""Queue list/clear/delete and scrape/discover POST endpoints."""

from __future__ import annotations

from http.server import BaseHTTPRequestHandler
from urllib.parse import ParseResult, parse_qs

from api_http import json_response
from config import DEFAULT_DISCOVER_MAX, QUEUE_PAGE_SIZE
from db import clear_queue, delete_queue_url, init_db, list_queue_page, queue_stats


def handle_get_queue(handler: BaseHTTPRequestHandler, parsed: ParseResult) -> None:
    init_db()
    qs = parse_qs(parsed.query)
    offset = int((qs.get("offset") or ["0"])[0] or 0)
    limit = int((qs.get("limit") or [str(QUEUE_PAGE_SIZE)])[0] or QUEUE_PAGE_SIZE)
    status = (qs.get("status") or [""])[0].strip()
    q = (qs.get("q") or [""])[0].strip()
    page = list_queue_page(offset=offset, limit=limit, status=status, q=q)
    json_response(handler, 200, {**queue_stats(), **page})


def handle_post_queue_clear(handler: BaseHTTPRequestHandler, body: dict) -> None:
    init_db()
    clear_queue()
    from logutil import log_event

    log_event("queue_cleared", level="info", job="queue")
    json_response(handler, 200, {"ok": True, **queue_stats()})


def handle_post_queue_delete(handler: BaseHTTPRequestHandler, body: dict) -> None:
    init_db()
    ok = delete_queue_url(body.get("url", ""))
    json_response(handler, 200 if ok else 400, {"ok": ok, "error": None if ok else "not_found"})


def handle_post_queue_priority(handler: BaseHTTPRequestHandler, body: dict) -> None:
    """Jump a URL to the front of the scrape (next free worker if running)."""
    from pipeline_scrape import prioritize_queue_url

    body = body if isinstance(body, dict) else {}
    result = prioritize_queue_url(body.get("url") or "", title=str(body.get("title") or ""))
    code = 200 if result.get("ok") else 400
    json_response(handler, code, result)


def handle_post_discover(handler: BaseHTTPRequestHandler, body: dict) -> None:
    from pipeline import start_discover

    result = start_discover(
        body.get("url", ""),
        max_items=body.get("max_items", DEFAULT_DISCOVER_MAX),
    )
    code = 200 if result.get("ok") else 409
    json_response(handler, code, result)


def handle_post_scrape(handler: BaseHTTPRequestHandler, body: dict) -> None:
    from pipeline import start_scrape

    max_videos = body.get("max_videos", "all")
    workers = body.get("workers", 2)
    result = start_scrape(max_videos=max_videos, workers=workers)
    code = 200 if result.get("ok") else 409
    json_response(handler, code, result)
