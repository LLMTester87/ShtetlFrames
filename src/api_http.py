"""HTTP response helpers for the local web server."""

from __future__ import annotations

import json
import mimetypes
from http.server import BaseHTTPRequestHandler
from pathlib import Path


def cors(handler: BaseHTTPRequestHandler) -> None:
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
    handler.send_header("Access-Control-Allow-Headers", "Content-Type")


def json_response(handler: BaseHTTPRequestHandler, code: int, payload: object) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(code)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    cors(handler)
    handler.end_headers()
    handler.wfile.write(body)


def bytes_response(
    handler: BaseHTTPRequestHandler,
    code: int,
    data: bytes,
    content_type: str,
    *,
    no_cache: bool = False,
) -> None:
    handler.send_response(code)
    handler.send_header("Content-Type", content_type)
    handler.send_header("Content-Length", str(len(data)))
    if no_cache:
        handler.send_header("Cache-Control", "no-store, max-age=0")
    cors(handler)
    handler.end_headers()
    handler.wfile.write(data)


def file_response(
    handler: BaseHTTPRequestHandler,
    path: Path,
    content_type: str | None = None,
) -> None:
    if not path.exists() or not path.is_file():
        json_response(handler, 404, {"error": "not found", "path": str(path)})
        return
    ctype = content_type or mimetypes.guess_type(str(path))[0] or "application/octet-stream"
    data = path.read_bytes()
    # Avoid stale Max=/step= constraints after UI updates
    no_cache = path.suffix.lower() in {".html", ".js", ".css"} or "html" in (ctype or "")
    bytes_response(handler, 200, data, ctype, no_cache=no_cache)


def parse_json_body(handler: BaseHTTPRequestHandler) -> dict | None:
    length = int(handler.headers.get("Content-Length") or 0)
    raw = handler.rfile.read(length) if length else b"{}"
    try:
        body = json.loads(raw.decode("utf-8") or "{}")
    except json.JSONDecodeError:
        json_response(handler, 400, {"error": "invalid json"})
        return None
    return body
