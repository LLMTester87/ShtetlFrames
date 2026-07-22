"""Short ±2s crop strips: queue, status, list for download page."""

from __future__ import annotations

from http.server import BaseHTTPRequestHandler
from urllib.parse import ParseResult, parse_qs

from api_http import json_response
from db import db, init_db


def _enrich(items: list[dict]) -> list[dict]:
    """Attach candidate metadata when available."""
    ids = [int(x["id"]) for x in items if x.get("id") is not None]
    meta: dict[int, dict] = {}
    if ids:
        init_db()
        placeholders = ",".join("?" * len(ids))
        with db() as conn:
            rows = conn.execute(
                f"SELECT id, video_id, start_sec, end_sec, source_url, decision, best_cue "
                f"FROM candidates WHERE id IN ({placeholders})",
                ids,
            ).fetchall()
        meta = {int(r["id"]): dict(r) for r in rows}
    out = []
    for item in items:
        cid = int(item["id"])
        row = dict(item)
        m = meta.get(cid) or {}
        row["video_id"] = m.get("video_id") or ""
        row["start_sec"] = m.get("start_sec")
        row["end_sec"] = m.get("end_sec")
        row["source_url"] = m.get("source_url") or ""
        row["decision"] = m.get("decision") or ""
        row["best_cue"] = m.get("best_cue") or ""
        out.append(row)
    return out


def handle_get_crops(handler: BaseHTTPRequestHandler, parsed: ParseResult) -> None:
    from frame_strip import crop_status, list_crop_jobs

    qs = parse_qs(parsed.query)
    cid_raw = (qs.get("id") or [""])[0].strip()
    if cid_raw:
        try:
            cid = int(cid_raw)
        except ValueError:
            json_response(handler, 400, {"ok": False, "error": "id must be int"})
            return
        st = crop_status(cid)
        rows = _enrich([{**st, "id": cid}])
        json_response(handler, 200, {"ok": True, "crop": rows[0]})
        return
    items = _enrich(list_crop_jobs())
    json_response(handler, 200, {"ok": True, "crops": items, "total": len(items)})


def handle_post_crop(handler: BaseHTTPRequestHandler, body: dict) -> None:
    from frame_strip import crop_status, enqueue_crop_for_candidate

    raw = body.get("id") if isinstance(body, dict) else None
    if raw is None:
        raw = body.get("key") if isinstance(body, dict) else None
    try:
        cid = int(raw)
    except (TypeError, ValueError):
        json_response(handler, 400, {"ok": False, "error": "id required"})
        return

    force = bool(body.get("force")) if isinstance(body, dict) else False
    st = crop_status(cid)
    if not force and st["status"] == "ready":
        rows = _enrich([{**st, "id": cid}])
        json_response(
            handler,
            200,
            {"ok": True, "queued": False, "already": True, "crop": rows[0]},
        )
        return
    if st["status"] == "queued":
        rows = _enrich([{**st, "id": cid}])
        json_response(
            handler,
            200,
            {"ok": True, "queued": False, "already": True, "crop": rows[0]},
        )
        return

    init_db()
    with db() as conn:
        row = conn.execute("SELECT id FROM candidates WHERE id=?", (cid,)).fetchone()
    if not row:
        json_response(handler, 404, {"ok": False, "error": "candidate_not_found"})
        return

    result = enqueue_crop_for_candidate(cid, force=force)
    rows = _enrich([{**result, "id": cid}])
    json_response(
        handler,
        200,
        {
            "ok": True,
            "queued": bool(result.get("queued")),
            "already": False,
            "crop": rows[0],
        },
    )
