"""GET /api/candidates, GET /api/review/label_stats, POST /api/review."""

from __future__ import annotations

from http.server import BaseHTTPRequestHandler
from urllib.parse import ParseResult, parse_qs

from api_http import json_response
from db import init_db, list_candidates, update_review


def load_candidates() -> list[dict]:
    init_db()
    return list_candidates(limit=5000)


def handle_get_label_stats(handler: BaseHTTPRequestHandler) -> None:
    init_db()
    try:
        from label_feedback import compute_label_stats

        json_response(handler, 200, {"ok": True, **compute_label_stats()})
    except Exception as e:
        json_response(handler, 500, {"ok": False, "error": str(e)[:240]})


def handle_get_candidates(handler: BaseHTTPRequestHandler, parsed: ParseResult) -> None:
    qs = parse_qs(parsed.query)
    rows = load_candidates()
    status = (qs.get("status") or [""])[0].lower().strip()
    openai = (qs.get("openai") or [""])[0].lower().strip()
    show_openai_drop = openai == "drop" or status == "openai_drop"
    show_openai_keep = openai == "keep" or status == "openai_keep"
    show_openai_uncertain = openai == "uncertain" or status == "openai_uncertain"
    show_pending = status == "pending"
    # Default: when OpenAI verify is on, Review shows keeps only.
    # "To check" (pending) = needs human Keep/Pass — include OpenAI drops too.
    # openai=keep / openai=drop: explicit OpenAI pass / fail filters.
    try:
        from openai_verify import (
            notes_openai_approved,
            notes_openai_dropped,
            notes_openai_uncertain,
            openai_verify_enabled,
        )

        if show_openai_drop:
            rows = [r for r in rows if notes_openai_dropped(r.get("notes"))]
        elif show_openai_keep:
            rows = [r for r in rows if notes_openai_approved(r.get("notes"))]
        elif show_openai_uncertain:
            rows = [r for r in rows if notes_openai_uncertain(r.get("notes"))]
        elif show_pending:
            # Do not hide OpenAI drops — human still may want to review them.
            pass
        elif openai_verify_enabled():
            rows = [r for r in rows if notes_openai_approved(r.get("notes"))]
    except Exception:
        pass
    q = (qs.get("q") or [""])[0].lower().strip()
    if show_pending:
        rows = [r for r in rows if not (r.get("decision") or "").strip()]
    elif not show_openai_drop and not show_openai_keep and not show_openai_uncertain:
        if status == "accept":
            rows = [r for r in rows if r.get("decision") == "accept"]
        elif status == "reject":
            rows = [r for r in rows if r.get("decision") == "reject"]
    if q:
        rows = [
            r
            for r in rows
            if q in (r.get("video_id") or "").lower()
            or q in (r.get("best_cue") or "").lower()
        ]
    limit = int((qs.get("limit") or ["2000"])[0])
    json_response(handler, 200, {"candidates": rows[:limit], "total": len(rows)})


def handle_post_review(handler: BaseHTTPRequestHandler, body: dict) -> None:
    key = body.get("key")
    decision = body.get("decision", "")
    notes = body.get("notes", "")
    if not key:
        json_response(handler, 400, {"error": "key required"})
        return
    if decision not in ("", "accept", "reject", "clear"):
        json_response(handler, 400, {"error": "invalid decision"})
        return
    try:
        cand_id = int(key)
    except (TypeError, ValueError):
        json_response(handler, 400, {"error": "key must be candidate id"})
        return
    init_db()
    decision_final = "" if decision in ("", "clear") else decision
    update_review(cand_id, decision_final, notes or "")
    # Invalidate few-shot cache so next verify picks up new Keep/Pass stills.
    try:
        from label_feedback import build_fewshot_content_parts

        build_fewshot_content_parts(force=True)
    except Exception:
        pass
    if decision_final == "accept":
        try:
            from frame_strip import enqueue_strip_for_keep

            enqueue_strip_for_keep(cand_id)
        except Exception:
            pass
    json_response(
        handler,
        200,
        {"ok": True, "key": key, "decision": decision_final},
    )
