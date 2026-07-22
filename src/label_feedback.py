"""Human Keep/Pass feedback for OpenAI verify — stats + few-shot stills."""

from __future__ import annotations

import base64
import json
import os
import re
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Any

# Cap vision few-shots so verify payloads stay small.
# Bias toward Pass / false-keep examples — soft CLIP floods Pathé false positives.
_FEWSHOT_KEEP = 3
_FEWSHOT_PASS = 3
_FEWSHOT_CACHE_TTL_SEC = 300.0
_REASON_SAMPLE = 8

_cache_lock = threading.Lock()
_fewshot_cache: dict[str, Any] = {"built_at": 0.0, "parts": [], "meta": {}}


def _openai_tag(notes: str | None) -> str:
    """Return 'keep', 'drop', 'uncertain', 'skip', or '' (openai: or vlm:)."""
    for line in (notes or "").splitlines():
        low = line.strip().lower()
        for prov in ("openai:", "vlm:"):
            if low.startswith(prov + "keep"):
                return "keep"
            if low.startswith(prov + "uncertain"):
                return "uncertain"
            if low.startswith(prov + "drop"):
                return "drop"
            if low.startswith(prov + "skip"):
                return "skip"
    return ""


def _openai_reason(notes: str | None) -> str:
    for line in (notes or "").splitlines():
        s = line.strip()
        low = s.lower()
        for prefix in (
            "openai:keep",
            "openai:drop",
            "openai:uncertain",
            "openai:skip",
            "vlm:keep",
            "vlm:drop",
            "vlm:uncertain",
            "vlm:skip",
        ):
            if low.startswith(prefix):
                rest = s[len(prefix) :].strip()
                rest = re.sub(r"^conf=\d+(?:\.\d+)?\s*", "", rest, flags=re.I)
                return rest[:160]
    return ""


def _bucket(decision: str, ai: str) -> str:
    d = (decision or "").strip().lower()
    if d not in ("accept", "reject"):
        return "pending"
    if ai == "keep":
        return "agree_keep" if d == "accept" else "false_keep"
    if ai == "drop":
        return "false_drop" if d == "accept" else "agree_drop"
    if ai == "uncertain":
        return "uncertain_labeled"
    return "labeled_no_openai"


def compute_label_stats(*, reason_samples: int = _REASON_SAMPLE) -> dict[str, Any]:
    """Agreement between human decision and openai:keep/drop notes."""
    from db import db, init_db

    init_db()
    with db() as conn:
        rows = conn.execute(
            "SELECT id, decision, notes, image_url, video_id, peak_score "
            "FROM candidates ORDER BY id DESC"
        ).fetchall()

    counts: Counter[str] = Counter()
    false_keep_reasons: Counter[str] = Counter()
    false_drop_reasons: Counter[str] = Counter()
    false_keep_ids: list[int] = []
    false_drop_ids: list[int] = []

    for r in rows:
        d = dict(r)
        ai = _openai_tag(d.get("notes"))
        b = _bucket(str(d.get("decision") or ""), ai)
        counts[b] += 1
        reason = _openai_reason(d.get("notes")) or "(no reason)"
        if b == "false_keep":
            false_keep_reasons[reason] += 1
            if len(false_keep_ids) < reason_samples:
                false_keep_ids.append(int(d["id"]))
        elif b == "false_drop":
            false_drop_reasons[reason] += 1
            if len(false_drop_ids) < reason_samples:
                false_drop_ids.append(int(d["id"]))

    labeled = (
        counts["agree_keep"]
        + counts["agree_drop"]
        + counts["false_keep"]
        + counts["false_drop"]
    )
    agree = counts["agree_keep"] + counts["agree_drop"]
    agreement_rate = (agree / labeled) if labeled else None

    return {
        "n_total": len(rows),
        "n_labeled": labeled,
        "n_pending": counts["pending"],
        "agree_keep": counts["agree_keep"],
        "agree_drop": counts["agree_drop"],
        "false_keep": counts["false_keep"],
        "false_drop": counts["false_drop"],
        "uncertain_labeled": counts["uncertain_labeled"],
        "labeled_no_openai": counts["labeled_no_openai"],
        "agreement_rate": agreement_rate,
        "false_keep_reasons": false_keep_reasons.most_common(12),
        "false_drop_reasons": false_drop_reasons.most_common(12),
        "sample_false_keep_ids": false_keep_ids,
        "sample_false_drop_ids": false_drop_ids,
    }


def min_keep_confidence() -> float:
    try:
        v = float(os.environ.get("OPENAI_MIN_KEEP_CONF") or "0.70")
    except (TypeError, ValueError):
        v = 0.70
    return max(0.0, min(0.95, v))


def apply_confidence_gate(verdict: dict[str, Any]) -> dict[str, Any]:
    """Downgrade low-confidence keeps to uncertain (not auto-shown in Review keeps)."""
    if verdict.get("skipped"):
        return verdict
    out = dict(verdict)
    conf = float(out.get("confidence") or 0.0)
    floor = min_keep_confidence()
    if out.get("keep") and conf < floor:
        out["keep"] = False
        out["uncertain"] = True
        prior = str(out.get("reason") or "")
        out["reason"] = f"low_conf<{floor:.2f} {prior}".strip()[:240]
    return out


def _still_bytes_for_candidate(cand_id: int, image_url: str | None = None) -> tuple[bytes, str] | None:
    from still_store import candidate_still_path

    path = candidate_still_path(int(cand_id))
    if path.is_file() and path.stat().st_size > 200:
        raw = path.read_bytes()
        if raw[:3] == b"\xff\xd8\xff":
            return raw, "image/jpeg"
        if raw[:8] == b"\x89PNG\r\n\x1a\n":
            return raw, "image/png"
        return raw, "image/jpeg"
    url = (image_url or "").strip()
    if url.startswith(("http://", "https://")):
        try:
            from openai_verify import _fetch_image_bytes

            return _fetch_image_bytes(url)
        except Exception:
            return None
    return None


def _select_labeled_examples(
    *,
    decision: str,
    limit: int,
    prefer_false_keep: bool = False,
    prefer_false_drop: bool = False,
) -> list[dict[str, Any]]:
    """Recent Keep/Pass rows that have a usable still.

    prefer_false_keep (Pass examples): AI keep + human Pass — teach what NOT to keep.
    prefer_false_drop (Keep examples): AI drop + human Keep — teach real Hasidic stills
    the model wrongly rejected (shtreimel / Orthodox fedora).
    """
    from db import db, init_db

    init_db()
    with db() as conn:
        rows = conn.execute(
            "SELECT id, decision, notes, image_url, video_id FROM candidates "
            "WHERE decision=? ORDER BY id DESC LIMIT ?",
            (decision, max(limit * 16, 48)),
        ).fetchall()

    scored: list[tuple[int, dict[str, Any]]] = []
    for r in rows:
        d = dict(r)
        got = _still_bytes_for_candidate(int(d["id"]), d.get("image_url"))
        if not got:
            continue
        raw, mime = got
        # Prefer compact JPEGs for few-shot.
        if len(raw) > 1_200_000:
            continue
        notes = str(d.get("notes") or "")
        low = notes.lower()
        ai = _openai_tag(notes)
        priority = 0
        if prefer_false_keep:
            # Best teaching signal: model kept, human passed.
            if ai == "keep":
                priority += 100
            if "head=no" in low or "bare" in low:
                priority += 20
            if any(
                w in low
                for w in ("coat", "secular", "suit", "uniform", "mitre", "clergy")
            ):
                priority += 10
        elif prefer_false_drop:
            # Best teaching signal: model dropped, human kept (false drops).
            if ai == "drop":
                priority += 120
            if "head=no" in low and "jewish=yes" in low:
                priority += 40  # classic shtreimel/fedora miss
            if any(w in low for w in ("shtreimel", "fur", "spodik", "fedora", "brim")):
                priority += 20
            if ai == "keep" and "head=yes" in low:
                priority += 30  # also include clear true keeps
        else:
            # Prefer clear head-covered keeps; skip bare-head openai keeps.
            if ai == "keep" and "head=yes" in low:
                priority += 100
            elif ai == "keep":
                priority += 40
            if any(w in low for w in ("shtreimel", "yarmulke", "black hat", "fedora")):
                priority += 15
            if "head=no" in low or "bare" in low:
                priority -= 50
        scored.append(
            (
                priority,
                {
                    "id": int(d["id"]),
                    "decision": decision,
                    "raw": raw,
                    "mime": mime,
                    "notes": notes,
                },
            )
        )

    scored.sort(key=lambda t: (-t[0], -t[1]["id"]))
    return [item for _, item in scored[:limit]]


def build_fewshot_content_parts(
    *,
    n_keep: int = _FEWSHOT_KEEP,
    n_pass: int = _FEWSHOT_PASS,
    force: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Vision content parts: labeled examples then the live still is appended by caller."""
    flag = (os.environ.get("OPENAI_FEWSHOT") or "1").strip().lower()
    if flag in ("0", "false", "off", "no", "none"):
        return [], {"enabled": False, "n_keep": 0, "n_pass": 0}

    now = time.monotonic()
    with _cache_lock:
        if (
            not force
            and _fewshot_cache["parts"]
            and (now - float(_fewshot_cache["built_at"])) < _FEWSHOT_CACHE_TTL_SEC
        ):
            return list(_fewshot_cache["parts"]), dict(_fewshot_cache["meta"])

    try:
        keeps = _select_labeled_examples(
            decision="accept", limit=n_keep, prefer_false_drop=True
        )
        passes = _select_labeled_examples(
            decision="reject", limit=n_pass, prefer_false_keep=True
        )
    except Exception as e:
        return [], {"enabled": True, "error": str(e)[:160], "n_keep": 0, "n_pass": 0}

    parts: list[dict[str, Any]] = []
    if keeps or passes:
        parts.append(
            {
                "type": "text",
                "text": (
                    "Examples labeled by a human reviewer for this project. "
                    "KEEP = person overall looks Jewish/Orthodox AND wears a Jewish "
                    "head covering — including shtreimel/spodik (fur Hasidic hat) and "
                    "Orthodox black fedora. Fur shtreimel counts. "
                    "PASS = not a match — bare heads, secular Pathé crowds, uniforms, other faiths. "
                    "Match this bar."
                ),
            }
        )
    for ex in keeps:
        b64 = base64.standard_b64encode(ex["raw"]).decode("ascii")
        parts.append(
            {
                "type": "text",
                "text": (
                    f"Example KEEP (human): candidate #{ex['id']} — "
                    "Orthodox/Hasidic look with Jewish head covering "
                    "(yarmulke / Orthodox hat / shtreimel/spodik)."
                ),
            }
        )
        parts.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:{ex['mime']};base64,{b64}"},
            }
        )
    for ex in passes:
        b64 = base64.standard_b64encode(ex["raw"]).decode("ascii")
        parts.append(
            {
                "type": "text",
                "text": (
                    f"Example PASS / reject (human): candidate #{ex['id']} — "
                    "do NOT keep stills like this."
                ),
            }
        )
        parts.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:{ex['mime']};base64,{b64}"},
            }
        )

    meta = {
        "enabled": True,
        "n_keep": len(keeps),
        "n_pass": len(passes),
        "keep_ids": [x["id"] for x in keeps],
        "pass_ids": [x["id"] for x in passes],
    }
    with _cache_lock:
        _fewshot_cache["built_at"] = now
        _fewshot_cache["parts"] = list(parts)
        _fewshot_cache["meta"] = meta
    return parts, meta


def export_fewshot_cache(path: Path | str | None = None) -> Path | None:
    """Write a JSON manifest of few-shot ids (no image bytes) for debugging."""
    from config import OUTPUT_DIR

    parts, meta = build_fewshot_content_parts(force=True)
    dest = Path(path) if path else OUTPUT_DIR / "label_fewshot_meta.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(
        json.dumps({"meta": meta, "n_parts": len(parts)}, indent=2),
        encoding="utf-8",
    )
    return dest
