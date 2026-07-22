"""Rescore all Review candidates with current OpenCLIP CueScorer.

Writes output/candidate_rescore_report.json (+ .csv) with old peak vs new score.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from config import OUTPUT_DIR, load_env  # noqa: E402
from db import db, init_db  # noqa: E402
from label_feedback import _openai_tag  # noqa: E402
from openai_verify import notes_openai_approved  # noqa: E402
from shtetl_core.cues import (  # noqa: E402
    DEFAULT_SCORE_THRESHOLD,
    MIN_HEADCOVER_SCORE,
    MIN_POS_SCORE,
)
from shtetl_core.scoring import CueScorer  # noqa: E402
from still_store import candidate_still_path  # noqa: E402

OUT_JSON = OUTPUT_DIR / "candidate_rescore_report.json"
OUT_CSV = OUTPUT_DIR / "candidate_rescore_report.csv"


def _load_pil(cand_id: int, image_url: str | None):
    from PIL import Image

    path = candidate_still_path(cand_id)
    if path.is_file() and path.stat().st_size > 200:
        return Image.open(path).convert("RGB"), str(path)
    url = (image_url or "").strip()
    if url.startswith(("http://", "https://")):
        import requests

        r = requests.get(url, timeout=45)
        if r.status_code == 200 and len(r.content) > 200:
            from io import BytesIO

            return Image.open(BytesIO(r.content)).convert("RGB"), url
    return None, None


def main() -> int:
    load_env()
    init_db()
    with db() as conn:
        rows = [
            dict(r)
            for r in conn.execute(
                "SELECT id, video_id, start_sec, end_sec, peak_score, mean_score, "
                "rank_score, best_cue, decision, notes, image_url, source_url "
                "FROM candidates ORDER BY id ASC"
            ).fetchall()
        ]

    print(f"Loading CueScorer for {len(rows)} candidate(s)…", flush=True)
    scorer = CueScorer()
    thr = float(DEFAULT_SCORE_THRESHOLD)

    report_rows: list[dict] = []
    for i, d in enumerate(rows, 1):
        cid = int(d["id"])
        print(f"[{i}/{len(rows)}] #{cid}…", flush=True)
        img, src = _load_pil(cid, d.get("image_url"))
        old_peak = d.get("peak_score")
        try:
            old_peak_f = float(old_peak) if old_peak is not None else None
        except (TypeError, ValueError):
            old_peak_f = None

        row_out: dict = {
            "id": cid,
            "video_id": d.get("video_id"),
            "start_sec": d.get("start_sec"),
            "end_sec": d.get("end_sec"),
            "decision": (d.get("decision") or "") or "(pending)",
            "openai": _openai_tag(d.get("notes")) or "(none)",
            "old_peak_score": old_peak_f,
            "old_best_cue": d.get("best_cue") or "",
            "source": src,
            "image_url": d.get("image_url") or "",
            "source_url": d.get("source_url") or "",
        }
        if img is None:
            row_out.update(
                {
                    "error": "no_still",
                    "new_score": None,
                    "pos_score": None,
                    "neg_score": None,
                    "passes_new_gate": None,
                    "new_best_cue": None,
                }
            )
            report_rows.append(row_out)
            print("  SKIP no still", flush=True)
            continue

        score, pos, neg, cue = scorer.score_image(img)
        passes = score >= thr
        row_out.update(
            {
                "error": None,
                "new_score": round(score, 4),
                "pos_score": round(pos, 4),
                "neg_score": round(neg, 4),
                "passes_new_gate": passes,
                "new_best_cue": cue,
                "delta": (
                    round(score - old_peak_f, 4) if old_peak_f is not None else None
                ),
            }
        )
        report_rows.append(row_out)
        print(
            f"  old={old_peak_f} new={score:.4f} pos={pos:.3f} neg={neg:.3f} "
            f"pass={passes} decision={row_out['decision']} openai={row_out['openai']}",
            flush=True,
        )

    scored = [r for r in report_rows if r.get("new_score") is not None]
    passed = [r for r in scored if r.get("passes_new_gate")]
    failed = [r for r in scored if not r.get("passes_new_gate")]
    skipped = [r for r in report_rows if r.get("error")]

    summary = {
        "n_total": len(report_rows),
        "n_scored": len(scored),
        "n_skipped_no_still": len(skipped),
        "n_pass_new_gate": len(passed),
        "n_fail_new_gate": len(failed),
        "score_threshold": thr,
        "min_pos_score": MIN_POS_SCORE,
        "min_headcover_score": MIN_HEADCOVER_SCORE,
        "pass_rate": (len(passed) / len(scored)) if scored else None,
        "by_decision_pass": {},
        "by_openai_pass": {},
    }
    for r in scored:
        k = str(r.get("decision") or "")
        b = summary["by_decision_pass"].setdefault(k, {"n": 0, "pass": 0, "fail": 0})
        b["n"] += 1
        if r.get("passes_new_gate"):
            b["pass"] += 1
        else:
            b["fail"] += 1
    for r in scored:
        k = str(r.get("openai") or "")
        b = summary["by_openai_pass"].setdefault(k, {"n": 0, "pass": 0, "fail": 0})
        b["n"] += 1
        if r.get("passes_new_gate"):
            b["pass"] += 1
        else:
            b["fail"] += 1

    payload = {"summary": summary, "rows": report_rows}
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    fields = [
        "id",
        "video_id",
        "decision",
        "openai",
        "old_peak_score",
        "new_score",
        "delta",
        "pos_score",
        "neg_score",
        "passes_new_gate",
        "new_best_cue",
        "old_best_cue",
        "error",
        "source_url",
        "image_url",
    ]
    with OUT_CSV.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in report_rows:
            w.writerow(r)

    print("\n=== SUMMARY ===", flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    print(f"\nJSON {OUT_JSON}", flush=True)
    print(f"CSV  {OUT_CSV}", flush=True)
    print("\n=== PER CANDIDATE ===", flush=True)
    for r in report_rows:
        if r.get("error"):
            print(
                f"#{r['id']:>4}  ERROR={r['error']}  decision={r['decision']}  "
                f"openai={r['openai']}  old={r['old_peak_score']}",
                flush=True,
            )
        else:
            print(
                f"#{r['id']:>4}  new={r['new_score']:.4f}  old={r['old_peak_score']}  "
                f"Δ={r.get('delta')}  pass={r['passes_new_gate']}  "
                f"decision={r['decision']}  openai={r['openai']}  "
                f"pos={r['pos_score']} neg={r['neg_score']}",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
