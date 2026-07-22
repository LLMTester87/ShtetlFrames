"""Calibrate score threshold against known-good segments; document FP modes."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from ultralytics import YOLO

from config import DATA_DIR, DEFAULT_SCORE_THRESHOLD, OUTPUT_DIR, VIDEOS_DIR, YOLO_WEIGHTS
from detect import CueScorer, scan_video


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold-grid", default="0.02,0.05,0.08,0.10,0.12,0.15")
    args = ap.parse_args()

    cal_path = DATA_DIR / "calibration_segments.csv"
    if not cal_path.exists():
        raise SystemExit(f"Missing {cal_path}")

    yolo = YOLO(YOLO_WEIGHTS)
    scorer = CueScorer()

    rows = list(csv.DictReader(cal_path.open(encoding="utf-8")))
    video_ids = sorted({r["video_id"] for r in rows})
    hits_by_vid: dict[str, tuple[Path, list]] = {}
    for vid in video_ids:
        matches = [
            p
            for p in VIDEOS_DIR.glob(f"{vid}.*")
            if p.suffix.lower() in {".mp4", ".webm", ".mkv", ".avi", ".mov"}
        ]
        if not matches:
            print(f"Missing video for {vid}")
            continue
        path = matches[0]
        print(f"Scoring calibration video {vid}...")
        hits = scan_video(
            path,
            video_id=vid,
            scorer=scorer,
            yolo=yolo,
            sample_fps=1.0,
            score_threshold=-1.0,
            save_crops_dir=None,
        )
        hits_by_vid[vid] = (path, hits)
        print(f"  raw person-frame hits: {len(hits)}")

    grid = [float(x) for x in args.threshold_grid.split(",")]
    per_seg = []
    for r in rows:
        vid = r["video_id"]
        if vid not in hits_by_vid:
            continue
        _, hits = hits_by_vid[vid]
        start, end = float(r["start_sec"]), float(r["end_sec"])
        scores = [h.score for h in hits if start <= h.time_sec <= end]
        per_seg.append(
            {
                "video_id": vid,
                "start_sec": start,
                "end_sec": end,
                "label": r["label"],
                "n_hits_in_window": len(scores),
                "mean_score": float(np.mean(scores)) if scores else None,
                "max_score": float(np.max(scores)) if scores else None,
                "p50": float(np.percentile(scores, 50)) if scores else None,
            }
        )

    outside_scores = []
    for vid, (_, hits) in hits_by_vid.items():
        windows = [
            (float(r["start_sec"]), float(r["end_sec"])) for r in rows if r["video_id"] == vid
        ]
        for h in hits:
            if any(a <= h.time_sec <= b for a, b in windows):
                continue
            outside_scores.append(h.score)

    pos_means = [s["mean_score"] for s in per_seg if s["mean_score"] is not None]
    report = {
        "default_threshold": DEFAULT_SCORE_THRESHOLD,
        "positive_segment_stats": per_seg,
        "positive_mean_of_means": float(np.mean(pos_means)) if pos_means else None,
        "outside_window_score_mean": float(np.mean(outside_scores)) if outside_scores else None,
        "outside_window_n": len(outside_scores),
        "threshold_grid_recall_proxy": [],
        "recommended_threshold": DEFAULT_SCORE_THRESHOLD,
        "false_positive_modes": [
            "Blurry crowd scenes where modern coats score as kapote.",
            "Bearded clergy or academics of other religions in dark garb (rare in this corpus).",
            "Close-ups of fur hats or dark fabrics without a person (if detector misfires).",
            "Children in cheder may score positively but are not rabbis; human review must reject.",
            "Secular men in dark coats/hats in period crowds.",
        ],
        "notes": (
            "Litvish / non-Hasidic Orthodox dress is an intentional positive class "
            "(indistinguishable from Hasidic in B&W newsreels), not a false positive. "
            "Recall proxy: fraction of labeled positive windows with at least one hit above threshold. "
            "Outside-window scores are imperfect negatives because labeled windows do not cover all true positives."
        ),
    }

    best_t = DEFAULT_SCORE_THRESHOLD
    best_score = -1.0
    for t in grid:
        recalled = 0
        for s in per_seg:
            vid = s["video_id"]
            if vid not in hits_by_vid:
                continue
            _, hits = hits_by_vid[vid]
            scores = [
                h.score
                for h in hits
                if s["start_sec"] <= h.time_sec <= s["end_sec"] and h.score >= t
            ]
            if scores:
                recalled += 1
        recall = recalled / max(len(per_seg), 1)
        outside_rate = (
            float(np.mean([1 if x >= t else 0 for x in outside_scores])) if outside_scores else 0.0
        )
        utility = recall - 0.5 * outside_rate
        report["threshold_grid_recall_proxy"].append(
            {
                "threshold": t,
                "positive_windows_recalled": recalled,
                "positive_windows_total": len(per_seg),
                "recall_proxy": round(recall, 3),
                "outside_hit_rate": round(outside_rate, 3),
                "utility": round(utility, 3),
            }
        )
        if utility > best_score:
            best_score = utility
            best_t = t

    report["recommended_threshold"] = best_t
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / "calibration_report.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
