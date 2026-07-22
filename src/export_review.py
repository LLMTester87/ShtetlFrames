"""Export human review queue CSV + pack from candidates.jsonl."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from config import CANDIDATES_PATH, CONTACT_DIR, OUTPUT_DIR, REVIEW_CSV


def load_candidates(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    rows.sort(key=lambda r: r.get("rank_score", 0), reverse=True)
    return rows


def contact_sheets_for_video(video_path: Path, items: list[tuple[int, dict]], out_dir: Path) -> None:
    """One sequential decode pass; attach contact_sheet paths onto candidate dicts."""
    if not items:
        return
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 24.0)
    needed: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for rank_i, cand in items:
        times = np.linspace(cand["start_sec"], max(cand["start_sec"], cand["end_sec"] - 0.1), 4)
        for slot, t in enumerate(times):
            needed[int(t * fps)].append((rank_i, slot))

    buffers: dict[int, dict[int, object]] = defaultdict(dict)
    max_frame = max(needed.keys()) if needed else 0
    frame_idx = 0
    while frame_idx <= max_frame:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx in needed:
            thumb = cv2.resize(frame, (320, 240))
            for rank_i, slot in needed[frame_idx]:
                buffers[rank_i][slot] = thumb
        frame_idx += 1
    cap.release()

    out_dir.mkdir(parents=True, exist_ok=True)
    for rank_i, cand in items:
        thumbs = [buffers[rank_i][s] for s in range(4) if s in buffers[rank_i]]
        if not thumbs:
            continue
        sheet = np.hstack(thumbs)
        name = f"{cand['video_id']}_{rank_i:03d}_{cand['start_sec']:.0f}-{cand['end_sec']:.0f}.jpg"
        path = out_dir / name
        cv2.imwrite(str(path), sheet)
        cand["contact_sheet"] = str(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", type=Path, default=CANDIDATES_PATH)
    ap.add_argument("--out", type=Path, default=REVIEW_CSV)
    ap.add_argument("--sheet-top", type=int, default=40)
    args = ap.parse_args()

    rows = load_candidates(args.candidates)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    top = rows[: args.sheet_top]
    by_video: dict[str, list[tuple[int, dict]]] = defaultdict(list)
    for i, r in enumerate(top):
        by_video[r.get("source_path") or ""].append((i, r))
    for src, items in by_video.items():
        if not src or not Path(src).exists():
            continue
        print(f"Contact sheets for {Path(src).name} ({len(items)} clips)")
        contact_sheets_for_video(Path(src), items, CONTACT_DIR)

    fieldnames = [
        "rank",
        "video_id",
        "start_sec",
        "end_sec",
        "peak_score",
        "mean_score",
        "rank_score",
        "hit_count",
        "best_cue",
        "source_path",
        "contact_sheet",
        "label",
        "human_accept_reject",
        "reviewer_notes",
    ]
    with args.out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for i, r in enumerate(rows, 1):
            w.writerow(
                {
                    "rank": i,
                    "video_id": r.get("video_id"),
                    "start_sec": r.get("start_sec"),
                    "end_sec": r.get("end_sec"),
                    "peak_score": r.get("peak_score"),
                    "mean_score": r.get("mean_score"),
                    "rank_score": r.get("rank_score"),
                    "hit_count": r.get("hit_count"),
                    "best_cue": r.get("best_cue"),
                    "source_path": r.get("source_path"),
                    "contact_sheet": r.get("contact_sheet", ""),
                    "label": r.get("label", "visual_candidate_not_identity"),
                    "human_accept_reject": "",
                    "reviewer_notes": "",
                }
            )

    pack = {
        "n_candidates": len(rows),
        "review_csv": str(args.out),
        "candidates_jsonl": str(args.candidates),
        "top_10": rows[:10],
        "instructions": (
            "Review contact sheets and accept/reject in review_queue.csv. "
            "Outputs are visual dress/appearance candidates only — not rabbinical identification."
        ),
    }
    pack_path = OUTPUT_DIR / "review_pack.json"
    pack_path.write_text(json.dumps(pack, indent=2), encoding="utf-8")
    print(f"Wrote {args.out} ({len(rows)} rows)")
    print(f"Wrote {pack_path}")


if __name__ == "__main__":
    main()
