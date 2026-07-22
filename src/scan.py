"""CLI: scan local videos for Hasidic/Orthodox visual candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ultralytics import YOLO

from config import (
    CANDIDATES_PATH,
    DEFAULT_FPS,
    DEFAULT_SCORE_THRESHOLD,
    OUTPUT_DIR,
    VIDEOS_DIR,
    YOLO_WEIGHTS,
)
from detect import (
    CueScorer,
    aggregate_segments,
    segments_to_jsonl,
    scan_video,
)


VIDEO_EXTS = {".mp4", ".webm", ".mkv", ".avi", ".mov", ".mpg", ".mpeg", ".ogv"}


def find_videos(video_dir: Path, only_ids: set[str] | None = None) -> list[tuple[str, Path]]:
    found = []
    for p in sorted(video_dir.iterdir()):
        if p.suffix.lower() not in VIDEO_EXTS:
            continue
        vid = p.stem
        # Strip IA prefixes awkwardly if needed — use stem as id
        if only_ids and vid not in only_ids and not any(vid.startswith(i) for i in only_ids):
            # match known prefixes
            matched = False
            for i in only_ids:
                if i in vid:
                    vid = i
                    matched = True
                    break
            if not matched:
                continue
        found.append((vid, p))
    return found


def main() -> None:
    ap = argparse.ArgumentParser(description="Scan videos for traditional Orthodox/Hasidic visual cues")
    ap.add_argument("--video-dir", type=Path, default=VIDEOS_DIR)
    ap.add_argument("--only", nargs="*", help="Video IDs to scan")
    ap.add_argument("--fps", type=float, default=DEFAULT_FPS)
    ap.add_argument("--threshold", type=float, default=DEFAULT_SCORE_THRESHOLD)
    ap.add_argument("--out", type=Path, default=CANDIDATES_PATH)
    ap.add_argument("--fresh", action="store_true", help="Overwrite candidates.jsonl")
    args = ap.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    only = set(args.only) if args.only else None
    videos = find_videos(args.video_dir, only)
    if not videos:
        print(f"No videos in {args.video_dir}")
        return

    if args.fresh and args.out.exists():
        args.out.unlink()

    print(f"Loading YOLO + CLIP on device...")
    yolo = YOLO(YOLO_WEIGHTS)
    scorer = CueScorer()
    print(f"Device: {scorer.device}")

    all_segments = []
    for video_id, path in videos:
        print(f"Scanning {video_id}: {path.name}", flush=True)
        hits = scan_video(
            path,
            video_id=video_id,
            scorer=scorer,
            yolo=yolo,
            sample_fps=args.fps,
            score_threshold=args.threshold,
            save_crops_dir=None,
        )
        segs = aggregate_segments(hits, source_path=str(path))
        segments_to_jsonl(segs, args.out, append=True)
        all_segments.extend(segs)
        print(f"  hits={len(hits)} segments={len(segs)}", flush=True)

    summary = {
        "videos_scanned": len(videos),
        "segments": len(all_segments),
        "threshold": args.threshold,
        "fps": args.fps,
        "candidates_path": str(args.out),
    }
    (OUTPUT_DIR / "scan_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
