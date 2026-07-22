"""Frame sampling, person detection, CLIP cue scoring, temporal aggregation.

Implementation lives in `shtetl_core` (shared with the RunPod worker).
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

from shtetl_core import (
    CandidateSegment,
    CueScorer,
    FrameHit,
    aggregate_segments,
    sample_frame_indices,
    scan_video,
    write_sheet_from_crops,
)

__all__ = [
    "CandidateSegment",
    "CueScorer",
    "FrameHit",
    "aggregate_segments",
    "sample_frame_indices",
    "scan_video",
    "segments_to_jsonl",
    "write_contact_sheet",
    "write_sheet_from_crops",
]


def write_contact_sheet(
    video_path: Path,
    segment: CandidateSegment,
    out_path: Path,
    n_thumbs: int = 4,
) -> Path | None:
    """Build contact sheet via sequential decode (seek-safe for WebM)."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 24.0)
    targets = list(
        np.linspace(segment.start_sec, max(segment.start_sec, segment.end_sec - 0.1), n_thumbs)
    )
    target_idxs = {int(t * fps) for t in targets}
    thumbs = []
    frame_idx = 0
    max_idx = max(target_idxs) if target_idxs else 0
    while frame_idx <= max_idx:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx in target_idxs:
            thumbs.append(cv2.resize(frame, (320, 240)))
        frame_idx += 1
    cap.release()
    if not thumbs:
        return None
    sheet = np.hstack(thumbs)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), sheet)
    return out_path


def segments_to_jsonl(segments: list[CandidateSegment], path: Path, append: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append and path.exists() else "w"
    with path.open(mode, encoding="utf-8") as f:
        for seg in segments:
            f.write(json.dumps(asdict(seg), ensure_ascii=False) + "\n")
