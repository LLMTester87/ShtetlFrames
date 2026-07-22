"""Temporal aggregation of frame hits into ranked candidate segments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from shtetl_core.cues import MAX_GAP_SEC, MAX_SEGMENTS_PER_VIDEO, MIN_SEGMENT_SEC
from shtetl_core.scoring import FrameHit


@dataclass
class CandidateSegment:
    video_id: str
    source_path: str
    start_sec: float
    end_sec: float
    peak_score: float
    mean_score: float
    hit_count: int
    best_cue: str
    rank_score: float
    contact_sheet: str | None = None
    label: str = "orthodox_dress_candidate_not_identity"


def _scale_to_height(img: np.ndarray, target_h: int = 320) -> np.ndarray:
    """Resize preserving aspect ratio (no stretch) to a common collage height."""
    h, w = img.shape[:2]
    if h <= 0 or w <= 0 or h == target_h:
        return img
    scale = target_h / float(h)
    nw = max(1, int(round(w * scale)))
    return cv2.resize(img, (nw, target_h), interpolation=cv2.INTER_AREA)


def best_crop_path(hits: list[FrameHit]) -> Path | None:
    """Highest-scoring on-disk person crop (prefer for vision verify)."""
    with_crops = [h for h in hits if h.crop_path and Path(h.crop_path).exists()]
    if not with_crops:
        return None
    best = max(with_crops, key=lambda h: h.score)
    return Path(best.crop_path)


def write_sheet_from_crops(
    hits: list[FrameHit],
    out_path: Path,
    n_thumbs: int = 4,
) -> Path | None:
    """Collage top-scoring person crops into a review still (no video seek).

    Single-crop sheets keep native resolution. Multi-crop sheets scale to a
    shared height while preserving aspect ratio — never stretch to a fixed
    WxH (that warped profile kippot and caused OpenAI false drops).
    """
    with_crops = [h for h in hits if h.crop_path and Path(h.crop_path).exists()]
    if not with_crops:
        return None
    top = sorted(with_crops, key=lambda h: -h.score)[:n_thumbs]
    top = sorted(top, key=lambda h: h.time_sec)
    thumbs: list[np.ndarray] = []
    for hit in top:
        img = cv2.imread(str(hit.crop_path))
        if img is None:
            continue
        thumbs.append(img)
    if not thumbs:
        return None
    if len(thumbs) == 1:
        sheet = thumbs[0]
    else:
        sheet = np.hstack([_scale_to_height(t, 320) for t in thumbs])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), sheet)
    return out_path


def _group_hits(
    hits: list[FrameHit],
    *,
    min_seg: float,
    max_gap: float,
) -> list[tuple[list[FrameHit], dict]]:
    """Cluster nearby hits; return (group, stats) pairs sorted by rank."""
    if not hits:
        return []
    ordered = sorted(hits, key=lambda h: h.time_sec)
    groups: list[list[FrameHit]] = [[ordered[0]]]
    for hit in ordered[1:]:
        if hit.time_sec - groups[-1][-1].time_sec <= max_gap:
            groups[-1].append(hit)
        else:
            groups.append([hit])

    packed: list[tuple[list[FrameHit], dict]] = []
    for group in groups:
        start = group[0].time_sec
        end = group[-1].time_sec
        if end - start < min_seg:
            mid = (start + end) / 2
            start = max(0.0, mid - min_seg / 2)
            end = start + min_seg
        scores = [h.score for h in group]
        size_weight = float(np.mean([(h.bbox[3] - h.bbox[1]) for h in group])) / 200.0
        size_weight = max(0.5, min(2.0, size_weight))
        peak = max(scores)
        mean = float(np.mean(scores))
        duration = max(end - start, 0.1)
        # Rank: peak × density × apparent person size × duration factor.
        rank = peak * (1.0 + 0.1 * len(group)) * size_weight * min(duration / 5.0, 2.0)
        best = max(group, key=lambda h: h.score)
        packed.append(
            (
                group,
                {
                    "video_id": group[0].video_id,
                    "start_sec": round(start, 2),
                    "end_sec": round(end, 2),
                    "peak_score": round(peak, 4),
                    "mean_score": round(mean, 4),
                    "hit_count": len(group),
                    "best_cue": best.best_cue,
                    "rank_score": round(rank, 4),
                },
            )
        )
    packed.sort(key=lambda item: item[1]["rank_score"], reverse=True)
    return packed


def aggregate_segments(
    hits: list[FrameHit],
    source_path: str,
    min_seg: float = MIN_SEGMENT_SEC,
    max_gap: float = MAX_GAP_SEC,
    sheet_dir: Path | None = None,
    max_segments: int = MAX_SEGMENTS_PER_VIDEO,
) -> list[CandidateSegment]:
    """Local pipeline shape: CandidateSegment (+ optional contact sheet)."""
    segs: list[CandidateSegment] = []
    packed = _group_hits(hits, min_seg=min_seg, max_gap=max_gap)
    if max_segments and max_segments > 0:
        packed = packed[: int(max_segments)]
    for group, stats in packed:
        contact = None
        if sheet_dir is not None:
            name = (
                f"{group[0].video_id}_{int(stats['start_sec'])}_"
                f"{int(stats['end_sec'])}_{abs(hash(round(stats['peak_score'], 4))) % 100000}.jpg"
            )
            wrote = write_sheet_from_crops(group, sheet_dir / name)
            if wrote:
                contact = str(wrote)
            else:
                best = max(group, key=lambda h: h.score)
                if best.crop_path and Path(best.crop_path).exists():
                    dest = sheet_dir / name
                    sheet_dir.mkdir(parents=True, exist_ok=True)
                    img = cv2.imread(str(best.crop_path))
                    if img is not None:
                        cv2.imwrite(str(dest), img)
                        contact = str(dest)
        segs.append(
            CandidateSegment(
                video_id=stats["video_id"],
                source_path=source_path,
                start_sec=stats["start_sec"],
                end_sec=stats["end_sec"],
                peak_score=stats["peak_score"],
                mean_score=stats["mean_score"],
                hit_count=stats["hit_count"],
                best_cue=stats["best_cue"],
                rank_score=stats["rank_score"],
                contact_sheet=contact,
            )
        )
    return segs


def aggregate_segments_dicts(
    hits: list[FrameHit],
    video_id: str,
    min_seg: float = MIN_SEGMENT_SEC,
    max_gap: float = MAX_GAP_SEC,
    max_segments: int = MAX_SEGMENTS_PER_VIDEO,
) -> list[dict]:
    """Worker shape: dict segments with `_hits` for still upload."""
    out: list[dict] = []
    packed = _group_hits(hits, min_seg=min_seg, max_gap=max_gap)
    if max_segments and max_segments > 0:
        packed = packed[: int(max_segments)]
    for group, stats in packed:
        row = dict(stats)
        row["video_id"] = video_id
        row["_hits"] = group
        out.append(row)
    return out
