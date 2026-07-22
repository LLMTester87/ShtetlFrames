"""Shared vision scoring used by local scans and the RunPod worker."""

from shtetl_core.cues import (
    CLIP_MODEL,
    CLIP_PRETRAINED,
    DEFAULT_FPS,
    DEFAULT_SCORE_THRESHOLD,
    HEADCOVER_PROMPTS,
    MAX_GAP_SEC,
    MIN_HEADCOVER_SCORE,
    MIN_PERSON_AREA,
    MIN_POS_SCORE,
    MIN_SEGMENT_SEC,
    NEGATIVE_PROMPTS,
    POSITIVE_PROMPTS,
    TOP_K_CUES,
    YOLO_CONF,
    YOLO_WEIGHTS,
)
from shtetl_core.scoring import CueScorer, FrameHit, clamp_weak_score, clamp_without_headcover
from shtetl_core.segments import (
    CandidateSegment,
    aggregate_segments,
    aggregate_segments_dicts,
    write_sheet_from_crops,
)
from shtetl_core.scan import sample_frame_indices, scan_video
from shtetl_core.textutil import slugify
from shtetl_core.upload import upload_image

__all__ = [
    "CLIP_MODEL",
    "CLIP_PRETRAINED",
    "DEFAULT_FPS",
    "DEFAULT_SCORE_THRESHOLD",
    "HEADCOVER_PROMPTS",
    "MAX_GAP_SEC",
    "MIN_HEADCOVER_SCORE",
    "MIN_PERSON_AREA",
    "MIN_POS_SCORE",
    "MIN_SEGMENT_SEC",
    "NEGATIVE_PROMPTS",
    "POSITIVE_PROMPTS",
    "TOP_K_CUES",
    "YOLO_CONF",
    "YOLO_WEIGHTS",
    "CandidateSegment",
    "CueScorer",
    "FrameHit",
    "aggregate_segments",
    "aggregate_segments_dicts",
    "clamp_weak_score",
    "clamp_without_headcover",
    "sample_frame_indices",
    "scan_video",
    "slugify",
    "upload_image",
    "write_sheet_from_crops",
]
