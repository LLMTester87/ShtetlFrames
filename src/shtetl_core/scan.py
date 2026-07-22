"""Sequential video decode + YOLO person crops + CLIP scoring."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

import cv2
from PIL import Image
from ultralytics import YOLO

from shtetl_core.cues import DEFAULT_FPS, DEFAULT_SCORE_THRESHOLD, MIN_PERSON_AREA, YOLO_CONF
from shtetl_core.scoring import CueScorer, FrameHit

ProgressCallback = Callable[[float, float, int], None]


def sample_frame_indices(n_frames: int, fps: float, sample_fps: float) -> list[int]:
    if n_frames <= 0 or fps <= 0:
        return []
    step = max(1, int(round(fps / sample_fps)))
    return list(range(0, n_frames, step))


def scan_video(
    video_path: Path,
    video_id: str,
    scorer: CueScorer,
    yolo: YOLO,
    sample_fps: float = DEFAULT_FPS,
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
    save_crops_dir: Path | None = None,
    on_progress: ProgressCallback | None = None,
) -> list[FrameHit]:
    """
    Walk the video sequentially (seek is unreliable on archival WebM/AVI).
    Keep the best person crop per sampled frame above score_threshold.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 24.0)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = (n_frames / fps) if n_frames > 0 and fps > 0 else 0.0
    frame_interval = max(1, int(round(fps / max(sample_fps, 0.1))))
    hits: list[FrameHit] = []

    if save_crops_dir:
        save_crops_dir.mkdir(parents=True, exist_ok=True)

    frame_idx = 0
    last_prog = -1
    last_wall = time.time()
    if on_progress is not None:
        try:
            on_progress(0.0, duration, 0)
        except Exception:
            pass

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        if frame_idx % frame_interval != 0:
            frame_idx += 1
            continue
        time_sec = frame_idx / fps
        now = time.time()
        if on_progress is not None:
            bucket = int(time_sec // 5)
            if bucket != last_prog or (now - last_wall) >= 3.0:
                last_prog = bucket
                last_wall = now
                try:
                    on_progress(time_sec, duration, len(hits))
                except Exception:
                    pass
        results = yolo.predict(frame, conf=YOLO_CONF, classes=[0], verbose=False)
        frame_idx += 1
        if not results:
            continue
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            continue

        frame_best: FrameHit | None = None
        for box in boxes:
            xyxy = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = map(int, xyxy)
            width, height = x2 - x1, y2 - y1
            if width * height < MIN_PERSON_AREA:
                continue
            # Prefer upper body for clothing / hat / payot cues.
            y2b = y1 + max(height // 2, min(height, int(height * 0.75)))
            crop = frame[
                max(0, y1) : min(frame.shape[0], y2b),
                max(0, x1) : min(frame.shape[1], x2),
            ]
            if crop.size == 0:
                continue
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)
            score, pos_s, neg_s, cue = scorer.score_image(pil)
            if score < score_threshold:
                continue
            crop_path = None
            if save_crops_dir is not None:
                crop_path = str(save_crops_dir / f"{video_id}_{frame_idx}_{x1}_{y1}.jpg")
                pil.save(crop_path, quality=85)
            hit = FrameHit(
                video_id=video_id,
                time_sec=time_sec,
                frame_idx=frame_idx,
                score=score,
                pos_score=pos_s,
                neg_score=neg_s,
                best_cue=cue,
                bbox=[float(x1), float(y1), float(x2), float(y2)],
                crop_path=crop_path,
            )
            if frame_best is None or hit.score > frame_best.score:
                frame_best = hit
        if frame_best is not None:
            hits.append(frame_best)

    cap.release()
    return hits
