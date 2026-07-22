"""Rescore local munkacs_1933_yt.mp4 with current cues; OpenAI multi-crop verify."""

from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "debug-30525a.log"
sys.path.insert(0, str(ROOT / "src"))

import cv2  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from ultralytics import YOLO  # noqa: E402

from config import YOLO_WEIGHTS, load_env  # noqa: E402
from settings_store import apply_settings_to_environ  # noqa: E402
from shtetl_core.cues import (  # noqa: E402
    DEFAULT_SCORE_THRESHOLD,
    MIN_PERSON_AREA,
    YOLO_CONF,
)
from shtetl_core.scoring import CueScorer  # noqa: E402
from shtetl_core.segments import (  # noqa: E402
    aggregate_segments_dicts,
    write_sheet_from_crops,
)

VIDEO_ID = "munkacs_1933_yt"
SOURCE_URL = "https://www.youtube.com/watch?v=tdkNbcpCTc0"


def _log(hypothesis_id: str, location: str, message: str, data: dict) -> None:
    # #region agent log
    payload = {
        "sessionId": "30525a",
        "runId": "rescore-munkacs",
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data,
        "timestamp": int(time.time() * 1000),
    }
    with LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    # #endregion


def main() -> int:
    load_env()
    apply_settings_to_environ()
    load_env()

    from openai_verify import (
        format_verdict_notes,
        notes_openai_approved,
        notes_openai_dropped,
        openai_verify_enabled,
        verify_stills_any,
    )

    video = ROOT / "data" / "videos" / f"{VIDEO_ID}.mp4"
    if not video.is_file():
        print(f"missing {video} — downloading…", flush=True)
        from download import download_entry

        info = download_entry(
            SOURCE_URL,
            "Jewish Life in Munkatch - March 1933 (complete)",
            video_id=VIDEO_ID,
        )
        path = info.get("path")
        if not path or not Path(path).is_file():
            print(f"download failed: {info}", flush=True)
            return 1
        video = Path(path)

    thr = float(DEFAULT_SCORE_THRESHOLD)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"video={video.name} thr={thr} device={device}", flush=True)
    print(f"openai={openai_verify_enabled()}", flush=True)

    yolo = YOLO(YOLO_WEIGHTS)
    scorer = CueScorer(device=device)
    from shtetl_core.cues import MAX_NEG_TO_POS_RATIO, NEG_SCORE_WEIGHT

    print(
        f"NEG_SCORE_WEIGHT={NEG_SCORE_WEIGHT} MAX_NEG_TO_POS_RATIO={MAX_NEG_TO_POS_RATIO}",
        flush=True,
    )

    cap = cv2.VideoCapture(str(video))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    sample_fps = 0.5
    interval = max(1, int(round(fps / sample_fps)))
    hits = []
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        if frame_idx % interval != 0:
            frame_idx += 1
            continue
        t = frame_idx / fps
        results = yolo.predict(frame, conf=YOLO_CONF, classes=[0], verbose=False)
        frame_idx += 1
        if not results or results[0].boxes is None:
            continue
        frame_best = None
        for box in results[0].boxes:
            xyxy = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = map(int, xyxy)
            w, h = x2 - x1, y2 - y1
            if w * h < MIN_PERSON_AREA:
                continue
            y2b = y1 + max(h // 2, min(h, int(h * 0.75)))
            crop = frame[
                max(0, y1) : min(frame.shape[0], y2b),
                max(0, x1) : min(frame.shape[1], x2),
            ]
            if crop.size == 0:
                continue
            pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            score, pos_s, neg_s, cue = scorer.score_image(pil)
            if score < thr:
                continue
            from shtetl_core.scoring import FrameHit

            hit = FrameHit(
                video_id=VIDEO_ID,
                time_sec=t,
                frame_idx=frame_idx,
                score=score,
                pos_score=pos_s,
                neg_score=neg_s,
                best_cue=cue,
                bbox=[float(x1), float(y1), float(x2), float(y2)],
                crop_path=None,
            )
            hit._pil = pil  # type: ignore[attr-defined]
            if frame_best is None or hit.score > frame_best.score:
                frame_best = hit
        if frame_best is not None:
            hits.append(frame_best)
            print(
                f"HIT t={frame_best.time_sec:.1f}s score={frame_best.score:.3f} "
                f"pos={frame_best.pos_score:.3f} neg={frame_best.neg_score:.3f} "
                f"cue={frame_best.best_cue[:60]}",
                flush=True,
            )

    cap.release()
    print(f"frame_hits={len(hits)}", flush=True)
    _log(
        "M",
        "_rescore_munkacs_local.py:hits",
        "frame_hits",
        {"n": len(hits), "thr": thr, "video": video.name},
    )
    if not hits:
        print("WOULD_ENTER_REVIEW_POSITIVE=False (no CLIP hits)", flush=True)
        return 0

    segs = aggregate_segments_dicts(hits, VIDEO_ID)
    print(f"segments={len(segs)}", flush=True)

    rows = []
    with tempfile.TemporaryDirectory(prefix="munkacs_") as td:
        tmp = Path(td)
        for hi, h in enumerate(hits):
            pil = getattr(h, "_pil", None)
            if pil is None:
                continue
            cp = tmp / f"hit_{hi}_{h.time_sec:.1f}.jpg"
            pil.save(cp, quality=90)
            h.crop_path = str(cp)

        for i, seg in enumerate(segs, 1):
            group = seg.get("_hits") or []
            if not group:
                t0 = float(seg.get("start_sec") or 0)
                t1 = float(seg.get("end_sec") or t0)
                group = [
                    h
                    for h in hits
                    if h.time_sec >= t0 - 0.5 and h.time_sec <= t1 + 0.5
                ]
            sheet_path = tmp / f"seg_{i}_sheet.jpg"
            wrote = write_sheet_from_crops(group, sheet_path) if group else None
            crop_paths = [
                h.crop_path
                for h in sorted(group, key=lambda x: -x.score)
                if getattr(h, "crop_path", None)
            ]
            row = {
                "video_id": VIDEO_ID,
                "start_sec": seg.get("start_sec"),
                "end_sec": seg.get("end_sec"),
                "peak_score": seg.get("peak_score"),
                "mean_score": seg.get("mean_score"),
                "rank_score": seg.get("rank_score"),
                "hit_count": seg.get("hit_count"),
                "best_cue": seg.get("best_cue"),
                "source_url": SOURCE_URL,
                "_local_still": str(wrote) if wrote else None,
            }
            if crop_paths and openai_verify_enabled():
                v = verify_stills_any(crop_paths, max_attempts=3)
                row["notes"] = format_verdict_notes(v)
                _log(
                    "M",
                    "_rescore_munkacs_local.py:seg",
                    f"verdict_seg_{i}",
                    {
                        "seg": i,
                        "start": row["start_sec"],
                        "end": row["end_sec"],
                        "peak": row["peak_score"],
                        "verify": v.get("verified_path"),
                        "attempts": v.get("verify_attempts"),
                        "n_crops": len(crop_paths),
                        "keep": v.get("keep"),
                        "looks_jewish": v.get("looks_jewish"),
                        "head_covered": v.get("head_covered"),
                        "confidence": v.get("confidence"),
                        "notes": (row["notes"] or "")[:240],
                    },
                )
                print(
                    f"seg#{i} {row['start_sec']}-{row['end_sec']}s "
                    f"peak={row['peak_score']:.3f} {row['notes'][:200]}",
                    flush=True,
                )
            else:
                print(
                    f"seg#{i} {row['start_sec']}-{row['end_sec']}s "
                    f"peak={row['peak_score']:.3f} (no openai)",
                    flush=True,
                )
            rows.append(row)

    keeps = [i for i, r in enumerate(rows, 1) if notes_openai_approved(r.get("notes"))]
    drops = [i for i, r in enumerate(rows, 1) if notes_openai_dropped(r.get("notes"))]
    print("---", flush=True)
    print(f"OpenAI keeps: {keeps}", flush=True)
    print(f"OpenAI drops: {drops}", flush=True)
    print(f"WOULD_ENTER_REVIEW_POSITIVE={bool(keeps)}", flush=True)
    _log(
        "M",
        "_rescore_munkacs_local.py:summary",
        "rescore_done",
        {
            "keeps": keeps,
            "drops": drops,
            "would_review": bool(keeps),
            "n_segs": len(rows),
            "n_hits": len(hits),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
