"""Debug why OpenAI dropped t=20s on asset 71170 — crop vs full-frame."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "debug-30525a.log"
sys.path.insert(0, str(ROOT / "src"))

import cv2  # noqa: E402
from PIL import Image  # noqa: E402
from ultralytics import YOLO  # noqa: E402

from config import YOLO_WEIGHTS, load_env  # noqa: E402
from settings_store import apply_settings_to_environ  # noqa: E402
from shtetl_core.cues import MIN_PERSON_AREA, YOLO_CONF  # noqa: E402


def _log(
    hypothesis_id: str,
    location: str,
    message: str,
    data: dict,
    *,
    run_id: str = "post-fix",
) -> None:
    # #region agent log
    payload = {
        "sessionId": "30525a",
        "runId": run_id,
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

    from openai_verify import format_verdict_notes, verify_still
    from shtetl_core.segments import write_sheet_from_crops
    from shtetl_core.scoring import CueScorer, FrameHit

    video = ROOT / "data" / "videos" / "asset_71170.mp4"
    out_dir = ROOT / "output" / "debug_71170_t20"
    out_dir.mkdir(parents=True, exist_ok=True)

    _log("setup", "_debug_71170_t20.py:main", "start", {"video_exists": video.is_file()})
    if not video.is_file():
        print("missing video", flush=True)
        return 1

    cap = cv2.VideoCapture(str(video))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    target_t = 20.0
    frame_idx = int(round(target_t * fps))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        _log("A", "_debug_71170_t20.py", "frame_read_fail", {"frame_idx": frame_idx})
        return 1

    h, w = frame.shape[:2]
    full_path = out_dir / "full_t20.jpg"
    cv2.imwrite(str(full_path), frame)
    _log(
        "A",
        "_debug_71170_t20.py:frame",
        "full_frame_saved",
        {"path": str(full_path), "w": w, "h": h, "fps": fps, "frame_idx": frame_idx},
    )

    yolo = YOLO(YOLO_WEIGHTS)
    results = yolo.predict(frame, conf=YOLO_CONF, classes=[0], verbose=False)
    boxes = []
    if results and results[0].boxes is not None:
        for box in results[0].boxes:
            xyxy = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = map(int, xyxy)
            area = max(0, x2 - x1) * max(0, y2 - y1)
            boxes.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2, "area": area})
    _log("A", "_debug_71170_t20.py:yolo", "person_boxes", {"n": len(boxes), "boxes": boxes[:8]})

    scorer = CueScorer()
    best = None
    for b in boxes:
        x1, y1, x2, y2 = b["x1"], b["y1"], b["x2"], b["y2"]
        width, height = x2 - x1, y2 - y1
        if width * height < MIN_PERSON_AREA:
            continue
        # Same crop as scan.py
        y2b = y1 + max(height // 2, min(height, int(height * 0.75)))
        # Pad top — kippah often sits at/above YOLO y1 in profile
        y1_pad = max(0, y1 - int(0.12 * height))
        crop_std = frame[max(0, y1) : min(h, y2b), max(0, x1) : min(w, x2)]
        crop_pad = frame[y1_pad : min(h, y2b), max(0, x1) : min(w, x2)]
        if crop_std.size == 0:
            continue
        pil_std = Image.fromarray(cv2.cvtColor(crop_std, cv2.COLOR_BGR2RGB))
        score, pos_s, neg_s, cue = scorer.score_image(pil_std)
        cand = {
            "bbox": b,
            "y2b": y2b,
            "y1_pad": y1_pad,
            "score": score,
            "pos": pos_s,
            "neg": neg_s,
            "cue": cue,
            "crop_std_h": crop_std.shape[0],
            "crop_std_w": crop_std.shape[1],
            "top_margin_px": y1,  # distance from frame top to bbox top
            "pil_std": pil_std,
            "pil_pad": Image.fromarray(cv2.cvtColor(crop_pad, cv2.COLOR_BGR2RGB))
            if crop_pad.size
            else pil_std,
        }
        if best is None or score > best["score"]:
            best = cand

    if best is None:
        _log("A", "_debug_71170_t20.py", "no_person_crop", {})
        print("no person", flush=True)
        return 1

    std_path = out_dir / "crop_std_t20.jpg"
    pad_path = out_dir / "crop_pad_t20.jpg"
    best["pil_std"].save(std_path, quality=90)
    best["pil_pad"].save(pad_path, quality=90)
    _log(
        "A",
        "_debug_71170_t20.py:crop",
        "best_crop_geometry",
        {
            "bbox": best["bbox"],
            "y2b": best["y2b"],
            "y1_pad": best["y1_pad"],
            "top_margin_px": best["top_margin_px"],
            "score": best["score"],
            "pos": best["pos"],
            "neg": best["neg"],
            "cue": best["cue"][:80],
            "std_path": str(std_path),
            "pad_path": str(pad_path),
            "kippah_risk": best["top_margin_px"] < 8,
        },
    )

    # Contact-sheet path used on pod (write_sheet_from_crops)
    hit = FrameHit(
        video_id="asset_71170",
        time_sec=20.0,
        frame_idx=frame_idx,
        score=best["score"],
        pos_score=best["pos"],
        neg_score=best["neg"],
        best_cue=best["cue"],
        bbox=[
            float(best["bbox"]["x1"]),
            float(best["bbox"]["y1"]),
            float(best["bbox"]["x2"]),
            float(best["bbox"]["y2"]),
        ],
        crop_path=str(std_path),
    )
    sheet_path = out_dir / "sheet_t20.jpg"
    wrote = write_sheet_from_crops([hit], sheet_path)
    sheet_shape = None
    if wrote and Path(wrote).is_file():
        _im = cv2.imread(str(wrote))
        sheet_shape = list(_im.shape[:2]) if _im is not None else None
    _log(
        "C",
        "_debug_71170_t20.py:sheet",
        "contact_sheet",
        {
            "wrote": str(wrote) if wrote else None,
            "exists": bool(wrote and Path(wrote).is_file()),
            "shape_hw": sheet_shape,
            "runId_tag": "post-fix",
        },
    )

    for label, path, hyp in (
        ("crop_std", std_path, "A"),
        ("crop_pad", pad_path, "A"),
        ("full_frame", full_path, "B"),
        ("sheet", sheet_path if wrote else std_path, "C"),
    ):
        v = verify_still(image_path=path, timeout=45.0)
        note = format_verdict_notes(v)
        # #region agent log
        _log(
            hyp,
            "_debug_71170_t20.py:openai",
            f"verdict_{label}",
            {
                "label": label,
                "path": str(path),
                "keep": v.get("keep"),
                "looks_jewish": v.get("looks_jewish"),
                "head_covered": v.get("head_covered"),
                "confidence": v.get("confidence"),
                "uncertain": v.get("uncertain"),
                "skipped": v.get("skipped"),
                "reason": (v.get("reason") or "")[:240],
                "notes": note[:240],
                "runId_tag": "post-fix",
            },
        )
        # #endregion
        print(f"{label}: {note}", flush=True)

    print(f"logged → {LOG}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
