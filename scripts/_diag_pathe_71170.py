"""Diagnose why asset 71170 scored 0 hits — log unclamped CLIP gates on person crops."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import cv2  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from ultralytics import YOLO  # noqa: E402

from config import VIDEOS_DIR, YOLO_WEIGHTS, load_env  # noqa: E402
from download import download_britishpathe  # noqa: E402
from shtetl_core.cues import (  # noqa: E402
    HEADCOVER_PROMPTS,
    MAX_NEG_TO_POS_RATIO,
    MIN_HEADCOVER_SCORE,
    MIN_PERSON_AREA,
    MIN_POS_SCORE,
    NEGATIVE_PROMPTS,
    NEG_SCORE_WEIGHT,
    POSITIVE_PROMPTS,
    YOLO_CONF,
)
from shtetl_core.scoring import CueScorer  # noqa: E402


def main() -> int:
    load_env()
    url = "https://www.britishpathe.com/asset/71170/"
    print("downloading…", flush=True)
    path = download_britishpathe(url, VIDEOS_DIR, "asset_71170", title="asset-71170")
    if not path:
        print("download failed", flush=True)
        return 1
    path = Path(path)
    print(f"video: {path} ({path.stat().st_size // 1024} KB)", flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"loading YOLO+CLIP on {device}…", flush=True)
    yolo = YOLO(YOLO_WEIGHTS)
    scorer = CueScorer(device=device)

    cap = cv2.VideoCapture(str(path))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 24.0)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = n_frames / fps if n_frames and fps else 0
    # Denser than pod 0.5 fps for diagnosis.
    sample_fps = 1.0
    interval = max(1, int(round(fps / sample_fps)))
    print(f"duration≈{duration:.1f}s fps={fps:.2f} sample={sample_fps} interval={interval}", flush=True)

    best = None
    n_people = 0
    n_scored = 0
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
        if not results or results[0].boxes is None or len(results[0].boxes) == 0:
            continue
        for box in results[0].boxes:
            xyxy = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = map(int, xyxy)
            w, h = x2 - x1, y2 - y1
            if w * h < MIN_PERSON_AREA:
                continue
            n_people += 1
            y2b = y1 + max(h // 2, min(h, int(h * 0.75)))
            crop = frame[max(0, y1) : min(frame.shape[0], y2b), max(0, x1) : min(frame.shape[1], x2)]
            if crop.size == 0:
                continue
            pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            # Raw similarities (before clamps).
            image = scorer.preprocess(pil).unsqueeze(0).to(scorer.device)
            with torch.no_grad():
                img_feat = scorer.model.encode_image(image)
                img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
                pos_sims = (img_feat @ scorer.pos_feat.T).squeeze(0)
                neg_sims = (img_feat @ scorer.neg_feat.T).squeeze(0)
                head_sims = (img_feat @ scorer.head_feat.T).squeeze(0)
                pos_score = float(pos_sims.max().item())
                neg_score = float(torch.topk(neg_sims, min(3, neg_sims.numel())).values.mean().item())
                head_score = float(head_sims.max().item())
                best_pos = POSITIVE_PROMPTS[int(pos_sims.argmax().item())]
                best_head = HEADCOVER_PROMPTS[int(head_sims.argmax().item())]
                best_neg = NEGATIVE_PROMPTS[int(neg_sims.argmax().item())]
            raw = pos_score - NEG_SCORE_WEIGHT * neg_score
            gated_score, _, _, _ = scorer.score_image(pil)
            n_scored += 1
            row = {
                "t": t,
                "raw": raw,
                "gated": gated_score,
                "pos": pos_score,
                "neg": neg_score,
                "head": head_score,
                "best_pos": best_pos,
                "best_head": best_head,
                "best_neg": best_neg,
                "fail_pos": pos_score < MIN_POS_SCORE,
                "fail_head": head_score < MIN_HEADCOVER_SCORE,
                "fail_neg": (neg_score / pos_score) >= MAX_NEG_TO_POS_RATIO if pos_score > 1e-6 else True,
                "pass_thr": gated_score >= 0.08,
            }
            if best is None or row["gated"] > best["gated"] or (
                row["gated"] == best["gated"] and row["raw"] > best["raw"]
            ):
                best = row
            if n_scored <= 8 or row["pass_thr"] or (row["head"] >= 0.14 and row["pos"] >= 0.15):
                print(
                    f"t={t:.1f}s gated={gated_score:.3f} raw={raw:.3f} "
                    f"pos={pos_score:.3f} head={head_score:.3f} neg={neg_score:.3f} "
                    f"fail_pos={row['fail_pos']} fail_head={row['fail_head']} fail_neg={row['fail_neg']}",
                    flush=True,
                )
                print(f"  + {best_pos[:70]}", flush=True)
                print(f"  head: {best_head[:70]}", flush=True)
                print(f"  - {best_neg[:70]}", flush=True)

    cap.release()
    print("---", flush=True)
    print(f"person_boxes_scored={n_scored} (area-ok people seen≈{n_people})", flush=True)
    if not best:
        print("NO_PERSON_CROPS — YOLO never found a large enough person.", flush=True)
        return 0
    print("BEST_CROP:", flush=True)
    for k, v in best.items():
        print(f"  {k}: {v}", flush=True)
    print(
        f"gates: need pos≥{MIN_POS_SCORE}, head≥{MIN_HEADCOVER_SCORE}, "
        f"neg/pos<{MAX_NEG_TO_POS_RATIO}, gated≥0.08",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
