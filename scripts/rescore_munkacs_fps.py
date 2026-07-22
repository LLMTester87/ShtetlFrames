"""Rescore Munkács positives + known false-positive assets with headcover gate."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import requests
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from shtetl_core.cues import (  # noqa: E402
    DEFAULT_SCORE_THRESHOLD,
    MIN_HEADCOVER_SCORE,
    MIN_POS_SCORE,
)
from shtetl_core.scoring import CueScorer  # noqa: E402
from shtetl_core.scan import scan_video  # noqa: E402
from ultralytics import YOLO  # noqa: E402
from shtetl_core.cues import YOLO_WEIGHTS  # noqa: E402

OUT_DIR = ROOT / "output" / "headcover_rescore"
STILLS_DIR = OUT_DIR / "stills"
REPORT = OUT_DIR / "results.json"

MUNKACS_STILLS = [
    ("munkacs_peak_0.1414", 0.1414, "https://files.catbox.moe/7i7c5z.jpg"),
    ("munkacs_peak_0.1623", 0.1623, "https://files.catbox.moe/vmmdw6.jpg"),
    ("munkacs_peak_0.1191", 0.1191, "https://files.catbox.moe/9al9mv.jpg"),
    ("munkacs_peak_0.1040", 0.1040, "https://files.catbox.moe/wimcyk.jpg"),
]

FP_STILLS = [
    ("fp_eOqy744ViSc_old_0.1013", 0.1013, "https://files.catbox.moe/sinuif.jpg"),
    ("fp_-dEISdL6kMU_a", 0.0631, "https://files.catbox.moe/hr9k8w.jpg"),
    ("fp_-dEISdL6kMU_b", 0.0604, "https://files.catbox.moe/obqopx.jpg"),
    ("fp_-dEISdL6kMU_c", 0.0545, "https://files.catbox.moe/lg8k8b.jpg"),
    ("fp_-dEISdL6kMU_d", 0.0544, "https://files.catbox.moe/8tynhw.jpg"),
    ("fp_-dEISdL6kMU_e", 0.0530, "https://files.catbox.moe/2wjij5.jpg"),
    ("fp_-dEISdL6kMU_f", 0.0521, "https://files.catbox.moe/on080d.jpg"),
    ("fp_-dEISdL6kMU_g", 0.0509, "https://files.catbox.moe/6eg70w.jpg"),
    ("fp_w0Mar_FI4Vk_old_0.0582", 0.0582, "https://files.catbox.moe/tu650e.jpg"),
]

VIDEO_SCANS = [
    {
        "name": "reject_eOqy744ViSc",
        "path": ROOT / "output" / "fp_clips" / "reject_eOqy744ViSc.mp4",
        "old_peak": 0.1013,
        "kind": "false_positive",
    },
    {
        "name": "reject_-dEISdL6kMU",
        "path": ROOT / "output" / "fp_clips" / "reject_-dEISdL6kMU.mp4",
        "old_peak": 0.0631,
        "kind": "false_positive",
    },
    {
        "name": "reject_w0Mar_FI4Vk",
        "path": ROOT / "output" / "fp_clips" / "reject_w0Mar_FI4Vk.mp4",
        "old_peak": 0.0582,
        "kind": "false_positive",
    },
    {
        "name": "pathe_polish_refugees",
        "path": ROOT
        / "data"
        / "videos"
        / "ww2_polish_refugees_reach_iran_1943_archive_highlights.f398.mp4",
        "old_peak": 0.0705,
        "kind": "false_positive_control",
    },
]


def download(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 1000:
        return dest
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    dest.write_bytes(r.content)
    return dest


@torch.no_grad()
def score_still(scorer: CueScorer, path: Path) -> dict:
    pil = Image.open(path).convert("RGB")
    image = scorer.preprocess(pil).unsqueeze(0).to(scorer.device)
    img_feat = scorer.model.encode_image(image)
    img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
    pos_sims = (img_feat @ scorer.pos_feat.T).squeeze(0)
    neg_sims = (img_feat @ scorer.neg_feat.T).squeeze(0)
    head_sims = (img_feat @ scorer.head_feat.T).squeeze(0)
    score, pos_score, neg_score, cue = scorer.score_image(pil)
    return {
        "score": round(score, 4),
        "pos": round(pos_score, 4),
        "neg": round(neg_score, 4),
        "headcover": round(float(head_sims.max().item()), 4),
        "best_cue": cue,
        "passes_gate": score >= DEFAULT_SCORE_THRESHOLD,
        "pos_ok": pos_score >= MIN_POS_SCORE,
        "head_ok": float(head_sims.max().item()) >= MIN_HEADCOVER_SCORE,
        "raw_pos_top": round(float(pos_sims.max().item()), 4),
        "raw_neg_top": round(float(neg_sims.max().item()), 4),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    STILLS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"device will load CueScorer… threshold={DEFAULT_SCORE_THRESHOLD} "
          f"min_pos={MIN_POS_SCORE} min_head={MIN_HEADCOVER_SCORE}", flush=True)
    t0 = time.time()
    scorer = CueScorer()
    print(f"models ready in {time.time() - t0:.1f}s on {scorer.device}", flush=True)

    still_rows = []
    for name, old_peak, url in MUNKACS_STILLS + FP_STILLS:
        kind = "munkacs_positive" if name.startswith("munkacs") else "false_positive_still"
        dest = STILLS_DIR / f"{name}.jpg"
        print(f"still {name}…", flush=True)
        try:
            download(url, dest)
            detail = score_still(scorer, dest)
            row = {
                "name": name,
                "kind": kind,
                "old_peak": old_peak,
                "url": url,
                "path": str(dest),
                **detail,
            }
        except Exception as e:
            row = {
                "name": name,
                "kind": kind,
                "old_peak": old_peak,
                "url": url,
                "error": str(e)[:300],
            }
        still_rows.append(row)
        print(
            f"  → score={row.get('score')} head={row.get('headcover')} "
            f"pass={row.get('passes_gate')} cue={str(row.get('best_cue') or row.get('error'))[:60]}",
            flush=True,
        )

    print("loading YOLO…", flush=True)
    yolo = YOLO(YOLO_WEIGHTS)
    if torch.cuda.is_available():
        yolo.to("cuda")

    video_rows = []
    for item in VIDEO_SCANS:
        path: Path = item["path"]
        print(f"video {item['name']} ({path.name})…", flush=True)
        if not path.exists():
            video_rows.append({**item, "path": str(path), "error": "missing_file"})
            print("  → missing", flush=True)
            continue
        t1 = time.time()
        # Keep all person-frame scores so we can report peaks below the gate.
        soft = scan_video(
            path,
            item["name"],
            scorer,
            yolo,
            sample_fps=1.0,
            score_threshold=-1.0,
            save_crops_dir=None,
        )
        soft_top = sorted(soft, key=lambda h: -h.score)[:8]
        hits = [h for h in soft if h.score >= DEFAULT_SCORE_THRESHOLD]
        row = {
            "name": item["name"],
            "kind": item["kind"],
            "path": str(path),
            "old_peak": item["old_peak"],
            "elapsed_sec": round(time.time() - t1, 1),
            "n_person_frames": len(soft),
            "n_hits_at_gate": len(hits),
            "soft_top_scores": [round(h.score, 4) for h in soft_top],
            "soft_top_cues": [h.best_cue for h in soft_top],
            "soft_top_pos": [round(h.pos_score, 4) for h in soft_top],
            "passes_any": len(hits) > 0,
        }
        video_rows.append(row)
        print(
            f"  → hits={row['n_hits_at_gate']} soft_top={row['soft_top_scores']}",
            flush=True,
        )

    munkacs = [r for r in still_rows if r.get("kind") == "munkacs_positive"]
    fps = [r for r in still_rows if r.get("kind") == "false_positive_still"]
    summary = {
        "gates": {
            "score_threshold": DEFAULT_SCORE_THRESHOLD,
            "min_pos_score": MIN_POS_SCORE,
            "min_headcover_score": MIN_HEADCOVER_SCORE,
        },
        "munkacs_stills": {
            "n": len(munkacs),
            "n_pass": sum(1 for r in munkacs if r.get("passes_gate")),
            "scores": [r.get("score") for r in munkacs],
            "headcovers": [r.get("headcover") for r in munkacs],
        },
        "fp_stills": {
            "n": len(fps),
            "n_pass": sum(1 for r in fps if r.get("passes_gate")),
            "scores": [r.get("score") for r in fps],
            "headcovers": [r.get("headcover") for r in fps],
        },
        "fp_videos": {
            "n": len(video_rows),
            "n_with_hits": sum(1 for r in video_rows if r.get("passes_any")),
            "soft_peaks": {r["name"]: (r.get("soft_top_scores") or [None])[0] for r in video_rows},
        },
    }
    report = {
        "summary": summary,
        "stills": still_rows,
        "videos": video_rows,
    }
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    print(f"wrote {REPORT}", flush=True)


if __name__ == "__main__":
    main()
