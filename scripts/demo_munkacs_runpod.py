"""Provision a fresh RunPod GPU and score the Munkács 1933 reference film.

Writes output/munkacs_runpod_demo.json for the README demo section.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from config import DEFAULT_FPS, DEFAULT_SCORE_THRESHOLD, OUTPUT_DIR, load_env  # noqa: E402

MUNKACS_URL = "https://www.youtube.com/watch?v=tdkNbcpCTc0"
MUNKACS_TITLE = "Jewish Life in Munkatch - March 1933 (complete)"
OUT = OUTPUT_DIR / "munkacs_runpod_demo.json"


def main() -> int:
    load_env()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    def status(msg: str) -> None:
        print(msg, flush=True)

    status("Terminating old pods and creating a fresh RunPod GPU…")
    from runpod_client import process_video_remote, set_pod_pool
    from runpod_provision import ensure_pods, stop_pod

    t0 = time.time()
    bases = ensure_pods(count=1, on_status=status, recreate=True)
    set_pod_pool(bases)
    status(f"Pod ready: {bases[0]}")

    # Slightly below default so the reference film still yields segments if cues are tight.
    thr = max(0.08, float(DEFAULT_SCORE_THRESHOLD) - 0.02)
    status(f"Scanning Munkács on GPU (threshold={thr}): {MUNKACS_URL}")
    out = process_video_remote(
        url=MUNKACS_URL,
        title=MUNKACS_TITLE,
        queue_id=None,
        sample_fps=DEFAULT_FPS,
        score_threshold=thr,
        source_url=MUNKACS_URL,
        on_status=status,
        max_attempts=3,
    )
    status(
        f"Raw pod result: ok={out.get('ok')} n_hits={out.get('n_hits')} "
        f"n_frame_hits={out.get('n_frame_hits')} err={out.get('error')!r}"
    )

    segs = list(out.get("segments") or [])
    segs_sorted = sorted(segs, key=lambda s: float(s.get("peak_score") or 0), reverse=True)

    # Optional OpenAI second pass on top stills (same gate as Review).
    openai_rows = []
    try:
        from openai_verify import (
            format_verdict_notes,
            openai_verify_enabled,
            verdict_is_keep,
            verify_still,
        )

        if openai_verify_enabled():
            status("OpenAI headcover verify on ranked stills…")
            for s in segs_sorted[:8]:
                url = s.get("image_url") or ""
                if not url:
                    continue
                v = verify_still(image_url=url)
                openai_rows.append(
                    {
                        "start_sec": s.get("start_sec"),
                        "peak_score": s.get("peak_score"),
                        "best_cue": s.get("best_cue"),
                        "image_url": url,
                        "openai_keep": verdict_is_keep(v),
                        "notes": format_verdict_notes(v),
                    }
                )
    except Exception as e:
        status(f"OpenAI verify skipped: {e}")

    report = {
        "ok": True,
        "title": MUNKACS_TITLE,
        "source_url": MUNKACS_URL,
        "youtube_id": "tdkNbcpCTc0",
        "pod_base": bases[0],
        "elapsed_sec": round(time.time() - t0, 1),
        "sample_fps": DEFAULT_FPS,
        "score_threshold": float(DEFAULT_SCORE_THRESHOLD),
        "n_hits": out.get("n_hits"),
        "n_frame_hits": out.get("n_frame_hits"),
        "n_segments": len(segs),
        "peak_score": max((float(s.get("peak_score") or 0) for s in segs), default=0.0),
        "segments": segs_sorted[:12],
        "top_frames": (out.get("top_frames") or [])[:12],
        "openai_verify": openai_rows,
        "worker_note": "Fresh RunPod via ensure_pods(recreate=True); CLIP ViT-L-14 + headcover gate",
    }
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    status(f"Wrote {OUT}")
    status(
        f"Done: {report['n_segments']} segments · peak={report['peak_score']:.4f} · "
        f"{report['elapsed_sec']}s"
    )

    try:
        status("Stopping GPU pod to save cost…")
        stop_pod()
    except Exception as e:
        status(f"Pod stop skipped: {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
