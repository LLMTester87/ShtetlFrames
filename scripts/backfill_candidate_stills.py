"""Backfill missing Review stills by grabbing a frame at each candidate timestamp.

Downloads each source video once (YouTube / Pathé / direct), extracts JPEGs with
ffmpeg (OpenCV fallback), saves to output/contact_sheets/cand_{id}.jpg, then
deletes the local video.
"""

from __future__ import annotations

import sys
import tempfile
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from config import VIDEOS_DIR, load_env  # noqa: E402
from db import db, init_db  # noqa: E402
from still_ensure import extract_frame  # noqa: E402
from still_store import local_still_url, save_candidate_still  # noqa: E402


def _download_source(url: str, video_id: str, title: str) -> Path | None:
    from download import download_britishpathe, download_entry
    from serve import find_video_file

    existing = find_video_file(video_id)
    if existing and existing.is_file():
        return existing

    if "britishpathe.com" in (url or "").lower():
        VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
        path = download_britishpathe(url, VIDEOS_DIR, video_id, title=title or video_id)
        return path if path and Path(path).is_file() else None

    result = download_entry(url, title or video_id, video_id=video_id)
    if result.get("error") or not result.get("path"):
        print(f"  download failed: {result.get('error') or 'no path'}", flush=True)
        return None
    path = Path(result["path"])
    return path if path.is_file() else None


def missing_candidates() -> list[dict]:
    with db() as conn:
        rows = conn.execute(
            """
            SELECT id, video_id, start_sec, end_sec, source_url, image_url, notes
            FROM candidates
            ORDER BY video_id, start_sec
            """
        ).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        if local_still_url(d["id"]):
            continue
        url = (d.get("source_url") or "").strip()
        if not url:
            continue
        out.append(d)
    return out


def main() -> int:
    load_env()
    init_db()
    rows = missing_candidates()
    if not rows:
        print("Nothing to backfill — all candidates with source URLs already have local stills.")
        return 0

    by_key: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        vid = (r.get("video_id") or "unknown").strip() or "unknown"
        url = (r.get("source_url") or "").strip()
        by_key[(vid, url)].append(r)

    print(f"Backfilling {len(rows)} still(s) across {len(by_key)} video(s)…", flush=True)
    ok_n = 0
    fail_n = 0

    for (vid, url), group in by_key.items():
        print(f"\n=== {vid} · {len(group)} moment(s) ===", flush=True)
        print(f"  {url}", flush=True)
        path = _download_source(url, vid, vid)
        if not path:
            fail_n += len(group)
            continue
        print(f"  video: {path.name} ({path.stat().st_size // 1024} KB)", flush=True)
        try:
            with tempfile.TemporaryDirectory(prefix="shtetl_backfill_") as tmp:
                tmpdir = Path(tmp)
                for r in group:
                    cid = int(r["id"])
                    t0 = float(r.get("start_sec") or 0.0)
                    t1 = float(r.get("end_sec") or t0)
                    # Prefer mid-window of the hit segment.
                    t = t0 if t1 <= t0 else (t0 + t1) / 2.0
                    tmp_jpg = tmpdir / f"{cid}.jpg"
                    print(f"  #{cid} @ {t:.2f}s …", end=" ", flush=True)
                    if not extract_frame(path, t, tmp_jpg):
                        print("EXTRACT FAIL", flush=True)
                        fail_n += 1
                        continue
                    saved = save_candidate_still(cid, path=tmp_jpg)
                    if saved:
                        print(f"OK → {saved.name}", flush=True)
                        ok_n += 1
                    else:
                        print("SAVE FAIL", flush=True)
                        fail_n += 1
        finally:
            # Drop downloaded media for this video_id (keep disk light).
            try:
                for p in VIDEOS_DIR.glob(f"{vid}*"):
                    if p.is_file() and p.suffix.lower() in {
                        ".mp4",
                        ".webm",
                        ".mkv",
                        ".avi",
                        ".mov",
                        ".part",
                    }:
                        p.unlink(missing_ok=True)
            except OSError:
                pass

    print(f"\nDone. saved={ok_n} failed={fail_n}")
    still_missing = [r["id"] for r in missing_candidates()]
    if still_missing:
        print(f"Still missing local images: {still_missing}")
    return 0 if ok_n else 1


if __name__ == "__main__":
    raise SystemExit(main())
