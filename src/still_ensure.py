"""Ensure every Review candidate has a durable local still (cand_{id}.jpg).

Call sites:
- insert_candidates (sync save + enqueue fallback)
- list_candidates (fast URL hydrate + enqueue video extract)
- scripts/backfill_candidate_stills.py
"""

from __future__ import annotations

import queue
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

from config import CONTACT_DIR, VIDEOS_DIR
from still_store import candidate_still_path, local_still_url, save_candidate_still

_ensure_q: queue.Queue[dict[str, Any]] = queue.Queue()
_ensure_seen: set[int] = set()
_ensure_lock = threading.Lock()
_worker_started = False


def extract_frame(video: Path, time_sec: float, out: Path) -> bool:
    """Grab one JPEG frame at time_sec (ffmpeg, OpenCV fallback)."""
    out.parent.mkdir(parents=True, exist_ok=True)
    t = max(0.0, float(time_sec))
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{t:.3f}",
        "-i",
        str(video),
        "-frames:v",
        "1",
        "-q:v",
        "2",
        str(out),
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=120)
        if r.returncode == 0 and out.is_file() and out.stat().st_size > 200:
            return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    try:
        import cv2

        cap = cv2.VideoCapture(str(video))
        if not cap.isOpened():
            return False
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ok, frame = cap.read()
        cap.release()
        if not ok or frame is None:
            return False
        return bool(cv2.imwrite(str(out), frame)) and out.is_file() and out.stat().st_size > 200
    except Exception:
        return False


def _download_source(url: str, video_id: str) -> Path | None:
    from download import download_britishpathe, download_entry
    from serve import find_video_file

    existing = find_video_file(video_id)
    if existing and existing.is_file():
        return existing
    if "britishpathe.com" in (url or "").lower():
        VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
        path = download_britishpathe(url, VIDEOS_DIR, video_id, title=video_id)
        return path if path and Path(path).is_file() else None
    result = download_entry(url, video_id, video_id=video_id)
    if result.get("error") or not result.get("path"):
        return None
    path = Path(result["path"])
    return path if path.is_file() else None


def ensure_candidate_still(
    cand_id: int,
    *,
    source_url: str = "",
    video_id: str = "",
    start_sec: float = 0.0,
    end_sec: float | None = None,
    image_url: str | None = None,
    download_video: bool = True,
) -> Path | None:
    """Return local still path, creating it from URL or source video if needed."""
    cid = int(cand_id)
    existing = candidate_still_path(cid)
    if existing.is_file() and existing.stat().st_size > 200:
        return existing

    # Fast: re-download Catbox / files.catbox if still live.
    url = (image_url or "").strip()
    if url.startswith(("http://", "https://")) and "litter.catbox" not in url.lower():
        saved = save_candidate_still(cid, image_url=url)
        if saved:
            return saved

    if not download_video:
        return None
    src = (source_url or "").strip()
    if not src:
        return None
    vid = (video_id or f"cand_{cid}").strip() or f"cand_{cid}"
    t0 = float(start_sec or 0.0)
    t1 = float(end_sec if end_sec is not None else t0)
    mid = t0 if t1 <= t0 else (t0 + t1) / 2.0

    from serve import find_video_file

    existing = find_video_file(vid)
    owned = False
    if existing and existing.is_file():
        video = existing
    else:
        video = _download_source(src, vid)
        owned = bool(video)
    if not video:
        return None
    try:
        with tempfile.TemporaryDirectory(prefix=f"still_{cid}_") as td:
            tmp = Path(td) / f"{cid}.jpg"
            if not extract_frame(video, mid, tmp):
                return None
            return save_candidate_still(cid, path=tmp)
    finally:
        if owned and video and Path(video).is_file():
            try:
                Path(video).unlink(missing_ok=True)
            except OSError:
                pass


def enqueue_ensure_still(row: dict[str, Any]) -> None:
    """Queue a background video-frame extract for a missing still."""
    global _worker_started
    try:
        cid = int(row["id"])
    except (KeyError, TypeError, ValueError):
        return
    if local_still_url(cid):
        return
    if not (row.get("source_url") or "").strip():
        return
    with _ensure_lock:
        if cid in _ensure_seen:
            return
        _ensure_seen.add(cid)
        if not _worker_started:
            _worker_started = True
            threading.Thread(
                target=_ensure_worker, daemon=True, name="still-ensure"
            ).start()
    _ensure_q.put(
        {
            "id": cid,
            "source_url": (row.get("source_url") or "").strip(),
            "video_id": (row.get("video_id") or "").strip(),
            "start_sec": row.get("start_sec") or 0,
            "end_sec": row.get("end_sec"),
            "image_url": row.get("image_url"),
        }
    )


def _ensure_worker() -> None:
    while True:
        row = _ensure_q.get()
        try:
            cid = int(row["id"])
            if local_still_url(cid):
                continue
            ensure_candidate_still(
                cid,
                source_url=str(row.get("source_url") or ""),
                video_id=str(row.get("video_id") or ""),
                start_sec=float(row.get("start_sec") or 0),
                end_sec=row.get("end_sec"),
                image_url=row.get("image_url"),
                download_video=True,
            )
        except Exception:
            pass
        finally:
            with _ensure_lock:
                try:
                    _ensure_seen.discard(int(row.get("id") or 0))
                except Exception:
                    pass
            time.sleep(0.15)


def missing_still_ids(*, limit: int = 5000) -> list[int]:
    from db import db, init_db

    init_db()
    CONTACT_DIR.mkdir(parents=True, exist_ok=True)
    with db() as conn:
        rows = conn.execute(
            "SELECT id FROM candidates ORDER BY id DESC LIMIT ?",
            (max(1, int(limit)),),
        ).fetchall()
    return [int(r["id"]) for r in rows if not local_still_url(int(r["id"]))]
