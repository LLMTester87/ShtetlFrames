"""Build labeled timeline strips around a Review hit.

Long strip: ±10s @ 0.5s (overlay labels). Short crop: ±2s @ 0.5s with
caption bar below each frame (no overlap), JPEG capped at 399KB.
"""

from __future__ import annotations

import io
import re
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from config import CONTACT_DIR, VIDEOS_DIR
from still_store import candidate_crop_path, candidate_strip_path

PAD_SEC = 10.0
INTERVAL_SEC = 0.5
THUMB_H = 144
# ~41 frames × ~scaled width; keep JPEG readable in Review.
JPEG_QUALITY = 85

# Short Review crop: ±2s @ 0.5s, caption below frame, under 399KB.
CROP_PAD_SEC = 2.0
CROP_THUMB_H = 280
CROP_MAX_BYTES = 399_000
CROP_JPEG_QUALITY = 92

_strip_lock = threading.Lock()
_inflight: set[int] = set()

_crop_lock = threading.Lock()
_crop_inflight: set[int] = set()
_crop_errors: dict[int, str] = {}


def hit_mid_sec(start_sec: float, end_sec: float) -> float:
    a = float(start_sec or 0.0)
    b = float(end_sec or a)
    if b < a:
        a, b = b, a
    return (a + b) / 2.0


def sample_times(mid_sec: float, *, pad: float = PAD_SEC, step: float = INTERVAL_SEC) -> list[float]:
    start = max(0.0, mid_sec - pad)
    end = mid_sec + pad
    times: list[float] = []
    t = start
    # Inclusive end within a small epsilon.
    while t <= end + 1e-9:
        times.append(round(t, 3))
        t += step
    if times and abs(times[-1] - end) > 0.05:
        times.append(round(end, 3))
    return times


def _fmt_ts(sec: float) -> str:
    s = max(0, int(round(float(sec))))
    m, r = divmod(s, 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}:{m:02d}:{r:02d}"
    return f"{m}:{r:02d}"


def _source_label(video_id: str | None, source_url: str | None) -> str:
    url = (source_url or "").strip()
    host = ""
    if url:
        try:
            host = (urlparse(url).netloc or "").lower()
            if host.startswith("www."):
                host = host[4:]
        except Exception:
            host = ""
    if "youtube" in host or "youtu.be" in host:
        src = "YouTube"
    elif "britishpathe" in host:
        src = "British Pathé"
    elif host:
        src = host[:40]
    else:
        src = "archive"
    vid = re.sub(r"[_\s]+", " ", (video_id or "").strip())[:48]
    if vid:
        return f"{src} · {vid}"
    return src


def _ffmpeg_extract_window(
    video: Path,
    *,
    start_sec: float,
    duration_sec: float,
    out_dir: Path,
    thumb_h: int = THUMB_H,
) -> list[Path]:
    """Extract frames at INTERVAL_SEC over [start, start+duration) via fps filter."""
    out_dir.mkdir(parents=True, exist_ok=True)
    pattern = out_dir / "f_%04d.jpg"
    fps = 1.0 / INTERVAL_SEC
    vf = f"fps={fps:.6f},scale=-2:{int(thumb_h)}"
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{max(0.0, start_sec):.3f}",
        "-i",
        str(video),
        "-t",
        f"{max(0.1, duration_sec):.3f}",
        "-vf",
        vf,
        "-q:v",
        "3",
        str(pattern),
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=300)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []
    if r.returncode != 0:
        return []
    return sorted(out_dir.glob("f_*.jpg"))


def _opencv_extract_times(
    video: Path,
    times: list[float],
    out_dir: Path,
    *,
    thumb_h: int = THUMB_H,
) -> list[tuple[float, Path]]:
    import cv2

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        return []
    out_dir.mkdir(parents=True, exist_ok=True)
    got: list[tuple[float, Path]] = []
    for i, t in enumerate(times):
        cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, t) * 1000.0)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        h, w = frame.shape[:2]
        if h > 0 and thumb_h > 0 and h != thumb_h:
            nw = max(1, int(round(w * (thumb_h / float(h)))))
            frame = cv2.resize(frame, (nw, thumb_h), interpolation=cv2.INTER_AREA)
        dest = out_dir / f"f_{i:04d}.jpg"
        if cv2.imwrite(str(dest), frame) and dest.is_file() and dest.stat().st_size > 200:
            got.append((t, dest))
    cap.release()
    return got


def _label_frame(
    img,
    *,
    source: str,
    timestamp: str,
    mark: bool = False,
    caption_below: bool = False,
):
    """Stamp source + timestamp. caption_below=True keeps the bar under the frame pixels."""
    from PIL import Image, ImageDraw, ImageFont

    frame = img.convert("RGB")
    try:
        font = ImageFont.truetype("arial.ttf", 14)
        font_sm = ImageFont.truetype("arial.ttf", 12)
    except Exception:
        font = ImageFont.load_default()
        font_sm = font

    w, h = frame.size
    bar_h = 36
    line1 = timestamp + (" ★" if mark else "")
    line2 = (source or "")[:70]

    if caption_below:
        canvas = Image.new("RGB", (w, h + bar_h), (0, 0, 0))
        canvas.paste(frame, (0, 0))
        draw = ImageDraw.Draw(canvas)
        draw.text((6, h + 3), line1, fill=(255, 220, 120), font=font)
        draw.text((6, h + 18), line2, fill=(230, 230, 230), font=font_sm)
        if mark:
            draw.rectangle([0, 0, w - 1, h - 1], outline=(255, 200, 80), width=3)
        return canvas

    im = frame
    draw = ImageDraw.Draw(im)
    draw.rectangle([0, h - bar_h, w, h], fill=(0, 0, 0))
    draw.text((6, h - bar_h + 3), line1, fill=(255, 220, 120), font=font)
    draw.text((6, h - bar_h + 18), line2, fill=(230, 230, 230), font=font_sm)
    if mark:
        draw.rectangle([0, 0, w - 1, h - 1], outline=(255, 200, 80), width=3)
    return im


def stitch_labeled_strip(
    frames: list[tuple[float, Any]],
    *,
    source: str,
    mid_sec: float,
    caption_below: bool = False,
) -> Any:
    """Horizontal stitch of labeled PIL images."""
    from PIL import Image

    if not frames:
        raise ValueError("no_frames")
    labeled = []
    for t, im in frames:
        mark = abs(float(t) - float(mid_sec)) <= (INTERVAL_SEC / 2 + 0.05)
        labeled.append(
            _label_frame(
                im,
                source=source,
                timestamp=_fmt_ts(t),
                mark=mark,
                caption_below=caption_below,
            )
        )
    heights = [im.height for im in labeled]
    widths = [im.width for im in labeled]
    h = max(heights)
    total_w = sum(widths)
    canvas = Image.new("RGB", (total_w, h), (10, 9, 8))
    x = 0
    for im in labeled:
        if im.height != h:
            try:
                resample = Image.Resampling.LANCZOS
            except AttributeError:
                resample = Image.LANCZOS
            nw = max(1, int(round(im.width * (h / float(im.height)))))
            im = im.resize((nw, h), resample)
        canvas.paste(im, (x, 0))
        x += im.width
    return canvas


def _extract_labeled_frames(
    video: Path,
    *,
    mid_sec: float,
    pad: float,
    thumb_h: int,
    work_dir: Path,
) -> list[tuple[float, Any]]:
    from PIL import Image

    times = sample_times(mid_sec, pad=pad)
    if not times:
        return []
    win_start = times[0]
    duration = max(INTERVAL_SEC, times[-1] - times[0] + INTERVAL_SEC)
    paths = _ffmpeg_extract_window(
        video,
        start_sec=win_start,
        duration_sec=duration,
        out_dir=work_dir,
        thumb_h=thumb_h,
    )
    paired: list[tuple[float, Any]] = []
    if paths:
        n = min(len(paths), len(times))
        for i in range(n):
            try:
                im = Image.open(paths[i]).convert("RGB")
            except Exception:
                continue
            paired.append((times[i], im))
    if len(paired) < 3:
        paired = []
        for t, p in _opencv_extract_times(
            video, times, work_dir / "cv", thumb_h=thumb_h
        ):
            try:
                im = Image.open(p).convert("RGB")
            except Exception:
                continue
            paired.append((t, im))
    return paired


def _save_jpeg_under(img: Any, dest: Path, *, max_bytes: int, quality: int = JPEG_QUALITY) -> Path | None:
    """Write the largest JPEG that still fits under max_bytes."""
    from PIL import Image

    if not isinstance(img, Image.Image):
        return None
    CONTACT_DIR.mkdir(parents=True, exist_ok=True)
    work = img.convert("RGB")
    try:
        resample = Image.Resampling.LANCZOS
    except AttributeError:
        resample = Image.LANCZOS

    # Prefer higher quality first; only shrink when the quality floor still overflows.
    qualities = list(range(int(quality), 49, -4))
    if 45 not in qualities:
        qualities.append(45)
    scales = [1.0, 0.94, 0.88, 0.80, 0.72, 0.64, 0.56]
    best_data: bytes | None = None
    best_size = 0

    for scale in scales:
        cur = work
        if scale < 1.0:
            nw = max(32, int(round(work.width * scale)))
            nh = max(24, int(round(work.height * scale)))
            cur = work.resize((nw, nh), resample)
        for q in qualities:
            buf = io.BytesIO()
            cur.save(buf, format="JPEG", quality=int(q), optimize=True)
            data = buf.getvalue()
            n = len(data)
            if 500 < n <= max_bytes and n >= best_size:
                best_data = data
                best_size = n
                # Close enough to the budget — stop searching smaller.
                if n >= int(max_bytes * 0.90):
                    dest.write_bytes(data)
                    return dest
        if best_data is not None:
            # Full-scale (or this scale) already produced a fit; keep best, don't downscale more.
            dest.write_bytes(best_data)
            return dest
    if best_data is not None:
        dest.write_bytes(best_data)
        return dest
    # Last resort: smallest encode we can write.
    work.save(dest, format="JPEG", quality=40, optimize=True)
    return dest if dest.is_file() and dest.stat().st_size > 500 else None


def build_strip_for_video(
    video: Path,
    *,
    cand_id: int,
    start_sec: float,
    end_sec: float,
    video_id: str = "",
    source_url: str = "",
) -> Path | None:
    """Extract ±PAD_SEC @ INTERVAL_SEC, label, stitch → cand_{id}_strip.jpg."""
    mid = hit_mid_sec(start_sec, end_sec)
    source = _source_label(video_id, source_url)

    with tempfile.TemporaryDirectory(prefix=f"strip_{cand_id}_") as td:
        paired = _extract_labeled_frames(
            video, mid_sec=mid, pad=PAD_SEC, thumb_h=THUMB_H, work_dir=Path(td)
        )
        if len(paired) < 2:
            return None
        strip = stitch_labeled_strip(paired, source=source, mid_sec=mid)
        dest = candidate_strip_path(cand_id)
        strip.save(dest, format="JPEG", quality=JPEG_QUALITY, optimize=True)
        return dest if dest.is_file() and dest.stat().st_size > 500 else None


def build_crop_for_video(
    video: Path,
    *,
    cand_id: int,
    start_sec: float,
    end_sec: float,
    video_id: str = "",
    source_url: str = "",
) -> Path | None:
    """Extract ±2s @ 0.5s, caption below frame, stitch → cand_{id}_crop.jpg ≤399KB."""
    mid = hit_mid_sec(start_sec, end_sec)
    source = _source_label(video_id, source_url)

    with tempfile.TemporaryDirectory(prefix=f"crop_{cand_id}_") as td:
        paired = _extract_labeled_frames(
            video,
            mid_sec=mid,
            pad=CROP_PAD_SEC,
            thumb_h=CROP_THUMB_H,
            work_dir=Path(td),
        )
        if len(paired) < 2:
            return None
        strip = stitch_labeled_strip(
            paired, source=source, mid_sec=mid, caption_below=True
        )
        dest = candidate_crop_path(cand_id)
        return _save_jpeg_under(
            strip, dest, max_bytes=CROP_MAX_BYTES, quality=CROP_JPEG_QUALITY
        )


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
        return None
    path = Path(result["path"])
    return path if path.is_file() else None


def generate_strip_for_candidate(
    row: dict[str, Any],
    *,
    video_path: Path | None = None,
    download_if_needed: bool = True,
    delete_video_after: bool = False,
) -> Path | None:
    """Build strip for one candidate dict (id, start_sec, end_sec, video_id, source_url)."""
    cid = int(row["id"])
    dest = candidate_strip_path(cid)
    if dest.is_file() and dest.stat().st_size > 500:
        return dest

    url = (row.get("source_url") or "").strip()
    vid = (row.get("video_id") or "").strip() or f"cand_{cid}"
    video = video_path
    owned = False
    if video is None or not Path(video).is_file():
        if not download_if_needed or not url:
            return None
        video = _download_source(url, vid, vid)
        owned = True
    if not video or not Path(video).is_file():
        return None
    try:
        return build_strip_for_video(
            Path(video),
            cand_id=cid,
            start_sec=float(row.get("start_sec") or 0),
            end_sec=float(row.get("end_sec") or 0),
            video_id=vid,
            source_url=url,
        )
    finally:
        if delete_video_after and owned and video and Path(video).is_file():
            try:
                from run_archive import delete_local_video

                delete_local_video(Path(video))
            except Exception:
                try:
                    Path(video).unlink(missing_ok=True)
                except Exception:
                    pass


def enqueue_strip_for_keep(cand_id: int) -> None:
    """Fire-and-forget strip build after human Keep (non-blocking)."""
    cid = int(cand_id)
    with _strip_lock:
        if cid in _inflight:
            return
        if candidate_strip_path(cid).is_file():
            return
        _inflight.add(cid)

    def _run() -> None:
        try:
            from db import db, init_db

            init_db()
            with db() as conn:
                row = conn.execute(
                    "SELECT id, video_id, start_sec, end_sec, source_url, decision "
                    "FROM candidates WHERE id=?",
                    (cid,),
                ).fetchone()
            if not row:
                return
            d = dict(row)
            if (d.get("decision") or "") != "accept":
                return
            generate_strip_for_candidate(d, download_if_needed=True, delete_video_after=True)
        except Exception:
            pass
        finally:
            with _strip_lock:
                _inflight.discard(cid)

    threading.Thread(target=_run, daemon=True, name=f"strip-{cid}").start()


def crop_ready(cand_id: int) -> bool:
    path = candidate_crop_path(int(cand_id))
    return path.is_file() and path.stat().st_size > 500


def crop_status(cand_id: int) -> dict[str, Any]:
    """Return status for Review button: ready | queued | error | none."""
    from still_store import local_crop_url

    cid = int(cand_id)
    if crop_ready(cid):
        path = candidate_crop_path(cid)
        return {
            "status": "ready",
            "crop_url": local_crop_url(cid),
            "bytes": path.stat().st_size,
            "error": None,
        }
    with _crop_lock:
        if cid in _crop_inflight:
            return {"status": "queued", "crop_url": None, "bytes": 0, "error": None}
        err = _crop_errors.get(cid)
    if err:
        return {"status": "error", "crop_url": None, "bytes": 0, "error": err}
    return {"status": "none", "crop_url": None, "bytes": 0, "error": None}


def generate_crop_for_candidate(
    row: dict[str, Any],
    *,
    video_path: Path | None = None,
    download_if_needed: bool = True,
    delete_video_after: bool = True,
    force: bool = False,
) -> Path | None:
    """Build short ±2s crop for one candidate dict."""
    cid = int(row["id"])
    dest = candidate_crop_path(cid)
    if not force and dest.is_file() and dest.stat().st_size > 500:
        return dest
    if force and dest.is_file():
        try:
            dest.unlink()
        except OSError:
            pass

    url = (row.get("source_url") or "").strip()
    vid = (row.get("video_id") or "").strip() or f"cand_{cid}"
    video = video_path
    owned = False
    if video is None or not Path(video).is_file():
        if not download_if_needed or not url:
            return None
        video = _download_source(url, vid, vid)
        owned = True
    if not video or not Path(video).is_file():
        return None
    try:
        return build_crop_for_video(
            Path(video),
            cand_id=cid,
            start_sec=float(row.get("start_sec") or 0),
            end_sec=float(row.get("end_sec") or 0),
            video_id=vid,
            source_url=url,
        )
    finally:
        if delete_video_after and owned and video and Path(video).is_file():
            try:
                from run_archive import delete_local_video

                delete_local_video(Path(video))
            except Exception:
                try:
                    Path(video).unlink(missing_ok=True)
                except Exception:
                    pass


def enqueue_crop_for_candidate(cand_id: int, *, force: bool = False) -> dict[str, Any]:
    """Queue short crop build. Refuses if already ready or in flight (unless force)."""
    cid = int(cand_id)
    if not force:
        st = crop_status(cid)
        if st["status"] == "ready":
            return {"ok": True, "queued": False, **st}
    with _crop_lock:
        if cid in _crop_inflight:
            return {"ok": True, "queued": False, "status": "queued", "crop_url": None, "bytes": 0, "error": None}
        if not force and crop_ready(cid):
            return {"ok": True, "queued": False, **crop_status(cid)}
        _crop_errors.pop(cid, None)
        _crop_inflight.add(cid)

    def _run() -> None:
        try:
            from db import db, init_db

            init_db()
            with db() as conn:
                row = conn.execute(
                    "SELECT id, video_id, start_sec, end_sec, source_url "
                    "FROM candidates WHERE id=?",
                    (cid,),
                ).fetchone()
            if not row:
                with _crop_lock:
                    _crop_errors[cid] = "candidate_not_found"
                return
            path = generate_crop_for_candidate(
                dict(row),
                download_if_needed=True,
                delete_video_after=True,
                force=force,
            )
            if not path:
                with _crop_lock:
                    _crop_errors[cid] = "build_failed"
        except Exception as e:
            with _crop_lock:
                _crop_errors[cid] = str(e)[:200] or "build_failed"
        finally:
            with _crop_lock:
                _crop_inflight.discard(cid)

    threading.Thread(target=_run, daemon=True, name=f"crop-{cid}").start()
    return {"ok": True, "queued": True, "status": "queued", "crop_url": None, "bytes": 0, "error": None}


def list_crop_jobs() -> list[dict[str, Any]]:
    """All ready crops + currently queued ids (for Crops page)."""
    from still_store import local_crop_url

    out: list[dict[str, Any]] = []
    seen: set[int] = set()
    CONTACT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(CONTACT_DIR.glob("cand_*_crop.jpg")):
        m = re.match(r"cand_(\d+)_crop\.jpg$", path.name)
        if not m or path.stat().st_size <= 500:
            continue
        cid = int(m.group(1))
        seen.add(cid)
        out.append(
            {
                "id": cid,
                "status": "ready",
                "crop_url": local_crop_url(cid),
                "bytes": path.stat().st_size,
                "error": None,
            }
        )
    with _crop_lock:
        queued = sorted(_crop_inflight)
        errors = dict(_crop_errors)
    for cid in queued:
        if cid in seen:
            continue
        out.append(
            {
                "id": cid,
                "status": "queued",
                "crop_url": None,
                "bytes": 0,
                "error": None,
            }
        )
        seen.add(cid)
    for cid, err in errors.items():
        if cid in seen:
            continue
        out.append(
            {
                "id": cid,
                "status": "error",
                "crop_url": None,
                "bytes": 0,
                "error": err,
            }
        )
    out.sort(key=lambda r: (-1 if r["status"] == "queued" else 0, -int(r["id"])))
    return out
