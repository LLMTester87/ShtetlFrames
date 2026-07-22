"""Sparse / dense YouTube section helpers to cut full-video proxy bandwidth.

yt-dlp ``--download-sections`` downloads short timed slices instead of the whole file.
When multiple sections are concatenated, output timeline starts at 0 — remap with
``remap_segments_to_source``.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from typing import Any


def _env_float(key: str, default: float) -> float:
    try:
        return float(os.environ.get(key) or default)
    except ValueError:
        return default


def sparse_section_sec() -> float:
    return max(5.0, _env_float("SPARSE_SECTION_SEC", 20.0))


def sparse_stride_sec() -> float:
    return max(sparse_section_sec(), _env_float("SPARSE_STRIDE_SEC", 60.0))


def dense_pad_sec() -> float:
    return max(5.0, _env_float("DENSE_PAD_SEC", 30.0))


def max_dense_sec() -> float:
    return max(60.0, _env_float("MAX_DENSE_SEC", 600.0))


def format_section(start: float, end: float) -> str:
    """yt-dlp section spec: *START-END in seconds."""
    s = max(0.0, float(start))
    e = max(s + 0.5, float(end))
    return f"*{s:.2f}-{e:.2f}"


def build_sparse_windows(
    duration: float,
    *,
    section_sec: float | None = None,
    stride_sec: float | None = None,
) -> list[tuple[float, float]]:
    """Return (start, end) windows covering the video sparsely."""
    dur = max(0.0, float(duration))
    if dur <= 0:
        return []
    sec = float(section_sec if section_sec is not None else sparse_section_sec())
    stride = float(stride_sec if stride_sec is not None else sparse_stride_sec())
    sec = max(5.0, sec)
    stride = max(sec, stride)

    # Short enough that sparse ≈ full — caller should skip sparse and download whole file.
    if dur <= sec + 1.0:
        return [(0.0, dur)]

    starts: list[float] = []
    t = 0.0
    while t < dur:
        starts.append(t)
        t += stride
    # Ensure a window near the end so the tail is not missed.
    last_start = max(0.0, dur - sec)
    if not starts or last_start > starts[-1] + 1.0:
        starts.append(last_start)

    windows: list[tuple[float, float]] = []
    for start in starts:
        end = min(dur, start + sec)
        if end > start:
            windows.append((start, end))
    return windows


def should_skip_sparse(duration: float | None) -> bool:
    """True when full download is cheaper / similar to sparse (short clips)."""
    if duration is None or duration <= 0:
        return True
    # One or two windows worth — full file is fine.
    return float(duration) <= sparse_stride_sec() + sparse_section_sec()


def sections_from_windows(windows: list[tuple[float, float]]) -> list[str]:
    return [format_section(s, e) for s, e in windows if e > s]


def build_dense_windows(
    hit_times: list[float],
    duration: float,
    *,
    pad_sec: float | None = None,
    max_total_sec: float | None = None,
) -> list[tuple[float, float]]:
    """Merge padded windows around hit timestamps; cap total covered length."""
    if not hit_times:
        return []
    dur = max(0.0, float(duration)) if duration and duration > 0 else 0.0
    pad = float(pad_sec if pad_sec is not None else dense_pad_sec())
    cap = float(max_total_sec if max_total_sec is not None else max_dense_sec())
    pad = max(5.0, pad)
    cap = max(60.0, cap)

    raw: list[tuple[float, float, float]] = []  # start, end, center
    for t in sorted(float(x) for x in hit_times):
        start = max(0.0, t - pad)
        end = t + pad
        if dur > 0:
            end = min(dur, end)
        if end <= start:
            continue
        raw.append((start, end, t))

    if not raw:
        return []

    # Merge overlaps
    raw.sort(key=lambda w: w[0])
    merged: list[list[float]] = [[raw[0][0], raw[0][1], raw[0][2]]]
    for start, end, center in raw[1:]:
        if start <= merged[-1][1] + 1.0:
            merged[-1][1] = max(merged[-1][1], end)
            # keep earliest center for scoring stability
        else:
            merged.append([start, end, center])

    windows = [(m[0], m[1]) for m in merged]
    total = sum(e - s for s, e in windows)
    if total <= cap:
        return windows

    # Prefer windows whose center is closest to original hits (already ordered).
    # Drop from the end until under cap, but always keep at least one.
    kept: list[tuple[float, float]] = []
    used = 0.0
    for s, e in windows:
        length = e - s
        if kept and used + length > cap:
            remain = cap - used
            if remain >= 10.0:
                kept.append((s, s + remain))
            break
        kept.append((s, e))
        used += length
        if used >= cap:
            break
    return kept or windows[:1]


def remap_time(t_out: float, windows: list[tuple[float, float]]) -> float:
    """Map timestamp in concatenated section file back to source timeline."""
    if not windows:
        return float(t_out)
    t = max(0.0, float(t_out))
    cum = 0.0
    for start, end in windows:
        length = max(0.0, end - start)
        if t <= cum + length + 1e-6:
            return start + (t - cum)
        cum += length
    # Past end — clamp to last window end
    return windows[-1][1]


def remap_segments_to_source(
    segments: list[dict[str, Any]],
    windows: list[tuple[float, float]],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for seg in segments or []:
        s = dict(seg)
        try:
            start = float(s.get("start_sec") or 0)
            end = float(s.get("end_sec") or start)
        except (TypeError, ValueError):
            out.append(s)
            continue
        s["start_sec"] = round(remap_time(start, windows), 3)
        s["end_sec"] = round(remap_time(end, windows), 3)
        out.append(s)
    return out


def hit_times_from_result(out: dict[str, Any]) -> list[float]:
    times: list[float] = []
    for seg in out.get("segments") or []:
        try:
            times.append(float(seg.get("start_sec") or 0))
            times.append(float(seg.get("end_sec") or seg.get("start_sec") or 0))
        except (TypeError, ValueError):
            pass
    for fr in out.get("top_frames") or []:
        try:
            times.append(float(fr.get("t") or 0))
        except (TypeError, ValueError):
            pass
    return times


def result_has_hits(out: dict[str, Any] | None) -> bool:
    if not out:
        return False
    if int(out.get("n_hits") or 0) > 0:
        return True
    if int(out.get("n_frame_hits") or 0) > 0:
        return True
    if out.get("segments"):
        return True
    return False


def probe_duration(
    url: str,
    *,
    cookies_path: str | None = None,
    proxy_url: str | None = None,
    proxy_insecure: bool = False,
) -> float | None:
    """Best-effort duration via yt-dlp -J (prefer no proxy — metadata only)."""
    url = (url or "").strip()
    if not url:
        return None
    cmd = [
        sys.executable,
        "-m",
        "yt_dlp",
        "-J",
        "--no-playlist",
        "--skip-download",
        "-4",
        "--socket-timeout",
        "20",
    ]
    if cookies_path and os.path.isfile(cookies_path):
        cmd.extend(["--cookies", cookies_path])
    if proxy_url:
        cmd.extend(["--proxy", proxy_url])
        if proxy_insecure:
            cmd.append("--no-check-certificates")
    cmd.append(url)
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=90)
    except Exception:
        return None
    if proc.returncode != 0 or not (proc.stdout or "").strip():
        return None
    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return None
    dur = data.get("duration")
    try:
        d = float(dur)
    except (TypeError, ValueError):
        return None
    return d if d > 0 else None


def probe_duration_smart(url: str) -> float | None:
    """Try cookies without proxy, then proxy if configured."""
    cookies_path = None
    try:
        from yt_cookies import cookies_path as _cp, ensure_cookies_for_scrape

        ensure_cookies_for_scrape()
        p = _cp()
        if p.is_file():
            cookies_path = str(p)
    except Exception:
        pass

    dur = probe_duration(url, cookies_path=cookies_path, proxy_url=None)
    if dur is not None:
        return dur
    try:
        from yt_proxy import proxy_configured, proxy_needs_insecure_ssl, residential_proxy_url

        if proxy_configured():
            return probe_duration(
                url,
                cookies_path=cookies_path,
                proxy_url=residential_proxy_url(),
                proxy_insecure=proxy_needs_insecure_ssl(),
            )
    except Exception:
        pass
    return None
