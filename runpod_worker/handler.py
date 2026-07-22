"""ShtetlFrames RunPod worker — download, GPU scan, upload stills.

Vision scoring lives in `shtetl_core` (same package as the local app).
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
import traceback
from pathlib import Path
from typing import Any

import torch
from ultralytics import YOLO

from shtetl_core import (
    CLIP_MODEL,
    CLIP_PRETRAINED,
    DEFAULT_SCORE_THRESHOLD,
    YOLO_WEIGHTS,
    CueScorer,
    aggregate_segments_dicts,
    scan_video as core_scan_video,
    slugify,
    upload_image,
    write_sheet_from_crops,
)

WORKDIR = Path("/tmp/shtetlframes")
VIDEOS = WORKDIR / "videos"
CROPS = WORKDIR / "crops"
STILLS = WORKDIR / "stills"

# Worker samples slower than local DEFAULT_FPS (1.5) to cut pod time/cost.
DEFAULT_FPS = 0.5
DEFAULT_MAX_SCAN_SEC = 480  # reserved for future section limits
USER_AGENT = "ShtetlFrames-RunPod/1.0"

_progress_lock = threading.Lock()
_gpu_lock = threading.Lock()  # YOLO/CLIP only — downloads may run in parallel
_tls = threading.local()
_job_progress: dict[str, dict[str, Any]] = {}
_job_results: dict[str, dict[str, Any]] = {}
_job_results_lock = threading.Lock()
_ollama_prepared = False
_ollama_prep_lock = threading.Lock()
_IDLE_PROGRESS: dict[str, Any] = {
    "phase": "idle",
    "message": "",
    "queue_id": None,
    "title": "",
    "url": "",
    "pct": None,
    "detail": "",
    "updated_at": 0.0,
}

_yolo = None
_scorer = None

_YTDLP_PCT_RE = re.compile(r"\[download\]\s+([\d.]+)%")
_YTDLP_ETA_RE = re.compile(r"ETA\s+(\d+:\d+(?::\d+)?)")
_YTDLP_SPEED_RE = re.compile(r"at\s+(\S+/s)")
_YTDLP_SIZE_RE = re.compile(r"of\s+~?\s*([\d.]+\s*[KMG]i?B)", re.I)


def _job_key(queue_id: Any) -> str:
    return str(queue_id) if queue_id is not None else "_default"


def set_progress(
    phase: str,
    message: str,
    *,
    pct: float | None = None,
    detail: str = "",
    queue_id: Any = None,
    title: str | None = None,
    url: str | None = None,
) -> None:
    """Update live job progress (polled by local UI via GET /progress)."""
    if queue_id is None:
        queue_id = getattr(_tls, "queue_id", None)
    key = _job_key(queue_id)
    with _progress_lock:
        cur = dict(_job_progress.get(key) or _IDLE_PROGRESS)
        if queue_id is not None:
            cur["queue_id"] = queue_id
        if title is not None:
            cur["title"] = title
        if url is not None:
            cur["url"] = url
        cur["phase"] = phase
        cur["message"] = (message or "")[:240]
        cur["detail"] = (detail or "")[:400]
        cur["pct"] = None if pct is None else round(float(pct), 1)
        cur["updated_at"] = time.time()
        _job_progress[key] = cur
    line = f"[shtetl] q={key} {phase}: {message}"
    if detail:
        line = f"{line} · {detail}"
    print(line, flush=True)


def get_progress(queue_id: Any = None) -> dict[str, Any]:
    with _progress_lock:
        if queue_id is not None:
            return dict(_job_progress.get(_job_key(queue_id)) or _IDLE_PROGRESS)
        active = [
            dict(v)
            for v in _job_progress.values()
            if (v.get("phase") or "") not in ("", "idle", "done")
        ]
        for pref in ("scan", "download", "upload", "queued"):
            for v in active:
                if v.get("phase") == pref:
                    return v
        if active:
            return active[0]
        return dict(_IDLE_PROGRESS)


def clear_progress(queue_id: Any = None) -> None:
    with _progress_lock:
        if queue_id is not None:
            _job_progress.pop(_job_key(queue_id), None)
        else:
            _job_progress.clear()


def store_job_result(queue_id: Any, result: dict[str, Any]) -> None:
    with _job_results_lock:
        _job_results[_job_key(queue_id)] = dict(result or {})


def get_job_result(queue_id: Any, *, consume: bool = False) -> dict[str, Any] | None:
    key = _job_key(queue_id)
    with _job_results_lock:
        if consume:
            return _job_results.pop(key, None)
        cur = _job_results.get(key)
        return dict(cur) if cur is not None else None


def clear_job_result(queue_id: Any = None) -> None:
    with _job_results_lock:
        if queue_id is not None:
            _job_results.pop(_job_key(queue_id), None)
        else:
            _job_results.clear()


def still_file_path(queue_id: Any, index: int) -> Path:
    return STILLS / _job_key(queue_id) / f"{int(index)}.jpg"


def clear_job_stills(queue_id: Any = None) -> None:
    """Drop on-disk Review stills for a finished/abandoned job."""
    try:
        if queue_id is not None:
            shutil.rmtree(STILLS / _job_key(queue_id), ignore_errors=True)
        elif STILLS.is_dir():
            shutil.rmtree(STILLS, ignore_errors=True)
    except OSError:
        pass


def _encode_review_still(src: Path) -> tuple[bytes | None, str | None]:
    """Compact JPEG bytes + base64 for the PC. Prefer small payloads (proxy limits)."""
    try:
        import base64
        import io

        from PIL import Image

        with Image.open(src) as im:
            rgb = im.convert("RGB")
            # Must fit inline in /result JSON — Catbox often fails and /still is flaky.
            # Target <~120KB raw so base64 stays under the client inline cap.
            for max_side, quality in ((720, 72), (640, 65), (512, 58), (420, 50), (360, 42)):
                work = rgb.copy()
                work.thumbnail((max_side, max_side))
                buf = io.BytesIO()
                work.save(buf, format="JPEG", quality=quality, optimize=True)
                raw = buf.getvalue()
                if len(raw) <= 100_000:
                    return raw, base64.standard_b64encode(raw).decode("ascii")
            return raw, base64.standard_b64encode(raw).decode("ascii")
    except Exception:
        try:
            import base64

            raw = Path(src).read_bytes()
            if len(raw) > 500_000:
                return None, None
            return raw, base64.standard_b64encode(raw).decode("ascii")
        except Exception:
            return None, None


def reset_models() -> None:
    """Drop cached YOLO/CLIP so the next scan reloads after a bad numpy/torch state."""
    global _yolo, _scorer
    _yolo = None
    _scorer = None


def _models():
    global _yolo, _scorer
    if _yolo is None:
        _yolo = YOLO(YOLO_WEIGHTS)
        if torch.cuda.is_available():
            _yolo.to("cuda")
    if _scorer is None:
        _scorer = CueScorer()
    return _yolo, _scorer


def _ytdlp_error_summary(log_tail: list[str]) -> str:
    """Prefer real ERROR lines; ignore Python deprecation noise from yt-dlp."""
    skip = ("deprecated feature", "please update to python", "support for python version")
    errors = [
        ln
        for ln in log_tail
        if "error" in ln.lower() and not any(s in ln.lower() for s in skip)
    ]
    if errors:
        return " | ".join(errors[-3:])[-700:]
    useful = [ln for ln in log_tail if not any(s in ln.lower() for s in skip)]
    return ("\n".join(useful) or "\n".join(log_tail) or "download_failed")[-700:]


def _is_permanent_ytdlp_error(msg: str) -> bool:
    low = (msg or "").lower()
    markers = (
        "members-only",
        "members only",
        "this video is available to this channel's members",
        "join this channel to get access",
        "private video",
        "this video is private",
        "video unavailable",
        "this video is not available",
        "copyright",
        "has been removed",
        "this video has been removed",
        "account associated with this video has been terminated",
        "who has blocked you",
        "sign in to confirm your age",
        "confirm your age",
        "age-restricted",
        "login required",
    )
    return any(m in low for m in markers)


def _write_cookies_file(cookies_text: str | None) -> Path | None:
    text = (cookies_text or "").strip()
    if len(text) < 40:
        return None
    low = text.lower()
    if "youtube.com" not in low and ".google.com" not in low:
        return None
    VIDEOS.mkdir(parents=True, exist_ok=True)
    path = WORKDIR / f"yt_cookies_{os.getpid()}_{threading.get_ident()}.txt"
    path.write_text(text, encoding="utf-8")
    return path


def _is_google_block_error(msg: str) -> bool:
    low = (msg or "").lower()
    markers = (
        "not a bot",
        "sign in to confirm",
        "confirm you're not a bot",
        "confirm you are not a bot",
        "login_required",
        "could not complete the youtube request",
    )
    return any(m in low for m in markers)


def _is_proxy_throttle_error(msg: str) -> bool:
    low = (msg or "").lower()
    markers = (
        "retry-after",
        "retry after",
        "max_request_rate",
        "max request rate",
        "err::throttle",
        "too many requests",
        "http error 429",
        "status code 429",
        "429 too many",
        "rate exceeded",
        "rate limit",
        "x-scrapfly-reject",
    )
    return any(m in low for m in markers)


def _parse_retry_after_seconds(msg: str | None) -> float | None:
    import re

    if not msg:
        return None
    for pat in (
        r"(?i)retry-after['\"\s:=]+(\d+(?:\.\d+)?)",
        r"(?i)retry after['\"\s:=]+(\d+(?:\.\d+)?)",
        r"(?i)retry_after[=:\s]+(\d+(?:\.\d+)?)",
    ):
        m = re.search(pat, msg)
        if not m:
            continue
        try:
            sec = float(m.group(1))
        except ValueError:
            continue
        if sec > 0:
            return min(sec, 300.0)
    return None


def _wait_retry_after(msg: str, *, label: str = "proxy") -> float:
    """Sleep for Scrapfly Retry-After (default 60s). Returns seconds waited."""
    sec = _parse_retry_after_seconds(msg)
    if sec is None:
        sec = 60.0
    sec = max(1.0, min(float(sec), 300.0))
    set_progress(
        "download",
        f"{label} Retry-After — waiting {sec:.0f}s…",
        detail=f"retry_after={sec:.0f}",
    )
    time.sleep(sec)
    return sec


def _resolve_proxy_url(proxy_url: str | None, *, allow_env: bool = False) -> str | None:
    """Residential proxy from the job payload.

    Env fallbacks are opt-in (force_proxy only). Pods often have YT_PROXY_URL /
    Scrapfly baked in — using that on every job burns Scrapfly even for cookies-only.
    """
    val = (proxy_url or "").strip()
    if val:
        return val if "://" in val else f"http://{val}"
    if not allow_env:
        return None
    for candidate in (
        os.environ.get("SCRAPFLY_PROXY_URL"),
        os.environ.get("SCRAPINGDOG_PROXY_URL"),
        os.environ.get("YT_PROXY_URL"),
    ):
        env_val = (candidate or "").strip()
        if env_val:
            return env_val if "://" in env_val else f"http://{env_val}"
    scrapfly = (os.environ.get("SCRAPFLY_API_KEY") or "").strip()
    if scrapfly:
        from urllib.parse import quote

        country = (os.environ.get("SCRAPFLY_COUNTRY") or "us").strip().lower() or "us"
        opts = (
            os.environ.get("SCRAPFLY_PROXY_OPTS")
            or f"country-{country}-asp-true-renderJs-false-proxyPool-public_residential_pool"
        ).strip()
        return f"http://{quote(opts, safe='-._')}:{quote(scrapfly, safe='')}@proxy.scrapfly.io:7777"
    dog = (os.environ.get("SCRAPINGDOG_API_KEY") or "").strip()
    if dog:
        from urllib.parse import quote

        return f"http://scrapingdog:{quote(dog, safe='')}@proxy.scrapingdog.com:8081"
    return None


def _download_once(
    url: str,
    vid: str,
    *,
    player_client: str | None,
    download_sections: list[str] | None,
    cookies_path: Path | None = None,
    proxy_url: str | None = None,
    proxy_label: str = "proxy",
    proxy_insecure: bool = False,
    referer: str | None = None,
    format_selector: str | None = None,
) -> Path:
    out_tmpl = str(VIDEOS / f"{vid}.%(ext)s")
    fmt = format_selector or "bv*[height<=720]+ba/b[height<=720]/b"
    cmd = [
        "yt-dlp",
        "-f",
        # Cap 720p to cut residential-proxy bandwidth (matches local download.py).
        fmt,
        "--merge-output-format",
        "mp4",
        "-o",
        out_tmpl,
        "--no-playlist",
        "--retries",
        "5",
        "--fragment-retries",
        "5",
        "--sleep-requests",
        "1",
        "--newline",
        # Prefer IPv4 — some pod IPv6 routes trip YouTube bot checks harder.
        "-4",
    ]
    if proxy_url:
        cmd.extend(["--proxy", proxy_url])
        # Scrapfly proxy mode terminates TLS; ScrapingDog keeps normal verify.
        if proxy_insecure:
            cmd.append("--no-check-certificates")
    if referer:
        cmd.extend(["--referer", referer])
        try:
            from urllib.parse import urlparse as _urlparse

            origin = f"{_urlparse(referer).scheme}://{_urlparse(referer).netloc}"
            if origin.startswith("http"):
                cmd.extend(["--add-header", f"Origin:{origin}"])
        except Exception:
            pass
    # android_* clients do not support cookies (yt-dlp skips them).
    cookie_ok_clients = {"web", "web_safari", "mweb", "tv", "tv_embedded", None}
    use_cookies = bool(cookies_path and cookies_path.is_file()) and not referer
    if use_cookies and player_client in cookie_ok_clients:
        cmd.extend(["--cookies", str(cookies_path)])
    elif use_cookies and player_client not in cookie_ok_clients:
        use_cookies = False
    if player_client:
        cmd.extend(["--extractor-args", f"youtube:player_client={player_client}"])
    for sec in download_sections or []:
        section = (sec or "").strip()
        if section:
            cmd.extend(["--download-sections", section, "--force-keyframes-at-cuts"])
    cmd.append(url)
    client_label = (player_client or "default") + ("+cookies" if use_cookies else "")
    if referer and "britishpathe" in (referer + url).lower():
        client_label = "britishpathe+hls"
    elif referer:
        client_label += "+referer"
    if proxy_url:
        client_label += f"+{proxy_label or 'proxy'}"
    set_progress("download", f"yt-dlp ({client_label})", pct=0, detail=url[:120])
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    log_tail: list[str] = []
    t0 = time.time()
    last_ui = 0.0
    try:
        for raw in proc.stdout:
            line = (raw or "").rstrip()
            if not line:
                continue
            log_tail.append(line)
            if len(log_tail) > 60:
                log_tail = log_tail[-60:]
            now = time.time()
            pct_m = _YTDLP_PCT_RE.search(line)
            if pct_m or now - last_ui >= 2.0 or line.startswith("[download]") or "Merging" in line:
                pct = float(pct_m.group(1)) if pct_m else None
                eta_m = _YTDLP_ETA_RE.search(line)
                speed_m = _YTDLP_SPEED_RE.search(line)
                size_m = _YTDLP_SIZE_RE.search(line)
                eta = eta_m.group(1) if eta_m else ""
                speed = speed_m.group(1) if speed_m else ""
                size = size_m.group(1) if size_m else ""
                bits = [
                    x
                    for x in (
                        f"{pct:.1f}%" if pct is not None else None,
                        size,
                        speed,
                        f"ETA {eta}" if eta else None,
                    )
                    if x
                ]
                detail = " · ".join(bits) if bits else line[-160:]
                if "Destination:" in line or "Merging" in line:
                    detail = line[-160:]
                    if "Merging" in line:
                        pct = 99.0
                set_progress(
                    "download",
                    "downloading" if pct is not None else f"yt-dlp ({client_label})",
                    pct=pct,
                    detail=detail,
                )
                last_ui = now
            if now - t0 > 1800:
                proc.kill()
                raise TimeoutError("yt-dlp timeout after 1800s")
        rc = proc.wait(timeout=60)
    except Exception:
        try:
            proc.kill()
        except OSError:
            pass
        raise
    candidates = [
        p
        for p in VIDEOS.glob(f"{vid}.*")
        if p.suffix.lower() in {".mp4", ".webm", ".mkv", ".avi"} and p.stat().st_size > 0
    ]
    if candidates:
        path = max(candidates, key=lambda p: p.stat().st_size)
        # Accept a substantial file even if yt-dlp exited non-zero after warnings.
        if rc == 0 or path.stat().st_size > 100 * 1024:
            return path
    err = _ytdlp_error_summary(log_tail)
    raise RuntimeError(f"yt-dlp failed ({client_label}): {err}")


def download_video(
    url: str,
    title: str,
    download_sections: list[str] | None = None,
    cookies_text: str | None = None,
    proxy_url: str | None = None,
    force_proxy: bool = False,
    proxy_provider: str = "proxy",
    proxy_insecure: bool = False,
    referer: str | None = None,
    source: str | None = None,
    m3u8_url: str | None = None,
) -> Path:
    """Download on the GPU pod only — cookies and/or residential proxy, never the user's PC."""
    VIDEOS.mkdir(parents=True, exist_ok=True)
    WORKDIR.mkdir(parents=True, exist_ok=True)
    vid = slugify(title)
    src = (source or "").strip().lower()
    download_url = (m3u8_url or url or "").strip()
    ref = (referer or "").strip() or None
    # British Pathé preview HLS — no YouTube cookies / residential proxy.
    is_pathe = (
        src == "britishpathe"
        or bool(m3u8_url)
        or "britishpathe.com" in download_url.lower()
        or (ref and "britishpathe.com" in ref.lower())
        or download_url.lower().endswith(".m3u8")
    )
    if is_pathe:
        set_progress(
            "download",
            "British Pathé HLS (no YouTube proxy)…",
            detail=(ref or download_url)[:120],
        )
        for p in VIDEOS.glob(f"{vid}.*"):
            if p.suffix.lower() in {".part", ".ytdl", ".temp"}:
                try:
                    p.unlink()
                except OSError:
                    pass
        # Prefer small preview rungs — full Pathé masters are 200MB+/15min and starve the pool.
        return _download_once(
            download_url,
            vid,
            player_client=None,
            download_sections=None,
            cookies_path=None,
            proxy_url=None,
            referer=ref,
            format_selector=(
                "bv*[height<=360][tbr<=800]+ba/b[height<=360][tbr<=800]/"
                "bv*[height<=480]+ba/b[height<=480]/worst"
            ),
        )

    is_youtube = "youtube.com" in url.lower() or "youtu.be" in url.lower()
    cookies_path = _write_cookies_file(cookies_text)
    has_cookies = cookies_path is not None
    # Only pull baked-in pod env proxy when the job explicitly wants proxy.
    proxy = _resolve_proxy_url(proxy_url, allow_env=bool(force_proxy))
    label = (proxy_provider or "proxy").strip().lower() or "proxy"
    if label in ("none", "auto", ""):
        label = "proxy"
    if proxy and "scrapfly" in proxy.lower():
        label = "scrapfly"
        proxy_insecure = True  # Scrapfly terminates TLS with its own cert.
    # Cookie-capable clients first when we have a jar; then guest clients.
    if is_youtube and has_cookies:
        clients: list[str | None] = ["tv", "web", "mweb", "tv_embedded", "android_vr", "android", None]
    elif is_youtube:
        clients = ["android_vr", "android", "ios", "tv", "mweb", "web", None]
    else:
        clients = [None]

    # Phases stay on the pod: cookies → residential proxy (if configured) → fail.
    phases: list[tuple[str, str | None, bool]] = []
    if force_proxy and proxy:
        phases.append(("proxy", proxy, has_cookies))
        if has_cookies:
            phases.append(("cookies", None, True))
    else:
        if has_cookies:
            phases.append(("cookies", None, True))
        if proxy:
            phases.append(("proxy", proxy, has_cookies))
        if not has_cookies and not proxy:
            phases.append(("guest", None, False))
        elif not has_cookies:
            phases.append(("guest", None, False))

    last_err = "download_failed"
    try:
        attempt_i = 0
        for phase_name, phase_proxy, use_cookie_jar in phases:
            phase_clients = clients
            if phase_name == "proxy":
                set_progress(
                    "download",
                    f"YouTube blocked — {label} on GPU…",
                    detail=label,
                )
                # android first — tv/web often false-DRM / format misses through residential.
                phase_clients = (
                    ["android", "android_vr", "tv", "mweb", "web", None] if is_youtube else [None]
                )
            for client in phase_clients:
                attempt_i += 1
                try:
                    for p in VIDEOS.glob(f"{vid}.*"):
                        if p.suffix.lower() in {".part", ".ytdl", ".temp"}:
                            try:
                                p.unlink()
                            except OSError:
                                pass
                    path = _download_once(
                        url,
                        vid,
                        player_client=client,
                        download_sections=download_sections,
                        cookies_path=cookies_path if use_cookie_jar else None,
                        proxy_url=phase_proxy,
                        proxy_label=label if phase_proxy else "proxy",
                        proxy_insecure=bool(proxy_insecure and phase_proxy),
                    )
                    mb = path.stat().st_size / (1024 * 1024)
                    set_progress(
                        "download",
                        "download complete",
                        pct=100,
                        detail=f"{mb:.1f} MB · {path.name} · {phase_name}",
                    )
                    return path
                except Exception as e:
                    last_err = str(e)
                    set_progress(
                        "download",
                        f"retry download ({phase_name} {attempt_i})",
                        detail=last_err[:200],
                    )
                    if _is_permanent_ytdlp_error(last_err):
                        break
                    # Scrapfly / 429: fail FAST so the PC client can switch to ScrapingDog.
                    # Do not sit on Retry-After for minutes burning the scrape UI.
                    if phase_proxy and _is_proxy_throttle_error(last_err):
                        ra = _parse_retry_after_seconds(last_err) or 60.0
                        set_progress(
                            "download",
                            "scrapfly throttled — failing for ScrapingDog fallback",
                            detail=f"retry_after={ra:.0f}",
                        )
                        raise RuntimeError(
                            f"scrapfly_throttled retry_after={ra:.0f}: {last_err[:240]}"
                        )
                    time.sleep(min(8.0, 1.5 * attempt_i))
            # After a cookie-phase Google block, fall through to residential proxy on this pod.
            if phase_name == "cookies" and proxy and _is_google_block_error(last_err):
                continue
            if _is_permanent_ytdlp_error(last_err):
                break
        set_progress("download", "download failed", detail=last_err[:200])
        raise RuntimeError(last_err)
    finally:
        if cookies_path is not None:
            try:
                cookies_path.unlink(missing_ok=True)
            except OSError:
                pass


def _scan_with_progress(
    video_path: Path,
    video_id: str,
    sample_fps: float,
    score_threshold: float,
    save_crops_dir: Path,
):
    yolo, scorer = _models()
    last_ui = {"t": 0.0}

    def on_progress(time_sec: float, duration: float, n_hits: int) -> None:
        now = time.time()
        if now - last_ui["t"] < 2.0 and time_sec > 0:
            return
        last_ui["t"] = now
        if duration > 0:
            pct = min(99.0, 100.0 * time_sec / duration)
            detail = f"{time_sec:.0f}s / {duration:.0f}s · {n_hits} hits"
        else:
            pct = None
            detail = f"{time_sec:.0f}s · {n_hits} hits"
        set_progress("scan", "scanning on GPU", pct=pct, detail=detail)

    set_progress(
        "scan",
        "GPU scan starting",
        pct=0,
        detail=f"{sample_fps:g} fps sample",
    )
    hits = core_scan_video(
        video_path,
        video_id,
        scorer,
        yolo,
        sample_fps=sample_fps,
        score_threshold=score_threshold,
        save_crops_dir=save_crops_dir,
        on_progress=on_progress,
    )
    set_progress(
        "scan",
        "scan complete",
        pct=100,
        detail=f"{len(hits)} frame hits",
    )
    return hits


def _is_retryable_job_error(msg: str) -> bool:
    low = (msg or "").lower()
    if _is_permanent_ytdlp_error(low):
        return False
    markers = (
        "numpy is not available",
        "no module named 'numpy",
        "numpy._core",
        "cuda out of memory",
        "cublas",
        "temporary failure",
        "connection reset",
        "connection aborted",
        "timed out",
        "timeout",
        "http error 5",
        "503",
        "502",
        "429",
        "fragment",
        "unable to download",
    )
    return any(m in low for m in markers)


def process_job(inp: dict) -> dict:
    global _ollama_prepared
    url = (inp.get("url") or "").strip()
    title = inp.get("title") or "video"
    queue_id = inp.get("queue_id")
    if not url.startswith("http"):
        return {"ok": False, "error": "url_required", "segments": []}
    sample_fps = float(inp.get("sample_fps") or DEFAULT_FPS)
    score_threshold = float(inp.get("score_threshold") or DEFAULT_SCORE_THRESHOLD)
    source_url = inp.get("source_url") or url
    raw_sections = inp.get("download_sections") or []
    if isinstance(raw_sections, str):
        download_sections = [raw_sections]
    elif isinstance(raw_sections, list):
        download_sections = [str(s) for s in raw_sections if str(s).strip()]
    else:
        download_sections = []
    cookies_text = inp.get("cookies_text") or inp.get("cookies") or ""
    if not isinstance(cookies_text, str):
        cookies_text = ""
    proxy_url = inp.get("proxy_url") or ""
    if not isinstance(proxy_url, str):
        proxy_url = ""
    proxy_url = proxy_url.strip() or None
    force_proxy = bool(inp.get("force_proxy"))
    proxy_provider = str(inp.get("proxy_provider") or "proxy").strip() or "proxy"
    proxy_insecure = bool(inp.get("proxy_insecure"))
    referer = inp.get("referer") or inp.get("http_referer") or ""
    if not isinstance(referer, str):
        referer = ""
    referer = referer.strip() or None
    source = str(inp.get("source") or "").strip() or None
    m3u8_url = inp.get("m3u8_url") or ""
    if not isinstance(m3u8_url, str):
        m3u8_url = ""
    m3u8_url = m3u8_url.strip() or None
    preloaded = (inp.get("video_path") or "").strip()
    skip_download = bool(inp.get("skip_download") or preloaded)
    video_id = slugify(title)
    path = None
    crop_dir = CROPS / video_id
    _tls.queue_id = queue_id
    set_progress(
        "queued",
        "job accepted",
        pct=0,
        detail=title[:120],
        queue_id=queue_id,
        title=title,
        url=url,
    )
    max_attempts = 3
    last_err = ""
    last_tb = ""
    try:
        WORKDIR.mkdir(parents=True, exist_ok=True)
        for attempt in range(1, max_attempts + 1):
            path = None
            try:
                if skip_download and preloaded:
                    path = Path(preloaded)
                    if not path.is_file() or path.stat().st_size < 1024:
                        raise RuntimeError(f"preloaded_video_missing: {preloaded}")
                    set_progress("download", "using uploaded video", pct=100, detail=path.name)
                else:
                    # Downloads may overlap across concurrent /scan requests.
                    # Always on this GPU pod (cookies / residential proxy) — never the user's PC.
                    path = download_video(
                        url,
                        title,
                        download_sections=download_sections or None,
                        cookies_text=cookies_text or None,
                        proxy_url=proxy_url,
                        force_proxy=force_proxy,
                        proxy_provider=proxy_provider,
                        proxy_insecure=proxy_insecure,
                        referer=referer,
                        source=source,
                        m3u8_url=m3u8_url,
                    )
                video_id = path.stem
                crop_dir = CROPS / video_id
                with _gpu_lock:
                    try:
                        hits = _scan_with_progress(
                            path, video_id, sample_fps, score_threshold, crop_dir
                        )
                    except Exception as scan_err:
                        # Recover from broken torch/numpy bridge by reloading models.
                        if "numpy" in str(scan_err).lower():
                            reset_models()
                            hits = _scan_with_progress(
                                path, video_id, sample_fps, score_threshold, crop_dir
                            )
                        else:
                            raise
                # Free CLIP activations before Ollama shares the GPU.
                try:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass
                raw_segs = aggregate_segments_dicts(hits, video_id)
                segments = []
                n_segs = len(raw_segs)
                for i, seg in enumerate(raw_segs, 1):
                    group = seg.pop("_hits")
                    set_progress(
                        "upload",
                        f"uploading stills {i}/{n_segs}" if n_segs else "no segments",
                        pct=(100.0 * i / n_segs) if n_segs else 100,
                        detail=f"{len(hits)} frame hits → segments",
                        queue_id=queue_id,
                    )
                    image_url = None
                    notes = ""
                    still_b64 = None
                    still_flags: list[str] = []
                    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                        tmp_path = Path(tmp.name)
                    wrote = write_sheet_from_crops(group, tmp_path)
                    if wrote:
                        # Vision verify on the pod with the local JPEG — Catbox URLs often 0-byte / blocked.
                        backend_raw = (
                            str(inp.get("verify_backend") or os.environ.get("VERIFY_BACKEND") or "openai")
                            .strip()
                            .lower()
                        )
                        if backend_raw in (
                            "ollama_then_openai",
                            "cascade",
                            "ollama+openai",
                            "vlm_then_openai",
                            "open_vlm_then_openai",
                        ):
                            backend = "ollama_then_openai"
                        elif backend_raw in ("open_vlm", "vlm", "ollama", "qwen"):
                            backend = "open_vlm"
                        else:
                            backend = "openai"
                        os.environ["VERIFY_BACKEND"] = backend
                        os.environ.setdefault("OPENAI_VERIFY", "1")
                        vlm_base = (
                            inp.get("open_vlm_base_url") or os.environ.get("OPEN_VLM_BASE_URL") or ""
                        ).strip()
                        oai_key = (inp.get("openai_api_key") or os.environ.get("OPENAI_API_KEY") or "").strip()
                        if backend in ("open_vlm", "ollama_then_openai"):
                            can_verify = bool(vlm_base)
                        else:
                            can_verify = bool(oai_key)
                        if can_verify:
                            if backend in ("open_vlm", "ollama_then_openai"):
                                os.environ["SHTETL_POD"] = "1"
                                os.environ["OPEN_VLM_BASE_URL"] = vlm_base or "http://127.0.0.1:11434/v1"
                                vlm_model = (
                                    inp.get("open_vlm_model")
                                    or os.environ.get("OPEN_VLM_MODEL")
                                    or ""
                                ).strip()
                                if vlm_model:
                                    os.environ["OPEN_VLM_MODEL"] = vlm_model
                                vlm_key = (
                                    inp.get("open_vlm_api_key")
                                    or os.environ.get("OPEN_VLM_API_KEY")
                                    or ""
                                ).strip()
                                if vlm_key:
                                    os.environ["OPEN_VLM_API_KEY"] = vlm_key
                                # Once per process: serve + wait for model (bg pull from warm).
                                with _ollama_prep_lock:
                                    if not _ollama_prepared:
                                        try:
                                            from ollama_pod import (
                                                ensure_ollama,
                                                start_background_pull,
                                                wait_for_model,
                                            )

                                            start_background_pull()
                                            ensure_ollama(pull=False, use_cache=True)
                                            ok = wait_for_model(timeout_sec=180.0)
                                            _ollama_prepared = bool(ok)
                                            print(
                                                f"[shtetl] ollama prepare ok={ok}",
                                                flush=True,
                                            )
                                        except Exception as ol_err:
                                            print(
                                                f"[shtetl] ollama ensure: {ol_err}",
                                                flush=True,
                                            )
                            if backend in ("openai", "ollama_then_openai") and oai_key:
                                os.environ["OPENAI_API_KEY"] = oai_key
                                oai_model = (
                                    inp.get("openai_model") or os.environ.get("OPENAI_MODEL") or ""
                                ).strip()
                                if oai_model:
                                    os.environ["OPENAI_MODEL"] = oai_model
                            label = {
                                "ollama_then_openai": "Ollama→OpenAI",
                                "open_vlm": "Open VLM",
                            }.get(backend, "OpenAI")
                            try:
                                from openai_verify import format_verdict_notes, verify_still

                                set_progress(
                                    "upload",
                                    f"{label} verify {i}/{n_segs}",
                                    pct=(100.0 * i / n_segs) if n_segs else 100,
                                    detail=f"local still → {label}",
                                    queue_id=queue_id,
                                )
                                notes = format_verdict_notes(verify_still(image_path=wrote))
                            except Exception as oai_err:
                                prefix = "vlm" if backend == "open_vlm" else "openai"
                                notes = f"{prefix}:drop conf=0.00 pod_verify:{oai_err}"[:500]
                        still_raw, still_b64 = _encode_review_still(Path(wrote))
                        # Durable on-disk still for GET /still (survives large JSON truncation).
                        try:
                            dest = still_file_path(queue_id, i)
                            dest.parent.mkdir(parents=True, exist_ok=True)
                            if still_raw:
                                dest.write_bytes(still_raw)
                            else:
                                shutil.copyfile(wrote, dest)
                        except OSError:
                            still_flags.append("still_disk_fail")
                        image_url = upload_image(wrote, user_agent=USER_AGENT)
                        if not image_url:
                            still_flags.append("upload_failed")
                        if not still_b64:
                            still_flags.append("still_b64_missing")
                    try:
                        tmp_path.unlink(missing_ok=True)
                    except OSError:
                        pass
                    # Prefix flags so they are not truncated by long OpenAI notes.
                    if still_flags:
                        notes = (" ".join(still_flags) + (" " + notes if notes else "")).strip()[
                            :500
                        ]
                    seg_out = {
                        "video_id": video_id,
                        "start_sec": seg["start_sec"],
                        "end_sec": seg["end_sec"],
                        "peak_score": seg["peak_score"],
                        "mean_score": seg["mean_score"],
                        "rank_score": seg["rank_score"],
                        "hit_count": seg["hit_count"],
                        "best_cue": seg["best_cue"],
                        "source_url": source_url,
                        "image_url": image_url,
                        "notes": notes,
                        "still_index": i,
                    }
                    # Always inline when possible — Catbox expires; /still is backup.
                    # Cap raised: PC hydrate also GETs /still if this is omitted.
                    if still_b64 and len(still_b64) <= 550_000:
                        seg_out["still_b64"] = still_b64
                    elif still_b64:
                        seg_out["notes"] = (
                            f"still_b64_too_large {seg_out.get('notes') or ''}".strip()
                        )[:500]
                    segments.append(seg_out)
                set_progress(
                    "done",
                    f"done · {len(segments)} segments",
                    pct=100,
                    detail=f"{len(hits)} frame hits",
                    queue_id=queue_id,
                )
                top_frames = [
                    {
                        "t": round(h.time_sec, 2),
                        "score": round(h.score, 4),
                        "pos": round(h.pos_score, 4),
                        "neg": round(h.neg_score, 4),
                        "cue": h.best_cue,
                    }
                    for h in sorted(hits, key=lambda x: -x.score)[:20]
                ]
                return {
                    "ok": True,
                    "video_id": video_id,
                    "segments": segments,
                    "n_hits": len(segments),
                    "n_frame_hits": len(hits),
                    "top_frames": top_frames,
                    "model": CLIP_MODEL,
                    "pretrained": CLIP_PRETRAINED,
                    "device": "cuda" if torch.cuda.is_available() else "cpu",
                    "error": None,
                }
            except Exception as e:
                last_err = str(e)[:1000]
                last_tb = traceback.format_exc()[-2000:]
                if path and path.exists():
                    try:
                        path.unlink()
                    except OSError:
                        pass
                    path = None
                shutil.rmtree(crop_dir, ignore_errors=True)
                if attempt < max_attempts and _is_retryable_job_error(last_err):
                    set_progress(
                        "queued",
                        f"retrying job ({attempt + 1}/{max_attempts})",
                        detail=last_err[:200],
                        queue_id=queue_id,
                    )
                    if "numpy" in last_err.lower():
                        reset_models()
                    time.sleep(2.0 * attempt)
                    continue
                break
        set_progress("error", "job failed", detail=last_err[:200], queue_id=queue_id)
        return {
            "ok": False,
            "error": last_err or "job_failed",
            "detail": last_tb,
            "segments": [],
        }
    finally:
        shutil.rmtree(crop_dir, ignore_errors=True)
        if path and path.exists():
            try:
                path.unlink()
            except OSError:
                pass
        for p in VIDEOS.glob(f"{video_id}*"):
            try:
                p.unlink()
            except OSError:
                pass
        time.sleep(0.2)
        # Keep progress until /result consumes the job (async client polls).
        _tls.queue_id = None


def handler(event: dict) -> dict:
    inp = event.get("input") or event
    return process_job(inp if isinstance(inp, dict) else {})


if __name__ == "__main__":
    import runpod

    runpod.serverless.start({"handler": handler})
