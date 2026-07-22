"""Live CMD dashboard — friendly copy + rich ANSI graphics. Hot-reloads on save."""

from __future__ import annotations

import copy
import importlib
import os
import re
import shutil
import sys
import threading
import time
from typing import Any

_lock = threading.Lock()
_enabled = False
_state: dict[str, Any] = {
    "mode": "idle",  # idle | discover | setup | scrape | both | done | error
    "headline": "Waiting for you to start in the browser",
    "sub": "",
    "discovering": False,
    "discover_headline": "",
    "discover_sub": "",
    "scraping": False,
    "done": 0,
    "total": 0,
    "hits": 0,
    "errors": 0,
    "live": [],  # list[{title, phase, detail}]
    "note": "",
    "note_until": 0.0,
    "url": "http://127.0.0.1:8787",
    "review": "http://127.0.0.1:8787/review",
    "last_draw": 0.0,
    "started_at": time.time(),
}
_last_frame_lines = 0
_FIXED_HEIGHT = 48
_SRC_MTIME = 0.0
_ticker: threading.Thread | None = None
_ticker_stop = threading.Event()
_use_color = True

# ── ANSI helpers ─────────────────────────────────────────────────────────────
_RESET = "\033[0m"
_BOLD = "\033[1m"
_DIM = "\033[2m"
_ITALIC = "\033[3m"


def _c(code: str, text: str) -> str:
    if not _use_color:
        return text
    return f"{code}{text}{_RESET}"


def _fg(n: int) -> str:
    return f"\033[38;5;{n}m"


def _bg(n: int) -> str:
    return f"\033[48;5;{n}m"


# Palette (256-color — works in Windows Terminal / modern conhost with VT)
_GOLD = _fg(178)
_CREAM = _fg(223)
_TEAL = _fg(73)
_SKY = _fg(74)
_GREEN = _fg(114)
_AMBER = _fg(214)
_ROSE = _fg(168)
_SLATE = _fg(245)
_WHITE = _fg(255)
_MUTED = _fg(242)

_PHASE_FRIENDLY = {
    "queued": "In line",
    "download": "Getting the video",
    "downloading": "Getting the video",
    "scan": "Looking carefully",
    "scanning": "Looking carefully",
    "upload": "Saving pictures",
    "uploading": "Saving pictures",
    "done": "Finished",
    "error": "Skipped",
    "idle": "Idle",
}

_PHASE_COLOR = {
    "queued": _SLATE,
    "download": _SKY,
    "downloading": _SKY,
    "scan": _TEAL,
    "scanning": _TEAL,
    "upload": _GREEN,
    "uploading": _GREEN,
    "done": _GREEN,
    "error": _ROSE,
    "idle": _MUTED,
}

_SPINNER = ("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")
_PULSE = ("·", "•", "●", "•")


def _visible_len(text: str) -> int:
    return len(re.sub(r"\033\[[0-9;]*m", "", text or ""))


def _pad_visible(text: str, width: int) -> str:
    """Left-justify accounting for ANSI escape length."""
    vis = _visible_len(text)
    if vis >= width:
        # Clip by visible chars while keeping codes mostly intact
        return _clip_ansi(text, width)
    return text + (" " * (width - vis))


def _clip_ansi(text: str, n: int) -> str:
    """Clip to n visible chars; append ellipsis; keep reset at end."""
    if n <= 0:
        return ""
    out = []
    vis = 0
    i = 0
    s = text or ""
    while i < len(s) and vis < n:
        if s[i] == "\033":
            m = re.match(r"\033\[[0-9;]*m", s[i:])
            if m:
                out.append(m.group(0))
                i += len(m.group(0))
                continue
        out.append(s[i])
        vis += 1
        i += 1
    result = "".join(out)
    if i < len(s) and n > 1:
        # Replace last visible char with ellipsis
        result = _clip_ansi(text, n - 1) + "…"
    return result + (_RESET if _use_color and "\033[" in result else "")


def _clean_text(text: str) -> str:
    t = (text or "").replace("\n", " ").replace("\r", " ")
    # Fix common mojibake / replacement glyphs from bad encodings
    t = t.replace("\ufffd", "'").replace("�", "'")
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _clip(text: str, n: int) -> str:
    t = _clean_text(text)
    if len(t) <= n:
        return t
    if n <= 1:
        return t[:n]
    return t[: n - 1] + "…"


def _console_width() -> int:
    try:
        cols = shutil.get_terminal_size(fallback=(110, 40)).columns
    except Exception:
        cols = 110
    return max(76, min(int(cols) - 4, 118))


def _spinner(t: float | None = None) -> str:
    t = time.time() if t is None else t
    return _SPINNER[int(t * 10) % len(_SPINNER)]


def _pulse(t: float | None = None) -> str:
    t = time.time() if t is None else t
    return _PULSE[int(t * 3) % len(_PULSE)]


def _block_bar(frac: float, width: int = 28, *, fill_color: str = "") -> str:
    """Smooth unicode block progress bar."""
    frac = max(0.0, min(1.0, float(frac)))
    blocks = " ▏▎▍▌▋▊▉█"
    exact = frac * width
    full = int(exact)
    rem = exact - full
    partial = blocks[min(8, int(rem * 8) + (1 if rem > 0 else 0))] if full < width else ""
    empty = width - full - (1 if partial else 0)
    filled = "█" * full + partial
    blank = "░" * max(0, empty)
    if _use_color and fill_color:
        return f"{fill_color}{filled}{_MUTED}{blank}{_RESET}"
    return f"{filled}{blank}"


def _pct(done: int, total: int) -> float:
    total = max(int(total), 1)
    done = max(0, min(int(done), total))
    return 100.0 * done / total


def _enable_vt() -> None:
    global _use_color
    if os.name == "nt":
        try:
            import ctypes

            kernel32 = ctypes.windll.kernel32
            handle = kernel32.GetStdHandle(-11)
            mode = ctypes.c_uint32()
            if kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
                kernel32.SetConsoleMode(handle, mode.value | 0x0004)
                _use_color = True
                return
        except Exception:
            pass
    _use_color = sys.stdout.isatty() if hasattr(sys.stdout, "isatty") else True


def _hot_reload() -> bool:
    """If this file changed on disk, reload module and restore live state."""
    global _SRC_MTIME
    try:
        m = os.path.getmtime(__file__)
    except OSError:
        return False
    if _SRC_MTIME <= 0:
        _SRC_MTIME = m
        return False
    if m <= _SRC_MTIME:
        return False
    # Keep ticker thread + stop event across reload (avoid duplicate animators).
    saved_state = copy.deepcopy(_state)
    saved_enabled = _enabled
    saved_ticker = _ticker
    saved_stop = _ticker_stop
    mod = importlib.reload(sys.modules[__name__])
    mod._state.clear()
    mod._state.update(saved_state)
    mod._enabled = saved_enabled
    mod._SRC_MTIME = m
    mod._ticker = saved_ticker
    mod._ticker_stop = saved_stop
    mod._enable_vt()
    # Brief on-screen confirmation that the new look was picked up live.
    mod._state["note"] = "Dashboard updated live (no restart)."
    mod._state["note_until"] = time.time() + 5.0
    return True


def refresh_from_jobs() -> bool:
    """Resync dashboard counters / live rows from the DB + in-memory Pathé live map."""
    if not is_enabled():
        return False
    try:
        from db import get_job
    except Exception:
        return False

    try:
        pj = get_job("pathe_scrape") or {}
    except Exception:
        pj = {}
    try:
        yj = get_job("scrape") or {}
    except Exception:
        yj = {}

    pathe_run = str(pj.get("status") or "") == "running"
    yt_run = str(yj.get("status") or "") == "running"
    if not pathe_run and not yt_run:
        return False

    live: list[dict[str, str]] = []
    if pathe_run:
        try:
            from pipeline_pathe import pathe_live_snapshot

            live = [
                {
                    "title": str(info.get("title") or ""),
                    "phase": str(info.get("phase") or ""),
                    "detail": str(info.get("detail") or ""),
                }
                for info in (pathe_live_snapshot() or [])[:8]
            ]
        except Exception:
            live = list(_state.get("live") or [])
        set_scrape(
            done=int(pj.get("completed") or 0),
            total=max(int(pj.get("total") or 0), 1),
            hits=int(pj.get("hits") or 0),
            errors=0,
            live=live,
            headline="Looking through British Pathé…",
            sub=str(pj.get("message") or "")[:120],
        )
        return True

    # YouTube scrape — keep existing live rows; just heal the counters.
    set_scrape(
        done=int(yj.get("completed") or 0),
        total=max(int(yj.get("total") or 0), 1),
        hits=int(yj.get("hits") or 0),
        errors=0,
        live=list(_state.get("live") or []),
        headline="Looking through videos…",
        sub=str(yj.get("message") or "")[:120],
    )
    return True


def _ensure_ticker() -> None:
    """Animate spinner / pick up hot-reloads even when scrape status is quiet."""
    global _ticker
    if _ticker is not None and _ticker.is_alive():
        return
    _ticker_stop.clear()

    def _loop() -> None:
        # ~1 Hz is enough for spinner; faster redraws burn CPU in Windows conhost.
        ticks = 0
        while not _ticker_stop.wait(1.0):
            try:
                if _hot_reload():
                    _sys_draw = sys.modules[__name__].draw
                    _sys_draw(force=True)
                    continue
                if not is_enabled():
                    continue
                ticks += 1
                # Heal stale counters every ~3s (e.g. after a stuck frame).
                if ticks % 3 == 0:
                    refresh_from_jobs()
                with _lock:
                    mode = _state.get("mode")
                if mode in ("scrape", "setup", "discover", "both"):
                    draw(force=True)
            except Exception:
                pass

    _ticker = threading.Thread(target=_loop, name="console-dash", daemon=True)
    _ticker.start()


def enable() -> None:
    global _enabled
    _enable_vt()
    with _lock:
        _enabled = True
        _state["started_at"] = time.time()
    _ensure_ticker()
    draw(force=True)


def is_enabled() -> bool:
    with _lock:
        return _enabled


def friendly_phase(phase: str) -> str:
    p = (phase or "").strip().lower()
    return _PHASE_FRIENDLY.get(p, "Working")


def friendly_setup_line(raw: str) -> str:
    s = (raw or "").strip()
    low = s.lower()
    if not s:
        return ""
    if "terminat" in low:
        return "Cleaning up the old cloud computer…"
    if "recreat" in low or "creating" in low:
        return "Starting a cloud computer…"
    if "waiting for gpu" in low or "first boot" in low:
        m = re.search(r"(\d+)s", s)
        secs = m.group(1) if m else ""
        extra = f" ({secs}s so far)" if secs else ""
        return f"Almost ready — first start can take a few minutes{extra}"
    if "pod ready" in low or "gpu pod ready" in low or "pods ready" in low:
        return "Cloud computer is ready"
    if "trying nvidia" in low or "image " in low:
        return "Picking a machine…"
    if "api key" in low:
        return "Need your cloud key in Settings"
    if "error" in low or "fail" in low:
        return _friendly_detail(s)[:160]
    if "proxy.runpod" in low or "http" in low:
        return "Setting things up…"
    return s[:140]


def _plain_discover(message: str) -> str:
    s = (message or "").strip()
    low = s.lower()
    if "pathé" in low or "pathe" in low:
        m = re.search(r"(\d[\d,]*)\s*/\s*(\d[\d,]*)\s*urls", low)
        if m:
            return f"Finding Pathé clips · {m.group(1)} so far (goal {m.group(2)})…"
        m = re.search(r"queued\s+(\d[\d,]*)", low)
        if m:
            return f"Added {m.group(1)} Pathé clips to your list…"
        if "page" in low:
            return "Browsing the British Pathé catalog…"
        if "empty window" in low or "retry" in low:
            return "Pathé page was blank — trying again…"
        if "done" in low or "found" in low:
            m = re.search(r"(\d[\d,]*)\s*unique", low) or re.search(
                r"found\s+(\d[\d,]*)", low
            )
            if m:
                return f"Finished Pathé discover · {m.group(1)} clips"
            return "Finished Pathé discover"
        if "scrape started" in low:
            return "Pathé list growing — scrape started alongside…"
        return "Finding British Pathé clips…"
    if "youtube" in low or "listing" in low:
        return "Reading the channel list…"
    if "crawl" in low:
        return "Looking through the page…"
    if "insert" in low or "queue" in low or "added" in low:
        return "Adding videos to your list…"
    if "done" in low or "complete" in low:
        return "Finished finding videos"
    return s[:140] if s else "Working…"


def _friendly_detail(detail: str) -> str:
    """Turn pod/yt-dlp noise into short, calm status lines."""
    d = _clean_text(detail)
    if not d:
        return ""
    low = d.lower()

    # Hide / rewrite known technical noise
    if "deprecated feature" in low or "support for python version" in low:
        return "Getting the video…"
    if "numpy is not available" in low:
        return "Cloud hiccup — trying again…"
    if "members-only" in low or "members only" in low:
        return "This video is members-only (skipped)"
    if "private video" in low:
        return "Private video (skipped)"
    if "video unavailable" in low:
        return "Video unavailable (skipped)"
    if "resolving british pathé" in low or "resolving british pathe" in low:
        return "Looking up the Pathé preview stream…"
    if "pathé hls" in low or "pathe hls" in low or "m3u8" in low:
        return "Downloading Pathé preview (HLS)…"
    if "http_503" in low or "http_502" in low or "gpu busy" in low:
        return "GPU proxy busy — retrying…"
    if "http_524" in low or "gateway timeout" in low:
        return "Cloud gateway timed out — retrying…"
    if "pod ready ·" in low or "creating " in low and "pod" in low:
        return "Waiting for a free GPU…"
    if "british pathe" in low or "britishpathe" in low or "pathé" in low:
        if "waiting for discover" in low:
            return "Waiting for more Pathé clips to be found…"
        if "scrape" in low and ("done" in low or "hits" in low):
            return d[:90]
        return "Working on a British Pathé clip…"
    if (
        "falling back to scrapingdog" in low
        or ("scrapfly failed" in low and "scrapingdog" in low)
        or ("scrapfly" in low and "scrapingdog" in low and "switch" in low)
        or "failing for scrapingdog" in low
        or "falling back to iproyal" in low  # legacy status lines
    ):
        return "Scrapfly failed — switching to ScrapingDog…"
    if "retry-after" in low or ("scrapfly" in low and "waiting" in low):
        return "Scrapfly rate limit — waiting before the next request…"
    if any(p in low for p in ("scrapfly", "scrapingdog")) and "gpu" in low:
        if "blocked" in low or "retry" in low:
            return "YouTube blocked — residential proxy download on the GPU…"
        return "Downloading via residential proxy on the GPU…"
    if "sign in to confirm" in low or "not a bot" in low:
        return "YouTube blocked the cloud IP — retrying with proxy/cookies on GPU…"
    if "local_fallback" in low or "local_pc_download" in low:
        return "Cloud download failed (PC download is off)"
    if "downloading on this pc" in low or "browser cookies" in low:
        # Legacy status from older runs — should not appear after restart.
        return "Downloading on the GPU (cookies)…"
    if "uploading" in low and ("mb" in low or "gpu" in low):
        return "Finishing on GPU…"
    if "members-only" in low or "members only" in low:
        return "Members-only video — skipped"
    if "yt-dlp failed" in low or "download_failed" in low:
        if "private" in low:
            return "Private video — skipped"
        if "unavailable" in low:
            return "Video unavailable — skipped"
        return "Couldn't download — YouTube blocked or link broken"
    if "pod_bad_json" in low or "524" in low:
        return "Cloud timed out — retrying…"
    if "pod_scan" in low or "pod retry" in low:
        return "Retrying on the cloud computer…"
    if "saving segments" in low or "sqlite" in low:
        return "Saving the best stills…"
    if "download complete" in low:
        m = re.search(r"([\d.]+)\s*MB", d, re.I)
        return f"Downloaded{(' · ' + m.group(1) + ' MB') if m else ''}"
    if "scanning on gpu" in low or (low.startswith("scan") and "hit" in low):
        # Keep "515s / 1209s · 0 hits" style if present
        m = re.search(r"([\d.]+s\s*/\s*[\d.]+s.*?)$", d)
        if m:
            return f"Watching frames · {m.group(1)}"
        hits = re.search(r"(\d+)\s*hits?", d, re.I)
        return "Watching frames" + (f" · {hits.group(0)}" if hits else "")
    if "uploading stills" in low:
        return "Uploading stills…"
    if "job failed" in low:
        rest = re.sub(r"(?i)job failed\s*·?\s*", "", d).strip(" ·")
        return _friendly_detail(rest) if rest else "Had a problem with this video"
    if "error ·" in low:
        rest = d.split("·", 1)[-1].strip()
        return _friendly_detail(rest) if rest else "Had a problem"

    # Keep compact progress crumbs
    if re.search(r"\d+(?:\.\d+)?%", d) or "eta" in low or "mib" in low or "mb" in low:
        # Prefer "42% · 21 MB · ETA 00:12" style
        pct = re.search(r"(\d+(?:\.\d+)?)%", d)
        size = re.search(r"([\d.]+\s*[KMG]i?B)", d, re.I)
        eta = re.search(r"ETA\s+(\S+)", d, re.I)
        speed = re.search(r"([\d.]+\s*[KMG]i?B/s)", d, re.I)
        bits = []
        if pct:
            bits.append(f"{pct.group(1)}%")
        if size:
            bits.append(size.group(1).replace("iB", "B"))
        if speed:
            bits.append(speed.group(1).replace("iB", "B"))
        if eta:
            bits.append(f"ETA {eta.group(1)}")
        if bits:
            return " · ".join(bits)
    if len(d) > 90:
        return d[:87] + "…"
    return d


def _sync_mode_locked() -> None:
    """Pick dashboard mode from discover/scrape flags (caller holds _lock)."""
    discovering = bool(_state.get("discovering"))
    scraping = bool(_state.get("scraping"))
    if discovering and scraping:
        _state["mode"] = "both"
        _state["headline"] = "Discover + scrape running together…"
        _state["sub"] = ""
    elif scraping:
        _state["mode"] = "scrape"
        if not _state.get("headline") or "Finding" in str(_state.get("headline") or ""):
            _state["headline"] = "Looking through videos…"
    elif discovering:
        _state["mode"] = "discover"
        _state["headline"] = _state.get("discover_headline") or "Finding videos…"
        _state["sub"] = _state.get("discover_sub") or ""
    # else leave mode alone (setup/done/error/idle set explicitly)


def set_idle(*, note: str = "") -> None:
    with _lock:
        _state.update(
            {
                "mode": "idle",
                "headline": "Waiting for you to start in the browser",
                "sub": "Discover videos, then press Go",
                "discovering": False,
                "discover_headline": "",
                "discover_sub": "",
                "scraping": False,
                "done": 0,
                "total": 0,
                "hits": 0,
                "errors": 0,
                "live": [],
                "note": note or "",
            }
        )
    draw(force=True)


def set_setup(message: str) -> None:
    with _lock:
        _state["mode"] = "setup"
        _state["headline"] = "Getting ready…"
        _state["sub"] = friendly_setup_line(message) or "Please wait"
        # Keep scrape live rows if a scrape is already going.
        if not _state.get("scraping"):
            _state["live"] = []
    draw()


def set_discover(message: str, *, headline: str | None = None) -> None:
    with _lock:
        low = (message or "").lower()
        done_msg = (
            "discover done" in low
            or ("found " in low and "added" in low)
            or ("unique asset" in low and "done" in low)
        )
        _state["discovering"] = not done_msg
        _state["discover_headline"] = headline or "Finding videos…"
        _state["discover_sub"] = _plain_discover(message)
        _sync_mode_locked()
        if _state["mode"] == "discover":
            _state["headline"] = _state["discover_headline"]
            _state["sub"] = _state["discover_sub"]
    draw()


def set_scrape(
    *,
    done: int,
    total: int,
    hits: int,
    errors: int,
    live: list[dict[str, str]],
    headline: str | None = None,
    sub: str = "",
    reset_session: bool = False,
) -> None:
    with _lock:
        _state["scraping"] = True
        if reset_session or not _state.get("started_at"):
            _state["started_at"] = time.time()
        _state["done"] = int(done)
        _state["total"] = int(total)
        _state["hits"] = int(hits)
        _state["errors"] = int(errors)
        _state["live"] = live[:8]
        if headline:
            _state["headline"] = headline
        elif not _state.get("discovering"):
            _state["headline"] = "Looking through videos…"
        if sub or not _state.get("discovering"):
            _state["sub"] = sub or ""
        _sync_mode_locked()
        if _state["mode"] == "scrape" and headline:
            _state["headline"] = headline
    # Do not draw() here — scrape threads were blocking on Windows stdout while
    # the console ticker already redraws ~1 Hz.


def touch_live(live: list[dict[str, str]]) -> None:
    """Refresh in-flight worker rows without changing done/hits counters."""
    with _lock:
        if not _state.get("scraping"):
            return
        _state["live"] = live[:8]


def set_done(*, hits: int, errors: int, done: int) -> None:
    with _lock:
        _state["scraping"] = False
        _state["hits"] = int(hits)
        _state["errors"] = int(errors)
        _state["done"] = int(done)
        _state["live"] = []
        # If discover is still running, keep showing it instead of "all done".
        if _state.get("discovering"):
            _sync_mode_locked()
            _state["headline"] = _state.get("discover_headline") or "Finding videos…"
            _state["sub"] = (
                f"Scrape finished · {hits} clip(s). "
                + (_state.get("discover_sub") or "Still discovering…")
            )
        else:
            _state["mode"] = "done"
            _state["headline"] = "All done for now"
            if hits:
                _state["sub"] = f"Found {hits} possible clip(s) — open Review to check them"
            else:
                _state["sub"] = (
                    "No strong matches this round — try more videos or Review later"
                )
    draw(force=True)


def set_error(message: str) -> None:
    with _lock:
        _state["scraping"] = False
        _state["discovering"] = False
        _state["mode"] = "error"
        _state["headline"] = "Something went wrong"
        _state["sub"] = _friendly_detail(message or "Unknown problem")[:160]
        _state["live"] = []
    draw(force=True)


def note_from_status(message: str, *, job: str = "") -> bool:
    if not is_enabled():
        return False
    msg = (message or "").strip()
    j = (job or "").strip().lower()
    if j in ("discover", "pathe_discover"):
        set_discover(
            msg,
            headline=(
                "Finding British Pathé clips…"
                if "pathe" in j or "pathé" in msg.lower() or "pathe" in msg.lower()
                else None
            ),
        )
        return True
    if j == "pathe_scrape":
        low = msg.lower()
        if "done ·" in low or low.startswith("pathé scrape done") or "pathe scrape done" in low:
            # Final line handled by pipeline via set_done; keep dashboard calm.
            return True
        if "error" in low and "hits" not in low:
            # Per-item errors stay in live rows; don't flip whole dash to error.
            return True
        # Progress is published via set_scrape from pipeline_pathe.
        return True
    if j in ("scrape", "runpod", "go", ""):
        low = msg.lower()
        if any(
            k in low
            for k in (
                "pod",
                "gpu",
                "runpod",
                "creating",
                "terminat",
                "waiting for",
                "bootstrap",
                "docker",
                "image ",
                "workers",
                "model",
            )
        ) and "done #" not in low and "error #" not in low:
            with _lock:
                mode = _state["mode"]
            if mode in ("idle", "setup", "discover", "both") or "spinning" in low or "ensuring" in low or "creating" in low:
                set_setup(msg)
            return True
        if msg.startswith("DONE ") or (" → " in msg and "hit" in low):
            return True
        if msg.startswith("ERROR ") or msg.startswith("  →"):
            return True
        if re.match(r"^\d+/\d+ done", msg):
            return True
    return False


def draw(*, force: bool = False) -> None:
    global _last_frame_lines
    if _hot_reload():
        sys.modules[__name__].draw(force=True)
        return
    if not is_enabled():
        return
    now = time.time()
    with _lock:
        # Allow animation ticks (~3/sec) without starving scrape updates
        if not force and (now - float(_state["last_draw"])) < 0.28:
            return
        _state["last_draw"] = now
        snap = dict(_state)
        snap["live"] = list(_state["live"])

    width = _console_width()
    lines = _build(snap, width, now)
    while len(lines) < _FIXED_HEIGHT:
        lines.append("")
    lines = lines[:_FIXED_HEIGHT]

    buf = "\033[H\033[2J\033[3J" + "\n".join(lines) + "\n"
    try:
        sys.stdout.write(buf)
        sys.stdout.flush()
        _last_frame_lines = len(lines)
    except Exception:
        try:
            os.system("cls" if os.name == "nt" else "clear")
            print("\n".join(lines), flush=True)
            _last_frame_lines = len(lines)
        except Exception:
            pass


def _phase_badge(phase: str) -> str:
    key = (phase or "").strip().lower()
    label = friendly_phase(key)
    color = _PHASE_COLOR.get(key, _SLATE)
    return _c(f"{_BOLD}{color}", f"● {label}")


def _stat_chip(label: str, value: str, color: str) -> str:
    return f"{_c(_MUTED, label)} {_c(f'{_BOLD}{color}', value)}"


def _mini_bar_from_detail(detail: str, width: int = 12) -> str:
    m = re.search(r"(\d+(?:\.\d+)?)%", detail or "")
    if not m:
        return ""
    frac = float(m.group(1)) / 100.0
    return _block_bar(frac, width, fill_color=_TEAL)


def _elapsed_str(started: float) -> str:
    secs = max(0, int(time.time() - started))
    if secs < 60:
        return f"{secs}s"
    m, s = divmod(secs, 60)
    if m < 60:
        return f"{m}m {s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h {m:02d}m"


def _build(s: dict[str, Any], width: int, now: float) -> list[str]:
    inner = width
    brand = _c(f"{_BOLD}{_GOLD}", "◆ ShtetlFrames")
    tagline = _c(_CREAM, "Find old film that looks traditionally Orthodox")
    spin = _c(_TEAL, _spinner(now))
    pulse = _c(_AMBER, _pulse(now))

    top = _c(_GOLD, "╔" + "═" * (inner + 2) + "╗")
    bot = _c(_GOLD, "╚" + "═" * (inner + 2) + "╝")
    div = _c(_MUTED, "╟" + "─" * (inner + 2) + "╢")

    def row(text: str = "", *, raw: bool = False) -> str:
        # text may include ANSI; pad by visible width
        body = text if raw else text
        padded = _pad_visible(body, inner)
        border = _c(_GOLD, "║")
        return f"{border} {padded} {border}"

    lines = [
        top,
        row(f"{brand}  {pulse}  {_c(_DIM + _SLATE, 'live')}"),
        row(tagline),
        div,
        row(
            f"{_c(_MUTED, 'Browser')}  {_c(_SKY, str(s.get('url') or ''))}"
            f"    {_c(_MUTED, 'Review')}  {_c(_SKY, str(s.get('review') or ''))}"
        ),
        div,
    ]

    mode = s.get("mode") or "idle"
    headline = str(s.get("headline") or "")
    if mode in ("scrape", "setup", "discover", "both"):
        lines.append(row(f"{spin}  {_c(f'{_BOLD}{_WHITE}', headline)}"))
    elif mode == "done":
        lines.append(row(f"{_c(_GREEN, '✓')}  {_c(f'{_BOLD}{_GREEN}', headline)}"))
    elif mode == "error":
        lines.append(row(f"{_c(_ROSE, '✕')}  {_c(f'{_BOLD}{_ROSE}', headline)}"))
    else:
        lines.append(row(f"{_c(_AMBER, '○')}  {_c(f'{_BOLD}{_CREAM}', headline)}"))

    if s.get("sub") and mode not in ("both", "scrape"):
        lines.append(row(_c(_SLATE, str(s["sub"]))))

    def _append_discover_block() -> None:
        dh = str(s.get("discover_headline") or "Finding videos…")
        ds = str(s.get("discover_sub") or s.get("sub") or "")
        lines.append(row(""))
        lines.append(row(f"{_c(_SKY, '①')}  {_c(_BOLD + _CREAM, dh)}"))
        if ds:
            lines.append(row(f"    {_c(_SLATE, ds)}"))
        w = min(36, max(22, inner - 20))
        pos = int(now * 8) % max(1, w)
        shimmer = ["░"] * w
        for j in range(4):
            shimmer[(pos + j) % w] = "▓"
        lines.append(row(f"    {_c(_SKY, ''.join(shimmer))}  {_c(_MUTED, 'searching')}"))

    def _append_scrape_block(*, compact: bool = False) -> None:
        done = int(s.get("done") or 0)
        total = max(int(s.get("total") or 0), 1)
        hits = int(s.get("hits") or 0)
        errors = int(s.get("errors") or 0)
        pct = _pct(done, total)
        frac = done / total
        bar_w = min(36, max(22, inner - 36))
        bar = _block_bar(frac, bar_w, fill_color=_GOLD)
        lines.append(row(""))
        if compact:
            lines.append(
                row(f"{_c(_AMBER, '②')}  {_c(_BOLD + _CREAM, 'Scraping Pathé / queue…')}")
            )
            if s.get("sub"):
                lines.append(row(f"    {_c(_SLATE, str(s['sub']))}"))
        lines.append(
            row(
                f"{bar}  {_c(f'{_BOLD}{_GOLD}', f'{pct:5.1f}%')}  "
                f"{_c(_MUTED, f'{done:,}/{total:,}')}"
            )
        )
        lines.append(row(""))
        chips = "   ".join(
            [
                _stat_chip("checked", f"{done:,}", _CREAM),
                _stat_chip("clips", f"{hits:,}", _GREEN),
                _stat_chip("failed", f"{errors:,}", _ROSE if errors else _MUTED),
                _stat_chip("active", f"{len(s.get('live') or []):,}", _TEAL),
            ]
        )
        lines.append(row(chips))
        started = float(s.get("started_at") or now)
        lines.append(row(_c(_MUTED, f"session  {_elapsed_str(started)}")))

        live = s.get("live") or []
        if live:
            lines.append(row(""))
            lines.append(
                row(f"{_c(_BOLD + _CREAM, 'Right now')} {_c(_MUTED, '— cloud workers')}")
            )
            for item in live:
                title = _clip(item.get("title") or "video", max(28, inner - 6))
                phase = (item.get("phase") or "").strip().lower()
                detail = _friendly_detail(item.get("detail") or "")
                badge = _phase_badge(phase)
                mini = _mini_bar_from_detail(item.get("detail") or "", 10)
                lines.append(row(f"  {_c(_WHITE, '▸')} {_c(_BOLD + _CREAM, title)}"))
                status_bits = [badge]
                if mini:
                    status_bits.append(mini)
                if detail:
                    status_bits.append(_c(_SLATE, detail))
                lines.append(row("    " + "  ".join(status_bits)))

    if mode == "both":
        _append_discover_block()
        _append_scrape_block(compact=True)
    elif mode == "scrape":
        _append_scrape_block(compact=False)
    elif mode == "done":
        lines.append(row(""))
        bar = _block_bar(1.0, min(36, inner - 20), fill_color=_GREEN)
        lines.append(row(f"{bar}  {_c(f'{_BOLD}{_GREEN}', '100%')}"))
        lines.append(row(""))
        lines.append(
            row(
                "   ".join(
                    [
                        _stat_chip("videos", f"{int(s.get('done') or 0):,}", _CREAM),
                        _stat_chip("clips", f"{int(s.get('hits') or 0):,}", _GREEN),
                        _stat_chip("failed", f"{int(s.get('errors') or 0):,}", _ROSE),
                    ]
                )
            )
        )
        if int(s.get("hits") or 0):
            lines.append(row(""))
            lines.append(row(_c(_TEAL, "→ Open Review in the browser to browse the clips")))
    elif mode == "setup":
        lines.append(row(""))
        # Indeterminate shimmer bar
        w = min(36, max(22, inner - 20))
        pos = int(now * 6) % max(1, w)
        shimmer = ["░"] * w
        for j in range(5):
            k = (pos + j) % w
            shimmer[k] = "█"
        bar = _c(_TEAL, "".join(shimmer))
        lines.append(row(f"{bar}  {_c(_MUTED, 'warming up')}"))
        lines.append(row(_c(_MUTED, "This only happens once in a while.")))
    elif mode == "discover":
        lines.append(row(""))
        w = min(36, max(22, inner - 20))
        pos = int(now * 8) % max(1, w)
        shimmer = ["░"] * w
        for j in range(4):
            shimmer[(pos + j) % w] = "▓"
        lines.append(row(f"{_c(_SKY, ''.join(shimmer))}  {_c(_MUTED, 'searching')}"))
    elif mode == "idle":
        lines.append(row(""))
        lines.append(row(_c(_MUTED, "Tip: leave this window open while you work in the browser.")))
        lines.append(
            row(_c(_MUTED, "YouTube or British Pathé progress appears here automatically."))
        )
        pathe_url = (str(s.get("url") or "http://127.0.0.1:8787").rstrip("/") + "/pathe")
        lines.append(row(f"{_c(_MUTED, 'British Pathé')}  {_c(_SKY, pathe_url)}"))
    elif mode == "error":
        lines.append(row(""))
        lines.append(row(_c(_ROSE, "You can try again from the browser.")))

    note = str(s.get("note") or "")
    note_until = float(s.get("note_until") or 0)
    if note and (note_until <= 0 or now <= note_until):
        lines.append(row(""))
        lines.append(row(f"{_c(_AMBER, 'ⓘ')}  {_c(_CREAM, note)}"))

    lines.append(row(""))
    lines.append(row(_c(_MUTED, "Press Ctrl+C here to stop the app.")))
    lines.append(bot)
    return lines
