"""Pull worker code from GitHub and hot-reload without recreating the RunPod.

Handler / shtetl_core / openai_verify reload in-process. If entry.py itself
changes, schedule a soft process recycle (same GPU pod, uvicorn re-exec) once
in-flight scans drain.
"""

from __future__ import annotations

import hashlib
import importlib
import os
import sys
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parent
# jsDelivr tracks main faster/more reliably than raw.githubusercontent.com CDN.
GITHUB_RAW = os.environ.get(
    "SHTETL_SYNC_BASE",
    "https://cdn.jsdelivr.net/gh/AIQAEngineer/ShtetlFrames@main",
).rstrip("/")
POLL_SEC = float(os.environ.get("SHTETL_SYNC_POLL_SEC") or "120")

_SYNC_FILES: tuple[tuple[str, str], ...] = (
    ("entry.py", f"{GITHUB_RAW}/runpod_worker/entry.py"),
    ("handler.py", f"{GITHUB_RAW}/runpod_worker/handler.py"),
    ("worker_sync.py", f"{GITHUB_RAW}/runpod_worker/worker_sync.py"),
    ("ollama_pod.py", f"{GITHUB_RAW}/runpod_worker/ollama_pod.py"),
    ("openai_verify.py", f"{GITHUB_RAW}/src/openai_verify.py"),
    ("label_feedback.py", f"{GITHUB_RAW}/src/label_feedback.py"),
    ("shtetl_core/__init__.py", f"{GITHUB_RAW}/src/shtetl_core/__init__.py"),
    ("shtetl_core/cues.py", f"{GITHUB_RAW}/src/shtetl_core/cues.py"),
    ("shtetl_core/scoring.py", f"{GITHUB_RAW}/src/shtetl_core/scoring.py"),
    ("shtetl_core/scan.py", f"{GITHUB_RAW}/src/shtetl_core/scan.py"),
    ("shtetl_core/segments.py", f"{GITHUB_RAW}/src/shtetl_core/segments.py"),
    ("shtetl_core/textutil.py", f"{GITHUB_RAW}/src/shtetl_core/textutil.py"),
    ("shtetl_core/upload.py", f"{GITHUB_RAW}/src/shtetl_core/upload.py"),
)

_lock = threading.Lock()
_state: dict[str, Any] = {
    "last_check": 0.0,
    "last_ok": 0.0,
    "last_changed": [],
    "last_error": None,
    "pending_soft_recycle": False,
    "poll_sec": POLL_SEC,
}
_inflight_fn: Callable[[], int] | None = None
_started = False


def set_inflight_checker(fn: Callable[[], int]) -> None:
    global _inflight_fn
    _inflight_fn = fn


def cues_snapshot() -> dict[str, Any]:
    """Live CLIP knobs currently loaded in this process."""
    try:
        import shtetl_core.cues as c

        return {
            "DEFAULT_SCORE_THRESHOLD": float(c.DEFAULT_SCORE_THRESHOLD),
            "MIN_POS_SCORE": float(c.MIN_POS_SCORE),
            "MIN_HEADCOVER_SCORE": float(c.MIN_HEADCOVER_SCORE),
            "MAX_NEG_TO_POS_RATIO": float(c.MAX_NEG_TO_POS_RATIO),
            "NEG_SCORE_WEIGHT": float(c.NEG_SCORE_WEIGHT),
            "TOP_K_NEGS": int(c.TOP_K_NEGS),
            "MAX_SEGMENTS_PER_VIDEO": int(getattr(c, "MAX_SEGMENTS_PER_VIDEO", -1)),
            "YOLO_CONF": float(c.YOLO_CONF),
        }
    except Exception as e:
        return {"error": str(e)[:200]}


def sync_status() -> dict[str, Any]:
    with _lock:
        out = dict(_state)
    out["cues"] = cues_snapshot()
    return out


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _fetch(url: str) -> bytes:
    # raw.githubusercontent.com often serves stale main for minutes; bust cache.
    sep = "&" if "?" in url else "?"
    bust = f"{url}{sep}t={int(time.time())}"
    req = urllib.request.Request(
        bust,
        headers={
            "User-Agent": "ShtetlFrames-pod-sync/1.0",
            "Cache-Control": "no-cache",
            "Pragma": "no-cache",
        },
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read()


def _reload_modules(changed: list[str]) -> list[str]:
    """Reload imported modules affected by downloaded files."""
    reloaded: list[str] = []
    # Order: leaves first-ish, then packages, then handler.
    order = [
        "shtetl_core.cues",
        "shtetl_core.textutil",
        "shtetl_core.upload",
        "shtetl_core.scoring",
        "shtetl_core.scan",
        "shtetl_core.segments",
        "shtetl_core",
        "openai_verify",
        "label_feedback",
        "ollama_pod",
        "handler",
        "worker_sync",
    ]
    want = set()
    for rel in changed:
        if rel.startswith("shtetl_core/"):
            mod = "shtetl_core." + rel.split("/", 1)[1].replace(".py", "")
            if mod.endswith(".__init__"):
                mod = "shtetl_core"
            want.add(mod)
            want.add("shtetl_core")
        elif rel == "handler.py":
            want.add("handler")
        elif rel == "openai_verify.py":
            want.add("openai_verify")
        elif rel == "label_feedback.py":
            want.add("label_feedback")
            want.add("openai_verify")
        elif rel == "ollama_pod.py":
            want.add("ollama_pod")
            want.add("handler")
        elif rel == "worker_sync.py":
            want.add("worker_sync")

    for name in order:
        if name not in want:
            continue
        mod = sys.modules.get(name)
        if mod is None:
            continue
        try:
            importlib.reload(mod)
            reloaded.append(name)
        except Exception as e:
            print(f"[shtetl] reload failed {name}: {e}", flush=True)

    if "handler" in reloaded or any(x.startswith("shtetl_core") for x in reloaded):
        try:
            import handler as h

            if hasattr(h, "reset_models"):
                h.reset_models()
                print("[shtetl] models cache cleared after sync", flush=True)
        except Exception as e:
            print(f"[shtetl] reset_models after sync: {e}", flush=True)

    # entry.py owns FastAPI + inflight globals — must soft-recycle the process.
    # sync_push used to write entry.py without scheduling this, leaving stale limits.
    if "entry.py" in changed:
        with _lock:
            _state["pending_soft_recycle"] = True
        print("[shtetl] entry.py changed — soft-recycle armed", flush=True)
        _try_soft_recycle()
    return reloaded


def _soft_recycle() -> None:
    """Replace this process with a fresh uvicorn (same pod / disk / GPU)."""
    print("[shtetl] soft-recycle uvicorn after entry.py update…", flush=True)
    py = sys.executable
    os.chdir(str(ROOT))
    os.environ["PYTHONPATH"] = str(ROOT)
    os.execv(py, [py, "-m", "uvicorn", "entry:app", "--host", "0.0.0.0", "--port", "8000"])


def sync_from_github(*, force: bool = False) -> dict[str, Any]:
    """Download changed files from GitHub main; hot-reload modules."""
    with _lock:
        _state["last_check"] = time.time()
    changed: list[str] = []
    try:
        for rel, url in _SYNC_FILES:
            dest = ROOT / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            remote = _fetch(url)
            local = dest.read_bytes() if dest.is_file() else b""
            if _sha(remote) == _sha(local) and not force:
                continue
            dest.write_bytes(remote)
            changed.append(rel)
            print(f"[shtetl] synced {rel} ({len(remote)} bytes)", flush=True)

        entry_changed = "entry.py" in changed
        code_changed = [c for c in changed if c != "entry.py"]
        reloaded = _reload_modules(code_changed) if code_changed else []

        pending_recycle = False
        if entry_changed:
            # entry.py owns FastAPI routes — needs process recycle when idle.
            pending_recycle = True
            with _lock:
                _state["pending_soft_recycle"] = True

        with _lock:
            _state["last_ok"] = time.time()
            _state["last_changed"] = list(changed)
            _state["last_error"] = None
            _state["last_reloaded"] = reloaded

        if pending_recycle:
            _try_soft_recycle()

        return {
            "ok": True,
            "changed": changed,
            "reloaded": reloaded,
            "pending_soft_recycle": pending_recycle,
        }
    except Exception as e:
        with _lock:
            _state["last_error"] = str(e)[:300]
        return {"ok": False, "error": str(e)[:300], "changed": changed}


def _try_soft_recycle() -> None:
    with _lock:
        if not _state.get("pending_soft_recycle"):
            return
    alive = 0
    if _inflight_fn:
        try:
            alive = int(_inflight_fn())
        except Exception:
            alive = 0
    if alive > 0:
        print(
            f"[shtetl] soft-recycle waiting — {alive} scan(s) in flight",
            flush=True,
        )
        return
    with _lock:
        _state["pending_soft_recycle"] = False
    _soft_recycle()


def _poll_loop() -> None:
    # Initial delay so warm-up finishes first.
    time.sleep(45.0)
    while True:
        try:
            sync_from_github(force=False)
            _try_soft_recycle()
        except Exception as e:
            print(f"[shtetl] sync poll error: {e}", flush=True)
        time.sleep(max(30.0, POLL_SEC))


def start_background_sync() -> None:
    global _started
    if _started:
        return
    _started = True
    threading.Thread(target=_poll_loop, name="shtetl-github-sync", daemon=True).start()
    print(f"[shtetl] GitHub sync poller started (every {POLL_SEC:.0f}s)", flush=True)
