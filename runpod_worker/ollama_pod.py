"""Install / start Ollama on the RunPod GPU for vision verify cascade."""

from __future__ import annotations

import os
import shutil
import subprocess
import threading
import time
from typing import Any

import requests

DEFAULT_MODEL = "qwen2.5vl:3b"
OLLAMA_HOST = "127.0.0.1:11434"
OLLAMA_BASE = f"http://{OLLAMA_HOST}"

_lock = threading.Lock()
_ensure_cache: dict[str, Any] | None = None
_pull_thread: threading.Thread | None = None
_pull_started = False


def ollama_model_name() -> str:
    return (os.environ.get("OPEN_VLM_MODEL") or DEFAULT_MODEL).strip() or DEFAULT_MODEL


def ollama_ready(*, timeout: float = 2.0) -> bool:
    try:
        r = requests.get(f"{OLLAMA_BASE}/api/tags", timeout=timeout)
        return r.status_code == 200
    except requests.RequestException:
        return False


def ollama_has_model(name: str | None = None) -> bool:
    name = name or ollama_model_name()
    return _have_model(name)


def ollama_status() -> dict[str, Any]:
    model = ollama_model_name()
    ready = ollama_ready()
    return {
        "ready": ready,
        "model": model,
        "model_ready": bool(ready and _have_model(model)),
        "pulling": bool(_pull_thread is not None and _pull_thread.is_alive()),
        "cached_ok": bool(_ensure_cache and _ensure_cache.get("ok")),
    }


def _have_model(name: str) -> bool:
    try:
        r = requests.get(f"{OLLAMA_BASE}/api/tags", timeout=10)
        if r.status_code != 200:
            return False
        models = (r.json() or {}).get("models") or []
        want = name.lower()
        for m in models:
            mid = str(m.get("name") or m.get("model") or "").lower()
            if mid == want or mid.startswith(want + ":") or want.startswith(mid):
                return True
            if mid.split(":")[0] == want.split(":")[0] and want in mid:
                return True
        return any(want in str(m.get("name") or "").lower() for m in models)
    except Exception:
        return False


def _install_ollama() -> bool:
    if shutil.which("ollama"):
        return True
    print("[shtetl] installing Ollama…", flush=True)
    try:
        subprocess.run(
            ["bash", "-lc", "curl -fsSL https://ollama.com/install.sh | sh"],
            check=False,
            timeout=600,
        )
    except Exception as e:
        print(f"[shtetl] ollama install failed: {e}", flush=True)
        return False
    return bool(shutil.which("ollama"))


def _start_serve() -> None:
    if ollama_ready():
        return
    env = os.environ.copy()
    env["OLLAMA_HOST"] = OLLAMA_HOST
    env.setdefault("OLLAMA_KEEP_ALIVE", "30m")
    print(f"[shtetl] starting ollama serve on {OLLAMA_HOST}…", flush=True)
    subprocess.Popen(
        ["ollama", "serve"],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    for _ in range(60):
        if ollama_ready():
            print("[shtetl] ollama serve ready", flush=True)
            return
        time.sleep(1.0)
    print("[shtetl] ollama serve did not become ready", flush=True)


def _pull_model(name: str) -> bool:
    if _have_model(name):
        print(f"[shtetl] ollama model present: {name}", flush=True)
        return True
    print(f"[shtetl] ollama pull {name} (GPU)…", flush=True)
    try:
        r = subprocess.run(
            ["ollama", "pull", name],
            capture_output=True,
            text=True,
            timeout=3600,
            env={**os.environ, "OLLAMA_HOST": OLLAMA_HOST},
        )
    except Exception as e:
        print(f"[shtetl] ollama pull error: {e}", flush=True)
        return False
    ok = r.returncode == 0 and _have_model(name)
    if ok:
        print(f"[shtetl] ollama pull ok: {name}", flush=True)
    else:
        err = (r.stderr or r.stdout or "")[:300]
        print(f"[shtetl] ollama pull failed: {err}", flush=True)
    return ok


def ensure_ollama(*, pull: bool = True, use_cache: bool = True) -> dict[str, Any]:
    """Make sure Ollama is installed, serving, and (optionally) has the VLM model.

    Cached per process when ``use_cache`` and a prior successful ensure exists.
    """
    global _ensure_cache
    os.environ.setdefault("SHTETL_POD", "1")
    os.environ.setdefault("OPEN_VLM_BASE_URL", f"{OLLAMA_BASE}/v1")
    model = ollama_model_name()

    if use_cache and _ensure_cache is not None:
        cached = dict(_ensure_cache)
        # Refresh model_ready cheaply — pull may have finished in background.
        if cached.get("ready") and ollama_ready():
            cached["model_ready"] = _have_model(model)
            cached["model"] = model
            cached["ok"] = bool(cached["ready"] and (cached["model_ready"] or not pull))
            if cached["ok"] or not pull:
                _ensure_cache = cached
                return cached
        elif cached.get("ok") and not pull:
            return cached

    out: dict[str, Any] = {
        "ok": False,
        "model": model,
        "host": OLLAMA_HOST,
        "installed": False,
        "ready": False,
        "model_ready": False,
    }
    if not _install_ollama():
        out["error"] = "install_failed"
        _ensure_cache = out
        return out
    out["installed"] = True
    _start_serve()
    out["ready"] = ollama_ready()
    if not out["ready"]:
        out["error"] = "serve_not_ready"
        _ensure_cache = out
        return out
    if pull:
        out["model_ready"] = _pull_model(model)
        if not out["model_ready"]:
            out["error"] = "pull_failed"
            _ensure_cache = out
            return out
    else:
        out["model_ready"] = _have_model(model)
    out["ok"] = bool(out["ready"] and (out["model_ready"] or not pull))
    _ensure_cache = out
    return out


def start_background_pull() -> None:
    """Install + serve + pull model without blocking YOLO/CLIP warm."""
    global _pull_thread, _pull_started, _ensure_cache
    with _lock:
        if _pull_started and _pull_thread is not None and _pull_thread.is_alive():
            return
        if ollama_ready() and ollama_has_model():
            _ensure_cache = {
                "ok": True,
                "model": ollama_model_name(),
                "host": OLLAMA_HOST,
                "installed": True,
                "ready": True,
                "model_ready": True,
            }
            _pull_started = True
            return
        _pull_started = True

        def _run() -> None:
            try:
                st = ensure_ollama(pull=True, use_cache=False)
                print(f"[shtetl] ollama_bg {st}", flush=True)
            except Exception as e:
                print(f"[shtetl] ollama_bg_err: {e}", flush=True)

        _pull_thread = threading.Thread(target=_run, daemon=True, name="ollama-pull")
        _pull_thread.start()


def wait_for_model(*, timeout_sec: float = 120.0) -> bool:
    """Block until the VLM model is present (for first verify after bg pull)."""
    if ollama_has_model():
        return True
    # Kick serve/install without a blocking pull if bg thread is already pulling.
    ensure_ollama(pull=False, use_cache=True)
    deadline = time.time() + max(5.0, float(timeout_sec))
    while time.time() < deadline:
        if ollama_has_model():
            return True
        # If no bg pull, do a foreground pull once.
        with _lock:
            bg_alive = _pull_thread is not None and _pull_thread.is_alive()
        if not bg_alive and not ollama_has_model():
            return bool(ensure_ollama(pull=True, use_cache=False).get("model_ready"))
        time.sleep(1.0)
    return ollama_has_model()
