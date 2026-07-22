"""Build & push runpod_worker image from the UI (local Docker required)."""

from __future__ import annotations

import shutil
import subprocess
import threading
import time
from typing import Any

from config import ROOT, load_env
import config as app_config

WORKER_DIR = ROOT / "runpod_worker"

_lock = threading.Lock()
_job: dict[str, Any] = {
    "status": "idle",  # idle|running|done|error
    "phase": "",
    "message": "",
    "log": "",
    "image": "",
    "started_at": 0.0,
    "finished_at": 0.0,
}


def build_status() -> dict[str, Any]:
    with _lock:
        return dict(_job)


def _append(line: str) -> None:
    with _lock:
        _job["log"] = (_job["log"] + line + "\n")[-12000:]
        _job["message"] = line[:300]


def _set(**kwargs: Any) -> None:
    with _lock:
        _job.update(kwargs)


def _run(cmd: list[str], *, timeout: int = 3600) -> int:
    _append("$ " + " ".join(cmd))
    proc = subprocess.Popen(
        cmd,
        cwd=str(WORKER_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    assert proc.stdout is not None
    t0 = time.time()
    while True:
        if time.time() - t0 > timeout:
            proc.kill()
            _append("TIMEOUT")
            return 124
        line = proc.stdout.readline()
        if not line and proc.poll() is not None:
            break
        if line:
            _append(line.rstrip("\n"))
    return int(proc.wait() or 0)


def resolve_image_tag(explicit: str = "") -> str:
    load_env()
    import os

    tag = (explicit or app_config.RUNPOD_DOCKER_IMAGE or "").strip()
    if tag:
        return tag
    user = (os.environ.get("DOCKER_HUB_USER") or "").strip()
    if user:
        return f"{user}/shtetlframes-runpod:latest"
    raise RuntimeError(
        "Set Docker Hub user (or full Docker image) in Settings, then Build & push"
    )


def start_build_and_push(*, image: str = "", push: bool = True) -> dict[str, Any]:
    """Kick off docker build (+ optional push) in a background thread."""
    with _lock:
        if _job["status"] == "running":
            return {"ok": False, "error": "busy", "job": dict(_job)}
        _job.update(
            {
                "status": "running",
                "phase": "starting",
                "message": "Starting Docker build…",
                "log": "",
                "image": "",
                "started_at": time.time(),
                "finished_at": 0.0,
            }
        )

    t = threading.Thread(target=_build_job, args=(image, push), daemon=True)
    t.start()
    return {"ok": True, "job": build_status()}


def _build_job(image: str, push: bool) -> None:
    try:
        if not shutil.which("docker"):
            raise RuntimeError("Docker not found on PATH — install Docker Desktop and try again")
        if not WORKER_DIR.is_dir():
            raise RuntimeError(f"Missing worker folder: {WORKER_DIR}")

        tag = resolve_image_tag(image)
        _set(image=tag, phase="build", message=f"Building {tag}…")

        code = _run(["docker", "build", "-t", tag, "."], timeout=7200)
        if code != 0:
            raise RuntimeError(f"docker build failed (exit {code})")

        if push:
            load_env()
            import os

            user = (os.environ.get("DOCKER_HUB_USER") or "").strip()
            token = (os.environ.get("DOCKER_HUB_TOKEN") or "").strip()
            if user and token:
                _set(phase="login", message="Docker Hub login…")
                login = subprocess.run(
                    ["docker", "login", "-u", user, "--password-stdin"],
                    input=token + "\n",
                    text=True,
                    capture_output=True,
                    timeout=120,
                )
                if login.returncode != 0:
                    _append((login.stderr or login.stdout or "login failed")[-500:])
                    raise RuntimeError("docker login failed — check Docker Hub user/token")
            _set(phase="push", message=f"Pushing {tag}…")
            code = _run(["docker", "push", tag], timeout=7200)
            if code != 0:
                raise RuntimeError(
                    f"docker push failed (exit {code}). Log in with `docker login` or set Docker Hub token in Settings."
                )

            # Persist image tag into settings
            try:
                from settings_store import set_settings

                set_settings({"RUNPOD_DOCKER_IMAGE": tag, "SCAN_BACKEND": "runpod"})
                load_env()
            except Exception as e:
                _append(f"settings_update_warning: {e}")

        _set(
            status="done",
            phase="done",
            message=f"Ready — image {tag}" + (" (pushed)" if push else " (local only)"),
            finished_at=time.time(),
        )
    except Exception as e:
        _set(
            status="error",
            phase="error",
            message=str(e)[:500],
            finished_at=time.time(),
        )
        _append(f"ERROR: {e}")
