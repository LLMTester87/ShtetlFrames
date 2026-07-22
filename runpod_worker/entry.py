"""HTTP entrypoint for RunPod GPU Pods (auto-provisioned by ShtetlFrames)."""

from __future__ import annotations

import tempfile
import threading
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, Query, UploadFile
from fastapi.responses import JSONResponse

app = FastAPI(title="ShtetlFrames Pod Worker")
_ready = False
_warm_err = ""
_scan_threads_lock = threading.Lock()
_scan_threads: dict[str, threading.Thread] = {}

# Adaptive per-pod concurrency: try high, ratchet down on overload / hard failures.
_MAX_INFLIGHT_SCANS = 3
_HARD_PATHE_INFLIGHT_CAP = 6  # absolute ceiling (Settings PATHE_STACK_MAX may raise toward this)
_MAX_PATHE_INFLIGHT = 3
_yt_inflight_limit = _MAX_INFLIGHT_SCANS
_pathe_inflight_limit = _MAX_PATHE_INFLIGHT
_yt_ok_streak = 0
_pathe_ok_streak = 0
_SCALE_UP_AFTER_OK = 2  # successes at current limit before trying +1 again

_OVERLOAD_ERR_MARKERS = (
    "pod_saturated",
    "http_503",
    "http_524",
    "http_502",
    "gateway time-out",
    "gateway timeout",
    "cuda out of memory",
    "out of memory",
    "oom",
    "worker_died",
    "connection reset",
    "connection aborted",
    "broken pipe",
)


def _is_pathe_payload(inp: dict) -> bool:
    return str(inp.get("source") or "").lower() == "britishpathe"


def _current_inflight_limit(*, is_pathe: bool) -> int:
    # May be called while holding _scan_threads_lock — do not re-acquire.
    if is_pathe:
        return max(1, min(_pathe_inflight_limit, _MAX_PATHE_INFLIGHT, _HARD_PATHE_INFLIGHT_CAP))
    return max(1, min(_yt_inflight_limit, _MAX_INFLIGHT_SCANS))


def _apply_pathe_inflight_hint(inp: dict) -> None:
    """Raise Pathé inflight ceiling when the client Settings value is higher."""
    global _MAX_PATHE_INFLIGHT, _pathe_inflight_limit
    raw = inp.get("pathe_max_inflight")
    if raw is None:
        raw = inp.get("max_pathe_inflight")
    if raw is None:
        return
    try:
        want = max(1, min(_HARD_PATHE_INFLIGHT_CAP, int(raw)))
    except (TypeError, ValueError):
        return
    with _scan_threads_lock:
        if want > _MAX_PATHE_INFLIGHT:
            _MAX_PATHE_INFLIGHT = want
            print(f"[shtetl] pathe inflight ceiling → {_MAX_PATHE_INFLIGHT}", flush=True)
        if want > _pathe_inflight_limit:
            _pathe_inflight_limit = want
            print(f"[shtetl] pathe inflight limit → {_pathe_inflight_limit}", flush=True)


def _note_job_outcome(*, is_pathe: bool, ok: bool, err: str = "") -> None:
    """Scale concurrency: down hard on overload/OOM, up slowly after a streak of OK."""
    global _yt_inflight_limit, _pathe_inflight_limit, _yt_ok_streak, _pathe_ok_streak
    err_l = (err or "").lower()
    overload = (not ok) and any(m in err_l for m in _OVERLOAD_ERR_MARKERS)
    with _scan_threads_lock:
        if is_pathe:
            if ok:
                _pathe_ok_streak += 1
                if (
                    _pathe_ok_streak >= _SCALE_UP_AFTER_OK
                    and _pathe_inflight_limit < _MAX_PATHE_INFLIGHT
                ):
                    _pathe_inflight_limit += 1
                    _pathe_ok_streak = 0
                    print(
                        f"[shtetl] pathe inflight scale-up → {_pathe_inflight_limit}",
                        flush=True,
                    )
            else:
                _pathe_ok_streak = 0
                if overload or _pathe_inflight_limit > 1:
                    # Ease down by 1 (overload and soft fails) — avoid cliff to 1.
                    new_lim = max(1, _pathe_inflight_limit - 1)
                    if new_lim != _pathe_inflight_limit:
                        _pathe_inflight_limit = new_lim
                        print(
                            f"[shtetl] pathe inflight scale-down → {_pathe_inflight_limit}"
                            f" ({err_l[:80]})",
                            flush=True,
                        )
        else:
            if ok:
                _yt_ok_streak += 1
                if (
                    _yt_ok_streak >= _SCALE_UP_AFTER_OK
                    and _yt_inflight_limit < _MAX_INFLIGHT_SCANS
                ):
                    _yt_inflight_limit += 1
                    _yt_ok_streak = 0
                    print(
                        f"[shtetl] yt inflight scale-up → {_yt_inflight_limit}",
                        flush=True,
                    )
            else:
                _yt_ok_streak = 0
                if overload or _yt_inflight_limit > 1:
                    new_lim = 1 if overload else max(1, _yt_inflight_limit - 1)
                    if new_lim != _yt_inflight_limit:
                        _yt_inflight_limit = new_lim
                        print(
                            f"[shtetl] yt inflight scale-down → {_yt_inflight_limit}"
                            f" ({err_l[:80]})",
                            flush=True,
                        )


def _warm_once() -> None:
    import numpy
    import torch

    # Fail fast if torch/numpy ABI is broken (avoids "Numpy is not available" mid-scan).
    torch.from_numpy(numpy.zeros(1, dtype=numpy.float32))
    from handler import _models, reset_models

    reset_models()
    _models()
    # Ollama on this GPU — background pull so YOLO/CLIP models_ready is not blocked.
    try:
        import os

        os.environ.setdefault("SHTETL_POD", "1")
        os.environ.setdefault("OPEN_VLM_BASE_URL", "http://127.0.0.1:11434/v1")
        os.environ.setdefault("OPEN_VLM_MODEL", "qwen2.5vl:3b")
        os.environ.setdefault("VERIFY_BACKEND", "openai")
        # Ollama warm skipped while VERIFY_BACKEND=openai (re-enable with ollama_then_openai).
        if (os.environ.get("VERIFY_BACKEND") or "").strip().lower() in (
            "ollama_then_openai",
            "open_vlm",
            "vlm",
            "ollama",
        ):
            from ollama_pod import ollama_status, start_background_pull

            start_background_pull()
            print(f"[shtetl] ollama_warm_bg {ollama_status()}", flush=True)
        else:
            print("[shtetl] ollama_warm skipped (VERIFY_BACKEND=openai)", flush=True)
    except Exception as ollama_err:
        print(f"[shtetl] ollama_warm_skip: {ollama_err}", flush=True)
    print(
        f"[shtetl] warm ok cuda={torch.cuda.is_available()} numpy={numpy.__version__}",
        flush=True,
    )


def _inflight_count() -> int:
    with _scan_threads_lock:
        return sum(1 for t in _scan_threads.values() if t is not None and t.is_alive())


@app.on_event("startup")
def _warm() -> None:
    global _ready, _warm_err
    last = ""
    for attempt in range(1, 4):
        try:
            _warm_once()
            _ready = True
            _warm_err = ""
            try:
                import worker_sync

                worker_sync.set_inflight_checker(_inflight_count)
                worker_sync.start_background_sync()
            except Exception as sync_err:
                print(f"[shtetl] github sync not started: {sync_err}", flush=True)
            return
        except Exception as e:
            last = str(e)[:500]
            print(f"[shtetl] warm_failed attempt={attempt}: {e}", flush=True)
            time.sleep(2.0 * attempt)
    _warm_err = last
    _ready = False


@app.get("/health")
def health() -> dict:
    try:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        device = "unknown"
    try:
        from handler import get_progress

        progress = get_progress()
    except Exception:
        progress = None
    with _scan_threads_lock:
        alive = sum(1 for t in _scan_threads.values() if t is not None and t.is_alive())
        yt_lim = _yt_inflight_limit
        pathe_lim = _pathe_inflight_limit
    sync = {}
    try:
        import worker_sync

        sync = worker_sync.sync_status()
    except Exception:
        pass
    ollama = {}
    try:
        from ollama_pod import ollama_status

        ollama = ollama_status()
    except Exception:
        ollama = {"ready": False, "model_ready": False}
    return {
        # ok tracks models_ready so wait_healthy does not accept a cold/broken warm.
        "ok": bool(_ready),
        "service": "shtetlframes",
        "device": device,
        "models_ready": _ready,
        "warm_error": _warm_err or None,
        "progress": progress,
        "inflight": alive,
        "inflight_limit_yt": yt_lim,
        "inflight_limit_pathe": pathe_lim,
        "github_sync": sync,
        "ollama": ollama,
    }


@app.post("/reload")
def reload_from_github() -> dict:
    """Pull latest worker files from GitHub main and hot-reload (no pod recreate)."""
    try:
        import worker_sync

        worker_sync.set_inflight_checker(_inflight_count)
        return worker_sync.sync_from_github(force=False)
    except Exception as e:
        return {"ok": False, "error": str(e)[:300]}


@app.post("/sync_push")
def sync_push(payload: dict) -> dict:
    """Accept file bodies from the PC (bypass stale GitHub CDN) and hot-reload."""
    import worker_sync
    from pathlib import Path

    inp = dict(payload) if isinstance(payload, dict) else {}
    files = inp.get("files") if isinstance(inp.get("files"), dict) else {}
    if not files:
        return {"ok": False, "error": "files_required"}
    root = Path(__file__).resolve().parent
    changed: list[str] = []
    for rel, content in files.items():
        name = str(rel).replace("\\", "/").lstrip("/")
        if ".." in name or name.startswith("/"):
            continue
        if not (
            name.endswith(".py")
            and (
                name
                in {
                    "entry.py",
                    "handler.py",
                    "worker_sync.py",
                    "ollama_pod.py",
                    "openai_verify.py",
                    "label_feedback.py",
                }
                or name.startswith("shtetl_core/")
            )
        ):
            continue
        raw = content if isinstance(content, (bytes, bytearray)) else str(content).encode("utf-8")
        dest = root / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(raw)
        changed.append(name)
    if not changed:
        return {"ok": False, "error": "no_valid_files"}
    worker_sync.set_inflight_checker(_inflight_count)
    reloaded = worker_sync._reload_modules(changed)
    pending = "entry.py" in changed
    return {
        "ok": True,
        "changed": changed,
        "reloaded": reloaded,
        "pending_soft_recycle": pending
        or bool((worker_sync.sync_status() or {}).get("pending_soft_recycle")),
    }


@app.get("/progress")
def progress(queue_id: str | None = Query(default=None)) -> dict:
    """Live download/scan/upload status (optionally for one queue item)."""
    from handler import get_progress

    qid: str | int | None = queue_id
    if queue_id is not None and str(queue_id).isdigit():
        qid = int(queue_id)
    return {"ok": True, **get_progress(qid)}


@app.post("/scan")
def scan(payload: dict) -> JSONResponse:
    """Accept a scan job and return immediately (avoids RunPod proxy ~100s 524 timeouts).

    Client polls GET /result?queue_id=… until pending=false.
    """
    from handler import (
        clear_job_result,
        clear_job_stills,
        clear_progress,
        process_job,
        set_progress,
        store_job_result,
    )

    if not _ready:
        return JSONResponse(
            {
                "ok": False,
                "accepted": False,
                "error": f"models_not_ready:{_warm_err or 'warming'}",
            },
            status_code=503,
        )

    inp = dict(payload) if isinstance(payload, dict) else {}
    qid = inp.get("queue_id")
    if qid is None or qid == "":
        qid = f"anon-{uuid.uuid4().hex[:12]}"
        inp["queue_id"] = qid
    key = str(qid)

    if _is_pathe_payload(inp):
        _apply_pathe_inflight_hint(inp)

    # Reject duplicate in-flight job for same queue id; cap total concurrency.
    with _scan_threads_lock:
        t_old = _scan_threads.get(key)
        if t_old is not None and t_old.is_alive():
            return JSONResponse(
                {"ok": True, "accepted": True, "async": True, "queue_id": qid, "duplicate": True},
                status_code=202,
            )
        alive = sum(1 for t in _scan_threads.values() if t is not None and t.is_alive())
        is_pathe = _is_pathe_payload(inp)
        # Adaptive: Pathé up to Settings ceiling (≤6), YouTube up to 3.
        max_inflight = _current_inflight_limit(is_pathe=is_pathe)
        if alive >= max_inflight:
            return JSONResponse(
                {
                    "ok": False,
                    "accepted": False,
                    "error": f"pod_saturated:{alive}/{max_inflight}",
                },
                status_code=503,
            )

    is_pathe = _is_pathe_payload(inp)
    clear_job_result(qid)
    # Do NOT clear on-disk stills at scan start — the PC may still be hydrating
    # the previous job via GET /still. New writes overwrite the same indices.
    clear_progress(qid)
    set_progress(
        "queued",
        "accepted",
        pct=0,
        detail=str(inp.get("title") or "")[:120],
        queue_id=qid,
        title=str(inp.get("title") or ""),
        url=str(inp.get("url") or ""),
    )

    def _run() -> None:
        try:
            result = process_job(inp)
        except Exception as e:
            result = {"ok": False, "error": str(e)[:1000], "segments": []}
        if not isinstance(result, dict):
            result = {"ok": False, "error": "bad_result", "segments": []}
        result = dict(result)
        result["queue_id"] = qid
        ok = bool(result.get("ok", True)) and not result.get("error")
        _note_job_outcome(
            is_pathe=is_pathe,
            ok=ok,
            err=str(result.get("error") or ""),
        )
        store_job_result(qid, result)
        set_progress(
            "done" if ok else "error",
            "job complete" if ok else "job failed",
            pct=100 if ok else None,
            detail=str(result.get("error") or "")[:200],
            queue_id=qid,
        )
        with _scan_threads_lock:
            _scan_threads.pop(key, None)

        # Give the PC time to GET /still before freeing disk.
        def _deferred_clear() -> None:
            time.sleep(180.0)
            try:
                clear_job_stills(qid)
            except Exception:
                pass

        threading.Thread(
            target=_deferred_clear, daemon=True, name=f"still-clear-{key}"
        ).start()

    t = threading.Thread(target=_run, name=f"scan-{key}", daemon=True)
    with _scan_threads_lock:
        _scan_threads[key] = t
    t.start()
    return JSONResponse(
        {"ok": True, "accepted": True, "async": True, "queue_id": qid},
        status_code=202,
    )


@app.get("/result")
def result(queue_id: str = Query(...)) -> dict:
    """Final job payload once finished; pending=true while still running."""
    from handler import clear_progress, get_job_result, get_progress

    qid: str | int = queue_id
    if str(queue_id).isdigit():
        qid = int(queue_id)
    key = str(qid)
    done = get_job_result(qid, consume=False)
    if done is None:
        with _scan_threads_lock:
            t = _scan_threads.get(key)
            alive = t is not None and t.is_alive()
        prog = get_progress(qid)
        phase = str(prog.get("phase") or "")
        # Thread vanished without storing a result (OOM / kill) — fail fast for client retry.
        if not alive and phase not in ("", "idle"):
            clear_progress(qid)
            return {
                "ok": False,
                "pending": False,
                "queue_id": qid,
                "error": "worker_died",
                "segments": [],
            }
        return {"ok": True, "pending": True, "queue_id": qid, **prog}
    # Consume so memory does not grow; clear live progress.
    done = get_job_result(qid, consume=True) or done
    clear_progress(qid)
    out = dict(done)
    out["pending"] = False
    return out


@app.get("/still")
def still(queue_id: str = Query(...), index: int = Query(1)):
    """JPEG Review still for one segment (backup when still_b64 is missing from /result)."""
    from fastapi.responses import FileResponse

    from handler import still_file_path

    try:
        idx = int(index)
    except (TypeError, ValueError):
        idx = 1
    path = still_file_path(queue_id, idx)
    if not path.is_file() or path.stat().st_size < 200:
        return JSONResponse(
            {"ok": False, "error": "still_not_found"},
            status_code=404,
        )
    return FileResponse(path, media_type="image/jpeg", filename=path.name)


@app.post("/verify_still")
def verify_still_http(payload: dict) -> dict:
    """Run vision verify on a JPEG (image_b64). backend=open_vlm|openai.

    Used from the PC to A/B Ollama-on-GPU vs OpenAI on identical crops.
    """
    import base64
    import os
    import tempfile

    inp = dict(payload) if isinstance(payload, dict) else {}
    b64 = (inp.get("image_b64") or "").strip()
    if not b64:
        return {"ok": False, "error": "image_b64_required"}
    backend = (inp.get("backend") or "open_vlm").strip().lower()
    if backend in ("ollama", "vlm", "qwen", "open-vlm"):
        backend = "open_vlm"
    if backend not in ("open_vlm", "openai"):
        return {"ok": False, "error": f"unsupported_backend:{backend}"}

    os.environ["SHTETL_POD"] = "1"
    os.environ["OPENAI_VERIFY"] = "1"
    os.environ["VERIFY_BACKEND"] = backend
    if backend == "open_vlm":
        os.environ["OPEN_VLM_BASE_URL"] = "http://127.0.0.1:11434/v1"
        model = (inp.get("open_vlm_model") or os.environ.get("OPEN_VLM_MODEL") or "").strip()
        if model:
            os.environ["OPEN_VLM_MODEL"] = model
        try:
            from ollama_pod import ensure_ollama, start_background_pull, wait_for_model

            start_background_pull()
            ensure_ollama(pull=False, use_cache=True)
            if not wait_for_model(timeout_sec=float(inp.get("ollama_wait_sec") or 240)):
                from ollama_pod import ollama_status

                return {
                    "ok": False,
                    "error": "ollama_model_not_ready",
                    "ollama": ollama_status(),
                }
        except Exception as e:
            return {"ok": False, "error": f"ollama_ensure:{e}"[:240]}
    else:
        key = (inp.get("openai_api_key") or os.environ.get("OPENAI_API_KEY") or "").strip()
        if key:
            os.environ["OPENAI_API_KEY"] = key
        oai_model = (inp.get("openai_model") or os.environ.get("OPENAI_MODEL") or "").strip()
        if oai_model:
            os.environ["OPENAI_MODEL"] = oai_model

    try:
        raw = base64.standard_b64decode(b64)
    except Exception as e:
        return {"ok": False, "error": f"b64_decode:{e}"[:160]}
    if len(raw) < 200:
        return {"ok": False, "error": "image_too_small"}

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp.write(raw)
            tmp_path = Path(tmp.name)
        # Fresh import path so sync_push'd openai_verify is used.
        import importlib

        import openai_verify as ov

        importlib.reload(ov)
        # Clear any prior hard-disable from other backends in this process.
        ov._disabled_reason = None  # type: ignore[attr-defined]
        verdict = ov.verify_still(image_path=tmp_path, timeout=float(inp.get("timeout") or 90))
        notes = ov.format_verdict_notes(verdict)
        return {
            "ok": True,
            "backend": backend,
            "model": (
                ov.open_vlm_model() if backend == "open_vlm" else ov.openai_model()
            ),
            "keep": bool(verdict.get("keep")),
            "looks_jewish": verdict.get("looks_jewish"),
            "head_covered": verdict.get("head_covered"),
            "confidence": verdict.get("confidence"),
            "skipped": verdict.get("skipped"),
            "reason": (verdict.get("reason") or "")[:300],
            "notes": notes[:400],
            "error": verdict.get("error"),
        }
    except Exception as e:
        return {"ok": False, "error": str(e)[:300]}
    finally:
        if tmp_path is not None:
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass


@app.post("/pathe_list")
def pathe_list(payload: dict) -> JSONResponse:
    """Fetch a British Pathé search/listing page via Scrapfly (no GPU / models).

    Used so catalog discover can run on a dedicated pod and not fight the PC's
    Scrapfly asset-resolve calls used by Pathé scrape.
    """
    import json
    import urllib.parse
    import urllib.request

    inp = dict(payload) if isinstance(payload, dict) else {}
    page_url = (inp.get("url") or "").strip()
    key = (inp.get("scrapfly_api_key") or inp.get("scrapfly_key") or "").strip()
    if not page_url:
        return JSONResponse({"ok": False, "error": "url_required"}, status_code=400)
    if not key:
        return JSONResponse(
            {"ok": False, "error": "scrapfly_api_key_required"}, status_code=400
        )
    try:
        wait = int(inp.get("rendering_wait") or 4000)
    except (TypeError, ValueError):
        wait = 4000
    wait = max(500, min(wait, 20000))
    auto_scroll = bool(inp.get("auto_scroll"))
    country = (inp.get("country") or "us").strip() or "us"
    params = {
        "key": key,
        "url": page_url,
        "asp": "true",
        "country": country,
        "render_js": "true",
        "rendering_wait": str(wait),
    }
    if auto_scroll:
        params["auto_scroll"] = "true"
    api = "https://api.scrapfly.io/scrape?" + urllib.parse.urlencode(params)
    try:
        req = urllib.request.Request(
            api,
            headers={"User-Agent": "ShtetlFrames-pod/1.0 (pathe_list)"},
        )
        with urllib.request.urlopen(req, timeout=180) as resp:
            data = json.loads(resp.read().decode("utf-8", "replace"))
    except Exception as e:
        return JSONResponse(
            {"ok": False, "error": f"scrapfly_http: {e}"[:300]},
            status_code=502,
        )
    result = data.get("result") or {}
    if not result.get("success"):
        reason = (
            result.get("error")
            or result.get("reason")
            or data.get("message")
            or "scrape_failed"
        )
        return JSONResponse(
            {"ok": False, "error": f"scrapfly_pathe_page: {reason}"[:300]},
            status_code=502,
        )
    status = int(result.get("status_code") or 0)
    html = result.get("content") or ""
    if status >= 400 or not html:
        return JSONResponse(
            {"ok": False, "error": f"scrapfly_pathe_page_http_{status}"},
            status_code=502,
        )
    return JSONResponse(
        {
            "ok": True,
            "url": page_url,
            "html": html,
            "status_code": status,
            "nbytes": len(html),
        }
    )


@app.post("/scan_file")
async def scan_file(
    video: UploadFile = File(...),
    title: str = Form("video"),
    queue_id: str = Form(""),
    source_url: str = Form(""),
    sample_fps: float = Form(0.5),
    score_threshold: float = Form(0.10),
) -> JSONResponse:
    """Scan an uploaded video file (legacy). Prefer /scan — downloads stay on the pod."""
    from handler import process_job, slugify

    suffix = Path(video.filename or "video.mp4").suffix or ".mp4"
    vid = slugify(title or "upload")
    dest = Path(tempfile.gettempdir()) / "shtetlframes" / "videos" / f"{vid}{suffix}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("wb") as out:
        while True:
            chunk = await video.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)
    qid: str | int | None = queue_id.strip() or None
    if qid is not None and str(qid).isdigit():
        qid = int(qid)
    result = process_job(
        {
            "url": source_url or f"file://{dest.name}",
            "title": title or vid,
            "queue_id": qid,
            "sample_fps": float(sample_fps or 0.5),
            "score_threshold": float(score_threshold or 0.10),
            "source_url": source_url or "",
            "skip_download": True,
            "video_path": str(dest),
        }
    )
    code = 200 if result.get("ok") else 500
    return JSONResponse(result, status_code=code)


@app.get("/cues_config")
def cues_config() -> dict:
    try:
        from shtetl_core import cues as c

        return {
            "ok": True,
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
        return {"ok": False, "error": str(e)[:200]}


if __name__ == "__main__":
    import uvicorn

    # Multiple threads so concurrent /scan downloads are not blocked by one GPU job.
    uvicorn.run("entry:app", host="0.0.0.0", port=8000, workers=1)
