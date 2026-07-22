"""A/B vision verify: RunPod Ollama (qwen2.5vl) vs local OpenAI on the same stills."""

from __future__ import annotations

import base64
import json
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "debug-30525a.log"
sys.path.insert(0, str(ROOT / "src"))

from config import load_env  # noqa: E402
from settings_store import apply_settings_to_environ  # noqa: E402


def _log(message: str, data: dict, *, hypothesis_id: str = "O") -> None:
    # #region agent log
    payload = {
        "sessionId": "30525a",
        "runId": "ollama-vs-openai",
        "hypothesisId": hypothesis_id,
        "location": "_compare_ollama_openai_pod.py",
        "message": message,
        "data": data,
        "timestamp": int(time.time() * 1000),
    }
    with LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    # #endregion


def _push_worker(base: str) -> None:
    from runpod_client import _local_worker_files_for_push

    files = _local_worker_files_for_push()
    r = requests.post(
        f"{base.rstrip('/')}/sync_push",
        json={"files": files},
        timeout=120,
    )
    print(f"sync_push {r.status_code} {r.text[:200]}", flush=True)
    r.raise_for_status()


def _wait_health(base: str, *, timeout_sec: float = 300) -> dict:
    t0 = time.time()
    last: dict = {}
    while time.time() - t0 < timeout_sec:
        try:
            r = requests.get(f"{base.rstrip('/')}/health", timeout=15)
            last = r.json() if r.ok else {"ok": False, "http": r.status_code}
            print(
                f"health models_ready={last.get('models_ready')} "
                f"ollama={last.get('ollama')}",
                flush=True,
            )
            if last.get("models_ready"):
                return last
        except Exception as e:
            last = {"ok": False, "error": str(e)[:160]}
            print(f"health err: {e}", flush=True)
        time.sleep(5)
    return last


def _b64(path: Path) -> str:
    return base64.standard_b64encode(path.read_bytes()).decode("ascii")


def _openai_local(path: Path) -> dict:
    import os

    # Force OpenAI path regardless of Settings.
    os.environ["VERIFY_BACKEND"] = "openai"
    os.environ["OPENAI_VERIFY"] = "1"
    from openai_verify import (
        _disabled_reason,
        format_verdict_notes,
        verify_still,
    )
    import openai_verify as ov

    ov._disabled_reason = None
    v = verify_still(image_path=path, timeout=60)
    return {
        "backend": "openai",
        "keep": bool(v.get("keep")),
        "looks_jewish": v.get("looks_jewish"),
        "head_covered": v.get("head_covered"),
        "confidence": v.get("confidence"),
        "skipped": v.get("skipped"),
        "reason": (v.get("reason") or "")[:240],
        "notes": format_verdict_notes(v)[:300],
        "error": v.get("error"),
        "disabled": _disabled_reason,
    }


def _ollama_pod(base: str, path: Path, *, model: str) -> dict:
    r = requests.post(
        f"{base.rstrip('/')}/verify_still",
        json={
            "backend": "open_vlm",
            "open_vlm_model": model,
            "image_b64": _b64(path),
            "timeout": 120,
            "ollama_wait_sec": 300,
        },
        timeout=420,
    )
    try:
        data = r.json()
    except Exception:
        data = {"ok": False, "error": f"http_{r.status_code}:{r.text[:200]}"}
    data["http_status"] = r.status_code
    return data


def _collect_stills() -> list[tuple[str, Path]]:
    out: list[tuple[str, Path]] = []
    dbg = ROOT / "output" / "debug_71170_t20"
    for name, label in (
        ("crop_std_t20.jpg", "71170_t20_keep_expect"),
        ("compare_t22.jpg", "71170_t22_drop_expect"),
        ("full_t20.jpg", "71170_full_t20"),
    ):
        p = dbg / name
        if p.is_file():
            out.append((label, p))

    # Extract a couple Munkács peak frames for the positive-control set.
    video = ROOT / "data" / "videos" / "munkacs_1933_yt.mp4"
    if video.is_file():
        import cv2

        dest = ROOT / "output" / "debug_munkacs_compare"
        dest.mkdir(parents=True, exist_ok=True)
        cap = cv2.VideoCapture(str(video))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
        for t in (14.0, 42.0, 60.0):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(t * fps)))
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            p = dest / f"munkacs_t{int(t)}.jpg"
            cv2.imwrite(str(p), frame)
            out.append((f"munkacs_t{int(t)}", p))
        cap.release()
    return out


def main() -> int:
    load_env()
    apply_settings_to_environ()
    load_env()

    import os

    from runpod_provision import ensure_pod, find_shtetl_pods, stop_pod

    stills = _collect_stills()
    if not stills:
        print("no stills found", flush=True)
        return 1
    print(f"stills={len(stills)}", flush=True)
    _log("start", {"n_stills": len(stills), "labels": [s[0] for s in stills]})

    def status(msg: str) -> None:
        print(msg, flush=True)

    # Prefer 4090 for this A/B (3090 boots were stuck on 502 / never warmed).
    # Settings/load_env would overwrite RUNPOD_GPU_TYPE — pin create order instead.
    # Prefer high-VRAM first (Ollama VLM); keep 3090 last — recent boots hung on 502.
    gpu_order = [
        "NVIDIA GeForce RTX 4090",
        "NVIDIA RTX A6000",
        "NVIDIA L40",
        "NVIDIA RTX A5000",
        "NVIDIA RTX A4500",
        "NVIDIA GeForce RTX 3090",
    ]
    print(f"gpu_try_order={gpu_order}", flush=True)
    import runpod_provision as rp
    import runpod_bootstrap as rb

    # cu128 dies on many hosts (ports=0). Pin the proven cu124 image only.
    good_image = "runpod/pytorch:0.7.0-cu1241-torch260-ubuntu2204"
    os.environ["RUNPOD_DOCKER_IMAGE"] = good_image
    rp._gpu_try_order = lambda preferred: list(gpu_order)  # type: ignore[assignment]
    rp.GPU_FALLBACKS = list(gpu_order)
    rb.IMAGE_CANDIDATES = [good_image]
    rb.DEFAULT_BASE_IMAGE = good_image
    _orig_find = rp.find_shtetl_pods

    def _find_rewrite_image():
        pods = _orig_find()
        for p in pods:
            # Prevent ensure_pods from preferring a dead cu128 "proven" image.
            p["imageName"] = good_image
        return pods

    rp.find_shtetl_pods = _find_rewrite_image  # type: ignore[assignment]
    try:
        import config as app_config

        app_config.RUNPOD_DOCKER_IMAGE = good_image
    except Exception:
        pass
    print(f"image_forced={good_image}", flush=True)

    # Drop zombie pods so ensure_pods creates a fresh machine.
    for p in find_shtetl_pods():
        pid = p.get("id")
        if pid:
            status(f"stopping stale pod {pid}…")
            try:
                stop_pod(pid)
            except Exception as e:
                status(f"stop {pid}: {e}")

    base = ensure_pod(on_status=status, recreate=True)
    print(f"pod={base}", flush=True)
    _push_worker(base)
    health = _wait_health(base, timeout_sec=360)
    _log("pod_ready", {"base": base, "health": health})

    model = (
        __import__("os").environ.get("OPEN_VLM_MODEL") or "qwen2.5vl:3b"
    ).strip() or "qwen2.5vl:3b"
    print(f"ollama_model={model}", flush=True)

    rows = []
    agree = 0
    disagree = 0
    for label, path in stills:
        print(f"--- {label} ---", flush=True)
        oai = _openai_local(path)
        print(f"  openai: keep={oai.get('keep')} {oai.get('notes', '')[:160]}", flush=True)
        vlm = _ollama_pod(base, path, model=model)
        print(
            f"  ollama: ok={vlm.get('ok')} keep={vlm.get('keep')} "
            f"{(vlm.get('notes') or vlm.get('error') or '')[:160]}",
            flush=True,
        )
        same = None
        if vlm.get("ok") and oai.get("keep") is not None and vlm.get("keep") is not None:
            same = bool(oai.get("keep")) == bool(vlm.get("keep"))
            if same:
                agree += 1
            else:
                disagree += 1
        row = {
            "label": label,
            "path": str(path),
            "openai": oai,
            "ollama": {
                "ok": vlm.get("ok"),
                "keep": vlm.get("keep"),
                "looks_jewish": vlm.get("looks_jewish"),
                "head_covered": vlm.get("head_covered"),
                "confidence": vlm.get("confidence"),
                "notes": (vlm.get("notes") or "")[:300],
                "error": vlm.get("error"),
                "model": vlm.get("model"),
            },
            "agree": same,
        }
        rows.append(row)
        _log("pair", row, hypothesis_id="O")

    summary = {
        "agree": agree,
        "disagree": disagree,
        "n": len(rows),
        "model": model,
        "pod": base,
    }
    print("---", flush=True)
    print(f"agree={agree} disagree={disagree} / {len(rows)}", flush=True)
    for r in rows:
        print(
            f"  {r['label']}: openai={r['openai'].get('keep')} "
            f"ollama={r['ollama'].get('keep')} agree={r['agree']}",
            flush=True,
        )
    _log("summary", summary, hypothesis_id="O")

    out = ROOT / "output" / "ollama_vs_openai_compare.json"
    out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2), encoding="utf-8")
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
