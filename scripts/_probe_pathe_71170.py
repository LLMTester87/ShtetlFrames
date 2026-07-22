"""One-off: score britishpathe.com/asset/71170/ and report keep/drop."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from config import load_env  # noqa: E402
import config as app_config  # noqa: E402
from settings_store import apply_settings_to_environ  # noqa: E402


def main() -> int:
    load_env()
    apply_settings_to_environ()
    load_env()

    from britishpathe import prepare_pathe_job
    from openai_verify import (
        filter_candidates_openai,
        notes_openai_approved,
        notes_openai_dropped,
        openai_verify_enabled,
    )
    from runpod_client import _attach_vision_verify_payload, segments_to_candidate_rows
    from runpod_provision import find_shtetl_pods, pod_proxy_url

    url = "https://www.britishpathe.com/asset/71170/"
    pods = [p for p in find_shtetl_pods() if (p.get("desiredStatus") or "") == "RUNNING"]
    print(f"running pods={len(pods)}", flush=True)
    base = None
    for p in pods:
        b = pod_proxy_url(p["id"])
        try:
            h = requests.get(f"{b.rstrip('/')}/health", timeout=10)
            data = h.json() if h.content else {}
            ready = bool(data.get("models_ready"))
            inf = data.get("inflight")
            print(
                f"  {p.get('name')} ready={ready} inflight={inf} http={h.status_code}",
                flush=True,
            )
            if base is None and ready and int(inf or 0) == 0:
                base = b
        except Exception as e:
            print(f"  {p.get('name')} health fail: {e}", flush=True)
    if not base and pods:
        base = pod_proxy_url(pods[0]["id"])
    if not base:
        print("no pod", flush=True)
        return 1

    thr = float(getattr(app_config, "SCORE_THRESHOLD", None) or 0.04)
    print(
        f"using {base.split('//')[-1][:40]} thr={thr} "
        f"verify={os.environ.get('VERIFY_BACKEND')}",
        flush=True,
    )

    def on_status(msg: str, **_kw) -> None:
        print(f"  status: {msg}", flush=True)

    print("resolving Pathé HLS…", flush=True)
    job = prepare_pathe_job(url, "asset-71170", on_status=on_status)
    if not job:
        print("resolve failed", flush=True)
        return 1

    qid = f"probe-71170-{int(time.time())}"
    payload = {
        "url": job["download_url"],
        "title": job.get("title") or "asset-71170",
        "queue_id": qid,
        "sample_fps": 0.5,
        "score_threshold": thr,
        "source_url": url,
        "source": "britishpathe",
        "m3u8_url": job["m3u8_url"],
        "referer": job["referer"],
        "force_proxy": False,
        "proxy_provider": "none",
        "proxy_insecure": False,
        "pathe_max_inflight": 3,
    }
    _attach_vision_verify_payload(payload)

    print("POST /scan…", flush=True)
    r = requests.post(f"{base.rstrip('/')}/scan", json=payload, timeout=120)
    print(f"scan status={r.status_code} body={(r.text or '')[:240]}", flush=True)
    if r.status_code not in (200, 202):
        return 1

    deadline = time.time() + 900
    out = None
    while time.time() < deadline:
        pr = requests.get(
            f"{base.rstrip('/')}/result",
            params={"queue_id": qid},
            timeout=60,
        )
        data = pr.json() if pr.content else {}
        if data.get("pending"):
            prog = data.get("detail") or data.get("message") or data.get("phase")
            print(f"  pending… {prog}", flush=True)
            time.sleep(4)
            continue
        out = data
        break

    if not out:
        print("timeout waiting for result", flush=True)
        return 1

    print(f"ok={out.get('ok')} error={out.get('error')}", flush=True)
    segs = out.get("segments") or []
    print(f"segments={len(segs)}", flush=True)
    for i, s in enumerate(segs, 1):
        print(
            f"  seg#{i} {s.get('start_sec')}-{s.get('end_sec')} "
            f"peak={s.get('peak_score')} notes={(s.get('notes') or '')[:160]}",
            flush=True,
        )

    rows = segments_to_candidate_rows(out, source_url=url)
    if rows and openai_verify_enabled():
        rows = filter_candidates_openai(
            rows, on_status=lambda m: print(f"  filter: {m}", flush=True)
        )

    keeps: list[int] = []
    drops: list[int] = []
    for i, row in enumerate(rows, 1):
        notes = row.get("notes") or ""
        print(
            f"#{i} {row.get('start_sec')}-{row.get('end_sec')}s "
            f"peak={row.get('peak_score')} rank={row.get('rank_score')} "
            f"cue={row.get('best_cue')} notes={str(notes)[:220]}",
            flush=True,
        )
        if notes_openai_approved(notes):
            keeps.append(i)
        elif notes_openai_dropped(notes):
            drops.append(i)

    print("---", flush=True)
    print(f"CLIP hits returned: {len(segs)}", flush=True)
    print(f"OpenAI keeps: {keeps}", flush=True)
    print(f"OpenAI drops: {drops}", flush=True)
    print(f"WOULD_ENTER_REVIEW_POSITIVE={bool(keeps)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
