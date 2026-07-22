"""Read-only 10-minute pod workload sampler."""
from __future__ import annotations

import json
import sqlite3
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from config import load_env

load_env()
from runpod_provision import find_shtetl_pods, pod_proxy_url

OUT = ROOT / "output" / "pod_monitor_10m.jsonl"
SAMPLES = 10
INTERVAL_SEC = 60.0


def _queue_stats() -> dict:
    db = ROOT / "output" / "shtetlframes.db"
    out: dict = {}
    if not db.exists():
        return out
    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    try:
        for label, where in (
            ("pathe", "url LIKE '%britishpathe.com%'"),
            ("other", "url NOT LIKE '%britishpathe.com%'"),
        ):
            row = conn.execute(
                f"""
                SELECT
                  SUM(CASE WHEN status='done' THEN 1 ELSE 0 END) AS n_done,
                  SUM(CASE WHEN status='error' THEN 1 ELSE 0 END) AS n_error,
                  SUM(CASE WHEN status='pending' THEN 1 ELSE 0 END) AS n_pending,
                  SUM(CASE WHEN status IN ('queued','scanning','downloading','uploading')
                           THEN 1 ELSE 0 END) AS n_active
                FROM queue_items WHERE {where}
                """
            ).fetchone()
            out[label] = {
                "n_done": int(row["n_done"] or 0),
                "n_error": int(row["n_error"] or 0),
                "n_pending": int(row["n_pending"] or 0),
                "n_active": int(row["n_active"] or 0),
            }
        jobs = {}
        for r in conn.execute(
            "SELECT id, status, phase, message, workers, progress FROM jobs"
        ):
            jobs[r["id"]] = {
                "status": r["status"],
                "phase": r["phase"],
                "message": (r["message"] or "")[:160],
                "workers": r["workers"],
            }
        out["jobs"] = jobs
    except Exception as e:
        out["err"] = str(e)[:200]
    finally:
        conn.close()
    return out


def _probe(name: str, pid: str, base: str) -> dict:
    row: dict = {"name": name, "id": pid[:18], "base": base}
    try:
        hr = requests.get(base.rstrip("/") + "/health", timeout=8)
        row["health_status"] = hr.status_code
        hj = hr.json() if hr.status_code == 200 and hr.content else {}
        row["ok"] = hj.get("ok")
        row["inflight"] = hj.get("inflight")
        row["inflight_limit_pathe"] = hj.get("inflight_limit_pathe")
        row["inflight_limit_yt"] = hj.get("inflight_limit_yt")
        sync = hj.get("github_sync") if isinstance(hj.get("github_sync"), dict) else {}
        row["sync_changed"] = sync.get("last_changed")
        row["sync_error"] = sync.get("last_error")
    except Exception as e:
        row["health_err"] = type(e).__name__
    try:
        pr = requests.get(base.rstrip("/") + "/progress", timeout=8)
        pj = pr.json() if pr.status_code == 200 and pr.content else {}
        row["phase"] = pj.get("phase")
        row["message"] = (pj.get("message") or "")[:80]
        row["title"] = (pj.get("title") or pj.get("detail") or "")[:120]
        row["queue_id"] = pj.get("queue_id")
        row["pct"] = pj.get("pct")
    except Exception as e:
        row["progress_err"] = type(e).__name__
    return row


def _local_job() -> dict:
    try:
        r = requests.get("http://127.0.0.1:8787/api/jobs", timeout=6)
        j = (r.json() or {}).get("jobs") or {}
        return j.get("pathe_scrape") or {}
    except Exception as e:
        return {"err": type(e).__name__}


def sample_once(i: int) -> dict:
    pods = find_shtetl_pods()
    probes: list[dict] = []
    with ThreadPoolExecutor(max_workers=max(1, len(pods) or 1)) as ex:
        futs = [
            ex.submit(
                _probe,
                p.get("name") or (p.get("id") or "")[:12],
                p.get("id") or "",
                pod_proxy_url(p.get("id") or ""),
            )
            for p in pods
            if p.get("id")
        ]
        for fut in as_completed(futs):
            probes.append(fut.result())
    probes.sort(key=lambda r: r.get("name") or "")
    return {
        "i": i,
        "ts": time.time(),
        "n_pods": len(pods),
        "queue": _queue_stats(),
        "job": _local_job(),
        "pods": probes,
    }


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    print(f"sampling {SAMPLES}x every {INTERVAL_SEC:.0f}s → {OUT}", flush=True)
    t0 = time.time()
    with OUT.open("w", encoding="utf-8") as fh:
        for i in range(1, SAMPLES + 1):
            snap = sample_once(i)
            fh.write(json.dumps(snap, ensure_ascii=False) + "\n")
            fh.flush()
            busy = sum(
                1
                for p in snap["pods"]
                if (p.get("phase") or "") not in ("", "idle", "done", None)
                or (isinstance(p.get("inflight"), int) and p["inflight"] > 0)
            )
            stacked = sum(
                1
                for p in snap["pods"]
                if isinstance(p.get("inflight"), int) and p["inflight"] > 1
            )
            pq = (snap.get("queue") or {}).get("pathe") or {}
            print(
                f"[{i}/{SAMPLES}] pods={snap['n_pods']} busy={busy} stacked={stacked} "
                f"pathe_active={pq.get('n_active')} done={pq.get('n_done')} "
                f"elapsed={time.time()-t0:.0f}s",
                flush=True,
            )
            if i < SAMPLES:
                time.sleep(INTERVAL_SEC)
    print(f"done in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
