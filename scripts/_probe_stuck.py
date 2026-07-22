import json
import sqlite3
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

sys.path.insert(0, "src")
from config import load_env

load_env()
from runpod_provision import find_shtetl_pods, pod_proxy_url


def probe(p):
    pid = p.get("id") or ""
    name = p.get("name") or "?"
    des = p.get("desiredStatus")
    rt = p.get("runtime")
    ports = (rt or {}).get("ports") if isinstance(rt, dict) else None
    base = pod_proxy_url(pid)
    out = {
        "name": name,
        "id": pid[:14],
        "desired": des,
        "ports": bool(ports),
        "base": base,
    }
    try:
        h = requests.get(base + "/health", timeout=6)
        out["health"] = h.status_code
        if h.status_code == 200 and h.content:
            d = h.json()
            out["ok"] = d.get("ok")
            out["ready"] = d.get("models_ready")
            out["inflight"] = d.get("inflight")
            out["pathe_lim"] = d.get("inflight_limit_pathe")
            out["warm_err"] = (d.get("warm_error") or "")[:80]
            sync = d.get("github_sync") or {}
            if isinstance(sync, dict):
                out["sync_err"] = (sync.get("last_error") or "")[:60]
                out["sync_pending"] = sync.get("pending_soft_recycle")
                out["sync_changed"] = sync.get("last_changed")
    except Exception as e:
        out["health"] = type(e).__name__
    try:
        pr = requests.get(base + "/progress", timeout=6)
        if pr.status_code == 200 and pr.content:
            pj = pr.json()
            out["phase"] = pj.get("phase")
            out["msg"] = (pj.get("message") or "")[:70]
            out["title"] = (pj.get("title") or pj.get("detail") or "")[:50]
            out["qid"] = pj.get("queue_id")
            out["pct"] = pj.get("pct")
        else:
            out["phase"] = f"http_{pr.status_code}"
    except Exception as e:
        out["phase"] = type(e).__name__
    return out


pods = find_shtetl_pods()
print("PODS", len(pods))
with ThreadPoolExecutor(max_workers=8) as ex:
    rows = list(ex.map(probe, pods))
for r in sorted(rows, key=lambda x: x["name"]):
    print(json.dumps(r, ensure_ascii=False))

db = Path("output/shtetlframes.db")
if db.exists():
    c = sqlite3.connect(str(db))
    c.row_factory = sqlite3.Row
    for jid in ("pathe_scrape", "pathe_discover", "scrape"):
        row = c.execute(
            "select id,status,phase,message,workers,progress from jobs where id=?",
            (jid,),
        ).fetchone()
        if row:
            print("JOB", dict(row))
    q = c.execute(
        "select status, count(*) n from queue_items "
        "where url like '%britishpathe%' group by status"
    ).fetchall()
    print("PATHE_Q", [dict(r) for r in q])
    stuck = c.execute(
        "select id, status, detail, substr(title,1,40) t from queue_items "
        "where url like '%britishpathe%' and status in "
        "('queued','scanning','downloading','uploading') "
        "order by id desc limit 15"
    ).fetchall()
    print("ACTIVE_ROWS", [dict(r) for r in stuck])
