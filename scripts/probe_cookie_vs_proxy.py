"""One-shot RunPod probe: cookies-only first, then Scrapfly if needed. Heavy logging."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

LOG = ROOT / "data" / "probe_cookie_vs_proxy.log"


def log(msg: str, **data) -> None:
    line = {"t": time.strftime("%H:%M:%S"), "msg": msg, **data}
    text = json.dumps(line, ensure_ascii=False)
    print(text, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(text + "\n")


def main() -> int:
    from config import load_env

    load_env()
    from runpod_client import set_pod_pool
    from runpod_provision import ensure_pods, find_shtetl_pods
    from yt_cookies import ensure_cookies_for_scrape, read_cookies_text
    from yt_proxy import configured_provider, residential_proxy_url
    import requests

    url = "https://www.youtube.com/watch?v=eIjzXJR1Rag"
    title = "probe Smuggling Illegal Migrants (1959)"
    queue_id = 900001

    if len(sys.argv) > 1:
        url = sys.argv[1]

    LOG.write_text("", encoding="utf-8")
    ck = ensure_cookies_for_scrape()
    cookies = read_cookies_text()
    provider = configured_provider()
    proxy = residential_proxy_url()
    log(
        "start",
        url=url,
        cookies_ok=bool(cookies),
        cookies_bytes=len(cookies or ""),
        ensure=ck.get("message"),
        provider=provider,
        has_proxy=bool(proxy),
    )
    if not cookies:
        log("abort_no_cookies")
        return 2

    existing = find_shtetl_pods()
    log("existing_pods", n=len(existing), names=[(p.get("name"), p.get("desiredStatus")) for p in existing[:6]])

    def on_status(m: str) -> None:
        log("status", text=m[:220])

    log("ensure_pods_begin")
    bases = ensure_pods(count=1, on_status=on_status, min_ready=1, extra_fill_sec=0)
    set_pod_pool(bases)
    base = bases[0]
    log("pod_ready", base_tail=base.split("//")[-1][:40])

    # Health
    try:
        hr = requests.get(f"{base}/health", timeout=20)
        log("health", status=hr.status_code, body=(hr.text or "")[:300])
    except Exception as e:
        log("health_err", err=str(e)[:200])
        return 3

    def submit(payload: dict, label: str) -> dict:
        log("submit", label=label, force_proxy=payload.get("force_proxy"), has_proxy_url=bool(payload.get("proxy_url")), has_cookies=bool(payload.get("cookies_text")))
        t0 = time.time()
        r = requests.post(f"{base}/scan", json=payload, timeout=90)
        log("scan_accept", label=label, http=r.status_code, body=(r.text or "")[:400])
        try:
            out = r.json()
        except Exception:
            return {"ok": False, "error": f"bad_json:{r.status_code}"}
        if not (out.get("accepted") or r.status_code == 202 or out.get("async")):
            # sync or immediate error
            return out if isinstance(out, dict) else {"ok": False, "error": "bad"}
        qid = out.get("queue_id") or payload.get("queue_id")
        deadline = time.time() + 900
        last_prog = ""
        while time.time() < deadline:
            try:
                pr = requests.get(f"{base}/progress", params={"queue_id": qid}, timeout=15)
                if pr.status_code == 200 and pr.content:
                    prog = pr.json()
                    line = f"{prog.get('phase')}|{prog.get('message')}|{prog.get('detail')}|{prog.get('pct')}"
                    if line != last_prog:
                        last_prog = line
                        log("progress", label=label, phase=prog.get("phase"), message=str(prog.get("message") or "")[:120], detail=str(prog.get("detail") or "")[:160], pct=prog.get("pct"))
                rr = requests.get(f"{base}/result", params={"queue_id": qid}, timeout=30)
                if rr.status_code == 200 and rr.content:
                    data = rr.json()
                    if isinstance(data, dict) and not data.get("pending", True):
                        log("result", label=label, ok=data.get("ok"), error=str(data.get("error") or "")[:300], n_hits=data.get("n_hits"), elapsed=int(time.time() - t0))
                        return data
            except Exception as e:
                log("poll_err", label=label, err=str(e)[:160])
            time.sleep(2.5)
        return {"ok": False, "error": "timeout", "elapsed": int(time.time() - t0)}

    # Phase A: cookies only (no Scrapfly)
    payload_a = {
        "url": url,
        "title": title,
        "queue_id": queue_id,
        "sample_fps": 1.0,
        "score_threshold": 0.10,
        "source_url": url,
        "force_proxy": False,
        "proxy_provider": provider,
        "cookies_text": cookies,
        # deliberately omit proxy_url
    }
    log("phase_A_cookies_only")
    out_a = submit(payload_a, "cookies_only")
    if out_a.get("ok"):
        log("SUCCESS_cookies_only", n_hits=out_a.get("n_hits"))
        return 0

    err_a = str(out_a.get("error") or out_a)[:400]
    log("cookies_only_failed", error=err_a)

    if not proxy:
        log("no_proxy_configured_cannot_escalate")
        return 4

    # Phase B: force Scrapfly
    payload_b = dict(payload_a)
    payload_b["queue_id"] = queue_id + 1
    payload_b["force_proxy"] = True
    payload_b["proxy_url"] = proxy
    payload_b["proxy_insecure"] = provider == "scrapfly"
    log("phase_B_force_scrapfly")
    out_b = submit(payload_b, "force_proxy")
    if out_b.get("ok"):
        log("SUCCESS_force_proxy", n_hits=out_b.get("n_hits"))
        return 0
    log("force_proxy_failed", error=str(out_b.get("error") or out_b)[:400])
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
