"""Summarize output/pod_monitor_10m.jsonl."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
p = ROOT / "output" / "pod_monitor_10m.jsonl"
rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
print("samples", len(rows))
t0, t1 = rows[0]["ts"], rows[-1]["ts"]
dur = (t1 - t0) / 60.0
j0, j1 = rows[0].get("job") or {}, rows[-1].get("job") or {}
c0, c1 = int(j0.get("completed") or 0), int(j1.get("completed") or 0)
q0 = (rows[0].get("queue") or {}).get("pathe") or {}
q1 = (rows[-1].get("queue") or {}).get("pathe") or {}
d0, d1 = int(q0.get("n_done") or 0), int(q1.get("n_done") or 0)
print(f"duration_min={dur:.1f}")
print(f"job_completed {c0} -> {c1}  delta={c1 - c0}  rate={(c1 - c0) / max(dur, 0.01):.2f}/min")
print(f"queue_done {d0} -> {d1}  delta={d1 - d0}  rate={(d1 - d0) / max(dur, 0.01):.2f}/min")
print(
    f"pending {q0.get('n_pending')} -> {q1.get('n_pending')}  "
    f"active {q0.get('n_active')} -> {q1.get('n_active')}"
)
print(f"workers {j0.get('workers')} -> {j1.get('workers')}")
print(f"msg_end={(j1.get('message') or '')[:120]}")

busy_hist, idle_hist, stack_hist = [], [], []
phases = Counter()
lims = Counter()
per_pod_idle: Counter[str] = Counter()
per_pod_busy: Counter[str] = Counter()
for r in rows:
    pods = r.get("pods") or []
    busy = idle = stacked = 0
    for pod in pods:
        ph = pod.get("phase") or "idle"
        phases[ph] += 1
        inf = pod.get("inflight")
        lim = pod.get("inflight_limit_pathe")
        if lim is not None:
            lims[str(lim)] += 1
        name = pod.get("name") or "?"
        is_busy = ph not in ("", "idle", "done", None) or (
            isinstance(inf, int) and inf > 0
        )
        if is_busy:
            busy += 1
            per_pod_busy[name] += 1
        else:
            idle += 1
            per_pod_idle[name] += 1
        if isinstance(inf, int) and inf > 1:
            stacked += 1
    busy_hist.append(busy)
    idle_hist.append(idle)
    stack_hist.append(stacked)
    phc = Counter((p.get("phase") or "idle") for p in pods)
    print(
        f"  [{r['i']:2d}] pods={r['n_pods']} busy={busy} idle={idle} stacked={stacked} "
        f"job_done={(r.get('job') or {}).get('completed')} "
        f"active={((r.get('queue') or {}).get('pathe') or {}).get('n_active')} "
        f"phases={dict(phc)}"
    )

n = len(busy_hist)
print(f"busy avg={sum(busy_hist)/n:.1f} min={min(busy_hist)} max={max(busy_hist)}")
print(f"idle avg={sum(idle_hist)/n:.1f}")
print(f"stacked avg={sum(stack_hist)/n:.1f} max={max(stack_hist)}")
print("phase totals", dict(phases))
print("pathe inflight limits seen", dict(lims))
print("per-pod idle ticks", dict(per_pod_idle))
print("per-pod busy ticks", dict(per_pod_busy))
rate = (d1 - d0) / max(dur, 0.01)
pend = int(q1.get("n_pending") or 0)
if rate > 0:
    print(f"ETA_hours_at_this_rate={pend / rate / 60.0:.1f}")
else:
    print("ETA n/a")
