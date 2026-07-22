"""Compare cheaper OpenAI models vs existing gpt-5.6-sol verdicts (read-only)."""
from __future__ import annotations

import os
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from config import load_env
from db import db, init_db
from openai_verify import format_verdict_notes, verdict_is_keep, verify_still
from still_store import candidate_still_path

load_env()
init_db()

MODELS = [
    "gpt-5.6-terra",  # ~0.5x Sol
    "gpt-5.6-luna",  # ~0.2x Sol
    "gpt-5.4-mini",  # ~0.15x Sol input
]


def parse_tag(notes: str) -> dict:
    low = (notes or "").lower()
    if "openai:keep" in low:
        tag = "keep"
    elif "openai:drop" in low:
        tag = "drop"
    elif "openai:uncertain" in low:
        tag = "uncertain"
    else:
        tag = "none"
    j = re.search(r"jewish=(yes|no)", low)
    h = re.search(r"head=(yes|no)", low)
    return {
        "tag": tag,
        "jewish": j.group(1) if j else None,
        "head": h.group(1) if h else None,
        "notes": notes or "",
    }


def pick_sample(limit_keep: int = 12, limit_drop: int = 18) -> list[dict]:
    with db() as conn:
        rows = [
            dict(r)
            for r in conn.execute(
                "SELECT id, decision, notes FROM candidates ORDER BY id ASC"
            )
        ]
    keeps, drops, special = [], [], []
    for r in rows:
        p = parse_tag(r.get("notes") or "")
        if p["tag"] not in ("keep", "drop"):
            continue
        if not candidate_still_path(int(r["id"])).is_file():
            continue
        r["_sol"] = p
        decision = (r.get("decision") or "").strip()
        # prioritize disagreements / labeled
        if decision in ("accept", "reject"):
            special.append(r)
        elif p["tag"] == "keep":
            keeps.append(r)
        else:
            drops.append(r)
    out: list[dict] = []
    seen: set[int] = set()
    for r in special:
        if r["id"] not in seen:
            out.append(r)
            seen.add(r["id"])
    for r in keeps:
        if len([x for x in out if x["_sol"]["tag"] == "keep"]) >= limit_keep:
            break
        if r["id"] not in seen:
            out.append(r)
            seen.add(r["id"])
    for r in drops:
        if len([x for x in out if x["_sol"]["tag"] == "drop"]) >= limit_drop:
            break
        if r["id"] not in seen:
            out.append(r)
            seen.add(r["id"])
    return out


def run_model(model: str, sample: list[dict]) -> list[dict]:
    os.environ["OPENAI_MODEL"] = model
    # Clear few-shot cache so each model gets a fair prompt pack.
    try:
        from label_feedback import build_fewshot_content_parts

        build_fewshot_content_parts(force=True)
    except Exception:
        pass
    results = []
    for i, r in enumerate(sample, 1):
        cid = int(r["id"])
        path = candidate_still_path(cid)
        print(f"  [{model}] {i}/{len(sample)} #{cid}…", flush=True)
        v = verify_still(image_path=path, timeout=60.0)
        if v.get("skipped") or v.get("error"):
            pred = {
                "tag": "error",
                "jewish": None,
                "head": None,
                "reason": str(v.get("reason") or v.get("error") or "")[:120],
            }
        else:
            ok = verdict_is_keep(v)
            pred = {
                "tag": "uncertain" if v.get("uncertain") else ("keep" if ok else "drop"),
                "jewish": "yes" if v.get("looks_jewish") else "no",
                "head": "yes" if v.get("head_covered") else "no",
                "reason": format_verdict_notes(v)[:140],
            }
        results.append({"id": cid, "sol": r["_sol"], "pred": pred, "decision": r.get("decision")})
    return results


def summarize(model: str, rows: list[dict]) -> None:
    comparable = [r for r in rows if r["pred"]["tag"] in ("keep", "drop", "uncertain")]
    agree = sum(1 for r in comparable if r["pred"]["tag"] == r["sol"]["tag"])
    n = len(comparable)
    print(f"\n=== {model} vs Sol keep/drop ===")
    print(f"n={n}  tag_agree={agree}/{n} ({(agree/n if n else 0):.1%})")
    # confusion
    pairs = Counter((r["sol"]["tag"], r["pred"]["tag"]) for r in comparable)
    for (a, b), c in sorted(pairs.items()):
        print(f"  Sol {a} → {model} {b}: {c}")
    # flag agreement among cases where both have flags
    j_ok = h_ok = j_n = h_n = 0
    for r in comparable:
        if r["sol"]["jewish"] and r["pred"]["jewish"]:
            j_n += 1
            j_ok += int(r["sol"]["jewish"] == r["pred"]["jewish"])
        if r["sol"]["head"] and r["pred"]["head"]:
            h_n += 1
            h_ok += int(r["sol"]["head"] == r["pred"]["head"])
    if j_n:
        print(f"  jewish flag agree: {j_ok}/{j_n} ({j_ok/j_n:.1%})")
    if h_n:
        print(f"  head flag agree: {h_ok}/{h_n} ({h_ok/h_n:.1%})")
    # vs human where labeled
    labeled = [r for r in comparable if (r.get("decision") or "") in ("accept", "reject")]
    if labeled:
        # human accept→keep, reject→drop
        hum_agree = 0
        for r in labeled:
            want = "keep" if r["decision"] == "accept" else "drop"
            if r["pred"]["tag"] == want:
                hum_agree += 1
        print(f"  vs human decision: {hum_agree}/{len(labeled)} ({hum_agree/len(labeled):.1%})")
        sol_hum = sum(
            1
            for r in labeled
            if r["sol"]["tag"] == ("keep" if r["decision"] == "accept" else "drop")
        )
        print(f"  Sol vs human (same subset): {sol_hum}/{len(labeled)} ({sol_hum/len(labeled):.1%})")
    # mismatches
    mism = [r for r in comparable if r["pred"]["tag"] != r["sol"]["tag"]]
    print(f"  mismatches ({len(mism)}):")
    for r in mism[:12]:
        print(
            f"    #{r['id']} Sol={r['sol']['tag']} j={r['sol']['jewish']} h={r['sol']['head']}"
            f" → {r['pred']['tag']} j={r['pred']['jewish']} h={r['pred']['head']}"
        )


def main() -> int:
    sample = pick_sample()
    print(
        f"Sample size={len(sample)} "
        f"keeps={sum(1 for r in sample if r['_sol']['tag']=='keep')} "
        f"drops={sum(1 for r in sample if r['_sol']['tag']=='drop')} "
        f"labeled={sum(1 for r in sample if (r.get('decision') or '') in ('accept','reject'))}",
        flush=True,
    )
    for model in MODELS:
        rows = run_model(model, sample)
        summarize(model, rows)
    print(
        "\nPricing context (per 1M tokens, approx): "
        "Sol $5/$30 · Terra $2.50/$15 · Luna $1/$6 · 5.4-mini $0.75/$4.50",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
