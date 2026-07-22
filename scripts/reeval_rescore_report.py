"""Re-apply loosened clamps to existing rescore report (no CLIP reload)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from shtetl_core.cues import (  # noqa: E402
    DEFAULT_SCORE_THRESHOLD,
    MAX_NEG_TO_POS_RATIO,
    MIN_POS_SCORE,
)
from shtetl_core.scoring import clamp_strong_negative, clamp_weak_score  # noqa: E402

REPORT = ROOT / "output" / "candidate_rescore_report.json"


def main() -> int:
    data = json.loads(REPORT.read_text(encoding="utf-8"))
    thr = DEFAULT_SCORE_THRESHOLD
    rows = []
    for r in data["rows"]:
        if r.get("pos_score") is None:
            rows.append(r)
            continue
        pos = float(r["pos_score"])
        neg = float(r["neg_score"])
        score = pos - neg
        score = clamp_weak_score(
            score, pos, min_pos_score=MIN_POS_SCORE, score_threshold=thr
        )
        score = clamp_strong_negative(
            score,
            pos,
            neg,
            max_neg_to_pos_ratio=MAX_NEG_TO_POS_RATIO,
            score_threshold=thr,
        )
        r2 = dict(r)
        r2["new_score"] = round(score, 4)
        r2["passes_new_gate"] = score >= thr
        old = r.get("old_peak_score")
        r2["delta"] = round(score - float(old), 4) if old is not None else None
        rows.append(r2)

    scored = [r for r in rows if r.get("pos_score") is not None]
    passed = [r for r in scored if r["passes_new_gate"]]
    acc = [r for r in scored if r["decision"] == "accept"]
    rej = [r for r in scored if r["decision"] == "reject"]
    acc_p = sum(1 for r in acc if r["passes_new_gate"])
    rej_p = sum(1 for r in rej if r["passes_new_gate"])

    summary = {
        "n_total": len(rows),
        "n_scored": len(scored),
        "n_pass_new_gate": len(passed),
        "n_fail_new_gate": len(scored) - len(passed),
        "score_threshold": thr,
        "min_pos_score": MIN_POS_SCORE,
        "max_neg_to_pos_ratio": MAX_NEG_TO_POS_RATIO,
        "accept_pass": f"{acc_p}/{len(acc)}",
        "reject_leak": f"{rej_p}/{len(rej)}",
    }
    REPORT.write_text(
        json.dumps({"summary": summary, "rows": rows}, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    print("--- accepts ---")
    for r in acc:
        print(
            f"#{r['id']} score={r['new_score']:.4f} pass={r['passes_new_gate']} "
            f"pos={r['pos_score']} neg={r['neg_score']}"
        )
    print("--- reject leaks ---")
    for r in rej:
        if r["passes_new_gate"]:
            print(f"#{r['id']} score={r['new_score']:.4f} openai={r['openai']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
