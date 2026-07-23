"""Re-verify Munkács positives vs current Pathé openai:keep false alarms.

Expect: Munkács stills KEEP; Pathé FP ids DROP under the hardened prompt.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from config import load_env  # noqa: E402
from openai_verify import format_verdict_notes, verdict_is_keep, verify_still  # noqa: E402

# Must-keep Munkács stills (historical Review accepts / clear workshop+yeshiva frames).
# Soft peak 0.1040 is intentionally omitted — too blurry; CLIP gate already weak there.
MUNKACS = [
    ROOT / "output" / "debug_munkacs_compare" / "munkacs_peak_0.1623.jpg",
    ROOT / "output" / "debug_munkacs_compare" / "munkacs_peak_0.1414.jpg",
    ROOT / "output" / "debug_munkacs_compare" / "munkacs_peak_0.1191.jpg",
    ROOT / "output" / "debug_munkacs_compare" / "munkacs_t60.jpg",
    ROOT / "output" / "debug_munkacs_compare" / "munkacs_t42.jpg",
    ROOT / "output" / "debug_munkacs_compare" / "munkacs_t14.jpg",
]

# Current Review openai:keep rows that should fail under the harder gate.
FP_IDS = [1806, 1822, 1824, 1825, 1831, 1832, 1833]


def main() -> int:
    load_env()
    rows: list[dict] = []
    ok_tp = ok_fp = 0
    n_tp = n_fp = 0

    print("=== TRUE POSITIVES (Munkács) — expect KEEP ===", flush=True)
    for path in MUNKACS:
        n_tp += 1
        if not path.is_file():
            print(f"MISSING {path}", flush=True)
            rows.append({"path": str(path), "expect": "keep", "ok": False, "error": "missing"})
            continue
        v = verify_still(image_path=path)
        kept = verdict_is_keep(v)
        ok = kept
        ok_tp += int(ok)
        note = format_verdict_notes(v)
        print(f"{'OK' if ok else 'FAIL'} {path.name}: {note}", flush=True)
        rows.append(
            {
                "path": str(path),
                "expect": "keep",
                "ok": ok,
                "kept": kept,
                "verdict": {
                    "keep": v.get("keep"),
                    "looks_jewish": v.get("looks_jewish"),
                    "head_covered": v.get("head_covered"),
                    "marker": v.get("marker"),
                    "confidence": v.get("confidence"),
                    "reason": v.get("reason"),
                },
            }
        )

    print("\n=== FALSE KEEPS (Pathé) — expect DROP ===", flush=True)
    for cid in FP_IDS:
        n_fp += 1
        path = ROOT / "output" / "contact_sheets" / f"cand_{cid}.jpg"
        if not path.is_file():
            print(f"MISSING {path}", flush=True)
            rows.append({"id": cid, "expect": "drop", "ok": False, "error": "missing"})
            continue
        v = verify_still(image_path=path)
        kept = verdict_is_keep(v)
        ok = not kept
        ok_fp += int(ok)
        note = format_verdict_notes(v)
        print(f"{'OK' if ok else 'FAIL'} cand_{cid}: {note}", flush=True)
        rows.append(
            {
                "id": cid,
                "path": str(path),
                "expect": "drop",
                "ok": ok,
                "kept": kept,
                "verdict": {
                    "keep": v.get("keep"),
                    "looks_jewish": v.get("looks_jewish"),
                    "head_covered": v.get("head_covered"),
                    "marker": v.get("marker"),
                    "confidence": v.get("confidence"),
                    "reason": v.get("reason"),
                },
            }
        )

    summary = {
        "munkacs_keep": f"{ok_tp}/{n_tp}",
        "pathe_fp_drop": f"{ok_fp}/{n_fp}",
        "pass": ok_tp == n_tp and ok_fp == n_fp,
        "rows": rows,
    }
    out = ROOT / "output" / "_harden_validate.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSummary: munkacs_keep={ok_tp}/{n_tp} pathe_fp_drop={ok_fp}/{n_fp}", flush=True)
    print(f"Wrote {out}", flush=True)
    return 0 if summary["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
