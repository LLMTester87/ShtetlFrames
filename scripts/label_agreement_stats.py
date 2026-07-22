"""Print Keep/Pass vs OpenAI agreement stats (read-only)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from config import load_env  # noqa: E402
from label_feedback import compute_label_stats, export_fewshot_cache  # noqa: E402


def main() -> int:
    load_env()
    stats = compute_label_stats()
    print(json.dumps(stats, indent=2))
    rate = stats.get("agreement_rate")
    if rate is not None:
        print(
            f"\nagreement_rate={rate:.1%} "
            f"({stats['agree_keep'] + stats['agree_drop']}/{stats['n_labeled']} labeled)",
            flush=True,
        )
    try:
        path = export_fewshot_cache()
        if path:
            print(f"fewshot_meta={path}", flush=True)
    except Exception as e:
        print(f"fewshot_meta_skip={e}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
