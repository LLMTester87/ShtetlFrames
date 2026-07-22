"""Re-run OpenAI Orthodox-appearance gate on existing Review candidates.

Rewrites notes to openai:keep / openai:drop / openai:uncertain from the current
prompt (+ optional Keep/Pass few-shots). Does not overwrite human decision.

Usage:
  python scripts/reverify_review_openai.py              # all with image
  python scripts/reverify_review_openai.py --disagreements
  python scripts/reverify_review_openai.py --pending
  python scripts/reverify_review_openai.py --limit 50
  python scripts/reverify_review_openai.py --dry-run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from config import load_env  # noqa: E402
from db import db, init_db  # noqa: E402
from label_feedback import compute_label_stats  # noqa: E402
from openai_verify import (  # noqa: E402
    format_verdict_notes,
    notes_openai_approved,
    notes_openai_dropped,
    openai_verify_enabled,
    verdict_is_keep,
    verify_still,
)
from still_store import candidate_still_path  # noqa: E402


def _ai_tag(notes: str | None) -> str:
    low = (notes or "").lower()
    if "openai:keep" in low:
        return "keep"
    if "openai:drop" in low:
        return "drop"
    if "openai:uncertain" in low:
        return "uncertain"
    return ""


def _select_rows(
    *,
    disagreements: bool,
    pending: bool,
    limit: int | None,
) -> list[dict]:
    with db() as conn:
        rows = [
            dict(r)
            for r in conn.execute(
                "SELECT id, decision, notes, image_url FROM candidates ORDER BY id ASC"
            ).fetchall()
        ]
    out: list[dict] = []
    for d in rows:
        decision = (d.get("decision") or "").strip()
        ai = _ai_tag(d.get("notes"))
        if pending and decision:
            continue
        if disagreements:
            if not (
                (decision == "reject" and ai == "keep")
                or (decision == "accept" and ai == "drop")
            ):
                continue
        out.append(d)
        if limit is not None and len(out) >= limit:
            break
    return out


def _resolve_image(row: dict) -> tuple[Path | None, str | None]:
    cid = int(row["id"])
    local = candidate_still_path(cid)
    if local.is_file() and local.stat().st_size > 200:
        return local, None
    url = (row.get("image_url") or "").strip()
    if url.startswith(("http://", "https://")):
        return None, url
    return None, None


def main() -> int:
    load_env()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--disagreements",
        action="store_true",
        help="Only rows where human Keep/Pass disagrees with openai keep/drop",
    )
    parser.add_argument(
        "--pending",
        action="store_true",
        help="Only rows with no human decision yet",
    )
    parser.add_argument("--limit", type=int, default=None, help="Max candidates")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Verify but do not write notes",
    )
    args = parser.parse_args()

    init_db()
    if not openai_verify_enabled():
        print("OpenAI verify is off or key missing — nothing to do.")
        return 1

    before = compute_label_stats()
    print(
        "BEFORE agreement_rate="
        + (
            f"{before['agreement_rate']:.1%}"
            if before.get("agreement_rate") is not None
            else "n/a"
        )
        + f" labeled={before['n_labeled']} "
        f"false_keep={before['false_keep']} false_drop={before['false_drop']}",
        flush=True,
    )

    rows = _select_rows(
        disagreements=bool(args.disagreements),
        pending=bool(args.pending),
        limit=args.limit,
    )
    print(f"Re-verifying {len(rows)} candidate(s)…", flush=True)

    kept = 0
    dropped = 0
    uncertain = 0
    skipped = 0
    flipped = 0
    for d in rows:
        cid = d["id"]
        path, url = _resolve_image(d)
        if not path and not url:
            skipped += 1
            print(f"#{cid} skip — no still", flush=True)
            continue
        print(f"#{cid} verifying…", flush=True)
        verdict = verify_still(
            image_path=path if path else None,
            image_url=url if url else None,
        )
        if verdict.get("skipped") or verdict.get("error") in (
            "no_image",
            "image_fetch_failed",
        ):
            skipped += 1
            print(
                f"  SKIP ({verdict.get('reason') or verdict.get('error')}) "
                f"— leaving prior notes",
                flush=True,
            )
            continue
        note = format_verdict_notes(verdict)
        was_keep = notes_openai_approved(d.get("notes"))
        was_drop = notes_openai_dropped(d.get("notes"))
        ok = verdict_is_keep(verdict)
        is_unc = bool(verdict.get("uncertain"))
        if not args.dry_run:
            with db(write=True) as conn:
                conn.execute("UPDATE candidates SET notes=? WHERE id=?", (note, cid))
        if is_unc:
            uncertain += 1
            print(f"  UNCERTAIN {note[:120]}", flush=True)
        elif ok:
            kept += 1
            if was_drop:
                flipped += 1
            print(f"  KEEP {note[:120]}", flush=True)
        else:
            dropped += 1
            if was_keep:
                flipped += 1
            print(f"  DROP (was_keep={was_keep}) {note[:120]}", flush=True)

    after = compute_label_stats() if not args.dry_run else before
    print(
        f"Done. keep={kept} drop={dropped} uncertain={uncertain} "
        f"skipped={skipped} flipped={flipped} dry_run={args.dry_run}",
        flush=True,
    )
    print(
        "AFTER agreement_rate="
        + (
            f"{after['agreement_rate']:.1%}"
            if after.get("agreement_rate") is not None
            else "n/a"
        )
        + f" labeled={after['n_labeled']} "
        f"false_keep={after['false_keep']} false_drop={after['false_drop']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
