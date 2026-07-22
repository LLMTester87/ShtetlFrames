"""Quick look at recent candidates (no network)."""
from db import db, init_db
from openai_verify import notes_openai_approved, openai_verify_enabled

init_db()
print("verify_enabled", openai_verify_enabled())
with db() as c:
    total = c.execute("SELECT count(*) AS n FROM candidates").fetchone()["n"]
    keep = c.execute(
        "SELECT count(*) AS n FROM candidates WHERE lower(coalesce(notes,'')) LIKE 'openai:keep%'"
    ).fetchone()["n"]
    print("total", total, "openai_keep_prefix", keep)
    rows = c.execute(
        "SELECT id, best_cue, peak_score, notes FROM candidates ORDER BY id DESC LIMIT 40"
    ).fetchall()
for r in rows:
    d = dict(r)
    n = (d.get("notes") or "").replace("\n", " ")[:110]
    print(
        d["id"],
        "ok",
        notes_openai_approved(d.get("notes")),
        "peak",
        round(float(d["peak_score"] or 0), 3),
        "cue",
        (d.get("best_cue") or "")[:40],
        "|",
        n,
    )
