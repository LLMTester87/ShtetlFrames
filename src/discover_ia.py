"""Discover Internet Archive movie items for batch scanning.

Prefer `discover_archives.py` for relevance-gated bulk_queue.csv.
This module remains for quick IA-only sweeps.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import requests

from config import BATCH_IA_CSV, OUTPUT_DIR

USER_AGENT = "ShtetlFrames/1.0 (research)"

QUERIES = [
    '(Jewish OR Hasidic OR Hassidic OR Agudah OR Agudas OR Munkacs OR Munkacz OR Yiddish) AND mediatype:movies AND year:[1900 TO 1950]',
    'Jews AND (Poland OR Krakow OR Warsaw OR Galicia OR Vilna OR Lwow) AND mediatype:movies AND year:[1900 TO 1950]',
    '(synagogue OR shtetl OR Orthodox) AND Jewish AND mediatype:movies AND year:[1900 TO 1950]',
    '"displaced persons" AND Jewish AND mediatype:movies AND year:[1945 TO 1950]',
    'Movietone AND Jewish AND mediatype:movies',
]

# Do not auto-download Nazi propaganda or camp atrocity reels for this tooling.
BLOCK_KEYWORDS = (
    "eternal jew",
    "ewige jude",
    "peril juif",
    "der ewige",
    "hitler",
    "auschwitz",
    "theresienstadt",
    "treblinka",
    "majdanek",
    "propaganda",
    "jude satt",
)


def search_ia(query: str, rows: int = 50, page: int = 1) -> list[dict]:
    url = "https://archive.org/advancedsearch.php"
    params = {
        "q": query,
        "fl[]": ["identifier", "title", "year", "description", "licenseurl", "downloads"],
        "rows": rows,
        "page": page,
        "output": "json",
        "sort[]": "downloads desc",
    }
    # requests with list fl[] needs special handling
    r = requests.get(
        url,
        params=[
            ("q", query),
            ("fl[]", "identifier"),
            ("fl[]", "title"),
            ("fl[]", "year"),
            ("fl[]", "description"),
            ("fl[]", "licenseurl"),
            ("rows", str(rows)),
            ("page", str(page)),
            ("output", "json"),
            ("sort[]", "downloads desc"),
        ],
        timeout=90,
        headers={"User-Agent": USER_AGENT},
    )
    r.raise_for_status()
    data = r.json()
    return data.get("response", {}).get("docs", [])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=40)
    ap.add_argument("--download-top", type=int, default=15, help="Also download top N unique items")
    args = ap.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    seen = set()
    rows_out = []
    for q in QUERIES:
        print(f"Query: {q[:80]}...")
        try:
            docs = search_ia(q, rows=args.rows)
        except Exception as e:
            print(f"  failed: {e}")
            continue
        for d in docs:
            ident = d.get("identifier")
            if not ident or ident in seen:
                continue
            seen.add(ident)
            title = d.get("title") or ""
            if isinstance(title, list):
                title = title[0]
            year = d.get("year")
            if isinstance(year, list):
                year = year[0]
            desc = d.get("description") or ""
            if isinstance(desc, list):
                desc = " ".join(desc)[:300]
            else:
                desc = str(desc)[:300]
            blob = f"{title} {desc} {ident}".lower()
            blocked = any(k in blob for k in BLOCK_KEYWORDS)
            rows_out.append(
                {
                    "identifier": ident,
                    "title": title,
                    "year": year or "",
                    "url": f"https://archive.org/details/{ident}",
                    "licenseurl": d.get("licenseurl") or "",
                    "query": q[:120],
                    "description_snip": desc.replace("\n", " "),
                    "blocked_from_autodownload": str(blocked).lower(),
                }
            )
        print(f"  accumulated unique: {len(rows_out)}")

    BATCH_IA_CSV.parent.mkdir(parents=True, exist_ok=True)
    with BATCH_IA_CSV.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "identifier",
                "title",
                "year",
                "url",
                "licenseurl",
                "query",
                "description_snip",
                "blocked_from_autodownload",
            ],
        )
        w.writeheader()
        w.writerows(rows_out)
    print(f"Wrote {BATCH_IA_CSV} ({len(rows_out)} items)")

    (OUTPUT_DIR / "ia_batch_discoveries.json").write_text(
        json.dumps(rows_out, indent=2), encoding="utf-8"
    )

    if args.download_top > 0:
        from download import download_archive_org
        from config import VIDEOS_DIR

        VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
        # Prefer titles that look relevant
        keywords = (
            "jewish",
            "jews",
            "yiddish",
            "hasid",
            "agud",
            "munk",
            " poland",
            "warsaw",
            "krakow",
            "galicia",
            "synagogue",
            "shtetl",
            "orthodox",
            "palestine",
            "jerusalem",
            "hebrew",
        )
        eligible = [r for r in rows_out if r.get("blocked_from_autodownload") != "true"]
        ranked = sorted(
            eligible,
            key=lambda r: sum(k in (r["title"] + " " + r["description_snip"]).lower() for k in keywords),
            reverse=True,
        )
        downloaded = []
        for item in ranked[: args.download_top]:
            print(f"Downloading IA {item['identifier']}...")
            try:
                path = download_archive_org(item["identifier"], VIDEOS_DIR)
                downloaded.append({"identifier": item["identifier"], "path": str(path) if path else None})
                print(f"  -> {path}")
            except Exception as e:
                downloaded.append({"identifier": item["identifier"], "error": str(e)})
                print(f"  -> error {e}")
        (OUTPUT_DIR / "ia_download_manifest.json").write_text(
            json.dumps(downloaded, indent=2), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
