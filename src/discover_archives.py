"""Relevance-gated discovery across IA + Commons + known multi-video seed packs.

Writes output/bulk_queue.csv ranked by relevance (not download popularity).
Does not auto-download large batches — review the queue first.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import requests

from config import BULK_QUEUE_CSV, OUTPUT_DIR

USER_AGENT = "ShtetlFramesDiscover/1.0 (research; respectful archival use)"

RELEVANCE_TERMS = [
    ("hasid", 5),
    ("hassid", 5),
    ("chasid", 5),
    ("agudah", 5),
    ("agudas", 5),
    ("agudath", 5),
    ("munkacs", 5),
    ("munkacz", 5),
    ("mukachevo", 4),
    ("rebbe", 4),
    ("shtetl", 4),
    ("payot", 4),
    ("shtreimel", 5),
    ("yiddish", 3),
    ("orthodox", 3),
    ("synagogue", 2),
    ("jewish", 2),
    ("jews", 2),
    ("jew ", 2),
    ("galicia", 3),
    ("poland", 2),
    ("warsaw", 3),
    ("krakow", 3),
    ("cracow", 3),
    ("lwow", 3),
    ("lvov", 3),
    ("lviv", 2),
    ("vilna", 3),
    ("vilnius", 2),
    ("bialystok", 3),
    ("kolbuszowa", 4),
    ("carpathian", 3),
    ("palestine", 2),
    ("jerusalem", 2),
    ("meah shearim", 4),
    ("cheder", 3),
    ("rabbi", 2),
    ("landsmanshaft", 3),
]

NEGATIVE_TERMS = [
    ("banana", 8),
    ("chaplin", 8),
    ("betty boop", 8),
    ("baseball", 5),
    ("olympics", 4),
    ("bomber", 4),
    ("b-29", 5),
    ("b-17", 5),
    ("cartoon", 6),
    ("animation", 4),
    ("western", 3),
    ("gangster", 3),
    ("scarface", 8),
    ("birth of a nation", 10),
]

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
    "jude satt",
    "concentration camp",
    "extermination",
    "death mills",
    "nazi concentration",
)

IA_QUERIES = [
    'title:(Jewish OR Jews OR Yiddish OR Hasidic OR Hasidim OR Agudah OR Munkacs OR Munkacz) AND mediatype:movies AND year:[1900 TO 1950]',
    'title:(Poland OR Warsaw OR Krakow OR Cracow OR Galicia OR Vilna OR Bialystok) AND (Jewish OR Jews OR Yiddish) AND mediatype:movies AND year:[1900 TO 1950]',
    'description:(Hasidic OR Hasidim OR Agudah OR "Orthodox Jew" OR shtetl OR rebbe) AND mediatype:movies AND year:[1900 TO 1950]',
    'title:(synagogue OR cheder OR rabbi) AND (Jewish OR Jews) AND mediatype:movies AND year:[1900 TO 1950]',
    '"displaced persons" AND (Jewish OR Jews) AND mediatype:movies AND year:[1945 TO 1950]',
    'creator:(PeriscopeFilm) AND (Jewish OR Israel OR Palestine) AND mediatype:movies AND year:[1900 TO 1950]',
]

# Known multi-video seed packs (direct URLs where possible)
SEED_PACKS = [
    {
        "source": "Wikimedia Commons",
        "url": "https://upload.wikimedia.org/wikipedia/commons/f/ff/Historic_Chofetz_Chaim_Video_Almost_Unnoticed_For_Over_A_Decade_2.webm",
        "title": "World Congress of Agudas Yisroel - Vienna 1923 (Fox outtakes)",
        "year": "1923",
        "tier": "A-seed",
        "relevance": 100,
        "notes": "Proven high-density Orthodox leadership; USC MIRC provenance",
        "downloadable": "yes",
    },
    {
        "source": "YouTube",
        "url": "https://www.youtube.com/watch?v=rp1OeIf0D0w",
        "title": "Jewish Life in Munkatch - March 1933 (complete)",
        "year": "1933",
        "tier": "A-seed",
        "relevance": 100,
        "notes": "Munkacs wedding crowds + Minchas Elazar; NARA lineage",
        "downloadable": "yes",
    },
    {
        "source": "YouTube",
        "url": "https://www.youtube.com/watch?v=hdf6-qnr11s",
        "title": "Spielberg Archive - Jewish Life in Krakow 1939 (Five Cities)",
        "year": "1939",
        "tier": "A-seed",
        "relevance": 95,
        "notes": "Goskind Five Cities; HUJI/Spielberg YouTube mirror",
        "downloadable": "yes",
    },
    {
        "source": "YouTube",
        "url": "https://www.youtube.com/results?search_query=Spielberg+Jewish+Film+Archive+Jewish+Life+in+Lwow",
        "title": "Five Cities - Jewish Life in Lwow 1939 (search)",
        "year": "1939",
        "tier": "A-seed",
        "relevance": 90,
        "notes": "Resolve to concrete watch URL before download",
        "downloadable": "search",
    },
    {
        "source": "YouTube",
        "url": "https://www.youtube.com/results?search_query=Spielberg+Jewish+Film+Archive+Jewish+Life+in+Warsaw+1939",
        "title": "Five Cities - A Day in Warsaw 1939 (search)",
        "year": "1939",
        "tier": "A-seed",
        "relevance": 90,
        "notes": "Resolve to concrete watch URL before download",
        "downloadable": "search",
    },
    {
        "source": "YouTube",
        "url": "https://www.youtube.com/results?search_query=Spielberg+Jewish+Film+Archive+Jewish+Life+in+Vilna+1939",
        "title": "Five Cities - Jewish Life in Vilna 1939 (search)",
        "year": "1939",
        "tier": "A-seed",
        "relevance": 90,
        "notes": "Resolve to concrete watch URL before download",
        "downloadable": "search",
    },
    {
        "source": "YouTube",
        "url": "https://www.youtube.com/results?search_query=Spielberg+Jewish+Film+Archive+Jewish+Life+in+Bialystok+1939",
        "title": "Five Cities - Jewish Life in Bialystok 1939 (search)",
        "year": "1939",
        "tier": "A-seed",
        "relevance": 90,
        "notes": "Resolve to concrete watch URL before download",
        "downloadable": "search",
    },
    {
        "source": "USHMM catalog",
        "url": "https://resources.ushmm.org/film",
        "title": "USHMM Film & Video Archive (~3500+ streamed clips)",
        "year": "1920-1950",
        "tier": "A-catalog",
        "relevance": 85,
        "notes": "Browse keywords: Jewish life before, Munkacs, Orthodox, Poland",
        "downloadable": "stream",
    },
    {
        "source": "Spielberg / JFC",
        "url": "https://jfc.org.il/en/compilation/the-steven-spielberg-jewish-film-archive-collection/",
        "title": "Spielberg Jewish Film Archive (~20k catalog / 600-2000 online)",
        "year": "1911-1950",
        "tier": "A-catalog",
        "relevance": 85,
        "notes": "Largest Jewish doc vault; stream online titles",
        "downloadable": "stream",
    },
    {
        "source": "USC MIRC",
        "url": "https://digital.library.sc.edu/collections/fox-movietone-news-the-war-years/",
        "title": "Fox Movietone News (USC MIRC) ~23k titles / 8k+ online",
        "year": "1919-1944",
        "tier": "A-catalog",
        "relevance": 80,
        "notes": "Search Agudah, Orthodox Jew, Vienna, rabbi",
        "downloadable": "stream",
    },
    {
        "source": "YIVO",
        "url": "https://polishjews.yivo.org/videos",
        "title": "YIVO digitized Poland home movies (~75)",
        "year": "1920-1939",
        "tier": "B",
        "relevance": 92,
        "notes": "Highest shtetl density; limited public bulk",
        "downloadable": "stream",
    },
]


def _as_text(val) -> str:
    if val is None:
        return ""
    if isinstance(val, list):
        return " ".join(str(x) for x in val)
    return str(val)


def relevance_score(title: str, description: str = "", identifier: str = "") -> int:
    blob = f"{title} {description} {identifier}".lower()
    if any(k in blob for k in BLOCK_KEYWORDS):
        return -1000
    score = 0
    for term, w in RELEVANCE_TERMS:
        if term in blob:
            score += w
    for term, w in NEGATIVE_TERMS:
        if term in blob:
            score -= w
    # Must have at least one Jewish-world anchor
    anchors = (
        "jewish",
        "jew",
        "jews",
        "yiddish",
        "hasid",
        "hassid",
        "chasid",
        "agudah",
        "synagogue",
        "rabbi",
        "israel",
        "palestine",
        "hebrew",
    )
    if not any(a in blob for a in anchors):
        score -= 20
    return score


def year_ok(year_val, lo: int = 1900, hi: int = 1955) -> bool:
    try:
        y = int(_as_text(year_val)[:4])
        return lo <= y <= hi
    except (TypeError, ValueError):
        return True  # unknown year: keep, score will decide


def search_ia(query: str, rows: int = 50, page: int = 1) -> list[dict]:
    r = requests.get(
        "https://archive.org/advancedsearch.php",
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
            ("sort[]", "week desc"),
        ],
        timeout=90,
        headers={"User-Agent": USER_AGENT},
    )
    r.raise_for_status()
    return r.json().get("response", {}).get("docs", [])


def discover_ia(rows: int = 50) -> list[dict]:
    seen = set()
    out = []
    for q in IA_QUERIES:
        print(f"IA: {q[:90]}...")
        try:
            docs = search_ia(q, rows=rows)
        except Exception as e:
            print(f"  failed: {e}")
            continue
        for d in docs:
            ident = d.get("identifier")
            if not ident or ident in seen:
                continue
            seen.add(ident)
            title = _as_text(d.get("title"))
            desc = _as_text(d.get("description"))[:500]
            year = _as_text(d.get("year"))
            if not year_ok(year):
                continue
            score = relevance_score(title, desc, ident)
            if score < 4:
                continue
            blocked = score <= -100 or any(
                k in f"{title} {desc} {ident}".lower() for k in BLOCK_KEYWORDS
            )
            out.append(
                {
                    "source": "Internet Archive",
                    "url": f"https://archive.org/details/{ident}",
                    "title": title[:200],
                    "year": year[:10],
                    "tier": "A-ia",
                    "relevance": score,
                    "notes": desc[:180].replace("\n", " "),
                    "downloadable": "no" if blocked else "yes",
                    "identifier": ident,
                    "licenseurl": _as_text(d.get("licenseurl")),
                    "blocked": str(blocked).lower(),
                }
            )
        print(f"  unique kept so far: {len(out)}")
    return out


def discover_commons(limit: int = 40) -> list[dict]:
    """MediaWiki API search for video files related to Jewish / Agudah / Poland."""
    out = []
    queries = [
        "filetype:video Jewish Poland",
        "filetype:video Agudah OR Agudas",
        "filetype:video Hasidic OR Hasidim",
        "filetype:video Yiddish newsreel",
        "filetype:video Munkacs OR Munkacz",
    ]
    api = "https://commons.wikimedia.org/w/api.php"
    seen = set()
    for q in queries:
        print(f"Commons: {q}")
        try:
            r = requests.get(
                api,
                params={
                    "action": "query",
                    "list": "search",
                    "srsearch": q,
                    "srnamespace": 6,  # File
                    "srlimit": min(20, limit),
                    "format": "json",
                },
                timeout=60,
                headers={"User-Agent": USER_AGENT},
            )
            r.raise_for_status()
            hits = r.json().get("query", {}).get("search", [])
        except Exception as e:
            print(f"  failed: {e}")
            continue
        for h in hits:
            title = h.get("title", "")
            if not title.lower().startswith("file:"):
                continue
            # Prefer video extensions in title
            low = title.lower()
            if not any(ext in low for ext in (".webm", ".ogv", ".mp4", ".avi", ".mkv")):
                # still may be video without ext in title — keep if snippet relevant
                pass
            key = title.lower()
            if key in seen:
                continue
            seen.add(key)
            snippet = re.sub(r"<[^>]+>", "", h.get("snippet", ""))
            score = relevance_score(title, snippet)
            if score < 3:
                continue
            file_url = f"https://commons.wikimedia.org/wiki/{title.replace(' ', '_')}"
            out.append(
                {
                    "source": "Wikimedia Commons",
                    "url": file_url,
                    "title": title.replace("File:", "", 1)[:200],
                    "year": "",
                    "tier": "A-commons",
                    "relevance": score,
                    "notes": snippet[:180],
                    "downloadable": "yes",
                    "identifier": title,
                    "licenseurl": "",
                    "blocked": "false",
                }
            )
    return out[:limit]


# Curated IA identifiers known/likely relevant (used when live search is flaky)
CURATED_IA = [
    ("Yiddish-film-scenes", "Yiddish film scenes (Dybbuk / Tevye excerpts)", "1937", 12, "Orthodox dress in Yiddish feature excerpts"),
    ("chicago-tribune-judaism-exposition-1933-newsreel-footage", "Chicago Tribune Judaism Exposition 1933", "1933", 8, "Public Jewish event newsreel"),
    ("TheGolem_893", "The Golem (1920)", "1920", 6, "Jewish period dress — secondary yield"),
    ("UptownNewYork", "Uptown New York (1932)", "1932", 5, "NYC Jewish family drama — limited traditional dress"),
    ("56124-the-eternal-light", "Eternal Light: The Remnant (1969)", "1969", -5, "Post-1950 — keep listed but year-gated low"),
]


def curated_ia_rows() -> list[dict]:
    out = []
    for ident, title, year, bonus, notes in CURATED_IA:
        score = relevance_score(title, notes, ident) + bonus
        if not year_ok(year, lo=1900, hi=1955) and int(year[:4]) > 1955:
            # still list as catalog pointer with low score / not download priority
            downloadable = "no"
        else:
            downloadable = "yes" if score >= 4 else "no"
        out.append(
            {
                "source": "Internet Archive (curated)",
                "url": f"https://archive.org/details/{ident}",
                "title": title,
                "year": year,
                "tier": "A-ia-curated",
                "relevance": score,
                "notes": notes,
                "downloadable": downloadable,
                "identifier": ident,
                "licenseurl": "",
                "blocked": "false",
            }
        )
    return out


def rescore_prior_discoveries() -> list[dict]:
    """Re-rank previous ia_batch_discoveries.csv with relevance scoring."""
    prior = OUTPUT_DIR / "ia_batch_discoveries.csv"
    if not prior.exists():
        return []
    out = []
    with prior.open(encoding="utf-8") as f:
        for d in csv.DictReader(f):
            title = d.get("title") or ""
            desc = d.get("description_snip") or ""
            ident = d.get("identifier") or ""
            year = d.get("year") or ""
            if year and not year_ok(year):
                continue
            score = relevance_score(title, desc, ident)
            if score < 4:
                continue
            blocked = (d.get("blocked_from_autodownload") or "").lower() == "true" or score <= -100
            out.append(
                {
                    "source": "Internet Archive (rescored)",
                    "url": d.get("url") or f"https://archive.org/details/{ident}",
                    "title": title[:200],
                    "year": year,
                    "tier": "A-ia",
                    "relevance": score,
                    "notes": desc[:180],
                    "downloadable": "no" if blocked else "yes",
                    "identifier": ident,
                    "licenseurl": d.get("licenseurl") or "",
                    "blocked": str(blocked).lower(),
                }
            )
    out.sort(key=lambda x: x["relevance"], reverse=True)
    return out


def write_queue(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "source",
        "url",
        "title",
        "year",
        "tier",
        "relevance",
        "downloadable",
        "notes",
        "identifier",
        "licenseurl",
        "blocked",
    ]
    # Deduplicate by URL
    seen = set()
    deduped = []
    for r in sorted(rows, key=lambda x: (-int(x.get("relevance") or 0), x.get("title") or "")):
        u = r.get("url") or ""
        if u in seen:
            continue
        seen.add(u)
        # normalize missing keys
        row = {f: r.get(f, "") for f in fields}
        deduped.append(row)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(deduped)
    print(f"Wrote {path} ({len(deduped)} rows)")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build relevance-gated bulk_queue.csv")
    ap.add_argument("--rows", type=int, default=40, help="IA rows per query")
    ap.add_argument("--ia-top", type=int, default=40, help="Keep top N IA by relevance")
    ap.add_argument("--commons-limit", type=int, default=30)
    ap.add_argument("--skip-ia", action="store_true")
    ap.add_argument("--skip-commons", action="store_true")
    args = ap.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    # Seed packs first
    for s in SEED_PACKS:
        rows.append(
            {
                **s,
                "identifier": "",
                "licenseurl": "",
                "blocked": "false",
            }
        )

    if not args.skip_ia:
        ia = discover_ia(rows=args.rows)
        ia.sort(key=lambda x: x["relevance"], reverse=True)
        rows.extend(ia[: args.ia_top])

    # Always rescore prior discoveries + curated IDs (works offline when IA search is down)
    rows.extend(rescore_prior_discoveries()[: args.ia_top])
    rows.extend(curated_ia_rows())

    if not args.skip_commons:
        rows.extend(discover_commons(limit=args.commons_limit))

    write_queue(rows, BULK_QUEUE_CSV)
    # Also JSON for the web UI
    payload = list(csv.DictReader(BULK_QUEUE_CSV.open(encoding="utf-8")))
    (OUTPUT_DIR / "bulk_queue.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    summary = {
        "n_queue": len(payload),
        "n_downloadable": sum(1 for r in payload if r.get("downloadable") == "yes"),
        "n_stream": sum(1 for r in payload if r.get("downloadable") == "stream"),
        "n_seed": sum(1 for r in payload if str(r.get("tier", "")).startswith("A-seed")),
        "top_10": payload[:10],
    }
    (OUTPUT_DIR / "bulk_queue_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2)[:800])


if __name__ == "__main__":
    main()
