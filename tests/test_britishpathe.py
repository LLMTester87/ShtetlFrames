"""Unit tests for British Pathé helpers (no network)."""

import urllib.parse

from britishpathe import (
    asset_id_from_url,
    asset_page_url,
    extract_m3u8,
    extract_og_title,
    is_britishpathe_asset_url,
    is_britishpathe_title,
    is_britishpathe_url,
    load_local_catalog,
    normalize_asset_url,
    parse_assets_from_html,
    prepare_pathe_job,
    search_url,
)
from crawl.common import is_crawlable
from queue_manage import guess_downloadable, guess_source


SAMPLE_HTML = """
<html><head>
<meta property="og:title" content="PRINCESS MARGARET'S WEDDING (1960)" />
</head><body>
<a href="https://www.britishpathe.com/asset/197462" class="card-v3__link" aria-label="PRINCESS MARGARET WEDDING" title="12.3">
<video></video></a>
<a href="/asset/36553/">TUG MASTER</a>
<script>var x={"url":"https://www.britishpathe.com/fe-cdn/britishpathe/video/659/197462/abc/2/def/playlist.m3u8","asset_id":197462};</script>
</body></html>
"""


def test_asset_url_helpers():
    assert is_britishpathe_url("https://www.britishpathe.com/asset/197462/")
    assert is_britishpathe_asset_url("https://www.britishpathe.com/asset/197462/")
    assert asset_id_from_url("https://www.britishpathe.com/asset/197462/") == "197462"
    assert asset_page_url(197462).endswith("/asset/197462/")
    assert normalize_asset_url("https://britishpathe.com/asset/197462") == (
        "https://www.britishpathe.com/asset/197462/"
    )
    assert normalize_asset_url("https://www.britishpathe.com/asset/197462/") == (
        "https://www.britishpathe.com/asset/197462/"
    )


def test_extract_m3u8_prefers_fe_cdn():
    m3u8 = extract_m3u8(SAMPLE_HTML)
    assert m3u8 is not None
    assert "fe-cdn" in m3u8
    assert m3u8.endswith("playlist.m3u8")


def test_extract_og_title():
    assert "PRINCESS MARGARET" in extract_og_title(SAMPLE_HTML)


def test_parse_assets_from_html():
    entries = parse_assets_from_html(SAMPLE_HTML, year="1960")
    assert any(e["identifier"] == "197462" for e in entries)
    hit = next(e for e in entries if e["identifier"] == "197462")
    assert "Margaret" in hit["title"] or "MARGARET" in hit["title"]
    assert hit["year"] == "1960"
    assert hit["downloadable"] == "yes"
    tug = next(e for e in entries if e["identifier"] == "36553")
    assert "TUG" in tug["title"].upper()


def test_search_url_builds():
    all_u = search_url("", page=1)
    assert "searchQuery=" in all_u
    assert "page=null" in all_u
    assert "refined[]=" in all_u
    assert "selection=" in all_u
    u = search_url("wedding", page=2)
    assert "wedding" in u
    assert "page=2" in u


def test_prepare_pathe_job_ignores_youtube():
    # Dedicated Pathé page only — YouTube titles are not auto-mapped here.
    assert (
        prepare_pathe_job(
            "https://www.youtube.com/watch?v=abc",
            "Clip | British Pathe",
        )
        is None
    )


def test_pathe_youtube_title_helper():
    assert is_britishpathe_title(
        "Smuggling Illegal Migrants Into Australia (1959) | Unissued N'13 | British Pathe"
    )
    assert not is_britishpathe_title("Random travel vlog")


def test_load_local_catalog_reads_jsonl(tmp_path, monkeypatch):
    import britishpathe as bp

    cat = tmp_path / "pathe_catalog.jsonl"
    cat.write_text(
        "\n".join(
            [
                '{"url":"https://www.britishpathe.com/asset/111/","title":"ONE","identifier":"111","source":"British Pathé","downloadable":"yes"}',
                '{"url":"https://www.britishpathe.com/asset/111/","title":"ONE DUPE","identifier":"111","source":"British Pathé","downloadable":"yes"}',
                '{"url":"https://www.britishpathe.com/asset/222/","title":"TWO","identifier":"222","source":"British Pathé","downloadable":"yes"}',
                "not-json",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(bp, "_CATALOG_PATH", cat)
    rows = load_local_catalog(max_items=10)
    assert len(rows) == 2
    assert {r["identifier"] for r in rows} == {"111", "222"}
    assert rows[0]["url"].endswith("/asset/111/")


def test_queue_guess_pathe_asset():
    url = "https://www.britishpathe.com/asset/197462/"
    assert guess_downloadable(url) == "yes"
    assert guess_source(url) == "British Pathé (user)"
    assert is_crawlable(url) is False
