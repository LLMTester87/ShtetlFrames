"""Hub vs single-video URL classification."""

from crawl import is_crawlable


def test_youtube_watch_not_crawlable():
    assert is_crawlable("https://www.youtube.com/watch?v=tdkNbcpCTc0") is False


def test_youtube_channel_crawlable():
    assert is_crawlable("https://www.youtube.com/@britishpathe/videos") is True


def test_youtube_playlist_crawlable():
    assert is_crawlable("https://www.youtube.com/playlist?list=PLxxxxxx") is True


def test_direct_file_not_crawlable():
    assert is_crawlable("https://upload.wikimedia.org/wikipedia/commons/a/a0/foo.webm") is False


def test_archive_search_crawlable():
    assert is_crawlable("https://archive.org/search?query=jewish") is True


def test_non_http_not_crawlable():
    assert is_crawlable("ftp://example.com/x") is False
