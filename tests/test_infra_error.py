"""Infra vs permanent YouTube failure classification."""

from runpod_client import is_infra_error, is_permanent_youtube_skip


def test_pod_not_ready_is_infra_not_permanent():
    msg = "GPU pod not ready — scrape should call ensure_pod first"
    assert is_infra_error(msg)
    assert not is_permanent_youtube_skip(msg)


def test_http_404_proxy_is_infra():
    assert is_infra_error("http_404")
    assert is_infra_error("http_404 all GPU proxies dead — need ensure_pods")


def test_members_only_is_permanent_not_infra():
    msg = "Join this channel to get access to members-only content"
    assert is_permanent_youtube_skip(msg)
    assert not is_infra_error(msg)


def test_private_is_permanent():
    assert is_permanent_youtube_skip("ERROR: Private video")
    assert not is_infra_error("ERROR: Private video")
