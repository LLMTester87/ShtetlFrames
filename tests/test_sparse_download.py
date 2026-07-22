"""Unit tests for sparse / dense section helpers."""

from sparse_download import (
    build_dense_windows,
    build_sparse_windows,
    format_section,
    remap_segments_to_source,
    remap_time,
    should_skip_sparse,
)


def test_format_section():
    assert format_section(0, 20).startswith("*0.00-20.00")


def test_sparse_windows_stride():
    wins = build_sparse_windows(180, section_sec=20, stride_sec=60)
    assert wins[0] == (0.0, 20.0)
    assert (60.0, 80.0) in wins
    assert wins[-1][1] == 180.0
    assert wins[-1][0] == 160.0  # tail window


def test_should_skip_short():
    assert should_skip_sparse(30) is True
    assert should_skip_sparse(None) is True
    assert should_skip_sparse(600) is False


def test_remap_concatenated():
    windows = [(0.0, 20.0), (60.0, 80.0)]
    assert abs(remap_time(5.0, windows) - 5.0) < 0.01
    assert abs(remap_time(25.0, windows) - 65.0) < 0.01
    segs = remap_segments_to_source(
        [{"start_sec": 25.0, "end_sec": 28.0, "peak_score": 0.1}],
        windows,
    )
    assert abs(segs[0]["start_sec"] - 65.0) < 0.02


def test_dense_merge_and_cap():
    wins = build_dense_windows([100.0, 110.0, 400.0], 1000.0, pad_sec=30, max_total_sec=120)
    assert wins
    total = sum(e - s for s, e in wins)
    assert total <= 120.0 + 1.0
