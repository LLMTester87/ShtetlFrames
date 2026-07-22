"""Unit tests for timeline strip helpers (no ffmpeg / download)."""

from PIL import Image

from frame_strip import (
    CROP_MAX_BYTES,
    _fmt_ts,
    _label_frame,
    _save_jpeg_under,
    _source_label,
    hit_mid_sec,
    sample_times,
)


def test_hit_mid():
    assert hit_mid_sec(10, 14) == 12.0
    assert hit_mid_sec(14, 10) == 12.0


def test_sample_times_span():
    times = sample_times(20.0, pad=10.0, step=0.5)
    assert times[0] == 10.0
    assert times[-1] == 30.0
    assert abs(times[1] - times[0] - 0.5) < 1e-9
    # 10..30 inclusive every 0.5 → 41 samples
    assert len(times) == 41


def test_fmt_ts():
    assert _fmt_ts(65) == "1:05"
    assert _fmt_ts(3661) == "1:01:01"


def test_source_label_youtube():
    s = _source_label("my_video_slug", "https://www.youtube.com/watch?v=abc")
    assert "YouTube" in s
    assert "my video slug" in s or "my_video_slug" in s.replace(" ", "_")


def test_label_caption_below_does_not_shrink_frame():
    img = Image.new("RGB", (160, 120), (40, 80, 120))
    out = _label_frame(img, source="YouTube · demo", timestamp="1:05", caption_below=True)
    assert out.width == 160
    assert out.height == 120 + 36
    # Top of canvas should still be the original frame color (not black bar).
    assert out.getpixel((80, 60)) == (40, 80, 120)
    # Caption bar lives under the frame.
    assert out.getpixel((80, 130)) == (0, 0, 0)


def test_save_jpeg_under_respects_crop_cap(tmp_path):
    # Wide strip similar to a short crop stitch.
    img = Image.new("RGB", (1800, 316), (90, 70, 50))
    dest = tmp_path / "crop.jpg"
    path = _save_jpeg_under(img, dest, max_bytes=CROP_MAX_BYTES, quality=92)
    assert path is not None
    assert path.is_file()
    assert 500 < path.stat().st_size <= CROP_MAX_BYTES
