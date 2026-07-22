"""Temporal aggregation ranking / gap clustering."""

from shtetl_core.scoring import FrameHit
from shtetl_core.segments import aggregate_segments, aggregate_segments_dicts


def _hit(t: float, score: float, video_id: str = "v1") -> FrameHit:
    return FrameHit(
        video_id=video_id,
        time_sec=t,
        frame_idx=int(t * 10),
        score=score,
        pos_score=0.3,
        neg_score=0.1,
        best_cue="test cue",
        bbox=[0.0, 0.0, 100.0, 200.0],
        crop_path=None,
    )


def test_nearby_hits_merge_into_one_segment():
    hits = [_hit(10.0, 0.12), _hit(11.0, 0.15), _hit(12.0, 0.11)]
    segs = aggregate_segments(hits, source_path="/tmp/v.mp4", min_seg=3.0, max_gap=2.5)
    assert len(segs) == 1
    assert segs[0].hit_count == 3
    assert segs[0].peak_score == 0.15


def test_gap_splits_segments():
    hits = [_hit(10.0, 0.12), _hit(20.0, 0.14)]  # 10s gap > 2.5
    segs = aggregate_segments(hits, source_path="/tmp/v.mp4", min_seg=3.0, max_gap=2.5)
    assert len(segs) == 2
    assert segs[0].peak_score >= segs[1].peak_score  # sorted by rank


def test_dicts_keep_hits_for_worker():
    hits = [_hit(5.0, 0.13)]
    rows = aggregate_segments_dicts(hits, video_id="abc")
    assert len(rows) == 1
    assert rows[0]["video_id"] == "abc"
    assert len(rows[0]["_hits"]) == 1
