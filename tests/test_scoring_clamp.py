"""Pure helpers for weak-match score clamping."""

from shtetl_core.scoring import (
    clamp_strong_negative,
    clamp_weak_score,
    clamp_without_headcover,
)


def test_clamp_weak_pos_forces_below_threshold():
    out = clamp_weak_score(0.15, 0.20, min_pos_score=0.28, score_threshold=0.10)
    assert out == 0.05  # threshold - 0.05


def test_strong_pos_keeps_score():
    out = clamp_weak_score(0.14, 0.32, min_pos_score=0.0, score_threshold=-0.15)
    assert out == 0.14


def test_already_low_score_unchanged_when_weak():
    out = clamp_weak_score(0.01, 0.10, min_pos_score=0.28, score_threshold=0.10)
    assert out == 0.01


def test_no_headcover_clamps_below_threshold():
    out = clamp_without_headcover(0.12, 0.20, min_headcover_score=0.25, score_threshold=0.10)
    assert out == 0.05


def test_with_headcover_keeps_score():
    out = clamp_without_headcover(0.12, 0.32, min_headcover_score=0.0, score_threshold=-0.15)
    assert out == 0.12


def test_strong_negative_ratio_clamps():
    out = clamp_strong_negative(
        0.12, 0.30, 0.25, max_neg_to_pos_ratio=0.78, score_threshold=0.10
    )
    assert out == 0.05


def test_soft_neg_ratio_keeps_score():
    out = clamp_strong_negative(
        -0.05, 0.20, 0.25, max_neg_to_pos_ratio=99.0, score_threshold=-0.15
    )
    assert out == -0.05
