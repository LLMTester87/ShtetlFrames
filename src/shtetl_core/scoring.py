"""OpenCLIP cue scorer: positive dress cues minus strong negatives."""

from __future__ import annotations

from dataclasses import dataclass

import open_clip
import torch
from PIL import Image

from shtetl_core.cues import (
    CLIP_MODEL,
    CLIP_PRETRAINED,
    DEFAULT_SCORE_THRESHOLD,
    HEADCOVER_PROMPTS,
    MAX_NEG_TO_POS_RATIO,
    MIN_HEADCOVER_SCORE,
    MIN_POS_SCORE,
    NEGATIVE_PROMPTS,
    NEG_SCORE_WEIGHT,
    POSITIVE_PROMPTS,
    TOP_K_CUES,
    TOP_K_NEGS,
)


@dataclass
class FrameHit:
    video_id: str
    time_sec: float
    frame_idx: int
    score: float
    pos_score: float
    neg_score: float
    best_cue: str
    bbox: list[float]
    crop_path: str | None = None


def clamp_weak_score(
    score: float,
    pos_score: float,
    *,
    min_pos_score: float = MIN_POS_SCORE,
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
) -> float:
    """Force weak absolute matches below the hit gate (pure helper for tests)."""
    if pos_score < min_pos_score:
        return min(score, score_threshold - 0.05)
    return score


def clamp_without_headcover(
    score: float,
    headcover_score: float,
    *,
    min_headcover_score: float = MIN_HEADCOVER_SCORE,
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
) -> float:
    """Reject sidelocks/curls-only matches that lack a visible head covering."""
    if headcover_score < min_headcover_score:
        return min(score, score_threshold - 0.05)
    return score


def clamp_strong_negative(
    score: float,
    pos_score: float,
    neg_score: float,
    *,
    max_neg_to_pos_ratio: float = MAX_NEG_TO_POS_RATIO,
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
) -> float:
    """Reject when negatives nearly match the positive (secular coat / clergy FP)."""
    if pos_score <= 1e-6:
        return min(score, score_threshold - 0.05)
    if (neg_score / pos_score) >= max_neg_to_pos_ratio:
        return min(score, score_threshold - 0.05)
    return score


class CueScorer:
    """Encode prompts once; score person crops as pos_sim - neg_sim."""

    def __init__(self, device: str | None = None) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            CLIP_MODEL, pretrained=CLIP_PRETRAINED
        )
        self.tokenizer = open_clip.get_tokenizer(CLIP_MODEL)
        self.model = self.model.to(self.device).eval()
        with torch.no_grad():
            pos_tok = self.tokenizer(POSITIVE_PROMPTS).to(self.device)
            neg_tok = self.tokenizer(NEGATIVE_PROMPTS).to(self.device)
            head_tok = self.tokenizer(HEADCOVER_PROMPTS).to(self.device)
            self.pos_feat = self.model.encode_text(pos_tok)
            self.neg_feat = self.model.encode_text(neg_tok)
            self.head_feat = self.model.encode_text(head_tok)
            self.pos_feat = self.pos_feat / self.pos_feat.norm(dim=-1, keepdim=True)
            self.neg_feat = self.neg_feat / self.neg_feat.norm(dim=-1, keepdim=True)
            self.head_feat = self.head_feat / self.head_feat.norm(dim=-1, keepdim=True)

    @torch.no_grad()
    def score_image(self, pil_img: Image.Image) -> tuple[float, float, float, str]:
        image = self.preprocess(pil_img).unsqueeze(0).to(self.device)
        img_feat = self.model.encode_image(image)
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
        pos_sims = (img_feat @ self.pos_feat.T).squeeze(0)
        neg_sims = (img_feat @ self.neg_feat.T).squeeze(0)
        head_sims = (img_feat @ self.head_feat.T).squeeze(0)
        # Best positive vs strongest negatives (stricter than mean-of-many).
        k_pos = min(max(1, TOP_K_CUES), pos_sims.numel())
        top_pos, top_idx = torch.topk(pos_sims, k_pos)
        pos_score = float(top_pos.mean().item())
        k_neg = min(max(1, TOP_K_NEGS), neg_sims.numel())
        neg_score = float(torch.topk(neg_sims, k_neg).values.mean().item())
        headcover_score = float(head_sims.max().item())
        # Soften negative pull so OpenAI sees more borderline crops.
        score = pos_score - float(NEG_SCORE_WEIGHT) * neg_score
        best_cue = POSITIVE_PROMPTS[int(top_idx[0].item())]
        score = clamp_weak_score(score, pos_score)
        score = clamp_without_headcover(score, headcover_score)
        score = clamp_strong_negative(score, pos_score, neg_score)
        return score, pos_score, neg_score, best_cue
