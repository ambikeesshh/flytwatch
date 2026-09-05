"""Stage 1 — always-on cheap anomaly scorer.

CLIP ViT-B/32 scores each sampled frame against per-class prompt ensembles.
Runs every frame of the feed; cost target is a few ms/frame on GPU and
well under 100ms on CPU. High recall is the goal — stage 2 filters.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from transformers import CLIPModel, CLIPProcessor

from src.config import (
    ANOMALY_LABELS,
    STAGE1_EMA_ALPHA,
    STAGE1_MODEL,
    STAGE1_TRIGGER_CONSEC,
    STAGE1_TRIGGER_THRESHOLD,
)
from src.prompts import CLASS_PROMPTS, NORMAL_PROMPTS


def _pooled(out):
    """transformers 5.x returns BaseModelOutputWithPooling from get_*_features;
    older versions return the projected tensor directly."""
    return out.pooler_output if hasattr(out, "pooler_output") else out


@dataclass
class Stage1Output:
    t: float                       # timestamp of the scored sample
    scores: dict[str, float]       # per-class anomaly probability (post-softmax vs normal)
    top_class: str
    top_score: float
    smoothed_top: float            # EMA-smoothed top anomaly score
    trigger: bool                  # armed a stage-2 verification this sample


class Stage1CLIP:
    def __init__(self, model_name: str = STAGE1_MODEL, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name).to(self.device).eval()
        self.labels = list(ANOMALY_LABELS)
        self._build_text_features()
        self._ema: dict[str, float] = {}
        self._consec = 0

    def _build_text_features(self) -> None:
        texts = NORMAL_PROMPTS + [p for c in self.labels for p in CLASS_PROMPTS[c]]
        with torch.no_grad():
            enc = self.processor(text=texts, return_tensors="pt", padding=True, truncation=True).to(self.device)
            feats = _pooled(self.model.get_text_features(**enc))
            feats = feats / feats.norm(dim=-1, keepdim=True)
        n_t = len(NORMAL_PROMPTS)  # prompts per class
        self.normal_feat = feats[:n_t].mean(0)
        self.normal_feat /= self.normal_feat.norm()
        self.class_feats = []
        for i, _ in enumerate(self.labels):
            block = feats[n_t + i * n_t: n_t + (i + 1) * n_t].mean(0)
            block = block / block.norm()
            self.class_feats.append(block)
        self.class_feats = torch.stack(self.class_feats)  # [C, D]

    @torch.no_grad()
    def score_frame(self, frame_rgb: np.ndarray, t: float, logit_scale: float = 100.0) -> Stage1Output:
        enc = self.processor(images=frame_rgb, return_tensors="pt").to(self.device)
        img = _pooled(self.model.get_image_features(**enc))
        img = img / img.norm(dim=-1, keepdim=True)

        sims_normal = (img @ self.normal_feat).flatten()           # [1]
        sims_classes = (img @ self.class_feats.T).flatten()        # [C]
        logits = torch.cat([sims_normal, sims_classes]) * logit_scale
        probs = torch.softmax(logits, dim=0).cpu().numpy()
        class_probs = probs[1:]                                    # normal takes index 0

        best = int(class_probs.argmax())
        top_class = self.labels[best]
        top_score = float(class_probs[best])
        prev = self._ema.get(top_class, top_score)
        smoothed = STAGE1_EMA_ALPHA * top_score + (1 - STAGE1_EMA_ALPHA) * prev
        self._ema[top_class] = smoothed

        self._consec = self._consec + 1 if smoothed >= STAGE1_TRIGGER_THRESHOLD else 0
        return Stage1Output(
            t=t,
            scores={c: float(p) for c, p in zip(self.labels, class_probs)},
            top_class=top_class,
            top_score=top_score,
            smoothed_top=smoothed,
            trigger=self._consec >= STAGE1_TRIGGER_CONSEC,
        )
