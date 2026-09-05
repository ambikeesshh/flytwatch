"""Stage 2 — small VLM verifier (triggered only by stage 1).

Qwen2.5-VL-3B looks at a short frame window around a trigger and returns a
structured verdict. Because it runs only when stage 1 arms it, its per-call
latency does not threaten the real-time budget.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

import numpy as np
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from src.config import STAGE2_MAX_NEW_TOKENS, STAGE2_MODEL
from src.prompts import STAGE2_SYSTEM, STAGE2_USER


@dataclass
class Stage2Output:
    is_anomaly: bool
    class_name: str
    confidence: float
    description: str
    latency_ms: float


class Stage2VLM:
    def __init__(self, model_name: str = STAGE2_MODEL, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = (
            Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_name, torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32
            )
            .to(self.device)
            .eval()
        )

    @torch.no_grad()
    def verify(self, frames_rgb: list[np.ndarray], t: float, video_id: str,
               candidate: str, classes: list[str]) -> Stage2Output:
        user = STAGE2_USER.format(ts=t, video_id=video_id, candidate=candidate, classes=", ".join(classes))
        content = [{"type": "image"} for _ in frames_rgb] + [{"type": "text", "text": user}]
        prompt = self.processor.apply_chat_template(
            [{"role": "system", "content": STAGE2_SYSTEM},
             {"role": "user", "content": content}],
            tokenize=False, add_generation_prompt=True,
        )
        enc = self.processor(
            text=[prompt], images=frames_rgb, return_tensors="pt"
        ).to(self.device)

        import time
        t0 = time.perf_counter()
        out = self.model.generate(**enc, max_new_tokens=STAGE2_MAX_NEW_TOKENS, do_sample=False)
        latency_ms = (time.perf_counter() - t0) * 1000

        text = self.processor.batch_decode(out[:, enc.input_ids.shape[1]:],
                                           skip_special_tokens=True)[0]
        parsed = _parse_json(text)
        return Stage2Output(
            is_anomaly=bool(parsed.get("is_anomaly", False)),
            class_name=str(parsed.get("class_name", "normal")),
            confidence=float(parsed.get("confidence", 0.0)),
            description=str(parsed.get("description", "")),
            latency_ms=latency_ms,
        )


def _parse_json(text: str) -> dict:
    """Models occasionally wrap JSON in prose or code fences — recover it."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return {}
