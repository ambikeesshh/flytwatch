"""Cascade pipeline: ingest → stage-1 scoring → (on trigger) stage-2 VLM
verification → temporal aggregation → events.

One object processes a video file at STAGE1_SAMPLE_FPS and produces both the
event list (predictions JSON for the eval harness) and a per-sample trace
(stage-1 scores over time — the demo timeline is built from it).
"""

from __future__ import annotations

import json
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from src.config import (
    ANOMALY_LABELS,
    RUNS_DIR,
    STAGE1_SAMPLE_FPS,
    STAGE2_CLIP_FRAMES,
    STAGE2_CLIP_SPREAD_SEC,
    STAGE2_CONFIRM_THRESHOLD,
    TEST_DIR,
)
from src.stage1_clip import Stage1CLIP
from src.stage2_vlm import Stage2VLM
from src.temporal import TemporalAggregator


@dataclass
class VideoResult:
    video_id: str
    duration_sec: float
    events: list[dict] = field(default_factory=list)
    trace: list[dict] = field(default_factory=list)   # per-sample stage-1 scores
    timing: dict = field(default_factory=dict)
    clip_scores: dict[str, float] = field(default_factory=dict)  # mean per-class score over whole clip

    @property
    def is_anomaly(self) -> bool:
        return bool(self.events)


class CascadePipeline:
    def __init__(self, stage1: Stage1CLIP, stage2: Stage2VLM | None = None):
        self.stage1 = stage1
        self.stage2 = stage2
        self._frame_buf: deque[tuple[float, np.ndarray]] = deque(maxlen=64)

    def process_video(self, path: Path, video_id: str | None = None) -> VideoResult:
        video_id = video_id or path.stem
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise RuntimeError(f"cannot open {path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        step = max(1, int(round(fps / STAGE1_SAMPLE_FPS)))
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = n_frames / fps if n_frames else 0.0

        agg = TemporalAggregator()
        res = VideoResult(video_id=video_id, duration_sec=duration)
        s1_ms: list[float] = []
        s2_ms: list[float] = []
        score_sums: dict[str, float] = {}

        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if i % step == 0:
                t = i / fps
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self._frame_buf.append((t, rgb))

                t0 = time.perf_counter()
                s1 = self.stage1.score_frame(rgb, t)
                s1_ms.append((time.perf_counter() - t0) * 1000)
                res.trace.append({"t": t, "top_class": s1.top_class,
                                  "top_score": s1.top_score, "smoothed": s1.smoothed_top})
                for c, v in s1.scores.items():
                    score_sums[c] = score_sums.get(c, 0.0) + v

                if s1.trigger:
                    window = self._window(t)
                    if self.stage2 is not None and len(window) >= 2:
                        t0 = time.perf_counter()
                        v = self.stage2.verify(window, t, video_id, s1.top_class, ANOMALY_LABELS)
                        s2_ms.append((time.perf_counter() - t0) * 1000)
                        if v.is_anomaly and v.confidence >= STAGE2_CONFIRM_THRESHOLD:
                            agg.update(t, v.class_name, v.confidence, v.description)
                    else:
                        agg.update(t, s1.top_class, s1.smoothed_top)
            i += 1
        cap.release()

        for ev in agg.confirmed_events():
            res.events.append({
                "video_id": video_id,
                "is_anomaly": True,
                "class_name": ev.class_name,
                "start_time_sec": round(ev.start_t, 2),
                "end_time_sec": round(ev.last_t, 2),
                "score": round(ev.max_confidence, 4),
                "description": ev.description,
            })
        n_samples = len(res.trace)
        if n_samples:
            res.clip_scores = {c: round(s / n_samples, 5) for c, s in score_sums.items()}

        wall = sum(s1_ms) / 1000 + sum(s2_ms) / 1000
        res.timing = {
            "stage1_ms_samples": [round(m, 3) for m in s1_ms],
            "stage2_ms_calls": [round(m, 3) for m in s2_ms],
            "stage1_ms_per_sample": float(np.mean(s1_ms)) if s1_ms else 0.0,
            "stage2_ms_per_call": float(np.mean(s2_ms)) if s2_ms else 0.0,
            "stage2_calls": len(s2_ms),
            "samples": len(res.trace),
            "video_duration_sec": duration,
            "processing_sec": wall,
            "realtime_factor": wall / duration if duration else 0.0,  # <1 means faster than real time
        }
        return res

    def _window(self, t: float) -> list[np.ndarray]:
        horizon = t - STAGE2_CLIP_SPREAD_SEC
        frames = [f for (ft, f) in self._frame_buf if ft >= horizon]
        if len(frames) > STAGE2_CLIP_FRAMES:
            idx = np.linspace(0, len(frames) - 1, STAGE2_CLIP_FRAMES).astype(int)
            frames = [frames[j] for j in idx]
        return frames


def run_directory(videos_dir: Path, out_path: Path, stage2: bool = True,
                  raw_path: Path | None = None) -> list[dict]:
    """Process every video in a directory; write the predictions JSON and a
    richer raw-results file (events + full timing lists) the submission
    builder consumes."""
    s1 = Stage1CLIP()
    s2 = Stage2VLM() if stage2 else None
    pipe = CascadePipeline(s1, s2)
    preds: list[dict] = []
    raw: list[dict] = []
    timings: list[dict] = []
    videos = sorted(videos_dir.glob("*.mp4"))
    for v in videos:
        r = pipe.process_video(v)
        preds.extend(r.events)
        timings.append({"video_id": r.video_id, **{
            k: v2 for k, v2 in r.timing.items() if not k.endswith("_ms_samples") and k != "stage2_ms_calls"
        }})
        raw.append({
            "video_id": r.video_id,
            "duration_sec": r.duration_sec,
            "events": r.events,
            "timing": r.timing,
            "clip_scores": r.clip_scores,
        })
        print(f"{r.video_id}: {len(r.events)} events | RTF {r.timing['realtime_factor']:.2f}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(preds, indent=2))
    (out_path.parent / "timings.json").write_text(json.dumps(timings, indent=2))
    if raw_path is not None:
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_text(json.dumps(raw, indent=2))
    return preds


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=TEST_DIR / "videos")
    ap.add_argument("--output", type=Path, default=RUNS_DIR / "predictions.json")
    ap.add_argument("--no-stage2", action="store_true", help="stage-1-only quick run")
    args = ap.parse_args()
    raw_path = args.output.parent / "raw_results.json"
    preds = run_directory(args.input, args.output, stage2=not args.no_stage2, raw_path=raw_path)
    print(f"\n{len(preds)} events written to {args.output}")
