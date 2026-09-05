# AHC Visual Intelligence Hackathon — Real-Time Video Anomaly Detection

Two-stage cascade for detecting contextual anomalies in drone/CCTV/dashcam
video in real time, built around a small vision-language model as the runtime
detector. Large hosted models are used only during development (comparison,
distillation, data generation) — never in the runtime path.

## Architecture

```
video feed ──▶ sampler (2 fps)
                 │
                 ▼
        ┌──────────────────┐   every sample, ~5ms GPU / ~60ms CPU
        │ Stage 1: CLIP    │   prompt-ensemble scoring, EMA smoothing
        │ ViT-B/32 scorer  │   high recall, cheap
        └────────┬─────────┘
                 │ trigger (score ≥ τ for N consecutive samples)
                 ▼
        ┌──────────────────┐   triggered only, ~0.5-1.5s on T4
        │ Stage 2: Qwen2.5 │   6-frame window → structured JSON verdict
        │ VL-3B verifier   │   contextual reasoning, false-alarm filter
        └────────┬─────────┘
                 │ confirmed (confidence ≥ τ₂)
                 ▼
        ┌──────────────────┐
        │ Temporal event   │   per-class debounce, cooldown, dedup
        │ state machine    │   → alerts + event intervals
        └──────────────────┘
```

Why a cascade: the problem statement prizes real-time throughput across many
feeds and low false alarms. Stage 1 runs always-on at a few ms per frame;
stage 2's expensive reasoning runs only when something is worth verifying
(typically <5% of samples). The VLM supplies the context judgment the problem
demands (parked car: normal in a bay, anomalous on a shoulder); the state
machine supplies the temporal judgment (accidents are instant, congestion
builds, stalled vehicles become anomalous only after a while).

## Layout

- `src/stage1_clip.py` — always-on CLIP scorer (prompt ensembles, EMA, trigger)
- `src/stage2_vlm.py` — Qwen2.5-VL-3B verifier, structured JSON output
- `src/temporal.py` — event state machine (confirm/clear/cooldown)
- `src/pipeline.py` — cascade orchestrator + CLI, emits predictions JSON
- `src/eval/evaluate.py` — Levels 1/2/3 scoring against ground truth
- `src/prompts.py` — all prompts in one place
- `scripts/download_dataset.py` — multi-mirror resilient Drive downloader
- `notebooks/` — Colab/Kaggle fine-tuning (stage-1 probe, stage-2 LoRA)

## Run

```bash
.venv/bin/python -m src.pipeline --input data/test/videos --output runs/predictions.json
.venv/bin/python -m src.eval.evaluate runs/predictions.json
```

`--no-stage2` gives a fast stage-1-only baseline.

## Metrics

- Level 1: video-level AUC / F1 / accuracy (`is_anomaly`)
- Level 2: event precision / recall / F1 (tolerant temporal matching), false alarms per hour
- Level 3: class accuracy on matched events + confusion matrix
- Runtime: per-stage latency, real-time factor (processing sec / video sec; <1 is real-time)
