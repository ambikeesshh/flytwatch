# FlytWatch — Cascade Architecture for Real-Time Video Anomaly Detection

## Design goal

Detect contextual anomalies in drone/CCTV/dashcam video in real time, cheaply
enough to run across many feeds, with low false alarms. The problem statement
pins the two tensions: anomaly is **contextual** (a parked car is normal in a
bay, anomalous on a shoulder), and inference must stay **cheap** — large
hosted models are development-only tools.

The cascade resolves both: a lightweight always-on stage watches every frame,
and a small vision-language model — the only class of model that can reason
about context in language — verifies only what matters.

## Pipeline

```
video feed (any fps)
   │  sampling @ 2 fps
   ▼
┌───────────────────────────────┐  every sample · ~5 ms GPU / ~60 ms CPU
│ Stage 1 — CLIP ViT-B/32 probe │  image embedding → linear head over
│ (trained on 3,173 labeled     │  11 anomaly classes + normal
│  train-set events)            │  EMA smoothing · consecutive-trigger gate
└───────────────┬───────────────┘
                │ trigger (score ≥ τ for N consecutive samples)
                ▼
┌───────────────────────────────┐  on trigger only · ~0.5–1.5 s on T4
│ Stage 2 — Qwen2.5-VL-3B       │  6-frame window → structured JSON verdict
│ LoRA verifier (fine-tuned on  │  {is_anomaly, class, confidence, why}
│ description summaries)        │  contextual reasoning, FA filter
└───────────────┬───────────────┘
                │ confirmed (confidence ≥ τ₂)
                ▼
┌───────────────────────────────┐
│ Temporal event state machine  │  per-class confirm/clear windows,
│                               │  cooldown dedup → alerts + intervals
└───────────────────────────────┘
```

**What runs per frame vs per clip:** stage 1 runs on every sampled frame
(2 fps); stage 2 runs per clip-window only when stage 1 arms it — typically
under 5% of samples. Temporal logic is O(1) per observation.

## Why each piece

- **CLIP linear probe (stage 1).** Zero-shot prompt similarity reached
  ~50% class accuracy on this taxonomy; training a linear head over CLIP
  embeddings on the labeled train set is minutes of compute for a large jump,
  and keeps per-frame cost at a single ViT-B/32 forward pass.
- **Small VLM verifier (stage 2).** The problem statement's own criterion:
  class membership alone cannot decide anomaly; context can. The verifier
  receives the candidate class and answers in structured JSON with an
  explanation, which doubles as the submission's reasoning field.
- **Temporal state machine.** Anomalies unfold at different scales (an
  accident lasts a second; congestion builds; a stalled vehicle is anomalous
  only after a while). Per-class confirm/clear windows and cooldowns turn
  per-sample scores into events and suppress duplicate alerts.

## Training (Colab, T4)

1. **Probe:** CLIP embeddings of ~900 train clips (events + normals) →
   multinomial logistic regression, 3-fold CV.
2. **Verifier:** LoRA (r=16, language layers only, frozen vision encoder) on
   (6-frame clip, candidate class) → JSON verdict pairs built from
   ground-truth `description_summary` fields — the distillation path the
   dataset doc itself recommends. Large hosted models are never used at
   runtime.

## Runtime characteristics (measured, local CPU)

- Stage 1: ~61 ms/frame (12-core CPU, no GPU) → 7× faster than real time
  end-to-end on the 34-video pack; RTF ≈ 0.13. On T4, stage 1 drops to
  single-digit ms.
- Latency stats are recorded per video (`runtime_metadata`) and reported
  honestly in every submission.

## Evaluation

Local harness scores all three benchmark tiers against the provided public
test ground truth before every submission: video-level AUC/F1 (D1), tolerant
temporal event matching with 0.5 overlap (D2/D3), false alarms per hour, and
per-class confusion — the same shape as the platform's scoring.
