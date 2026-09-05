# FlytWatch — Real-Time Video Anomaly Detection

FlytWatch is a two-stage cascade for detecting contextual anomalies in
drone / CCTV / dashcam video in near real time on modest hardware:

- **Stage 1 — CLIP ViT-B/32 trained probe.** A linear probe on frozen CLIP
  image embeddings replaces the zero-shot prompt head. It scores every
  sampled frame at **~60 ms/frame on CPU** with **72.3% cross-validation
  accuracy** (trained on the D1/D2 training split), giving a cheap,
  high-recall anomaly trigger.
- **Stage 2 — Qwen2.5-VL-3B LoRA verifier.** A LoRA fine-tuned 3B
  vision-language model runs only on triggered windows, reasoning over a
  6-frame clip to emit a structured JSON verdict (class, confidence,
  time span) and filter false alarms.

A temporal event state machine on top handles confirm/clear/cooldown, so
accidents fire instantly, congestion builds gradually, and a parked car is
normal in a bay but anomalous on a road shoulder.

```
video ─▶ sampler (2 fps) ─▶ Stage 1: CLIP probe (always-on, ~60ms CPU)
                              │ score ≥ τ for N samples
                              ▼
                           Stage 2: Qwen2.5-VL-3B LoRA (triggered, T4)
                              │ confidence ≥ τ₂
                              ▼
                           event state machine ─▶ alerts + intervals
```

## Repository layout

```
src/          core package
  stage1_clip.py      CLIP scorer + trained linear probe head
  stage2_vlm.py       Qwen2.5-VL-3B LoRA verifier (structured JSON out)
  temporal.py         event state machine (confirm/clear/cooldown)
  pipeline.py         cascade orchestrator + CLI
  submission.py       predictions JSON → platform submission format
  eval/evaluate.py    Level 1/2/3 scoring vs ground truth
  demo/               Flask dashboard (live alerts + event log)
scripts/       run + scoring + deck tooling
  run_v2_local.py     CPU end-to-end run of the probe cascade (v2)
  predict_score.py    platform score predictor (reverse-engineered rubric)
  build_deck2.js      builds the 2-slide submission deck (pptxgenjs)
  build_colab_notebook.py / download_dataset.py
notebooks/     Colab/Kaggle training (stage-1 probe, stage-2 LoRA)
data/          datasets, manifests (gitignored — not in the repo)
runs/          predictions, scores, artifacts (gitignored)
submission/    final deliverables (deck pptx/pdf; jpgs gitignored)
```

## Setup

```bash
uv venv                          # or: python -m venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt   # torch, transformers, opencv, pillow
```

Put dataset videos under `data/test/` and `data/eval/`
(`scripts/download_dataset.py` fetches from Drive mirrors).

## Run

End-to-end cascade (stage 2 needs a GPU or will be slow):

```bash
python -m src.pipeline --input data/test/videos --output runs/predictions.json
python -m src.eval.evaluate runs/predictions.json
```

Fast CPU path used for the eval round — stage-1 probe only, event state
machine, and platform-oriented post-processing:

```bash
python scripts/run_v2_local.py --help
```

Score a predictions file against the platform rubric:

```bash
python scripts/predict_score.py runs/predictions.json
```

Build the submission deck (7 slides, needs `npm install` once):

```bash
node scripts/build_deck_final.js
```

## Submission

`submission/` contains the final deliverables:

- `FlytWatch_Final.pptx` / `.pdf` — the judging deck: architecture flow,
  temporal-logic schematic, example detections, results charts, learnings.

## Metrics

- Level 1: video-level AUC / F1 / accuracy
- Level 2: event precision / recall / F1 (IoU ≥ 0.5 matching), false alarms/hr
- Level 3: class accuracy on matched events
- Runtime: per-stage latency and real-time factor (< 1 means real-time)
