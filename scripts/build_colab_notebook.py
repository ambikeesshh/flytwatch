#!/usr/bin/env python3
"""Generates notebooks/colab_finetune.ipynb — the Colab training entry point.

Part A extracts CLIP features from train frames and fits a linear probe for
stage 1. Part B LoRA-fine-tunes Qwen2.5-VL-3B for stage 2 with Unsloth.
Artifacts land in Drive under ahc_hackathon_artifacts/ for local use.
"""

import json
from pathlib import Path

cells = []

def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text})

def code(text):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                  "outputs": [], "source": text})

md("""# AHC Hackathon — Fine-tune on Colab (T4)

**Setup before running:** open the dataset mirror link in your browser, open the shared
folder, and use **Add shortcut to Drive** so it appears in *My Drive*. Then mount Drive
below — no quota problems, because Colab reads with your authenticated session.

Produces:
- `stage1_linear_probe.pt` — linear head over CLIP embeddings (stage 1 upgrade)
- `stage2_lora_adapter/` — LoRA adapter for Qwen2.5-VL-3B (stage 2 upgrade)
""")

code("""# 1 — environment
!pip -q install unsloth_zoo
!pip -q install --no-deps unsloth
!pip -q install --no-deps --upgrade xformers

import torch, os
assert torch.cuda.is_available(), "Runtime → Change runtime type → T4 GPU"
print("GPU:", torch.cuda.get_device_name(0))

from google.colab import drive
drive.mount('/content/drive')
""")

code("""# 2 — locate the dataset on Drive (shortcut added from a mirror)
import glob, os

CANDIDATES = glob.glob('/content/drive/MyDrive/**/train', recursive=True)
CANDIDATES = [c for c in CANDIDATES if os.path.isdir(os.path.join(os.path.dirname(c), 'test'))]
assert CANDIDATES, "train/test not found — add the shared folder as a Drive shortcut"
DATA_ROOT = os.path.dirname(CANDIDATES[0])
TRAIN, TEST = os.path.join(DATA_ROOT, 'train'), os.path.join(DATA_ROOT, 'test')
print("DATA_ROOT =", DATA_ROOT)
print("classes:", sorted(os.listdir(TRAIN)))
""")

code("""# 3 — read all ground-truth CSVs into one frame
import pandas as pd, glob, os

rows = []
for csv in glob.glob(os.path.join(TRAIN, '*', 'ground_truth.csv')):
    cls = os.path.basename(os.path.dirname(csv))
    df = pd.read_csv(csv)
    if 'class_name' not in df.columns: df['class_name'] = cls
    df['source_class_dir'] = cls
    rows.append(df)
train_gt = pd.concat(rows, ignore_index=True)
for col in ('start_time_sec', 'end_time_sec'):
    if col in train_gt: train_gt[col] = pd.to_numeric(train_gt[col], errors='coerce')
train_gt['is_anomaly'] = train_gt.get('is_anomaly', True).astype(bool) if 'is_anomaly' in train_gt else True
print(train_gt.shape); print(train_gt['class_name'].value_counts())
train_gt.head(3)
""")

code("""# 4 — clip sampling utilities
import cv2, numpy as np

def video_path_for(video_id):
    for pat in (os.path.join(TRAIN, '*', 'videos', video_id + '.mp4'),
                os.path.join(TRAIN, '*', 'videos', video_id + '.mov')):
        hits = glob.glob(pat)
        if hits: return hits[0]
    return None

def sample_clip(path, start, end, n=6):
    \"\"\"n frames evenly spaced across [start, end].\"\"\"
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    end = max(end if end == end else start + 1.0, start + 0.5)  # NaN-safe
    ts = np.linspace(max(0, start), end, n)
    frames, got = [], []
    for t in ts:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, fr = cap.read()
        if ok:
            frames.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)); got.append(t)
    cap.release()
    return frames, got

def sample_normal_clip(path, n=6, seed=0):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS); nfr = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    dur = nfr / fps if fps and nfr else 10.0
    cap.release()
    rng = np.random.default_rng(seed)
    start = float(rng.uniform(0, max(0.1, dur - 2)))
    return sample_clip(path, start, start + 2.0, n)
""")

md("""## Part A — stage-1 linear probe (CLIP features + logistic head)

Extracts CLIP embeddings for event clips and normal clips, then fits a
one-vs-rest linear classifier over the 11 anomaly classes + normal. This
replaces hand-written prompt similarity with a head trained on this data —
usually a large stage-1 accuracy jump for minutes of work.
""")

code("""# 5 — feature extraction (event clips + normal clips)
from transformers import CLIPModel, CLIPProcessor
from tqdm.auto import tqdm

LABELS = ["normal","traffic_accident","traffic_congestion","stalled_or_broken_down_vehicle",
          "vehicle_blocking_traffic","wrong_way_driving","road_spill_or_debris",
          "waterlogging_or_flood","fire","smoke","fighting_or_violence",
          "loitering_or_suspicious_presence"]
ANOM = [l for l in LABELS if l != 'normal']

clip_model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32').cuda().eval()
clip_proc = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')

def embed(frames):
    with torch.no_grad():
        enc = clip_proc(images=frames, return_tensors='pt').to('cuda')
        f = clip_model.get_image_features(**enc)
        f = f.pooler_output if hasattr(f, 'pooler_output') else f
        return (f / f.norm(dim=-1, keepdim=True)).mean(0).cpu().numpy()

X, y = [], []
events = train_gt[train_gt['is_anomaly'] & train_gt['class_name'].isin(ANOM)].head(600)
for _, r in tqdm(events.iterrows(), total=len(events)):
    p = video_path_for(str(r['video_id']))
    if not p: continue
    s = r.get('start_time_sec'); e = r.get('end_time_sec')
    frames, _ = sample_clip(p, float(s) if s == s else 0.0, float(e) if e == e else float(s) + 2 if s == s else 2.0)
    if len(frames) >= 2:
        X.append(embed(frames)); y.append(r['class_name'])

normal_df = train_gt[train_gt['class_name'] == 'normal'].head(300)
for _, r in tqdm(normal_df.iterrows(), total=len(normal_df)):
    p = video_path_for(str(r['video_id']))
    if not p: continue
    frames, _ = sample_normal_clip(p)
    if len(frames) >= 2:
        X.append(embed(frames)); y.append('normal')

X = np.stack(X); y = np.array(y)
print("dataset:", X.shape, dict(zip(*np.unique(y, return_counts=True))))
""")

code("""# 6 — fit + export the linear probe
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import pickle, os

clf = LogisticRegression(max_iter=2000, C=1.0)
print("cv accuracy:", cross_val_score(clf, X, y, cv=3).mean())
clf.fit(X, y)

ART = '/content/drive/MyDrive/ahc_hackathon_artifacts'
os.makedirs(ART, exist_ok=True)
with open(os.path.join(ART, 'stage1_linear_probe.pkl'), 'wb') as f:
    pickle.dump({'classes': list(clf.classes_), 'coef': clf.coef_, 'intercept': clf.intercept_}, f)
print("saved →", ART + '/stage1_linear_probe.pkl')
""")

md("""## Part B — stage-2 LoRA fine-tune (Qwen2.5-VL-3B with Unsloth)

Trains the verifier to emit the exact JSON verdict format on clips from the
training set. `description_summary` from ground truth is used as the
supervision for the description field when present.
""")

code("""# 7 — build SFT pairs: (clip images, prompt) → JSON answer
import json

SYSTEM = ("You are a video anomaly verification agent watching footage from a camera "
          "over an urban area. You receive a short sequence of frames around a potential "
          "anomaly. Anomaly depends on context: a parked car is normal in a parking bay "
          "but anomalous on a highway shoulder. Respond ONLY with a JSON object.")

def user_msg(vid, ts, candidate):
    return (f"Frames from around timestamp {ts:.1f}s of video {vid} are shown in time order.\\n"
            f"Candidate anomaly type flagged by a cheap detector: {candidate}\\n"
            f"Classes you may use: {', '.join(LABELS)}\\n"
            'Respond with exactly this JSON schema:\\n'
            '{"is_anomaly": true/false, "class_name": "<class or normal>", '
            '"confidence": <0.0-1.0>, "description": "<one sentence>"}')

pairs = []
for _, r in events.iterrows():
    p = video_path_for(str(r['video_id']))
    if not p: continue
    s = r.get('start_time_sec'); e = r.get('end_time_sec')
    s = float(s) if s == s else 0.0; e = float(e) if e == e else s + 2.0
    frames, got = sample_clip(p, s, e, n=6)
    if len(frames) < 2: continue
    desc = str(r.get('description_summary') or '').strip() or f"A {r['class_name'].replace('_',' ')} event is visible."
    ans = {"is_anomaly": True, "class_name": r['class_name'], "confidence": 0.95, "description": desc}
    pairs.append({"frames": frames, "user": user_msg(r['video_id'], got[0], r['class_name']),
                  "answer": json.dumps(ans)})

for _, r in normal_df.iterrows():
    p = video_path_for(str(r['video_id']))
    if not p: continue
    frames, got = sample_normal_clip(p, n=6)
    if len(frames) < 2: continue
    ans = {"is_anomaly": False, "class_name": "normal", "confidence": 0.95,
           "description": "Routine activity, nothing anomalous."}
    pairs.append({"frames": frames, "user": user_msg(r['video_id'], got[0], 'unknown'),
                  "answer": json.dumps(ans)})

print(len(pairs), "SFT pairs")
""")

code("""# 8 — Unsloth LoRA training
from unsloth import FastVisionModel
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig

model, processor = FastVisionModel.from_pretrained(
    "unsloth/Qwen2.5-VL-3B-Instruct-bnb-4bit",
    load_in_4bit=True, use_gradient_checkpointing="unsloth")

model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers=False,
    finetune_language_layers=True,
    r=16, lora_alpha=16, lora_dropout=0,
    target_modules="all-linear",
)

def to_convo(p):
    return {
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": [{"type": "image", "image": im} for im in p["frames"]]
                                       + [{"type": "text", "text": p["user"]}]},
            {"role": "assistant", "content": [{"type": "text", "text": p["answer"]}]},
        ]
    }

dataset = [to_convo(p) for p in pairs]
FastVisionModel.for_training(model)

trainer = SFTTrainer(
    model=model,
    data_collator=UnslothVisionDataCollator(model, processor),
    train_dataset=dataset,
    args=SFTConfig(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        num_train_epochs=1,
        warmup_steps=5,
        learning_rate=1e-4,
        logging_steps=5,
        output_dir="stage2_lora_runs",
        bf16=True,
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
        dataset_num_proc=2,
        max_seq_length=2048,
        remove_unused_columns=False,
    ),
)
trainer.train()
""")

code("""# 9 — export the adapter to Drive
model.save_pretrained(os.path.join(ART, 'stage2_lora_adapter'))
processor.save_pretrained(os.path.join(ART, 'stage2_lora_adapter'))
print("saved →", ART + '/stage2_lora_adapter')
""")

code("""# 10 — sanity-check the fine-tuned verifier on one training clip
FastVisionModel.for_inference(model)
p = pairs[0]
convo = [{"role": "system", "content": SYSTEM},
         {"role": "user", "content": [{"type": "image", "image": im} for im in p["frames"]]
                                    + [{"type": "text", "text": p["user"]}]}]
text = processor.apply_chat_template(convo, tokenize=False, add_generation_prompt=True)
enc = processor(text=[text], images=p["frames"], return_tensors="pt").to("cuda")
out = model.generate(**enc, max_new_tokens=128, do_sample=False)
print(processor.batch_decode(out[:, enc.input_ids.shape[1]:], skip_special_tokens=True)[0])
print("expected:", p["answer"])
""")

md("""## Part C — run the full detector over the 34 test videos

Uses the fine-tuned cascade on GPU: linear-probe stage 1 (cheap, every sample)
+ LoRA stage 2 verifier (on triggers). Emits `raw_results.json` with events,
clip scores and timing, mirroring the local pipeline format. The next cell
copies it to the Drive artifacts folder — download it locally and run the
submission builder.
""")

code("""# 11 — full cascade inference over the test pack
import cv2, numpy as np, json, time, glob, os
import torch

TEST_VIDEOS = os.path.join(TEST, 'videos')
LABELS_ALL = LABELS  # from Part A
ANOM = [l for l in LABELS if l != 'normal']

probe = pickle.load(open(os.path.join(ART, 'stage1_linear_probe.pkl'), 'rb'))
PROBE_CLASSES = probe['classes']
PROBE_W = torch.tensor(probe['coef'], dtype=torch.float32).cuda()
PROBE_B = torch.tensor(probe['intercept'], dtype=torch.float32).cuda()

def embed_frames(frames):
    with torch.no_grad():
        enc = clip_proc(images=frames, return_tensors='pt').to('cuda')
        f = clip_model.get_image_features(**enc)
        f = f.pooler_output if hasattr(f, 'pooler_output') else f
        return f / f.norm(dim=-1, keepdim=True)

def probe_scores(feat):  # feat [D] normalized
    logits = feat @ PROBE_W.T + PROBE_B
    return torch.softmax(logits, dim=-1)

SAMPLE_FPS, EMA_A, TRIG_T, TRIG_N = 2.0, 0.6, 0.50, 2
CONFIRM_S, CLEAR_S = 1.0, 5.0

def process_video(path, vid):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(fps / SAMPLE_FPS)))
    nfr = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    dur = nfr / fps if nfr else 0.0
    events, s1_ms, s2_ms = [], [], []
    ema, consec, buf = {}, 0, []
    i = 0
    while True:
        ok, fr = cap.read()
        if not ok: break
        if i % step == 0:
            t = i / fps
            rgb = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
            buf.append((t, rgb))
            if len(buf) > 48: buf.pop(0)
            t0 = time.perf_counter()
            feat = embed_frames([rgb])[0]
            probs = probe_scores(feat).cpu().numpy()
            cls_probs = {c: float(p) for c, p in zip(PROBE_CLASSES, probs) if c != 'normal'}
            top = max(cls_probs, key=cls_probs.get)
            prev = ema.get(top, cls_probs[top])
            sm = EMA_A * cls_probs[top] + (1 - EMA_A) * prev
            ema[top] = sm
            s1_ms.append((time.perf_counter() - t0) * 1000)
            consec = consec + 1 if sm >= TRIG_T else 0
            if consec >= TRIG_N:
                window = [f for (ft, f) in buf if ft >= t - 2.0][-6:]
                if len(window) >= 2:
                    t0 = time.perf_counter()
                    v = stage2_verify(window, t, vid, top)
                    s2_ms.append((time.perf_counter() - t0) * 1000)
                    if v and v['is_anomaly'] and v.get('confidence', 0) >= 0.5:
                        ev = next((e for e in events if e['class_name'] == v['class_name'] and t - e['last_t'] <= CLEAR_S), None)
                        if ev:
                            ev['last_t'] = t; ev['score'] = max(ev['score'], v.get('confidence', 0))
                            ev.setdefault('desc', v.get('description', ''))
                        else:
                            events.append({'class_name': v['class_name'], 'start_t': t, 'last_t': t,
                                           'score': v.get('confidence', 0), 'desc': v.get('description', '')})
        i += 1
    cap.release()
    confirmed = [e for e in events if e['last_t'] - e['start_t'] >= 0 or e['score'] >= 0.6]
    return {'video_id': vid, 'duration_sec': dur,
            'events': [{'video_id': vid, 'is_anomaly': True, 'class_name': e['class_name'],
                        'start_time_sec': round(e['start_t'], 2), 'end_time_sec': round(e['last_t'], 2),
                        'score': round(e['score'], 4), 'description': e.get('desc', '')} for e in confirmed],
            'timing': {'stage1_ms_samples': [round(m,3) for m in s1_ms],
                       'stage2_ms_calls': [round(m,3) for m in s2_ms],
                       'samples': len(s1_ms), 'stage2_calls': len(s2_ms),
                       'processing_sec': sum(s1_ms)/1000 + sum(s2_ms)/1000,
                       'video_duration_sec': dur}}

def stage2_verify(window, t, vid, candidate):
    user = (f"Frames from around timestamp {t:.1f}s of video {vid} are shown in time order.\\n"
            f"Candidate anomaly type flagged by a cheap detector: {candidate}\\n"
            f"Classes you may use: {', '.join(LABELS)}\\n"
            'Respond with exactly this JSON schema:\\n'
            '{"is_anomaly": true/false, "class_name": "<class or normal>", '
            '"confidence": <0.0-1.0>, "description": "<one sentence>"}')
    convo = [{"role": "system", "content": SYSTEM},
             {"role": "user", "content": [{"type": "image", "image": im} for im in window]
                                        + [{"type": "text", "text": user}]}]
    text = processor.apply_chat_template(convo, tokenize=False, add_generation_prompt=True)
    enc = processor(text=[text], images=window, return_tensors="pt").to("cuda")
    out = model.generate(**enc, max_new_tokens=128, do_sample=False)
    ans = processor.batch_decode(out[:, enc.input_ids.shape[1]:], skip_special_tokens=True)[0]
    try:
        return json.loads(ans)
    except Exception:
        import re
        m = re.search(r'\\{.*\\}', ans, re.DOTALL)
        try: return json.loads(m.group(0)) if m else None
        except Exception: return None

raw_out = []
for path in sorted(glob.glob(os.path.join(TEST_VIDEOS, '*.mp4'))):
    vid = os.path.basename(path)[:-4]
    r = process_video(path, vid)
    raw_out.append(r)
    print(vid, len(r['events']), 'events')

with open('/content/raw_results.json', 'w') as f:
    json.dump(raw_out, f)
print('saved /content/raw_results.json')
""")

code("""# 12 — copy results to Drive artifacts
import shutil
shutil.copy('/content/raw_results.json', os.path.join(ART, 'raw_results.json'))
print('saved →', ART + '/raw_results.json')
print('Next: download this file from Drive into runs/ locally, then run the submission builder.')
""")

nb = {"nbformat": 4, "nbformat_minor": 5,
      "metadata": {"colab": {"gpuType": "T4", "provenance": []},
                   "kernelspec": {"name": "python3", "display_name": "Python 3"},
                   "language_info": {"name": "python"},
                   "accelerator": "GPU"},
      "cells": cells}

out = Path(__file__).resolve().parents[1] / "notebooks" / "colab_finetune.ipynb"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(nb, indent=1))
print(f"wrote {out} ({len(cells)} cells)")
