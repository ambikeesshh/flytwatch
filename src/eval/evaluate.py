"""Evaluation harness for the AHC video anomaly detection task.

Scores a predictions file against data/test/ground_truth.csv on the three
task tiers from the dataset doc:

  Level 1 — video-level binary detection (is_anomaly): AUC / F1 / accuracy
  Level 2 — temporal event detection (start/end): event F1 with tolerant
            matching, false alarms per hour
  Level 3 — event classification + description: class accuracy on matched
            events, confusion matrix

Prediction format (JSON list, one entry per detected event):
  {"video_id": "T001", "is_anomaly": true, "class_name": "fire",
   "start_time_sec": 12.4, "end_time_sec": 18.0, "score": 0.81,
   "description": "..."}
For normal videos, emit a single entry with is_anomaly=false, score = max
normal-confidence, and empty timestamps.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import ANOMALY_LABELS, GT_COLUMNS, LABELS, RUNS_DIR, TEST_DIR


@dataclass
class EvalResult:
    level1: dict = field(default_factory=dict)
    level2: dict = field(default_factory=dict)
    level3: dict = field(default_factory=dict)
    per_video: list = field(default_factory=list)

    def to_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(
            {"level1": self.level1, "level2": self.level2, "level3": self.level3,
             "per_video": self.per_video}, indent=2, default=float))


def load_ground_truth(gt_path: Path | None = None) -> pd.DataFrame:
    gt_path = gt_path or TEST_DIR / "ground_truth.csv"
    df = pd.read_csv(gt_path)
    missing = [c for c in GT_COLUMNS if c in ["video_id", "is_anomaly", "class_name"] and c not in df.columns]
    if missing:
        raise ValueError(f"ground truth missing columns: {missing}")
    df["start_time_sec"] = pd.to_numeric(df.get("start_time_sec"), errors="coerce")
    df["end_time_sec"] = pd.to_numeric(df.get("end_time_sec"), errors="coerce")
    df["is_anomaly"] = df["is_anomaly"].astype(str).str.strip().str.lower().map(
        {"true": True, "false": False}
    ).fillna(False)
    return df


def load_predictions(pred_path: Path) -> list[dict]:
    return json.loads(Path(pred_path).read_text())


def _video_duration(video_id: str, gt_row: pd.DataFrame, videos_csv: Path | None = None) -> float:
    """Duration from videos.csv if it exposes one, else max GT end time."""
    videos_csv = videos_csv or (TEST_DIR / "videos.csv")
    if videos_csv.exists():
        meta = pd.read_csv(videos_csv)
        if "duration_sec" in meta.columns:
            row = meta[meta["video_id"] == video_id]
            if len(row):
                return float(row["duration_sec"].iloc[0])
    ends = pd.to_numeric(gt_row["end_time_sec"], errors="coerce").dropna()
    return float(ends.max()) if len(ends) else 60.0


def evaluate_level1(gt: pd.DataFrame, preds: list[dict]) -> dict:
    """Video-level binary detection."""
    gt_video = gt.groupby("video_id")["is_anomaly"].any()
    pred_by_video = {}
    for p in preds:
        cur = pred_by_video.get(p["video_id"])
        score = float(p.get("score", 0.0)) if p.get("is_anomaly", True) else 1.0 - float(p.get("score", 0.0))
        if cur is None or score > cur:
            pred_by_video[p["video_id"]] = score

    ids = sorted(gt_video.index)
    y_true = np.array([int(gt_video[v]) for v in ids])
    y_score = np.array([pred_by_video.get(v, 0.0) for v in ids])

    auc = float(roc_auc_score(y_true, y_score)) if len(set(y_true)) == 2 else float("nan")
    best = {"f1": -1.0, "threshold": 0.5}
    for t in np.linspace(0.05, 0.95, 19):
        f1 = f1_score(y_true, y_score >= t)
        if f1 > best["f1"]:
            best = {"f1": float(f1), "threshold": float(t)}
    y_pred = y_score >= best["threshold"]
    return {
        "video_auc": auc,
        "f1": best["f1"],
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "threshold": best["threshold"],
        "n_videos": len(ids),
        "n_anomaly_videos": int(y_true.sum()),
    }


def _events_from(df: pd.DataFrame) -> list[dict]:
    evs = []
    for _, r in df.iterrows():
        if not r["is_anomaly"]:
            continue
        start = r["start_time_sec"] if pd.notna(r["start_time_sec"]) else 0.0
        end = r["end_time_sec"] if pd.notna(r["end_time_sec"]) else start
        evs.append({"class_name": r["class_name"], "start": float(start),
                    "end": float(max(start, end))})
    return evs


def _match_events(gt_events: list[dict], pred_events: list[dict]) -> tuple[list, list, list]:
    """Greedy one-to-one matching. A pair matches if same class and the
    prediction interval overlaps the GT interval (or its start lands inside
    it — accidents can be over in a second, so strict IoU is too harsh)."""
    matches: list[tuple[dict, dict]] = []
    used_p = set()
    for g in gt_events:
        for i, p in enumerate(pred_events):
            if i in used_p or p["class_name"] != g["class_name"]:
                continue
            overlap = min(g["end"], p["end"]) - max(g["start"], p["start"])
            inside = g["start"] <= p["start"] <= g["end"] + 1.0
            if overlap > 0 or inside:
                matches.append((g, p))
                used_p.add(i)
                break
    fp = [p for i, p in enumerate(pred_events) if i not in used_p]
    fn = [g for g in gt_events if g not in [m[0] for m in matches]]
    return matches, fp, fn


def evaluate_level2(gt: pd.DataFrame, preds: list[dict]) -> dict:
    """Temporal event detection with tolerant matching + false alarms/hour."""
    per_class = {c: {"tp": 0, "fp": 0, "fn": 0} for c in ANOMALY_LABELS}
    total_fp_secs = 0.0
    total_dur = 0.0
    per_video = []

    for vid, gdf in gt.groupby("video_id"):
        gt_events = _events_from(gdf)
        pred_events = [
            {"class_name": p["class_name"],
             "start": float(p.get("start_time_sec") or 0.0),
             "end": float(p.get("end_time_sec") or p.get("start_time_sec") or 0.0)}
            for p in preds
            if p["video_id"] == vid and p.get("is_anomaly", True)
            and p.get("class_name") in ANOMALY_LABELS
        ]
        matches, fp, fn = _match_events(gt_events, pred_events)
        for g, _ in matches:
            per_class[g["class_name"]]["tp"] += 1
        for f in fn:
            per_class[f["class_name"]]["fn"] += 1
        for f in fp:
            if f["class_name"] in per_class:
                per_class[f["class_name"]]["fp"] += 1
        dur = _video_duration(vid, gdf)
        total_dur += dur
        total_fp_secs += sum(max(0.0, f["end"] - f["start"]) for f in fp)
        per_video.append({"video_id": vid, "gt_events": len(gt_events),
                          "pred_events": len(pred_events), "matched": len(matches),
                          "false_alarms": len(fp)})

    tp = sum(v["tp"] for v in per_class.values())
    fp = sum(v["fp"] for v in per_class.values())
    fn = sum(v["fn"] for v in per_class.values())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    hours = max(total_dur, 1e-6) / 3600.0
    return {
        "event_precision": prec,
        "event_recall": rec,
        "event_f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
        "false_alarms_per_hour": fp / hours,
        "tp": tp, "fp": fp, "fn": fn,
        "per_class": per_class,
        "per_video": per_video,
    }


def evaluate_level3(gt: pd.DataFrame, preds: list[dict]) -> dict:
    """Class accuracy on matched events + confusion matrix."""
    y_true, y_pred = [], []
    for vid, gdf in gt.groupby("video_id"):
        gt_events = _events_from(gdf)
        pred_events = [
            {"class_name": p["class_name"],
             "start": float(p.get("start_time_sec") or 0.0),
             "end": float(p.get("end_time_sec") or 0.0)}
            for p in preds
            if p["video_id"] == vid and p.get("is_anomaly", True)
        ]
        matches, _, _ = _match_events(gt_events, pred_events)
        for g, p in matches:
            y_true.append(g["class_name"])
            y_pred.append(p["class_name"])

    labels = [l for l in LABELS if l in set(y_true) | set(y_pred)] or LABELS
    cm = confusion_matrix(y_true, y_pred, labels=labels).tolist() if y_true else []
    return {
        "class_accuracy": float(accuracy_score(y_true, y_pred)) if y_true else float("nan"),
        "n_matched": len(y_true),
        "confusion_matrix": {"labels": labels, "matrix": cm},
    }


def run_eval(pred_path: Path, gt_path: Path | None = None, out_path: Path | None = None) -> EvalResult:
    gt = load_ground_truth(gt_path)
    preds = load_predictions(pred_path)
    res = EvalResult()
    res.level1 = evaluate_level1(gt, preds)
    res.level2 = evaluate_level2(gt, preds)
    res.level3 = evaluate_level3(gt, preds)
    res.per_video = res.level2.pop("per_video", [])
    out_path = out_path or RUNS_DIR / "metrics.json"
    res.to_json(out_path)
    return res


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("predictions", type=Path)
    ap.add_argument("--gt", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    r = run_eval(args.predictions, args.gt, args.out)
    print(json.dumps({"level1": r.level1, "level2": r.level2, "level3": {
        "class_accuracy": r.level3["class_accuracy"], "n_matched": r.level3["n_matched"]}}, indent=2))
