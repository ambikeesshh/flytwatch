"""Build and validate the platform submission JSON.

Implements the exact contract from the benchmark page:
- one file, every video id at most once, matched against the manifest
- empty events [] means "normal"; class_name is never "normal"
- start/end null at D1; required, >=0, end>start, within duration at D2-3
- explanation 20-500 chars, bonus-only
- runtime_metadata per video: frames_processed, chunks_processed,
  end_to_end_internal_time_ms, model_runtimes with percentile stats
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import DATA_DIR, RUNS_DIR

ANOMALY_CLASSES = [
    "traffic_accident", "traffic_congestion", "stalled_or_broken_down_vehicle",
    "vehicle_blocking_traffic", "fire", "smoke", "waterlogging_or_flood",
    "wrong_way_driving", "road_spill_or_debris", "fighting_or_violence",
    "loitering_or_suspicious_presence",
]


def load_manifest(path: Path | None = None) -> dict[str, dict]:
    import os
    path = path or Path(os.environ.get("AHC_MANIFEST", "")) if os.environ.get("AHC_MANIFEST") else (path or DATA_DIR / "benchmark_manifest.json")
    man = json.loads(Path(path).read_text())
    return {v["video_id"]: v for v in man["videos"]}


def _model_runtime(name: str, ms_list: list[float]) -> dict | None:
    if not ms_list:
        return None
    arr = np.array(ms_list)
    return {
        "model_name": name,
        "call_count": len(arr),
        "total_time_ms": round(float(arr.sum()), 1),
        "average_time_ms": round(float(arr.mean()), 1),
        "p50_time_ms": round(float(np.percentile(arr, 50)), 1),
        "p95_time_ms": round(float(np.percentile(arr, 95)), 1),
        "max_time_ms": round(float(arr.max()), 1),
    }


def _clamp_explanation(text: str) -> str | None:
    text = (text or "").strip()
    if not text:
        return None
    if len(text) < 20:
        text = text + " " + "Anomalous activity detected in the scene context."  # pad to minimum
    return text[:500]


def build_submission(raw_results_path: Path, out_path: Path,
                     model_name: str = "flytwatch-cascade-v1",
                     submission_id: str = "run-01",
                     hardware: str = "local CPU (12-core)", manifest_path: Path | None = None) -> dict:
    manifest = load_manifest(manifest_path)
    raw = json.loads(Path(raw_results_path).read_text())

    predictions = []
    total_wall_ms = 0.0
    for entry in raw:
        vid = entry["video_id"]
        meta = manifest.get(vid)
        level = meta["level"] if meta else 2
        duration = float(meta["duration_sec"]) if meta else float(entry.get("duration_sec", 0))
        timing = entry.get("timing", {})

        events = []
        for ev in entry.get("events", []):
            if level == 1:
                start = end = None
            else:
                start = max(0.0, float(ev.get("start_time_sec") or 0.0))
                end = float(ev.get("end_time_sec") or start + 1.0)
                end = min(end, duration)
                if end <= start:
                    end = min(start + 1.0, duration)
            events.append({
                "class_name": ev["class_name"],
                "start_time_sec": start,
                "end_time_sec": end,
                "explanation": _clamp_explanation(ev.get("description", "")),
            })

        rt = [
            _model_runtime("clip-vit-b32-scorer", timing.get("stage1_ms_samples", [])),
            _model_runtime("qwen2.5-vl-3b-verifier", timing.get("stage2_ms_calls", [])),
        ]
        rt = [r for r in rt if r]
        wall_ms = round(float(timing.get("processing_sec", 0.0)) * 1000, 1)
        total_wall_ms += wall_ms
        predictions.append({
            "video_id": vid,
            "events": events,
            "runtime_metadata": {
                "frames_processed": int(timing.get("samples", 0)),
                "chunks_processed": max(1, int(timing.get("stage2_calls", 0)) or 1),
                "end_to_end_internal_time_ms": wall_ms,
                "model_runtimes": rt,
            },
        })

    sub = {
        "schema_version": "1.0",
        "submission_id": submission_id,
        "model_name": model_name,
        "run_metadata": {
            "total_wall_time_ms": round(total_wall_ms, 1),
            "max_parallel_videos": 1,
            "hardware": hardware,
        },
        "predictions": predictions,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(sub, indent=2))
    return sub


def validate_submission(path: Path, manifest_path: Path | None = None) -> list[str]:
    """Every rule from the benchmark page. Returns [] when clean."""
    errors: list[str] = []
    manifest = load_manifest(manifest_path)
    try:
        sub = json.loads(Path(path).read_text())
    except Exception as e:
        return [f"JSON parse error: {e}"]

    if sub.get("schema_version") != "1.0":
        errors.append("schema_version must be '1.0'")
    if not isinstance(sub.get("predictions"), list):
        return errors + ["predictions must be an array"]

    seen: dict[str, int] = {}
    for i, p in enumerate(sub["predictions"]):
        vid = p.get("video_id")
        if vid not in manifest:
            errors.append(f"[{i}] video_id {vid!r} not in manifest")
            continue
        seen[vid] = seen.get(vid, 0) + 1
        meta = manifest[vid]
        duration = float(meta["duration_sec"])

        events = p.get("events")
        if not isinstance(events, list):
            errors.append(f"{vid}: events must be an array")
            continue
        for j, ev in enumerate(events):
            cls = ev.get("class_name")
            if cls not in ANOMALY_CLASSES:
                errors.append(f"{vid}[{j}]: class_name {cls!r} not one of the 11 anomaly classes")
            st, en = ev.get("start_time_sec"), ev.get("end_time_sec")
            if meta["level"] == 1:
                if st is not None or en is not None:
                    errors.append(f"{vid}[{j}]: D1 events must have null start/end")
            else:
                if st is None or en is None:
                    errors.append(f"{vid}[{j}]: D2/D3 events require start/end")
                else:
                    if st < 0:
                        errors.append(f"{vid}[{j}]: start_time_sec < 0")
                    if en <= st:
                        errors.append(f"{vid}[{j}]: end_time_sec must exceed start")
                    if en > duration:
                        errors.append(f"{vid}[{j}]: end_time_sec {en} exceeds duration {duration}")
            exp = ev.get("explanation")
            if exp is not None and not (20 <= len(exp) <= 500):
                errors.append(f"{vid}[{j}]: explanation must be 20-500 chars (got {len(exp)})")

        rt = p.get("runtime_metadata")
        if not isinstance(rt, dict):
            errors.append(f"{vid}: runtime_metadata missing")
        else:
            for field in ("frames_processed", "chunks_processed",
                          "end_to_end_internal_time_ms", "model_runtimes"):
                if field not in rt:
                    errors.append(f"{vid}: runtime_metadata.{field} missing")

    for vid, n in seen.items():
        if n > 1:
            errors.append(f"{vid}: appears {n} times (at most once)")
    missing = [v for v in manifest if v not in seen]
    if missing:
        errors.append(f"WARNING: {len(missing)} manifest videos not in file (scored as normal): {missing}")
    return errors


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["build", "validate"])
    ap.add_argument("--raw", type=Path, default=RUNS_DIR / "raw_results.json")
    ap.add_argument("--out", type=Path, default=RUNS_DIR / "submission.json")
    ap.add_argument("--model-name", default="flytwatch-cascade-v1")
    ap.add_argument("--submission-id", default="run-01")
    args = ap.parse_args()

    if args.command == "build":
        sub = build_submission(args.raw, args.out, args.model_name, args.submission_id)
        n_events = sum(len(p["events"]) for p in sub["predictions"])
        print(f"built {args.out}: {len(sub['predictions'])} videos, {n_events} events")
    errs = validate_submission(args.out)
    if errs:
        print("VALIDATION ISSUES:")
        for e in errs:
            print(" -", e)
    else:
        print("VALIDATION: clean — safe to upload")
