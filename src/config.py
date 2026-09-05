"""Central configuration: label set, paths, thresholds.

Label strings must match ground_truth.csv exactly (per the dataset doc).
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
TEST_DIR = DATA_DIR / "test"
TRAIN_DIR = DATA_DIR / "train"
RUNS_DIR = PROJECT_ROOT / "runs"

# Exact label strings from the dataset doc — do not edit.
LABELS: list[str] = [
    "normal",
    "traffic_accident",
    "traffic_congestion",
    "stalled_or_broken_down_vehicle",
    "vehicle_blocking_traffic",
    "wrong_way_driving",
    "road_spill_or_debris",
    "waterlogging_or_flood",
    "fire",
    "smoke",
    "fighting_or_violence",
    "loitering_or_suspicious_presence",
]
ANOMALY_LABELS = [l for l in LABELS if l != "normal"]

# Ground-truth CSV columns (doc-specified): video_id, level, is_anomaly,
# class_name, start_time_sec, end_time_sec, description_summary.
GT_COLUMNS = [
    "video_id",
    "level",
    "is_anomaly",
    "class_name",
    "start_time_sec",
    "end_time_sec",
    "description_summary",
]

# --- Stage 1 (always-on cheap detector) ---
STAGE1_MODEL = "openai/clip-vit-base-patch32"
STAGE1_SAMPLE_FPS = 2.0          # frames/sec fed to stage 1
STAGE1_TRIGGER_THRESHOLD = 0.5   # anomaly score that arms verification
STAGE1_EMA_ALPHA = 0.6           # temporal smoothing of per-class scores
STAGE1_TRIGGER_CONSEC = 2        # consecutive above-threshold samples to trigger

# --- Stage 2 (small VLM verifier, triggered) ---
STAGE2_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
STAGE2_CLIP_FRAMES = 6           # frames per verification window
STAGE2_CLIP_SPREAD_SEC = 2.0     # spread of the window around trigger time
STAGE2_MAX_NEW_TOKENS = 128
STAGE2_CONFIRM_THRESHOLD = 0.55  # parsed confidence needed to confirm an event

# --- Temporal event state machine ---
EVENT_CONFIRM_SECS = 1.0         # sustained anomaly before an alert fires
EVENT_CLEAR_SECS = 5.0           # quiet time before an event is closed
EVENT_COOLDOWN_SECS = 10.0       # suppress duplicate alerts for same class
