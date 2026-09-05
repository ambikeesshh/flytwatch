"""Prompt sets for stage 1 (CLIP scoring) and stage 2 (VLM verification).

CLIP benefits from template ensembles; the VLM needs an explicit,
structured task description that forces contextual reasoning and JSON output.
"""

from __future__ import annotations

from src.config import ANOMALY_LABELS

# Phrases per anomaly class, tuned for aerial/CCTV/dashcam viewpoints.
# v2: discriminative phrasing to break CLIP confusions seen in run-01
# (fire↔smoke, fighting↔loitering, accident↔wrong_way/stalled).
CLASS_PHRASES: dict[str, str] = {
    "traffic_accident": "a crashed vehicle and collision aftermath with wrecked damaged cars on a road",
    "traffic_congestion": "a long queue of vehicles stuck in a traffic jam on a road",
    "stalled_or_broken_down_vehicle": "one vehicle stopped alone on the roadside shoulder of a highway",
    "vehicle_blocking_traffic": "a single vehicle stopped in the middle of the road blocking the lane while other cars steer around it",
    "wrong_way_driving": "a vehicle driving against oncoming traffic on the wrong side of the road",
    "road_spill_or_debris": "loose debris rocks or spilled material scattered on the road surface",
    "waterlogging_or_flood": "a street flooded with standing water covering the road",
    "fire": "large bright orange flames of fire burning on or near the road",
    "smoke": "thick smoke rising over the area with no visible flames",
    "fighting_or_violence": "two or more people physically fighting and hitting each other in the street",
    "loitering_or_suspicious_presence": "a person standing idle and lingering on the sidewalk doing nothing",
    "normal": "an ordinary everyday street scene with normal traffic and nothing unusual happening",
}

TEMPLATES = [
    "a photo of {}",
    "a video frame of {}",
    "drone footage of {}",
    "surveillance camera view of {}",
]

NORMAL_PROMPTS = [t.format(CLASS_PHRASES["normal"]) for t in TEMPLATES]

# label -> ensemble of rendered prompts (normal handled separately)
CLASS_PROMPTS: dict[str, list[str]] = {
    c: [t.format(CLASS_PHRASES[c]) for t in TEMPLATES] for c in ANOMALY_LABELS
}

STAGE2_SYSTEM = (
    "You are a video anomaly verification agent watching footage from a camera "
    "over an urban area (the camera may be a drone, CCTV or dashcam). You receive "
    "a short sequence of frames captured around a potential anomaly. Decide "
    "whether a genuinely anomalous event is visible, bearing in mind that "
    "anomaly depends on context: a parked car is normal in a parking bay but "
    "anomalous on a highway shoulder; stopped traffic at a red light is normal; "
    "congestion on an open highway is not. Respond ONLY with a JSON object, no "
    "other text."
)

STAGE2_USER = """Frames from around timestamp {ts:.1f}s of video {video_id} are shown in time order.

Candidate anomaly type flagged by a cheap detector: {candidate}

Consider the full scene context, not just the candidate. Classes you may use:
{classes}

Respond with exactly this JSON schema:
{{"is_anomaly": true/false,
  "class_name": "<one of the classes above, or normal>",
  "confidence": <0.0-1.0>,
  "description": "<one sentence describing what is visible and why it is or is not anomalous>"}}"""
