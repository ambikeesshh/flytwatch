"""Live demo dashboard — plays test videos with real-time detection overlay.

Serves the test videos and the detector's predictions; the browser plays a
video at 1x while the frontend pops alerts exactly when our system flagged
them (the same events as in the submission JSON), with the stage-1 score
timeline underneath. This is the "someone can act while the drone is still
overhead" story made visible.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from flask import Flask, Response, jsonify, render_template, request

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import RUNS_DIR, TEST_DIR

app = Flask(__name__)

VIDEOS_DIR = TEST_DIR / "videos"
MANIFEST = PROJECT_ROOT / "data" / "benchmark_manifest.json"


def load_predictions() -> dict[str, list[dict]]:
    """video_id -> events, from the latest submission file."""
    for cand in (RUNS_DIR / "submission.json", RUNS_DIR / "predictions.json"):
        if cand.exists():
            sub = json.loads(cand.read_text())
            preds = sub.get("predictions", sub if isinstance(sub, list) else [])
            out: dict[str, list[dict]] = {}
            for p in preds:
                out[p["video_id"]] = p.get("events", [])
            return out
    return {}


@app.get("/")
def index():
    videos = []
    if MANIFEST.exists():
        man = json.loads(MANIFEST.read_text())
        preds = load_predictions()
        for v in man["videos"]:
            videos.append({
                "id": v["video_id"],
                "level": v["level"],
                "duration": v["duration_sec"],
                "n_events": len(preds.get(v["video_id"], [])),
            })
    return render_template("index.html", videos=videos)


@app.get("/api/events/<video_id>")
def events(video_id: str):
    return jsonify(load_predictions().get(video_id, []))


@app.get("/video/<video_id>")
def video(video_id: str):
    path = VIDEOS_DIR / f"{video_id}.mp4"
    if not path.exists():
        return {"error": "not found"}, 404

    def stream(start: int, end: int):
        with open(path, "rb") as f:
            f.seek(start)
            remaining = end - start + 1
            while chunk := f.read(min(1024 * 512, remaining)):
                remaining -= len(chunk)
                yield chunk

    size = path.stat().st_size
    range_header = request.headers.get("Range")
    if range_header:
        start_s, end_s = range_header.replace("bytes=", "").split("-")
        start = int(start_s) if start_s else 0
        end = int(end_s) if end_s else size - 1
        return Response(stream(start, end), 206, mimetype="video/mp4", content_type="video/mp4",
                        headers={"Content-Range": f"bytes {start}-{end}/{size}",
                                 "Accept-Ranges": "bytes", "Content-Length": str(end - start + 1)})
    return Response(stream(0, size - 1), 200, mimetype="video/mp4",
                    headers={"Accept-Ranges": "bytes", "Content-Length": str(size)})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5050, threaded=True)
