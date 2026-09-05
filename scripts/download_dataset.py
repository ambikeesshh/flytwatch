#!/usr/bin/env python3
"""Resilient multi-mirror downloader for the AHC hackathon dataset.

Google Drive rate-limits anonymous downloads; the organisers provide 5 mirror
copies. Each mirror is an independent file set with its own quota, so when one
mirror 429s we fall through to the next. Pass --cookies with a browser export
(NetScape cookies.txt format) to use your authenticated Google session, which
bypasses the anonymous quota entirely.

Usage:
  python scripts/download_dataset.py --part test --out data/
  python scripts/download_dataset.py --part all --out data/ --cookies cookies.txt
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

MIRRORS: dict[str, dict[str, str]] = {
    "m1": {
        "test": "1m2yA3eNZqPNNhLaMDCC6c70UOxbN3fFE",
        "train": "1M1NPZtcjkwtWDVvJ0YR-mftJRcBn4Qb0",
    },
    "m2": {
        "test": "1m3UCdbgCfJ7gH4wuVG4ZhpKXx680j7ae",
        "train": "1qzo2TUbu2hhsvLNVNb98RkTbuQdMcva4",
    },
    "m3": {
        "test": "1Kzql4M36X4jG_WtIL7mn6B7qKaip5Qnr",
        "train": "1c63OfTChcHVHI3YKJjd2xccvJVJB3Wum",
    },
    "m4": {
        "test": "1MzCwLXT0kusPDw75FakJtwCIhW4b103Z",
        "train": "1QdwbFBYdqwhGuGSfd8lUXbNAIG8P1NlB",
    },
    "m5": {
        "test": "1ZsaYxA4XfKYzjNraIDvaI-xQWerL0wyp",
        "train": "1K4uLHawEKcrV0pryhOdp7niGTZTAPj6Q",
    },
}

QUOTA_MARKERS = ("Too many users", "cannot retrieve", "Failed to retrieve")


def run_gdown(url: str, out: Path, cookies: Path | None, tries: int = 2) -> tuple[bool, str]:
    """Download one Drive folder/file. Returns (ok, log)."""
    cmd = [sys.executable, "-m", "gdown", "--folder", url, "-O", str(out), "--continue"]
    if cookies:
        cmd += ["--cookies", str(cookies)]
    log = ""
    for _ in range(tries):
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
        log = proc.stdout + proc.stderr
        if proc.returncode == 0 and not any(m in log for m in QUOTA_MARKERS):
            return True, log
        if any(m in log for m in QUOTA_MARKERS):
            return False, log  # quota: fall to next mirror immediately
        time.sleep(5)
    return False, log


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["test", "train", "all"], default="test")
    ap.add_argument("--out", type=Path, default=Path("data"))
    ap.add_argument("--cookies", type=Path, default=None, help="cookies.txt for authenticated downloads")
    ap.add_argument("--mirrors", nargs="*", default=list(MIRRORS), help="mirror order, e.g. m2 m3")
    args = ap.parse_args()

    parts = ["test", "train"] if args.part == "all" else [args.part]
    status = {}
    for part in parts:
        done = False
        for m in args.mirrors:
            if m not in MIRRORS:
                continue
            fid = MIRRORS[m][part]
            print(f"[{part}] trying mirror {m} ({fid}) ...")
            ok, log = run_gdown(f"https://drive.google.com/drive/folders/{fid}", args.out / part, args.cookies)
            status[f"{part}:{m}"] = "ok" if ok else "quota/failed"
            print(f"[{part}] mirror {m}: {'OK' if ok else 'failed'}")
            if ok:
                done = True
                break
        if not done:
            print(f"[{part}] ALL mirrors failed. Wait a few minutes and rerun, or use --cookies.")
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (args.out / "download_status.json").write_text(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
