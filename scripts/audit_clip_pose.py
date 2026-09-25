"""Remove clips where the presenter is not facing the camera.

The project plan admits only "single-anchor shots where the presenter faces the
camera directly", but pipeline 2 checked face count and identity and never pose,
so profile shots got in -- a Fatih Altayli clip measured yaw 0.83, effectively a
90-degree profile of him talking to someone off camera. There is no readable mouth
in a frame like that; it is mislabelled training data, not merely weak data.

presenter_filter.MAX_FACE_YAW now gates new clips. This cleans up what the older
filter already wrote, for each clip removing:
  * the clip mp4
  * its per-clip Whisper transcript JSON
  * its row in data/reporter_clips.csv

A PySceneDetect scene is one shot, so a clip's camera angle is constant and a
handful of frames classify the whole clip. The median across those frames is used
rather than the max, so a single glance mid-sentence does not condemn an otherwise
frontal shot.

    uv run python scripts/audit_clip_pose.py --dry-run
    uv run python scripts/audit_clip_pose.py --dry-run --reporter fatih_altayli
    uv run python scripts/audit_clip_pose.py --apply
"""

import argparse
import collections
import csv
import json
import statistics
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

import face_recognition  # noqa: E402
from presenter_filter import (face_yaw, presenter_sized_faces,  # noqa: E402
                               MAX_FACE_YAW, MIN_FACE_FRACTION)

CLIPS_CSV = ROOT / "data" / "reporter_clips.csv"
CACHE = ROOT / "data" / "_clip_yaw_cache.json"
TRANSCRIPTS = ROOT / "data" / "reporter_transcripts"
CLIP_FIELDS = ["reporter", "source_url", "source_video", "clip_file", "start_sec",
               "end_sec", "duration_sec", "transcript_text", "transcript_json"]


def clip_yaw(path: Path, frames: int = 3):
    """Median yaw over `frames` samples, or None if no frame could be measured.

    HOG runs on a half-size copy and the box is scaled back up for the landmark
    pass: detection is the expensive half and is ~4x cheaper at half resolution,
    while yaw still comes off full-resolution landmarks. Over thousands of clips
    that is the difference between forty minutes and most of a day.
    """
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return None
    yaws = []
    for i in range(frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(total * (i + 0.5) / frames))
        ok, frame = cap.read()
        if not ok:
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        half = cv2.resize(rgb, (0, 0), fx=0.5, fy=0.5)
        floor_px = half.shape[0] * MIN_FACE_FRACTION
        boxes = [tuple(v * 2 for v in b)
                 for b in face_recognition.face_locations(half, model="hog")
                 if (b[2] - b[0]) >= floor_px]
        if len(boxes) != 1:
            continue
        y = face_yaw(rgb, boxes[0])
        if y is not None:
            yaws.append(y)
    cap.release()
    return statistics.median(yaws) if yaws else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="actually delete")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--reporter", action="append", help="limit to these reporters")
    ap.add_argument("--max-yaw", type=float, default=MAX_FACE_YAW)
    ap.add_argument("--frames", type=int, default=3)
    args = ap.parse_args()
    if not args.apply and not args.dry_run:
        ap.error("pass --dry-run or --apply")

    with open(CLIPS_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    # Measuring 5000+ clips costs ~40 minutes, so the per-clip yaw is cached: a
    # later run at a different threshold, and the --apply pass itself, reuse it
    # instead of re-measuring. Delete the file to force a fresh measurement.
    cache = {}
    if CACHE.exists():
        cache = json.loads(CACHE.read_text(encoding="utf-8"))
        print(f"cache: {len(cache)} clips already measured")

    keep, drop, unmeasured = [], [], []
    stats = collections.defaultdict(lambda: [0, 0, 0])   # kept, dropped, unmeasured
    measured = 0
    for i, r in enumerate(rows, 1):
        rep = r["reporter"]
        if args.reporter and rep not in args.reporter:
            keep.append(r)
            continue
        key = r["clip_file"]
        path = ROOT / key
        if not path.exists():
            # Clip already gone; its row is stale either way.
            drop.append((r, None))
            stats[rep][1] += 1
            continue
        if key in cache:
            y = cache[key]
        else:
            y = clip_yaw(path, args.frames)
            cache[key] = y
            measured += 1
            if measured % 200 == 0:
                CACHE.write_text(json.dumps(cache), encoding="utf-8")
                print(f"  ... {i}/{len(rows)} ({measured} newly measured)", flush=True)
        if y is None:
            # No measurable presenter-sized face in any sampled frame. That fails
            # the same bar the live filter now applies, so it goes too.
            unmeasured.append(r)
            drop.append((r, None))
            stats[rep][2] += 1
        elif y > args.max_yaw:
            drop.append((r, y))
            stats[rep][1] += 1
        else:
            keep.append(r)
            stats[rep][0] += 1
    CACHE.write_text(json.dumps(cache), encoding="utf-8")

    # What other thresholds would cost, so the call can be made from one pass.
    ys = [v for k, v in cache.items() if v is not None]
    if ys:
        print("\nesik taramasi (olculen tum klipler):")
        for thr in (0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60):
            n = sum(1 for v in ys if v > thr)
            print(f"  yaw > {thr:.2f}: {n:5d} / {len(ys)} klip atilir ({100*n/len(ys):5.1f}%)")

    print(f"\n{'reporter':20} {'kalan':>7} {'yan cekim':>10} {'olculemedi':>11} {'atilan %':>9}")
    for rep in sorted(stats):
        k, d, u = stats[rep]
        tot = k + d + u
        print(f"{rep:20} {k:7d} {d:10d} {u:11d} {100*(d+u)/max(1,tot):8.1f}%")
    print(f"\ntotal {len(rows)} clips -> keep {len(keep)}, drop {len(drop)} "
          f"({len(unmeasured)} of them unmeasurable), threshold yaw > {args.max_yaw}")

    if not args.apply:
        print("dry run, nothing deleted")
        return

    removed_files = removed_json = 0
    for r, _ in drop:
        p = ROOT / r["clip_file"]
        if p.exists():
            p.unlink()
            removed_files += 1
        tj = (r.get("transcript_json") or "").strip()
        if tj:
            jp = ROOT / tj
            if jp.exists():
                jp.unlink()
                removed_json += 1

    tmp = CLIPS_CSV.with_suffix(".csv.tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CLIP_FIELDS)
        w.writeheader()
        for r in keep:
            w.writerow({k: r.get(k, "") for k in CLIP_FIELDS})
    tmp.replace(CLIPS_CSV)

    print(f"deleted {removed_files} clips and {removed_json} transcripts; "
          f"{CLIPS_CSV.name} now holds {len(keep)} rows")


if __name__ == "__main__":
    main()
