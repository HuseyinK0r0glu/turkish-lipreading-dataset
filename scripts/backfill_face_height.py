"""Fill face_height_px on manifest rows written before the column existed.

New clips get it from `presenter_filter.solo_segments`, which already measures the
face while sampling. Rows produced earlier have the field empty, which would leave
Phase 2/3 unable to apply a size threshold to most of the corpus, so this measures
them from the clip files themselves.

A clip is one camera shot, so a few frames characterise it; the median is used so a
frame where the detector misses does not skew the value. Rows already carrying a
value are left alone, which makes the script resumable -- it writes the manifest
every BATCH rows, so an interrupted run loses at most that much work.

DO NOT RUN THIS WHILE run_all.py IS RUNNING. Both write reporter_clips.csv -- this
one rewrites the whole file from a snapshot it took at startup, run_all appends to
it -- so a concurrent pass would silently drop every clip produced in the meantime.
Stop the run first. There is no hurry: new clips already carry the column, so this
is pure catch-up on older rows and can wait for any convenient pause. Measured at
roughly 3 s per clip on cold files, so budget hours rather than minutes for a full
corpus.

    uv run python scripts/backfill_face_height.py --dry-run
    uv run python scripts/backfill_face_height.py
"""

import argparse
import collections
import csv
import statistics
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from presenter_filter import presenter_sized_faces  # noqa: E402

CLIPS_CSV = ROOT / "data" / "reporter_clips.csv"
CLIP_FIELDS = ["reporter", "source_url", "source_video", "clip_file", "start_sec",
               "end_sec", "duration_sec", "face_height_px", "transcript_text",
               "transcript_json"]
BATCH = 200


def measure(path: Path, frames: int = 3):
    """Median presenter face height in px, or None if no frame could be measured."""
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return None
    heights = []
    for i in range(frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(total * (i + 0.5) / frames))
        ok, frame = cap.read()
        if not ok:
            continue
        boxes = presenter_sized_faces(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if len(boxes) == 1:
            heights.append(boxes[0][2] - boxes[0][0])
    cap.release()
    return int(statistics.median(heights)) if heights else None


def write_all(rows):
    tmp = CLIPS_CSV.with_suffix(".csv.tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CLIP_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in CLIP_FIELDS})
    tmp.replace(CLIPS_CSV)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--frames", type=int, default=3)
    args = ap.parse_args()

    with open(CLIPS_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    todo = [r for r in rows if not (r.get("face_height_px") or "").strip()]
    print(f"{CLIPS_CSV.name}: {len(rows)} rows, {len(todo)} missing face_height_px")
    if args.dry_run:
        print("dry run, nothing written")
        return
    if not todo:
        print("nothing to do")
        return

    done = missing = 0
    for i, r in enumerate(todo, 1):
        p = ROOT / r["clip_file"]
        h = measure(p, args.frames) if p.exists() else None
        # Empty, not 0: 0 would read as a measured tiny face. Empty means unknown,
        # which is what an unreadable or vanished clip actually is.
        r["face_height_px"] = h if h else ""
        if h:
            done += 1
        else:
            missing += 1
        if i % BATCH == 0:
            write_all(rows)
            print(f"  ... {i}/{len(todo)} ({done} measured, {missing} unmeasurable)",
                  flush=True)
    write_all(rows)

    vals = [int(r["face_height_px"]) for r in rows
            if (r.get("face_height_px") or "").strip()]
    print(f"\nmeasured {done}, unmeasurable {missing}")
    if vals:
        vals.sort()
        print(f"face height px: median {statistics.median(vals)}  "
              f"p10 {vals[len(vals)//10]}  p90 {vals[9*len(vals)//10]}")
        for thr in (150, 220, 260):
            n = sum(1 for v in vals if v < thr)
            print(f"  under {thr}px: {n}/{len(vals)} ({100*n/len(vals):.0f}%)")
        by = collections.defaultdict(list)
        for r in rows:
            v = (r.get("face_height_px") or "").strip()
            if v:
                by[r["reporter"]].append(int(v))
        print()
        for rep in sorted(by, key=lambda x: statistics.median(by[x])):
            print(f"  {rep:20} n={len(by[rep]):5d}  median {statistics.median(by[rep]):.0f}px")


if __name__ == "__main__":
    main()
