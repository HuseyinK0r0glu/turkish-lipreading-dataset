"""Clear `completed` on pipeline-2 rows that produced no clips, so they are retried.

`run_pipeline2` marks a row completed whether or not any clip came out of it, so a
row the filter rejected outright is never retried. That was fine while a rejection
meant "this video really has no solo footage", but the all-or-nothing version of
`scene_has_only_target` also rejected whole cut-free broadcasts over a single
two-face frame (see CLAUDE.md invariant 2). Those rows are recoverable now that
`solo_segments` cuts scenes instead of voting on them.

Only rows with ZERO clips in data/reporter_clips.csv are reset. A row that yielded
some clips is left alone: re-running it would export overlapping segments alongside
the ones already in the manifest.

    uv run python scripts/reset_empty_rows.py --dry-run
    uv run python scripts/reset_empty_rows.py
"""

import argparse
import collections
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from presenter_filter import extract_video_id  # noqa: E402

VIDEO_LINKS = ROOT / "video_links.csv"
CLIPS_CSV = ROOT / "data" / "reporter_clips.csv"
LINK_FIELDS = ["url", "channel", "start", "end", "pipeline", "reporter", "completed"]


def write_all_rows(rows, path: Path):
    tmp = path.with_suffix(".csv.tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=LINK_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: (r.get(k) or "") for k in LINK_FIELDS})
    tmp.replace(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    clips = collections.Counter()
    if CLIPS_CSV.exists():
        with open(CLIPS_CSV, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                vid = extract_video_id((r.get("source_url") or "").strip())
                if vid:
                    clips[vid] += 1

    with open(VIDEO_LINKS, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    reset = []
    for r in rows:
        if (r.get("pipeline") or "").strip() != "2":
            continue
        if (r.get("completed") or "").strip() != "1":
            continue
        vid = extract_video_id((r.get("url") or "").strip())
        if vid and clips.get(vid, 0) == 0:
            reset.append(r)

    by_rep = collections.Counter((r.get("reporter") or "") for r in reset)
    print(f"completed pipeline-2 rows with zero clips: {len(reset)}")
    for rep, n in by_rep.most_common():
        print(f"   {rep:20} {n}")

    if args.dry_run:
        print("dry run, nothing written")
        return
    if not reset:
        print("nothing to reset")
        return

    for r in reset:
        r["completed"] = ""
    write_all_rows(rows, VIDEO_LINKS)
    still = sum(1 for r in rows if (r.get("completed") or "").strip() == "1")
    print(f"reset {len(reset)} rows to pending; {still} rows still completed")


if __name__ == "__main__":
    main()
