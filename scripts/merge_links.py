"""Append rows from another links CSV into video_links.csv, by video id.

Idempotent and safe to re-run. Rows already present in video_links.csv are never
touched, so every `completed=1` flag survives -- which matters because
`run_all.mark_completed` rewrites the WHOLE file from the snapshot it read at
startup. A merge done while a run is live is undone the next time any row
finishes; re-running this script afterwards puts the new rows back without
resetting the progress made in between.

    uv run python scripts/merge_links.py video_links_balanced.csv
    uv run python scripts/merge_links.py video_links_balanced.csv --dry-run
"""

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from presenter_filter import extract_video_id  # noqa: E402

VIDEO_LINKS = ROOT / "video_links.csv"
LINK_FIELDS = ["url", "channel", "start", "end", "pipeline", "reporter", "completed"]


def read_rows(path: Path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_all_rows(rows, path: Path):
    # Write beside the target and replace, so an interrupted write cannot leave
    # video_links.csv truncated -- it is the only record of what is done.
    tmp = path.with_suffix(".csv.tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=LINK_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: (r.get(k) or "") for k in LINK_FIELDS})
    tmp.replace(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source", help="CSV whose rows should be added to video_links.csv")
    ap.add_argument("--target", default=str(VIDEO_LINKS))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    target = Path(args.target)
    source = Path(args.source)
    if not source.is_absolute():
        source = ROOT / source

    base = read_rows(target)
    incoming = read_rows(source)

    have = set()
    for r in base:
        vid = extract_video_id((r.get("url") or "").strip())
        if vid:
            have.add(vid)

    added, dup, bad = [], 0, 0
    for r in incoming:
        vid = extract_video_id((r.get("url") or "").strip())
        if not vid:
            bad += 1
            continue
        if vid in have:
            dup += 1
            continue
        have.add(vid)
        added.append({k: (r.get(k) or "") for k in LINK_FIELDS})

    done = sum(1 for r in base if (r.get("completed") or "").strip() == "1")
    print(f"{target.name}: {len(base)} rows ({done} completed)")
    print(f"{source.name}: {len(incoming)} rows -> {len(added)} new, "
          f"{dup} already present, {bad} unparseable")

    if args.dry_run:
        print("dry run, nothing written")
        return
    if not added:
        print("nothing to add")
        return

    write_all_rows(base + added, target)
    print(f"{target.name}: now {len(base) + len(added)} rows "
          f"({done} completed, untouched)")


if __name__ == "__main__":
    main()
