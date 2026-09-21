"""Push one reporter's pending rows to the end of video_links.csv.

`run_all.run_pipeline2` walks pending rows in file order, so a reporter that owns
a contiguous block in the middle of the CSV is processed before everything after
it. cuneyt_ozdemir is 4407 of 6602 rows, which means an unfiltered run spends
months on that one speaker before it reaches anything else -- exactly the
imbalance the rest of the corpus is meant to correct.

This only reorders. No row is added, dropped or edited, and every `completed`
flag travels with its row, so it is safe to run between runs:

    uv run python scripts/reorder_links.py --last cuneyt_ozdemir
    uv run python scripts/reorder_links.py --last cuneyt_ozdemir --dry-run

Completed rows are grouped first (they are skipped anyway), then pending rows for
everyone else, then the deprioritised reporter's pending rows.
"""

import argparse
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VIDEO_LINKS = ROOT / "video_links.csv"
LINK_FIELDS = ["url", "channel", "start", "end", "pipeline", "reporter", "completed"]


def read_rows(path: Path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_all_rows(rows, path: Path):
    # Write beside the target and replace: video_links.csv is the only record of
    # what is done, so a half-written file would be unrecoverable.
    tmp = path.with_suffix(".csv.tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=LINK_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: (r.get(k) or "") for k in LINK_FIELDS})
    tmp.replace(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--last", action="append", required=True,
                    help="reporter whose pending rows go to the end (repeatable)")
    ap.add_argument("--target", default=str(VIDEO_LINKS))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    target = Path(args.target)
    last = set(args.last)
    rows = read_rows(target)

    done, pending, deferred = [], [], []
    for r in rows:
        if (r.get("completed") or "").strip() == "1":
            done.append(r)
        elif (r.get("reporter") or "").strip() in last:
            deferred.append(r)
        else:
            pending.append(r)

    ordered = done + pending + deferred
    assert len(ordered) == len(rows)

    print(f"{target.name}: {len(rows)} rows")
    print(f"  completed (skipped by run_all) : {len(done)}")
    print(f"  pending, runs first            : {len(pending)}")
    print(f"  pending, deferred to the end   : {len(deferred)}  ({', '.join(sorted(last))})")

    if args.dry_run:
        print("dry run, nothing written")
        return
    write_all_rows(ordered, target)
    print("reordered")


if __name__ == "__main__":
    main()
