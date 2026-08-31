"""Smoke test: run BOTH pipelines over the first N rows of video_links.csv.

Nothing in the real dataset is touched:
  * the selected rows are copied to data/test_run/video_links_test.csv, so the
    `completed` flags in the real video_links.csv stay untouched;
  * every output (videos, clips, transcripts) lands under data/test_run/.

It then verifies the disk-space rule that the real run depends on: pipeline-2
source videos must be gone once their clips are exported.

Run:  venv/Scripts/python scripts/test_data_pipe.py
      venv/Scripts/python scripts/test_data_pipe.py --rows 4 --whisper-model small
      venv/Scripts/python scripts/test_data_pipe.py --keep-source   # inspect sources
"""

import argparse
import csv
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_all  # noqa: E402

PROJECT = Path(__file__).resolve().parent.parent
TEST_ROOT = PROJECT / "data" / "test_run"
TEST_CSV = TEST_ROOT / "video_links_test.csv"
VIDEO_EXT = (".mp4", ".webm", ".mkv", ".mov")


def make_test_csv(rows_wanted):
    """Copy the first N rows of the real CSV into an isolated test CSV."""
    all_rows = run_all.read_rows(run_all.VIDEO_LINKS)
    selected = all_rows[:rows_wanted]
    TEST_ROOT.mkdir(parents=True, exist_ok=True)
    with open(TEST_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=run_all.LINK_FIELDS)
        w.writeheader()
        for r in selected:
            row = {k: (r.get(k) or "") for k in run_all.LINK_FIELDS}
            row["completed"] = ""       # always re-run the test rows
            w.writerow(row)
    return selected


def describe(selected):
    print(f"\n=== test selection: first {len(selected)} rows of video_links.csv ===")
    for i, r in enumerate(selected, start=1):
        pl = run_all.col(r, "pipeline")
        who = run_all.col(r, "reporter") or run_all.col(r, "channel")
        start, end = run_all.col(r, "start"), run_all.col(r, "end")
        span = f"{start}-{end}" if start or end else "full video"
        print(f"  {i:2d}. p{pl}  {who:20s} {run_all.col(r, 'url')}  {span}")
    p1 = sum(1 for r in selected if run_all.col(r, "pipeline") == "1")
    p2 = sum(1 for r in selected if run_all.col(r, "pipeline") == "2")
    print(f"  -> pipeline1={p1}, pipeline2={p2}\n")


def count_files(folder: Path, suffixes=None):
    if not folder.exists():
        return []
    return [p for p in folder.rglob("*")
            if p.is_file() and (suffixes is None or p.suffix.lower() in suffixes)]


def dir_size_mb(folder: Path):
    if not folder.exists():
        return 0.0
    return sum(p.stat().st_size for p in folder.rglob("*") if p.is_file()) / (1024 * 1024)


def report(delete_source):
    """Summarize what the run produced and check the source-deletion rule."""
    p2_raw = TEST_ROOT / "pipeline2_raw"
    leftovers = count_files(p2_raw, VIDEO_EXT)
    clips = [p for p in TEST_ROOT.glob("*_solo_clips/*.mp4")]
    p1_videos = count_files(TEST_ROOT / "raw_videos", VIDEO_EXT)
    p1_json = count_files(TEST_ROOT / "transcripts", {".json"})
    p2_json = count_files(TEST_ROOT / "reporter_transcripts", {".json"})
    manifest = TEST_ROOT / "reporter_clips.csv"
    manifest_rows = 0
    if manifest.exists():
        with open(manifest, newline="", encoding="utf-8") as f:
            manifest_rows = sum(1 for _ in csv.DictReader(f))

    print("\n" + "=" * 62)
    print("TEST SUMMARY")
    print("=" * 62)
    print(f"pipeline1  videos downloaded : {len(p1_videos)}")
    print(f"pipeline1  transcripts (json): {len(p1_json)}")
    print(f"pipeline2  clips exported    : {len(clips)}")
    print(f"pipeline2  transcripts (json): {len(p2_json)}")
    print(f"pipeline2  manifest rows     : {manifest_rows}  ({manifest.name})")
    print(f"test dir total size          : {dir_size_mb(TEST_ROOT):.1f} MB")

    print("\n--- disk-space check: pipeline-2 source videos ---")
    if not delete_source:
        print(f"SKIPPED (--keep-source): {len(leftovers)} source videos kept in {p2_raw}")
        return True
    if leftovers:
        print(f"FAIL: {len(leftovers)} source video(s) still on disk in {p2_raw}:")
        for p in leftovers:
            print(f"   - {p.name}  ({p.stat().st_size / (1024 * 1024):.1f} MB)")
        return False
    print(f"PASS: no source videos left in {p2_raw} "
          f"(size {dir_size_mb(p2_raw):.1f} MB) - clips only")
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rows", type=int, default=10,
                    help="how many rows from the top of video_links.csv (default 10)")
    ap.add_argument("--whisper-model", default="base",
                    help="Whisper model for the test (default 'base' - fast; the real "
                         "run uses large-v3)")
    ap.add_argument("--keep-source", action="store_true",
                    help="keep pipeline-2 source videos (skips the disk-space check)")
    ap.add_argument("--fresh", action="store_true",
                    help="wipe data/test_run/ before running")
    args = ap.parse_args()

    if args.fresh and TEST_ROOT.exists():
        print(f"[test] wiping {TEST_ROOT}")
        shutil.rmtree(TEST_ROOT)

    selected = make_test_csv(args.rows)
    describe(selected)
    print(f"[test] outputs   -> {TEST_ROOT}")
    print(f"[test] real CSV  -> untouched ({run_all.VIDEO_LINKS.name})")
    print(f"[test] whisper   -> {args.whisper_model}\n")

    run_all.run_full(args.whisper_model, links_csv=TEST_CSV, data_root=TEST_ROOT,
                     limit=args.rows, delete_source=not args.keep_source)

    ok = report(not args.keep_source)
    print("\nRESULT:", "OK - ready for the full run" if ok else "PROBLEM - see above")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
