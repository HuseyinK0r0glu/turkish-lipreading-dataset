"""Unified entry point: read video_links.csv and dispatch each row to pipeline 1 or 2.

video_links.csv columns: url, channel, start, end, pipeline, reporter
  pipeline == 1  -> download + Whisper ASR (existing pipeline, untouched)
  pipeline == 2  -> solo-presenter filter -> clips -> per-clip Whisper transcripts

Full run (processes every row):
    venv/bin/python scripts/run_all.py
    venv/bin/python scripts/run_all.py --whisper-model small

Quick test (shortest video per reporter for pipeline 2 + shortest pipeline-1 video):
    venv/bin/python scripts/run_all.py --test
    venv/bin/python scripts/run_all.py --test --reporters fatih_portakal pipeline1 --whisper-model base

Pipeline-2 results land in a single CSV (data/reporter_clips.csv): one row per kept
clip with its transcript text + transcript JSON path. Test outputs go to data/quick_test/.

NOTE: pipeline 1 (download_videos.py) shells out to a bare `yt-dlp`, so run with the
venv active (or venv/bin on PATH) when pipeline-1 rows are involved.
"""

import argparse
import csv
import os
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from download_videos import download_videos, parse_time_to_seconds  # noqa: E402
import run_whisper  # noqa: E402
import pipeline2  # noqa: E402
from presenter_filter import load_reference_encoding, extract_video_id  # noqa: E402

PROJECT = Path(__file__).resolve().parent.parent
VIDEO_LINKS = PROJECT / "video_links.csv"
DATA = PROJECT / "data"
QUICK_TEST = DATA / "quick_test"
CLIP_FIELDS = ["reporter", "source_url", "source_video", "clip_file",
               "start_sec", "end_sec", "duration_sec",
               "transcript_text", "transcript_json"]
LINK_FIELDS = ["url", "channel", "start", "end", "pipeline", "reporter", "completed"]


# ---------- shared helpers ----------

def read_rows():
    with open(VIDEO_LINKS, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def col(row, key):
    return (row.get(key) or "").strip()


def by_pipeline(rows, value):
    return [r for r in rows if col(r, "pipeline") == value]


def is_completed(row):
    return col(row, "completed") == "1"


def write_all_rows(rows):
    """Persist the full video_links.csv (all columns), preserving the completed flag."""
    with open(VIDEO_LINKS, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=LINK_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: (r.get(k) or "") for k in LINK_FIELDS})


def mark_completed(rows, row):
    """Flag one row completed and rewrite the CSV immediately (interrupt-safe resume)."""
    row["completed"] = "1"
    write_all_rows(rows)


def write_pipeline1_csv(rows, out_csv: Path):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["url", "channel", "start", "end"])
        w.writeheader()
        for r in rows:
            w.writerow({k: col(r, k) for k in ["url", "channel", "start", "end"]})


def append_clip_rows(csv_path: Path, rows):
    if not rows:
        return
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CLIP_FIELDS)
        if is_new:
            w.writeheader()
        for r in rows:
            w.writerow(r)


def get_durations_batch(urls):
    """Fetch durations for many URLs in a single yt-dlp call. Returns {video_id: seconds}."""
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8")
    try:
        tmp.write("\n".join(urls))
        tmp.close()
        res = subprocess.run(
            [sys.executable, "-m", "yt_dlp", "--no-warnings",
             "--print", "%(id)s\t%(duration)s", "--batch-file", tmp.name],
            capture_output=True, text=True,
        )
    finally:
        os.unlink(tmp.name)
    result = {}
    for line in res.stdout.splitlines():
        parts = line.strip().split("\t")
        if len(parts) == 2 and parts[1].isdigit():
            result[parts[0]] = int(parts[1])
    return result


def transcribe_clips_into(rows, transcripts_dir: Path, model_name):
    for cr in rows:
        clip = PROJECT / cr["clip_file"]
        out_json = transcripts_dir / (Path(cr["clip_file"]).stem + ".json")
        cr["transcript_text"] = pipeline2.transcribe_clip(clip, out_json, model_name)
        cr["transcript_json"] = str(out_json.relative_to(PROJECT))


# ---------- full run ----------

def run_pipeline1(rows, video_dir, output_dir, model_name):
    pipe1 = [r for r in by_pipeline(rows, "1") if not is_completed(r)]
    if not pipe1:
        print("[pipeline1] no pending rows")
        return
    tmp = DATA / "_pipeline1_links.csv"
    write_pipeline1_csv(pipe1, tmp)
    print(f"[pipeline1] {len(pipe1)} pending rows -> download + Whisper")
    download_videos(set(), {}, link_file=str(tmp), video_dir=video_dir)
    run_whisper.transcribe_videos(video_dir=video_dir, output_dir=output_dir,
                                  model_name=model_name)
    for r in pipe1:
        r["completed"] = "1"
    write_all_rows(rows)
    print(f"[pipeline1] marked {len(pipe1)} rows completed")


def run_pipeline2(rows, model_name, clips_csv, downloads_dir, clips_root, transcripts_dir):
    pipe2 = [r for r in by_pipeline(rows, "2") if not is_completed(r)]
    if not pipe2:
        print("[pipeline2] no pending rows")
        return
    print(f"[pipeline2] {len(pipe2)} pending rows")
    enc_cache = {}
    for r in pipe2:
        reporter, url = col(r, "reporter"), col(r, "url")
        if not reporter or not url:
            continue
        img = pipeline2.reporter_image(reporter)
        if not img.exists():
            print(f"[pipeline2] skip {reporter}: missing image {img}")
            continue
        if reporter not in enc_cache:
            enc_cache[reporter] = load_reference_encoding(img)
        clips_dir = clips_root / f"{reporter}_solo_clips"
        clip_rows = pipeline2.select_clips(url, reporter, enc_cache[reporter],
                                           downloads_dir, clips_dir)
        transcribe_clips_into(clip_rows, transcripts_dir, model_name)
        append_clip_rows(clips_csv, clip_rows)
        mark_completed(rows, r)   # persist after each video -> safe to interrupt/resume
        print(f"[pipeline2] {reporter}: {len(clip_rows)} clips from {url} (completed)")


def run_full(model_name):
    rows = read_rows()
    run_pipeline1(rows, video_dir=str(DATA / "raw_videos"),
                  output_dir=str(DATA / "transcripts"), model_name=model_name)
    run_pipeline2(rows, model_name=model_name,
                  clips_csv=DATA / "reporter_clips.csv",
                  downloads_dir=DATA / "pipeline2_raw",
                  clips_root=DATA,
                  transcripts_dir=DATA / "reporter_transcripts")


# ---------- quick test ----------

def run_test(model_name, only=None):
    rows = read_rows()
    QUICK_TEST.mkdir(parents=True, exist_ok=True)

    # pipeline 2: shortest video per reporter
    pipe2 = by_pipeline(rows, "2")
    by_reporter = {}
    for r in pipe2:
        by_reporter.setdefault(col(r, "reporter"), []).append(r)

    print("=== pipeline 2: shortest video per reporter ===")
    for reporter, reporter_rows in by_reporter.items():
        if only and reporter not in only:
            continue
        img = pipeline2.reporter_image(reporter)
        if not img.exists():
            print(f"[test] {reporter}: missing image {img}")
            continue
        urls = [col(r, "url") for r in reporter_rows]
        print(f"[test] {reporter}: fetching durations for {len(urls)} videos (one batch call)...")
        dur_map = get_durations_batch(urls)
        best, best_d = None, float("inf")
        for r in reporter_rows:
            vid_id = extract_video_id(col(r, "url"))
            d = dur_map.get(vid_id)
            if d and d < best_d:
                best, best_d = r, d
        if not best:
            print(f"[test] {reporter}: could not read any durations")
            continue
        enc = load_reference_encoding(img)
        work = QUICK_TEST / reporter
        clip_rows = pipeline2.select_clips(col(best, "url"), reporter, enc,
                                           work / "raw", work / "clips")
        transcribe_clips_into(clip_rows, work / "transcripts", model_name)
        append_clip_rows(QUICK_TEST / "reporter_clips_test.csv", clip_rows)
        print(f"[test] {reporter}: shortest={col(best,'url')} ({best_d}s) "
              f"-> {len(clip_rows)} clips kept")

    # pipeline 1: single shortest video
    if not only or "pipeline1" in only:
        pipe1 = by_pipeline(rows, "1")
        if pipe1:
            # pipeline-1 rows already have start/end times; fall back to batch query
            needs_fetch = [r for r in pipe1 if not (col(r, "start") and col(r, "end"))]
            if needs_fetch:
                print(f"[test] pipeline1: fetching durations for {len(needs_fetch)} rows...")
                dur_map = get_durations_batch([col(r, "url") for r in needs_fetch])
            else:
                dur_map = {}

            def pipe1_len(row):
                s, e = col(row, "start"), col(row, "end")
                if s and e:
                    a, b = parse_time_to_seconds(s), parse_time_to_seconds(e)
                    if a is not None and b is not None and b > a:
                        return b - a
                vid_id = extract_video_id(col(row, "url"))
                return dur_map.get(vid_id, float("inf"))

        if pipe1:
            shortest = min(pipe1, key=pipe1_len)
            print(f"\n=== pipeline 1: shortest video {col(shortest,'url')} "
                  f"({col(shortest,'start') or 'full'}-{col(shortest,'end')}) ===")
            tmp = QUICK_TEST / "_pipeline1_links.csv"
            write_pipeline1_csv([shortest], tmp)
            vdir = str(QUICK_TEST / "pipeline1_raw")
            download_videos(set(), {}, link_file=str(tmp), video_dir=vdir)
            run_whisper.transcribe_videos(video_dir=vdir,
                                          output_dir=str(QUICK_TEST / "pipeline1_transcripts"),
                                          model_name=model_name)

    print(f"\nTest outputs in {QUICK_TEST}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--test", action="store_true",
                    help="shortest video per reporter (pipeline 2) + shortest pipeline-1 video")
    ap.add_argument("--whisper-model", default="large-v3",
                    help="Whisper model (large-v3 default; use base/small for a fast test)")
    ap.add_argument("--reporters", nargs="*",
                    help="limit --test to these reporter slugs and/or 'pipeline1'")
    args = ap.parse_args()

    if args.test:
        run_test(args.whisper_model, only=set(args.reporters) if args.reporters else None)
    else:
        run_full(args.whisper_model)


if __name__ == "__main__":
    main()
