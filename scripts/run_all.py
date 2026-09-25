"""Unified entry point: read video_links.csv and dispatch each row to pipeline 1 or 2.

video_links.csv columns: url, channel, start, end, pipeline, reporter
  pipeline == 1  -> download + Whisper ASR (existing pipeline, untouched)
  pipeline == 2  -> solo-presenter filter -> clips -> per-clip Whisper transcripts

Pipeline-2 source videos are DELETED as soon as their clips are exported, so a
full-CSV run never accumulates raw footage on disk (--keep-source opts out).

Full run (processes every row):
    uv run python scripts/run_all.py
    uv run python scripts/run_all.py --whisper-model small
    uv run python scripts/run_all.py --limit 10        # first 10 rows after filtering
    uv run python scripts/run_all.py --exclude-reporters cuneyt_ozdemir
    uv run python scripts/run_all.py --reporters cem_ogretir can_okanar
    uv run python scripts/run_all.py --pipeline 2

Quick test (shortest video per reporter for pipeline 2 + shortest pipeline-1 video):
    uv run python scripts/run_all.py --test
    uv run python scripts/run_all.py --test --reporters fatih_portakal pipeline1 --whisper-model base

For a full end-to-end smoke test over the first N CSV rows, use scripts/test_data_pipe.py.

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
               "start_sec", "end_sec", "duration_sec", "face_height_px",
               "transcript_text", "transcript_json"]
LINK_FIELDS = ["url", "channel", "start", "end", "pipeline", "reporter", "completed"]


# ---------- shared helpers ----------

def read_rows(links_csv=VIDEO_LINKS):
    with open(links_csv, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def col(row, key):
    return (row.get(key) or "").strip()


def by_pipeline(rows, value):
    return [r for r in rows if col(r, "pipeline") == value]


def is_completed(row):
    return col(row, "completed") == "1"


def write_all_rows(rows, links_csv=VIDEO_LINKS):
    """Persist the full video_links.csv (all columns), preserving the completed flag."""
    with open(links_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=LINK_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: (r.get(k) or "") for k in LINK_FIELDS})


def mark_completed(rows, row, links_csv=VIDEO_LINKS):
    """Flag one row completed and rewrite the CSV immediately (interrupt-safe resume)."""
    row["completed"] = "1"
    write_all_rows(rows, links_csv)


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
        cr["transcript_json"] = out_json.relative_to(PROJECT).as_posix()


# ---------- full run ----------

def run_pipeline1(selected, all_rows, links_csv, data_root, video_dir, output_dir,
                  model_name):
    pipe1 = [r for r in by_pipeline(selected, "1") if not is_completed(r)]
    if not pipe1:
        print("[pipeline1] no pending rows")
        return
    tmp = data_root / "_pipeline1_links.csv"
    write_pipeline1_csv(pipe1, tmp)
    print(f"[pipeline1] {len(pipe1)} pending rows -> download + Whisper")
    ok_keys = download_videos(set(), {}, link_file=str(tmp), video_dir=video_dir)
    run_whisper.transcribe_videos(video_dir=video_dir, output_dir=output_dir,
                                  model_name=model_name)
    # Only rows whose download actually succeeded are completed; a failed row stays
    # pending so the next run retries it instead of silently losing the video.
    done = [r for r in pipe1
            if (col(r, "url"), col(r, "start"), col(r, "end")) in ok_keys]
    for r in done:
        r["completed"] = "1"
    write_all_rows(all_rows, links_csv)
    failed = len(pipe1) - len(done)
    print(f"[pipeline1] marked {len(done)} rows completed"
          + (f"; {failed} row(s) FAILED and stay pending" if failed else ""))


def run_pipeline2(selected, all_rows, links_csv, model_name, clips_csv, downloads_dir,
                  clips_root, transcripts_dir, delete_source=True):
    pipe2 = [r for r in by_pipeline(selected, "2") if not is_completed(r)]
    if not pipe2:
        print("[pipeline2] no pending rows")
        return
    print(f"[pipeline2] {len(pipe2)} pending rows "
          f"(source videos {'deleted after clipping' if delete_source else 'KEPT'})")
    enc_cache = {}
    done = failed = 0
    total = len(pipe2)
    for i, r in enumerate(pipe2, start=1):
        reporter, url = col(r, "reporter"), col(r, "url")
        if not reporter or not url:
            continue
        img = pipeline2.reporter_image(reporter)
        if not img.exists():
            print(f"[pipeline2] skip {reporter}: missing image {img}")
            continue
        try:
            if reporter not in enc_cache:
                enc_cache[reporter] = load_reference_encoding(img)
            clips_dir = clips_root / f"{reporter}_solo_clips"
            clip_rows = pipeline2.select_clips(url, reporter, enc_cache[reporter],
                                               downloads_dir, clips_dir,
                                               delete_source=delete_source)
            transcribe_clips_into(clip_rows, transcripts_dir, model_name)
            append_clip_rows(clips_csv, clip_rows)
            # persist after each video -> safe to interrupt/resume
            mark_completed(all_rows, r, links_csv)
            done += 1
            print(f"[pipeline2] [{i}/{total}] {reporter}: {len(clip_rows)} clips "
                  f"from {url} (completed)")
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            # A single unavailable/geo-blocked/deleted video must not abort a
            # 4000-row run. The row stays pending so a later run retries it.
            failed += 1
            print(f"[pipeline2] [{i}/{total}] FAILED {reporter} {url}: "
                  f"{type(exc).__name__}: {exc}")
    print(f"[pipeline2] finished: {done} completed, {failed} failed (left pending)")


def select_rows(all_rows, pipelines=None, reporters=None, exclude_reporters=None,
                limit=None):
    """Filter first, then take the first `limit` of what survives.

    Same order as colab_run_all.ipynb, and it matters: cuneyt_ozdemir occupies CSV rows
    235-4641, so without a reporter filter a plain run reaches cem_ogretir/can_okanar
    only after ~4400 videos of one speaker.
    """
    rows = all_rows
    if pipelines:
        rows = [r for r in rows if col(r, "pipeline") in pipelines]
    if reporters:
        rows = [r for r in rows
                if col(r, "pipeline") != "2" or col(r, "reporter") in reporters]
    if exclude_reporters:
        rows = [r for r in rows
                if col(r, "pipeline") != "2" or col(r, "reporter") not in exclude_reporters]
    return rows[:limit] if limit else rows


def run_full(model_name, links_csv=VIDEO_LINKS, data_root=DATA, limit=None,
             delete_source=True, pipelines=None, reporters=None,
             exclude_reporters=None):
    all_rows = read_rows(links_csv)
    selected = select_rows(all_rows, pipelines, reporters, exclude_reporters, limit)
    if len(selected) != len(all_rows):
        pending = sum(1 for r in selected if not is_completed(r))
        print(f"[run_all] {len(selected)} of {len(all_rows)} CSV rows selected "
              f"({pending} still pending)")
    run_pipeline1(selected, all_rows, links_csv, data_root,
                  video_dir=str(data_root / "raw_videos"),
                  output_dir=str(data_root / "transcripts"), model_name=model_name)
    run_pipeline2(selected, all_rows, links_csv, model_name=model_name,
                  clips_csv=data_root / "reporter_clips.csv",
                  downloads_dir=data_root / "pipeline2_raw",
                  clips_root=data_root,
                  transcripts_dir=data_root / "reporter_transcripts",
                  delete_source=delete_source)


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
                    help="only these reporter slugs (with --test also accepts 'pipeline1')")
    ap.add_argument("--exclude-reporters", nargs="*",
                    help="skip these reporter slugs, e.g. --exclude-reporters cuneyt_ozdemir")
    ap.add_argument("--pipeline", nargs="*", choices=["1", "2"],
                    help="only rows of these pipelines")
    ap.add_argument("--limit", type=int,
                    help="process only the first N rows left after the other filters")
    ap.add_argument("--keep-source", action="store_true",
                    help="keep pipeline-2 source videos after clipping (default: delete "
                         "them so disk usage stays flat)")
    args = ap.parse_args()

    if args.test:
        run_test(args.whisper_model, only=set(args.reporters) if args.reporters else None)
    else:
        run_full(args.whisper_model, limit=args.limit,
                 delete_source=not args.keep_source,
                 pipelines=set(args.pipeline) if args.pipeline else None,
                 reporters=set(args.reporters) if args.reporters else None,
                 exclude_reporters=(set(args.exclude_reporters)
                                    if args.exclude_reporters else None))


if __name__ == "__main__":
    main()
