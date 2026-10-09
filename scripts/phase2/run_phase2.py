"""Phase 2 entry point: solo clips -> 88x88 mouth crops + word/sentence manifests.

Reads a snapshot of data/reporter_clips.csv (never writes it), skips source clips
already in data/phase2/face_track_report.csv, and processes the rest one clip per
worker process. Each worker runs MediaPipe on the CPU and SyncNet on the GPU.

    uv run python scripts/phase2/run_phase2.py --reporters kubra_par --limit 20
    uv run python scripts/phase2/run_phase2.py --exclude-reporters cuneyt_ozdemir
    uv run python scripts/phase2/run_phase2.py --workers 6

Outputs (PHASE2_PLAN §2): data/lip_crops/<reporter>/<clip stem>.npz and
data/phase2/{word_clips,sentence_clips,syncnet_scores,discarded,face_track_report}.csv.
"""
import argparse
import os
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]  # phase2 modules, then scripts/ (Phase 1)

import torch  # noqa: E402

import manifests  # noqa: E402
import segments  # noqa: E402
import source_clip  # noqa: E402

SYNC_MIN = 3.0                # project plan: discard source clips with SyncNet score < 3
# Off-screen speech (PHASE2_PLAN §4.5.1). calibrate_offscreen.py, 38 clips over 23
# reporters with the second half's audio swapped for another reporter's: own-audio
# words median 6.3 (p5 3.6), foreign-audio words median -0.3 (p95 2.0). At 2.5 (the
# Youden optimum) 98.8% of real words are kept and 97.6% of foreign ones dropped.
# The clip-level score alone let 21 of those 38 half-foreign clips through.
WORD_SYNC_MIN = 2.5
LIP_MOTION_MIN = None         # recorded per word; no calibrated threshold yet
SENT_OFFSCREEN_MAX_FRAC = 0.3
DEFAULT_WORKERS = max(1, min(6, (os.cpu_count() or 4) // 3))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--clips-csv", type=Path, default=manifests.DATA / "reporter_clips.csv")
    p.add_argument("--links-csv", type=Path, default=manifests.PROJECT / "video_links.csv")
    p.add_argument("--out-dir", type=Path, default=manifests.PHASE2_DIR)
    p.add_argument("--crops-dir", type=Path, default=manifests.CROPS_DIR)
    p.add_argument("--reporters", nargs="+")
    p.add_argument("--exclude-reporters", nargs="+")
    p.add_argument("--limit", type=int, help="first N pending clips after filtering")
    p.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--sync-min", type=float, default=SYNC_MIN)
    p.add_argument("--word-sync-min", type=float, default=WORD_SYNC_MIN)
    p.add_argument("--lip-motion-min", type=float, default=LIP_MOTION_MIN)
    p.add_argument("--sent-offscreen-max-frac", type=float, default=SENT_OFFSCREEN_MAX_FRAC)
    return p.parse_args()


def make_job(row: dict, args, thresholds) -> source_clip.Job:
    stem = manifests.stem_of(row["clip_file"])
    return source_clip.Job(
        stem=stem,
        clip_path=str(manifests.PROJECT / row["clip_file"]),
        transcript_path=str(manifests.PROJECT / row["transcript_json"]),
        npz_path=str(args.crops_dir / row["reporter"] / f"{stem}.npz"),
        device=args.device, sync_min=args.sync_min, thresholds=thresholds)


def main():
    args = parse_args()
    removed = manifests.drop_unfinished(args.out_dir)
    if removed:
        print(f"[phase2] removed {removed} rows of a clip that did not finish last time")

    rows = manifests.snapshot_clips(args.clips_csv, args.links_csv)
    no_channel = sum(1 for r in rows if not r["channel"])
    done = manifests.done_clips(args.out_dir)
    pending = [r for r in manifests.select(rows, args.reporters, args.exclude_reporters)
               if r["clip_file"] not in done]
    if args.limit:
        pending = pending[:args.limit]
    hours = sum(float(r["duration_sec"]) for r in pending) / 3600
    print(f"[phase2] {len(rows)} clips in snapshot ({no_channel} without a channel match), "
          f"{len(done)} already done, {len(pending)} to process ({hours:.2f} h) "
          f"with {args.workers} workers on {args.device}")
    if not pending:
        return

    thresholds = segments.Thresholds(word_sync_min=args.word_sync_min,
                                     lip_motion_min=args.lip_motion_min,
                                     sent_offscreen_max_frac=args.sent_offscreen_max_frac)
    totals = {"ok": 0, "failed": 0, "words": 0, "sentences": 0}
    t0, done_sec = time.time(), 0.0
    queue = iter(pending)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        running = {}

        def submit_next():
            row = next(queue, None)
            if row is not None:
                job = make_job(row, args, thresholds)
                running[pool.submit(source_clip.process, job)] = (row, job)

        for _ in range(args.workers * 2):  # bounded: rows are not all pickled up front
            submit_next()
        while running:
            finished, _ = wait(running, return_when=FIRST_COMPLETED)
            for fut in finished:
                row, job = running.pop(fut)
                submit_next()
                try:
                    result = fut.result()
                except Exception as exc:  # one bad clip must never stop the run
                    totals["failed"] += 1
                    print(f"[phase2] FAILED {row['clip_file']}: {exc!r}", file=sys.stderr)
                    continue
                manifests.write_result(args.out_dir, row, Path(job.npz_path), result)
                totals["ok"] += 1
                totals["words"] += len(result["words"])
                totals["sentences"] += len(result["sentences"])
                done_sec += float(row["duration_sec"])
                n = totals["ok"] + totals["failed"]
                rate = done_sec / max(time.time() - t0, 1e-6)
                eta_h = (hours * 3600 - done_sec) / max(rate, 1e-6) / 3600
                sync = (result.get("sync") or {}).get("syncnet_score", "-")
                print(f"[phase2] {n}/{len(pending)} {result['status']:<10} sync={sync} "
                      f"words {len(result['words'])}/{result['n_words']} "
                      f"sents {len(result['sentences'])}  {result.get('seconds')}s  "
                      f"{rate:.2f}x realtime, ETA {eta_h:.1f} h  {job.stem}", flush=True)
    print(f"[phase2] finished: {totals['ok']} clips, {totals['failed']} failed (retried "
          f"next run), {totals['words']} word clips, {totals['sentences']} sentence clips")


if __name__ == "__main__":
    main()
