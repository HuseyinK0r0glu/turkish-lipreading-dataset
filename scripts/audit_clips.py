"""Re-audit every produced pipeline-2 clip against the CURRENT solo-presenter rules.

Clips written before the per-sample YuNet filter (presenter_filter.analyze_frames /
solo_segments) were certified by one HOG frame per 30 s, and that let split screens,
video walls, interpreter boxes and a few seconds of another person through. This
runs every clip through exactly the code the pipeline now uses and makes the
manifest agree with it:

  trim     some of it passes                         -> each passing part becomes
                                                        its own clip, the old one
                                                        is removed
  drop     nothing in it passes (or it is < 2 s)     -> clip, transcript and
                                                        manifest row removed

There is no "keep as-is": every clip is at least trimmed by SCENE_EDGE_TRIM, because
its edges are scene cuts and those are where transitions sit.

Trimmed clips are cut from the existing clip (no re-download) and their transcript is
SLICED from the existing Whisper JSON instead of re-running Whisper: words entirely
inside the kept range are kept and shifted to clip-local time, words that straddle
the cut are dropped. transcript_text is rebuilt from what remains.

Stage 1 (analysis) caches one JSON per clip in data/_clip_audit/, so re-running with
different thresholds costs seconds. Stage 3 (apply) journals every clip it finishes
to data/_clip_audit/applied.jsonl before deleting anything, so an interrupted apply
resumes where it stopped. A manifest backup is written before the first change.

    uv run python scripts/audit_clips.py                 # analyse + report, no changes
    uv run python scripts/audit_clips.py --apply         # ... then rewrite the corpus
    uv run python scripts/audit_clips.py --reporters hadi_ozisik --workers 8
"""

import argparse
import collections
import csv
import json
import os
import re
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cv2  # noqa: E402

PROJECT = Path(__file__).resolve().parent.parent
DATA = PROJECT / "data"
MANIFEST = DATA / "reporter_clips.csv"
AUDIT_DIR = DATA / "_clip_audit"
JOURNAL = AUDIT_DIR / "applied.jsonl"
CLIP_NAME_RE = re.compile(r"^(?P<prefix>.*_scene_\d{4})_[\d.]+-[\d.]+$")

_encodings = {}


def _worker_init():
    cv2.setNumThreads(1)


def _analyze(clip_file: str, reporter: str) -> str:
    from presenter_filter import analyze_frames, load_reference_encoding, sample_frames
    import pipeline2
    if reporter not in _encodings:
        _encodings[reporter] = load_reference_encoding(pipeline2.reporter_image(reporter))
    records = analyze_frames(sample_frames(PROJECT / clip_file), _encodings[reporter])
    out = cache_path(clip_file)
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(records, separators=(",", ":")), encoding="utf-8")
    os.replace(tmp, out)
    return clip_file


def cache_path(clip_file: str) -> Path:
    return AUDIT_DIR / (Path(clip_file).stem + ".json")


def read_manifest():
    with open(MANIFEST, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_manifest(rows, fields):
    tmp = MANIFEST.with_suffix(".csv.tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, MANIFEST)


def plan_clip(row, records):
    """(verdict, segments) under the current rules; segments are clip-local."""
    from presenter_filter import IDENTITY_TOLERANCE, solo_segments
    dur = float(row["duration_sec"])
    segs = solo_segments(records, 0.0, dur, IDENTITY_TOLERANCE)
    if not segs:
        return "drop", []
    return "trim", segs


def slice_transcript(src_json: Path, a: float, b: float, new_name: str):
    data = json.loads(src_json.read_text(encoding="utf-8"))
    out = {"video": new_name, "segments": []}
    for seg in data.get("segments", []):
        words = [dict(w, start_sec=round(w["start_sec"] - a, 3),
                      end_sec=round(w["end_sec"] - a, 3))
                 for w in seg.get("words", [])
                 if w["start_sec"] >= a and w["end_sec"] <= b]
        if not words:
            continue
        out["segments"].append({
            "start_sec": words[0]["start_sec"],
            "end_sec": words[-1]["end_sec"],
            "text": " ".join(w["surface_form"] for w in words).strip(),
            "words": words,
        })
    return out, " ".join(s["text"] for s in out["segments"]).strip()


def apply_clip(row, segs):
    """Cut `segs` out of the clip, write their transcripts; return new manifest rows."""
    from presenter_filter import export_clip
    clip = PROJECT / row["clip_file"]
    stem = Path(row["clip_file"]).stem
    m = CLIP_NAME_RE.match(stem)
    prefix = m.group("prefix") if m else stem
    src_start = float(row["start_sec"])
    old_json = PROJECT / row["transcript_json"] if row.get("transcript_json") else None

    new_rows = []
    for a, b, face_h in segs:
        abs_a, abs_b = src_start + a, src_start + b
        new_clip = clip.with_name(f"{prefix}_{abs_a:.2f}-{abs_b:.2f}.mp4")
        export_clip(clip, new_clip, a, b)
        new_row = dict(row)
        new_row.update({
            "clip_file": new_clip.relative_to(PROJECT).as_posix(),
            "start_sec": round(abs_a, 3),
            "end_sec": round(abs_b, 3),
            "duration_sec": round(b - a, 3),
            "face_height_px": face_h,
        })
        if old_json and old_json.exists():
            new_json = old_json.with_name(new_clip.stem + ".json")
            data, text = slice_transcript(old_json, a, b, new_clip.name)
            new_json.write_text(json.dumps(data, ensure_ascii=False, indent=2),
                                encoding="utf-8")
            new_row["transcript_json"] = new_json.relative_to(PROJECT).as_posix()
            new_row["transcript_text"] = text
        new_rows.append(new_row)
    return new_rows


def remove_clip_files(row, keep=()):
    for key in ("clip_file", "transcript_json"):
        rel = row.get(key)
        if rel and rel not in keep:
            p = PROJECT / rel
            if p.exists():
                p.unlink()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apply", action="store_true", help="rewrite clips and manifest")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 4))
    ap.add_argument("--cut-workers", type=int, default=6,
                    help="parallel ffmpeg re-encodes during --apply")
    ap.add_argument("--reporters", nargs="*")
    ap.add_argument("--report", type=Path, help="write per-clip verdicts as CSV here")
    args = ap.parse_args()

    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    rows = read_manifest()
    fields = list(rows[0].keys()) if rows else []
    done = {}
    if JOURNAL.exists():
        for line in JOURNAL.read_text(encoding="utf-8").splitlines():
            if line.strip():
                j = json.loads(line)
                done[j["old"]] = j["new"]
    todo = [r for r in rows if r["clip_file"] not in done
            and (not args.reporters or r["reporter"] in args.reporters)]

    # ---- stage 1: analysis (cached) ----
    need = [r for r in todo if not cache_path(r["clip_file"]).exists()]
    print(f"[audit] {len(todo)} clips to audit, {len(need)} need analysis "
          f"({sum(float(r['duration_sec']) for r in need) / 3600:.1f} h of video, "
          f"{args.workers} workers)")
    if need:
        t0 = time.time()
        total_s = sum(float(r["duration_sec"]) for r in need)
        done_s = 0.0
        dur = {r["clip_file"]: float(r["duration_sec"]) for r in need}
        with ProcessPoolExecutor(args.workers, initializer=_worker_init) as ex:
            futs = [ex.submit(_analyze, r["clip_file"], r["reporter"]) for r in need]
            for n, fut in enumerate(as_completed(futs), 1):
                try:
                    cf = fut.result()
                    done_s += dur[cf]
                except Exception as exc:  # an unreadable clip is dropped in stage 2
                    print(f"[audit] analysis failed: {type(exc).__name__}: {exc}")
                if n % 200 == 0 or n == len(futs):
                    el = time.time() - t0
                    eta = el / max(done_s, 1) * (total_s - done_s)
                    print(f"[audit] analysed {n}/{len(futs)} "
                          f"({done_s / 3600:.1f}/{total_s / 3600:.1f} h) "
                          f"elapsed {el / 60:.0f} min, ETA {eta / 60:.0f} min", flush=True)

    # ---- stage 2: plan ----
    plans = []
    stats = collections.defaultdict(lambda: collections.Counter())
    for r in todo:
        cp = cache_path(r["clip_file"])
        records = json.loads(cp.read_text(encoding="utf-8")) if cp.exists() else []
        verdict, segs = plan_clip(r, records)
        kept = sum(b - a for a, b, _ in segs)
        s = stats[r["reporter"]]
        s["clips"] += 1
        s[verdict] += 1
        s["in_h"] += float(r["duration_sec"])
        s["out_h"] += kept
        s["out_clips"] += len(segs)
        plans.append((r, verdict, segs))

    print(f"\n{'reporter':20s} {'clips':>6s} {'drop':>6s} {'trim':>6s} "
          f"{'->clips':>8s} {'hours':>7s} {'->hours':>8s} {'kept%':>6s}")
    tot = collections.Counter()
    for rep in sorted(stats):
        s = stats[rep]
        tot.update(s)
        print(f"{rep:20s} {s['clips']:6d} {s['drop']:6d} {s['trim']:6d} "
              f"{s['out_clips']:8d} {s['in_h'] / 3600:7.2f} {s['out_h'] / 3600:8.2f} "
              f"{100 * s['out_h'] / max(1, s['in_h']):5.0f}%")
    print(f"{'TOTAL':20s} {tot['clips']:6d} {tot['drop']:6d} {tot['trim']:6d} "
          f"{tot['out_clips']:8d} {tot['in_h'] / 3600:7.2f} {tot['out_h'] / 3600:8.2f} "
          f"{100 * tot['out_h'] / max(1, tot['in_h']):5.0f}%")

    if args.report:
        with open(args.report, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["reporter", "clip_file", "duration_sec", "verdict",
                        "kept_sec", "segments"])
            for r, verdict, segs in plans:
                w.writerow([r["reporter"], r["clip_file"], r["duration_sec"], verdict,
                            round(sum(b - a for a, b, _ in segs), 2),
                            ";".join(f"{a:.2f}-{b:.2f}" for a, b, _ in segs)])
        print(f"[audit] per-clip report -> {args.report}")

    if not args.apply:
        print("\n[audit] dry run; re-run with --apply to rewrite the corpus")
        return

    # ---- stage 3: apply ----
    backup = MANIFEST.with_name(f"reporter_clips.backup-{time.strftime('%Y%m%d-%H%M%S')}.csv")
    shutil.copy2(MANIFEST, backup)
    print(f"[audit] manifest backup -> {backup.name}")
    # Re-encoding is the cost here (~60 h of video through libx264), so clips are cut
    # in parallel threads -- each is an ffmpeg subprocess. The journal is written only
    # from this thread, and a clip's old files are deleted only after its journal line
    # is on disk, so an interrupted apply never loses a clip.
    from concurrent.futures import ThreadPoolExecutor
    with open(JOURNAL, "a", encoding="utf-8") as jf, \
            ThreadPoolExecutor(args.cut_workers) as pool:
        jobs = [(r, pool.submit(apply_clip, r, segs) if segs else None)
                for r, verdict, segs in plans]
        for n, (r, fut) in enumerate(jobs, 1):
            new_rows = fut.result() if fut else []
            jf.write(json.dumps({"old": r["clip_file"], "new": new_rows},
                                ensure_ascii=False) + "\n")
            jf.flush()
            os.fsync(jf.fileno())
            remove_clip_files(r, keep={x["clip_file"] for x in new_rows}
                              | {x.get("transcript_json") for x in new_rows})
            done[r["clip_file"]] = new_rows
            if n % 250 == 0:
                print(f"[audit] applied {n}/{len(plans)}", flush=True)

    out_rows = []
    for r in rows:
        out_rows.extend(done.get(r["clip_file"], [r]))
    write_manifest(out_rows, fields)

    # Clip files no manifest row points at: exported by a row that was interrupted
    # before its manifest append. That row is still pending and will re-create them.
    listed = {r["clip_file"] for r in out_rows}
    orphans = [p for p in DATA.glob("*_solo_clips/*.mp4")
               if p.relative_to(PROJECT).as_posix() not in listed]
    for p in orphans:
        p.unlink()
        tj = DATA / "reporter_transcripts" / (p.stem + ".json")
        if tj.exists():
            tj.unlink()
    print(f"[audit] manifest now {len(out_rows)} rows "
          f"({sum(float(r['duration_sec']) for r in out_rows) / 3600:.1f} h); "
          f"removed {len(orphans)} orphan clip files")


if __name__ == "__main__":
    main()
