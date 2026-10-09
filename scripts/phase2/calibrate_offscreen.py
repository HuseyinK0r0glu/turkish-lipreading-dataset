"""Choose WORD_SYNC_MIN from spliced audio (PHASE2_PLAN §4.5.1).

Off-screen speech is speech the visible mouth did not produce. This builds it on
purpose: for each sampled solo clip the second half of the audio is replaced by
another reporter's speech, and the clip is scored twice from ONE decode -- with its
own audio and with the spliced audio -- through the same SyncScorer and the same
segments.word_items that run_phase2 uses.

Three groups of word_sync_conf come out:
  real          every word, own audio                      -> should be kept
  spliced_keep  first-half words of a spliced clip          -> should be kept
  spliced_drop  second-half words (foreign audio)           -> should be dropped

It prints how many of each a threshold keeps, and Youden's best one. Spliced audio is
the easy case -- the mouth still moves, just to other words -- so it calibrates the
SyncNet signal only; a silent presenter under a voice-over also has low lip_motion.
The plan's hand-labelled set remains the check on real broadcasts.

    uv run python scripts/phase2/calibrate_offscreen.py --clips 40
"""
import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]

import numpy as np  # noqa: E402
import torch  # noqa: E402

import manifests  # noqa: E402
import segments  # noqa: E402
import video_io  # noqa: E402
from run_phase2 import SYNC_MIN  # noqa: E402
from source_clip import track_clip  # noqa: E402

SAMPLES_PER_FRAME = video_io.AUDIO_RATE // video_io.FPS
# Words whose window straddles the splice point belong to neither group.
GROUPS = ("real", "spliced_keep", "spliced_drop")


def pick_clips(rows, n, min_sec, seed):
    """Round-robin over reporters so no single speaker dominates the calibration."""
    rng = random.Random(seed)
    by_rep = defaultdict(list)
    for r in rows:
        if float(r["duration_sec"]) >= min_sec:
            by_rep[r["reporter"]].append(r)
    for v in by_rep.values():
        rng.shuffle(v)
    picked, reps = [], sorted(by_rep)
    while len(picked) < n and any(by_rep.values()):
        for rep in reps:
            if by_rep[rep] and len(picked) < n:
                picked.append(by_rep[rep].pop())
    return picked, by_rep


def word_confs(stem, transcript, roi, res):
    track = segments.Track(good=roi.good, aperture=roi.aperture,
                           mouth_width_px=roi.mouth_width_px, yaw=roi.yaw, pitch=roi.pitch,
                           frame_conf=res.frame_conf, offset=res.offset)
    th = segments.Thresholds(word_sync_min=None, lip_motion_min=None, sent_offscreen_max_frac=1)
    kept, _, _ = segments.word_items(stem, transcript, track, th)
    return [(w["frame_start"], w["frame_end"], w["word_sync_conf"]) for w in kept
            if w["word_sync_conf"] != ""]


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--clips", type=int, default=40)
    p.add_argument("--min-sec", type=float, default=10.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", type=Path, default=manifests.PHASE2_DIR / "calibration_offscreen.csv")
    args = p.parse_args()

    rows = manifests.read_csv(manifests.DATA / "reporter_clips.csv")
    clips, spare = pick_clips(rows, args.clips, args.min_sec, args.seed)
    rng = random.Random(args.seed + 1)
    conf = {g: [] for g in GROUPS}
    clip_scores = []  # (own-audio score, spliced score)
    out_rows = []
    for i, row in enumerate(clips, 1):
        donors = [r for r in rows if r["reporter"] != row["reporter"]
                  and float(r["duration_sec"]) >= float(row["duration_sec"]) / 2 + 1]
        donor = rng.choice(donors)
        clip = manifests.PROJECT / row["clip_file"]
        with open(manifests.PROJECT / row["transcript_json"], encoding="utf-8") as f:
            transcript = json.load(f)
        roi, scorer = track_clip(clip, args.device)
        own = video_io.read_audio(clip)
        half = len(roi.good) // 2
        split = half * SAMPLES_PER_FRAME
        foreign = video_io.read_audio(manifests.PROJECT / donor["clip_file"])[:len(own) - split]
        spliced = own.copy()
        spliced[split:split + len(foreign)] = foreign
        real, fake = scorer.finish(own), scorer.finish(spliced)
        if real is None or fake is None or real.conf < SYNC_MIN:
            print(f"[{i}/{len(clips)}] skip {row['reporter']} (would be discarded at clip level)")
            continue
        clip_scores.append((real.conf, fake.conf))
        stem = manifests.stem_of(row["clip_file"])
        for a, b, c in word_confs(stem, transcript, roi, real):
            conf["real"].append(c)
            out_rows.append((stem, row["reporter"], "real", a, b, c))
        for a, b, c in word_confs(stem, transcript, roi, fake):
            group = "spliced_keep" if b <= half else "spliced_drop" if a >= half else None
            if group:
                conf[group].append(c)
                out_rows.append((stem, row["reporter"], group, a, b, c))
        print(f"[{i}/{len(clips)}] {row['reporter']:<20} score own={real.conf:.2f} "
              f"spliced={fake.conf:.2f}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("source_clip,speaker_id,group,frame_start,frame_end,word_sync_conf\n")
        f.writelines(",".join(map(str, r)) + "\n" for r in out_rows)

    arr = {g: np.array(v, np.float64) for g, v in conf.items()}
    print(f"\nwrote {args.out}")
    for g in GROUPS:
        if len(arr[g]):
            q = np.percentile(arr[g], [1, 5, 25, 50, 75, 95])
            print(f"{g:<13} n={len(arr[g]):<5} p1/p5/p25/p50/p75/p95 = "
                  + " ".join(f"{x:.2f}" for x in q))
    fake_pass = sum(1 for _, s in clip_scores if s >= SYNC_MIN)
    print(f"spliced clips still passing the clip-level SYNC_MIN={SYNC_MIN}: "
          f"{fake_pass}/{len(clip_scores)}")
    if not (len(arr["real"]) and len(arr["spliced_drop"])):
        return
    print("\nthreshold  real kept  spliced_keep kept  spliced_drop caught")
    best = (-1.0, None)
    for t in np.arange(0.0, 6.01, 0.5):
        kept = (arr["real"] >= t).mean()
        caught = (arr["spliced_drop"] < t).mean()
        keep2 = (arr["spliced_keep"] >= t).mean() if len(arr["spliced_keep"]) else float("nan")
        print(f"{t:>9.1f}  {kept:>9.3f}  {keep2:>17.3f}  {caught:>19.3f}")
        if kept + caught - 1 > best[0]:
            best = (kept + caught - 1, t)
    print(f"\nYouden-best WORD_SYNC_MIN = {best[1]:.1f}")


if __name__ == "__main__":
    main()
