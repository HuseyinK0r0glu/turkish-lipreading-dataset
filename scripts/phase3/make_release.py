"""Phase 3 step 7b: assemble data/phase3/release/ - the ONLY part that is public.

Metadata only: labels, timings, speaker, channel, split and the reference into the lip
crops. No raw video, no full-face frames, no audio, no annotation previews. The crops
themselves (88x88 gray uint8, data/lip_crops/) ship alongside as `lip_crops/`; they are
referenced here, not copied, and stay un-normalised: `norm_stats.json` holds the global
mean / std (over a seeded sample of TRAIN crops) to apply at load time.

  words_{train,val,test}.csv   clip_id, label_surface, label_lemma, in_vocab_*, surface_form,
                               lemma, word_start_sec, word_end_sec, speaker_id, channel, split,
                               npz_path, frame_start, frame_end, n_frames, fps
                               (+ verified, label_fix once verification/verified.csv exists)
  sentences_{val,test}.csv     clip_id, text, word_spans, speaker_id, channel, split, npz ref
  vocab_*.csv  surface_to_lemma.csv  splits.json  speaker_table.csv
  spotting_gt/                 queries, negatives, classes (+ test_gt_verified.csv if present)
  stats/                       class frequencies and the other tables of dataset_stats.py
  norm_stats.json  README.md

`word_start_sec` / `word_end_sec` are Whisper times in the source clip's audio; the
frames are `np.load(npz_path)["frames"][frame_start:frame_end]`, +-15 frames of context
around the word, already shifted by the audio/video offset.

    uv run python scripts/phase3/make_release.py
    uv run python scripts/phase3/make_release.py --norm-sample 0     # skip the crop statistics
"""
import argparse
import json
import shutil

import numpy as np
import pandas as pd

import common as C

WORD_COLS = ["clip_id", "label_surface", "label_lemma", "in_vocab_surface", "in_vocab_lemma",
             "surface_form", "lemma", "word_start_sec", "word_end_sec", "speaker_id", "channel",
             "split", "npz_path", "frame_start", "frame_end", "n_frames", "fps"]
SENT_COLS = ["clip_id", "text", "word_spans", "speaker_id", "channel", "split", "npz_path",
             "frame_start", "frame_end", "n_frames", "fps"]
CROPS_PREFIX = "data/"   # release paths are relative to the package: lip_crops/<speaker>/<stem>.npz

README = """# Turkish News Lip Reading Spotting Benchmark - metadata release

Built {created} by `scripts/phase3/make_release.py`.

Contents: word-clip and sentence-clip metadata with speaker-independent splits, two
vocabularies (surface form and lemma), the spotting query lists and frequency statistics.
The only pixels of the dataset are the 88x88 grayscale mouth crops in `lip_crops/`
(distributed next to this folder); no raw video, full-face frames or audio are released.

## Splits

| split | speakers | word clips | sentences |
|-------|----------|-----------:|----------:|
{split_rows}

Split by speaker. No channel is shared between train and test. Validation and test rows
with ASR confidence < 0.7 are excluded; train keeps them.

## Vocabulary

| labeling | classes | train clips | val clips | test clips |
|----------|--------:|------------:|----------:|-----------:|
{vocab_rows}

A class has at least {min_train} training examples from at least {min_speakers} training
speakers. `label_*` is empty and `in_vocab_*` is 0 for word clips outside the vocabulary.
Labels are Whisper large-v3 output (ASR, not human transcripts){verified_note}.
`lemma` is a context-free zeyrek lemma: a grouping key, not gold morphology.

## Loading a clip

```python
frames = np.load(npz_path)["frames"][frame_start:frame_end]   # (T, 88, 88) uint8, 25 fps
x = (frames / 255.0 - mean) / std                              # norm_stats.json
```

Sentence `word_spans` times are seconds from the sentence clip's first frame.

## Spotting

`spotting_gt/queries.csv` lists every occurrence of a vocabulary class in the val/test
sentence clips (automatic, from ASR word timings); `negatives.csv` sampled sentences
without the class; `classes.csv` the counts and the `evaluable` flag.
{gt_note}
"""


def crop_stats(paths, sample, seed):
    rng = np.random.default_rng(seed)
    paths = sorted(paths)
    pick = rng.choice(len(paths), size=min(sample, len(paths)), replace=False)
    n = total = total_sq = 0.0
    used = 0
    for i in pick:
        try:
            f = np.load(C.ROOT / paths[i])["frames"].astype(np.float64) / 255.0
        except Exception:
            continue
        n += f.size
        total += f.sum()
        total_sq += (f * f).sum()
        used += 1
    mean = total / n
    return {"mean": round(mean, 6), "std": round(float(np.sqrt(total_sq / n - mean * mean)), 6),
            "scale": "pixel / 255", "source": "train source clips, all frames",
            "npz_files_sampled": used, "seed": seed}


def release_path(p: pd.Series) -> pd.Series:
    return p.str.removeprefix(CROPS_PREFIX)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--phase3", type=C.Path, default=C.PHASE3)
    ap.add_argument("--norm-sample", type=int, default=2000,
                    help="train npz files to estimate the global mean/std from (0 = skip)")
    args = ap.parse_args()
    C.utf8_stdout()
    rep = C.Report()
    p3 = args.phase3
    rel = p3 / "release"
    info = json.loads(C.require(p3 / "splits.json", "build_splits.py").read_text("utf-8"))
    seed = info["parameters"]["seed"]
    if rel.exists():
        shutil.rmtree(rel)   # rebuilt from scratch: nothing stale survives a re-run
    rel.mkdir(parents=True)

    verified = None
    vpath = p3 / "verification" / "verified.csv"
    if vpath.exists():
        verified = pd.read_csv(vpath, dtype=str, keep_default_na=False)
        verified = verified[verified.kind == "word"][["clip_id", "verified", "label_fix"]]

    rep("Phase 3 make_release")
    train_npz, n_words, n_sents = [], {}, {}
    for split in C.SPLITS:
        w = pd.read_csv(C.require(p3 / f"word_{split}.csv", "build_splits.py"), dtype=str,
                        keep_default_na=False, usecols=WORD_COLS)[WORD_COLS]
        if split == "train":
            train_npz = w.npz_path.unique()
        w["npz_path"] = release_path(w.npz_path)
        if verified is not None:
            w = w.merge(verified, on="clip_id", how="left").fillna({"verified": "",
                                                                    "label_fix": ""})
        n_words[split] = len(w)
        C.write_csv(w, rel / f"words_{split}.csv")
        rep(f"  words_{split}.csv: {len(w):,}")
    for split in ["val", "test"]:
        s = pd.read_csv(C.require(p3 / f"sentence_{split}.csv", "build_spotting_set.py"),
                        dtype=str, keep_default_na=False, usecols=SENT_COLS)[SENT_COLS]
        s["npz_path"] = release_path(s.npz_path)
        n_sents[split] = len(s)
        C.write_csv(s, rel / f"sentences_{split}.csv")
        rep(f"  sentences_{split}.csv: {len(s):,}")

    for name in ["vocab_surface.csv", "vocab_lemma.csv", "surface_to_lemma.csv",
                 "speaker_table.csv"]:
        shutil.copy2(p3 / name, rel / name)
    public = {k: v for k, v in info.items() if k != "input"}   # no local paths in the release
    public["input_rows"] = info["input"]["rows"]
    public["input_sha256"] = info["input"]["sha256"]
    public.get("spotting", {}).pop("input", None)
    C.write_text(rel / "splits.json", json.dumps(public, ensure_ascii=False, indent=2) + "\n")

    (rel / "spotting_gt").mkdir()
    has_gt = False
    for name in ["queries.csv", "negatives.csv", "classes.csv", "test_gt_verified.csv"]:
        src = p3 / "spotting_gt" / name
        if src.exists():
            shutil.copy2(src, rel / "spotting_gt" / name)
            has_gt |= name == "test_gt_verified.csv"
    (rel / "stats").mkdir()
    stats = sorted((p3 / "stats").glob("class_frequency_*.csv")) + [
        p3 / "stats" / n for n in ["per_speaker.csv", "per_channel.csv", "word_length.csv"]]
    for src in stats:
        if src.exists():
            shutil.copy2(src, rel / "stats" / src.name)
    if not list((rel / "stats").iterdir()):
        rep("  stats/ is empty: run dataset_stats.py first")

    if args.norm_sample:
        norm = crop_stats(train_npz, args.norm_sample, seed)
        C.write_text(rel / "norm_stats.json", json.dumps(norm, indent=2) + "\n")
        rep(f"  norm_stats.json: mean {norm['mean']}, std {norm['std']} "
            f"({norm['npz_files_sampled']} train npz files)")

    kappa = p3 / "verification" / "kappa.txt"
    if kappa.exists():
        shutil.copy2(kappa, rel / "verification_kappa.txt")
    C.write_text(rel / "README.md", README.format(
        created=info["created"],
        split_rows="\n".join(
            f"| {s} | {len(info['speakers'][s])} | {n_words[s]:,} | "
            f"{n_sents.get(s, 0) and format(n_sents[s], ',') or '-'} |" for s in C.SPLITS),
        vocab_rows="\n".join(
            f"| {lab} | {a['classes']} | {a['train_clips_in_vocab']:,} | "
            f"{a['val_clips_in_vocab']:,} | {a['test_clips_in_vocab']:,} |"
            for lab, a in info["achieved"].items()),
        min_train=info["parameters"]["min_train"],
        min_speakers=info["parameters"]["min_speakers"],
        verified_note=("; a hand-checked sample carries `verified` / `label_fix`, agreement "
                       "in `verification_kappa.txt`" if verified is not None else
                       "; the manual verification sample is not included yet"),
        gt_note=("`test_gt_verified.csv` holds the hand-corrected test occurrences."
                 if has_gt else
                 "The hand-corrected test ground truth is not included yet.")))

    # the copyright rule, checked rather than trusted
    banned = [p for p in rel.rglob("*") if p.suffix.lower() in
              {".mp4", ".wav", ".m4a", ".mp3", ".png", ".jpg", ".jpeg", ".npz"}]
    assert not banned, f"media in the release folder: {banned[:3]}"
    size = sum(p.stat().st_size for p in rel.rglob("*") if p.is_file())
    rep(f"\nrelease/: {sum(1 for p in rel.rglob('*') if p.is_file())} files, "
        f"{size / 1e6:.0f} MB, no media files")
    rep("OPEN DECISION (project owner): speaker slugs and channel names are released as-is "
        "(public news presenters); they are also part of every clip_id and npz path.")
    rep.save(p3 / "stats" / "release_report.txt")


if __name__ == "__main__":
    main()
