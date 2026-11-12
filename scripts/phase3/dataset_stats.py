"""Phase 3 step 7a: distribution tables for the paper and the Phase 5 error analysis.

Reads data/phase3/word_*.csv, sentence_*.csv and vocab_*.csv; writes CSVs into
data/phase3/stats/ (tables only: plotting is left to the paper's notebooks, the repo
has no plotting dependency):

  class_frequency_{surface,lemma}.csv   per class: counts per split, speakers, length, syllables
  per_speaker.csv  per_channel.csv      word clips (all / in vocab), sentences, hours per split
  word_length.csv                       classes and clips by character and syllable count
  hist_<column>.csv                     per-split histograms: face_height_px, mouth_width_px,
                                        confidence, word_sync_conf, syncnet_score, n_frames,
                                        word duration
  phase2_discards.csv                   what Phase 2 dropped, by level and reason
  summary.txt                           headline numbers

    uv run python scripts/phase3/dataset_stats.py
"""
import argparse

import numpy as np
import pandas as pd

import common as C

HISTS = {  # column -> bin edges
    "face_height_px": np.arange(0, 1001, 20),
    "mouth_width_px": np.arange(0, 201, 5),
    "confidence": np.round(np.arange(0, 1.0001, 0.05), 2),
    "word_sync_conf": np.arange(0, 15.01, 0.5),
    "syncnet_score": np.arange(0, 15.01, 0.5),
    "n_frames": np.arange(0, 91, 5),
    "word_sec": np.round(np.arange(0, 2.0001, 0.08), 2),
}
COLS = ["clip_id", "speaker_id", "channel", "npz_path", "split", "label_surface", "label_lemma",
        "in_vocab_surface", "in_vocab_lemma", "face_height_px", "mouth_width_px", "confidence",
        "word_sync_conf", "syncnet_score", "n_frames", "word_start_sec", "word_end_sec",
        "manual_review"]


def histogram(df, column, edges):
    values = pd.to_numeric(df[column], errors="coerce")
    rows = []
    for split in C.SPLITS:
        v = values[df.split == split].dropna().to_numpy()
        counts, _ = np.histogram(np.clip(v, edges[0], edges[-1]), bins=edges)
        rows.append(pd.DataFrame({"split": split, "bin_start": edges[:-1], "bin_end": edges[1:],
                                  "count": counts, "share": counts / max(1, len(v))}))
    return pd.concat(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--phase3", type=C.Path, default=C.PHASE3)
    args = ap.parse_args()
    C.utf8_stdout()
    rep = C.Report()
    p3, out = args.phase3, args.phase3 / "stats"

    w = pd.concat([pd.read_csv(C.require(p3 / f"word_{s}.csv", "build_splits.py"), dtype=str,
                               keep_default_na=False, usecols=COLS) for s in C.SPLITS])
    w["word_sec"] = w.word_end_sec.astype(float) - w.word_start_sec.astype(float)
    for lab in C.LABELINGS:
        w[f"in_vocab_{lab}"] = w[f"in_vocab_{lab}"].astype(int)
    rep(f"Phase 3 dataset_stats: {len(w):,} word clips")

    # class frequency
    for lab in C.LABELINGS:
        v = pd.read_csv(p3 / f"vocab_{lab}.csv", keep_default_na=False, dtype={"class": str})
        v["total_n"] = v.train_n + v.val_n + v.test_n
        v["chars"] = v["class"].str.len()
        v["syllables"] = v["class"].map(C.syllables)
        top = (w[(w.split == "train") & (w[f"in_vocab_{lab}"] == 1)]
               .groupby(f"label_{lab}").speaker_id
               .agg(lambda s: s.value_counts(normalize=True).iloc[0]))
        v["top_speaker_share"] = v["class"].map(top).round(3)
        C.write_csv(v, out / f"class_frequency_{lab}.csv")
        rep(f"  [{lab}] {len(v)} classes; train examples per class median "
            f"{int(v.train_n.median())}, max {int(v.train_n.max())}; "
            f"{int((v.chars <= 2).sum())} classes of 1-2 letters, "
            f"{int((v.syllables <= 1).sum())} of one syllable; "
            f"{int((v.top_speaker_share > 0.5).sum())} classes where one speaker holds "
            f"over half the train examples")

    # word length
    rows = []
    for lab in C.LABELINGS:
        part = w[w[f"in_vocab_{lab}"] == 1]
        label = part[f"label_{lab}"]
        for measure, values in [("chars", label.str.len()), ("syllables", label.map(
                {c: C.syllables(c) for c in label.unique()}))]:
            t = pd.DataFrame({"value": values, "cls": label}).groupby("value").agg(
                clips=("cls", "size"), classes=("cls", "nunique")).reset_index()
            rows.append(t.assign(labeling=lab, measure=measure))
    C.write_csv(pd.concat(rows)[["labeling", "measure", "value", "classes", "clips"]],
                out / "word_length.csv")

    # per speaker / per channel
    sents = []
    for s in ["train", "val", "test"]:
        path = p3 / f"sentence_{s}.csv"
        if path.exists():
            sents.append(pd.read_csv(path, dtype=str, keep_default_na=False,
                                     usecols=["speaker_id", "channel", "n_frames", "split"]))
    sent = pd.concat(sents) if sents else pd.DataFrame(
        columns=["speaker_id", "channel", "n_frames", "split"])
    sent["hours"] = sent.n_frames.astype(float) / C.FPS / 3600
    for key, name in [("speaker_id", "per_speaker"), ("channel", "per_channel")]:
        t = w.groupby([key, "split"]).agg(
            word_clips=("clip_id", "size"), source_clips=("npz_path", "nunique"),
            in_vocab_surface=("in_vocab_surface", "sum"), in_vocab_lemma=("in_vocab_lemma", "sum"),
            manual_review=("manual_review", lambda s: int((s == "1").sum())),
            median_face_height_px=("face_height_px",
                                   lambda s: pd.to_numeric(s, errors="coerce").median()))
        if key == "channel":
            t["speakers"] = w.groupby([key, "split"]).speaker_id.nunique()
        st = sent.groupby([key, "split"]).agg(sentences=("hours", "size"),
                                              sentence_hours=("hours", "sum"))
        t = t.join(st).fillna({"sentences": 0, "sentence_hours": 0})
        t["sentences"] = t.sentences.astype(int)
        t["sentence_hours"] = t.sentence_hours.round(2)
        C.write_csv(t.reset_index().sort_values(["split", "word_clips"], ascending=[True, False]),
                    out / f"{name}.csv")

    for column, edges in HISTS.items():
        C.write_csv(histogram(w, column, edges), out / f"hist_{column}.csv")

    if C.DISCARDED.exists():
        d = pd.read_csv(C.DISCARDED, dtype=str, keep_default_na=False, usecols=["level", "reason"])
        C.write_csv(d.value_counts().rename("count").reset_index(), out / "phase2_discards.csv")

    rep("\nPer split:")
    for split in C.SPLITS:
        part = w[w.split == split]
        fh = pd.to_numeric(part.face_height_px, errors="coerce")
        rep(f"  {split:5} {len(part):9,} word clips, {part.speaker_id.nunique():2d} speakers, "
            f"{part.channel.nunique():2d} channels; in vocab surface "
            f"{int(part.in_vocab_surface.sum()):,} / lemma {int(part.in_vocab_lemma.sum()):,}; "
            f"face height median {fh.median():.0f} px; word {part.word_sec.median():.2f} s median")
    if "train" not in set(sent.split):
        rep("  (train sentence hours need build_spotting_set.py --with-train)")
    rep.save(out / "summary.txt")


if __name__ == "__main__":
    main()
