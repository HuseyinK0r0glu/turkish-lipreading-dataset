"""Phase 3 step 5: sentence pools and the automatic spotting ground truth.

Spotting is evaluated on SENTENCE clips of the held-out speakers (word clips are for
training). Reads data/phase2/sentence_clips.csv plus build_splits.py's splits.json and
vocab_*.csv, and writes into data/phase3/:

  sentence_val.csv sentence_test.csv   Phase 2 sentence rows + split (--with-train adds
                                       sentence_train.csv for later weak supervision)
  spotting_gt/queries.csv     one row per occurrence of a vocabulary class in a sentence:
                              class, labeling, split, clip_id, start_sec, end_sec,
                              start_frame, end_frame, surface_form
  spotting_gt/negatives.csv   per class, sampled sentences that do NOT contain it
  spotting_gt/classes.csv     positives / negatives per class and split, `evaluable` flag

Times are relative to the sentence clip's first frame (word_spans are already shifted
by the SyncNet offset); frames index `frames[frame_start:frame_end]` of that clip.
A span that sticks out of the clip (2% of sentences, by up to 4 frames) is clamped.

This is the AUTO SEED: Whisper timings, not human-verified. The hand-corrected test
ground truth is spotting_gt/test_gt_verified.csv (make_verification_set.py -> Label
Studio -> score_kappa.py).

    uv run python scripts/phase3/build_spotting_set.py
    uv run python scripts/phase3/build_spotting_set.py --min-positives 5 --neg-per-class 100
"""
import argparse
import json
import zlib

import numpy as np
import pandas as pd

import common as C


def occurrences(sent: pd.DataFrame, vocab: dict):
    """Every word span of every sentence that is a vocabulary class, per labeling."""
    rows, bad_json, clamped = [], 0, 0
    for clip_id, split, n_frames, spans in sent[
            ["clip_id", "split", "n_frames", "word_spans"]].itertuples(index=False):
        try:
            spans = json.loads(spans)
        except ValueError:
            bad_json += 1
            continue
        length = int(n_frames) / C.FPS
        for w in spans:
            surface_form = w.get("surface_form", "")
            if not surface_form or surface_form[:1] in C.APOSTROPHES:
                continue
            keys = {"surface": C.surface_class(surface_form),
                    "lemma": C.lemma_class(w.get("lemma", ""))}
            hits = [lab for lab in C.LABELINGS if keys[lab] in vocab[lab]]
            if not hits:
                continue
            start, end = float(w["start_sec"]), float(w["end_sec"])
            s, e = min(max(start, 0.0), length), min(max(end, 0.0), length)
            clamped += (s, e) != (start, end)
            f0 = min(int(round(s * C.FPS)), int(n_frames) - 1)
            f1 = min(max(f0 + 1, int(round(e * C.FPS))), int(n_frames))
            for lab in hits:
                rows.append((keys[lab], lab, split, clip_id, round(s, 3), round(e, 3),
                             f0, f1, surface_form))
    q = pd.DataFrame(rows, columns=["class", "labeling", "split", "clip_id", "start_sec",
                                    "end_sec", "start_frame", "end_frame", "surface_form"])
    return q, bad_json, clamped


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--sentences", type=C.Path, default=C.SENTENCES)
    ap.add_argument("--out", type=C.Path, default=C.PHASE3)
    ap.add_argument("--min-positives", type=int, default=5,
                    help="positive TEST sentences a class needs to be evaluable")
    ap.add_argument("--neg-per-class", type=int, default=100,
                    help="negative sentences sampled per class and split")
    ap.add_argument("--with-train", action="store_true", help="also write sentence_train.csv")
    args = ap.parse_args()
    C.utf8_stdout()
    rep = C.Report()

    info = json.loads(C.require(args.out / "splits.json", "build_splits.py").read_text("utf-8"))
    seed = info["parameters"]["seed"]
    split_of = {s: name for name, spk in info["speakers"].items() for s in spk}
    vocab = {lab: set(pd.read_csv(args.out / f"vocab_{lab}.csv", dtype=str,
                                  keep_default_na=False)["class"]) for lab in C.LABELINGS}

    sent, n_bad = C.load_sentences(args.sentences)
    n_input = len(sent)
    rep(f"Phase 3 build_spotting_set\ninput: {args.sentences.name}  {n_input:,} rows "
        f"({n_bad} malformed rows skipped)")
    assert sent.clip_id.is_unique, "clip_id is not unique in the sentence manifest"
    sent["split"] = sent.speaker_id.map(split_of).fillna("dropped")
    test_channels = set(info["channels"]["test"])
    leaked = (sent.split == "train") & sent.channel.isin(test_channels)
    sent.loc[leaked, "split"] = "dropped"
    empty = sent.n_words.astype(int) == 0
    rep(f"  sentences without words: {int(empty.sum()):,} (left out)")
    sent = sent[~empty]

    held = sent[sent.split.isin(["val", "test"])]
    for name in ["val", "test"] + (["train"] if args.with_train else []):
        part = sent[sent.split == name]
        C.write_csv(part, args.out / f"sentence_{name}.csv")
        hours = part.n_frames.astype(int).sum() / C.FPS / 3600
        rep(f"  sentence_{name}.csv: {len(part):,} sentences, {hours:.1f} h, "
            f"{part.speaker_id.nunique()} speakers ({', '.join(sorted(part.speaker_id.unique())[:6])}"
            f"{'...' if part.speaker_id.nunique() > 6 else ''})")
    assert not set(held.loc[held.split == "test", "channel"]) & set(info["channels"]["train"])

    q, bad_json, clamped = occurrences(held, vocab)
    rep(f"\noccurrences of vocabulary classes in val/test sentences: {len(q):,} "
        f"({bad_json} unreadable word_spans, {clamped:,} spans clamped to the clip)")
    q = q.sort_values(["labeling", "class", "split", "clip_id", "start_sec"])
    C.write_csv(q, args.out / "spotting_gt" / "queries.csv")

    # per class: positives, available negatives, evaluable flag
    pos = (q.groupby(["labeling", "class", "split"])
           .agg(occurrences=("clip_id", "size"), pos_sentences=("clip_id", "nunique")))
    pool = {name: held.loc[held.split == name, "clip_id"].to_numpy() for name in ["val", "test"]}
    rows, negatives = [], []
    containing = q.groupby(["labeling", "class", "split"]).clip_id.agg(set).to_dict()
    for lab in C.LABELINGS:
        for cls in sorted(vocab[lab]):
            row = {"class": cls, "labeling": lab}
            for name in ["val", "test"]:
                key = (lab, cls, name)
                has = containing.get(key, set())
                row[f"{name}_occurrences"] = int(pos.occurrences.get(key, 0))
                row[f"{name}_pos_sentences"] = len(has)
                row[f"{name}_neg_sentences"] = len(pool[name]) - len(has)
            row["evaluable"] = int(row["test_pos_sentences"] >= args.min_positives)
            rows.append(row)
            if not row["evaluable"]:
                continue
            for name in ["val", "test"]:
                has = containing.get((lab, cls, name), set())
                rng = np.random.default_rng([seed, zlib.crc32(f"{lab}/{cls}/{name}".encode())])
                # oversample, then drop the positives: cheaper than a set difference per class
                draw = rng.choice(pool[name], size=min(len(pool[name]),
                                                       args.neg_per_class + len(has)),
                                  replace=False)
                neg = [c for c in draw if c not in has][:args.neg_per_class]
                negatives += [(cls, lab, name, c) for c in neg]
    classes = pd.DataFrame(rows)
    C.write_csv(classes, args.out / "spotting_gt" / "classes.csv")
    C.write_csv(pd.DataFrame(negatives, columns=["class", "labeling", "split", "clip_id"]),
                args.out / "spotting_gt" / "negatives.csv")

    rep(f"\nEvaluable = at least {args.min_positives} positive TEST sentences:")
    for lab in C.LABELINGS:
        c = classes[classes.labeling == lab]
        ok = c[c.evaluable == 1]
        out = c[c.evaluable == 0]
        rep(f"  {lab:8} {len(ok)} of {len(c)} classes evaluable; test positives per class "
            f"median {int(ok.test_pos_sentences.median()) if len(ok) else 0}, "
            f"min {int(ok.test_pos_sentences.min()) if len(ok) else 0}; "
            f"val positives median {int(ok.val_pos_sentences.median()) if len(ok) else 0}")
        if len(out):
            rep(f"           not evaluable: " + ", ".join(
                f"{r['class']} ({r.test_pos_sentences})" for _, r in out.iterrows()))
        no_val = ok[ok.val_pos_sentences == 0]
        if len(no_val):
            rep(f"           evaluable but absent from val: " + ", ".join(no_val["class"]))
    rep(f"  negatives sampled: {len(negatives):,} ({args.neg_per_class} per class and split)")
    rep("\nGround truth here is the Whisper + SyncNet auto seed, not human-verified.")

    info["spotting"] = {
        "input": {"path": args.sentences.resolve().relative_to(C.ROOT).as_posix(),
                  "rows": n_input, "sha256": C.sha256(args.sentences)},
        "min_positives": args.min_positives, "neg_per_class": args.neg_per_class,
        "sentences": {name: int(len(pool[name])) for name in pool},
        "evaluable_classes": {lab: int(classes[(classes.labeling == lab)].evaluable.sum())
                              for lab in C.LABELINGS},
    }
    C.write_text(args.out / "splits.json", json.dumps(info, ensure_ascii=False, indent=2) + "\n")
    rep.save(args.out / "stats" / "spotting_report.txt")


if __name__ == "__main__":
    main()
