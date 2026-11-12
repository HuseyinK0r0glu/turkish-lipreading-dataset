"""Phase 3 steps 2-4: quality filter, speaker split, train cap, vocabulary.

Reads data/phase2/word_clips.csv (never written) and writes into data/phase3/:

  word_train.csv word_val.csv word_test.csv   Phase 2 word rows + label_surface, label_lemma,
                                              split, in_vocab_surface, in_vocab_lemma
  vocab_surface.csv vocab_lemma.csv           class, train_n, val_n, test_n, n_speakers, n_channels
  surface_to_lemma.csv                        which surface forms fall under which lemma class
  speaker_table.csv                           per speaker: channels, clips, hours, words, split
  dropped.csv                                 clip_id + reason for every word row left out
  splits.json                                 speakers per split, every parameter, input hashes
  stats/build_report.txt                      this script's console report
  stats/split_search.csv                      alternative channel-disjoint splits, ranked

The split is BY SPEAKER. Speakers that share a channel (and channels that share a
speaker) form one unit that moves as a whole, so the test set is channel-independent
by construction; a train row on a test channel is dropped and then asserted absent.
The default split is the hand pick the vocabulary numbers in CLAUDE.md refer to.

    uv run python scripts/phase3/build_splits.py
    uv run python scripts/phase3/build_splits.py --val levent_gultekin --test yilmaz_ozdil ozlem_gurses
    uv run python scripts/phase3/build_splits.py --min-train 300 --min-speakers 5 --cap-share 0.15
"""
import argparse
import itertools
import json
from datetime import date

import numpy as np
import pandas as pd

import common as C

# ---- every Phase 3 decision is a default here and a line in splits.json ----
DEFAULTS = dict(
    val=["levent_gultekin"],
    test=["yilmaz_ozdil", "ozlem_gurses"],
    seed=2026,
    require_track_ok=True,      # padded span free of tracking gaps
    min_frames=9,               # item length incl. padding; clip-edge words are shorter
    min_word_sec=0.04,          # Whisper spans under one frame are collapsed alignments
    eval_drop_manual_review=True,   # ASR confidence < 0.7 stays out of val/test, kept in train
    cap_share=0.15,             # no train speaker above this share of the train rows
    min_train=300,              # train examples a class needs
    min_speakers=5,             # train speakers a class needs (not one person's tic)
    keep_digits=False,          # "3", "10": spelled as digits or words at Whisper's whim
    target_classes=500,
    target_clips=150_000,
)


def units_of(df):
    """Connected components of the speaker-channel graph -> list of speaker sets."""
    parent = {}

    def find(x):
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for spk, ch in df[["speaker_id", "channel"]].drop_duplicates().itertuples(index=False):
        parent[find(("s", spk))] = find(("c", ch))
    groups = {}
    for node in list(parent):
        if node[0] == "s":
            groups.setdefault(find(node), set()).add(node[1])
    return sorted((sorted(g) for g in groups.values()), key=lambda g: g[0])


def search_splits(df, chosen_val, chosen_test, min_train, lo=0.06, hi=0.14, max_units=3,
                  top=200):
    """Rank channel-disjoint val/test choices by distance from 10% / 10%.

    Units move whole, so every row is a valid split. Class counts are on the rows as
    they stand before the eval-only filters, the cap and the speaker minimum: a
    comparison between splits, not the final vocabulary.
    """
    units = units_of(df)
    unit_of = {s: i for i, u in enumerate(units) for s in u}
    uid = df.speaker_id.map(unit_of).to_numpy()
    share = np.bincount(uid, minlength=len(units)) / len(df)
    mats = {}
    for lab in C.LABELINGS:
        col = df[C.class_column(lab)]
        total = col.value_counts()
        keep = total.index[total >= min_train]
        m = df[col.isin(keep)]
        mats[lab] = (pd.crosstab(m.speaker_id.map(unit_of), m[C.class_column(lab)])
                     .reindex(range(len(units)), fill_value=0).to_numpy())
    combos = [c for k in range(1, max_units + 1)
              for c in itertools.combinations(range(len(units)), k)
              if lo <= share[list(c)].sum() <= hi]
    chosen = (frozenset(unit_of[s] for s in chosen_val), frozenset(unit_of[s] for s in chosen_test))
    rows = []
    for t in combos:
        for v in combos:
            if set(t) & set(v):
                continue
            held = list(t) + list(v)
            ts, vs = share[list(t)].sum(), share[list(v)].sum()
            row = {
                "score": round(100 * (abs(ts - 0.1) + abs(vs - 0.1)), 2),
                "test_pct": round(100 * ts, 2), "val_pct": round(100 * vs, 2),
                "test_speakers": " ".join(s for i in t for s in units[i]),
                "val_speakers": " ".join(s for i in v for s in units[i]),
                "n_test_speakers": sum(len(units[i]) for i in t),
                "n_val_speakers": sum(len(units[i]) for i in v),
                "chosen": int((frozenset(v), frozenset(t)) == chosen),
            }
            for lab, m in mats.items():
                row[f"{lab}_classes"] = int(((m.sum(0) - m[held].sum(0)) >= min_train).sum())
            rows.append(row)
    out = pd.DataFrame(rows).sort_values(
        ["score", "lemma_classes"], ascending=[True, False]).reset_index(drop=True)
    mine = out[out.chosen == 1]
    closer = int((out.score < mine.score.iloc[0]).sum()) if len(mine) else -1
    return pd.concat([mine, out[out.chosen == 0].head(top)]), closer, len(out)


def cap_train(train, cap_share, seed, rep):
    """Subsample speakers above `cap_share` of the train rows, evenly across lemma classes."""
    limit = int(cap_share * len(train))
    sizes = train.speaker_id.value_counts()
    over = sizes[sizes > limit]
    rep(f"\nTrain cap: no speaker above {cap_share:.0%} of train = {limit:,} rows")
    if over.empty:
        rep(f"  no speaker exceeds it (largest: {sizes.index[0]} {sizes.iloc[0]:,} = "
            f"{sizes.iloc[0] / len(train):.1%}); nothing removed")
        return train.index[:0], {}
    drop, capped = [], {}
    for spk, n in over.items():
        rows = train[train.speaker_id == spk]
        kept = rows.groupby("lemma_cls", group_keys=False).sample(frac=limit / n, random_state=seed)
        drop.append(rows.index.difference(kept.index))
        capped[spk] = {"before": int(n), "after": int(len(kept))}
        rep(f"  {spk}: {n:,} -> {len(kept):,}")
    return drop[0].append(drop[1:]) if len(drop) > 1 else drop[0], capped


def vocabulary(df, labeling, args, stop, rep):
    """-> vocab frame for one labeling; prints what each rule removes."""
    col = C.class_column(labeling)
    n = pd.crosstab(df[col], df["split"]).reindex(columns=C.SPLITS, fill_value=0)
    train = df[df.split == "train"]
    v = pd.DataFrame({
        "train_n": n["train"], "val_n": n["val"], "test_n": n["test"],
        "n_speakers": train.groupby(col).speaker_id.nunique(),
        "n_channels": train.groupby(col).channel.nunique(),
    }).fillna(0).astype(int)
    v = v[v.train_n >= args.min_train]
    rep(f"\n[{labeling}] classes with >= {args.min_train} train examples: {len(v)}")
    few = v.index[v.n_speakers < args.min_speakers]
    rep(f"  - {len(few)} with < {args.min_speakers} train speakers"
        + (": " + ", ".join(few[:20]) if len(few) else ""))
    v = v.drop(few)
    digits = v.index[v.index.str.contains(r"\d")] if not args.keep_digits else v.index[:0]
    rep(f"  - {len(digits)} containing a digit" + (": " + ", ".join(digits) if len(digits) else ""))
    v = v.drop(digits)
    stopped = v.index[v.index.isin(stop)]
    rep(f"  - {len(stopped)} on stoplist.txt" + (": " + ", ".join(stopped) if len(stopped) else ""))
    v = v.drop(stopped).sort_values(["train_n", "val_n"], ascending=False)
    v.index.name = "class"
    flag = "OK " if len(v) >= args.target_classes else "LOW"
    rep(f"  {flag} {len(v)} classes (target {args.target_classes}+), "
        f"{int(v.train_n.sum()):,} train / {int(v.val_n.sum()):,} val / "
        f"{int(v.test_n.sum()):,} test clips in them")
    for split in ["val", "test"]:
        zero = v.index[v[f"{split}_n"] == 0]
        rep(f"  classes with {split}_n == 0 (not evaluable there): {len(zero)}"
            + (": " + ", ".join(zero[:20]) if len(zero) else ""))
    rep(f"  rarest kept: " + ", ".join(f"{k} {r.train_n}" for k, r in v.tail(5).iterrows()))
    return v.reset_index()


def speaker_table(raw_counts, df, split_of):
    t = pd.DataFrame({"words_phase2": raw_counts})
    t["channels"] = df.groupby("speaker_id").channel.agg(lambda s: " | ".join(sorted(set(s))))
    if C.TRACK_REPORT.exists():
        r = pd.read_csv(C.TRACK_REPORT, dtype=str, keep_default_na=False,
                        usecols=["speaker_id", "status", "n_frames"])
        r = r[r.status == "ok"]
        frames = pd.to_numeric(r.n_frames, errors="coerce").fillna(0)
        t["source_clips"] = r.groupby("speaker_id").size()
        t["hours"] = (frames.groupby(r.speaker_id).sum() / C.FPS / 3600).round(2)
    else:
        t["source_clips"] = df.groupby("speaker_id").npz_path.nunique()
    kept = df[df.split.isin(C.SPLITS)]
    t["words_kept"] = kept.groupby("speaker_id").size()
    for lab in C.LABELINGS:
        t[f"words_in_vocab_{lab}"] = kept[kept[f"in_vocab_{lab}"] == 1].groupby("speaker_id").size()
    t["split"] = pd.Series(split_of)
    t = t.fillna({"channels": "", "split": "dropped"}).fillna(0)
    for c in t.columns.difference(["channels", "split", "hours"]):
        t[c] = t[c].astype(int)
    t.index.name = "speaker_id"
    return t.sort_values("words_phase2", ascending=False).reset_index()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    d = DEFAULTS
    ap.add_argument("--words", type=C.Path, default=C.WORDS)
    ap.add_argument("--out", type=C.Path, default=C.PHASE3)
    ap.add_argument("--val", nargs="+", default=d["val"])
    ap.add_argument("--test", nargs="+", default=d["test"])
    ap.add_argument("--seed", type=int, default=d["seed"])
    ap.add_argument("--keep-track-gaps", action="store_true", help="do not require track_ok == 1")
    ap.add_argument("--min-frames", type=int, default=d["min_frames"])
    ap.add_argument("--min-word-sec", type=float, default=d["min_word_sec"])
    ap.add_argument("--keep-manual-review-in-eval", action="store_true",
                    help="keep manual_review == 1 rows in val/test")
    ap.add_argument("--cap-share", type=float, default=d["cap_share"],
                    help="max share of train rows per speaker (1 = no cap)")
    ap.add_argument("--min-train", type=int, default=d["min_train"])
    ap.add_argument("--min-speakers", type=int, default=d["min_speakers"])
    ap.add_argument("--keep-digits", action="store_true")
    ap.add_argument("--stoplist", type=C.Path, default=C.STOPLIST)
    ap.add_argument("--target-classes", type=int, default=d["target_classes"])
    ap.add_argument("--target-clips", type=int, default=d["target_clips"])
    ap.add_argument("--no-search", action="store_true", help="skip stats/split_search.csv")
    args = ap.parse_args()
    C.utf8_stdout()
    rep = C.Report()

    rep(f"Phase 3 build_splits  {date.today().isoformat()}")
    df, n_bad = C.load_words(args.words)
    n_input = len(df)
    phase2_columns = list(df.columns)
    rep(f"input: {args.words.name}  {n_input:,} rows ({n_bad} malformed rows skipped)")
    assert df.clip_id.is_unique, "clip_id is not unique in the word manifest"
    C.add_classes(df)
    raw_counts = df.groupby("speaker_id").size()
    reason = pd.Series("", index=df.index)

    def drop(mask, why):
        reason[mask & (reason == "")] = why

    # ---- step 2: quality filter. Every rule is reported on its own first. ----
    n_frames = df.n_frames.astype(int)
    dur = df.word_end_sec.astype(float) - df.word_start_sec.astype(float)
    rules = [
        ("no_label", ~C.label_mask(df), "empty label or bare apostrophe suffix"),
        ("track_gap", (df.track_ok != "1") & (not args.keep_track_gaps), "track_ok != 1"),
        ("too_few_frames", n_frames < args.min_frames, f"n_frames < {args.min_frames}"),
        ("word_too_short", dur < args.min_word_sec, f"word span < {args.min_word_sec} s"),
    ]
    rep("\nQuality filter (rows each rule would remove on its own):")
    for why, mask, text in rules:
        rep(f"  {why:16} {int(mask.sum()):8,}  {text}")
    for why, mask, _ in rules:
        drop(mask, why)
    rep(f"  removed together: {int((reason != '').sum()):,}")
    mr = df.manual_review == "1"
    rep(f"  manual_review == 1 (ASR confidence < 0.7): {int(mr.sum()):,} = {mr.mean():.1%} "
        f"of rows; " + ("kept everywhere" if args.keep_manual_review_in_eval
                        else "kept in train, removed from val/test"))
    fh = pd.to_numeric(df.face_height_px, errors="coerce")
    rep(f"  face_height_px (recorded, never gated on): {int(fh.isna().sum()):,} empty, "
        f"{(fh < 150).mean():.1%} under 150 px, {(fh < 220).mean():.1%} under 220 px")

    # ---- step 3: speaker split ----
    speakers = set(df.speaker_id)
    for name, chosen in [("val", args.val), ("test", args.test)]:
        unknown = set(chosen) - speakers
        if unknown:
            raise SystemExit(f"--{name}: unknown speakers {sorted(unknown)}")
    if set(args.val) & set(args.test):
        raise SystemExit("a speaker cannot be in both val and test")
    split_of = {s: "train" for s in speakers}
    split_of.update({s: "val" for s in args.val})
    split_of.update({s: "test" for s in args.test})
    df["split"] = df.speaker_id.map(split_of)

    test_channels = set(df.loc[df.split == "test", "channel"])
    leaked = (df.split == "train") & df.channel.isin(test_channels)
    if leaked.any():
        rep(f"\n{int(leaked.sum()):,} train rows are on a test channel "
            f"({', '.join(sorted(set(df.loc[leaked, 'speaker_id'])))}): dropped")
    drop(leaked, "test_channel")
    if not args.keep_manual_review_in_eval:
        drop(mr & df.split.isin(["val", "test"]), "manual_review_eval")

    if not args.no_search:
        table, closer, total = search_splits(df[reason.isin(["", "manual_review_eval"])],
                                             args.val, args.test, args.min_train)
        C.write_csv(table, args.out / "stats" / "split_search.csv")
        rep(f"\nSplit search: {total:,} channel-disjoint val/test choices with 6-14% each; "
            + (f"{closer:,} are closer to 10%/10% than the chosen one" if closer >= 0
               else "the chosen one is outside that range")
            + " (best 200 in stats/split_search.csv, chosen split first)")

    keep = reason == ""
    cap_drop, capped = cap_train(df[keep & (df.split == "train")], args.cap_share, args.seed, rep)
    reason[cap_drop] = "train_cap"
    keep = reason == ""
    df.loc[~keep, "split"] = "dropped"
    kept = df[keep]

    rep(f"\nSplit ({len(kept):,} word clips kept of {n_input:,}):")
    ratios = {}
    for name in C.SPLITS:
        part = kept[kept.split == name]
        ratios[name] = len(part) / len(kept)
        spk = sorted(part.speaker_id.unique())
        rep(f"  {name:5} {len(part):9,} ({ratios[name]:5.1%})  {len(spk):2d} speakers, "
            f"{part.channel.nunique():2d} channels"
            + ("" if name == "train" else "  " + ", ".join(spk)))
    top = kept[kept.split == "train"].speaker_id.value_counts(normalize=True)
    rep(f"  largest train speakers: " + ", ".join(f"{k} {v:.1%}" for k, v in top.head(4).items()))

    # ---- step 4: vocabulary, counted on the final train split ----
    stop = C.load_stoplist(args.stoplist)
    vocab = {}
    for lab in C.LABELINGS:
        vocab[lab] = vocabulary(kept, lab, args, stop, rep)
        in_vocab = df[C.class_column(lab)].isin(set(vocab[lab]["class"])) & keep
        df[f"in_vocab_{lab}"] = in_vocab.astype(int)
        df[f"label_{lab}"] = df[C.class_column(lab)].where(in_vocab, "")
    kept = df[keep]

    # ---- hard assertions (plan step 3 and the definition of done) ----
    per_speaker = kept.groupby("speaker_id").split.nunique()
    assert (per_speaker == 1).all(), "a speaker is in two splits"
    ch = {name: set(kept.loc[kept.split == name, "channel"]) for name in C.SPLITS}
    assert not ch["train"] & ch["test"], f"train/test share channels: {ch['train'] & ch['test']}"
    for lab in C.LABELINGS:
        assert (vocab[lab].train_n >= args.min_train).all()
        assert (vocab[lab].n_speakers >= args.min_speakers).all()
    rep("\nAssertions passed: speaker-disjoint splits, no channel shared by train and test, "
        "every class meets the train and speaker minimum.")
    rep(f"  train/val channel overlap (reported, not required): "
        f"{sorted(ch['train'] & ch['val']) or 'none'}")
    rep(f"  val/test channel overlap: {sorted(ch['val'] & ch['test']) or 'none'}")

    # ---- write ----
    out_columns = phase2_columns + ["label_surface", "label_lemma", "split",
                                    "in_vocab_surface", "in_vocab_lemma"]
    for name in C.SPLITS:
        C.write_csv(kept.loc[kept.split == name, out_columns], args.out / f"word_{name}.csv")
    for lab in C.LABELINGS:
        C.write_csv(vocab[lab], args.out / f"vocab_{lab}.csv")
    mapping = (kept[(kept.in_vocab_surface == 1) | (kept.in_vocab_lemma == 1)]
               .groupby(["surface", "lemma_cls"]).agg(
                   n=("clip_id", "size"), in_vocab_surface=("in_vocab_surface", "max"),
                   in_vocab_lemma=("in_vocab_lemma", "max"))
               .reset_index().rename(columns={"lemma_cls": "lemma"})
               .sort_values(["lemma", "n"], ascending=[True, False]))
    C.write_csv(mapping, args.out / "surface_to_lemma.csv")
    dropped = df.loc[~keep, ["clip_id", "speaker_id"]].assign(reason=reason[~keep])
    C.write_csv(dropped, args.out / "dropped.csv")
    C.write_csv(speaker_table(raw_counts, df, split_of), args.out / "speaker_table.csv")

    # ---- targets ----
    rep("\nAchieved vs targets:")
    achieved = {}
    for lab in C.LABELINGS:
        n_cls = len(vocab[lab])
        n_clips = int((kept[f"in_vocab_{lab}"] == 1).sum())
        achieved[lab] = {
            "classes": n_cls, "word_clips_in_vocab": n_clips,
            "train_clips_in_vocab": int(vocab[lab].train_n.sum()),
            "val_clips_in_vocab": int(vocab[lab].val_n.sum()),
            "test_clips_in_vocab": int(vocab[lab].test_n.sum()),
            "min_train_per_class": int(vocab[lab].train_n.min()) if n_cls else 0,
        }
        rep(f"  {lab:8} classes {n_cls} (target {args.target_classes}+: "
            f"{'met' if n_cls >= args.target_classes else 'NOT met'});  in-vocab word clips "
            f"{n_clips:,} (target {args.target_clips:,}+: "
            f"{'met' if n_clips >= args.target_clips else 'NOT met'});  "
            f"min train/class {achieved[lab]['min_train_per_class']}")
    rep(f"  split ratio train/val/test: " + " / ".join(f"{ratios[s]:.1%}" for s in C.SPLITS)
        + "  (target 80/10/10)")

    info = {
        "created": date.today().isoformat(),
        "input": {"path": args.words.resolve().relative_to(C.ROOT).as_posix(),
                  "rows": n_input, "sha256": C.sha256(args.words)},
        "speakers": {name: sorted(s for s, v in split_of.items() if v == name)
                     for name in C.SPLITS},
        "channels": {name: sorted(ch[name]) for name in C.SPLITS},
        "parameters": {
            "seed": args.seed, "require_track_ok": not args.keep_track_gaps,
            "min_frames": args.min_frames, "min_word_sec": args.min_word_sec,
            "eval_drop_manual_review": not args.keep_manual_review_in_eval,
            "cap_share": args.cap_share, "min_train": args.min_train,
            "min_speakers": args.min_speakers, "keep_digits": args.keep_digits,
            "stoplist": sorted(stop),
        },
        "train_cap": capped,
        "dropped": {k: int(v) for k, v in reason[reason != ""].value_counts().items()},
        "rows": {name: int((kept.split == name).sum()) for name in C.SPLITS},
        "ratios": {name: round(ratios[name], 4) for name in C.SPLITS},
        "achieved": achieved,
    }
    C.write_text(args.out / "splits.json", json.dumps(info, ensure_ascii=False, indent=2) + "\n")
    rep.save(args.out / "stats" / "build_report.txt")
    print(f"\nwrote {args.out.relative_to(C.ROOT).as_posix()}/")


if __name__ == "__main__":
    main()
