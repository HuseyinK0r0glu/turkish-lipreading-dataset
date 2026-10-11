"""How far the word clips are from the vocabulary target (plan: 500+ classes with
>= 300 TRAINING examples each, 150k+ word clips).

Read-only over data/phase2/word_clips.csv. Counts classes two ways, the two labelings
the plan's ablation compares:

  surface  the spoken form, Turkish-lower-cased with edge punctuation stripped, so
           "Bir", "bir" and "bir." are one class ("ekonomide" != "ekonomi")
  lemma    the zeyrek lemma (scripts/lemmatize.py), so "ekonomi", "ekonomide",
           "ekonominin" are one class

"Training examples" means after a speaker-independent split, not the whole pool:
validation speakers and test speakers (whose channels must not occur in train) are
held out first. The default split holds out personal-channel speakers, which keeps
the test set channel-independent; pass your own with --val/--test.

    uv run python scripts/phase3/vocab_report.py
    uv run python scripts/phase3/vocab_report.py --min-train 300 --top 30
    uv run python scripts/phase3/vocab_report.py --test yilmaz_ozdil ozlem_gurses --val levent_gultekin
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from lemmatize import _strip_punct, lemmatize_word, turkish_lower  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
WORDS = ROOT / "data" / "phase2" / "word_clips.csv"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--words", type=Path, default=WORDS)
    ap.add_argument("--test", nargs="+", default=["yilmaz_ozdil", "ozlem_gurses"])
    ap.add_argument("--val", nargs="+", default=["levent_gultekin"])
    ap.add_argument("--min-train", type=int, default=300)
    ap.add_argument("--target", type=int, default=500)
    ap.add_argument("--top", type=int, default=0, help="also list the N classes at the cut")
    ap.add_argument("--relemmatize", action="store_true",
                    help="recompute lemmas with the current lemmatize.py instead of "
                         "trusting the manifest (for manifests written before it)")
    args = ap.parse_args()

    w = pd.read_csv(args.words, usecols=["surface_form", "lemma", "speaker_id", "channel"],
                    dtype=str, keep_default_na=False)
    w["surface"] = w.surface_form.map(lambda s: turkish_lower(_strip_punct(s)))
    if args.relemmatize:
        lem = {s: lemmatize_word(s) for s in w.surface_form.unique()}
        w["lemma"] = w.surface_form.map(lem)
    else:
        w["lemma"] = w.lemma.map(turkish_lower)
    # bare apostrophe suffixes ("'nin") from transcripts segments.py did not join
    w = w[(w.surface != "") & ~w.surface_form.str[:1].isin(["'", "’"])]

    test_channels = set(w.loc[w.speaker_id.isin(args.test), "channel"])
    split = pd.Series("train", index=w.index)
    split[w.speaker_id.isin(args.val)] = "val"
    split[w.speaker_id.isin(args.test)] = "test"
    leaked = w[(split == "train") & w.channel.isin(test_channels)]
    split[leaked.index] = "dropped"  # train rows on a test channel break channel independence

    n = len(w)
    print(f"word clips: {n:,}   speakers: {w.speaker_id.nunique()}")
    for name in ["train", "val", "test", "dropped"]:
        k = int((split == name).sum())
        if k:
            spk = sorted(w.loc[split == name, "speaker_id"].unique())
            print(f"  {name:7} {k:9,} ({100 * k / n:4.1f}%)  "
                  + (", ".join(spk) if name != "train" else f"{len(spk)} speakers"))
    if leaked.size:
        print(f"  ({len(leaked):,} train rows dropped: their channel is a test channel)")

    train = w[split == "train"]
    print(f"\nclasses with >= {args.min_train} training examples (target {args.target}+):")
    for col in ["surface", "lemma"]:
        c = train[col].value_counts()
        ok = c[c >= args.min_train]
        covered = int(ok.sum())
        flag = "OK " if len(ok) >= args.target else "LOW"
        print(f"  {flag} {col:8} {len(ok):5d} classes, {covered:,} train clips in them "
              f"(unique {col}s: {len(c):,})")
        if args.top:
            cut = c.iloc[max(0, args.target - args.top // 2): args.target + args.top // 2]
            print("       around rank", args.target, ":",
                  ", ".join(f"{k} {v}" for k, v in cut.items()))


if __name__ == "__main__":
    main()
