"""Phase 3 step 1: audit the Phase 2 manifests before anything is built on them.

Read-only over data/phase2/. Asserts the invariants Phase 3 relies on (unique clip_id,
fps 25, frame_end - frame_start == n_frames, every npz on disk, no empty speaker) and
writes a one-page summary to data/phase3/stats/audit.txt.

    uv run python scripts/phase3/audit_manifests.py
    uv run python scripts/phase3/audit_manifests.py --npz-sample 1000   # quick npz check
"""
import argparse
import json

import pandas as pd

import common as C


def check(rep, ok: bool, text: str, failures: list):
    rep(f"  {'ok  ' if ok else 'FAIL'} {text}")
    if not ok:
        failures.append(text)


def audit(name, df, n_bad, rep, failures, npz_sample, seed):
    rep(f"\n== {name}: {len(df):,} rows ({n_bad} malformed rows skipped) ==")
    check(rep, df.clip_id.is_unique, "clip_id unique", failures)
    check(rep, bool((df.fps == "25").all()), "fps == 25 on every row", failures)
    n = df.n_frames.astype(int)
    span = df.frame_end.astype(int) - df.frame_start.astype(int)
    check(rep, bool((span == n).all()), "frame_end - frame_start == n_frames", failures)
    check(rep, bool((n > 0).all()), "n_frames > 0", failures)
    check(rep, bool((df.speaker_id != "").all()), "no empty speaker_id", failures)
    empty_ch = int((df.channel == "").sum())
    check(rep, empty_ch == 0,
          f"no empty channel ({empty_ch:,} empty: they would break channel independence)",
          failures)
    check(rep, bool(df.npz_path.map(lambda p: "\\" not in p).all()), "npz_path is posix-style",
          failures)

    paths = pd.Series(df.npz_path.unique())
    if npz_sample and npz_sample < len(paths):
        paths = paths.sample(npz_sample, random_state=seed)
    missing = [p for p in paths if not (C.ROOT / p).exists()]
    check(rep, not missing,
          f"npz files on disk ({len(paths):,} checked, {len(missing)} missing)", failures)
    for p in missing[:5]:
        rep(f"       missing: {p}")

    empties = {c: int((df[c] == "").sum()) for c in df.columns if (df[c] == "").any()}
    rep(f"  empty values per column: {empties or 'none'}")
    rep(f"  source clips (npz): {df.npz_path.nunique():,}   speakers: {df.speaker_id.nunique()}"
        f"   channels: {df.channel.nunique()}")
    rep(f"  frames in items: {int(n.sum()):,} ({n.sum() / C.FPS / 3600:.1f} h incl. overlap)")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--npz-sample", type=int, default=0,
                    help="check only this many npz paths per manifest (0 = all)")
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()
    C.utf8_stdout()
    rep, failures = C.Report(), []

    rep("Phase 3 audit of the Phase 2 manifests")
    w, w_bad = C.load_words()
    audit("word_clips.csv", w, w_bad, rep, failures, args.npz_sample, args.seed)
    for col in ["track_ok", "manual_review"]:
        rep(f"  {col}: {w[col].value_counts().to_dict()}")
    C.add_classes(w)
    rep(f"  rows without a word label (empty / bare suffix): {int((~C.label_mask(w)).sum()):,}")

    per_speaker = w.groupby("speaker_id").agg(
        words=("clip_id", "size"), source_clips=("npz_path", "nunique"),
        channels=("channel", lambda s: " | ".join(sorted(set(s)))))
    del w
    s, s_bad = C.load_sentences()
    audit("sentence_clips.csv", s, s_bad, rep, failures, args.npz_sample, args.seed)
    bad_spans = 0
    for spans in s.word_spans.sample(min(2000, len(s)), random_state=args.seed):
        try:
            assert isinstance(json.loads(spans), list)
        except (ValueError, AssertionError):
            bad_spans += 1
    check(rep, bad_spans == 0, f"word_spans parses as JSON (2000 sampled, {bad_spans} bad)",
          failures)
    per_speaker["sentences"] = s.groupby("speaker_id").size()
    per_speaker["sentences"] = per_speaker["sentences"].fillna(0).astype(int)

    rep("\n== per speaker ==")
    rep(per_speaker.sort_values("words", ascending=False).to_string())
    multi = per_speaker[per_speaker.channels.str.contains(r" \| ")]
    if len(multi):
        rep(f"\nspeakers on more than one channel (they tie those channels together in a "
            f"split): {', '.join(multi.index)}")
    rep("\n== per channel (sentences) ==")
    rep(s.groupby("channel").agg(sentences=("clip_id", "size"),
                                 speakers=("speaker_id", "nunique")).to_string())

    for path in [C.TRACK_REPORT, C.DISCARDED]:
        if path.exists():
            col = "status" if path == C.TRACK_REPORT else "reason"
            d = pd.read_csv(path, dtype=str, keep_default_na=False, usecols=[col])
            rep(f"\n{path.name} {col}: {d[col].value_counts().to_dict()}")

    rep(f"\n{'ALL CHECKS PASSED' if not failures else 'FAILED: ' + '; '.join(failures)}")
    rep.save(C.PHASE3 / "stats" / "audit.txt")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
