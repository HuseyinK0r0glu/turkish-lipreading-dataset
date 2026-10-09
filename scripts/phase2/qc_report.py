"""Phase 2 quality control (PHASE2_PLAN §7): contact sheets + distribution summary.

    uv run python scripts/phase2/qc_report.py
    uv run python scripts/phase2/qc_report.py --per-speaker 12 --speakers kubra_par

Writes data/phase2/qc/montage_<speaker>.png -- one row per random word clip, frames
sampled evenly across its window, the word written under the middle frame (which
should show the mouth mid-articulation) -- and data/phase2/qc/summary.txt.
"""
import argparse
import sys
import unicodedata
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import manifests  # noqa: E402

STRIP_FRAMES = 7
TILE = 88
LABEL_H = 18


def ascii_label(text: str) -> str:
    """cv2's Hershey fonts have no Turkish letters: fold them for the overlay."""
    text = text.replace("ı", "i").replace("İ", "I")
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()


def strip(frames: np.ndarray, label: str) -> np.ndarray:
    idx = np.linspace(0, len(frames) - 1, STRIP_FRAMES).round().astype(int)
    row = np.hstack([frames[i] for i in idx])
    out = np.full((TILE + LABEL_H, row.shape[1]), 255, np.uint8)
    out[:TILE] = row
    mid_x = (STRIP_FRAMES // 2) * TILE + 2
    cv2.putText(out, ascii_label(label)[:24], (mid_x, TILE + 13), cv2.FONT_HERSHEY_SIMPLEX,
                0.4, 0, 1, cv2.LINE_AA)
    return out


def montages(words: pd.DataFrame, qc_dir: Path, per_speaker: int, seed: int) -> None:
    cache = {}
    for speaker, group in words.groupby("speaker_id"):
        sample = group.sample(min(per_speaker, len(group)), random_state=seed)
        rows = []
        for r in sample.itertuples():
            if r.npz_path not in cache:
                cache[r.npz_path] = np.load(manifests.PROJECT / r.npz_path)["frames"]
            rows.append(strip(cache[r.npz_path][int(r.frame_start):int(r.frame_end)],
                              r.surface_form))
        path = qc_dir / f"montage_{speaker}.png"
        cv2.imwrite(str(path), np.vstack(rows))
        print(f"wrote {path}")
        cache.clear()


def describe(series: pd.Series) -> str:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return "n/a"
    q = s.quantile([0.05, 0.5, 0.95])
    return f"n={len(s)}  p5={q[0.05]:.2f}  median={q[0.5]:.2f}  p95={q[0.95]:.2f}"


def summary(out_dir: Path) -> str:
    words = pd.read_csv(out_dir / "word_clips.csv")
    sents = pd.read_csv(out_dir / "sentence_clips.csv")
    sync = pd.read_csv(out_dir / "syncnet_scores.csv")
    disc = pd.read_csv(out_dir / "discarded.csv")
    report = pd.read_csv(out_dir / "face_track_report.csv")
    lines = [
        f"source clips processed: {len(report)}",
        "  by status: " + ", ".join(f"{k}={v}" for k, v in report["status"].value_counts().items()),
        f"  good-frame share: {report['good_frames'].sum() / max(report['n_frames'].sum(), 1):.3f}",
        f"word clips: {len(words)}   distinct surface forms: {words['surface_form'].nunique()}",
        f"  duration (s):   {describe(words['n_frames'] / 25)}",
        f"  word_sync_conf: {describe(words['word_sync_conf'])}",
        f"  lip_motion:     {describe(words['lip_motion'])}",
        f"  mouth_width_px: {describe(words['mouth_width_px'])}",
        f"  manual_review share: {pd.to_numeric(words['manual_review']).mean():.3f}",
        f"sentence clips: {len(sents)}",
        f"  duration (s):   {describe(sents['n_frames'] / 25)}",
        f"syncnet_score:    {describe(sync['syncnet_score'])}",
        f"syncnet_offset:   {describe(sync['syncnet_offset'])}",
        "discards:",
    ]
    if not disc.empty:
        for (level, reason), n in disc.groupby(["level", "reason"]).size().items():
            lines.append(f"  {level:<12} {reason:<20} {n}")
    lines.append("per speaker:  clips  words  word_h  sentences")
    for spk, g in report.groupby("speaker_id"):
        w = words[words["speaker_id"] == spk]
        lines.append(f"  {spk:<20} {len(g):>6} {len(w):>7} {w['n_frames'].sum() / 25 / 3600:>7.2f} "
                     f"{(sents['speaker_id'] == spk).sum():>9}")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--out-dir", type=Path, default=manifests.PHASE2_DIR)
    p.add_argument("--per-speaker", type=int, default=8)
    p.add_argument("--speakers", nargs="+")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    qc_dir = args.out_dir / "qc"
    qc_dir.mkdir(parents=True, exist_ok=True)
    words = pd.read_csv(args.out_dir / "word_clips.csv", keep_default_na=False)
    if args.speakers:
        words = words[words["speaker_id"].isin(args.speakers)]
    montages(words, qc_dir, args.per_speaker, args.seed)
    text = summary(args.out_dir)
    (qc_dir / "summary.txt").write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
