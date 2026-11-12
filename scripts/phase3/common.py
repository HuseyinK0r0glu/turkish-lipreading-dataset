"""Shared Phase 3 code: paths, tolerant manifest loaders, label normalisation.

Phase 3 is manifest work (plans/PHASE3_PLAN.md): it reads data/phase2/*.csv and writes
data/phase3/. Nothing here opens a video, and nothing writes into data/phase2/.

Label classes are built exactly as vocab_report.py builds them:

  surface  turkish_lower(_strip_punct(surface_form))   "Bir", "bir." -> "bir"
  lemma    turkish_lower(lemma)                        the zeyrek lemma in the manifest

Never str.lower(): Turkish İ/I. Rows whose surface is empty after stripping, or that
are a bare apostrophe suffix ("'nin"), carry no word and are dropped by `label_mask`.
"""
import hashlib
import io
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "phase2"))
from lemmatize import _strip_punct, turkish_lower  # noqa: E402
from manifests import SENTENCE_FIELDS, WORD_FIELDS  # noqa: E402

DATA = ROOT / "data"
PHASE2 = DATA / "phase2"
PHASE3 = DATA / "phase3"
WORDS = PHASE2 / "word_clips.csv"
SENTENCES = PHASE2 / "sentence_clips.csv"
TRACK_REPORT = PHASE2 / "face_track_report.csv"
DISCARDED = PHASE2 / "discarded.csv"
SYNC_SCORES = PHASE2 / "syncnet_scores.csv"
STOPLIST = Path(__file__).with_name("stoplist.txt")

FPS = 25
SPLITS = ["train", "val", "test"]
LABELINGS = ["surface", "lemma"]
APOSTROPHES = ("'", "’")
VOWELS = set("aeıioöuüâîû")


def utf8_stdout():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


class Report:
    """print() that also keeps the lines, so every script can leave its report on disk."""

    def __init__(self):
        self.lines = []

    def __call__(self, text=""):
        print(text, flush=True)
        self.lines.append(str(text))

    def save(self, path: Path):
        write_text(path, "\n".join(self.lines) + "\n")


def write_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def write_csv(df: pd.DataFrame, path: Path):
    """Atomic UTF-8 CSV: a re-run replaces the file or leaves the old one intact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    df.to_csv(tmp, index=False, encoding="utf-8", lineterminator="\n")
    os.replace(tmp, path)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def read_manifest(path: Path, fields, usecols=None):
    """A Phase 2 CSV as an all-string frame -> (frame, n_malformed_rows_dropped).

    Phase 2 appended these on Windows through power cuts; `manifests.strip_nul_tail`
    repairs them, but Phase 3 must not write there, so a NUL tail or a final row cut
    off mid-line is tolerated here instead: stripped in memory, counted, reported.
    """
    raw = Path(path).read_bytes()
    clean = raw.rstrip(b"\0")
    if not clean.endswith(b"\n"):
        clean = clean[:clean.rfind(b"\n") + 1]
    cut = len(raw) != len(clean)
    df = pd.read_csv(io.BytesIO(clean), dtype=str, keep_default_na=False, na_filter=False,
                     encoding="utf-8", on_bad_lines="warn", usecols=usecols)
    del raw, clean
    missing = [c for c in (usecols or fields) if c not in df.columns]
    if missing:
        raise SystemExit(f"{path}: missing columns {missing}")
    df = df.fillna("")
    bad = (df["clip_id"] == "") | df["clip_id"].str.contains("\0", regex=False)
    if "fps" in df.columns:
        bad |= df["fps"] == ""
    n_bad = int(bad.sum()) + int(cut)
    return df[~bad].reset_index(drop=True), n_bad


def load_words(path: Path = WORDS, usecols=None):
    return read_manifest(path, WORD_FIELDS, usecols)


def load_sentences(path: Path = SENTENCES, usecols=None):
    return read_manifest(path, SENTENCE_FIELDS, usecols)


def surface_class(surface_form: str) -> str:
    return turkish_lower(_strip_punct(surface_form))


def lemma_class(lemma: str) -> str:
    return turkish_lower(lemma)


def add_classes(df: pd.DataFrame) -> pd.DataFrame:
    """Add `surface` and `lemma_cls` (the class keys); mapped over unique values."""
    surf = {s: surface_class(s) for s in df["surface_form"].unique()}
    lem = {s: lemma_class(s) for s in df["lemma"].unique()}
    df["surface"] = df["surface_form"].map(surf)
    df["lemma_cls"] = df["lemma"].map(lem)
    return df


def label_mask(df: pd.DataFrame) -> pd.Series:
    """True for rows that carry a word (see module docstring)."""
    return (df["surface"] != "") & ~df["surface_form"].str[:1].isin(APOSTROPHES)


def class_column(labeling: str) -> str:
    return {"surface": "surface", "lemma": "lemma_cls"}[labeling]


def syllables(word: str) -> int:
    """Turkish syllable count = vowel count."""
    return sum(1 for c in word if c in VOWELS)


def load_stoplist(path: Path = STOPLIST) -> set:
    if not path.exists():
        return set()
    out = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            out.add(turkish_lower(line))
    return out


def show(path: Path) -> str:
    """Repo-relative posix path for messages; the path as given when it is elsewhere."""
    try:
        return Path(path).resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return Path(path).as_posix()


def require(path: Path, made_by: str) -> Path:
    if not path.exists():
        raise SystemExit(f"missing {show(path)} - run {made_by} first")
    return path
