"""Turkish lemmatization for the transcript JSONs (zeyrek), plus a backfill CLI.

Whisper gives surface forms ("evlerde", "kararına"); Phase 3 needs to decide whether
those are one word class or several. This fills the `lemma` field that the
transcribers used to leave equal to `surface_form`.

zeyrek has no sentence-level disambiguation, so `lemmatize_word` picks ONE lemma per
word, deterministically and without context:

  1. punctuation is stripped from the ends of the token ("karar," -> "karar");
  2. the lemma proposed by most of zeyrek's analyses wins, ties go to the analyzer's
     own order. "karara" is therefore "kararmak" or "karar" depending on that order --
     the choice is stable, not always linguistically right. Phase 3 should treat
     `lemma` as a grouping key, not as ground truth;
  3. a token zeyrek cannot analyse (numbers, "%50", foreign names, ASR noise) falls
     back to its Turkish-lowercased surface form, so "Xyz" and "xyz" do not become two
     word classes.

Words are analysed with zeyrek's private `_parse`, not its public `lemmatize`: the
latter runs an NLTK sentence tokenizer that needs a separately downloaded `punkt_tab`
resource, and Whisper tokens are already single words.

Backfill over transcripts written before this existed (idempotent -- the result is
deterministic, and a file is only rewritten when a lemma actually changes):

    uv run python scripts/lemmatize.py --dry-run
    uv run python scripts/lemmatize.py

The write is atomic (temp file + rename), so an interrupted run never leaves a
half-written JSON. Prefer not to run it while run_all.py is live: a clip that is
being transcribed at that moment is written by run_all with lemmas already filled in,
but the two writers should not race on the same file.
"""

import argparse
import collections
import json
import logging
import os
import sys
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TRANSCRIPT_DIRS = [ROOT / "data" / "transcripts", ROOT / "data" / "reporter_transcripts"]

_analyzer = None


def _get_analyzer():
    global _analyzer
    if _analyzer is None:
        import zeyrek
        # zeyrek logs every analysis candidate ("APPENDING RESULT: ...") at WARNING,
        # which floods the console for a corpus-sized run.
        logging.getLogger("zeyrek").setLevel(logging.ERROR)
        _analyzer = zeyrek.MorphAnalyzer()
    return _analyzer


def turkish_lower(text: str) -> str:
    """str.lower() maps "I" to "i" and "İ" to "i̇" (two code points); Turkish needs ı / i."""
    return text.replace("İ", "i").replace("I", "ı").lower()


def _strip_punct(token: str) -> str:
    start, end = 0, len(token)
    while start < end and not token[start].isalnum():
        start += 1
    while end > start and not token[end - 1].isalnum():
        end -= 1
    return token[start:end]


@lru_cache(maxsize=200_000)
def lemmatize_word(surface: str) -> str:
    """One lemma for one Whisper word token. Never empty for a non-empty input."""
    word = _strip_punct(surface)
    if not word:
        return surface  # pure punctuation: nothing to lemmatize

    analyses = _get_analyzer()._parse(word)

    votes = collections.Counter()
    first_seen = {}  # lower-cased lemma -> first spelling, in analyzer order
    for a in analyses:
        lemma = a.dict_item.lemma
        key = turkish_lower(lemma)
        votes[key] += 1
        first_seen.setdefault(key, lemma)
    if not votes:
        return turkish_lower(word)

    # max() returns the first maximal key in iteration order, i.e. the earliest seen.
    best = max(first_seen, key=lambda k: votes[k])
    return first_seen[best]


def lemmatize_transcript(data: dict) -> int:
    """Fill `lemma` on every word in place; return how many words changed."""
    changed = 0
    for seg in data.get("segments", []):
        for w in seg.get("words", []):
            lemma = lemmatize_word(w["surface_form"])
            if w.get("lemma") != lemma:
                w["lemma"] = lemma
                changed += 1
    return changed


def backfill(dirs, dry_run=False):
    files = changed_files = changed_words = 0
    for d in dirs:
        if not d.is_dir():
            print(f"skip (missing): {d}")
            continue
        for path in sorted(d.glob("*.json")):
            files += 1
            data = json.loads(path.read_text(encoding="utf-8"))
            n = lemmatize_transcript(data)
            if not n:
                continue
            changed_files += 1
            changed_words += n
            if not dry_run:
                tmp = path.with_suffix(".json.tmp")
                tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2),
                               encoding="utf-8")
                os.replace(tmp, path)
            if files % 1000 == 0:
                print(f"  {files} files scanned, {changed_files} updated", flush=True)
    verb = "would update" if dry_run else "updated"
    print(f"{files} transcripts scanned; {verb} {changed_files} files "
          f"({changed_words} words)")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dirs", nargs="+", type=Path, default=TRANSCRIPT_DIRS,
                    help="transcript directories (default: data/transcripts and "
                         "data/reporter_transcripts)")
    ap.add_argument("--dry-run", action="store_true", help="count changes, write nothing")
    ap.add_argument("--word", nargs="+", help="print the lemma of these words and exit")
    args = ap.parse_args()

    if args.word:
        for w in args.word:
            print(f"{w} -> {lemmatize_word(w)}")
        return
    backfill(args.dirs, args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
