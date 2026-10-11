"""Turkish lemmatization for the transcript JSONs (zeyrek), plus a backfill CLI.

Whisper gives surface forms ("evlerde", "kararına"); Phase 3 needs to decide whether
those are one word class or several. This fills the `lemma` field that the
transcribers used to leave equal to `surface_form`.

zeyrek has no sentence-level disambiguation, so `lemmatize_word` picks ONE lemma per
word, deterministically and without context:

  1. punctuation is stripped from the ends of the token ("karar," -> "karar");
  2. among zeyrek's analyses the most plausible for news speech wins: fewest
     morphemes rare in news (optative, imperative, 2nd person; abbreviations; a
     proper-noun reading of a lower-case token; a two-letter open-class root), then
     the shortest analysis, then a closed-class reading (pronoun, conjunction,
     postposition...), then a root left intact at the start of the word, then the
     lemma most analyses propose, then the longer lemma, then the analyzer's order.
     Majority vote alone picked "iç" for "için", "bun" for "bunu", "türki" for
     "Türkiye", "bilemek" for "biliyor".
     Still context-free -- Phase 3 should treat `lemma` as a grouping key, not as
     ground truth;
  3. the lemma is Turkish-lower-cased and circumflex-folded ("âmâ" -> "ama");
  4. a token zeyrek cannot analyse (numbers, "%50", foreign names, ASR noise) falls
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

# Disambiguation (see lemmatize_word). Morphemes that are rare in news speech: an
# analysis that needs one loses to one that does not ("karara" is karar+Dat, not the
# optative of kararmak; "bunu" is bu+Acc, not bun+Acc).
_RARE_IN_NEWS = {"Opt", "Imp", "P2sg", "P2pl", "A2sg", "Neces", "Desr", "Cond"}
_CLOSED_CLASS = {"Pron", "Conj", "Postp", "Det", "Ques", "Adv", "Interj", "Num"}
# zeyrek's lexicon spells some lemmas with circumflexes ("âmâ", "siyasî", "hâlâ");
# speech and Whisper do not, so "ama" and "âmâ" must not be two classes.
_FOLD_CIRCUMFLEX = str.maketrans("âîûÂÎÛ", "aiuAİU")


def _patch_shared_attrs():
    """zeyrek 0.1.3 bug: results depend on which words were analysed before.

    SearchPath.initial hands the stem transition's own `attrs` set to the new search
    path, and the search mutates the path's set in place -- so analysing "vermek"
    permanently edits the lexicon entry for "ver", and "veriyor" afterwards gets no
    analysis at all (and fell back to its surface form). Over a corpus that silently
    made a word's lemma depend on file order. Give every path its own copy.
    """
    from zeyrek.morphotactics import SearchPath
    if getattr(SearchPath, "_attrs_copied", False):
        return
    original = SearchPath.initial.__func__

    def initial(cls, stem_transition, tail):
        path = original(cls, stem_transition, tail)
        path.phonetic_attributes = set(path.phonetic_attributes)
        return path

    SearchPath.initial = classmethod(initial)
    SearchPath._attrs_copied = True


def _get_analyzer():
    global _analyzer
    if _analyzer is None:
        import zeyrek
        # zeyrek logs every analysis candidate ("APPENDING RESULT: ...") at WARNING,
        # which floods the console for a corpus-sized run.
        logging.getLogger("zeyrek").setLevel(logging.ERROR)
        _patch_shared_attrs()
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
    if not analyses:
        return turkish_lower(word).translate(_FOLD_CIRCUMFLEX)

    votes = collections.Counter(turkish_lower(a.dict_item.lemma) for a in analyses)
    lowered = turkish_lower(word)
    lowercase_input = word[:1] == lowered[:1]

    def rank(item):
        order, a = item
        ids = {m[0].id_ for m in a.morphemes}
        pos = a.dict_item.primary_pos.value
        # a pronoun carries its own person ("sana" = sen+A2sg+Dat): not a rare suffix
        penalty = 0 if pos == "Pron" else len(ids & _RARE_IN_NEWS)
        secondary = str(a.dict_item.secondary_pos)  # "SecondaryPos.ProperNoun", ...
        if "Abbreviation" in secondary:
            penalty += 1
        if lowercase_input and "ProperNoun" in secondary:
            penalty += 1
        lemma = a.dict_item.lemma
        low = turkish_lower(lemma)
        open_class = pos not in _CLOSED_CLASS
        if open_class and len(low) <= 2 and low != lowered:
            penalty += 1  # "dolar" -> do+A3pl, "aslında" -> as: two-letter roots are noise
        root = low[:-3] if pos == "Verb" and low.endswith(("mak", "mek")) else low
        intact = lowered.startswith(root)  # "biliyor": bil(mek) yes, bile(mek) no
        return (penalty, len(a.morphemes), open_class, not intact,
                -votes[low], -len(lemma), order)

    best = min(enumerate(analyses), key=rank)[1]
    # Lower-cased: zeyrek returns "Kemal" (proper) and "kemal" (noun) in an order that
    # follows set iteration, i.e. Python's per-process hash seed, so the spelling of a
    # tie was not reproducible between runs. A grouping key needs no capital.
    return turkish_lower(best.dict_item.lemma).translate(_FOLD_CIRCUMFLEX)


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
