# Phase 3 — Dataset compilation and benchmark design

Written 2026-11-09 on a machine that has no Phase 2 output. Implementation happens on the
machine where `data/` lives. Everything below is derived from the Phase 2 *code*
(`scripts/phase2/manifests.py`, `segments.py`, `source_clip.py`, `run_phase2.py`), not from
`PHASE2_PLAN.md`, which is partly stale (see §1.3).

Phase 3 is **manifest work**: it reads CSVs, decides labels / splits / vocabulary, and writes
new CSVs. It never touches video, never re-runs Whisper or SyncNet, and never rewrites a
Phase 2 file. The `.npz` crops are only *referenced* by path; they are loaded in Phase 4.

---

## 1. Inputs (read-only) — exactly what Phase 2 leaves behind

All paths relative to the repo root. All CSVs are UTF-8; **sentence rows carry a
`word_spans` JSON string, so call `csv.field_size_limit(2**31 - 1)`** before using the `csv`
module (pandas is fine).

| Path | Phase 3 use |
|------|-------------|
| `data/phase2/word_clips.csv` | THE word-level input: one row per word item |
| `data/phase2/sentence_clips.csv` | THE sentence-level input (spotting eval + `word_spans` ground-truth seed) |
| `data/phase2/syncnet_scores.csv` | per source clip; `passed` flag; optional QC |
| `data/phase2/discarded.csv` | what Phase 2 dropped and why (stats only) |
| `data/phase2/face_track_report.csv` | per source clip `status` (ok / low_sync / no_words); stats only |
| `data/lip_crops/<reporter>/<clip stem>.npz` | NOT read in Phase 3; paths are carried through |
| `video_links.csv` | NOT needed: `channel` is already in the Phase 2 manifests |
| `data/reporter_clips.csv` | NOT needed (Phase 2 already joined what matters) |

### 1.1 Word manifest columns (`WORD_FIELDS`)

`clip_id, npz_path, frame_start, frame_end, speaker_id, channel, source_clip, source_video,
face_height_px, mouth_width_px, clip_start_sec, clip_end_sec, n_frames, fps,
syncnet_score, syncnet_offset, lip_motion, mean_yaw, mean_pitch, surface_form, lemma,
word_start_sec, word_end_sec, confidence, manual_review, word_sync_conf, track_ok`

### 1.2 Sentence manifest columns (`SENTENCE_FIELDS`)

Same spine, plus `text, sentence_start_sec, sentence_end_sec, split_index, n_words,
sentence_sync_conf, word_spans`.

### 1.3 Facts that differ from PHASE2_PLAN.md (trust these)

1. **Crops are stored once per SOURCE clip**, not per word: `npz_path` is
   `data/lip_crops/<reporter>/<stem>.npz` and a word/sentence item is the frame range
   `[frame_start, frame_end)` into its `frames` array (CLAUDE.md invariant 11). Many rows
   share one `npz_path`. Phase 3 must **keep `npz_path`, `frame_start`, `frame_end`** in every
   output row; without them Phase 4 cannot find the pixels. The npz holds
   `frames (T,88,88) uint8`, `good`, `sync_conf`, `fps`.
2. `lemma` is real (zeyrek), already in the manifest. No separate lemma file.
3. There is no `sync_ok` column. Rows in these manifests **already passed** the clip-level
   (`SYNC_MIN` 3.0) and per-word (`WORD_SYNC_MIN` 2.5) sync gates; failures are in
   `discarded.csv`. Phase 3 may filter *further* but does not need to redo those gates.
4. `word_spans[].start_sec / end_sec` inside a sentence row are **relative to that sentence
   clip's start** (already shifted by `syncnet_offset` and by the window start). They are
   not source-clip times. Convert to frames with `round(t * fps)` (+ `frame_start` for an
   index into the npz).
5. `manual_review` is `1` when Whisper `confidence < 0.7`; `track_ok` is 0/1 (no tracking
   gap in the padded span); `face_height_px` may be empty (= unmeasurable, not zero);
   `lip_motion` has no calibrated threshold.
6. `surface_form` is NFC + stripped but **not** case-folded or punctuation-stripped.
   Reuse `scripts/lemmatize.py` helpers exactly as `scripts/phase3/vocab_report.py` does:
   `turkish_lower(_strip_punct(s))` for the surface class, `turkish_lower(lemma)` for the
   lemma class. Do not use `str.lower()` (Turkish İ/I).
7. Drop word rows whose `surface_form` is empty after stripping or starts with `'`/`’`
   (bare suffixes), as `vocab_report.py` does.

### 1.4 First thing to do on the data machine: sanity check (before any code)

```powershell
uv run python scripts/phase2/phase2_status.py          # confirm run 3 finished, 0 failed
uv run python scripts/phase3/vocab_report.py           # must print 367 surface / 507 lemma
```

Expected (CLAUDE.md, 2026-10-19): 72,845 source clips; 1,142,706 word and 174,653 sentence
items (before the 20-clip ozlem_gurses removal, so slightly fewer). If row counts are far
off, stop and find out why before building on them.

---

## 2. Outputs — where Phase 3 writes

Everything goes to a new folder **`data/phase3/`** (gitignored, as all of `data/`). Phase 3
creates nothing anywhere else and edits no Phase 1/2 file. Code lives in `scripts/phase3/`
(its own folder, importing `scripts/lemmatize.py`, never editing Phase 1/2 code, same rule
Phase 2 followed).

```
data/phase3/
├── vocab_surface.csv        class list under surface labeling
├── vocab_lemma.csv          class list under lemma labeling
├── speaker_table.csv        per speaker: channel, clips, hours, words, assigned split
├── splits.json              speaker -> split assignment + parameters used (reproducibility)
├── word_train.csv  word_val.csv  word_test.csv        (word items, with label columns)
├── sentence_val.csv  sentence_test.csv                (spotting eval pool; see §4)
├── spotting_gt/             spotting ground truth (auto seed + manually corrected)
├── verification/            Label Studio sample + annotations + kappa
├── stats/                   frequency tables, distribution plots/CSVs for the paper
└── release/                 ONLY what is public (metadata + split files + vocab)
```

### 2.1 Output row schema

Each `word_*.csv` row = the Phase 2 word row (all columns kept, so nothing needs re-joining)
plus:

| new column | meaning |
|------------|---------|
| `label_surface` | stripped, Turkish-lower-cased surface class, empty if not in vocab |
| `label_lemma` | lower-cased lemma class, empty if not in vocab |
| `split` | `train` / `val` / `test` |
| `in_vocab_surface`, `in_vocab_lemma` | 0/1 |

Sentence rows keep the Phase 2 columns plus `split`. Phase 4 loads pixels via
`np.load(npz_path)["frames"][frame_start:frame_end]`; **the data machine must therefore keep
`data/lip_crops/` at the same repo-relative location**, and paths stay posix-style.

Clip IDs are stable and unique (`<stem>_w<idx>` / `<stem>_s<idx>`): use them as keys,
never row numbers.

---

## 3. Steps

### Step 1 — Load, clean, audit  (`scripts/phase3/common.py`)
- Single loader for both manifests (dtype=str then cast), adds `label_surface`/`label_lemma`.
- Assert: unique `clip_id`; every `npz_path` exists on disk (sample-check 1000 if slow, full
  check once); `fps == 25`; `frame_end - frame_start == n_frames`; no empty `speaker_id` or
  `channel` (report count of empty channels — they would break channel-independence).
- Write a one-page audit to `data/phase3/stats/audit.txt` (counts, empties, per-channel and
  per-speaker totals).

### Step 2 — Quality filter (word level), parameters in one config block
Candidates, each reported as "rows removed" before applying, so each choice is visible:
- `track_ok == 1` (padded span free of tracking gaps) — recommended on.
- `manual_review`: keep for **train**, but **exclude from val/test** (low-confidence
  Whisper labels would make eval noisy). Decide after seeing the share.
- `face_height_px` floor: **do not apply globally** (it would delete whole studios,
  CLAUDE.md). Report accuracy-relevant breakdown instead; optionally expose a flag.
- Word duration sanity: Phase 2 already drops > 2 s; additionally consider a minimum
  (e.g. `n_frames >= 9`) — check the distribution first.

### Step 3 — Speaker split (the core decision)
Constraints (project plan): val speaker-independent; **test speaker- AND channel-independent**;
train must not contain any channel that is in test.
- Build `speaker_table.csv` (speaker -> channel, clips, hours, words).
- Split **by speaker**, never by clip. Approach: small search (greedy/ILP) choosing val and
  test speaker sets that (a) put each *channel* wholly in one split, (b) approach 10% of word
  clips each, (c) leave test with enough sentence clips and enough in-vocab words for
  spotting. Start from the existing hand pick (`val levent_gultekin`, `test yilmaz_ozdil +
  ozlem_gurses`) and report its actual percentages; it is unlikely to be 10/10.
- Speaker imbalance: `cuneyt_ozdemir` is ~60%+ of rows. Apply a **per-speaker cap on train**
  (e.g. a max share of train clips, drawn per class at random with a fixed seed). Caps apply
  to train only; val/test keep all their clips. Make the cap a parameter; log the before/after
  vocab sizes since capping reduces per-class counts.
- Record everything in `splits.json` (speakers, seed, caps) and write the `split` column.
- Hard assertions: no speaker in two splits; no channel shared between train and test;
  report (not assert) train/val channel overlap.

### Step 4 — Vocabulary (both labelings, the plan's ablation)
- Count **training-split** examples per class under surface and under lemma (after the cap).
- Keep classes with `>= 300` train examples (`MIN_TRAIN`, parameter).
- Write `vocab_surface.csv` / `vocab_lemma.csv`: `class, train_n, val_n, test_n, n_speakers,
  n_channels`. Also a surface -> lemma mapping table for the release.
- Add a minimum-speakers-per-class requirement (suggest >= 5 train speakers) so a class is not
  one person's tic; report how many classes it removes.
- Targets: 500+ classes, 150k+ word clips. If surface misses 500, report it honestly; lemma is
  the expected primary labeling, surface is the ablation. Review both lists by eye for Whisper
  junk (numbers, fillers, foreign tokens) and add an explicit `stoplist.txt` if needed.
- Classes must also appear in val/test to be evaluable; report classes with `val_n == 0` or
  `test_n == 0`.
- Reuse `vocab_report.py` logic; either import its helpers or fold it into `common.py`, and
  keep its CLI working.

### Step 5 — Sentence pool for spotting evaluation
- Spotting evaluation uses **sentence clips only** (project plan, BSL-1K style); word clips
  are for training. So the held-out speakers' *sentence* items form the eval pool:
  `sentence_val.csv`, `sentence_test.csv`. Train speakers' sentences are not needed for the
  benchmark (optionally keep as `sentence_train.csv` for later weak-supervision work).
- For each vocab class, find test sentences that contain it, using `word_spans`
  (match `surface_form` or `lemma` per labeling). Output `spotting_gt/queries.csv`:
  `class, labeling, clip_id, start_sec, end_sec` (clip-relative, from `word_spans`).
- Also pick **negative** sentences per class (class absent) so precision is measurable;
  report positives/negatives per class and drop classes with too few test positives.
- This is the **auto seed** for the ground truth; it comes from Whisper + SyncNet offset,
  so it is not yet human-verified.

### Step 6 — Manual verification (Label Studio)
- Random 5% sample, stratified by split/speaker, with a fixed seed → `verification/sample.csv`
  + Label Studio import JSON. Items: word clips (is the label what is mouthed/said?) and a
  sentence subset (is the transcript right?). For sentence clips and word clips Label Studio
  needs something viewable: **render short preview videos from the crops** (88×88 gray, 25 fps)
  — no raw video, copyright rule — into `verification/media/`.
- Include the Phase 2 pending checks: hand-label off-screen speech on a sample, and the
  Phase 2 QC contact-sheet review (`scripts/phase2/qc_report.py`).
- Double-annotate an overlap subset; compute Cohen's κ (`sklearn.metrics.cohen_kappa_score`)
  into `verification/kappa.txt`. Apply the corrected labels back as a `verified` / `label_fix`
  column; never overwrite the original columns.
- Test-set spotting GT (Step 5) gets the stricter pass: annotators correct the boundaries of
  each target occurrence; store the corrected file as `spotting_gt/test_gt_verified.csv`.

### Step 7 — Statistics and release package
- `stats/`: class frequency (both labelings), clips/hours per speaker and channel, word
  length / syllable distribution, `face_height_px` / `mouth_width_px` distributions per split,
  `confidence` and sync distributions. These feed the Phase 5 error analysis.
- `release/` contains **only**: metadata CSVs (label, lemma, start/end sec, speaker_id, channel,
  split, plus npz reference), split files, vocab + surface->lemma mapping, frequency stats.
  **No raw video, full-face frames or audio.** Crops (88×88 gray) are the only pixels, and
  global mean/var normalisation is applied at release time, not in the stored crops.
- Consider whether speaker slugs are released as-is (they are public news presenters) — a
  decision for the project owner, flag it rather than assume.

---

## 4. Scripts to write (all in `scripts/phase3/`)

| Script | Does |
|--------|------|
| `common.py` | loaders, label normalisation, assertions, paths (single source of `DATA/phase2`, `DATA/phase3`) |
| `audit_manifests.py` | Step 1 report |
| `build_splits.py` | Steps 2–4: filter, speaker split, cap, vocab, writes word_*.csv + vocab + splits.json |
| `build_spotting_set.py` | Step 5: sentence pools + query/negative lists |
| `make_verification_set.py` | Step 6: sample, preview renders, Label Studio import; `score_kappa.py` for agreement |
| `make_release.py` | Step 7: assemble `release/` from the above |
| `vocab_report.py` | existing; keep working |

Conventions (from CLAUDE.md): match existing scripts, UTF-8 with
`sys.stdout.reconfigure(encoding="utf-8")`, argparse with sensible defaults rooted at the
repo (`ROOT = Path(__file__).resolve().parents[2]`), deps via `pyproject.toml`/uv only,
deterministic (fixed seeds, recorded in `splits.json`), idempotent re-runs, never write into
`data/phase2/`.

---

## 5. Known risks to handle

1. **Speaker/channel coupling and imbalance** — the 10/10 ratio may be unattainable with
   channel-disjoint test; document the achieved ratio rather than forcing it.
2. **Train/val/test counts after capping** may drop surface classes below 300; the lemma
   labeling is the safety net.
3. **Whisper label noise** — word labels are ASR, not human; the 5% audit + κ quantifies it.
   Prefer excluding `manual_review == 1` from eval splits.
4. **Phase 2 re-runs** — if Phase 2 is re-run, `word_clips.csv` changes and so do clip IDs'
   membership. Phase 3 outputs are only valid for the manifest they were built from: store the
   input file's row count and a hash in `splits.json`.
5. **Windows paths / power loss** — Phase 2 manifests were written on Windows and may have had
   NUL-tail repair; the loader should tolerate a trailing partial row (use
   `manifests.strip_nul_tail` logic or just `on_bad_lines="warn"` and report count).
6. **Disk** — Phase 3 writes only CSV/JSON plus small preview clips; no crop copies.

## 6. Definition of done

- `data/phase3/` contains the splits, both vocab lists, speaker table, `splits.json`.
- Asserts pass: speaker-disjoint splits, test channel-disjoint from train, all `npz_path`
  resolve, all classes meet the train threshold and speaker minimum.
- Report states achieved numbers vs targets: classes (500+), word clips (150k+), train
  examples per class (300), split ratios.
- 5% verification done with κ reported; spotting test GT corrected by hand.
- CLAUDE.md Status table updated for Phase 3 (and `data/phase3` added to the "What lands in
  `data/`" table).
