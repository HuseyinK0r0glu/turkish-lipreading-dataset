# Turkish News Lip Reading Spotting Benchmark

**Goal:** First Turkish visual word-spotting lip-reading benchmark from YouTube news. Word + sentence clips, Whisper ASR, then baseline classifiers + sliding-window spotting. Full plan: `TurkishNewsBrodcasts-LipReadingSpottingBenchmark-ProjectPlan.pdf` (read only if task requires it). Phase 2 design: `plans/PHASE2_PLAN.md`.

---

## Repo Layout

```
turkish-lipreading-dataset/
├── video_links.csv              ← SINGLE input for both pipelines
├── cookies.txt                  ← OPTIONAL, gitignored. Netscape-format YouTube
│                                  session; without it age-restricted videos fail
├── pyproject.toml               ← uv-managed dependencies
├── .python-version              ← 3.12
├── uv.lock                      ← committed lockfile
├── pipeline2_reporter_pictures/ ← reference face photo per reporter (<slug>.jpeg)
├── models/                     ← YuNet (committed); MediaPipe + SyncNet weights auto-download, gitignored
├── plans/PHASE2_PLAN.md, PHASE3_PLAN.md
├── colab_run_all.ipynb          ← Colab GPU runner (self-contained reimplementation)
├── scripts/
│   ├── run_all.py               ← MAIN ENTRY POINT (dispatches rows to pipeline 1 / 2)
│   ├── run_status.py            ← read-only progress/ETA snapshot of a live run
│   ├── test_data_pipe.py        ← isolated smoke test over the first N CSV rows
│   ├── download_videos.py       ← pipeline 1 downloader
│   ├── run_whisper.py           ← Whisper ASR + NFC/strip normalization
│   ├── pipeline2.py             ← pipeline 2 orchestration + per-clip Whisper
│   ├── presenter_filter.py      ← pipeline 2 ENGINE (download, scene detect, face filter)
│   ├── lemmatize.py             ← zeyrek lemma per word + transcript backfill CLI
│   ├── run_more_data.cmd        ← 2026-10-11 chain: five new presenters, then Phase 2 again
│   ├── merge_links.py           ← idempotent append into video_links.csv, keeps `completed`
│   ├── reorder_links.py         ← push a reporter's pending rows to the end of the queue
│   ├── reset_empty_rows.py      ← un-complete rows that yielded zero clips, so they retry
│   ├── defer_low_res.py         ← probe pending rows, move sub-720p ones to the end
│   ├── audit_clips.py           ← re-apply the CURRENT solo rules to produced clips
│   ├── backfill_face_height.py  ← fill face_height_px on pre-existing manifest rows
│   └── phase2/                  ← Phase 2 (own folder; imports Phase 1 code, never edits it)
│       ├── run_phase2.py        ← ENTRY POINT: reporter_clips.csv snapshot → crops + manifests
│       ├── source_clip.py       ← one solo clip end to end (the worker task)
│       ├── video_io.py / landmarks.py / mouth_roi.py / syncnet.py / segments.py
│       ├── manifests.py         ← input join (channel), output CSV schemas, resume
│       ├── syncnet_model.py     ← vendored SyncNet network (MIT)
│       ├── phase2_status.py     ← read-only progress/ETA snapshot of a live run (`--follow`)
│       ├── qc_report.py         ← contact sheets + distribution summary
│       └── calibrate_offscreen.py ← spliced-audio evidence for WORD_SYNC_MIN
│   └── phase3/                  ← Phase 3 (manifest work only; never writes data/phase2/)
│       ├── common.py            ← paths, tolerant manifest loaders, label classes
│       ├── audit_manifests.py   ← step 1: checks on the Phase 2 manifests
│       ├── build_splits.py      ← steps 2-4: filter, speaker split, cap, vocabulary
│       ├── build_spotting_set.py ← step 5: sentence pools + spotting queries/negatives
│       ├── make_verification_set.py / score_kappa.py ← step 6: Label Studio sample, kappa
│       ├── dataset_stats.py / make_release.py ← step 7: stats tables, public metadata
│       ├── stoplist.txt         ← classes kept out of the vocabulary
│       └── vocab_report.py      ← classes with >= 300 TRAIN examples under a speaker split
└── data/                        ← ALL generated; gitignored. Nothing is placed here by hand.
```

`video_links.csv` is the **only** source of URLs. The per-reporter `reporter_links/*.txt`
files and the `build_video_links.py` / `run_reporters.py` / `quick_test_reporters.py`
scripts that consumed them were removed once the CSV became authoritative — add new
videos by appending rows to the CSV.

---

## Environment (uv)

**Python 3.11 or 3.12 required** — `face_recognition`/dlib have no 3.13+ wheels.

```powershell
uv sync                              # create .venv from pyproject.toml + uv.lock
uv run python scripts/run_all.py     # run anything inside the env
```

Three things in `pyproject.toml` exist for a reason — do not "simplify" them away:

1. `override-dependencies = ["dlib; sys_platform == 'never'"]` — `face_recognition` requires `dlib`, whose PyPI sdist compiles from source and needs VS C++ Build Tools on Windows. `dlib-bin` provides the identical module as a prebuilt wheel.
2. `setuptools<81` — `face_recognition_models` imports `pkg_resources`, removed in setuptools 81.
3. `torch` pinned to the `pytorch-cu121` index — PyPI torch is **CPU-only on Windows**. cu121 matches this machine's CUDA 12.2 driver (`nvidia-smi`); cu126/cu128 wheels assume newer drivers.

FFmpeg must be on PATH (`winget install Gyan.FFmpeg`). Both downloaders fall back to the `imageio-ffmpeg` bundled binary via `--ffmpeg-location`, but yt-dlp also wants `ffprobe`, so a real install is preferred.

---

## The two pipelines

`video_links.csv` columns: `url,channel,start,end,pipeline,reporter,completed`.
The `pipeline` column decides which path a row takes.

### Pipeline 1 — segment ASR (8 rows)

Download the named time range (or the whole video when `start`/`end` are empty) →
Whisper large-v3 → word-level JSON. **No face filtering, no speaker identity.**
Overlapping intervals on the same URL are detected and skipped.

### Pipeline 2 — solo-presenter clips (7109 rows)

Download the full broadcast → PySceneDetect scene cuts → sample every 0.2 s and keep
only the stretches where **exactly one person** is on screen, it matches
`pipeline2_reporter_pictures/<reporter>.jpeg` and faces the camera (invariant #3)
→ export each kept stretch (>= 2 s) as its own mp4 → Whisper per clip → append to
`data/reporter_clips.csv`. **The source video is deleted immediately after clip export**
so disk usage stays flat across a 7000-row run.

Pipeline 2 is the one that matters: its clips are already single-speaker, face-verified
and carry a `reporter` label (= `speaker_id`), which is what Phases 2/3 need. Pipeline 1
output cannot enter speaker-independent splits until it passes the same filter.

Both are resumable: a row is flagged `completed=1` in `video_links.csv` the moment it
finishes and the CSV is rewritten immediately. Failed rows stay pending and are retried.

**Age-restricted videos** ("Sign in to confirm your age") need a logged-in session.
`presenter_filter.cookie_args` looks for `cookies.txt` at the repo root first, then
`$YTDLP_COOKIES_FROM_BROWSER` (e.g. `edge`). Prefer the file: on this machine Chrome's
App-Bound Encryption blocks extraction (yt-dlp #10927) and a running Edge holds a lock
on its cookie DB (#7271). A cookie source that turns out to be broken is dropped after
the first failure and the download is retried without it — otherwise one bad cookie
source would fail *every* row, not just the age-restricted ones.

---

## What lands in `data/`

Everything here is generated. Inputs live at the repo root.

| Path | Written by | Contents |
|------|-----------|----------|
| `raw_videos/` | pipeline 1 | `<channel>_<title>_<HHMMSS-HHMMSS>.mp4`. **Kept** — no clip replaces them. |
| `transcripts/<video>.json` | pipeline 1 | Word-level Whisper JSON for each raw video. |
| `pipeline2_raw/` | pipeline 2 | Transient source broadcasts. **Deleted after clipping** → normally empty. |
| `<reporter>_solo_clips/` | pipeline 2 | `<reporter>_<srcvideo>_scene_<idx>_<start>-<end>.mp4` — the dataset material. |
| `reporter_transcripts/<clip>.json` | pipeline 2 | Per-clip Whisper JSON. Timestamps are **clip-local** (start at 0). |
| `reporter_clips.csv` | `run_all.append_clip_rows` | Master manifest, one row per kept clip. **The Phase 1 → Phase 2 hand-off.** |
| `_pipeline1_links.csv` | `run_all` | Temp CSV of pending pipeline-1 rows handed to the downloader. |
| `test_run/` | `test_data_pipe.py` | Isolated copy of all of the above; the real CSV is never touched. |
| `phase3/` | `scripts/phase3/*` | `word_{train,val,test}.csv` (Phase 2 word rows + `label_surface`, `label_lemma`, `split`, `in_vocab_*`), `sentence_{val,test}.csv`, `vocab_{surface,lemma}.csv`, `surface_to_lemma.csv`, `speaker_table.csv`, `dropped.csv`, `splits.json` (speakers, parameters, input hash), `spotting_gt/`, `verification/` (previews WITH audio: internal), `stats/`, `release/` (metadata only: the public part). |

`reporter_clips.csv` columns: `reporter, source_url, source_video, clip_file, start_sec,
end_sec, duration_sec, face_height_px, transcript_text, transcript_json`. Paths are stored
posix-style so a manifest written on Windows is readable on Colab/Linux.

`face_height_px` is the median presenter face height over the cells a clip spans —
**recorded, never gated on**. The 88×88 lip crop wants a face around 260 px, but a hard
floor there would delete whole reporters: measured across the corpus, 18% of clips sit
under 150 px and 43% under 220 px, and the small ones are the TV studios (`serdar_cebe`
126 px, `kubra_par` 137 px — 1079 clips at 0.4% pose rejection, the cleanest block there
is) while the large ones are personal channels (`yilmaz_ozdil` 535 px, `fatih_altayli`
534 px). So the number travels into the manifest and Phase 2/3 choose their own
threshold. Empty means unmeasurable, not zero. `scripts/backfill_face_height.py` fills
it for rows written before the column existed.

Transcript JSON schema (read-only contract for Phase 2):
`segments[] → {start_sec, end_sec, text, words[]}`, each word
`{surface_form, lemma, start_sec, end_sec, confidence, manual_review}`.
`lemma` = `lemmatize.lemmatize_word(surface_form)` (zeyrek, lower-cased; backfilled into all
transcripts 2026-10-11). `manual_review = 1` when `confidence < 0.7`.

---

## Status

| Phase | Status |
|-------|--------|
| 1 — Download + Whisper ASR | Resumed 2026-10-11 (`scripts/run_more_data.cmd`): five NEW presenters for speaker/channel diversity, capped 2026-10-17 to shorten the run — ekrem_acikel (Sözcü TV, 11 rows), sinem_fistikoglu (Halk TV, 19), seda_selek (KOZA TV, first 5 of 27), irfan_degirmenci and sule_aydin (own channels, first 20 each); all done 2026-10-18 (+5,302 clips, 72,865 total), then Phase 2 again over everything into `data/phase2/`. irfan_degirmenci yields little (258 clips / 0.4 h from 20 rows, 76% sync). selcuk_tepeli (NOW, 50 rows in the CSV) was dropped after 11 episodes gave 1 clip: NOW keeps a sign-language interpreter box in every frame (~5% of frame height, above MIN_FACE_FRACTION), so invariant #3 never sees one person; exclude him from runs. Their reference photos were picked automatically (dominant face across episodes) and still need a human check. 971 older pending rows remain for later. Was paused at the 180 h mark: 1511 of 7117 rows completed (all 8 pipeline-1, 1503 of 7109 pipeline-2); `reporter_clips.csv` holds 67,563 clips / 180.0 h from 24 reporters (2026-10-09). |
| 2 — Lip extraction + SyncNet | Run 3 done 2026-10-19 over all 72,865 clips with the fixed segments.py (0 failed): 66,912 ok / 5,720 low_sync / 233 no_words; 1,142,706 word clips and 174,653 sentence clips; crops in `data/lip_crops/` rewritten. The v1 manifests (2026-10-10) were superseded and deleted. 2026-10-19 cleanup: channel names unified (Halktv → Halk TV, SÖZCÜ Televizyonu → Sözcü TV, CNN_TURK_5N1K → CNN Türk) in `video_links.csv` and the Phase 2 manifests, and irfan_degirmenci's split-screen video with test speaker ozlem_gurses (Lp9NGIgjUFE, 20 clips) removed everywhere, its row dropped from the CSV so `reset_empty_rows.py` cannot bring it back: 72,845 clips. Progress: `scripts/phase2/phase2_status.py`. Off-screen hand check (2026-11): 100 boundary words (`word_sync_conf` just over 2.5) labelled, 97 on-screen; a second 200-word set (boundary / far below / far above, `data/phase2/qc/offscreen_check_2.csv`) awaits labelling and decides whether `WORD_SYNC_MIN` stays 2.5. QC montages reviewed by hand 2026-11-12. Alignment audit repeated on run 3 (`data/audit50_run3/README.md`, 50 words, automatic rules all pass). Only the labelling of the 200-word off-screen set is still open before Phase 2 is "done"; see PHASE2_PLAN §10. |
| 3 — Dataset splits + benchmark | Code done and run 2026-11-12 on the run 3 manifests (`scripts/phase3/`, plan `plans/PHASE3_PLAN.md`); manual verification NOT done. Split by speaker: val levent_gultekin, test yilmaz_ozdil + ozlem_gurses, train 26 speakers; 1,118,242 word clips kept of 1,142,556 = 82.1 / 8.4 / 9.5 % (dropped: 8,217 track gaps, 6,617 word spans < 0.04 s, 61 without label, 9,419 `manual_review` rows in val/test; train keeps `manual_review`). The 15% per-speaker train cap removes nothing (largest: cuneyt_ozdemir 12.3%). Vocabulary (>= 300 train examples, >= 5 train speakers, no digit tokens, `stoplist.txt`): **354 surface / 496 lemma classes** — lemma is 4 short of 500 because the seven digit classes (1-6, 10) and "eee" are excluded (`--keep-digits` gives 503); 466,503 / 757,173 in-vocab word clips. Spotting pool: 14,164 val / 19,367 test sentence clips, 352 surface / 494 lemma classes with >= 5 positive test sentences; `spotting_gt/queries.csv` is the Whisper auto seed. Open: the Label Studio pass (`make_verification_set.py` → annotate → `score_kappa.py`: 5% sample, kappa, hand-corrected `spotting_gt/test_gt_verified.csv`; only a 12-item trial was rendered), then `make_release.py` again; owner decision on releasing speaker slugs as-is. Val is one speaker and test two, all personal channels with large faces (no TV studio in eval); `stats/split_search.csv` ranks alternatives. |
| 4–5 — Models + release | Not started |

Vocabulary (`vocab_report.py`, val levent_gultekin, test yilmaz_ozdil + ozlem_gurses): classes with
>= 300 train examples went from 342 surface / 482 lemma (v1, 2026-10-11) to 367 surface / 507 lemma
(run 3, 2026-10-19) — lemma labeling meets the 500 target, surface does not. Those are raw counts;
the Phase 3 vocabulary (`build_splits.py`, after the quality filter, digit rule and stoplist) is
354 / 496. Audit of 50 random v1 items: `data/audit50/README.md`.

Known gaps: `channel` is not propagated into
`reporter_clips.csv` (see PHASE2_PLAN §1.2); 4622 of 7109 pipeline-2 rows are a single
speaker (Cüneyt Özdemir), which strains speaker-independent splits.

---

## Non-obvious invariants — do not regress these

1. **`get_scene_list(start_in_scene=True)`** in `presenter_filter.detect_scenes`. Without
   it PySceneDetect returns an EMPTY list for a video with no cuts, so single-shot
   talking-head broadcasts — the best lip-reading material — silently yield zero clips.
2. **`solo_segments` cuts a scene, it does not vote on it.** Because of #1 a cut-free
   broadcast is ONE scene covering the whole file, so a whole-scene "does only the
   target appear?" test is all-or-nothing at broadcast granularity — one two-face
   insert graphic discarded 34.7 minutes of clean single-face monologue. Each sample
   certifies only itself; failing samples cut the scene and survivors are merged.
3. **Every 0.2 s sample must pass all gates** (`analyze_frames` + `solo_segments`):
   - **exactly one person on set**, counted with **YuNet**, not HOG
     (`FACE_SCORE_MIN` 0.80, `MIN_FACE_FRACTION` 0.045). The old filter checked ONE
     HOG frame per 30 s on a half-scale copy; HOG there misses faces under ~120 px,
     so split screens, a presenter beside a video wall of three politicians, and
     sign-language interpreter boxes passed as "solo". 0 faces fails too.
   - **identity voted per track** (same face, no jump > `TRACK_MAX_SHIFT` widths):
     median dlib distance <= `IDENTITY_TOLERANCE` (0.6, dlib's own). Per-sample distances of the right person
     wander 0.45-0.83, so a per-sample test shreds clips; a person cut in elsewhere
     in the frame starts a new track and is voted on alone.
   - **`face_yaw <= MAX_FACE_YAW` on the sample itself** — the plan's "presenter faces
     the camera directly". 0.3-0.5 is a clear look away, 0.5+ often a hand over the
     mouth, 0.9 a profile. Natural head turns do fragment monologues; that is the
     rule working, not a bug.
   - `SCENE_EDGE_TRIM` (0.3 s) comes off every scene end: cuts land on transitions
     (hadi_ozisik glitch wipes, dissolves), and one clip ended in 0.4 s of another man.
   - clips shorter than `MIN_CLIP_DURATION` (2.0 s) are not exported: Phase 2 word
     clips are timestamp +-0.6 s, so shorter clips carry no usable word.
   dlib runs on a box derived from YuNet's (`_dlib_box`, agrees with HOG to ~0.01).
   Identity encoding is ~200 ms, so it runs every `IDENTITY_EVERY` samples and at
   each track start; `analyze_video` splits a broadcast into 5-min chunks across CPU
   cores. `colab_run_all.ipynb` carries the same functions; keep them in sync.
   `scripts/audit_clips.py` re-applies these rules to already-produced clips (trims
   to passing parts, slices the transcript without re-running Whisper, drops the rest).
4. **`load_reference_encoding` takes the LARGEST face, not `encodings[0]`.**
   Detection order is not size order, so on a photo with background art the first
   encoding can be the artwork — and then every scene of that reporter is matched
   against a painting, the rows complete with zero clips, and nothing looks broken.
5. **`DOWNLOAD_FORMAT` requires >=720p**, deliberately. Sub-720p sources are skipped
   (row left pending), not downscaled into the dataset.
   It is also capped at 1080p when the video has it: YouTube serves ~300-400 KB/s per
   video here, and 4K is ~4x the bytes for pixels the 1080p analysis throws away.
6. **`run_pipeline2` catches per-row exceptions.** One unavailable video must never abort
   a 7000-row run.
7. **Only successfully downloaded pipeline-1 rows are marked completed** —
   `download_videos()` returns the set of keys that need no further work.
8. **`normalize_text` strips.** Whisper word tokens carry a leading space; without the
   strip, " Yargı" and "Yargı" become two word classes in Phase 3.

### Phase 2 (`scripts/phase2/`)

9. **Phase 2 never writes `reporter_clips.csv`.** It reads a snapshot; Phase 1 may be
   appending. Resume state is Phase 2's own `face_track_report.csv`, written LAST per
   source clip; `manifests.drop_unfinished` clears rows of a clip that crashed mid-write.
10. **Heads are counted with Phase 1's YuNet rule, not MediaPipe.** MediaPipe's built-in
   detector is short-range: on a full studio frame (face ~130 px of 1080) it finds
   nothing, so `kubra_par` came out 1 good frame in 713. MediaPipe only runs on a crop
   around the YuNet face. YuNet runs every 5 frames (0.2 s, Phase 1's cadence, and >half
   the runtime when run on every frame); a multi-face sample marks the frames back to
   the previous sample (`RoiStream._spread_head_counts`).
11. **Crops are stored once per SOURCE clip** (`lip_crops/<reporter>/<stem>.npz`) and
   items are `frame_start`/`frame_end` ranges into it. Per-word files would store each
   frame ~4x (±15-frame padding): ~1.2 M words ≈ 340 GB raw, more than the disk.
12. **Word spans are shifted by `syncnet_offset`** before mapping to frames: Whisper
   times are audio time; the mouth lags the audio by that many frames.
13. **`syncnet.py` must keep reproducing the reference demo** (`--selftest example.avi`
   → offset 3, conf ~10.0). It is a streaming rewrite of `SyncNetInstance.evaluate`;
   the clip score, the window→frame mapping and the medfilt(9) are load-bearing.
14. **`segments.segment_words` joins apostrophe suffixes** ("Erdoğan" + "'ın" →
   "Erdoğan'ın"): Whisper splits them, and 2.4% of word items were bare suffixes.
   Word and sentence items both read it, so word indices stay aligned with `sync_ok`.
15. **Sentence windows cover their words**, not just Whisper's segment bounds: the
   segment start is often after the first word's start, and 9.2% of v1 sentence
   clips opened mid-word. Words longer than `WORD_MAX_SEC` (2 s) and non-Latin labels
   are Whisper failures and are discarded.
16. **`lemmatize._patch_shared_attrs` must stay.** zeyrek 0.1.3 shares a lexicon set
   with every search path and mutates it: after "vermek", "veriyor" has no analysis.
   Without the patch a word's lemma depends on which files were processed before it.

---

## Conventions for AI Assistants

1. **Minimal scope** — only the requested phase/step.
2. **Match existing scripts** — paths, JSON schema, Turkish NFC normalization.
3. **Never commit** `data/`, `*.mp4`, raw footage.
4. **Copyright** — publish mouth ROI crops only, not raw video.
5. **Deps** — `pyproject.toml` via uv, not bare pip.

## Key Targets

| Item | Target |
|------|--------|
| Raw video | 100+ h |
| Word classes | 500+ |
| Word clips | 150,000+ |
| Min examples/class | 300 (train) |
| Lip crop | 88×88 grayscale |
| Source resolution | ≥720p |
| ASR review flag | confidence < 0.7 |
| SyncNet discard | score < 3 |

---

## Runtime reality (measured, RTX A4000 + CUDA 12.2)

Whisper large-v3 runs at **2.63x realtime on GPU**, ~0.4x on CPU. A 35-video sample of
`video_links.csv` averages 20.5 min a video. At that rate the 5606 pending pipeline-2 rows
(2026-10-09) are ~1915 h of broadcast: **~4-5 weeks serial for everything, ~1 week
excluding `cuneyt_ozdemir`** (1202 rows). Upper bounds: pipeline 2 only transcribes the
kept clips, not the whole broadcast.

`cuneyt_ozdemir` is 4622 of 7117 rows (65%); 4404 of his are still pending and sit
mostly in the back half of the CSV (rows 2466-7118, after `reorder_links.py`). Use the
row filters to control what runs:

```powershell
uv run python scripts/run_all.py --exclude-reporters cuneyt_ozdemir
uv run python scripts/run_all.py --reporters cem_ogretir can_okanar
uv run python scripts/run_all.py --pipeline 2 --limit 50
uv run python scripts/run_all.py --pipeline 2 --reporters cuneyt_ozdemir --max-hours 15
```

Filters apply before `--limit`, matching `colab_run_all.ipynb`'s MAX_ROWS semantics.
`--max-hours` stops pipeline 2 once the selected reporters' clips in
`reporter_clips.csv` reach that many hours; clips already there count.

Newscasts are much slower than that average (2026-10-13): a 1h50 Halk TV / KOZA episode is
1080p60, ~2.2 GB, ~2.5 h to download at the ~250 KB/s YouTube serves here, ~1.2 h of
scene detection (every frame) and ~2-3 h of face analysis -- ~6 h a row. `run_pipeline2`
therefore downloads the next row while the current one is analysed (one download ahead).
