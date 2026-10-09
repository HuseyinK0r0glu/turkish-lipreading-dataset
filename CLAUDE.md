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
├── models/                     ← YuNet face detector (auto-downloaded if missing)
├── plans/PHASE2_PLAN.md
├── colab_run_all.ipynb          ← Colab GPU runner (self-contained reimplementation)
├── scripts/
│   ├── run_all.py               ← MAIN ENTRY POINT (dispatches rows to pipeline 1 / 2)
│   ├── run_status.py            ← read-only progress/ETA snapshot of a live run
│   ├── test_data_pipe.py        ← isolated smoke test over the first N CSV rows
│   ├── download_videos.py       ← pipeline 1 downloader
│   ├── run_whisper.py           ← Whisper ASR + NFC/strip normalization
│   ├── pipeline2.py             ← pipeline 2 orchestration + per-clip Whisper
│   ├── presenter_filter.py      ← pipeline 2 ENGINE (download, scene detect, face filter)
│   ├── merge_links.py           ← idempotent append into video_links.csv, keeps `completed`
│   ├── reorder_links.py         ← push a reporter's pending rows to the end of the queue
│   ├── reset_empty_rows.py      ← un-complete rows that yielded zero clips, so they retry
│   ├── defer_low_res.py         ← probe pending rows, move sub-720p ones to the end
│   ├── audit_clips.py           ← re-apply the CURRENT solo rules to produced clips
│   └── backfill_face_height.py  ← fill face_height_px on pre-existing manifest rows
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
`lemma == surface_form` for now (TODO: zeyrek). `manual_review = 1` when `confidence < 0.7`.

---

## Status

| Phase | Status |
|-------|--------|
| 1 — Download + Whisper ASR | Paused at the 180 h mark. 1511 of 7117 rows completed (all 8 pipeline-1, 1503 of 7109 pipeline-2); `reporter_clips.csv` holds 67,563 clips / 180.0 h from 24 reporters (2026-10-09). |
| 2 — Lip extraction + SyncNet | Designed in `plans/PHASE2_PLAN.md`, no code |
| 3 — Dataset splits + benchmark | Not started |
| 4–5 — Models + release | Not started |

Known gaps: `lemma == surface_form` (zeyrek pending); `channel` is not propagated into
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
