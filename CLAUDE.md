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
├── plans/PHASE2_PLAN.md
├── colab_run_all.ipynb          ← Colab GPU runner (self-contained reimplementation)
├── scripts/
│   ├── run_all.py               ← MAIN ENTRY POINT (dispatches rows to pipeline 1 / 2)
│   ├── run_status.py            ← read-only progress/ETA snapshot of a live run
│   ├── test_data_pipe.py        ← isolated smoke test over the first N CSV rows
│   ├── download_videos.py       ← pipeline 1 downloader
│   ├── run_whisper.py           ← Whisper ASR + NFC/strip normalization
│   ├── download_and_transcribe.py  ← older pipeline-1-only entry point
│   ├── pipeline2.py             ← pipeline 2 orchestration + per-clip Whisper
│   ├── presenter_filter.py      ← pipeline 2 ENGINE (download, scene detect, face filter)
│   ├── merge_links.py           ← idempotent append into video_links.csv, keeps `completed`
│   ├── reorder_links.py         ← push a reporter's pending rows to the end of the queue
│   ├── reset_empty_rows.py      ← un-complete rows that yielded zero clips, so they retry
│   ├── defer_low_res.py         ← probe pending rows, move sub-720p ones to the end
│   ├── audit_clip_pose.py       ← re-audit produced clips against MAX_FACE_YAW
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

### Pipeline 2 — solo-presenter clips (4691 rows)

Download the full broadcast → PySceneDetect scene cuts → keep only scenes where
**exactly one face** appears and it matches `pipeline2_reporter_pictures/<reporter>.jpeg`
→ export each kept scene as its own mp4 → Whisper per clip → append to
`data/reporter_clips.csv`. **The source video is deleted immediately after clip export**
so disk usage stays flat across a 4691-row run.

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
| 1 — Download + Whisper ASR | Both pipelines verified end-to-end by `test_data_pipe.py`. Full run not yet done (`completed` empty on all 4699 rows). |
| 2 — Lip extraction + SyncNet | Designed in `plans/PHASE2_PLAN.md`, no code |
| 3 — Dataset splits + benchmark | Not started |
| 4–5 — Models + release | Not started |

Known gaps: `lemma == surface_form` (zeyrek pending); `channel` is not propagated into
`reporter_clips.csv` (see PHASE2_PLAN §1.2); 4407 of 4691 pipeline-2 rows are a single
speaker (Cüneyt Özdemir), which strains speaker-independent splits.

---

## Non-obvious invariants — do not regress these

1. **`get_scene_list(start_in_scene=True)`** in `presenter_filter.detect_scenes`. Without
   it PySceneDetect returns an EMPTY list for a video with no cuts, so single-shot
   talking-head broadcasts — the best lip-reading material — silently yield zero clips.
2. **`solo_segments` cuts a scene, it does not vote on it.** Because of #1 a cut-free
   broadcast is ONE scene covering the whole file, so a whole-scene "does only the
   target appear?" test is all-or-nothing at broadcast granularity — one two-face
   insert graphic discarded 34.7 minutes of clean single-face monologue and a
   Deniz Zeyrek video yielded zero clips. `sample_grid` therefore divides the scene
   into ~30 s cells with one identity check at each centre, and `solo_segments`
   drops only the failing cells and merges the survivors. A cell with 0 faces or an
   undecodable frame counts as failing: no mouth on screen, no lip-reading signal,
   and a video OpenCV cannot seek must yield nothing rather than unverified clips.
   `MAX_SAMPLES` (480) is now only a guard against a multi-hour livestream — it is
   no longer a cost cap, because samples are the cut grid, not a vote.
   `colab_run_all.ipynb` carries the same two functions; keep them in sync.
3. **A cell must pass all four gates**, in `solo_segments`: exactly one
   *presenter-sized* face (`presenter_sized_faces`, `MIN_FACE_FRACTION` — HOG finds
   36 px "faces" in wall art and one of those used to disqualify a whole broadcast),
   that face matches the reporter, and `face_yaw <= MAX_FACE_YAW`. The yaw gate is
   the plan's "presenter faces the camera directly": without it a 90-degree profile
   of the presenter talking to an off-camera guest passed every other test, and an
   audit of 5521 existing clips found 857 (15.5%) over the line — 82% of
   `osman_gokcek`, 49% of `fatih_altayli`, 40% of `ismail_kucukkaya`, against 0.4%
   of `kubra_par`. The difference is the studio camera angle, not the presenter.
   `scripts/audit_clip_pose.py` re-runs that audit over produced clips.
4. **`load_reference_encoding` takes the LARGEST face, not `encodings[0]`.**
   Detection order is not size order, so on a photo with background art the first
   encoding can be the artwork — and then every scene of that reporter is matched
   against a painting, the rows complete with zero clips, and nothing looks broken.
5. **`DOWNLOAD_FORMAT` requires >=720p**, deliberately. Sub-720p sources are skipped
   (row left pending), not downscaled into the dataset.
6. **`run_pipeline2` catches per-row exceptions.** One unavailable video must never abort
   a 4691-row run.
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
`video_links.csv` averages 20.5 min, so the whole CSV is ~1600 h of audio:
**~5-6 weeks serial for everything, ~2.5 days excluding `cuneyt_ozdemir`.**

`cuneyt_ozdemir` occupies CSV rows 235-4641 and is 94% of the corpus, so an unfiltered
run reaches `cem_ogretir`/`can_okanar` only at the very end. Use the row filters:

```powershell
uv run python scripts/run_all.py --exclude-reporters cuneyt_ozdemir
uv run python scripts/run_all.py --reporters cem_ogretir can_okanar
uv run python scripts/run_all.py --pipeline 2 --limit 50
```

Filters apply before `--limit`, matching `colab_run_all.ipynb`'s MAX_ROWS semantics.
