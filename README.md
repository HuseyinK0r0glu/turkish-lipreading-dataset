# Turkish News Lip-Reading Spotting Benchmark

A dataset pipeline for building the first Turkish visual word-spotting lip-reading benchmark from YouTube news broadcasts. The pipeline collects Turkish news anchor footage, filters it to solo-anchor scenes, transcribes with Whisper ASR, and produces word- and sentence-level clips for lip-reading research.

| Target | Value |
|--------|-------|
| Raw video | 100+ hours |
| Word classes | 500+ |
| Word clips | 150,000+ |
| Min examples/class | 300 (train) |
| Lip crop resolution | 88×88 grayscale |
| ASR review flag | confidence < 0.7 |

---

## Repository Layout

```
turkish-lipreading-dataset/
├── video_links.csv                    ← THE input for both pipelines (url, channel, start, end, pipeline, reporter, completed)
├── pipeline2_reporter_pictures/       ← reference face photos, one <slug>.jpeg per reporter
├── pyproject.toml                     ← uv-managed dependencies
├── .python-version                    ← 3.12
├── uv.lock                            ← committed lockfile
├── colab_run_all.ipynb                ← Colab GPU runner (both pipelines)
└── scripts/
    ├── run_all.py                     ← unified local runner (pipeline 1 + 2)
    ├── pipeline2.py                   ← solo-presenter scene filter + clip export
    ├── presenter_filter.py            ← shared building blocks (download, scene detect, face filter)
    ├── download_videos.py             ← pipeline 1 downloader
    ├── download_and_transcribe.py     ← pipeline 1 entry point (download + Whisper)
    ├── run_whisper.py                 ← Whisper ASR wrapper
    └── test_data_pipe.py              ← smoke test over the first N CSV rows (isolated)
```

---

## Disk usage

Pipeline-2 source videos are **deleted as soon as their clips are exported**, in
both the local runner and the Colab notebook. Only the trimmed clips survive, so a
full 4691-row run does not accumulate raw broadcasts on disk.

| Runner | Flag | Default |
|--------|------|---------|
| `scripts/run_all.py` | `--keep-source` | delete |
| `scripts/test_data_pipe.py` | `--keep-source` | delete |
| `colab_run_all.ipynb` | `KEEP_P2_SOURCE_VIDEO` | `False` (delete) |

Pipeline-1 raw videos in `data/raw_videos/` are **kept** — they have no clip
artifact standing in for them, and pipeline 1 is only a handful of short segments.

---

## Two Pipelines

`video_links.csv` is the single input; its `pipeline` column decides which path each row
takes. `scripts/run_all.py` reads the CSV and dispatches — you never run the two
separately.

### Pipeline 1 — segment ASR (8 rows)

Downloads the time range named by `start`/`end` (or the whole video when both are empty)
and transcribes it with Whisper. Overlapping intervals on the same URL are detected and
skipped. **No face filtering and no speaker identity**, so its output cannot go into
speaker-independent splits until it passes the pipeline-2 filter.

### Pipeline 2 — solo-presenter clips (4691 rows)

Downloads the full broadcast → PySceneDetect scene cuts → keeps only scenes where
**exactly one face** appears *and* it matches that reporter's reference photo → exports
each kept scene as its own mp4 → transcribes each clip → appends a row to
`data/reporter_clips.csv`. **The source broadcast is deleted as soon as its clips are
exported.**

This is the pipeline that matters: its clips are single-speaker, face-verified, and
carry a `reporter` label that becomes `speaker_id` in Phases 2–3.

Sources below 720p are **rejected**, not downscaled — the mouth ROI would not survive the
88×88 lip crop. Such a row is logged and left pending rather than aborting the run
(relax `DOWNLOAD_FORMAT` in `scripts/presenter_filter.py` if you want them).

#### Reporters

Each reporter needs `pipeline2_reporter_pictures/<slug>.jpeg` as the reference face.
Row counts are from `video_links.csv` (`pipeline=2`):

| Reporter | Slug | Channel | Rows |
|----------|------|---------|-----:|
| Cüneyt Özdemir | `cuneyt_ozdemir` | Cüneyt Özdemir | 4407 |
| Fatih Portakal | `fatih_portakal` | Fatih Portakal TV | 76 |
| Kübra Par | `kubra_par` | TV100 | 60 |
| Gülşah Ekinci | `gulsah_ekinci` | Halk TV | 39 |
| Can Okanar | `can_okanar` | A Haber | 30 |
| Cem Öğretir | `cem_ogretir` | atv Haber | 28 |
| Serdar Cebe | `serdar_cebe` | Sözcü TV | 26 |
| Ece Üner | `ece_uner` | Halk TV | 16 |
| İsmail Küçükkaya | `ismail_kucukkaya` | Halk TV | 9 |

> One speaker is 94% of the corpus. Phase 3 needs speaker-independent val/test splits,
> so the smaller reporters are what make those splits possible — worth growing.

---

## Local Setup (uv)

Dependencies are managed by [uv](https://docs.astral.sh/uv/) via `pyproject.toml` +
`uv.lock`. Python 3.11 or 3.12 is required (`face_recognition`/`dlib` have no 3.13+
wheels); uv installs the right interpreter itself from `.python-version`.

```powershell
winget install --id astral-sh.uv -e     # once
winget install --id Gyan.FFmpeg -e      # once - yt-dlp needs ffmpeg + ffprobe
uv sync
```

That is the whole setup. Verify:

```powershell
uv run python -c "import cv2, whisper, scenedetect, face_recognition, torch; print('ok', torch.cuda.is_available())"
```

`uv run <cmd>` executes inside the project environment — no activation step, and it
re-syncs automatically if `pyproject.toml` changed.

### Why `pyproject.toml` looks the way it does

Three settings are load-bearing; removing them breaks the install:

| Setting | Reason |
|---------|--------|
| `override-dependencies = ["dlib; sys_platform == 'never'"]` | `face_recognition` requires `dlib`, whose PyPI sdist compiles from source and needs VS C++ Build Tools on Windows. `dlib-bin` supplies the identical module as a prebuilt wheel. |
| `setuptools<81` | `face_recognition_models` imports `pkg_resources`, which setuptools 81 removed. |
| `torch` from the `pytorch-cu121` index | PyPI torch is **CPU-only on Windows**, which makes Whisper large-v3 unusably slow. cu121 matches a CUDA 12.2 driver. |

**Match the CUDA build to your driver.** Run `nvidia-smi` and read the `CUDA Version`
field, then set the index URL in `pyproject.toml` accordingly (`cu121`, `cu124`,
`cu126`, `cu128`) and re-run `uv sync`. No NVIDIA GPU? Point the source at
`https://download.pytorch.org/whl/cpu` instead.

### First run

1. Place one clear frontal face photo per reporter in `pipeline2_reporter_pictures/<slug>.jpeg`  
   (e.g. `cem_ogretir.jpeg`, `can_okanar.jpeg`).

2. Smoke-test the plumbing on the first 10 CSV rows before committing to a long run.
   Everything lands in `data/test_run/` and the real `video_links.csv` is not touched:

```powershell
uv run python scripts/test_data_pipe.py --rows 10 --whisper-model base --fresh
```

   It ends with a `RESULT: OK` line and an explicit check that no pipeline-2 source
   videos were left behind.

3. Full run over the whole CSV:

```powershell
uv run python scripts/run_all.py
```

   `run_all.py` is **resumable**: each row is flagged `completed=1` in `video_links.csv`
   the moment it finishes, so you can stop it with Ctrl+C and re-run to continue. Rows
   whose download fails stay pending and are retried on the next run.

```powershell
uv run python scripts/run_all.py --exclude-reporters cuneyt_ozdemir  # the other 8 speakers
uv run python scripts/run_all.py --reporters cem_ogretir can_okanar  # named speakers only
uv run python scripts/run_all.py --pipeline 2                        # skip pipeline-1 rows
uv run python scripts/run_all.py --limit 50            # first 50 rows after filtering
uv run python scripts/run_all.py --whisper-model small # faster, lower quality
uv run python scripts/run_all.py --keep-source         # don't delete source videos
```

---

## How long a full run takes

Measured on this repo's hardware (RTX A4000 16 GB, CUDA 12.2, 12-thread CPU), with a
35-video sample of `video_links.csv` averaging **20.5 min** (median 12.8, max 120):

| Stage | Rate | Whole CSV (4691 videos ≈ 1600 h audio) | Without `cuneyt_ozdemir` (284 videos ≈ 97 h) |
|-------|------|---------------------------------------:|---------------------------------------------:|
| Whisper large-v3 (GPU) | 2.63× realtime | ~610 h | ~37 h |
| Scene detection | ~600 fps | ~65 h | ~4 h |
| Face identity checks | ≤20 × ~1 s / video | ~26 h | ~2 h |
| Clip export (libx264 re-encode) | ~5× realtime | ~235 h | ~14 h |
| **Total (serial)** | | **~5–6 weeks** | **~2.5 days** |

For reference, Whisper on **CPU** measures 2.45× realtime for `small` and ~0.4× for
`large-v3` — roughly 6× slower, which is why the CUDA index in `pyproject.toml` matters.

Two consequences worth planning around:

- **Reporter order.** `cuneyt_ozdemir` occupies CSV rows 235–4641, so an unfiltered run
  reaches `cem_ogretir` and `can_okanar` only after ~4400 videos of a single speaker.
  Since that speaker is already 94% of the corpus and Phase 3 needs speaker-independent
  splits, run `--exclude-reporters cuneyt_ozdemir` first.
- **Clip export is ~25% of the runtime** because `export_clip` re-encodes with libx264.
  Single-shot videos now produce one clip spanning the whole file, where `-c copy` would
  be near-instant and lossless — but stream copy snaps to keyframes, which can pull a
  frame from the previous shot into a scene-cut clip and break the single-face guarantee.
  Re-encoding is the safe default; switch per-case if runtime matters more.

The run is resumable, so it does not need to finish in one sitting.

---

## Google Colab (alternative)

`colab_run_all.ipynb` is a self-contained reimplementation of both pipelines for a Colab
GPU. It predates the local CUDA setup above and is still useful if you want to run on a
second machine, but a local NVIDIA GPU is now the simpler path: no Drive sync, no session
timeouts, no YouTube bot-check on a datacenter IP.

Set `MAX_ROWS = None` in the config cell before a real run — it ships at `11` for testing.

**One-time Drive setup** — create `MyDrive/turkish_lipreading/` with:

```
turkish_lipreading/
├── video_links.csv
├── cookies.txt                  ← optional but recommended (see Troubleshooting cell)
└── pipeline2_reporter_pictures/
    ├── cem_ogretir.jpeg
    ├── can_okanar.jpeg
    └── ...                      ← one file per reporter slug
```

Everything else (`raw_videos/`, `transcripts/`, `clips/`, `reporter_clips.csv`) is created automatically. The run is **resumable** — completed rows are flagged and skipped on re-run, so Colab timeouts are harmless.

See the notebook's **Troubleshooting** cell if YouTube downloads fail (the bot-check workaround using `cookies.txt`).

---

## What lands in `data/`

Everything under `data/` is **generated** and gitignored — nothing is placed there by
hand. The inputs live at the repo root (`video_links.csv`, `pipeline2_reporter_pictures/`).

| Path | Written by | Contents |
|------|-----------|----------|
| `raw_videos/` | pipeline 1 | `<channel>_<title>_<HHMMSS-HHMMSS>.mp4`. **Kept** — nothing else stands in for them. |
| `transcripts/<video>.json` | pipeline 1 | Word-level Whisper JSON per downloaded segment. |
| `pipeline2_raw/` | pipeline 2 | Transient source broadcasts. **Deleted right after clipping** → normally empty. |
| `<reporter>_solo_clips/` | pipeline 2 | `<reporter>_<srcvideo>_scene_<idx>_<start>-<end>.mp4` — the actual dataset material. |
| `reporter_transcripts/<clip>.json` | pipeline 2 | Per-clip Whisper JSON. Timestamps are **clip-local** (start at 0). |
| `reporter_clips.csv` | `run_all.py` | Master manifest, one row per kept clip. **The Phase 1 → Phase 2 hand-off.** |
| `_pipeline1_links.csv` | `run_all.py` | Scratch CSV of pending pipeline-1 rows handed to the downloader. |
| `test_run/` | `test_data_pipe.py` | Isolated copy of all of the above; the real CSV is never touched. |

`reporter_clips.csv` columns: `reporter, source_url, source_video, clip_file, start_sec,
end_sec, duration_sec, transcript_text, transcript_json`. Paths are posix-style so a
manifest written on Windows can be read on Colab/Linux.

Transcript JSON (both pipelines share the schema):

```jsonc
{
  "video": "<file>",
  "segments": [{
    "start_sec": 0.0, "end_sec": 2.72, "text": "Yargıtay emsal bir karara imza attı.",
    "words": [{
      "surface_form": "Yargıtay",   // NFC-normalized, whitespace stripped
      "lemma": "Yargıtay",          // == surface_form for now (TODO: zeyrek)
      "start_sec": 0.0, "end_sec": 0.7,
      "confidence": 0.695732,
      "manual_review": 1            // 1 when confidence < 0.7
    }]
  }]
}
```
