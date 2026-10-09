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
├── cookies.txt                        ← optional, gitignored; YouTube session for age-restricted videos
├── pipeline2_reporter_pictures/       ← reference face photos, one <slug>.jpeg per reporter
├── models/                            ← YuNet, MediaPipe and SyncNet weights (auto-downloaded if missing)
├── plans/PHASE2_PLAN.md               ← Phase 2 design (lip extraction + SyncNet)
├── pyproject.toml                     ← uv-managed dependencies
├── .python-version                    ← 3.12
├── uv.lock                            ← committed lockfile
├── colab_run_all.ipynb                ← Colab GPU runner (both pipelines)
└── scripts/
    ├── run_all.py                     ← unified local runner (pipeline 1 + 2)
    ├── run_status.py                  ← read-only progress/ETA snapshot of a live run
    ├── pipeline2.py                   ← pipeline 2 orchestration + per-clip Whisper
    ├── presenter_filter.py            ← shared building blocks (download, scene detect, face filter)
    ├── download_videos.py             ← pipeline 1 downloader
    ├── run_whisper.py                 ← Whisper ASR wrapper
    ├── test_data_pipe.py              ← smoke test over the first N CSV rows (isolated)
    ├── merge_links.py                 ← idempotent append into video_links.csv, keeps `completed`
    ├── reorder_links.py               ← push a reporter's pending rows to the end of the queue
    ├── reset_empty_rows.py            ← un-complete rows that yielded zero clips, so they retry
    ├── defer_low_res.py               ← probe pending rows, move sub-720p ones to the end
    ├── audit_clips.py                 ← re-apply the current solo rules to produced clips
    ├── backfill_face_height.py        ← fill face_height_px on older manifest rows
    └── phase2/                        ← Phase 2: mouth crops + SyncNet (see "Phase 2" below)
        ├── run_phase2.py              ← entry point: reporter_clips.csv → crops + manifests
        ├── source_clip.py             ← one solo clip end to end (the worker task)
        ├── video_io.py                ← 25 fps ffmpeg decode, 16 kHz audio
        ├── landmarks.py               ← YuNet head count + MediaPipe Face Landmarker
        ├── mouth_roi.py               ← tracking gaps, smoothed 88×88 mouth crops
        ├── syncnet.py                 ← streaming SyncNet scoring (clip + per-frame)
        ├── syncnet_model.py           ← SyncNet network, vendored (MIT)
        ├── segments.py                ← word / sentence slicing rules
        ├── manifests.py               ← input snapshot + channel join, output CSVs, resume
        ├── phase2_status.py           ← read-only progress/ETA snapshot of a live run
        └── qc_report.py               ← contact sheets + distribution summary
```

---

## Disk usage

Pipeline-2 source videos are **deleted as soon as their clips are exported**, in
both the local runner and the Colab notebook. Only the trimmed clips survive, so a
full 7109-row run does not accumulate raw broadcasts on disk.

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

### Pipeline 2 — solo-presenter clips (7109 rows)

Downloads the full broadcast → PySceneDetect scene cuts → samples every 0.2 s and keeps
only the stretches where **exactly one person** is on screen (YuNet), it matches that
reporter's reference photo (dlib, voted per face track) and faces the camera → exports
each kept stretch of at least 2 s as its own mp4 → transcribes each clip → appends a row
to `data/reporter_clips.csv`. A failing sample cuts a scene rather than discarding it, so
one insert graphic does not cost a whole single-shot monologue. **The source broadcast
is deleted as soon as its clips are exported.**

This is the pipeline that matters: its clips are single-speaker, face-verified, and
carry a `reporter` label that becomes `speaker_id` in Phases 2–3.

Sources below 720p are **rejected**, not downscaled — the mouth ROI would not survive the
88×88 lip crop. Such a row is logged and left pending rather than aborting the run
(relax `DOWNLOAD_FORMAT` in `scripts/presenter_filter.py` if you want them).

#### Reporters

Each reporter needs `pipeline2_reporter_pictures/<slug>.jpeg` as the reference face.
Row counts are from `video_links.csv` (`pipeline=2`); clips and hours from
`data/reporter_clips.csv` (2026-10-09):

| Reporter | Slug | Channel | Rows | Done | Clips | Hours |
|----------|------|---------|-----:|-----:|------:|------:|
| Cüneyt Özdemir | `cuneyt_ozdemir` | Cüneyt Özdemir | 4622 | 218 | 4376 | 15.03 |
| Turgay Güler | `turgay_guler` | Turgay Güler | 340 | 166 | 7217 | 10.91 |
| Hadi Özışık | `hadi_ozisik` | Hadi Özışık | 320 | 160 | 3318 | 14.05 |
| Ardan Zentürk | `ardan_zenturk` | Ardan Zentürk | 300 | 67 | 4298 | 6.09 |
| Fatih Portakal | `fatih_portakal` | Fatih Portakal TV | 246 | 208 | 6300 | 11.50 |
| Osman Gökçek | `osman_gokcek` | Osman Gökçek | 240 | 9 | 6 | 0.00 |
| Enver Aysever | `enver_aysever` | Enver Aysever | 180 | 96 | 1722 | 4.84 |
| Ruşen Çakır | `rusen_cakir` | Medyascope | 110 | 77 | 2367 | 10.38 |
| Sabahattin Önkibar | `sabahattin_onkibar` | Sabahattin Önkibar | 110 | 78 | 8539 | 12.19 |
| Nagehan Alçı | `nagehan_alci` | Nagehan Alçı | 74 | 2 | 346 | 0.83 |
| Ece Üner | `ece_uner` | Halk TV | 61 | 35 | 1015 | 2.95 |
| Kübra Par | `kubra_par` | TV100 | 60 | 59 | 1727 | 8.17 |
| Can Okanar | `can_okanar` | A Haber | 55 | 37 | 904 | 3.55 |
| Deniz Zeyrek | `deniz_zeyrek` | Deniz Zeyrek | 55 | 37 | 3553 | 11.69 |
| İsmail Küçükkaya | `ismail_kucukkaya` | Halk TV | 49 | 21 | 1959 | 5.23 |
| Levent Gültekin | `levent_gultekin` | Levent Gültekin | 40 | 31 | 5027 | 18.38 |
| Yılmaz Özdil | `yilmaz_ozdil` | Yılmaz Özdil | 40 | 34 | 3739 | 17.12 |
| Gülşah Ekinci | `gulsah_ekinci` | Halk TV | 39 | 39 | 2633 | 5.31 |
| Fatih Altaylı | `fatih_altayli` | Fatih Altaylı | 35 | 24 | 6825 | 14.54 |
| Özlem Gürses | `ozlem_gurses` | Özlem Gürses | 35 | 35 | 468 | 2.47 |
| Cem Öğretir | `cem_ogretir` | atv Haber | 28 | 24 | 65 | 0.14 |
| Serdar Cebe | `serdar_cebe` | Sözcü TV | 26 | 26 | 606 | 2.21 |
| Ersan Şen | `ersan_sen` | Ersan Şen | 25 | 3 | 294 | 0.64 |
| Sinan Burhan | `sinan_burhan` | Sinan Burhan | 19 | 17 | 259 | 1.82 |
| **Total** | | | **7109** | **1503** | **67,563** | **180.0** |

> `cuneyt_ozdemir` is 65% of the rows but only 8% of the clip hours so far: most of his
> rows are still pending. Phase 3 needs speaker-independent val/test splits, so keeping
> the other 23 reporters growing matters more than finishing his queue.

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
uv run python scripts/run_all.py --exclude-reporters cuneyt_ozdemir  # the other 23 speakers
uv run python scripts/run_all.py --reporters cem_ogretir can_okanar  # named speakers only
uv run python scripts/run_all.py --pipeline 2                        # skip pipeline-1 rows
uv run python scripts/run_all.py --limit 50            # first 50 rows after filtering
uv run python scripts/run_all.py --pipeline 2 --reporters cuneyt_ozdemir --max-hours 15
uv run python scripts/run_all.py --whisper-model small # faster, lower quality
uv run python scripts/run_all.py --keep-source         # don't delete source videos
```

   Filters apply before `--limit`. `--max-hours` stops pipeline 2 once the selected
   reporters' clips in `reporter_clips.csv` reach that many hours (existing clips count).
   `uv run python scripts/run_status.py` prints progress and an ETA for a run in progress.
   It prints `DURMUS OLABILIR` ("may have stalled") once the log has been quiet for a while, but a long
   download writes nothing to the log, so check `data/pipeline2_raw/` before killing the run.

   Age-restricted videos ("Sign in to confirm your age") need a logged-in session: put a
   Netscape-format `cookies.txt` at the repo root, or set `YTDLP_COOKIES_FROM_BROWSER`
   (e.g. `edge`). Prefer the file; Chrome's App-Bound Encryption blocks cookie extraction.

---

## How long a full run takes

Measured on this repo's hardware (RTX A4000 16 GB, CUDA 12.2, 12-thread CPU): Whisper
large-v3 runs at **2.63× realtime on GPU**, and a 35-video sample of `video_links.csv`
averages **20.5 min** a video (median 12.8, max 120).

At that rate the 5606 pending pipeline-2 rows (2026-10-09) are ~1915 h of broadcast:
**~4–5 weeks serial for everything, ~1 week excluding `cuneyt_ozdemir`** (1202 rows).
Those are upper bounds, since pipeline 2 only transcribes the kept clips, not the whole
broadcast. Downloads are a real share of it: YouTube serves ~300–400 KB/s per video
here, so a 2.4 GB two-hour 1080p broadcast takes about two and a half hours to arrive.

Whisper on **CPU** measures 2.45× realtime for `small` and ~0.4× for `large-v3`,
roughly 6× slower, which is why the CUDA index in `pyproject.toml` matters.

Two consequences worth planning around:

- **Reporter order.** 4404 of `cuneyt_ozdemir`'s 4622 rows are pending and sit at the
  back of the CSV (rows 2466–7118, after `scripts/reorder_links.py`), so an unfiltered run
  works through the other reporters first. `--exclude-reporters cuneyt_ozdemir` makes
  that explicit.
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
end_sec, duration_sec, face_height_px, transcript_text, transcript_json`. Paths are
posix-style so a manifest written on Windows can be read on Colab/Linux.

`face_height_px` is the median presenter face height over the clip. It is recorded,
never used as a filter: TV studio reporters sit at 126–137 px while personal channels
reach ~535 px, so a hard floor would remove whole speakers. Phases 2 and 3 pick their own
threshold. An empty value means unmeasurable, not zero.

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

---

## Phase 2 — mouth crops + SyncNet

`scripts/phase2/run_phase2.py` turns the solo clips in `data/reporter_clips.csv` into
88×88 grayscale mouth crops at a fixed 25 fps, scores audio-visual sync with SyncNet, and
cuts word and sentence items out of each clip's transcript. Design and rationale:
`plans/PHASE2_PLAN.md`. Phase 1 code is imported, never modified; Phase 2 reads
`reporter_clips.csv` as a snapshot and never writes it, so it can run while Phase 1 is
still appending.

```powershell
uv run python scripts/phase2/run_phase2.py --reporters kubra_par --limit 20  # pilot
uv run python scripts/phase2/run_phase2.py --exclude-reporters cuneyt_ozdemir
uv run python scripts/phase2/run_phase2.py --workers 8
uv run python scripts/phase2/phase2_status.py --follow    # progress/ETA of a live run
uv run python scripts/phase2/qc_report.py                 # contact sheets + summary
uv run python scripts/phase2/calibrate_offscreen.py      # WORD_SYNC_MIN evidence
```

It is resumable the same way Phase 1 is: a source clip is finished once it has a row in
`data/phase2/face_track_report.csv` (written last), and a clip that crashed half-way has
its partial rows removed on the next start. The NUL padding a power cut leaves at the end
of those CSVs is cut off at that point too.

**Setup.** `uv sync` installs `mediapipe`, `python-speech-features` and `scipy`. The
MediaPipe face landmarker (`models/face_landmarker.task`) and the SyncNet weights
(`models/syncnet_v2.model`, 54 MB, from the authors' VGG page) download on first use and
are gitignored. Only SyncNet's network definition is vendored
(`scripts/phase2/syncnet_model.py`, MIT, from
[joonson/syncnet_python](https://github.com/joonson/syncnet_python)); the scoring is a
streaming rewrite of its `evaluate` that reproduces the reference demo
(`example.avi`: offset 3, confidence 10.05 vs the published 10.02):

```powershell
uv run python scripts/phase2/syncnet.py --selftest path\to\example.avi
```

**Per clip:**

1. Decode at 25 fps through one ffmpeg pipe; frame *k* is time *k*/25.
2. Count people on set with Phase 1's YuNet rule (`presenter_filter.detect_faces` +
   `is_on_set`) every 5 frames, Phase 1's 0.2 s cadence. MediaPipe's Face Landmarker
   runs on a crop around the presenter's face: its own detector is short-range and finds
   nothing on a full studio frame, where the face is ~130 px of 1080.
3. Mark gaps: no face, a second person, a mouth jump of more than half a face width, or
   implausible lip landmarks. Detector blinks of up to 2 frames are bridged.
4. Crop the mouth from a smoothed lip box grown 0.4× on each side → 88×88 grayscale.
   `mouth_width_px` is recorded in source pixels and never used as a filter.
5. SyncNet on smoothed 224×224 face crops. Clips scoring below 3 are discarded. The best
   A/V offset shifts every word window, and the framewise confidence gives each word its
   `word_sync_conf`, the off-screen-speech signal. Words below `WORD_SYNC_MIN = 2.5` are
   dropped. `calibrate_offscreen.py` measured that threshold by splicing another
   reporter's audio into half of each clip: it keeps 98.8% of real words and drops 97.6%
   of foreign-audio ones, while the clip-level score alone passed 21 of 38 half-foreign
   clips. A sentence is dropped when more than 30% of its words fail.
6. Words: the Whisper span ±15 frames. A word is dropped when its own span crosses a gap;
   the padding stops early at a gap (`track_ok = 0`). Sentences: Whisper segments of
   1–8 s, longer ones split at the pause nearest the middle.

**Storage.** One `data/lip_crops/<reporter>/<clip stem>.npz` per source clip holds every
frame once (`frames` uint8 `(T, 88, 88)`, plus `good` and the framewise `sync_conf`).
Word and sentence rows point into it with `npz_path` + `frame_start`/`frame_end`
(end-exclusive). A file per word would store each frame about 4 times over, because of
the ±15-frame padding. At ~1.2 M words that is ~340 GB uncompressed, more than this disk
holds. Per-class LRW-style files can be cut for the classes Phase 3 picks.

| File in `data/phase2/` | One row per |
|------|------|
| `word_clips.csv` | kept word: label, speaker, channel, frame range, Whisper confidence, `word_sync_conf`, `lip_motion`, pose, sizes |
| `sentence_clips.csv` | kept sentence: `text`, frame range, `word_spans` (JSON, times relative to the sentence start) |
| `syncnet_scores.csv` | scored source clip: `syncnet_score`, `syncnet_offset`, passed |
| `discarded.csv` | dropped source clip, word or sentence, with a reason |
| `face_track_report.csv` | processed source clip: frame status counts, words/sentences kept |

Throughput on this machine (RTX A4000, 20 threads): ~1.5× realtime per worker. 6 workers
process 0.18 h of clips in 86 s including start-up, and 10 workers are no faster because
MediaPipe's own threads already fill the CPU. The 180 h corpus is roughly a day of
running.

`channel` comes from joining `reporter_clips.csv` to `video_links.csv` on the YouTube
video id, and is written into every Phase 2 manifest.
