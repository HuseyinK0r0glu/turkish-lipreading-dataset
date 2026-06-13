# Turkish News Lip Reading Spotting Benchmark

**Goal:** First Turkish visual word-spotting lip-reading benchmark from YouTube news. Word + sentence clips, Whisper ASR, then baseline classifiers + sliding-window spotting. Full plan: `TurkishNewsBrodcasts-LipReadingSpottingBenchmark-ProjectPlan.pdf` (read only if task requires it).

---

## Repo Layout

```
turkish-lipreading-dataset/
├── video_links.csv          ← main pipeline input (url, channel, start, end)
├── linkCNN.txt              ← presenter pilot URLs (one per line)
├── c.webp                   ← reference face for presenter pilot
├── requirements.txt
├── scripts/
│   ├── download_and_transcribe.py   ← Phase 1 entry point
│   ├── download_videos.py
│   ├── run_whisper.py
│   └── pipeline_cuneyt_solo.py      ← presenter/scene filter pilot
└── data/                    ← gitignored except tracked CSVs
    ├── raw_videos/
    ├── transcripts/         ← per-video JSON from Whisper
    └── cuneyt_solo_clips/   ← filtered clips + scene_boundaries.csv
```

---

## Status

| Phase | Status |
|-------|--------|
| 1 — Download + Whisper ASR | In progress (scripts done, small CSV sample) |
| 1b — Presenter solo filter | Started (CNN/Cüneyt pilot) |
| 2 — Lip extraction + SyncNet | Not started |
| 3 — Dataset splits + benchmark | Not started |
| 4–5 — Models + release | Not started |

---
    
## Two Ingestion Pipelines

### A — Main (CSV → ASR)

`video_links.csv` columns: `url,channel,start,end`. Empty start/end = full video. Overlapping intervals on same URL are skipped automatically.

Channels (target ≥25 h each): TRT Haber, CNN Türk, NTV, Habertürk.

Run: `python scripts/download_and_transcribe.py`

Transcript JSON (`data/transcripts/<video>.json`): segments with `words[]` → `surface_form`, `lemma` (TODO: zeyrek), `start_sec`, `end_sec`, `confidence`, `manual_review` (1 if confidence < 0.7).

### B — Presenter solo filter (pilot)

Downloads videos from `linkCNN.txt`, detects scene cuts (PySceneDetect), samples 3 frames per scene, and keeps only scenes where exactly one face appears and it matches `c.webp` (face_recognition, tolerance 0.47). Passing clips are exported via ffmpeg.

Run: `python scripts/pipeline_cuneyt_solo.py`

Outputs: `data/cuneyt_solo_clips/*.mp4`, `scene_boundaries.csv` (kept scenes + `TOTAL_KEPT` duration row).

**Requires Python 3.11 or 3.12** (face_recognition/dlib not supported on 3.13+).

---

## Phase Checklist

### Phase 1 — Data collection & ASR
- [x] yt-dlp downloader (overlap skip, existing-file skip)
- [x] Whisper large-v3 + word timestamps + NFC normalization
- [x] Presenter solo pilot pipeline
- [ ] ≥100 h qualifying video across main channels
- [ ] ASR on all segments
- [ ] Lemma extraction via zeyrek (currently `lemma == surface_form`)

### Phase 2 — Visual processing
- [ ] MediaPipe Face Mesh → lip crop 88×88 grayscale (LRW standard)
- [ ] Face re-verify every 15 frames; SyncNet discard if score < 3
- [ ] Word clips (±15 frames) → `word_clips/`; sentence clips (1–8 s) → `sentence_clips/`

### Phase 3 — Benchmark
- [ ] Vocab: ≥300 examples/class → 500+ classes, 150k+ word clips
- [ ] Splits 80/10/10 (val: speaker-independent; test: speaker + channel independent)

### Phases 4–5
Baselines: 3D CNN+BiGRU, ResNet-18+MS-TCN, VTP/SwinLip, AV-HuBERT+LoRA → sliding-window spotting (29-frame window, stride 4) → HuggingFace release (lip crops + metadata only, no raw video/audio).

---

## Conventions for AI Assistants

1. **Minimal scope** — only the requested phase/step.
2. **Match existing scripts** — paths, JSON schema, Turkish NFC normalization.
3. **Never commit** `data/raw_videos/`, `data/transcripts/`, `*.mp4`, raw footage.
4. **Copyright** — publish mouth ROI crops only, not raw video.
5. **Deps** — `requirements.txt` (yt-dlp, whisper, torch, opencv, scenedetect, face_recognition, imageio-ffmpeg).

## Key Targets

| Item | Target |
|------|--------|
| Raw video | 100+ h |
| Word classes | 500+ |
| Word clips | 150,000+ |
| Min examples/class | 300 (train) |
| Lip crop | 88×88 grayscale |
| ASR review flag | confidence < 0.7 |
| SyncNet discard | score < 3 |
