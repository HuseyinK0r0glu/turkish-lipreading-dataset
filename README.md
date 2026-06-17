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
├── video_links.csv                    ← unified pipeline input (url, channel, start, end, pipeline, reporter)
├── reporter_links/                    ← per-reporter YouTube URL lists (pipeline 2 input)
│   ├── links_cuneyt_ozdemir.txt
│   ├── links_fatih_portakal.txt
│   ├── links_kubra_par.txt
│   ├── links_gulsah_ekinci.txt
│   ├── links_serdar_cebe.txt
│   ├── links_ece_uner.txt
│   ├── links_ismail_kucukkaya.txt
│   ├── links_cem_ogretir.txt
│   └── links_can_okanar.txt
├── pipeline2_reporter_pictures/       ← reference face photos, one <slug>.jpeg per reporter
├── requirements.txt
├── colab_run_all.ipynb                ← Colab GPU runner (both pipelines, recommended)
└── scripts/
    ├── build_video_links.py           ← rebuild video_links.csv from reporter_links/
    ├── run_all.py                     ← unified local runner (pipeline 1 + 2)
    ├── run_reporters.py               ← run pipeline 2 for selected reporters locally
    ├── pipeline2.py                   ← solo-presenter scene filter + clip export
    ├── pipeline_cuneyt_solo.py        ← shared building blocks (download, scene detect, face filter)
    ├── download_videos.py             ← pipeline 1 downloader
    ├── download_and_transcribe.py     ← pipeline 1 entry point (download + Whisper)
    └── run_whisper.py                 ← Whisper ASR wrapper
```

---

## Two Pipelines

### Pipeline 1 — General ASR

Downloads news clips defined in `video_links.csv` (rows with `pipeline=1`) and transcribes them with Whisper large-v3. Produces word-level JSON transcripts.

```bash
python scripts/run_all.py
# or just the ASR step:
python scripts/download_and_transcribe.py
```

### Pipeline 2 — Solo-Presenter Filter

For each reporter in `video_links.csv` (rows with `pipeline=2`), downloads the full broadcast, detects scene cuts, and keeps only scenes where exactly one face appears and it matches the reporter's reference photo. Kept scenes are clipped and transcribed with Whisper.

#### Reporters

| Reporter | Channel | Links file |
|----------|---------|------------|
| Cüneyt Özdemir | YouTube | `links_cuneyt_ozdemir.txt` |
| Fatih Portakal | Fatih Portakal TV | `links_fatih_portakal.txt` |
| Kübra Par | TV100 | `links_kubra_par.txt` |
| Gülşah Ekinci | Halk TV | `links_gulsah_ekinci.txt` |
| Serdar Cebe | Sözcü TV | `links_serdar_cebe.txt` |
| Ece Üner | Halk TV | `links_ece_uner.txt` |
| İsmail Küçükkaya | Halk TV | `links_ismail_kucukkaya.txt` |
| Cem Öğretir | atv Haber | `links_cem_ogretir.txt` |
| Can Okanar | A Haber | `links_can_okanar.txt` |

---

## Local Setup

**Python 3.11 or 3.12 required** — `face_recognition`/`dlib` are not compatible with 3.13+.

```bash
python3.11 -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

On Windows, `dlib` requires Visual Studio C++ Build Tools (Desktop development with C++ workload) and CMake on PATH. If `dlib` fails to install:

```bash
pip install --no-cache-dir dlib
pip install -r requirements.txt
```

### First run

1. Place one clear frontal face photo per reporter in `pipeline2_reporter_pictures/<slug>.jpeg`  
   (e.g. `cem_ogretir.jpeg`, `can_okanar.jpeg`).

2. Rebuild `video_links.csv` from the reporter link files:

```bash
python scripts/build_video_links.py
```

3. Run:

```bash
# Full run (both pipelines)
python scripts/run_all.py

# Pipeline 2 only (all reporters)
python scripts/run_reporters.py

# Pipeline 2, single reporter
python scripts/run_reporters.py cem_ogretir

# Quick smoke test (shortest video per reporter)
python scripts/run_all.py --test
```

---

## Google Colab (Recommended)

`colab_run_all.ipynb` runs both pipelines on a free Colab GPU (Whisper large-v3 is fast on T4).

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

## Outputs

| Output | Location | Description |
|--------|----------|-------------|
| Pipeline 1 videos | `data/raw_videos/` | Downloaded clips |
| Pipeline 1 transcripts | `data/transcripts/<video>.json` | Word-level Whisper output |
| Pipeline 2 clips | `data/<reporter>_solo_clips/` | Trimmed solo-anchor scenes |
| Pipeline 2 transcripts | `data/reporter_transcripts/` | Per-clip Whisper JSON |
| Pipeline 2 manifest | `data/reporter_clips.csv` | All kept clips + transcript text |
