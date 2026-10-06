# Phase 2 — Visual Processing & Clip Generation: Implementation Plan

**Scope:** Phase 2 only (Visual Processing and Clip Generation, Months 2–5 of the project plan).
This document is a design/implementation plan — **no code yet**. It is written to align
with the Phase 1 artifacts already in this repo and to feed cleanly into Phase 3
(Dataset Compilation & Benchmark Design).

**Input available today** (`data/reporter_clips.csv`, measured 2026-10-06): 61,346 solo
clips, 168.9 h, 23 reporters, `face_height_px` filled on every row. That is already past
the 100 h raw-video target, so Phase 2 can start on the existing manifest while the
Phase 1 run continues to append to it.

---

## 0. Goal of Phase 2 (from the project plan)

Turn the time-aligned transcripts + qualifying videos from Phase 1 into two pools of
**mouth-ROI lip clips**:

- `word_clips/` — one clip per Whisper word, ±15 frames of coarticulation context.
- `sentence_clips/` — one clip per Whisper sentence segment (1–8 s), labeled with its
  full transcript. These become the **spotting evaluation set** in Phase 3.

Each clip is a sequence of **88×88 grayscale mouth crops** (LRW convention), produced by:

1. Face landmarks (MediaPipe Face Landmarker, 468/478 landmarks).
2. Face tracking; drop frames/segments on tracking failure (§4.3).
3. Lip-region crop from the ~20 mouth landmarks → 88×88 grayscale.
4. Audio-visual sync check (SyncNet); discard clips with score < 3.

---

## 1. What Phase 2 consumes (inputs from Phase 1)

| Input | Produced by | Notes |
|-------|-------------|-------|
| **`data/reporter_clips.csv`** (the master handoff table) | `run_all.py` → `pipeline2.py` | **The single Phase 1 → Phase 2 entry point** — one row per kept solo clip linking the clip mp4, its transcript JSON, and the reporter (see §1.2) |
| Solo-presenter clip mp4s + reporter identity | `pipeline2.select_clips` → `data/<reporter>_solo_clips/*.mp4` | Single-face, identity-verified, frontal (§1.1). **Named** `<reporter>_<srcvideo>_scene_<idx>_<start>-<end>.mp4` (clip stem encodes everything) |
| Per-clip transcripts | `pipeline2.transcribe_clip` → `data/reporter_transcripts/<clip_stem>.json` | **Clip-local timestamps** (each clip transcribed alone → times start at 0 within the clip). Same JSON schema below |
| (Stream A) full-video transcripts + raw videos | `run_whisper.py` → `data/transcripts/`, `data/raw_videos/` | 8 pipeline-1 rows. No face filter / no identity — see §1.1 |

Phase 1 transcript schema (must be treated as read-only contract):

```json
{
  "video": "<file>",
  "segments": [
    {
      "start_sec": 0.0, "end_sec": 0.0, "text": "...",
      "words": [
        {"surface_form": "...", "lemma": "...", "start_sec": 0.0,
         "end_sec": 0.0, "confidence": 0.0, "manual_review": 0}
      ]
    }
  ]
}
```

`surface_form` and `text` are NFC-normalized and stripped (`run_whisper.normalize_text`).
`lemma == surface_form` until zeyrek is wired in.

### 1.1 Key alignment decision: which stream feeds Phase 2

There are **two Phase 1 streams**. They differ in one Phase-3-critical way: **speaker identity**.

- **Stream B (reporter solo clips, `pipeline2.py`)** — already satisfies Phase 2's
  inclusion criteria and carries a `reporter` label → the natural `speaker_id`.
  **Primary input for Phase 2.** What each clip is already guaranteed to have passed
  (`presenter_filter.analyze_frames` + `solo_segments`, see CLAUDE.md invariant #3),
  checked on a sample **every 0.2 s**:
  - exactly one person on screen, counted with YuNet (score ≥ 0.80, face ≥ 4.5% of
    frame height) — split screens, video walls and interpreter boxes are already out;
  - identity voted per face track (median dlib distance ≤ 0.6 vs the reference photo);
  - head yaw ≤ 0.30 on every sample (frontal);
  - 0.3 s trimmed off each scene edge, clips ≥ 2.0 s;
  - source ≥ 720p (capped at 1080p).

  Clips written before these rules existed have been re-applied to them with
  `scripts/audit_clips.py`, so the whole manifest is on the same rule set.
  **Not** checked: pitch (looking down at a script), graphic overlays across the mouth,
  and face size (recorded as `face_height_px`, never gated — see §4.4).
- **Stream A (pipeline 1, `run_whisper.py`)** — 8 rows; transcripts cover *full videos*
  with no per-shot face filtering and **no speaker identity**. To use it, first run it
  through the same filter (`presenter_filter.detect_scenes` + `analyze_video` +
  `solo_segments` against each candidate reporter's reference encoding). Low priority:
  8 rows vs 61k Stream B clips.

> **Recommendation:** Build Phase 2 to consume **solo, identity-tagged clips** as the unit of
> work. Make the lip-extraction code source-agnostic: its contract is
> "(clip video with audio) + (matching transcript JSON) + (speaker_id, channel)".

### 1.2 Verified Phase 1 → Phase 2 contract (actual artifacts, confirmed in code)

`run_all.py` writes **`data/reporter_clips.csv`** as the master manifest of Stream B. Its
columns (from `run_all.CLIP_FIELDS`) are the literal handoff Phase 2 should read:

```
reporter, source_url, source_video, clip_file, start_sec, end_sec,
duration_sec, face_height_px, transcript_text, transcript_json
```

- `clip_file` → relative posix path to the solo clip `.mp4`.
- `transcript_json` → relative posix path to that clip's per-clip transcript JSON.
- `reporter` → the **speaker_id** (slug, e.g. `fatih_portakal`).
- `start_sec`/`end_sec` here are the clip's span **within the source video** (provenance);
  the timestamps *inside* `transcript_json` are **clip-local** and are what Phase 2 slices on.
- `face_height_px` → median presenter face height in **source** pixels. Empty means
  unmeasurable, not zero.

So Phase 2's input adapter is simply: *iterate `reporter_clips.csv`, and for each row open
`clip_file` + `transcript_json`.* No need to re-derive anything from the pipeline internals.

**The manifest is append-only while Phase 1 runs.** Phase 2 must read it as a snapshot
and never rewrite it (same hazard as `backfill_face_height.py`: a rewrite drops every
row `run_all` appended in the meantime). Resumability comes from Phase 2's own outputs
(§2.1), so re-reading the grown manifest later just picks up the new clips.

**⚠ Gap to fix — `channel` is not propagated.** `video_links.csv` has a `channel` column,
but `pipeline2.select_clips` does **not** copy it into `reporter_clips.csv`. Phase 3's
**channel-independent test split** needs it. Two options:
1. **Add `channel` to `run_all.CLIP_FIELDS`** and pass it through `pipeline2.select_clips`
   (and the Colab notebook's copy) so every future clip row carries its channel. Existing
   rows still need option 2, and adding a column to a CSV that `append_clip_rows` appends
   to without rewriting the header has to be done while no run is live.
2. **(Recommended for Phase 2, no upstream change)** In Phase 2's input adapter, **join**
   `reporter_clips.csv` to `video_links.csv` on video id (`presenter_filter.extract_video_id`
   on `source_url` and `url` — URLs are not always spelled the same way).

Either way, Phase 2 persists `channel` into its own manifests (§2.3) so Phase 3 never
has to go back to `video_links.csv`.

> **Speaker↔channel coupling caveat (for Phase 3, note now):** each reporter is tied to one
> program/channel (e.g. Kübra Par → TV100). So a *speaker*-independent hold-out is often
> also *channel*-independent by construction. Also note the speaker imbalance:
> `cuneyt_ozdemir` is the large majority of `video_links.csv`; Phase 3 will need per-speaker
> caps, and `run_all.py --max-hours` exists to stop his share growing further.

---

## 2. Output design (what Phase 2 produces)

### 2.1 Directory layout (extends existing `data/`, stays gitignored)

```
data/
├── raw_videos/                  (existing, stream A)
├── transcripts/                 (existing, stream A full-video)
├── pipeline2_raw/               (existing, transient — normally empty)
├── <reporter>_solo_clips/       (existing, stream B clip mp4s)
├── reporter_transcripts/        (existing, stream B per-clip JSON)
├── reporter_clips.csv           (existing, stream B master manifest = Phase 2 input)
├── lip_crops/                   ← NEW Phase 2 root
│   ├── word_clips/<clip_id>.npz
│   └── sentence_clips/<clip_id>.npz
├── phase2/                      ← NEW manifests & QC
│   ├── word_clips.csv           (per-word-clip metadata manifest)
│   ├── sentence_clips.csv       (per-sentence-clip metadata manifest)
│   ├── syncnet_scores.csv       (per source-clip AV sync score)
│   ├── discarded.csv            (what was dropped + reason)
│   └── face_track_report.csv    (per source-clip tracking stats)
```

`data/` is already gitignored as a whole, so nothing new is needed in `.gitignore`.

**`clip_id` convention:** reuse the existing solo-clip stem (e.g.
`fatih_portakal_<srcvideo>_scene_0007_12.30-18.40`) and suffix `_w<idx>` for word clips /
`_s<idx>` for sentence clips. This keeps Phase 2 IDs traceable straight back to the source
clip and its `reporter_transcripts/*.json` — no new ID namespace to reconcile.

**Resumability:** a source clip is done when it has a row in `face_track_report.csv`
(written last, after its `.npz` files and manifest rows). Skip those on restart; write
manifest rows incrementally. Same convention as `run_whisper.py`'s "skip if JSON exists".

- **Do NOT store word/sentence crops as flat per-class folders yet.** Class assignment
  (surface-form vs lemma, the 300-example threshold, the 500-class cut) is a **Phase 3**
  decision. Phase 2 stores clips flat + records `surface_form`/`lemma` in the manifest.

### 2.2 Per-clip storage format

Store each clip as a **compressed NumPy archive** (`.npz`), not mp4:

- `frames`: `uint8` array, shape `(T, 88, 88)` — raw grayscale mouth crops (no global
  mean/var normalization yet; that is applied at **release time** in Phase 5, matching the
  LRW-1000 convention, so the on-disk data stays reusable).
- Keep `T` variable (word clips ~15–30 frames, sentence clips ~25–200 frames). Padding/
  fixed-length tensoring is a training-time concern (Phase 4), not storage.

Rationale: lossless, no codec smearing on the lips, trivial to load for training. mp4 is
only needed transiently for SyncNet (which wants A/V together) — see §4.5.

### 2.3 Manifest schema (CSV, the Phase 2 → Phase 3 contract)

`word_clips.csv` — one row per word clip:

| column | source | notes |
|--------|--------|-------|
| `clip_id` | generated | source clip stem + `_w<idx>` (§2.1) |
| `npz_path` | generated | relative posix path |
| `surface_form` | transcript | NFC-normalized, stripped |
| `lemma` | transcript | currently == surface_form (zeyrek TODO from Phase 1) |
| `speaker_id` | `reporter_clips.csv:reporter` | the reporter slug |
| `channel` | `video_links.csv` join on video id | §1.2 |
| `source_clip` | `reporter_clips.csv:clip_file` | parent solo clip (provenance) |
| `source_video` | `reporter_clips.csv:source_video` | original YouTube video |
| `face_height_px` | `reporter_clips.csv:face_height_px` | carried through for Phase 3/4 size thresholds |
| `mouth_width_px` | face landmarks | median mouth width in source pixels **before** resize — the number that actually says how much the 88×88 crop was upscaled |
| `word_start_sec`, `word_end_sec` | transcript | original Whisper word span |
| `clip_start_sec`, `clip_end_sec` | computed | after ±15-frame padding |
| `n_frames` | computed | T |
| `fps` | computed | 25 after normalization (§3) |
| `confidence` | transcript | Whisper word prob |
| `manual_review` | transcript | 1 if confidence < 0.7 |
| `syncnet_score`, `syncnet_offset` | SyncNet | for the parent source clip |
| `mean_yaw`, `mean_pitch` | face landmarks | pose stats (precompute for Phase 5 error analysis) |
| `track_ok` | tracker | no tracking gap inside the word span |

`sentence_clips.csv` — same spine, replace word columns with:
`text` (full transcript), `sentence_start_sec`, `sentence_end_sec`, `split_index`
(if a >8 s sentence was split), and a `word_spans` reference (JSON list of
`{surface_form, lemma, start_sec, end_sec}` within the clip) — this is the raw material
for Phase 3's **spotting ground-truth annotation**.

---

## 3. Frame-rate & timestamp alignment (do this first — it underpins everything)

This is the single biggest correctness risk and the main "alignment" gap between Phase 1
(seconds) and Phase 2 (frames).

- The plan assumes **25 fps** ("±15 frames ≈ 0.6 s at 25 fps", "29-frame window" in Phase 4).
- YouTube broadcast downloads are typically 25 or 30 fps (sometimes 50/60). Whisper gives
  times in **seconds**.

**Plan:** normalize every source clip to a **canonical 25 fps** before lip extraction.
Do it the way `presenter_filter.sample_frames` already reads video: one ffmpeg process
(`presenter_filter.ffmpeg_executable()`) with `-vf fps=25` piping raw frames, rather than
re-encoding an intermediate mp4 or seeking with OpenCV. Then frame index
`= round(t_sec * 25)`. This makes ±15 frames, 1–8 s sentence bounds, and the Phase 4
29-frame/stride-4 sliding window all exact and consistent across the dataset.

Document the chosen fps in every manifest row (`fps` column) so nothing downstream has to
guess.

---

## 4. The processing pipeline (per source clip)

Process **one solo, identity-tagged source clip at a time**. Steps:

### 4.1 Decode & fps-normalize
25 fps constant frame rate via the ffmpeg pipe (§3). Keep the clip's audio track
available for SyncNet (read it from the original clip mp4; no re-encode needed).

### 4.2 Face landmarks — MediaPipe
- Use the **Tasks API `FaceLandmarker`** (`.task` model file in `models/`, auto-downloaded
  like YuNet). The legacy `mp.solutions.face_mesh` API is deprecated in current MediaPipe
  releases; do not build on it.
- Set `num_faces=2`: Stream B guarantees one face only on its 0.2 s samples, so a second
  face appearing between samples is caught here for free and the frame is marked bad.
- 0 faces on a frame → gap. Mean yaw/pitch come from the landmarker's facial
  transformation matrix.
- `uv add mediapipe` (not pip; CLAUDE.md "Deps"). Python 3.11/3.12 — same constraint as
  dlib. Check that the lock still resolves with the pinned `torch==2.5.1` / numpy.

### 4.3 Face tracking (re-verification mostly done upstream)
The original project plan asks for identity re-verification every 15 frames. Phase 1 now
does stronger than that (count + yaw every 0.2 s = every 5 frames, dlib identity every
2 s and at every new track), so a per-15-frame dlib re-encode in Phase 2 would mostly
repeat work at ~200 ms a face. Instead:
- Track continuity from landmarks: mouth-centre jump > 0.5 face-widths between frames, or
  a frame with 0 / 2 faces, marks a **gap**.
- Words/sentences overlapping a gap are dropped and logged to `discarded.csv`
  (`reason = track_fail`).
- **Optional** (off by default, behind a flag): dlib re-verification at gap boundaries
  only, reusing `presenter_filter.load_reference_encoding` / `IDENTITY_TOLERANCE`.

### 4.4 Lip-region extraction → 88×88 grayscale
- From the landmarks, take the **mouth landmark set** (MediaPipe `LIPS` indices, outer +
  inner lip; ~20 points around the mouth as the plan specifies).
- Compute a tight mouth bounding box, then **expand by a fixed margin** (~0.4×) so the
  lips never touch the crop edge across speech.
- **Stabilize the box temporally** (moving average of centre + size over a small window).
- Make the crop **square**, grayscale, resize to **88×88**. Store as `uint8`.
- Record `mouth_width_px` (pre-resize). **Do not gate on face/mouth size**: studio
  reporters sit at 126–137 px faces (`serdar_cebe`, `kubra_par`), so a size floor would
  delete whole speakers. Phase 3/4 decide thresholds from the recorded numbers.
- Sanity guard: reject frames where the mouth box is implausible relative to the face box
  (landmark failure) → gap.

### 4.5 Audio-visual sync check — SyncNet
- Compute a **SyncNet score per source clip** (needs the clip with audio + face crops at
  25 fps). Use `syncnet_python` (joonson) vendored under e.g. `third_party/syncnet_python/`
  with its pretrained weights downloaded on first use (not a pip package — document the
  setup). It runs on the existing cu121 torch.
- **Discard clips with score < 3** (plan threshold). Record `syncnet_offset` so a small
  constant A/V offset can be corrected rather than discarded.
- **Ordering:** SyncNet runs **before** word/sentence splitting. After it passes, audio is
  no longer needed (release is video-crops-only, CLAUDE.md "Copyright").
- Keep the threshold a config parameter — Phase 4 Stage C ablations study the sync
  threshold, and `syncnet_score` is stored per clip so re-thresholding needs no recompute.

### 4.6 Word-level clip generation
For each word in the (clip-local) transcript:
- `clip_start = word_start_sec − 15 frames`, `clip_end = word_end_sec + 15 frames`
  (clamped to clip bounds) → ~0.6–1.2 s, consistent with LRW's 29-frame standard while
  allowing LRW-1000-style variance. Phase 1's 2.0 s minimum clip length exists for this.
- Slice the already-extracted 88×88 frame stack → `(T,88,88)` → write `.npz`.
- Skip words whose frames overlap a tracking gap.
- Carry `confidence`/`manual_review` into the manifest (do **not** drop low-confidence
  words automatically — flag them; Phase 3 decides).

### 4.7 Sentence-level clip generation
- Keep Whisper segments with `1 s ≤ duration ≤ 8 s` as whole clips.
- For `duration > 8 s`, **split at natural pause boundaries** — use the inter-word gaps
  already present in `words[]` (largest silence gap nearest the midpoint, recurse until all
  pieces ≤ 8 s). Record `split_index`.
- Drop segments `< 1 s`.
- Label each sentence clip with its full `text` and the embedded `word_spans`.

---

## 5. Suggested script layout (mirrors existing modular style)

Reuse Phase 1 building blocks; do not duplicate them.

```
scripts/
├── phase2_extract_lips.py   # ffmpeg 25fps pipe → FaceLandmarker → tracking → 88x88 crops
├── phase2_sync_filter.py    # SyncNet wrapper → syncnet_scores.csv, discard < 3
├── phase2_make_clips.py     # word + sentence slicing from crops + transcript → .npz + manifests
└── phase2_run.py            # orchestrator: snapshot data/reporter_clips.csv, join channel,
                             #   iterate (clip_file + transcript_json), call the above
```

- Match existing conventions: `argparse` CLI with defaulted paths, `pathlib`, posix paths
  in manifests, NFC via `run_whisper.normalize_text`, ffmpeg via
  `presenter_filter.ffmpeg_executable`, video ids via `presenter_filter.extract_video_id`.
- Row filters like `run_all.py` (`--reporters`, `--exclude-reporters`, `--limit`), so a
  first pass can run on a few small, clean reporters (`kubra_par`, `cem_ogretir`) before
  the bulk.
- CPU parallelism like `presenter_filter.analyze_video` (process pool, one clip per task);
  SyncNet batched on the GPU.
- Keep `phase2_run.py` standalone; it does not need to be part of `run_all.py`.

---

## 6. Dependencies to add

| Package | Purpose | Note |
|---------|---------|------|
| `mediapipe` | Face Landmarker | `uv add mediapipe`; Python 3.11/3.12 |
| SyncNet (`syncnet_python`, vendored) | AV sync scoring | external repo + pretrained weights; document setup, not pip |
| (already present) `opencv-python`, `face_recognition`, `imageio-ffmpeg`, `numpy`, `tqdm`, `torch` (cu121) | — | reuse |

Add a short "Phase 2 / SyncNet setup" section to README documenting the vendored SyncNet +
weights.

---

## 7. Quality control & validation (before declaring Phase 2 done)

- **Visual spot-check:** contact sheet of N random word/sentence clips per reporter —
  crops centered, lips fully inside, 88×88. Include the smallest-face reporters.
- **Distribution checks:** clip durations (words 0.6–1.2 s; sentences 1–8 s),
  frames-per-clip, SyncNet scores, `mouth_width_px`, discard reasons.
- **Alignment audit:** for a sample, overlay the word text on the middle frame and verify the
  mouth is mid-articulation (catches fps/timestamp drift early).
- **Coverage report:** hours and word clips retained vs discarded per speaker/channel —
  feeds Phase 3's 500-class, 150k-clip targets.
- Everything logged to `data/phase2/*.csv` so QC is reproducible.

---

## 8. Scale, performance, storage

- **Input size:** ~169 h of clips today (§ top) → on the order of 15M frames at 25 fps.
  Face landmarks are the per-frame hot loop; budget a pilot on one reporter to measure
  frames/s before the full run.
- **Storage estimate:** 150k word clips × ~25 frames × 88×88 bytes ≈ ~29 GB raw before
  `.npz` compression; sentence clips add to this. Keep on local/gitignored storage, never
  committed (CLAUDE.md rule #3).

---

## 9. Open decisions to confirm (recommended defaults in **bold**)

0. **Channel:** **join on video id in the Phase 2 adapter** (§1.2 option 2); patching
   `CLIP_FIELDS` can follow when a run is stopped.
1. **Primary input stream:** **Stream B via `data/reporter_clips.csv`**; Stream A (8 rows)
   only after it passes the same filter.
2. **Canonical fps:** **25 fps**.
3. **Crop margin around lip landmarks:** **~0.4× expansion + square + temporal smoothing**
   (tune on the spot-check montage).
4. **Storage:** **`.npz` of `uint8 (T,88,88)`**, raw; global mean/var deferred to Phase 5.
5. **Low-confidence words:** **keep + flag** (`manual_review`), don't auto-drop.
6. **SyncNet:** vendor `syncnet_python`; threshold **< 3 discard**, score stored.
7. **Identity re-verification:** **off by default** (Phase 1 already does it), flag to
   enable at gap boundaries.
8. **Face/mouth size:** **record, don't gate** (`face_height_px`, `mouth_width_px`).

---

## 10. Definition of done (Phase 2)

- `data/lip_crops/word_clips/` and `.../sentence_clips/` populated with 88×88 grayscale
  `.npz` clips for all qualifying, sync-passing, identity-tagged source clips.
- `data/phase2/word_clips.csv` + `sentence_clips.csv` manifests complete, carrying
  `speaker_id`, `channel`, labels, timing, confidence, SyncNet score, pose stats, size
  stats and (for sentences) `word_spans`.
- `syncnet_scores.csv`, `discarded.csv`, `face_track_report.csv` written.
- QC montage + distribution report reviewed.
- README updated with Phase 2 run instructions + SyncNet setup; `mediapipe` in
  `pyproject.toml` / `uv.lock`.
- `CLAUDE.md` Status table: Phase 2 marked done.

> Net result: Phase 3 can start immediately — it only needs the two manifests + the `.npz`
> pools to do frequency analysis, vocabulary selection, speaker/channel-independent splits,
> and spotting ground-truth annotation, with **zero reprocessing of video**.
