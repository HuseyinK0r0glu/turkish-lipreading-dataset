# Phase 2 — Visual Processing & Clip Generation: Implementation Plan

**Scope:** Phase 2 only (Visual Processing and Clip Generation, Months 2–5 of the project plan).
This document is a design/implementation plan — **no code yet**. It is written to align
with the Phase 1 artifacts already in this repo and to feed cleanly into Phase 3
(Dataset Compilation & Benchmark Design).

---

## 0. Goal of Phase 2 (from the project plan)

Turn the time-aligned transcripts + qualifying videos from Phase 1 into two pools of
**mouth-ROI lip clips**:

- `word_clips/` — one clip per Whisper word, ±15 frames of coarticulation context.
- `sentence_clips/` — one clip per Whisper sentence segment (1–8 s), labeled with its
  full transcript. These become the **spotting evaluation set** in Phase 3.

Each clip is a sequence of **88×88 grayscale mouth crops** (LRW convention), produced by:

1. Face detection (MediaPipe Face Mesh, 468 landmarks).
2. Face tracking + re-verification every 15 frames (drop on tracking failure).
3. Lip-region crop from the ~20 mouth landmarks → 88×88 grayscale.
4. Audio-visual sync check (SyncNet); discard clips with score < 3.

---

## 1. What Phase 2 consumes (inputs from Phase 1)

| Input | Produced by | Notes |
|-------|-------------|-------|
| **`data/reporter_clips.csv`** (the master handoff table) | `run_all.py` → `pipeline2.py` | **The single Phase 1 → Phase 2 entry point** — one row per kept solo clip linking the clip mp4, its transcript JSON, and the reporter (see §1.2) |
| Solo-presenter clip mp4s + reporter identity | `pipeline2.select_clips` → `data/<reporter>_solo_clips/*.mp4` | Already single-face, face-verified, **named** `<reporter>_<srcvideo>_scene_<idx>_<start>-<end>.mp4` (clip stem encodes everything) |
| Per-clip transcripts | `pipeline2.transcribe_clip` → `data/reporter_transcripts/<clip_stem>.json` | **Clip-local timestamps** (each clip transcribed alone → times start at 0 within the clip). Same JSON schema below |
| (Stream A) full-video transcripts + raw videos | `run_whisper.py` → `data/transcripts/`, `data/raw_videos/` | No face filter / no identity yet — see §1.1 |

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

### 1.1 Key alignment decision: which stream feeds Phase 2

There are **two Phase 1 streams**. They differ in one Phase-3-critical way: **speaker identity**.

- **Stream B (reporter solo clips, `pipeline2.py`)** — already satisfies Phase 2's
  inclusion criteria (single anchor, faces camera, no split-screen) *and* carries a
  `reporter` label → this is the natural `speaker_id`. **Recommended primary input for
  Phase 2.** Phase 2 runs face-mesh/lip-crop on these clips + their per-clip transcripts.
- **Stream A (main CSV → full-video ASR, `run_whisper.py`)** — larger volume but transcripts
  cover *full videos* with no per-shot face filtering and **no speaker identity**. To use
  this stream in Phase 2 you must first run the same scene-detect + single-face-verify gate
  (the `pipeline_cuneyt_solo` building blocks) **and** resolve speaker identity (match each
  kept scene's face against the known reporter reference encodings). Until identity is
  resolved, Stream A clips cannot go into speaker-independent val/test splits.

> **Recommendation:** Build Phase 2 to consume **solo, identity-tagged clips** as the unit of
> work (Stream B today; Stream A after it passes the same solo-filter + identity match).
> Make the lip-extraction code source-agnostic: its contract is "(clip video with audio) +
> (matching transcript JSON) + (speaker_id, channel)". This keeps Phase 2 decoupled from
> which downloader produced the clip.

### 1.2 Verified Phase 1 → Phase 2 contract (actual artifacts, confirmed in code)

`run_all.py` writes **`data/reporter_clips.csv`** as the master manifest of Stream B. Its
columns (from `run_all.CLIP_FIELDS`) are the literal handoff Phase 2 should read:

```
reporter, source_url, source_video, clip_file, start_sec, end_sec,
duration_sec, transcript_text, transcript_json
```

- `clip_file` → relative path to the solo clip `.mp4`.
- `transcript_json` → relative path to that clip's per-clip transcript JSON.
- `reporter` → the **speaker_id** (slug, e.g. `fatih_portakal`).
- `start_sec`/`end_sec` here are the clip's span **within the source video** (provenance);
  the timestamps *inside* `transcript_json` are **clip-local** and are what Phase 2 slices on.

So Phase 2's input adapter is simply: *iterate `reporter_clips.csv`, and for each row open
`clip_file` + `transcript_json`.* No need to re-derive anything from the pipeline internals.

**⚠ Gap to fix — `channel` is not propagated.** `video_links.csv` has a `channel` column,
but `pipeline2.select_clips` does **not** copy it into `reporter_clips.csv`. Phase 3's
**channel-independent test split** needs it. Two options (do one in Phase 2):
1. **(Preferred, tiny change)** Add `channel` to `run_all.CLIP_FIELDS` and pass it through
   `pipeline2.select_clips` so every future clip row carries its channel.
2. **(No upstream change)** In Phase 2's input adapter, **join** `reporter_clips.csv` to
   `video_links.csv` on `source_url == url` to recover `channel`.

Either way, Phase 2 must persist `channel` into its own manifests (§2.3) so Phase 3 never
has to go back to `video_links.csv`.

> **Speaker↔channel coupling caveat (for Phase 3, note now):** each reporter is tied to one
> program/channel (e.g. Kübra Par → TV100). So a *speaker*-independent hold-out is often
> also *channel*-independent by construction — good for the test split, but it means you need
> **enough distinct speakers per channel** if you want val (speaker-independent) and test
> (speaker+channel-independent) to be cleanly separable. Worth tracking the
> reporter→channel map as Phase 2 builds manifests.

---

## 2. Output design (what Phase 2 produces)

### 2.1 Directory layout (extends existing `data/`, stays gitignored)

```
data/
├── raw_videos/                  (existing, stream A)
├── transcripts/                 (existing, stream A full-video)
├── pipeline2_raw/               (existing, stream B downloaded sources)
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

**`clip_id` convention:** reuse the existing solo-clip stem (e.g.
`fatih_portakal_<srcvideo>_scene_0007_12.30-18.40`) and suffix `_w<idx>` for word clips /
`_s<idx>` for sentence clips. This keeps Phase 2 IDs traceable straight back to the source
clip and its `reporter_transcripts/*.json` — no new ID namespace to reconcile.

**Resumability:** mirror `run_all.py`'s interrupt-safe pattern (it rewrites
`video_links.csv` `completed` flag after each video). Phase 2 should **skip any clip whose
`.npz` + manifest row already exist**, and write manifest rows incrementally, so a long run
can be killed and resumed — same convention as `run_whisper.py`'s "skip if JSON exists".

- **Do NOT store word/sentence crops as flat per-class folders yet.** Class assignment
  (surface-form vs lemma, the 300-example threshold, the 500-class cut) is a **Phase 3**
  decision. Phase 2 stores clips flat + records `surface_form`/`lemma` in the manifest;
  Phase 3 groups them. This keeps Phase 2 scope minimal (CLAUDE.md rule #1).
- Add `data/lip_crops/` and `data/phase2/` to `.gitignore` (binary crops, large).

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
| `npz_path` | generated | relative to repo root |
| `surface_form` | transcript | NFC-normalized |
| `lemma` | transcript | currently == surface_form (zeyrek TODO from Phase 1) |
| `speaker_id` | `reporter_clips.csv:reporter` | the reporter slug |
| `channel` | **`video_links.csv` join on `source_url`** | not in `reporter_clips.csv` today — recover per §1.2 gap fix |
| `source_clip` | `reporter_clips.csv:clip_file` | parent solo clip (provenance) |
| `source_video` | `reporter_clips.csv:source_video` | original YouTube video |
| `word_start_sec`, `word_end_sec` | transcript | original Whisper word span |
| `clip_start_sec`, `clip_end_sec` | computed | after ±15-frame padding |
| `n_frames` | computed | T |
| `fps` | computed | should be 25 after normalization (§4.1) |
| `confidence` | transcript | Whisper word prob |
| `manual_review` | transcript | 1 if confidence < 0.7 |
| `syncnet_score` | SyncNet | for the parent source clip |
| `mean_yaw`, `mean_pitch` | face mesh | pose stats (precompute for Phase 5 error analysis) |
| `track_ok` | tracker | passed re-verification |

`sentence_clips.csv` — same spine, replace word columns with:
`text` (full transcript), `sentence_start_sec`, `sentence_end_sec`, `split_index`
(if a >8 s sentence was split), and a `word_spans` reference (JSON list of
`{surface_form, lemma, start_sec, end_sec}` within the clip) — this is the raw material
for Phase 3's **spotting ground-truth annotation**.

> Precomputing `mean_yaw`/`mean_pitch` and `word_spans` now is cheap and saves a full
> reprocess in Phase 5 (head-pose error analysis) and Phase 3 (spotting GT). Aligned with
> the plan's later stages without expanding Phase 2's deliverable.

---

## 3. Frame-rate & timestamp alignment (do this first — it underpins everything)

This is the single biggest correctness risk and the main "alignment" gap between Phase 1
(seconds) and Phase 2 (frames).

- The plan assumes **25 fps** ("±15 frames ≈ 0.6 s at 25 fps", "29-frame window" in Phase 4).
- YouTube broadcast downloads are typically 25 or 30 fps (sometimes 50/60). Whisper gives
  times in **seconds**.

**Plan:** normalize every source clip to a **canonical 25 fps** before lip extraction
(re-encode once with ffmpeg, the executable resolver already exists in
`pipeline_cuneyt_solo.ffmpeg_executable()`). Then frame index `= round(t_sec * 25)`.
This makes ±15 frames, 1–8 s sentence bounds, and the Phase 4 29-frame/stride-4 sliding
window all exact and consistent across the dataset.

Document the chosen fps in every manifest row (`fps` column) so nothing downstream has to
guess.

---

## 4. The processing pipeline (per source clip)

Process **one solo, identity-tagged source clip at a time**. Steps:

### 4.1 Decode & fps-normalize
Re-encode to 25 fps (constant frame rate), keep audio for now (needed by SyncNet). Read
frames with OpenCV (already a dependency).

### 4.2 Face detection — MediaPipe Face Mesh
- Run Face Mesh per frame → 468 landmarks (+ optionally `FaceLandmarker` blendshapes/pose).
- Expect exactly one face (Stream B already guarantees this). If 0 faces on a frame,
  mark a gap; if >1, that frame violates inclusion criteria → flag the clip.
- Add `mediapipe` to `requirements.txt`. Confirm Python 3.11/3.12 (MediaPipe wheels, same
  constraint as dlib already documented in CLAUDE.md).

### 4.3 Face tracking + re-verification (every 15 frames)
- Track the single face across frames (landmark centroid continuity / IoU of face bbox).
- **Every 15 frames**, re-verify identity by re-encoding the detected face and comparing
  against the reporter reference encoding (reuse `face_recognition` +
  `pipeline2_reporter_pictures/<reporter>.jpeg`, same tolerance regime as Phase 1b).
- If re-verification fails (wrong/lost face) → **discard the affected segment** and log to
  `discarded.csv` with reason `track_fail`. This both enforces the plan's tracking rule and
  catches scene-cut leakage from the upstream detector.

### 4.4 Lip-region extraction → 88×88 grayscale
- From Face Mesh, take the **mouth landmark set** (MediaPipe `LIPS` connection indices,
  ~outer+inner lip points; select ~20 surrounding the mouth as the plan specifies).
- Compute a tight mouth bounding box, then **expand by a fixed margin** (e.g. ~0.4×) so the
  lips never touch the crop edge across speech.
- **Stabilize the box temporally** (moving average of center + size over a small window) to
  remove per-frame jitter — critical for clean training crops.
- Make the crop **square** (pad/extend the shorter side), convert to grayscale, resize to
  **88×88** (LRW). Store as `uint8`.
- Sanity guard: reject frames where the mouth box is implausibly small/large (off-frame
  pose) → contributes to the gap/discard logic.

### 4.5 Audio-visual sync check — SyncNet
- Compute a **SyncNet score per source clip** (needs the *full clip with audio* + face
  crops at 25 fps). Use the standard `syncnet_python` (joonson) reference implementation
  as an external/vendored tool (it is not a pip package — document the setup).
- **Discard clips with score < 3** (plan threshold). Also record `syncnet_offset` if
  available so a small constant A/V offset can be corrected rather than discarded.
- **Ordering:** SyncNet must run **before** word/sentence splitting and before audio is
  stripped. After it passes, audio is no longer needed (release is video-crops-only).
- Keep the threshold a config parameter — Phase 4 Stage C ablations explicitly study
  "impact of AV sync filtering threshold," so it must be tunable, and `syncnet_score` is
  stored per clip to allow re-thresholding without recompute.

### 4.6 Word-level clip generation
For each word in the (clip-local) transcript:
- `clip_start = word_start_sec − 15 frames`, `clip_end = word_end_sec + 15 frames`
  (clamped to clip bounds) → ~0.6–1.2 s, consistent with LRW's 29-frame standard while
  allowing LRW-1000-style variance.
- Slice the already-extracted 88×88 grayscale frame stack → `(T,88,88)` → write `.npz`.
- Skip/flag words whose frames overlap a tracking-gap or discarded segment.
- Carry `confidence`/`manual_review` into the manifest (do **not** drop low-confidence
  words automatically — flag them; Phase 3 manual verification decides).

### 4.7 Sentence-level clip generation
- Keep Whisper segments with `1 s ≤ duration ≤ 8 s` as whole clips.
- For `duration > 8 s`, **split at natural pause boundaries** — use the inter-word gaps
  already present in `words[]` (largest silence gap nearest the midpoint, recurse until all
  pieces ≤ 8 s). Record `split_index`.
- Drop segments `< 1 s`.
- Label each sentence clip with its full `text` and the embedded `word_spans` (for Phase 3
  spotting GT).

---

## 5. Suggested script layout (mirrors existing modular style)

Reuse Phase 1 building blocks; do not duplicate them.

```
scripts/
├── phase2_extract_lips.py   # decode→fps-norm→facemesh→track/reverify→88x88 crops
│                            #   (imports ffmpeg_executable, scene/face helpers, normalize_text)
├── phase2_sync_filter.py    # SyncNet wrapper → syncnet_scores.csv, discard < 3
├── phase2_make_clips.py     # word + sentence slicing from crops + transcript → .npz + manifests
└── phase2_run.py            # orchestrator: read data/reporter_clips.csv, iterate each
                             #   (clip_file + transcript_json), call the above, write manifests
```

- Match existing conventions: `argparse` CLI with defaulted paths, `pathlib`, NFC via
  `run_whisper.normalize_text`, ffmpeg via `pipeline_cuneyt_solo.ffmpeg_executable`,
  face matching via the Phase 1b reference-encoding helpers.
- Integrate `phase2_run.py` into the existing `run_all.py` orchestration as a stage that
  runs after transcripts exist (optional — can stay standalone first).

---

## 6. Dependencies to add

| Package | Purpose | Note |
|---------|---------|------|
| `mediapipe` | Face Mesh / landmarks | Python 3.11/3.12 |
| SyncNet (`syncnet_python`, vendored) | AV sync scoring | external repo + pretrained weights; document setup, not pip |
| (already present) `opencv-python`, `face_recognition`, `imageio-ffmpeg`, `numpy`, `tqdm` | — | reuse |

Add `mediapipe` to `requirements.txt`; add a short "Phase 2 / SyncNet setup" section to
README documenting the vendored SyncNet + weights + the 3.11/3.12 requirement.

---

## 7. Quality control & validation (before declaring Phase 2 done)

- **Visual spot-check:** dump a contact-sheet (montage) of N random word/sentence clips to
  confirm crops are centered, lips fully inside, grayscale, 88×88.
- **Distribution checks:** histogram of clip durations (words 0.6–1.2 s; sentences 1–8 s),
  frames-per-clip, SyncNet scores, discard reasons.
- **Alignment audit:** for a sample, overlay the word text on the middle frame and verify the
  mouth is mid-articulation (catches fps/timestamp drift early).
- **Coverage report:** hours retained vs discarded per speaker/channel — feeds Phase 1's
  ≥100 h / Phase 3's 500-class, 150k-clip targets.
- Everything logged to `data/phase2/*.csv` so QC is reproducible, not ad hoc.

---

## 8. Scale, performance, storage

- **Throughput:** Face Mesh + crop is the per-frame hot loop; batch by clip, optional GPU.
  SyncNet is per-clip and cheaper. Consider the existing Colab notebook for GPU runs.
- **Storage estimate:** 150k word clips × ~25 frames × 88×88 bytes ≈ ~29 GB raw before
  `.npz` compression; sentence clips add to this. Keep on local/gitignored storage, never
  committed (CLAUDE.md rule #3).
- **Resumability:** skip clips whose `.npz` + manifest row already exist (mirror the
  existing "skip if exists" pattern in `run_whisper.py` / downloaders).

---

## 9. Open decisions to confirm (recommended defaults in **bold**)

0. **Channel propagation (do this first, §1.2):** **patch `run_all.CLIP_FIELDS` +
   `pipeline2.select_clips` to carry `channel`** into `reporter_clips.csv` (preferred), or
   join on `source_url` in the Phase 2 adapter. Required for Phase 3's channel-independent test split.
1. **Primary input stream:** **Stream B via `data/reporter_clips.csv`** first; fold in
   Stream A only after it passes the same solo-filter + identity match.
2. **Canonical fps:** **25 fps** (re-encode all clips). Matches every downstream frame count.
3. **Crop margin around lip landmarks:** **~0.4× expansion + square + temporal smoothing**
   (tune on the spot-check montage).
4. **Storage:** **`.npz` of `uint8 (T,88,88)`**, raw (un-normalized); global mean/var
   deferred to Phase 5 release.
5. **Low-confidence words:** **keep + flag** (`manual_review`), don't auto-drop.
6. **SyncNet:** vendor `syncnet_python`; threshold **< 3 discard**, but store the score so
   the Phase 4 ablation can re-threshold without recompute.

---

## 10. Definition of done (Phase 2)

- `data/lip_crops/word_clips/` and `.../sentence_clips/` populated with 88×88 grayscale
  `.npz` clips for all qualifying, sync-passing, identity-tagged source clips.
- `data/phase2/word_clips.csv` + `sentence_clips.csv` manifests complete, carrying
  `speaker_id`, `channel`, labels, timing, confidence, SyncNet score, pose stats, and
  (for sentences) `word_spans`.
- `syncnet_scores.csv`, `discarded.csv`, `face_track_report.csv` written.
- QC montage + distribution report reviewed.
- README updated with Phase 2 run instructions + SyncNet setup; `mediapipe` in
  `requirements.txt`; new dirs in `.gitignore`.
- Phase checklist in `CLAUDE.md` Phase 2 boxes ticked.

> Net result: Phase 3 can start immediately — it only needs the two manifests + the `.npz`
> pools to do frequency analysis, vocabulary selection, speaker/channel-independent splits,
> and spotting ground-truth annotation, with **zero reprocessing of video**.
