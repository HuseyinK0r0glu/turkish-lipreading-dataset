"""Phase 3 step 6: the manual verification set for Label Studio.

Draws a seeded random sample (default 5%), stratified by split and speaker, of

  words      in-vocabulary word clips of all three splits: is the label what is said,
             and is it said by the mouth on screen? (covers the off-screen-speech check)
  sentences  val/test sentence clips: is the transcript right?
  spotting   test occurrences from spotting_gt/queries.csv (up to --gt-per-class per
             class): the annotator corrects the start/end of the target word

and renders a preview for each into data/phase3/verification/media/: the 88x88 gray
crop at 25 fps, upscaled, muxed with the matching audio cut from the local source clip
(shifted by the SyncNet offset, like the frames). The audio is what makes a word label
checkable at all; the previews are INTERNAL annotation material under the gitignored
data/ and are never part of release/ (--no-audio renders crops only).

Two annotators: every item goes to A or B, and --overlap of them to both; score_kappa.py
computes Cohen's kappa on that overlap from the two exports.

    uv run python scripts/phase3/make_verification_set.py --limit 20     # try it first
    uv run python scripts/phase3/make_verification_set.py                # full 5%, resumable
    uv run python scripts/phase3/make_verification_set.py --no-render    # lists only

Rendering skips previews that already exist, so an interrupted run continues.
See data/phase3/verification/README.md (written by this script) for the Label Studio steps.
"""
import argparse
import hashlib
import json
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

import common as C

VERDICTS = ["correct", "wrong_word", "offscreen", "bad_crop", "unsure"]

CONFIG_WORDS = """<View>
  <Header value="Is this word said in the clip, by the mouth you see?"/>
  <Text name="prompt" value="$prompt"/>
  <Video name="video" value="$video"/>
  <Choices name="verdict" toName="video" choice="single" required="true" showInline="true">
    <Choice value="correct" hotkey="1"/>
    <Choice value="wrong_word" hotkey="2"/>
    <Choice value="offscreen" hotkey="3"/>
    <Choice value="bad_crop" hotkey="4"/>
    <Choice value="unsure" hotkey="5"/>
  </Choices>
  <TextArea name="label_fix" toName="video" maxSubmissions="1" editable="true"
            placeholder="wrong_word only: the word that is actually said"/>
</View>
"""
CONFIG_SENTENCES = """<View>
  <Header value="Is this transcript what is said, by the mouth you see?"/>
  <Text name="prompt" value="$prompt"/>
  <Video name="video" value="$video"/>
  <Choices name="verdict" toName="video" choice="single" required="true" showInline="true">
    <Choice value="correct" hotkey="1"/>
    <Choice value="wrong_word" hotkey="2"/>
    <Choice value="offscreen" hotkey="3"/>
    <Choice value="bad_crop" hotkey="4"/>
    <Choice value="unsure" hotkey="5"/>
  </Choices>
  <TextArea name="label_fix" toName="video" maxSubmissions="1" editable="true"
            placeholder="wrong_word only: the corrected transcript"/>
</View>
"""
CONFIG_SPOTTING = """<View>
  <Header value="Move the region so it covers exactly the target word"/>
  <Text name="prompt" value="$prompt"/>
  <Video name="video" value="$video" sync="audio"/>
  <Audio name="audio" value="$video" sync="video" zoom="true"/>
  <Labels name="occ" toName="audio">
    <Label value="target" background="#d33"/>
  </Labels>
  <Choices name="verdict" toName="audio" choice="single" required="true" showInline="true">
    <Choice value="present" hotkey="1"/>
    <Choice value="absent" hotkey="2"/>
    <Choice value="unsure" hotkey="3"/>
  </Choices>
</View>
"""
README = """# Phase 3 manual verification

Written by `scripts/phase3/make_verification_set.py`. Everything here is internal
annotation material (the previews carry audio): never copy `media/` into `release/`.

- `sample.csv` - every sampled item: kind, clip_id, split, speaker, labels, media, annotator
- `tasks_words_A.json`, `tasks_words_B.json` - word tasks per annotator (overlap items in both)
- `tasks_sentences_A.json`, `tasks_sentences_B.json` - sentence transcript tasks
- `tasks_spotting.json` - test occurrences with the automatic span pre-filled as a region
- `label_config_*.xml` - paste into the project's Labeling Interface (Code view)
- `media/` - previews: 88x88 crops upscaled, 25 fps, source audio

## Label Studio (not a project dependency; run it from its own environment)

```powershell
$env:LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED = "true"
$env:LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT = "{root}"
uvx label-studio start
```

One project per task file. In each project: Settings > Labeling Interface > Code, paste
the matching `label_config_*.xml`; Settings > Cloud Storage > Add Source Storage > Local
files with absolute path `{root}/media` (this only grants read access, do not sync it);
then Import the `tasks_*.json`.

Verdicts: `correct`; `wrong_word` (type what is said into the text box); `offscreen` (the
voice is not the mouth on screen: the Phase 2 off-screen check); `bad_crop` (mouth not
visible / wrong region); `unsure`. A word clip shows +-15 frames of context, so its
neighbours are audible too: judge the word in the middle.

Spotting: drag the pre-filled `target` region on the waveform until it covers exactly
the target word; `absent` if the word is not said in the clip.

When done, export each project as JSON into `exports/` and run:

```powershell
uv run python scripts/phase3/score_kappa.py
```

It writes `kappa.txt`, `verified.csv` (clip_id, verified, label_fix; the manifests are
never overwritten) and `../spotting_gt/test_gt_verified.csv`.
"""


def media_name(kind: str, clip_id: str) -> str:
    return f"{kind}_{hashlib.sha1(clip_id.encode('utf-8')).hexdigest()[:16]}.mp4"


def ffmpeg_exe() -> str:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def render(job, out_dir, ffmpeg, audio, scale):
    """One preview. -> '' when written or already there, else an error string."""
    out = out_dir / job["media"]
    if out.exists():
        return ""
    try:
        f0, f1 = int(job["frame_start"]), int(job["frame_end"])
        frames = np.load(C.ROOT / job["npz_path"])["frames"][f0:f1]
        if not len(frames):
            return f"{job['clip_id']}: empty frame range"
        h, w = frames.shape[1:]
        dur = len(frames) / C.FPS
        tmp = out.with_name(out.stem + ".part.mp4")
        source = C.ROOT / job["source_clip"]
        video_in = ["-f", "rawvideo", "-pix_fmt", "gray", "-s", f"{w}x{h}", "-r", str(C.FPS),
                    "-i", "-"]
        video_out = ["-vf", f"scale={w * scale}:{h * scale}:flags=neighbor", "-c:v", "libx264",
                     "-pix_fmt", "yuv420p", "-crf", "18", "-movflags", "+faststart",
                     "-t", f"{dur:.3f}", str(tmp)]
        attempts = []
        if audio and source.exists():
            # frame f shows the mouth of audio time (f - offset) / fps (CLAUDE.md invariant 12)
            t0 = (f0 - int(float(job.get("syncnet_offset") or 0))) / C.FPS
            delay = max(0.0, -t0)
            a_in = ["-ss", f"{max(0.0, t0):.3f}", "-t", f"{dur:.3f}", "-i", str(source)]
            a_out = ["-map", "0:v", "-map", "1:a", "-c:a", "aac", "-b:a", "64k", "-ac", "1"]
            if delay:
                a_out += ["-af", f"adelay={int(round(delay * 1000))}:all=1"]
            attempts.append(video_in + a_in + a_out + video_out)
        attempts.append(video_in + ["-map", "0:v"] + video_out)
        err = ""
        for cmd in attempts:
            p = subprocess.run([ffmpeg, "-v", "error", "-y"] + cmd, input=frames.tobytes(),
                               capture_output=True)
            if p.returncode == 0 and tmp.exists():
                tmp.replace(out)
                return ""
            err = p.stderr.decode("utf-8", "replace").strip()[-200:]
        return f"{job['clip_id']}: {err}"
    except Exception as e:  # one unreadable npz must not stop a 30,000-clip render
        return f"{job['clip_id']}: {type(e).__name__}: {e}"


def stratified(df, fraction, seed, limit):
    part = df.groupby(["split", "speaker_id"], group_keys=False).sample(frac=fraction,
                                                                       random_state=seed)
    part = part.sample(frac=1, random_state=seed)  # shuffle: annotators do not go by speaker
    return part.head(limit) if limit else part


def assign(df, overlap, seed):
    """annotator column: 'both' for the first `overlap` share, then A/B alternating."""
    n_both = int(round(overlap * len(df)))
    who = np.where(np.arange(len(df)) % 2 == 0, "A", "B").astype(object)
    who[:n_both] = "both"
    return df.assign(annotator=who)


def url(media: str) -> str:
    return f"/data/local-files/?d=media/{media}"


def write_tasks(path, tasks):
    C.write_text(path, json.dumps(tasks, ensure_ascii=False, indent=1) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--phase3", type=C.Path, default=C.PHASE3)
    ap.add_argument("--fraction", type=float, default=0.05, help="share of in-vocab word clips")
    ap.add_argument("--sentence-fraction", type=float, default=0.05,
                    help="share of val/test sentence clips")
    ap.add_argument("--overlap", type=float, default=0.10,
                    help="share of the sample both annotators label (for kappa)")
    ap.add_argument("--gt-per-class", type=int, default=20,
                    help="test occurrences per class and labeling to boundary-correct (0 = none)")
    ap.add_argument("--limit", type=int, default=0, help="at most this many items per kind")
    ap.add_argument("--seed", type=int, default=None, help="default: the seed in splits.json")
    ap.add_argument("--no-render", action="store_true")
    ap.add_argument("--no-audio", action="store_true")
    ap.add_argument("--scale", type=int, default=4, help="preview upscale factor")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    C.utf8_stdout()
    rep = C.Report()
    p3 = args.phase3
    out = p3 / "verification"
    info = json.loads(C.require(p3 / "splits.json", "build_splits.py").read_text("utf-8"))
    seed = info["parameters"]["seed"] if args.seed is None else args.seed
    render_cols = ["clip_id", "npz_path", "frame_start", "frame_end", "source_clip",
                   "syncnet_offset", "speaker_id"]

    # ---- words ----
    cols = render_cols + ["surface_form", "label_surface", "label_lemma", "split",
                          "in_vocab_surface", "in_vocab_lemma", "confidence", "manual_review"]
    words = pd.concat([pd.read_csv(C.require(p3 / f"word_{s}.csv", "build_splits.py"), dtype=str,
                                   keep_default_na=False, usecols=cols) for s in C.SPLITS])
    pool = words[(words.in_vocab_surface == "1") | (words.in_vocab_lemma == "1")]
    del words
    w = assign(stratified(pool, args.fraction, seed, args.limit), args.overlap, seed)
    w = w.assign(kind="word", media=[media_name("w", c) for c in w.clip_id])
    rep(f"Phase 3 make_verification_set (seed {seed})")
    rep(f"words: {len(w):,} sampled of {len(pool):,} in-vocab word clips "
        f"({args.fraction:.0%}, by split and speaker)")
    rep("  per split: " + ", ".join(f"{k} {v:,}" for k, v in w.split.value_counts().items()))

    # ---- sentences ----
    s_cols = render_cols + ["text", "split"]
    sents = pd.concat([pd.read_csv(C.require(p3 / f"sentence_{s}.csv", "build_spotting_set.py"),
                                   dtype=str, keep_default_na=False, usecols=s_cols)
                       for s in ["val", "test"]])
    s = assign(stratified(sents, args.sentence_fraction, seed, args.limit), args.overlap, seed)
    s = s.assign(kind="sentence", media=[media_name("s", c) for c in s.clip_id])
    rep(f"sentences: {len(s):,} sampled of {len(sents):,} val/test sentence clips")

    # ---- spotting ground truth (test only) ----
    g = pd.DataFrame()
    if args.gt_per_class:
        q = pd.read_csv(C.require(p3 / "spotting_gt" / "queries.csv", "build_spotting_set.py"),
                        dtype=str, keep_default_na=False)
        q = q[q.split == "test"]
        picked = (q.groupby(["labeling", "class"], group_keys=False)
                  .apply(lambda d: d.sample(min(len(d), args.gt_per_class), random_state=seed)))
        key = ["clip_id", "start_sec", "end_sec"]
        # one task per spoken occurrence, whichever labelings picked it
        g = (q.merge(picked[key].drop_duplicates(), on=key)
             .groupby(key + ["surface_form"], as_index=False)
             .agg(classes=("class", lambda c: " / ".join(sorted(set(c))))))
        g = g.sample(frac=1, random_state=seed)
        if args.limit:
            g = g.head(args.limit)
        g = g.merge(sents[render_cols + ["text"]], on="clip_id", how="left")
        g = g.assign(kind="spotting", split="test", annotator="A",
                     media=[media_name("s", c) for c in g.clip_id])
        rep(f"spotting: {len(g):,} test occurrences to boundary-correct "
            f"(up to {args.gt_per_class} per class and labeling, of {len(q):,})")

    # ---- task files ----
    def word_task(r):
        return {"data": {
            "video": url(r.media), "clip_id": r.clip_id, "kind": "word", "split": r.split,
            "speaker_id": r.speaker_id, "surface_form": r.surface_form,
            "label_surface": r.label_surface, "label_lemma": r.label_lemma,
            "prompt": f"{r.surface_form}     (lemma class: {r.label_lemma or '-'}, "
                      f"surface class: {r.label_surface or '-'})"}}

    def sentence_task(r):
        return {"data": {"video": url(r.media), "clip_id": r.clip_id, "kind": "sentence",
                         "split": r.split, "speaker_id": r.speaker_id, "prompt": r.text}}

    def spotting_task(r):
        region = {"from_name": "occ", "to_name": "audio", "type": "labels",
                  "value": {"start": float(r.start_sec), "end": float(r.end_sec),
                            "labels": ["target"]}}
        return {"data": {"video": url(r.media), "clip_id": r.clip_id, "kind": "spotting",
                         "auto_start_sec": r.start_sec, "auto_end_sec": r.end_sec,
                         "target": r.surface_form, "classes": r.classes,
                         "prompt": f"target: {r.surface_form}     in: {r.text}"},
                "predictions": [{"model_version": "auto_seed", "result": [region]}]}

    for name, df, make in [("words", w, word_task), ("sentences", s, sentence_task)]:
        for who in "AB":
            part = df[df.annotator.isin([who, "both"])]
            write_tasks(out / f"tasks_{name}_{who}.json", [make(r) for r in part.itertuples()])
            rep(f"  tasks_{name}_{who}.json: {len(part):,} tasks "
                f"({int((part.annotator == 'both').sum()):,} shared)")
    if len(g):
        write_tasks(out / "tasks_spotting.json", [spotting_task(r) for r in g.itertuples()])
    for name, xml in [("words", CONFIG_WORDS), ("sentences", CONFIG_SENTENCES),
                      ("spotting", CONFIG_SPOTTING)]:
        C.write_text(out / f"label_config_{name}.xml", xml)
    C.write_text(out / "README.md", README.replace("{root}", out.resolve().as_posix()))
    (out / "exports").mkdir(parents=True, exist_ok=True)

    sample_cols = ["kind", "clip_id", "split", "speaker_id", "annotator", "media",
                   "surface_form", "label_surface", "label_lemma", "text", "start_sec", "end_sec"]
    sample = pd.concat([w, s, g]).reindex(columns=sample_cols).fillna("")
    C.write_csv(sample, out / "sample.csv")

    # ---- previews ----
    jobs = (pd.concat([w, s, g])[render_cols + ["media"]].drop_duplicates("media")
            .to_dict("records"))
    media = out / "media"
    media.mkdir(parents=True, exist_ok=True)
    for part in media.glob("*.part.mp4"):
        part.unlink()
    todo = [j for j in jobs if not (media / j["media"]).exists()]
    rep(f"previews: {len(jobs):,} needed, {len(jobs) - len(todo):,} already rendered")
    if args.no_render:
        rep("  --no-render: skipped")
    elif todo:
        ffmpeg, errors, done = ffmpeg_exe(), [], 0
        with ThreadPoolExecutor(args.workers) as pool_:
            for err in pool_.map(lambda j: render(j, media, ffmpeg, not args.no_audio,
                                                  args.scale), todo):
                done += 1
                if err:
                    errors.append(err)
                if done % 500 == 0:
                    print(f"  rendered {done:,}/{len(todo):,} ({len(errors)} failed)", flush=True)
        rep(f"  rendered {len(todo) - len(errors):,}, failed {len(errors)}")
        for e in errors[:10]:
            rep(f"    {e}")
    rep.save(out / "sample_report.txt")
    print(f"\nnext: {(out / 'README.md').relative_to(C.ROOT).as_posix()}")


if __name__ == "__main__":
    main()
