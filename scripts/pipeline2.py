"""Pipeline 2 (solo-presenter) processing for the unified CSV flow.

Per pipeline-2 video:
  download -> scene detect -> keep scenes where ONLY the reporter appears
  (matched against pipeline2_reporter_pictures/<reporter>.jpeg) -> export clips
  -> transcribe each clip with Whisper (same JSON schema as pipeline 1).

Reuses the building blocks from presenter_filter.py and run_whisper.py so the
behaviour stays consistent with the existing scripts.
"""

import gc
import json
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))
from presenter_filter import (  # noqa: E402
    detect_scenes,
    download_video,
    export_clip,
    safe_name,
    scene_has_only_target,
)
from run_whisper import normalize_text  # noqa: E402

PROJECT = Path(__file__).resolve().parent.parent
REPORTER_PICS = PROJECT / "pipeline2_reporter_pictures"

_model = None


def reporter_image(reporter: str) -> Path:
    return REPORTER_PICS / f"{reporter}.jpeg"


def _get_model(model_name: str):
    global _model
    if _model is None:
        import whisper
        print(f"[pipeline2] loading Whisper model '{model_name}' ...")
        _model = whisper.load_model(model_name)
    return _model


def remove_source_video(video_path: Path) -> None:
    """Delete a source video once its clips are exported, so disk usage stays flat.

    The clips are self-contained mp4s, so the multi-hundred-MB source is dead weight
    after export. Retried once because Windows can still hold the handle briefly
    after cv2/scenedetect release it.
    """
    try:
        freed_mb = video_path.stat().st_size / (1024 * 1024)
    except OSError:
        return  # already gone

    for attempt in (1, 2):
        try:
            video_path.unlink()
            print(f"[pipeline2] deleted source video {video_path.name} "
                  f"(freed {freed_mb:.1f} MB)")
            return
        except OSError as exc:
            if attempt == 1:
                gc.collect()  # drop any lingering decoder handle
                continue
            print(f"[pipeline2] WARNING could not delete {video_path.name}: {exc}")


def select_clips(url, reporter, ref_encoding, downloads_dir: Path, clips_dir: Path,
                 scene_threshold=30.0, sample_count=5, tolerance=0.5, min_duration=1.2,
                 delete_source=True):
    """Download the video and export clips where only the reporter appears.

    With delete_source=True (the default) the downloaded source video is removed as
    soon as its clips are exported, so a full-CSV run never accumulates raw videos.
    Mirrors KEEP_P2_SOURCE_VIDEO=False in colab_run_all.ipynb.
    """
    video_path = download_video(url, downloads_dir)
    scenes = detect_scenes(video_path, threshold=scene_threshold)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    base = safe_name(video_path.stem)
    clips_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    try:
        for idx, (start_sec, end_sec) in enumerate(scenes, start=1):
            if end_sec - start_sec < min_duration:
                continue
            if not scene_has_only_target(cap, start_sec, end_sec, ref_encoding,
                                         sample_count, tolerance):
                continue
            clip = clips_dir / f"{reporter}_{base}_scene_{idx:04d}_{start_sec:.2f}-{end_sec:.2f}.mp4"
            export_clip(video_path, clip, start_sec, end_sec)
            rows.append({
                "reporter": reporter,
                "source_url": url,
                "source_video": video_path.name,
                # as_posix(): the manifest is the Phase 2 hand-off and is often written
                # on Windows but read on Colab/Linux, where a backslash path is not a
                # path separator at all.
                "clip_file": clip.relative_to(PROJECT).as_posix(),
                "start_sec": round(start_sec, 3),
                "end_sec": round(end_sec, 3),
                "duration_sec": round(end_sec - start_sec, 3),
            })
    finally:
        cap.release()

    if delete_source:
        remove_source_video(video_path)
    return rows


def transcribe_clip(clip_path: Path, out_json: Path, model_name="large-v3") -> str:
    """Transcribe one clip; write pipeline-1-style JSON; return the plain text."""
    model = _get_model(model_name)
    result = model.transcribe(str(clip_path), word_timestamps=True,
                              language="tr", verbose=False)

    data = {"video": clip_path.name, "segments": []}
    texts = []
    for seg in result.get("segments", []):
        seg_data = {
            "start_sec": round(seg["start"], 3),
            "end_sec": round(seg["end"], 3),
            "text": normalize_text(seg["text"]),
            "words": [],
        }
        texts.append(seg_data["text"])
        for w in seg.get("words", []):
            surface = normalize_text(w["word"])
            confidence = round(w.get("probability", 1.0), 6)
            seg_data["words"].append({
                "surface_form": surface,
                "lemma": surface,  # TODO: zeyrek, mirrors run_whisper.py
                "start_sec": round(w["start"], 3),
                "end_sec": round(w["end"], 3),
                "confidence": confidence,
                "manual_review": int(confidence < 0.7),
            })
        data["segments"].append(seg_data)

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return " ".join(texts).strip()
