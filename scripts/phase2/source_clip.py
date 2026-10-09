"""Phase 2 for ONE solo clip: decode -> landmarks -> mouth crops + SyncNet -> items.

`process(job)` is what the worker pool runs. It writes the clip's crop stack (.npz)
and returns plain dicts; it knows nothing about the manifests' columns or files,
which run_phase2 owns.

Storage: one .npz per SOURCE clip holding every 25 fps frame once, not one file per
word. Word windows overlap by +-15 frames, so per-word files would store each frame
~4x -- for ~1.5M words that is several hundred GB. Items are frame ranges into the
stack (`frame_start`, `frame_end` in the manifests).
"""
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

import landmarks
import mouth_roi
import segments
import syncnet
import video_io


@dataclass
class Job:
    stem: str
    clip_path: str
    transcript_path: str
    npz_path: str
    device: str
    sync_min: float
    thresholds: segments.Thresholds


def _count_words(transcript: dict) -> int:
    return sum(len(s.get("words", [])) for s in transcript.get("segments", []))


def track_clip(clip: Path, device: str):
    """Decode once -> (RoiTrack, SyncScorer holding the clip's visual features).

    The scorer is not finished: the caller decides which audio it is scored against.
    """
    width, height, scale = video_io.decode_size(clip)
    scorer = syncnet.SyncScorer(device)

    def on_face(cx, cy, half, frame):
        scorer.add(syncnet.face_crop(frame, cx, cy, half))

    stream = mouth_roi.RoiStream(scale, on_face)
    with landmarks.ClipLandmarker() as lm:
        for k, frame in enumerate(video_io.read_frames(clip, width, height)):
            stream.push(frame, lm.detect(frame, k))
    return stream.finish(), scorer


def process(job: Job) -> dict:
    t0 = time.time()
    cv2.setNumThreads(1)  # one clip per process; let the pool own the cores
    with open(job.transcript_path, encoding="utf-8") as f:
        transcript = json.load(f)
    result = {"stem": job.stem, "status": "", "words": [], "sentences": [],
              "discards": [], "sync": None, "track": {},
              "n_words": _count_words(transcript),
              "n_segments": len(transcript.get("segments", []))}
    if result["n_words"] == 0:
        result["status"] = "no_words"
        result["seconds"] = round(time.time() - t0, 1)
        return result

    clip = Path(job.clip_path)
    roi, scorer = track_clip(clip, job.device)
    sync: Optional[syncnet.SyncResult] = scorer.finish(video_io.read_audio(clip))

    status_counts = {name: int((roi.status == code).sum())
                     for code, name in mouth_roi.STATUS_NAMES.items()}
    result["track"] = {
        "n_frames": len(roi.status),
        "good_frames": int(roi.good.sum()),
        **status_counts,
        "mouth_width_px": (round(float(np.nanmedian(roi.mouth_width_px)), 1)
                           if np.isfinite(roi.mouth_width_px).any() else ""),
    }
    if sync is None:
        result["status"] = "unscorable"
        result["discards"].append({"level": "source_clip", "item_id": job.stem, "text": "",
                                   "reason": "too_short_for_sync", "detail": ""})
        result["seconds"] = round(time.time() - t0, 1)
        return result
    result["sync"] = {"syncnet_score": round(sync.conf, 4), "syncnet_offset": sync.offset,
                      "min_dist": round(sync.min_dist, 4)}
    if sync.conf < job.sync_min:
        result["status"] = "low_sync"
        result["discards"].append({"level": "source_clip", "item_id": job.stem, "text": "",
                                   "reason": "low_sync",
                                   "detail": f"syncnet_score={sync.conf:.3f}"})
        result["seconds"] = round(time.time() - t0, 1)
        return result

    track = segments.Track(good=roi.good, aperture=roi.aperture,
                           mouth_width_px=roi.mouth_width_px, yaw=roi.yaw, pitch=roi.pitch,
                           frame_conf=sync.frame_conf, offset=sync.offset)
    words, w_disc, sync_ok = segments.word_items(job.stem, transcript, track, job.thresholds)
    sents, s_disc = segments.sentence_items(job.stem, transcript, track, job.thresholds,
                                            sync_ok)
    result.update(words=words, sentences=sents, discards=w_disc + s_disc)
    if words or sents:
        npz = Path(job.npz_path)
        npz.parent.mkdir(parents=True, exist_ok=True)
        tmp = npz.with_name(npz.stem + ".tmp.npz")
        np.savez_compressed(tmp, frames=roi.crops, good=roi.good,
                            sync_conf=sync.frame_conf, fps=np.int16(video_io.FPS))
        tmp.replace(npz)
    result["status"] = "ok"
    result["seconds"] = round(time.time() - t0, 1)
    return result
