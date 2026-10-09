"""Word and sentence items from a clip's transcript + per-frame track (PHASE2_PLAN §4.6, §4.7).

Pure functions: no I/O, no models. Every item is a frame range [frame_start,
frame_end) into the source clip's 25 fps crop stack. Whisper times are in the audio
timeline; SyncNet's offset says the mouth lags the audio by `offset` frames, so every
span is shifted by it before it is mapped to frames.
"""
import json
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from run_whisper import normalize_text
from video_io import FPS, frame_of

PAD_FRAMES = 15        # +-15 frames of coarticulation context around a word
SENT_MIN_SEC = 1.0
SENT_MAX_SEC = 8.0
# A pause this close to the middle of an over-long sentence is preferred over a
# slightly longer one at its edge: pieces should come out balanced.
SPLIT_CENTRE_ZONE = 0.5


@dataclass
class Thresholds:
    word_sync_min: Optional[float]     # None = record word_sync_conf, never drop on it
    lip_motion_min: Optional[float]    # None = record lip_motion, never drop on it
    sent_offscreen_max_frac: float     # sentence dropped when more of its words fail


@dataclass
class Track:
    """What segments needs from mouth_roi + syncnet, as plain arrays of length T."""
    good: np.ndarray
    aperture: np.ndarray
    mouth_width_px: np.ndarray
    yaw: np.ndarray
    pitch: np.ndarray
    frame_conf: np.ndarray
    offset: int

    @property
    def n(self) -> int:
        return len(self.good)


def _nanmean(a) -> float:
    a = np.asarray(a, np.float64)
    return float(np.nanmean(a)) if np.isfinite(a).any() else float("nan")


def _nanstd(a) -> float:
    a = np.asarray(a, np.float64)
    return float(np.nanstd(a)) if np.isfinite(a).any() else float("nan")


def _round(x: float, nd: int = 4):
    return "" if x is None or not np.isfinite(x) else round(float(x), nd)


def span_frames(track: Track, start_sec: float, end_sec: float) -> Optional[Tuple[int, int]]:
    """Audio-time span -> [f0, f1) video frames, shifted by the A/V offset, clamped."""
    f0 = frame_of(start_sec) + track.offset
    f1 = max(f0 + 1, frame_of(end_sec) + track.offset)
    f0, f1 = max(0, f0), min(track.n, f1)
    return (f0, f1) if f0 < f1 else None


def padded_window(track: Track, f0: int, f1: int) -> Tuple[int, int, bool]:
    """[f0, f1) grown by PAD_FRAMES each side, stopping at clip edges and track gaps.

    The third value is False when a gap (not a clip edge) cut the padding short.
    """
    want0, want1 = max(0, f0 - PAD_FRAMES), min(track.n, f1 + PAD_FRAMES)
    a = f0
    while a > want0 and track.good[a - 1]:
        a -= 1
    b = f1
    while b < want1 and track.good[b]:
        b += 1
    return a, b, (a == want0 and b == want1)


def _stats(track: Track, a: int, b: int) -> dict:
    return {
        "mouth_width_px": _round(np.nanmedian(track.mouth_width_px[a:b])
                                 if np.isfinite(track.mouth_width_px[a:b]).any() else np.nan, 1),
        "mean_yaw": _round(_nanmean(track.yaw[a:b]), 2),
        "mean_pitch": _round(_nanmean(track.pitch[a:b]), 2),
    }


def _discard(level, item_id, text, reason, detail=""):
    return {"level": level, "item_id": item_id, "text": text, "reason": reason,
            "detail": detail}


def word_items(stem: str, transcript: dict, track: Track, th: Thresholds):
    """-> (kept word dicts, discard dicts, {word_index: passes_sync})."""
    kept, discards, sync_ok = [], [], {}
    idx = -1
    for seg in transcript.get("segments", []):
        for w in seg.get("words", []):
            idx += 1
            item_id = f"{stem}_w{idx:04d}"
            surface = normalize_text(w.get("surface_form", ""))
            if not surface:
                discards.append(_discard("word", item_id, "", "empty_label"))
                continue
            span = span_frames(track, w["start_sec"], w["end_sec"])
            if span is None:
                discards.append(_discard("word", item_id, surface, "out_of_range"))
                continue
            f0, f1 = span
            if not track.good[f0:f1].all():
                discards.append(_discard("word", item_id, surface, "track_fail"))
                continue
            a, b, full_pad = padded_window(track, f0, f1)
            conf = _nanmean(track.frame_conf[a:b])
            motion = _nanstd(track.aperture[a:b])
            passes = not (th.word_sync_min is not None and np.isfinite(conf)
                          and conf < th.word_sync_min)
            sync_ok[idx] = passes
            if not passes:
                discards.append(_discard("word", item_id, surface, "offscreen_speech",
                                         f"word_sync_conf={conf:.3f}"))
                continue
            if th.lip_motion_min is not None and np.isfinite(motion) and motion < th.lip_motion_min:
                discards.append(_discard("word", item_id, surface, "offscreen_speech",
                                         f"lip_motion={motion:.4f}"))
                sync_ok[idx] = False
                continue
            kept.append({
                "clip_id": item_id,
                "surface_form": surface,
                "lemma": normalize_text(w.get("lemma", "") or surface),
                "word_start_sec": round(float(w["start_sec"]), 3),
                "word_end_sec": round(float(w["end_sec"]), 3),
                "frame_start": a, "frame_end": b,
                "clip_start_sec": round(a / FPS, 3), "clip_end_sec": round(b / FPS, 3),
                "n_frames": b - a,
                "confidence": w.get("confidence", ""),
                "manual_review": w.get("manual_review", ""),
                "word_sync_conf": _round(conf),
                "lip_motion": _round(motion),
                "track_ok": int(full_pad),
                **_stats(track, a, b),
            })
    return kept, discards, sync_ok


def _split(words: List[dict], start: float, end: float) -> List[Tuple[float, float, List[dict]]]:
    """Split [start, end] at pauses until every piece is <= SENT_MAX_SEC."""
    if end - start <= SENT_MAX_SEC or len(words) < 2:
        return [(start, end, words)]
    mid, half = (start + end) / 2, (end - start) / 2
    gaps = [(words[i + 1]["start_sec"] - words[i]["end_sec"], i) for i in range(len(words) - 1)]
    central = [g for g in gaps if abs(words[g[1]]["end_sec"] - mid) <= SPLIT_CENTRE_ZONE * half]
    if central:
        _, i = max(central)
    else:
        _, i = min(gaps, key=lambda g: abs(words[g[1]]["end_sec"] - mid))
    left, right = words[:i + 1], words[i + 1:]
    return (_split(left, start, left[-1]["end_sec"])
            + _split(right, right[0]["start_sec"], end))


def sentence_items(stem: str, transcript: dict, track: Track, th: Thresholds, sync_ok: dict):
    """-> (kept sentence dicts, discard dicts). `sync_ok` comes from word_items."""
    kept, discards = [], []
    widx = 0
    for s_idx, seg in enumerate(transcript.get("segments", [])):
        words = [dict(w, _idx=widx + i) for i, w in enumerate(seg.get("words", []))]
        widx += len(words)
        pieces = _split(words, seg["start_sec"], seg["end_sec"])
        for p_idx, (start, end, pw) in enumerate(pieces):
            split = p_idx if len(pieces) > 1 else ""
            item_id = f"{stem}_s{s_idx:03d}" + (f"_{p_idx}" if len(pieces) > 1 else "")
            text = (normalize_text(seg.get("text", "")) if len(pieces) == 1
                    else " ".join(normalize_text(w["surface_form"]) for w in pw))
            if not (SENT_MIN_SEC <= end - start <= SENT_MAX_SEC):
                discards.append(_discard("sentence", item_id, text, "duration",
                                         f"{end - start:.2f}s"))
                continue
            span = span_frames(track, start, end)
            if span is None:
                discards.append(_discard("sentence", item_id, text, "out_of_range"))
                continue
            a, b = span
            if not track.good[a:b].all():
                discards.append(_discard("sentence", item_id, text, "track_fail"))
                continue
            judged = [sync_ok[w["_idx"]] for w in pw if w["_idx"] in sync_ok]
            failed = sum(1 for ok in judged if not ok)
            if judged and failed / len(judged) > th.sent_offscreen_max_frac:
                discards.append(_discard("sentence", item_id, text, "offscreen_speech",
                                         f"{failed}/{len(judged)} words failed sync"))
                continue
            offset_sec = track.offset / FPS
            spans = [{"surface_form": normalize_text(w["surface_form"]),
                      "lemma": normalize_text(w.get("lemma", "") or w["surface_form"]),
                      "start_sec": round(w["start_sec"] + offset_sec - a / FPS, 3),
                      "end_sec": round(w["end_sec"] + offset_sec - a / FPS, 3)}
                     for w in pw]
            kept.append({
                "clip_id": item_id,
                "text": text,
                "sentence_start_sec": round(float(start), 3),
                "sentence_end_sec": round(float(end), 3),
                "split_index": split,
                "frame_start": a, "frame_end": b,
                "clip_start_sec": round(a / FPS, 3), "clip_end_sec": round(b / FPS, 3),
                "n_frames": b - a,
                "n_words": len(pw),
                "sentence_sync_conf": _round(_nanmean(track.frame_conf[a:b])),
                "lip_motion": _round(_nanstd(track.aperture[a:b])),
                "word_spans": json.dumps(spans, ensure_ascii=False),
                **_stats(track, a, b),
            })
    return kept, discards
