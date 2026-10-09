"""Decode a solo clip at the canonical 25 fps and read its audio (PHASE2_PLAN §3, §4.1).

Every Phase 2 timestamp is a frame index at FPS: frame k covers [k/FPS, (k+1)/FPS).
The clip is decoded once, through one ffmpeg pipe, the same way
presenter_filter.sample_frames reads video -- no intermediate mp4, no OpenCV seeks.
"""
import subprocess
from pathlib import Path
from typing import Iterator, Tuple

import cv2
import numpy as np

from presenter_filter import ANALYSIS_MAX_HEIGHT, ffmpeg_executable

FPS = 25
AUDIO_RATE = 16000  # SyncNet's MFCC front end is trained on 16 kHz mono


def frame_of(t_sec: float) -> int:
    """Frame index of a time in seconds at the canonical FPS."""
    return int(round(t_sec * FPS))


def decode_size(video_path: Path) -> Tuple[int, int, float]:
    """(width, height, scale) the frames are decoded at.

    Frames taller than ANALYSIS_MAX_HEIGHT are scaled down, like Phase 1's analysis;
    `scale` converts decoded pixels back to source pixels (source = decoded / scale).
    """
    cap = cv2.VideoCapture(str(video_path))
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if src_w <= 0 or src_h <= 0:
        raise RuntimeError(f"Cannot read frame size of {video_path}")
    scale = min(1.0, ANALYSIS_MAX_HEIGHT / src_h)
    w = max(2, int(round(src_w * scale / 2)) * 2)
    h = max(2, int(round(src_h * scale / 2)) * 2)
    return w, h, h / src_h


def read_frames(video_path: Path, width: int, height: int) -> Iterator[np.ndarray]:
    """Yield BGR frames at a constant FPS, sized (height, width)."""
    cmd = [ffmpeg_executable(), "-v", "error", "-nostdin", "-i", str(video_path),
           "-an", "-sn", "-vf", f"fps={FPS},scale={width}:{height}",
           "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    size = width * height * 3
    try:
        while True:
            buf = proc.stdout.read(size)
            if len(buf) < size:
                break
            yield np.frombuffer(buf, np.uint8).reshape(height, width, 3)
    finally:
        proc.stdout.close()
        proc.kill()
        proc.wait()


def read_audio(video_path: Path) -> np.ndarray:
    """Mono 16 kHz int16 samples of the clip's audio track (empty if it has none)."""
    cmd = [ffmpeg_executable(), "-v", "error", "-nostdin", "-i", str(video_path),
           "-vn", "-ac", "1", "-ar", str(AUDIO_RATE), "-f", "s16le", "-acodec", "pcm_s16le", "-"]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    return np.frombuffer(res.stdout, np.int16)
