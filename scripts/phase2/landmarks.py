"""Face landmarks for a clip's frames (PHASE2_PLAN §4.2).

Two detectors, each doing what it is good at:

- Phase 1's YuNet (presenter_filter.detect_faces + is_on_set) counts the people on
  set and finds the presenter, with exactly Phase 1's score and size floors.
- MediaPipe's Face Landmarker (Tasks API, VIDEO mode) runs on a square crop around
  that face. Its built-in detector is short-range: on full studio frames, where a
  presenter's face is ~130 px of 1080, it finds nothing at all. On the crop the face
  fills the image.

YuNet costs ~23 ms a frame -- more than everything else together -- so it runs every
DETECT_EVERY frames, Phase 1's own 0.2 s cadence; in between, the crop follows the
previous frame's landmarks. mouth_roi spreads each sample's head count over the
frames around it.
"""
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision

from presenter_filter import detect_faces, is_on_set
from video_io import FPS

MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "face_landmarker.task"
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
             "face_landmarker/float16/latest/face_landmarker.task")
CROP_FACTOR = 2.0  # landmarker input = this many face-box sides around the face
CROP_PX = 384      # ...resampled to this square
DETECT_EVERY = 5   # YuNet head count every 5 frames = 0.2 s at 25 fps


@dataclass
class Face:
    points: np.ndarray  # (478, 2) float32, decoded-frame pixels
    yaw: float          # degrees, 0 = facing the camera
    pitch: float        # degrees

    @property
    def box(self):
        """(x0, y0, x1, y1) of the whole landmark mesh."""
        x0, y0 = self.points.min(axis=0)
        x1, y1 = self.points.max(axis=0)
        return float(x0), float(y0), float(x1), float(y1)


@dataclass
class FrameFaces:
    n_on_set: Optional[int]  # people on set, counted like Phase 1; None = not counted here
    face: Optional[Face]     # landmarks of the presenter; None if none or MediaPipe failed


def ensure_model() -> Path:
    if not MODEL_PATH.exists():
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading MediaPipe face landmarker -> {MODEL_PATH}")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    return MODEL_PATH


def _pose(matrix) -> tuple:
    """(yaw, pitch) in degrees from the landmarker's facial transformation matrix."""
    rot = np.asarray(matrix, dtype=np.float64)[:3, :3]
    rot = rot / np.linalg.norm(rot, axis=0, keepdims=True)  # drop any scale
    pitch, yaw, _roll = cv2.RQDecomp3x3(rot)[0]
    return float(yaw), float(pitch)


class ClipLandmarker:
    """One landmarker per clip: VIDEO mode needs timestamps that only increase."""

    def __init__(self):
        options = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(ensure_model())),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            output_facial_transformation_matrixes=True,
        )
        self._lm = vision.FaceLandmarker.create_from_options(options)
        self._box = None  # (x, y, w, h) the next crop is centred on

    def detect(self, frame_bgr: np.ndarray, frame_idx: int) -> FrameFaces:
        n = None
        if frame_idx % DETECT_EVERY == 0 or self._box is None:
            h = frame_bgr.shape[0]
            on_set = [f for f in detect_faces(frame_bgr) if is_on_set(f, h)]
            n = len(on_set)
            if not on_set:
                self._box = None
                return FrameFaces(0, None)
            self._box = max(on_set, key=lambda f: f[3] * f[4])[1:]
        face = self._landmarks(frame_bgr, frame_idx)
        if face is not None:
            x0, y0, x1, y1 = face.box
            self._box = (x0, y0, x1 - x0, y1 - y0)
        return FrameFaces(n, face)

    def _landmarks(self, frame_bgr: np.ndarray, frame_idx: int) -> Optional[Face]:
        x, y, w, bh = self._box
        side = CROP_FACTOR * max(w, bh)
        x0, y0 = x + w / 2 - side / 2, y + bh / 2 - side / 2
        s = CROP_PX / side
        m = np.float32([[s, 0, -x0 * s], [0, s, -y0 * s]])
        crop = cv2.warpAffine(frame_bgr, m, (CROP_PX, CROP_PX), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT)
        rgb = np.ascontiguousarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        res = self._lm.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb),
                                        int(frame_idx * 1000 / FPS))
        if not res.face_landmarks:
            return None
        pts = np.array([(p.x * CROP_PX / s + x0, p.y * CROP_PX / s + y0)
                        for p in res.face_landmarks[0]], dtype=np.float32)
        yaw, pitch = (_pose(res.facial_transformation_matrixes[0])
                      if res.facial_transformation_matrixes else (float("nan"),) * 2)
        return Face(pts, yaw, pitch)

    def close(self):
        self._lm.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
