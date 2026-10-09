"""Face tracking and mouth-ROI extraction over a clip (PHASE2_PLAN §4.3, §4.4).

Each frame is classified (one presenter face / none / two / a jump / implausible
landmarks), then cropped twice from temporally smoothed boxes:

- an 88x88 grayscale mouth crop -- the dataset material;
- a 224x224 colour face crop handed to a callback (SyncNet).

Smoothing looks SMOOTH_LAG frames ahead, so frames are buffered that long before
their crops are made; a clip is never held in memory as full frames.
"""
from collections import deque
from dataclasses import dataclass
from typing import Callable, List, Optional

import cv2
import numpy as np
from mediapipe.tasks.python import vision

from presenter_filter import TRACK_MAX_SHIFT

LIP_IDX = sorted({i for c in vision.FaceLandmarksConnections.FACE_LANDMARKS_LIPS
                  for i in (c.start, c.end)})  # 40 outer + inner lip points
MOUTH_LEFT, MOUTH_RIGHT = 61, 291              # mouth corners
INNER_TOP, INNER_BOTTOM = 13, 14               # inner-lip midpoints

CROP_SIZE = 88
# The crop side is the lip box's longer side grown by MOUTH_MARGIN on each side,
# so the lips stay inside the crop when the mouth opens.
MOUTH_MARGIN = 0.4
MOUTH_SMOOTH = 2   # +-frames, moving average of mouth centre and size
FACE_SMOOTH = 6    # +-frames, median of face box (run_pipeline's kernel 13)
SMOOTH_LAG = max(MOUTH_SMOOTH, FACE_SMOOTH)

# Lips narrower/wider than this share of the face box are a landmark failure.
MOUTH_FACE_RATIO = (0.20, 0.75)
# A run of at most this many no-face frames between good ones is a detector blink,
# not a gap: the crop is held through it.
MAX_FILL_GAP = 2

GOOD, NO_FACE, MULTI_FACE, JUMP, BAD_LANDMARKS = range(5)
STATUS_NAMES = {GOOD: "good", NO_FACE: "no_face", MULTI_FACE: "multi_face",
                JUMP: "jump", BAD_LANDMARKS: "bad_landmarks"}


@dataclass
class FrameObs:
    status: int
    mouth_cx: float = np.nan
    mouth_cy: float = np.nan
    mouth_side: float = np.nan   # longer side of the tight lip box (decoded px)
    mouth_width: float = np.nan  # corner-to-corner (decoded px)
    aperture: float = np.nan     # inner-lip opening / mouth width
    face_cx: float = np.nan
    face_cy: float = np.nan
    face_half: float = np.nan    # half the face box's longer side
    yaw: float = np.nan
    pitch: float = np.nan


@dataclass
class RoiTrack:
    """Per-frame results of one clip; every array has length T."""
    crops: np.ndarray        # (T, 88, 88) uint8
    status: np.ndarray       # (T,) int8, STATUS codes before gap filling
    good: np.ndarray         # (T,) bool, after filling detector blinks
    mouth_width_px: np.ndarray  # (T,) float32, SOURCE pixels
    aperture: np.ndarray     # (T,) float32
    yaw: np.ndarray
    pitch: np.ndarray


def observe(faces, n_on_set: int, ref: Optional[FrameObs]) -> FrameObs:
    """Classify one frame (a landmarks.FrameFaces) given its head count.

    `ref` is the last single-face frame, for track continuity.
    """
    if n_on_set == 0 or faces.face is None:
        return FrameObs(NO_FACE)
    face = faces.face
    pts = face.points
    lips = pts[LIP_IDX]
    lx0, ly0 = lips.min(axis=0)
    lx1, ly1 = lips.max(axis=0)
    fx0, fy0, fx1, fy1 = face.box
    width = float(np.linalg.norm(pts[MOUTH_RIGHT] - pts[MOUTH_LEFT]))
    obs = FrameObs(
        GOOD,
        mouth_cx=float((lx0 + lx1) / 2), mouth_cy=float((ly0 + ly1) / 2),
        mouth_side=float(max(lx1 - lx0, ly1 - ly0)), mouth_width=width,
        aperture=float(np.linalg.norm(pts[INNER_BOTTOM] - pts[INNER_TOP]) / max(width, 1e-6)),
        face_cx=(fx0 + fx1) / 2, face_cy=(fy0 + fy1) / 2,
        face_half=max(fx1 - fx0, fy1 - fy0) / 2, yaw=face.yaw, pitch=face.pitch,
    )
    if n_on_set > 1:
        obs.status = MULTI_FACE
    elif not (MOUTH_FACE_RATIO[0] <= width / max(fx1 - fx0, 1e-6) <= MOUTH_FACE_RATIO[1]):
        obs.status = BAD_LANDMARKS
    elif ref is not None and (np.hypot(obs.mouth_cx - ref.mouth_cx, obs.mouth_cy - ref.mouth_cy)
                              > TRACK_MAX_SHIFT * 2 * ref.face_half):
        obs.status = JUMP
    return obs


def fill_blinks(status: np.ndarray) -> np.ndarray:
    """Good mask, with short no-face runs between good frames counted as good."""
    good = status == GOOD
    t, n = 0, len(status)
    while t < n:
        if status[t] != NO_FACE:
            t += 1
            continue
        end = t
        while end < n and status[end] == NO_FACE:
            end += 1
        if 0 < t and end < n and good[t - 1] and good[end] and end - t <= MAX_FILL_GAP:
            good[t:end] = True
        t = end
    return good


def _window(obs: List[FrameObs], k: int, radius: int) -> List[FrameObs]:
    lo, hi = max(0, k - radius), min(len(obs), k + radius + 1)
    return [o for o in obs[lo:hi] if o.status == GOOD]


def mouth_crop(gray: np.ndarray, cx: float, cy: float, side: float) -> np.ndarray:
    s = CROP_SIZE / side
    m = np.float32([[s, 0, CROP_SIZE / 2 - cx * s], [0, s, CROP_SIZE / 2 - cy * s]])
    return cv2.warpAffine(gray, m, (CROP_SIZE, CROP_SIZE), flags=cv2.INTER_AREA
                          if s < 1 else cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


class RoiStream:
    """push() frames with their landmarks in order, then finish().

    `on_face_crop(cx, cy, half, frame_bgr)` is called once per frame, in order,
    with the smoothed face box -- the SyncNet hook.
    """

    def __init__(self, scale: float, on_face_crop: Optional[Callable] = None):
        self.scale = scale  # decoded px = source px * scale
        self.on_face_crop = on_face_crop
        self.obs: List[FrameObs] = []
        self._frames = deque()
        self._crops: List[np.ndarray] = []
        self._ref: Optional[FrameObs] = None
        self._first_good: Optional[FrameObs] = None
        self._samples: List[tuple] = []  # (frame index, head count) where one was taken
        self._n_on_set = 1
        self._last_mouth = None
        self._last_face = None

    def push(self, frame_bgr: np.ndarray, faces) -> None:
        if faces.n_on_set is not None:
            self._n_on_set = faces.n_on_set
            self._samples.append((len(self.obs), faces.n_on_set))
        o = observe(faces, self._n_on_set, self._ref)
        if o.status in (GOOD, JUMP):
            self._ref = o
        if o.status == GOOD and self._first_good is None:
            self._first_good = o
        self.obs.append(o)
        self._frames.append(frame_bgr)
        if len(self._frames) > SMOOTH_LAG:
            self._emit(len(self.obs) - 1 - SMOOTH_LAG)

    def _emit(self, k: int) -> None:
        frame = self._frames.popleft()
        near = _window(self.obs, k, MOUTH_SMOOTH)
        if near:
            self._last_mouth = (np.mean([o.mouth_cx for o in near]),
                                np.mean([o.mouth_cy for o in near]),
                                np.mean([o.mouth_side for o in near]) * (1 + 2 * MOUTH_MARGIN))
        near = _window(self.obs, k, FACE_SMOOTH)
        if near:
            self._last_face = (np.median([o.face_cx for o in near]),
                               np.median([o.face_cy for o in near]),
                               np.median([o.face_half for o in near]))
        # Before the first good frame there is nothing to hold; borrow the next one.
        mouth = self._last_mouth or self._first_good_mouth()
        face = self._last_face or self._first_good_face()
        if mouth is None:
            self._crops.append(np.zeros((CROP_SIZE, CROP_SIZE), np.uint8))
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            self._crops.append(mouth_crop(gray, *mouth))
        if self.on_face_crop is not None:
            h, w = frame.shape[:2]
            self.on_face_crop(*(face or (w / 2, h / 2, h / 4)), frame)

    def _first_good_mouth(self):
        o = self._first_good
        return None if o is None else (o.mouth_cx, o.mouth_cy,
                                       o.mouth_side * (1 + 2 * MOUTH_MARGIN))

    def _first_good_face(self):
        o = self._first_good
        return None if o is None else (o.face_cx, o.face_cy, o.face_half)

    def _spread_head_counts(self, status: np.ndarray) -> None:
        """A second person seen at a sample is assumed present since the previous one.

        Counting runs every few frames (landmarks.DETECT_EVERY); push() can only carry
        a count forward, so frames just BEFORE a multi-face sample are fixed here.
        """
        prev = 0
        for idx, n in self._samples:
            if n > 1:
                seg = status[prev:idx]
                seg[seg == GOOD] = MULTI_FACE
            prev = idx + 1

    def finish(self) -> RoiTrack:
        while self._frames:
            self._emit(len(self.obs) - len(self._frames))
        status = np.array([o.status for o in self.obs], np.int8)
        self._spread_head_counts(status)
        f32 = lambda attr: np.array([getattr(o, attr) for o in self.obs], np.float32)  # noqa: E731
        crops = (np.stack(self._crops) if self._crops
                 else np.zeros((0, CROP_SIZE, CROP_SIZE), np.uint8))
        return RoiTrack(crops=crops, status=status, good=fill_blinks(status),
                        mouth_width_px=f32("mouth_width") / self.scale,
                        aperture=f32("aperture"), yaw=f32("yaw"), pitch=f32("pitch"))
