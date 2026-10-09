"""Audio-visual sync scoring with SyncNet (PHASE2_PLAN §4.5, §4.5.1).

Same maths as syncnet_python's SyncNetInstance.evaluate -- 5-frame 224x224 face
windows vs 20-step MFCC windows, distances over +-VSHIFT frame offsets, clip
confidence = median - min of the mean distance curve, framewise confidence at the
best offset median-filtered with kernel 9 -- but streaming: face crops go to the
GPU in batches as the clip is decoded, so a 470 s clip never holds 12k crops in RAM
and nothing is written to disk.

Verified against the reference demo (data/example.avi -> offset 3, conf ~10.0) by
`python scripts/phase2/syncnet.py --selftest <example.avi>`.
"""
import argparse
import math
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import python_speech_features
import torch
from scipy import signal

if __name__ == "__main__":  # run directly for the self-test: make scripts/ importable
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from syncnet_model import S
from video_io import AUDIO_RATE, FPS

MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "syncnet_v2.model"
MODEL_URL = "http://www.robots.ox.ac.uk/~vgg/software/lipsync/data/syncnet_v2.model"

FACE_SIZE = 224        # SyncNet's visual input
CROP_SCALE = 0.40      # run_pipeline.py --crop_scale default
VSHIFT = 15            # run_syncnet.py --vshift default: +-0.6 s searched
WINDOW = 5             # video frames per SyncNet window
MFCC_PER_FRAME = 4     # 100 MFCC steps/s vs 25 fps
BATCH = 64
FRAME_CONF_KERNEL = 9  # medfilt of the framewise confidence, as in evaluate()
PAD_VALUE = 110        # run_pipeline.py pads frames with grey 110


@dataclass
class SyncResult:
    offset: int              # frames; the mouth lags the audio by this much
    conf: float              # clip score: median(mean dist) - min(mean dist)
    min_dist: float
    frame_conf: np.ndarray   # (T,) float32 per frame, NaN where no window covers it


def ensure_model() -> Path:
    if not MODEL_PATH.exists():
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading SyncNet weights -> {MODEL_PATH}")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    return MODEL_PATH


_model = None


def load_model(device: str) -> S:
    """Per-process SyncNet, loaded once."""
    global _model
    if _model is None:
        net = S(num_layers_in_fc_layers=1024)
        state = torch.load(ensure_model(), map_location="cpu", weights_only=True)
        net.load_state_dict(state)
        _model = net.to(device).eval()
    return _model


def face_crop(frame_bgr: np.ndarray, cx: float, cy: float, half: float) -> np.ndarray:
    """run_pipeline.crop_video's crop: square, shifted toward the lower face, 224x224.

    `half` is half the face box's longer side. The crop spans x in cx +- half*(1+cs)
    and y in [cy - half, cy + half*(1+2cs)], padded with grey where it leaves the frame.
    """
    side = 2 * half * (1 + CROP_SCALE)
    x0, y0 = cx - half * (1 + CROP_SCALE), cy - half
    s = FACE_SIZE / side
    m = np.float32([[s, 0, -x0 * s], [0, s, -y0 * s]])
    return cv2.warpAffine(frame_bgr, m, (FACE_SIZE, FACE_SIZE), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=(PAD_VALUE,) * 3)


class SyncScorer:
    """Feed 224x224 BGR face crops in frame order, then `finish(audio)`."""

    def __init__(self, device: str):
        self.device = device
        self.net = load_model(device)
        self._pending: List[np.ndarray] = []
        self._feats: List[torch.Tensor] = []
        self.n_frames = 0

    def add(self, crop_bgr: np.ndarray) -> None:
        self._pending.append(crop_bgr)
        self.n_frames += 1
        if len(self._pending) >= BATCH + WINDOW - 1:
            self._flush(BATCH)

    @torch.no_grad()
    def _flush(self, n_windows: int) -> None:
        if n_windows <= 0:
            return
        x = np.stack(self._pending[:n_windows + WINDOW - 1])            # (F, H, W, 3)
        t = torch.from_numpy(x).to(self.device).float().permute(3, 0, 1, 2)  # (3, F, H, W)
        win = t.unfold(1, WINDOW, 1).permute(1, 0, 4, 2, 3).contiguous()   # (n, 3, 5, H, W)
        self._feats.append(self.net.forward_lip(win).cpu())
        self._pending = self._pending[n_windows:]

    @torch.no_grad()
    def finish(self, audio: np.ndarray) -> Optional[SyncResult]:
        """None when the clip is too short (or silent) to score.

        The visual features are kept, so this can be called again with other audio
        (calibrate_offscreen.py scores one decode against real and spliced audio).
        """
        self._flush(len(self._pending) - (WINDOW - 1))
        min_length = min(self.n_frames, math.floor(len(audio) / (AUDIO_RATE / FPS)))
        n = min_length - WINDOW  # evaluate() stops one window short; keep that
        if n <= 2 * VSHIFT or not self._feats:
            return None
        im_feat = torch.cat(self._feats, 0)[:n]

        mfcc = python_speech_features.mfcc(audio, AUDIO_RATE).T          # (13, steps)
        cct = torch.from_numpy(mfcc.astype(np.float32))[None, None]      # (1, 1, 13, steps)
        cc_feat = []
        for i in range(0, n, BATCH):
            idx = range(i, min(n, i + BATCH))
            batch = torch.cat([cct[:, :, :, v * MFCC_PER_FRAME:v * MFCC_PER_FRAME + 20]
                               for v in idx], 0)
            cc_feat.append(self.net.forward_aud(batch.to(self.device)).cpu())
        cc_feat = torch.cat(cc_feat, 0)

        dists = _offset_distances(im_feat, cc_feat)                       # (n, 2V+1)
        mdist = dists.mean(0)
        minval, minidx = torch.min(mdist, 0)
        med = torch.median(mdist)
        fconf = (med - dists[:, minidx]).numpy()
        fconfm = signal.medfilt(fconf, kernel_size=FRAME_CONF_KERNEL)

        # Window i spans frames i..i+4; credit its confidence to the centre frame.
        frame_conf = np.full(self.n_frames, np.nan, np.float32)
        frame_conf[WINDOW // 2:WINDOW // 2 + n] = fconfm
        return SyncResult(offset=int(VSHIFT - minidx), conf=float(med - minval),
                          min_dist=float(minval), frame_conf=frame_conf)


def _offset_distances(feat1: torch.Tensor, feat2: torch.Tensor) -> torch.Tensor:
    """calc_pdist, vectorised: row i = distances of video i to audio i-V..i+V."""
    win = 2 * VSHIFT + 1
    feat2p = torch.nn.functional.pad(feat2, (0, 0, VSHIFT, VSHIFT))
    windows = feat2p.unfold(0, win, 1)                                    # (n, D, win)
    # pairwise_distance adds eps=1e-6 to the difference; match it.
    return torch.linalg.vector_norm(feat1[:, :, None] - windows + 1e-6, dim=1)


def _selftest(video: Path, device: str) -> None:
    """Score an already-cropped face video (the syncnet_python demo input) as-is."""
    from video_io import read_audio
    cap = cv2.VideoCapture(str(video))
    scorer = SyncScorer(device)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        scorer.add(cv2.resize(frame, (FACE_SIZE, FACE_SIZE)))
    cap.release()
    res = scorer.finish(read_audio(video))
    print(f"AV offset: {res.offset}  Min dist: {res.min_dist:.3f}  Confidence: {res.conf:.3f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="SyncNet self-test on a pre-cropped face video.")
    p.add_argument("--selftest", type=Path, required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    _selftest(args.selftest, args.device)
