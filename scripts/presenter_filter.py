import argparse
import csv
import math
import os
import re
import shutil
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import cv2
import numpy as np
from scenedetect import SceneManager, open_video
from scenedetect.detectors import ContentDetector

try:
    import face_recognition
except Exception as exc:
    raise RuntimeError(
        "face_recognition import edilemedi. "
        "Windows tarafinda Python 3.11/3.12 + Visual Studio C++ Build Tools gerekli."
    ) from exc

if sys.version_info >= (3, 13):
    raise RuntimeError(
        "Python surumu cok yeni: "
        f"{sys.version.split()[0]}. "
        "face_recognition/dlib bu surumde genelde wheel sunmuyor. "
        "Python 3.11 veya 3.12 kullanin."
    )


# >=720p is a HARD requirement: below that the mouth ROI is too small to survive the
# 88x88 lip crop. A video that caps at 480p fails with "Requested format is not
# available"; run_all.run_pipeline2 catches that, logs it, and leaves the row pending
# rather than aborting the run. Append "/bestvideo+bestaudio/best" to accept lower
# resolutions instead.
#
# Capped at 1080p when the video has it. Nothing downstream uses more: analysis is
# scaled to ANALYSIS_MAX_HEIGHT and the lip crop is 88x88. YouTube serves this
# machine at ~300-400 KB/s per video whatever the line speed, and a 4K stream is ~4x
# the bytes (a 10-minute cuneyt_ozdemir video: 619 MB at 2160p, 153 MB at 1080p), so
# uncapped "best" spent most of the run downloading pixels that were thrown away.
# Sources that only exist above 1080p still download, via the uncapped fallback.
DOWNLOAD_FORMAT = ("bestvideo[height>=720][height<=1080]+bestaudio"
                   "/best[height>=720][height<=1080]"
                   "/bestvideo[height>=720]+bestaudio/best[height>=720]")

# Every verdict is taken on sampled frames read in a single sequential ffmpeg pass
# (see sample_frames). This used to be one frame per 30 s, each certifying 15 s
# either side of itself, and an audit of the first 8063 clips found that that is
# exactly where they leaked: a split screen, a presenter standing beside a video
# wall of three politicians, or a sign-language interpreter box that the one sample
# happened to miss, and the whole clip was labelled "solo".
#
# Faces are COUNTED, and yaw is measured, every SECONDS_PER_SAMPLE (0.2 s): YuNet
# costs ~10 ms a frame and the 68-point landmark fit behind face_yaw ~1 ms. The dlib
# identity encoding is the expensive part -- ~200 ms per face on this machine -- so
# it runs once per IDENTITY_EVERY samples and additionally on the first sample of
# every new track (see analyze_frames), so a different person cut in for half a
# second still gets an identity check of its own. That case is real: a hadi_ozisik
# clip ended in 0.4 s of another man because the scene cut fell on a glitch
# transition, not on the change of shot. Identity is voted per track (see
# TRACK_MAX_SHIFT), so one encoding every 2 s is plenty.
SECONDS_PER_SAMPLE = 0.2
IDENTITY_EVERY = 10

# Trimmed off both ends of every scene. Cuts land on transitions -- hadi_ozisik's
# glitch wipes corrupt the first 3-5 frames of each shot, dissolves blend two shots
# -- and those frames carry no clean mouth. 0.3 s covers every transition measured.
SCENE_EDGE_TRIM = 0.3

# Shortest clip worth exporting. Phase 2 cuts a word clip as its Whisper timestamp
# +-0.6 s (the LRW-style 29-frame window) and keeps sentences of 1-8 s, so a clip
# under ~2 s rarely contains even one word with its full context. 569 of the first
# 8063 clips were under 2 s, 401 of them hadi_ozisik's jump cuts.
MIN_CLIP_DURATION = 2.0

# Frames are analysed at no more than this height. fatih_altayli uploads 2160p, and a
# 4K frame costs 4x the decode and pipe bandwidth for a face that is already ~500 px
# tall at 1080. Heights reported back (face_height_px) are rescaled to the SOURCE
# resolution, so they stay comparable with the rows written before this existed.
ANALYSIS_MAX_HEIGHT = 1080

# Face COUNTING uses OpenCV's YuNet detector, not dlib HOG. HOG on a half-scale frame
# -- which is what the 30 s grid could afford -- does not see a face under ~120 px:
# measured on kubra_par's studio shots it found her 137 px face in 0 of 6 frames, and
# on a Serdar Cebe frame it found him but not the three ~90 px politicians on the
# video wall beside him, so the frame passed as "exactly one face". YuNet finds all
# of them at ~10 ms a frame on a 960 px copy, which is what makes per-second sampling
# affordable. dlib is still used for identity and yaw, on a box derived from YuNet's
# (see _dlib_box): measured on 24 frames the distances and yaws agree with a native
# HOG box to within ~0.01.
YUNET_MODEL = Path(__file__).resolve().parent.parent / "models" / "face_detection_yunet_2023mar.onnx"
YUNET_URL = ("https://github.com/opencv/opencv_zoo/raw/main/models/"
             "face_detection_yunet/face_detection_yunet_2023mar.onnx")
DETECT_WIDTH = 960
# Detections below this score are not even recorded. The counting threshold is
# FACE_SCORE_MIN; this one only keeps the per-frame record small.
YUNET_MIN_SCORE = 0.5

# A detection counts as a PERSON ON SET -- and so disqualifies the frame unless it
# is the only one -- when its score is at least FACE_SCORE_MIN and it is at least
# MIN_FACE_FRACTION of the frame height. Both were set by reviewing detections on
# the first 8063 clips (scripts/audit_clips.py).
#
# The score floor removes YuNet's false positives, which on this corpus are
# furniture and decor: a tree trunk (0.69), a desk figurine (0.60), a bust on a
# bookshelf (0.61). Real faces, including webcam guests in a split screen and an
# interpreter in a corner box, score 0.85+.
#
# The size floor is what separates "another person is on screen" from "a photo on
# the shelf behind him". It is deliberately LOW: a sign-language interpreter box is
# ~6% of frame height and a four-way split-screen guest ~9%, and both are the
# violations this gate exists for.
FACE_SCORE_MIN = 0.80
MIN_FACE_FRACTION = 0.045

# Identity is voted per TRACK, not per frame. The reference photos match their own
# presenter at distances that wander 0.45-0.60 frame to frame (lighting, expression,
# a studio at 137 px), so a per-second test against tolerance 0.5 would shred a
# clean monologue into 1-2 s fragments. A track is a run of consecutive samples
# with exactly one person on set whose face centre moves by less than
# TRACK_MAX_SHIFT face-widths between samples; it is the reporter when its MEDIAN
# distance is within tolerance. A different person cut in mid-scene lands somewhere
# else in the frame, so it starts a new track and is voted on by itself. There is
# deliberately no per-sample ceiling: on hadi_ozisik every sample reviewed at
# distance 0.65-0.83 was the presenter himself (profile, sunglasses, outdoor light)
# or a detector false positive -- a ceiling only discarded good tracks.
TRACK_MAX_SHIFT = 0.5

# Median track distance at which a track is the reporter. 0.6 is dlib's own
# same-person threshold. It used to be 0.5, which only worked because the old filter
# looked at ONE frame per 30 s and a borderline frame sometimes fell under it: the
# reference photos are not studio stills, and kubra_par's clean, frontal, single-face
# tracks have a median of 0.50-0.60 (a 40 s clip at 0.5165 was dropped whole at 0.5).
# Reviewing every track with a median of 0.58-0.78 on fatih_portakal and kubra_par
# found the presenter in all of them; other people alone on screen (field reporters,
# interviewees) sit above 0.7.
IDENTITY_TOLERANCE = 0.6

# Maximum lateral head turn for a frame to count as "presenter faces the camera
# directly", which the project plan requires of every clip. face_yaw returns 0 for
# a face pointed at the lens and 1 for a full profile. Calibrated on this corpus:
#   hand-picked frontal reference photos      median 0.064, p90 0.152
#   clips that read as frontal on review      <= 0.20
#   clips that read as clearly turned away    >= 0.32
#   talking to an off-camera guest, profile    0.61 - 0.83
# 0.30 sits in that gap: twice the frontal p90, under anything that looked turned.
# A 90-degree profile has no readable mouth at all, so those clips are worse than
# useless -- they are mislabelled training data.
#
# YAW ONLY. An anchor glancing DOWN at a script is pitch, which this does not
# measure; nor does it police graphic overlays or a face too small for the 88x88
# lip crop (MIN_FACE_FRACTION is a false-positive floor, not a quality gate).
MAX_FACE_YAW = 0.30

# Wall-clock ceilings for the two steps that can block forever. Neither raises on
# its own: yt-dlp waits on the network, and OpenCV spins in C code no Python
# exception can reach, so a single bad video hangs an unattended 4691-row run.
# 4 h, not 1: a 3 GB 1080p60 broadcast on a ~850 KB/s link needs well over an hour,
# and the 1 h ceiling this started at killed one such download mid-transfer. The
# timeout only has to catch a download that will never end (a livestream), so it can
# afford to be generous.
DOWNLOAD_TIMEOUT = 14400.0
PROBE_TIMEOUT = 30.0

# Age-restricted videos fail with "Sign in to confirm your age" unless yt-dlp can
# present a logged-in YouTube session. Two sources, in this order:
#   1. cookies.txt at the repo root            -> --cookies
#   2. $YTDLP_COOKIES_FROM_BROWSER (e.g. "edge") -> --cookies-from-browser
# The file is preferred because live browser extraction is unreliable on Windows:
# Chrome 127+ encrypts its cookie DB with App-Bound Encryption (yt-dlp #10927) and
# any running Chromium holds a lock on it (#7271). Neither works on this machine.
COOKIES_FILE = Path(__file__).resolve().parent.parent / "cookies.txt"
COOKIES_FROM_BROWSER = os.environ.get("YTDLP_COOKIES_FROM_BROWSER", "").strip()

# yt-dlp failures that mean "the cookie source itself is broken", not "this video
# is unavailable". Matched against stderr so a dead cookie source can be dropped.
# "no longer valid" fires when the exported cookies got rotated (YouTube does this
# within minutes if the source browser stays signed in); "confirm you're not a bot"
# is what a stale/rotated cookie gets treated as — worse than sending no cookies at
# all, so both must drop the source instead of poisoning every remaining row.
COOKIE_ERROR_RE = re.compile(
    "cookie database|DPAPI|cookies-from-browser|could not (copy|find|decrypt)"
    "|no longer valid|confirm (you.re|you are) not a bot",
    re.IGNORECASE,
)

# yt-dlp writes one file per stream before merging (*.f136.mp4, *.f251.webm) and
# leaves them behind when the merge fails. They are never the final output, and the
# audio-only one is the exact decoder trap PROBE_TIMEOUT exists for, so skip them.
FORMAT_FRAGMENT_RE = re.compile(r"\.f\d+$")


def read_links(links_file: Path) -> List[str]:
    if not links_file.exists():
        raise FileNotFoundError(f"Link file not found: {links_file}")

    urls: List[str] = []
    for raw_line in links_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        urls.append(line)
    return urls


def run_cmd(cmd: List[str]) -> None:
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def yt_dlp_base_cmd() -> List[str]:
    return [sys.executable, "-m", "yt_dlp"]


# Flipped to True the first time a cookie source proves unusable; see download_video.
_cookies_disabled = False


def cookie_args() -> List[str]:
    """yt-dlp flags for a logged-in YouTube session, or [] when none is configured."""
    if _cookies_disabled:
        return []
    if COOKIES_FILE.exists():
        return ["--cookies", str(COOKIES_FILE)]
    if COOKIES_FROM_BROWSER:
        return ["--cookies-from-browser", COOKIES_FROM_BROWSER]
    return []


def ffmpeg_executable() -> str:
    path = shutil.which("ffmpeg")
    if path:
        return path

    try:
        from imageio_ffmpeg import get_ffmpeg_exe  # type: ignore

        return get_ffmpeg_exe()
    except Exception as exc:
        raise RuntimeError(
            "ffmpeg executable not found. Install FFmpeg and add it to PATH, or run: "
            f"{sys.executable} -m pip install imageio-ffmpeg"
        ) from exc


def extract_video_id(url: str) -> Optional[str]:
    patterns = [
        r"youtu\.be/([A-Za-z0-9_-]{11})",
        r"[?&]v=([A-Za-z0-9_-]{11})",
        r"/shorts/([A-Za-z0-9_-]{11})",
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


# Semicolons, not newlines: this has to survive as one `-c` argument.
_PROBE_SRC = (
    "import sys, cv2; "
    "cap = cv2.VideoCapture(sys.argv[1]); "
    "ok = cap.isOpened() and cap.read()[0]; "
    "cap.release(); "
    "sys.exit(0 if ok else 1)"
)


def is_video_readable(video_path: Path) -> bool:
    """True if OpenCV can actually decode a frame - probed in a killable child.

    An audio-only leftover from a failed merge (*.f251.webm) reports isOpened() ==
    True, and cap.read() then spins forever inside FFmpeg looking for a video packet
    that does not exist. That is a C loop: no timeout, thread or signal reaches it,
    and one such file froze a 284-row run for nine hours at 100% CPU. Give the probe
    its own process so it can be killed.
    """
    try:
        result = subprocess.run(
            [sys.executable, "-c", _PROBE_SRC, str(video_path)],
            capture_output=True, timeout=PROBE_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        print(f"Unreadable (decode probe timed out after {PROBE_TIMEOUT:.0f}s): "
              f"{video_path.name}")
        return False
    return result.returncode == 0


def choose_preferred_video(candidates: Iterable[Path]) -> Optional[Path]:
    # Prefer common video containers; skip transient partial files and the
    # per-stream fragments yt-dlp leaves behind when a merge fails.
    video_ext_order = {".mp4": 0, ".mkv": 1, ".webm": 2, ".mov": 3}
    ordered = sorted(
        {p.resolve() for p in candidates
         if p and p.exists() and p.suffix.lower() != ".part"
         and not FORMAT_FRAGMENT_RE.search(p.stem)},
        key=lambda p: (video_ext_order.get(p.suffix.lower(), 99), -p.stat().st_mtime),
    )
    for path in ordered:
        if is_video_readable(path):
            return path
    return None


def resolve_downloaded_video_path(url: str, downloads_dir: Path, stdout_lines: List[str]) -> Optional[Path]:
    prefixes = ("after_move:", "before_dl:", "filename:")
    candidates: List[Path] = []

    for line in stdout_lines:
        for prefix in prefixes:
            if line.startswith(prefix):
                raw_path = line.replace(prefix, "", 1).strip()
                if raw_path:
                    parsed = Path(raw_path)
                    if not parsed.is_absolute():
                        parsed = (downloads_dir / parsed).resolve()
                    candidates.append(parsed)
                break

    preferred = choose_preferred_video(candidates)
    if preferred:
        return preferred

    # Fallback for cases where yt-dlp output path hooks are missing.
    video_id = extract_video_id(url)
    if not video_id:
        try:
            id_cmd = [*yt_dlp_base_cmd(), "--get-id", "--no-warnings", url]
            id_result = subprocess.run(id_cmd, check=True, capture_output=True, text=True)
            video_id = id_result.stdout.strip()
        except subprocess.SubprocessError:
            video_id = ""
    if not video_id:
        return None

    globbed = list(downloads_dir.glob(f"*_{video_id}*"))
    return choose_preferred_video(globbed)


def download_video(url: str, downloads_dir: Path) -> Path:
    global _cookies_disabled
    downloads_dir.mkdir(parents=True, exist_ok=True)
    out_template = str(downloads_dir / "%(title).120s_%(id)s.%(ext)s")

    cmd = [
        *yt_dlp_base_cmd(),
        "-f",
        DOWNLOAD_FORMAT,
        "--merge-output-format",
        "mp4",
        "--retries",
        "3",
        "--continue",
        "--restrict-filenames",
        "-o",
        out_template,
        "--print",
        "after_move:%(filepath)s",
        "--print",
        "before_dl:%(filepath)s",
        "--print",
        "filename:%(filename)s",
    ]

    # Merging bestvideo+bestaudio needs ffmpeg; without it yt-dlp reports
    # "Requested format is not available". Point it at the bundled binary when
    # there is no system install.
    if not shutil.which("ffmpeg"):
        cmd += ["--ffmpeg-location", ffmpeg_executable()]

    def attempt(extra: List[str]) -> subprocess.CompletedProcess:
        try:
            return subprocess.run(cmd + extra + [url], check=False,
                                  capture_output=True, text=True,
                                  timeout=DOWNLOAD_TIMEOUT)
        except subprocess.TimeoutExpired as exc:
            # Left pending, not lost: run_all.run_pipeline2 logs the row and moves on.
            raise RuntimeError(
                f"yt-dlp timed out after {DOWNLOAD_TIMEOUT:.0f}s for URL: {url}"
            ) from exc

    cookies = cookie_args()
    result = attempt(cookies)
    if cookies and result.returncode != 0 and COOKIE_ERROR_RE.search(result.stderr or ""):
        # A dead cookie source must not take the run down with it. Without this it
        # fails EVERY row; having no cookies at all only costs the age-restricted
        # ones. Drop it for the rest of the process and retry once.
        _cookies_disabled = True
        tail = (result.stderr or "").strip().splitlines()
        print("Cookie source unusable, continuing without cookies: "
              + (tail[-1] if tail else "(no stderr)"))
        result = attempt([])

    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    downloaded = resolve_downloaded_video_path(url, downloads_dir, lines)
    if result.returncode != 0 and downloaded:
        print(
            "yt-dlp returned non-zero, but an existing video file was found. "
            f"Continuing with: {downloaded}"
        )
        return downloaded

    if not downloaded:
        tail = "\n".join(lines[-12:]) if lines else "(no --print output from yt-dlp)"
        stderr_tail = "\n".join(
            [line for line in result.stderr.splitlines() if line.strip()][-12:]
        ) or "(no stderr output from yt-dlp)"
        raise RuntimeError(
            "Could not resolve downloaded video file path for URL: "
            f"{url}\nyt-dlp output tail:\n{tail}\nyt-dlp error tail:\n{stderr_tail}"
        )

    print(f"Downloaded: {downloaded}")
    return downloaded


def load_reference_encoding(reference_image: Path):
    if not reference_image.exists():
        raise FileNotFoundError(f"Reference image not found: {reference_image}")

    image = face_recognition.load_image_file(str(reference_image))
    # Largest face, not encodings[0]. HOG finds faces in background art -- the
    # nagehan_alci photo carries a 36 px detection on a painting next to her 268 px
    # face -- and detection order is not size order, so taking the first encoding
    # risks matching every scene of a 300-row reporter against a painting. That
    # would fail silently: the row completes with zero clips and looks like a video
    # that simply had no solo footage.
    locations = face_recognition.face_locations(image, model="hog")
    if not locations:
        raise RuntimeError(
            f"No face found in reference image: {reference_image}. "
            "Use a frontal, clear face photo."
        )
    locations.sort(key=lambda b: b[2] - b[0], reverse=True)
    if len(locations) > 1:
        print(f"Reference {reference_image.name}: {len(locations)} faces detected, "
              f"using the largest ({locations[0][2] - locations[0][0]} px)")
    encodings = face_recognition.face_encodings(image, known_face_locations=locations[:1])
    if not encodings:
        raise RuntimeError(
            f"Face found but not encodable in reference image: {reference_image}."
        )
    return encodings[0]


def detect_scenes(video_path: Path, threshold: float) -> List[Tuple[float, float]]:
    video = open_video(str(video_path))
    manager = SceneManager()
    manager.add_detector(ContentDetector(threshold=threshold))
    manager.detect_scenes(video=video, show_progress=True)
    # start_in_scene=True is essential: by default PySceneDetect returns an EMPTY
    # list when it finds no cuts, so a single-shot talking-head broadcast - the best
    # possible lip-reading material - silently yields zero clips. With it, a cut-free
    # video comes back as one scene spanning the whole file.
    scene_list = manager.get_scene_list(start_in_scene=True)

    scenes: List[Tuple[float, float]] = []
    for start_tc, end_tc in scene_list:
        start_sec = start_tc.get_seconds()
        end_sec = end_tc.get_seconds()
        if end_sec > start_sec:
            scenes.append((start_sec, end_sec))
    return scenes


def sample_frames(video_path: Path, step: float = SECONDS_PER_SAMPLE,
                  start: float = 0.0, duration: Optional[float] = None):
    """Yield (time_sec, frame_bgr, source_height) every `step` seconds, sequentially.

    `start`/`duration` restrict the pass to one chunk (see analyze_video); times are
    still absolute. `start` should sit on the step grid so chunks tile it.

    One ffmpeg decode instead of an OpenCV seek per sample: a seek decodes from the
    previous keyframe, which at 5 samples a second would decode most of the video
    several times over. Frames taller than ANALYSIS_MAX_HEIGHT are scaled down;
    source_height lets callers report sizes in source pixels.
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

    window = ((["-ss", f"{start:.3f}"] if start > 0 else [])
              + (["-t", f"{duration:.3f}"] if duration is not None else []))
    cmd = [ffmpeg_executable(), "-v", "error", "-nostdin", *window, "-i", str(video_path),
           "-an", "-sn", "-vf", f"fps={1.0 / step:g}:round=near,scale={w}:{h}",
           "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    size = w * h * 3
    k = 0
    try:
        while True:
            buf = proc.stdout.read(size)
            if len(buf) < size:
                break
            yield start + k * step, np.frombuffer(buf, np.uint8).reshape(h, w, 3), src_h
            k += 1
    finally:
        proc.stdout.close()
        proc.kill()
        proc.wait()


_detector = None


def yunet_detector():
    """Per-process YuNet instance; downloads the 230 KB model on first use."""
    global _detector
    if _detector is None:
        if not YUNET_MODEL.exists():
            import urllib.request
            YUNET_MODEL.parent.mkdir(parents=True, exist_ok=True)
            print(f"Downloading YuNet face model -> {YUNET_MODEL}")
            urllib.request.urlretrieve(YUNET_URL, YUNET_MODEL)
        _detector = cv2.FaceDetectorYN.create(str(YUNET_MODEL), "", (320, 320),
                                              score_threshold=YUNET_MIN_SCORE)
    return _detector


def detect_faces(frame_bgr) -> List[Tuple[float, float, float, float, float]]:
    """YuNet detections as (score, x, y, w, h) in frame pixels, largest first."""
    fh, fw = frame_bgr.shape[:2]
    s = min(1.0, DETECT_WIDTH / fw)
    small = cv2.resize(frame_bgr, (int(fw * s), int(fh * s))) if s < 1.0 else frame_bgr
    det = yunet_detector()
    det.setInputSize((small.shape[1], small.shape[0]))
    _, raw = det.detect(small)
    faces = [] if raw is None else [
        (float(r[14]), float(r[0]) / s, float(r[1]) / s, float(r[2]) / s, float(r[3]) / s)
        for r in raw
    ]
    faces.sort(key=lambda f: f[4], reverse=True)
    return faces


def _dlib_box(face, frame_shape) -> Tuple[int, int, int, int]:
    """(top, right, bottom, left) for dlib from a YuNet box.

    YuNet's box runs from the hairline; dlib's HOG box is roughly square, brow to
    chin. Taking the bottom-anchored square of YuNet's width reproduces it closely
    enough that encodings and face_yaw match a native HOG box (see YUNET_MODEL).
    """
    _, x, y, w, h = face
    fh, fw = frame_shape[:2]
    top = int(max(0, y + h - w))
    bottom = int(min(fh - 1, y + h))
    left = int(max(0, x))
    right = int(min(fw - 1, x + w))
    return top, right, bottom, left


def is_on_set(face, frame_h: int) -> bool:
    """A detection that counts as a person on screen. See FACE_SCORE_MIN."""
    return face[0] >= FACE_SCORE_MIN and face[4] >= MIN_FACE_FRACTION * frame_h


def analyze_frames(frames, target_encoding) -> List[dict]:
    """Per-sample face record for the output of sample_frames.

    Each record: t, src_h, frame_h, frame_w and `faces`, a list of
    [score, x, y, w, h, dist, yaw] with the box as fractions of the frame, for every
    plausible face (score >= YUNET_MIN_SCORE, >= 3% of frame height). yaw is
    measured on every sample (None when landmarks cannot be fitted). dist is
    measured on every IDENTITY_EVERY-th sample, and also whenever the largest face
    jumped or the face count changed since the previous sample -- i.e. at the first
    sample of a new track -- and is None otherwise.

    The record is independent of the counting thresholds, so an audit can cache it
    and re-apply solo_segments with different settings without decoding again.
    """
    records: List[dict] = []
    prev = None
    for i, (t, frame, src_h) in enumerate(frames):
        fh, fw = frame.shape[:2]
        faces = [f for f in detect_faces(frame) if f[4] >= 0.03 * fh]
        jumped = prev is None or len(faces) != len(prev) or (
            faces and _shift(prev[0], faces[0]) >= TRACK_MAX_SHIFT)
        measure = i % IDENTITY_EVERY == 0 or jumped
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if faces else None
        out = []
        for f in faces:
            box = _dlib_box(f, frame.shape)
            yaw = face_yaw(rgb, box)
            yaw = None if yaw is None else round(yaw, 4)
            dist = None
            if measure:
                enc = face_recognition.face_encodings(rgb, known_face_locations=[box])
                if enc:
                    dist = round(float(face_recognition.face_distance(
                        [target_encoding], enc[0])[0]), 4)
            out.append([round(f[0], 3), round(f[1] / fw, 4), round(f[2] / fh, 4),
                        round(f[3] / fw, 4), round(f[4] / fh, 4), dist, yaw])
        records.append({"t": round(t, 3), "src_h": src_h, "frame_h": fh,
                        "frame_w": fw, "faces": out})
        prev = faces
    return records


# Parallel chunks for analyze_video. Each worker loads its own dlib models (~100 MB).
ANALYSIS_WORKERS = max(1, min(12, (os.cpu_count() or 4) - 4))
ANALYSIS_CHUNK_SEC = 300.0


def _analyze_chunk(video_path: str, target_encoding, start: float,
                   duration: float) -> List[dict]:
    cv2.setNumThreads(1)
    return analyze_frames(sample_frames(Path(video_path), start=start, duration=duration),
                          target_encoding)


def analyze_video(video_path: Path, target_encoding,
                  workers: int = ANALYSIS_WORKERS) -> List[dict]:
    """analyze_frames over a whole video, split into time chunks across processes.

    Single-threaded the analysis runs at ~4x realtime, i.e. half an hour for a
    two-hour bulletin; chunked over the CPU cores it keeps pace with Whisper.
    """
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    cap.release()
    total = frames / fps if fps else 0.0
    if workers <= 1 or total <= ANALYSIS_CHUNK_SEC:
        return analyze_frames(sample_frames(video_path), target_encoding)

    from concurrent.futures import ProcessPoolExecutor
    n = math.ceil(total / ANALYSIS_CHUNK_SEC)
    step = round(ANALYSIS_CHUNK_SEC / SECONDS_PER_SAMPLE) * SECONDS_PER_SAMPLE
    # The last chunk gets no -t, so a container whose frame count under-reports the
    # duration still has its tail analysed.
    jobs = [(i * step, step if i < n - 1 else None) for i in range(n)]
    with ProcessPoolExecutor(min(workers, n)) as ex:
        parts = list(ex.map(_analyze_chunk, [str(video_path)] * n,
                            [target_encoding] * n, *zip(*jobs)))
    return [r for part in parts for r in part]


def _shift(a, b) -> float:
    """Centre displacement between two (score, x, y, w, h) boxes, in face widths."""
    ax, ay = a[1] + a[3] / 2, a[2] + a[4] / 2
    bx, by = b[1] + b[3] / 2, b[2] + b[4] / 2
    return math.hypot(ax - bx, ay - by) / max(1.0, min(a[3], b[3]))


def face_yaw(frame_rgb, box) -> Optional[float]:
    """Lateral head turn: 0.0 facing the camera, 1.0 full profile.

    Taken from where the nose tip sits horizontally inside the jaw outline. On a
    frontal face it is midway between the two jaw edges; as the head turns, the far
    edge collapses towards it. Cheap (one landmark pass on a box already detected)
    and stable down to fairly small faces.

    Returns None when landmarks cannot be fitted; callers must treat that as a
    failure, not a pass -- an unmeasurable face is not a verified frontal one.
    """
    marks = face_recognition.face_landmarks(frame_rgb, [box])
    if not marks:
        return None
    chin = marks[0].get("chin")
    nose = marks[0].get("nose_tip")
    if not chin or not nose:
        return None
    nose_x = sum(p[0] for p in nose) / len(nose)
    left_x, right_x = chin[0][0], chin[-1][0]
    d_left, d_right = abs(nose_x - left_x), abs(right_x - nose_x)
    if d_left + d_right <= 0:
        return 1.0
    return abs(d_right - d_left) / (d_right + d_left)


def presenter_sized_faces(frame_rgb) -> List[Tuple[int, int, int, int]]:
    """Faces that count as a person on set, as dlib (top, right, bottom, left) boxes.

    Kept for the one-off maintenance script backfill_face_height;
    the pipeline itself goes through analyze_frames/solo_segments.
    """
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    return [_dlib_box(f, frame_rgb.shape) for f in detect_faces(frame_bgr)
            if is_on_set(f, frame_rgb.shape[0])]


def solo_segments(
    records: List[dict],
    start_sec: float,
    end_sec: float,
    tolerance: float,
    min_duration: float = MIN_CLIP_DURATION,
) -> List[Tuple[float, float, int]]:
    """Sub-ranges of one scene in which only the target reporter is on screen.

    `records` are analyze_frames output; only those inside the scene, after
    SCENE_EDGE_TRIM is taken off both ends, are considered. A sample passes when:

      1. exactly one detection is a person on set (is_on_set) -- a second face, a
         split-screen guest, an interpreter box, or NO face all fail;
      2. it belongs to a track (run of such samples whose face does not jump by
         TRACK_MAX_SHIFT face-widths) whose median identity distance is within
         `tolerance`; a track with no identity measurement at all fails;
      3. its own yaw is <= MAX_FACE_YAW ("the presenter faces the camera
         directly" -- a profile to an off-camera guest has no readable mouth); a
         face whose landmarks cannot be fitted fails.

    Passing samples are merged into runs; each run becomes (start, end, face_px),
    extended by half a sample step either side and clamped to the trimmed scene,
    and dropped when shorter than min_duration. face_px is the median on-set face
    height in SOURCE pixels -- recorded, never gated on (see CLIP_FIELDS).

    Why not a whole-scene vote: get_scene_list(start_in_scene=True) makes a cut-free
    broadcast ONE scene, and one two-face insert then discarded 34.7 minutes of
    clean Deniz Zeyrek monologue. Cutting at the failing samples keeps the rest.
    """
    lo, hi = start_sec + SCENE_EDGE_TRIM, end_sec - SCENE_EDGE_TRIM
    recs = [r for r in records if lo <= r["t"] <= hi]
    if hi - lo < min_duration or not recs:
        return []

    # 1. exactly one person on set
    solo: List[Optional[list]] = []
    for r in recs:
        on_set = [f for f in r["faces"] if f[0] >= FACE_SCORE_MIN and f[4] >= MIN_FACE_FRACTION]
        solo.append(on_set[0] if len(on_set) == 1 else None)

    # 2. tracks, identity voted per track
    def px(r, f):
        return (f[0], f[1] * r["frame_w"], f[2] * r["frame_h"],
                f[3] * r["frame_w"], f[4] * r["frame_h"])

    good = [False] * len(recs)
    i = 0
    while i < len(recs):
        if solo[i] is None:
            i += 1
            continue
        j = i
        while (j + 1 < len(recs) and solo[j + 1] is not None
               and _shift(px(recs[j], solo[j]), px(recs[j + 1], solo[j + 1])) < TRACK_MAX_SHIFT):
            j += 1
        track = range(i, j + 1)
        dists = [solo[k][5] for k in track if solo[k][5] is not None]
        if dists and statistics.median(dists) <= tolerance:
            # 3. yaw, per sample
            for k in track:
                good[k] = solo[k][6] is not None and solo[k][6] <= MAX_FACE_YAW
        i = j + 1

    half = SECONDS_PER_SAMPLE / 2.0
    out: List[Tuple[float, float, int]] = []
    k = 0
    while k < len(recs):
        if not good[k]:
            k += 1
            continue
        m = k
        while m + 1 < len(recs) and good[m + 1]:
            m += 1
        a = max(lo, recs[k]["t"] - half)
        b = min(hi, recs[m]["t"] + half)
        if b - a >= min_duration:
            hs = [solo[q][4] * recs[q]["src_h"] for q in range(k, m + 1)]
            out.append((a, b, int(statistics.median(hs))))
        k = m + 1
    return out


def safe_name(raw: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", raw)
    cleaned = cleaned.strip("._")
    return cleaned or "video"


def export_clip(
    input_video: Path,
    output_video: Path,
    start_sec: float,
    end_sec: float,
) -> None:
    output_video.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg_executable(),
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{start_sec:.3f}",
        "-to",
        f"{end_sec:.3f}",
        "-i",
        str(input_video),
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "22",
        "-c:a",
        "aac",
        str(output_video),
    ]
    run_cmd(cmd)


def process_video(
    video_path: Path,
    output_dir: Path,
    target_encoding,
    threshold: float,
    tolerance: float,
    min_duration: float,
    scene_rows: List[dict],
) -> None:
    print(f"\nProcessing video: {video_path}")
    scenes = detect_scenes(video_path, threshold=threshold)
    print(f"Detected scenes: {len(scenes)}")
    records = analyze_video(video_path, target_encoding)

    kept = 0
    skipped = 0
    base = safe_name(video_path.stem)

    for idx, (scene_start, scene_end) in enumerate(scenes, start=1):
        segments = solo_segments(records, scene_start, scene_end, tolerance,
                                 min_duration=min_duration)
        if not segments:
            skipped += 1
            continue

        # One scene can yield several clips; start-end in the name keeps them
        # distinct, so the scene index still means the source scene.
        for start_sec, end_sec, face_h in segments:
            duration = end_sec - start_sec
            clip_name = f"{base}_scene_{idx:04d}_{start_sec:.2f}-{end_sec:.2f}.mp4"
            output_video = output_dir / clip_name
            export_clip(video_path, output_video, start_sec, end_sec)
            scene_rows.append(
                {
                    "video": video_path.name,
                    "scene_index": idx,
                    "start_sec": round(start_sec, 3),
                    "end_sec": round(end_sec, 3),
                    "duration_sec": round(duration, 3),
                    "face_height_px": face_h,
                }
            )
            kept += 1
            print(f"Kept: {output_video.name}")

    print(f"Finished {video_path.name} | kept={kept} skipped={skipped}")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Download videos from links file, detect scenes, and keep clips where "
            "only the target person appears."
        )
    )
    parser.add_argument("--links-file", required=True,
                        help="Path to a reporter links file, e.g. reporter_links/links_cem_ogretir.txt")
    parser.add_argument("--reference-image", required=True,
                        help="Path to reference face photo, e.g. pipeline2_reporter_pictures/cem_ogretir.jpeg")
    parser.add_argument("--downloads-dir", default="data/raw_videos")
    parser.add_argument("--output-dir", default="data/reporter_solo_clips")
    parser.add_argument("--scene-threshold", type=float, default=30.0)
    parser.add_argument("--face-tolerance", type=float, default=IDENTITY_TOLERANCE)
    parser.add_argument("--min-duration", type=float, default=MIN_CLIP_DURATION)
    return parser.parse_args()


def main():
    args = parse_args()
    links_file = Path(args.links_file)
    reference_image = Path(args.reference_image)
    downloads_dir = Path(args.downloads_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    scene_csv_path = output_dir / "scene_boundaries.csv"

    urls = read_links(links_file)
    if not urls:
        raise RuntimeError(f"No URL found in {links_file}")

    target_encoding = load_reference_encoding(reference_image)

    downloaded_files: List[Path] = []
    for url in urls:
        downloaded_files.append(download_video(url, downloads_dir))

    scene_rows: List[dict] = []
    for video_path in downloaded_files:
        process_video(
            video_path=video_path,
            output_dir=output_dir,
            target_encoding=target_encoding,
            threshold=args.scene_threshold,
            tolerance=args.face_tolerance,
            min_duration=max(0.1, args.min_duration),
            scene_rows=scene_rows,
        )

    with scene_csv_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["video", "scene_index", "start_sec", "end_sec", "duration_sec",
                      "face_height_px"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(scene_rows)
        total_kept_duration = round(
            sum(float(row["duration_sec"]) for row in scene_rows), 3
        )
        writer.writerow(
            {
                "video": "TOTAL_KEPT",
                "scene_index": "",
                "start_sec": "",
                "end_sec": "",
                "duration_sec": total_kept_duration,
            }
        )

    print("\nPipeline finished successfully.")
    print(f"Output clips: {output_dir.resolve()}")
    print(f"Scene CSV: {scene_csv_path.resolve()}")


if __name__ == "__main__":
    main()
