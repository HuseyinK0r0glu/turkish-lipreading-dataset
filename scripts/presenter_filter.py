import argparse
import csv
import math
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import cv2
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
DOWNLOAD_FORMAT = "bestvideo[height>=720]+bestaudio/best[height>=720]"

# Identity-check sampling density for long scenes (see sample_times).
SECONDS_PER_SAMPLE = 30.0
MAX_SAMPLES = 20

# Wall-clock ceilings for the two steps that can block forever. Neither raises on
# its own: yt-dlp waits on the network, and OpenCV spins in C code no Python
# exception can reach, so a single bad video hangs an unattended 4691-row run.
DOWNLOAD_TIMEOUT = 3600.0
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
COOKIE_ERROR_RE = re.compile(
    "cookie database|DPAPI|cookies-from-browser|could not (copy|find|decrypt)",
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
    encodings = face_recognition.face_encodings(image)
    if not encodings:
        raise RuntimeError(
            f"No face found in reference image: {reference_image}. "
            "Use a frontal, clear face photo."
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


def sample_times(start_sec: float, end_sec: float, sample_count: int) -> List[float]:
    """Frame times used to certify "only the target appears" in a scene.

    sample_count is a floor, not a fixed number: now that cut-free videos come back
    as one scene spanning the whole file, a fixed 5 samples would be certifying a
    40-minute broadcast off five frames. Scale with duration, but bound it - every
    sample costs a HOG face detection (~1 s at 1080p) across thousands of videos.
    """
    duration = end_sec - start_sec
    if duration <= 0:
        return []

    effective = max(sample_count, math.ceil(duration / SECONDS_PER_SAMPLE))
    effective = min(effective, MAX_SAMPLES)
    if effective <= 1:
        return [start_sec + duration * 0.5]

    times: List[float] = []
    for i in range(effective):
        ratio = (i + 1) / (effective + 1)
        times.append(start_sec + (duration * ratio))
    return times


def extract_frame_rgb(cap: cv2.VideoCapture, time_sec: float):
    cap.set(cv2.CAP_PROP_POS_MSEC, max(0, time_sec * 1000))
    ok, frame = cap.read()
    if not ok or frame is None:
        return None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def scene_has_only_target(
    cap: cv2.VideoCapture,
    start_sec: float,
    end_sec: float,
    target_encoding,
    sample_count: int,
    tolerance: float,
) -> bool:
    times = sample_times(start_sec, end_sec, sample_count)
    found_target_single = False

    for t in times:
        frame_rgb = extract_frame_rgb(cap, t)
        if frame_rgb is None:
            continue

        locations = face_recognition.face_locations(frame_rgb, model="hog")
        face_count = len(locations)

        if face_count > 1:
            return False

        if face_count == 1:
            encoding = face_recognition.face_encodings(
                frame_rgb, known_face_locations=locations
            )
            if not encoding:
                continue
            is_match = face_recognition.compare_faces(
                [target_encoding], encoding[0], tolerance=tolerance
            )[0]
            if is_match:
                found_target_single = True

    return found_target_single


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
    sample_count: int,
    tolerance: float,
    min_duration: float,
    scene_rows: List[dict],
) -> None:
    print(f"\nProcessing video: {video_path}")
    scenes = detect_scenes(video_path, threshold=threshold)
    print(f"Detected scenes: {len(scenes)}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    kept = 0
    skipped = 0
    base = safe_name(video_path.stem)

    try:
        for idx, (start_sec, end_sec) in enumerate(scenes, start=1):
            duration = end_sec - start_sec
            if duration < min_duration:
                skipped += 1
                continue

            keep = scene_has_only_target(
                cap=cap,
                start_sec=start_sec,
                end_sec=end_sec,
                target_encoding=target_encoding,
                sample_count=sample_count,
                tolerance=tolerance,
            )
            if not keep:
                skipped += 1
                continue

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
                }
            )
            kept += 1
            print(f"Kept: {output_video.name}")
    finally:
        cap.release()

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
    parser.add_argument("--sample-count", type=int, default=3)
    parser.add_argument("--face-tolerance", type=float, default=0.47)
    parser.add_argument("--min-duration", type=float, default=1.2)
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
            sample_count=max(1, args.sample_count),
            tolerance=args.face_tolerance,
            min_duration=max(0.1, args.min_duration),
            scene_rows=scene_rows,
        )

    with scene_csv_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["video", "scene_index", "start_sec", "end_sec", "duration_sec"]
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
