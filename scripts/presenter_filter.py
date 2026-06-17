import argparse
import csv
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


def is_video_readable(video_path: Path) -> bool:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        return False
    ok, _ = cap.read()
    cap.release()
    return bool(ok)


def choose_preferred_video(candidates: Iterable[Path]) -> Optional[Path]:
    # Prefer common video containers and avoid transient partial files.
    video_ext_order = {".mp4": 0, ".mkv": 1, ".webm": 2, ".mov": 3}
    ordered = sorted(
        {p.resolve() for p in candidates if p and p.exists() and p.suffix.lower() != ".part"},
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
    downloads_dir.mkdir(parents=True, exist_ok=True)
    out_template = str(downloads_dir / "%(title).120s_%(id)s.%(ext)s")

    cmd = [
        *yt_dlp_base_cmd(),
        "-f",
        "bestvideo[height>=720]+bestaudio/best/best",
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
        url,
    ]

    result = subprocess.run(cmd, check=False, capture_output=True, text=True)
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
    scene_list = manager.get_scene_list()

    scenes: List[Tuple[float, float]] = []
    for start_tc, end_tc in scene_list:
        start_sec = start_tc.get_seconds()
        end_sec = end_tc.get_seconds()
        if end_sec > start_sec:
            scenes.append((start_sec, end_sec))
    return scenes


def sample_times(start_sec: float, end_sec: float, sample_count: int) -> List[float]:
    duration = end_sec - start_sec
    if duration <= 0:
        return []
    if sample_count <= 1:
        return [start_sec + duration * 0.5]

    times: List[float] = []
    for i in range(sample_count):
        ratio = (i + 1) / (sample_count + 1)
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
