# scripts/download_videos.py
import csv
import os
import shutil
import subprocess
import sys
import math
from urllib.parse import parse_qs, urlparse

def _yt_dlp():
    return [sys.executable, "-m", "yt_dlp"]


def ffmpeg_location():
    """Path to hand yt-dlp via --ffmpeg-location, or None if ffmpeg is on PATH.

    yt-dlp needs ffmpeg both for --download-sections and for merging
    bestvideo+bestaudio; without it every row dies with "ffmpeg is not installed"
    or "Requested format is not available". Falls back to the binary bundled with
    imageio-ffmpeg so no system install is strictly required.

    Deliberately duplicated from presenter_filter.ffmpeg_executable() so that
    pipeline 1 does not have to import face_recognition.
    """
    if shutil.which("ffmpeg"):
        return None  # already on PATH; yt-dlp finds it by itself
    try:
        from imageio_ffmpeg import get_ffmpeg_exe

        return get_ffmpeg_exe()
    except Exception:
        return None

VIDEO_DIR = "data/raw_videos"
LINK_FILE = "video_links.csv"

# Same policy as presenter_filter.DOWNLOAD_FORMAT: >=720p is a hard requirement, so a
# lower-resolution source is skipped (row left pending) instead of polluting the
# dataset with mouth crops upscaled from 480p.
DOWNLOAD_FORMAT = "bestvideo[height>=720]+bestaudio/best[height>=720]"


def normalize_video_url(url):
    parsed = urlparse(url.strip())
    host = parsed.netloc.lower().replace("www.", "").replace("m.", "")
    path = parsed.path.rstrip("/")

    if host in {"youtube.com", "youtu.be"}:
        if host == "youtu.be":
            video_id = path.lstrip("/")
            if video_id:
                return f"https://youtube.com/watch?v={video_id}"
        if path == "/watch":
            video_id = parse_qs(parsed.query).get("v", [""])[0]
            if video_id:
                return f"https://youtube.com/watch?v={video_id}"
        if path.startswith("/shorts/"):
            short_id = path.split("/shorts/", 1)[1].split("/", 1)[0]
            if short_id:
                return f"https://youtube.com/shorts/{short_id}"

    return f"{parsed.scheme}://{host}{parsed.path}"


def parse_time_to_seconds(time_text):
    value = time_text.strip()
    if not value:
        return None

    parts = value.split(":")
    try:
        if len(parts) == 3:
            return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
        if len(parts) == 2:
            return float(parts[0]) * 60 + float(parts[1])
        if len(parts) == 1:
            return float(parts[0])
    except ValueError:
        return None
    return None


def format_seconds(total_seconds):
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = total_seconds - (hours * 3600 + minutes * 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"


def format_hhmmss(total_seconds):
    whole = int(math.ceil(total_seconds))
    hours = whole // 3600
    minutes = (whole % 3600) // 60
    seconds = whole % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def get_video_duration_seconds(url):
    try:
        output = subprocess.check_output(
            [*_yt_dlp(), "--print", "%(duration)s", url],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        return float(output)
    except (subprocess.CalledProcessError, FileNotFoundError, ValueError):
        return None


def segment_tag(start, end):
    return f"_{start.replace(':', '')}-{end.replace(':', '')}"


def segment_already_downloaded(video_dir, channel, start, end):
    expected_suffix = f"{segment_tag(start, end)}.mp4"
    expected_prefix = f"{channel}_"

    for name in os.listdir(video_dir):
        if not name.startswith(expected_prefix):
            continue
        if name.endswith(expected_suffix):
            return True
    return False


def increment_overlap_count(overlapping_video_intervals, overlap_key):
    overlapping_video_intervals[overlap_key] = overlapping_video_intervals.get(overlap_key, 0) + 1


def download_videos(seen_video_intervals, overlapping_video_intervals, link_file=LINK_FILE, video_dir=VIDEO_DIR):
    """Download every row of link_file.

    Returns the set of (url, start, end) keys — taken verbatim from the CSV — that
    need no further work: downloaded now, already on disk, or intentionally skipped
    as an overlapping duplicate. Rows that FAILED are absent, so the caller must not
    mark them completed. (Callers that ignore the return value keep working.)
    """
    os.makedirs(video_dir, exist_ok=True)
    seen_video_intervals.clear()
    overlapping_video_intervals.clear()

    with open(link_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        entries = list(reader)

    intervals_by_url = {}
    duration_cache = {}
    completed_keys = set()
    ffmpeg_path = ffmpeg_location()

    for row_index, entry in enumerate(entries, start=2):
        url = entry["url"].strip()
        channel = entry["channel"].strip()
        start = entry["start"].strip()
        end = entry["end"].strip()

        if not url:
            continue

        # start/end are rewritten below for full-video rows; keep the CSV's own
        # values so the caller can match the key back to its row.
        row_key = (url, start, end)

        normalized_url = normalize_video_url(url)

        should_skip_download = False
        overlap_start_fmt = None
        overlap_end_fmt = None

        is_full_video = False
        if not start and not end:
            if normalized_url not in duration_cache:
                duration_cache[normalized_url] = get_video_duration_seconds(url)
            duration_seconds = duration_cache[normalized_url]
            if duration_seconds is None or duration_seconds <= 0:
                print(f"[{channel}] Skipping row {row_index}: could not read duration for full-video URL")
                continue
            start = "00:00:00"
            end = format_hhmmss(duration_seconds)
            is_full_video = True
        elif bool(start) != bool(end):
            print(f"[{channel}] Skipping row {row_index}: both start and end must be provided together")
            continue

        start_sec = parse_time_to_seconds(start)
        end_sec = parse_time_to_seconds(end)

        if start_sec is None or end_sec is None:
            print(f"[{channel}] Skipping row {row_index}: invalid time format ({start} -> {end})")
            continue
        if end_sec <= start_sec:
            print(f"[{channel}] Skipping row {row_index}: invalid interval ({start} -> {end})")
            continue

        prior_intervals = intervals_by_url.setdefault(normalized_url, [])
        for prev_start, prev_end, prev_start_raw, prev_end_raw, prev_is_full_video in prior_intervals:
            if prev_is_full_video:
                overlap_start_fmt = format_seconds(start_sec)
                overlap_end_fmt = format_seconds(end_sec)

                increment_overlap_count(
                    overlapping_video_intervals,
                    (normalized_url, start, end, overlap_start_fmt, overlap_end_fmt),
                )
                should_skip_download = True
                break

            overlap_start = max(start_sec, prev_start)
            overlap_end = min(end_sec, prev_end)
            if (overlap_end - overlap_start) >= 1.0:
                overlap_start_fmt = format_seconds(overlap_start)
                overlap_end_fmt = format_seconds(overlap_end)

                increment_overlap_count(
                    overlapping_video_intervals,
                    (normalized_url, start, end, overlap_start_fmt, overlap_end_fmt),
                )
                should_skip_download = True
                break

        if should_skip_download:
            print(
                f"[{channel}] Skipping due to overlap: {url} "
                f"({start} -> {end}) overlaps {overlap_start_fmt} -> {overlap_end_fmt}"
            )
            completed_keys.add(row_key)  # duplicate coverage: nothing left to do
            continue

        # Only keep intervals that are not skipped due to overlap.
        seen_video_intervals.add((normalized_url, start, end, is_full_video))
        prior_intervals.append((start_sec, end_sec, start, end, is_full_video))

        filename = f"{channel}_%(title)s"
        if start and end:
            filename += segment_tag(start, end)
        filename += ".%(ext)s"

        if segment_already_downloaded(video_dir, channel, start, end):
            print(f"[{channel}] Skipping (already downloaded): {url} ({start} -> {end})")
            completed_keys.add(row_key)
            continue

        print(f"[{channel}] Downloading {url} {f'{start} → {end}' if start else '(full video)'}")

        cmd = [
            *_yt_dlp(),
            "-f",
            DOWNLOAD_FORMAT,
            "--merge-output-format",
            "mp4",
            "--retries",
            "3",
            "--continue",
            "-o",
            os.path.join(video_dir, filename),
            url,
        ]

        cmd += [
            "--download-sections",
            f"*{start}-{end}",
            "--force-keyframes-at-cuts",
        ]

        if ffmpeg_path:
            cmd += ["--ffmpeg-location", ffmpeg_path]

        result = subprocess.run(cmd)
        if result.returncode == 0:
            completed_keys.add(row_key)
        else:
            print(f"[{channel}] DOWNLOAD FAILED (yt-dlp exit {result.returncode}): "
                  f"{url} ({start} -> {end}) - row left pending for a re-run")

    return completed_keys


if __name__ == "__main__":
    download_videos(set(), {})
