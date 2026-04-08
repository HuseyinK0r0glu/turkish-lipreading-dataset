# scripts/download_videos.py
import os
import csv
import subprocess

VIDEO_DIR = "data/raw_videos"
LINK_FILE = "video_links.csv"

os.makedirs(VIDEO_DIR, exist_ok=True)

with open(LINK_FILE, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    entries = list(reader)

for entry in entries:
    url     = entry["url"].strip()
    channel = entry["channel"].strip()
    start   = entry["start"].strip()
    end     = entry["end"].strip()

    if not url:
        continue

    # Build output filename using channel name instead of uploader
    filename = f"{channel}_%(title)s"
    if start and end:
        filename += f"_{start.replace(':', '')}-{end.replace(':', '')}"
    filename += ".%(ext)s"

    print(f"[{channel}] Downloading {url} {f'{start} → {end}' if start else '(full video)'}")

    cmd = [
        "yt-dlp",
        "-f", "bestvideo[height>=720]+bestaudio/best",
        "--merge-output-format", "mp4",
        "--retries", "3",
        "--continue",
        "-o", os.path.join(VIDEO_DIR, filename),
        url
    ]

    if start and end:
        cmd += [
            "--download-sections", f"*{start}-{end}",
            "--force-keyframes-at-cuts"
        ]

    subprocess.run(cmd)