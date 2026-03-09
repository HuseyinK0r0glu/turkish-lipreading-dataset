# scripts/download_videos.py
import os
import subprocess

VIDEO_DIR = "data/raw_videos"
LINK_FILE = "video_links.txt"

# make sure directory exists
os.makedirs(VIDEO_DIR, exist_ok=True)

# read links
with open(LINK_FILE, "r") as f:
    links = [line.strip() for line in f if line.strip()]

# download each video with yt-dlp
for url in links:
    print(f"Downloading {url} ...")
    cmd = [
        "yt-dlp",
        "-f", "bestvideo[height>=720]+bestaudio/best",  # video>=720 + audio
        "--merge-output-format", "mp4",                 # merge video+audio to single mp4
        "--retries", "3",                               # retry if fail
        "--continue",                                   # resume if partially downloaded
        "-o", os.path.join(VIDEO_DIR, "%(uploader)s_%(title)s.%(ext)s"),
        url
    ]
    subprocess.run(cmd)