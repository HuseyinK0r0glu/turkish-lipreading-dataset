# scripts/download_and_transcribe.py
import os
import subprocess
import sys

from download_videos import download_videos
from run_whisper import run_whisper_pipeline

# Global sets shared across pipeline steps.
SEEN_VIDEO_INTERVALS = set()
# Map: overlap_tuple -> occurrence_count
OVERLAPPING_VIDEO_INTERVALS = {}


def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(project_root)

    print("Installing requirements...")
    subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

    print("Starting video download...")
    download_videos(SEEN_VIDEO_INTERVALS, OVERLAPPING_VIDEO_INTERVALS)

    print("Starting Whisper transcription ...")
    run_whisper_pipeline(SEEN_VIDEO_INTERVALS, OVERLAPPING_VIDEO_INTERVALS)


if __name__ == "__main__":
    main()
