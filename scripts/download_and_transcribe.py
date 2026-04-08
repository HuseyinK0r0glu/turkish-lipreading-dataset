# scripts/download_and_transcribe.py
import subprocess
import sys
import os

# Always run from the project root regardless of where you call this from
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_ROOT)

# 1. Install requirements
print("Installing requirements...")
subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

# 2. Download videos
print("Starting video download...")
subprocess.run([sys.executable, "scripts/download_videos.py"])

# 3. Run Whisper transcription
print("Starting Whisper transcription ...")
subprocess.run([sys.executable, "scripts/run_whisper.py"])