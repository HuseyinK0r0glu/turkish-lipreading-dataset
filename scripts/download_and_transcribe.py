# scripts/download_and_transcribe.py
import subprocess
import sys
import os

# 1️⃣ Install requirements
print("Installing requirements...")
subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

# 2️⃣ Call download_videos.py
print("Starting video download...")
subprocess.run([sys.executable, "scripts/download_videos.py"])

# 3. Run Whisper transcription
print("Starting Whisper transcription ...")
subprocess.run([sys.executable, "scripts/run_whisper.py"])