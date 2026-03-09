# scripts/run_whisper.py
import os
import whisper
import json
import unicodedata

VIDEO_DIR = "data/raw_videos"
OUTPUT_DIR = "data/transcripts"

os.makedirs(OUTPUT_DIR, exist_ok=True)

model = whisper.load_model("large-v3")


def normalize_text(text):
    return unicodedata.normalize("NFC", text)


for file in os.listdir(VIDEO_DIR):
    if not file.endswith((".mp4", ".webm", ".mkv")):
        continue

    video_path = os.path.join(VIDEO_DIR, file)
    out_file = os.path.join(OUTPUT_DIR, f"{os.path.splitext(file)[0]}.json")

    # Skip already-transcribed files
    if os.path.exists(out_file):
        print(f"Skipping (already transcribed): {file}")
        continue

    print(f"Transcribing {video_path} ...")
    result = model.transcribe(
        video_path,
        word_timestamps=True,
        language="tr",
        verbose=False
    )

    custom_result = {
        "video": file,
        "segments": []
    }

    for seg in result.get("segments", []):
        segment_data = {
            "start_sec": round(seg["start"], 3),
            "end_sec": round(seg["end"], 3),
            "text": normalize_text(seg["text"]),
            "words": []
        }

        for w in seg.get("words", []):
            surface = normalize_text(w["word"])
            confidence = round(w.get("probability", 1.0), 6)

            word_data = {
                "surface_form": surface,
                "lemma": surface,  # TODO: add lemma later (e.g. via zeyrek MorphAnalyzer)
                "start_sec": round(w["start"], 3),
                "end_sec": round(w["end"], 3),
                "confidence": confidence,
                "manual_review": int(confidence < 0.7)
            }
            segment_data["words"].append(word_data)

        custom_result["segments"].append(segment_data)

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(custom_result, f, ensure_ascii=False, indent=2)

    print(f"  Saved → {out_file}")

print("Done.")