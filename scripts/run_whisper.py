# scripts/run_whisper.py
import json
import os
import unicodedata

import whisper

VIDEO_DIR = "data/raw_videos"
OUTPUT_DIR = "data/transcripts"


def normalize_text(text):
    return unicodedata.normalize("NFC", text)


def transcribe_videos(video_dir=VIDEO_DIR, output_dir=OUTPUT_DIR, model_name="large-v3"):
    os.makedirs(output_dir, exist_ok=True)
    model = whisper.load_model(model_name)

    for file in os.listdir(video_dir):
        if not file.endswith((".mp4", ".webm", ".mkv")):
            continue

        video_path = os.path.join(video_dir, file)
        out_file = os.path.join(output_dir, f"{os.path.splitext(file)[0]}.json")

        if os.path.exists(out_file):
            continue

        result = model.transcribe(
            video_path,
            word_timestamps=True,
            language="tr",
            verbose=False,
        )

        custom_result = {"video": file, "segments": []}

        for seg in result.get("segments", []):
            segment_data = {
                "start_sec": round(seg["start"], 3),
                "end_sec": round(seg["end"], 3),
                "text": normalize_text(seg["text"]),
                "words": [],
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
                    "manual_review": int(confidence < 0.7),
                }
                segment_data["words"].append(word_data)

            custom_result["segments"].append(segment_data)

        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(custom_result, f, ensure_ascii=False, indent=2)


def print_interval_report(seen_video_intervals, overlapping_video_intervals):
    print("\n--- CSV Interval Check (download stage) ---")
    print(f"Tracked unique URL/time intervals: {len(seen_video_intervals)}")
    if overlapping_video_intervals:
        total_overlap_skips = sum(overlapping_video_intervals.values())
        print(f"Total rows skipped due to overlap: {total_overlap_skips}")
        print("Overlapping intervals found and skipped from download (same video, overlap >= 1 second):")
        for item, count in sorted(overlapping_video_intervals.items()):
            video_url, start_raw, end_raw, overlap_start, overlap_end = item
            print(
                f"- URL: {video_url} | interval: {start_raw} -> {end_raw} | "
                f"overlap: {overlap_start} -> {overlap_end} | occurs {count} times"
            )
    else:
        print("No overlaps found.")


def print_total_transcribed_summary(output_dir=OUTPUT_DIR):
    total_seconds = 0.0

    for file in os.listdir(output_dir):
        if not file.endswith(".json"):
            continue
        with open(os.path.join(output_dir, file), "r", encoding="utf-8") as f:
            data = json.load(f)
        if data["segments"]:
            total_seconds += data["segments"][-1]["end_sec"]

    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)

    print(f"\nTotal transcribed audio: {hours}h {minutes}m {seconds}s ({total_seconds:.1f} seconds)")


def run_whisper_pipeline(seen_video_intervals, overlapping_video_intervals):
    transcribe_videos()
    print_interval_report(seen_video_intervals, overlapping_video_intervals)
    print_total_transcribed_summary()


if __name__ == "__main__":
    run_whisper_pipeline(set(), {})
