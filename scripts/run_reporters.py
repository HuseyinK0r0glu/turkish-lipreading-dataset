"""Run the solo-presenter pipeline for each reporter using YOUR reference photos.

Setup (you do this once):
  1. Put one clear, frontal face photo per reporter in  data/reference_faces/
     with these exact names:
        fatih.jpeg  kubra.jpeg  gulsah.jpeg  serdar.jpeg  ece.jpeg  ismail.jpeg
  2. Run:  venv/bin/python scripts/run_reporters.py
     (or run a single reporter:  ... scripts/run_reporters.py fatih_portakal)

For each reporter it calls scripts/pipeline_cuneyt_solo.py on that reporter's
links file with that reporter's reference image, writing kept solo clips to
data/<slug>_solo_clips/ plus a scene_boundaries.csv.
"""

import subprocess
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
PIPELINE = PROJECT / "scripts" / "pipeline_cuneyt_solo.py"
REF_DIR = PROJECT / "data" / "reference_faces"

# slug, links file, reference image filename (placed by you in data/reference_faces/)
REPORTERS = [
    ("fatih_portakal",   "links_fatih_portakal.txt",   "fatih.jpeg"),
    ("kubra_par",        "links_kubra_par.txt",        "kubra.jpeg"),
    ("gulsah_ekinci",    "links_gulsah_ekinci.txt",    "gulsah.jpeg"),
    ("serdar_cebe",      "links_serdar_cebe.txt",      "serdar.jpeg"),
    ("ece_uner",         "links_ece_uner.txt",         "ece.jpeg"),
    ("ismail_kucukkaya", "links_ismail_kucukkaya.txt", "ismail.jpeg"),
]

# Filter knobs passed to the pipeline (override the script defaults of 3 / 0.47).
# sample-count 5  -> a brief head-turn in one sampled frame no longer drops the scene.
# face-tolerance 0.5 -> slightly looser so the same anchor isn't over-rejected on angle.
SAMPLE_COUNT = "5"
FACE_TOLERANCE = "0.5"


def run_one(slug: str, links_name: str, ref_name: str) -> None:
    ref_path = REF_DIR / ref_name
    links_path = PROJECT / links_name
    out_dir = PROJECT / "data" / f"{slug}_solo_clips"

    if not ref_path.exists():
        print(f"[skip] {slug}: put a face photo at {ref_path}")
        return
    if not links_path.exists():
        print(f"[skip] {slug}: links file missing at {links_path}")
        return

    print(f"\n===== {slug} =====")
    subprocess.run(
        [
            sys.executable, str(PIPELINE),
            "--links-file", str(links_path),
            "--reference-image", str(ref_path),
            "--output-dir", str(out_dir),
            "--sample-count", SAMPLE_COUNT,
            "--face-tolerance", FACE_TOLERANCE,
        ],
        check=False,
    )


def main() -> None:
    REF_DIR.mkdir(parents=True, exist_ok=True)
    wanted = sys.argv[1:]  # optional: run only the given slugs
    selected = [r for r in REPORTERS if not wanted or r[0] in wanted]
    if not selected:
        print(f"No matching reporter. Choose from: {[r[0] for r in REPORTERS]}")
        return
    for slug, links_name, ref_name in selected:
        run_one(slug, links_name, ref_name)


if __name__ == "__main__":
    main()
