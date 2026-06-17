"""Quick smoke test for the solo-presenter pipeline across multiple reporters.

For each reporter links file this script:
  1. Probes the first few videos and picks the SHORTEST as the test video.
  2. Downloads a short section from a DIFFERENT video, auto-extracts the
     reporter's dominant frontal face -> saves it as the reference photo.
  3. Downloads a short section of the shortest video and runs the existing
     solo-filter (scene detect + single-face identity match + clip export).
  4. Prints a summary so you can see, per reporter, whether it works.

Reference and test come from different videos, so a "kept" clip means the
auto-extracted reference photo actually generalizes across broadcasts.

Run:  venv/bin/python scripts/quick_test_reporters.py
"""

import subprocess
import sys
from pathlib import Path

import cv2
import face_recognition

sys.path.insert(0, str(Path(__file__).resolve().parent))
from presenter_filter import (  # noqa: E402
    detect_scenes,
    export_clip,
    ffmpeg_executable,
    safe_name,
    scene_has_only_target,
)

PROJECT = Path(__file__).resolve().parent.parent
REF_DIR = PROJECT / "data" / "reference_faces"
TEST_DIR = PROJECT / "data" / "quick_test"

# (display name, links file, reference image filename)
REPORTERS = [
    ("Fatih Portakal",    "reporter_links/links_fatih_portakal.txt",   "fatih.jpeg"),
    ("Kubra Par",         "reporter_links/links_kubra_par.txt",        "kubra.jpeg"),
    ("Gulsah Ekinci",     "reporter_links/links_gulsah_ekinci.txt",    "gulsah.jpeg"),
    ("Serdar Cebe",       "reporter_links/links_serdar_cebe.txt",      "serdar.jpeg"),
    ("Ece Uner",          "reporter_links/links_ece_uner.txt",         "ece.jpeg"),
    ("Ismail Kucukkaya",  "reporter_links/links_ismail_kucukkaya.txt", "ismail.jpeg"),
    ("Cem Ogretir",       "reporter_links/links_cem_ogretir.txt",      "cem_ogretir.jpeg"),
    ("Can Okanar",        "reporter_links/links_can_okanar.txt",       "can_okanar.jpeg"),
]

PROBE_N = 6          # how many videos to probe for the shortest
REF_SECONDS = 120    # length of reference-extraction clip
TEST_SECONDS = 240   # length of the test clip
REF_SAMPLES = 24     # frames sampled when learning the reference face
FACE_TOLERANCE = 0.47
SCENE_THRESHOLD = 30.0


def head_offset(duration: int) -> int:
    # News anchors read the headlines on-camera at the very start; sample there
    # (skipping only the short title sequence) instead of the footage-heavy middle.
    return 20 if duration and duration > 80 else 0


def ytdlp(args):
    return subprocess.run(
        [sys.executable, "-m", "yt_dlp", "--no-warnings", *args],
        capture_output=True, text=True,
    )


def read_urls(links_file: Path):
    urls = []
    for line in links_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            urls.append(line)
    return urls


def get_duration(url: str):
    res = ytdlp(["--print", "%(duration)s", url])
    val = res.stdout.strip().splitlines()
    if val and val[0].isdigit():
        return int(val[0])
    return None


def download_section(url: str, start: int, length: int, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    section = f"*{start}-{start + length}"
    res = ytdlp([
        "-f", "best[height<=720]/best",
        "--download-sections", section,
        "--force-keyframes-at-cuts",
        "--force-overwrites",
        "--no-part",
        "--no-playlist",
        "-o", str(out_path),
        url,
    ])
    return out_path.exists() and out_path.stat().st_size > 0, res


def clip_duration(path: Path):
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    cap.release()
    return frames / fps if fps else 0.0


def extract_reference_face(clip_path: Path, out_img: Path, samples: int = 12):
    """Sample frames, cluster faces, save the largest crop of the dominant face."""
    cap = cv2.VideoCapture(str(clip_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    if total <= 0:
        cap.release()
        return False

    clusters = []  # list of dict(enc, count, best_area, best_frame, best_loc)
    for i in range(samples):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(total * (i + 0.5) / samples))
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        locs = face_recognition.face_locations(rgb, model="hog")
        if len(locs) != 1:          # only learn from clean single-face frames
            continue
        enc = face_recognition.face_encodings(rgb, known_face_locations=locs)
        if not enc:
            continue
        enc = enc[0]
        top, right, bottom, left = locs[0]
        area = (bottom - top) * (right - left)
        placed = False
        for c in clusters:
            if face_recognition.compare_faces([c["enc"]], enc, tolerance=0.5)[0]:
                c["count"] += 1
                if area > c["best_area"]:
                    c.update(best_area=area, best_frame=rgb.copy(), best_loc=locs[0])
                placed = True
                break
        if not placed:
            clusters.append(dict(enc=enc, count=1, best_area=area,
                                 best_frame=rgb.copy(), best_loc=locs[0]))
    cap.release()
    if not clusters:
        return False

    dom = max(clusters, key=lambda c: c["count"])
    top, right, bottom, left = dom["best_loc"]
    h, w = dom["best_frame"].shape[:2]
    my, mx = int((bottom - top) * 0.4), int((right - left) * 0.4)
    crop = dom["best_frame"][max(0, top - my):min(h, bottom + my),
                             max(0, left - mx):min(w, right + mx)]
    out_img.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_img), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
    return True


def run_solo_filter(clip_path: Path, ref_img: Path, out_dir: Path):
    image = face_recognition.load_image_file(str(ref_img))
    encs = face_recognition.face_encodings(image)
    if not encs:
        return None
    ref_enc = encs[0]

    scenes = detect_scenes(clip_path, threshold=SCENE_THRESHOLD)
    if not scenes:  # short clip with no hard cuts -> treat as one scene
        scenes = [(0.0, clip_duration(clip_path))]

    cap = cv2.VideoCapture(str(clip_path))
    kept = 0
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        for idx, (s, e) in enumerate(scenes, 1):
            if e - s < 0.8:
                continue
            if scene_has_only_target(cap, s, e, ref_enc, sample_count=3,
                                     tolerance=FACE_TOLERANCE):
                export_clip(clip_path, out_dir / f"kept_{idx:03d}.mp4", s, e)
                kept += 1
    finally:
        cap.release()
    return len(scenes), kept


def main():
    print(f"{'Reporter':18} {'ref?':5} {'test vid':12} {'scenes':7} {'kept':5} status")
    print("-" * 70)
    for name, links_name, ref_name in REPORTERS:
        links_file = PROJECT / links_name
        slug = safe_name(name)
        work = TEST_DIR / slug
        try:
            urls = read_urls(links_file)
            if len(urls) < 2:
                print(f"{name:18} {'-':5} {'-':12} {'-':7} {'-':5} need >=2 urls")
                continue

            # probe first PROBE_N for durations -> pick shortest as test video
            durs = [(u, get_duration(u)) for u in urls[:PROBE_N]]
            durs = [(u, d) for u, d in durs if d]
            if not durs:
                print(f"{name:18} {'-':5} {'-':12} {'-':7} {'-':5} no durations")
                continue
            durs.sort(key=lambda x: x[1])
            test_url, test_dur = durs[0]
            ref_url = next((u for u, _ in durs if u != test_url), urls[0])
            ref_dur = dict(durs).get(ref_url, 600)

            # reference: opening section of a different video
            ref_clip = work / "ref_src.mp4"
            ok, _ = download_section(ref_url, head_offset(ref_dur),
                                     REF_SECONDS, ref_clip)
            ref_img = REF_DIR / ref_name
            ref_ok = ok and extract_reference_face(ref_clip, ref_img, REF_SAMPLES)
            if not ref_ok:
                print(f"{name:18} {'NO':5} {'-':12} {'-':7} {'-':5} reference extract failed")
                continue

            # test: opening section of the shortest video (anchor reads headlines)
            test_clip = work / "test.mp4"
            ok, _ = download_section(test_url, head_offset(test_dur),
                                     TEST_SECONDS, test_clip)
            if not ok:
                print(f"{name:18} {'YES':5} {'-':12} {'-':7} {'-':5} test download failed")
                continue

            result = run_solo_filter(test_clip, ref_img, work / "kept")
            vid = test_url.split("/")[-1][:11]
            if result is None:
                print(f"{name:18} {'YES':5} {vid:12} {'-':7} {'-':5} no face in reference")
            else:
                scenes, kept = result
                status = "OK works" if kept > 0 else "no solo match"
                print(f"{name:18} {'YES':5} {vid:12} {scenes:<7} {kept:<5} {status}")
        except Exception as exc:  # keep going across reporters
            print(f"{name:18} {'?':5} {'-':12} {'-':7} {'-':5} ERROR {type(exc).__name__}: {exc}")

    print("-" * 70)
    print(f"Reference photos: {REF_DIR}")
    print(f"Kept solo clips : {TEST_DIR}/<reporter>/kept/")


if __name__ == "__main__":
    main()
