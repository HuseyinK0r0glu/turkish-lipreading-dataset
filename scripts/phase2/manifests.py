"""Phase 2 inputs and outputs as CSV (PHASE2_PLAN §1.2, §2.1, §2.3).

Input: a SNAPSHOT of data/reporter_clips.csv, joined to video_links.csv on video id
for `channel`. Phase 1 may be appending to that manifest, so it is only ever read.

Output: data/phase2/*.csv, appended one source clip at a time. A source clip is done
once its face_track_report row exists -- that row is written last -- so a crash
mid-clip leaves rows that `drop_unfinished` removes on the next start.
"""
import csv
from pathlib import Path
from typing import Dict, Iterable, List, Set

PROJECT = Path(__file__).resolve().parents[2]
DATA = PROJECT / "data"
PHASE2_DIR = DATA / "phase2"
CROPS_DIR = DATA / "lip_crops"

TRACK_STATUS_FIELDS = ["good", "no_face", "multi_face", "jump", "bad_landmarks"]

SPINE = ["clip_id", "npz_path", "frame_start", "frame_end", "speaker_id", "channel",
         "source_clip", "source_video", "face_height_px", "mouth_width_px",
         "clip_start_sec", "clip_end_sec", "n_frames", "fps",
         "syncnet_score", "syncnet_offset", "lip_motion", "mean_yaw", "mean_pitch"]
WORD_FIELDS = SPINE + ["surface_form", "lemma", "word_start_sec", "word_end_sec",
                       "confidence", "manual_review", "word_sync_conf", "track_ok"]
SENTENCE_FIELDS = SPINE + ["text", "sentence_start_sec", "sentence_end_sec", "split_index",
                           "n_words", "sentence_sync_conf", "word_spans"]
SYNC_FIELDS = ["source_clip", "speaker_id", "channel", "syncnet_score", "syncnet_offset",
               "min_dist", "passed"]
DISCARD_FIELDS = ["source_clip", "speaker_id", "level", "item_id", "text", "reason", "detail"]
REPORT_FIELDS = (["source_clip", "speaker_id", "channel", "status", "n_frames", "good_frames"]
                 + TRACK_STATUS_FIELDS[1:]
                 + ["mouth_width_px", "words_total", "words_kept", "segments_total",
                    "sentences_kept", "seconds"])

OUTPUTS = {
    "word_clips.csv": WORD_FIELDS,
    "sentence_clips.csv": SENTENCE_FIELDS,
    "syncnet_scores.csv": SYNC_FIELDS,
    "discarded.csv": DISCARD_FIELDS,
    "face_track_report.csv": REPORT_FIELDS,  # written LAST per source clip
}


def rel(path: Path) -> str:
    """Posix path relative to the repo, like reporter_clips.csv's paths."""
    return Path(path).resolve().relative_to(PROJECT).as_posix()


def stem_of(clip_file: str) -> str:
    return Path(clip_file).stem


def read_csv(path: Path) -> List[dict]:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ---------- input ----------

def snapshot_clips(clips_csv: Path, links_csv: Path) -> List[dict]:
    """reporter_clips.csv rows with a `channel` key joined from video_links.csv."""
    from presenter_filter import extract_video_id
    from run_all import read_clip_rows, read_rows

    channel_by_id = {}
    for r in read_rows(links_csv):
        vid = extract_video_id(r.get("url") or "")
        if vid and r.get("channel"):
            channel_by_id.setdefault(vid, r["channel"].strip())
    rows = []
    for r in read_clip_rows(clips_csv):
        vid = extract_video_id(r.get("source_url") or "")
        rows.append(dict(r, channel=channel_by_id.get(vid, "")))
    return rows


def select(rows: List[dict], reporters=None, exclude_reporters=None, limit=None) -> List[dict]:
    """Same filter-then-limit order as run_all.select_rows."""
    if reporters:
        rows = [r for r in rows if r["reporter"] in reporters]
    if exclude_reporters:
        rows = [r for r in rows if r["reporter"] not in exclude_reporters]
    return rows[:limit] if limit else rows


# ---------- output ----------

def done_clips(out_dir: Path) -> Set[str]:
    """clip_file of every source clip already finished."""
    return {r["source_clip"] for r in read_csv(out_dir / "face_track_report.csv")}


def strip_nul_tail(path: Path) -> int:
    """Cut the NUL padding a power cut leaves at the end of a file being appended.

    NTFS had already extended the file but the last block's data never reached the
    disk, so the tail reads back as zeros. Left in, it parses as one junk row whose
    source_clip is a run of NULs -- in face_track_report that row counts as "done".
    Also drops a final line cut off mid-row. Returns the bytes removed.
    """
    if not path.exists():
        return 0
    data = path.read_bytes()
    clean = data.rstrip(b"\0")
    if not clean.endswith(b"\n"):
        clean = clean[:clean.rfind(b"\n") + 1]
    if len(clean) != len(data):
        path.write_bytes(clean)
    return len(data) - len(clean)


def drop_unfinished(out_dir: Path) -> int:
    """Remove rows of source clips that never got their report row (a crash mid-write)."""
    for name in OUTPUTS:
        cut = strip_nul_tail(out_dir / name)
        if cut:
            print(f"[phase2] {name}: cut {cut} bytes of power-loss padding")
    done = done_clips(out_dir)
    removed = 0
    for name, fields in OUTPUTS.items():
        path = out_dir / name
        rows = read_csv(path)
        keep = [r for r in rows if r["source_clip"] in done]
        if len(keep) != len(rows):
            removed += len(rows) - len(keep)
            _write(path, fields, keep, mode="w")
    return removed


def _write(path: Path, fields: List[str], rows: Iterable[dict], mode: str = "a") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new = mode == "w" or not path.exists()
    with open(path, mode, newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerows(rows)


def write_result(out_dir: Path, clip_row: dict, npz_path: Path, result: Dict) -> None:
    """Append every row one processed source clip produced; the report row last."""
    source = clip_row["clip_file"]
    sync = result.get("sync") or {}
    spine = {
        "npz_path": rel(npz_path),
        "speaker_id": clip_row["reporter"],
        "channel": clip_row.get("channel", ""),
        "source_clip": source,
        "source_video": clip_row.get("source_video", ""),
        "face_height_px": clip_row.get("face_height_px", ""),
        "fps": 25,
        "syncnet_score": sync.get("syncnet_score", ""),
        "syncnet_offset": sync.get("syncnet_offset", ""),
    }
    _write(out_dir / "word_clips.csv", WORD_FIELDS, [{**spine, **w} for w in result["words"]])
    _write(out_dir / "sentence_clips.csv", SENTENCE_FIELDS,
           [{**spine, **s} for s in result["sentences"]])
    if sync:
        _write(out_dir / "syncnet_scores.csv", SYNC_FIELDS, [{
            "source_clip": source, "speaker_id": clip_row["reporter"],
            "channel": clip_row.get("channel", ""), **sync,
            "passed": int(result["status"] == "ok")}])
    _write(out_dir / "discarded.csv", DISCARD_FIELDS,
           [{"source_clip": source, "speaker_id": clip_row["reporter"], **d}
            for d in result["discards"]])
    track = result.get("track") or {}
    _write(out_dir / "face_track_report.csv", REPORT_FIELDS, [{
        "source_clip": source, "speaker_id": clip_row["reporter"],
        "channel": clip_row.get("channel", ""), "status": result["status"],
        **{k: track.get(k, "") for k in ["n_frames", "good_frames", "mouth_width_px"]
           + TRACK_STATUS_FIELDS[1:]},
        "words_total": result["n_words"], "words_kept": len(result["words"]),
        "segments_total": result["n_segments"], "sentences_kept": len(result["sentences"]),
        "seconds": result.get("seconds", ""),
    }])
