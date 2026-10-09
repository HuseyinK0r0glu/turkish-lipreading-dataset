"""Snapshot of a running (or finished) run_phase2.py job -- run_status.py for Phase 2.

Read-only: it reads only the files the run already writes, so it is safe to call
at any moment against a live run.

    uv run python scripts/phase2/phase2_status.py
    uv run python scripts/phase2/phase2_status.py --follow        # refresh every 30 s
    uv run python scripts/phase2/phase2_status.py --follow 120    # ... every 120 s

Progress comes from face_track_report.csv (one row per finished source clip, across
every run), not from the log, so it survives restarts. The log carries no
timestamps, so the rate is taken from the mtimes of the crop .npz files the most
recent clips wrote, and it stops at the first long gap -- a window reaching back
across a power cut or a restart would otherwise average in the hours the machine
was off.
"""

import csv
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
csv.field_size_limit(2**31 - 1)  # sentence rows carry word_spans JSON

PROJECT = Path(__file__).resolve().parents[2]
DATA = PROJECT / "data"
CLIPS_CSV = DATA / "reporter_clips.csv"
OUT = DATA / "phase2"
REPORT = OUT / "face_track_report.csv"
CROPS = DATA / "lip_crops"
RECENT_CLIPS = 400       # window for the rate
GAP = 10 * 60            # a pause this long between two clips ends the rate window
STALE_AFTER = 10 * 60    # no clip finished for this long -> probably not running
LOG_TAIL = 256 * 1024


def read_csv(path):
    """Tolerant read: the run appends to these files while we look at them."""
    if not path.exists():
        return []
    text = path.read_bytes().rstrip(b"\0").decode("utf-8", errors="replace")
    try:
        return list(csv.DictReader(text.splitlines()))
    except csv.Error:
        return []


def bar(done, total, width=22):
    if not total:
        return "-" * width
    filled = int(width * done / total)
    return "#" * filled + "." * (width - filled)


def human(hours):
    if hours < 1:
        return f"{hours * 60:.0f} dk"
    if hours < 48:
        return f"{hours:.1f} saat"
    return f"{hours / 24:.1f} gun"


def latest_log():
    logs = sorted(DATA.glob("phase2_run*.log"), key=lambda p: p.stat().st_mtime)
    return logs[-1] if logs else None


def log_tail(path):
    """Last [phase2] lines of the log. `*>` in Windows PowerShell 5.1 writes UTF-16."""
    with open(path, "rb") as f:
        bom = f.read(2)
        size = f.seek(0, 2)
        f.seek(max(2, size - LOG_TAIL))
        raw = f.read()
    if bom in (b"\xff\xfe", b"\xfe\xff"):
        enc = "utf-16-le" if bom == b"\xff\xfe" else "utf-16-be"
        raw = raw[len(raw) % 2:] if (size - len(raw)) % 2 else raw
        text = raw.decode(enc, errors="replace")
    else:
        text = raw.decode("utf-8", errors="replace")
    return [l.strip() for l in text.replace("\r", "\n").splitlines()
            if l.startswith("[phase2]")]


def finish_times(report, crop_path):
    """(finish mtime, row index) of the newest clips that wrote a crop file."""
    out = []
    for i in range(len(report) - 1, max(-1, len(report) - 1 - RECENT_CLIPS), -1):
        p = crop_path(report[i])
        try:
            out.append((p.stat().st_mtime, i))
        except OSError:
            continue  # low_sync / no_words clips write no crops
    out.sort()
    # keep only the stretch after the last long pause
    for k in range(len(out) - 1, 0, -1):
        if out[k][0] - out[k - 1][0] > GAP:
            return out[k:]
    return out


def main():
    follow = "--follow" in sys.argv
    every = 30
    for a in sys.argv[1:]:
        if a.isdigit():
            every = int(a)

    while True:
        if follow:
            print("\033[2J\033[H", end="")
        report()
        if not follow:
            return
        print(f"\n({every} sn'de bir yenileniyor - Ctrl+C ile cik)")
        time.sleep(every)


def report():
    print(f"=== Phase 2  {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
    if not REPORT.exists():
        print("data/phase2/face_track_report.csv yok - phase 2 henuz hic klip bitirmemis.")
        return

    # --- liveness -------------------------------------------------------
    log = latest_log()
    mtimes = [REPORT.stat().st_mtime] + ([log.stat().st_mtime] if log else [])
    idle = time.time() - max(mtimes)
    state = "CALISIYOR" if idle < STALE_AFTER else f"DURMUS OLABILIR (son cikti {idle/60:.0f} dk once)"
    print(f"Durum : {state}   (son yazim {idle/60:.1f} dk once)")

    # --- progress -------------------------------------------------------
    clips = read_csv(CLIPS_CSV)
    dur = {r["clip_file"]: float(r["duration_sec"] or 0) for r in clips}
    total_by = Counter(r["reporter"] for r in clips)
    hours_by = defaultdict(float)
    for r in clips:
        hours_by[r["reporter"]] += dur[r["clip_file"]] / 3600

    rep = [r for r in read_csv(REPORT) if r.get("source_clip") in dur]
    done_by = Counter(r["speaker_id"] for r in rep)
    done_h_by = defaultdict(float)
    words_by = Counter()
    for r in rep:
        done_h_by[r["speaker_id"]] += dur[r["source_clip"]] / 3600
        words_by[r["speaker_id"]] += int(r.get("words_kept") or 0)
    tot_h, done_h = sum(hours_by.values()), sum(done_h_by.values())

    print(f"Ilerleme: {len(rep)}/{len(clips)} klip  [{bar(len(rep), len(clips), 30)}] "
          f"{100 * len(rep) / max(len(clips), 1):.1f}%")
    print(f"          {done_h:.1f}/{tot_h:.1f} saat malzeme\n")

    st = Counter(r["status"] for r in rep)
    words = sum(int(r.get("words_kept") or 0) for r in rep)
    words_all = sum(int(r.get("words_total") or 0) for r in rep)
    sents = sum(int(r.get("sentences_kept") or 0) for r in rep)
    print("Sonuc  : " + "  ".join(f"{k} {v}" for k, v in st.most_common()))
    print(f"Kelime : {words:,} klip tutuldu / {words_all:,} kelime "
          f"({100 * words / max(words_all, 1):.1f}%)   Cumle: {sents:,}\n")

    print(f"  {'muhabir':18} {'klip':>24}  {'saat':>11}  {'kelime':>8}  ok%")
    ok_by = Counter(r["speaker_id"] for r in rep if r["status"] == "ok")
    for rp in sorted(total_by):
        d, t = done_by[rp], total_by[rp]
        flag = "<" if 0 < d < t else " "
        okp = f"{100 * ok_by[rp] / d:3.0f}" if d else "  -"
        print(f"  {rp:18} [{bar(d, t, 12)}] {d:5d}/{t:<5d} "
              f"{done_h_by[rp]:5.1f}/{hours_by[rp]:<5.1f} {words_by[rp]:8d}  {okp} {flag}")

    # --- rate + ETA -----------------------------------------------------
    stem = lambda r: Path(r["source_clip"]).stem  # noqa: E731
    win = finish_times(rep, lambda r: CROPS / r["speaker_id"] / f"{stem(r)}.npz")
    left_clips, left_h = len(clips) - len(rep), tot_h - done_h
    if len(win) >= 2 and left_clips > 0:
        (t0, i0), (t1, i1) = win[0], win[-1]
        span_h = (t1 - t0) / 3600
        n = i1 - i0  # report rows between the two, crop-less clips included
        audio_h = sum(dur[rep[i]["source_clip"]] for i in range(i0 + 1, i1 + 1)) / 3600
        if span_h > 0 and n > 0:
            print(f"\nHiz (son {n} klip, {span_h * 60:.0f} dk): {n / span_h:.0f} klip/saat, "
                  f"{audio_h / span_h:.1f}x realtime")
            print(f"Kalan: {left_clips} klip / {left_h:.1f} saat malzeme -> "
                  f"~{human(left_h / (audio_h / span_h))}")

    # --- log ------------------------------------------------------------
    if log:
        lines = log_tail(log)
        fails = [l for l in lines if l.startswith("[phase2] FAILED")]
        print(f"\nLog: {log.relative_to(PROJECT).as_posix()}")
        for l in [l for l in lines if re.match(r"\[phase2\] \d+/\d+", l)][-3:]:
            print(f"  {l[:130]}")
        if fails:
            print(f"  son hatalar ({len(fails)} log sonunda):")
            for l in fails[-3:]:
                print(f"    {l[:126]}")
        if lines and lines[-1].startswith("[phase2] finished"):
            print(f"  {lines[-1]}")


if __name__ == "__main__":
    main()
