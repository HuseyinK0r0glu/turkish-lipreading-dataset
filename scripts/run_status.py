"""Snapshot of a running (or finished) run_all.py job.

Read-only: it touches nothing but the files the run already writes, so it is safe
to call at any moment against a live run.

    uv run python scripts/run_status.py
    uv run python scripts/run_status.py --follow        # refresh every 30 s
    uv run python scripts/run_status.py --follow 120    # ... every 120 s

The ETA is deliberately reported twice. The naive average over the whole run is
misleading here: short reporter uploads and 2-hour ana haber broadcasts differ by
an order of magnitude in cost, so a run that just finished a batch of short videos
reports an ETA several times too optimistic. The second figure uses only the last
few rows, which is what the remaining ones will actually look like.
"""

import csv
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
LINKS = PROJECT / "video_links.csv"
DATA = PROJECT / "data"
CLIPS_CSV = DATA / "reporter_clips.csv"
LOG = DATA / "run_all.log"
ERR = DATA / "run_all.err"
RECENT_ROWS = 5          # window for the "recent" rate
STALE_AFTER = 20 * 60    # no log write for this long -> probably not running


def read_csv(path):
    """Tolerant read: the run appends to these files while we look at them."""
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        try:
            return list(csv.DictReader(f))
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


def row_finish_times(clips):
    """When each source video finished, from its transcripts' mtimes.

    The log carries no timestamps, but a row's per-clip transcripts are written
    last, so the newest one dates the row's completion.
    """
    per_video = defaultdict(float)
    for r in clips:
        p = PROJECT / (r.get("transcript_json") or "")
        try:
            per_video[r["source_video"]] = max(per_video[r["source_video"]], p.stat().st_mtime)
        except OSError:
            continue
    return sorted(t for t in per_video.values() if t)


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
    print(f"=== {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")

    # --- liveness -------------------------------------------------------
    mtimes = [p.stat().st_mtime for p in (LOG, ERR) if p.exists()]
    if not mtimes:
        print("data/run_all.log yok - run bu makinede loglanarak baslatilmamis.")
        return
    idle = time.time() - max(mtimes)
    state = "CALISIYOR" if idle < STALE_AFTER else f"DURMUS OLABILIR (son cikti {idle/60:.0f} dk once)"
    print(f"Durum : {state}   (son log yazimi {idle/60:.1f} dk once)")

    # --- CSV progress ---------------------------------------------------
    rows = read_csv(LINKS)
    log = LOG.read_text(encoding="utf-8", errors="replace") if LOG.exists() else ""
    marks = re.findall(r"^\[pipeline2\] \[(\d+)/(\d+)\]", log, re.M)
    cur, total = (int(marks[-1][0]), int(marks[-1][1])) if marks else (0, 0)
    fails = len(re.findall(r"^\[pipeline2\] \[\d+/\d+\] FAILED", log, re.M))

    # Which reporters this run covers. The log does not record the --reporters /
    # --exclude-reporters flags, but a reporter with more pending rows than the run
    # has rows in total cannot possibly be part of it - that is what filters out
    # cuneyt_ozdemir (4407 rows) on an --exclude-reporters run without guessing.
    p2 = [r for r in rows if (r.get("pipeline") or "").strip() == "2"]
    pending = Counter((r.get("reporter") or "").strip() for r in p2
                      if (r.get("completed") or "").strip() != "1")
    sel = [r for r in p2
           if not total or pending[(r.get("reporter") or "").strip()] <= total]
    done = sum(1 for r in sel if (r.get("completed") or "").strip() == "1")
    # `done` counts the CSV flag, so it includes rows finished by earlier runs;
    # `cur` counts only what this process walked past. They differ after a restart.
    print(f"Ilerleme: bu run {cur}/{total} satir isledi ({fails} basarisiz)")
    print(f"          CSV toplami {done}/{len(sel)} satir tamam (onceki runlar dahil)\n")

    seen = set(re.findall(r"^\[pipeline2\] \[\d+/\d+\] (?:FAILED )?(\w+)", log, re.M))
    c = Counter(((r.get("reporter") or "").strip(),
                 (r.get("completed") or "").strip() == "1") for r in sel)
    for rp in sorted({k[0] for k in c}):
        t, k = c[(rp, True)], c[(rp, False)]
        flag = "<" if rp in seen and k else " "
        print(f"  {rp:18} [{bar(t, t + k)}] {t:3d}/{t + k} {flag}")

    # --- output ---------------------------------------------------------
    clips = read_csv(CLIPS_CSV)
    if clips:
        tot = sum(float(r["duration_sec"]) for r in clips if r.get("duration_sec"))
        print(f"\nUretilen: {len(clips)} klip / {tot/3600:.2f} saat malzeme")
        by = Counter(r["reporter"] for r in clips)
        for k, v in by.most_common():
            d = sum(float(r["duration_sec"]) for r in clips
                    if r["reporter"] == k and r.get("duration_sec"))
            print(f"  {k:18} {v:4d} klip  {d/3600:5.2f} h")

    # --- rate + ETA -----------------------------------------------------
    finishes = row_finish_times(clips)
    remaining = total - cur
    if len(finishes) >= 2 and remaining > 0:
        span = (finishes[-1] - finishes[0]) / 3600
        overall = (len(finishes) - 1) / span if span else 0
        print(f"\nHiz (tum run) : {overall:.2f} satir/saat -> kalan {remaining} satir "
              f"~{human(remaining / overall) if overall else '?'}")
        w = finishes[-min(RECENT_ROWS + 1, len(finishes)):]
        rspan = (w[-1] - w[0]) / 3600
        recent = (len(w) - 1) / rspan if rspan else 0
        if recent:
            print(f"Hiz (son {len(w)-1})    : {recent:.2f} satir/saat -> kalan {remaining} satir "
                  f"~{human(remaining / recent)}   <- gercekci olan bu")

    # --- what it is chewing on right now --------------------------------
    raw = sorted((DATA / "pipeline2_raw").glob("*")) if (DATA / "pipeline2_raw").exists() else []
    if raw:
        f = raw[-1]
        print(f"\nSu an isleniyor: {f.name[:70]} ({f.stat().st_size/1e9:.2f} GB)")
    if ERR.exists():
        tail = ERR.read_text(encoding="utf-8", errors="replace")[-600:].replace("\r", "\n")
        last = [l for l in tail.splitlines() if l.strip()]
        if last:
            print(f"Asama          : {last[-1].strip()[:100]}")
    for line in [l for l in log.splitlines() if l.startswith("[pipeline2] [")][-3:]:
        print(f"  {line[:110]}")


if __name__ == "__main__":
    main()
