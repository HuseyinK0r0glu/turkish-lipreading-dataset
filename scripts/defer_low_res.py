"""Move pending rows whose best video stream is under 720p to the end of the CSV.

`DOWNLOAD_FORMAT` demands >=720p, so a sub-720p source fails with "Requested
format is not available", the row stays pending, and every later run retries it --
14 such rows sat at the head of the queue and were re-probed on each restart. They
are cheap individually (a metadata fetch, no download) but they are noise in the
log and they never succeed. Half of enver_aysever's 176 pending rows measured
sub-720p.

Rows are moved, never deleted: if the >=720p requirement is ever relaxed, or if a
channel re-uploads, they are still there. Completed rows are untouched.

The measured height is cached in data/_row_height_cache.json, so a rerun only
probes rows it has not seen.

    uv run python scripts/defer_low_res.py --dry-run
    uv run python scripts/defer_low_res.py --reporter enver_aysever
    uv run python scripts/defer_low_res.py
"""

import argparse
import collections
import csv
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from presenter_filter import extract_video_id, yt_dlp_base_cmd, cookie_args  # noqa: E402

VIDEO_LINKS = ROOT / "video_links.csv"
CACHE = ROOT / "data" / "_row_height_cache.json"
LINK_FIELDS = ["url", "channel", "start", "end", "pipeline", "reporter", "completed"]
CHUNK = 60


def probe(urls, cache):
    """Probe `urls` into `cache`, persisting after every chunk.

    Written per chunk, not once at the end: probing 1700 rows takes the better part
    of an hour, and a single write at the end means an interrupt throws all of it
    away. The cache is the only reason a rerun is cheap.
    """
    out = {}
    for i in range(0, len(urls), CHUNK):
        batch = urls[i:i + CHUNK]
        bf = ROOT / "data" / "_height_batch.txt"
        bf.write_text("\n".join(batch), encoding="utf-8")
        cmd = [*yt_dlp_base_cmd(), *cookie_args(), "--no-warnings", "--skip-download",
               "--ignore-errors", "--socket-timeout", "20", "--sleep-requests", "0.3",
               "--print", "%(id)s\t%(height)s", "--batch-file", str(bf)]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                               errors="replace", cwd=ROOT, timeout=3600)
        except subprocess.TimeoutExpired:
            print("  probe chunk timed out, skipping"); continue
        for line in r.stdout.splitlines():
            f = line.split("\t")
            if len(f) < 2:
                continue
            try:
                out[f[0]] = int(float(f[1]))
            except ValueError:
                out[f[0]] = 0
        bf.unlink(missing_ok=True)
        cache.update(out)
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        CACHE.write_text(json.dumps(cache), encoding="utf-8")
        print(f"  probed {min(i + CHUNK, len(urls))}/{len(urls)}", flush=True)
    return out


def write_all(rows):
    tmp = VIDEO_LINKS.with_suffix(".csv.tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=LINK_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: (r.get(k) or "") for k in LINK_FIELDS})
    tmp.replace(VIDEO_LINKS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--reporter", action="append", help="limit probing to these")
    ap.add_argument("--min-height", type=int, default=720)
    args = ap.parse_args()

    with open(VIDEO_LINKS, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    cache = json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.exists() else {}
    print(f"cache: {len(cache)} rows already probed")

    pending = [r for r in rows
               if (r.get("pipeline") or "").strip() == "2"
               and (r.get("completed") or "").strip() != "1"]
    if args.reporter:
        pending = [r for r in pending
                   if (r.get("reporter") or "").strip() in set(args.reporter)]

    need = []
    for r in pending:
        vid = extract_video_id((r.get("url") or "").strip())
        if vid and vid not in cache:
            need.append(r["url"])
    print(f"{len(pending)} pending rows in scope, {len(need)} to probe")

    # Dry run still probes: without it the first dry run has nothing to report,
    # which is the one case where a preview actually matters.
    if need:
        probe(need, cache)

    low = set()
    unknown = 0
    for r in pending:
        vid = extract_video_id((r.get("url") or "").strip())
        h = cache.get(vid)
        if h is None:
            unknown += 1
        elif h < args.min_height:
            low.add(id(r))

    by = collections.Counter((r.get("reporter") or "") for r in pending if id(r) in low)
    print(f"\nunder {args.min_height}p: {len(low)} rows  (unprobed: {unknown})")
    for rep, n in by.most_common():
        tot = sum(1 for r in pending if (r.get("reporter") or "") == rep)
        print(f"   {rep:20} {n:4d}/{tot:<4d} ({100*n/tot:.0f}%)")

    if args.dry_run:
        print("dry run, nothing written")
        return
    if not low:
        print("nothing to move")
        return

    # completed first (run_all skips them), then usable pending, then the low-res tail
    done = [r for r in rows if (r.get("completed") or "").strip() == "1"]
    keep = [r for r in rows
            if (r.get("completed") or "").strip() != "1" and id(r) not in low]
    tail = [r for r in rows
            if (r.get("completed") or "").strip() != "1" and id(r) in low]
    assert len(done) + len(keep) + len(tail) == len(rows)
    write_all(done + keep + tail)
    print(f"moved {len(tail)} sub-{args.min_height}p rows to the end; "
          f"{len(keep)} pending rows run first")


if __name__ == "__main__":
    main()
