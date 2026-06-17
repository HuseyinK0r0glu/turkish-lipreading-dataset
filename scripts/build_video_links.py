"""(Re)build video_links.csv as the single source for both pipelines.

Columns: url, channel, start, end, pipeline, reporter
  pipeline 1 rows -> kept from the existing video_links.csv (download + Whisper)
  pipeline 2 rows -> regenerated from the per-reporter links_*.txt files in reporter_links/

Idempotent: existing pipeline-1 rows are preserved; pipeline-2 rows are always
rebuilt from the txt files, so you can edit the txts and re-run this safely.

Run:  venv/bin/python scripts/build_video_links.py
"""

import csv
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
VIDEO_LINKS = PROJECT / "video_links.csv"
FIELDS = ["url", "channel", "start", "end", "pipeline", "reporter", "completed"]

# reporter slug -> (links file, broadcaster channel name)
REPORTERS = {
    "fatih_portakal":   ("reporter_links/links_fatih_portakal.txt",   "Fatih Portakal TV"),
    "kubra_par":        ("reporter_links/links_kubra_par.txt",        "TV100"),
    "gulsah_ekinci":    ("reporter_links/links_gulsah_ekinci.txt",    "Halk TV"),
    "serdar_cebe":      ("reporter_links/links_serdar_cebe.txt",      "Sözcü TV"),
    "ece_uner":         ("reporter_links/links_ece_uner.txt",         "Halk TV"),
    "ismail_kucukkaya": ("reporter_links/links_ismail_kucukkaya.txt", "Halk TV"),
    "cuneyt_ozdemir":   ("reporter_links/links_cuneyt_ozdemir.txt",   "Cüneyt Özdemir"),
    "cem_ogretir":      ("reporter_links/links_cem_ogretir.txt",      "atv Haber"),
    "can_okanar":       ("reporter_links/links_can_okanar.txt",       "A Haber"),
}


def load_existing_rows():
    """Return (pipeline1_rows, completed_by_url) from the current CSV, if any.

    completed_by_url preserves the 'completed' flag across rebuilds so a rebuild
    never re-queues videos that were already fully processed.
    """
    if not VIDEO_LINKS.exists():
        return [], {}
    with open(VIDEO_LINKS, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    completed_by_url = {}
    kept = []
    for r in rows:
        url = (r.get("url") or "").strip()
        completed = (r.get("completed") or "").strip()
        if url and completed:
            completed_by_url[url] = completed
        # rows from the original 4-column file have no 'pipeline' -> treat as 1
        if (r.get("pipeline") or "1").strip() == "1":
            kept.append({
                "url": url,
                "channel": (r.get("channel") or "").strip(),
                "start": (r.get("start") or "").strip(),
                "end": (r.get("end") or "").strip(),
                "pipeline": "1",
                "reporter": "",
                "completed": completed,
            })
    return kept, completed_by_url


def pipeline2_rows(completed_by_url):
    rows = []
    for slug, (links_name, channel) in REPORTERS.items():
        links_file = PROJECT / links_name
        if not links_file.exists():
            print(f"[warn] missing {links_name}")
            continue
        for line in links_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            rows.append({"url": line, "channel": channel, "start": "", "end": "",
                         "pipeline": "2", "reporter": slug,
                         "completed": completed_by_url.get(line, "")})
    return rows


def main():
    pipeline1, completed_by_url = load_existing_rows()
    rows = pipeline1 + pipeline2_rows(completed_by_url)
    with open(VIDEO_LINKS, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    p1 = sum(1 for r in rows if r["pipeline"] == "1")
    p2 = sum(1 for r in rows if r["pipeline"] == "2")
    print(f"Wrote {VIDEO_LINKS.name}: {len(rows)} rows  (pipeline1={p1}, pipeline2={p2})")


if __name__ == "__main__":
    main()
