"""Phase 3 step 6, after annotation: agreement, verified labels, corrected spotting GT.

Reads the Label Studio JSON exports (the full "JSON" format, not JSON-MIN) from
data/phase3/verification/exports/ and writes:

  verification/kappa.txt             Cohen's kappa per item kind on the doubly-annotated
                                     overlap, raw agreement, verdict shares per split
  verification/verified.csv          clip_id, kind, n_annotators, verdicts, verified, label_fix
  spotting_gt/test_gt_verified.csv   the hand-corrected test occurrences, per class/labeling

Nothing is overwritten in the manifests: `verified` / `label_fix` live in their own file,
keyed by clip_id, and make_release.py joins them on.

Annotators are told apart by export file when several files hold the same task
(tasks_*_A / tasks_*_B imported into separate projects), else by Label Studio's
`completed_by`.

    uv run python scripts/phase3/score_kappa.py
    uv run python scripts/phase3/score_kappa.py --exports a.json b.json
"""
import argparse
import json
from collections import defaultdict
from itertools import combinations

import pandas as pd
from sklearn.metrics import cohen_kappa_score

import common as C


def annotator_of(ann, file_tag):
    by = ann.get("completed_by")
    if isinstance(by, dict):
        by = by.get("email") or by.get("id")
    return f"{file_tag}:{by}"


def read_exports(paths):
    """-> list of dicts: kind, clip_id, annotator, verdict, label_fix, start, end, data."""
    rows = []
    for path in paths:
        tasks = json.loads(path.read_text(encoding="utf-8"))
        if tasks and "annotations" not in tasks[0]:
            raise SystemExit(f"{path.name}: export as 'JSON', not JSON-MIN / CSV")
        for task in tasks:
            data = task.get("data", {})
            for ann in task.get("annotations", []):
                if ann.get("was_cancelled"):
                    continue
                row = {"kind": data.get("kind", ""), "clip_id": data.get("clip_id", ""),
                       "annotator": annotator_of(ann, path.stem), "verdict": "",
                       "label_fix": "", "start": None, "end": None, "data": data}
                for res in ann.get("result", []):
                    value = res.get("value", {})
                    if res.get("from_name") == "verdict" and value.get("choices"):
                        row["verdict"] = value["choices"][0]
                    elif res.get("from_name") == "label_fix" and value.get("text"):
                        row["label_fix"] = " ".join(value["text"]).strip()
                    elif res.get("from_name") == "occ" and "start" in value:
                        row["start"], row["end"] = float(value["start"]), float(value["end"])
                if row["verdict"]:
                    rows.append(row)
    return rows


def agreement(rows, kind, rep):
    by_item = defaultdict(dict)
    for r in rows:
        if r["kind"] == kind:
            by_item[r["clip_id"]][r["annotator"]] = r["verdict"]
    annotators = sorted({a for v in by_item.values() for a in v})
    rep(f"\n[{kind}] {len(by_item):,} items annotated by {len(annotators)} annotator(s)")
    found = False
    for a, b in combinations(annotators, 2):
        both = [(v[a], v[b]) for v in by_item.values() if a in v and b in v]
        if len(both) < 2:
            continue
        found = True
        ya, yb = zip(*both)
        raw = sum(x == y for x, y in both) / len(both)
        kappa = cohen_kappa_score(ya, yb) if len(set(ya) | set(yb)) > 1 else float("nan")
        rep(f"  {a} vs {b}: {len(both):,} shared items, raw agreement {raw:.3f}, "
            f"Cohen's kappa {kappa:.3f}")
        rep("  confusion (rows " + a + ", columns " + b + "):")
        rep("    " + pd.crosstab(pd.Series(ya, name=""), pd.Series(yb, name=""))
            .to_string().replace("\n", "\n    "))
    if not found:
        rep("  no doubly-annotated items: kappa not computable")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--phase3", type=C.Path, default=C.PHASE3)
    ap.add_argument("--exports", type=C.Path, nargs="+",
                    help="Label Studio JSON exports (default: verification/exports/*.json)")
    args = ap.parse_args()
    C.utf8_stdout()
    rep = C.Report()
    ver = args.phase3 / "verification"
    paths = args.exports or sorted((ver / "exports").glob("*.json"))
    if not paths:
        raise SystemExit(f"no exports in {(ver / 'exports').as_posix()} - annotate first "
                         f"(see {(ver / 'README.md').as_posix()})")
    rows = read_exports(paths)
    rep(f"Phase 3 score_kappa: {len(rows):,} annotations from {len(paths)} export file(s)")

    for kind in ["word", "sentence"]:
        agreement(rows, kind, rep)

    # ---- verified.csv: one row per item ----
    items = defaultdict(list)
    for r in rows:
        if r["kind"] in ("word", "sentence"):
            items[(r["kind"], r["clip_id"])].append(r)
    out = []
    for (kind, clip_id), anns in items.items():
        verdicts = sorted({a["verdict"] for a in anns})
        fixes = [a["label_fix"] for a in anns if a["label_fix"]]
        out.append({"clip_id": clip_id, "kind": kind,
                    "split": anns[0]["data"].get("split", ""),
                    "speaker_id": anns[0]["data"].get("speaker_id", ""),
                    "n_annotators": len({a["annotator"] for a in anns}),
                    "verdicts": "|".join(verdicts),
                    "verified": int(verdicts == ["correct"]),
                    "label_fix": fixes[0] if fixes else ""})
    verified = pd.DataFrame(out, columns=["clip_id", "kind", "split", "speaker_id",
                                          "n_annotators", "verdicts", "verified", "label_fix"])
    C.write_csv(verified, ver / "verified.csv")
    for kind in ["word", "sentence"]:
        v = verified[verified.kind == kind]
        if v.empty:
            continue
        rep(f"\n[{kind}] label accuracy on the sample (verified = every annotator said "
            f"correct): {v.verified.mean():.1%} of {len(v):,}")
        rep("  by split: " + ", ".join(f"{k} {g.verified.mean():.1%} (n={len(g)})"
                                       for k, g in v.groupby("split")))
        rep("  verdicts: " + ", ".join(f"{k} {n}" for k, n in v.verdicts.value_counts().items()))

    # ---- corrected spotting ground truth ----
    spot = [r for r in rows if r["kind"] == "spotting"]
    if spot:
        g = pd.DataFrame([{
            "clip_id": r["clip_id"], "auto_start_sec": str(r["data"].get("auto_start_sec", "")),
            "auto_end_sec": str(r["data"].get("auto_end_sec", "")),
            "present": int(r["verdict"] == "present"), "verdict": r["verdict"],
            "start_sec": r["start"], "end_sec": r["end"]} for r in spot])
        g = g.drop_duplicates(["clip_id", "auto_start_sec", "auto_end_sec"])
        q = pd.read_csv(C.require(args.phase3 / "spotting_gt" / "queries.csv",
                                  "build_spotting_set.py"), dtype=str, keep_default_na=False)
        q = q[q.split == "test"].rename(columns={"start_sec": "auto_start_sec",
                                                 "end_sec": "auto_end_sec"})
        gt = q[["class", "labeling", "clip_id", "auto_start_sec", "auto_end_sec",
                "surface_form"]].merge(g, on=["clip_id", "auto_start_sec", "auto_end_sec"])
        gt = gt[gt.verdict != "unsure"]
        has = gt.present == 1
        gt.loc[~has, ["start_sec", "end_sec"]] = None
        gt["start_frame"] = (gt.start_sec * C.FPS).round().astype("Int64")
        gt["end_frame"] = (gt.end_sec * C.FPS).round().astype("Int64")
        gt = gt[["class", "labeling", "clip_id", "start_sec", "end_sec", "start_frame",
                 "end_frame", "present", "auto_start_sec", "auto_end_sec", "surface_form"]]
        C.write_csv(gt.sort_values(["labeling", "class", "clip_id"]),
                    args.phase3 / "spotting_gt" / "test_gt_verified.csv")
        moved = gt[has]
        shift_s = (moved.start_sec - moved.auto_start_sec.astype(float)).abs()
        shift_e = (moved.end_sec - moved.auto_end_sec.astype(float)).abs()
        rep(f"\n[spotting] {len(g):,} occurrences checked -> {len(gt):,} class rows in "
            f"test_gt_verified.csv; present {g.present.mean():.1%}; boundary moved by "
            f"mean {shift_s.mean():.3f} s (start) / {shift_e.mean():.3f} s (end), "
            f"over 0.1 s in {((shift_s > 0.1) | (shift_e > 0.1)).mean():.1%}")
    else:
        rep("\n[spotting] no spotting annotations in the exports")
    rep.save(ver / "kappa.txt")


if __name__ == "__main__":
    main()
