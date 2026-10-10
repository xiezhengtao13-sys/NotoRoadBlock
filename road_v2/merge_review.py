"""Merge the blind visual review into final labels for dataset_noto2024_v0.

Rules (paper subset: no tunnels, no post-earthquake detour roads, January photo available). A segment counts as
blocked when the two independent sources agree: GSI mapped a slope failure at the road (candidate) AND the blind
review saw the road surface covered or cut.
  candidate, reviewed blocked:            1
  candidate, reviewed clear:              0
  non-candidate, reviewed clear:          0
  non-candidate, not in the review set:   0 (no mapped slope failure within 5 m)
  non-candidate, reviewed blocked:        excluded (obstruction that cannot be attributed to a mapped slope failure)
  reviewed undeterminable ('unclear'):    excluded
  in the review set but not reviewed yet: excluded, and reported (the final run needs a complete review)
The column label_reviewed keeps the plain review answer (1/0, -1 if excluded or not reviewed) for sensitivity checks.
With several reviewers, the primary reviewer's answer is used and the others give agreement statistics.

Usage:
  python -I road_v2/merge_review.py [primary_reviewer]
Writes dataset_noto2024_v0/labels_verified.csv and review/review_summary.json.
"""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent
DS = ROOT / "dataset_noto2024_v0"
REVIEW = DS / "review"
CODE = {"blocked": 1, "clear": 0, "unclear": -1}


def read_reviewer(path):
    latest = {}
    with open(path, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            latest[r["sample_id"]] = r["label"]  # newest row wins
    return latest


def kappa(a, b):
    cats = sorted(set(a) | set(b)); n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((a.count(c) / n) * (b.count(c) / n) for c in cats)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def main() -> int:
    reviewers = {p.stem[len("labels_"):]: read_reviewer(p) for p in sorted(REVIEW.glob("labels_*.csv"))
                 if not p.stem.endswith("_recheck")}
    if not reviewers:
        print("no review files in", REVIEW); return 1
    primary = sys.argv[1] if len(sys.argv) > 1 else max(reviewers, key=lambda k: len(reviewers[k]))
    ans = dict(reviewers[primary])
    review_ids = {it["id"] for it in json.loads((REVIEW / "review_list.json").read_text(encoding="utf-8"))}
    # disputed candidates (first answer disagrees with the DEM-based deposits) were rechecked with a stricter rule
    recheck_list = REVIEW / "review_list_recheck.json"
    recheck_ids = {it["id"] for it in json.loads(recheck_list.read_text(encoding="utf-8"))} if recheck_list.exists() else set()
    recheck_file = REVIEW / f"labels_{primary}_recheck.csv"
    recheck = read_reviewer(recheck_file) if recheck_file.exists() else {}
    first_pass = {s: ans.get(s, "") for s in recheck_ids}
    for s in recheck_ids:
        ans[s] = recheck.get(s, "")  # an unanswered recheck counts as not reviewed yet
    rows = [r for r in csv.DictReader(open(DS / "samples.csv", encoding="utf-8-sig"))
            if r["post_date"] and r["tunnel"] == "0" and r["post_event_road"] == "0"]
    out, missing = [], Counter()
    for r in rows:
        sid, in_rev = r["sample_id"], r["sample_id"] in review_ids
        cand = r["sample_type"] == "candidate"
        if in_rev:
            a = ans.get(sid, "")
            if not a:
                missing[r["sample_type"]] += 1
                label = reviewed = -2
            else:
                reviewed = CODE[a]
                label = -3 if (not cand and a == "blocked") else reviewed  # -3: conflict, excluded
        else:
            a, label, reviewed = "", 0, -1
        out.append(dict(sample_id=sid, sample_type=r["sample_type"], in_review=int(in_rev), review_label=a, label=label,
                        label_reviewed=reviewed))
    with open(DS / "labels_verified.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)

    summary = dict(primary=primary, reviewers={k: len(v) for k, v in reviewers.items()},
                   review_set_in_paper_subset=sum(o["in_review"] for o in out), missing=dict(missing))
    if recheck_ids:
        pairs = Counter((first_pass[s], recheck.get(s, "(open)")) for s in recheck_ids)
        summary["recheck"] = dict(n=len(recheck_ids), answered=sum(s in recheck for s in recheck_ids),
                                  first_vs_recheck={f"{a} -> {b}": n for (a, b), n in sorted(pairs.items())})
    for t in ("candidate", "near", "random"):
        rev = [o for o in out if o["sample_type"] == t and o["in_review"] and o["review_label"]]
        summary[f"{t}_reviewed"] = dict(Counter(o["review_label"] for o in rev), n=len(rev))
    final = [o for o in out if o["label"] >= 0]
    summary["excluded"] = dict(undeterminable=sum(o["label"] == -1 for o in out),
                               noncandidate_blocked=sum(o["label"] == -3 for o in out),
                               not_reviewed=sum(o["label"] == -2 for o in out))
    summary["final"] = dict(n=len(final), blocked=sum(o["label"] == 1 for o in final),
                            by_stratum={t: dict(Counter(o["label"] for o in final if o["sample_type"] == t)) for t in ("candidate", "near", "random")})
    others = [k for k in reviewers if k != primary]
    for k in others:
        common = sorted(set(ans) & set(reviewers[k]) & review_ids)
        if common:
            a = [ans[s] for s in common]; b = [reviewers[k][s] for s in common]
            summary[f"agreement_{primary}_vs_{k}"] = dict(n=len(common), percent=round(100 * sum(x == y for x, y in zip(a, b)) / len(common), 1),
                                                         cohen_kappa=round(kappa(a, b), 3))
    (REVIEW / "review_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    if missing:
        print(f"\n[warning] {sum(missing.values())} samples of the review set are not reviewed yet; they are excluded for now.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
