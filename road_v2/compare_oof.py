"""Compare fine-tuned configurations on their out-of-fold predictions, with the definitions used in the manuscript
(paper_results.py): fold-mean metrics, pooled AUROC/AUPRC, AUROC against each kind of negative, inspected share of the
network at 50% / 80% recall, and block-bootstrap differences to a reference configuration.

Usage (sam3 env):
  python -I road_v2/compare_oof.py convnext_post convnext_siamese [more configs] [--ref convnext_siamese] [--loro]
Prints a Markdown report to stdout.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paper_results as pr  # noqa: E402

SRC = "verified"


def fold_metrics(name, loro):
    p = pr.RES / f"ft_{name}_{SRC}{'_loro' if loro else ''}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def main() -> int:
    args = sys.argv[1:]
    loro = "--loro" in args
    ref = args[args.index("--ref") + 1] if "--ref" in args else None
    names = [a for i, a in enumerate(args) if not a.startswith("--") and not (i > 0 and args[i - 1] == "--ref")]
    if ref and ref not in names:
        names.insert(0, ref)
    data = {n: pr.oof(n, SRC, loro) for n in names}
    missing = [n for n, d in data.items() if d is None]
    if missing:
        print("missing out-of-fold files for:", ", ".join(missing))
        return 1
    print(f"## {'Leave-one-region-out' if loro else 'Spatially blocked 5-fold CV'} ({SRC} labels)\n")
    print("| config | fold-mean AUROC | fold-mean AUPRC | fold-mean F1 | pooled AUROC | pooled AUPRC | "
          "AUROC vs cand.-clear | vs near | vs random |" + ("" if loro else " share @50% recall | share @80% recall |"))
    print("|---|---|---|---|---|---|---|---|---|" + ("" if loro else "---|---|"))
    for n in names:
        d, j = data[n], fold_metrics(n, loro)
        f = j["folds"] if j else {}
        row = [n, f"{f.get('AUROC_mean', float('nan')):.3f}", f"{f.get('AUPRC_mean', float('nan')):.3f}",
               f"{f.get('F1_mean', float('nan')):.3f}", f"{roc_auc_score(d['y'], d['s']):.3f}",
               f"{average_precision_score(d['y'], d['s']):.3f}"]
        row += [f"{pr.auroc_vs(d, k):.3f}" for k in ("candidate", "near", "random")]
        if not loro:
            row += [f"{pr.at_recall(d, t)[0]:.2f}%" for t in (0.5, 0.8)]
        print("| " + " | ".join(row) + " |")
    if loro:
        print("\nPer held-out region (AUROC):\n")
        regions = None
        for n in names:
            j = fold_metrics(n, loro)
            if j:
                regions = regions or list(j["per_group"])
                print(f"- {n}: " + ", ".join(f"{g} {j['per_group'][g]['AUROC']:.3f}" for g in regions))
    if ref:
        print(f"\n### Differences to {ref} (block bootstrap over 2 km blocks, 1000 resamples, 95% interval)\n")
        print("| config | Δ pooled AUROC | Δ pooled AUPRC | Δ AUROC vs cand.-clear |")
        print("|---|---|---|---|")
        r = data[ref]
        for n in names:
            if n == ref:
                continue
            d = data[n]
            assert (d["id"] == r["id"]).all(), "out-of-fold files list the segments in different orders"

            def diff(sel, d=d):
                return np.array([roc_auc_score(d["y"][sel], d["s"][sel]) - roc_auc_score(r["y"][sel], r["s"][sel]),
                                 average_precision_score(d["y"][sel], d["s"][sel]) - average_precision_score(r["y"][sel], r["s"][sel]),
                                 pr.auroc_vs(d, "candidate", sel) - pr.auroc_vs(r, "candidate", sel)])
            obs = diff(np.arange(len(d["y"])))
            lo, hi = pr.block_bootstrap(d["id"], diff, n_boot=1000)
            print(f"| {n} | " + " | ".join(f"{o:+.3f} ({a:+.3f} to {b:+.3f})" for o, a, b in zip(obs, lo, hi)) + " |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
