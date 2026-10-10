"""Per-seed metrics (mean ± SD) and seed-averaged out-of-fold scores for fine-tuned configurations.

Runs of a configuration: <config> (seeds 100+k), <config>_s2 (200+k), <config>_s3 (300+k), as written by
finetune_noto.py / finetune_noto_v2.py with FT_SEED_OFFSET and FT_NAME_SUFFIX. Writes
results_noto/oof_ft_<config>_seedavg_verified.csv, which compare_oof.py reads as configuration <config>_seedavg.

Usage (sam3 env):
  python -I road_v2/aggregate_seeds.py convnext_post convnext_siamese_v2
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paper_results as pr  # noqa: E402

for cfg in sys.argv[1:]:
    runs = [n for n in (cfg, cfg + "_s2", cfg + "_s3") if (pr.RES / f"oof_ft_{n}_verified.csv").exists()]
    ds = [pr.oof(n, "verified") for n in runs]
    for d in ds[1:]:
        assert (d["id"] == ds[0]["id"]).all(), "runs list the segments in different orders"
    m = np.array([[roc_auc_score(d["y"], d["s"]), average_precision_score(d["y"], d["s"]), pr.auroc_vs(d, "candidate"),
                   pr.at_recall(d, 0.8)[0]] for d in ds])
    print(f"## {cfg}: {len(ds)} seeds ({', '.join(runs)})")
    for n, r in zip(runs, m):
        print(f"- {n}: pooled AUROC {r[0]:.4f}, AUPRC {r[1]:.4f}, AUROC vs cand.-clear {r[2]:.4f}, share@80% {r[3]:.2f}%")
    print(f"- mean ± SD: AUROC {m[:, 0].mean():.4f} ± {m[:, 0].std():.4f}, AUPRC {m[:, 1].mean():.4f} ± {m[:, 1].std():.4f}, "
          f"cand.-clear {m[:, 2].mean():.4f} ± {m[:, 2].std():.4f}, share@80% {m[:, 3].mean():.2f} ± {m[:, 3].std():.2f}\n")
    rows = list(csv.DictReader(open(pr.RES / f"oof_ft_{cfg}_verified.csv", encoding="utf-8")))
    mean_score = np.mean([d["s"] for d in ds], axis=0)
    with open(pr.RES / f"oof_ft_{cfg}_seedavg_verified.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["sample_id", "sample_type", "label", "group", "score"])
        for r, s in zip(rows, mean_score):
            w.writerow([r["sample_id"], r["sample_type"], r["label"], r["group"], round(float(s), 5)])
