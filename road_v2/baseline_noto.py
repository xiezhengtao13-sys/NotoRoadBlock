"""Frozen pretrained features + logistic regression on dataset_noto2024_v0 (the paper's frozen-feature models).

Paper subset: road pieces in tunnels and post-earthquake detour roads are excluded. Labels: the GSI-derived
candidates (--labels candidate, default) or the blind visual review merged by merge_review.py (--labels verified).
Evaluation: the dataset's 5 spatial folds (2 km blocks; neighbours never split), C chosen by inner grouped CV on the
training blocks; metrics pooled over out-of-fold scores and as mean +- SD over folds; leave-one-region-out for the
two best feature sets.
Feature sets: road class only (no pixels), colour statistics, ConvNeXt-T (ImageNet-12k), DINOv2 ViT-B/14;
post-event only or pre+post; full patch, centre crop (~60 m) or patch tokens pooled over the road mask.

Usage (GPU env with timm and a local Hugging Face cache, e.g. sam3):
  python -I road_v2/baseline_noto.py [--labels candidate|verified]
Caches features in road_v2/cache_noto/; writes road_v2/results_noto/frozen_<labels>.csv/.json,
oof_frozen_<labels>.csv and loro_frozen_<labels>.csv.
"""
from __future__ import annotations

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent
DS = ROOT / "dataset_noto2024_v0"
CACHE = ROOT / "cache_noto"
RES = ROOT / "results_noto"
BACKBONES = {
    "convnext_tiny": dict(name="convnext_tiny.in12k_ft_in1k"),
    "dinov2_vitb14": dict(name="vit_base_patch14_dinov2.lvd142m", img_size=224),
}
C_GRID = (0.01, 0.1, 1.0)


def load_rows():
    rows = list(csv.DictReader(open(DS / "samples.csv", encoding="utf-8-sig")))
    return [r for r in rows if r["post_date"]]  # one sample has no January photo


def load_imgs(ids, kind):
    return np.stack([np.asarray(Image.open(DS / "chips" / kind / f"{i}.jpg").convert("RGB")) for i in ids])


# ---------- features ----------
def extract(ids):
    import timm
    import torch
    import torch.nn.functional as F
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    CACHE.mkdir(exist_ok=True)
    masks = np.stack([np.asarray(Image.open(DS / "masks/road" / f"{i}.png")) > 0 for i in ids])
    for key, spec in BACKBONES.items():
        out = CACHE / f"feats_{key}.npz"
        if out.exists() and list(np.load(out)["ids"]) == ids:
            print(f"[cached] {out.name}", flush=True)
            continue
        t0 = time.time()
        model = timm.create_model(spec["name"], pretrained=True, num_classes=0,
                                  **{k: v for k, v in spec.items() if k != "name"}).to(device).eval()
        cfg = model.pretrained_cfg
        mean = torch.tensor(cfg["mean"], device=device).view(1, 3, 1, 1)
        std = torch.tensor(cfg["std"], device=device).view(1, 3, 1, 1)
        res = {}
        for kind in ("pre", "post"):
            feats = {"full": [], "center": [], "road": []}
            for i in range(0, len(ids), 64):
                x = torch.from_numpy(load_imgs(ids[i:i + 64], kind)).to(device).permute(0, 3, 1, 2).float() / 255
                for view, xv in (("full", x), ("center", x[:, :, 64:192, 64:192])):
                    xv = (F.interpolate(xv, size=(224, 224), mode="bilinear", align_corners=False, antialias=True) - mean) / std
                    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                        if key.startswith("dinov2"):
                            t = model.forward_features(xv).float()
                            cls, patch = t[:, 0], t[:, model.num_prefix_tokens:]
                            feats[view].append(torch.cat([cls, patch.mean(1)], 1).cpu().numpy())
                            if view == "full":  # patch tokens averaged over the target road piece
                                g = int(round(patch.shape[1] ** 0.5))
                                m = torch.from_numpy(masks[i:i + 64].astype(np.float32)).to(device)[:, None]
                                w = F.adaptive_avg_pool2d(m, g).flatten(1)
                                w = w / w.sum(1, keepdim=True).clamp_min(1e-6)
                                feats["road"].append(torch.einsum("bn,bnc->bc", w, patch).cpu().numpy())
                        else:
                            feats[view].append(model(xv).float().cpu().numpy())
            for view, lst in feats.items():
                if lst:
                    res[f"{kind}_{view}"] = np.concatenate(lst).astype(np.float32)
        np.savez_compressed(out, ids=np.array(ids), **res)
        print(f"[ok] {key}: {', '.join(f'{k} {v.shape[1]}' for k, v in res.items())} in {time.time() - t0:.0f} s", flush=True)
        del model
        torch.cuda.empty_cache()


def colour_stats(ids, kind):
    out = []
    for i in ids:
        a = np.asarray(Image.open(DS / "chips" / kind / f"{i}.jpg").convert("RGB")).astype(np.float32) / 255
        c, g = a[64:192, 64:192], a.mean(2)
        grad = np.abs(np.diff(g, axis=0)).mean() + np.abs(np.diff(g, axis=1)).mean()
        out.append(np.concatenate([a.mean((0, 1)), a.std((0, 1)), c.mean((0, 1)), c.std((0, 1)), [grad, (g > 0.75).mean()]]))
    return np.array(out, np.float32)


# ---------- evaluation ----------
def make_model(d, C):
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    steps = [StandardScaler()] + ([PCA(n_components=256, random_state=0)] if d > 256 else [])
    return make_pipeline(*steps, LogisticRegression(C=C, class_weight="balanced", max_iter=5000))


def fit_predict(X, y, groups, tr, te):
    from sklearn.metrics import average_precision_score
    from sklearn.model_selection import GroupKFold
    scores = {}
    for C in C_GRID:
        s = []
        for itr, ite in GroupKFold(n_splits=3).split(X[tr], y[tr], groups[tr]):
            m = make_model(X.shape[1], C).fit(X[tr][itr], y[tr][itr])
            s.append(average_precision_score(y[tr][ite], m.predict_proba(X[tr][ite])[:, 1]))
        scores[C] = np.mean(s)
    C = max(scores, key=scores.get)
    return make_model(X.shape[1], C).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]


def metrics(y, p, types):
    from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
    yhat = p >= 0.5
    out = dict(n=int(len(y)), n_pos=int(y.sum()), chance_AUPRC=round(float(y.mean()), 3),
               AUROC=roc_auc_score(y, p), AUPRC=average_precision_score(y, p),
               F1=f1_score(y, yhat), precision=precision_score(y, yhat, zero_division=0), recall=recall_score(y, yhat))
    for t in ("near", "random"):  # hard negatives (same terrain, <= 300 m away) vs random negatives
        s = (types == "candidate") | (types == t)
        out[f"AUROC_vs_{t}"] = roc_auc_score(y[s], p[s])
    return {k: (round(float(v), 3) if isinstance(v, float) else v) for k, v in out.items()}


def run_cv(X, y, folds, groups, types):
    """Out-of-fold scores over the spatial folds; pooled metrics plus mean and SD over folds."""
    oof = np.zeros(len(y)); per = []
    for f in sorted(set(folds)):
        te = np.flatnonzero(folds == f); tr = np.flatnonzero(folds != f)
        oof[te] = fit_predict(X, y, groups, tr, te)
        per.append(metrics(y[te], oof[te], types[te]))
    m = {f"pooled_{k}": v for k, v in metrics(y, oof, types).items()}
    for k in ("AUROC", "AUPRC", "F1", "AUROC_vs_near", "AUROC_vs_random", "chance_AUPRC"):
        vals = [p[k] for p in per]
        m[f"{k}_mean"] = round(float(np.mean(vals)), 3); m[f"{k}_sd"] = round(float(np.std(vals)), 3)
    return m, oof


def loro(f, reg, X, y, groups, types, regions):
    """Train on four GSI interpretation areas, test on the fifth."""
    te = np.flatnonzero(regions == reg); tr = np.flatnonzero(regions != reg)
    p = fit_predict(X, y, groups, tr, te)
    return dict(features=f, region=reg, **metrics(y[te], p, types[te])), te, p


def label_source():
    if "--labels" in sys.argv:
        src = sys.argv[sys.argv.index("--labels") + 1]
        assert src in ("candidate", "verified"), src
        return src
    return "candidate"


def main() -> int:
    from joblib import Parallel, delayed
    src = label_source()
    rows = load_rows()
    ids = [r["sample_id"] for r in rows]
    print(f"[1] features for {len(ids)} samples (labels: {src})", flush=True)
    extract(ids)
    cn = np.load(CACHE / "feats_convnext_tiny.npz"); dn = np.load(CACHE / "feats_dinov2_vitb14.npz")
    hw_names = sorted({r["highway"] for r in rows})
    hw = np.array([[r["highway"] == h for h in hw_names] for r in rows], np.float32)
    col_post, col_pre = colour_stats(ids, "post"), colour_stats(ids, "pre")
    ad = lambda a, b: np.abs(a - b)
    FS = {
        "road class only (no pixels)": hw,
        "colour stats, after": col_post,
        "colour stats, before+after": np.hstack([col_post, col_pre, col_post - col_pre]),
        "ConvNeXt-T, after": cn["post_full"],
        "ConvNeXt-T, after, full+centre": np.hstack([cn["post_full"], cn["post_center"]]),
        "ConvNeXt-T, before+after": np.hstack([cn["post_full"], cn["post_center"], cn["pre_full"], cn["pre_center"],
                                               ad(cn["post_full"], cn["pre_full"]), ad(cn["post_center"], cn["pre_center"])]),
        "DINOv2-B, after": dn["post_full"],
        "DINOv2-B, after + road-pooled": np.hstack([dn["post_full"], dn["post_road"]]),
        "DINOv2-B, before+after + road-pooled": np.hstack([dn["post_full"], dn["post_road"], dn["pre_full"], dn["pre_road"],
                                                           ad(dn["post_full"], dn["pre_full"]), ad(dn["post_road"], dn["pre_road"])]),
        "ConvNeXt-T + DINOv2-B, after": np.hstack([cn["post_full"], cn["post_center"], dn["post_full"], dn["post_road"]]),
    }
    # paper subset: no tunnels, no post-earthquake detour roads; labels from the GSI candidates or the verified review
    keep = np.array([r["tunnel"] == "0" and r["post_event_road"] == "0" for r in rows])
    if src == "verified":
        ver = {r["sample_id"]: int(r["label"]) for r in csv.DictReader(open(DS / "labels_verified.csv", encoding="utf-8"))}
        y_all = np.array([ver.get(i, -9) for i in ids])
        keep &= y_all >= 0
    else:
        y_all = np.array([int(r["label_aerial"]) for r in rows])
    y = y_all[keep]
    types = np.array([r["sample_type"] for r in rows])[keep]
    folds = np.array([int(r["fold"]) for r in rows])[keep]; groups = np.array([r["block_id"] for r in rows])[keep]
    regions = np.array([r["region"] for r in rows])[keep]
    kept_ids = [i for i, k in zip(ids, keep) if k]
    print(f"    {keep.sum()} samples, {y.sum()} blocked", flush=True)

    print("[2] 5-fold spatial CV", flush=True)
    t0 = time.time()
    names = list(FS)
    res = Parallel(n_jobs=6)(delayed(run_cv)(FS[f][keep], y, folds, groups, types) for f in names)
    print(f"    done in {time.time() - t0:.0f} s", flush=True)
    RES.mkdir(exist_ok=True)
    table = [dict(features=f, **m) for f, (m, _) in zip(names, res)]
    with open(RES / f"frozen_{src}.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(table[0].keys())); w.writeheader(); w.writerows(table)
    for t in table:
        print(f"  {t['features']:40s} AUROC {t['AUROC_mean']:.3f}±{t['AUROC_sd']:.3f}  AUPRC {t['AUPRC_mean']:.3f}±{t['AUPRC_sd']:.3f} "
              f"(chance {t['chance_AUPRC_mean']:.3f})  F1 {t['F1_mean']:.3f}  vs-near {t['AUROC_vs_near_mean']:.3f}")
    with open(RES / f"oof_frozen_{src}.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh); w.writerow(["sample_id", "sample_type", "label", "fold"] + names)
        for k, sid in enumerate(kept_ids):
            w.writerow([sid, types[k], int(y[k]), int(folds[k])] + [round(float(o[k]), 5) for _, o in res])

    print("[3] leave-one-region-out", flush=True)
    best = [t["features"] for t in sorted(table, key=lambda t: -t["AUPRC_mean"])[:2]]
    jobs = [(f, reg) for f in best for reg in sorted(set(regions))]
    lo = Parallel(n_jobs=6)(delayed(loro)(f, reg, FS[f][keep], y, groups, types, regions) for f, reg in jobs)
    lo_rows = [r for r, _, _ in lo]
    with open(RES / f"loro_frozen_{src}.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(lo_rows[0].keys())); w.writeheader(); w.writerows(lo_rows)
    for t in lo_rows:
        print(f"  {t['features']:40s} test={t['region']:16s} n={t['n']:4d} pos={t['n_pos']:3d} "
              f"AUROC {t['AUROC']:.3f}  AUPRC {t['AUPRC']:.3f} (chance {t['chance_AUPRC']:.3f})")
    (RES / f"frozen_{src}.json").write_text(json.dumps(dict(labels=src, n=int(keep.sum()), n_pos=int(y.sum()), cv=table,
                                                            loro=lo_rows), ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
