"""Road-focused, misregistration-tolerant change features from frozen DINOv2 patch tokens.

Why: concatenating global pre- and post-event features did not help, because most pre/post differences come from
season, snow and illumination, and the change that matters is local to the road. Here the comparison is restricted
to the road and made robust to the several-metre offsets between pre- and post-event orthophotos:
  1. DINOv2 ViT-B/14 patch tokens (16 x 16 grid, ~7 m per token) for the pre- and post-event patches;
  2. for every post token, the best cosine similarity to the pre tokens in its 3 x 3 neighbourhood (tolerates offsets);
  3. similarities averaged over the road (weights from the road mask) and over the background, and their difference,
     which removes the scene-wide change caused by season and snow.
Evaluation: same protocol as baseline_noto.py (5 spatial folds, inner grouped CV for C), on the verified labels:
post-only features vs. post-only + change features vs. change features alone.

Usage (runs on the CPU so that it can share the machine with GPU training):
  python -I road_v2/change_features.py
Caches tokens in road_v2/cache_noto/change_dinov2.npz; writes results_noto/change_verified.json.
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
G = 16  # token grid


def neighbour_mask(g=G, r=1):
    idx = np.arange(g * g); i, j = idx // g, idx % g
    return (np.abs(i[:, None] - i[None, :]) <= r) & (np.abs(j[:, None] - j[None, :]) <= r)


def change_feats(post_tok, pre_tok, w, nb):
    """post_tok, pre_tok: (256, C) L2-normalised; w: (256,) road weights in [0, 1]."""
    s = post_tok @ pre_tok.T
    best = np.where(nb, s, -np.inf).max(1)  # best match within +-1 token (~7 m)
    road = w > 0.15; bg = w == 0
    rw = w / max(w.sum(), 1e-6)
    s_road = float((best * rw).sum()); s_bg = float(best[bg].mean()) if bg.any() else s_road
    vals = best[road] if road.any() else best
    return [s_road, s_bg, s_road - s_bg, float(np.percentile(vals, 10)), float(vals.min()),
            float(np.percentile(best[bg], 10)) if bg.any() else s_road]


def extract(ids):
    out = CACHE / "change_dinov2.npz"
    if out.exists() and list(np.load(out)["ids"]) == ids:
        print("[cached]", out.name, flush=True)
        return np.load(out)["feats"]
    import timm
    import torch
    import torch.nn.functional as F
    torch.set_num_threads(max(1, os.cpu_count() - 2))
    model = timm.create_model("vit_base_patch14_dinov2.lvd142m", pretrained=True, num_classes=0, img_size=224).eval()
    cfg = model.pretrained_cfg
    mean = torch.tensor(cfg["mean"]).view(1, 3, 1, 1); std = torch.tensor(cfg["std"]).view(1, 3, 1, 1)
    nb = neighbour_mask()

    def tokens(kind, batch):
        x = torch.from_numpy(np.stack([np.asarray(Image.open(DS / "chips" / kind / f"{i}.jpg").convert("RGB")) for i in batch]))
        x = x.permute(0, 3, 1, 2).float() / 255
        x = (F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False, antialias=True) - mean) / std
        with torch.no_grad():
            t = model.forward_features(x)[:, model.num_prefix_tokens:]
        return F.normalize(t, dim=-1).numpy()

    feats, t0 = [], time.time()
    for k in range(0, len(ids), 32):
        batch = ids[k:k + 32]
        tp, tq = tokens("post", batch), tokens("pre", batch)
        for b, sid in enumerate(batch):
            m = torch.from_numpy((np.asarray(Image.open(DS / "masks/road" / f"{sid}.png")) > 0).astype(np.float32))[None, None]
            w = F.adaptive_avg_pool2d(m, G).flatten().numpy()
            feats.append(change_feats(tp[b], tq[b], w, nb))
        if k % 640 == 0:
            print(f"  {k + len(batch)}/{len(ids)} ({time.time() - t0:.0f} s)", flush=True)
    feats = np.array(feats, np.float32)
    np.savez_compressed(out, ids=np.array(ids), feats=feats)
    return feats


def main() -> int:
    from sklearn.compose import ColumnTransformer
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    ver = {r["sample_id"]: int(r["label"]) for r in csv.DictReader(open(DS / "labels_verified.csv", encoding="utf-8"))}
    rows = [r for r in csv.DictReader(open(DS / "samples.csv", encoding="utf-8-sig")) if ver.get(r["sample_id"], -9) >= 0]
    ids = [r["sample_id"] for r in rows]
    y = np.array([ver[i] for i in ids]); folds = np.array([int(r["fold"]) for r in rows]); groups = np.array([r["block_id"] for r in rows])
    types = np.array([r["sample_type"] for r in rows])
    print(f"[1] change features for {len(ids)} samples ({y.sum()} blocked)", flush=True)
    ch = extract(ids)
    dn = np.load(CACHE / "feats_dinov2_vitb14.npz")
    pos = {s: k for k, s in enumerate(dn["ids"])}
    sel = np.array([pos[i] for i in ids])
    post = np.hstack([dn["post_full"][sel], dn["post_road"][sel]])
    d = post.shape[1]

    def model(kind, C):
        if kind == "change":
            return make_pipeline(StandardScaler(), LogisticRegression(C=C, class_weight="balanced", max_iter=5000))
        ct = ColumnTransformer([("deep", make_pipeline(StandardScaler(), PCA(256, random_state=0)), list(range(d)))] +
                               ([("chg", StandardScaler(), list(range(d, d + ch.shape[1])))] if kind == "post+change" else []))
        return make_pipeline(ct, LogisticRegression(C=C, class_weight="balanced", max_iter=5000))

    sets = {"post (DINOv2 + road-pooled)": ("post", post), "post + road change": ("post+change", np.hstack([post, ch])),
            "road change only": ("change", ch)}
    results = {}
    for name, (kind, X) in sets.items():
        per = []
        for f in range(5):
            tr, te = np.flatnonzero(folds != f), np.flatnonzero(folds == f)
            best, bestC = -1, None
            for C in (0.01, 0.1, 1.0):
                sc = []
                for itr, ite in GroupKFold(3).split(X[tr], y[tr], groups[tr]):
                    m = model(kind, C).fit(X[tr][itr], y[tr][itr])
                    sc.append(average_precision_score(y[tr][ite], m.predict_proba(X[tr][ite])[:, 1]))
                if np.mean(sc) > best:
                    best, bestC = np.mean(sc), C
            p = model(kind, bestC).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
            near = (types[te] == "candidate") | (types[te] == "near")
            per.append((roc_auc_score(y[te], p), average_precision_score(y[te], p), roc_auc_score(y[te][near], p[near]),
                        f1_score(y[te], p >= 0.5), y[te].mean()))
        a = np.array(per)
        results[name] = {f"{m}_{s}": round(float(fn(a[:, k])), 3) for k, m in enumerate(("AUROC", "AUPRC", "AUROC_vs_near", "F1", "chance_AUPRC"))
                         for s, fn in (("mean", np.mean), ("sd", np.std))}
        print(f"  {name:28s} AUROC {a[:, 0].mean():.3f}±{a[:, 0].std():.3f}  AUPRC {a[:, 1].mean():.3f}±{a[:, 1].std():.3f}  "
              f"vs-near {a[:, 2].mean():.3f}", flush=True)
    for k, name in enumerate(["s_road", "s_bg", "s_road-s_bg", "p10_road", "min_road", "p10_bg"]):
        print(f"    {name:12s} blocked {ch[y == 1, k].mean():.3f}  clear {ch[y == 0, k].mean():.3f}")
    RES.mkdir(exist_ok=True)
    (RES / "change_verified.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
