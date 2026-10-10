"""End-to-end fine-tuning of Siamese v2 (with local change map and road-corridor pooling) on dataset_noto2024_v0.

Configurations:
  convnext_siamese_v2_aux   Siamese v2 with auxiliary LiDAR change prediction loss
  convnext_siamese_v2       Siamese v2 without auxiliary LiDAR loss

Usage:
  python -I road_v2/finetune_noto_v2.py [config ...] --labels verified [--folds K]
"""
from __future__ import annotations

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import finetune_noto as base

sys.stdout.reconfigure(encoding="utf-8")

CONFIGS = {
    "convnext_siamese_v2_aux": dict(aux=True, batch=16),
    "convnext_siamese_v2": dict(aux=False, batch=16),
    # ablations of the proposed network, one component removed each
    "convnext_siamese_v2_noalign": dict(aux=False, batch=16, align=False),  # co-located cells only, no 3x3 search
    "convnext_siamese_v2_nocorridor": dict(aux=False, batch=16, corridor=False),  # uniform instead of road weights
    "convnext_siamese_v2_nolocal": dict(aux=False, batch=16, local=False),  # global features only, no change map
}


def batch_views(post, pre, road_mask, lidar_mask, train, gen):
    """uint8 NHWC tensors on GPU -> float NCHW crops with identical geometry."""
    post = post.permute(0, 3, 1, 2).float() / 255.0
    pre = pre.permute(0, 3, 1, 2).float() / 255.0 if pre is not None else None
    road_mask = road_mask[:, None].float() if road_mask is not None else None
    lidar_mask = lidar_mask[:, None].float() if lidar_mask is not None else None

    if train:
        oy, ox = (int(v) for v in torch.randint(0, 256 - base.CROP + 1, (2,), generator=gen))
        k = int(torch.randint(0, 4, (1,), generator=gen))
        flip = bool(torch.randint(0, 2, (1,), generator=gen))
    else:
        oy = ox = (256 - base.CROP) // 2
        k = 0
        flip = False

    def geo(t):
        if t is None:
            return None
        t = t[:, :, oy:oy + base.CROP, ox:ox + base.CROP]
        t = torch.rot90(t, k, (2, 3))
        return t.flip(3) if flip else t

    post = geo(post)
    pre = geo(pre)
    road_mask = geo(road_mask)
    lidar_mask = geo(lidar_mask)

    if train:
        def jitter(t):
            if t is None:
                return None
            n = t.shape[0]
            b = 1.0 + (torch.rand(n, 1, 1, 1, device=base.DEVICE) - 0.5) * 0.3
            c = 1.0 + (torch.rand(n, 1, 1, 1, device=base.DEVICE) - 0.5) * 0.3
            m = t.mean((1, 2, 3), keepdim=True)
            return ((t - m) * c + m * b).clamp(0.0, 1.0)

        post = jitter(post)
        pre = jitter(pre)

    return post, pre, road_mask, lidar_mask


class NetV2(nn.Module):
    def __init__(self, aux: bool = False, align: bool = True, corridor: bool = True, local: bool = True):
        super().__init__()
        self.aux, self.align, self.corridor, self.local = aux, align, corridor, local
        self.enc = timm.create_model(
            "convnext_tiny.in12k_ft_in1k",
            pretrained=True,
            features_only=True,
            out_indices=(2, 3),
        )
        cfg = self.enc.pretrained_cfg
        self.register_buffer("mean", torch.tensor(cfg["mean"]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(cfg["std"]).view(1, 3, 1, 1))

        self.ln = nn.LayerNorm(768)
        self.proj = nn.Conv2d(384, 128, kernel_size=1)
        self.conv_change = nn.Conv2d(257, 128, kernel_size=1)
        self.gelu = nn.GELU()

        self.head_f = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(2560 if local else 2304, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1),
        )
        self.head_p = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(768, 1),
        )
        if self.aux:
            self.head_aux = nn.Conv2d(128, 1, kernel_size=1)

    def norm(self, x):
        return (x - self.mean) / self.std

    def forward(self, post, pre, road_mask):
        B = post.shape[0]
        # Encoder forward pass on concatenated post and pre
        x = torch.cat([self.norm(post), self.norm(pre)], dim=0)
        f3_all, f4_all = self.enc(x)
        f3_post, f3_pre = f3_all.chunk(2, dim=0)
        f4_post, f4_pre = f4_all.chunk(2, dim=0)

        # Global features
        f_post = self.ln(f4_post.mean(dim=(-2, -1)))
        f_pre = self.ln(f4_pre.mean(dim=(-2, -1)))

        # Local change map on F3 (H = W = 14)
        H, W = f3_post.shape[2], f3_post.shape[3]
        q = F.normalize(self.proj(f3_post), dim=1)
        k = F.normalize(self.proj(f3_pre), dim=1)

        if self.align:
            kn = F.unfold(k, kernel_size=3, padding=1).view(B, 128, 9, H, W)
            corr = (q.unsqueeze(2) * kn).sum(dim=1)  # [B, 9, H, W]
            w = F.softmax(corr / 0.1, dim=1)
            k_al = (kn * w.unsqueeze(1)).sum(dim=2)  # [B, 128, H, W]
            s = corr.max(dim=1, keepdim=True).values  # [B, 1, H, W]
        else:  # ablation: compare co-located cells only
            k_al = k
            s = (q * k).sum(dim=1, keepdim=True)

        C = self.gelu(self.conv_change(torch.cat([q, (q - k_al).abs(), 1.0 - s], dim=1)))  # [B, 128, H, W]

        # Road-corridor pooling
        m_dilated = F.max_pool2d(road_mask, kernel_size=31, stride=1, padding=15)
        a = F.adaptive_avg_pool2d(m_dilated, 14) if self.corridor else torch.ones_like(C[:, :1])
        g_road = (C * a).sum(dim=(2, 3)) / (a.sum(dim=(2, 3)) + 1e-6)
        g_all = C.mean(dim=(2, 3))

        feat_fused = torch.cat([f_post, f_pre, (f_post - f_pre).abs()] + ([g_road, g_all] if self.local else []), dim=1)
        logit_f = self.head_f(feat_fused).squeeze(1)
        logit_p = self.head_p(f_post).squeeze(1)

        aux_logit = self.head_aux(C) if self.aux else None
        return logit_f, logit_p, aux_logit


def train_and_predict(cfg, data, tr, te, seed, fold_idx=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    gen = torch.Generator().manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    aux = cfg["aux"]
    bs = cfg.get("batch", 16)
    epochs = base.EPOCHS

    model = NetV2(aux=aux, align=cfg.get("align", True), corridor=cfg.get("corridor", True),
                  local=cfg.get("local", True)).to(base.DEVICE)

    enc_params = list(model.enc.parameters())
    enc_ids = {id(p) for p in enc_params}
    head_params = [p for p in model.parameters() if id(p) not in enc_ids]

    opt = torch.optim.AdamW(
        [
            {"params": enc_params, "lr": base.LR_BACKBONE},
            {"params": head_params, "lr": base.LR_HEAD},
        ],
        weight_decay=base.WD,
    )

    steps = epochs * math.ceil(len(tr) / bs)
    warm = math.ceil(len(tr) / bs)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt,
        lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm))),
    )

    y = data["y"]
    pos_weight = torch.tensor((y[tr] == 0).sum() / max(1, (y[tr] == 1).sum()), device=base.DEVICE)

    model.train()
    for ep in range(epochs):
        t_ep_start = time.time()
        perm = tr[torch.randperm(len(tr), generator=gen).numpy()]
        for i in range(0, len(perm), bs):
            b = perm[i:i + bs]
            post, pre, road_mask, lidar_mask = batch_views(
                base.to_gpu(data["post"][b]),
                base.to_gpu(data["pre"][b]),
                base.to_gpu(data["road_mask"][b]),
                base.to_gpu(data["lidar_mask"][b]) if aux else None,
                True,
                gen,
            )
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=base.DEVICE.type == "cuda"):
                logit_f, logit_p, aux_logit = model(post, pre, road_mask)

            y_b = torch.from_numpy(y[b]).float().to(base.DEVICE)
            loss = (
                F.binary_cross_entropy_with_logits(logit_f.float(), y_b, pos_weight=pos_weight)
                + F.binary_cross_entropy_with_logits(logit_p.float(), y_b, pos_weight=pos_weight)
            )
            if aux and aux_logit is not None:
                t = F.adaptive_avg_pool2d(lidar_mask.float(), 14)
                loss = loss + 0.3 * F.binary_cross_entropy_with_logits(aux_logit.float(), t)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()

        sec_ep = time.time() - t_ep_start
        peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0.0
        print(f"    fold {fold_idx} epoch {ep + 1}/{epochs}: {sec_ep:.1f} s | peak GPU mem: {peak_mb:.1f} MB", flush=True)

    model.eval()
    out_p, out_fused, out_post = [], [], []
    with torch.no_grad():
        for i in range(0, len(te), 64):
            b = te[i:i + 64]
            post, pre, road_mask, _ = batch_views(
                base.to_gpu(data["post"][b]),
                base.to_gpu(data["pre"][b]),
                base.to_gpu(data["road_mask"][b]),
                None,
                False,
                gen,
            )
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=base.DEVICE.type == "cuda"):
                logit_f, logit_p, _ = model(post, pre, road_mask)

            sf = torch.sigmoid(logit_f.float()).cpu().numpy()
            sp = torch.sigmoid(logit_p.float()).cpu().numpy()
            p = 0.5 * sf + 0.5 * sp

            out_p.append(p)
            out_fused.append(sf)
            out_post.append(sp)

    del model, opt
    torch.cuda.empty_cache()
    return np.concatenate(out_p), np.concatenate(out_fused), np.concatenate(out_post)


def main() -> int:
    argv = sys.argv[1:]
    src = "verified"
    if "--labels" in argv:
        k = argv.index("--labels")
        src = argv[k + 1]
        del argv[k:k + 2]
        assert src in ("candidate", "verified"), src

    loro = "--loro" in argv  # leave-one-region-out instead of the 5 spatial folds
    fold_filter = None
    if "--folds" in argv:
        k = argv.index("--folds")
        fold_filter = int(argv[k + 1])
        del argv[k:k + 2]

    names = [a for a in argv if not a.startswith("--")] or list(CONFIGS)
    rows = base.paper_rows(src)
    ids = [r["sample_id"] for r in rows]

    t0 = time.time()
    data = dict(
        post=base.load(ids, "post"),
        pre=base.load(ids, "pre"),
        road_mask=np.stack([np.asarray(Image.open(base.DS / "masks/road" / f"{i}.png")) > 0 for i in ids]).astype(np.uint8),
        lidar_mask=np.stack([np.asarray(Image.open(base.DS / "masks/slide_lidar" / f"{i}.png")) > 0 for i in ids]).astype(np.uint8),
        y=np.array([r["y"] for r in rows]),
    )
    types = np.array([r["sample_type"] for r in rows])
    groups = np.array([r["region"] for r in rows]) if loro else np.array([int(r["fold"]) for r in rows])

    print(
        f"[data] {len(ids)} samples ({data['y'].sum()} positive) loaded in {time.time() - t0:.0f} s; "
        f"labels={src}, folds={'single fold ' + str(fold_filter) if fold_filter is not None else '5 spatial folds'}",
        flush=True,
    )
    base.RES.mkdir(exist_ok=True)

    is_smoke = fold_filter is not None

    for name in names:
        cfg = CONFIGS[name]
        tag = f"{name}{base.NAME_SUFFIX}_{src}{'_loro' if loro else ''}{'_smoke' if is_smoke else ''}"
        oof = np.zeros(len(ids))
        oof_fused = np.zeros(len(ids))
        oof_post = np.zeros(len(ids))
        per = {}
        t1 = time.time()

        all_folds = sorted(set(groups))
        folds_to_run = [fold_filter] if is_smoke else all_folds

        for k, g in enumerate(all_folds):
            if g not in folds_to_run:
                continue
            te = np.flatnonzero(groups == g)
            tr = np.flatnonzero(groups != g)
            pred_p, pred_f, pred_post = train_and_predict(cfg, data, tr, te, seed=100 + k + base.SEED_OFFSET, fold_idx=g)
            oof[te] = pred_p
            oof_fused[te] = pred_f
            oof_post[te] = pred_post
            per[str(g)] = base.metrics(data["y"][te], oof[te], types[te])
            peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0.0
            print(
                f"  {tag} fold={g}: AUROC {per[str(g)]['AUROC']:.3f} AUPRC {per[str(g)]['AUPRC']:.3f} "
                f"({time.time() - t1:.0f} s elapsed, peak GPU mem: {peak_mb:.1f} MB)",
                flush=True,
            )

        summary = {
            f"{m}_{s}": round(float(fn([g_val[m] for g_val in per.values()])), 4)
            for m in ("AUROC", "AUPRC", "F1", "AUROC_vs_near", "AUROC_vs_random", "chance_AUPRC")
            for s, fn in (("mean", np.mean), ("sd", np.std))
        }

        eval_indices = np.concatenate([np.flatnonzero(groups == g) for g in folds_to_run])
        overall_eval = base.metrics(data["y"][eval_indices], oof[eval_indices], types[eval_indices])

        res = dict(
            config=name,
            labels=src,
            **cfg,
            epochs=base.EPOCHS,
            overall=overall_eval,
            folds=summary,
            per_group=per,
            minutes=round((time.time() - t1) / 60, 2),
            peak_gpu_memory_mb=round(float(torch.cuda.max_memory_allocated() / (1024 ** 2)), 1) if torch.cuda.is_available() else 0.0,
        )

        (base.RES / f"ft_{tag}.json").write_text(json.dumps(res, indent=1), encoding="utf-8")

        with open(base.RES / f"oof_ft_{tag}.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["sample_id", "sample_type", "label", "group", "score"])
            w.writerows(
                [[i, t, int(v), g, round(float(p), 5)] for i, t, v, g, p in zip(ids, types, data["y"], groups, oof)]
            )

        with open(base.RES / f"oof_ft_{tag}_heads.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["sample_id", "score_fused", "score_post"])
            w.writerows(
                [[i, round(float(sf), 5), round(float(sp), 5)] for i, sf, sp in zip(ids, oof_fused, oof_post)]
            )

        o = res["overall"]
        print(
            f"[done] {tag}: AUROC {o['AUROC']:.3f} AUPRC {o['AUPRC']:.3f} (chance {o['chance_AUPRC']}) "
            f"F1 {o['F1']:.3f} vs-near {o['AUROC_vs_near']:.3f} in {res['minutes']} min\n",
            flush=True,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
