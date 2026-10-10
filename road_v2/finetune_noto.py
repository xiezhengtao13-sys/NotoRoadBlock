"""End-to-end fine-tuning on dataset_noto2024_v0, evaluated with the dataset's 5 spatial folds.

Configs (name -> backbone, input):
  resnet50_post        ResNet-50, post-event image (RGB)
  convnext_post        ConvNeXt-T, post-event image
  swin_post            Swin-T, post-event image
  convnext_post_mask   ConvNeXt-T, post-event image + target-road mask as a 4th input channel (zero-initialised)
  convnext_siamese     ConvNeXt-T, pre- and post-event images through one shared encoder;
                       head on [f_post, f_pre, |f_post - f_pre|]
Paper subset: road pieces in tunnels, post-earthquake detour roads and the one sample without a January
photo are excluded.
Training: random 224-px crop of the 256-px chip, flips and 90-degree rotations, mild brightness/contrast
jitter (all on the GPU); AdamW (backbone 1e-4, head 1e-3, wd 0.05), 1 warm-up epoch + cosine decay,
10 epochs, class-weighted BCE, bf16 autocast. The model after the last epoch is evaluated on the held-out
fold (centre crop); there is no early stopping on test data.

Usage (GPU env with timm and a local Hugging Face cache, e.g. sam3):
  python -I road_v2/finetune_noto.py [config ...] [--labels candidate|verified]   # 5 spatial folds
  python -I road_v2/finetune_noto.py --loro convnext_post [--labels verified]     # leave-one-region-out
Labels: GSI-derived candidates (default) or the blind review merged by merge_review.py (verified).
Writes road_v2/results_noto/ft_<config>_<labels>[_loro].json and oof_ft_<config>_<labels>[_loro].csv.
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
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent
DS = ROOT / "dataset_noto2024_v0"
RES = ROOT / "results_noto"
CONFIGS = {
    "resnet50_post": dict(backbone="resnet50.a1_in1k", mode="post"),
    "convnext_post": dict(backbone="convnext_tiny.in12k_ft_in1k", mode="post"),
    "swin_post": dict(backbone="swin_tiny_patch4_window7_224.ms_in1k", mode="post"),
    "convnext_post_mask": dict(backbone="convnext_tiny.in12k_ft_in1k", mode="post_mask"),
    "convnext_siamese": dict(backbone="convnext_tiny.in12k_ft_in1k", mode="siamese", batch=16),
}
EPOCHS = int(os.environ.get("FT_EPOCHS", 10))  # FT_EPOCHS=1 for a quick smoke test
SEED_OFFSET = int(os.environ.get("FT_SEED_OFFSET", 0))  # extra seeds: fold k uses seed 100 + k + offset
NAME_SUFFIX = os.environ.get("FT_NAME_SUFFIX", "")  # e.g. "_s2": results are written as <config>_s2_<labels>...
BATCH, LR_BACKBONE, LR_HEAD, WD, CROP = 32, 1e-4, 1e-3, 0.05, 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def paper_rows(src):
    """Paper subset with an integer label 'y': GSI candidates, or the merged blind review (undeterminable dropped)."""
    rows = [r for r in csv.DictReader(open(DS / "samples.csv", encoding="utf-8-sig"))
            if r["post_date"] and r["tunnel"] == "0" and r["post_event_road"] == "0"]
    if src == "verified":
        ver = {r["sample_id"]: int(r["label"]) for r in csv.DictReader(open(DS / "labels_verified.csv", encoding="utf-8"))}
        rows = [dict(r, y=ver[r["sample_id"]]) for r in rows if ver.get(r["sample_id"], -9) >= 0]
    else:
        rows = [dict(r, y=int(r["label_aerial"])) for r in rows]
    return rows


def load(ids, kind):
    return np.stack([np.asarray(Image.open(DS / "chips" / kind / f"{i}.jpg").convert("RGB")) for i in ids])


class Net(nn.Module):
    def __init__(self, backbone, mode):
        super().__init__()
        self.mode = mode
        self.enc = timm.create_model(backbone, pretrained=True, num_classes=0)
        if mode == "post_mask":
            add_input_channel(self.enc)
        d = self.enc.num_features
        cfg = self.enc.pretrained_cfg
        self.register_buffer("mean", torch.tensor(cfg["mean"]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(cfg["std"]).view(1, 3, 1, 1))
        if mode == "siamese":
            self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(3 * d, 256), nn.GELU(), nn.Dropout(0.2), nn.Linear(256, 1))
        else:
            self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(d, 1))

    def norm(self, x):
        return (x - self.mean) / self.std

    def forward(self, post, pre=None, mask=None):
        if self.mode == "siamese":
            f = self.enc(torch.cat([self.norm(post), self.norm(pre)]))
            fp, fq = f.chunk(2)
            return self.head(torch.cat([fp, fq, (fp - fq).abs()], 1)).squeeze(1)
        x = self.norm(post)
        if self.mode == "post_mask":
            x = torch.cat([x, mask * 2 - 1], 1)
        return self.head(self.enc(x)).squeeze(1)


def add_input_channel(model):
    """Give the first 3-channel conv a 4th input channel with zero weights (pretrained RGB behaviour kept at start)."""
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Conv2d) and mod.in_channels == 3:
            new = nn.Conv2d(4, mod.out_channels, mod.kernel_size, mod.stride, mod.padding, mod.dilation, mod.groups,
                            mod.bias is not None)
            with torch.no_grad():
                new.weight.zero_(); new.weight[:, :3] = mod.weight
                if mod.bias is not None:
                    new.bias.copy_(mod.bias)
            parent = model.get_submodule(name.rsplit(".", 1)[0]) if "." in name else model
            setattr(parent, name.rsplit(".", 1)[-1], new)
            return
    raise RuntimeError("no 3-channel conv found")


def to_gpu(a):
    return torch.from_numpy(a).to(DEVICE)


def batch_views(post, pre, mask, train, gen):
    """uint8 NHWC tensors on the GPU -> float NCHW crops with identical geometry for post/pre/mask."""
    post = post.permute(0, 3, 1, 2).float() / 255
    pre = pre.permute(0, 3, 1, 2).float() / 255 if pre is not None else None
    mask = mask[:, None].float() if mask is not None else None
    if train:
        oy, ox = (int(v) for v in torch.randint(0, 256 - CROP + 1, (2,), generator=gen))
        k = int(torch.randint(0, 4, (1,), generator=gen)); flip = bool(torch.randint(0, 2, (1,), generator=gen))
    else:
        oy = ox = (256 - CROP) // 2; k = 0; flip = False

    def geo(t):
        if t is None:
            return None
        t = t[:, :, oy:oy + CROP, ox:ox + CROP]
        t = torch.rot90(t, k, (2, 3))
        return t.flip(3) if flip else t

    post, pre, mask = geo(post), geo(pre), geo(mask)
    if train:
        def jitter(t):
            if t is None:
                return None
            n = t.shape[0]
            b = 1 + (torch.rand(n, 1, 1, 1, device=DEVICE) - 0.5) * 0.3
            c = 1 + (torch.rand(n, 1, 1, 1, device=DEVICE) - 0.5) * 0.3
            m = t.mean((1, 2, 3), keepdim=True)
            return ((t - m) * c + m * b).clamp(0, 1)
        post, pre = jitter(post), jitter(pre)
    return post, pre, mask


def train_and_predict(cfg, data, tr, te, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    gen = torch.Generator().manual_seed(seed)
    model = Net(cfg["backbone"], cfg["mode"]).to(DEVICE)
    bs = cfg.get("batch", BATCH)
    head = list(model.head.parameters()); hid = {id(p) for p in head}
    opt = torch.optim.AdamW([{"params": [p for p in model.parameters() if id(p) not in hid], "lr": LR_BACKBONE},
                             {"params": head, "lr": LR_HEAD}], weight_decay=WD)
    steps = EPOCHS * math.ceil(len(tr) / bs); warm = math.ceil(len(tr) / bs)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm))))
    y = data["y"]
    pos_weight = torch.tensor((y[tr] == 0).sum() / max(1, (y[tr] == 1).sum()), device=DEVICE)
    need_pre, need_mask = cfg["mode"] == "siamese", cfg["mode"] == "post_mask"
    model.train()
    for ep in range(EPOCHS):
        perm = tr[torch.randperm(len(tr), generator=gen).numpy()]
        for i in range(0, len(perm), bs):
            b = perm[i:i + bs]
            post, pre, mask = batch_views(to_gpu(data["post"][b]), to_gpu(data["pre"][b]) if need_pre else None,
                                          to_gpu(data["mask"][b]) if need_mask else None, True, gen)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=DEVICE.type == "cuda"):
                logit = model(post, pre, mask)
            loss = F.binary_cross_entropy_with_logits(logit.float(), torch.from_numpy(y[b]).float().to(DEVICE), pos_weight=pos_weight)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
    model.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(te), 64):
            b = te[i:i + 64]
            post, pre, mask = batch_views(to_gpu(data["post"][b]), to_gpu(data["pre"][b]) if need_pre else None,
                                          to_gpu(data["mask"][b]) if need_mask else None, False, gen)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=DEVICE.type == "cuda"):
                out.append(torch.sigmoid(model(post, pre, mask).float()).cpu().numpy())
    del model, opt
    torch.cuda.empty_cache()
    return np.concatenate(out)


def metrics(y, p, types):
    out = dict(n=int(len(y)), n_pos=int(y.sum()), chance_AUPRC=round(float(y.mean()), 3), AUROC=roc_auc_score(y, p),
               AUPRC=average_precision_score(y, p), F1=f1_score(y, p >= 0.5))
    for t in ("near", "random"):
        s = (types == "candidate") | (types == t)
        out[f"AUROC_vs_{t}"] = roc_auc_score(y[s], p[s])
    return {k: (round(float(v), 4) if isinstance(v, float) else v) for k, v in out.items()}


def main() -> int:
    argv = sys.argv[1:]
    src = "candidate"
    if "--labels" in argv:
        k = argv.index("--labels"); src = argv[k + 1]; del argv[k:k + 2]
        assert src in ("candidate", "verified"), src
    loro = "--loro" in argv
    names = [a for a in argv if not a.startswith("--")] or list(CONFIGS)
    rows = paper_rows(src); ids = [r["sample_id"] for r in rows]
    t0 = time.time()
    data = dict(post=load(ids, "post"), pre=load(ids, "pre"),
                mask=np.stack([np.asarray(Image.open(DS / "masks/road" / f"{i}.png")) > 0 for i in ids]).astype(np.uint8),
                y=np.array([r["y"] for r in rows]))
    types = np.array([r["sample_type"] for r in rows])
    groups = np.array([int(r["fold"]) for r in rows]) if not loro else np.array([r["region"] for r in rows])
    print(f"[data] {len(ids)} samples ({data['y'].sum()} positive) loaded in {time.time() - t0:.0f} s; "
          f"{'leave-one-region-out' if loro else '5 spatial folds'}", flush=True)
    RES.mkdir(exist_ok=True)
    for name in names:
        cfg = CONFIGS[name]; tag = f"{name}{NAME_SUFFIX}_{src}{'_loro' if loro else ''}"
        oof = np.zeros(len(ids)); per = {}; t1 = time.time()
        for k, g in enumerate(sorted(set(groups))):
            te = np.flatnonzero(groups == g); tr = np.flatnonzero(groups != g)
            oof[te] = train_and_predict(cfg, data, tr, te, seed=100 + k + SEED_OFFSET)
            per[str(g)] = metrics(data["y"][te], oof[te], types[te])
            print(f"  {tag} test={g}: AUROC {per[str(g)]['AUROC']:.3f} AUPRC {per[str(g)]['AUPRC']:.3f} "
                  f"({time.time() - t1:.0f} s)", flush=True)
        summary = {f"{m}_{s}": round(float(fn([g[m] for g in per.values()])), 4)
                   for m in ("AUROC", "AUPRC", "F1", "AUROC_vs_near", "AUROC_vs_random", "chance_AUPRC")
                   for s, fn in (("mean", np.mean), ("sd", np.std))}
        res = dict(config=name, labels=src, **cfg, epochs=EPOCHS, overall=metrics(data["y"], oof, types), folds=summary,
                   per_group=per, minutes=round((time.time() - t1) / 60, 1))
        (RES / f"ft_{tag}.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
        with open(RES / f"oof_ft_{tag}.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh); w.writerow(["sample_id", "sample_type", "label", "group", "score"])
            w.writerows([[i, t, int(v), g, round(float(p), 5)] for i, t, v, g, p in zip(ids, types, data["y"], groups, oof)])
        o = res["overall"]
        print(f"[done] {tag}: AUROC {o['AUROC']:.3f} AUPRC {o['AUPRC']:.3f} (chance {o['chance_AUPRC']}) F1 {o['F1']:.3f} "
              f"vs-near {o['AUROC_vs_near']:.3f} in {res['minutes']} min", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
