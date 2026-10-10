"""Results tables and figures for the Buildings manuscript from road_v2/results_noto/ (English labels).

The manuscript presents a segment-based bi-temporal change-detection method. The proposed network (convnext_siamese_v2,
finetune_noto_v2.py: global features and a local change map pooled along the road corridor, fused and post-event heads)
is the reference for every analysis; the global-fusion Siamese network (convnext_siamese), the single-temporal networks
and the v2 ablations are baselines and ablations.

Inputs (one label source at a time, --labels candidate|verified):
  frozen_<labels>.json, oof_frozen_<labels>.csv, loro_frozen_<labels>.csv, change_<labels>.json   (baseline_noto.py,
  change_features.py)
  ft_<config>_<labels>[_loro].json, oof_ft_<config>_<labels>[_loro].csv   (finetune_noto.py)
Outputs in paper_buildings/:
  tables/results_<labels>.tex         model comparison (mean +- SD over the spatial folds)
  tables/hardneg_<labels>.tex         AUROC against each kind of negative (pooled out-of-fold scores)
  tables/seeds_<labels>.tex           three seeds of the proposed and the single-temporal network
  tables/ablation_<labels>.tex        ablations of the proposed network (first seed)
  figures/fig4_models_<labels>.png    AUROC and AUPRC per model, chance level marked
  figures/fig5_loro_<labels>.png      leave-one-region-out AUROC/AUPRC per region, with the same region in spatial CV
  figures/fig6_screening_<labels>.png share of the network to inspect vs recall at the population prevalence
  figures/fig7_examples_<labels>.jpg  pre/post pairs of confident hits, false alarms and misses of the proposed model
  tables/key_numbers_<labels>.json    numbers quoted in the text
  tables/numbers_<labels>.tex         the same numbers as LaTeX macros (red unless --labels verified)
Usage:
  python -I road_v2/paper_results.py [--labels candidate|verified]
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageFilter
from sklearn.metrics import average_precision_score, roc_auc_score

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent
RES = ROOT / "results_noto"
DS = ROOT / "dataset_noto2024_v0"
PAPER = ROOT.parent / "paper_buildings"
POPULATION, POP_CANDIDATES = 102368, 1123  # road segments in the interpretation extent; GSI candidates among them
plt.rcParams.update({"font.family": ["Arial", "DejaVu Sans"], "font.size": 9, "axes.edgecolor": "#888888",
                     "axes.spines.top": False, "axes.spines.right": False})
FROZEN = [("road class only (no pixels)", "Road class only (no image)"),
          ("colour stats, after", "Colour statistics, post"),
          ("ConvNeXt-T, after", "ConvNeXt-T frozen, post"),
          ("DINOv2-B, after", "DINOv2-B frozen, post"),
          ("DINOv2-B, after + road-pooled", "DINOv2-B frozen, post, road-pooled"),
          ("DINOv2-B, before+after + road-pooled", "DINOv2-B frozen, pre + post, road-pooled")]
PROP_CFG = "convnext_siamese_v2"  # the proposed network
PROPOSED = "Proposed network, pre + post"
GLOBAL = "Siamese ConvNeXt-T, global fusion, pre + post"  # conventional late-fusion Siamese network
FT = [("resnet50_post", "ResNet-50 fine-tuned, post"), ("swin_post", "Swin-T fine-tuned, post"),
      ("convnext_post", "ConvNeXt-T fine-tuned, post"), ("convnext_post_mask", "ConvNeXt-T fine-tuned, post + road mask"),
      ("convnext_siamese", GLOBAL), (PROP_CFG, PROPOSED)]
ABLATIONS = [(PROP_CFG, "Full", "Proposed network"),
             (PROP_CFG + "_noalign", "Noalign", "Without neighbourhood search (co-located cells only)"),
             (PROP_CFG + "_nocorridor", "Nocorr", "Without road-corridor weighting"),
             (PROP_CFG + "_nolocal", "Nolocal", "Without local change branch"),
             ("convnext_siamese", "Glob", "Global fusion with a single head"),
             (PROP_CFG + "_aux", "Aux", "With auxiliary deposition-mask loss"),
             (PROP_CFG + ":fused", "Fused", "Fused head alone"),
             (PROP_CFG + ":post", "Posthead", "Post-event head alone"),
             ("convnext_post", "Post", "Without pre-event image (single-temporal)")]
REGION_EN = {"suzu-wajimaEAST": "Suzu–\nWajima East", "wajimaCENTRAL": "Wajima\nCentral", "wajimaWEST": "Wajima\nWest",
             "anamizu": "Anamizu", "nanao": "Nanao"}
REGION_TEX = {"suzu-wajimaEAST": "Suzu--Wajima East", "wajimaCENTRAL": "Wajima Central", "wajimaWEST": "Wajima West",
              "anamizu": "Anamizu", "nanao": "Nanao"}
BLUE, ORANGE, RED, GREY, INK, PURPLE = "#2f6fb0", "#d9822b", "#b03a1e", "#9a9a9a", "#222222", "#7b5ea7"
WORDS = {0: "none", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five"}


def labels_arg():
    a = sys.argv[1:]
    return a[a.index("--labels") + 1] if "--labels" in a else "candidate"


def read_csv(p):
    with open(p, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def blocks_of():
    with open(DS / "samples.csv", encoding="utf-8-sig") as f:
        return {r["sample_id"]: r for r in csv.DictReader(f)}


SAMPLES = blocks_of()


def oof(name, src, loro=False):
    """Out-of-fold scores of a fine-tuned configuration as arrays (ids, sample type, label, score); "<config>:fused" and
    "<config>:post" select one head of the proposed network."""
    name, _, head = name.partition(":")
    p = RES / f"oof_ft_{name}_{src}{'_loro' if loro else ''}.csv"
    if not p.exists():
        return None
    rows = read_csv(p)
    d = dict(id=np.array([r["sample_id"] for r in rows]), t=np.array([r["sample_type"] for r in rows]),
             y=np.array([int(r["label"]) for r in rows]), s=np.array([float(r["score"]) for r in rows]))
    if head:
        h = {r["sample_id"]: float(r[f"score_{head}"]) for r in read_csv(p.with_name(p.stem + "_heads.csv"))}
        d["s"] = np.array([h[i] for i in d["id"]])
    return d


def oof_frozen(column, src):
    rows = read_csv(RES / f"oof_frozen_{src}.csv")
    return dict(id=np.array([r["sample_id"] for r in rows]), t=np.array([r["sample_type"] for r in rows]),
                y=np.array([int(r["label"]) for r in rows]), s=np.array([float(r[column]) for r in rows]))


def collect(src):
    rows = []
    fz = json.loads((RES / f"frozen_{src}.json").read_text(encoding="utf-8"))
    by = {t["features"]: t for t in fz["cv"]}
    stats = ("AUROC", "AUPRC", "F1", "AUROC_vs_near", "chance_AUPRC")
    for key, name in FROZEN:
        t = by[key]
        rows.append(dict(name=name, kind="frozen", **{f"{m}_{s}": t[f"{m}_{s}"] for m in stats for s in ("mean", "sd")}))
    chg = RES / f"change_{src}.json"  # road-focused pre/post change features (change_features.py)
    if chg.exists():
        c = json.loads(chg.read_text(encoding="utf-8"))
        for key, name in (("road change only", "DINOv2-B frozen, road change only (pre vs. post)"),
                          ("post + road change", "DINOv2-B frozen, post, road-pooled + road change")):
            rows.append(dict(name=name, kind="frozen", **{f"{m}_{s}": c[key][f"{m}_{s}"] for m in stats for s in ("mean", "sd")}))
    for key, name in FT:
        p = RES / f"ft_{key}_{src}.json"
        if not p.exists():
            continue
        f = json.loads(p.read_text(encoding="utf-8"))["folds"]
        kind = "proposed" if name == PROPOSED else ("bitemporal" if name == GLOBAL else "fine-tuned")
        rows.append(dict(name=name, kind=kind, **{k: f[k] for k in f}))
    return rows, fz


def table_tex(rows, src):
    out, prev = [], None
    red = src != "verified"  # numbers from unverified labels are shown in red in the draft
    for r in rows:
        cell = lambda m: (f"\\textcolor{{red}}{{{r[m + '_mean']:.3f} $\\pm$ {r[m + '_sd']:.3f}}}" if red
                          else f"{r[m + '_mean']:.3f} $\\pm$ {r[m + '_sd']:.3f}")
        group = "base" if r["name"].startswith(("Road class", "Colour")) else r["kind"]
        if out and group != prev:
            out.append("\\midrule")
        prev = group
        name = f"\\textbf{{{r['name']}}}" if r["kind"] == "proposed" else r["name"]
        out.append(f"{name} & {cell('AUROC')} & {cell('AUPRC')} & {cell('F1')} & {cell('AUROC_vs_near')} \\\\")
    (PAPER / "tables").mkdir(exist_ok=True)
    # MDPI wide table: the table extends into the left margin (\fulllength) so that "mean +- SD" fits on one line
    head = ["\\begin{adjustwidth}{-\\extralength}{0cm}",
            "\\begin{tabularx}{\\fulllength}{>{\\raggedright\\arraybackslash}p{6.3cm}CCCC}", "\\toprule",
            "\\textbf{Model and Input} & \\textbf{AUROC} & \\textbf{AUPRC} & \\textbf{F1} & \\textbf{AUROC vs. Near} \\\\", "\\midrule"]
    tail = ["\\bottomrule", "\\end{tabularx}", "\\end{adjustwidth}"]
    (PAPER / "tables" / f"results_{src}.tex").write_text("\n".join(head + out + tail) + "\n", encoding="utf-8")


def fig_models(rows, src):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 4.5), sharey=True)
    yy = np.arange(len(rows))[::-1]
    colour = {"frozen": BLUE, "fine-tuned": ORANGE, "bitemporal": PURPLE, "proposed": RED}
    for ax, m in zip(axes, ("AUROC", "AUPRC")):
        v = [r[f"{m}_mean"] for r in rows]; e = [r[f"{m}_sd"] for r in rows]
        col = [GREY if r["name"].startswith(("Road class", "Colour")) else colour[r["kind"]] for r in rows]
        ax.barh(yy, v, xerr=e, color=col, height=0.65, error_kw=dict(lw=0.8, capsize=2))
        for y_, x_, e_ in zip(yy, v, e):
            ax.text(x_ + e_ + 0.012, y_, f"{x_:.2f}", va="center", fontsize=7.5)
        chance = 0.5 if m == "AUROC" else rows[0]["chance_AUPRC_mean"]
        ax.axvline(chance, color=INK, ls="--", lw=0.8)
        ax.text(chance + 0.01, yy[0] + 0.55, "chance", fontsize=7, color=INK)
        ax.set_xlim(0 if m == "AUPRC" else 0.4, 1.0); ax.set_title(m, fontsize=9.5)
    axes[0].set_yticks(yy, [r["name"] for r in rows], fontsize=8)
    for lab in axes[0].get_yticklabels():
        if lab.get_text() == PROPOSED:
            lab.set_fontweight("bold")
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color=GREY, label="No deep features"), Patch(color=BLUE, label="Frozen features + logistic regression"),
                        Patch(color=ORANGE, label="Fine-tuned, single-temporal"),
                        Patch(color=PURPLE, label="Fine-tuned, bi-temporal, global fusion"), Patch(color=RED, label="Proposed network")],
               fontsize=7.5, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.55, 0.0))
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    out = PAPER / "figures" / f"fig4_models_{src}.png"
    fig.savefig(out, dpi=300, facecolor="white"); plt.close(fig)
    return out


def block_bootstrap(ids, fn, n_boot=2000, seed=0):
    """Percentile interval of fn(selection) over resamples of the 2 km blocks."""
    blocks = np.array([SAMPLES[i]["block_id"] for i in ids])
    ub = np.unique(blocks); where = {u: np.flatnonzero(blocks == u) for u in ub}
    rng = np.random.default_rng(seed); vals = []
    for _ in range(n_boot):
        sel = np.concatenate([where[u] for u in rng.choice(ub, len(ub), replace=True)])
        try:
            v = fn(sel)
        except ValueError:
            continue
        if v is not None and np.all(np.isfinite(v)):
            vals.append(v)
    return np.percentile(np.array(vals), [2.5, 97.5], axis=0)


def auroc_vs(d, kind, sel=None):
    """AUROC of the blocked segments against one kind of negative: candidate-clear, near, random or all."""
    idx = np.arange(len(d["y"])) if sel is None else sel
    y, t, s = d["y"][idx], d["t"][idx], d["s"][idx]
    neg = (y == 0) if kind == "all" else (y == 0) & (t == kind)
    m = (y == 1) | neg
    if y[m].min() == y[m].max():
        raise ValueError("one class")
    return roc_auc_score(y[m], s[m])


def hard_negatives(src):
    """AUROC against each kind of negative for the key models, and the Siamese gain on candidate-clear negatives."""
    models = [("DINOv2-B frozen, post, road-pooled", oof_frozen("DINOv2-B, after + road-pooled", src)),
              ("ConvNeXt-T fine-tuned, post", oof("convnext_post", src)),
              ("ConvNeXt-T fine-tuned, post + road mask", oof("convnext_post_mask", src)),
              (GLOBAL, oof("convnext_siamese", src)),
              (PROPOSED, oof(PROP_CFG, src))]
    models = [(n, d) for n, d in models if d is not None]
    rows = []
    for name, d in models:
        rows.append(dict(name=name, **{k: auroc_vs(d, k) for k in ("all", "candidate", "near", "random")}))
    d0, d1 = oof("convnext_post", src), oof(PROP_CFG, src)
    gain = {}
    if d0 is not None and d1 is not None:
        assert (d0["id"] == d1["id"]).all()
        for kind in ("candidate", "near", "random", "all"):
            obs = auroc_vs(d1, kind) - auroc_vs(d0, kind)
            lo, hi = block_bootstrap(d1["id"], lambda sel, k=kind: auroc_vs(d1, k, sel) - auroc_vs(d0, k, sel))
            gain[kind] = dict(gain=obs, lo=lo, hi=hi)
    counts = {k: int(((models[0][1]["y"] == 0) & (models[0][1]["t"] == k)).sum()) for k in ("candidate", "near", "random")}
    red = src != "verified"
    f = (lambda v: f"\\textcolor{{red}}{{{v:.3f}}}") if red else (lambda v: f"{v:.3f}")
    lines = ["\\begin{tabularx}{\\textwidth}{>{\\raggedright\\arraybackslash}p{5.6cm}CCCC}", "\\toprule",
             "\\textbf{Model and Input} & \\textbf{All Negatives} & \\textbf{Candidate-Clear} & \\textbf{Near} & \\textbf{Random} \\\\",
             f" & ({sum(counts.values())}) & ({counts['candidate']}) & ({counts['near']}) & ({counts['random']}) \\\\", "\\midrule"]
    for r in rows:
        name = f"\\textbf{{{r['name']}}}" if r["name"] == PROPOSED else r["name"]
        lines.append(f"{name} & {f(r['all'])} & {f(r['candidate'])} & {f(r['near'])} & {f(r['random'])} \\\\")
    lines += ["\\bottomrule", "\\end{tabularx}"]
    (PAPER / "tables" / f"hardneg_{src}.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return rows, gain, counts


def screening(d):
    """Reweight the random stratum to the population; return recall, share of the network inspected, precision, and the
    sample types in ranking order."""
    keep = (d["t"] == "candidate") | (d["t"] == "random")
    y, s, t = d["y"][keep], d["s"][keep], d["t"][keep]
    n_cand = (t == "candidate").sum(); n_rand = (t == "random").sum()
    # candidates stand for all GSI candidates in the population, random segments for the remaining population
    w = np.where(t == "candidate", POP_CANDIDATES / n_cand, (POPULATION - POP_CANDIDATES) / n_rand)
    order = np.argsort(-s)
    y, w, t = y[order], w[order], t[order]
    tp = np.cumsum(w * y); flagged = np.cumsum(w)
    return tp / tp[-1], flagged / w.sum(), tp / flagged, tp[-1] / w.sum(), t, y


def at_recall(d, target):
    r, sh, pr, _, t, y = screening(d)
    i = int(np.searchsorted(r, target))
    return sh[i] * 100, pr[i] * 100, int(((t[:i + 1] == "random") & (y[:i + 1] == 0)).sum())


def screening_numbers(d):
    out = {}
    for target in (0.5, 0.8):
        share, prec, n_rand = at_recall(d, target)
        lo, hi = block_bootstrap(d["id"], lambda sel, tg=target: at_recall({k: v[sel] for k, v in d.items()}, tg)[0])
        out[int(target * 100)] = dict(share=share, precision=prec, random_above=n_rand, lo=lo, hi=hi)
    out["prevalence"] = screening(d)[3] * 100
    return out


def fig_screening(src):
    curves = [(oof(PROP_CFG, src), "Proposed network", RED, "-"),
              (oof("convnext_post", src), "ConvNeXt-T fine-tuned, post", ORANGE, "--")]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1))
    prev = None
    for d, name, col, ls in curves:
        if d is None:
            continue
        r, sh, pr, prev, _, _ = screening(d)
        axes[0].plot(sh * 100, r * 100, color=col, lw=1.6, ls=ls, label=name)
        axes[1].plot(r * 100, pr * 100, color=col, lw=1.6, ls=ls)
    xr = np.logspace(-1, 2, 200)  # random inspection finds blocked segments in proportion to the share inspected
    axes[0].plot(xr, xr, color=GREY, ls=":", lw=0.9, label="Random inspection")
    axes[0].set_xscale("log"); axes[0].set_xlim(0.1, 100); axes[0].set_ylim(0, 100)
    axes[0].set_xlabel("Share of road segments inspected (%)"); axes[0].set_ylabel("Blocked segments found (%)")
    axes[0].legend(fontsize=7, frameon=False, loc="upper left")
    axes[1].set_xlabel("Recall (%)"); axes[1].set_ylabel("Precision (%)"); axes[1].set_ylim(0, 100)
    if prev is not None:
        axes[1].axhline(prev * 100, color=GREY, ls=":", lw=0.9)
        axes[1].text(2, prev * 100 + 2, "random inspection", fontsize=7, color=GREY)
    fig.tight_layout()
    out = PAPER / "figures" / f"fig6_screening_{src}.png"
    fig.savefig(out, dpi=300, facecolor="white"); plt.close(fig)
    return out


def loro_numbers(src, fz, n_boot=2000):
    """Leave-one-region-out: per-region and pooled scores, the same region's AUROC within spatial cross-validation, and
    block-bootstrap differences to the single-temporal network."""
    out = {}
    for name in (PROP_CFG, "convnext_siamese", "convnext_post", PROP_CFG + ":fused", PROP_CFG + ":post"):
        d = oof(name, src, loro=True)
        if d is None:
            continue
        out[name + "_pooled"] = dict(AUROC=roc_auc_score(d["y"], d["s"]), AUPRC=average_precision_score(d["y"], d["s"]),
                                     Cand=auroc_vs(d, "candidate"))
        if ":" not in name:
            out[name] = json.loads((RES / f"ft_{name}_{src}_loro.json").read_text(encoding="utf-8"))["per_group"]
    d = oof(PROP_CFG, src)
    if d is not None:
        reg = np.array([SAMPLES[i]["region"] for i in d["id"]])
        out["spatial_cv_by_region"] = {g: roc_auc_score(d["y"][reg == g], d["s"][reg == g]) for g in np.unique(reg)}
        pre = np.array([SAMPLES[i]["pre_date"] for i in d["id"]])
        out["old_pre_share_by_region"] = {g: float(np.isin(pre[reg == g], ["2010", "2014"]).mean()) for g in np.unique(reg)}
    out["frozen"] = {t["region"]: t for t in fz["loro"] if t["features"] == "DINOv2-B, after + road-pooled"}
    for name, ref in ((PROP_CFG, "convnext_post"), ("convnext_siamese", "convnext_post"), (PROP_CFG, "convnext_siamese")):
        a, b = oof(name, src, loro=True), oof(ref, src, loro=True)
        if a is None or b is None:
            continue
        assert (a["id"] == b["id"]).all()
        diff = lambda sel, a=a, b=b: np.array([
            roc_auc_score(a["y"][sel], a["s"][sel]) - roc_auc_score(b["y"][sel], b["s"][sel]),
            average_precision_score(a["y"][sel], a["s"][sel]) - average_precision_score(b["y"][sel], b["s"][sel]),
            auroc_vs(a, "candidate", sel) - auroc_vs(b, "candidate", sel)])
        lo, hi = block_bootstrap(a["id"], diff, n_boot)
        out[f"{name}_vs_{ref}"] = dict(obs=diff(np.arange(len(a["y"]))).tolist(), lo=lo.tolist(), hi=hi.tolist())
    return out


def fig_loro(src, lo):
    regs = list(REGION_EN)
    series = [(lo.get(PROP_CFG), "Proposed network", RED),
              (lo.get("convnext_siamese"), "Siamese, global fusion", PURPLE),
              (lo.get("convnext_post"), "ConvNeXt-T fine-tuned, post", ORANGE),
              (lo.get("frozen"), "DINOv2-B frozen, road-pooled", BLUE)]
    series = [s for s in series if s[0]]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1))
    x = np.arange(len(regs)); w = 0.8 / len(series)
    n_pos = {g: lo["frozen"][g]["n_pos"] for g in regs}
    for ax, m in zip(axes, ("AUROC", "AUPRC")):
        for k, (data, name, col) in enumerate(series):
            ax.bar(x - 0.4 + w * (k + 0.5), [data[g][m] for g in regs], w, color=col, label=name)
        if m == "AUROC" and "spatial_cv_by_region" in lo:
            cv = lo["spatial_cv_by_region"]
            ax.scatter(x - 0.4 + w * 0.5, [cv[g] for g in regs], marker="D", s=16, color=INK, zorder=5,
                       label="Same region, spatial CV (proposed)")
        if m == "AUPRC":
            ax.scatter(x, [lo["frozen"][g]["chance_AUPRC"] for g in regs], marker="_", s=300, color=INK, zorder=5,
                       label="Chance (prevalence)")
        ax.set_xticks(x, [f"{REGION_EN[g]}\n({n_pos[g]})" for g in regs], fontsize=7.5)
        ax.set_ylim(0.5 if m == "AUROC" else 0.0, 1.0); ax.set_title(m, fontsize=9.5)
    handles, labels_ = [], []
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        for hh, ll in zip(h, l):
            if ll not in labels_:
                handles.append(hh); labels_.append(ll)
    fig.legend(handles, labels_, fontsize=7, frameon=False, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.07))
    fig.text(0.5, -0.03, "Held-out region (number of blocked segments)", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    out = PAPER / "figures" / f"fig5_loro_{src}.png"
    fig.savefig(out, dpi=300, facecolor="white", bbox_inches="tight"); plt.close(fig)
    return out


def outline(sid):
    m = Image.open(DS / "masks/road" / f"{sid}.png")
    corridor = np.array(m.filter(ImageFilter.MaxFilter(15))) > 0
    inner = np.array(Image.fromarray(corridor.astype(np.uint8) * 255).filter(ImageFilter.MinFilter(5))) > 0
    return corridor & ~inner


def fig_examples(src, n=2, overrides=None):
    d = oof(PROP_CFG, src)
    rows = [dict(sample_id=i, label=int(y), score=float(s)) for i, y, s in zip(d["id"], d["y"], d["s"])]
    rng = np.random.default_rng(3)
    pick = lambda cand: [cand[i] for i in sorted(rng.choice(len(cand), min(n, len(cand)), replace=False))]
    groups = [("Detected", pick([r for r in rows if r["label"] == 1 and r["score"] > 0.9])),
              ("False alarm", pick([r for r in rows if r["label"] == 0 and r["score"] > 0.8])),
              ("Missed", pick([r for r in rows if r["label"] == 1 and r["score"] < 0.2]))]
    # A visually selected replacement for the second false-alarm pair in the final figure.
    replacements = {(1, 1): 'noto_01082'} if overrides is None and src == 'verified' and n == 2 else (overrides or {})
    by_id = {r['sample_id']: r for r in rows}
    for (i, j), sid in replacements.items():
        r = by_id[sid]
        assert (i == 0 and r['label'] == 1 and r['score'] > .9) or (i == 1 and r['label'] == 0 and r['score'] > .8) or (i == 2 and r['label'] == 1 and r['score'] < .2)
        groups[i][1][j] = r
    records = [dict(r, row=i+1, image_columns=[2*j+1, 2*j+2], category=g,
                    selection='visually selected replacement' if (i, j) in replacements else 'seeded random draw (seed 3)')
               for i, (g, items) in enumerate(groups) for j, r in enumerate(items)]
    fig, axes = plt.subplots(3, 2 * n, figsize=(7.2, 5.6))
    for i, (g, items) in enumerate(groups):
        for j in range(n):
            for k, kind in enumerate(("pre", "post")):
                a = axes[i, 2 * j + k]; a.set_xticks([]); a.set_yticks([])
                if j >= len(items):
                    a.axis("off"); continue
                r = items[j]
                rgb = np.array(Image.open(DS / f"chips/{kind}" / f"{r['sample_id']}.jpg").convert("RGB"))
                rgb[outline(r["sample_id"])] = (255, 43, 214)
                a.imshow(rgb)
                a.set_title(("Pre-event" if kind == "pre" else f"Post-event, score {r['score']:.2f}"), fontsize=7.5, pad=2)
                if j == 0 and k == 0:
                    a.set_ylabel(g, fontsize=9)
    fig.tight_layout(h_pad=0.6, w_pad=0.3)
    out = PAPER / "figures" / f"fig7_examples_{src}.jpg"
    fig.savefig(out, dpi=300, facecolor="white", pil_kwargs={"quality": 92}); plt.close(fig)
    (PAPER / 'figures' / f'qualitative_provenance_{src}.json').write_text(json.dumps(records, indent=2), encoding='utf-8')
    return out


def siamese_gain(src, n_boot=2000):
    """Siamese (pre + post) minus post-only ConvNeXt-T: folds won, and a block bootstrap (2 km blocks) of the
    difference in pooled out-of-fold AUROC and AUPRC."""
    pa, pb = RES / f"oof_ft_convnext_post_{src}.csv", RES / f"oof_ft_{PROP_CFG}_{src}.csv"
    if not (pa.exists() and pb.exists()):
        return {}
    a = {r["sample_id"]: r for r in read_csv(pa)}; b = {r["sample_id"]: r for r in read_csv(pb)}
    ids = sorted(a)
    y = np.array([int(a[i]["label"]) for i in ids]); s0 = np.array([float(a[i]["score"]) for i in ids])
    s1 = np.array([float(b[i]["score"]) for i in ids]); g = np.array([a[i]["group"] for i in ids])
    won = sum(roc_auc_score(y[g == f], s1[g == f]) > roc_auc_score(y[g == f], s0[g == f]) for f in np.unique(g))
    diff = lambda sel: (roc_auc_score(y[sel], s1[sel]) - roc_auc_score(y[sel], s0[sel]),
                        average_precision_score(y[sel], s1[sel]) - average_precision_score(y[sel], s0[sel]))
    obs = diff(np.arange(len(y)))
    lo, hi = block_bootstrap(np.array(ids), lambda sel: np.array(diff(sel)) if y[sel].min() != y[sel].max() else None, n_boot)
    return dict(folds_won=int(won), n_folds=int(len(np.unique(g))), auroc=obs[0], auroc_lo=lo[0], auroc_hi=hi[0],
                auprc=obs[1], auprc_lo=lo[1], auprc_hi=hi[1])


def signed(v):
    t = f"{v:+.3f}"
    return "$-$" + t[1:] if t[0] == "-" else t


def seed_stats(src, n_boot=2000):
    """Pooled metrics of the three seeds of the proposed and the single-temporal network (mean and SD), and a block
    bootstrap of the difference between their seed-averaged scores."""
    out, avg = {}, {}
    for cfg in ("convnext_post", PROP_CFG):
        runs = [d for d in (oof(n, src) for n in (cfg, cfg + "_s2", cfg + "_s3")) if d is not None]
        if len(runs) < 2:
            return {}
        for d in runs[1:]:
            assert (d["id"] == runs[0]["id"]).all()
        m = np.array([[roc_auc_score(d["y"], d["s"]), average_precision_score(d["y"], d["s"]), auroc_vs(d, "candidate"),
                       at_recall(d, 0.8)[0]] for d in runs])
        out[cfg] = dict(n=len(runs), per_seed=m.tolist(), mean=m.mean(0).tolist(), sd=m.std(0, ddof=1).tolist())
        avg[cfg] = dict(runs[0], s=np.mean([d["s"] for d in runs], axis=0))
    a, b = avg[PROP_CFG], avg["convnext_post"]
    diff = lambda sel: np.array([roc_auc_score(a["y"][sel], a["s"][sel]) - roc_auc_score(b["y"][sel], b["s"][sel]),
                                 average_precision_score(a["y"][sel], a["s"][sel]) - average_precision_score(b["y"][sel], b["s"][sel]),
                                 auroc_vs(a, "candidate", sel) - auroc_vs(b, "candidate", sel)])
    lo, hi = block_bootstrap(a["id"], diff, n_boot)
    out["gain"] = dict(obs=diff(np.arange(len(a["y"]))).tolist(), lo=lo.tolist(), hi=hi.tolist())
    pp, po = np.array(out[PROP_CFG]["per_seed"]), np.array(out["convnext_post"]["per_seed"])
    out["all_runs_better"] = [bool(pp[:, j].min() > po[:, j].max()) for j in range(3)]
    return out


def seeds_tex(st, src):
    lines = ["\\begin{adjustwidth}{-\\extralength}{0cm}",
             "\\begin{tabularx}{\\fulllength}{>{\\raggedright\\arraybackslash}p{5.2cm}CCCC}", "\\toprule",
             "\\textbf{Model and Input} & \\textbf{AUROC} & \\textbf{AUPRC} & \\textbf{AUROC vs. Cand.-Clear} & "
             "\\textbf{Share at 80\\% Recall (\\%)} \\\\", "\\midrule"]
    for name, cfg in (("ConvNeXt-T fine-tuned, post", "convnext_post"), (PROPOSED, PROP_CFG)):
        r = st[cfg]
        cells = [f"{r['mean'][k]:.3f} $\\pm$ {r['sd'][k]:.3f}" for k in range(3)] + [f"{r['mean'][3]:.2f} $\\pm$ {r['sd'][3]:.2f}"]
        lines.append((f"\\textbf{{{name}}}" if name == PROPOSED else name) + " & " + " & ".join(cells) + " \\\\")
    g = st["gain"]
    lines += ["\\midrule", "Difference of the seed-averaged scores & " + " & ".join(
        f"{signed(g['obs'][k])} [{signed(g['lo'][k])}, {signed(g['hi'][k])}]" for k in range(3)) + " & -- \\\\",
              "\\bottomrule", "\\end{tabularx}", "\\end{adjustwidth}"]
    (PAPER / "tables" / f"seeds_{src}.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def ablations(src, n_boot=2000):
    """Ablations of the proposed network (spatially blocked CV, first seed): pooled AUROC, AUPRC and AUROC against
    candidate-clear negatives, and block-bootstrap differences to the full network."""
    full = oof(PROP_CFG, src)
    if full is None:
        return []
    rows = []
    for cfg, tag, name in ABLATIONS:
        d = oof(cfg, src)
        if d is None:
            continue
        assert (d["id"] == full["id"]).all()
        r = dict(tag=tag, name=name, AUROC=roc_auc_score(d["y"], d["s"]), AUPRC=average_precision_score(d["y"], d["s"]),
                 Cand=auroc_vs(d, "candidate"))
        if cfg != PROP_CFG:
            diff = lambda sel, d=d: np.array([roc_auc_score(d["y"][sel], d["s"][sel]) - roc_auc_score(full["y"][sel], full["s"][sel]),
                                              auroc_vs(d, "candidate", sel) - auroc_vs(full, "candidate", sel)])
            lo, hi = block_bootstrap(d["id"], diff, n_boot)
            r.update(dAUROC=r["AUROC"] - rows[0]["AUROC"], dAUROCLo=lo[0], dAUROCHi=hi[0],
                     dCand=r["Cand"] - rows[0]["Cand"], dCandLo=lo[1], dCandHi=hi[1])
        rows.append(r)
    ci = lambda r, k: f"{signed(r[k])} [{signed(r[k + 'Lo'])}, {signed(r[k + 'Hi'])}]" if k in r else "--"
    lines = ["\\begin{adjustwidth}{-\\extralength}{0cm}",
             "\\begin{tabularx}{\\fulllength}{>{\\raggedright\\arraybackslash}p{5.4cm}CCCCC}", "\\toprule",
             "\\textbf{Configuration} & \\textbf{AUROC} & \\textbf{AUPRC} & \\textbf{AUROC vs. Cand.-Clear} & "
             "\\textbf{$\\Delta$AUROC [95\\% CI]} & \\textbf{$\\Delta$Cand.-Clear [95\\% CI]} \\\\", "\\midrule"]
    for r in rows:
        name = f"\\textbf{{{r['name']}}}" if r["tag"] == "Full" else r["name"]
        lines.append(f"{name} & {r['AUROC']:.3f} & {r['AUPRC']:.3f} & {r['Cand']:.3f} & {ci(r, 'dAUROC')} & {ci(r, 'dCand')} \\\\")
        if r["tag"] in ("Full", "Glob", "Aux", "Posthead"):
            lines.append("\\midrule")
    lines += ["\\bottomrule", "\\end{tabularx}", "\\end{adjustwidth}"]
    (PAPER / "tables" / f"ablation_{src}.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return rows


def number_macros(rows, key, fz, src):
    """LaTeX macros for every number quoted in the text; red while the labels are not verified."""
    wrap = (lambda v: f"\\textcolor{{red}}{{{v}}}") if src != "verified" else (lambda v: f"{v}")
    by = {r["name"]: r for r in rows}
    fzr = [r for r in rows if r["kind"] == "frozen" and not r["name"].startswith(("Road class", "Colour"))
           and "road change" not in r["name"]]  # the change-feature rows are an ablation, not a separate model
    best_fz = max(fzr, key=lambda r: r["AUPRC_mean"])
    m = dict(NSeg=fz["n"], NBlk=fz["n_pos"], Prev=f"{fz['n_pos'] / fz['n']:.2f}",
             RCauroc=f"{by['Road class only (no image)']['AUROC_mean']:.2f}",
             COLauroc=f"{by['Colour statistics, post']['AUROC_mean']:.2f}",
             FZbestName=best_fz["name"].replace(", post", "").replace(" frozen", ""),
             FZauroc=f"{best_fz['AUROC_mean']:.3f}", FZauprc=f"{best_fz['AUPRC_mean']:.3f}",
             DINOglobalAUROC=f"{by['DINOv2-B frozen, post']['AUROC_mean']:.3f}",
             DINOroadAUROC=f"{by['DINOv2-B frozen, post, road-pooled']['AUROC_mean']:.3f}",
             DINObiAUROC=f"{by['DINOv2-B frozen, pre + post, road-pooled']['AUROC_mean']:.3f}",
             CNXfrozenAUROC=f"{by['ConvNeXt-T frozen, post']['AUROC_mean']:.3f}")
    if PROPOSED in by:
        p = by[PROPOSED]
        m.update(FTauroc=f"{p['AUROC_mean']:.3f}", FTaurocSD=f"{p['AUROC_sd']:.3f}", FTauprc=f"{p['AUPRC_mean']:.3f}",
                 FTauprcSD=f"{p['AUPRC_sd']:.3f}", FTfone=f"{p['F1_mean']:.2f}", FTnear=f"{p['AUROC_vs_near_mean']:.3f}",
                 FTrandom=f"{p['AUROC_vs_random_mean']:.3f}")
    for k, name in (("FTpost", "ConvNeXt-T fine-tuned, post"), ("FTmask", "ConvNeXt-T fine-tuned, post + road mask"),
                    ("FTres", "ResNet-50 fine-tuned, post"), ("FTswin", "Swin-T fine-tuned, post"), ("FTglob", GLOBAL)):
        if name in by:
            m[f"{k}AUROC"] = f"{by[name]['AUROC_mean']:.3f}"; m[f"{k}AUPRC"] = f"{by[name]['AUPRC_mean']:.3f}"
            m[f"{k}Fone"] = f"{by[name]['F1_mean']:.2f}"; m[f"{k}Near"] = f"{by[name]['AUROC_vs_near_mean']:.3f}"
    if "DINOv2-B frozen, road change only (pre vs. post)" in by:
        m["CHGonlyAUROC"] = f"{by['DINOv2-B frozen, road change only (pre vs. post)']['AUROC_mean']:.3f}"
        m["CHGplusAUROC"] = f"{by['DINOv2-B frozen, post, road-pooled + road change']['AUROC_mean']:.3f}"
    sg = key.get("siamese_gain")
    if sg:
        m.update(SiamFoldsWon=WORDS.get(sg["folds_won"], str(sg["folds_won"])), SiamGainAUROC=f"{sg['auroc']:+.3f}",
                 SiamGainAUROCLo=f"{sg['auroc_lo']:+.3f}", SiamGainAUROCHi=f"{sg['auroc_hi']:+.3f}",
                 SiamGainAUPRC=f"{sg['auprc']:+.3f}", SiamGainAUPRCLo=f"{sg['auprc_lo']:+.3f}",
                 SiamGainAUPRCHi=f"{sg['auprc_hi']:+.3f}")
    hn = {r["name"]: r for r in key.get("hard_negatives", [])}
    if PROPOSED in hn:
        m.update(HNsiam=f"{hn[PROPOSED]['candidate']:.3f}", HNpost=f"{hn['ConvNeXt-T fine-tuned, post']['candidate']:.3f}",
                 HNmask=f"{hn['ConvNeXt-T fine-tuned, post + road mask']['candidate']:.3f}",
                 HNfz=f"{hn['DINOv2-B frozen, post, road-pooled']['candidate']:.3f}", HNglob=f"{hn[GLOBAL]['candidate']:.3f}",
                 PNsiamNear=f"{hn[PROPOSED]['near']:.3f}", PNsiamRandom=f"{hn[PROPOSED]['random']:.3f}",
                 PNpostNear=f"{hn['ConvNeXt-T fine-tuned, post']['near']:.3f}",
                 PNpostRandom=f"{hn['ConvNeXt-T fine-tuned, post']['random']:.3f}")
        hg = key["hard_negative_gain"]
        m.update(HNgain=f"{hg['candidate']['gain']:+.3f}", HNgainLo=f"{hg['candidate']['lo']:+.3f}",
                 HNgainHi=f"{hg['candidate']['hi']:+.3f}", NCandClear=key["negative_counts"]["candidate"],
                 NNear=key["negative_counts"]["near"], NRandom=key["negative_counts"]["random"])
    lo = key.get("loro", {})
    if PROP_CFG in lo:
        s = lo[PROP_CFG]; v = {g: s[g]["AUROC"] for g in s}
        lo_k = min(v, key=v.get); hi_k = max(v, key=v.get)
        cv = lo["spatial_cv_by_region"]; loss = {g: cv[g] - v[g] for g in v}
        pooled = lo[PROP_CFG + "_pooled"]
        m.update(LOROmin=f"{v[lo_k]:.2f}", LOROmax=f"{v[hi_k]:.2f}", LOROminArea=REGION_TEX[lo_k],
                 LOROmaxArea=REGION_TEX[hi_k], LOROminAreaPos=str(s[lo_k]["n_pos"]), LOROmaxAreaPos=str(s[hi_k]["n_pos"]),
                 LOROpooled=f"{pooled['AUROC']:.3f}", LOROpooledAUPRC=f"{pooled['AUPRC']:.3f}",
                 LOROpooledCand=f"{pooled['Cand']:.3f}",
                 LOROlossMean=f"{np.mean([loss[g] for g in loss if g != 'nanao']):.3f}",
                 LOROlossMin=f"{min(loss[g] for g in loss if g != 'nanao'):.3f}",
                 LOROlossMax=f"{max(loss[g] for g in loss if g != 'nanao'):.3f}",
                 CVregionMin=f"{min(cv.values()):.2f}", CVregionMax=f"{max(cv.values()):.2f}")
        for tag, cap, name in (("post", "Post", "convnext_post"), ("glob", "Glob", "convnext_siamese")):
            if name in lo:
                q = lo[name + "_pooled"]
                m.update({f"LORO{tag}Pooled": f"{q['AUROC']:.3f}", f"LORO{tag}PooledAUPRC": f"{q['AUPRC']:.3f}",
                          f"LORO{tag}PooledCand": f"{q['Cand']:.3f}"})
                better = sum(s[g]["AUROC"] > lo[name][g]["AUROC"] for g in s)
                m[f"LOROpropBetter{cap}"] = f"{WORDS.get(better, better)} of the {WORDS[len(s)]}"
        for tag, name, ref in (("Prop", PROP_CFG, "convnext_post"), ("Glob", "convnext_siamese", "convnext_post"),
                               ("PropGlob", PROP_CFG, "convnext_siamese")):
            dv = lo.get(f"{name}_vs_{ref}")
            for k, nm in enumerate(("AUROC", "AUPRC", "Cand")) if dv else ():
                m[f"LORO{tag}d{nm}"] = f"{dv['obs'][k]:+.3f}"
                m[f"LORO{tag}d{nm}Lo"] = f"{dv['lo'][k]:+.3f}"; m[f"LORO{tag}d{nm}Hi"] = f"{dv['hi'][k]:+.3f}"
        for tag, name in (("Fused", PROP_CFG + ":fused"), ("Posthead", PROP_CFG + ":post")):
            if name + "_pooled" in lo:
                m[f"LORO{tag}Pooled"] = f"{lo[name + '_pooled']['AUROC']:.3f}"
    if "old_pre_share_by_region" in lo:
        m["NanaoOldPre"] = f"{100 * lo['old_pre_share_by_region']['nanao']:.0f}"
    fv = [t["AUROC"] for t in lo.get("frozen", {}).values()]
    if fv:
        m.update(LOROfzMin=f"{min(fv):.2f}", LOROfzMax=f"{max(fv):.2f}", NanaoPos=str(lo["frozen"]["nanao"]["n_pos"]))
    sc = key.get("screening", {})
    for tag, name in (("", PROP_CFG), ("Post", "convnext_post")):
        if name not in sc:
            continue
        s = sc[name]
        m[f"Scr{tag}ShareFifty"] = f"{s['50']['share']:.2f}"
        m[f"Scr{tag}ShareFiftyLo"] = f"{s['50']['lo']:.2f}"; m[f"Scr{tag}ShareFiftyHi"] = f"{s['50']['hi']:.2f}"
        m[f"Scr{tag}PrecFifty"] = f"{s['50']['precision']:.1f}"
        m[f"Scr{tag}ShareEighty"] = f"{s['80']['share']:.2f}"
        m[f"Scr{tag}ShareEightyLo"] = f"{s['80']['lo']:.2f}"; m[f"Scr{tag}ShareEightyHi"] = f"{s['80']['hi']:.2f}"
        m[f"Scr{tag}PrecEighty"] = f"{s['80']['precision']:.1f}"
        m[f"Scr{tag}RandFifty"] = str(s["50"]["random_above"]); m[f"Scr{tag}RandEighty"] = str(s["80"]["random_above"])
        m[f"Scr{tag}FactorFifty"] = f"{50 / s['50']['share']:.0f}"; m[f"Scr{tag}FactorEighty"] = f"{80 / s['80']['share']:.0f}"
        m[f"Scr{tag}FactorEightyLo"] = f"{80 / s['80']['hi']:.0f}"; m[f"Scr{tag}FactorEightyHi"] = f"{80 / s['80']['lo']:.0f}"
        m["PopPrev"] = f"{s['prevalence']:.1f}"
    st = key.get("seeds")
    if st:
        for tag, cfg in (("Post", "convnext_post"), ("Prop", PROP_CFG)):
            mm, sd = st[cfg]["mean"], st[cfg]["sd"]
            for k, nm in enumerate(("AUROC", "AUPRC", "Cand")):
                m[f"Seed{tag}{nm}"] = f"{mm[k]:.3f}"; m[f"Seed{tag}{nm}SD"] = f"{sd[k]:.3f}"
            m[f"Seed{tag}Share"] = f"{mm[3]:.2f}"; m[f"Seed{tag}ShareSD"] = f"{sd[3]:.2f}"
        m["NSeeds"] = WORDS[st[PROP_CFG]["n"]]
        g = st["gain"]
        for k, nm in enumerate(("AUROC", "AUPRC", "Cand")):
            m[f"SeedGain{nm}"] = f"{g['obs'][k]:+.3f}"
            m[f"SeedGain{nm}Lo"] = f"{g['lo'][k]:+.3f}"; m[f"SeedGain{nm}Hi"] = f"{g['hi'][k]:+.3f}"
    for r in key.get("ablations", []):
        t = r["tag"]
        m.update({f"AB{t}AUROC": f"{r['AUROC']:.3f}", f"AB{t}AUPRC": f"{r['AUPRC']:.3f}", f"AB{t}Cand": f"{r['Cand']:.3f}"})
        for nm in ("dAUROC", "dAUROCLo", "dAUROCHi", "dCand", "dCandLo", "dCandHi") if "dAUROC" in r else ():
            m[f"AB{t}{nm}"] = f"{r[nm]:+.3f}"
    for k, v in m.items():  # typographic minus for negative signed numbers
        if isinstance(v, str) and v.startswith("-") and v[1:2].isdigit():
            m[k] = "$-$" + v[1:]
    lines = [f"% generated by road_v2/paper_results.py --labels {src}; do not edit"]
    lines += [f"\\newcommand{{\\{k}}}{{{wrap(v)}}}" for k, v in m.items()]
    (PAPER / "tables" / f"numbers_{src}.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return m


def main() -> int:
    src = labels_arg()
    (PAPER / "figures").mkdir(exist_ok=True)
    rows, fz = collect(src)
    table_tex(rows, src)
    key = dict(n=fz["n"], n_blocked=fz["n_pos"])
    key["models"] = {r["name"]: {k: r[k] for k in r if k.endswith(("_mean", "_sd"))} for r in rows}
    hn_rows, hn_gain, counts = hard_negatives(src)
    key.update(hard_negatives=hn_rows, hard_negative_gain=hn_gain, negative_counts=counts)
    key["siamese_gain"] = siamese_gain(src)
    key["seeds"] = seed_stats(src)
    if key["seeds"]:
        seeds_tex(key["seeds"], src)
    key["ablations"] = ablations(src)
    key["loro"] = loro_numbers(src, fz)
    key["screening"] = {name: {str(k): v for k, v in screening_numbers(d).items()}
                        for name, d in ((PROP_CFG, oof(PROP_CFG, src)), ("convnext_post", oof("convnext_post", src)))
                        if d is not None}
    outs = [fig_models(rows, src), fig_loro(src, key["loro"]), fig_screening(src), fig_examples(src)]
    macros = number_macros(rows, key, fz, src)
    (PAPER / "tables" / f"key_numbers_{src}.json").write_text(json.dumps(key, indent=1, ensure_ascii=False, default=float),
                                                              encoding="utf-8")
    for o in outs:
        print("[done]", o)
    print(json.dumps(macros, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
