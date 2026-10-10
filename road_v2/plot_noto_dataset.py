"""Figures for road_v2/dataset_noto2024_v0: random example pairs (pre | post) and dataset statistics.

Usage:
  python -I road_v2/plot_noto_dataset.py
Writes figures/ inside the dataset folder (NOTO_DS overrides the dataset path):
  examples_candidate.png  15 random candidate-blocked samples (before | January | April)
  examples_clear.png      6 random near-clear + 6 random random-clear samples
  dataset_stats.png       map, road classes, photo dates, areas, label agreement, overlap depth, folds
Overlays: target road piece (red = candidate blocked, green = clear), GSI aerial landslide/deposit polygon
outline (orange), GSI LiDAR deposit polygon outline (blue). Imagery: "出典：国土地理院".
"""
from __future__ import annotations

import csv
import json
import os
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageFilter

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent
DS = Path(os.environ.get("NOTO_DS", ROOT / "dataset_noto2024_v0"))
FIG = DS / "figures"
RED, GREEN, ORANGE, BLUE, GREY, INK = "#e34948", "#1baf7a", "#eda100", "#3d8fd6", "#9a9a9a", "#2b2b2b"
BG = "#fcfcfb"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": "#cccccc", "axes.labelcolor": INK, "xtick.color": INK,
                     "ytick.color": INK, "font.family": ["Microsoft YaHei", "DejaVu Sans"]})


def load():
    with open(DS / "samples.csv", encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    geo = json.loads((DS / "samples.geojson").read_text(encoding="utf-8"))["features"]
    lines = {f["properties"]["sample_id"]: f["geometry"]["coordinates"] for f in geo}
    return rows, lines


def px(coords, x0, y0):
    lon, lat = np.asarray(coords).T
    n = 256 * 2 ** 18
    return (lon + 180) / 360 * n - x0, (1 - np.arcsinh(np.tan(np.radians(lat))) / np.pi) / 2 * n - y0


def outline(mask_path, rgb, color):
    m = np.array(Image.open(mask_path)) > 0
    if not m.any():
        return rgb
    er = np.array(Image.fromarray(m.astype(np.uint8) * 255).filter(ImageFilter.MinFilter(5))) > 0
    edge = m & ~er
    out = rgb.copy()
    out[edge] = np.array([int(color[i:i + 2], 16) for i in (1, 3, 5)], np.uint8)
    return out


def panel(ax, r, kind, line):
    path = DS / "chips" / kind / f"{r['sample_id']}.jpg"
    if path.exists():
        rgb = np.array(Image.open(path).convert("RGB"))
        rgb = outline(DS / "masks/slide_lidar" / f"{r['sample_id']}.png", rgb, BLUE)
        rgb = outline(DS / "masks/slide_aerial" / f"{r['sample_id']}.png", rgb, ORANGE)
        ax.imshow(rgb)
    else:
        ax.imshow(np.full((256, 256, 3), 235, np.uint8))
        ax.text(128, 128, "no image", ha="center", va="center", color=GREY)
    x, y = px(line, int(r["x0"]), int(r["y0"]))
    ax.plot(x, y, color=RED if r["label_aerial"] == "1" else GREEN, lw=2.2, alpha=0.9, solid_capstyle="butt")
    ax.set_xlim(0, 256); ax.set_ylim(256, 0); ax.set_xticks([]); ax.set_yticks([])
    date = r[f"{kind}_date"] or "–"
    ax.set_title({"pre": "before", "post": "after", "post_apr": "later"}[kind] + f" {date}", fontsize=7.5, color=INK, pad=2)


def triplet_sheet(rows, lines, title, out, ncol=3):
    kinds = ("pre", "post", "post_apr")
    nrow = int(np.ceil(len(rows) / ncol))
    fig, axes = plt.subplots(nrow, ncol * 3, figsize=(ncol * 3 * 2.05, nrow * 2.45), facecolor=BG)
    axes = np.atleast_2d(axes)
    for k in range(nrow * ncol):
        trio = [axes[k // ncol, 3 * (k % ncol) + j] for j in range(3)]
        if k >= len(rows):
            for a in trio: a.axis("off")
            continue
        r = rows[k]
        for a, kind in zip(trio, kinds):
            panel(a, r, kind, lines[r["sample_id"]])
        flags = [r["highway"]]
        if r["tunnel"] == "1": flags.append("TUNNEL")
        if r["bridge"] == "1": flags.append("bridge")
        if r["label_aerial"] == "1": flags.append(f"{float(r['slide_overlap_m']):.0f} m inside polygon")
        flags.append(f"LiDAR {'yes' if r['label_lidar'] == '1' else 'no'}")
        flags.append(f"MLIT pt {float(r['mlit_dist_m']):.0f} m")
        trio[0].text(0, 1.21, f"{r['sample_id']} · {r['region']}", transform=trio[0].transAxes, fontsize=8, color=INK, weight="bold")
        trio[0].text(0, 1.105, " · ".join(flags), transform=trio[0].transAxes, fontsize=7, color="#555555")
    fig.suptitle(title, fontsize=11, color=INK, y=0.997)
    fig.text(0.5, 0.003, "Each sample: before (GSI yearly photo 2010–2022) | after (GSI disaster orthophoto, Jan 2024) | later (GSI, Apr 2024); "
             "~121 m × 121 m, 0.47 m/px.  Line: target 50 m road piece (red = candidate blocked, green = clear).  "
             "Orange: GSI aerial landslide/deposit polygon.  Blue: GSI LiDAR deposit polygon.  Imagery: 出典 国土地理院",
             ha="center", fontsize=7.5, color="#555555")
    fig.subplots_adjust(left=0.006, right=0.994, top=1 - 0.98 / (nrow * 2.45), bottom=0.22 / (nrow * 2.45) + 0.012,
                        wspace=0.05, hspace=0.48)
    for j in range(1, ncol):  # thin separators between samples
        x = (axes[0, 3 * j - 1].get_position().x1 + axes[0, 3 * j].get_position().x0) / 2
        fig.add_artist(plt.Line2D([x, x], [0.02, 0.975], color="#d8d8d8", lw=0.8))
    fig.savefig(out, dpi=105, facecolor=BG)
    plt.close(fig)
    print("[done]", out)


def stats_figure(rows, out):
    fig = plt.figure(figsize=(19.5, 9.4), facecolor=BG)
    gs = fig.add_gridspec(2, 4, width_ratios=[1.3, 1, 1, 1], hspace=0.42, wspace=0.36)
    cand = [r for r in rows if r["sample_type"] == "candidate"]
    # (a) map
    ax = fig.add_subplot(gs[:, 0])
    osm = json.loads((ROOT / "external/osm/noto2024_roads.geojson").read_text(encoding="utf-8"))["features"]
    for f in osm:
        c = np.asarray(f["geometry"]["coordinates"])
        if c.ndim == 2:
            ax.plot(c[:, 0], c[:, 1], color="#dddddd", lw=0.3, zorder=1)
    for typ, color, size, z in (("random", GREY, 4, 2), ("near", "#6f6f6f", 4, 3), ("candidate", RED, 6, 4)):
        sub = [r for r in rows if r["sample_type"] == typ]
        ax.scatter([float(r["lon"]) for r in sub], [float(r["lat"]) for r in sub], s=size, color=color, zorder=z,
                   label={"random": "random clear", "near": "near clear (≤300 m)", "candidate": "candidate blocked"}[typ] + f" ({len(sub)})",
                   linewidths=0)
    ax.set_aspect(1 / np.cos(np.radians(37.3)))
    ax.legend(loc="upper left", frameon=False, markerscale=2.5, fontsize=8)
    ax.set_title("(a) samples on the OSM road network (grey)", loc="left", color=INK)
    ax.tick_params(labelsize=7.5)
    # (b) road class
    ax = fig.add_subplot(gs[0, 1])
    classes = [c for c, _ in Counter(r["highway"] for r in rows).most_common()]
    pos = [sum(r["highway"] == c and r["label_aerial"] == "1" for r in rows) for c in classes]
    neg = [sum(r["highway"] == c and r["label_aerial"] == "0" for r in rows) for c in classes]
    yy = np.arange(len(classes))
    ax.barh(yy - 0.2, pos, 0.4, color=RED, label="candidate blocked"); ax.barh(yy + 0.2, neg, 0.4, color=GREEN, label="clear")
    for y_, a, b in zip(yy, pos, neg):
        ax.text(a, y_ - 0.2, f" {a}", va="center", fontsize=7.5); ax.text(b, y_ + 0.2, f" {b}", va="center", fontsize=7.5)
    ax.set_yticks(yy, classes); ax.invert_yaxis(); ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.set_xlim(0, max(neg + pos) * 1.18)
    ax.set_title("(b) OSM road class", loc="left", color=INK)
    # (c) image dates
    ax = fig.add_subplot(gs[0, 2])
    post = Counter(r["post_date"] or "none" for r in rows); pre = Counter(r["pre_date"] or "none" for r in rows)
    pk = sorted(post); qk = sorted(pre, reverse=True)
    ax.barh(np.arange(len(pk)), [post[k] for k in pk], color="#c25a3a")
    ax.barh(np.arange(len(qk)) + len(pk) + 1, [pre[k] for k in qk], color="#5b7fa8")
    ax.set_yticks(list(range(len(pk))) + list(np.arange(len(qk)) + len(pk) + 1), [f"after {k}" for k in pk] + [f"before {k}" for k in qk])
    for i, k in enumerate(pk): ax.text(post[k], i, f" {post[k]}", va="center", fontsize=7.5)
    for i, k in enumerate(qk): ax.text(pre[k], i + len(pk) + 1, f" {pre[k]}", va="center", fontsize=7.5)
    ax.invert_yaxis(); ax.tick_params(axis="y", labelsize=7.5)
    ax.set_xlim(0, max(list(post.values()) + list(pre.values())) * 1.18)
    ax.set_title("(c) photo date per sample (+ Apr 2024 for all)", loc="left", color=INK)
    # (d) regions
    ax = fig.add_subplot(gs[0, 3])
    regs = [c for c, _ in Counter(r["region"] for r in rows).most_common()]
    pos = [sum(r["region"] == c and r["label_aerial"] == "1" for r in rows) for c in regs]
    neg = [sum(r["region"] == c and r["label_aerial"] == "0" for r in rows) for c in regs]
    yy = np.arange(len(regs))
    ax.barh(yy - 0.2, pos, 0.4, color=RED); ax.barh(yy + 0.2, neg, 0.4, color=GREEN)
    for y_, a, b in zip(yy, pos, neg):
        ax.text(a, y_ - 0.2, f" {a}", va="center", fontsize=7.5); ax.text(b, y_ + 0.2, f" {b}", va="center", fontsize=7.5)
    ax.set_yticks(yy, [r or "(none)" for r in regs]); ax.invert_yaxis(); ax.set_xlim(0, max(neg + pos) * 1.2)
    ax.set_title("(d) GSI interpretation area", loc="left", color=INK)
    # (e) label agreement
    ax = fig.add_subplot(gs[1, 1])
    groups = [("candidate", "candidate\nblocked"), ("near", "near clear\n(≤300 m)"), ("random", "random\nclear")]
    tests = [("LiDAR deposit ≤5 m", lambda r: r["label_lidar"] == "1", BLUE),
             ("MLIT damage point ≤100 m", lambda r: float(r["mlit_dist_m"]) <= 100, "#8a5cc2"),
             ("OSM tunnel", lambda r: r["tunnel"] == "1", "#555555")]
    for j, (name, fn, color) in enumerate(tests):
        vals = []
        for typ, _ in groups:
            sub = [r for r in rows if r["sample_type"] == typ]
            vals.append(100 * np.mean([fn(r) for r in sub]) if sub else 0)
        xs = np.arange(len(groups)) + (j - 1) * 0.26
        ax.bar(xs, vals, 0.26, color=color, label=name)
        for x_, v in zip(xs, vals): ax.text(x_, v, f"{v:.0f}", ha="center", va="bottom", fontsize=7.5)
    ax.set_xticks(range(len(groups)), [g for _, g in groups]); ax.set_ylabel("% of samples"); ax.set_ylim(0, 100)
    ax.legend(frameon=False, fontsize=7.5, loc="upper right")
    ax.set_title("(e) do independent sources agree?", loc="left", color=INK)
    # (f) overlap
    ax = fig.add_subplot(gs[1, 2])
    ov = np.array([float(r["slide_overlap_m"]) for r in cand])
    bins = [0, 0.01, 5, 10, 20, 30, 40, 50.5]
    h, _ = np.histogram(ov, bins)
    labels = ["0 (edge\n≤5 m)", "0–5", "5–10", "10–20", "20–30", "30–40", "40–50"]
    ax.bar(range(len(h)), h, color=RED)
    for i, v in enumerate(h): ax.text(i, v, f"{v}", ha="center", va="bottom", fontsize=7.5)
    ax.set_xticks(range(len(h)), labels, fontsize=7.5); ax.set_xlabel("metres of the 50 m piece inside a GSI polygon")
    ax.set_title("(f) how deep is the candidate in the slide?", loc="left", color=INK)
    # (g) folds
    ax = fig.add_subplot(gs[1, 3])
    folds = sorted({r["fold"] for r in rows})
    p_ = [sum(r["fold"] == f and r["label_aerial"] == "1" for r in rows) for f in folds]
    n_ = [sum(r["fold"] == f and r["label_aerial"] == "0" for r in rows) for f in folds]
    ax.bar(np.arange(len(folds)), n_, color=GREEN, label="clear"); ax.bar(np.arange(len(folds)), p_, bottom=n_, color=RED, label="candidate blocked")
    for i, (a, b) in enumerate(zip(p_, n_)): ax.text(i, a + b, f"{a}/{a + b}", ha="center", va="bottom", fontsize=7.5)
    ax.set_xticks(range(len(folds)), [f"fold {f}" for f in folds]); ax.set_ylim(0, max(a + b for a, b in zip(p_, n_)) * 1.25)
    ax.legend(frameon=False, fontsize=7.5, loc="upper left", ncol=2)
    nb = len({r["block_id"] for r in rows})
    ax.set_title(f"(g) 5 spatial folds ({nb} blocks of 2 km)", loc="left", color=INK)
    fig.suptitle(f"Noto 2024 road-blockage dataset v0: {len(rows)} samples (50 m OSM road pieces)", fontsize=12.5, color=INK, x=0.06, ha="left")
    fig.savefig(out, dpi=105, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    print("[done]", out)


def main() -> int:
    FIG.mkdir(exist_ok=True)
    rows, lines = load()
    rng = np.random.default_rng(1)
    pick = lambda typ, n: [rows[i] for i in sorted(rng.choice([i for i, r in enumerate(rows) if r["sample_type"] == typ],
                                                                min(n, sum(r["sample_type"] == typ for r in rows)), replace=False))]
    triplet_sheet(pick("candidate", 15), lines, "Candidate blocked road pieces: 15 random samples (not hand-picked)", FIG / "examples_candidate.png")
    triplet_sheet(pick("near", 6) + pick("random", 6), lines, "Clear road pieces: rows 1–2 near a candidate (≤300 m), rows 3–4 random",
                  FIG / "examples_clear.png")
    stats_figure(rows, FIG / "dataset_stats.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
