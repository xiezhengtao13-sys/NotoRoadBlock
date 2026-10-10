"""Are the GSI post-event (and April) orthophotos shifted against the pre-event photos?

For each sample, the translation between the pre-event patch and the post-event patch (and the April patch) is
estimated by phase correlation on gradient-magnitude images (robust to snow and colour changes). Shifts are summarised
per post-event tile layer, because each layer was orthorectified separately.

Usage:
  python -I road_v2/check_registration.py [n_per_layer]
Writes results_noto/registration_shifts.csv and prints a per-layer summary (shift = how far the post image content
must be moved to match the pre image; +x east, +y south; 1 px = 0.47 m).
"""
from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.registration import phase_cross_correlation

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent
DS = ROOT / "dataset_noto2024_v0"
RES = ROOT / "results_noto"


def grad(path):
    g = np.asarray(Image.open(path).convert("L")).astype(np.float32)
    gy, gx = np.gradient(g)
    m = np.hypot(gx, gy)
    win = np.outer(np.hanning(256), np.hanning(256))
    return (m - m.mean()) * win


def shift(a, b):
    """(dy, dx) that moves image b onto image a, plus the registration error (lower = more reliable)."""
    s, err, _ = phase_cross_correlation(a, b, upsample_factor=4, normalization=None)
    return float(s[0]), float(s[1]), float(err)


def visual_check(n=24):
    """Pre with the OSM box | post with the OSM box | post with the box moved by the estimated shift."""
    import json
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = {r["sample_id"]: r for r in csv.DictReader(open(DS / "samples.csv", encoding="utf-8-sig"))}
    geo = {f["properties"]["sample_id"]: f["geometry"]["coordinates"]
           for f in json.loads((DS / "samples.geojson").read_text(encoding="utf-8"))["features"]}
    sh = list(csv.DictReader(open(RES / "registration_shifts.csv", encoding="utf-8")))
    sh = [s for s in sh if 8 < np.hypot(float(s["dx_post"]), float(s["dy_post"])) < 60]
    rng = np.random.default_rng(1)
    pick = [sh[i] for i in rng.choice(len(sh), min(n, len(sh)), replace=False)]
    npx = 256 * 2 ** 18
    fig, axes = plt.subplots(len(pick) // 2, 6, figsize=(13, len(pick) // 2 * 2.3))
    for k, s in enumerate(pick):
        sid = s["sample_id"]; r = rows[sid]; x0, y0 = int(r["x0"]), int(r["y0"])
        c = np.asarray(geo[sid])
        lx = (c[:, 0] + 180) / 360 * npx - x0
        ly = (1 - np.arcsinh(np.tan(np.radians(c[:, 1]))) / np.pi) / 2 * npx - y0
        dx, dy = float(s["dx_post"]), float(s["dy_post"])
        for j, (kind, ox, oy, title) in enumerate((("pre", 0, 0, "pre, OSM box"), ("post", 0, 0, "post, OSM box"),
                                                   ("post", -dx, -dy, f"post, moved ({-dx:+.0f},{-dy:+.0f}) px"))):
            a = axes.flat[6 * (k // 2) + 3 * (k % 2) + j]
            a.imshow(Image.open(DS / "chips" / kind / f"{sid}.jpg"))
            a.plot(lx + ox, ly + oy, color="#ff2bd6" if j < 2 else "yellow", lw=2)
            a.set_xlim(0, 256); a.set_ylim(256, 0); a.set_xticks([]); a.set_yticks([])
            a.set_title(f"{sid} {title}" if j == 0 else title, fontsize=6.5)
    fig.tight_layout()
    out = RES / "registration_check.png"
    fig.savefig(out, dpi=100); print("[done]", out)
    return 0


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "visual":
        return visual_check()
    n_per = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    rows = [r for r in csv.DictReader(open(DS / "samples.csv", encoding="utf-8-sig"))
            if r["post_date"] and r["tunnel"] == "0" and r["post_event_road"] == "0"]
    rng = np.random.default_rng(0)
    by_layer = defaultdict(list)
    for r in rows:
        by_layer[r["post_layer"]].append(r)
    out = []
    for layer, rs in sorted(by_layer.items()):
        pick = [rs[i] for i in rng.choice(len(rs), min(n_per, len(rs)), replace=False)]
        for r in pick:
            sid = r["sample_id"]
            pre = grad(DS / "chips/pre" / f"{sid}.jpg")
            dy, dx, err = shift(pre, grad(DS / "chips/post" / f"{sid}.jpg"))
            ay, ax_, aerr = shift(pre, grad(DS / "chips/post_apr" / f"{sid}.jpg"))
            out.append(dict(sample_id=sid, post_layer=layer, pre_date=r["pre_date"], dx_post=dx, dy_post=dy, err_post=err,
                            dx_apr=ax_, dy_apr=ay, err_apr=aerr))
    RES.mkdir(exist_ok=True)
    with open(RES / "registration_shifts.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
    print(f"{'post layer':36s} {'n':>3s}  median shift post->pre (px)   |shift| median  >10 px   April->pre median |shift|")
    for layer in sorted({o["post_layer"] for o in out}):
        o = [x for x in out if x["post_layer"] == layer]
        dx = np.array([x["dx_post"] for x in o]); dy = np.array([x["dy_post"] for x in o])
        mag = np.hypot(dx, dy); amag = np.hypot([x["dx_apr"] for x in o], [x["dy_apr"] for x in o])
        print(f"{layer:36s} {len(o):3d}  dx {np.median(dx):+6.1f}  dy {np.median(dy):+6.1f}        {np.median(mag):5.1f}       "
              f"{np.mean(mag > 10):4.0%}   {np.median(amag):5.1f}")
    allmag = np.hypot([x["dx_post"] for x in out], [x["dy_post"] for x in out])
    print(f"all: median |shift| {np.median(allmag):.1f} px ({np.median(allmag) * 0.47:.1f} m), 90th pct {np.percentile(allmag, 90):.1f} px")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
