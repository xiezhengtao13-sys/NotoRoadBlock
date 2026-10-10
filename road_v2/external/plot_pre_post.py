"""Side-by-side pre/post GSI photos for one site, with the derived road pieces drawn on both.

Usage:
  python road_v2/external/plot_pre_post.py [lat lon pre_layer post_layer]
Defaults: the Wajima west-coast close-up (pre nendophoto2020, post GSI 2024-01-11 disaster photo).
Writes prepost_<lat>_<lon>.png next to this script. Tiles: "出典：国土地理院".
"""
from __future__ import annotations

import glob
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon as MplPolygon
from shapely.geometry import box, shape

from gsi_noto.demo_overlay import lonlat_to_px
from plot_sample_closeup import Z, mosaic

HERE = Path(__file__).resolve().parent
W_M, H_M = 1200, 800


def main() -> int:
    lat, lon = (float(sys.argv[1]), float(sys.argv[2])) if len(sys.argv) > 2 else (37.3759, 136.7929)
    pre_layer = sys.argv[3] if len(sys.argv) > 3 else "nendophoto2020"
    post_layer = sys.argv[4] if len(sys.argv) > 4 else "20240102noto_wazimanishi_0111do"
    w = W_M / (111320 * math.cos(math.radians(lat))); h = H_M / 110540
    win = box(lon - w / 2, lat - h / 2, lon + w / 2, lat + h / 2)
    x0, y1 = lonlat_to_px(lon - w / 2, lat - h / 2, Z)
    x1, y0 = lonlat_to_px(lon + w / 2, lat + h / 2, Z)
    feats = json.loads((HERE / "derived/noto2024_gsi_aerial_pieces.geojson").read_text(encoding="utf-8"))["features"]
    pieces = [(shape(f["geometry"]), f["properties"]["blocked"]) for f in feats]
    pieces = [(g, b) for g, b in pieces if g.intersects(win)]
    slides = []
    for p in glob.glob(str(HERE / "gsi_noto/noto_*/hokai*.geojson")):
        d = json.loads(Path(p).read_text(encoding="utf-8-sig"))
        slides += [shape(f["geometry"]) for f in d["features"] if shape(f["geometry"]).intersects(win)]
    to_px = lambda xy: np.array([lonlat_to_px(a, b, Z) for a, b in xy]) - [x0, y0]

    fig, axes = plt.subplots(1, 2, figsize=(16, 5.8), facecolor="#fcfcfb")
    for ax, layer, label in zip(axes, (pre_layer, post_layer), ("before", "after")):
        img, got, total = mosaic(layer, x0, y0, x1, y1)
        ax.imshow(img)
        for g in slides:
            for poly in ([g] if g.geom_type == "Polygon" else list(g.geoms)):
                ax.add_patch(MplPolygon(to_px(poly.exterior.coords), closed=True, fc="none", ec="#eda100", lw=1.2))
        for g, b in pieces:
            xy = to_px(g.coords)
            ax.plot(xy[:, 0], xy[:, 1], color="#e34948" if b else "#1baf7a", lw=3 if b else 2, alpha=0.85)
        ax.set_xlim(0, img.size[0]); ax.set_ylim(img.size[1], 0)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{label}: GSI {layer}  ({got}/{total} tiles)", fontsize=10)
    fig.suptitle(f"{lat:.4f}N {lon:.4f}E, {W_M} m × {H_M} m. Red = candidate blocked road piece, green = clear, "
                 f"orange = GSI landslide/deposit polygon. Source: GSI", fontsize=10)
    fig.tight_layout()
    out = HERE / f"prepost_{lat:.4f}_{lon:.4f}.png"
    fig.savefig(out, dpi=120, facecolor="#fcfcfb")
    print("[done]", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
