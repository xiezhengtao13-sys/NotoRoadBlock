"""Data figures for the Buildings manuscript (paper_buildings/figures/), English labels.

  fig1_study_area.png  GSI hillshade base map, OSM roads, the five GSI interpretation areas, slope-failure polygons,
                       dataset segments, scale bar and an inset of Japan
  fig3_examples.jpg    pre / post / later patches of example segments with the target segment outlined

Usage:
  python -I road_v2/paper_figures.py [blocked_id ...] -- [clear_id ...]
Base map (hillshade) and imagery: GSI tiles ("出典：国土地理院").
"""
from __future__ import annotations

import csv
import io
import json
import math
import sys
import urllib.error
import urllib.request
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon as MplPolygon, Rectangle
from PIL import Image, ImageFilter
from shapely.geometry import shape

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent
DS = ROOT / "dataset_noto2024_v0"
EXT = ROOT / "external"
OUT = ROOT.parent / "paper_buildings" / "figures"
plt.rcParams.update({"font.family": ["Arial", "DejaVu Sans"], "font.size": 9})
REGION_EN = {"suzu-wajimaEAST": "Suzu–Wajima East", "wajimaCENTRAL": "Wajima Central", "wajimaWEST": "Wajima West",
             "anamizu": "Anamizu", "nanao": "Nanao"}


def gpx(lon, lat, z):
    n = 256 * 2 ** z
    lon, lat = np.asarray(lon, float), np.asarray(lat, float)
    return (lon + 180) / 360 * n, (1 - np.arcsinh(np.tan(np.radians(lat))) / np.pi) / 2 * n


def tile(layer, z, x, y, ext="png"):
    try:
        with urllib.request.urlopen(f"https://cyberjapandata.gsi.go.jp/xyz/{layer}/{z}/{x}/{y}.{ext}", timeout=40) as r:
            return Image.open(io.BytesIO(r.read())).convert("RGBA")
    except urllib.error.HTTPError:
        return None


def basemap(layer, z, lon0, lat0, lon1, lat1):
    (x0, x1), (y1, y0) = gpx([lon0, lon1], [lat0, lat1], z)
    tx0, ty0, tx1, ty1 = int(x0 // 256), int(y0 // 256), int(x1 // 256), int(y1 // 256)
    img = Image.new("RGBA", ((tx1 - tx0 + 1) * 256, (ty1 - ty0 + 1) * 256), (0, 0, 0, 0))
    for tx in range(tx0, tx1 + 1):
        for ty in range(ty0, ty1 + 1):
            t = tile(layer, z, tx, ty)
            if t is not None:
                img.paste(t, ((tx - tx0) * 256, (ty - ty0) * 256))
    return img.crop((int(x0 - tx0 * 256), int(y0 - ty0 * 256), int(x1 - tx0 * 256), int(y1 - ty0 * 256))), x0, y0


def shaded(rgba, strength):
    """GSI hillshade tiles are transparent over the sea: lighten the land, paint the sea light blue."""
    a = np.asarray(rgba).astype(float)
    rgb = 255 - (255 - a[..., :3]) * strength
    rgb[a[..., 3] == 0] = [214, 230, 245]
    return Image.fromarray(rgb.clip(0, 255).astype(np.uint8))


def load_geojson(paths):
    out = []
    for p in paths:
        d = json.loads(Path(p).read_text(encoding="utf-8-sig"))
        out += [shape(f["geometry"]) for f in d["features"] if f.get("geometry")]
    return out


def fig_study_area(rows):
    Z = 11
    lon0, lat0, lon1, lat1 = 136.66, 36.98, 137.38, 37.56
    img, x0, y0 = basemap("hillshademap", Z, lon0, lat0, lon1, lat1)
    img = shaded(img, 0.45)
    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    ax.imshow(img, zorder=0)
    to = lambda lon, lat: (gpx(lon, lat, Z)[0] - x0, gpx(lon, lat, Z)[1] - y0)
    for f in json.loads((EXT / "osm/noto2024_roads.geojson").read_text(encoding="utf-8"))["features"]:
        c = np.asarray(f["geometry"]["coordinates"])
        if c.ndim == 2:
            x, y = to(c[:, 0], c[:, 1]); ax.plot(x, y, color="#8c8c8c", lw=0.25, zorder=1)
    g = EXT / "gsi_noto"
    for poly in load_geojson(sorted(g.glob("noto_*/hokai*.geojson"))):
        for p in getattr(poly, "geoms", [poly]):
            x, y = to(*np.asarray(p.exterior.coords)[:, :2].T)
            ax.add_patch(MplPolygon(np.column_stack([x, y]), closed=True, fc="#f08c00", ec="#f08c00", lw=0.3, zorder=2))
    for p in sorted(g.glob("noto_*/判読範囲*.geojson")):
        name = REGION_EN[p.parent.name.split("_")[1]]
        for poly in load_geojson([p]):
            for q in getattr(poly, "geoms", [poly]):
                x, y = to(*np.asarray(q.exterior.coords)[:, :2].T)
                ax.plot(x, y, color="black", lw=0.9, zorder=3)
        c = poly.centroid
    labels = {"Suzu–Wajima East": (137.20, 37.47), "Wajima Central": (136.93, 37.43), "Wajima West": (136.74, 37.31),
              "Anamizu": (136.80, 37.17), "Nanao": (137.08, 37.10)}
    for name, (lo, la) in labels.items():
        x, y = to(lo, la)
        ax.text(x, y, name, fontsize=8.5, weight="bold", ha="center", va="center", zorder=6,
                bbox=dict(fc="white", ec="none", alpha=0.75, pad=1.2))
    for lab, color, s, zo, name in ((0, "#2f6fb0", 2.2, 4, "Clear"), (1, "#d62728", 3.2, 5, "Blocked")):
        sub = [r for r in rows if r["y"] == lab]
        x, y = to(np.array([float(r["lon"]) for r in sub]), np.array([float(r["lat"]) for r in sub]))
        ax.scatter(x, y, s=s, color=color, lw=0, zorder=zo, label=f"{name} segments ({len(sub)})")
    ax.add_patch(Rectangle((0, 0), 0, 0, fc="#f08c00", label="GSI slope-failure polygons"))
    ax.plot([], [], color="black", lw=0.9, label="GSI interpretation areas")
    ax.plot([], [], color="#8c8c8c", lw=0.6, label="OpenStreetMap roads")
    ax.legend(loc="lower right", fontsize=7.5, framealpha=0.9, markerscale=3)
    # scale bar (10 km) and north arrow
    mpp = 40075016.7 * math.cos(math.radians(37.25)) / (256 * 2 ** Z)
    L = 10000 / mpp; W = img.size[0]; bx, by = W - 40 - L, 175
    ax.plot([bx, bx + L], [by, by], color="black", lw=2.5, zorder=7)
    ax.text(bx + L / 2, by - 8, "10 km", ha="center", va="bottom", fontsize=8, zorder=7)
    ax.annotate("N", xy=(W - 60, 40), xytext=(W - 60, 105), ha="center", fontsize=10, weight="bold",
                arrowprops=dict(arrowstyle="-|>", color="black", lw=1.5), zorder=7)
    ax.set_xlim(0, img.size[0]); ax.set_ylim(img.size[1], 0); ax.set_xticks([]); ax.set_yticks([])
    # lon/lat ticks
    for lo in (136.8, 137.0, 137.2):
        x, _ = to(lo, lat0); ax.text(x, img.size[1] + 4, f"{lo:.1f}°E", ha="center", va="top", fontsize=7.5)
    for la in (37.1, 37.3, 37.5):
        _, y = to(lon0, la); ax.text(-4, y, f"{la:.1f}°N", ha="right", va="center", fontsize=7.5)
    # inset: Japan
    ins = fig.add_axes([0.105, 0.70, 0.22, 0.22])
    jimg, jx0, jy0 = basemap("hillshademap", 5, 128.0, 30.0, 146.5, 45.8)
    ins.imshow(shaded(jimg, 0.6))
    (rx0, rx1), (ry1, ry0) = gpx([lon0, lon1], [lat0, lat1], 5)
    ins.add_patch(Rectangle((rx0 - jx0, ry0 - jy0), rx1 - rx0, ry1 - ry0, fill=False, ec="#d62728", lw=1.5))
    ins.set_xticks([]); ins.set_yticks([])
    for s in ins.spines.values():
        s.set_edgecolor("#555555")
    fig.savefig(OUT / "fig1_study_area.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("[done]", OUT / "fig1_study_area.png")


def outline(sid):
    m = Image.open(DS / "masks/road" / f"{sid}.png")
    corridor = np.array(m.filter(ImageFilter.MaxFilter(15))) > 0
    inner = np.array(Image.fromarray(corridor.astype(np.uint8) * 255).filter(ImageFilter.MinFilter(5))) > 0
    return corridor & ~inner


def fig_examples(rows, blocked, clear):
    by = {r["sample_id"]: r for r in rows}
    ids = [(s, "Blocked") for s in blocked] + [(s, "Clear") for s in clear]
    fig, axes = plt.subplots(len(ids), 3, figsize=(6.6, 2.25 * len(ids)))
    for i, (sid, lab) in enumerate(ids):
        r = by[sid]; e = outline(sid)
        for j, (kind, title) in enumerate((("pre", "Pre-event"), ("post", "Post-event"), ("post_apr", "Later"))):
            a = axes[i, j]
            rgb = np.array(Image.open(DS / "chips" / kind / f"{sid}.jpg").convert("RGB"))
            rgb[e] = (255, 43, 214)
            a.imshow(rgb); a.set_xticks([]); a.set_yticks([])
            date = r[{"pre": "pre_date", "post": "post_date", "post_apr": "post_apr_date"}[kind]]
            a.set_title(f"{title} ({date})", fontsize=8, pad=2)
        road = r["highway"].replace("_", " ")
        axes[i, 0].set_ylabel(f"({chr(97 + i)}) {lab}\n{road}", fontsize=8.5)
    fig.tight_layout(h_pad=0.6, w_pad=0.3)
    fig.savefig(OUT / "fig3_examples.jpg", dpi=300, facecolor="white", pil_kwargs={"quality": 92})
    plt.close(fig)
    print("[done]", OUT / "fig3_examples.jpg")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    # final dataset: verified labels (merge_review.py); excluded segments are not shown
    ver = {r["sample_id"]: int(r["label"]) for r in csv.DictReader(open(DS / "labels_verified.csv", encoding="utf-8"))}
    rows = [dict(r, y=ver[r["sample_id"]]) for r in csv.DictReader(open(DS / "samples.csv", encoding="utf-8-sig"))
            if ver.get(r["sample_id"], -9) >= 0]
    args = sys.argv[1:]
    if "--" in args:
        k = args.index("--"); blocked, clear = args[:k], args[k + 1:]
    else:  # all verified by the interpreter; noto_00796 lies on the same motorway as noto_00794, next to a slope failure
        blocked, clear = ["noto_00794", "noto_00028", "noto_00999"], ["noto_00796", "noto_02759"]
    fig_study_area(rows)
    fig_examples(rows, blocked, clear)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
