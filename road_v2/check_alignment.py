"""How well do the OSM road pieces line up with the roads in the GSI orthophotos?

Reference: GSI road centrelines (experimental_rdcl GeoJSON tiles, zoom 16), which come from the same agency as the
orthophotos. For each sample, the offset is the mean distance from the OSM piece to the nearest GSI centreline.

Usage:
  python -I road_v2/check_alignment.py sheet [n]   # contact sheet of the first n samples in review order
  python -I road_v2/check_alignment.py stats       # offsets for all samples of the paper subset
Tiles are cached in road_v2/cache_noto/rdcl/. Writes results_noto/alignment_*.png / alignment_offsets.csv.
"""
from __future__ import annotations

import csv
import json
import math
import sys
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from shapely import STRtree
from shapely.geometry import LineString, shape

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent
DS = ROOT / "dataset_noto2024_v0"
CACHE = ROOT / "cache_noto" / "rdcl"
RES = ROOT / "results_noto"
Z = 16


def tile_xy(lon, lat, z=Z):
    n = 2 ** z
    return int((lon + 180) / 360 * n), int((1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * n)


def rdcl_tile(x, y):
    p = CACHE / f"{x}_{y}.geojson"
    if not p.exists():
        try:
            with urllib.request.urlopen(f"https://cyberjapandata.gsi.go.jp/xyz/experimental_rdcl/{Z}/{x}/{y}.geojson", timeout=40) as r:
                p.write_bytes(r.read())
        except urllib.error.HTTPError:
            p.write_text('{"type":"FeatureCollection","features":[]}', encoding="utf-8")
    d = json.loads(p.read_text(encoding="utf-8"))
    return [shape(f["geometry"]) for f in d.get("features", []) if f.get("geometry")]


def centrelines_for(points_lonlat):
    """GSI centrelines in the zoom-16 tiles around the given points (3 x 3 tiles each)."""
    keys = set()
    for lon, lat in points_lonlat:
        x, y = tile_xy(lon, lat)
        keys |= {(x + dx, y + dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)}
    CACHE.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(8) as pool:
        parts = list(pool.map(lambda k: rdcl_tile(*k), sorted(keys)))
    lines = []
    for part in parts:
        for g in part:
            lines += list(getattr(g, "geoms", [g]))
    return lines


def to_metric(coords, lat0):
    c = np.asarray(coords)[:, :2]
    return np.column_stack([c[:, 0] * 111320 * math.cos(math.radians(lat0)), c[:, 1] * 110540])


def offsets(rows, geo, lines):
    lat0 = 37.3
    lm = [LineString(to_metric(l.coords, lat0)) for l in lines if len(l.coords) > 1]
    tree = STRtree(lm)
    out = {}
    for r in rows:
        seg = LineString(to_metric(geo[r["sample_id"]], lat0))
        pts = [seg.interpolate(t, normalized=True) for t in np.linspace(0, 1, 6)]
        idx = tree.nearest(seg)
        d_near = min(lm[idx].distance(p) for p in pts)
        out[r["sample_id"]] = (float(np.mean([lm[idx].distance(p) for p in pts])), float(lm[idx].distance(seg)), d_near)
    return out


SEARCH_PX, STRIP_PX, MIN_PROMINENCE = 34, 3, 0.06  # +-16 m search, ~3 m wide sampling strip


def road_likeness(rgb):
    """Grey road surface: low saturation, not black shadow. Values 0..1."""
    from PIL import Image
    hsv = np.asarray(Image.fromarray(rgb).convert("HSV")).astype(np.float32) / 255
    return (1 - hsv[..., 1]) * (hsv[..., 2] > 0.15)


def snap_offset(rgb, line_px):
    """Shift (in px, along the normal) that puts the piece on the most road-like parallel band; 0 if unclear."""
    R = road_likeness(rgb)
    p = np.asarray(line_px, float)
    c = p - p.mean(0)
    t = np.linalg.svd(c, full_matrices=False)[2][0]; nrm = np.array([-t[1], t[0]])
    dense = np.concatenate([np.linspace(p[i], p[i + 1], 12, endpoint=False) for i in range(len(p) - 1)] + [p[-1:]])
    prof = []
    for d in range(-SEARCH_PX, SEARCH_PX + 1):
        vals = []
        for w in range(-STRIP_PX, STRIP_PX + 1):
            q = dense + (d + w) * nrm
            ok = (q[:, 0] >= 0) & (q[:, 0] < 256) & (q[:, 1] >= 0) & (q[:, 1] < 256)
            if ok.sum() < len(q) * 0.7:
                vals = None; break
            vals.append(R[q[ok, 1].astype(int), q[ok, 0].astype(int)].mean())
        prof.append(np.mean(vals) if vals is not None else np.nan)
    prof = np.array(prof)
    if np.isnan(prof).all():
        return 0.0, nrm, prof
    k = int(np.nanargmax(prof)); d = k - SEARCH_PX
    prominence = prof[k] - np.nanmedian(prof)
    return (float(d) if prominence >= MIN_PROMINENCE else 0.0), nrm, prof


def main() -> int:
    mode = sys.argv[1] if len(sys.argv) > 1 else "sheet"
    if mode == "snaptest":
        return snaptest(int(sys.argv[2]) if len(sys.argv) > 2 else 32)
    rows = {r["sample_id"]: r for r in csv.DictReader(open(DS / "samples.csv", encoding="utf-8-sig"))}
    geo = {f["properties"]["sample_id"]: f["geometry"]["coordinates"]
           for f in json.loads((DS / "samples.geojson").read_text(encoding="utf-8"))["features"]}
    RES.mkdir(exist_ok=True)
    if mode == "sheet":
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from PIL import Image
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 16
        order = [it["id"] for it in json.loads((DS / "review/review_list.json").read_text(encoding="utf-8"))][:n]
        lines = centrelines_for([(float(rows[s]["lon"]), float(rows[s]["lat"])) for s in order])
        off = offsets([rows[s] for s in order], geo, lines)
        cols = 4; nrow = math.ceil(n / cols)
        fig, axes = plt.subplots(nrow, cols, figsize=(cols * 3.2, nrow * 3.4))
        npx = 256 * 2 ** 18
        gx = lambda lon: (np.asarray(lon) + 180) / 360 * npx
        gy = lambda lat: (1 - np.arcsinh(np.tan(np.radians(np.asarray(lat)))) / np.pi) / 2 * npx
        for k, sid in enumerate(order):
            a = axes.flat[k]; r = rows[sid]; x0, y0 = int(r["x0"]), int(r["y0"])
            a.imshow(Image.open(DS / "chips/pre" / f"{sid}.jpg"))
            for l in lines:
                c = np.asarray(l.coords)
                a.plot(gx(c[:, 0]) - x0, gy(c[:, 1]) - y0, color="cyan", lw=1.4)
            c = np.asarray(geo[sid])
            a.plot(gx(c[:, 0]) - x0, gy(c[:, 1]) - y0, color="#ff2bd6", lw=2.2)
            a.set_xlim(0, 256); a.set_ylim(256, 0); a.set_xticks([]); a.set_yticks([])
            a.set_title(f"#{k + 1} {sid}  offset {off[sid][0]:.1f} m", fontsize=8)
        for a in list(axes.flat)[n:]:
            a.axis("off")
        fig.suptitle("Pre-event photo; magenta = OSM piece (our box), cyan = GSI road centrelines", fontsize=10)
        fig.tight_layout()
        out = RES / f"alignment_first{n}.png"
        fig.savefig(out, dpi=110); print("[done]", out)
        print({sid: round(off[sid][0], 1) for sid in order})
    else:
        sub = [r for r in rows.values() if r["post_date"] and r["tunnel"] == "0" and r["post_event_road"] == "0"]
        lines = centrelines_for([(float(r["lon"]), float(r["lat"])) for r in sub])
        off = offsets(sub, geo, lines)
        with open(RES / "alignment_offsets.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh); w.writerow(["sample_id", "sample_type", "highway", "mean_offset_m", "min_dist_m", "nearest_point_m"])
            for r in sub:
                w.writerow([r["sample_id"], r["sample_type"], r["highway"]] + [round(v, 2) for v in off[r["sample_id"]]])
        v = np.array([off[r["sample_id"]][0] for r in sub])
        print(f"n={len(v)} mean offset: median {np.median(v):.1f} m, 75% {np.percentile(v, 75):.1f} m, 90% {np.percentile(v, 90):.1f} m, "
              f"share > 5 m {np.mean(v > 5):.1%}, > 10 m {np.mean(v > 10):.1%}, > 20 m {np.mean(v > 20):.1%}")
        for h in sorted({r["highway"] for r in sub}):
            vv = np.array([off[r["sample_id"]][0] for r in sub if r["highway"] == h])
            print(f"  {h:15s} n={len(vv):4d} median {np.median(vv):5.1f} m  >10 m {np.mean(vv > 10):.0%}")
    return 0


def snaptest(n):
    """Visual check of the image-based snapping on the first n samples in review order (pre and post photos)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    rows = {r["sample_id"]: r for r in csv.DictReader(open(DS / "samples.csv", encoding="utf-8-sig"))}
    geo = {f["properties"]["sample_id"]: f["geometry"]["coordinates"]
           for f in json.loads((DS / "samples.geojson").read_text(encoding="utf-8"))["features"]}
    order = [it["id"] for it in json.loads((DS / "review/review_list.json").read_text(encoding="utf-8"))][:n]
    npx = 256 * 2 ** 18
    gx = lambda lon: (np.asarray(lon) + 180) / 360 * npx
    gy = lambda lat: (1 - np.arcsinh(np.tan(np.radians(np.asarray(lat)))) / np.pi) / 2 * npx
    cols = 8; nrow = math.ceil(n / 4)
    fig, axes = plt.subplots(nrow, cols, figsize=(cols * 2.1, nrow * 2.3))
    for k, sid in enumerate(order):
        r = rows[sid]; x0, y0 = int(r["x0"]), int(r["y0"])
        c = np.asarray(geo[sid]); line = np.column_stack([gx(c[:, 0]) - x0, gy(c[:, 1]) - y0])
        pre = np.asarray(Image.open(DS / "chips/pre" / f"{sid}.jpg").convert("RGB"))
        d, nrm, _ = snap_offset(pre, line)
        new = line + d * nrm
        for j, kind in enumerate(("pre", "post")):
            a = axes.flat[2 * k + j]
            a.imshow(Image.open(DS / "chips" / kind / f"{sid}.jpg"))
            a.plot(line[:, 0], line[:, 1], color="#ff2bd6", lw=1.8)
            a.plot(new[:, 0], new[:, 1], color="yellow", lw=1.4, ls="--")
            a.set_xlim(0, 256); a.set_ylim(256, 0); a.set_xticks([]); a.set_yticks([])
            a.set_title(f"#{k + 1} {kind}  shift {d * 0.47:+.1f} m" if j == 0 else f"#{k + 1} post", fontsize=7)
    fig.suptitle("magenta = OSM piece, yellow dashed = snapped to the most road-like band in the pre-event photo", fontsize=9)
    fig.tight_layout()
    out = RES / f"snaptest_first{n}.png"
    fig.savefig(out, dpi=100); print("[done]", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
