"""Build the Noto 2024 road-blockage dataset v0 from the open data in road_v2/external.

One sample = one 50 m OSM road piece inside the GSI aerial interpretation extent (cloud areas removed),
with a pre-event and a post-event GSI orthophoto chip, pixel masks, and labels from independent sources.

Samples: every candidate-blocked piece (its 5 m buffer touches a GSI aerial landslide/deposit polygon)
plus twice as many clear pieces: half "near" (within 300 m of a candidate piece, same terrain) and half
drawn at random from the whole extent.

Labels and evidence per sample (samples.csv):
  label_aerial    1 if the piece's 5 m buffer touches a GSI aerial-photo landslide/deposit polygon
  slide_dist_m    distance to the nearest such polygon; slide_overlap_m = piece length inside polygons
  label_lidar     1 if within 5 m of a GSI LiDAR-DEM deposit polygon (codes 2/4/6/8/10)
  mlit_dist_m     distance to the nearest MLIT road-damage point (major damage sites, aerial-survey
                  damage points, road-clearing points), with its layer and name
  tunnel, bridge  OSM tags of the way (a tunnel under a landslide is not blocked on the photo)
  post_event_road 1 if the OSM name marks a temporary / detour road built after the quake (仮設, 迂回, 応急)
  block_id, fold  2 km x 2 km spatial block and a 5-fold split grouped by block (no neighbour leakage)
Imagery (GSI tiles, zoom 18, ~0.47 m/px; chips 256 x 256 px = ~121 m, centred on the piece):
  post      earliest January 2024 GSI disaster orthophoto (1/2-1/17) covering >= 97 % of the chip and at
            least 75 % as sharp as the sharpest January option (skips haze and fog)
  post_apr  GSI April 2024 orthophoto (whole peninsula): was the road cleared by then?
  pre       most recent GSI yearly photo before the quake, same sharpness rule (2019-2022 preferred, else to 2010)
Masks (PNG, 0/255): road = target piece drawn 6 m wide; slide_aerial / slide_lidar = polygons.

Usage:
  python -I road_v2/build_noto_dataset.py            # full build
  python -I road_v2/build_noto_dataset.py 30         # quick test with 30 positives
Writes road_v2/dataset_noto2024_v0/. Imagery: "出典：国土地理院" (GSI tiles).
"""
from __future__ import annotations

import csv
import io
import json
import math
import re
import sqlite3
import sys
import time
import urllib.error
import urllib.request
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import shapely
from PIL import Image, ImageDraw
from rasterio.warp import transform as rio_transform
from shapely import STRtree, wkb
from shapely.geometry import Point, shape
from shapely.ops import unary_union
from sklearn.model_selection import StratifiedGroupKFold

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent
EXT = ROOT / "external"
OUT = ROOT / "dataset_noto2024_v0"
CRS_M = "EPSG:6675"
Z, CHIP = 18, 256
ROAD_W_M, NEAR_M, BLOCK_M, MIN_VALID = 6.0, 300.0, 2000.0, 0.97
LIDAR_CODES = {2, 4, 6, 8, 10}
POST_EVENT_ROAD = "仮設|迂回|応急"  # OSM is the 2026 network: temporary / detour roads built after the quake
POST_LAYERS = [  # (layer, capture date), earliest first; GSI disaster orthophotos for the 2024 Noto quake
    ("20240102noto_wazimahigashi_0102do", "2024-01-02"), ("20240102noto_suzu_0102do", "2024-01-02"),
    ("20240102noto_wazimanaka_0102do", "2024-01-02"), ("20240102_noto_suzu_0105do", "2024-01-05"),
    ("20240102_noto_wazimanaka_0105do", "2024-01-05"), ("20240102_noto_anamizu_0105do", "2024-01-05"),
    ("20240102_noto_nanao_0105do", "2024-01-05"), ("20240102noto_wazimanaka_0111do", "2024-01-11"),
    ("20240102noto_wazimanishi_0111do", "2024-01-11"), ("20240102noto_anamizu_0111do", "2024-01-11"),
    ("20240102noto_suzu_0114do", "2024-01-14"), ("20240102noto_wazimahigashi_0114do", "2024-01-14"),
    ("20240102noto_anamizu_0114do", "2024-01-14"), ("20240102noto_wazimanishi_0117do", "2024-01-17"),
    ("20240102noto_anamizu_0117do", "2024-01-17"), ("20240102noto_nanao_0117do", "2024-01-17"),
    ("20240102noto_0405_0426do", "2024-04"),
]
PRE_LAYERS = [(f"nendophoto{y}", str(y)) for y in range(2022, 2009, -1)]
UA = {"User-Agent": "road-blockage-dataset-builder/0.1 (research; python urllib)"}


# ---------- geometry helpers ----------
def project(geoms, src, dst):
    arr = np.array(geoms, dtype=object)
    xy = shapely.get_coordinates(arr)
    if len(xy) == 0:
        return list(arr)
    xs, ys = rio_transform(src, dst, xy[:, 0].tolist(), xy[:, 1].tolist())
    return list(shapely.set_coordinates(arr.copy(), np.column_stack([xs, ys])))


def load_geojson(paths):
    geoms = []
    for p in paths:
        d = json.loads(Path(p).read_text(encoding="utf-8-sig"))
        geoms += [shape(f["geometry"]) for f in d["features"] if f.get("geometry")]
    return [shapely.make_valid(g) for g in geoms]


def load_gpkg(path, codes):
    con = sqlite3.connect(path)
    out = []
    for code, blob in con.execute("select code, geom from landslidepolygon"):
        if code in codes:
            env = (blob[3] >> 1) & 7
            out.append(shapely.make_valid(wkb.loads(bytes(blob[8 + {0: 0, 1: 32, 2: 48, 3: 48, 4: 64}[env]:]))))
    return out


def global_px(lon, lat):
    n = 256 * 2 ** Z
    lon, lat = np.asarray(lon, float), np.asarray(lat, float)
    return (lon + 180) / 360 * n, (1 - np.arcsinh(np.tan(np.radians(lat))) / np.pi) / 2 * n


def nearest(tree, geoms):
    """Distance from each geom to the nearest tree item, and that item's index."""
    (gi, ti), d = tree.query_nearest(geoms, return_distance=True, all_matches=False)
    dist = np.full(len(geoms), np.inf); idx = np.full(len(geoms), -1)
    dist[gi], idx[gi] = d, ti
    return dist, idx


# ---------- tiles ----------
def fetch(layer, z, x, y, stats):
    url = f"https://cyberjapandata.gsi.go.jp/xyz/{layer}/{z}/{x}/{y}.png"
    for attempt in range(4):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=40) as r:
                stats["ok"] += 1
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                stats["404"] += 1
                return None
        except Exception:
            pass
        time.sleep(1.5 * (attempt + 1))
    stats["failed"] += 1
    return None


def decode(b):
    return None if b is None else np.array(Image.open(io.BytesIO(b)).convert("RGBA"))


def chip_tiles(x0, y0):
    return [(tx, ty) for tx in range(x0 // 256, (x0 + CHIP - 1) // 256 + 1) for ty in range(y0 // 256, (y0 + CHIP - 1) // 256 + 1)]


def compose(tiles, x0, y0):
    rgba = np.zeros((CHIP, CHIP, 4), np.uint8)
    for tx, ty in chip_tiles(x0, y0):
        t = tiles.get((tx, ty))
        if t is None:
            continue
        sx0, sx1 = max(x0, tx * 256), min(x0 + CHIP, tx * 256 + 256)
        sy0, sy1 = max(y0, ty * 256), min(y0 + CHIP, ty * 256 + 256)
        rgba[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = t[sy0 - ty * 256:sy1 - ty * 256, sx0 - tx * 256:sx1 - tx * 256]
    return rgba[..., :3], float((rgba[..., 3] > 0).mean())


def sharpness(rgb):
    """Mean absolute grey-level gradient: haze, fog and deep shadow all make it small."""
    g = rgb.astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32)
    return float(np.abs(np.diff(g, axis=0)).mean() + np.abs(np.diff(g, axis=1)).mean())


def choose_post(cands):
    """Earliest January photo that is at least 75 % as sharp as the sharpest January photo (skips hazy ones)."""
    pool = [c for c in cands if c[1].startswith("2024-01")]
    if not pool:
        return None
    best = max(c[3] for c in pool)
    return min((c for c in pool if c[3] >= 0.75 * best), key=lambda c: (c[1], -c[3]))


def choose_pre(cands):
    """Most recent pre-event photo that is at least 75 % as sharp as the sharpest one; 2019-2022 preferred."""
    pool = [c for c in cands if int(c[1]) >= 2019] or cands
    if not pool:
        return None
    best = max(c[3] for c in pool)
    return max((c for c in pool if c[3] >= 0.75 * best), key=lambda c: (c[1], c[3]))


def fill_imagery(samples, kind, layers, choose, pool, stats):
    """Per spatial chunk, render the chip from every layer that covers >= MIN_VALID of it, then keep one."""
    has14 = {}
    order = sorted(range(len(samples)), key=lambda i: (samples[i]["x0"] // 4096, samples[i]["y0"] // 4096))
    for c in range(0, len(order), 200):
        chunk = order[c:c + 200]
        cands = defaultdict(list)  # sample -> [(layer, date, valid, sharpness, rgb)]
        for layer, date in layers:
            k14 = {i: (layer, (samples[i]["x0"] + CHIP // 2) // 4096, (samples[i]["y0"] + CHIP // 2) // 4096) for i in chunk}
            new = list(set(k14.values()) - has14.keys())
            for k, b in zip(new, pool.map(lambda k: fetch(k[0], 14, k[1], k[2], stats), new)):
                has14[k] = b is not None
            cand = [i for i in chunk if has14[k14[i]]]
            need = sorted({t for i in cand for t in chip_tiles(samples[i]["x0"], samples[i]["y0"])})
            tiles = dict(zip(need, pool.map(lambda t: decode(fetch(layer, Z, t[0], t[1], stats)), need)))
            for i in cand:
                rgb, valid = compose(tiles, samples[i]["x0"], samples[i]["y0"])
                if valid >= MIN_VALID:
                    cands[i].append((layer, date, valid, sharpness(rgb), rgb))
        for i in chunk:
            s, got = samples[i], choose(cands[i])
            s[f"{kind}_options"] = ";".join(f"{d}:{sh:.1f}" for _, d, _, sh, _ in sorted(cands[i], key=lambda x: x[1]))
            if got is None:
                s.update({f"{kind}_layer": "", f"{kind}_date": "", f"{kind}_valid": 0.0, f"{kind}_sharpness": 0.0})
                continue
            layer, date, valid, sh, rgb = got
            Image.fromarray(rgb).save(OUT / "chips" / kind / f"{s['sample_id']}.jpg", quality=90)
            s.update({f"{kind}_layer": layer, f"{kind}_date": date, f"{kind}_valid": round(valid, 3), f"{kind}_sharpness": round(sh, 1)})
        print(f"  {kind}: {min(c + 200, len(order))}/{len(order)} chips, tiles {dict(stats)}", flush=True)


# ---------- masks ----------
def ring_px(coords, x0, y0):
    x, y = global_px(*np.asarray(coords)[:, :2].T)
    return list(zip((x - x0).tolist(), (y - y0).tolist()))


def poly_mask(polys_ll, x0, y0):
    m = Image.new("L", (CHIP, CHIP), 0); d = ImageDraw.Draw(m)
    for g in polys_ll:
        for p in getattr(g, "geoms", [g]):
            if p.geom_type != "Polygon":
                continue
            d.polygon(ring_px(p.exterior.coords, x0, y0), fill=255)
            for h in p.interiors:
                d.polygon(ring_px(h.coords, x0, y0), fill=0)
    return m


def main() -> int:
    n_pos_limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    rng = np.random.default_rng(0)
    for sub in ("chips/pre", "chips/post", "chips/post_apr", "masks/road", "masks/slide_aerial", "masks/slide_lidar"):
        (OUT / sub).mkdir(parents=True, exist_ok=True)

    print("[1] loading vectors", flush=True)
    feats = json.loads((EXT / "derived/noto2024_gsi_aerial_pieces.geojson").read_text(encoding="utf-8"))["features"]
    pieces_ll = [shape(f["geometry"]) for f in feats]
    blocked = np.array([f["properties"]["blocked"] for f in feats])
    highway = [f["properties"]["highway"] for f in feats]
    g = EXT / "gsi_noto"
    slides_ll = load_geojson(sorted(g.glob("noto_*/hokai*.geojson")))
    regions = []
    for p in sorted(g.glob("noto_*/判読範囲*.geojson")):
        name = p.parent.name.split("_")[1]
        regions.append((name, unary_union(load_geojson([p]))))
    lidar_m = load_gpkg(EXT / "gsi_noto_dem/20240101noto_landslidepolygon.gpkg", LIDAR_CODES)
    osm = json.loads((EXT / "osm/noto2024_roads.geojson").read_text(encoding="utf-8"))["features"]
    osm_lines = [shape(f["geometry"]) for f in osm]
    z = zipfile.ZipFile(EXT / "mlit_noto2024_roads/data_full_bundle.zip", metadata_encoding="cp932")
    mlit = []
    for layer in ("damaged_area", "damage_situation_aerial_photography", "keikai"):
        for f in json.loads(z.read(f"road/r6noto/map/json/{layer}.geojson").decode("utf-8-sig"))["features"]:
            mlit.append((layer, f["properties"].get("name", ""), Point(f["geometry"]["coordinates"][:2])))
    print(f"    pieces {len(pieces_ll)} (candidate blocked {blocked.sum()}), aerial polygons {len(slides_ll)}, "
          f"LiDAR deposit polygons {len(lidar_m)}, OSM ways {len(osm)}, MLIT points {len(mlit)}", flush=True)

    print("[2] selecting samples", flush=True)
    pieces_m = project(pieces_ll, "EPSG:4326", CRS_M)
    mids_m = [p.interpolate(0.5, normalized=True) for p in pieces_m]
    pos = np.flatnonzero(blocked)
    if n_pos_limit:
        pos = rng.choice(pos, n_pos_limit, replace=False)
    clear = np.flatnonzero(~blocked)
    pos_tree = STRtree([mids_m[i] for i in np.flatnonzero(blocked)])
    near_hit = pos_tree.query([mids_m[i] for i in clear], predicate="dwithin", distance=NEAR_M)[0]
    near_pool = clear[np.unique(near_hit)]
    near = rng.choice(near_pool, min(len(pos), len(near_pool)), replace=False)
    rand_pool = np.setdiff1d(clear, near)
    rand = rng.choice(rand_pool, len(pos), replace=False)
    sel = np.concatenate([pos, near, rand])
    kind = ["candidate"] * len(pos) + ["near"] * len(near) + ["random"] * len(rand)
    print(f"    {len(pos)} candidate blocked + {len(near)} near-clear (pool {len(near_pool)}) + {len(rand)} random-clear", flush=True)

    print("[3] labels and evidence", flush=True)
    sel_m = [pieces_m[i] for i in sel]
    slides_m = project(slides_ll, "EPSG:4326", CRS_M)
    s_tree, l_tree = STRtree(slides_m), STRtree(lidar_m)
    slide_dist, _ = nearest(s_tree, sel_m)
    lidar_dist, _ = nearest(l_tree, sel_m)
    overlap = []
    for p in sel_m:
        hit = s_tree.query(p, predicate="intersects")
        overlap.append(p.intersection(unary_union([slides_m[k] for k in hit])).length if len(hit) else 0.0)
    mlit_m = project([m[2] for m in mlit], "EPSG:4326", CRS_M)
    mlit_dist, mlit_idx = nearest(STRtree(mlit_m), sel_m)
    osm_tree = STRtree(osm_lines)
    mids_ll = project([mids_m[i] for i in sel], CRS_M, "EPSG:4326")
    _, osm_idx = nearest(osm_tree, mids_ll)
    print(f"    aerial label re-check: {int(((slide_dist <= 5.0 + 1e-6) != blocked[sel]).sum())} mismatches of {len(sel)}", flush=True)

    samples = []
    for k, (i, m_ll) in enumerate(zip(sel, mids_ll)):
        lon, lat = m_ll.x, m_ll.y
        px, py = global_px(lon, lat)
        mx, my = mids_m[i].x, mids_m[i].y
        region = next((n for n, geom in regions if geom.contains(m_ll)), "")
        o = osm[osm_idx[k]]["properties"]
        samples.append(dict(
            sample_id=f"noto_{k:05d}", sample_type=kind[k], lon=round(lon, 7), lat=round(lat, 7),
            highway=highway[i], osm_id=o["osm_id"], road_name=o["name"] or "", tunnel=int(o["tunnel"] == "yes"),
            bridge=int(o["bridge"] == "yes"), post_event_road=int(bool(re.search(POST_EVENT_ROAD, o["name"] or ""))),
            region=region,
            label_aerial=int(blocked[i]), slide_dist_m=round(float(slide_dist[k]), 1), slide_overlap_m=round(float(overlap[k]), 1),
            label_lidar=int(lidar_dist[k] <= 5.0), lidar_dist_m=round(float(lidar_dist[k]), 1),
            mlit_dist_m=round(float(mlit_dist[k]), 1), mlit_layer=mlit[mlit_idx[k]][0], mlit_name=mlit[mlit_idx[k]][1],
            block_id=f"{int(mx // BLOCK_M)}_{int(my // BLOCK_M)}",
            x0=int(round(float(px))) - CHIP // 2, y0=int(round(float(py))) - CHIP // 2))
    y = np.array([s["label_aerial"] for s in samples]); groups = [s["block_id"] for s in samples]
    for f, (_, te) in enumerate(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0).split(np.zeros(len(y)), y, groups)):
        for t in te:
            samples[t]["fold"] = f

    print("[4] masks", flush=True)
    for k, s in enumerate(samples):
        x0, y0 = s["x0"], s["y0"]
        line = ring_px(pieces_ll[sel[k]].coords, x0, y0)
        m = Image.new("L", (CHIP, CHIP), 0)
        ImageDraw.Draw(m).line(line, fill=255, width=max(1, round(ROAD_W_M / (40075016.7 * math.cos(math.radians(s["lat"])) / 256 / 2 ** Z))))
        m.save(OUT / "masks/road" / f"{s['sample_id']}.png")
        win_m = mids_m[sel[k]].buffer(CHIP * 0.6)
        poly_mask([slides_ll[j] for j in s_tree.query(win_m)], x0, y0).save(OUT / "masks/slide_aerial" / f"{s['sample_id']}.png")
        near_l = [lidar_m[j] for j in l_tree.query(win_m)]
        poly_mask(project(near_l, CRS_M, "EPSG:4326") if near_l else [], x0, y0).save(OUT / "masks/slide_lidar" / f"{s['sample_id']}.png")

    print("[5] imagery", flush=True)
    stats = Counter()
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=8) as pool:
        fill_imagery(samples, "post", POST_LAYERS[:-1], choose_post, pool, stats)
        fill_imagery(samples, "post_apr", POST_LAYERS[-1:], lambda c: c[0] if c else None, pool, stats)
        fill_imagery(samples, "pre", PRE_LAYERS, choose_pre, pool, stats)
    print(f"    imagery done in {time.time() - t0:.0f} s, tiles {dict(stats)}", flush=True)

    cols = list(samples[0].keys())
    cols = [c for c in cols if c not in ("x0", "y0")] + ["x0", "y0"]
    with open(OUT / "samples.csv", "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=cols); w.writeheader(); w.writerows(samples)
    gj = dict(type="FeatureCollection", features=[
        dict(type="Feature", properties={c: s[c] for c in cols}, geometry=shapely.geometry.mapping(pieces_ll[sel[k]]))
        for k, s in enumerate(samples)])
    (OUT / "samples.geojson").write_text(json.dumps(gj, ensure_ascii=False), encoding="utf-8")
    summary = dict(
        n=len(samples), by_type=Counter(s["sample_type"] for s in samples),
        label_aerial=Counter(s["label_aerial"] for s in samples), label_lidar=Counter(s["label_lidar"] for s in samples),
        post_date=Counter(s["post_date"] for s in samples), post_apr=Counter(s["post_apr_date"] for s in samples), pre_date=Counter(s["pre_date"] for s in samples),
        region=Counter(s["region"] for s in samples), tiles=dict(stats))
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=dict), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, default=dict))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
