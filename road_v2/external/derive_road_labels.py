"""Derive candidate road-blockage labels: OSM roads cut into 50 m pieces x official landslide polygons.

A piece is a candidate "blocked by landslide" when its 5 m buffer touches a GSI landslide/deposit
polygon. Only pieces inside the GSI interpretation extent (minus cloud-masked areas) are kept, so the
negatives were actually inspected. These are candidates, not ground truth: a polygon touching a road
does not prove the carriageway was impassable.

Usage:
  python road_v2/external/derive_road_labels.py
Writes derived/<event>_<source>_pieces.geojson and derived/summary.json
"""
from __future__ import annotations

import json
import sqlite3
from collections import Counter
from pathlib import Path

import numpy as np
from rasterio.warp import transform as rio_transform
from shapely import STRtree, wkb
from shapely.geometry import LineString, mapping, shape
from shapely.ops import transform as shp_transform, unary_union

HERE = Path(__file__).resolve().parent
OUT = HERE / "derived"
PIECE_M, BUFFER_M = 50.0, 5.0


def project(geoms, src, dst):
    """Reproject many geometries with one PROJ call (shapely 2 vectorized coordinate access)."""
    import shapely
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
    return geoms


def load_gpkg(path, codes=None):
    con = sqlite3.connect(path)
    out = []
    for code, blob in con.execute("select code, geom from landslidepolygon"):
        if codes and code not in codes:
            continue
        env = (blob[3] >> 1) & 7
        out.append(wkb.loads(bytes(blob[8 + {0: 0, 1: 32, 2: 48, 3: 48, 4: 64}[env]:])))
    return out


def road_pieces(roads_geojson, crs):
    d = json.loads(Path(roads_geojson).read_text(encoding="utf-8"))
    lines = [shape(f["geometry"]) for f in d["features"]]
    hw = [f["properties"]["highway"] for f in d["features"]]
    lines_m = project(lines, "EPSG:4326", crs)
    pieces, cls = [], []
    for g, h in zip(lines_m, hw):
        n = max(1, int(round(g.length / PIECE_M)))
        for k in range(n):
            seg = LineString([g.interpolate(t) for t in np.linspace(k / n, (k + 1) / n, 6) * g.length])
            pieces.append(seg); cls.append(h)
    return pieces, cls


def label(pieces, cls, slides_m, extent_m, crs, event, source):
    tree = STRtree(slides_m)
    keep, blocked = [], []
    ext_tree_geom = extent_m
    for i, p in enumerate(pieces):
        if ext_tree_geom is not None and not ext_tree_geom.contains(p.centroid):
            continue
        hit = tree.query(p.buffer(BUFFER_M), predicate="intersects")
        keep.append(i); blocked.append(len(hit) > 0)
    keep = np.array(keep); blocked = np.array(blocked)
    by_class = Counter(cls[i] for i, b in zip(keep, blocked) if b)
    summary = dict(event=event, source=source, pieces_in_extent=int(len(keep)), candidate_blocked=int(blocked.sum()),
                   blocked_share=round(float(blocked.mean()), 4) if len(keep) else None,
                   blocked_by_class=dict(by_class.most_common()))
    # write the labeled pieces back in lon/lat
    geoms_ll = project([pieces[i] for i in keep], crs, "EPSG:4326")
    feats = [dict(type="Feature", properties=dict(highway=cls[i], blocked=bool(b), event=event, source=source),
                  geometry=mapping(g)) for i, b, g in zip(keep, blocked, geoms_ll)]
    OUT.mkdir(exist_ok=True)
    (OUT / f"{event}_{source}_pieces.geojson").write_text(json.dumps(dict(type="FeatureCollection", features=feats)), encoding="utf-8")
    return summary


def extent(paths_extent, paths_cloud, crs):
    ext = unary_union(project(load_geojson(paths_extent), "EPSG:4326", crs))
    if paths_cloud:
        ext = ext.difference(unary_union(project(load_geojson(paths_cloud), "EPSG:4326", crs)))
    return ext


def main() -> int:
    import sys
    only = set(sys.argv[1:])
    prev = OUT / "summary.json"
    summaries = [x for x in json.loads(prev.read_text(encoding="utf-8")) if x["event"] not in only] if (only and prev.exists()) else []
    if only and "noto2024" not in only:
        return run_rest(summaries, only)
    g = HERE / "gsi_noto"
    noto_slides = sorted(g.glob("noto_*/hokai*.geojson"))
    noto_ext = sorted(g.glob("noto_*/判読範囲*.geojson"))
    noto_cloud = sorted(g.glob("noto_*/cloud_poly*.geojson"))
    crs = "EPSG:6675"
    pieces, cls = road_pieces(HERE / "osm/noto2024_roads.geojson", crs)
    ext = extent(noto_ext, noto_cloud, crs)
    summaries.append(label(pieces, cls, project(load_geojson(noto_slides), "EPSG:4326", crs), ext, crs, "noto2024", "gsi_aerial"))
    print(summaries[-1], flush=True)
    # LiDAR-DEM inventory: depositional / flow parts are what can bury a road
    dem = load_gpkg(HERE / "gsi_noto_dem/20240101noto_landslidepolygon.gpkg", codes={2, 4, 6, 8, 10})
    summaries.append(label(pieces, cls, dem, None, crs, "noto2024", "gsi_lidar_deposits"))
    print(summaries[-1], flush=True)
    return run_rest(summaries, only)


def run_rest(summaries, only):
    if not only or "kumamoto2026" in only:
        summaries.append(kumamoto2026())
        print(summaries[-1], flush=True)
    if not only or "hokkaido2018" in only:
        summaries.append(hokkaido2018())
        print(summaries[-1], flush=True)
    OUT.mkdir(exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


def kumamoto2026():
    k = HERE / "gsi_kumamoto2026/kumamoto_yatsushiro0814"
    crs = "EPSG:6670"
    pieces, cls = road_pieces(HERE / "osm/kumamoto2026_roads.geojson", crs)
    ext = extent(sorted(k.glob("*判読範囲*.geojson")), sorted(k.glob("*雲*.geojson")), crs)
    return label(pieces, cls, project(load_geojson(sorted(k.glob("*土砂*.geojson"))), "EPSG:4326", crs),
                 ext, crs, "kumamoto2026", "gsi_aerial")


def hokkaido2018():
    h = next((HERE / "gsi_hokkaido2018").rglob("*斜面崩壊・堆積範囲.geojson"))
    crs = "EPSG:6680"
    pieces, cls = road_pieces(HERE / "osm/hokkaido2018_roads.geojson", crs)
    cloud = list((HERE / "gsi_hokkaido2018").rglob("*雲による未判読範囲.geojson"))
    ext_files = [p for p in (HERE / "gsi_hokkaido2018").rglob("*判読範囲.geojson") if "未判読" not in p.name]
    from shapely.geometry import Polygon
    ext_poly = [Polygon(l.coords) for l in load_geojson(ext_files) if l.geom_type == "LineString"]
    ext = unary_union(project(ext_poly, "EPSG:4326", crs))
    if cloud:
        ext = ext.difference(unary_union(project(load_geojson(cloud), "EPSG:4326", crs)))
    return label(pieces, cls, project(load_geojson([h]), "EPSG:4326", crs), ext, crs, "hokkaido2018", "gsi_aerial")


if __name__ == "__main__":
    raise SystemExit(main())
