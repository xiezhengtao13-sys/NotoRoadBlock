"""Unzip every downloaded archive (fixing Shift-JIS file names) and summarize each vector dataset.

Usage:
  python road_v2/external/inspect_all.py
Writes inventory.json next to this script and prints a summary.
"""
from __future__ import annotations

import json
import math
import sqlite3
import struct
import zipfile
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

from shapely import wkb
from shapely.geometry import shape

HERE = Path(__file__).resolve().parent


def fix_name(info: zipfile.ZipInfo) -> str:
    """Zip entries without the UTF-8 flag were decoded as cp437; GSI archives use Shift-JIS or UTF-8."""
    if info.flag_bits & 0x800:
        return info.filename
    try:
        raw = info.filename.encode("cp437")
    except UnicodeEncodeError:  # already proper Unicode (e.g. from a Unicode path extra field)
        return info.filename
    for enc in ("utf-8", "cp932"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return info.filename


def unzip_all():
    for z in HERE.glob("gsi_*/*.zip"):
        out = z.with_suffix("")
        if out.exists():
            continue
        with zipfile.ZipFile(z) as zf:
            names = [i.filename for i in zf.infolist()]
            nested = all(n.startswith(z.stem + "/") for n in names)  # archive already has its own top folder
            for info in zf.infolist():
                name = fix_name(info)
                dest = (z.parent if nested else out) / name
                if name.endswith("/"):
                    dest.mkdir(parents=True, exist_ok=True)
                    continue
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(zf.read(info))


def area_m2(geom) -> float:
    lat = geom.centroid.y
    kx, ky = 111320 * math.cos(math.radians(lat)), 110540
    from shapely.affinity import scale
    return scale(geom, kx, ky, origin=(0, 0)).area


def summarize(geoms, props, src):
    types = Counter(g.geom_type for g in geoms)
    keys = Counter(k for p in props for k in p)
    polys = [g for g in geoms if g.geom_type in ("Polygon", "MultiPolygon")]
    lines = [g for g in geoms if g.geom_type in ("LineString", "MultiLineString")]
    out = dict(source=str(src.relative_to(HERE)), n=len(geoms), types=dict(types), property_keys=list(keys)[:15])
    if geoms:
        xs = [c for g in geoms for c in (g.bounds[0], g.bounds[2])]
        ys = [c for g in geoms for c in (g.bounds[1], g.bounds[3])]
        out["bbox"] = [round(min(xs), 4), round(min(ys), 4), round(max(xs), 4), round(max(ys), 4)]
    if polys:
        a = sorted(area_m2(g) for g in polys)
        out["area_m2"] = dict(median=round(a[len(a) // 2]), max=round(a[-1]), total_ha=round(sum(a) / 1e4, 1))
    if lines:
        out["total_length_km"] = round(sum(area_len(g) for g in lines) / 1000, 1)
    # categorical attributes worth showing (few distinct values)
    cats = {}
    for k in keys:
        vals = Counter(str(p.get(k)) for p in props)
        if 1 < len(vals) <= 12 and not k.startswith("_"):
            cats[k] = dict(vals.most_common(12))
    if cats:
        out["categories"] = cats
    return out


def area_len(g) -> float:
    lat = g.centroid.y
    from shapely.affinity import scale
    return scale(g, 111320 * math.cos(math.radians(lat)), 110540, origin=(0, 0)).length


def read_geojson(p):
    d = json.loads(p.read_text(encoding="utf-8-sig"))
    if not isinstance(d, dict) or "features" not in d:  # e.g. code_table.json, not a FeatureCollection
        return None, None
    feats = d.get("features", [])
    return [shape(f["geometry"]) for f in feats if f.get("geometry")], [f.get("properties") or {} for f in feats]


def read_kml(p):
    ns = {"k": "http://www.opengis.net/kml/2.2"}
    root = ET.parse(p).getroot()
    geoms, props = [], []
    from shapely.geometry import Polygon, LineString, Point
    for pm in root.iter("{http://www.opengis.net/kml/2.2}Placemark"):
        name = pm.findtext("k:name", default="", namespaces=ns)
        for tag, ctor in (("Polygon", "poly"), ("LineString", "line"), ("Point", "pt")):
            for el in pm.iter(f"{{http://www.opengis.net/kml/2.2}}{tag}"):
                txt = el.find(".//k:coordinates", ns)
                if txt is None or not txt.text:
                    continue
                pts = [tuple(map(float, c.split(",")[:2])) for c in txt.text.split()]
                try:
                    g = Polygon(pts) if ctor == "poly" else LineString(pts) if ctor == "line" else Point(pts[0])
                except Exception:
                    continue
                geoms.append(g); props.append({"name": name})
                break
    return geoms, props


def read_gpkg(p):
    con = sqlite3.connect(p)
    out = []
    for (table,) in con.execute("select table_name from gpkg_contents where data_type='features'"):
        gcol = con.execute("select column_name from gpkg_geometry_columns where table_name=?", (table,)).fetchone()[0]
        cols = [r[1] for r in con.execute(f'pragma table_info("{table}")')]
        geoms, props = [], []
        for row in con.execute(f'select * from "{table}"'):
            rec = dict(zip(cols, row))
            blob = rec.pop(gcol)
            if not blob:
                continue
            flags = blob[3]
            env = (flags >> 1) & 0x07
            header = 8 + {0: 0, 1: 32, 2: 48, 3: 48, 4: 64}[env]
            geoms.append(wkb.loads(bytes(blob[header:])))
            props.append(rec)
        out.append((table, geoms, props))
    return out


def main() -> int:
    unzip_all()
    inv = []
    for p in sorted(HERE.rglob("*")):
        if p.suffix.lower() in (".geojson", ".json") and "inventory" not in p.name:
            g, pr = read_geojson(p)
            if g is not None:
                inv.append(summarize(g, pr, p))
        elif p.suffix.lower() == ".kml":
            g, pr = read_kml(p)
            inv.append(summarize(g, pr, p))
        elif p.suffix.lower() == ".gpkg":
            for table, g, pr in read_gpkg(p):
                s = summarize(g, pr, p)
                s["table"] = table
                inv.append(s)
    (HERE / "inventory.json").write_text(json.dumps(inv, ensure_ascii=False, indent=1), encoding="utf-8")
    for s in inv:
        print(json.dumps(s, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
