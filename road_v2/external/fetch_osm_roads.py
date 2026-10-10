"""Download OpenStreetMap road centrelines for the study areas via the Overpass API.

Roads only (motorway ... residential, plus links); no tracks, service roads or footpaths.
Data © OpenStreetMap contributors, ODbL 1.0.

Usage:
  python road_v2/external/fetch_osm_roads.py
Writes osm/<area>_roads.geojson
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
OUT = HERE / "osm"
AREAS = {  # south, west, north, east (from the GSI interpretation extents)
    "noto2024": (36.99, 136.64, 37.56, 137.40),
    "kumamoto2026": (32.24, 130.43, 32.90, 131.07),
    "hokkaido2018": (42.54, 141.68, 42.96, 142.13),
}
HIGHWAY = "motorway|trunk|primary|secondary|tertiary|unclassified|residential|motorway_link|trunk_link|primary_link|secondary_link|tertiary_link"
URLS = ["https://overpass-api.de/api/interpreter", "https://overpass.kumi.systems/api/interpreter",
        "https://maps.mail.ru/osm/tools/overpass/api/interpreter"]


def main() -> int:
    OUT.mkdir(exist_ok=True)
    for name, (s, w, n, e) in AREAS.items():
        dest = OUT / f"{name}_roads.geojson"
        if dest.exists():
            print("skip", dest.name)
            continue
        q = f'[out:json][timeout:300];way["highway"~"^({HIGHWAY})$"]({s},{w},{n},{e});out tags geom;'
        t0 = time.time()
        els = None
        for url in URLS:  # public servers are often busy (429/504): fall back to mirrors
            try:
                r = requests.post(url, data={"data": q}, timeout=600, headers={"User-Agent": "road-blockage-research/0.1"})
                r.raise_for_status()
                els = r.json()["elements"]
                break
            except Exception as exc:
                print(f"  {url}: {exc}", flush=True)
        if els is None:
            print(f"{name}: all Overpass servers failed")
            continue
        feats = [dict(type="Feature",
                      properties=dict(osm_id=el["id"], highway=el["tags"].get("highway"), name=el["tags"].get("name"),
                                      bridge=el["tags"].get("bridge"), tunnel=el["tags"].get("tunnel")),
                      geometry=dict(type="LineString", coordinates=[[p["lon"], p["lat"]] for p in el["geometry"]]))
                 for el in els if el.get("type") == "way" and len(el.get("geometry", [])) >= 2]
        dest.write_text(json.dumps(dict(type="FeatureCollection", features=feats), ensure_ascii=False), encoding="utf-8")
        print(f"{name}: {len(feats)} ways, {dest.stat().st_size / 1e6:.1f} MB, {time.time() - t0:.0f}s", flush=True)
        time.sleep(5)  # be polite to the public Overpass server
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
