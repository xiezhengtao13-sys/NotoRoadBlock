"""Which GSI photo layers cover each study site, and when were they taken?

For sample points in each earthquake area, request one zoom-17 tile from every yearly aerial-photo layer
(nendophoto2007 ... nendophoto2026), the seamless "latest" photo and the base ortho, and read the
capture-date metadata from the matching *_spec / .geojson tiles where they exist.

Usage:
  python road_v2/external/probe_pre_event.py
Writes pre_event_coverage.json next to this script.
"""
from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
SITES = {  # event date, sample points (lat, lon)
    "noto2024 (2024-01-01)": [("Wajima W coast road", 37.3759, 136.7929), ("Anamizu valley", 37.2526, 136.8768),
                              ("Suzu centre", 37.4360, 137.2600)],
    "kumamoto2026 (2026-07-28)": [("Yatsushiro", 32.4500, 130.6000), ("Misato hills", 32.6400, 130.8500)],
    "hokkaido2018 (2018-09-06)": [("Atsuma landslides", 42.7600, 141.9500)],
}
LAYERS = ["seamlessphoto", "ort"] + [f"nendophoto{y}" for y in range(2026, 2006, -1)]


def tile(lat, lon, z):
    n = 2 ** z
    return int((lon + 180) / 360 * n), int((1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * n)


def get(url) -> bytes | None:
    curl = os.environ.get('CURL_BIN') or shutil.which('curl')
    if not curl: raise RuntimeError('Install curl or set CURL_BIN to its executable.')
    r = subprocess.run([curl, "-s", "-f", url], capture_output=True)
    return r.stdout if r.returncode == 0 and r.stdout else None


def spec(layer, lat, lon):
    """Capture metadata: GSI serves it as GeoJSON tiles (layer itself or <layer>_spec), zoom varies."""
    for name in (layer, layer + "_spec") if layer.startswith("nendophoto") else (layer + "_spec",):
        for z in (16, 15, 14, 13, 12, 11, 10):
            x, y = tile(lat, lon, z)
            data = get(f"https://maps.gsi.go.jp/xyz/{name}/{z}/{x}/{y}.geojson")
            if not data:
                continue
            try:
                feats = json.loads(data.decode("utf-8-sig")).get("features", [])
            except Exception:
                continue
            props = [f.get("properties", {}) for f in feats]
            dates = sorted({str(v) for p in props for k, v in p.items() if any(s in k for s in ("撮影", "date", "Date", "年月"))})
            if dates:
                return dates[:4]
    return None


def main() -> int:
    out = {}
    for event, pts in SITES.items():
        out[event] = {}
        for name, lat, lon in pts:
            x, y = tile(lat, lon, 17)
            hits = {}
            for lay in LAYERS:
                for ext in ("jpg", "png"):
                    if get(f"https://cyberjapandata.gsi.go.jp/xyz/{lay}/17/{x}/{y}.{ext}"):
                        hits[lay] = spec(lay, lat, lon)
                        break
            out[event][name] = hits
            print(f"{event:28s} {name:22s} -> " + ", ".join(f"{k}{'' if v is None else ' ' + str(v)}" for k, v in hits.items()), flush=True)
    (HERE / "pre_event_coverage.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
