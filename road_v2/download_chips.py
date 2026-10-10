"""Reconstruct selected original GSI chips from saved layer and zoom-18 crop metadata."""
from pathlib import Path
import argparse, concurrent.futures, csv, io, json, re, time
import urllib.error, urllib.request
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
DS = ROOT / 'dataset_noto2024_v0'
CACHE = ROOT / 'tile_cache'
Z = 18

def fetch(job):
    layer, x, y = job
    assert re.fullmatch(r'[A-Za-z0-9_-]+', layer)
    path = CACHE / f'{layer}_{Z}_{x}_{y}.png'
    if path.exists(): return path
    request = urllib.request.Request(f'https://cyberjapandata.gsi.go.jp/xyz/{layer}/{Z}/{x}/{y}.png',
                                     headers={'User-Agent': 'NotoRoadBlock-research-reconstruction/1.0'})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=40) as response: data = response.read()
            image = Image.open(io.BytesIO(data)); image.load()
            assert image.size == (256, 256)
            path.write_bytes(data); return path
        except urllib.error.HTTPError as error:
            if error.code == 404: return None
            if attempt == 2: raise
        except (OSError, TimeoutError):
            if attempt == 2: raise
        time.sleep(attempt + 1)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ids', nargs='+')
    parser.add_argument('--paper-only', action='store_true')
    parser.add_argument('--kinds', nargs='+', choices=('pre', 'post', 'post_apr'), default=['pre', 'post'])
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    if not args.ids and not args.paper_only: parser.error('Choose --ids or --paper-only; no implicit bulk download.')
    rows = list(csv.DictReader(open(DS / 'samples.csv', encoding='utf-8-sig')))
    labels = {r['sample_id']: int(r['label']) for r in csv.DictReader(open(DS / 'labels_verified.csv', encoding='utf-8-sig'))}
    if args.ids:
        missing = set(args.ids) - {r['sample_id'] for r in rows}
        if missing: parser.error('Unknown sample IDs: ' + ', '.join(sorted(missing)))
        rows = [r for r in rows if r['sample_id'] in args.ids]
    if args.paper_only: rows = [r for r in rows if labels.get(r['sample_id'], -9) >= 0]
    targets, jobs = [], set()
    for row in rows:
        assert re.fullmatch(r'noto_\d{5}', row['sample_id'])
        x0, y0 = int(row['x0']), int(row['y0'])
        for kind in args.kinds:
            layer = row[f'{kind}_layer']
            if not layer: continue
            target = DS / 'chips' / kind / f"{row['sample_id']}.jpg"
            if target.exists(): continue
            coordinates = [(x, y) for x in range(x0//256, (x0+255)//256+1)
                           for y in range(y0//256, (y0+255)//256+1)]
            jobs.update((layer, x, y) for x, y in coordinates)
            targets.append((row, kind, target, coordinates))
    CACHE.mkdir(exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        paths = dict(zip(sorted(jobs), pool.map(fetch, sorted(jobs))))
    for row, kind, target, coordinates in targets:
        x0, y0 = int(row['x0']), int(row['y0']); layer = row[f'{kind}_layer']
        rgba = np.zeros((256, 256, 4), dtype=np.uint8)
        for x, y in coordinates:
            path = paths[(layer, x, y)]
            if path is None: continue
            tile = np.asarray(Image.open(path).convert('RGBA'))
            sx0, sx1 = max(x0,x*256), min(x0+256,x*256+256)
            sy0, sy1 = max(y0,y*256), min(y0+256,y*256+256)
            rgba[sy0-y0:sy1-y0,sx0-x0:sx1-x0] = tile[sy0-y*256:sy1-y*256,sx0-x*256:sx1-x*256]
        valid = float((rgba[:,:,3] > 0).mean())
        if valid < float(row[f'{kind}_valid']) - .001:
            raise RuntimeError(f"Reduced source coverage: {row['sample_id']} {kind}: {valid:.4f}")
        target.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(rgba[:,:,:3]).save(target, quality=90)
    print(json.dumps({'chips_restored': len(targets), 'tiles_requested': len(jobs)}))

if __name__ == '__main__': main()
