"""Draw genuine GSI views, continuous road annotations and a blockage close-up."""
from pathlib import Path
import concurrent.futures
import io, json, math, os, shutil, subprocess
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.colors import to_rgb
from shapely.geometry import LineString, box
from shapely.ops import unary_union

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'paper_revision/figures'
OUT.mkdir(parents=True, exist_ok=True)
CACHE = ROOT / 'paper_revision/tile_cache'
CACHE.mkdir(exist_ok=True)
DS = ROOT / 'dataset_noto2024_v0'
Z = 17
LAT, LON = 37.2641044, 136.9172309
W_M, H_M = 1200, 800
LAYERS = ['nendophoto2020', '20240102noto_anamizu_0111do']
RED, GREEN = '#e52c3c', '#15b957'
MASK_ALPHA, CORRIDOR_WIDTH_M = .48, 10

def px(lon, lat, zoom=Z):
    lon, lat = np.asarray(lon), np.asarray(lat)
    return np.stack([(lon+180)/360*256*2**zoom,
                     (1-np.arcsinh(np.tan(np.radians(lat)))/np.pi)/2*256*2**zoom], axis=-1)

gsd = math.cos(math.radians(LAT))*2*math.pi*6378137/(256*2**Z)
x0, y0 = np.rint(px(LON, LAT)-[W_M/gsd/2, H_M/gsd/2]).astype(int)
width, height = round(W_M/gsd), round(H_M/gsd)
viewport = box(0, 0, width, height)

def fetch(job):
    layer, zoom, x, y = job
    target = CACHE / f'{layer}_{zoom}_{x}_{y}.png'
    if not target.exists():
        url = f'https://cyberjapandata.gsi.go.jp/xyz/{layer}/{zoom}/{x}/{y}.png'
        curl_bin = os.environ.get('CURL_BIN') or shutil.which('curl')
        if not curl_bin:
            raise RuntimeError('Install curl or set CURL_BIN to its executable path.')
        result = subprocess.run([curl_bin, '-f', '-sS', '--connect-timeout', '10',
                                 '--max-time', '35', '--retry', '1', url], capture_output=True)
        if result.returncode:
            raise RuntimeError(f'Tile unavailable: {url}; '+result.stderr.decode(errors='replace')[:120])
        assert Image.open(io.BytesIO(result.stdout)).size == (256, 256)
        target.write_bytes(result.stdout)
    assert Image.open(target).size == (256, 256)
    return target

def compose(layer, zoom, origin, size):
    ox, oy = map(int, origin); w, h = map(int, size)
    tiles = [(x, y) for x in range(ox//256, (ox+w-1)//256+1)
             for y in range(oy//256, (oy+h-1)//256+1)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        list(pool.map(fetch, [(layer, zoom, x, y) for x, y in tiles]))
    canvas = Image.new('RGB', ((max(x for x, y in tiles)-ox//256+1)*256,
                              (max(y for x, y in tiles)-oy//256+1)*256))
    for x, y in tiles:
        canvas.paste(Image.open(CACHE/f'{layer}_{zoom}_{x}_{y}.png').convert('RGB'),
                     ((x-ox//256)*256, (y-oy//256)*256))
    return canvas.crop((ox%256, oy%256, ox%256+w, oy%256+h))

images = [compose(layer, Z, (x0, y0), (width, height)) for layer in LAYERS]
for layer, img in zip(LAYERS, images):
    img.save(OUT/f'landscape_{layer}.jpg', quality=96)

# Exact local subset makes the full regional OSM download optional on reproduction.
road_subset = OUT / 'landscape_roads.geojson'
full_roads = ROOT / 'external/osm/noto2024_roads.geojson'
road_source = full_roads if full_roads.exists() else road_subset
road_features = json.loads(road_source.read_text(encoding='utf-8'))['features']
roads, selected_features = [], []
for f in road_features:
    g = np.asarray(f['geometry']['coordinates'])
    line = LineString(px(g[:, 0], g[:, 1])-[x0, y0])
    if line.intersects(viewport):
        roads.append(line.intersection(viewport)); selected_features.append(f)
road_subset.write_text(json.dumps({'type': 'FeatureCollection', 'features': selected_features},
                                  ensure_ascii=False, separators=(',', ':')), encoding='utf-8')
assert roads, 'No mapped roads in this extent'

labels = pd.read_csv(DS/'labels_verified.csv').set_index('sample_id').label.to_dict()
features = json.loads((DS/'samples.geojson').read_text(encoding='utf-8'))['features']
blocked = []
for f in features:
    sid = f['properties']['sample_id']
    if labels.get(sid) != 1:
        continue
    g = np.asarray(f['geometry']['coordinates'])
    line = LineString(px(g[:, 0], g[:, 1])-[x0, y0])
    if line.intersects(viewport):
        blocked.append((sid, line.intersection(viewport)))
assert blocked, 'No blocked segment in this extent'

def raster_mask(geometry):
    mask = Image.new('L', (width, height), 0); draw = ImageDraw.Draw(mask)
    polygons = [geometry] if geometry.geom_type == 'Polygon' else list(geometry.geoms)
    for polygon in polygons:
        if polygon.geom_type != 'Polygon':
            continue
        draw.polygon(list(polygon.exterior.coords), fill=255)
        for hole in polygon.interiors:
            draw.polygon(list(hole.coords), fill=0)
    return np.asarray(mask) > 0

radius = CORRIDOR_WIDTH_M/2/gsd
road_mask = raster_mask(unary_union([r.buffer(radius) for r in roads]).intersection(viewport))
blocked_mask = raster_mask(unary_union([r.buffer(radius) for _, r in blocked]).intersection(viewport))
overlay = np.zeros((height, width, 4), dtype=float)
overlay[road_mask, :3] = to_rgb(GREEN); overlay[road_mask, 3] = MASK_ALPHA
overlay[blocked_mask, :3] = to_rgb(RED); overlay[blocked_mask, 3] = MASK_ALPHA

# Same post-event layer/date, at the native zoom-18 resolution.
bounds = unary_union([r for _, r in blocked]).bounds
cx, cy = (bounds[0]+bounds[2])/2, (bounds[1]+bounds[3])/2
zoom_w_m, zoom_h_m = 220, 220*2/3
zoom_w, zoom_h = zoom_w_m/gsd, zoom_h_m/gsd
zx, zy = cx-zoom_w/2, cy-zoom_h/2
close_origin = np.rint(2*np.array([x0+zx, y0+zy])).astype(int)
close_size = (round(2*zoom_w), round(2*zoom_h))
closeup = compose(LAYERS[1], 18, close_origin, close_size)
closeup.save(OUT/'landscape_blockage_closeup.jpg', quality=96)

plt.rcParams.update({'font.family': 'Arial', 'font.size': 8})
fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.65), facecolor='white')
titles = ['(a) Pre-event | 2020', '(b) Post-event | 11 January 2024',
          '(c) Road blockage annotations', '(d) Post-event blockage close-up']
for n, (ax, title) in enumerate(zip(axes.flat, titles)):
    img = images[0 if n == 0 else 1] if n < 3 else closeup
    ax.imshow(img); w, h = img.size
    ax.set(xlim=(0, w), ylim=(h, 0)); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, loc='left', fontsize=8.5, fontweight='bold', pad=5)
    for spine in ax.spines.values():
        spine.set_visible(False)
    scale = gsd if n < 3 else gsd/2; bar_m = 100 if n < 3 else 20
    ax.plot([w*.055, w*.055+bar_m/scale], [h*.94]*2, color='white', lw=3)
    ax.text(w*.055+bar_m/scale/2, h*.91, f'{bar_m} m', color='white', ha='center', fontsize=7.5,
            path_effects=[pe.withStroke(linewidth=2, foreground='black')])
    ax.text(w*.96, h*.08, 'N', color='white', ha='center', fontsize=8,
            path_effects=[pe.withStroke(linewidth=2, foreground='black')])
axes[1, 0].imshow(overlay)
axes[1, 0].legend(handles=[Line2D([0], [0], color=RED, lw=5, alpha=.65, label='Blocked'),
                          Line2D([0], [0], color=GREEN, lw=5, alpha=.65, label='Unblocked')],
                  loc='upper left', fontsize=7, framealpha=.95)
axes[1, 0].add_patch(Rectangle((zx, zy), zoom_w, zoom_h, fill=False, edgecolor='white',
                             linewidth=1, linestyle='--',
                             path_effects=[pe.withStroke(linewidth=2, foreground='#333333')]))
axes[1, 0].text(zx, zy-9, 'Detail in (d)', color='white', fontsize=7,
                 path_effects=[pe.withStroke(linewidth=2, foreground='#333333')])
fig.subplots_adjust(left=.005, right=.995, top=.96, bottom=.105, hspace=.16, wspace=.035)
fig.text(.01, .045, 'Source: GSI orthophotos | OSM roads', fontsize=7, color='#526575')
fig.text(.01, .022, 'Road annotations: blocked segments in red; remaining mapped roads in green',
         fontsize=7, color='#526575')
fig.savefig(OUT/'fig_landscape.jpg', dpi=400, pil_kwargs={'quality': 96})
fig.savefig(OUT/'fig_landscape.pdf', dpi=300)
metadata = {
    'centre_lat': LAT, 'centre_lon': LON, 'width_m': W_M, 'height_m': H_M,
    'zoom': Z, 'ground_sampling_m': gsd, 'layers': LAYERS,
    'mapped_road_features': len(roads), 'blocked_segments': [sid for sid, _ in blocked],
    'road_geometry': 'landscape_roads.geojson (local subset of original OSM download)',
    'mask_width_m': CORRIDOR_WIDTH_M, 'mask_alpha': MASK_ALPHA,
    'annotation_rule': 'label=1 segments red; all remaining mapped road portions green as requested',
    'role': 'landscape annotation and source-image illustration, not model prediction or new evaluation labels',
    'closeup': {'layer': LAYERS[1], 'date': '2024-01-11', 'zoom': 18,
                'x0': int(close_origin[0]), 'y0': int(close_origin[1]),
                'width_px': close_size[0], 'height_px': close_size[1],
                'width_m': zoom_w_m, 'height_m': zoom_h_m},
    'performance_metrics_changed': False,
}
(OUT/'landscape_provenance.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
print(json.dumps(metadata, ensure_ascii=False))
