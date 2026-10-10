"""Regenerate Figure 5 only, using actual verified GSI image chips."""
from pathlib import Path
import csv, importlib.util, json

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
spec = importlib.util.spec_from_file_location('paper_figures', ROOT / 'paper_figures.py')
pf = importlib.util.module_from_spec(spec); spec.loader.exec_module(pf)
pf.OUT = HERE / 'figures'
pf.OUT.mkdir(parents=True, exist_ok=True)
ver = {r['sample_id']: int(r['label']) for r in csv.DictReader(open(pf.DS / 'labels_verified.csv', encoding='utf-8-sig'))}
rows = [dict(r, y=ver[r['sample_id']]) for r in csv.DictReader(open(pf.DS / 'samples.csv', encoding='utf-8-sig'))
        if ver.get(r['sample_id'], -9) >= 0]
blocked = ['noto_00794', 'noto_00028', 'noto_00999']
clear = ['noto_00796', 'noto_02759']
by = {r['sample_id']: r for r in rows}
assert all(by[s]['y'] == 1 for s in blocked)
assert all(by[s]['y'] == 0 for s in clear)
pf.fig_examples(rows, blocked, clear)
records = [{k: by[s][k] for k in ('sample_id', 'y', 'highway', 'region', 'pre_date', 'post_date', 'post_apr_date')}
           for s in blocked + clear]
(pf.OUT / 'examples_provenance.json').write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding='utf-8')
print('Figure 5(c): noto_00999, verified blocked, primary road, 2 January 2024 orthophoto.')
