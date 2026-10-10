"""Verify saved evaluation outputs without retraining or downloading imagery."""
from pathlib import Path
import csv, json, math, zipfile
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
DS = ROOT / 'road_v2/dataset_noto2024_v0'
RES = ROOT / 'road_v2/results_noto'
def read(path):
    with open(path, encoding='utf-8-sig') as handle: return list(csv.DictReader(handle))

samples = {r['sample_id']: r for r in read(DS / 'samples.csv')}
labels = {r['sample_id']: r for r in read(DS / 'labels_verified.csv')}
final = {s: r for s, r in labels.items() if int(r['label']) >= 0}
assert len(final) == 2750 and sum(int(r['label']) for r in final.values()) == 671
block_folds = {}
for sid in final:
    sample = samples[sid]; block, fold = sample['block_id'], sample['fold']
    assert sample['tunnel'] == '0' and sample['post_event_road'] == '0' and sample['post_date']
    assert block not in block_folds or block_folds[block] == fold
    block_folds[block] = fold
assert set(block_folds.values()) == {'0', '1', '2', '3', '4'}
checked = []
models = {}
for path in sorted(RES.glob('oof_ft_*verified*.csv')):
    if path.stem.endswith('_heads'): continue
    rows = read(path); ids = [r['sample_id'] for r in rows]
    assert len(set(ids)) == len(ids) and set(ids) == set(final), path.name
    for row in rows:
        assert row['label'] == final[row['sample_id']]['label']
        assert row['sample_type'] == samples[row['sample_id']]['sample_type']
        score = float(row['score']); assert math.isfinite(score) and 0 <= score <= 1
        if path.stem.endswith('_loro'): assert row['group'] == samples[row['sample_id']]['region']
        else: assert row['group'] == samples[row['sample_id']]['fold']
    y = np.array([int(r['label']) for r in rows]); score = np.array([float(r['score']) for r in rows])
    folds = [sorted(set(r['group'] for r in rows))][0]
    per = []
    for fold in folds:
        idx = np.array([r['group'] == fold for r in rows]); per.append((roc_auc_score(y[idx],score[idx]), average_precision_score(y[idx],score[idx])))
    model = {'pooled_AUROC': float(roc_auc_score(y, score)), 'pooled_AUPRC': float(average_precision_score(y,score)),
             'fold_mean_AUROC': float(np.mean(per,axis=0)[0]), 'fold_mean_AUPRC': float(np.mean(per,axis=0)[1])}
    result_file = RES / path.name.replace('oof_ft_', 'ft_').replace('.csv', '.json')
    if result_file.exists():
        saved = json.loads(result_file.read_text(encoding='utf-8'))['folds']
        assert abs(saved['AUROC_mean'] - model['fold_mean_AUROC']) < .001
        assert abs(saved['AUPRC_mean'] - model['fold_mean_AUPRC']) < .001
    heads_file = path.with_name(path.stem + '_heads.csv')
    if heads_file.exists():
        heads = {r['sample_id']: r for r in read(heads_file)}
        assert set(heads) == set(final)
        for row in rows:
            head = heads[row['sample_id']]
            assert abs(float(row['score']) - (float(head['score_fused'])+float(head['score_post']))/2) < .00002
    checked.append(path.name); models[path.stem] = model
with zipfile.ZipFile(ROOT / 'data/noto_masks.zip') as archive:
    assert archive.testzip() is None
    mask_names = set(archive.namelist())
    for sid in final:
        for kind in ('road', 'slide_aerial', 'slide_lidar'): assert f'masks/{kind}/{sid}.png' in mask_names
summary = {'final_segments': len(final), 'blocked': 671, 'clear': 2079, 'spatial_blocks': len(block_folds),
           'checked_oof_files': len(checked), 'models': models}
(ROOT / 'audit_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
print(json.dumps({k:v for k,v in summary.items() if k != 'models'}, indent=2))
print('Proposed:', json.dumps(models['oof_ft_convnext_siamese_v2_verified']))
