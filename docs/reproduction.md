# Reproduction guide

## Experiment map

| ID | Question | Scripts / inputs |
|---|---|---|
| E1 | Overall model comparison | `baseline_noto.py`, `finetune_noto.py`, `finetune_noto_v2.py`; verified labels and five spatial folds |
| E2 | Bi-temporal and road evidence | Frozen road/global feature variants, `change_features.py`, post-only and post+mask configurations; metrics by negative stratum in `paper_results.py` |
| E3 | Component ablations | `convnext_siamese_v2_noalign`, `_nocorridor`, `_nolocal`, `_aux`; saved `_heads.csv` for fused/post head outputs |
| E4 | Training stability | Proposed and post-only models with seeds 100–104, 200–204 and 300–304; `aggregate_seeds.py` |
| E5 | Regional transfer | `--loro` in the fine-tuning scripts; saved frozen-model regional results |
| E6 | Inspection effort | Saved out-of-fold probabilities; population weighting and block-bootstrap functions in `paper_results.py` and `compare_oof.py` |

These are checks of one earthquake, with spatial and regional holdouts. They do not demonstrate transfer to a new earthquake or measure field deployment performance.

## Training settings

The base fold seed is 100 + fold index. The second and third runs use offsets 100 and 200, respectively. This is taken from the original `run_v2_loro_seeds.ps1` launcher. All three use the same saved folds. Additional seed runs are available only for the proposed and post-only networks.

PowerShell example for the second seed:

```powershell
$env:FT_SEED_OFFSET = '100'
$env:FT_NAME_SUFFIX = '_s2'
python road_v2/finetune_noto.py convnext_post --labels verified
python road_v2/finetune_noto_v2.py convnext_siamese_v2 --labels verified
Remove-Item Env:FT_SEED_OFFSET
Remove-Item Env:FT_NAME_SUFFIX
```

For the third run use offset 200 and suffix `_s3`. Never run GPU training jobs simultaneously on an 8 GB GPU. The original protocol uses 10 epochs, AdamW, separate encoder/head learning rates, a warm-up epoch, cosine decay, shared geometric augmentation and independent mild photometric jitter. See the source for exact values. The final epoch is evaluated; there is no test-fold early stopping.

## Analysis and figures

Saved verified-label OOF outputs allow result auditing without imagery or pretrained weights:

```bash
python scripts/audit_results.py
python road_v2/compare_oof.py convnext_post convnext_siamese_v2 --ref convnext_post
```

The second command recomputes block-bootstrap comparisons and can take several minutes. To regenerate all original result tables and figures, restore all final-subset image chips and masks, create `paper_buildings/figures`, then run `python road_v2/paper_results.py --labels verified`. Output tables require the MDPI template's tabular/adjustwidth macros when included in LaTeX.

The network and dataset workflow diagrams use Arial, with a platform fallback if Arial is unavailable. `paper_revision/update_examples.py` creates the data-example figure; row (c) uses verified blocked sample `noto_00950`. `paper_revision/draw_landscape.py` uses genuine GSI images and saved OOF scores. It requires `curl` on PATH or `CURL_BIN` to point to the executable.

The scripts reproduce the reported workflow; the record is not a claim of bitwise repeatability across CUDA versions or future changes to the upstream data services. The source figures and text must be checked by the authors before submission.
