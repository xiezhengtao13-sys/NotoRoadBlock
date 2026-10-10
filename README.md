# NotoRoadBlock

Research code and supporting data for segment-based Siamese detection of earthquake-induced road blockage from aerial orthophotos, developed for the 2024 Noto Peninsula earthquake.

The final analysis subset contains **2,750 segments: 671 blocked and 2,079 clear**. Roads are represented by 50 m segments and 256 × 256 pre/post image chips. The proposed network combines global encoder features with a local change representation pooled along the target road corridor.

## Included

- Data preparation, review-label merging, baseline, training, ablation, regional-transfer and evaluation scripts.
- Sample metadata, road-segment geometries, final labels, 2 km spatial blocks and five-fold assignments.
- A compact archive of the original road, aerial-deposit and DEM-deposit masks.
- Verified-label evaluation outputs, including out-of-fold probabilities, three training seeds and individual prediction-head scores.
- Scripts and TeX tables for checking the reported results and drawing figures.

Full orthophoto chips, trained weights and the manuscript are not bundled. The chips can be reconstructed from the recorded GSI layer identifiers and pixel coordinates. This repository is a research snapshot, not a live road-closure service.

## Quick verification without a GPU

```bash
python -m pip install -r requirements-analysis.txt
python scripts/audit_results.py
```

The audit checks sample/label alignment, spatial-fold consistency, pooled and fold-mean metrics, and the proposed network's two-head averaging. It writes `audit_summary.json`. Pooled metrics and fold-mean metrics answer different questions; the manuscript's main comparison uses fold means.

## Restore model inputs

```bash
python scripts/unpack_masks.py
# Small example download; use --paper-only to reconstruct all final-analysis chips.
python road_v2/download_chips.py --ids noto_00794 noto_00950 --kinds pre post
```

The downloader uses the original selected layers and zoom-18 crop coordinates, checks coverage and saves JPEGs at the original quality setting. It does not reselect dates or alter labels/folds. GSI availability and upstream imagery updates can affect reconstruction; original acquisition dates and sources are retained in the metadata.

## Training and experiments

Install a suitable PyTorch build for your hardware and `requirements-training.txt`. Pretrained model weights are obtained through `timm`; the original scripts default to `HF_HUB_OFFLINE=1`. Set it to `0` for the first run if the weights are not already cached. GPU runs in the original work used an RTX 4060 Laptop with 8 GB VRAM.

```bash
python road_v2/baseline_noto.py --labels verified
python road_v2/change_features.py
python road_v2/finetune_noto.py convnext_post convnext_siamese resnet50_post swin_post convnext_post_mask --labels verified
python road_v2/finetune_noto_v2.py convnext_siamese_v2 convnext_siamese_v2_noalign convnext_siamese_v2_nocorridor convnext_siamese_v2_nolocal convnext_siamese_v2_aux --labels verified
python road_v2/finetune_noto.py --loro convnext_post convnext_siamese --labels verified
python road_v2/finetune_noto_v2.py --loro convnext_siamese_v2 --labels verified
```

Run additional seeds separately using `FT_SEED_OFFSET=100` with `FT_NAME_SUFFIX=_s2`, and `FT_SEED_OFFSET=200` with `FT_NAME_SUFFIX=_s3`, for `convnext_post` and `convnext_siamese_v2`. Then run:

```bash
python road_v2/aggregate_seeds.py convnext_post convnext_siamese_v2
python road_v2/compare_oof.py convnext_post convnext_siamese_v2
```

The selected seed offsets are recorded in `docs/reproduction.md`. Actual training environments are recorded in `docs/tested_environments.json`. CPU analysis and GIS preparation were performed in a separate environment from GPU training.

## Data and labels

Read [the dataset card](docs/dataset_card.md) before using `labels_verified.csv`. The filename describes the merged label protocol; it does not mean that every clear segment was independently inspected. Exclusion codes and the treatment of non-candidate obstructions are documented there. Use the saved spatial folds, not a random image-patch split.

`build_noto_dataset.py` records the original full reconstruction pipeline, which requires the external OSM, GSI and MLIT files described in [the source list](docs/data_sources.md). For reproducing the published experiment subset, use the included metadata, masks and selected-layer downloader. MLIT-derived fields are omitted from the public sample metadata; they are not model inputs or evaluation variables.

## Sources and attribution

Road geometries and OSM attributes: **© OpenStreetMap contributors**, under [ODbL 1.0](https://opendatacommons.org/licenses/odbl/1-0/). See [OpenStreetMap copyright](https://www.openstreetmap.org/copyright).

GSI imagery and interpreted deposit polygons: **Source: Geospatial Information Authority of Japan**. Masks and label evidence are derived products prepared by the research authors, not official GSI road-status products. See [GSI terms of use](https://www.gsi.go.jp/ENGLISH/page_e30286.html) and [data-source details](docs/data_sources.md).

See [DATA_LICENSE.md](DATA_LICENSE.md) for the distinction between third-party data terms and software. A software licence has not yet been selected by the authors.
