# Dataset card: NotoRoadBlock research snapshot

## Unit and imagery

One sample is an OSM road segment of nominal length 50 m. Image chips are 256 × 256 pixels at zoom 18, approximately 0.47 m/pixel around the study area. Pre-event imagery is from GSI annual orthophotos (2010–2022); post-event imagery is selected from January 2024 disaster orthophotos. April 2024 imagery supported visual interpretation and is not an input to the reported two-temporal networks.

The saved sample frame has 3,369 segments. The final experiment subset excludes tunnels, post-event detour roads, a missing January image, undeterminable labels and conflicting non-candidate obstructions, leaving 2,750 segments (671 positive, 2,079 negative). `sample_type` is candidate, near or random. It describes sampling, not the final blockage class.

## Label protocol and codes

- Candidate status is based on a 5 m road buffer intersecting GSI interpreted slope-failure/deposit polygons.
- Blind visual review uses pre/post/later images without displaying candidate labels or slope polygons. Disputed candidate cases were re-inspected with stricter road-surface criteria.
- Label 1: blocked; label 0: clear.
- Label -1: undeterminable; -2: required review not completed; -3: a non-candidate obstruction excluded by the study's slope-failure-focused label protocol.
- Many non-candidate negatives retain label 0 without individual visual review. `in_review`, `review_label` and `label_reviewed` preserve that distinction. Do not describe all 2,750 labels as independently confirmed ground truth.

Labels describe visible road-surface obstruction in the selected imagery, including partial debris coverage. They are not field-verified vehicle-passability or official closure labels.

## Evaluation

Five spatial folds group complete 2 km blocks; region names support leave-one-region-out testing. Closely spaced segments can have overlapping imagery. Random patch splitting is inappropriate for spatial independence. Positive sampling is enriched, so the sample prevalence is not the road-network prevalence. Inspection-effort estimates use population weights and their uncertainty depends on the number of sampled high-scoring negatives.

## Files

- `samples.csv`: layer/date/pixel coordinates, sampling stratum, region, block and fold.
- `samples.geojson`: corresponding road-segment geometries and metadata.
- `labels_verified.csv`: merged label outcomes, including exclusions.
- `data/noto_masks.zip`: original `masks/road`, `masks/slide_aerial`, `masks/slide_lidar` raster masks.
- `results_noto/*verified*.csv/.json`: saved evaluation outputs. Existing results are a reference snapshot; rerunning training may not reproduce bitwise identical predictions.

The original OSM download reflects the road network at preparation time, rather than a historical pre-earthquake snapshot. This and single-event evaluation limit interpretation. Data-source attribution and terms are in `DATA_LICENSE.md`.
