# Source data

- GSI 2024 Noto earthquake disaster information: https://www.gsi.go.jp/BOUSAI/20240101_noto_earthquake.html
- GSI aerial slope-failure interpretation data: https://www.gsi.go.jp/BOUSAI/20240101_noto_earthquake.html (the five interpretation areas used in the manuscript).
- GSI pre-event annual orthophotos: `nendophoto<year>` layers, selected layer/year recorded per segment.
- GSI January and April disaster orthophotos: selected layer identifiers recorded per segment, e.g. `20240102noto_anamizu_0111do` and `20240102noto_0405_0426do`.
- GSI tile endpoint: `https://cyberjapandata.gsi.go.jp/xyz/<layer>/<zoom>/<x>/<y>.png`.
- GSI DEM deposit inventory: `20240101noto_landslidepolygon.gpkg`, EPSG:6675; deposition codes 2, 4, 6, 8 and 10. Original full builder expects it under `road_v2/external/gsi_noto_dem/`.
- OpenStreetMap roads: https://www.openstreetmap.org/ ; the saved sample geometry and folds should be used to reproduce this snapshot, because a new live OSM download can change the sample frame.
- MLIT road recovery map (source evidence used in the full preparation pipeline): https://www.mlit.go.jp/road/noto/index.html . Raw MLIT files and derived MLIT attributes are not distributed here.

The full builder expects the GSI interpretation GeoJSON files under `road_v2/external/gsi_noto/noto_*`, OSM roads under `road_v2/external/osm/`, and the original MLIT bundle under `road_v2/external/mlit_noto2024_roads/`. Download these from their providers if rebuilding candidate selection from scratch. For the final experiment snapshot, the included metadata and masks avoid regenerating the sample frame or review decisions.
