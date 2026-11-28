# MacaoSpatialEquity Phase A to Phase B data dictionary

This note is the schema contract for the frozen bundle that Phase B reads.
Analysis stays in EPSG:32649. Statistical estimates are not redefined here.
They live in `outputs/tables/` and `outputs/metrics/phase_a_ml_baselines.json`.

## Coordinate reference system

- Every analysis GeoPackage uses **EPSG:32649** (WGS 84 / UTM zone 49N).
- DSEC census geometries are read in native EPSG:8433 (`always_xy=True`) and
  transformed locally. The broken server-side `outSR=4326` path is not used.
- Maritime zone Z26 is excluded. Terrestrial 2021 census totals used as the
  population control are 681,293 residents and 82,751 residents aged 65 or older.
- H3 cell population is an area-weighted sum of DSEC building-group counts.
  It is not a uniform building-height dasymetric surface. Grouped high-rise
  footprints can still spread people evenly inside one polygon.

## Frozen files

- `cells.gpkg`: canonical populated H3 resolution-9 cells. M1 columns are kept
  and M2 `acc10_`, `acc15_`, and `lisa_` columns are appended.
- `cells_res8.gpkg`: canonical populated H3 resolution-8 cells for the MAUP
  check, with the same class of appended columns when the interim layer has them.
- `cells_base_res9.gpkg` and `cells_base_res8.gpkg`: M1 base grids. Phase B
  should prefer `cells.gpkg` / `cells_res8.gpkg`.
- `zones_23.gpkg`: 23 terrestrial DSEC zones (Z01–Z23), EPSG:32649.
- `parishes_7.gpkg`: 7 official parishes dissolved from those zones, including
  Taipa (Nossa Senhora do Carmo / 嘉模堂區).
- `facilities_clean.gpkg`: cleaned park, health centre, library, bus stop, and
  EV charger points after cross-source 25 m deduplication. Same-source
  neighbours, including opposite bus poles, are not collapsed.
- `lrt_stations.gpkg`: operational MLM stations inside the SAR.
- `graph_edges.csv`: 10-minute pedestrian edges with columns `src_cell_id`,
  `dst_cell_id`, `walk_time_min`, `weight`, `same_block`.
- `spatial_folds.csv`: frozen geographic blocks with columns `cell_id`,
  `spatial_block` (0–4, seed 42).
- `split_metadata.json`: block sizes and positive-cell counts.
- `cell_poi_texts.csv`: columns `cell_id`, `poi_count_raw`, `poi_count_filtered`,
  `poi_text`. POI text has EV and charging tokens removed for a later text model.
- `feature_manifest.json`: leak-free `ml_features` whitelist.
- `FREEZE_MANIFEST.json`: SHA-256 of every other file in this directory.

## Anti-leakage

Machine-learning and later graph models may use only `ml_features` from
`feature_manifest.json`. The target and audit columns are `has_ev_charger`,
`ev_station_count`, and `ev_charger_capacity`. Columns that start with
`acc10_`, `acc15_`, or `lisa_` are spatial-analysis outputs. They are not
features for predicting whether a cell has a charger. `cell_poi_texts.csv`
is stripped of EV, charger, and charging-station strings before any text model
sees it. Imputation and scaling for the Phase A baselines are fit inside each
training block, not on the full grid.

`spatial_block` is the frozen cross-validation label. It is not a covariate.

## `cells.gpkg` columns

- `cell_id`
- `centroid_x`
- `centroid_y`
- `area_km2`
- `zone_id`
- `zone_name`
- `parish_id`
- `parish_name`
- `pop_total`
- `pop_65plus`
- `pop_density`
- `elderly_share`
- `intersection_density`
- `road_density_km_per_km2`
- `res_bldg_share`
- `com_bldg_share`
- `bldg_coverage_ratio`
- `dist_nearest_lrt_m`
- `dist_nearest_bus_m`
- `has_ev_charger`
- `ev_station_count`
- `ev_charger_capacity`
- `spatial_block`
- `acc10_park`
- `acc10_health_centre`
- `acc10_library`
- `acc10_bus_stop`
- `acc10_ev_charger`
- `acc10_ev_unweighted`
- `acc10_composite`
- `acc15_park`
- `acc15_health_centre`
- `acc15_library`
- `acc15_bus_stop`
- `acc15_ev_charger`
- `acc15_ev_unweighted`
- `acc15_composite`
- `lisa_i_acc10_park`
- `lisa_p_acc10_park`
- `lisa_q_fdr_acc10_park`
- `lisa_cluster_acc10_park`
- `lisa_i_acc10_health_centre`
- `lisa_p_acc10_health_centre`
- `lisa_q_fdr_acc10_health_centre`
- `lisa_cluster_acc10_health_centre`
- `lisa_i_acc10_library`
- `lisa_p_acc10_library`
- `lisa_q_fdr_acc10_library`
- `lisa_cluster_acc10_library`
- `lisa_i_acc10_bus_stop`
- `lisa_p_acc10_bus_stop`
- `lisa_q_fdr_acc10_bus_stop`
- `lisa_cluster_acc10_bus_stop`
- `lisa_i_acc10_ev_charger`
- `lisa_p_acc10_ev_charger`
- `lisa_q_fdr_acc10_ev_charger`
- `lisa_cluster_acc10_ev_charger`
- `lisa_i_acc10_composite`
- `lisa_p_acc10_composite`
- `lisa_q_fdr_acc10_composite`
- `lisa_cluster_acc10_composite`
- `lisa_i_acc15_park`
- `lisa_p_acc15_park`
- `lisa_q_fdr_acc15_park`
- `lisa_cluster_acc15_park`
- `lisa_i_acc15_health_centre`
- `lisa_p_acc15_health_centre`
- `lisa_q_fdr_acc15_health_centre`
- `lisa_cluster_acc15_health_centre`
- `lisa_i_acc15_library`
- `lisa_p_acc15_library`
- `lisa_q_fdr_acc15_library`
- `lisa_cluster_acc15_library`
- `lisa_i_acc15_bus_stop`
- `lisa_p_acc15_bus_stop`
- `lisa_q_fdr_acc15_bus_stop`
- `lisa_cluster_acc15_bus_stop`
- `lisa_i_acc15_ev_charger`
- `lisa_p_acc15_ev_charger`
- `lisa_q_fdr_acc15_ev_charger`
- `lisa_cluster_acc15_ev_charger`
- `lisa_i_acc15_composite`
- `lisa_p_acc15_composite`
- `lisa_q_fdr_acc15_composite`
- `lisa_cluster_acc15_composite`

## `cells_res8.gpkg` columns

- `cell_id`
- `area_km2`
- `zone_id`
- `parish_id`
- `parish_name`
- `pop_total`
- `pop_65plus`
- `pop_density`
- `elderly_share`
- `has_ev_charger`
- `acc10_park`
- `acc10_health_centre`
- `acc10_library`
- `acc10_bus_stop`
- `acc10_ev_charger`
- `acc10_ev_unweighted`
- `acc10_composite`
- `acc15_park`
- `acc15_health_centre`
- `acc15_library`
- `acc15_bus_stop`
- `acc15_ev_charger`
- `acc15_ev_unweighted`
- `acc15_composite`
- `lisa_i_acc10_park`
- `lisa_p_acc10_park`
- `lisa_q_fdr_acc10_park`
- `lisa_cluster_acc10_park`
- `lisa_i_acc10_health_centre`
- `lisa_p_acc10_health_centre`
- `lisa_q_fdr_acc10_health_centre`
- `lisa_cluster_acc10_health_centre`
- `lisa_i_acc10_library`
- `lisa_p_acc10_library`
- `lisa_q_fdr_acc10_library`
- `lisa_cluster_acc10_library`
- `lisa_i_acc10_bus_stop`
- `lisa_p_acc10_bus_stop`
- `lisa_q_fdr_acc10_bus_stop`
- `lisa_cluster_acc10_bus_stop`
- `lisa_i_acc10_ev_charger`
- `lisa_p_acc10_ev_charger`
- `lisa_q_fdr_acc10_ev_charger`
- `lisa_cluster_acc10_ev_charger`
- `lisa_i_acc10_composite`
- `lisa_p_acc10_composite`
- `lisa_q_fdr_acc10_composite`
- `lisa_cluster_acc10_composite`
- `lisa_i_acc15_park`
- `lisa_p_acc15_park`
- `lisa_q_fdr_acc15_park`
- `lisa_cluster_acc15_park`
- `lisa_i_acc15_health_centre`
- `lisa_p_acc15_health_centre`
- `lisa_q_fdr_acc15_health_centre`
- `lisa_cluster_acc15_health_centre`
- `lisa_i_acc15_library`
- `lisa_p_acc15_library`
- `lisa_q_fdr_acc15_library`
- `lisa_cluster_acc15_library`
- `lisa_i_acc15_bus_stop`
- `lisa_p_acc15_bus_stop`
- `lisa_q_fdr_acc15_bus_stop`
- `lisa_cluster_acc15_bus_stop`
- `lisa_i_acc15_ev_charger`
- `lisa_p_acc15_ev_charger`
- `lisa_q_fdr_acc15_ev_charger`
- `lisa_cluster_acc15_ev_charger`
- `lisa_i_acc15_composite`
- `lisa_p_acc15_composite`
- `lisa_q_fdr_acc15_composite`
- `lisa_cluster_acc15_composite`

## Reading rule for Phase B

Join on `cell_id`. Do not recompute the published Gini, Theil, regression, or
cross-validation numbers from scratch if the goal is to quote Phase A. Those
numbers are the committed tables. This bundle freezes the cell attributes those
tables were built from, plus the accessibility and LISA fields needed for maps
and for a later graph model that must not treat them as charger-prediction features.
