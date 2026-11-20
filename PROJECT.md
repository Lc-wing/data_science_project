# Project: MacaoSpatialEquity — Plan 5 Phase A (CS482 Data Science)

> **Single Source of Truth** maintained by the top-level Project Orchestrator (`orchestrator_r1`).
> Scope: Complete **all Phase A (阶段A) tasks** of `plans/plan_5_macao_facilities_ev_graph_dl.md` (v2) in full compliance with `plans/00_review_report.md` and CS482 `Project-requirements` (Google Drive ID `1-F3ff-pTu3-pbU5cIzW343U1u-AgWrfs-gJx7qZfOV8`), culminating in verified frozen Phase A→B artifacts, programmatic Figure 1 / Table 1, a 3.5–3.8 A4 page Phase A report draft passing the 12/03 Go/No-Go checkpoint, and 100% E2E + Tier-5 adversarial verification.

---

## Architecture

### System Overview & Data Flow
```text
[Live Public Sources (Unauthenticated, Verified 2026-11-20)]
  ├── DSEC 2021 Census ArcGIS REST (24 zones + 5,682 buildings; native EPSG:8433 -> EPSG:32649)
  ├── CEM Live EV API (84 stations, 1,610 unique chargers; swap lat/lon -> EPSG:32649)
  ├── Geofabrik Macau OSM (.osm.pbf + .gpkg.zip) + overpass.openstreetmap.fr (walk graph, POIs, buildings)
  └── Official Web Registries (gov.mo SSM, library.gov.mo IC, iam.gov.mo, mlm.com.mo) for whitelisting
          │
          ▼  [M1: Ingest, CRS Cleaning, 25m Cross-Source Dedup, H3 Grids, Walk Graph & Frozen Splits]
  data/frozen/:
    cells_base_res9.gpkg, cells_base_res8.gpkg, zones_23.gpkg, parishes_7.gpkg,
    facilities_clean.gpkg, lrt_stations.gpkg, graph_edges.csv, spatial_folds.csv,
    split_metadata.json, cell_poi_texts.csv (EV-stripped), feature_manifest.json
          │
          ├──────────────────────────────────────────────────────────┐
          ▼                                                          ▼
  [M2: Spatial Analytics & Econometrics]                 [M3: Leak-Free ML Baselines]
    • Gaussian 2SFCA (10 & 15 min; r9, r8, zones)          • Strictly consumes M1 non-EV features
    • EDA summaries & Lorenz curve data                    • Frozen 5-fold Spatial Block CV
    • H1: Pop-weighted Gini, Theil's T (within/between     • Naive, Logistic Regression, Random Forest
          parish) + 2000-rep bootstrap CI, LISA (999+FDR)  • Metrics: ROC-AUC, PR-AUC, Brier,
    • H2: Paired bootstrap Gini diffs (+ Holm p)             Top-20% Hit Rate, class prevalence
    • H3: N=23 zone OLS(HC3) + spreg SEM & SLM (side-by-side)
    • MAUP sensitivity: H3 res-8 vs res-9
          │                                                          │
          └────────────────────────────┬─────────────────────────────┘
                                       ▼
  [M4: Contract Freeze, Publication Viz, Unified Replay CLI & CS482 Report Draft]
    • Freeze final cells.gpkg (base + 2SFCA + LISA + target + fold) + cells_res8.gpkg + FREEZE_MANIFEST.json
    • Programmatic 300-DPI figures (EDA, Lorenz, Fig 1 dual-panel, MAUP) & Table 1 (zero hand-typed numbers)
    • Compile CS482 A4 PDF Phase A draft (3.5–3.8p main text, 11pt, 1.5 spacing + cover/refs/contrib/AI decl)
    • Numeric audit & 12/03 Go/No-Go report
                                       │
                                       ▼
  [M5: Final Milestone — Tiers 1–4 E2E Verification (against E2E Track TEST_READY.md) + Tier 5 Hardening]
```

### Verified Critical Data & Engineering Safeguards (From Step 0 Survey)
1. **DSEC CRS Trap (`wkid: 3064` → `EPSG:8433`)**: DSEC ArcGIS 10.03 silently breaks `outSR=4326` (Gulf of Guinea / untransformed meters). ALWAYS fetch native coordinates without `outSR`, assign `EPSG:8433` (`Macao 1920 / Macao Grid`, `always_xy=True`), transform to `EPSG:32649` (analysis) and `EPSG:4326` (H3), parse Esri multi-ring winding order (CW=outer, CCW=hole), and assert landmark alignment `< 50 m` (e.g., `逸景樓` / `湖畔大廈` vs OSM/CEM).
2. **DSEC Building-Level Census Exact Totals**: `c2021_Building_P5` / `Census2021_Population_TW_v2` (5,682 polygons: 3,733 populated terrestrial + 1 maritime `OBJECTID=1` pop 777/61 + 1,948 zero-pop) sums to **exact 2021 Census totals: `682,070` total / `82,812` aged 65+** (terrestrial: **`681,293` / `82,751`**). Aggregate building-level census counts by area-weighted intersection into H3 r9/r8 cells (conserving exact terrestrial totals within `< 0.05%`), eliminating uniform-height dasymetric distortion while explicitly documenting 2021-vs-2026 temporal mismatch and building-group caveats.
3. **CEM EV API Traps**: `https://ev.cem-macau.com/Ajax/GetMapData?lang=en` returns 84 public stations where `stationlongitude` holds **LATITUDE** (`~22.1–22.2`) and `stationlatitude` holds **LONGITUDE** (`~113.5x`) on all 84/84 records — MUST swap and assert Macao bounds. Fast chargers list 3 connector heads per physical charger in `pilerecords` — MUST use `capacity = nunique(chargerid)` (fallback to `pilenum` when `pilerecords` is empty, e.g., station `9488798`), while preserving `pilenum` and `total_kw` for sensitivity checks. Document 4 newly reclaimed Zona A stations explicitly.
4. **Overpass Outage & Disconnected Walking Network**: Default `overpass-api.de` returns 502/406. Use deterministic Geofabrik `macau-latest.osm.pbf` (2.8 MB) + `macau-latest-free.gpkg.zip` (3.8 MB) and/or `overpass.openstreetmap.fr`. Build OSMnx pedestrian graph with **`retain_all=True`** (otherwise non-walkable cross-harbor bridges cause default largest-component filtering to drop Taipa & Coloane!), prune tiny disconnected artifacts (<10 nodes) while retaining BOTH Peninsula and Taipa–Cotai–Coloane main components, and add centroid-to-nearest-node snap walk time (at 4.5 km/h = 75 m/min) so `walk_time_min > 0` strictly for all distinct cell pairs.
5. **7 Official Parishes Partition**: OSM Geofabrik `admin_level6` misses Taipa (`嘉模堂區`). Derive the 7 parishes by dissolving DSEC's 23 terrestrial `Zona` polygons (`Z01–05` Fátima, `Z06–10` Santo António, `Z11–12` São Lázaro, `Z13–15` Sé, `Z16–17` São Lourenço, `Z18–22` Nossa Senhora do Carmo / Taipa, `Z23` São Francisco Xavier / Coloane+Cotai), verified against OSM parish relations — ensuring 100% consistency between H1 parish Theil decomposition and H3 zone regression.
6. **Facility Whitelisting & Cross-Source-Only 25 m Dedup**:
   - Health centres: strictly official SSM terrestrial primary health centres/stations (11 verified against `gov.mo`), excluding private clinics.
   - Public libraries: official IC public libraries (~15–16 verified against `library.gov.mo`), excluding university/school/museum libraries (MUST, UM, MAM).
   - Parks/sitting-out areas: public `park`/`garden`/`playground`/`recreation_ground` clipped to Macao SAR, excluding `access=private`, keeping polygon area (`m²`) for capacity weighting, deduplicating `pois`/`pois_a`/`landuse_a`.
   - Bus stops: cross-source 25 m dedup only — NEVER blindly merge opposite-side official bus stops on narrow streets!
   - LRT stations (covariate): operational MLM stations clipped to Macao SAR, excluding Mainland Zhuhai CRH stations and under-construction East Line, deduplicating `Lotus`/`蓮花`.
7. **Statistical & Anti-Leakage Methodological Guardrails (`00_review_report.md` C1–C15)**:
   - Use **Theil's T (GE(1))** (convention $0 \ln 0 = 0$) so cells/parishes with $A_i = 0$ do not diverge; assert exact `Theil_total == Theil_within + Theil_between` within `1e-9`.
   - **No pretest switching** on H3 ($N=23$ zones): pre-specify and report OLS (HC3 SE), `spreg.ML_Error`, and `spreg.ML_Lag` side-by-side with connected spatial weights $W$ (primary: row-standardized `k=4` NN; sensitivity: bridge-augmented Queen), coefficients, 95% CIs, VIF, and residual Moran's I.
   - **Zero Target Leakage**: `feature_manifest.json` whitelists strictly non-EV features for ML/GNN; `cell_poi_texts.csv` strips all EV/charging/充電/充电/Tesla/CEM-EV tokens; sklearn `Pipeline` fits imputation/scaling strictly inside training folds; all 5 spatial blocks MUST contain both `EV=1` and `EV=0` cells.
   - **Package Compatibility**: Verify `pandas` × PySAL (`spreg`/`libpysal`/`esda`/`inequality`) immediately in M1; pin `pandas>=2.2,<3.0` if `pandas 3.x` triggers PySAL incompatibilities. Use `h3` v4 API and `osmnx` v2 API.

---

## Code Layout

All project code, data, outputs, report, and tests live at `/home/antigravity/Lc-wing/data_science_project` (NEVER place source code, data, or tests inside `.agents/teamwork/`):

```text
/home/antigravity/Lc-wing/data_science_project/
├── .gitignore
├── pyproject.toml                   # uv + Python 3.11+, ruff (line-length=100), pytest, MPLBACKEND=Agg
├── uv.lock
├── PROJECT.md                       # Living top-level architecture & milestones (Orchestrator owned)
├── TEST_INFRA.md                    # E2E test infrastructure & feature coverage plan (E2E Track owned)
├── TEST_READY.md                    # E2E suite readiness signal & runner command (E2E Track owned)
├── README.md                        # Project overview & reproducible Phase A CLI instructions
├── plans/                           # Existing read-only planning blueprints
├── data/
│   ├── raw/                         # Timestamped raw snapshots + data_provenance.json (URLs, dates, SHA-256)
│   ├── interim/                     # Intermediate EPSG:32649 layers & analysis joins
│   └── frozen/                      # FROZEN Phase A -> Phase B immutable contract bundle
│       ├── cells_base_res9.gpkg     # M1 base H3 r9 populated cells + leak-free features + target + fold
│       ├── cells_base_res8.gpkg     # M1 base H3 r8 populated cells (MAUP)
│       ├── cells.gpkg               # M4 canonical frozen H3 r9 (M1 base + M2 2SFCA/LISA)
│       ├── cells_res8.gpkg          # M4 canonical frozen H3 r8 (M1 base + M2 2SFCA)
│       ├── zones_23.gpkg            # 23 terrestrial DSEC zones (EPSG:32649)
│       ├── parishes_7.gpkg          # 7 official parishes dissolved from zones_23 (EPSG:32649)
│       ├── facilities_clean.gpkg    # 5 cleaned deduplicated facility layers (EPSG:32649)
│       ├── lrt_stations.gpkg        # Operational MLM LRT stations (EPSG:32649)
│       ├── graph_edges.csv          # 10-min pedestrian edges (src_cell_id, dst_cell_id, walk_time_min, weight, same_block)
│       ├── spatial_folds.csv        # Frozen 5 geographic CV blocks (cell_id, spatial_block)
│       ├── split_metadata.json      # Clustering method, seed, per-block n_cells / n_pos_ev / n_neg_ev
│       ├── cell_poi_texts.csv       # EV-stripped OSM POI text per cell (Phase B Route B ready)
│       ├── feature_manifest.json    # Explicit whitelist of leak-free ML/GNN features vs excluded/target cols
│       ├── FREEZE_MANIFEST.json     # M4 SHA-256 checksums of all frozen Phase A->B deliverables
│       └── data_dictionary.md       # Complete schema & anti-leakage specification
├── src/
│   └── macao_equity/
│       ├── __init__.py              # M1
│       ├── config.py                # M1: CRS, seeds, walk speed (4.5 km/h), cutoffs (10/15m), paths, parish map
│       ├── ingest/                  # M1: dsec.py, cem_ev.py, osm.py
│       ├── preprocess/              # M1: clean.py, h3_grid.py, graph_and_folds.py
│       ├── analysis/                # M2: sfca.py, inequality.py, spatial_stats.py, regression.py, maup.py
│       ├── ml/                      # M3: baselines.py (exclusive M3 namespace for safe parallel execution with M2)
│       ├── freeze_bundle.py         # M4: canonical cells.gpkg merge & SHA-256 FREEZE_MANIFEST.json
│       └── viz/                     # M4: figures.py, tables.py
├── scripts/
│   ├── 01_ingest_and_prepare.py     # M1 CLI
│   ├── 02_run_spatial_analysis.py   # M2 CLI
│   ├── 03_run_ml_baselines.py       # M3 CLI
│   ├── 04_freeze_and_report.py      # M4 CLI
│   └── run_phase_a.py               # M4 Unified deterministic end-to-end CLI (--skip-download supported)
├── outputs/
│   ├── tables/                      # M2/M3/M4 verified CSV/JSON/MD/TEX statistical & ML outputs
│   ├── metrics/                     # M1/M3 data_provenance.json, phase_a_ml_baselines.json
│   └── figures/                     # M4 300-DPI PNG/PDF figures (EDA, Lorenz, Fig 1 dual-panel, MAUP)
├── report/
│   ├── phase_a_draft.tex            # M4 CS482 LaTeX source (A4, single-col, 11pt, 1.5 spacing)
│   ├── references.bib               # M4 BibTeX references (incl. K-Dense citation)
│   ├── CS482_GroupX_MacaoSpatialEquity_PhaseA_Draft.pdf  # M4 compiled PDF (3.5–3.8p main text)
│   └── go_nogo_audit.md             # M4 numeric traceability & 12/03 Go/No-Go checklist audit
└── tests/
    ├── test_m1_ingest_crs_graph.py  # M1 unit/integration tests
    ├── test_m2_sfca_stats_maup.py   # M2 unit/integration tests
    ├── test_m3_ml_leakage.py        # M3 unit/integration tests
    ├── test_m4_contract_and_report.py # M4 contract & report verification tests
    └── e2e/                         # E2E Testing Track (opaque-box Tiers 1–4 + Tier 5 hardening)
```

---

## Feature Inventory

Every single requirement from Step 0 Survey (`spec_miner_survey_1`, `explorer_survey_1`, `explorer_survey_2`) is assigned a milestone (ZERO unassigned):

| # | Feature | Description | Milestone | Source |
|---|---|---|---|---|
| A01 | Geospatial Python Environment | Reproducible `uv` Python 3.11+ `.venv`, `pyproject.toml`, `ruff` (100 chars), `pytest`, `MPLBACKEND=Agg`, `.gitignore`, verified binary wheels & PySAL/pandas compatibility | M1 | survey (PR #1, plan5 §2) |
| A02 | Day-1 Live Data Acquisition & Provenance | Fetch & snapshot DSEC 2021 REST (24 zones + 5,682 buildings), CEM EV JSON (84 stations), Geofabrik Macau OSM + `overpass.openstreetmap.fr`, official registry cross-checks; log `data_provenance.json` (URLs, UTC+8 dates, SHA-256, `data.gov.mo` login-gate note) | M1 | survey (plan5 §2.1, §9) |
| A03 | CRS Standardization & Cross-Source 25m Dedup | Local `EPSG:8433` (`always_xy=True`) → `EPSG:32649` for DSEC + Esri ring winding fix + landmark alignment `<50 m`; CEM swapped lat/lon fix & `nunique(chargerid)` capacity (`pilenum` fallback for `9488798`); official SSM/IC/IAM/MLM whitelisting; cross-source-only 25 m dedup | M1 | survey (plan5 §2.2–2.3, traps 1–10) |
| A04 | Spatial Units & Exact Building Census Allocation | `zones_23.gpkg` (23 terrestrial zones, excluding `Z26` maritime), `parishes_7.gpkg` (7 official parishes dissolved from zones), populated H3 res-9 (~250–400 cells) & H3 res-8 grids; area-weighted DSEC building-level census aggregation conserving exact `681,293` total / `82,751` aged 65+ (<0.05% drift); drop unpopulated cells; document UM Hengqin & Zona A | M1 | survey (plan5 §2.1–2.3, §5) |
| A05 | Walkable Network, Leak-Free Features & EV-Stripped POI Text | OSMnx walk graph (`retain_all=True`, Peninsula + Taipa–Cotai–Coloane main components, ≥95% cell snap <300 m, landmark walk-time sanity check); strictly non-EV cell covariates (`feature_manifest.json`); EV-stripped `cell_poi_texts.csv` + regex audit log for Phase B Route B | M1 | survey (plan5 §2.1–2.2, Phase B, §9) |
| A06 | 10-Min Pedestrian Graph & Frozen 5 Spatial Blocks | `graph_edges.csv` (undirected pairs with `0 < walk_time_min <= 10.0` incl. snap time, `weight = 1/walk_time_min`, `same_block`); deterministic 5 geographic CV blocks (`spatial_folds.csv`, `split_metadata.json`, seed=42) verified `n_pos >= 1` and `n_neg >= 1` in all 5 folds | M1 | survey (plan5 §2.3, §4, Phase B) |
| A07 | Gaussian 2SFCA Accessibility (10 & 15 min) | Network Gaussian 2SFCA for all 5 facility types (capacity-weighted park area & EV chargers + unweighted EV sensitivity) at 10-min & 15-min cutoffs on H3 r9, H3 r8, and DSEC 23 zones; composite public-facility index; zero-demand guard (`R_j=0`) & supply conservation unit tests | M2 | survey (plan5 §2.2, §5, §6) |
| A08 | Exploratory Data Analysis (EDA & Lorenz) | EDA summary statistics & Lorenz curve data across 5 facility types + population density (M2), rendered into publication-grade EDA maps & Lorenz curve figures (M4) | M2, M4 | survey (plan5 §3; CS482 §2.3) |
| A09 | H1 Confirmatory Inequality & LISA Inference | Pop-weighted Gini + 2,000-rep bootstrap 95% CI; Theil's T (total, within-parish, between-parish, exact additivity, $0\ln 0=0$) + bootstrap 95% CI; Global Moran's I & LISA (999 perms, Benjamini-Hochberg FDR) with documented island-safe spatial weights $W$ | M2 | survey (plan5 §1, §2.2, §4) |
| A10 | H2 Confirmatory Paired Bootstrap Gini Diffs | Paired bootstrap (2,000 reps, identical resampled H3 cells, fixed seed) 95% CIs + Holm-adjusted p-values for `Gini_EV − Gini_{Park,Health,Library,Bus}` (10 & 15 min) | M2 | survey (plan5 §1, §4; review C7/C15) |
| A11 | H3 Confirmatory Zone Spatial Regression ($N=23$) | Pre-specified side-by-side OLS (HC3 SE), `spreg.ML_Error`, `spreg.ML_Lag` at DSEC 23-zone level (`elderly_share` + `log_pop_density`, connected `k=4` NN `W` + bridge-Queen sensitivity), reporting coefs, 95% CIs, VIF, fit stats, residual Moran's I (zero pretest switching) | M2 | survey (plan5 §1, §2.2, §4; review C3/C15) |
| A12 | MAUP Sensitivity Analysis (H3 r8 vs r9) | Compare H3 res-8 vs H3 res-9 Gini, Theil decomposition, and Global Moran's I under identical population & facility inputs; tabulate & quantify scale sensitivity | M2 | survey (plan5 §5) |
| A13 | Phase A Leak-Free ML Baselines (Spatial Block CV) | Binary classification of `has_ev_charger` on H3 r9 using strictly M1 non-EV features & frozen 5 spatial blocks: `DummyClassifier`, Logistic Regression, Random Forest (`Pipeline` fit strictly on train folds; inner CV tuning); report prevalence, ROC-AUC, PR-AUC, Brier, Top-20% Hit Rate | M3 | survey (plan5 §0, §4, §5, Phase B; review C6) |
| A14 | Canonical Phase A→B Frozen Bundle & Replay CLI | Merge M1+M2 into canonical frozen `cells.gpkg` & `cells_res8.gpkg`; generate `FREEZE_MANIFEST.json` (SHA-256) & `data_dictionary.md`; unified `scripts/run_phase_a.py --skip-download` & `README.md` | M4 | survey (plan5 §2.3, Phase B, §9) |
| A15 | Programmatic Figure 1 & Table 1 Generation | High-DPI (≥300 DPI) **Figure 1** (dual-panel: Composite accessibility choropleth + FDR LISA cluster map), EDA & MAUP figures, and **Table 1** (Gini/Theil + H3 regression + ML CV) generated 100% programmatically from verified CSV/JSON | M4 | survey (plan5 §6; CS482 §3) |
| A16 | CS482 Phase A Report Draft & 12/03 Go/No-Go Audit | Compile `report/phase_a_draft.tex` → `CS482_GroupX_MacaoSpatialEquity_PhaseA_Draft.pdf` (3.5–3.8 A4 pages main text, single-col, 11pt, 1.5 spacing; all 5 Minimum Requirements; correlational framing; all 4 limitations; Cover/Refs w/ K-Dense/Contribution/AI Declaration); automated numeric audit in `report/go_nogo_audit.md` | M4 | survey (CS482 §2–5; plan5 §6–9; review C1–C15) |
| A17 | Opaque-Box E2E Test Suite (Tiers 1–4) | Independent requirement-driven E2E test infrastructure (`TEST_INFRA.md`), Tiers 1–4 test suites (`tests/e2e/`), and `TEST_READY.md` covering all A01–A16 user requirements | E2E Track | Project Pattern Dual Track |
| A18 | Final Milestone: E2E Pass + Tier 5 Hardening | Pass 100% of Tiers 1–4 E2E tests sequentially, then run Tier 5 white-box adversarial coverage hardening (Challenger-initiated loop) + whole-project Forensic Audit | M5 | Project Pattern Final Milestone |

---

## Milestones

| # | Name | Scope | Dependencies | Owner / Sub-Orch | Status |
|---|---|---|---|---|---|
| E2E | E2E Testing Track | `TEST_INFRA.md`, opaque-box `tests/e2e/` (Tiers 1–4), `TEST_READY.md` | none | `e2e_orch_1_gen1` (`3bc21da4e7eae3025e566998dc714177`) | IN_PROGRESS |
| M1 | Env, Data Ingestion, CRS Cleaning & Frozen Graph/Splits | A01–A06 | none | `sub_orch_m1_gen1` (`7654a19bedae570012b1eac979ae2379`) | IN_PROGRESS |
| M2 | Gaussian 2SFCA, EDA, H1/H2/H3 Spatial Stats & MAUP | A07, A08 (data), A09–A12 | M1 | `sub_orch_m2` | PLANNED |
| M3 | Leak-Free Spatial-Block CV ML Baselines | A13 | M1 (parallel w/ M2) | `sub_orch_m3` | PLANNED |
| M4 | Contract Freeze, Fig 1 / Table 1, CLI & CS482 Report Draft | A08 (figs), A14–A16 | M2, M3 | `sub_orch_m4` | PLANNED |
| M5 | Final Milestone: E2E Tiers 1–4 Pass + Tier 5 Hardening | A18 | M4, E2E (`TEST_READY.md`) | `sub_orch_m5` | PLANNED |

---

## Interface Contracts

### M1 ↔ M2, M3, M4 (Frozen Base Spatial & Feature Contract)
- **CRS**: All `.gpkg` files strictly **`EPSG:32649`** (`crs.to_epsg() == 32649`), valid non-empty geometries.
- **`data/frozen/cells_base_res9.gpkg`** (H3 resolution 9, ~250–400 populated terrestrial cells):
  - Required columns:
    - Identity & Admin: `cell_id` (str, unique PK, H3 v4 res-9 hex ID), `geometry` (Polygon, EPSG:32649), `centroid_x` (float), `centroid_y` (float), `area_km2` (float), `zone_id` (str, e.g. `Z01`–`Z23`), `zone_name` (str), `parish_id` (str, `P01`–`P07`), `parish_name` (str).
    - Leak-Free Features (EXACT list also recorded in `data/frozen/feature_manifest.json` under `"ml_features"`): `pop_total` (float, >0), `pop_65plus` (float, >=0), `pop_density` (float, persons/km²), `elderly_share` (float, in `[0, 1]`), `intersection_density` (float, nodes/km²), `road_density_km_per_km2` (float), `res_bldg_share` (float, in `[0, 1]`), `com_bldg_share` (float, in `[0, 1]`), `bldg_coverage_ratio` (float), `dist_nearest_lrt_m` (float, >=0), `dist_nearest_bus_m` (float, >=0) (zero NaNs/Infs in ML features!).
    - Target & Audit Columns (EXACT list recorded in `feature_manifest.json` under `"excluded_target_and_audit"`): `has_ev_charger` (int, `{0, 1}`), `ev_station_count` (int, >=0), `ev_charger_capacity` (int, >=0, sum of unique `chargerid` with `pilenum` fallback).
    - CV Split: `spatial_block` (int, `0..4`, matching `spatial_folds.csv` 1-to-1).
- **`data/frozen/cells_base_res8.gpkg`** (H3 resolution 8, populated terrestrial cells for MAUP):
  - Required columns: `cell_id`, `geometry`, `area_km2`, `zone_id`, `parish_id`, `parish_name`, `pop_total`, `pop_65plus`, `pop_density`, `elderly_share`, `has_ev_charger`. Sum of `pop_total` equals `cells_base_res9.gpkg` sum (`681,293` ± 0.05%).
- **`data/frozen/zones_23.gpkg`** (23 terrestrial DSEC statistical zones, excluding `Z26` maritime):
  - Required columns: `zone_id`, `zone_name`, `parish_id`, `parish_name`, `area_km2`, `pop_total`, `pop_65plus`, `pop_density`, `log_pop_density`, `elderly_share`, `geometry`.
- **`data/frozen/parishes_7.gpkg`** (7 official parishes dissolved from `zones_23.gpkg`):
  - Required columns: `parish_id`, `parish_name`, `parish_name_zh`, `area_km2`, `pop_total`, `pop_65plus`, `geometry`.
- **`data/frozen/facilities_clean.gpkg`**:
  - Layers (or single table with `facility_type` enum ∈ `{park, health_centre, library, bus_stop, ev_charger}` — prefer single canonical layer `facilities` with column `facility_type` PLUS per-type attributes so downstream access is uniform):
  - Required columns: `facility_id` (unique str), `facility_type` (str), `name` (str), `source` (str), `capacity` (float, >0: park area in m² [capped/documented fallback if point], EV unique `chargerid` count [fallback `pilenum`], 1.0 for health_centre/library/bus_stop), `capacity_unit` (str), `capacity_imputed` (bool), `pilenum` (nullable float, EV only), `total_kw` (nullable float, EV only), `in_zona_a` (bool), `geometry` (Point in EPSG:32649).
- **`data/raw/osm_walk_graph.graphml`**:
  - Directed/undirected OSMnx pedestrian network containing both Macao Peninsula and Taipa–Cotai–Coloane components, edge `length` in meters, walking speed constant `4.5 km/h` (`75.0 m/min`) in `config.py`.
- **`data/frozen/graph_edges.csv`**:
  - Columns: `src_cell_id,dst_cell_id,walk_time_min,weight,same_block`.
  - Invariants: `src_cell_id != dst_cell_id`, `0 < walk_time_min <= 10.0`, `weight == 1.0 / walk_time_min` (finite), symmetric undirected pairs `(u,v)` and `(v,u)` present, every ID exists in `cells_base_res9.gpkg`.
- **`data/frozen/spatial_folds.csv`** & **`data/frozen/split_metadata.json`**:
  - Columns: `cell_id,spatial_block` (`0..4`), exact 1-to-1 match with `cells_base_res9.gpkg`.
  - Invariant: every block `0..4` has `n_pos_ev >= 1` and `n_neg_ev >= 1`.
- **`data/frozen/cell_poi_texts.csv`**:
  - Columns: `cell_id,poi_count_raw,poi_count_filtered,poi_text`.
  - Invariant: ZERO substrings matching EV/charging regex (`(?i)\b(ev|evse|supercharger|tesla)\b|charg|充電|充电|電動車|电动车`).
- **`data/frozen/feature_manifest.json`**:
  - Keys: `ml_features` (list[str]), `excluded_target_and_audit` (list[str]), `analysis_prefix_excluded` (`["acc10_", "acc15_", "lisa_"]`), `random_seed` (`42`).

### M2 ↔ M4 (Spatial Analytics & Econometrics Contract)
- **`data/interim/cells_r9_with_2sfca_lisa.gpkg`** & **`data/interim/cells_r8_with_2sfca.gpkg`** & **`data/interim/zones_23_with_2sfca.gpkg`**:
  - Preserves all M1 base columns verbatim + appends namespaced analysis columns:
    - `acc10_{park,health_centre,library,bus_stop,ev_charger,composite}` and `acc15_{park,health_centre,library,bus_stop,ev_charger,composite}` (float, `>= 0`, finite), plus unweighted EV sensitivity `acc10_ev_unweighted`, `acc15_ev_unweighted`.
    - LISA columns (for 15-min primary & 10-min, per facility + composite): `lisa_i_{metric}`, `lisa_p_{metric}`, `lisa_q_fdr_{metric}`, `lisa_cluster_{metric}` ∈ `{HH, LL, LH, HL, NS}`.
- **`outputs/tables/h1_inequality_lisa.csv`**:
  - Columns: `resolution,threshold_min,facility_type,gini,gini_ci_low,gini_ci_high,theil_total,theil_within,theil_between,theil_total_ci_low,theil_total_ci_high,theil_between_share,moran_i,moran_p,n_ll_fdr,n_hh_fdr,mean_acc,median_acc,zero_acc_pct`.
- **`outputs/tables/h2_gini_diff_bootstrap.csv`**:
  - Columns: `threshold_min,comparison,gini_ev,gini_other,gini_diff,ci_low,ci_high,p_boot_raw,p_holm,n_boot,seed`.
- **`outputs/tables/h3_spatial_regression.csv`** & **`outputs/tables/h3_regression_diagnostics.json`**:
  - Columns: `threshold_min,outcome,weights_spec,model_type,term,coef,se,ci_low,ci_high,p_value,r2_or_pseudo_r2,aic,resid_moran_i,resid_moran_p,n_obs`.
  - Required `model_type` values reported side-by-side for identical `outcome`/`n_obs=23`: `OLS_HC3`, `SEM_ML`, `SLM_ML`.
- **`outputs/tables/maup_r8_vs_r9.csv`**, **`outputs/tables/eda_summary.csv`**, **`outputs/tables/lorenz_curves.csv`**.

### M3 ↔ M4 (ML Baselines Contract)
- **`outputs/metrics/phase_a_ml_baselines.json`**:
  - Structure:
    ```json
    {
      "seed": 42,
      "n_cells": 300,
      "n_pos": 60,
      "prevalence": 0.2,
      "feature_columns": ["..."],
      "models": {
        "naive_prior": { "fold_metrics": [...], "mean_roc_auc": ..., "std_roc_auc": ..., "mean_pr_auc": ..., "std_pr_auc": ..., "mean_brier": ..., "std_brier": ..., "mean_top20_hit_rate": ..., "std_top20_hit_rate": ..., "oof_roc_auc": ..., "oof_pr_auc": ..., "oof_brier": ..., "oof_top20_hit_rate": ... },
        "logistic_regression": { ... },
        "random_forest": { ... }
      }
    }
    ```
- **`outputs/tables/ml_cv_results.csv`** & **`outputs/tables/ml_oof_predictions.csv`**.

### M4 ↔ M5 & Phase B (Final Frozen Bundle & Report Contract)
- **`data/frozen/cells.gpkg`** (canonical merged H3 r9 file matching M1+M2 schema) + **`data/frozen/cells_res8.gpkg`** + **`data/frozen/FREEZE_MANIFEST.json`** (SHA-256 checksums for all frozen files) + **`data/frozen/data_dictionary.md`**.
- **`scripts/run_phase_a.py`** (`--skip-download` reproduces all frozen analysis/tables/figures cleanly with exit code 0).
- **`outputs/figures/fig1_accessibility_lisa.png`**, **`outputs/figures/eda_facilities_pop.png`**, **`outputs/figures/eda_accessibility_lorenz.png`**, **`outputs/figures/maup_r8_vs_r9.png`** (≥300 DPI, non-empty).
- **`outputs/tables/table1_summary.{csv,md,tex}`** (100% generated programmatically from M2/M3 outputs).
- **`report/CS482_GroupX_MacaoSpatialEquity_PhaseA_Draft.pdf`**:
  - Main text strictly **3.0–3.8 A4 pages** (single-column, 11pt, 1.5 spacing; Cover Page, References, Contribution Statement, AI Declaration excluded from main-text page count).
  - **`report/go_nogo_audit.md`**: Automated verification that all 5 CS482 Minimum Requirements are met, every figure/table/number matches CSV/JSON outputs, and the 12/03 Go/No-Go checkpoint passes.
