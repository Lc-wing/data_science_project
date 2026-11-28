"""Realistic Macao-scale valid synthetic project builder + 18 atomic fault mutators."""

from __future__ import annotations

import csv
import json
import math
import random
from collections.abc import Callable
from pathlib import Path
from typing import Any

from e2e_lib.contracts import (
    FACILITY_TYPES,
    FROZEN_CORE_HASHED_FILES,
    GLOBAL_SEED,
    ML_FEATURES_REQUIRED,
    ML_METRICS,
    REPORT_AUDIT_REL,
    REPORT_BIB_REL,
    REPORT_PDF_REL,
    REPORT_TEX_REL,
    TARGET_EPSG,
    TERRESTRIAL_POP_65PLUS,
    TERRESTRIAL_POP_TOTAL,
    TRADITIONAL_FACILITY_TYPES,
)
from e2e_lib.oracle import (
    average_precision_score,
    bh_fdr,
    brier_score,
    fit_predict_ridge_prob,
    holm_bonferroni,
    ols_hc3_and_vif,
    roc_auc_score,
    top20_hit_rates,
    weighted_gini,
    weighted_theil_t,
)
from e2e_lib.readers import (
    file_sha256,
    read_csv_strict,
    read_gpkg,
    read_json_strict,
    write_gpkg,
    write_pdf,
    write_png,
)

# Official 7 parishes mapped to 23 terrestrial DSEC zones inside real Macao UTM 49N bounds
_PARISH_SPECS: tuple[dict[str, Any], ...] = (
    {
        "parish_id": "P01",
        "parish_name": "Nossa Senhora de Fatima",
        "parish_name_zh": "花地瑪堂區",
        "zones": ["Z01", "Z02", "Z03", "Z04", "Z05"],
        "bbox": (761_000.0, 2_458_000.0, 764_200.0, 2_459_500.0),
        "landmass": "peninsula",
    },
    {
        "parish_id": "P02",
        "parish_name": "Santo Antonio",
        "parish_name_zh": "花王堂區",
        "zones": ["Z06", "Z07", "Z08", "Z09", "Z10"],
        "bbox": (761_000.0, 2_456_800.0, 763_500.0, 2_458_000.0),
        "landmass": "peninsula",
    },
    {
        "parish_id": "P03",
        "parish_name": "Sao Lazaro",
        "parish_name_zh": "望德堂區",
        "zones": ["Z11", "Z12"],
        "bbox": (763_500.0, 2_456_800.0, 764_600.0, 2_458_000.0),
        "landmass": "peninsula",
    },
    {
        "parish_id": "P04",
        "parish_name": "Se",
        "parish_name_zh": "大堂區",
        "zones": ["Z13", "Z14", "Z15"],
        "bbox": (762_600.0, 2_455_400.0, 764_600.0, 2_456_800.0),
        "landmass": "peninsula",
    },
    {
        "parish_id": "P05",
        "parish_name": "Sao Lourenco",
        "parish_name_zh": "風順堂區",
        "zones": ["Z16", "Z17"],
        "bbox": (761_000.0, 2_455_400.0, 762_600.0, 2_456_800.0),
        "landmass": "peninsula",
    },
    {
        "parish_id": "P06",
        "parish_name": "Nossa Senhora do Carmo",
        "parish_name_zh": "嘉模堂區",
        "zones": ["Z18", "Z19", "Z20", "Z21", "Z22"],
        "bbox": (761_500.0, 2_451_000.0, 765_500.0, 2_453_200.0),
        "landmass": "islands",
    },
    {
        "parish_id": "P07",
        "parish_name": "Sao Francisco Xavier",
        "parish_name_zh": "聖方濟各堂區",
        "zones": ["Z23"],
        "bbox": (761_500.0, 2_448_000.0, 765_500.0, 2_450_500.0),
        "landmass": "islands",
    },
)

_ZONE_TO_BLOCK: dict[str, int] = {
    "Z01": 0,
    "Z02": 0,
    "Z03": 0,
    "Z04": 0,
    "Z05": 0,
    "Z06": 1,
    "Z07": 1,
    "Z08": 1,
    "Z09": 1,
    "Z10": 1,
    "Z11": 2,
    "Z12": 2,
    "Z13": 2,
    "Z14": 2,
    "Z15": 3,
    "Z16": 3,
    "Z17": 3,
    "Z18": 4,
    "Z19": 4,
    "Z20": 4,
    "Z21": 4,
    "Z22": 4,
    "Z23": 4,
}

_POI_LEXICON: tuple[str, ...] = (
    "bakery cafe noodle tea house",
    "supermarket grocery pharmacy clinic",
    "community park garden playground",
    "post office municipal market bank",
    "public library cultural center temple",
    "restaurant dim sum bakery convenience store",
    "sports ground primary school kindergarten",
    "heritage plaza museum pedestrian promenade",
    "residential tower shopping arcade bookstore",
    "bus terminal ferry pier waterfront garden",
)


def _rect_ring(x0: float, y0: float, x1: float, y1: float) -> list[tuple[float, float]]:
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]


def _hex_ring(cx: float, cy: float, radius: float) -> list[tuple[float, float]]:
    pts = [
        (cx + radius * math.cos(math.pi / 3.0 * k), cy + radius * math.sin(math.pi / 3.0 * k))
        for k in range(6)
    ]
    pts.append(pts[0])
    return pts


def _gaussian_weight(dist_min: float, cutoff_min: float) -> float:
    if dist_min >= cutoff_min:
        return 0.0
    num = math.exp(-0.5 * (dist_min / cutoff_min) ** 2) - math.exp(-0.5)
    den = 1.0 - math.exp(-0.5)
    return max(0.0, num / den)


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def _compute_2sfca(
    demands: list[dict[str, Any]],
    facilities: list[dict[str, Any]],
    cutoff_min: float,
    use_unit_capacity: bool = False,
) -> list[float]:
    """Run genuine Dai (2011) Gaussian 2SFCA respecting harbour separation."""
    n_dem = len(demands)
    n_fac = len(facilities)
    # Precompute walkable pairs (same landmass only; no cross-sea teleport)
    fac_neighbors: list[list[tuple[int, float]]] = [[] for _ in range(n_fac)]
    for j, fac in enumerate(facilities):
        fx, fy = fac["x"], fac["y"]
        f_land = fac["landmass"]
        for i, dem in enumerate(demands):
            if dem["landmass"] != f_land:
                continue
            dist_m = math.hypot(dem["x"] - fx, dem["y"] - fy)
            walk_min = (dist_m * 1.15) / 75.0
            if walk_min < cutoff_min:
                w = _gaussian_weight(walk_min, cutoff_min)
                if w > 0.0:
                    fac_neighbors[j].append((i, w))

    # Step 1: supply-to-demand ratio R_j
    ratios = [0.0] * n_fac
    for j, fac in enumerate(facilities):
        denom = sum(demands[i]["pop_total"] * w for i, w in fac_neighbors[j])
        if denom > 0.0:
            cap = 1.0 if use_unit_capacity else float(fac["capacity"])
            ratios[j] = cap / denom

    # Step 2: cell accessibility A_i
    acc = [0.0] * n_dem
    for j, nbrs in enumerate(fac_neighbors):
        rj = ratios[j]
        if rj <= 0.0:
            continue
        for i, w in nbrs:
            acc[i] += rj * w
    return acc


def _compute_synthetic_lisa(
    values: list[float],
    adj: dict[int, list[int]],
) -> tuple[list[float], list[float], list[float], list[str], float, float]:
    """Compute exact standardized spatial lag, local I_i, global Moran's I, BH-FDR q, clusters."""
    n = len(values)
    mean_v = sum(values) / n
    var_v = sum((v - mean_v) ** 2 for v in values) / n
    std_v = math.sqrt(var_v) if var_v > 1e-15 else 1.0
    z = [(v - mean_v) / std_v for v in values]

    wz = []
    for i in range(n):
        nbrs = adj.get(i, [])
        if nbrs:
            wz.append(sum(z[j] for j in nbrs) / len(nbrs))
        else:
            wz.append(0.0)

    local_i = [z[i] * wz[i] for i in range(n)]
    global_i = sum(local_i) / n
    global_i = max(-1.0, min(1.0, global_i))
    global_p = 0.001 if abs(global_i) > 0.15 else 0.042

    # Deterministic realistic permutation pseudo-p values (multiples of 1/1000 in [0.001, 1.0])
    pvals: list[float] = []
    for i in range(n):
        mag = abs(local_i[i])
        if mag > 1.35:
            p = 0.001
        elif mag > 0.95:
            p = 0.003
        elif mag > 0.65:
            p = 0.012
        elif mag > 0.40:
            p = 0.045
        else:
            p = min(0.999, max(0.08, 0.90 - 0.8 * mag))
        pvals.append(round(p, 6))

    qvals = bh_fdr(pvals)
    clusters: list[str] = []
    for i in range(n):
        if qvals[i] <= 0.05:
            if z[i] >= 0.0 and wz[i] >= 0.0:
                clusters.append("HH")
            elif z[i] < 0.0 and wz[i] < 0.0:
                clusters.append("LL")
            elif z[i] >= 0.0 and wz[i] < 0.0:
                clusters.append("HL")
            else:
                clusters.append("LH")
        else:
            clusters.append("NS")
    return local_i, pvals, qvals, clusters, global_i, global_p


def _composite_index(acc_by_type: dict[str, list[float]]) -> list[float]:
    """Compute non-negative min-max normalized average across the 4 public facilities."""
    n = len(acc_by_type["park"])
    comp = [0.0] * n
    for ftype in TRADITIONAL_FACILITY_TYPES:
        vals = acc_by_type[ftype]
        lo = min(vals)
        hi = max(vals)
        span = hi - lo if (hi - lo) > 1e-15 else 1.0
        for i in range(n):
            comp[i] += (vals[i] - lo) / span
    return [v / len(TRADITIONAL_FACILITY_TYPES) for v in comp]


def _build_synthetic_pdf_pages(
    headline: dict[str, Any],
    *,
    main_text_pages_mode: str = "valid",
) -> list[list[str]]:
    """Build deterministic multi-page CS482 Phase A report lines yielding exact fractional pages."""
    ev_g15 = headline["ev_gini_15"]
    park_g15 = headline["park_gini_15"]
    h3_coef = headline["h3_elderly_coef"]
    rf_auc = headline["rf_oof_roc_auc"]
    lr_auc = headline["lr_oof_roc_auc"]
    rf_pr = headline["rf_oof_pr_auc"]

    cover_page = [
        "CS482 Data Science - Phase A Project Report Draft",
        "Spatial Equity of Public Facilities and Public EV Charging Access in Macao SAR",
        "Group Number: Group X",
        "Course: CS482 Introduction to Data Science (University of Macau)",
        "Submission Stage: Phase A Deliverable & 12/03 Go/No-Go Checkpoint",
        "Team Members and Student IDs:",
        "Member 1: [Member Name 1] (Student ID: [DC00001])",
        "Member 2: [Member Name 2] (Student ID: [DC00002])",
        "Member 3: [Member Name 3] (Student ID: [DC00003])",
        "Member 4: [Member Name 4] (Student ID: [DC00004])",
    ]

    # Full page = 36 lines at 18pt pitch (~648 pt vertical height inside 841.89 pt A4)
    p2 = [
        "1. Introduction, Problem Definition, and Research Questions",
        "Rapid urban densification and ongoing electrification in the Macao Special",
        "Administrative Region create urgent spatial equity questions regarding 10-minute",
        "and 15-minute pedestrian accessibility to essential public services and public",
        "electric vehicle (EV) charging supply. This study investigates whether walkable",
        "access to parks, official SSM health centres, public IC libraries, bus stops,",
        "and CEM public EV chargers is distributed equitably across Macao's 681,293",
        "terrestrial residents and 82,751 residents aged 65+ spanning 7 official parishes",
        "and 23 terrestrial DSEC statistical zones. Primary beneficiaries include urban",
        "planning authorities (DSSCU, DSAT), aging residents, and EV siting planners.",
        "We formulate three pre-specified falsifiable hypotheses using strictly associational",
        "and correlational framing: H1 tests spatial clustering and inequality of walking",
        "accessibility via population-weighted Gini, parish-decomposed Theil's T, and",
        "FDR-adjusted LISA; H2 tests paired bootstrap Gini differentials between public EV",
        "charging and four traditional public facilities with Holm adjustment; H3 examines",
        "cross-sectional zone-level associations between elderly population share",
        "(controlling for log population density) and accessibility across all 23 terrestrial",
        "zones using side-by-side OLS-HC3, SEM, and SLM specifications.",
        "2. Data Collection, Preparation, and CRS Standardization",
        "Authoritative public datasets were acquired and logged with SHA-256 and UTC+8 dates:",
        "(1) DSEC 2021 Population Census building and zone layers fetched in native EPSG:8433",
        "and transformed locally with always_xy=True to EPSG:32649 (eliminating 315m datum",
        "shift and excluding maritime zone Z26 with 777 residents), conserving exact",
        "terrestrial totals of 681,293 residents and 82,751 seniors across 23 zones, 7",
        "parishes (including Taipa / Carmo), and 280 populated H3 resolution-9 hexagons",
        "(plus 45 H3 resolution-8 cells for MAUP sensitivity checks).",
        "(2) Live CEM public EV API (84 stations, 1,610 unique physical chargers after",
        "correcting swapped latitude/longitude coordinates, deduplicating connector heads",
        "via chargerid with pilenum fallback for station 9488798, and flagging 4 Zona A",
        "stations). (3) Geofabrik Macau OpenStreetMap extracts and official SSM, IC, IAM,",
        "and MLM registries (documenting the data.gov.mo login gate fallback), applying",
        "cross-source-only 25 m deduplication so opposite-side bus stops remain intact.",
        "All unpopulated cells were dropped, and pedestrian networks used retain_all=True",
        "so both Macao Peninsula and Taipa-Cotai-Coloane components remain intact without",
        "cross-harbour links. Strictly leak-free covariates (11 non-EV features) and",
        "EV-stripped OSM POI texts were frozen alongside 5 geographically clustered blocks",
        "and a 10-minute pedestrian cell graph verified via FREEZE_MANIFEST.json.",
    ]

    p3 = [
        "3. Spatial Analytics, Econometric Modeling, and Method Evaluation",
        "Our methodology integrates exploratory data analysis (EDA maps and Lorenz curves)",
        "with confirmatory spatial statistics and leak-free predictive modeling, avoiding",
        "descriptive-only reporting. First, Gaussian Two-Step Floating Catchment Area",
        "(2SFCA) scores were computed at 10-minute and 15-minute network walking thresholds",
        "(4.5 km/h = 75 m/min) for all five facility types plus capacity-weighted vs.",
        "unweighted EV sensitivity and a composite public accessibility index. Weighted",
        "supply conservation and 10-to-15 minute support monotonicity were verified",
        "algebraically across H3 resolution-9, resolution-8, and 23-zone geometries.",
        "Second, H1 inequality was evaluated using population-weighted Gini and parish",
        "Theil's T (GE(1) with 0 ln 0 = 0, preserving exact within + between additivity",
        "even when cells have zero access) with 2,000-replicate bootstrap 95% CIs.",
        "Spatial clustering was assessed with Global Moran's I and Local Moran's I (LISA)",
        "using 999 permutations and Benjamini-Hochberg FDR adjustment on connected weights.",
        "Third, H2 compared EV charging Gini against parks, health centres, libraries, and",
        "bus stops via paired cell-level bootstrap (2,000 replicates, seed=42) with Holm p.",
        "Fourth, H3 avoided pretest switching by reporting OLS (HC3 SE), spreg ML_Error",
        "(SEM), and spreg ML_Lag (SLM) side-by-side at N=23 zones under both connected k=4",
        "nearest-neighbor and bridge-augmented Queen spatial weights, inspecting VIF and",
        "residual Moran's I. Fifth, leak-free binary classification of has_ev_charger (56",
        "positive cells, 20.0% prevalence) was evaluated across frozen 5-fold Spatial Block",
        "CV using Naive Prior, Logistic Regression, and Random Forest inside strict",
        "train-only pipelines, reporting ROC-AUC, PR-AUC, Brier score, and Top-20% Hit Rate.",
        "Multi-resolution MAUP sensitivity contrasted H3 res-8 versus res-9.",
        "Figure 1 displays the dual-panel 15-minute composite accessibility choropleth",
        "alongside FDR-significant LISA clusters across Macao Peninsula, Taipa, and Coloane.",
        "Figure 1: Dual-panel 15-minute composite walking accessibility and FDR LISA map.",
        "Table 1 summarizes verified H1 inequality metrics, H3 side-by-side zone regression",
        "estimates, and Phase A spatial-block cross-validation baseline metrics.",
    ]

    p4 = [
        "4. Results, Table 1 Synthesis, and Multi-Axis Sensitivity Analysis",
        f"Across the 280 populated H3 res-9 cells, 15-minute EV charging Gini is {ev_g15:.3f}",
        f"compared with {park_g15:.3f} for parks (Table 1). Parish-level Theil's T decomposition",
        "quantifies both within-parish and between-parish disparities across all 7 parishes",
        "including Nossa Senhora do Carmo (Taipa), confirming exact additivity.",
        "FDR-adjusted LISA analysis highlights localized High-High access clusters and",
        "Low-Low coldspots concentrated in peripheral residential areas.",
        "Paired bootstrap comparisons (H2, 2,000 replicates, Holm-adjusted) report effect",
        "sizes and 95% confidence intervals honestly, interpreting non-significant contrasts",
        "without claiming absence of difference. In zone-level H3 regressions (N=23 zones),",
        f"elderly_share has an OLS-HC3 association coef of {h3_coef:.3f} controlling for",
        "log_pop_density, with consistent side-by-side SEM and SLM estimates across k=4 NN",
        "and bridge-Queen weights, low multicollinearity (finite VIF), and residual Moran's I.",
        f"Table 1 (Summary): H1 15-min EV Gini = {ev_g15:.3f}, Park Gini = {park_g15:.3f}; "
        f"H3 N=23 OLS-HC3 elderly_share = {h3_coef:.3f};",
        f"ML 5-fold Spatial Block CV: Logistic Regression OOF ROC-AUC = {lr_auc:.3f},",
        f"Random Forest OOF ROC-AUC = {rf_auc:.3f}, OOF PR-AUC = {rf_pr:.3f}, Naive = 0.500.",
        "Four-axis sensitivity checks confirm robustness: (a) MAUP comparison of H3 res-8",
        "(45 cells) versus res-9 (280 cells) preserves identical terrestrial census totals",
        "while quantifying scale effects on Gini, Theil's T, and Moran's I; (b) widening",
        "walking thresholds from 10 to 15 min monotonically expands served cell support;",
        "(c) capacity-weighted EV 2SFCA differs meaningfully from unweighted station counts;",
        "and (d) k=4 NN and bridge-Queen weights yield coherent H3 inference without",
        "disconnected island artifacts. All numbers trace verbatim to CSV/JSON files.",
        "5. Discussion, Planning Implications, Limitations, and Phase B Roadmap",
        "Our cross-sectional findings illustrate measurable spatial associations between",
        "urban form, demographic structure, and public service accessibility across Macao.",
        "Importantly, all results are strictly correlational and observational: this study",
        "does not establish causal mechanisms. Observed public EV charger placement reflects",
        "historical allocation decisions rather than true latent charging demand, so unserved",
        "cells flagged by spatial-block baselines represent candidate locations requiring",
        "on-site engineering and field verification before siting.",
        "Four key limitations must be considered when interpreting these findings:",
        "(1) High-rise population allocation and temporal mismatch: uniform building-height",
        "dasymetric assumptions fail in Macao's vertical high-rise districts; while DSEC",
        "2021 building-level counts mitigate footprint bias, building-group aggregation and",
        "2021-vs-2026 temporal mismatch (Zona A EV stations, UM Hengqin) remain caveats.",
        "(2) Incomplete facility capacity metadata: point-only parks require area imputation,",
        "station 9488798 uses pilenum fallback, and unit weights omit opening hours/quality.",
    ]

    # Page 5 (~20 lines = ~55% of 36-line page -> 3.0 + 0.55 = 3.55 exact main pages!)
    p5 = [
        "(3) Pedestrian routing simplifications: network walking times at 4.5 km/h omit",
        "elevation gradients, topography/slope, elevated pedestrian footbridges, overpasses,",
        "and non-walkable cross-sea harbour bridge crossings between Peninsula and Taipa.",
        "(4) Ecological fallacy and small-N statistical power: zone-level associations across",
        "N=23 statistical zones cannot be extrapolated to individual residents, and the small",
        "zone sample limits statistical power in spatial regression models.",
        "Policy recommendations for DSSCU and DSAT focus on auditing persistent FDR Low-Low",
        "coldspots in aging neighbourhoods and verifying grid capacity, curb space, and local",
        "resident demand through field surveys prior to infrastructure investment.",
        "Future Work (Phase B Extension): Using the immutable Phase A frozen data contract,",
        "10-minute pedestrian cell graph, EV-stripped POI text corpus, and frozen 5 spatial",
        "blocks, Phase B will benchmark graph neural networks (GATv2 / GraphSAGE) and",
        "text-attributed graph representations strictly against the verified Phase A Naive,",
        "Logistic Regression, and Random Forest baselines within the remaining page budget.",
        "In conclusion, Phase A delivers a reproducible, leak-free, multi-scale spatial",
        "equity audit of Macao public facilities and EV charging access that satisfies all",
        "five CS482 Minimum Requirements, enforces rigorous statistical and geospatial",
        "safeguards, and passes the binding 12/03 Go/No-Go checkpoint.",
        "End of Main Text (Phase A Draft).",
    ]

    backmatter_page = [
        "References",
        "Dai, D. (2011). Racial/ethnic and socioeconomic disparities in green space access.",
        "DSEC (2022). Results of 2021 Population Census of Macao SAR (CensusGIS REST API).",
        "CEM (2026). Companhia de Electricidade de Macau Public EV Charging Station Map API.",
        "OpenStreetMap Contributors & Geofabrik (2026). Macau SAR Pedestrian Network Extract.",
        "Kassis, T., et al. (2026). K-Dense: Automated Scientific Discovery. arXiv:2609.00065.",
        "Contribution Statement",
        "Member 1 ([DC00001]): DSEC/CEM/OSM data ingestion, EPSG:8433->32649 CRS, H3 grids.",
        "Member 2 ([DC00002]): Gaussian 2SFCA modeling, Gini/Theil, LISA, H2 bootstrap.",
        "Member 3 ([DC00003]): H3 zone spatial regressions (OLS-HC3/SEM/SLM), MAUP checks.",
        "Member 4 ([DC00004]): Leak-free ML spatial-block baselines, Figure 1/Table 1, report.",
        "AI Usage Declaration",
        "Tools Used: Large language model coding and writing assistants were used.",
        "Purposes: AI assisted with project ideation and planning, Python coding and",
        "debugging, and manuscript editing and LaTeX formatting.",
        "Independent Verification: All data ingestion, CRS alignment, statistical formulas,",
        "spatial weights, ML anti-leakage boundaries, citations, and report numbers were",
        "independently verified against raw public sources, automated tests, and SHA-256.",
    ]

    if main_text_pages_mode == "short_us_letter":
        # Deliberately violates page budget (< 2.0 pages) & includes TBD_NUM
        short_p2 = p2[:28] + ["Unresolved placeholder TBD_NUM in table."]
        return [cover_page, short_p2, backmatter_page]

    return [cover_page, p2, p3, p4, p5, backmatter_page]


def build_valid_synthetic_project(root: Path) -> Path:
    """Create a complete, mathematically coherent, PROJECT.md-compliant synthetic project root."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    rng = random.Random(GLOBAL_SEED)

    # 1. Zones (23) & Parishes (7) geometries
    zones_meta: list[dict[str, Any]] = []
    parishes_meta: list[dict[str, Any]] = []
    for pspec in _PARISH_SPECS:
        px0, py0, px1, py1 = pspec["bbox"]
        z_ids: list[str] = pspec["zones"]
        n_z = len(z_ids)
        dx = (px1 - px0) / n_z
        p_area_km2 = ((px1 - px0) * (py1 - py0)) / 1e6
        parishes_meta.append(
            {
                "parish_id": pspec["parish_id"],
                "parish_name": pspec["parish_name"],
                "parish_name_zh": pspec["parish_name_zh"],
                "bbox": (px0, py0, px1, py1),
                "area_km2": p_area_km2,
                "landmass": pspec["landmass"],
                "ring": _rect_ring(px0, py0, px1, py1),
            }
        )
        for z_idx, zid in enumerate(z_ids):
            zx0 = px0 + z_idx * dx
            zx1 = zx0 + dx
            z_area_km2 = ((zx1 - zx0) * (py1 - py0)) / 1e6
            zones_meta.append(
                {
                    "zone_id": zid,
                    "zone_name": f"Zona {zid} ({pspec['parish_name']})",
                    "parish_id": pspec["parish_id"],
                    "parish_name": pspec["parish_name"],
                    "bbox": (zx0, py0, zx1, py1),
                    "cx": 0.5 * (zx0 + zx1),
                    "cy": 0.5 * (py0 + py1),
                    "area_km2": z_area_km2,
                    "landmass": pspec["landmass"],
                    "ring": _rect_ring(zx0, py0, zx1, py1),
                }
            )

    # 2. 280 H3 res-9 cells strictly inside the 23 zones (12 per zone + 4 extra = 280)
    r9_per_zone = [12] * 23
    for idx in (0, 5, 17, 22):
        r9_per_zone[idx] += 1  # sum = 276 + 4 = 280

    hex_r9_radius = 165.0  # area ~ 0.0707 km^2, comfortably fits inside every zone
    cells_r9: list[dict[str, Any]] = []
    cell_counter = 0
    for z_idx, zinfo in enumerate(zones_meta):
        zx0, zy0, zx1, zy1 = zinfo["bbox"]
        count = r9_per_zone[z_idx]
        cols_n = 3 if count <= 12 else 4
        rows_n = math.ceil(count / cols_n)
        margin_x = min(hex_r9_radius + 15.0, (zx1 - zx0) * 0.22)
        margin_y = min(hex_r9_radius + 15.0, (zy1 - zy0) * 0.22)
        local_r = min(hex_r9_radius, margin_x - 5.0, margin_y - 5.0)
        for k in range(count):
            col_i = k % cols_n
            row_i = k // cols_n
            fx = 0.5 if cols_n == 1 else col_i / (cols_n - 1)
            fy = 0.5 if rows_n == 1 else row_i / (rows_n - 1)
            cx = (zx0 + margin_x) + fx * max(1.0, (zx1 - zx0) - 2.0 * margin_x)
            cy = (zy0 + margin_y) + fy * max(1.0, (zy1 - zy0) - 2.0 * margin_y)
            ring = _hex_ring(cx, cy, local_r)
            area_m2 = 1.5 * math.sqrt(3.0) * local_r * local_r
            area_km2 = area_m2 / 1e6
            # Deterministic varied population & elderly share per zone/cell
            base_w = (
                1.0
                + 0.6 * math.sin(cell_counter * 0.37)
                + (0.9 if zinfo["landmass"] == "peninsula" else 0.0)
            )
            elderly_prop = 0.08 + 0.13 * (0.5 + 0.5 * math.cos(z_idx * 0.61 + k * 0.2))
            cell_id = f"89411ca{cell_counter:08x}"
            cells_r9.append(
                {
                    "cell_id": cell_id,
                    "cx": cx,
                    "cy": cy,
                    "x": cx,
                    "y": cy,
                    "ring": ring,
                    "area_km2": area_km2,
                    "zone_id": zinfo["zone_id"],
                    "zone_name": zinfo["zone_name"],
                    "parish_id": zinfo["parish_id"],
                    "parish_name": zinfo["parish_name"],
                    "landmass": zinfo["landmass"],
                    "spatial_block": _ZONE_TO_BLOCK[zinfo["zone_id"]],
                    "raw_pop_w": max(0.35, base_w),
                    "raw_eld_prop": elderly_prop,
                }
            )
            cell_counter += 1

    # Scale res-9 populations to exact terrestrial totals (681,293 / 82,751)
    sum_w = sum(c["raw_pop_w"] for c in cells_r9)
    for c in cells_r9:
        c["pop_total"] = round(TERRESTRIAL_POP_TOTAL * (c["raw_pop_w"] / sum_w), 4)
    diff_tot = TERRESTRIAL_POP_TOTAL - sum(c["pop_total"] for c in cells_r9)
    cells_r9[0]["pop_total"] = round(cells_r9[0]["pop_total"] + diff_tot, 4)

    raw_eld = [c["pop_total"] * c["raw_eld_prop"] for c in cells_r9]
    sum_eld = sum(raw_eld)
    for idx, c in enumerate(cells_r9):
        c["pop_65plus"] = round(TERRESTRIAL_POP_65PLUS * (raw_eld[idx] / sum_eld), 4)
        c["pop_65plus"] = min(c["pop_65plus"], c["pop_total"] * 0.85)
    diff_eld = TERRESTRIAL_POP_65PLUS - sum(c["pop_65plus"] for c in cells_r9)
    cells_r9[0]["pop_65plus"] = round(cells_r9[0]["pop_65plus"] + diff_eld, 4)

    for c in cells_r9:
        c["pop_density"] = c["pop_total"] / c["area_km2"]
        c["elderly_share"] = c["pop_65plus"] / c["pop_total"]

    # Roll up exact zone and parish totals from res-9 cells
    for zinfo in zones_meta:
        z_cells = [c for c in cells_r9 if c["zone_id"] == zinfo["zone_id"]]
        zinfo["pop_total"] = sum(c["pop_total"] for c in z_cells)
        zinfo["pop_65plus"] = sum(c["pop_65plus"] for c in z_cells)
        zinfo["pop_density"] = zinfo["pop_total"] / zinfo["area_km2"]
        zinfo["log_pop_density"] = math.log(zinfo["pop_density"])
        zinfo["elderly_share"] = zinfo["pop_65plus"] / zinfo["pop_total"]
        zinfo["x"] = zinfo["cx"]
        zinfo["y"] = zinfo["cy"]

    for pinfo in parishes_meta:
        p_zones = [z for z in zones_meta if z["parish_id"] == pinfo["parish_id"]]
        pinfo["pop_total"] = sum(z["pop_total"] for z in p_zones)
        pinfo["pop_65plus"] = sum(z["pop_65plus"] for z in p_zones)

    # 3. 45 H3 res-8 cells partitioning exact zone populations
    cells_r8: list[dict[str, Any]] = []
    r8_counter = 0
    for z_idx, zinfo in enumerate(zones_meta):
        n_r8 = 2 if z_idx < 22 else 1  # 22*2 + 1 = 45 cells
        zx0, zy0, zx1, zy1 = zinfo["bbox"]
        for k in range(n_r8):
            fx = 0.5 if n_r8 == 1 else (0.28 if k == 0 else 0.72)
            cx = zx0 + fx * (zx1 - zx0)
            cy = 0.5 * (zy0 + zy1)
            rad = min(220.0, 0.24 * (zx1 - zx0), 0.35 * (zy1 - zy0))
            ring = _hex_ring(cx, cy, rad)
            area_km2 = (1.5 * math.sqrt(3.0) * rad * rad) / 1e6
            share = 1.0 if n_r8 == 1 else (0.55 if k == 0 else 0.45)
            ptot = zinfo["pop_total"] * share
            p65 = zinfo["pop_65plus"] * share
            cells_r8.append(
                {
                    "cell_id": f"88411ca{r8_counter:08x}",
                    "cx": cx,
                    "cy": cy,
                    "x": cx,
                    "y": cy,
                    "ring": ring,
                    "area_km2": area_km2,
                    "zone_id": zinfo["zone_id"],
                    "parish_id": zinfo["parish_id"],
                    "parish_name": zinfo["parish_name"],
                    "landmass": zinfo["landmass"],
                    "pop_total": ptot,
                    "pop_65plus": p65,
                    "pop_density": ptot / area_km2,
                    "elderly_share": p65 / ptot,
                    "has_ev_charger": 1 if (r8_counter % 3 == 0) else 0,
                }
            )
            r8_counter += 1

    # 4. Select 56 positive EV cells (exact 20% prevalence, >= 9 positives per block 0..4)
    # Correlate EV presence moderately with commercial share & density so ML models learn signal
    ev_pos_indices: set[int] = set()
    for blk in range(5):
        blk_idxs = [i for i, c in enumerate(cells_r9) if c["spatial_block"] == blk]
        target_k = 11 if blk < 4 else 12  # 4*11 + 12 = 56
        scored = sorted(blk_idxs, key=lambda i: (cells_r9[i]["pop_density"], i), reverse=True)
        chosen = scored[: target_k - 2] + scored[len(scored) // 2 : len(scored) // 2 + 2]
        ev_pos_indices.update(chosen[:target_k])
    ev_pos_sorted = sorted(ev_pos_indices)

    # 5. Facilities (5 types) & 14 LRT stations
    facilities: list[dict[str, Any]] = []
    fac_id = 0

    # 5a. 84 CEM EV stations: 80 inside the 56 positive EV cells + 4 in Zona A (1,610 capacity)
    # Distribute 80 stations across 56 positive cells (24 cells get 2 stations, 32 cells get 1)
    ev_cell_assignments: list[int] = list(ev_pos_sorted) + list(ev_pos_sorted[:24])
    ev_caps = [19] * 80 + [22, 22, 23, 23]  # 80*19 + 90 = 1520 + 90 = 1610 exact!
    cell_ev_count = [0] * len(cells_r9)
    cell_ev_cap = [0] * len(cells_r9)
    for idx_s, c_idx in enumerate(ev_cell_assignments):
        c = cells_r9[c_idx]
        # Small offset well inside cell hex radius
        off_x = 12.0 * math.cos(idx_s * 1.7)
        off_y = 12.0 * math.sin(idx_s * 1.7)
        cap = ev_caps[idx_s]
        cell_ev_count[c_idx] += 1
        cell_ev_cap[c_idx] += cap
        facilities.append(
            {
                "facility_id": f"EV_{fac_id:04d}",
                "facility_type": "ev_charger",
                "name": f"CEM Public EV Station {idx_s + 1:02d}",
                "source": "cem_ev_api",
                "capacity": float(cap),
                "capacity_unit": "chargers",
                "capacity_imputed": (idx_s == 79),
                "pilenum": float(cap + 7),
                "total_kw": float(cap * 22.0),
                "in_zona_a": False,
                "x": c["cx"] + off_x,
                "y": c["cy"] + off_y,
                "landmass": c["landmass"],
            }
        )
        fac_id += 1

    # 4 Zona A reclaimed EV stations (E > 764,000, outside 2021 populated zones)
    for za_i in range(4):
        cap = ev_caps[80 + za_i]
        facilities.append(
            {
                "facility_id": f"EV_{fac_id:04d}",
                "facility_type": "ev_charger",
                "name": f"CEM Zona A EV Station {za_i + 1}",
                "source": "cem_ev_api",
                "capacity": float(cap),
                "capacity_unit": "chargers",
                "capacity_imputed": False,
                "pilenum": float(cap + 8),
                "total_kw": float(cap * 30.0),
                "in_zona_a": True,
                "x": 765_000.0 + za_i * 220.0,
                "y": 2_457_800.0 + za_i * 150.0,
                "landmass": "peninsula",
            }
        )
        fac_id += 1

    # 5b. 80 Parks (area in m^2)
    for p_i in range(80):
        c = cells_r9[(p_i * 3 + 1) % len(cells_r9)]
        area_m2 = 850.0 + ((p_i * 733) % 14000)
        facilities.append(
            {
                "facility_id": f"PK_{fac_id:04d}",
                "facility_type": "park",
                "name": f"Macao Public Garden {p_i + 1:02d}",
                "source": "osm_iam" if p_i % 3 != 0 else "iam_gov_mo",
                "capacity": float(area_m2),
                "capacity_unit": "m2",
                "capacity_imputed": (p_i % 10 == 0),
                "pilenum": None,
                "total_kw": None,
                "in_zona_a": False,
                "x": c["cx"] + (18.0 if p_i % 3 != 0 else -18.0),
                "y": c["cy"] + (15.0 if p_i % 2 == 0 else -15.0),
                "landmass": c["landmass"],
            }
        )
        fac_id += 1

    # 5c. 11 Official SSM Health Centres
    health_cell_idxs = [4, 28, 55, 82, 110, 138, 165, 192, 215, 244, 271]
    for h_i, c_idx in enumerate(health_cell_idxs):
        c = cells_r9[c_idx]
        facilities.append(
            {
                "facility_id": f"HC_{fac_id:04d}",
                "facility_type": "health_centre",
                "name": f"SSM Centro de Saude {h_i + 1:02d}",
                "source": "ssm_gov_mo",
                "capacity": 1.0,
                "capacity_unit": "facility",
                "capacity_imputed": False,
                "pilenum": None,
                "total_kw": None,
                "in_zona_a": False,
                "x": c["cx"] + 5.0,
                "y": c["cy"] - 5.0,
                "landmass": c["landmass"],
            }
        )
        fac_id += 1

    # 5d. 15 Official IC Public Libraries (strictly public, zero university/museum names)
    lib_cell_idxs = [8, 24, 42, 61, 79, 98, 117, 134, 152, 171, 189, 208, 227, 249, 268]
    for l_i, c_idx in enumerate(lib_cell_idxs):
        c = cells_r9[c_idx]
        facilities.append(
            {
                "facility_id": f"LB_{fac_id:04d}",
                "facility_type": "library",
                "name": f"IC Public Library Branch {l_i + 1:02d}",
                "source": "ic_library_gov_mo",
                "capacity": 1.0,
                "capacity_unit": "facility",
                "capacity_imputed": False,
                "pilenum": None,
                "total_kw": None,
                "in_zona_a": False,
                "x": c["cx"] - 6.0,
                "y": c["cy"] + 6.0,
                "landmass": c["landmass"],
            }
        )
        fac_id += 1

    # 5e. 420 Bus Stops (2 sources in disjoint cells so cross-source pairs stay >= 35 m apart)
    bus_coords: list[tuple[float, float]] = []
    for b_i in range(420):
        c_idx = (b_i % 200) if b_i < 280 else (200 + (b_i % 80))
        c = cells_r9[c_idx]
        src = "osm_bus" if b_i < 280 else "dsat_bus"
        off_x = (14.0 if b_i % 2 == 0 else -14.0) + ((b_i % 5) - 2) * 2.0
        off_y = 10.0 if (b_i // 2) % 2 == 0 else -10.0
        bx = c["cx"] + off_x
        by = c["cy"] + off_y
        bus_coords.append((bx, by))
        facilities.append(
            {
                "facility_id": f"BS_{fac_id:04d}",
                "facility_type": "bus_stop",
                "name": f"DSAT Bus Stop M{b_i + 1:03d}",
                "source": src,
                "capacity": 1.0,
                "capacity_unit": "stop",
                "capacity_imputed": False,
                "pilenum": None,
                "total_kw": None,
                "in_zona_a": False,
                "x": bx,
                "y": by,
                "landmass": c["landmass"],
            }
        )
        fac_id += 1

    # 5f. 14 Operational MLM LRT stations (Peninsula Barra + Taipa/Cotai/Airport)
    lrt_coords: list[tuple[str, float, float]] = [
        ("Barra", 761_600.0, 2_455_800.0),
        ("Ocean", 761_900.0, 2_452_700.0),
        ("Jockey Club", 762_300.0, 2_452_300.0),
        ("Stadium", 762_700.0, 2_452_100.0),
        ("Pai Kok", 763_100.0, 2_451_800.0),
        ("Cotai West", 762_600.0, 2_450_100.0),
        ("Lotus", 762_900.0, 2_449_600.0),
        ("East Asian Games", 763_600.0, 2_449_900.0),
        ("Cotai East", 764_100.0, 2_450_300.0),
        ("MUST", 764_500.0, 2_451_400.0),
        ("Airport", 765_100.0, 2_451_700.0),
        ("Taipa Ferry Terminal", 765_200.0, 2_452_500.0),
        ("Union Hospital", 763_300.0, 2_449_300.0),
        ("Seac Pai Van", 763_500.0, 2_448_600.0),
    ]

    # 6. Populate leak-free ML covariates & target/audit columns on res-9 cells
    for idx, c in enumerate(cells_r9):
        has_ev = 1 if idx in ev_pos_indices else 0
        c["has_ev_charger"] = has_ev
        c["ev_station_count"] = cell_ev_count[idx]
        c["ev_charger_capacity"] = cell_ev_cap[idx]
        # Realistic urban morphology covariates (correlated subtly with EV presence)
        com_share = min(0.85, max(0.05, 0.22 + 0.18 * has_ev + rng.uniform(-0.10, 0.12)))
        res_share = min(0.92, max(0.08, 0.65 - 0.15 * has_ev + rng.uniform(-0.10, 0.10)))
        c["intersection_density"] = round(
            max(15.0, 85.0 + 35.0 * has_ev + rng.uniform(-25.0, 30.0)), 4
        )
        c["road_density_km_per_km2"] = round(
            max(4.0, 14.5 + 4.5 * has_ev + rng.uniform(-3.5, 4.0)), 4
        )
        c["res_bldg_share"] = round(res_share, 4)
        c["com_bldg_share"] = round(com_share, 4)
        c["bldg_coverage_ratio"] = round(
            min(0.85, max(0.08, 0.34 + 0.10 * has_ev + rng.uniform(-0.08, 0.08))), 4
        )
        c["dist_nearest_lrt_m"] = round(
            min(math.hypot(c["cx"] - lx, c["cy"] - ly) for _, lx, ly in lrt_coords), 4
        )
        c["dist_nearest_bus_m"] = round(
            min(math.hypot(c["cx"] - bx, c["cy"] - by) for bx, by in bus_coords), 4
        )

    # 7. Genuine Gaussian 2SFCA on cells_r9, cells_r8, and zones_23
    fac_by_type = {ft: [f for f in facilities if f["facility_type"] == ft] for ft in FACILITY_TYPES}
    for dataset in (cells_r9, cells_r8, zones_meta):
        for cutoff in (10, 15):
            acc_map: dict[str, list[float]] = {}
            for ft in FACILITY_TYPES:
                acc_vals = _compute_2sfca(dataset, fac_by_type[ft], float(cutoff), False)
                acc_map[ft] = acc_vals
                for i, val in enumerate(acc_vals):
                    dataset[i][f"acc{cutoff}_{ft}"] = val
            unw_vals = _compute_2sfca(dataset, fac_by_type["ev_charger"], float(cutoff), True)
            for i, val in enumerate(unw_vals):
                dataset[i][f"acc{cutoff}_ev_unweighted"] = val
            comp_vals = _composite_index(acc_map)
            for i, val in enumerate(comp_vals):
                dataset[i][f"acc{cutoff}_composite"] = val

    # 8. Walk graph (.graphml) & 10-min pedestrian cell edges (graph_edges.csv)
    graph_edges: list[dict[str, Any]] = []
    cell_adj: dict[int, list[int]] = {i: [] for i in range(len(cells_r9))}
    for i in range(len(cells_r9)):
        ci = cells_r9[i]
        for j in range(i + 1, len(cells_r9)):
            cj = cells_r9[j]
            if ci["landmass"] != cj["landmass"]:
                continue
            dist_m = math.hypot(ci["cx"] - cj["cx"], ci["cy"] - cj["cy"])
            walk_min = 0.65 + (dist_m * 1.12) / 75.0
            if walk_min <= 10.0:
                walk_r = round(walk_min, 6)
                w_val = round(1.0 / walk_r, 8)
                same_b = ci["spatial_block"] == cj["spatial_block"]
                graph_edges.append(
                    {
                        "src_cell_id": ci["cell_id"],
                        "dst_cell_id": cj["cell_id"],
                        "walk_time_min": walk_r,
                        "weight": w_val,
                        "same_block": same_b,
                    }
                )
                graph_edges.append(
                    {
                        "src_cell_id": cj["cell_id"],
                        "dst_cell_id": ci["cell_id"],
                        "walk_time_min": walk_r,
                        "weight": w_val,
                        "same_block": same_b,
                    }
                )
                cell_adj[i].append(j)
                cell_adj[j].append(i)

    # Ensure every cell has >= 1 neighbor for LISA spatial lag (using nearest same-landmass cell)
    lisa_adj: dict[int, list[int]] = {}
    for i, ci in enumerate(cells_r9):
        if cell_adj[i]:
            lisa_adj[i] = cell_adj[i]
        else:
            same_land = [
                j for j, cj in enumerate(cells_r9) if j != i and cj["landmass"] == ci["landmass"]
            ]
            same_land.sort(
                key=lambda j: math.hypot(ci["cx"] - cells_r9[j]["cx"], ci["cy"] - cells_r9[j]["cy"])
            )
            lisa_adj[i] = same_land[:4]

    # 9. LISA + BH-FDR on cells_r9 for 10 & 15 min metrics
    lisa_metrics = [
        f"acc{cutoff}_{ft}" for cutoff in (10, 15) for ft in (*FACILITY_TYPES, "composite")
    ]
    global_moran_cache: dict[str, tuple[float, float]] = {}
    for metric in lisa_metrics:
        vals = [c[metric] for c in cells_r9]
        li, lp, lq, lcl, g_i, g_p = _compute_synthetic_lisa(vals, lisa_adj)
        global_moran_cache[metric] = (g_i, g_p)
        for i, c in enumerate(cells_r9):
            c[f"lisa_i_{metric}"] = round(li[i], 6)
            c[f"lisa_p_{metric}"] = round(lp[i], 6)
            c[f"lisa_q_fdr_{metric}"] = round(lq[i], 6)
            c[f"lisa_cluster_{metric}"] = lcl[i]

    # 10. Write all GeoPackages (M1 base, M2 interim, M4 frozen)
    zones_base_cols = [
        ("zone_id", "TEXT"),
        ("zone_name", "TEXT"),
        ("parish_id", "TEXT"),
        ("parish_name", "TEXT"),
        ("area_km2", "REAL"),
        ("pop_total", "REAL"),
        ("pop_65plus", "REAL"),
        ("pop_density", "REAL"),
        ("log_pop_density", "REAL"),
        ("elderly_share", "REAL"),
    ]
    acc_cols_sql = [
        (f"acc{cutoff}_{ft}", "REAL")
        for cutoff in (10, 15)
        for ft in (*FACILITY_TYPES, "composite", "ev_unweighted")
    ]
    lisa_cols_sql: list[tuple[str, str]] = []
    for metric in lisa_metrics:
        lisa_cols_sql.extend(
            [
                (f"lisa_i_{metric}", "REAL"),
                (f"lisa_p_{metric}", "REAL"),
                (f"lisa_q_fdr_{metric}", "REAL"),
                (f"lisa_cluster_{metric}", "TEXT"),
            ]
        )

    write_gpkg(
        root / "data/frozen/zones_23.gpkg",
        "zones_23",
        zones_base_cols,
        zones_meta,
        [("POLYGON", [z["ring"]]) for z in zones_meta],
        geom_type="POLYGON",
        srs_id=TARGET_EPSG,
    )
    write_gpkg(
        root / "data/interim/zones_23_with_2sfca.gpkg",
        "zones_23_with_2sfca",
        zones_base_cols + acc_cols_sql,
        zones_meta,
        [("POLYGON", [z["ring"]]) for z in zones_meta],
        geom_type="POLYGON",
        srs_id=TARGET_EPSG,
    )

    parishes_cols = [
        ("parish_id", "TEXT"),
        ("parish_name", "TEXT"),
        ("parish_name_zh", "TEXT"),
        ("area_km2", "REAL"),
        ("pop_total", "REAL"),
        ("pop_65plus", "REAL"),
    ]
    write_gpkg(
        root / "data/frozen/parishes_7.gpkg",
        "parishes_7",
        parishes_cols,
        parishes_meta,
        [("POLYGON", [p["ring"]]) for p in parishes_meta],
        geom_type="POLYGON",
        srs_id=TARGET_EPSG,
    )

    # Prepare centroid_x / centroid_y for res-9 cells
    for c in cells_r9:
        c["centroid_x"] = c["cx"]
        c["centroid_y"] = c["cy"]

    r9_base_cols = [
        ("cell_id", "TEXT"),
        ("centroid_x", "REAL"),
        ("centroid_y", "REAL"),
        ("area_km2", "REAL"),
        ("zone_id", "TEXT"),
        ("zone_name", "TEXT"),
        ("parish_id", "TEXT"),
        ("parish_name", "TEXT"),
        *((f, "REAL") for f in ML_FEATURES_REQUIRED),
        ("has_ev_charger", "INTEGER"),
        ("ev_station_count", "INTEGER"),
        ("ev_charger_capacity", "INTEGER"),
        ("spatial_block", "INTEGER"),
    ]
    r9_full_cols = r9_base_cols + acc_cols_sql + lisa_cols_sql
    r9_geoms = [("POLYGON", [c["ring"]]) for c in cells_r9]

    write_gpkg(
        root / "data/frozen/cells_base_res9.gpkg",
        "cells_base_res9",
        r9_base_cols,
        cells_r9,
        r9_geoms,
        geom_type="POLYGON",
        srs_id=TARGET_EPSG,
    )
    write_gpkg(
        root / "data/interim/cells_r9_with_2sfca_lisa.gpkg",
        "cells_r9_with_2sfca_lisa",
        r9_full_cols,
        cells_r9,
        r9_geoms,
        geom_type="POLYGON",
        srs_id=TARGET_EPSG,
    )
    write_gpkg(
        root / "data/frozen/cells.gpkg",
        "cells",
        r9_full_cols,
        cells_r9,
        r9_geoms,
        geom_type="POLYGON",
        srs_id=TARGET_EPSG,
    )

    r8_base_cols = [
        ("cell_id", "TEXT"),
        ("area_km2", "REAL"),
        ("zone_id", "TEXT"),
        ("parish_id", "TEXT"),
        ("parish_name", "TEXT"),
        ("pop_total", "REAL"),
        ("pop_65plus", "REAL"),
        ("pop_density", "REAL"),
        ("elderly_share", "REAL"),
        ("has_ev_charger", "INTEGER"),
    ]
    r8_full_cols = r8_base_cols + acc_cols_sql
    r8_geoms = [("POLYGON", [c["ring"]]) for c in cells_r8]

    write_gpkg(
        root / "data/frozen/cells_base_res8.gpkg",
        "cells_base_res8",
        r8_base_cols,
        cells_r8,
        r8_geoms,
        geom_type="POLYGON",
        srs_id=TARGET_EPSG,
    )
    write_gpkg(
        root / "data/interim/cells_r8_with_2sfca.gpkg",
        "cells_r8_with_2sfca",
        r8_full_cols,
        cells_r8,
        r8_geoms,
        geom_type="POLYGON",
        srs_id=TARGET_EPSG,
    )
    write_gpkg(
        root / "data/frozen/cells_res8.gpkg",
        "cells_res8",
        r8_full_cols,
        cells_r8,
        r8_geoms,
        geom_type="POLYGON",
        srs_id=TARGET_EPSG,
    )

    fac_cols = [
        ("facility_id", "TEXT"),
        ("facility_type", "TEXT"),
        ("name", "TEXT"),
        ("source", "TEXT"),
        ("capacity", "REAL"),
        ("capacity_unit", "TEXT"),
        ("capacity_imputed", "INTEGER"),
        ("pilenum", "REAL"),
        ("total_kw", "REAL"),
        ("in_zona_a", "INTEGER"),
    ]
    write_gpkg(
        root / "data/frozen/facilities_clean.gpkg",
        "facilities",
        fac_cols,
        facilities,
        [("POINT", (f["x"], f["y"])) for f in facilities],
        geom_type="POINT",
        srs_id=TARGET_EPSG,
    )

    lrt_rows = [
        {"station_id": f"LRT_{i + 1:02d}", "name": name}
        for i, (name, _, _) in enumerate(lrt_coords)
    ]
    write_gpkg(
        root / "data/frozen/lrt_stations.gpkg",
        "lrt_stations",
        [("station_id", "TEXT"), ("name", "TEXT")],
        lrt_rows,
        [("POINT", (lx, ly)) for _, lx, ly in lrt_coords],
        geom_type="POINT",
        srs_id=TARGET_EPSG,
    )

    # 11. Walk Graph (.graphml) with Peninsula + Taipa-Cotai-Coloane components
    graphml_path = root / "data/raw/osm_walk_graph.graphml"
    graphml_path.parent.mkdir(parents=True, exist_ok=True)
    gml_lines = [
        '<?xml version="1.0" encoding="utf-8"?>',
        '<graphml xmlns="http://graphml.graphdrawing.org/xmlns">',
        '  <key id="d0" for="node" attr.name="x" attr.type="double"/>',
        '  <key id="d1" for="node" attr.name="y" attr.type="double"/>',
        '  <key id="d2" for="edge" attr.name="length" attr.type="double"/>',
        '  <graph edgedefault="undirected">',
    ]
    for i, c in enumerate(cells_r9):
        gml_lines.append(
            f'    <node id="n{i}"><data key="d0">{c["cx"]:.2f}</data>'
            f'<data key="d1">{c["cy"]:.2f}</data></node>'
        )
    seen_undir: set[tuple[int, int]] = set()
    for i, nbrs in lisa_adj.items():
        for j in nbrs:
            a, b = (i, j) if i < j else (j, i)
            if (a, b) in seen_undir:
                continue
            seen_undir.add((a, b))
            dist_m = max(
                15.0,
                math.hypot(
                    cells_r9[a]["cx"] - cells_r9[b]["cx"], cells_r9[a]["cy"] - cells_r9[b]["cy"]
                ),
            )
            gml_lines.append(
                f'    <edge source="n{a}" target="n{b}"><data key="d2">{dist_m:.2f}</data></edge>'
            )
    gml_lines.extend(["  </graph>", "</graphml>"])
    graphml_path.write_text("\n".join(gml_lines) + "\n", encoding="utf-8")

    # Raw snapshot files + data_provenance.json
    raw_dsec_z = root / "data/raw/dsec_2021_zones_raw.json"
    raw_dsec_b = root / "data/raw/dsec_2021_buildings_raw.json"
    raw_cem = root / "data/raw/cem_ev_raw.json"
    raw_osm = root / "data/raw/macau_osm_extract.gpkg.stub"
    raw_dsec_z.write_text('{"source":"DSEC_2021_zones","count":24}\n', encoding="utf-8")
    raw_dsec_b.write_text('{"source":"DSEC_2021_buildings","count":5682}\n', encoding="utf-8")
    raw_cem.write_text('{"source":"CEM_EV_API","count":84}\n', encoding="utf-8")
    raw_osm.write_text("Geofabrik Macau OSM snapshot stub\n", encoding="utf-8")

    provenance = {
        "generated_at_utc8": "2026-11-20T17:00:00+08:00",
        "datagovmo_note": (
            "data.gov.mo API requires account login/authentication token gate; "
            "all datasets acquired via authoritative unauthenticated DSEC CensusGIS REST, "
            "CEM public EV API, Geofabrik Macau OSM, and official SSM/IC/IAM/MLM registries."
        ),
        "sources": [
            {
                "name": "dsec_2021_census_zones",
                "family": "dsec_census",
                "url": "https://www.dsec.gov.mo/arcgis/rest/services/Census2021_Zones/MapServer/0",
                "extracted_at": "2026-11-20T16:30:00+08:00",
                "record_count": 24,
                "local_path": "data/raw/dsec_2021_zones_raw.json",
                "sha256": file_sha256(raw_dsec_z),
            },
            {
                "name": "dsec_2021_census_buildings",
                "family": "dsec_census",
                "url": "https://www.dsec.gov.mo/arcgis/rest/services/c2021_Building_P5/MapServer/0",
                "extracted_at": "2026-11-20T16:32:00+08:00",
                "record_count": 5682,
                "local_path": "data/raw/dsec_2021_buildings_raw.json",
                "sha256": file_sha256(raw_dsec_b),
            },
            {
                "name": "cem_public_ev_stations",
                "family": "cem_ev",
                "url": "https://ev.cem-macau.com/Ajax/GetMapData?lang=en",
                "extracted_at": "2026-11-20T16:35:00+08:00",
                "record_count": 84,
                "local_path": "data/raw/cem_ev_raw.json",
                "sha256": file_sha256(raw_cem),
            },
            {
                "name": "geofabrik_macau_osm",
                "family": "osm_geofabrik",
                "url": "https://download.geofabrik.de/asia/china/macau-latest.osm.pbf",
                "extracted_at": "2026-11-20T16:40:00+08:00",
                "record_count": 12450,
                "local_path": "data/raw/macau_osm_extract.gpkg.stub",
                "sha256": file_sha256(raw_osm),
            },
            {
                "name": "official_registries_ssm_ic_iam_mlm",
                "family": "official_registries",
                "url": (
                    "https://www.ssm.gov.mo/portal/ | "
                    "https://www.library.gov.mo/ | https://www.mlm.com.mo/"
                ),
                "extracted_at": "2026-11-20T16:45:00+08:00",
                "record_count": 40,
                "local_path": "data/raw/osm_walk_graph.graphml",
                "sha256": file_sha256(graphml_path),
            },
        ],
    }
    _write_json(root / "outputs/metrics/data_provenance.json", provenance)
    _write_json(root / "data/raw/data_provenance.json", provenance)

    # 12. Frozen CSV/JSON M1 contracts
    _write_csv(
        root / "data/frozen/graph_edges.csv",
        ["src_cell_id", "dst_cell_id", "walk_time_min", "weight", "same_block"],
        graph_edges,
    )
    _write_csv(
        root / "data/frozen/spatial_folds.csv",
        ["cell_id", "spatial_block"],
        [{"cell_id": c["cell_id"], "spatial_block": c["spatial_block"]} for c in cells_r9],
    )

    blocks_meta: dict[str, dict[str, int]] = {}
    for blk in range(5):
        b_cells = [c for c in cells_r9 if c["spatial_block"] == blk]
        n_pos_b = sum(c["has_ev_charger"] for c in b_cells)
        blocks_meta[str(blk)] = {
            "n_cells": len(b_cells),
            "n_pos_ev": n_pos_b,
            "n_neg_ev": len(b_cells) - n_pos_b,
        }
    _write_json(
        root / "data/frozen/split_metadata.json",
        {
            "method": "geographic_kmeans_spatial_blocks",
            "seed": GLOBAL_SEED,
            "n_blocks": 5,
            "n_cells": len(cells_r9),
            "blocks": blocks_meta,
        },
    )

    poi_rows: list[dict[str, Any]] = []
    for i, c in enumerate(cells_r9):
        if i % 6 == 0:
            raw_cnt, filt_cnt, txt = 0, 0, ""
        else:
            phrase = _POI_LEXICON[i % len(_POI_LEXICON)]
            extra = f"district {c['parish_name'].lower()} urban zone {c['zone_id'].lower()}"
            if c["has_ev_charger"]:
                txt = f"commercial plaza shopping arcade {phrase} {extra}"
                raw_cnt, filt_cnt = 7, 5
            else:
                txt = f"residential neighborhood {phrase} {extra}"
                raw_cnt, filt_cnt = 4, 4
        poi_rows.append(
            {
                "cell_id": c["cell_id"],
                "poi_count_raw": raw_cnt,
                "poi_count_filtered": filt_cnt,
                "poi_text": txt,
            }
        )
    _write_csv(
        root / "data/frozen/cell_poi_texts.csv",
        ["cell_id", "poi_count_raw", "poi_count_filtered", "poi_text"],
        poi_rows,
    )

    _write_json(
        root / "data/frozen/feature_manifest.json",
        {
            "ml_features": list(ML_FEATURES_REQUIRED),
            "excluded_target_and_audit": [
                "has_ev_charger",
                "ev_station_count",
                "ev_charger_capacity",
            ],
            "analysis_prefix_excluded": ["acc10_", "acc15_", "lisa_"],
            "random_seed": GLOBAL_SEED,
        },
    )

    # 13. M2 Analytics: h1_inequality_lisa.csv, lorenz_curves.csv, h2, h3, maup, eda
    h1_rows: list[dict[str, Any]] = []
    h1_lookup: dict[tuple[str, int, str], dict[str, Any]] = {}
    lorenz_rows: list[dict[str, Any]] = []

    for res_label, dataset in (("res9", cells_r9), ("res8", cells_r8)):
        pops = [c["pop_total"] for c in dataset]
        parishes = [c["parish_id"] for c in dataset]
        for cutoff in (10, 15):
            for ft in (*FACILITY_TYPES, "composite"):
                col = f"acc{cutoff}_{ft}"
                vals = [c[col] for c in dataset]
                g = weighted_gini(vals, pops)
                theil = weighted_theil_t(vals, pops, parishes)
                g_ci_lo = max(0.0, g - 0.025)
                g_ci_hi = min(1.0, g + 0.025)
                t_ci_lo = max(0.0, theil["total"] - 0.03)
                t_ci_hi = theil["total"] + 0.03
                if res_label == "res9":
                    g_moran, g_moran_p = global_moran_cache[col]
                    n_ll = sum(1 for c in dataset if c[f"lisa_cluster_{col}"] == "LL")
                    n_hh = sum(1 for c in dataset if c[f"lisa_cluster_{col}"] == "HH")
                else:
                    g_moran, g_moran_p = 0.28, 0.005
                    n_ll, n_hh = 0, 0
                sorted_v = sorted(vals)
                med = sorted_v[len(sorted_v) // 2]
                mean_v = sum(vals) / len(vals)
                zero_pct = 100.0 * sum(1 for v in vals if v <= 1e-12) / len(vals)
                row = {
                    "resolution": res_label,
                    "threshold_min": cutoff,
                    "facility_type": ft,
                    "gini": round(g, 8),
                    "gini_ci_low": round(g_ci_lo, 8),
                    "gini_ci_high": round(g_ci_hi, 8),
                    "theil_total": round(theil["total"], 10),
                    "theil_within": round(theil["within"], 10),
                    "theil_between": round(theil["between"], 10),
                    "theil_total_ci_low": round(t_ci_lo, 8),
                    "theil_total_ci_high": round(t_ci_hi, 8),
                    "theil_between_share": round(theil["between_share"], 8),
                    "moran_i": round(g_moran, 6),
                    "moran_p": round(g_moran_p, 6),
                    "n_ll_fdr": n_ll,
                    "n_hh_fdr": n_hh,
                    "mean_acc": round(mean_v, 8),
                    "median_acc": round(med, 8),
                    "zero_acc_pct": round(zero_pct, 4),
                }
                h1_rows.append(row)
                h1_lookup[(res_label, cutoff, ft)] = row

                # Exact Lorenz curve points for res9 individual facilities
                if res_label == "res9" and ft in FACILITY_TYPES:
                    pairs = sorted(zip(vals, pops, strict=True), key=lambda x: x[0])
                    tot_w = sum(pops)
                    tot_wy = sum(v * w for v, w in pairs)
                    cp, cq = 0.0, 0.0
                    lorenz_rows.append(
                        {
                            "threshold_min": cutoff,
                            "facility_type": ft,
                            "cum_pop_share": 0.0,
                            "cum_acc_share": 0.0,
                        }
                    )
                    for v, w in pairs:
                        cp += w / tot_w
                        cq += (w * v) / tot_wy if tot_wy > 0 else 0.0
                        lorenz_rows.append(
                            {
                                "threshold_min": cutoff,
                                "facility_type": ft,
                                "cum_pop_share": round(min(1.0, cp), 10),
                                "cum_acc_share": round(min(1.0, cq), 10),
                            }
                        )

    _write_csv(root / "outputs/tables/h1_inequality_lisa.csv", list(h1_rows[0].keys()), h1_rows)
    _write_csv(
        root / "outputs/tables/lorenz_curves.csv",
        ["threshold_min", "facility_type", "cum_pop_share", "cum_acc_share"],
        lorenz_rows,
    )

    # H2 paired bootstrap Gini diffs (4 traditional facilities x 2 thresholds = 8 rows)
    h2_rows: list[dict[str, Any]] = []
    for cutoff in (10, 15):
        g_ev = float(h1_lookup[("res9", cutoff, "ev_charger")]["gini"])
        raw_ps = [0.002, 0.015, 0.048, 0.190]
        holm_ps = holm_bonferroni(raw_ps)
        for idx_ft, ft in enumerate(TRADITIONAL_FACILITY_TYPES):
            g_oth = float(h1_lookup[("res9", cutoff, ft)]["gini"])
            diff = g_ev - g_oth
            ci_lo = max(-1.0, diff - 0.045)
            ci_hi = min(1.0, diff + 0.045)
            h2_rows.append(
                {
                    "threshold_min": cutoff,
                    "comparison": f"ev_charger_vs_{ft}",
                    "gini_ev": round(g_ev, 8),
                    "gini_other": round(g_oth, 8),
                    "gini_diff": round(diff, 10),
                    "ci_low": round(ci_lo, 8),
                    "ci_high": round(ci_hi, 8),
                    "p_boot_raw": raw_ps[idx_ft],
                    "p_holm": round(holm_ps[idx_ft], 8),
                    "n_boot": 2000,
                    "seed": GLOBAL_SEED,
                }
            )
    _write_csv(root / "outputs/tables/h2_gini_diff_bootstrap.csv", list(h2_rows[0].keys()), h2_rows)

    # H3 side-by-side spatial regression (N=23 zones) using exact OLS-HC3 oracle + SEM_ML + SLM_ML
    h3_rows: list[dict[str, Any]] = []
    h3_diag_vifs: dict[str, float] = {}
    x_eld = [z["elderly_share"] for z in zones_meta]
    x_lpd = [z["log_pop_density"] for z in zones_meta]
    for cutoff in (15, 10):
        outcome_col = f"acc{cutoff}_composite"
        y_zone = [z[outcome_col] for z in zones_meta]
        ols_res = ols_hc3_and_vif(y_zone, [x_eld, x_lpd], ["elderly_share", "log_pop_density"])
        h3_diag_vifs = {k: round(v, 6) for k, v in ols_res["vif"].items()}
        for w_spec in ("knn4", "bridge_queen"):
            w_shift = 0.0 if w_spec == "knn4" else 0.004
            for model_type in ("OLS_HC3", "SEM_ML", "SLM_ML"):
                m_scale = (
                    1.0 if model_type == "OLS_HC3" else (0.96 if model_type == "SEM_ML" else 0.94)
                )
                r2_val = float(ols_res["r2"]) * (1.0 if model_type == "OLS_HC3" else 1.03)
                aic_val = -18.5 + (0.0 if model_type == "OLS_HC3" else -1.2) + w_shift * 10
                rm_i = 0.14 if model_type == "OLS_HC3" else 0.03
                rm_p = 0.18 if model_type == "OLS_HC3" else 0.42
                for term in ("const", "elderly_share", "log_pop_density"):
                    coef = float(ols_res["coefs"][term]) * m_scale + (
                        0.0 if model_type == "OLS_HC3" else w_shift
                    )
                    se = max(
                        1e-5,
                        float(ols_res["se_hc3"][term]) * (1.0 if model_type == "OLS_HC3" else 0.95),
                    )
                    h3_rows.append(
                        {
                            "threshold_min": cutoff,
                            "outcome": outcome_col,
                            "weights_spec": w_spec,
                            "model_type": model_type,
                            "term": term,
                            "coef": round(coef, 8),
                            "se": round(se, 8),
                            "ci_low": round(coef - 1.96 * se, 8),
                            "ci_high": round(coef + 1.96 * se, 8),
                            "p_value": round(0.12 if term == "elderly_share" else 0.04, 6),
                            "r2_or_pseudo_r2": round(min(0.99, max(0.0, r2_val)), 6),
                            "aic": round(aic_val, 4),
                            "resid_moran_i": round(rm_i, 6),
                            "resid_moran_p": round(rm_p, 6),
                            "n_obs": 23,
                        }
                    )
                if model_type == "SEM_ML":
                    lam = 0.24 + w_shift
                    se_lam = 0.11
                    h3_rows.append(
                        {
                            "threshold_min": cutoff,
                            "outcome": outcome_col,
                            "weights_spec": w_spec,
                            "model_type": model_type,
                            "term": "lambda",
                            "coef": round(lam, 6),
                            "se": round(se_lam, 6),
                            "ci_low": round(lam - 1.96 * se_lam, 6),
                            "ci_high": round(lam + 1.96 * se_lam, 6),
                            "p_value": 0.029,
                            "r2_or_pseudo_r2": round(min(0.99, max(0.0, r2_val)), 6),
                            "aic": round(aic_val, 4),
                            "resid_moran_i": round(rm_i, 6),
                            "resid_moran_p": round(rm_p, 6),
                            "n_obs": 23,
                        }
                    )
                elif model_type == "SLM_ML":
                    rho = 0.21 + w_shift
                    se_rho = 0.10
                    h3_rows.append(
                        {
                            "threshold_min": cutoff,
                            "outcome": outcome_col,
                            "weights_spec": w_spec,
                            "model_type": model_type,
                            "term": "rho",
                            "coef": round(rho, 6),
                            "se": round(se_rho, 6),
                            "ci_low": round(rho - 1.96 * se_rho, 6),
                            "ci_high": round(rho + 1.96 * se_rho, 6),
                            "p_value": 0.036,
                            "r2_or_pseudo_r2": round(min(0.99, max(0.0, r2_val)), 6),
                            "aic": round(aic_val, 4),
                            "resid_moran_i": round(rm_i, 6),
                            "resid_moran_p": round(rm_p, 6),
                            "n_obs": 23,
                        }
                    )

    _write_csv(root / "outputs/tables/h3_spatial_regression.csv", list(h3_rows[0].keys()), h3_rows)
    _write_json(
        root / "outputs/tables/h3_regression_diagnostics.json",
        {
            "n_obs": 23,
            "vif": h3_diag_vifs,
            "weights": {
                "knn4": {"n_islands": 0, "min_neighbors": 4, "max_neighbors": 4},
                "bridge_queen": {"n_islands": 0, "min_neighbors": 2, "max_neighbors": 7},
            },
            "lm_lag_p": 0.14,
            "lm_error_p": 0.11,
            "breusch_pagan_p": 0.22,
        },
    )

    # MAUP table (r8 vs r9 across all 5 facilities + composite at 15 min & 10 min)
    maup_rows: list[dict[str, Any]] = []
    for cutoff in (15, 10):
        for ft in (*FACILITY_TYPES, "composite"):
            r8_row = h1_lookup[("res8", cutoff, ft)]
            r9_row = h1_lookup[("res9", cutoff, ft)]
            maup_rows.append(
                {
                    "threshold_min": cutoff,
                    "facility_type": ft,
                    "gini_r8": r8_row["gini"],
                    "gini_r9": r9_row["gini"],
                    "gini_delta_r9_minus_r8": round(r9_row["gini"] - r8_row["gini"], 8),
                    "theil_r8": r8_row["theil_total"],
                    "theil_r9": r9_row["theil_total"],
                    "theil_delta_r9_minus_r8": round(
                        r9_row["theil_total"] - r8_row["theil_total"], 8
                    ),
                    "moran_i_r8": r8_row["moran_i"],
                    "moran_i_r9": r9_row["moran_i"],
                    "moran_delta_r9_minus_r8": round(r9_row["moran_i"] - r8_row["moran_i"], 6),
                }
            )
    _write_csv(root / "outputs/tables/maup_r8_vs_r9.csv", list(maup_rows[0].keys()), maup_rows)

    # EDA summary table
    eda_rows: list[dict[str, Any]] = []
    eda_vars = [f"acc{c}_{ft}" for c in (10, 15) for ft in FACILITY_TYPES] + [
        "pop_density",
        "elderly_share",
    ]
    for col in eda_vars:
        vals = sorted(float(c[col]) for c in cells_r9)
        n_v = len(vals)
        mean_v = sum(vals) / n_v
        std_v = math.sqrt(sum((v - mean_v) ** 2 for v in vals) / n_v)
        eda_rows.append(
            {
                "variable": col,
                "n_cells": n_v,
                "min": round(vals[0], 8),
                "median": round(vals[n_v // 2], 8),
                "mean": round(mean_v, 8),
                "max": round(vals[-1], 8),
                "std": round(std_v, 8),
            }
        )
    _write_csv(root / "outputs/tables/eda_summary.csv", list(eda_rows[0].keys()), eda_rows)

    # 14. M3 Leak-Free Spatial-Block CV ML Baselines (strictly from the 11 ML features!)
    x_all = [[float(c[f]) for f in ML_FEATURES_REQUIRED] for c in cells_r9]
    y_all = [int(c["has_ev_charger"]) for c in cells_r9]
    folds_all = [int(c["spatial_block"]) for c in cells_r9]
    n_cells = len(cells_r9)
    n_pos = sum(y_all)
    prevalence = n_pos / n_cells

    oof_preds: dict[str, list[float]] = {
        "naive_prior": [0.0] * n_cells,
        "logistic_regression": [0.0] * n_cells,
        "random_forest": [0.0] * n_cells,
    }
    for blk in range(5):
        tr_idx = [i for i, f in enumerate(folds_all) if f != blk]
        te_idx = [i for i, f in enumerate(folds_all) if f == blk]
        x_tr = [x_all[i] for i in tr_idx]
        y_tr = [y_all[i] for i in tr_idx]
        x_te = [x_all[i] for i in te_idx]
        prior_p = sum(y_tr) / len(y_tr)
        lr_p = fit_predict_ridge_prob(x_tr, y_tr, x_te, alpha=5.0)
        rf_p = fit_predict_ridge_prob(x_tr, y_tr, x_te, alpha=0.8)
        for loc_i, orig_i in enumerate(te_idx):
            oof_preds["naive_prior"][orig_i] = round(prior_p, 8)
            oof_preds["logistic_regression"][orig_i] = round(lr_p[loc_i], 8)
            oof_preds["random_forest"][orig_i] = round(
                max(0.001, min(0.999, 0.92 * rf_p[loc_i] + 0.02)),
                8,
            )

    oof_csv_rows = [
        {
            "cell_id": cells_r9[i]["cell_id"],
            "spatial_block": folds_all[i],
            "has_ev_charger": y_all[i],
            "prob_naive_prior": oof_preds["naive_prior"][i],
            "prob_logistic_regression": oof_preds["logistic_regression"][i],
            "prob_random_forest": oof_preds["random_forest"][i],
        }
        for i in range(n_cells)
    ]
    _write_csv(
        root / "outputs/tables/ml_oof_predictions.csv",
        list(oof_csv_rows[0].keys()),
        oof_csv_rows,
    )

    models_json: dict[str, Any] = {}
    cv_csv_rows: list[dict[str, Any]] = []
    for m_name in ("naive_prior", "logistic_regression", "random_forest"):
        probs = oof_preds[m_name]
        fold_entries: list[dict[str, Any]] = []
        for blk in range(5):
            te_idx = [i for i, f in enumerate(folds_all) if f == blk]
            yt = [y_all[i] for i in te_idx]
            yp = [probs[i] for i in te_idx]
            rec20, _prec20 = top20_hit_rates(yt, yp)
            fold_entries.append(
                {
                    "fold": blk,
                    "roc_auc": round(roc_auc_score(yt, yp), 8),
                    "pr_auc": round(average_precision_score(yt, yp), 8),
                    "brier": round(brier_score(yt, yp), 8),
                    "top20_hit_rate": round(rec20, 8),
                }
            )
        agg: dict[str, Any] = {"fold_metrics": fold_entries}
        for met in ML_METRICS:
            fvals = [fe[met] for fe in fold_entries]
            mean_m = sum(fvals) / len(fvals)
            std_m = math.sqrt(sum((v - mean_m) ** 2 for v in fvals) / len(fvals))
            agg[f"mean_{met}"] = round(mean_m, 8)
            agg[f"std_{met}"] = round(std_m, 8)
        oof_rec20, _ = top20_hit_rates(y_all, probs)
        agg["oof_roc_auc"] = round(roc_auc_score(y_all, probs), 8)
        agg["oof_pr_auc"] = round(average_precision_score(y_all, probs), 8)
        agg["oof_brier"] = round(brier_score(y_all, probs), 8)
        agg["oof_top20_hit_rate"] = round(oof_rec20, 8)
        models_json[m_name] = agg

        cv_csv_rows.append(
            {
                "model": m_name,
                **{k: agg[k] for k in agg if k != "fold_metrics"},
            }
        )

    ml_json_payload = {
        "seed": GLOBAL_SEED,
        "n_cells": n_cells,
        "n_pos": n_pos,
        "prevalence": round(prevalence, 8),
        "feature_columns": list(ML_FEATURES_REQUIRED),
        "models": models_json,
    }
    _write_json(root / "outputs/metrics/phase_a_ml_baselines.json", ml_json_payload)
    _write_csv(root / "outputs/tables/ml_cv_results.csv", list(cv_csv_rows[0].keys()), cv_csv_rows)

    # 15. Programmatic Table 1 (CSV, MD, TEX) covering all 3 panels
    h3_ols_15_eld = next(
        r
        for r in h3_rows
        if r["threshold_min"] == 15
        and r["weights_spec"] == "knn4"
        and r["model_type"] == "OLS_HC3"
        and r["term"] == "elderly_share"
    )
    h3_sem_15_eld = next(
        r
        for r in h3_rows
        if r["threshold_min"] == 15
        and r["weights_spec"] == "knn4"
        and r["model_type"] == "SEM_ML"
        and r["term"] == "elderly_share"
    )
    h3_slm_15_eld = next(
        r
        for r in h3_rows
        if r["threshold_min"] == 15
        and r["weights_spec"] == "knn4"
        and r["model_type"] == "SLM_ML"
        and r["term"] == "elderly_share"
    )

    table1_rows: list[dict[str, Any]] = []
    for ft in FACILITY_TYPES:
        r = h1_lookup[("res9", 15, ft)]
        table1_rows.append(
            {
                "panel": "Panel_A_H1_Inequality_15min",
                "item": ft,
                "metric": "gini",
                "value": f"{r['gini']:.4f}",
                "aux_metric": "theil_total",
                "aux_value": f"{r['theil_total']:.4f}",
            }
        )
    for m_lbl, h3r in (
        ("OLS_HC3", h3_ols_15_eld),
        ("SEM_ML", h3_sem_15_eld),
        ("SLM_ML", h3_slm_15_eld),
    ):
        table1_rows.append(
            {
                "panel": "Panel_B_H3_Regression_N23",
                "item": f"{m_lbl}_elderly_share",
                "metric": "coef",
                "value": f"{h3r['coef']:.4f}",
                "aux_metric": "resid_moran_i",
                "aux_value": f"{h3r['resid_moran_i']:.4f}",
            }
        )
    for m_name in ("naive_prior", "logistic_regression", "random_forest"):
        m = models_json[m_name]
        table1_rows.append(
            {
                "panel": "Panel_C_ML_SpatialBlock_CV",
                "item": m_name,
                "metric": "oof_roc_auc",
                "value": f"{m['oof_roc_auc']:.4f}",
                "aux_metric": "oof_pr_auc",
                "aux_value": f"{m['oof_pr_auc']:.4f}",
            }
        )
        table1_rows.append(
            {
                "panel": "Panel_C_ML_SpatialBlock_CV",
                "item": f"{m_name}_calibration",
                "metric": "oof_brier",
                "value": f"{m['oof_brier']:.4f}",
                "aux_metric": "oof_top20_hit_rate",
                "aux_value": f"{m['oof_top20_hit_rate']:.4f}",
            }
        )

    _write_csv(
        root / "outputs/tables/table1_summary.csv",
        ["panel", "item", "metric", "value", "aux_metric", "aux_value"],
        table1_rows,
    )

    md_lines = [
        "# Table 1: Phase A Summary (H1 Inequality, H3 Spatial Regression, ML Spatial-Block CV)",
        "",
        "| Panel | Item | Metric | Value | Aux Metric | Aux Value |",
        "| --- | --- | --- | ---: | --- | ---: |",
    ]
    tex_lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        (
            r"\caption{Table 1: Phase A Summary of H1 Gini/Theil, "
            r"H3 OLS/SEM/SLM ($N=23$), and ML Spatial-Block CV Baselines.}"
        ),
        r"\label{tab:table1}",
        r"\begin{tabular}{lllr lr}",
        r"\toprule",
        r"Panel & Item & Metric & Value & Aux Metric & Aux Value \\",
        r"\midrule",
    ]
    for tr in table1_rows:
        md_lines.append(
            f"| {tr['panel']} | {tr['item']} | {tr['metric']} | {tr['value']} | "
            f"{tr['aux_metric']} | {tr['aux_value']} |"
        )
        p_esc = tr["panel"].replace("_", r"\_")
        i_esc = tr["item"].replace("_", r"\_")
        m_esc = tr["metric"].replace("_", r"\_")
        am_esc = tr["aux_metric"].replace("_", r"\_")
        tex_lines.append(
            f"{p_esc} & {i_esc} & {m_esc} & {tr['value']} & {am_esc} & {tr['aux_value']} \\\\"
        )
    tex_lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    (root / "outputs/tables/table1_summary.md").write_text(
        "\n".join(md_lines) + "\n", encoding="utf-8"
    )
    (root / "outputs/tables/table1_summary.tex").write_text(
        "\n".join(tex_lines) + "\n", encoding="utf-8"
    )

    # 16. Publication 300-DPI PNG figures
    write_png(root / "outputs/figures/fig1_accessibility_lisa.png", 2400, 1200, dpi=300, seed=101)
    write_png(root / "outputs/figures/eda_facilities_pop.png", 1800, 1200, dpi=300, seed=102)
    write_png(root / "outputs/figures/eda_accessibility_lorenz.png", 1800, 1200, dpi=300, seed=103)
    write_png(root / "outputs/figures/maup_r8_vs_r9.png", 1800, 1200, dpi=300, seed=104)

    # 17. data_dictionary.md & FREEZE_MANIFEST.json
    all_cells_cols = [c for c, _ in r9_full_cols]
    dd_lines = [
        "# MacaoSpatialEquity Phase A -> Phase B Frozen Data Dictionary",
        "",
        "## Coordinate Reference System & Geographic Scope",
        (
            "- All frozen GeoPackages strictly use **EPSG:32649** (WGS84 / UTM Zone 49N), "
            "transformed locally from DSEC EPSG:8433 (`always_xy=True`)."
        ),
        (
            "- Maritime zone `Z26` (777 residents / 61 aged 65+) is excluded; "
            "terrestrial 2021 census totals conserve exact `681,293` / `82,751`."
        ),
        (
            "- Newly reclaimed `Zona A` (0 residents in 2021 census vs. 4 2026 CEM EV stations) "
            "and `UM Hengqin` campus scope are explicitly documented."
        ),
        (
            "- `data.gov.mo` authentication login gate fallback uses unauthenticated DSEC "
            "CensusGIS REST, CEM EV API, Geofabrik OSM, and SSM/IC/IAM/MLM registries."
        ),
        "",
        "## Frozen Files",
        "- `cells.gpkg` / `cells_res8.gpkg`: Canonical H3 res-9 and res-8 populated cells.",
        (
            "- `zones_23.gpkg` / `parishes_7.gpkg`: 23 terrestrial zones and "
            "7 official parishes (incl. Taipa)."
        ),
        (
            "- `facilities_clean.gpkg` / `lrt_stations.gpkg`: 5 cleaned facility types "
            "(25 m cross-source dedup) and operational MLM LRT stations."
        ),
        (
            "- `graph_edges.csv`: `src_cell_id`, `dst_cell_id`, `walk_time_min`, `weight`, "
            "`same_block` (10-min pedestrian graph)."
        ),
        (
            "- `spatial_folds.csv`: `cell_id`, `spatial_block` "
            "(frozen 5-fold geographic blocks, seed=42)."
        ),
        (
            "- `cell_poi_texts.csv`: `cell_id`, `poi_count_raw`, `poi_count_filtered`, "
            "`poi_text` (EV-stripped for Phase B Route B)."
        ),
        "- `feature_manifest.json`: Explicit leak-free ML/GNN feature whitelist.",
        "",
        "## Anti-Leakage Contract",
        (
            "- Strictly excluded from ML/GNN inputs: target/audit columns (`has_ev_charger`, "
            "`ev_station_count`, `ev_charger_capacity`, `ev_*`) and M2 spatial analysis "
            "prefixes (`acc10_`, `acc15_`, `lisa_`)."
        ),
        "",
        "## Complete `cells.gpkg` Column Reference",
        *(f"- `{col}`" for col in all_cells_cols),
    ]
    (root / "data/frozen/data_dictionary.md").write_text(
        "\n".join(dd_lines) + "\n", encoding="utf-8"
    )

    freeze_files_map = {rel: file_sha256(root / rel) for rel in FROZEN_CORE_HASHED_FILES}
    _write_json(
        root / "data/frozen/FREEZE_MANIFEST.json",
        {
            "generated_at_utc8": "2026-11-20T17:10:00+08:00",
            "seed": GLOBAL_SEED,
            "crs": "EPSG:32649",
            "files": freeze_files_map,
        },
    )

    # 18. Project config & documentation files
    (root / "pyproject.toml").write_text(
        """[project]
name = "macao-spatial-equity"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "geopandas>=0.14",
    "shapely>=2.0",
    "pyproj>=3.6",
    "h3>=4.0",
    "osmnx>=2.0",
    "networkx>=3.2",
    "libpysal>=4.12",
    "esda>=2.6",
    "spreg>=1.5",
    "statsmodels>=0.14",
    "scikit-learn>=1.5",
    "matplotlib>=3.9",
    "pandas>=2.2,<3.0",
    "pytest>=8.0",
]

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-ra"
""",
        encoding="utf-8",
    )
    (root / "uv.lock").write_text(
        'version = 1\nrequires-python = ">=3.11"\n'
        '[[package]]\nname = "geopandas"\nversion = "1.0.1"\n'
        '[[package]]\nname = "h3"\nversion = "4.1.2"\n'
        '[[package]]\nname = "osmnx"\nversion = "2.0.1"\n'
        '[[package]]\nname = "spreg"\nversion = "1.8.1"\n',
        encoding="utf-8",
    )
    (root / ".gitignore").write_text(
        ".venv/\n__pycache__/\n.pytest_cache/\n.ruff_cache/\n.env\n.env.*\n*.pyc\n",
        encoding="utf-8",
    )
    (root / "README.md").write_text(
        """# MacaoSpatialEquity — Plan 5 Phase A (CS482 Data Science)

Reproducible spatial equity analysis of 10-minute and 15-minute pedestrian accessibility
across public facilities (parks, SSM health centres, IC public libraries, bus stops) and
CEM public EV chargers in Macao SAR.

## Environment Setup
```bash
uv sync
export MPLBACKEND=Agg
```

## Reproducible Offline Phase A Replay
```bash
python scripts/run_phase_a.py --skip-download
```

## Frozen Phase A -> Phase B Contract Bundle
All immutable Phase A->B deliverables live under `data/frozen/` with cryptographic SHA-256
verification in `data/frozen/FREEZE_MANIFEST.json` and schema documentation in
`data/frozen/data_dictionary.md`.
""",
        encoding="utf-8",
    )

    # 19. Deterministic CLI scripts (scripts/run_phase_a.py + 01..04)
    scripts_dir = root / "scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    for stage_name in (
        "01_ingest_and_prepare.py",
        "02_run_spatial_analysis.py",
        "03_run_ml_baselines.py",
        "04_freeze_and_report.py",
    ):
        (scripts_dir / stage_name).write_text(
            f'"""Stage CLI stub {stage_name} for synthetic E2E validation."""\n'
            "import argparse\n"
            "def main() -> int:\n"
            "    p = argparse.ArgumentParser()\n"
            '    p.add_argument("--skip-download", action="store_true")\n'
            "    p.parse_args()\n"
            "    return 0\n"
            'if __name__ == "__main__":\n'
            "    raise SystemExit(main())\n",
            encoding="utf-8",
        )

    (scripts_dir / "run_phase_a.py").write_text(
        '''"""Unified deterministic Phase A replay CLI (synthetic E2E harness)."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Run MacaoSpatialEquity Phase A pipeline.")
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Replay Phase A offline from cached raw/frozen snapshots without network access.",
    )
    args = parser.parse_args()
    root = Path(os.environ.get("MACAO_EQUITY_ROOT", Path(__file__).resolve().parents[1]))
    manifest_path = root / "data" / "frozen" / "FREEZE_MANIFEST.json"
    if not manifest_path.is_file():
        raise SystemExit(f"Missing manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    updated = {}
    for rel, expected in manifest.get("files", {}).items():
        fp = root / rel
        if not fp.is_file():
            raise SystemExit(f"Missing frozen file: {fp}")
        digest = _sha256(fp)
        if args.skip_download and digest != expected:
            raise SystemExit(f"Hash mismatch on {rel}")
        updated[rel] = digest
    manifest["files"] = updated
    out_txt = json.dumps(manifest, indent=2, ensure_ascii=False) + "\\n"
    manifest_path.write_text(out_txt, encoding="utf-8")
    print(f"Phase A replay complete (skip_download={args.skip_download}, files={len(updated)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
''',
        encoding="utf-8",
    )

    # 20. Report LaTeX, BibTeX, Compiled PDF, and Go/No-Go Audit
    headline = {
        "ev_gini_15": float(h1_lookup[("res9", 15, "ev_charger")]["gini"]),
        "park_gini_15": float(h1_lookup[("res9", 15, "park")]["gini"]),
        "h3_elderly_coef": float(h3_ols_15_eld["coef"]),
        "rf_oof_roc_auc": float(models_json["random_forest"]["oof_roc_auc"]),
        "lr_oof_roc_auc": float(models_json["logistic_regression"]["oof_roc_auc"]),
        "rf_oof_pr_auc": float(models_json["random_forest"]["oof_pr_auc"]),
    }

    (root / REPORT_TEX_REL).parent.mkdir(parents=True, exist_ok=True)
    (root / REPORT_TEX_REL).write_text(
        f"""\\documentclass[11pt,a4paper]{{article}}
\\usepackage[a4paper,margin=2.5cm]{{geometry}}
\\usepackage{{setspace}}
\\usepackage{{graphicx}}
\\usepackage{{booktabs}}
\\onehalfspacing
\\begin{{document}}
\\title{{Spatial Equity of Public Facilities and Public EV Charging Access in Macao SAR}}
\\author{{CS482 Group X}}
\\maketitle
\\newpage
\\section{{Introduction and Research Questions}}
We analyze 10-minute and 15-minute pedestrian accessibility across 681,293 residents and 82,751
residents aged 65+ spanning 23 terrestrial zones, 7 official parishes, and 280 H3 res-9 cells
with 84 CEM public EV stations (1,610 unique chargers).
\\section{{Results and Discussion}}
15-minute EV Gini is {headline["ev_gini_15"]:.3f}; Park Gini is {headline["park_gini_15"]:.3f};
H3 N=23 OLS-HC3 elderly\\_share coefficient is {headline["h3_elderly_coef"]:.3f};
Logistic Regression OOF ROC-AUC is {headline["lr_oof_roc_auc"]:.3f};
Random Forest OOF ROC-AUC is {headline["rf_oof_roc_auc"]:.3f} and
OOF PR-AUC is {headline["rf_oof_pr_auc"]:.3f}.
\\begin{{figure}}[htbp]
\\centering
\\includegraphics[width=\\textwidth]{{../outputs/figures/fig1_accessibility_lisa.png}}
\\caption{{Figure 1: Dual-panel 15-minute composite accessibility and FDR LISA clusters.}}
\\end{{figure}}
\\input{{../outputs/tables/table1_summary.tex}}
\\newpage
\\bibliographystyle{{plain}}
\\bibliography{{references}}
\\section*{{Contribution Statement}}
Team member contributions documented.
\\section*{{AI Usage Declaration}}
AI tools, planning/coding/writing purposes, and independent verification
documented \\cite{{kassis2026kdense}}.
\\end{{document}}
""",
        encoding="utf-8",
    )

    (root / REPORT_BIB_REL).write_text(
        """@article{dai2011racial,
  author = {Dai, Dajun},
  title = {Racial/ethnic and socioeconomic disparities in urban green space accessibility},
  journal = {Landscape and Urban Planning},
  year = {2011},
  volume = {102},
  pages = {234--244}
}

@misc{dsec2022census,
  author = {{DSEC}},
  title = {Results of 2021 Population Census of Macao SAR},
  year = {2022},
  howpublished = {Statistics and Census Service, Macao SAR}
}

@misc{cem2026ev,
  author = {{CEM}},
  title = {Public EV Charging Station Map Data API},
  year = {2026},
  howpublished = {Companhia de Electricidade de Macau}
}

@misc{osm2026macau,
  author = {{OpenStreetMap Contributors} and {Geofabrik}},
  title = {Macau SAR Geospatial Extract},
  year = {2026}
}

@article{kassis2026kdense,
  author = {Kassis, Timothy and others},
  title = {K-Dense: Automated Scientific Discovery and Verification},
  journal = {arXiv preprint arXiv:2609.00065},
  year = {2026}
}
""",
        encoding="utf-8",
    )

    pdf_pages = _build_synthetic_pdf_pages(headline, main_text_pages_mode="valid")
    write_pdf(
        root / REPORT_PDF_REL,
        pdf_pages,
        page_width=595.28,
        page_height=841.89,
        line_pitch=18.0,
        font_size=11.0,
        embed_image_on_page=3,
    )

    (root / REPORT_AUDIT_REL).write_text(
        f"""# 12/03 Go/No-Go Checkpoint & Numeric Traceability Audit

- **Verdict**: **GO** (PASS)
- **CS482 Minimum Requirements**: MR1 PASS, MR2 PASS, MR3 PASS, MR4 PASS, MR5 PASS
- **Main-Text Page Count**: 3.55 A4 pages (within 3.0–3.8 Phase A budget; 11pt, 1.5 spacing)
- **Frozen Bundle Checksums**: 12/12 core frozen artifacts verified in `FREEZE_MANIFEST.json`

## Verified Headline Numeric Traceability
- Terrestrial 2021 Census Total Population: `681,293` (aged 65+: `82,751`)
- Spatial Units: `23` zones, `7` parishes, `280` H3 res-9 cells (`56` EV-positive, 20.0%)
- Public EV Charging Supply: `84` stations, `1,610` unique physical chargers
- H1 15-min EV Gini: `{headline["ev_gini_15"]:.3f}`; Park Gini: `{headline["park_gini_15"]:.3f}`
- H3 Zone Regression (N=23) OLS-HC3 `elderly_share` coef: `{headline["h3_elderly_coef"]:.3f}`
- Phase A ML Baselines (5-fold Spatial Block CV):
  - Logistic Regression OOF ROC-AUC: `{headline["lr_oof_roc_auc"]:.3f}`
  - Random Forest OOF ROC-AUC: `{headline["rf_oof_roc_auc"]:.3f}`,
    OOF PR-AUC: `{headline["rf_oof_pr_auc"]:.3f}`
  - All 4 Phase B comparison metrics (ROC-AUC, PR-AUC, Brier, Top-20% Hit Rate) verified.
""",
        encoding="utf-8",
    )

    return root


# ---------------------------------------------------------------------------
# 18 Atomic Single-Invariant Fault Mutators (for Empirical Discrimination Tests)
# ---------------------------------------------------------------------------


def _rewrite_gpkg_from_layer(
    path: Path,
    *,
    srs_id: int | None = None,
    row_mutator: Callable[[list[dict[str, Any]]], list[dict[str, Any]]] | None = None,
    geom_mutator: Callable[[list[tuple[str, Any]]], list[tuple[str, Any]]] | None = None,
) -> None:
    layer = read_gpkg(path)
    rows = [dict(r) for r in layer.rows]
    geoms: list[tuple[str, Any]] = []
    for g in layer.geoms:
        if g.point_xy is not None:
            geoms.append(("POINT", g.point_xy))
        elif g.polygons:
            if len(g.polygons) == 1:
                _, outer, holes = g.polygons[0]
                geoms.append(("POLYGON", [list(outer), *[list(h) for h in holes]]))
            else:
                geoms.append(
                    (
                        "MULTIPOLYGON",
                        [
                            [list(outer), *[list(h) for h in holes]]
                            for _, outer, holes in g.polygons
                        ],
                    )
                )
    if row_mutator is not None:
        rows = row_mutator(rows)
    if geom_mutator is not None:
        geoms = geom_mutator(geoms)

    # Infer SQLite column types cleanly from existing values
    col_types: list[tuple[str, str]] = []
    for col in layer.columns:
        sample = next((r[col] for r in rows if r.get(col) is not None), None)
        if isinstance(sample, int) and not isinstance(sample, bool):
            col_types.append((col, "INTEGER"))
        elif isinstance(sample, float):
            col_types.append((col, "REAL"))
        else:
            col_types.append((col, "TEXT"))

    write_gpkg(
        path,
        layer.table_name,
        col_types,
        rows,
        geoms,
        geom_type=layer.geometry_type_name,
        srs_id=layer.srs_id if srs_id is None else srs_id,
    )


def _mut_wrong_crs_epsg4326(root: Path) -> None:
    def _to_lonlat(geoms: list[tuple[str, Any]]) -> list[tuple[str, Any]]:
        out = []
        for kind, data in geoms:
            if kind == "POLYGON":
                out.append(
                    (
                        "POLYGON",
                        [
                            [
                                (
                                    113.54 + (x - 762_000.0) / 100_000.0,
                                    22.19 + (y - 2_456_000.0) / 111_000.0,
                                )
                                for x, y in ring
                            ]
                            for ring in data
                        ],
                    )
                )
            else:
                out.append((kind, data))
        return out

    _rewrite_gpkg_from_layer(root / "data/frozen/cells.gpkg", srs_id=4326, geom_mutator=_to_lonlat)


def _mut_crs_datum_315m_shift(root: Path) -> None:
    # Shift zones_23 NW inside Macao UTM49N bbox so coarse bbox passes but cross-layer fails
    def _shift(geoms: list[tuple[str, Any]]) -> list[tuple[str, Any]]:
        return [
            ("POLYGON", [[(x - 900.0, y + 375.0) for x, y in ring] for ring in data])
            for _kind, data in geoms
        ]

    _rewrite_gpkg_from_layer(root / "data/frozen/zones_23.gpkg", geom_mutator=_shift)


def _mut_cem_latlon_swapped(root: Path) -> None:
    layer = read_gpkg(root / "data/frozen/facilities_clean.gpkg")
    ev_indices = {i for i, r in enumerate(layer.rows) if r["facility_type"] == "ev_charger"}

    def _swap(geoms: list[tuple[str, Any]]) -> list[tuple[str, Any]]:
        return [("POINT", (22.19, 113.54)) if i in ev_indices else g for i, g in enumerate(geoms)]

    _rewrite_gpkg_from_layer(root / "data/frozen/facilities_clean.gpkg", geom_mutator=_swap)


def _mut_maritime_z26_included(root: Path) -> None:
    def _add_z26_row(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        extra = dict(rows[-1])
        extra["zone_id"] = "Z26"
        extra["zone_name"] = "水域 Maritime"
        extra["pop_total"] = 777.0
        extra["pop_65plus"] = 61.0
        return [*rows, extra]

    def _add_z26_geom(geoms: list[tuple[str, Any]]) -> list[tuple[str, Any]]:
        return [*geoms, ("POLYGON", [_rect_ring(766_000.0, 2_454_000.0, 767_000.0, 2_455_000.0)])]

    _rewrite_gpkg_from_layer(
        root / "data/frozen/zones_23.gpkg",
        row_mutator=_add_z26_row,
        geom_mutator=_add_z26_geom,
    )


def _mut_zero_pop_and_pop_drift(root: Path) -> None:
    def _corrupt_pop(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        for i, r in enumerate(rows):
            if i == 0:
                r["pop_total"] = 0.0
                r["pop_65plus"] = 0.0
            else:
                r["pop_total"] = float(r["pop_total"]) * 1.02
        return rows

    _rewrite_gpkg_from_layer(root / "data/frozen/cells.gpkg", row_mutator=_corrupt_pop)


def _mut_ev_capacity_3x_double_count(root: Path) -> None:
    def _triple(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        for r in rows:
            if r["facility_type"] == "ev_charger":
                r["capacity"] = float(r["capacity"]) * 3.0
        return rows

    _rewrite_gpkg_from_layer(root / "data/frozen/facilities_clean.gpkg", row_mutator=_triple)


def _mut_ev_ml_feature_leak(root: Path) -> None:
    fm_path = root / "data/frozen/feature_manifest.json"
    fm = read_json_strict(fm_path)
    fm["ml_features"].append("ev_charger_capacity")
    _write_json(fm_path, fm)

    ml_path = root / "outputs/metrics/phase_a_ml_baselines.json"
    ml = read_json_strict(ml_path)
    ml["feature_columns"].append("ev_charger_capacity")
    _write_json(ml_path, ml)


def _mut_poi_text_ev_leak(root: Path) -> None:
    path = root / "data/frozen/cell_poi_texts.csv"
    headers, rows = read_csv_strict(path)
    rows[1]["poi_text"] = "shopping plaza CEM 充電站 Tesla Supercharger"
    _write_csv(path, headers, rows)


def _mut_graph_edge_zero_or_gt10(root: Path) -> None:
    path = root / "data/frozen/graph_edges.csv"
    headers, rows = read_csv_strict(path)
    rows[0]["walk_time_min"] = "0.0"
    rows[1]["walk_time_min"] = "12.5"
    rows.pop()  # break undirected symmetry as well
    _write_csv(path, headers, rows)


def _mut_graph_dropped_islands(root: Path) -> None:
    cells = read_gpkg(root / "data/frozen/cells.gpkg")
    island_ids = {r["cell_id"] for r in cells.rows if r["parish_id"] in {"P06", "P07"}}
    path = root / "data/frozen/graph_edges.csv"
    headers, rows = read_csv_strict(path)
    kept = [
        r for r in rows if r["src_cell_id"] not in island_ids and r["dst_cell_id"] not in island_ids
    ]
    _write_csv(path, headers, kept)


def _mut_fold_empty_positive_class(root: Path) -> None:
    cells = read_gpkg(root / "data/frozen/cells.gpkg")
    pos_in_blk4 = {
        r["cell_id"]
        for r in cells.rows
        if int(r["spatial_block"]) == 4 and int(r["has_ev_charger"]) == 1
    }

    def _move_blk(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        for r in rows:
            if r["cell_id"] in pos_in_blk4:
                r["spatial_block"] = 0
        return rows

    _rewrite_gpkg_from_layer(root / "data/frozen/cells.gpkg", row_mutator=_move_blk)

    folds_path = root / "data/frozen/spatial_folds.csv"
    headers, frows = read_csv_strict(folds_path)
    for r in frows:
        if r["cell_id"] in pos_in_blk4:
            r["spatial_block"] = "0"
    _write_csv(folds_path, headers, frows)

    sm_path = root / "data/frozen/split_metadata.json"
    sm = read_json_strict(sm_path)
    moved = len(pos_in_blk4)
    sm["blocks"]["4"]["n_pos_ev"] = 0
    sm["blocks"]["4"]["n_cells"] -= moved
    sm["blocks"]["0"]["n_pos_ev"] += moved
    sm["blocks"]["0"]["n_cells"] += moved
    _write_json(sm_path, sm)


def _mut_theil_additivity_broken(root: Path) -> None:
    path = root / "outputs/tables/h1_inequality_lisa.csv"
    headers, rows = read_csv_strict(path)
    rows[0]["theil_within"] = str(float(rows[0]["theil_within"]) + 0.015)
    _write_csv(path, headers, rows)


def _mut_h2_diff_and_holm_broken(root: Path) -> None:
    path = root / "outputs/tables/h2_gini_diff_bootstrap.csv"
    headers, rows = read_csv_strict(path)
    rows[0]["gini_diff"] = str(float(rows[0]["gini_diff"]) + 0.05)
    rows[0]["p_holm"] = str(float(rows[0]["p_boot_raw"]) * 0.5)
    _write_csv(path, headers, rows)


def _mut_h3_pretest_switching(root: Path) -> None:
    path = root / "outputs/tables/h3_spatial_regression.csv"
    headers, rows = read_csv_strict(path)
    kept = [r for r in rows if r["model_type"] != "SEM_ML"]
    _write_csv(path, headers, kept)


def _mut_freeze_sha256_tampered(root: Path) -> None:
    path = root / "data/frozen/graph_edges.csv"
    with path.open("a", encoding="utf-8") as fh:
        fh.write(" ")


def _mut_table1_numeric_drift(root: Path) -> None:
    csv_path = root / "outputs/tables/table1_summary.csv"
    headers, rows = read_csv_strict(csv_path)
    old_val = rows[0]["value"]
    new_val = f"{float(old_val) + 0.08:.4f}"
    rows[0]["value"] = new_val
    _write_csv(csv_path, headers, rows)
    for ext in ("md", "tex"):
        p = root / f"outputs/tables/table1_summary.{ext}"
        p.write_text(p.read_text(encoding="utf-8").replace(old_val, new_val, 1), encoding="utf-8")


def _mut_pdf_out_of_budget_and_us_letter(root: Path) -> None:
    headline = {
        "ev_gini_15": 0.450,
        "park_gini_15": 0.250,
        "h3_elderly_coef": -0.120,
        "rf_oof_roc_auc": 0.720,
        "lr_oof_roc_auc": 0.700,
        "rf_oof_pr_auc": 0.410,
    }
    short_pages = _build_synthetic_pdf_pages(headline, main_text_pages_mode="short_us_letter")
    write_pdf(
        root / REPORT_PDF_REL,
        short_pages,
        page_width=612.0,  # US Letter instead of A4
        page_height=792.0,
        line_pitch=18.0,
        font_size=11.0,
        embed_image_on_page=None,
    )


def _mut_figure_low_dpi_150(root: Path) -> None:
    write_png(root / "outputs/figures/fig1_accessibility_lisa.png", 2400, 1200, dpi=150, seed=999)


def _mut_graph_edge_and_island_drop(root: Path) -> None:
    _mut_graph_dropped_islands(root)
    _mut_graph_edge_zero_or_gt10(root)


def _mut_table1_and_pdf_report_violations(root: Path) -> None:
    _mut_table1_numeric_drift(root)
    _mut_pdf_out_of_budget_and_us_letter(root)
    _mut_figure_low_dpi_150(root)


FAULT_MUTATORS: dict[str, Callable[[Path], None]] = {
    "wrong_crs_epsg4326": _mut_wrong_crs_epsg4326,
    "crs_datum_315m_shift": _mut_crs_datum_315m_shift,
    "cem_latlon_swapped": _mut_cem_latlon_swapped,
    "maritime_z26_included": _mut_maritime_z26_included,
    "zero_pop_and_pop_drift": _mut_zero_pop_and_pop_drift,
    "ev_capacity_3x_double_count": _mut_ev_capacity_3x_double_count,
    "ev_ml_feature_leak": _mut_ev_ml_feature_leak,
    "poi_text_ev_leak": _mut_poi_text_ev_leak,
    "graph_edge_zero_or_gt10": _mut_graph_edge_zero_or_gt10,
    "graph_dropped_islands": _mut_graph_dropped_islands,
    "graph_edge_and_island_drop": _mut_graph_edge_and_island_drop,
    "fold_empty_positive_class": _mut_fold_empty_positive_class,
    "theil_additivity_broken": _mut_theil_additivity_broken,
    "h2_diff_and_holm_broken": _mut_h2_diff_and_holm_broken,
    "h3_pretest_switching": _mut_h3_pretest_switching,
    "freeze_sha256_tampered": _mut_freeze_sha256_tampered,
    "table1_numeric_drift": _mut_table1_numeric_drift,
    "pdf_out_of_budget_and_us_letter": _mut_pdf_out_of_budget_and_us_letter,
    "figure_low_dpi_150": _mut_figure_low_dpi_150,
    "table1_and_pdf_report_violations": _mut_table1_and_pdf_report_violations,
}


def apply_fault(root: Path, fault_id: str) -> None:
    """Apply a named single-fault mutation to a synthetic Macao project root."""
    if fault_id not in FAULT_MUTATORS:
        raise KeyError(f"Unknown fault_id {fault_id!r}; valid IDs: {sorted(FAULT_MUTATORS)}")
    FAULT_MUTATORS[fault_id](Path(root))
