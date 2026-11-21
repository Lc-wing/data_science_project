"""Tier 4 E2E Tests: 9 real-world stakeholder, Phase B consumer, and CLI replay scenarios."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any

from e2e_lib.contracts import (
    A4_HEIGHT_PT,
    A4_TOL_PT,
    A4_WIDTH_PT,
    ANALYTICAL_PLACEHOLDER_RE,
    EV_COL_LEAK_RE,
    EV_TEXT_LEAK_MULTILINGUAL_RE,
    FACILITY_TYPES,
    FROZEN_CANONICAL_FILES,
    FROZEN_CORE_HASHED_FILES,
    GLOBAL_SEED,
    H3_MODEL_TYPES,
    ML_METRICS,
    ML_MODEL_NAMES,
    PDF_MAIN_PAGES_MAX,
    PDF_MAIN_PAGES_MIN,
    PNG_MIN_BYTES,
    PNG_MIN_DPI,
    PNG_MIN_PIXEL_STD,
    POP_TOL_H3_TOTAL,
    REQUIRED_FIGURES,
    SECRET_TOKEN_RE,
    SHA256_RE,
    TABLE1_FILES,
    TABLE_PLACEHOLDER_RE,
    TAIPA_PARISH_RE,
    TARGET_EPSG,
    normalize_resolution_label,
    normalize_weights_label,
)
from e2e_lib.oracle import (
    average_precision_score,
    brier_score,
    fit_predict_ridge_prob,
    roc_auc_score,
    sentence_aware_unhedged_causal_hits,
    spearman_corr,
    top20_hit_rates,
)
from e2e_lib.readers import (
    GpkgLayer,
    PdfInfo,
    PngInfo,
    as_bool,
    as_float,
    file_sha256,
    read_csv_strict,
    read_gpkg,
    read_json_strict,
)


def _extract_freeze_file_map(manifest: dict[str, Any]) -> dict[str, str]:
    raw_map: dict[str, Any] = {}
    for key in ("files", "checksums", "sha256", "artifacts"):
        val = manifest.get(key)
        if isinstance(val, dict):
            raw_map = val
            break
    if not raw_map:
        raw_map = {
            k: v
            for k, v in manifest.items()
            if isinstance(v, str) and SHA256_RE.match(v.strip().lower())
        }
    out: dict[str, str] = {}
    for rel, val in raw_map.items():
        if isinstance(val, dict):
            digest = str(val.get("sha256", "")).strip().lower()
        else:
            digest = str(val).strip().lower()
        norm_rel = rel if rel.startswith("data/") else f"data/frozen/{rel}"
        out[norm_rel] = digest
    return out


def _extract_provenance_entries(prov_data: Any) -> list[dict[str, Any]]:
    if isinstance(prov_data, list):
        return [e for e in prov_data if isinstance(e, dict)]
    if isinstance(prov_data, dict):
        for key in ("sources", "datasets", "files", "entries"):
            val = prov_data.get(key)
            if isinstance(val, list):
                return [e for e in val if isinstance(e, dict)]
            if isinstance(val, dict):
                return [
                    {"name": k, **v} if isinstance(v, dict) else {"name": k, "value": v}
                    for k, v in val.items()
                ]
    return []


def _extract_floats(text: str) -> list[float]:
    return [float(m.group(0)) for m in re.finditer(r"-?\d+\.\d+", text)]


def test_scenario_01_cs482_instructor_grading_audit(
    report_pdf_info: PdfInfo,
    report_tex_text: str,
    report_bib_text: str,
    h1_table: tuple[list[str], list[dict[str, str]]],
    h3_table: tuple[list[str], list[dict[str, str]]],
    ml_baselines: dict[str, Any],
) -> None:
    """S1: CS482 instructor grading the single submitted PDF against the 100-pt rubric & MRs."""
    assert not report_pdf_info.encrypted
    assert report_pdf_info.total_pages >= 4
    for p_idx, (w, h) in enumerate(report_pdf_info.page_sizes_pt, start=1):
        assert abs(w - A4_WIDTH_PT) <= A4_TOL_PT and abs(h - A4_HEIGHT_PT) <= A4_TOL_PT, (
            f"Page {p_idx} is not A4 ({w}x{h} pt)"
        )
    assert report_pdf_info.single_column_ratio >= 0.65
    assert 14.0 <= report_pdf_info.median_line_pitch_pt <= 23.0
    assert PDF_MAIN_PAGES_MIN <= report_pdf_info.main_fractional_pages <= PDF_MAIN_PAGES_MAX
    assert report_pdf_info.fonts, "No fonts detected in compiled PDF"
    assert report_pdf_info.embedded_image_count >= 1 or "fig1_accessibility_lisa" in report_tex_text

    main_low = report_pdf_info.main_text.lower()
    # All 5 CS482 Minimum Requirements
    for req_tok in (
        "h1",
        "h2",
        "h3",
        "dsec",
        "cem",
        "32649",
        "2sfca",
        "gini",
        "theil",
        "lisa",
        "ols",
        "sem",
        "slm",
        "bootstrap",
        "fdr",
        "holm",
        "maup",
        "roc",
        "brier",
        "figure 1",
        "table 1",
    ):
        assert req_tok in main_low, f"Instructor audit: missing required topic {req_tok!r}"

    # All 4 excluded sections + K-Dense citation
    assert report_pdf_info.cover_pages and report_pdf_info.cover_text
    assert report_pdf_info.references_text
    assert report_pdf_info.contribution_text
    assert report_pdf_info.ai_declaration_text
    assert "2609.00065" in (report_pdf_info.references_text + "\n" + report_bib_text).lower()

    # Correlational framing & all 4 limitations
    assert any(w in main_low for w in ("correlational", "associational", "associated with"))
    assert not sentence_aware_unhedged_causal_hits(report_pdf_info.main_text)
    assert any(w in main_low for w in ("high-rise", "building-height", "dasymetric"))
    assert any(w in main_low for w in ("capacity metadata", "9488798", "imputed"))
    assert any(w in main_low for w in ("slope", "topograph", "footbridge", "overpass"))
    assert "ecological fallacy" in main_low

    # Spot-check >= 5 headline numbers against CSV/JSON
    _, h1_rows = h1_table
    ev_g15 = next(
        as_float(r["gini"])
        for r in h1_rows
        if normalize_resolution_label(r["resolution"]) == "res9"
        and int(float(r["threshold_min"])) == 15
        and r["facility_type"].strip() == "ev_charger"
    )
    park_g15 = next(
        as_float(r["gini"])
        for r in h1_rows
        if normalize_resolution_label(r["resolution"]) == "res9"
        and int(float(r["threshold_min"])) == 15
        and r["facility_type"].strip() == "park"
    )
    _, h3_rows = h3_table
    h3_eld = next(
        as_float(r["coef"])
        for r in h3_rows
        if int(float(r["threshold_min"])) == 15
        and r["model_type"].strip() == "OLS_HC3"
        and r["term"].strip().lower() == "elderly_share"
    )
    rf_auc = as_float(ml_baselines["models"]["random_forest"]["oof_roc_auc"])
    lr_auc = as_float(ml_baselines["models"]["logistic_regression"]["oof_roc_auc"])

    assert "681,293" in report_pdf_info.main_text or "681293" in report_pdf_info.main_text
    assert "23" in report_pdf_info.main_text
    pdf_floats = _extract_floats(report_pdf_info.main_text)
    for target in (ev_g15, park_g15, h3_eld, rf_auc, lr_auc):
        assert min(abs(v - target) for v in pdf_floats) <= 0.005


def test_scenario_02_go_nogo_checkpoint_gate(
    project_root: Path,
    report_pdf_info: PdfInfo,
    freeze_manifest: dict[str, Any],
    ml_baselines: dict[str, Any],
    go_nogo_audit_text: str,
) -> None:
    """S2: Project lead executing the binding 12/03 Go/No-Go checkpoint gate."""
    # 1. >= 3.0 A4 pages complete draft covering all 5 MRs
    assert PDF_MAIN_PAGES_MIN <= report_pdf_info.main_fractional_pages <= PDF_MAIN_PAGES_MAX

    # 2. All 13 canonical frozen files present and SHA-256 verified
    for rel in FROZEN_CANONICAL_FILES:
        p = project_root / rel
        assert p.is_file() and p.stat().st_size > 0, f"Missing frozen deliverable: {rel}"
    fmap = _extract_freeze_file_map(freeze_manifest)
    for rel in FROZEN_CORE_HASHED_FILES:
        assert rel in fmap and file_sha256(project_root / rel) == fmap[rel]

    # 3. ML baseline JSON supplies all 4 Phase B comparison metrics across all 3 models
    for m_name in ML_MODEL_NAMES:
        m = ml_baselines["models"][m_name]
        for met in ML_METRICS:
            assert 0.0 <= as_float(m[f"mean_{met}"]) <= 1.0
            assert 0.0 <= as_float(m[f"oof_{met}"]) <= 1.0

    # 4. Unambiguous GO in report/go_nogo_audit.md
    low = go_nogo_audit_text.lower()
    assert re.search(r"\b(go|pass)\b", low)
    assert not re.search(r"\b((?<!go/)no-go|fail|tbd_num|todo)\b", low)


def test_scenario_03_phase_b_route_a_gnn_consumer_dry_run(
    cells_r9: GpkgLayer,
    graph_edges: tuple[list[str], list[dict[str, str]]],
    spatial_folds: tuple[list[str], list[dict[str, str]]],
    feature_manifest: dict[str, Any],
    ml_baselines: dict[str, Any],
) -> None:
    """S3: Phase B Route A GNN researcher dry-run (inductive & transductive masks + 1-hop GNN)."""
    ml_feats = list(feature_manifest["ml_features"])
    assert ml_feats and not any(EV_COL_LEAK_RE.search(c) for c in ml_feats)

    node_ids = [str(r["cell_id"]) for r in cells_r9.rows]
    node_idx = {cid: i for i, cid in enumerate(node_ids)}
    n_nodes = len(node_ids)

    _, fold_rows = spatial_folds
    fold_map = {r["cell_id"]: int(float(r["spatial_block"])) for r in fold_rows}
    folds = [fold_map[cid] for cid in node_ids]
    y = [int(r["has_ev_charger"]) for r in cells_r9.rows]
    x_raw = [[as_float(r[c]) for c in ml_feats] for r in cells_r9.rows]

    _, edge_rows = graph_edges
    edges_parsed: list[tuple[int, int, float, bool]] = []
    for r in edge_rows:
        u = node_idx[r["src_cell_id"]]
        v = node_idx[r["dst_cell_id"]]
        w = as_float(r["weight"])
        sb = as_bool(r["same_block"])
        if sb:
            assert folds[u] == folds[v], "same_block=True edge crosses spatial blocks!"
        else:
            assert folds[u] != folds[v], "same_block=False edge lies within same block!"
        edges_parsed.append((u, v, w, sb))

    oof_gnn_probs = [0.0] * n_nodes
    for test_fold in range(5):
        train_mask = [f != test_fold for f in folds]
        test_mask = [f == test_fold for f in folds]

        # Inductive subgraph: strictly train-to-train edges
        inductive_edges = [
            (u, v, w) for u, v, w, _ in edges_parsed if train_mask[u] and train_mask[v]
        ]
        assert len(inductive_edges) > 0, f"Empty inductive training graph on fold {test_fold}"
        assert all(folds[u] != test_fold and folds[v] != test_fold for u, v, _ in inductive_edges)

        # Build 1-hop message passing features H = concat(X, D^{-1} A X) using train-only stats
        tr_indices = [i for i, m in enumerate(train_mask) if m]
        te_indices = [i for i, m in enumerate(test_mask) if m]
        y_tr = [y[i] for i in tr_indices]
        y_te = [y[i] for i in te_indices]
        assert sum(y_tr) >= 1 and sum(y_te) >= 1

        # Inductive neighbor aggregation for train; transductive full-graph for test evaluation
        adj_ind: dict[int, list[tuple[int, float]]] = defaultdict(list)
        for u, v, w in inductive_edges:
            adj_ind[u].append((v, w))
        adj_full: dict[int, list[tuple[int, float]]] = defaultdict(list)
        for u, v, w, _ in edges_parsed:
            adj_full[u].append((v, w))

        d_feat = len(ml_feats)

        def _agg_features(
            indices: list[int], adj_map: dict[int, list[tuple[int, float]]]
        ) -> list[list[float]]:
            out: list[list[float]] = []
            for idx in indices:
                nbrs = adj_map.get(idx, [])
                if not nbrs:
                    msg = list(x_raw[idx])
                else:
                    w_sum = sum(w for _, w in nbrs)
                    msg = [sum(w * x_raw[nbr][j] for nbr, w in nbrs) / w_sum for j in range(d_feat)]
                out.append([*x_raw[idx], *msg])
            return out

        h_tr = _agg_features(tr_indices, adj_ind)
        h_te = _agg_features(te_indices, adj_full)
        fold_probs = fit_predict_ridge_prob(h_tr, y_tr, h_te, alpha=3.0)
        for loc_i, node_i in enumerate(te_indices):
            oof_gnn_probs[node_i] = fold_probs[loc_i]

    gnn_roc = roc_auc_score(y, oof_gnn_probs)
    gnn_pr = average_precision_score(y, oof_gnn_probs)
    gnn_brier = brier_score(y, oof_gnn_probs)
    gnn_rec20, gnn_prec20 = top20_hit_rates(y, oof_gnn_probs)
    assert 0.0 <= gnn_roc <= 1.0 and 0.0 <= gnn_pr <= 1.0 and 0.0 <= gnn_brier <= 1.0
    assert 0.0 <= gnn_rec20 <= 1.0 and 0.0 <= gnn_prec20 <= 1.0
    assert int(ml_baselines["n_cells"]) == n_nodes


def test_scenario_04_phase_b_route_b_tag_dry_run(
    cells_r9: GpkgLayer,
    poi_texts: tuple[list[str], list[dict[str, str]]],
    spatial_folds: tuple[list[str], list[dict[str, str]]],
    feature_manifest: dict[str, Any],
) -> None:
    """S4: Phase B Route B Text-Attributed Graph (TAG) researcher dry-run."""
    _, poi_rows = poi_texts
    poi_by_id = {r["cell_id"]: r for r in poi_rows}
    _, fold_rows = spatial_folds
    fold_by_id = {r["cell_id"]: int(float(r["spatial_block"])) for r in fold_rows}

    cell_ids = [str(r["cell_id"]) for r in cells_r9.rows]
    assert set(cell_ids) == set(poi_by_id.keys()) == set(fold_by_id.keys())

    # Expanded multilingual EV-leakage scan (EN + ZH + PT) & non-trivial coverage per fold
    vocab_counter: Counter[str] = Counter()
    non_empty_per_fold: dict[int, int] = defaultdict(int)
    texts_aligned: list[str] = []
    for cid in cell_ids:
        txt = poi_by_id[cid]["poi_text"].strip()
        hit = EV_TEXT_LEAK_MULTILINGUAL_RE.search(txt)
        assert hit is None, f"Multilingual EV text leak in {cid}: {hit.group(0)!r}"
        texts_aligned.append(txt)
        tokens = re.findall(r"[a-zA-Z\u4e00-\u9fff]{2,}", txt.lower())
        vocab_counter.update(tokens)
        if txt:
            non_empty_per_fold[fold_by_id[cid]] += 1

    assert sum(non_empty_per_fold.values()) / len(cell_ids) >= 0.50
    assert len(vocab_counter) >= 15, f"POI text vocabulary diversity too low: {len(vocab_counter)}"
    for blk in range(5):
        assert non_empty_per_fold[blk] >= 1, f"Spatial fold {blk} has 0 non-empty POI text cells"

    # Leak-free train-fold-only bag-of-words + tabular features dry-run across 5 folds
    ml_feats = list(feature_manifest["ml_features"])
    x_tab = [[as_float(r[c]) for c in ml_feats] for r in cells_r9.rows]
    y = [int(r["has_ev_charger"]) for r in cells_r9.rows]
    folds = [fold_by_id[cid] for cid in cell_ids]
    oof_probs = [0.0] * len(cell_ids)

    for test_fold in range(5):
        tr_idx = [i for i, f in enumerate(folds) if f != test_fold]
        te_idx = [i for i, f in enumerate(folds) if f == test_fold]

        # Build vocabulary strictly from training fold texts
        tr_Counter: Counter[str] = Counter()
        for i in tr_idx:
            tr_Counter.update(re.findall(r"[a-zA-Z\u4e00-\u9fff]{2,}", texts_aligned[i].lower()))
        top_vocab = [w for w, _ in tr_Counter.most_common(12)]

        def _featurize(indices: list[int]) -> list[list[float]]:
            rows_out: list[list[float]] = []
            for i in indices:
                toks = Counter(re.findall(r"[a-zA-Z\u4e00-\u9fff]{2,}", texts_aligned[i].lower()))
                bow = [float(toks.get(w, 0)) for w in top_vocab]
                rows_out.append([*x_tab[i], *bow])
            return rows_out

        x_tr = _featurize(tr_idx)
        x_te = _featurize(te_idx)
        y_tr = [y[i] for i in tr_idx]
        probs = fit_predict_ridge_prob(x_tr, y_tr, x_te, alpha=4.0)
        for loc_i, orig_i in enumerate(te_idx):
            oof_probs[orig_i] = probs[loc_i]

    assert all(0.0 <= p <= 1.0 for p in oof_probs)
    assert 0.0 <= roc_auc_score(y, oof_probs) <= 1.0
    assert 0.0 <= average_precision_score(y, oof_probs) <= 1.0


def test_scenario_05_reproducible_cli_replay_sandbox(
    project_root: Path,
    tmp_path: Path,
) -> None:
    """S5: Sandboxed execution of scripts/run_phase_a.py --skip-download with zero live mutation."""
    # 1. Pre-replay SHA-256 snapshot of live project_root/data/frozen/
    live_frozen_dir = project_root / "data" / "frozen"
    pre_live_hashes = {
        p.relative_to(project_root).as_posix(): file_sha256(p)
        for p in sorted(live_frozen_dir.glob("*"))
        if p.is_file()
    }
    pre_manifest = read_json_strict(live_frozen_dir / "FREEZE_MANIFEST.json")
    pre_fmap = _extract_freeze_file_map(pre_manifest)
    for rel, exp_hash in pre_fmap.items():
        assert pre_live_hashes.get(rel) == exp_hash, f"Pre-replay hash mismatch on {rel}"

    # 2. Stage isolated sandbox copy
    sandbox = tmp_path / "phase_a_replay_sandbox"
    shutil.copytree(
        project_root,
        sandbox,
        ignore=shutil.ignore_patterns(
            ".venv",
            ".git",
            ".agents",
            "__pycache__",
            ".pytest_cache",
            ".ruff_cache",
        ),
    )

    # 3. Execute scripts/run_phase_a.py --skip-download in sandbox
    venv_py = project_root / ".venv" / "bin" / "python"
    python_bin = str(venv_py) if venv_py.is_file() else sys.executable
    env = dict(os.environ)
    env["MPLBACKEND"] = "Agg"
    env["MACAO_EQUITY_ROOT"] = str(sandbox)
    existing_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        f"{sandbox / 'src'}{os.pathsep}{existing_pp}" if existing_pp else str(sandbox / "src")
    )
    env["SOURCE_DATE_EPOCH"] = "1763596800"

    proc = subprocess.run(
        [python_bin, str(sandbox / "scripts" / "run_phase_a.py"), "--skip-download"],
        cwd=str(sandbox),
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert proc.returncode == 0, (
        f"Sandboxed run_phase_a.py --skip-download failed (exit={proc.returncode}):\n"
        f"STDOUT:\n{proc.stdout[-1500:]}\nSTDERR:\n{proc.stderr[-1500:]}"
    )

    # 4. Assert ZERO mutation of live project_root/data/frozen/
    post_live_hashes = {
        p.relative_to(project_root).as_posix(): file_sha256(p)
        for p in sorted(live_frozen_dir.glob("*"))
        if p.is_file()
    }
    assert post_live_hashes == pre_live_hashes, (
        "Live data/frozen/ files were mutated during replay!"
    )

    # 5. Verify sandbox FREEZE_MANIFEST.json integrity & deterministic reproduction
    sb_manifest = read_json_strict(sandbox / "data/frozen/FREEZE_MANIFEST.json")
    sb_fmap = _extract_freeze_file_map(sb_manifest)
    for rel, sb_sha in sb_fmap.items():
        assert file_sha256(sandbox / rel) == sb_sha, f"Sandbox manifest hash mismatch on {rel}"

    # Deterministic byte-identical text contracts
    for text_rel in (
        "data/frozen/graph_edges.csv",
        "data/frozen/spatial_folds.csv",
        "data/frozen/split_metadata.json",
        "data/frozen/cell_poi_texts.csv",
        "data/frozen/feature_manifest.json",
    ):
        assert file_sha256(sandbox / text_rel) == file_sha256(project_root / text_rel), (
            f"Non-deterministic text contract after replay: {text_rel}"
        )

    # Semantic & numerical determinism on GeoPackages and analytical tables
    live_cells = read_gpkg(project_root / "data/frozen/cells.gpkg")
    sb_cells = read_gpkg(sandbox / "data/frozen/cells.gpkg")
    assert live_cells.epsg == sb_cells.epsg == TARGET_EPSG
    assert len(live_cells.rows) == len(sb_cells.rows)
    assert set(live_cells.columns) == set(sb_cells.columns)

    _, live_h1 = read_csv_strict(project_root / "outputs/tables/h1_inequality_lisa.csv")
    _, sb_h1 = read_csv_strict(sandbox / "outputs/tables/h1_inequality_lisa.csv")
    assert len(live_h1) == len(sb_h1)
    for r1, r2 in zip(live_h1, sb_h1, strict=True):
        assert abs(as_float(r1["gini"]) - as_float(r2["gini"])) <= 1e-6


def test_scenario_06_urban_planner_coldspot_elderly_audit(
    cells_r9: GpkgLayer,
    zones_23: GpkgLayer,
    parishes_7: GpkgLayer,
    h1_table: tuple[list[str], list[dict[str, str]]],
    h2_table: tuple[list[str], list[dict[str, str]]],
    h3_table: tuple[list[str], list[dict[str, str]]],
    load_png: Callable[[str | Path], PngInfo],
    report_pdf_info: PdfInfo,
) -> None:
    """S6: Macao urban planner / DSSCU / DSAT coldspot and elderly-zone equity audit."""
    valid_zones = {str(r["zone_id"]): str(r["zone_name"]) for r in zones_23.rows}
    valid_parishes = {str(r["parish_id"]): str(r["parish_name"]) for r in parishes_7.rows}
    assert any(
        TAIPA_PARISH_RE.search(f"{r['parish_name']} {r['parish_name_zh']}") for r in parishes_7.rows
    )

    # Verify FDR Low-Low (LL) coldspots in cells.gpkg match h1_inequality_lisa.csv & zones/parishes
    _, h1_rows = h1_table
    for r in h1_rows:
        if normalize_resolution_label(r["resolution"]) != "res9":
            continue
        thr = int(float(r["threshold_min"]))
        ft = r["facility_type"].strip()
        c_col = f"lisa_cluster_acc{thr}_{ft}"
        if c_col not in cells_r9.columns:
            continue
        ll_cells = [row for row in cells_r9.rows if str(row[c_col]) == "LL"]
        assert len(ll_cells) == int(float(r["n_ll_fdr"]))
        for cell in ll_cells:
            assert str(cell["zone_id"]) in valid_zones
            assert str(cell["parish_id"]) in valid_parishes
            assert as_float(cell["pop_total"]) > 0.0
        # Theil decomposition additivity
        assert (
            abs(
                as_float(r["theil_total"])
                - (as_float(r["theil_within"]) + as_float(r["theil_between"]))
            )
            < 1e-6
        )

    # H2 paired bootstrap differentials & H3 side-by-side 3-model elderly-zone regressions
    _, h2_rows = h2_table
    assert len(h2_rows) >= 8
    _, h3_rows = h3_table
    models_15 = {
        r["model_type"].strip()
        for r in h3_rows
        if int(float(r["threshold_min"])) == 15 and int(float(r["n_obs"])) == 23
    }
    assert models_15 == H3_MODEL_TYPES

    fig1 = load_png("outputs/figures/fig1_accessibility_lisa.png")
    assert fig1.dpi_x is not None and fig1.dpi_x >= PNG_MIN_DPI and fig1.width > fig1.height

    main_low = report_pdf_info.main_text.lower()
    assert "historical" in main_low and ("field" in main_low or "candidate" in main_low)


def test_scenario_07_academic_integrity_provenance_audit(
    project_root: Path,
    provenance_info: tuple[list[Path], Any],
    freeze_manifest: dict[str, Any],
    feature_manifest: dict[str, Any],
    split_metadata: dict[str, Any],
    h2_table: tuple[list[str], list[dict[str, str]]],
    ml_baselines: dict[str, Any],
    report_bib_text: str,
    report_pdf_info: PdfInfo,
) -> None:
    """S7: Academic integrity reviewer verifying provenance, SHA-256, seeds, K-Dense, AI decl."""
    prov_paths, prov = provenance_info
    entries = _extract_provenance_entries(prov)
    assert len(entries) >= 4
    for p in prov_paths:
        assert not SECRET_TOKEN_RE.search(p.read_text(encoding="utf-8"))

    fmap = _extract_freeze_file_map(freeze_manifest)
    for rel, sha in fmap.items():
        assert file_sha256(project_root / rel) == sha

    _, h2_rows = h2_table
    assert (
        feature_manifest["random_seed"]
        == split_metadata.get("seed", split_metadata.get("random_seed"))
        == ml_baselines["seed"]
        == GLOBAL_SEED
    )
    assert {int(float(r["seed"])) for r in h2_rows} == {GLOBAL_SEED}

    assert not ANALYTICAL_PLACEHOLDER_RE.search(report_bib_text)
    assert "2609.00065" in report_bib_text or "kassis" in report_bib_text.lower()

    ai_low = report_pdf_info.ai_declaration_text.lower()
    assert any(w in ai_low for w in ("tool", "llm", "assistant", "model"))
    assert any(w in ai_low for w in ("plan", "ideation", "coding", "writing"))
    assert any(w in ai_low for w in ("verif", "check", "audit", "test"))

    # Scan all tables/metrics and PDF main_text + references for placeholder/NaN tokens
    for tbl_path in sorted((project_root / "outputs/tables").glob("*")):
        if tbl_path.is_file():
            txt = tbl_path.read_text(encoding="utf-8")
            assert not ANALYTICAL_PLACEHOLDER_RE.search(txt), f"Placeholder in {tbl_path.name}"
    assert not ANALYTICAL_PLACEHOLDER_RE.search(
        report_pdf_info.main_text + "\n" + report_pdf_info.references_text
    )


def test_scenario_08_maup_and_sensitivity_review(
    cells_r9: GpkgLayer,
    cells_r8: GpkgLayer,
    maup_table: tuple[list[str], list[dict[str, str]]],
    h1_table: tuple[list[str], list[dict[str, str]]],
    h2_table: tuple[list[str], list[dict[str, str]]],
    h3_table: tuple[list[str], list[dict[str, str]]],
    h3_diagnostics: dict[str, Any],
    load_png: Callable[[str | Path], PngInfo],
    report_pdf_info: PdfInfo,
) -> None:
    """S8: Quantitative spatial peer review of all 4 sensitivity axes (MAUP, cutoff, EV cap, W)."""
    # 1. Scale (MAUP: H3 r8 vs r9)
    pop_r9 = sum(as_float(r["pop_total"]) for r in cells_r9.rows)
    pop_r8 = sum(as_float(r["pop_total"]) for r in cells_r8.rows)
    assert abs(pop_r9 - pop_r8) / pop_r9 <= POP_TOL_H3_TOTAL
    _, m_rows = maup_table
    assert len(m_rows) >= 5
    maup_png = load_png("outputs/figures/maup_r8_vs_r9.png")
    assert maup_png.dpi_x is not None and maup_png.dpi_x >= PNG_MIN_DPI

    # 2. Travel-time cutoff (10 vs 15 min)
    _, h1_rows = h1_table
    h1_z = {
        (int(float(r["threshold_min"])), r["facility_type"].strip()): as_float(r["zero_acc_pct"])
        for r in h1_rows
        if normalize_resolution_label(r["resolution"]) == "res9"
    }
    for ft in FACILITY_TYPES:
        assert h1_z[(15, ft)] <= h1_z[(10, ft)] + 1e-6
    _, h2_rows = h2_table
    assert {int(float(r["threshold_min"])) for r in h2_rows} == {10, 15}

    # 3. EV capacity weighting (weighted vs unweighted)
    wt = [as_float(r["acc15_ev_charger"]) for r in cells_r9.rows]
    unw = [as_float(r["acc15_ev_unweighted"]) for r in cells_r9.rows]
    assert spearman_corr(wt, unw) > 0.50
    assert max(abs(a - b) for a, b in zip(wt, unw, strict=True)) > 1e-9

    # 4. Spatial weights matrix W (k=4 NN vs bridge-Queen)
    _, h3_rows = h3_table
    w_specs = {normalize_weights_label(r["weights_spec"]) for r in h3_rows}
    assert {"knn4", "bridge_queen"}.issubset(w_specs)
    for w_info in h3_diagnostics.get("weights", {}).values():
        if isinstance(w_info, dict) and "n_islands" in w_info:
            assert int(w_info["n_islands"]) == 0

    assert "maup" in report_pdf_info.main_text.lower()


def test_scenario_09_publication_visuals_and_table1_audit(
    project_root: Path,
    load_png: Callable[[str | Path], PngInfo],
    h1_table: tuple[list[str], list[dict[str, str]]],
    h3_table: tuple[list[str], list[dict[str, str]]],
    ml_baselines: dict[str, Any],
    report_tex_text: str,
    report_pdf_info: PdfInfo,
) -> None:
    """S9: Publication/typesetting audit of 4 300-DPI PNGs, Table 1 formats, and PDF integration."""
    for rel in REQUIRED_FIGURES:
        png = load_png(rel)
        assert png.crc_valid and png.size_bytes >= PNG_MIN_BYTES
        assert png.dpi_x is not None and png.dpi_x >= PNG_MIN_DPI
        assert png.pixel_std > PNG_MIN_PIXEL_STD
        if rel.endswith("fig1_accessibility_lisa.png"):
            assert png.width >= 2000 and png.width > png.height

    for rel in TABLE1_FILES:
        txt = (project_root / rel).read_text(encoding="utf-8")
        assert not TABLE_PLACEHOLDER_RE.search(txt)

    t1_csv_nums = _extract_floats(
        (project_root / "outputs/tables/table1_summary.csv").read_text(encoding="utf-8")
    )
    t1_tex_nums = _extract_floats(
        (project_root / "outputs/tables/table1_summary.tex").read_text(encoding="utf-8")
    )
    pdf_nums = _extract_floats(report_pdf_info.main_text)

    _, h1_rows = h1_table
    ev_g15 = next(
        as_float(r["gini"])
        for r in h1_rows
        if normalize_resolution_label(r["resolution"]) == "res9"
        and int(float(r["threshold_min"])) == 15
        and r["facility_type"].strip() == "ev_charger"
    )
    _, h3_rows = h3_table
    h3_eld = next(
        as_float(r["coef"])
        for r in h3_rows
        if int(float(r["threshold_min"])) == 15
        and r["model_type"].strip() == "OLS_HC3"
        and r["term"].strip().lower() == "elderly_share"
    )
    rf_auc = as_float(ml_baselines["models"]["random_forest"]["oof_roc_auc"])

    for target in (ev_g15, h3_eld, rf_auc):
        assert min(abs(v - target) for v in t1_csv_nums) <= 0.005
        assert min(abs(v - target) for v in t1_tex_nums) <= 0.005
        assert min(abs(v - target) for v in pdf_nums) <= 0.005

    assert "figure 1" in report_pdf_info.main_text.lower()
    assert "table 1" in report_pdf_info.main_text.lower()
    assert "fig1_accessibility_lisa" in report_tex_text
