"""One-command Colab L4/A100 Phase B Runner (PatchTST + Chronos + FinBERT + Gemma-4 / Qwen QLoRA).

Refuses to download weights or touch CUDA unless BOTH `--allow-weight-download` is passed AND
`torch.cuda.is_available()` is True. Reads Phase A `data/frozen/` read-only (verified via
`FREEZE_MANIFEST.json`) and writes all Phase B outputs under `outputs/phase_b/`.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from stock_volatility import config
from stock_volatility.phase_b.data_contract import (
    build_event_anchored_datasets,
    build_patchtst_windows,
    load_frozen_phase_b_inputs,
)
from stock_volatility.phase_b.fusion_and_eval import (
    evaluate_phase_b_h3_ablation,
    fit_2024_val_oof_meta_fusion,
)
from stock_volatility.phase_b.patchtst import (
    PATCHTST_1M_CONFIG,
    PATCHTST_15M_CONFIG,
    train_and_predict_patchtst,
)
from stock_volatility.phase_b.route_b_chronos import (
    DEFAULT_CHRONOS_MODELS,
    run_chronos_volatility_forecasts,
)
from stock_volatility.phase_b.route_b_finbert import (
    DEFAULT_FINBERT_MODEL,
    extract_finbert_event_scores,
)
from stock_volatility.phase_b.route_b_llm import (
    DEFAULT_L4_LLM_MODEL,
    DEFAULT_LLM_MODEL,
    require_cuda_and_download_flag,
    run_llm_zero_shot_and_qlora,
    select_default_llm_for_vram,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI options for Colab single-GPU (L4 24GB or A100 40GB/80GB) Phase B execution."""
    parser = argparse.ArgumentParser(
        description="Plan 6 Phase B Colab GPU Runner (PatchTST + FinBERT + LLM QLoRA + Chronos)."
    )
    parser.add_argument(
        "--allow-weight-download",
        action="store_true",
        help="Explicitly authorize Hugging Face weight downloads and CUDA execution on Colab GPU.",
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default="auto",
        help=(
            "Hugging Face LLM ID or 'auto' "
            f"(>=35GB VRAM -> {DEFAULT_LLM_MODEL}; <35GB L4 -> {DEFAULT_L4_LLM_MODEL})."
        ),
    )
    parser.add_argument(
        "--finbert-model",
        type=str,
        default=DEFAULT_FINBERT_MODEL,
        help=f"FinBERT baseline model ID (default: {DEFAULT_FINBERT_MODEL}).",
    )
    parser.add_argument(
        "--routes",
        nargs="+",
        choices=("all", "route_a", "finbert", "llm", "chronos"),
        default=["all"],
        help="Subsets of Phase B to run (default: all).",
    )
    parser.add_argument("--llm-epochs", type=int, default=1, help="QLoRA fine-tuning epochs.")
    parser.add_argument(
        "--llm-train-batch-size",
        type=int,
        default=8,
        help="Micro-batch size for QLoRA training (default: 8 on L4/A100, grad_accum adjusts).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse already-saved Stage 1-4 outputs in outputs/phase_b/ instead of recomputing.",
    )
    parser.add_argument(
        "--max-train-events",
        type=int,
        default=None,
        help="Optional fallback cap on Train event-anchored days (default: all 4,929 days).",
    )
    parser.add_argument(
        "--patchtst-seeds",
        type=int,
        nargs="+",
        default=[42, 43, 44, 45, 46],
        help="Random seeds for Route A PatchTST (default: 42 43 44 45 46).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory for Phase B artifacts (default: outputs/phase_b/).",
    )
    return parser.parse_args(argv)


CANONICAL_ARM_ORDER: tuple[str, ...] = (
    "patchtst_1m",
    "patchtst_15m",
    "chronos_2",
    "chronos_t5_large",
    "lgbm_plus_finbert",
    "lgbm_plus_llm_zs",
    "lgbm_plus_llm_qlora",
)


def _format_patchtst_label(base_label: str, cov80: float) -> str:
    """Format PatchTST table label matching Chronos semicolon convention."""
    stem = base_label[:-1] if base_label.endswith(")") else base_label
    return f"{stem}; 80% cov={100.0 * cov80:.1f}%)"


def hydrate_saved_phase_b_state(
    *,
    out_dir: Path,
    run_set: set[str],
    resolved_llm: str,
    summary: dict[str, Any],
    arms_for_ablation: dict[str, dict[str, Any]],
) -> None:
    """Preserve saved summary fields and unselected stage arms during subset `--routes` runs."""
    if "all" in run_set:
        return

    summary_path = out_dir / "phase_b_summary.json"
    if summary_path.is_file():
        prev_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        for k, v in prev_summary.items():
            summary.setdefault(k, v)

    if "route_a" not in run_set:
        for arm_id, lbl in (
            ("patchtst_1m", "PatchTST (1M, 5-seed)"),
            ("patchtst_15m", "PatchTST (15M, 5-seed)"),
        ):
            pred_p = out_dir / f"{arm_id}_test_predictions.parquet"
            met_p = out_dir / f"{arm_id}_metrics.json"
            if pred_p.is_file() and met_p.is_file():
                met_saved = json.loads(met_p.read_text(encoding="utf-8"))
                cov80 = float(met_saved["test_nominal_80_coverage"])
                arms_for_ablation[arm_id] = {
                    "label": _format_patchtst_label(lbl, cov80),
                    "test_pred_df": pd.read_parquet(pred_p),
                    "nominal_80_coverage": cov80,
                }
                if arm_id == "patchtst_15m":
                    summary["patchtst_15m_qlike"] = float(
                        met_saved["test_metrics_ensemble"]["qlike"]
                    )
                    summary["patchtst_15m_coverage_80"] = cov80

    if "chronos" not in run_set:
        for ch_id in DEFAULT_CHRONOS_MODELS:
            slug = ch_id.split("/")[-1].replace("-", "_")
            pred_p = out_dir / f"{slug}_test_predictions.parquet"
            met_p = out_dir / f"{slug}_metrics.json"
            if pred_p.is_file() and met_p.is_file():
                met_saved = json.loads(met_p.read_text(encoding="utf-8"))
                cov80_c = float(met_saved["test_nominal_80_coverage"])
                arms_for_ablation[slug] = {
                    "label": f"{ch_id.split('/')[-1]} (Zero-Shot; 80% cov={100.0 * cov80_c:.1f}%)",
                    "test_pred_df": pd.read_parquet(pred_p),
                    "nominal_80_coverage": cov80_c,
                }

    if "finbert" not in run_set:
        fb_p = out_dir / "lgbm_plus_finbert_test.parquet"
        if fb_p.is_file():
            arms_for_ablation["lgbm_plus_finbert"] = {
                "label": "LightGBM + FinBERT (110M)",
                "test_pred_df": pd.read_parquet(fb_p),
            }

    if "llm" not in run_set:
        llm_meta_p = out_dir / "llm_run_metadata.json"
        model_for_label = resolved_llm
        if llm_meta_p.is_file():
            llm_meta = json.loads(llm_meta_p.read_text(encoding="utf-8"))
            model_for_label = str(llm_meta.get("model_id", resolved_llm))
            summary["llm_model_id"] = model_for_label
        short_name = model_for_label.split("/")[-1]
        for arm_key, suffix in (
            ("lgbm_plus_llm_zs", "Zero-Shot"),
            ("lgbm_plus_llm_qlora", "4-bit QLoRA"),
        ):
            arm_p = out_dir / f"{arm_key}_test.parquet"
            if arm_p.is_file():
                arms_for_ablation[arm_key] = {
                    "label": f"LightGBM + {short_name} ({suffix})",
                    "test_pred_df": pd.read_parquet(arm_p),
                }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _free_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main(argv: list[str] | None = None) -> int:
    """Execute Colab GPU Phase B training, OOF fusion, H3 ablation & export artifacts."""
    args = parse_args(argv)
    require_cuda_and_download_flag(
        allow_weight_download=args.allow_weight_download,
        component="colab_phase_b",
    )

    gpu_name = str(torch.cuda.get_device_name(0))
    vram_gb = float(torch.cuda.get_device_properties(0).total_memory / 1e9)
    resolved_llm = (
        select_default_llm_for_vram(vram_gb) if args.llm_model == "auto" else args.llm_model
    )
    print(
        f"[PhaseB] GPU={gpu_name} ({vram_gb:.1f} GB VRAM) | LLM={resolved_llm} "
        f"| resume={args.resume}",
        flush=True,
    )

    root = config.PROJECT_ROOT
    out_dir = args.out_dir.resolve() if args.out_dir is not None else (root / "outputs" / "phase_b")
    out_dir.mkdir(parents=True, exist_ok=True)

    run_set = set(args.routes)
    run_all = "all" in run_set

    bundle = load_frozen_phase_b_inputs(root)
    panel_df = bundle["panel_df"]
    events_df = bundle["events_phase_b_df"]
    feat_m = bundle["feature_manifest"]
    val_a = pd.read_parquet(root / "outputs" / "tables" / "h2_val_predictions.parquet")
    test_a = pd.read_parquet(root / "outputs" / "tables" / "h2_test_predictions.parquet")
    ev_splits = build_event_anchored_datasets(panel_df, events_df)

    arms_for_ablation: dict[str, dict[str, Any]] = {}
    summary: dict[str, Any] = {
        "schema_version": "1.0.0",
        "milestone": "PhaseB",
        "gpu_name": gpu_name,
        "gpu_vram_gb": round(vram_gb, 2),
        "llm_model_id": resolved_llm,
        "finbert_model_id": args.finbert_model,
        "train_event_days": int(len(ev_splits["train"])),
        "val_event_days": int(len(ev_splits["val"])),
        "test_event_days": int(len(ev_splits["test"])),
    }
    hydrate_saved_phase_b_state(
        out_dir=out_dir,
        run_set=run_set,
        resolved_llm=resolved_llm,
        summary=summary,
        arms_for_ablation=arms_for_ablation,
    )

    # 1. Route A: PatchTST 1M vs 15M capacity ablation (5 seeds)
    if run_all or "route_a" in run_set:
        windows: dict[str, Any] | None = None
        for cfg_obj, arm_id, lbl in (
            (PATCHTST_1M_CONFIG, "patchtst_1m", "PatchTST (1M, 5-seed)"),
            (PATCHTST_15M_CONFIG, "patchtst_15m", "PatchTST (15M, 5-seed)"),
        ):
            pred_p = out_dir / f"{arm_id}_test_predictions.parquet"
            met_p = out_dir / f"{arm_id}_metrics.json"
            if args.resume and pred_p.is_file() and met_p.is_file():
                met_saved = json.loads(met_p.read_text(encoding="utf-8"))
                pred_saved = pd.read_parquet(pred_p)
                cov80 = float(met_saved["test_nominal_80_coverage"])
                print(
                    f"[PhaseB] Resumed {arm_id} from {pred_p.name} "
                    f"(QLIKE={met_saved['test_metrics_ensemble']['qlike']:.4f})",
                    flush=True,
                )
                arms_for_ablation[arm_id] = {
                    "label": _format_patchtst_label(lbl, cov80),
                    "test_pred_df": pred_saved,
                    "nominal_80_coverage": cov80,
                }
                if arm_id == "patchtst_15m":
                    summary["patchtst_15m_qlike"] = float(
                        met_saved["test_metrics_ensemble"]["qlike"]
                    )
                    summary["patchtst_15m_coverage_80"] = cov80
                continue

            if windows is None:
                print("[PhaseB] Stage 1/4: Route A PatchTST (1M & 15M, 5 seeds)...", flush=True)
                windows = build_patchtst_windows(panel_df, feat_m, lookback=60)
                print(
                    f"[PhaseB] Windows built: Train={len(windows['train']['x_seq']):,}, "
                    f"Val={len(windows['val']['x_seq']):,}, Test={len(windows['test']['x_seq']):,}",
                    flush=True,
                )
            res_pt = train_and_predict_patchtst(
                windows=windows,
                cfg=cfg_obj,
                seeds=args.patchtst_seeds,
                epochs=25,
                batch_size=512,
                device="cuda",
            )
            res_pt["test_pred_df"].to_parquet(pred_p, index=False, compression="zstd")
            _write_json(met_p, {k: v for k, v in res_pt.items() if not k.endswith("_df")})
            cov80_new = float(res_pt["test_nominal_80_coverage"])
            arms_for_ablation[arm_id] = {
                "label": _format_patchtst_label(lbl, cov80_new),
                "test_pred_df": res_pt["test_pred_df"],
                "nominal_80_coverage": cov80_new,
            }
            if arm_id == "patchtst_15m":
                summary["patchtst_15m_qlike"] = res_pt["test_metrics_ensemble"]["qlike"]
                summary["patchtst_15m_coverage_80"] = cov80_new
            _free_cuda()

    # 2. Route B (Chronos Foundation Time-Series Baselines)
    if run_all or "chronos" in run_set:
        for ch_id in DEFAULT_CHRONOS_MODELS:
            slug = ch_id.split("/")[-1].replace("-", "_")
            pred_p = out_dir / f"{slug}_test_predictions.parquet"
            met_p = out_dir / f"{slug}_metrics.json"
            if args.resume and pred_p.is_file() and met_p.is_file():
                met_saved = json.loads(met_p.read_text(encoding="utf-8"))
                pred_saved = pd.read_parquet(pred_p)
                print(
                    f"[PhaseB] Resumed {slug} from {pred_p.name} "
                    f"(QLIKE={met_saved['test_metrics']['qlike']:.4f})",
                    flush=True,
                )
                cov80_c = float(met_saved["test_nominal_80_coverage"])
                arms_for_ablation[slug] = {
                    "label": f"{ch_id.split('/')[-1]} (Zero-Shot; 80% cov={100.0 * cov80_c:.1f}%)",
                    "test_pred_df": pred_saved,
                    "nominal_80_coverage": cov80_c,
                }
                continue

            print(f"[PhaseB] Stage 2/4: Route B Chronos ({ch_id})...", flush=True)
            res_ch = run_chronos_volatility_forecasts(
                panel_df=panel_df,
                model_id=ch_id,
                allow_weight_download=args.allow_weight_download,
            )
            res_ch["test_pred_df"].to_parquet(pred_p, index=False, compression="zstd")
            _write_json(met_p, {k: v for k, v in res_ch.items() if not k.endswith("_df")})
            cov80_c = float(res_ch["test_nominal_80_coverage"])
            arms_for_ablation[slug] = {
                "label": f"{ch_id.split('/')[-1]} (Zero-Shot; 80% cov={100.0 * cov80_c:.1f}%)",
                "test_pred_df": res_ch["test_pred_df"],
                "nominal_80_coverage": cov80_c,
            }
            _free_cuda()

    # 3. Route B (FinBERT 110M Baseline + 2024-Val OOF Fusion)
    if run_all or "finbert" in run_set:
        fb_test_p = out_dir / "lgbm_plus_finbert_test.parquet"
        if args.resume and fb_test_p.is_file():
            print(f"[PhaseB] Resumed lgbm_plus_finbert from {fb_test_p.name}", flush=True)
            arms_for_ablation["lgbm_plus_finbert"] = {
                "label": "LightGBM + FinBERT (110M)",
                "test_pred_df": pd.read_parquet(fb_test_p),
            }
        else:
            print(f"[PhaseB] Stage 3/4: Route B FinBERT ({args.finbert_model})...", flush=True)
            res_fb = extract_finbert_event_scores(
                event_splits=ev_splits,
                model_id=args.finbert_model,
                allow_weight_download=args.allow_weight_download,
            )
            res_fb["val_scores_df"].to_parquet(out_dir / "finbert_val_scores.parquet", index=False)
            res_fb["test_scores_df"].to_parquet(
                out_dir / "finbert_test_scores.parquet", index=False
            )
            fused_fb = fit_2024_val_oof_meta_fusion(
                val_phase_a=val_a,
                test_phase_a=test_a,
                val_text_scores=res_fb["val_scores_df"],
                test_text_scores=res_fb["test_scores_df"],
                score_col="finbert_neg_logit",
                arm_name="lgbm_plus_finbert",
            )
            fused_fb["test_pred_df"].to_parquet(fb_test_p, index=False, compression="zstd")
            arms_for_ablation["lgbm_plus_finbert"] = {
                "label": "LightGBM + FinBERT (110M)",
                "test_pred_df": fused_fb["test_pred_df"],
            }
            _free_cuda()

    # 4. Route B Flagship: LLM Zero-Shot & 4-bit NF4 QLoRA + 2024-Val OOF Fusion
    if run_all or "llm" in run_set:
        llm_val_p = out_dir / "llm_val_scores.parquet"
        llm_test_p = out_dir / "llm_test_scores.parquet"
        llm_meta_p = out_dir / "llm_run_metadata.json"
        resumed_llm = False
        if args.resume and llm_val_p.is_file() and llm_test_p.is_file() and llm_meta_p.is_file():
            saved_meta = json.loads(llm_meta_p.read_text(encoding="utf-8"))
            if saved_meta.get("model_id") == resolved_llm:
                print(
                    f"[PhaseB] Resumed Stage 4 LLM ({resolved_llm}) from {llm_meta_p.name}",
                    flush=True,
                )
                val_llm_df = pd.read_parquet(llm_val_p)
                test_llm_df = pd.read_parquet(llm_test_p)
                summary.update(
                    {
                        "llm_revision": saved_meta.get("revision", "unknown"),
                        "gpu_name": saved_meta.get("gpu_name", gpu_name),
                        "peak_vram_gb": saved_meta.get("peak_vram_gb", 0.0),
                        "llm_elapsed_sec": saved_meta.get("elapsed_sec", 0.0),
                        "llm_trainable_params": saved_meta.get("trainable_params", 0),
                    }
                )
                resumed_llm = True

        if not resumed_llm:
            train_bs = max(1, int(args.llm_train_batch_size))
            grad_accum = max(1, (8 + train_bs - 1) // train_bs)
            print(
                f"[PhaseB] Stage 4/4: Route B LLM Zero-Shot & QLoRA ({resolved_llm}, "
                f"bs={train_bs}x{grad_accum})...",
                flush=True,
            )
            res_llm = run_llm_zero_shot_and_qlora(
                event_splits=ev_splits,
                model_id=resolved_llm,
                allow_weight_download=args.allow_weight_download,
                epochs=args.llm_epochs,
                train_batch_size=train_bs,
                grad_accum_steps=grad_accum,
                eval_batch_size=16,
                max_train_events=args.max_train_events,
            )
            val_llm_df = res_llm["val_scores_df"]
            test_llm_df = res_llm["test_scores_df"]
            val_llm_df.to_parquet(llm_val_p, index=False)
            test_llm_df.to_parquet(llm_test_p, index=False)
            _write_json(
                llm_meta_p,
                {k: v for k, v in res_llm.items() if not k.endswith("_df")},
            )
            summary.update(
                {
                    "llm_revision": res_llm["revision"],
                    "gpu_name": res_llm["gpu_name"],
                    "peak_vram_gb": res_llm["peak_vram_gb"],
                    "llm_elapsed_sec": res_llm["elapsed_sec"],
                    "llm_trainable_params": res_llm["trainable_params"],
                }
            )
            _free_cuda()

        short_name = resolved_llm.split("/")[-1]
        for col_name, arm_key, label in (
            ("llm_zs_logit", "lgbm_plus_llm_zs", f"LightGBM + {short_name} (Zero-Shot)"),
            ("llm_qlora_logit", "lgbm_plus_llm_qlora", f"LightGBM + {short_name} (4-bit QLoRA)"),
        ):
            fused = fit_2024_val_oof_meta_fusion(
                val_phase_a=val_a,
                test_phase_a=test_a,
                val_text_scores=val_llm_df,
                test_text_scores=test_llm_df,
                score_col=col_name,
                arm_name=arm_key,
            )
            fused["test_pred_df"].to_parquet(
                out_dir / f"{arm_key}_test.parquet", index=False, compression="zstd"
            )
            arms_for_ablation[arm_key] = {
                "label": label,
                "test_pred_df": fused["test_pred_df"],
            }

    # 5. Formal 2025 H3 Ablation Table in canonical arm order
    ordered_arms: dict[str, dict[str, Any]] = {
        k: arms_for_ablation[k] for k in CANONICAL_ARM_ORDER if k in arms_for_ablation
    }
    for k, v in arms_for_ablation.items():
        if k not in ordered_arms:
            ordered_arms[k] = v

    ablation = evaluate_phase_b_h3_ablation(
        phase_a_test_df=test_a,
        phase_b_arms=ordered_arms,
        block_len=5,
        n_boot=1000,
        seed=config.RANDOM_SEED,
    )
    comp_df = ablation["comparison_df"]
    comp_df.to_csv(out_dir / "phase_b_comparison.csv", index=False, lineterminator="\n")
    (out_dir / "phase_b_comparison.md").write_text(ablation["markdown_table"], encoding="utf-8")
    (out_dir / "phase_b_comparison.tex").write_text(ablation["latex_table"], encoding="utf-8")

    # Populate report-ready summary keys from verified comparison rows
    by_arm = comp_df.set_index("arm_id").to_dict(orient="index")
    if "lgbm_plus_finbert" in by_arm:
        summary["finbert_qlike_event"] = float(by_arm["lgbm_plus_finbert"]["qlike_event_days"])
        summary["finbert_pr_auc_event"] = float(by_arm["lgbm_plus_finbert"]["pr_auc_event_days"])
    if "lgbm_plus_llm_zs" in by_arm:
        summary["qwen_zs_pr_auc_event"] = float(by_arm["lgbm_plus_llm_zs"]["pr_auc_event_days"])
    if "lgbm_plus_llm_qlora" in by_arm:
        ql_r = by_arm["lgbm_plus_llm_qlora"]
        base_r = by_arm["phase_a_lgbm"]
        llm_id_for_note = str(summary.get("llm_model_id", resolved_llm))
        summary.update(
            {
                "qwen_qlora_qlike_full": float(ql_r["qlike_full"]),
                "qwen_qlora_qlike_event": float(ql_r["qlike_event_days"]),
                "qwen_qlora_pr_auc_full": float(ql_r["pr_auc_full"]),
                "qwen_qlora_pr_auc_event": float(ql_r["pr_auc_event_days"]),
                "qwen_qlora_ece": float(ql_r["ece"]),
                "qwen_qlora_dm_p_holm": float(ql_r["dm_qlike_p_holm"]),
                "h3_interpretation": (
                    f"Relative to Phase A price-only LightGBM (event-day PR-AUC "
                    f"{base_r['pr_auc_event_days']:.4f}, full QLIKE {base_r['qlike_full']:.4f}), "
                    f"{llm_id_for_note} QLoRA achieves event-day PR-AUC "
                    f"{ql_r['pr_auc_event_days']:.4f} and full QLIKE {ql_r['qlike_full']:.4f} "
                    f"(HLN-DM Holm p={ql_r['dm_qlike_p_holm']:.3f})."
                ),
            }
        )

    _write_json(out_dir / "phase_b_summary.json", summary)
    print(ablation["markdown_table"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
