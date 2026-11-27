"""Section 3.3 Calibration, 2024-Val OOF Cross-Modal Fusion & 2025 H3 Ablation Evaluation."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression, Ridge

from stock_volatility import config
from stock_volatility.analysis.panel_har_rv import holm_bonferroni_adjust
from stock_volatility.evaluation import (
    classification_metrics,
    date_block_bootstrap_ci,
    hln_dm_test,
    qlike_loss_array,
    regression_metrics,
)


def fit_temperature_scaler(logits_val: np.ndarray, y_val: np.ndarray) -> float:
    """Fit 1-parameter Temperature Scaling (`T > 0`) minimizing NLL strictly on 2024 Val."""
    z = np.asarray(logits_val, dtype=float)
    y = np.asarray(y_val, dtype=float)
    finite = np.isfinite(z) & np.isfinite(y)
    z, y = z[finite], y[finite]
    if len(z) < 10 or len(np.unique(y)) < 2:
        return 1.0

    def _nll(temp: float) -> float:
        p = np.clip(expit(z / temp), 1e-7, 1.0 - 1e-7)
        return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))

    res = minimize_scalar(_nll, bounds=(0.1, 20.0), method="bounded")
    return float(res.x) if res.success else 1.0


def apply_temperature_scaling(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Apply frozen Temperature Scaling `sigmoid(logits / T)` cleanly."""
    t_safe = max(float(temperature), 1e-3)
    return expit(np.asarray(logits, dtype=float) / t_safe)


def _safe_logit(p: np.ndarray) -> np.ndarray:
    return logit(np.clip(np.asarray(p, dtype=float), 1e-5, 1.0 - 1e-5))


def _assign_quarters(dates: pd.Series) -> np.ndarray:
    dt = pd.to_datetime(dates)
    return ((dt.dt.month - 1) // 3).to_numpy(dtype=int)


def fit_2024_val_oof_meta_fusion(
    *,
    val_base_df: pd.DataFrame,
    test_base_df: pd.DataFrame,
    val_text_scores: pd.DataFrame,
    test_text_scores: pd.DataFrame,
    score_col: str,
    arm_name: str,
) -> dict[str, Any]:
    """Fuse Global LightGBM predictions with event-day text/LLM logits strictly via 2024 Val OOF.

    Why 2024 Val OOF meta-fusion:
    - QLoRA adapters were trained on 2019-2023 Train, so Train-split LLM scores are in-sample.
    - Fitting Temperature Scaling and L2-regularized meta-weights strictly on 2024 Val (selecting
      regularization C / alpha via 4-quarter OOF CV strictly inside 2024 Val) prevents any
      in-sample QLoRA memorization from distorting fusion weights, while keeping 2025 Test 100%
      single-shot untouched.
    - Non-event trading days receive deterministic neutral score `0.0` paired with `has_news`.
    """
    val_df = val_base_df.merge(
        val_text_scores[["ticker", "date", score_col]],
        on=["ticker", "date"],
        how="left",
        validate="one_to_one",
    ).copy()
    test_df = test_base_df.merge(
        test_text_scores[["ticker", "date", score_col]],
        on=["ticker", "date"],
        how="left",
        validate="one_to_one",
    ).copy()

    val_raw_z = val_df[score_col].fillna(0.0).to_numpy(dtype=float)
    test_raw_z = test_df[score_col].fillna(0.0).to_numpy(dtype=float)
    val_has_ev = val_df["has_news"].to_numpy(dtype=float)
    test_has_ev = test_df["has_news"].to_numpy(dtype=float)

    # 1. Fit 1-parameter Temperature Scaling strictly on active 2024 Val event days
    ev_mask_val = val_has_ev == 1.0
    temp = fit_temperature_scaler(
        val_raw_z[ev_mask_val],
        val_df.loc[ev_mask_val, "y_tail"].to_numpy(dtype=int),
    )

    # Standardize active event logits using strictly 2024 Val event mean/std (non-event = 0.0)
    ev_mean = float(np.mean(val_raw_z[ev_mask_val])) if ev_mask_val.any() else 0.0
    ev_std = float(np.std(val_raw_z[ev_mask_val])) if ev_mask_val.any() else 1.0
    ev_std = max(ev_std, 1e-6)

    val_z_cal = np.where(ev_mask_val, (val_raw_z - ev_mean) / (ev_std * max(temp, 0.25)), 0.0)
    test_ev_mask = test_has_ev == 1.0
    test_z_cal = np.where(test_ev_mask, (test_raw_z - ev_mean) / (ev_std * max(temp, 0.25)), 0.0)

    # Feature matrix for meta-fusion: [base_lgbm_signal, text_z_cal, has_news]
    x_cls_val = np.column_stack(
        [_safe_logit(val_df["p_tail_lgbm_cal"].to_numpy()), val_z_cal, val_has_ev]
    )
    x_cls_test = np.column_stack(
        [_safe_logit(test_df["p_tail_lgbm_cal"].to_numpy()), test_z_cal, test_has_ev]
    )
    y_cls_val = val_df["y_tail"].to_numpy(dtype=int)

    x_reg_val = np.column_stack(
        [np.log(np.maximum(val_df["rv_hat_lgbm"].to_numpy(), 1e-4)), val_z_cal, val_has_ev]
    )
    x_reg_test = np.column_stack(
        [np.log(np.maximum(test_df["rv_hat_lgbm"].to_numpy(), 1e-4)), test_z_cal, test_has_ev]
    )
    y_reg_val = np.log(np.maximum(val_df["rv_5d_fwd"].to_numpy(), 1e-4))
    rv_val = val_df["rv_5d_fwd"].to_numpy(dtype=float)

    q_idx = _assign_quarters(val_df["date"])

    # 2. Select L2 regularization strictly via 4-quarter OOF CV inside 2024 Val
    best_c, best_oof_brier = 0.25, float("inf")
    for cand_c in (0.05, 0.25, 1.0, 5.0):
        oof_p = np.zeros(len(val_df), dtype=float)
        for q in range(4):
            tr_m = q_idx != q
            va_m = q_idx == q
            if not va_m.any() or len(np.unique(y_cls_val[tr_m])) < 2:
                continue
            clf = LogisticRegression(C=cand_c, max_iter=500, random_state=config.RANDOM_SEED)
            clf.fit(x_cls_val[tr_m], y_cls_val[tr_m])
            oof_p[va_m] = clf.predict_proba(x_cls_val[va_m])[:, 1]
        brier = float(np.mean((oof_p - y_cls_val) ** 2))
        if brier < best_oof_brier:
            best_oof_brier = brier
            best_c = cand_c

    final_clf = LogisticRegression(C=best_c, max_iter=500, random_state=config.RANDOM_SEED)
    final_clf.fit(x_cls_val, y_cls_val)
    p_tail_test = np.clip(final_clf.predict_proba(x_cls_test)[:, 1], 1e-6, 1.0 - 1e-6)

    best_alpha, best_oof_qlike = 10.0, float("inf")
    for cand_a in (1.0, 10.0, 100.0, 500.0):
        oof_rv = np.zeros(len(val_df), dtype=float)
        for q in range(4):
            tr_m = q_idx != q
            va_m = q_idx == q
            if not va_m.any():
                continue
            reg = Ridge(alpha=cand_a, random_state=config.RANDOM_SEED)
            reg.fit(x_reg_val[tr_m], y_reg_val[tr_m])
            resid = np.clip(y_reg_val[tr_m] - reg.predict(x_reg_val[tr_m]), -4, 4)
            smear = float(np.mean(np.exp(resid)))
            oof_rv[va_m] = np.maximum(np.exp(reg.predict(x_reg_val[va_m])) * smear, 1e-4)
        ql = float(np.mean(qlike_loss_array(rv_val, oof_rv)))
        if ql < best_oof_qlike:
            best_oof_qlike = ql
            best_alpha = cand_a

    final_reg = Ridge(alpha=best_alpha, random_state=config.RANDOM_SEED)
    final_reg.fit(x_reg_val, y_reg_val)
    final_smear = float(np.mean(np.exp(np.clip(y_reg_val - final_reg.predict(x_reg_val), -4, 4))))
    rv_hat_test = np.maximum(np.exp(final_reg.predict(x_reg_test)) * final_smear, 1e-4)

    test_out = test_df[["ticker", "date"]].copy()
    test_out["rv_hat"] = rv_hat_test
    test_out["p_tail"] = p_tail_test
    test_out["raw_text_logit"] = test_raw_z

    return {
        "arm_name": arm_name,
        "val_temperature": float(temp),
        "selected_cls_C": float(best_c),
        "selected_reg_alpha": float(best_alpha),
        "val_oof_brier": float(best_oof_brier),
        "val_oof_qlike": float(best_oof_qlike),
        "cls_weights": [float(w) for w in final_clf.coef_[0]],
        "reg_weights": [float(w) for w in final_reg.coef_],
        "test_pred_df": test_out,
    }


def _fmt_p(p: float) -> str:
    if not np.isfinite(p):
        return "—"
    if p < 1e-4:
        return "<0.0001"
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}"


def evaluate_h3_dl_ablation(
    *,
    base_test_df: pd.DataFrame,
    dl_arms: dict[str, dict[str, Any]],
    block_len: int = 5,
    n_boot: int = 1000,
    seed: int = config.RANDOM_SEED,
) -> dict[str, Any]:
    """Evaluate Global LightGBM vs all H3 DL/LLM arms on 2025 Full, Event-Day & Earnings subsets."""
    base = base_test_df.sort_values(["date", "ticker"], kind="mergesort").reset_index(drop=True)
    dates = base["date"].to_numpy()
    rv_true = base["rv_5d_fwd"].to_numpy(dtype=float)
    y_tail = base["y_tail"].to_numpy(dtype=int)
    ev_mask = base["has_news"].to_numpy() == 1
    earn_mask = base["has_8k_earnings"].to_numpy() == 1

    base_rv = base["rv_hat_lgbm"].to_numpy(dtype=float)
    base_p = base["p_tail_lgbm_cal"].to_numpy(dtype=float)
    base_ql_arr = qlike_loss_array(rv_true, base_rv)
    base_br_arr = (base_p - y_tail) ** 2

    def _eval_arm(
        arm_id: str,
        label: str,
        rv_hat: np.ndarray,
        p_hat: np.ndarray | None,
        cov80: float | None = None,
    ) -> dict[str, Any]:
        rm_full = regression_metrics(rv_true, rv_hat)
        rm_ev = regression_metrics(rv_true[ev_mask], rv_hat[ev_mask])

        ql_lo, ql_hi = date_block_bootstrap_ci(
            dates,
            lambda idx, rh=rv_hat: float(np.mean(qlike_loss_array(rv_true[idx], rh[idx]))),
            block_len=block_len,
            n_boot=n_boot,
            seed=seed,
        )
        if p_hat is not None:
            cm_full = classification_metrics(y_tail, p_hat)
            cm_ev = classification_metrics(y_tail[ev_mask], p_hat[ev_mask])
            cm_earn = classification_metrics(y_tail[earn_mask], p_hat[earn_mask])
            pr_lo, pr_hi = date_block_bootstrap_ci(
                dates,
                lambda idx, ph=p_hat: classification_metrics(y_tail[idx], ph[idx])["pr_auc"],
                block_len=block_len,
                n_boot=n_boot,
                seed=seed,
            )
            pr_full = cm_full["pr_auc"]
            pr_ev = cm_ev["pr_auc"]
            pr_earn = cm_earn["pr_auc"]
            brier_full = cm_full["brier"]
            ece_val = cm_full["ece"]
        else:
            pr_full = pr_lo = pr_hi = pr_ev = pr_earn = brier_full = ece_val = float("nan")

        return {
            "arm_id": arm_id,
            "model_label": label,
            "qlike_full": rm_full["qlike"],
            "qlike_ci95_low": ql_lo,
            "qlike_ci95_high": ql_hi,
            "qlike_event_days": rm_ev["qlike"],
            "rmse_full": rm_full["rmse"],
            "pr_auc_full": pr_full,
            "pr_auc_ci95_low": pr_lo,
            "pr_auc_ci95_high": pr_hi,
            "pr_auc_event_days": pr_ev,
            "pr_auc_earnings_days": pr_earn,
            "brier_full": brier_full,
            "ece": ece_val,
            "nominal_80_coverage": cov80 if cov80 is not None else float("nan"),
        }

    rows: list[dict[str, Any]] = [
        {
            **_eval_arm("global_lgbm", "Global LightGBM (Price-Only)", base_rv, base_p),
            "dm_qlike_stat": float("nan"),
            "dm_qlike_p_raw": float("nan"),
            "dm_qlike_p_holm": float("nan"),
            "dm_brier_stat": float("nan"),
            "dm_brier_p_raw": float("nan"),
            "dm_brier_p_holm": float("nan"),
        }
    ]

    dm_q_raw: list[float] = []
    dm_b_indices: list[int] = []
    dm_b_raw: list[float] = []
    arm_rows_tmp: list[dict[str, Any]] = []

    for arm_id, spec in dl_arms.items():
        merged = base[["ticker", "date"]].merge(
            spec["test_pred_df"], on=["ticker", "date"], how="left", validate="one_to_one"
        )
        rv_hat = merged["rv_hat"].fillna(pd.Series(base_rv)).to_numpy(dtype=float)
        has_cls = "p_tail" in merged.columns
        p_hat = (
            merged["p_tail"].fillna(pd.Series(base_p)).to_numpy(dtype=float) if has_cls else None
        )
        rec = _eval_arm(arm_id, str(spec["label"]), rv_hat, p_hat, spec.get("nominal_80_coverage"))

        # Multi-step h=5 HLN-DM tests against Global LightGBM (d_t = L(arm) - L(global_lgbm))
        d_ql = pd.Series(qlike_loss_array(rv_true, rv_hat) - base_ql_arr).groupby(dates).mean()
        dm_q = hln_dm_test(d_ql.to_numpy(), horizon=config.HORIZON_DAYS)
        rec["dm_qlike_stat"] = dm_q["dm_hln_stat"]
        rec["dm_qlike_p_raw"] = dm_q["p_value"]
        dm_q_raw.append(dm_q["p_value"])

        if p_hat is not None:
            d_br = pd.Series((p_hat - y_tail) ** 2 - base_br_arr).groupby(dates).mean()
            dm_b = hln_dm_test(d_br.to_numpy(), horizon=config.HORIZON_DAYS)
            rec["dm_brier_stat"] = dm_b["dm_hln_stat"]
            rec["dm_brier_p_raw"] = dm_b["p_value"]
            dm_b_indices.append(len(arm_rows_tmp))
            dm_b_raw.append(dm_b["p_value"])
        else:
            rec["dm_brier_stat"] = float("nan")
            rec["dm_brier_p_raw"] = float("nan")
            rec["dm_brier_p_holm"] = float("nan")

        arm_rows_tmp.append(rec)

    if arm_rows_tmp:
        q_holm = holm_bonferroni_adjust(dm_q_raw)
        b_holm_map: dict[int, float] = {}
        if dm_b_raw:
            for idx_b, p_bh in zip(dm_b_indices, holm_bonferroni_adjust(dm_b_raw), strict=True):
                b_holm_map[idx_b] = float(p_bh)
        for i_rec, (rec, pqh) in enumerate(zip(arm_rows_tmp, q_holm, strict=True)):
            rec["dm_qlike_p_holm"] = float(pqh)
            if i_rec in b_holm_map:
                rec["dm_brier_p_holm"] = b_holm_map[i_rec]
            rows.append(rec)

    comp_df = pd.DataFrame(rows)
    n_full_fmt = f"{len(base):,}".replace(",", "{,}")
    n_ev_fmt = f"{int(ev_mask.sum()):,}".replace(",", "{,}")

    md_lines = [
        "# Table 2: 2025 Holdout Deep Time-Series & Event-Anchored LLM Ablation (H3)",
        "",
        "| Model | 2025 Full QLIKE | Event-Day QLIKE | Tail PR-AUC (Event) | ECE | "
        "HLN-DM Holm p (Q / B vs LGBM) |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    tex_lines = [
        r"\begin{table}[!htb]",
        r"\centering",
        rf"\caption{{2025 Single-Shot Holdout Deep Time-Series and Event-Anchored LLM Ablation "
        rf"(H3; $N={n_full_fmt}$ full; ${n_ev_fmt}$ scorable pre-close event days; $h=5$ HLN-DM "
        r"Holm $p$-values vs.\ Global LightGBM; ``---'' indicates volatility-only models).}",
        r"\label{tab:h3_summary}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3.5pt}",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Model (Parameters) & Full QLIKE & Event QLIKE & Tail PR-AUC (Ev.) & ECE & "
        r"DM Holm $p$ (Q/B) \\",
        r"\midrule",
    ]

    for r in rows:
        if r["arm_id"] == "global_lgbm":
            dm_md = dm_tex = "Ref."
        else:
            pq_s = _fmt_p(r["dm_qlike_p_holm"])
            pb_s = _fmt_p(r["dm_brier_p_holm"])
            dm_md = f"{pq_s} / {pb_s}"
            dm_tex = f"{pq_s} / {pb_s.replace('—', '---')}"

        if np.isfinite(r["pr_auc_full"]):
            pr_md = pr_tex = f"{r['pr_auc_full']:.4f} ({r['pr_auc_event_days']:.4f})"
            ece_md = ece_tex = f"{r['ece']:.4f}"
        else:
            pr_md, pr_tex = "—", "---"
            ece_md, ece_tex = "—", "---"

        md_lines.append(
            f"| {r['model_label']} | `{r['qlike_full']:.4f}` | `{r['qlike_event_days']:.4f}` | "
            f"`{pr_md}` | `{ece_md}` | `{dm_md}` |"
        )
        lbl_tex = r["model_label"].replace("_", r"\_").replace("%", r"\%")
        tex_lines.append(
            f"{lbl_tex} & {r['qlike_full']:.4f} & {r['qlike_event_days']:.4f} & "
            f"{pr_tex} & {ece_tex} & {dm_tex} \\\\"
        )

    tex_lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return {
        "comparison_df": comp_df,
        "markdown_table": "\n".join(md_lines) + "\n",
        "latex_table": "\n".join(tex_lines) + "\n",
    }
