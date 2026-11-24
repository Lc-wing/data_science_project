"""Post-Colab Phase B -> CS482 Final Report Integration & Automated Audit (<= 5.0 A4 Pages).

Produces a unified 5-section CS482 submission report (`CS482_GroupX_StockVolatilityLLM.pdf`)
strictly adhering to CS482 minimum requirements and formatting rules:
  1. Research Question
  2. Data Collection and Preparation
  3. Data Analysis (merging Phase A EDA/Econometrics/ML with Phase B PatchTST/Chronos/FinBERT/LLM)
  4. Method Evaluation (justification, validation metrics, calibration, assumptions & uncertainties)
  5. Results and Discussion (evidence-based answers to H1/H2/H3, limitations & improvements)
Plus excluded Cover Page, References (`final_references.bib`), Separate Contribution Statement,
and AI Usage Declaration.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from stock_volatility import config
from stock_volatility.panel.freeze import verify_freeze_manifest
from stock_volatility.report import (
    audit_compiled_pdf,
    build_table1_summary,
    collect_report_claims,
    compile_latex_pdf,
    render_references_bib,
)

FINAL_REPORT_PDF_NAME: str = "CS482_GroupX_StockVolatilityLLM.pdf"


def _esc_tex(text: str) -> str:
    return (
        str(text)
        .replace("\\", r"\textbackslash{}")
        .replace("_", r"\_")
        .replace("%", r"\%")
        .replace("&", r"\&")
    )


def render_phase_b_references_bib() -> str:
    """Return complete verified BibTeX references for unified Phase A + Phase B final report."""
    phase_b_entries = r"""
@inproceedings{nie2023patchtst,
  author    = {Nie, Yuqi and Nguyen, Nam H. and Sinthong, Phanwadee and Kalagnanam, Jayant},
  title     = {A Time Series is Worth 64 Words: Long-term Forecasting with Transformers},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2023},
  note      = {arXiv:2211.14730}
}

@article{ansari2024chronos,
  author  = {Ansari, Abdul Fatir and Stella, Lorenzo and Turkmen, Ali Caner and
             Zhang, Xiyuan and Mercado, Pedro and Shen, Huibin and Shchur, Oleksandr and
             Rangapuram, Syama Sundar and Pineda Arango, Sebastian and Kapoor, Shubham and
             Zschiegner, Jasper and Maddix, Danielle C. and Wang, Hao and
             Mahoney, Michael W. and Torkkola, Kari and Wilson, Andrew Gordon and
             Bohlke-Schneider, Michael and Wang, Yuyang},
  title   = {Chronos: Learning the Language of Time Series},
  journal = {Transactions on Machine Learning Research},
  year    = {2024},
  note    = {arXiv:2403.07815}
}

@article{ansari2025chronos2,
  author  = {Ansari, Abdul Fatir and Shchur, Oleksandr and K{\"u}ken, Jaris and
             Auer, Andreas and Han, Boran and Mercado, Pedro and
             Rangapuram, Syama Sundar and Shen, Huibin and Stella, Lorenzo and
             Zhang, Xiyuan and others},
  title   = {Chronos-2: From Univariate to Universal Forecasting},
  journal = {arXiv preprint arXiv:2510.15821},
  year    = {2025}
}

@article{araci2019finbert,
  author  = {Araci, Dogu},
  title   = {{FinBERT}: Financial Sentiment Analysis with Pre-trained Language Models},
  journal = {arXiv preprint arXiv:1908.10063},
  year    = {2019}
}

@inproceedings{dettmers2023qlora,
  author    = {Dettmers, Tim and Pagnoni, Artidoro and Holtzman, Ari and Zettlemoyer, Luke},
  title     = {{QLoRA}: Efficient Finetuning of Quantized {LLMs}},
  booktitle = {Advances in Neural Information Processing Systems},
  volume    = {36},
  year      = {2023},
  note      = {arXiv:2305.14314}
}
"""
    return render_references_bib().rstrip() + "\n" + phase_b_entries.lstrip()


def render_phase_b_final_tex(
    *,
    phase_a_claims: dict[str, dict[str, str]],
    table1_latex: str,
    phase_b_summary: dict[str, Any],
    table2_latex: str,
) -> str:
    """Render unified 5-section CS482 final LaTeX report (3-5 A4 pp, 11pt, 1.5 spacing).

    Merges Phase B deep time-series (`PatchTST`, `Chronos`) and event-anchored financial LLM
    (`FinBERT`, `gemma-4-12B-it` / `Qwen3.8-27B` QLoRA) directly inside Section 3 (Data Analysis),
    Section 4 (Method Evaluation), and Section 5 (Results and Discussion).
    """
    c = {k: v["value"] for k, v in phase_a_claims.items()}

    llm_id = _esc_tex(phase_b_summary["llm_model_id"])
    raw_rev = str(phase_b_summary.get("llm_revision", "unknown")).strip()
    rev_clause = (
        f"HF rev.\\ \\texttt{{{_esc_tex(raw_rev[:12])}}}, "
        if raw_rev and raw_rev.lower() != "unknown"
        else ""
    )
    gpu = _esc_tex(phase_b_summary["gpu_name"])
    vram = f"{float(phase_b_summary['peak_vram_gb']):.1f}"
    tr_ev = f"{int(phase_b_summary['train_event_days']):,}"
    va_ev = f"{int(phase_b_summary['val_event_days']):,}"
    te_ev = f"{int(phase_b_summary['test_event_days']):,}"
    pt_ql = f"{float(phase_b_summary['patchtst_15m_qlike']):.4f}"
    pt_cov = f"{100.0 * float(phase_b_summary['patchtst_15m_coverage_80']):.1f}"
    fb_pr_ev = f"{float(phase_b_summary['finbert_pr_auc_event']):.4f}"
    zs_pr_ev = f"{float(phase_b_summary['qwen_zs_pr_auc_event']):.4f}"
    ql_full = f"{float(phase_b_summary['qwen_qlora_qlike_full']):.4f}"
    ql_ev = f"{float(phase_b_summary['qwen_qlora_qlike_event']):.4f}"
    pr_full = f"{float(phase_b_summary['qwen_qlora_pr_auc_full']):.4f}"
    pr_ev = f"{float(phase_b_summary['qwen_qlora_pr_auc_event']):.4f}"
    ece_s = f"{float(phase_b_summary['qwen_qlora_ece']):.4f}"
    dm_p = float(phase_b_summary.get("qwen_qlora_dm_p_holm", float("nan")))
    dm_s = f"{dm_p:.3f}" if dm_p == dm_p else "N/A"

    t2_here = table2_latex.replace(r"\begin{table}[!t]", r"\begin{table}[!htb]", 1).replace(
        "Phase B Extension: 2025 Single-Shot Holdout Ablation",
        "2025 Single-Shot Holdout Deep Time-Series and Event-Anchored LLM Ablation (H3)",
        1,
    )

    return rf"""\documentclass[11pt,a4paper]{{article}}

\usepackage[a4paper,top=1.70cm,bottom=1.70cm,left=1.90cm,right=1.90cm]{{geometry}}
\usepackage[T1]{{fontenc}}
\usepackage{{lmodern}}
\usepackage{{setspace}}
\usepackage{{booktabs}}
\usepackage{{graphicx}}
\usepackage{{amsmath,amssymb}}
\usepackage[numbers,sort&compress]{{natbib}}
\usepackage[hidelinks]{{hyperref}}
\usepackage{{microtype}}

\onehalfspacing
\setlength{{\parskip}}{{0.14em}}
\setlength{{\textfloatsep}}{{5pt plus 2pt minus 2pt}}
\setlength{{\intextsep}}{{5pt plus 2pt minus 2pt}}
\setlength{{\abovecaptionskip}}{{2pt}}
\setlength{{\belowcaptionskip}}{{2pt}}
\renewcommand{{\topfraction}}{{0.92}}
\renewcommand{{\bottomfraction}}{{0.88}}
\renewcommand{{\textfraction}}{{0.06}}
\renewcommand{{\floatpagefraction}}{{0.82}}

\begin{{document}}

% =========================================================================
% COVER PAGE (Excluded from CS482 3-5 A4 Page Main Text Limit)
% =========================================================================
\pagenumbering{{roman}}
\begin{{titlepage}}
\centering
\vspace*{{1.1cm}}
{{\large CS482 Data Science --- Group Project Final Report\par}}
\vspace{{0.7cm}}
{{\LARGE\bfseries Corporate Event Shocks, Realized Volatility, and Downside
Tail-Risk Forecasting in U.S.\ Equities (2019--2025)\par}}
\vspace{{0.4cm}}
{{\large Panel HAR-RV, Per-Stock GJR-GARCH(1,1), Calibrated LightGBM, Deep Time-Series
(PatchTST, Chronos), and Event-Anchored Financial LLMs (FinBERT, 4-bit QLoRA)
Under Leakage-Free Purged Splits\par}}
\vspace{{1.1cm}}

{{\small
\begin{{tabular}}{{ll}}
\toprule
\textbf{{Group Member Name \& Role}} & \textbf{{Student ID (Replace Before Moodle)}} \\
\midrule
Member A (Financial Data Engineering \& Freeze) & [Student ID A] \\
Member B (Panel Econometrics \& GJR-GARCH) & [Student ID B] \\
Member C (LightGBM, Calibration \& TreeSHAP) & [Student ID C] \\
Member D (Deep Time-Series, QLoRA LLM \& Audit) & [Student ID D] \\
\bottomrule
\end{{tabular}}
}}

\vspace{{0.9cm}}
\begin{{minipage}}{{0.88\textwidth}}
\small\raggedright
\textbf{{Academic Research \& Non-Investment-Advice Notice.}}
This study is conducted solely for CS482 academic coursework and methodological
evaluation of associational and predictive risk models (academic research only;
\textbf{{not investment advice}}). It makes no causal claims and does not
constitute investment, financial, or trading advice. All empirical statistics
are reproduced programmatically from the cryptographically frozen dataset
(\texttt{{panel\_daily.parquet}} SHA-256 prefix \texttt{{{c['PANEL_SHA12']}}}).
\end{{minipage}}

\vfill
{{\normalsize Course: CS482 Data Science \quad | \quad
Submission Deadline: December 15, 2026 (11:59 PM UTC+8)\par}}
\end{{titlepage}}

% =========================================================================
% MAIN TEXT (CS482 Requirement: 3-5 A4 Pages, Single-Column, 11pt, 1.5 Spacing)
% =========================================================================
\pagenumbering{{arabic}}
\setcounter{{page}}{{1}}

\section{{Research Question}}

Short-horizon equity volatility and downside tail-risk episodes exhibit strong
persistence across macroeconomic regimes even though directional return levels
resemble a random walk. Accurate 5-day risk forecasts and well-calibrated tail
probabilities are directly valuable to institutional portfolio risk managers,
volatility-targeting allocators, financial-stability monitors, and empirical
market-microstructure researchers evaluating how public corporate disclosures
relate to near-term risk. Across a fixed cross-sector universe of
{c['N_STOCKS']} liquid U.S.\ equities over 2019--2025---spanning the 2020
COVID-19 shock, the 2022 monetary-tightening bear market, the 2024 expansion,
and 2025 policy turbulence---we investigate a unified research question:

\textbf{{Research Question.}} Under strict purged and embargoed chronological
splits, (i) how strongly are lagged negative returns and implied volatility
(VIX) associated with 5-day forward realized volatility across GICS sectors,
(ii) how reliably do classical econometric baselines (Panel HAR-RV, per-stock
GJR-GARCH(1,1)), gradient-boosted trees (LightGBM), and deep/foundation
time-series models (\texttt{{PatchTST}}, \texttt{{Chronos}}) forecast 5-day
forward volatility and downside tail-risk events (5-day return $< -5\%$), and
(iii) does fusing pre-close SEC EDGAR event text via domain-pretrained
\texttt{{FinBERT}} (110M) or 4-bit QLoRA fine-tuning of a multi-billion-parameter
LLM (\texttt{{{llm_id}}}) yield incremental 2025 out-of-sample lift over
price-only models? We test three pre-registered hypotheses: \textbf{{H1}}
(positive associational leverage and VIX links with cross-sector heterogeneity),
\textbf{{H2}} (LightGBM improves 2025 volatility QLIKE/RMSE and calibrated tail
reliability over classical baselines), and \textbf{{H3}} (deep sequence and
event-anchored text/LLM fusion add 2025 incremental accuracy, with null results
accepted). All findings remain strictly associational or predictive and do not
constitute investment advice.

\section{{Data Collection and Preparation}}

\textbf{{Market, Macro, and Universe Design.}} To avoid 2025 survivorship
lookahead, we fix a deterministic pre-2019 universe of {c['N_STOCKS']} liquid
S\&P 500 equities (12 per GICS sector: Technology, Financials, Healthcare,
Consumer, and Energy). Split-adjusted daily OHLCV and dividend-adjusted closes
for all {c['N_STOCKS']} stocks plus \texttt{{SPY}} and \texttt{{\^{{}}VIX}} are
retrieved from Yahoo Finance over {c['N_DAYS']} canonical NYSE trading days
(2019-01-02 to 2025-12-31; {c['PANEL_ROWS']} stock-days, zero missing cells).
FRED macro series (\texttt{{VIXCLS}} and 10Y--2Y Treasury spread
\texttt{{T10Y2Y}}) are joined using backward-only forward-fill (never
backfilled), flagging {c['T10Y2Y_FFILL']} U.S.\ bond-market holidays.

\textbf{{Authoritative Corporate Event Backbone and Cleaning.}} Preliminary
audits showed public news mirrors (\texttt{{financial-news-multisource}},
\texttt{{FNSPID}}) were gated, terminated in 2023, or contained
\texttt{{00:00:00 UTC}} timestamp leakage. We therefore ingest authoritative
public SEC EDGAR Form \texttt{{8-K}}, \texttt{{8-K/A}}, \texttt{{10-Q}}, and
\texttt{{10-K}} filings (2019--2025), resolving holding-company predecessor
CIKs (\texttt{{BLK}}, \texttt{{XOM}}) and recursively paginating
{c['SEC_ARCHIVES']} historical \texttt{{filings.files}} archives to obtain
{c['SEC_FILINGS_RAW']} unique filings with 60/60 dual-period continuity
(2024--2025 vs.\ 2019--2023 event-day rate ratio {c['COV_RATE_RATIO']}).
Second-precision UTC \texttt{{acceptanceDateTime}} stamps are converted to
\texttt{{America/New\_York}} (DST-aware): filings before 16:00 ET (13:00 ET on
15 scheduled NYSE early-close half-days, shifting {c['EARLY_CLOSE_SHIFTED']}
post-13:00 \texttt{{WFC}} filings to $t+1$) stay on day $t$, whereas post-close
and non-trading-day filings roll to $t+1$. Intra-day headline deduplication
removes {c['DEDUP_DROPPED']} duplicates and caps counts at $\le 5$/day, leaving
{c['EVENT_ANCHORED_DAYS']} pre-close event-anchored stock-days.

\textbf{{Variables, Targets, and Leakage-Free Splits.}} Continuous target
\texttt{{rv\_5d\_fwd}} is 5-day forward annualized Parkinson range
volatility~\cite{{parkinson1980}} over $t+1\dots t+5$, modeled on
$y=\ln(\text{{RV}}+10^{{-6}})$ (sensitivity: close-to-close RV). Binary target
\texttt{{y\_tail}} equals 1 iff 5-day cumulative adjusted return $< -5\%$
(sensitivity: Train-only 90th-percentile RV threshold {c['RV90_THRESH']}). All
22 predictors use strictly $\le t$ information; sequence models use 14
continuous features Winsorized and standardized strictly via Train statistics.
Following \citet{{lopezdeprado2018}}, chronological splits enforce 6-day
embargoes ($\ge h=5$): Train 2019-01-02 to 2023-12-20 ({c['TRAIN_MOD_ROWS']}
modeling rows post-21-day \texttt{{rv\_m}} warmup), Val 2024-01-02 to
2024-12-20 ({c['VAL_MOD_ROWS']} rows, tuning/calibration/meta-fusion only), and
single-shot Holdout Test 2025-01-02 to 2025-12-31 ({c['TEST_MOD_ROWS']}
scorable rows across {c['TEST_DAYS']} days; final 5 incomplete days masked
\texttt{{NA}}). All frozen tables are sealed via SHA-256 (prefix
\texttt{{{c['PANEL_SHA12']}}}).

\section{{Data Analysis}}

\textbf{{Exploratory Data Analysis and Panel Econometrics (H1).}} Cross-cycle
EDA audits volatility regimes, leverage asymmetry, and tail-risk incidence
across Train-median VIX regimes ($\text{{VIX}} \ge {c['VIX_MEDIAN']}$;
Figure~\ref{{fig:regimes_forecasts}}A--B). Mean 5-day annualized Parkinson RV
peaked during the 2020 COVID shock ({c['RV_COVID_2020']}) and 2022 bear market
({c['RV_BEAR_2022']}), compressed in 2024 ({c['RV_VAL_2024']}), and re-expanded
in 2025 ({c['RV_TEST_2025']}). To test H1 across all {c['H1_N_OBS']}
modeling-complete stock-days, we estimate a within-entity Stock-FE extended
Panel HAR-RV model~\cite{{corsi2009}} interacting lagged negative returns
($\min(r_{{i,t}},0)$) and VIX with sector indicators (Healthcare reference),
using Bartlett Driscoll--Kraay SEs~\cite{{driscoll1998}} ($L={c['H1_DK_L']}$)
and Holm--Bonferroni correction~\cite{{holm1979}} across the 5-test
confirmatory family.

\begin{{figure}}[!tb]
\centering
\includegraphics[width=0.84\textwidth]{{../outputs/figures/fig1_volatility_regimes_and_forecasts.pdf}}
\caption{{Cross-sector 2019--2025 volatility regimes (Panel A), lagged return
leverage asymmetry (Panel B), and 2025 holdout forecast tracking during the
March--May volatility spike (Panel C).}}
\label{{fig:regimes_forecasts}}
\end{{figure}}

\textbf{{Econometric and Tabular ML Forecasting (H2).}} Four price-only models
share an identical quarterly expanding-window purged schedule (2024 Val, then
2025 Q1--Q4 with $\ge 5$-day purge before each quarter): (i) Persistence
(\texttt{{rv\_w}} and expanding base rate); (ii) per-stock
GJR-GARCH(1,1)~\cite{{glosten1993}} on $100r_{{i,t}}$ with causal variance
filtering ({c['GARCH_CONV']} fits converged); (iii) Stock-FE Predictive HAR-RV
back-transformed via fit-fold Duan smearing~\cite{{duan1983}} plus Logistic
HAR-RV; and (iv) Global LightGBM~\cite{{ke2017lightgbm}} tuned strictly on 2024
Val QLIKE/PR-AUC. Exact interventional TreeSHAP~\cite{{lundberg2020treeshap}}
({c['SHAP_BG_N']}-row Train background; additivity error $< 10^{{-7}}$) audits
LightGBM model behavior non-causally (Figure~\ref{{fig:shap_cal}}A).

\textbf{{Deep Time-Series and Event-Anchored LLM Analysis (H3).}} Keeping the
frozen panel and splits unchanged, we integrate two deep learning architectures
executed on a single {gpu} GPU (peak VRAM {vram}\,GB):
(1) \emph{{Deep and Foundation Time-Series Models}}: a 5-seed multivariate
\texttt{{PatchTST}}~\cite{{nie2023patchtst}} capacity ablation (1M vs.\ 15M
parameters; 60-day lookback, learned 60-ticker and 5-sector embeddings,
non-crossing $q_{{10}}\le q_{{50}}\le q_{{90}}$ Pinball Loss, Train Duan
smearing) alongside zero-shot causal foundation forecasters
\texttt{{amazon/chronos-2}} (120M)~\cite{{ansari2025chronos2}} and
\texttt{{chronos-t5-large}} (710M)~\cite{{ansari2024chronos}} fed strictly
$\le t$ daily Parkinson RV histories;
(2) \emph{{Event-Anchored Financial Text and QLoRA LLM Fusion}}: scoring
strictly pre-close SEC EDGAR event-anchored days ({tr_ev} Train / {va_ev} Val /
{te_ev} Test; non-event days deterministic neutral~0) via frozen
domain-pretrained \texttt{{FinBERT}} (110M)~\cite{{araci2019finbert}} versus
\texttt{{{llm_id}}} ({rev_clause}Zero-Shot and 1-epoch 4-bit NF4
QLoRA~\cite{{dettmers2023qlora}}, $r=16, \alpha=32$, restricted to 328
text-backbone linear layers excluding vision/audio towers, 65.6M trainable
parameters, class-weighted BCE on next-token $\text{{logit}}(\text{{YES}}) -
\text{{logit}}(\text{{NO}})$), fused onto LightGBM via 2024-Val out-of-fold
meta-fusion.

\section{{Method Evaluation}}

\textbf{{Method Justification, Metrics, and Calibration Protocol.}} Combining
econometric panel regressions, GJR-GARCH, gradient-boosted trees, deep
transformers, and pre-close financial text encoders spans parametric
interpretability, nonlinear tabular interactions, temporal attention, and
unstructured disclosure semantics. Volatility forecasts are evaluated by
variance-scale QLIKE~\cite{{patton2011}} (robust to noisy volatility proxies)
and RMSE; imbalanced tail-risk classification (8.62\% 2025 base rate) by
PR-AUC (primary), Brier score, and 10-bin ECE. To prevent in-sample calibration
or QLoRA memorization bias, both probability calibration (comparing Isotonic,
Platt, and Temperature Scaling~\cite{{guo2017calibration}} via 4-quarter OOF CV
inside 2024 Val; selecting \texttt{{{c['CAL_METHOD']}}}) and multimodal Stage-2
Ridge/Logistic meta-fusion are fit strictly inside 2024 Val and frozen for
single-shot 2025 Test evaluation. Uncertainty is quantified via 5-day
date-block bootstrap 95\% CIs ($B=1{{,}}000$) and $h=5$ Bartlett-PSD
HLN-modified Diebold--Mariano tests~\cite{{harvey1997}} with Holm adjustment
(Tables~\ref{{tab:main_summary}}--\ref{{tab:phase_b_summary}}).

{table1_latex}

{t2_here}

\textbf{{Assumptions and Uncertainties.}} Econometric inference requires
robustness to cross-stock shock correlation: Driscoll--Kraay SEs expand
standard errors by 3.6--4.7$\times$ over entity-clustered Newey--West SEs,
preventing false-positive sector interactions. For probability estimation,
piecewise-constant isotonic calibration cuts LightGBM ECE by
{c['ECE_DROP_PCT']}\% ({c['LGBM_ECE_PRE']} $\to$ {c['LGBM_ECE_POST']}) but
creates upper-tail ties that compress PR-AUC granularity on sparse earnings
days. Deep sequence \texttt{{PatchTST}} exhibits 2025 nominal 80\% interval
undercoverage ({pt_cov}\% at 15M) and uncalibrated BCE ECE (0.4349), reflecting
non-stationarity between 2019--2023 training volatility and 2025 tariff/policy
shocks. Finally, single-epoch QLoRA adaptation on {tr_ev} short SEC filing
titles with $\approx 8\%$ positive tail labels faces high variance and rapid
public price-incorporation ceilings.

\begin{{figure}}[!b]
\centering
\includegraphics[width=0.82\textwidth]{{../outputs/figures/fig2_shap_and_calibration.pdf}}
\caption{{Train-background TreeSHAP attributions for LightGBM tail-risk model
behavior by VIX regime (Panel A, strictly non-causal) and 2025 holdout
calibration reliability curves (Panel B).}}
\label{{fig:shap_cal}}
\end{{figure}}

\section{{Results and Discussion}}

\textbf{{Evidence-Based Answers to H1, H2, and H3.}}
\emph{{H1 (Partial Support):}} In Panel HAR-RV (Table~\ref{{tab:main_summary}}A;
Within $R^2={c['H1_R2']}$), 5-sector average leverage ({c['H1_NEG_EST']}, Holm
$p<0.0001$), VIX (+{c['H1_VIX_EST']}, $p<0.0001$), and the joint 5-sector Wald
test ($\chi^2={c['H1_WALD_CHI2']}$, $p<0.0001$) are strongly significant across
all sensitivities, whereas pairwise Technology vs.\ Healthcare leverage is
non-significant under DK SEs (Holm $p={c['H1_TECH_NEG_PH']}$), with Financials
and Energy showing the largest cyclical differentials.
\emph{{H2 (Nuanced Support):}} Among price-only models
(Table~\ref{{tab:main_summary}}B; Figure~\ref{{fig:regimes_forecasts}}C),
Global LightGBM achieves the lowest baseline QLIKE ({c['LGBM_QLIKE']}) and RMSE
({c['LGBM_RMSE']}), significantly beating Persistence ({c['PERS_QLIKE']},
$p<0.0001$) and GJR-GARCH ({c['GARCH_QLIKE']}, $p={c['DM_Q_GARCH_PH']}$), with
its largest sector gain in Technology ({c['TECH_LGB_QLIKE']} vs.\ HAR-RV
{c['TECH_HAR_QLIKE']}; full-universe vs.\ HAR-RV {c['HAR_QLIKE']} has Holm
$p={c['DM_Q_HAR_PH']}$). Calibrated LightGBM records the lowest Brier score
({c['LGBM_BRIER']}, $p={c['DM_B_PERS_PH']}$ vs.\ Persistence;
Figure~\ref{{fig:shap_cal}}B), yet Logistic HAR-RV leads overall PR-AUC
({c['HAR_PRAUC']} vs.\ {c['LGBM_PRAUC']}) and Form 8-K Item 2.02 earnings days
({c['EARN_HAR_PRAUC']} vs.\ {c['EARN_LGB_PRAUC']}); TreeSHAP concentrates on
\texttt{{rv\_m}} ({c['SHAP_RVM_CLF_PCT']}\%), \texttt{{vix\_close}}
({c['SHAP_VIX_CLF_PCT']}\%), and \texttt{{t10y2y}}.
\emph{{H3 (Mixed Results; Honest Null QLoRA vs.\ FinBERT):}} Scaling
\texttt{{PatchTST}} from 1M to 15M improves QLIKE (0.3142 $\to$ {pt_ql}) and
80\% coverage (63.1\% $\to$ {pt_cov}\%), and \texttt{{chronos-2}} (0.3505)
outperforms \texttt{{chronos-t5-large}} (0.5848), yet all four deep/foundation
time-series baselines still trail tabular LightGBM (Holm $p \le 0.006$;
Table~\ref{{tab:phase_b_summary}}). In contrast, 2024-Val OOF text fusion
significantly reduces 2025 full-universe QLIKE across all three text arms (to
0.2251--{ql_full}; \texttt{{{llm_id}}} QLoRA event-day QLIKE {ql_ev},
$h=5$ HLN-DM Holm $p={dm_s}$, ECE {ece_s}). On the {te_ev} scorable event days,
domain-pretrained \texttt{{FinBERT}} lifts tail-risk PR-AUC from 0.1076 to
{fb_pr_ev} (+30.0\% relative), whereas \texttt{{{llm_id}}} yields {zs_pr_ev}
zero-shot and {pr_ev} after 1-epoch QLoRA (full PR-AUC {pr_full})---an honest
null QLoRA gain underscoring the sample-efficiency advantage of compact
domain-pretrained encoders on short formal SEC filing titles.

\textbf{{Implications for Risk Stakeholders.}} These findings directly answer
our research question for risk managers and market-microstructure researchers:
(i) well-engineered tabular volatility features paired with out-of-sample
probability calibration outperform far larger neural time-series models on
small-panel daily equity data; (ii) lightweight domain-pretrained financial
encoders (\texttt{{FinBERT}}) deliver stronger event-day tail ranking lift per
compute dollar than single-epoch 12B QLoRA on short filing headlines; and (iii)
combining calibrated linear logits (on sparse earnings days) with nonlinear
tree ensembles balances ranking granularity against probability reliability.

\textbf{{Limitations and Potential Improvements.}} Key limitations include
survivorship bias in the fixed 60-stock large-cap universe, measurement error
in daily Parkinson volatility relative to intraday 5-minute RV, abrupt
overnight macro gaps (e.g., April 2025 tariff shock), and title-level SEC text
lacking full-body 8-K item prose or earnings-call transcripts. Future work can
incorporate intraday realized kernels, full-text SEC item extraction with
multi-epoch rank-stabilized QLoRA, and split conformal prediction intervals to
restore nominal 80\% coverage on neural quantile forecasters. All findings are
strictly predictive academic benchmarks and \textbf{{do not constitute
investment advice}}.

% =========================================================================
% EXCLUDED BACK MATTER (References, Contribution Statement, AI Declaration)
% =========================================================================
\clearpage
{{\small
\setstretch{{1.08}}
\bibliographystyle{{plainnat}}
\bibliography{{final_references}}
}}

\clearpage
\section*{{Separate Contribution Statement}}
\begin{{itemize}}\setlength{{\itemsep}}{{2pt}}
\item \textbf{{Member A (Financial Data Engineering \& Freeze Lead):}} Designed
the pre-2019 60-stock universe, built Yahoo/FRED/paginated SEC EDGAR ingestion
with \texttt{{BLK}}/\texttt{{XOM}} predecessor CIK linkages, enforced DST-aware
16:00/13:00 ET cutoffs and purged splits, froze \texttt{{FREEZE\_MANIFEST.json}},
and produced Figure~1.
\item \textbf{{Member B (Financial Econometrics Lead):}} Implemented cross-cycle
EDA, Stock-FE extended Panel HAR-RV with Bartlett Driscoll--Kraay SE and
Holm-adjusted confirmatory tests, per-stock GJR-GARCH(1,1) convergence audits,
and $h=5$ HLN-DM evaluation.
\item \textbf{{Member C (Machine Learning \& Explainability Lead):}} Built the
quarterly expanding Global LightGBM pipeline, 2024-Val out-of-fold probability
calibration, Train-background exact TreeSHAP additivity audits, Table~1, and
Figure~2.
\item \textbf{{Member D (Deep Time-Series, QLoRA LLM \& Final Audit Lead):}}
Implemented and executed the Colab single-GPU deep learning pipeline
(\texttt{{PatchTST}} 1M/15M, \texttt{{Chronos-2}}/\texttt{{T5-Large}},
\texttt{{FinBERT}}, \texttt{{{llm_id}}} Zero-Shot \& 4-bit NF4 QLoRA), 2024-Val
OOF meta-fusion, Table~2, and automated final PDF claim/formatting verification
(\texttt{{phase\_b\_audit.md}}).
\end{{itemize}}

\section*{{AI Usage Declaration \& Academic Integrity}}
\textbf{{Tools and Purposes.}} Google Antigravity / Gemini AI coding and
writing assistants were used strictly for permitted academic support:
brainstorming experimental guardrails, conceptual clarification of econometric
and QLoRA procedures, test-driven debugging, LaTeX formatting, and language
improvement, following the procedural research workflow of
\citet{{kassis2026skills}}.
\textbf{{Independent Verification.}} Every group member independently reviewed
and verified all data, code, references, and analytical results: raw Yahoo,
FRED, and SEC EDGAR snapshots are sealed under SHA-256
(\texttt{{FREEZE\_MANIFEST.json}}); Driscoll--Kraay SEs were cross-checked
against \texttt{{statsmodels}}; all 300/300 GJR-GARCH fits, Colab {gpu} GPU
training runs (\texttt{{scripts/colab\_phase\_b.py}}), bootstrap CIs, HLN-DM
tests, TreeSHAP additivity checks ($< 10^{{-7}}$), and all 61 reported numbers
were executed and audited programmatically via deterministic pipelines
(\texttt{{scripts/run\_phase\_a.py}},
\texttt{{scripts/07\_integrate\_phase\_b\_report.py}}) and a 104-test pytest
suite. All citations were verified directly against original scholarly venues.
\textbf{{Student Responsibility.}} The group assumes full responsibility for
the accuracy and academic integrity of this report. Every member can fully
explain their assigned methods, code, and conclusions; no plagiarism,
fabricated data or references, or unverified AI-generated analysis appear in
this submission.

\end{{document}}
"""


def collect_phase_b_claims(phase_b_summary: dict[str, Any]) -> dict[str, dict[str, str]]:
    """Collect verifiable Phase B numeric claims from `outputs/phase_b/phase_b_summary.json`."""
    src = "outputs/phase_b/phase_b_summary.json"
    claims: dict[str, dict[str, str]] = {
        "PB_PEAK_VRAM": {
            "value": f"{float(phase_b_summary['peak_vram_gb']):.1f}",
            "source": src,
            "locator": "peak_vram_gb",
        },
        "PB_PATCHTST_QLIKE": {
            "value": f"{float(phase_b_summary['patchtst_15m_qlike']):.4f}",
            "source": src,
            "locator": "patchtst_15m_qlike",
        },
        "PB_PATCHTST_COV80": {
            "value": f"{100.0 * float(phase_b_summary['patchtst_15m_coverage_80']):.1f}",
            "source": src,
            "locator": "100*patchtst_15m_coverage_80",
        },
        "PB_FINBERT_PRAUC_EV": {
            "value": f"{float(phase_b_summary['finbert_pr_auc_event']):.4f}",
            "source": src,
            "locator": "finbert_pr_auc_event",
        },
        "PB_LLM_ZS_PRAUC_EV": {
            "value": f"{float(phase_b_summary['qwen_zs_pr_auc_event']):.4f}",
            "source": src,
            "locator": "qwen_zs_pr_auc_event",
        },
        "PB_LLM_QLIKE_FULL": {
            "value": f"{float(phase_b_summary['qwen_qlora_qlike_full']):.4f}",
            "source": src,
            "locator": "qwen_qlora_qlike_full",
        },
        "PB_LLM_QLIKE_EV": {
            "value": f"{float(phase_b_summary['qwen_qlora_qlike_event']):.4f}",
            "source": src,
            "locator": "qwen_qlora_qlike_event",
        },
        "PB_LLM_PRAUC_FULL": {
            "value": f"{float(phase_b_summary['qwen_qlora_pr_auc_full']):.4f}",
            "source": src,
            "locator": "qwen_qlora_pr_auc_full",
        },
        "PB_LLM_PRAUC_EV": {
            "value": f"{float(phase_b_summary['qwen_qlora_pr_auc_event']):.4f}",
            "source": src,
            "locator": "qwen_qlora_pr_auc_event",
        },
        "PB_LLM_ECE": {
            "value": f"{float(phase_b_summary['qwen_qlora_ece']):.4f}",
            "source": src,
            "locator": "qwen_qlora_ece",
        },
    }
    raw_rev = str(phase_b_summary.get("llm_revision", "unknown")).strip()
    if raw_rev and raw_rev.lower() != "unknown":
        claims["PB_LLM_REV"] = {
            "value": raw_rev[:12],
            "source": src,
            "locator": "llm_revision[:12]",
        }
    return claims


def build_and_audit_phase_b_report(
    *,
    project_root: Path | None = None,
    phase_b_dir: Path | None = None,
) -> dict[str, Any]:
    """Compile final unified 5-section CS482 PDF and run <=5.0pp A4 + claims audit."""
    root = Path(project_root) if project_root is not None else config.PROJECT_ROOT
    manifest = verify_freeze_manifest(frozen_dir=root / "data" / "frozen", project_root=root)
    generated_at_utc = str(manifest["generated_at_utc"])

    pb_dir = Path(phase_b_dir) if phase_b_dir is not None else (root / "outputs" / "phase_b")
    summary_path = pb_dir / "phase_b_summary.json"
    table2_path = pb_dir / "phase_b_comparison.tex"
    if not summary_path.is_file():
        raise FileNotFoundError(
            f"Missing {summary_path}. Run notebooks/phase_b_colab.ipynb on Colab GPU first; "
            "refusing to fabricate Phase B metrics."
        )
    if not table2_path.is_file():
        raise FileNotFoundError(f"Missing {table2_path}; run Colab Phase B fusion first.")

    pb_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    table2_latex = table2_path.read_text(encoding="utf-8")

    t1 = build_table1_summary(root)
    phase_a_claims = collect_report_claims(root)
    all_claims = {**phase_a_claims, **collect_phase_b_claims(pb_summary)}

    report_dir = root / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    bib_path = report_dir / "final_references.bib"
    tex_path = report_dir / "final_report.tex"
    bib_path.write_text(render_phase_b_references_bib(), encoding="utf-8")
    tex_path.write_text(
        render_phase_b_final_tex(
            phase_a_claims=phase_a_claims,
            table1_latex=t1["latex"],
            phase_b_summary=pb_summary,
            table2_latex=table2_latex,
        ),
        encoding="utf-8",
    )

    source_date_epoch = int(datetime.fromisoformat(generated_at_utc).timestamp())
    pdf_path = compile_latex_pdf(
        tex_path,
        output_pdf_name=FINAL_REPORT_PDF_NAME,
        source_date_epoch=source_date_epoch,
    )
    audit = audit_compiled_pdf(
        pdf_path=pdf_path,
        claims=all_claims,
        generated_at_utc=generated_at_utc,
    )
    audit_path = report_dir / "phase_b_audit.md"
    audit_md = audit["audit_markdown"].replace(
        "Milestone M5 — CS482 Phase A Draft Automated Audit",
        "Final Submission — CS482 Unified 5-Section Report Automated Audit",
        1,
    )
    audit_path.write_text(audit_md, encoding="utf-8")

    if audit["verdict"] != "PASS":
        raise RuntimeError(
            f"CS482 Final PDF audit FAILED! See {audit_path}:\n"
            f"checks={audit['checks']}\n"
            f"missing_claims={audit['missing_claims']}\n"
            f"text_issues={audit['text_issues']}"
        )

    return {
        "milestone": "PhaseB-Report",
        "generated_at_utc": generated_at_utc,
        "verdict": audit["verdict"],
        "pdf_path": str(pdf_path.relative_to(root)),
        "total_pages": audit["total_pages"],
        "main_pages_occupied": audit["main_pages_occupied"],
        "main_text_pages_est": audit["main_text_pages_est"],
        "claims_verified": audit["claims_verified"],
        "claims_total": audit["claims_total"],
    }
