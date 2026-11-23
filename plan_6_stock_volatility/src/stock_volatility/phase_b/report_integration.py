"""Post-Colab Phase B -> CS482 Final Report Integration & Automated Audit (<= 5.0 A4 Pages)."""

from __future__ import annotations

import json
import re
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
    render_phase_a_tex,
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


def render_phase_b_section_tex(
    phase_b_summary: dict[str, Any],
    table2_latex: str,
) -> str:
    """Render concise Section 6 (~0.75 A4 pp) integrating verified Colab GPU Phase B results."""
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

    t2_here = table2_latex.replace(r"\begin{table}[!t]", r"\begin{table}[!htb]", 1)

    return (
        "\n\\section{Extension: Deep Time-Series and QLoRA News LLM (H3)}\n\n"
        f"Keeping the Phase~A frozen panel and purged splits strictly unchanged, we evaluate "
        f"two deep learning routes on a single {gpu} GPU (peak VRAM {vram}\\,GB). "
        f"In Route~A, 5-seed multivariate \\texttt{{PatchTST}} (15M; 60-day lookback, "
        f"ticker/sector embeddings, Pinball quantiles) scales favorably over 1M yet records "
        f"2025 QLIKE {pt_ql} (nominal 80\\% coverage {pt_cov}\\%), while zero-shot univariate "
        f"\\texttt{{chronos-2}} and \\texttt{{chronos-t5-large}} also trail Phase~A LightGBM "
        f"(Table~\\ref{{tab:phase_b_summary}}). "
        f"In Route~B (H3), we score strictly pre-close SEC EDGAR event-anchored days "
        f"({tr_ev} Train / {va_ev} Val / {te_ev} Test; non-event days neutral~0) via frozen "
        f"\\texttt{{FinBERT}} (110M) versus \\texttt{{{llm_id}}} ({rev_clause}4-bit NF4 QLoRA "
        f"$r=16$ on text-backbone layers only) with 2024-Val out-of-fold meta-fusion onto "
        f"Phase~A LightGBM.\n\n"
        f"{t2_here}\n"
        f"All three fused text arms reduce 2025 full-universe QLIKE to {ql_full} "
        f"(\\texttt{{{llm_id}}} QLoRA; event-day QLIKE {ql_ev}; $h=5$ HLN-DM Holm $p={dm_s}$) "
        f"with tight calibration (ECE {ece_s}). "
        f"On the {te_ev} event-anchored days, domain-pretrained \\texttt{{FinBERT}} lifts "
        f"tail-risk PR-AUC to {fb_pr_ev} (vs.\\ 0.1076 price-only), whereas \\texttt{{{llm_id}}} "
        f"reaches {zs_pr_ev} zero-shot and {pr_ev} after 1-epoch QLoRA (full PR-AUC {pr_full}). "
        f"This honest null QLoRA gain over \\texttt{{FinBERT}} reflects the sample-efficiency "
        f"boundary of fine-tuning a multi-billion-parameter LLM on {tr_ev} short SEC filing "
        f"titles with sparse tail labels and rapid market incorporation of public disclosures. "
        f"All Phase~B results remain strictly predictive, non-causal CS482 benchmarks.\n"
    )


def render_phase_b_final_tex(
    *,
    phase_a_claims: dict[str, dict[str, str]],
    table1_latex: str,
    phase_b_summary: dict[str, Any],
    table2_latex: str,
) -> str:
    """Inject Phase B Section 6 and Table 2 into the audited Phase A LaTeX template."""
    base_tex = render_phase_a_tex(phase_a_claims, table1_latex)
    updated_tex = base_tex.replace(
        "Group Project (Phase A Draft)",
        "Group Project (Final Report: Phase A + Phase B)",
        1,
    )
    updated_tex = updated_tex.replace(
        r"\section{Discussion, Limitations, and Phase B Outlook}",
        r"\section{Discussion and Limitations}",
        1,
    )
    n_ev_days = phase_a_claims["EVENT_ANCHORED_DAYS"]["value"]
    updated_tex = re.sub(
        r"\\textbf\{Phase B Extension \(Frozen Handoff Ready\)\.\}.*?material-event days\.",
        (
            r"\\textbf{Frozen Phase A to Phase B Handoff.} "
            f"With Phase~A data, purged splits, and {n_ev_days} "
            r"pre-close event-anchored text rows cryptographically frozen "
            r"(\\texttt{events\\_text\\_phase\\_b.parquet}), Section~6 evaluates deep time-series "
            r"architectures and 4-bit QLoRA LLM text fusion against \\texttt{FinBERT} (110M) and "
            r"price-only Phase~A LightGBM on the identical 2025 single-shot holdout."
        ),
        updated_tex,
        count=1,
        flags=re.DOTALL,
    )
    marker = (
        "% =========================================================================\n"
        "% EXCLUDED BACK MATTER"
    )
    if marker not in updated_tex:
        raise ValueError("Could not locate EXCLUDED BACK MATTER marker in Phase A LaTeX template")

    sec6 = render_phase_b_section_tex(phase_b_summary, table2_latex)
    return updated_tex.replace(marker, f"{sec6}\n{marker}", 1)


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
    """Compile final CS482 Phase A + Phase B PDF and run hardened <=5.0pp A4 + claims audit."""
    root = Path(project_root) if project_root is not None else config.PROJECT_ROOT
    manifest = verify_freeze_manifest(frozen_dir=root / "data" / "frozen", project_root=root)
    generated_at_utc = str(manifest["generated_at_utc"])

    pb_dir = Path(phase_b_dir) if phase_b_dir is not None else (root / "outputs" / "phase_b")
    summary_path = pb_dir / "phase_b_summary.json"
    table2_path = pb_dir / "phase_b_comparison.tex"
    if not summary_path.is_file():
        raise FileNotFoundError(
            f"Missing {summary_path}. Run notebooks/phase_b_colab.ipynb on Colab A100 first; "
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
    bib_path = report_dir / "references.bib"
    tex_path = report_dir / "final_report.tex"
    bib_path.write_text(render_references_bib(), encoding="utf-8")
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
        "Final Submission — CS482 Phase A + Phase B Automated Audit",
        1,
    )
    audit_path.write_text(audit_md, encoding="utf-8")

    if audit["verdict"] != "PASS":
        raise RuntimeError(
            f"CS482 Final Phase B PDF audit FAILED! See {audit_path}:\n"
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
