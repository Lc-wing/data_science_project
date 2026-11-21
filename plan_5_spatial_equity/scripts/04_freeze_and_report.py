#!/usr/bin/env python3
"""Milestone M4 CLI: freeze cells, draw figures, and compile the Phase A draft.

Numbers in Table 1 and the draft are copied from the committed M2 and M3
outputs. This script does not refit those models.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from macao_equity.freeze_bundle import (  # noqa: E402
    build_canonical_bundles,
    default_bundle_paths,
    write_data_dictionary,
    write_freeze_manifest,
)
from macao_equity.viz.figures import write_all_figures  # noqa: E402
from macao_equity.viz.report import (  # noqa: E402
    build_prose_claims,
    compile_pdf,
    measure_main_pages,
    render_audit,
    substitution_map,
    write_report_sources,
)
from macao_equity.viz.tables import write_table1  # noqa: E402

PAGE_LO = 3.55
PAGE_HI = 3.80


def _fit_main_text(root: Path, table_claims: list, prose_claims: list) -> float:
    """Compile the draft and nudge figure height and trailing space into 3.55–3.75 pages."""
    fig_frac = 0.24
    pad_pt = 0
    pages = 0.0
    for attempt in range(12):
        mapping = substitution_map(
            table_claims,
            prose_claims,
            fig_frac=fig_frac,
            pad_pt=pad_pt,
        )
        tex_path = write_report_sources(root, mapping)
        pdf_path = compile_pdf(tex_path)
        pages = measure_main_pages(pdf_path, root)
        print(
            f"report fit attempt {attempt}: main_pages={pages:.3f} "
            f"fig_frac={fig_frac:.2f} pad_pt={pad_pt}"
        )
        if PAGE_LO <= pages <= PAGE_HI:
            return pages
        # Positive space only. A negative skip pulls the references heading
        # up into the discussion and drops the last limitation out of the main text.
        if pages < PAGE_LO:
            pad_pt += max(12, int(round((PAGE_LO - pages) * 480)))
            pad_pt = min(pad_pt, 180)
        else:
            raise SystemExit(
                f"Main text is {pages:.2f} A4 pages, above {PAGE_HI:.2f}. "
                "Shorten the draft instead of pulling the references heading upward."
            )
    raise SystemExit(
        f"Main text stayed at {pages:.2f} A4 pages, outside {PAGE_LO:.2f}–{PAGE_HI:.2f}"
    )


def main(argv: list[str] | None = None) -> int:
    del argv
    root = PROJECT_ROOT
    paths = default_bundle_paths(root)
    r9, r8 = build_canonical_bundles(
        base_r9=paths["base_r9"],
        analysis_r9=paths["analysis_r9"],
        base_r8=paths["base_r8"],
        analysis_r8=paths["analysis_r8"],
        out_r9=paths["out_r9"],
        out_r8=paths["out_r8"],
    )
    write_data_dictionary(
        paths["dictionary"],
        r9_columns=list(r9.columns),
        r8_columns=list(r8.columns),
    )
    figures = write_all_figures(
        cells_r9=paths["out_r9"],
        facilities=root / "data" / "frozen" / "facilities_clean.gpkg",
        parishes=root / "data" / "frozen" / "parishes_7.gpkg",
        lorenz_csv=root / "outputs" / "tables" / "lorenz_curves.csv",
        maup_csv=root / "outputs" / "tables" / "maup_r8_vs_r9.csv",
        figures_dir=root / "outputs" / "figures",
    )
    _frame, table_claims = write_table1(root)
    prose_claims = build_prose_claims(root)
    pages = _fit_main_text(root, table_claims, prose_claims)
    audit = render_audit([*table_claims, *prose_claims], main_pages=pages)
    audit_path = root / "report" / "go_nogo_audit.md"
    audit_path.write_text(audit, encoding="utf-8")
    manifest = write_freeze_manifest(paths["frozen_dir"], project_root=root)
    print(f"wrote {paths['out_r9']}")
    print(f"wrote {paths['out_r8']}")
    for path in figures.values():
        print(f"wrote {path}")
    print(f"wrote {audit_path}")
    print(f"wrote {manifest}")
    print(f"main text {pages:.2f} A4 pages")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
