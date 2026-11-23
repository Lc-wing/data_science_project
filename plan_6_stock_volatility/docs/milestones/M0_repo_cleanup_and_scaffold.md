# Milestone M0: Repository Cleanup, Toolchain Provisioning & Plan 6 Scaffold (里程碑 M0：仓库整理、工具链配置与 Plan 6 脚手架)

- **Milestone**: `M0 — Repository Cleanup, Toolchain Provisioning & Plan 6 Scaffold (仓库整理、工具链配置与 Plan 6 脚手架)`
- **Status**: `COMPLETED & PUSHED TO GITHUB`
- **GitHub Commit SHA**: `124fc67eb8d578a0495733092598b60aa2b763e0` (Parent: `a85555445595c2e4fcceae43212c0d526b07225c`)
- **Commit Message**: `feat(plan6): M0 reorganize repo, provision toolchain, and scaffold plan_6_stock_volatility`
- **GitHub Commit URL**: `https://github.com/Lc-wing/data_science_project/commit/124fc67eb8d578a0495733092598b60aa2b763e0`
- **Verification Verdict**: `UNANIMOUS 5/5 PASS` (Forensic Auditor `CLEAN`, Reviewer 1 `APPROVE`, Reviewer 2 `APPROVE`, Challenger 1 `PASS`, Challenger 2 `PASS`; `26 passed` in `pytest -v`)

---

## 1. Milestone M0 Content & Deliverables (M0 里程碑交付内容)

### 1.1 Repository Reorganization & Sibling Isolation (Strategy 1 — Sibling Layout / 仓库整理与兄弟项目隔离)

- **Complete Sibling Isolation Preserved**:
  - `plan_2_tourism_hotel/` (`7/7` frozen SHA-256 hashes verified intact), `plan_5_spatial_equity/` (`14/14` frozen SHA-256 hashes and `plan_5_spatial_equity/.gitignore` verified intact), and `plans/` (`00_review_report.md` and `plan_1`..`plan_9`) remained **100% untouched** (`git diff` empty).
- **Root [`README.md`](../../../README.md) (`1,924 B`)**:
  - Updated implemented plan count from two to three (`plan_5_spatial_equity/`, `plan_2_tourism_hotel/`, `plan_6_stock_volatility/`).
  - Updated Plan 2 status to `"Phase A frozen, backtest complete, CS482 Phase A draft built"` (`macao_tourism`).
  - Added `plan_6_stock_volatility/` row (`stock_volatility`), updated `plans/` count from five to nine (`plan_1`–`plan_9`), and documented `pdflatex`, `bibtex`, and Poppler (`pdfinfo`, `pdftotext`) prerequisites.
- **Root [`.gitignore`](../../../.gitignore) (`475 B`)**:
  - Preserved lines 1–29 verbatim (keeping `.agents/` ignored at line 19).
  - Appended `.cache/`, `hf_cache/`, `checkpoints/`, `*.safetensors`, `*.pt`, `*.pth`, `*.ckpt`, and `*.gguf` (deliberately omitting global `*.zip` so Plan 5's tracked `macau-latest-free.gpkg.zip` stays unignored).
- **Local `.git/info/exclude` Configuration**:
  - Configured root-anchored `/PROJECT.md`, `/TEST_INFRA.md`, and `/TEST_READY.md` in local `.git/info/exclude` so internal multi-agent coordination documents never pollute GitHub while leaving tracked `plan_5_spatial_equity/{PROJECT,TEST_INFRA,TEST_READY}.md` completely unaffected.

### 1.2 System LaTeX & Poppler Toolchain Provisioning (TeX Live 与 Poppler 工具链安装)

Installed Debian TeX Live and Poppler packages non-interactively via `--no-install-recommends` (`texlive-latex-base`, `texlive-latex-recommended`, `texlive-latex-extra`, `texlive-fonts-recommended`, `texlive-plain-generic`, `lmodern`, `poppler-utils`, `poppler-data`):

| Binary Path | Version Verified | Purpose in Plan 6 |
| --- | --- | --- |
| `/usr/bin/pdflatex` | `pdfTeX 3.141592653-2.6-1.40.25 (TeX Live 2023/Debian)` | Compile CS482 Phase A Draft PDF (`A4`, `11pt`, `1.5` spacing) |
| `/usr/bin/bibtex` | `BibTeX 0.99d (TeX Live 2023/Debian)` | Compile academic bibliography (`report/references.bib`) |
| `/usr/bin/pdfinfo` | `pdfinfo version 24.02.0` | Verify A4 page dimensions (`595.276 x 841.89 pts`) and page count |
| `/usr/bin/pdftotext` | `pdftotext version 24.02.0` | Automated claims, page-budget, and forbidden causal wording audit |

### 1.3 Isolated Plan 6 `uv` Project Scaffold (`plan_6_stock_volatility/`)

All 9 committed files in Milestone M0 (`622,703 B` total, max file `590,321 B` < 25 MB limit):

| # | Repository Path | Byte Size (M0 Commit `124fc67`) | Contract & Role |
| ---: | --- | ---: | --- |
| 1 | `.gitignore` | `475 B` | Root cache (`.cache/`, `hf_cache/`, `checkpoints/`) and ML/LLM weight ignores |
| 2 | `README.md` | `1,924 B` | Root project index across Plans 2, 5, and 6 |
| 3 | `plan_6_stock_volatility/.gitignore` | `677 B` | Plan-6-specific `.cache/`, archive (`data/raw/*.{zip,tar.gz,tgz}`), weight, and LaTeX auxiliary ignores |
| 4 | `plan_6_stock_volatility/README.md` | `7,135 B` | Bilingual CS482 academic research & non-investment-advice disclaimer (`不构成任何投资或交易建议`), milestone table, setup commands, anti-leakage summary |
| 5 | `plan_6_stock_volatility/pyproject.toml` | `1,143 B` | `hatchling>=1.25`, `stock-volatility 0.1.0`, `requires-python = ">=3.11"`, 14 runtime + 2 dev deps, `constraint-dependencies = ["numba>=0.60"]`, `ruff` (`line-length = 100`), `pytest` |
| 6 | `plan_6_stock_volatility/uv.lock` | `590,321 B` | Universal lockfile verified with `--frozen --no-build` on both Python 3.11 and Python 3.12 (0 sdist-only packages) |
| 7 | `plan_6_stock_volatility/src/stock_volatility/__init__.py` | `175 B` | Exports `__version__ = "0.1.0"` |
| 8 | `plan_6_stock_volatility/src/stock_volatility/config.py` | `6,712 B` | Canonical paths, `RANDOM_SEED = 42`, split dates (`2019-01-02`..`2025-12-31`), `HORIZON_DAYS = 5`, `RV_EPS = 1e-6`, `TAIL_RETURN_THRESHOLD = -0.05`, `MAX_EVENTS_PER_DAY = 5`, `CUTOFF_TZ = "America/New_York"`, `CUTOFF_HOUR = 16`, deterministic 60-ticker `UNIVERSE_60`, SHA-256 & directory helpers, `MPLBACKEND=Agg`, `OMP_NUM_THREADS=1` |
| 9 | `plan_6_stock_volatility/tests/test_smoke.py` | `14,141 B` | 26 smoke & contract tests covering layout, `UNIVERSE_60`, split chronology/DST, all 14 core libraries, LightGBM + TreeSHAP additivity (`< 1e-6`) + `arch` GJR-GARCH(1,1), and A4 `pdflatex` compilation |

---

## 2. What the AI Agents Did in Milestone M0 (M0 里程碑内 AI 完成的工作全记录)

### 2.1 Step 0 Multi-Agent Repository & Specification Survey (前期三路并发调研)

Before creating the M0 scaffold, three read-only agents surveyed the project specifications, repository topology, and live upstream data feeds:
1. **`spec_miner_survey_1`**: Mined `plans/plan_6_stock_volatility_news_llm.md` (v2.2, commit `a855554`), cross-plan review guardrails `C1–C15` in `plans/00_review_report.md`, and the CS482 course requirements rubric.
2. **`explorer_survey_2`**: Audited git branch/PR topology, `.gitignore` rules, and sibling Plan 2 / Plan 5 isolation constraints, establishing **Strategy 1 (PR #8 Sibling Layout)** so Plans 2 and 5 remain untouched.
3. **`explorer_survey_3`**: Executed live environment and upstream data-source probes, discovering that `Brianferrell787/financial-news-multisource` on Hugging Face is gated (`HTTP 401 Unauthorized` without `HF_TOKEN`) and lacks continuous 2024 intraday coverage while `FNSPID` terminates in 2023 with `00:00:00 UTC` timestamp leakage — confirming that Plan 6 §2.1's pre-registered **SEC EDGAR Form `8-K`, `8-K/A`, `10-Q`, `10-K` fallback** is required.

### 2.2 M0 Three-Explorer Architecture & Toolchain Investigation (M0 三路专项探测与陷阱排查)

1. **`explorer_m0_r1_1`**: Audited root repo hygiene, `.gitignore`, `.git/info/exclude`, and verified all frozen SHA-256 hashes in Plan 2 (`7/7`) and Plan 5 (`14/14`).
2. **`explorer_m0_r1_2`**: Designed the `plan_6_stock_volatility/` package scaffold, `config.py` interface contract, and pre-2019 GICS sector taxonomy (`5` sectors × `12` pre-2019 liquid S&P 500 bellwethers, documenting `TGT` under pre-2019 Consumer Discretionary prior to its March 2023 GICS reclassification).
3. **`explorer_m0_r1_3`**: Uncovered and resolved **two critical system and dependency traps** before any production code was written:
   - **Trap 1 — TeX Live Recommends Bloat (~270 MiB OpenJDK/JRE):** Default `apt-get install texlive-*` pulls ~270 MiB of unnecessary Java/GUI recommends. Resolved by installing with `--no-install-recommends` while explicitly adding `lmodern`, `texlive-plain-generic`, and `poppler-data`.
   - **Trap 2 — Universal `uv.lock` Python 3.11 vs 3.12 Source-Build Trap (`numba>=0.60`):** Without version constraints, `uv`'s universal resolver included a legacy `numba==0.53.1` / `llvmlite==0.36.0` branch for `<3.12` that lacks prebuilt Python 3.11 wheels and fails under `--no-build`. Resolved by adding `[tool.uv] constraint-dependencies = ["numba>=0.60"]` in `pyproject.toml`, ensuring 100% prebuilt binary wheels on both Python 3.11 (`shap==0.50.0`, `numba==0.68.1`) and Python 3.12 (`shap==0.54.1`).

### 2.3 TDD Implementation by `worker_m0_r1` (测试驱动开发实现)

Following strict Test-Driven Development (`test-driven-development` skill):
- **RED Phase**: Created `tests/test_smoke.py` (`26` tests) first and verified failure (`6 failed, 20 passed`) before implementing `config.py`.
- **GREEN Phase**: Implemented `src/stock_volatility/__init__.py`, `src/stock_volatility/config.py`, `pyproject.toml`, `uv.lock`, `.gitignore`, and `README.md`; verified `26 passed in 5.25s` and `ruff check .` (`All checks passed!`).

### 2.4 Five-Way Independent Verification Gate (五路独立验证门禁)

All 5 independent verification agents audited M0 in parallel and returned unanimous pass verdicts:

| Verifier Role | Agent ID | Verdict | Independent Verification Scope & Findings |
| --- | --- | --- | --- |
| **Forensic Auditor** *(evaluated first)* | `auditor_m0_r1` | **`CLEAN`** | Verified genuine dpkg-owned ELF executables (`/usr/bin/{pdflatex,bibtex,pdfinfo,pdftotext}`, `dpkg -V` clean), authentic `uv.lock` (`uv lock --check` PASS), zero test skips/mocks/stubs, zero sibling diffs, `<25 MB` files, zero secrets. |
| **Reviewer 1 (Primary)** | `reviewer_m0_r1_1` | **`APPROVE`** | Full-suite `uv sync --all-groups --frozen --no-build && ruff check . && pytest -v` (`26 passed`), cross-plan `ruff check --no-cache` PASS, positive/negative `git check-ignore` checks, line-by-line review of all 9 files (`0` defects). |
| **Reviewer 2 (Secondary)** | `reviewer_m0_r1_2` | **`APPROVE`** | Character-by-character verification of `config.py` constants and `UNIVERSE_60` against `PROJECT.md`, zero sdist-only packages in `uv.lock`, valid relative Markdown links, bilingual non-investment-advice & non-causal wording audit (`0` defects). |
| **Challenger 1 (Toolchain & Py312)** | `challenger_m0_r1_1` | **`PASS`** | Cross-Python 3.12 (`shap==0.54.1`) and Python 3.11 (`shap==0.50.0`, `numba==0.68.1`) `--frozen --no-build` wheel sync + `26/26` pytest PASS; 3-page A4 CS482 `pdflatex -> bibtex -> pdflatex -> pdflatex` + 300-DPI matplotlib stress probe PASS. |
| **Challenger 2 (Hygiene & Contract)** | `challenger_m0_r1_2` | **`PASS`** | Throwaway git sandbox ignore/exclude check (`17/17` ignored, `12/12` unignored, `0` leakage); programmatic 60/60 `UNIVERSE_60` equality check; DST transition checks; Plan 2 (`7/7`) & Plan 5 (`14/14`) frozen SHA-256 checks PASS. |

### 2.5 Post-Gate GitHub Push (`worker_m0_git_push`)

After the 5-verifier gate passed unanimously, `worker_m0_git_push` staged the exact 9 M0 files, passed all pre-flight guardrails (no `.agents/`, no local coordination docs, `< 25 MB` blobs, zero secrets), committed `124fc67eb8d578a0495733092598b60aa2b763e0`, pushed to `origin/main`, and confirmed `git ls-remote origin refs/heads/main == HEAD`.

---

## 3. Reproducible Verification Commands (M0 独立验证命令)

```bash
cd /home/antigravity/Lc-wing/data_science_project/plan_6_stock_volatility
uv sync --all-groups --frozen --no-build
MPLBACKEND=Agg OMP_NUM_THREADS=1 uv run --no-sync ruff check .
MPLBACKEND=Agg OMP_NUM_THREADS=1 uv run --no-sync pytest -v tests/test_smoke.py
pdflatex --version && bibtex --version && pdfinfo -v && pdftotext -v
```
