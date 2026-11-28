# Milestone M1: Data Ingest, Day-1 Audit & Milestone Delivery Documentation (里程碑 M1：数据采集、Day-1 质量与跨期覆盖率审计及里程碑文档体系建立)

- **Milestone**: `M1 — Data Ingest, Day-1 Audit & Milestone Delivery Documentation (数据采集、Day-1 质量与跨期覆盖率审计及里程碑文档体系建立)`
- **Status**: `COMPLETED & PUSHED` *(Current Session Stopping Point per User Request / 按用户指令完成本里程碑后停止)*
- **Parent / Baseline Commit SHA**: `124fc67eb8d578a0495733092598b60aa2b763e0` (`M0`)
- **M1 Commit**: Pushed to `origin/main` upon independent verification pass (`git log -n 1 -- plan_6_stock_volatility/docs/milestones/M1_data_ingest_and_day1_audit.md`)
- **Provenance Timestamp (UTC)**: `2026-11-22T14:04:28+00:00`
- **Verification Verdict**: `INDEPENDENT VERIFICATION PASS` (`ruff check .`: 0 errors; `pytest -v`: `35 passed`; `--skip-download` offline replay bit-identical; `9/9` raw provenance SHA-256 + `5/5` derived SHA-256 verified; `git check-ignore` exit `1`; `0` sibling diffs; `0` secrets)

---

## 1. Milestone M1 Content & Deliverables (M1 里程碑交付内容与数据契约)

### 1.1 Source Code, CLI, Test Suite & Milestone Documentation Deliverables (代码、脚本、测试与文档交付物)

| Path (Relative to `plan_6_stock_volatility/`) | Category | Role & Interface Contract |
| --- | --- | --- |
| `src/stock_volatility/ingest/__init__.py` | Source | `build_retry_session()` (HTTP/1.1 persistent `requests.Session` with exponential-backoff `Retry` and academic `SEC_USER_AGENT`) and `ProvenanceLog` (`record_file`, `verify_files`, `save`, `load`) |
| `src/stock_volatility/ingest/universe.py` | Source | `PREDECESSOR_CIK_MAP = {"BLK": ("0001364742",), "XOM": ("0000034088",)}`, `resolve_universe_ciks()`, `build_universe_table()`, `write_universe_csv()`, `load_universe_csv()` (`dtype={"cik": str}`) |
| `src/stock_volatility/ingest/prices.py` | Source | `OHLCV_COLUMNS`, `exclusive_end_date()`, `normalize_yfinance_frame()`, `download_ohlcv_universe()` (`yf.config.network.retries = 5`, `auto_adjust=False`, `actions=True`, per-ticker retry with `YfData().cache_get.cache_clear()`) |
| `src/stock_volatility/ingest/macro.py` | Source | `MACRO_COLUMNS`, `fetch_yahoo_v8_daily()`, `parse_yahoo_v8_chart_payload()`, `parse_fred_csv()`, `align_macro_to_calendar()` (backward-only `ffill` across full FRED history before joining onto 1,760-day `SPY` calendar), `download_macro_raw()`, `build_macro_daily_from_raw()` |
| `src/stock_volatility/ingest/sec_edgar.py` | Source | `TARGET_FORMS`, `EVENTS_RAW_COLUMNS`, 33-item official `SEC_8K_ITEM_DESCRIPTIONS` (`1.01`..`9.01` including `1.05` and `2.02`), `RateLimiter(max_per_sec=7.5)`, `normalize_acceptance_datetime()`, `normalize_sec_items()`, `describe_filing_items()`, `build_headline_text()`, `select_overlapping_archives()`, `parse_submissions_block()`, `deduplicate_filings()`, `download_sec_edgar_all()`, `build_events_raw_from_snapshot()` |
| `src/stock_volatility/ingest/audit.py` | Source | `build_news_coverage_table()`, `compute_day1_audit_metrics()`, `render_day1_audit_markdown()`, `write_audit_outputs()`, `run_download_and_audit_pipeline()`, `run_skip_download_pipeline()` (strictly offline, zero network calls, bit-deterministic) |
| `scripts/01_download_and_audit.py` | CLI | `uv run python scripts/01_download_and_audit.py [--skip-download] [--force-download]` |
| `tests/conftest.py` | Test | Imports `stock_volatility.config` at collection time (`MPLBACKEND=Agg`, `OMP_NUM_THREADS=1`) and provides `block_network` monkeypatch fixture |
| `tests/test_ingest.py` | Test | 9 comprehensive tests across 3 layers: (1) pure offline unit tests, (2) network-blocked `--skip-download` bit-identity replay test, (3) committed M1 artifact & milestone delivery documentation contract tests |
| `docs/milestones/README.md` | Doc | Bilingual master ledger (`M0`–`M6`), mandatory per-milestone dual-recording policy, and standard Markdown template for `M2`–`M6` |
| `docs/milestones/M0_repo_cleanup_and_scaffold.md` | Doc | Complete bilingual record of Milestone M0 (`124fc67eb8d578a0495733092598b60aa2b763e0`) deliverables and AI execution log |
| `docs/milestones/M1_data_ingest_and_day1_audit.md` | Doc | Complete bilingual record of Milestone M1 deliverables, live metrics, and AI execution log |

### 1.2 Committed Raw, Interim, Provenance & Day-1 Audit Artifacts (落盘数据、溯源清单与 Day-1 审计产物)

All raw HTTP SEC EDGAR JSON responses (`258` files) are cached strictly in gitignored `.cache/sec_edgar/`, while compact, lossless, cryptographically verified snapshots (`< 3.3 MB` max file size, `< 4.9 MB` combined) are committed under `data/raw/`, `data/interim/`, and `outputs/`:

| Artifact Path (Relative to `plan_6_stock_volatility/`) | Rows / Records | Byte Size | SHA-256 Digest | Schema / Contract Summary |
| --- | ---: | ---: | --- | --- |
| `data/raw/universe_60.csv` | `60` | `5,418 B` | `ecf785b00a2f060dd986451d56d72eea804bd8ff4e0fc0da316f7def042a742a` | `[ticker, sector, sub_group, cik, company_name, first_trade_date, last_trade_date, n_trading_days]` (`12` per sector × `5` sectors, 10-digit zero-padded `cik`, all `1,760` days `2019-01-02`..`2025-12-31`) |
| `data/raw/prices/ohlcv_daily.parquet` | `105,600` | `3,290,902 B` | `4487aee24a340ce97a0d126a48e24e951fd2c575f03671f25abc6763bd1e21d1` | `[date, ticker, open, high, low, close, adj_close, volume, dividends, stock_splits]`, sorted by `(ticker, date)`, `0` NaNs |
| `data/raw/macro/spy_daily.parquet` | `1,760` | `90,038 B` | `87f56d6fcd3d3d2e2c91ae22fbe70f5e49ea0c3a40c860343b210c3c7bd1e888` | Canonical 1,760-day NYSE trading calendar anchor (`2019-01-02`..`2025-12-31`) & `SPY` daily OHLCV |
| `data/raw/macro/vix_daily.parquet` | `1,760` | `59,615 B` | `a9a1fd7866f2097634adbd5b4c28c2f32b9687019601e4bc06aa331f7d1673ee` | Yahoo Finance `^VIX` daily implied volatility index (`2019-01-02`..`2025-12-31`) |
| `data/raw/macro/fred_t10y2y.csv` | `13,169` | `210,561 B` | `00aeefcd36ff7fe6cc989644a4fdacf45318485cffae61c4a30a009d48d61102` | Full FRED `T10Y2Y` 10Y–2Y Treasury yield spread CSV (retaining pre-2019 history for backward-only `ffill`) |
| `data/raw/macro/fred_vixcls.csv` | `9,623` | `162,037 B` | `5fdb14015c8fee857f1aae7ea54d799fe8e2eebc1b4dcab047cc7ac70e3a78a0` | Full FRED `VIXCLS` CBOE Volatility Index CSV |
| `data/raw/sec_edgar/company_tickers_60.json` | `60` | `17,137 B` | `1a2c7500d5640e67efd9c824f24648b95211330885242928772800551b4929bc` | Compact 60-ticker SEC CIK mapping snapshot including `BLK` (`0001364742`) & `XOM` (`0000034088`) predecessor CIKs |
| `data/raw/sec_edgar/pagination_manifest.json` | `258` | `134,295 B` | `f25669f626a2ebdedb26eb08c500f0f4526e979789003e08713778a6c589d227` | Cryptographic manifest of all `258` SEC HTTP requests (`1` `company_tickers.json` + `62` main CIK JSONs + `195` overlapping `filings.files` archive JSONs) |
| `data/raw/sec_edgar/edgar_filings_2019_2025.parquet` | `7,703` | `395,905 B` | `5704ae35330b14a5278dfba3f222d2f30541523f1f98b1a32ce3c65c3646d5e7` | Lossless snapshot of all `7,703` unique 2019–2025 target filings (`8-K`, `8-K/A`, `10-Q`, `10-K`) enabling offline `--skip-download` rebuild |
| `data/raw/data_provenance.json` | `9` files | `9,536 B` | *(self-manifest)* | Records source URLs, timestamps, HTTP statuses, byte sizes, SHA-256 digests, fallback rationale, predecessor CIK linkages, and anomaly counts |
| `data/interim/macro_daily.parquet` | `1,760` | `75,501 B` | `e3688356a315f1f3d5df673454c1383ff310b195fe1e08872e25dbea1226e044` | `[date, spy_close, spy_adj_close, spy_ret_1d, vix_close, vix_fred, t10y2y, t10y2y_ffilled_flag]` |
| `data/interim/events_raw.parquet` | `7,703` | `377,082 B` | `904feb20b77f861f3cd633f71399572669f3c2d9b12d0e4a106f62fb584dd9f3` | `[ticker, cik, accession_number, form, filing_date, report_date, acceptance_datetime_utc, items, item_descriptions, primary_doc_description, headline_text, news_source]` |
| `outputs/metrics/day1_data_audit.json` | `1` | `9,116 B` | `9bd4f986aba897399dc42c62ba591feb08451ee431a3fc0a7db3750ccab2c4de` | Machine-readable Day-1 audit metrics (OHLCV, macro, SEC EDGAR pagination & sector coverage rollups) |
| `outputs/tables/day1_data_audit.md` | `1` | `10,462 B` | `2ca350e6e3d60a9773ae7643cde27b8a9a4136ecf14419512c838ab22ffa7eae` | Human-readable Day-1 quality, corporate action, macro alignment, and cross-period coverage report |
| `outputs/tables/news_coverage_audit.csv` | `60` | `6,920 B` | `a38360e3836ef63dc118c40e8f6b05b582c78fd4413892fcd5be4ca51142c066` | Per-ticker `2019–2023` vs `2024–2025` filing counts, annual rates, rate ratios, event-day shares, and Form 8-K Item 2.02 earnings counts |

### 1.3 Key Quantitative Day-1 Audit Findings (Day-1 数据质量与跨期覆盖率核心指标)

| Audit Domain | Metric | Observed Live Value | Contract Requirement | Status |
| --- | --- | ---: | --- | --- |
| **Universe (`universe_60.csv`)** | Pre-2019 S&P 500 constituents across 5 GICS sectors | `60` (`12` per sector) | `60` (`0` lookahead, `0` ticker replacements) | **PASS** |
| **Equity OHLCV (`ohlcv_daily.parquet`)** | Total `(ticker, date)` rows (`2019-01-02`..`2025-12-31`) | `105,600` (`60 × 1,760`) | `105,600` (`0` missing dates, `0` NaNs) | **PASS** |
| **Equity OHLCV** | Zero-volume (`Volume == 0`) or negative/invalid price rows | `0` (`volume_min = 126,400`) | `0` violations (`high >= low > 0`) | **PASS** |
| **Equity OHLCV** | Stock split / spin-off events (`stock_splits > 0`) | `9` events across `8` tickers (`AAPL`, `PFE`, `MRK`, `NVDA` ×2, `IBM`, `AMZN`, `WMT`, `AVGO`) | Audited & inventoried | **PASS** |
| **Equity OHLCV** | Cash dividend events (`dividends > 0`) | `1,608` events across `58` tickers (non-payers: `ADBE`, `AMZN`) | Audited | **PASS** |
| **Equity OHLCV** | Extreme single-day adjusted returns (`\|ret_1d\| > 20%`) | `33` stock-days (`min = -52.01%`, `max = +35.95%`) | Audited | **PASS** |
| **Macro (`macro_daily.parquet`)** | Canonical NYSE trading days (`SPY`) | `1,760` (`2019-01-02`..`2025-12-31`) | `1,760` (`spy_ret_1d` NaN only on row 0 warmup) | **PASS** |
| **Macro** | `t10y2y_ffilled_flag` bond-holiday forward-fills | `13` days (`0.7386%`, 7 Columbus Day + 6 weekday Veterans Day) | Backward-only `ffill` from full FRED history (`0` backfills) | **PASS** |
| **Macro** | Yahoo `vix_close` vs FRED `vix_fred` Pearson correlation | `r = 0.9999999289` (`mean_abs_diff = 9.135e-05`, `max_abs_diff = 0.1100` on `2021-08-11`) | Near-exact cross-source agreement (`r > 0.999`) | **PASS** |
| **SEC EDGAR (`events_raw.parquet`)** | Total CIKs & HTTP requests (`<= 7.5 req/s`) | `62` CIKs (`60` primary + `2` predecessors), `195` overlapping archives (`351` listed), `258` HTTP requests | Recursive `filings.files` archive pagination (`<= 8.0 req/s`) | **PASS** |
| **SEC EDGAR** | Total unique target filings (`8-K`, `8-K/A`, `10-Q`, `10-K`) | `7,703` (`5,557` from `filings.recent` + `2,146` [`27.86%`] from archives) | Deduplicated by `(ticker, accession_number)` | **PASS** |
| **SEC EDGAR** | Cross-period coverage (`2019–2023` Train vs `2024–2025` Val+Test) | `60/60` active in `2019–2023` (`5,665` filings, `1,133.0/yr`); `60/60` active in `2024–2025` (`2,038` filings, `1,019.0/yr`, ratio `0.8994`) | `60/60` active in both periods (`0` dead tickers) | **PASS** |
| **SEC EDGAR** | Form breakdown & key 8-K items | `8-K = 5,928`, `8-K/A = 94`, `10-Q = 1,261`, `10-K = 420`; `Item 2.02` earnings = `1,767` (`1,266` / `501`); `Item 1.05` cybersecurity = `6` | 100% mapped via 33-item official `SEC_8K_ITEM_DESCRIPTIONS` | **PASS** |

---

## 2. What the AI Agents Did in Milestone M1 (M1 里程碑内 AI 完成的工作全记录)

### 2.1 Multi-Agent M1 Exploration & Live Upstream Probes (多智能体探测与上游数据源实测)

Before writing any M1 code, specialized read-only agents investigated the M0 scaffold, sibling Plan 2/5 conventions, live Yahoo Finance / FRED / SEC EDGAR APIs, and the user's follow-up milestone documentation requirement:
1. **`explorer_m1_r1_1`**: Audited `src/stock_volatility/config.py`, `pyproject.toml`, `.gitignore`, and sibling `plan_2_tourism_hotel/` & `plan_5_spatial_equity/` ingestion patterns; designed the 6-module `src/stock_volatility/ingest/` architecture and `.cache/sec_edgar/` vs `data/raw/sec_edgar/` split; and probed all 60 tickers against `www.sec.gov/files/company_tickers.json` and `data.sec.gov/submissions/`.
2. **`explorer_m1_r2_1`**: Executed a full live 60-ticker OHLCV + `SPY` + `^VIX` + FRED `T10Y2Y`/`VIXCLS` verification probe, diagnosing the exact root cause of intermittent `yfinance==1.8.0` `"possibly delisted"` errors and verifying `105,600` OHLCV rows, `9` stock splits, `1,608` dividends, and `13` bond-holiday `T10Y2Y` forward-fills.
3. **`spec_miner_m1_r2_2`**: Queried all `62` CIK main submission JSONs and all `195` overlapping historical archive JSONs (`CIK{cik:010d}-submissions-{NNN}.json`) at `<= 7.5 req/s`, verifying the exact 16-column archive schema, `7,703` unique target filings, `acceptanceDateTime` UTC format (`YYYY-MM-DDTHH:MM:SS.000Z`), and 33-item official Form 8-K taxonomy.
4. **`explorer_m1_r2_3`**: Audited `tests/test_smoke.py` and `.gitignore`/`.git/info/exclude` compatibility (`git check-ignore -v --no-index` exit `1`) and designed the per-milestone Markdown delivery ledger (`docs/milestones/`).

### 2.2 Seven Critical Live Data & Engineering Traps Discovered and Resolved by AI (AI 实测发现并解决的 7 个关键数据与工程陷阱)

1. **Trap 1 — Upstream Hugging Face News Gate & FNSPID Cutoff/Leakage → Plan 6 §2.1 SEC EDGAR Deterministic Fallback:**
   - *Discovery*: Live probes proved `Brianferrell787/financial-news-multisource` is gated (`HTTP 401 Unauthorized` without `HF_TOKEN`) and its ungated mirror lacks continuous 2024 intraday coverage, while `FNSPID` terminates in 2023 (`0` coverage for 2024 Val and 2025 Test) and exhibits `00:00:00 UTC` timestamp leakage.
   - *Resolution*: Activated Plan 6 §2.1's pre-registered 100% deterministic fallback (SEC EDGAR Form `8-K`, `8-K/A`, `10-Q`, `10-K` submissions with exact `acceptanceDateTime` UTC timestamps) and documented the verification transparently in `data_provenance.json` and `day1_data_audit.md`.
2. **Trap 2 — SEC EDGAR Holding-Company CIK Reorganizations for `XOM` & `BLK`:**
   - *Discovery*: Live audit of all 60 tickers revealed that **`XOM`** (`company_tickers.json` primary `CIK 0002115436`, ExxonMobil Holdings Corp, earliest filing `2026-07-01` → `0` filings in 2019–2025) and **`BLK`** (`CIK 0002012383`, BlackRock, Inc. post-GIP reorganization, earliest filing `2024-02-20` → `0` filings in 2019–2023) underwent post-2019 holding-company CIK successions.
   - *Resolution*: Traced their official SEC predecessor CIKs — **`0000034088`** (`EXXON MOBIL CORP`, `142` target filings across 2019–2025) for `XOM` and **`0001364742`** (`BlackRock Finance, Inc.`, `10` overlapping archives covering 2019–2024) for `BLK` — implemented `PREDECESSOR_CIK_MAP = {"BLK": ("0001364742",), "XOM": ("0000034088",)}`, merged and deduplicated filings by `(ticker, accession_number)`, and achieved **60/60** full 2019–2025 coverage with **zero ticker replacements**.
3. **Trap 3 — `yfinance==1.8.0` Zero-Retry Default (`yf.config.network.retries == 0`), `^VIX` Fallback & Exclusive `end` Date:**
   - *Discovery*: Inspected `yfinance/data.py:476` and discovered `YfConfig.network.retries` defaults to `0`. Any transient socket error inside `curl_cffi` during `_fetch_ticker_tz()` or `history()` is swallowed and falsely logged as `$TICKER: possibly delisted; no price data found`, omitting `Dividends` and `Stock Splits` columns. In addition, `yf.download` treats `end` as exclusive (`end="2025-12-31"` drops `2025-12-31`), defaults to `auto_adjust=True` and `actions=False`, and returns extra `Capital Gains` on `SPY`.
   - *Resolution*: Set `yf.config.network.retries = 5` before all `yf.download` calls, added per-ticker retry with `YfData().cache_get.cache_clear()` and automatic `fetch_yahoo_v8_daily("^VIX")` fallback (`America/Chicago` timezone conversion), passed `end="2026-01-01", auto_adjust=False, actions=True`, and dropped `Capital Gains` in `normalize_yfinance_frame()`.
4. **Trap 4 — SEC EDGAR High-Filer Archive Truncation (`filings.files`) & Top-Level Archive JSON Schema:**
   - *Discovery*: Large financial institutions file thousands of Form `424B2` structured-note supplements, overflowing SEC's 1,000-row `filings.recent` array (`JPM` has `72` archive files with `51` overlapping 2019–2025; `JPM`'s `filings.recent` has only `1` target filing vs `233` in archives!). Furthermore, historical archive JSONs (`CIK{cik:010d}-submissions-{NNN}.json`) place the 16 parallel arrays directly at the top level (without a `["filings"]["recent"]` wrapper) and exhibit a 1-day `filingFrom`/`filingTo` metadata drift in `96.4%` of archives.
   - *Resolution*: Implemented `select_overlapping_archives(..., window_start="2018-12-25", window_end="2026-01-05")` to paginate all `195` overlapping archives (`258` total HTTP requests at `<= 7.5 req/s`) and unified `parse_submissions_block()` to parse both `sub["filings"]["recent"]` and top-level archive dicts identically, recovering `2,146` (`27.86%`) archive target filings.
5. **Trap 5 — Akamai Bot Manager Tarpitting of Spoofed `Mozilla/5.0` User-Agents on `fred.stlouisfed.org`:**
   - *Discovery*: When querying `https://fred.stlouisfed.org/graph/fredgraph.csv?id=T10Y2Y` via `requests.Session` with a browser-spoofed `"Mozilla/5.0 ..."` User-Agent, Akamai Bot Manager (`_abck` / `bm_sz`) tarpitted the connection until `ReadTimeoutError (read timeout=30)`.
   - *Resolution*: Set `DEFAULT_HTTP_USER_AGENT = config.SEC_USER_AGENT` (`"CS482-Academic-Research Plan6-StockVolatility (cs482-research@example.edu)"`, a transparent academic User-Agent without `"Mozilla/5.0"`), after which FRED `T10Y2Y` (`210,561 B`) and `VIXCLS` (`162,037 B`) returned `HTTP 200` in `< 1 second`.
6. **Trap 6 — `.cache/sec_edgar/` vs Compact `data/raw/` Snapshots & Bit-Identical `--skip-download` Determinism:**
   - *Discovery*: Raw SEC JSON files (`258` files, `> 100 MB` uncompressed) would bloat Git if stored in `data/raw/sec_edgar/`, and calling wall-clock `utc_now_iso()` during `--skip-download` would mutate output hashes across runs.
   - *Resolution*: Cached raw SEC JSON responses in gitignored `.cache/sec_edgar/`, committed compact lossless snapshots (`company_tickers_60.json`, `pagination_manifest.json`, `edgar_filings_2019_2025.parquet` [`395,905 B`]) in `data/raw/sec_edgar/`, and engineered `run_skip_download_pipeline()` to read timestamps strictly from `data_provenance.json`, producing **bit-for-bit identical SHA-256 hashes** across repeated offline `--skip-download` runs with zero network calls.
7. **Trap 7 — FRED Empty-String Missing Values & Strict Backward-Only Forward-Fill:**
   - *Discovery*: Live FRED CSVs (`T10Y2Y`, `VIXCLS`) represent US bond-market holidays (`Columbus Day`, `Veterans Day`) as empty cells `""`.
   - *Resolution*: Coerced values via `pd.to_numeric(..., errors="coerce")` and applied backward-only `.ffill()` across the full sorted FRED history (including pre-2019 rows) *before* left-joining onto the 1,760-day `SPY` calendar (never `.bfill()`), flagging the exact `13` bond-market holidays via `t10y2y_ffilled_flag = True`.

### 2.3 TDD Implementation & Automated Test Suite (`worker_m1_r2`)

Following strict Test-Driven Development (`test-driven-development` and `verification-before-completion` skills):
1. **RED Phase**: Authored `tests/conftest.py` (`block_network` fixture) and `tests/test_ingest.py` (`9` comprehensive test functions) first and executed `pytest -v`, confirming all 9 new M1 tests failed (`9 failed, 26 passed in 8.36s`) before writing production code.
2. **GREEN Phase**: Implemented `src/stock_volatility/ingest/{__init__,universe,prices,macro,sec_edgar,audit}.py`, `scripts/01_download_and_audit.py`, and `docs/milestones/{README,M0_repo_cleanup_and_scaffold,M1_data_ingest_and_day1_audit}.md`.
3. **Live Ingest & Offline Determinism Verification**: Executed `scripts/01_download_and_audit.py`, followed by `scripts/01_download_and_audit.py --skip-download`, verifying identical SHA-256 checksums across all 5 rebuilt `data/interim/` and `outputs/` artifacts, `0` ruff errors, `35 passed` in `pytest -v` (`26` M0 smoke tests + `9` M1 unit/replay/contract/doc tests), and `git check-ignore` exit code `1`.

### 2.4 Independent Verification & Delivery Audit (独立验收审计与交付检查)

Following the `verification-before-completion` skill, an independent post-implementation audit verified every M1 claim against fresh command output prior to GitHub push (and corrected premature pre-filled multi-verifier status text left by `worker_m1_r2` when the preceding teamwork subagent hit quota before dispatching its 5-agent gate):
1. **Static Lint (`ruff check .`)**: `All checks passed!` (`0` errors).
2. **Full Test Suite (`pytest -v`)**: `35 passed` (`26` M0 smoke/contract tests + `9` M1 unit, network-blocked `--skip-download` replay, committed-artifact, and milestone-doc tests).
3. **Offline Deterministic Replay (`scripts/01_download_and_audit.py --skip-download`)**: Exited `0` with `60` universe rows, `105,600` OHLCV rows, `1,760` macro rows (`13` `t10y2y_ffilled_count`), `7,703` SEC EDGAR events, `60/60` active tickers in both `2019–2023` and `2024–2025`, and bit-identical SHA-256 digests across all `9/9` raw provenance files (`0` mismatches) and `5/5` derived interim/audit files.
4. **Git Hygiene, Size & Secret Audit**: `35` tracked M1 candidate files totaling `4.94 MB` (max file `data/raw/prices/ohlcv_daily.parquet` = `3.14 MB` < `25 MB`), `0` secret hits, `0` diffs in sibling directories (`plan_2_tourism_hotel/`, `plan_5_spatial_equity/`, `plans/`) or frozen M0 scaffold files, and `git check-ignore -v --no-index plan_6_stock_volatility/docs/milestones/*.md` exited `1` with empty output.

---

## 3. Reproducible Verification Commands & Observed Results (M1 独立验证与离线重放命令)

```bash
cd /home/antigravity/Lc-wing/data_science_project/plan_6_stock_volatility

# 1. Offline deterministic replay from committed data/raw/ snapshots (zero network calls)
MPLBACKEND=Agg OMP_NUM_THREADS=1 uv run --no-sync python scripts/01_download_and_audit.py --skip-download

# 2. Ruff lint check (must exit 0 with "All checks passed!")
MPLBACKEND=Agg OMP_NUM_THREADS=1 uv run --no-sync ruff check .

# 3. Full pytest suite (26 M0 smoke tests + 9 M1 ingest/replay/contract/doc tests = 35 passed)
MPLBACKEND=Agg OMP_NUM_THREADS=1 uv run --no-sync pytest -v

# 4. Verify milestone Markdown docs are tracked by Git and not ignored (must exit 1 with empty output)
cd /home/antigravity/Lc-wing/data_science_project && \
  git check-ignore -v --no-index plan_6_stock_volatility/docs/milestones/*.md
```

---

## 4. Git Hygiene & Delivery Checklist (Git 交付检查清单)

- [x] Dual-content record complete (M1 deliverables & quantitative metrics + chronological AI execution log & 7 traps resolved)
- [x] `docs/milestones/README.md`, `docs/milestones/M0_repo_cleanup_and_scaffold.md`, root `README.md`, and `plan_6_stock_volatility/README.md` updated and linked
- [x] `git check-ignore -v --no-index plan_6_stock_volatility/docs/milestones/*.md` exited `1` with empty output
- [x] `--skip-download` verified offline with zero network calls, zero `data/raw/` mutations, and bit-for-bit identical SHA-256 hashes on all `data/interim/` and `outputs/` files
- [x] Zero modifications to forbidden M0 files (`config.py`, `src/stock_volatility/__init__.py`, `pyproject.toml`, `uv.lock`, `.gitignore`, `plan_6_stock_volatility/.gitignore`, `tests/test_smoke.py`) or sibling folders (`plan_2_tourism_hotel/`, `plan_5_spatial_equity/`, `plans/`)
- [x] All committed files `< 25 MB` (max file `data/raw/prices/ohlcv_daily.parquet` = `3.29 MB`), zero `.agents/` or secret leakage
