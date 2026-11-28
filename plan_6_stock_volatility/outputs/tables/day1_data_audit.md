# Milestone M1 — Day-1 Data Quality, Availability & Cross-Period Coverage Audit

- **Project**: `plan_6_stock_volatility` (`Milestone M1`)
- **Provenance Timestamp (UTC)**: `2026-11-22T14:04:28+00:00`
- **Canonical NYSE Trading Calendar**: `2019-01-02` to `2025-12-31` (`1760` trading days, anchored on `SPY`)
- **60-Stock Cross-Sector Universe**: `60` pre-2019 S&P 500 bellwethers (`5` GICS sectors × `12` stocks; `0` ticker replacements required)

---

## 1. Corporate Event Backbone Selection & Pre-Registered Fallback Verification

1. **Primary Candidate Probe (`Brianferrell787/financial-news-multisource`)**: Hugging Face dataset is gated (HTTP 401 Unauthorized without HF_TOKEN), and its ungated mirror lacks continuous 2024 ticker-tagged intraday coverage.
2. **Secondary Candidate Probe (`FNSPID`)**: Dataset ends in 2023 (0 coverage for 2024 Val and 2025 Test), requires 5.73-23.23 GB bulk download, and exhibits severe 00:00:00 UTC timestamp leakage.
3. **Activated Authoritative Backbone (`Plan 6 §2.1 SEC EDGAR 8-K/8-K/A/10-Q/10-K`)**: 100% public, zero authentication gate, exact second-precision UTC acceptanceDateTime for strict 16:00 America/New_York cutoff, and 60/60 ticker continuity across both 2019-2023 and 2024-2025.

### Predecessor CIK Reorganizations Resolved (`BLK` & `XOM`)

| Ticker | Primary CIK | Predecessor CIK | Resolution & 2019–2025 Continuity Note |
| --- | --- | --- | --- |
| `BLK` | `0002012383` (`BlackRock, Inc.`) | `0001364742` (`BlackRock Finance, Inc.`) | BlackRock completed its Global Infrastructure Partners (GIP) holding-company reorganization in 2024; 2019-2023 and early-2024 filings live under predecessor CIK 0001364742 while post-2024-02-20 filings live under CIK 0002012383. |
| `XOM` | `0002115436` (`ExxonMobil Holdings Corp`) | `0000034088` (`EXXON MOBIL CORP`) | ExxonMobil registered holding-company CIK 0002115436 in July 2026; 100% of 2019-2025 filings live under predecessor CIK 0000034088. |

---

## 2. Equity OHLCV Continuity & Splits Audit (`data/raw/prices/ohlcv_daily.parquet`)

| Audit Metric | Observed Value | Requirement | Status |
| --- | ---: | --- | --- |
| Total `(ticker, date)` rows | `105600` | `60 × 1760 = 105600` | PASS |
| Tickers matching canonical `SPY` calendar | `60 / 60` | `60 / 60` | PASS |
| NaN cells across OHLCV, adj_close, dividends, splits | `0` | `0` | PASS |
| Zero-volume (`Volume == 0`) or halt rows | `0` | Audited (`0`) | PASS |
| Non-positive price or `high < low` violations | `0` | `0` | PASS |
| Stock split / spin-off events (`stock_splits > 0`) | `9` across `8` tickers | Audited | PASS |
| Cash dividend events (`dividends > 0`) | `1608` across `58` tickers | Audited | PASS |
| Extreme daily adjusted returns (`|ret_1d| > 20%`) | `33` | Audited | PASS |

### Stock Split & Spin-Off Events Inventory (`2019-01-02`..`2025-12-31`)

| Date | Ticker | Split Ratio (`stock_splits`) | Raw `close` | `adj_close` |
| --- | --- | ---: | ---: | ---: |
| `2020-08-31` | `AAPL` | `4.0` | `129.0400` | `124.9626` |
| `2020-11-17` | `PFE` | `1.054` | `36.0400` | `26.4671` |
| `2021-06-03` | `MRK` | `1.048` | `73.9100` | `62.3289` |
| `2021-07-20` | `NVDA` | `4.0` | `18.6120` | `18.5271` |
| `2021-11-04` | `IBM` | `1.046` | `120.8500` | `99.1209` |
| `2022-06-06` | `AMZN` | `20.0` | `124.7900` | `124.7900` |
| `2024-02-26` | `WMT` | `3.0` | `59.6000` | `57.9666` |
| `2024-06-10` | `NVDA` | `10.0` | `121.7900` | `121.4437` |
| `2024-07-15` | `AVGO` | `10.0` | `171.4200` | `167.9498` |

---

## 3. Macro & Implied Volatility Alignment Audit (`data/interim/macro_daily.parquet`)

| Audit Metric | Observed Value | Notes |
| --- | ---: | --- |
| Canonical NYSE trading days (`SPY`) | `1760` | `2019-01-02`..`2025-12-31` |
| `spy_ret_1d` warmup NaN count | `1` | Row 0 (`2019-01-02`) only; inside M2 22-day `RV_m` warmup |
| `t10y2y_ffilled_flag` bond-holiday forward-fills | `13` (`0.74%`) | Backward-only forward-fill from full FRED history (never backfilled) |
| Yahoo `vix_close` vs FRED `vix_fred` Pearson `r` | `0.99999993` | Mean abs diff `0.000091`, max abs diff `0.1100` on `2021-08-11` |

- **Forward-filled `T10Y2Y` Bond-Holiday Dates (`13` days)**: `2019-10-14`, `2019-11-11`, `2020-10-12`, `2020-11-11`, `2021-10-11`, `2021-11-11`, `2022-10-10`, `2022-11-11`, `2023-10-09`, `2024-10-14`, `2024-11-11`, `2025-10-13`, `2025-11-11`

---

## 4. SEC EDGAR Pagination & Cross-Period Coverage Audit (`2019–2023` vs `2024–2025`)

- **Total CIKs Queried**: `62` (`60` primary + `2` predecessor CIKs)
- **Historical Archives Paginated (`filings.files`)**: `195` overlapping 2019–2025 archives out of `351` total archives (`258` total HTTP requests at `<= 7.5 req/s`)
- **Total Unique Target Filings (`8-K`, `8-K/A`, `10-Q`, `10-K`)**: `7703` (`5557` from `filings.recent` + `2146` from historical archives)
- **Active Tickers (`2019–2023` Train Window)**: `60 / 60` (`5665` filings, `1133.00` filings/year)
- **Active Tickers (`2024–2025` Val+Test Window)**: `60 / 60` (`2038` filings, `1019.00` filings/year, annual rate ratio `0.8994`)
- **Form Breakdown**: `8-K` = `5928`, `8-K/A` = `94`, `10-Q` = `1261`, `10-K` = `420` (`Item 2.02` earnings = `1767`, `Item 1.05` cybersecurity = `6`)

### Sector-Level Cross-Period Coverage Rollup

| Sector | Tickers | Active 19–23 | Active 24–25 | Filings 19–23 | Filings 24–25 | Total | Filings/Yr 19–23 | Filings/Yr 24–25 | Rate Ratio | Item 2.02 (19–23 / 24–25) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Technology | 12 | 12 | 12 | 972 | 357 | 1329 | 194.40 | 178.50 | 0.9182 | 247 / 96 |
| Financials | 12 | 12 | 12 | 1535 | 519 | 2054 | 307.00 | 259.50 | 0.8453 | 263 / 100 |
| Healthcare | 12 | 12 | 12 | 1098 | 363 | 1461 | 219.60 | 181.50 | 0.8265 | 259 / 100 |
| Consumer | 12 | 12 | 12 | 1008 | 383 | 1391 | 201.60 | 191.50 | 0.9499 | 244 / 97 |
| Energy | 12 | 12 | 12 | 1052 | 416 | 1468 | 210.40 | 208.00 | 0.9886 | 253 / 108 |

### Per-Ticker Coverage Summary (`outputs/tables/news_coverage_audit.csv`)

| Ticker | Sector | CIK | Filings 19–23 | Filings 24–25 | Total | Rate Ratio | Item 2.02 (19–23 / 24–25) |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `AAPL` | Technology | `0000320193` | 62 | 27 | 89 | 1.0887 | 21 / 8 |
| `MSFT` | Technology | `0000789019` | 73 | 25 | 98 | 0.8562 | 20 / 8 |
| `NVDA` | Technology | `0001045810` | 72 | 25 | 97 | 0.8681 | 22 / 8 |
| `AVGO` | Technology | `0001730168` | 98 | 29 | 127 | 0.7398 | 20 / 8 |
| `ORCL` | Technology | `0001341439` | 64 | 27 | 91 | 1.0547 | 21 / 8 |
| `CSCO` | Technology | `0000858877` | 86 | 32 | 118 | 0.9302 | 20 / 8 |
| `ADBE` | Technology | `0000796343` | 70 | 24 | 94 | 0.8571 | 20 / 8 |
| `CRM` | Technology | `0001108524` | 87 | 27 | 114 | 0.7759 | 20 / 8 |
| `INTC` | Technology | `0000050863` | 98 | 48 | 146 | 1.2245 | 21 / 8 |
| `TXN` | Technology | `0000097476` | 69 | 25 | 94 | 0.9058 | 20 / 8 |
| `QCOM` | Technology | `0000804328` | 67 | 30 | 97 | 1.1194 | 20 / 8 |
| `IBM` | Technology | `0000051143` | 126 | 38 | 164 | 0.7540 | 22 / 8 |
| `JPM` | Financials | `0000019617` | 172 | 62 | 234 | 0.9012 | 20 / 8 |
| `BAC` | Financials | `0000070858` | 84 | 36 | 120 | 1.0714 | 20 / 9 |
| `WFC` | Financials | `0000072971` | 336 | 47 | 383 | 0.3497 | 20 / 8 |
| `C` | Financials | `0000831001` | 139 | 58 | 197 | 1.0432 | 20 / 8 |
| `GS` | Financials | `0000886982` | 121 | 44 | 165 | 0.9091 | 22 / 8 |
| `MS` | Financials | `0000895421` | 77 | 30 | 107 | 0.9740 | 19 / 8 |
| `BLK` | Financials | `0002012383` | 78 | 40 | 118 | 1.2821 | 20 / 8 |
| `SCHW` | Financials | `0000316709` | 91 | 27 | 118 | 0.7418 | 20 / 8 |
| `AXP` | Financials | `0000004962` | 144 | 59 | 203 | 1.0243 | 20 / 8 |
| `USB` | Financials | `0000036104` | 107 | 41 | 148 | 0.9579 | 20 / 8 |
| `PNC` | Financials | `0000713676` | 118 | 46 | 164 | 0.9746 | 39 / 11 |
| `CB` | Financials | `0000896159` | 68 | 29 | 97 | 1.0662 | 23 / 8 |
| `JNJ` | Healthcare | `0000200406` | 86 | 31 | 117 | 0.9012 | 22 / 9 |
| `UNH` | Healthcare | `0000731766` | 96 | 45 | 141 | 1.1719 | 20 / 8 |
| `PFE` | Healthcare | `0000078003` | 92 | 29 | 121 | 0.7880 | 23 / 8 |
| `MRK` | Healthcare | `0000310158` | 63 | 22 | 85 | 0.8730 | 21 / 8 |
| `ABT` | Healthcare | `0000001800` | 73 | 25 | 98 | 0.8562 | 20 / 8 |
| `TMO` | Healthcare | `0000097745` | 97 | 27 | 124 | 0.6959 | 23 / 8 |
| `LLY` | Healthcare | `0000059478` | 73 | 29 | 102 | 0.9932 | 25 / 9 |
| `AMGN` | Healthcare | `0000318154` | 90 | 18 | 108 | 0.5000 | 20 / 8 |
| `BMY` | Healthcare | `0000014272` | 124 | 31 | 155 | 0.6250 | 21 / 8 |
| `GILD` | Healthcare | `0000882095` | 73 | 30 | 103 | 1.0274 | 22 / 8 |
| `CVS` | Healthcare | `0000064803` | 119 | 36 | 155 | 0.7563 | 22 / 10 |
| `CI` | Healthcare | `0001739940` | 112 | 40 | 152 | 0.8929 | 20 / 8 |
| `WMT` | Consumer | `0000104169` | 90 | 36 | 126 | 1.0000 | 20 / 8 |
| `PG` | Consumer | `0000080424` | 122 | 46 | 168 | 0.9426 | 20 / 8 |
| `KO` | Consumer | `0000021344` | 92 | 33 | 125 | 0.8967 | 20 / 8 |
| `PEP` | Consumer | `0000077476` | 86 | 32 | 118 | 0.9302 | 20 / 8 |
| `COST` | Consumer | `0000909832` | 80 | 32 | 112 | 1.0000 | 20 / 8 |
| `CL` | Consumer | `0000021665` | 71 | 23 | 94 | 0.8099 | 21 / 8 |
| `AMZN` | Consumer | `0001018724` | 73 | 23 | 96 | 0.7877 | 20 / 8 |
| `HD` | Consumer | `0000354950` | 82 | 32 | 114 | 0.9756 | 20 / 8 |
| `MCD` | Consumer | `0000063908` | 87 | 28 | 115 | 0.8046 | 22 / 8 |
| `NKE` | Consumer | `0000320187` | 66 | 29 | 95 | 1.0985 | 20 / 8 |
| `SBUX` | Consumer | `0000829224` | 82 | 39 | 121 | 1.1890 | 21 / 9 |
| `TGT` | Consumer | `0000027419` | 77 | 30 | 107 | 0.9740 | 20 / 8 |
| `XOM` | Energy | `0002115436` | 102 | 40 | 142 | 0.9804 | 21 / 8 |
| `CVX` | Energy | `0000093410` | 89 | 36 | 125 | 1.0112 | 21 / 9 |
| `COP` | Energy | `0001163165` | 69 | 30 | 99 | 1.0870 | 20 / 8 |
| `EOG` | Energy | `0000821189` | 79 | 38 | 117 | 1.2025 | 20 / 11 |
| `SLB` | Energy | `0000087347` | 94 | 30 | 124 | 0.7979 | 21 / 8 |
| `MPC` | Energy | `0001510295` | 90 | 28 | 118 | 0.7778 | 24 / 8 |
| `PSX` | Energy | `0001534701` | 82 | 39 | 121 | 1.1890 | 21 / 7 |
| `VLO` | Energy | `0001035002` | 81 | 27 | 108 | 0.8333 | 22 / 9 |
| `OXY` | Energy | `0000797468` | 102 | 44 | 146 | 1.0784 | 23 / 16 |
| `HAL` | Energy | `0000045012` | 75 | 28 | 103 | 0.9333 | 20 / 8 |
| `KMI` | Energy | `0001506307` | 116 | 43 | 159 | 0.9267 | 20 / 8 |
| `WMB` | Energy | `0000107263` | 73 | 33 | 106 | 1.1301 | 20 / 8 |
