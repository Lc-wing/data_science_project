# Table 1: H1 Panel HAR-RV Confirmatory Associations & H2 2025 Holdout Performance

## Panel A: H1 Stock-FE Extended Panel HAR-RV (Driscoll–Kraay SE, L=7, N=104,040)

| Confirmatory Term | Estimate / Stat | 95% CI / F-Stat | Raw p | Holm p | Holm Sig. |
| --- | ---: | --- | ---: | ---: | --- |
| Leverage AME (neg_ret_1d) | `-1.4458` | `[-2.0529, -0.8386]` | `<0.0001` | `<0.0001` | `Yes` |
| VIX AME (per +1 pt) | `+0.0141` | `[+0.0109, +0.0173]` | `<0.0001` | `<0.0001` | `Yes` |
| Tech x neg_ret_1d vs Health. | `-0.6479` | `[-1.3868, +0.0910]` | `0.086` | `0.171` | `No` |
| Tech x VIX vs Health. | `+0.0009` | `[-0.0011, +0.0029]` | `0.380` | `0.380` | `No` |
| Joint 5-Sector Wald (df=8) | `84.84` | `F=10.60` | `<0.0001` | `<0.0001` | `Yes` |

## Panel B: H2 Single-Shot 2025 Holdout Test (N=14,700; 245 Scorable Days)

| Model | QLIKE [95% CI] | RMSE | PR-AUC [95% CI] | Brier | ECE (Pre-Cal) | HLN-DM Holm p (QLIKE / Brier vs LGBM) |
| --- | --- | ---: | --- | ---: | ---: | ---: |
| Persistence (RV_w / Base) | `0.3144 [0.213,0.484]` | `0.1196` | `0.0721 [0.051,0.104]` | `0.0788` | `0.0023` | `<0.0001 / 0.047` |
| GJR-GARCH(1,1) Per-Stock | `0.3007 [0.241,0.386]` | `0.1362` | `0.1271 [0.098,0.169]` | `0.0796` | `0.0264` | `0.021 / 0.123` |
| Panel HAR-RV (+ Logistic) | `0.2383 [0.158,0.380]` | `0.0957` | `0.1422 [0.104,0.191]` | `0.0774` | `0.0070` | `0.134 / 0.716` |
| Global LightGBM (Cal.) | `0.2306 [0.154,0.372]` | `0.0951` | `0.1231 [0.083,0.177]` | `0.0773` | `0.0102 (0.380)` | `Ref.` |
