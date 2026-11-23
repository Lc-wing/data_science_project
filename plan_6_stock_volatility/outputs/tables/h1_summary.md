# Milestone M3 — H1 Stock-FE Extended Panel HAR-RV Regression

- **Provenance Timestamp (UTC)**: `2026-11-22T14:04:28+00:00`
- **Sample**: `104040` stock-days (`60` stocks × `1734` consecutive NYSE trading days, `2019-02-01`..`2025-12-23`)
- **Estimator**: Within-entity Stock-FE OLS with Bartlett Driscoll–Kraay SE (bandwidth `L = 7`, `df = 1733`); Within $R^2 = 0.4736$
- **Reference Sector**: `Healthcare` (defensive benchmark)
- **Sign Convention**: `neg_ret_1d = min(ret_1d, 0) <= 0`; a **negative** coefficient on `neg_ret_1d` indicates that larger negative returns are associated with **higher** forward 5-day log realized volatility (leverage association).
- **Scope Boundary**: Associational econometric evidence only; no causal claims.

---

## 1. Pre-Registered H1 Confirmatory Family (Holm–Bonferroni Adjusted)

| Test ID | Estimate / Stat | DK SE | `t` / `F` | Raw `p` | Holm `p` | 95% CI | Holm Sig (`α=0.05`) |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `neg_ret_1d` | `-1.44577` | `0.30956` | `-4.67` | `3.237e-06` | `9.711e-06` | [`-2.0529`, `-0.8386`] | `YES` |
| `vix_close` | `+0.01411` | `0.00165` | `+8.55` | `2.595e-17` | `1.298e-16` | [`+0.0109`, `+0.0173`] | `YES` |
| `tech_x_neg_ret_1d_vs_healthcare` | `-0.64789` | `0.37673` | `-1.72` | `8.566e-02` | `1.713e-01` | [`-1.3868`, `+0.0910`] | `NO` |
| `tech_x_vix_close_vs_healthcare` | `+0.00090` | `0.00103` | `+0.88` | `3.797e-01` | `3.797e-01` | [`-0.0011`, `+0.0029`] | `NO` |
| `joint_5sector_interaction_wald` | `+84.83506` | — | `+10.60` | `1.234e-14` | `4.935e-14` | — | `YES` |

---

## 2. Main Stock-FE Extended Panel HAR-RV Coefficients (Driscoll–Kraay SE)

| Term | Coef | DK SE | `t` | `p` | 95% CI |
| --- | ---: | ---: | ---: | ---: | --- |
| `log_rv_d` | `+0.09293` | `0.00860` | `+10.80` | `2.222e-26` | [`+0.07606`, `+0.10980`] |
| `log_rv_w` | `+0.16341` | `0.02723` | `+6.00` | `2.374e-09` | [`+0.11001`, `+0.21681`] |
| `log_rv_m` | `+0.21710` | `0.04136` | `+5.25` | `1.720e-07` | [`+0.13597`, `+0.29822`] |
| `neg_ret_1d` | `-1.03006` | `0.39801` | `-2.59` | `9.733e-03` | [`-1.81068`, `-0.24944`] |
| `vix_close` | `+0.01043` | `0.00177` | `+5.90` | `4.429e-09` | [`+0.00696`, `+0.01390`] |
| `technology_x_neg_ret_1d` | `-0.64789` | `0.37673` | `-1.72` | `8.566e-02` | [`-1.38679`, `+0.09101`] |
| `technology_x_vix_close` | `+0.00090` | `0.00103` | `+0.88` | `3.797e-01` | [`-0.00111`, `+0.00292`] |
| `financials_x_neg_ret_1d` | `-1.27307` | `0.54173` | `-2.35` | `1.888e-02` | [`-2.33558`, `-0.21057`] |
| `financials_x_vix_close` | `+0.00504` | `0.00114` | `+4.42` | `1.043e-05` | [`+0.00280`, `+0.00727`] |
| `consumer_x_neg_ret_1d` | `-0.41274` | `0.35636` | `-1.16` | `2.469e-01` | [`-1.11169`, `+0.28621`] |
| `consumer_x_vix_close` | `+0.00367` | `0.00096` | `+3.83` | `1.312e-04` | [`+0.00179`, `+0.00555`] |
| `energy_x_neg_ret_1d` | `+0.25514` | `0.41007` | `+0.62` | `5.339e-01` | [`-0.54913`, `+1.05942`] |
| `energy_x_vix_close` | `+0.00874` | `0.00145` | `+6.04` | `1.930e-09` | [`+0.00590`, `+0.01159`] |

---

## 3. Pre-Specified Sensitivity Checks (Zero Pretest Switching)

| Specification | `N` | Within $R^2$ | Term | Estimate | SE | `p` | Holm `p` |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| `train_only_2019_2023` | `73860` | `0.5293` | `ame_neg_ret_1d` | `-1.22512` | `0.34271` | `3.641e-04` | `1.092e-03` |
| `train_only_2019_2023` | `73860` | `0.5293` | `ame_vix_close` | `+0.01674` | `0.00175` | `5.180e-21` | `2.590e-20` |
| `train_only_2019_2023` | `73860` | `0.5293` | `tech_x_neg_ret_1d_vs_healthcare` | `-0.51011` | `0.42878` | `2.344e-01` | `3.849e-01` |
| `train_only_2019_2023` | `73860` | `0.5293` | `tech_x_vix_close_vs_healthcare` | `+0.00166` | `0.00127` | `1.925e-01` | `3.849e-01` |
| `train_only_2019_2023` | `73860` | `0.5293` | `joint_5sector_interaction_wald` | `+66.35678` | — | `5.321e-11` | `2.128e-10` |
| `close_to_close_rv_5d_cc_fwd` | `104040` | `0.2808` | `ame_neg_ret_1d` | `-1.54882` | `0.42459` | `2.723e-04` | `8.170e-04` |
| `close_to_close_rv_5d_cc_fwd` | `104040` | `0.2808` | `ame_vix_close` | `+0.01693` | `0.00218` | `1.390e-14` | `6.952e-14` |
| `close_to_close_rv_5d_cc_fwd` | `104040` | `0.2808` | `tech_x_neg_ret_1d_vs_healthcare` | `-0.03821` | `0.61503` | `9.505e-01` | `9.505e-01` |
| `close_to_close_rv_5d_cc_fwd` | `104040` | `0.2808` | `tech_x_vix_close_vs_healthcare` | `+0.00226` | `0.00166` | `1.744e-01` | `3.489e-01` |
| `close_to_close_rv_5d_cc_fwd` | `104040` | `0.2808` | `joint_5sector_interaction_wald` | `+33.08660` | — | `6.605e-05` | `2.642e-04` |
| `newey_west_entity_clustered_se` | `104040` | `0.4736` | `ame_neg_ret_1d` | `-1.44577` | `0.08706` | `6.547e-24` | `2.619e-23` |
| `newey_west_entity_clustered_se` | `104040` | `0.4736` | `ame_vix_close` | `+0.01411` | `0.00035` | `1.608e-44` | `8.038e-44` |
| `newey_west_entity_clustered_se` | `104040` | `0.4736` | `tech_x_neg_ret_1d_vs_healthcare` | `-0.64789` | `0.28793` | `2.818e-02` | `5.636e-02` |
| `newey_west_entity_clustered_se` | `104040` | `0.4736` | `tech_x_vix_close_vs_healthcare` | `+0.00090` | `0.00083` | `2.802e-01` | `2.802e-01` |
| `newey_west_entity_clustered_se` | `104040` | `0.4736` | `joint_5sector_interaction_wald` | `+197.32523` | — | `3.708e-16` | `1.112e-15` |
| `pooled_har_rv_no_interactions` | `104040` | `0.4694` | `pooled_neg_ret_1d` | `-1.53696` | `0.30781` | `6.538e-07` | — |
| `pooled_har_rv_no_interactions` | `104040` | `0.4694` | `pooled_vix_close` | `+0.01347` | `0.00169` | `2.446e-15` | — |
| `extended_with_macro_and_events` | `104040` | `0.4748` | `ame_neg_ret_1d` | `-0.97266` | `0.27878` | `4.969e-04` | `1.491e-03` |
| `extended_with_macro_and_events` | `104040` | `0.4748` | `ame_vix_close` | `+0.01384` | `0.00171` | `1.289e-15` | `6.446e-15` |
| `extended_with_macro_and_events` | `104040` | `0.4748` | `tech_x_neg_ret_1d_vs_healthcare` | `-0.60093` | `0.36799` | `1.027e-01` | `2.053e-01` |
| `extended_with_macro_and_events` | `104040` | `0.4748` | `tech_x_vix_close_vs_healthcare` | `+0.00096` | `0.00103` | `3.547e-01` | `3.547e-01` |
| `extended_with_macro_and_events` | `104040` | `0.4748` | `joint_5sector_interaction_wald` | `+86.03073` | — | `7.257e-15` | `2.903e-14` |
