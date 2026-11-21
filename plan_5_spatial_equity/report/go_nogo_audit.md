# Phase A 12/03 go/no-go audit

Checkpoint verdict: GO.
Numeric audit: PASS.
Every reported number below matches its source cell within rounding tolerance.
Main text length: 3.76 A4 pages.

## Minimum requirements

- MR1 Research question: H1, H2, and H3 are stated, with DSSCU, DSAT, and residents as the users of the description.
- MR2 Data collection and preparation: DSEC, CEM, and OpenStreetMap in EPSG:32649, with 25 m cross-source dedup. Terrestrial population control 681,293 across 23 zones and 7 parishes.
- MR3 Analysis: 2SFCA, Gini, Theil, LISA, side-by-side OLS / SEM / SLM, and spatial-block classifiers.
- MR4 Evaluation: bootstrap intervals, FDR, Holm adjustment, MAUP, ROC-AUC, PR-AUC, and Brier score.
- MR5 Results and discussion: Figure 1, Table 1, four limitations, correlational wording only.

## Model policy

H3 keeps OLS-HC3, SEM, and SLM side by side. A diagnostic p-value does not select a model.

## Limitations named in the draft

- Uniform building height in high-rise districts, including the residual dasymetric limit of building-group polygons.
- Incomplete facility capacity metadata, including imputed park area and station 9488798.
- No slope, incomplete footbridges and overpasses, and no cross-harbour walk link.
- Ecological fallacy: zone and cell associations are not individual associations.

## Claim trace

| Claim | Reported | Source value | Abs err | Status | Source | Locator |
| --- | ---: | ---: | ---: | --- | --- | --- |
| h1_15_park_gini | 0.6115 | 0.6115296382 | 2.96e-05 | PASS | outputs/tables/h1_inequality_lisa.csv | res9,15,park,gini |
| h1_15_park_gini_lo | 0.5288 | 0.5287589304 | 4.11e-05 | PASS | outputs/tables/h1_inequality_lisa.csv | res9,15,park,gini_ci_low |
| h1_15_park_gini_hi | 0.6771 | 0.6770734313 | 2.66e-05 | PASS | outputs/tables/h1_inequality_lisa.csv | res9,15,park,gini_ci_high |
| h1_15_park_theil | 1.2912 | 1.291193523 | 6.48e-06 | PASS | outputs/tables/h1_inequality_lisa.csv | res9,15,park,theil_total |
| h1_15_park_theil_lo | 0.9090 | 0.9089824985 | 1.75e-05 | PASS | outputs/tables/h1_inequality_lisa.csv | res9,15,park,theil_total_ci_low |
| h1_15_park_theil_hi | 1.6655 | 1.665462139 | 3.79e-05 | PASS | outputs/tables/h1_inequality_lisa.csv | res9,15,park,theil_total_ci_high |
| h1_15_health_centre_gini | 0.4923 | 0.4922984285 | 1.57e-06 | PASS | outputs/tables/h1_inequality_lisa.csv | res9,15,health_centre,gini |
| h1_15_health_centre_gini_lo | 0.4124 | 0.4124359908 | 3.6e-05 | PASS | outputs/tables/h1_inequality_lisa.csv | res9,15,health_centre,gini_ci_low |
| h1_15_health_centre_gini_hi | 0.5655 | 0.565539104 | 3.91e-05 | PASS | outputs/tables/h1_inequality_lisa.csv | res9,15,health_centre,gini_ci_high |
| h1_15_health_centre_theil | 0.5383 | 0.5382848402 | 1.52e-05 | PASS | outputs/tables/h1_inequality_lisa.csv | res9,15,health_centre,theil_total |
| h1_15_health_centre_theil_lo | 0.3384 | 0.3383926904 | 7.31e-06 | PASS | outputs/tables/h1_inequality_lisa.csv | res9,15,health_centre,theil_total_ci_low |
| h1_15_health_centre_theil_hi | 0.7629 | 0.7628564982 | 4.35e-05 | PASS | outputs/tables/h1_inequality_lisa.csv | res9,15,health_centre,theil_total_ci_high |
| h1_15_library_gini | 0.3249 | 0.3249450825 | 4.51e-05 | PASS | outputs/tables/h1_inequality_lisa.csv | res9,15,library,gini |
| h1_15_library_gini_lo | 0.2624 | 0.2624031132 | 3.11e-06 | PASS | outputs/tables/h1_inequality_lisa.csv | res9,15,library,gini_ci_low |
| h1_15_library_gini_hi | 0.3920 | 0.392008469 | 8.47e-06 | PASS | outputs/tables/h1_inequality_lisa.csv | res9,15,library,gini_ci_high |
| h1_15_library_theil | 0.2766 | 0.2765976976 | 2.3e-06 | PASS | outputs/tables/h1_inequality_lisa.csv | res9,15,library,theil_total |
| h1_15_library_theil_lo | 0.1486 | 0.1485655122 | 3.45e-05 | PASS | outputs/tables/h1_inequality_lisa.csv | res9,15,library,theil_total_ci_low |
| h1_15_library_theil_hi | 0.4396 | 0.4396440124 | 4.4e-05 | PASS | outputs/tables/h1_inequality_lisa.csv | res9,15,library,theil_total_ci_high |
| h1_15_bus_stop_gini | 0.3876 | 0.3875671754 | 3.28e-05 | PASS | outputs/tables/h1_inequality_lisa.csv | res9,15,bus_stop,gini |
| h1_15_bus_stop_gini_lo | 0.3326 | 0.3326298934 | 2.99e-05 | PASS | outputs/tables/h1_inequality_lisa.csv | res9,15,bus_stop,gini_ci_low |
| h1_15_bus_stop_gini_hi | 0.4411 | 0.441064716 | 3.53e-05 | PASS | outputs/tables/h1_inequality_lisa.csv | res9,15,bus_stop,gini_ci_high |
| h1_15_bus_stop_theil | 0.5972 | 0.5971865608 | 1.34e-05 | PASS | outputs/tables/h1_inequality_lisa.csv | res9,15,bus_stop,theil_total |
| h1_15_bus_stop_theil_lo | 0.4553 | 0.4553361795 | 3.62e-05 | PASS | outputs/tables/h1_inequality_lisa.csv | res9,15,bus_stop,theil_total_ci_low |
| h1_15_bus_stop_theil_hi | 0.7677 | 0.7677260459 | 2.6e-05 | PASS | outputs/tables/h1_inequality_lisa.csv | res9,15,bus_stop,theil_total_ci_high |
| h1_15_ev_charger_gini | 0.4984 | 0.4983717582 | 2.82e-05 | PASS | outputs/tables/h1_inequality_lisa.csv | res9,15,ev_charger,gini |
| h1_15_ev_charger_gini_lo | 0.4315 | 0.4314657599 | 3.42e-05 | PASS | outputs/tables/h1_inequality_lisa.csv | res9,15,ev_charger,gini_ci_low |
| h1_15_ev_charger_gini_hi | 0.5566 | 0.5566273144 | 2.73e-05 | PASS | outputs/tables/h1_inequality_lisa.csv | res9,15,ev_charger,gini_ci_high |
| h1_15_ev_charger_theil | 0.6706 | 0.6706185562 | 1.86e-05 | PASS | outputs/tables/h1_inequality_lisa.csv | res9,15,ev_charger,theil_total |
| h1_15_ev_charger_theil_lo | 0.4408 | 0.4407704387 | 2.96e-05 | PASS | outputs/tables/h1_inequality_lisa.csv | res9,15,ev_charger,theil_total_ci_low |
| h1_15_ev_charger_theil_hi | 0.9414 | 0.9414129481 | 1.29e-05 | PASS | outputs/tables/h1_inequality_lisa.csv | res9,15,ev_charger,theil_total_ci_high |
| h1_15_composite_gini | 0.3203 | 0.32025596 | 4.4e-05 | PASS | outputs/tables/h1_inequality_lisa.csv | res9,15,composite,gini |
| h1_15_composite_gini_lo | 0.2484 | 0.248368577 | 3.14e-05 | PASS | outputs/tables/h1_inequality_lisa.csv | res9,15,composite,gini_ci_low |
| h1_15_composite_gini_hi | 0.3937 | 0.3937260543 | 2.61e-05 | PASS | outputs/tables/h1_inequality_lisa.csv | res9,15,composite,gini_ci_high |
| h1_15_composite_theil | 0.2883 | 0.2883206076 | 2.06e-05 | PASS | outputs/tables/h1_inequality_lisa.csv | res9,15,composite,theil_total |
| h1_15_composite_theil_lo | 0.1293 | 0.1293354578 | 3.55e-05 | PASS | outputs/tables/h1_inequality_lisa.csv | res9,15,composite,theil_total_ci_low |
| h1_15_composite_theil_hi | 0.4828 | 0.4827728989 | 2.71e-05 | PASS | outputs/tables/h1_inequality_lisa.csv | res9,15,composite,theil_total_ci_high |
| h3_acc15_park_OLS_HC3_elderly_share_coef | 3.0492 | 3.049246811 | 4.68e-05 | PASS | outputs/tables/h3_spatial_regression.csv | 15,acc15_park,knn4,OLS_HC3,elderly_share,coef |
| h3_acc15_park_OLS_HC3_elderly_share_lo | -22.2117 | -22.21173669 | 3.67e-05 | PASS | outputs/tables/h3_spatial_regression.csv | 15,acc15_park,knn4,OLS_HC3,elderly_share,ci_low |
| h3_acc15_park_OLS_HC3_elderly_share_hi | 28.3102 | 28.31023031 | 3.03e-05 | PASS | outputs/tables/h3_spatial_regression.csv | 15,acc15_park,knn4,OLS_HC3,elderly_share,ci_high |
| h3_acc15_park_OLS_HC3_elderly_share_p | 0.8130 | 0.8129740292 | 2.6e-05 | PASS | outputs/tables/h3_spatial_regression.csv | 15,acc15_park,knn4,OLS_HC3,elderly_share,p_value |
| h3_acc15_park_OLS_HC3_log_pop_density_coef | -0.4191 | -0.4191385733 | 3.86e-05 | PASS | outputs/tables/h3_spatial_regression.csv | 15,acc15_park,knn4,OLS_HC3,log_pop_density,coef |
| h3_acc15_park_OLS_HC3_log_pop_density_lo | -1.1053 | -1.105305601 | 5.6e-06 | PASS | outputs/tables/h3_spatial_regression.csv | 15,acc15_park,knn4,OLS_HC3,log_pop_density,ci_low |
| h3_acc15_park_OLS_HC3_log_pop_density_hi | 0.2670 | 0.2670284542 | 2.85e-05 | PASS | outputs/tables/h3_spatial_regression.csv | 15,acc15_park,knn4,OLS_HC3,log_pop_density,ci_high |
| h3_acc15_park_OLS_HC3_log_pop_density_p | 0.2312 | 0.2312101885 | 1.02e-05 | PASS | outputs/tables/h3_spatial_regression.csv | 15,acc15_park,knn4,OLS_HC3,log_pop_density,p_value |
| h3_acc15_park_SEM_ML_elderly_share_coef | 5.7831 | 5.783127973 | 2.8e-05 | PASS | outputs/tables/h3_spatial_regression.csv | 15,acc15_park,knn4,SEM_ML,elderly_share,coef |
| h3_acc15_park_SEM_ML_elderly_share_lo | -10.6765 | -10.67648659 | 1.34e-05 | PASS | outputs/tables/h3_spatial_regression.csv | 15,acc15_park,knn4,SEM_ML,elderly_share,ci_low |
| h3_acc15_park_SEM_ML_elderly_share_hi | 22.2427 | 22.24274254 | 4.25e-05 | PASS | outputs/tables/h3_spatial_regression.csv | 15,acc15_park,knn4,SEM_ML,elderly_share,ci_high |
| h3_acc15_park_SEM_ML_elderly_share_p | 0.4910 | 0.4910428882 | 4.29e-05 | PASS | outputs/tables/h3_spatial_regression.csv | 15,acc15_park,knn4,SEM_ML,elderly_share,p_value |
| h3_acc15_park_SEM_ML_log_pop_density_coef | -0.4911 | -0.491125572 | 2.56e-05 | PASS | outputs/tables/h3_spatial_regression.csv | 15,acc15_park,knn4,SEM_ML,log_pop_density,coef |
| h3_acc15_park_SEM_ML_log_pop_density_lo | -0.9136 | -0.9136444417 | 4.44e-05 | PASS | outputs/tables/h3_spatial_regression.csv | 15,acc15_park,knn4,SEM_ML,log_pop_density,ci_low |
| h3_acc15_park_SEM_ML_log_pop_density_hi | -0.0686 | -0.06860670215 | 6.7e-06 | PASS | outputs/tables/h3_spatial_regression.csv | 15,acc15_park,knn4,SEM_ML,log_pop_density,ci_high |
| h3_acc15_park_SEM_ML_log_pop_density_p | 0.0227 | 0.02271133365 | 1.13e-05 | PASS | outputs/tables/h3_spatial_regression.csv | 15,acc15_park,knn4,SEM_ML,log_pop_density,p_value |
| h3_acc15_park_SLM_ML_elderly_share_coef | 1.9436 | 1.943643576 | 4.36e-05 | PASS | outputs/tables/h3_spatial_regression.csv | 15,acc15_park,knn4,SLM_ML,elderly_share,coef |
| h3_acc15_park_SLM_ML_elderly_share_lo | -15.3779 | -15.3779261 | 2.61e-05 | PASS | outputs/tables/h3_spatial_regression.csv | 15,acc15_park,knn4,SLM_ML,elderly_share,ci_low |
| h3_acc15_park_SLM_ML_elderly_share_hi | 19.2652 | 19.26521325 | 1.32e-05 | PASS | outputs/tables/h3_spatial_regression.csv | 15,acc15_park,knn4,SLM_ML,elderly_share,ci_high |
| h3_acc15_park_SLM_ML_elderly_share_p | 0.8259 | 0.8259252854 | 2.53e-05 | PASS | outputs/tables/h3_spatial_regression.csv | 15,acc15_park,knn4,SLM_ML,elderly_share,p_value |
| h3_acc15_park_SLM_ML_log_pop_density_coef | -0.5196 | -0.5195813411 | 1.87e-05 | PASS | outputs/tables/h3_spatial_regression.csv | 15,acc15_park,knn4,SLM_ML,log_pop_density,coef |
| h3_acc15_park_SLM_ML_log_pop_density_lo | -0.9965 | -0.9964998288 | 1.71e-07 | PASS | outputs/tables/h3_spatial_regression.csv | 15,acc15_park,knn4,SLM_ML,log_pop_density,ci_low |
| h3_acc15_park_SLM_ML_log_pop_density_hi | -0.0427 | -0.04266285344 | 3.71e-05 | PASS | outputs/tables/h3_spatial_regression.csv | 15,acc15_park,knn4,SLM_ML,log_pop_density,ci_high |
| h3_acc15_park_SLM_ML_log_pop_density_p | 0.0327 | 0.03273388021 | 3.39e-05 | PASS | outputs/tables/h3_spatial_regression.csv | 15,acc15_park,knn4,SLM_ML,log_pop_density,p_value |
| h3_acc15_composite_OLS_HC3_elderly_share_coef | -0.2375 | -0.2375115497 | 1.15e-05 | PASS | outputs/tables/h3_spatial_regression.csv | 15,acc15_composite,knn4,OLS_HC3,elderly_share,coef |
| h3_acc15_composite_OLS_HC3_elderly_share_lo | -2.2232 | -2.223178118 | 2.19e-05 | PASS | outputs/tables/h3_spatial_regression.csv | 15,acc15_composite,knn4,OLS_HC3,elderly_share,ci_low |
| h3_acc15_composite_OLS_HC3_elderly_share_hi | 1.7482 | 1.748155019 | 4.5e-05 | PASS | outputs/tables/h3_spatial_regression.csv | 15,acc15_composite,knn4,OLS_HC3,elderly_share,ci_high |
| h3_acc15_composite_OLS_HC3_elderly_share_p | 0.8146 | 0.8146422483 | 4.22e-05 | PASS | outputs/tables/h3_spatial_regression.csv | 15,acc15_composite,knn4,OLS_HC3,elderly_share,p_value |
| h3_acc15_composite_OLS_HC3_log_pop_density_coef | -0.0140 | -0.01399228004 | 7.72e-06 | PASS | outputs/tables/h3_spatial_regression.csv | 15,acc15_composite,knn4,OLS_HC3,log_pop_density,coef |
| h3_acc15_composite_OLS_HC3_log_pop_density_lo | -0.0746 | -0.07458449052 | 1.55e-05 | PASS | outputs/tables/h3_spatial_regression.csv | 15,acc15_composite,knn4,OLS_HC3,log_pop_density,ci_low |
| h3_acc15_composite_OLS_HC3_log_pop_density_hi | 0.0466 | 0.04659993044 | 6.96e-08 | PASS | outputs/tables/h3_spatial_regression.csv | 15,acc15_composite,knn4,OLS_HC3,log_pop_density,ci_high |
| h3_acc15_composite_OLS_HC3_log_pop_density_p | 0.6508 | 0.6508268808 | 2.69e-05 | PASS | outputs/tables/h3_spatial_regression.csv | 15,acc15_composite,knn4,OLS_HC3,log_pop_density,p_value |
| h3_acc15_composite_SEM_ML_elderly_share_coef | -0.1463 | -0.1463302563 | 3.03e-05 | PASS | outputs/tables/h3_spatial_regression.csv | 15,acc15_composite,knn4,SEM_ML,elderly_share,coef |
| h3_acc15_composite_SEM_ML_elderly_share_lo | -1.6411 | -1.641131346 | 3.13e-05 | PASS | outputs/tables/h3_spatial_regression.csv | 15,acc15_composite,knn4,SEM_ML,elderly_share,ci_low |
| h3_acc15_composite_SEM_ML_elderly_share_hi | 1.3485 | 1.348470834 | 2.92e-05 | PASS | outputs/tables/h3_spatial_regression.csv | 15,acc15_composite,knn4,SEM_ML,elderly_share,ci_high |
| h3_acc15_composite_SEM_ML_elderly_share_p | 0.8478 | 0.8478441331 | 4.41e-05 | PASS | outputs/tables/h3_spatial_regression.csv | 15,acc15_composite,knn4,SEM_ML,elderly_share,p_value |
| h3_acc15_composite_SEM_ML_log_pop_density_coef | -0.0198 | -0.01983259269 | 3.26e-05 | PASS | outputs/tables/h3_spatial_regression.csv | 15,acc15_composite,knn4,SEM_ML,log_pop_density,coef |
| h3_acc15_composite_SEM_ML_log_pop_density_lo | -0.0580 | -0.05800607016 | 6.07e-06 | PASS | outputs/tables/h3_spatial_regression.csv | 15,acc15_composite,knn4,SEM_ML,log_pop_density,ci_low |
| h3_acc15_composite_SEM_ML_log_pop_density_hi | 0.0183 | 0.01834088477 | 4.09e-05 | PASS | outputs/tables/h3_spatial_regression.csv | 15,acc15_composite,knn4,SEM_ML,log_pop_density,ci_high |
| h3_acc15_composite_SEM_ML_log_pop_density_p | 0.3085 | 0.3085375295 | 3.75e-05 | PASS | outputs/tables/h3_spatial_regression.csv | 15,acc15_composite,knn4,SEM_ML,log_pop_density,p_value |
| h3_acc15_composite_SLM_ML_elderly_share_coef | -0.3498 | -0.3497802438 | 1.98e-05 | PASS | outputs/tables/h3_spatial_regression.csv | 15,acc15_composite,knn4,SLM_ML,elderly_share,coef |
| h3_acc15_composite_SLM_ML_elderly_share_lo | -1.9324 | -1.932430803 | 3.08e-05 | PASS | outputs/tables/h3_spatial_regression.csv | 15,acc15_composite,knn4,SLM_ML,elderly_share,ci_low |
| h3_acc15_composite_SLM_ML_elderly_share_hi | 1.2329 | 1.232870316 | 2.97e-05 | PASS | outputs/tables/h3_spatial_regression.csv | 15,acc15_composite,knn4,SLM_ML,elderly_share,ci_high |
| h3_acc15_composite_SLM_ML_elderly_share_p | 0.6649 | 0.6648855182 | 1.45e-05 | PASS | outputs/tables/h3_spatial_regression.csv | 15,acc15_composite,knn4,SLM_ML,elderly_share,p_value |
| h3_acc15_composite_SLM_ML_log_pop_density_coef | -0.0210 | -0.02099562887 | 4.37e-06 | PASS | outputs/tables/h3_spatial_regression.csv | 15,acc15_composite,knn4,SLM_ML,log_pop_density,coef |
| h3_acc15_composite_SLM_ML_log_pop_density_lo | -0.0645 | -0.06450561778 | 5.62e-06 | PASS | outputs/tables/h3_spatial_regression.csv | 15,acc15_composite,knn4,SLM_ML,log_pop_density,ci_low |
| h3_acc15_composite_SLM_ML_log_pop_density_hi | 0.0225 | 0.02251436004 | 1.44e-05 | PASS | outputs/tables/h3_spatial_regression.csv | 15,acc15_composite,knn4,SLM_ML,log_pop_density,ci_high |
| h3_acc15_composite_SLM_ML_log_pop_density_p | 0.3443 | 0.344254298 | 4.57e-05 | PASS | outputs/tables/h3_spatial_regression.csv | 15,acc15_composite,knn4,SLM_ML,log_pop_density,p_value |
| ml_naive_prior_oof_roc_auc | 0.2171 | 0.21706865 | 3.13e-05 | PASS | outputs/metrics/phase_a_ml_baselines.json | models.naive_prior.oof_roc_auc |
| ml_naive_prior_oof_pr_auc | 0.1359 | 0.13590369 | 3.69e-06 | PASS | outputs/metrics/phase_a_ml_baselines.json | models.naive_prior.oof_pr_auc |
| ml_naive_prior_oof_brier | 0.1756 | 0.17564041 | 4.04e-05 | PASS | outputs/metrics/phase_a_ml_baselines.json | models.naive_prior.oof_brier |
| ml_naive_prior_oof_top20_hit_rate | 0.1071 | 0.10714286 | 4.29e-05 | PASS | outputs/metrics/phase_a_ml_baselines.json | models.naive_prior.oof_top20_hit_rate |
| ml_logistic_regression_oof_roc_auc | 0.8527 | 0.85273655 | 3.66e-05 | PASS | outputs/metrics/phase_a_ml_baselines.json | models.logistic_regression.oof_roc_auc |
| ml_logistic_regression_oof_pr_auc | 0.5320 | 0.53199178 | 8.22e-06 | PASS | outputs/metrics/phase_a_ml_baselines.json | models.logistic_regression.oof_pr_auc |
| ml_logistic_regression_oof_brier | 0.1160 | 0.11595394 | 4.61e-05 | PASS | outputs/metrics/phase_a_ml_baselines.json | models.logistic_regression.oof_brier |
| ml_logistic_regression_oof_top20_hit_rate | 0.5714 | 0.57142857 | 2.86e-05 | PASS | outputs/metrics/phase_a_ml_baselines.json | models.logistic_regression.oof_top20_hit_rate |
| ml_random_forest_oof_roc_auc | 0.8581 | 0.8580705 | 2.95e-05 | PASS | outputs/metrics/phase_a_ml_baselines.json | models.random_forest.oof_roc_auc |
| ml_random_forest_oof_pr_auc | 0.5780 | 0.577992 | 8e-06 | PASS | outputs/metrics/phase_a_ml_baselines.json | models.random_forest.oof_pr_auc |
| ml_random_forest_oof_brier | 0.1132 | 0.11316607 | 3.39e-05 | PASS | outputs/metrics/phase_a_ml_baselines.json | models.random_forest.oof_brier |
| ml_random_forest_oof_top20_hit_rate | 0.5536 | 0.55357143 | 2.86e-05 | PASS | outputs/metrics/phase_a_ml_baselines.json | models.random_forest.oof_top20_hit_rate |
| h3_first_ols_elderly_share_coef | 3.0492 | 3.049246811 | 4.68e-05 | PASS | outputs/tables/h3_spatial_regression.csv | 15,acc15_park,knn4,OLS_HC3,elderly_share,coef |
| comp_within_15 | 0.2041 | 0.2040564444 | 4.36e-05 | PASS | outputs/tables/h1_inequality_lisa.csv | res9,15,composite,theil_within |
| comp_between_15 | 0.0843 | 0.08426416319 | 3.58e-05 | PASS | outputs/tables/h1_inequality_lisa.csv | res9,15,composite,theil_between |
| comp_moran_15 | 0.6678 | 0.6678158237 | 1.58e-05 | PASS | outputs/tables/h1_inequality_lisa.csv | res9,15,composite,moran_i |
| comp_moran_p_15 | 0.0010 | 0.001 | 0 | PASS | outputs/tables/h1_inequality_lisa.csv | res9,15,composite,moran_p |
| comp_n_ll | 17 | 17 | 0 | PASS | outputs/tables/h1_inequality_lisa.csv | res9,15,composite,n_ll_fdr |
| comp_n_hh | 11 | 11 | 0 | PASS | outputs/tables/h1_inequality_lisa.csv | res9,15,composite,n_hh_fdr |
| ev_moran_15 | 0.3450 | 0.3449961536 | 3.85e-06 | PASS | outputs/tables/h1_inequality_lisa.csv | res9,15,ev_charger,moran_i |
| ev_zero_15 | 35.54 | 35.54006969 | 6.97e-05 | PASS | outputs/tables/h1_inequality_lisa.csv | res9,15,ev_charger,zero_acc_pct |
| ev_gini_10 | 0.5850 | 0.5849515981 | 4.84e-05 | PASS | outputs/tables/h1_inequality_lisa.csv | res9,10,ev_charger,gini |
| park_gini_10 | 0.6960 | 0.6959994345 | 5.65e-07 | PASS | outputs/tables/h1_inequality_lisa.csv | res9,10,park,gini |
| h2_park_diff | -0.1132 | -0.11315788 | 4.21e-05 | PASS | outputs/tables/h2_gini_diff_bootstrap.csv | 15,ev_charger_vs_park,gini_diff |
| h2_park_lo | -0.1954 | -0.1953575871 | 4.24e-05 | PASS | outputs/tables/h2_gini_diff_bootstrap.csv | 15,ev_charger_vs_park,ci_low |
| h2_park_hi | -0.0298 | -0.02984730151 | 4.73e-05 | PASS | outputs/tables/h2_gini_diff_bootstrap.csv | 15,ev_charger_vs_park,ci_high |
| h2_park_holm | 0.0550 | 0.05497251374 | 2.75e-05 | PASS | outputs/tables/h2_gini_diff_bootstrap.csv | 15,ev_charger_vs_park,p_holm |
| h2_lib_diff | 0.1734 | 0.1734266758 | 2.67e-05 | PASS | outputs/tables/h2_gini_diff_bootstrap.csv | 15,ev_charger_vs_library,gini_diff |
| h2_lib_holm | 0.0080 | 0.007996001999 | 4e-06 | PASS | outputs/tables/h2_gini_diff_bootstrap.csv | 15,ev_charger_vs_library,p_holm |
| h2_bus_diff | 0.1108 | 0.1108045828 | 4.58e-06 | PASS | outputs/tables/h2_gini_diff_bootstrap.csv | 15,ev_charger_vs_bus_stop,gini_diff |
| h2_bus_holm | 0.0080 | 0.007996001999 | 4e-06 | PASS | outputs/tables/h2_gini_diff_bootstrap.csv | 15,ev_charger_vs_bus_stop,p_holm |
| h2_health_diff | 0.0061 | 0.006073329742 | 2.67e-05 | PASS | outputs/tables/h2_gini_diff_bootstrap.csv | 15,ev_charger_vs_health_centre,gini_diff |
| h2_health_lo | -0.0815 | -0.08151632797 | 1.63e-05 | PASS | outputs/tables/h2_gini_diff_bootstrap.csv | 15,ev_charger_vs_health_centre,ci_low |
| h2_health_hi | 0.0936 | 0.09356354546 | 3.65e-05 | PASS | outputs/tables/h2_gini_diff_bootstrap.csv | 15,ev_charger_vs_health_centre,ci_high |
| h2_health_holm | 0.8926 | 0.8925537231 | 4.63e-05 | PASS | outputs/tables/h2_gini_diff_bootstrap.csv | 15,ev_charger_vs_health_centre,p_holm |
| maup_r8 | 0.3660 | 0.3660031036 | 3.1e-06 | PASS | outputs/tables/maup_r8_vs_r9.csv | 15,composite,gini_r8 |
| maup_r9 | 0.3203 | 0.32025596 | 4.4e-05 | PASS | outputs/tables/maup_r8_vs_r9.csv | 15,composite,gini_r9 |
| maup_delta | -0.0457 | -0.04574714363 | 4.71e-05 | PASS | outputs/tables/maup_r8_vs_r9.csv | 15,composite,gini_delta_r9_minus_r8 |
| vif_eld | 1.1585 | 1.158482627 | 1.74e-05 | PASS | outputs/tables/h3_regression_diagnostics.json | vif.elderly_share |
| vif_dens | 1.1585 | 1.158482627 | 1.74e-05 | PASS | outputs/tables/h3_regression_diagnostics.json | vif.log_pop_density |
| rf_mean_roc | 0.7728 | 0.77283316 | 3.32e-05 | PASS | outputs/metrics/phase_a_ml_baselines.json | models.random_forest.mean_roc_auc |
| lr_mean_roc | 0.7363 | 0.73630408 | 4.08e-06 | PASS | outputs/metrics/phase_a_ml_baselines.json | models.logistic_regression.mean_roc_auc |
| rf_std_roc | 0.1112 | 0.1112258 | 2.58e-05 | PASS | outputs/metrics/phase_a_ml_baselines.json | models.random_forest.std_roc_auc |
| imputed_cap | 12.0 | 12 | 0 | PASS | outputs/metrics/data_provenance.json | cem fallback_reason capacity=12.0 |
| pop_total | 681,293 | 681293 | 0 | PASS | data/frozen/cells_base_res9.gpkg | sum(pop_total) matches DSEC terrestrial control |
| pop_65 | 82,751 | 82751 | 0 | PASS | data/frozen/cells_base_res9.gpkg | sum(pop_65plus) matches DSEC terrestrial control |
| n_cells | 287 | 287 | 0 | PASS | outputs/metrics/phase_a_ml_baselines.json | n_cells |
| n_pos | 56 | 56 | 0 | PASS | outputs/metrics/phase_a_ml_baselines.json | n_pos |
| prevalence | 0.1951 | 0.1951219512 | 2.2e-05 | PASS | outputs/metrics/phase_a_ml_baselines.json | prevalence |
| n_res8 | 63 | 63 | 0 | PASS | data/frozen/cells_base_res8.gpkg | row_count |
| n_park | 240 | 240 | 0 | PASS | data/frozen/facilities_clean.gpkg | count(facility_type=park) |
| n_health | 11 | 11 | 0 | PASS | data/frozen/facilities_clean.gpkg | count(facility_type=health_centre) |
| n_library | 15 | 15 | 0 | PASS | data/frozen/facilities_clean.gpkg | count(facility_type=library) |
| n_bus | 587 | 587 | 0 | PASS | data/frozen/facilities_clean.gpkg | count(facility_type=bus_stop) |
| n_ev | 85 | 85 | 0 | PASS | data/frozen/facilities_clean.gpkg | count(facility_type=ev_charger) |
| walk_kmh | 4.5 | 4.5 | 0 | PASS | src/macao_equity/config.py | WALK_SPEED_KMH |
| seed | 42 | 42 | 0 | PASS | outputs/metrics/phase_a_ml_baselines.json | seed |
| n_boot | 2000 | 2000 | 0 | PASS | outputs/tables/h2_gini_diff_bootstrap.csv | n_boot |
| n_perm | 999 | 999 | 0 | PASS | outputs/tables/h3_regression_diagnostics.json | permutations |
| n_zones | 23 | 23 | 0 | PASS | data/frozen/zones_23.gpkg | row_count |
| n_parishes | 7 | 7 | 0 | PASS | data/frozen/parishes_7.gpkg | row_count |

Status of the numeric trace: PASS.
The frozen bundle checksum is in `data/frozen/FREEZE_MANIFEST.json`.
