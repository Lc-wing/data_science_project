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
| h1_15_composite_gini | 0.1882 | 0.1881723798 | 2.76e-05 | PASS | outputs/tables/h1_inequality_lisa.csv | res9,15,composite,gini |
| h1_15_composite_gini_lo | 0.1563 | 0.1562516381 | 4.84e-05 | PASS | outputs/tables/h1_inequality_lisa.csv | res9,15,composite,gini_ci_low |
| h1_15_composite_gini_hi | 0.2202 | 0.220244597 | 4.46e-05 | PASS | outputs/tables/h1_inequality_lisa.csv | res9,15,composite,gini_ci_high |
| h1_15_composite_theil | 0.0673 | 0.06726645729 | 3.35e-05 | PASS | outputs/tables/h1_inequality_lisa.csv | res9,15,composite,theil_total |
| h1_15_composite_theil_lo | 0.0479 | 0.04789693839 | 3.06e-06 | PASS | outputs/tables/h1_inequality_lisa.csv | res9,15,composite,theil_total_ci_low |
| h1_15_composite_theil_hi | 0.0910 | 0.09104708528 | 4.71e-05 | PASS | outputs/tables/h1_inequality_lisa.csv | res9,15,composite,theil_total_ci_high |
| h3_acc15_park_OLS_HC3_elderly_share_coef | 3.3187 | 3.318688397 | 1.16e-05 | PASS | outputs/tables/h3_spatial_regression.csv | 15,acc15_park,knn4,OLS_HC3,elderly_share,coef |
| h3_acc15_park_OLS_HC3_elderly_share_lo | -20.4267 | -20.42667972 | 2.03e-05 | PASS | outputs/tables/h3_spatial_regression.csv | 15,acc15_park,knn4,OLS_HC3,elderly_share,ci_low |
| h3_acc15_park_OLS_HC3_elderly_share_hi | 27.0641 | 27.06405651 | 4.35e-05 | PASS | outputs/tables/h3_spatial_regression.csv | 15,acc15_park,knn4,OLS_HC3,elderly_share,ci_high |
| h3_acc15_park_OLS_HC3_elderly_share_p | 0.7736 | 0.7736426376 | 4.26e-05 | PASS | outputs/tables/h3_spatial_regression.csv | 15,acc15_park,knn4,OLS_HC3,elderly_share,p_value |
| h3_acc15_park_OLS_HC3_log_pop_density_coef | -0.8733 | -0.8733214499 | 2.14e-05 | PASS | outputs/tables/h3_spatial_regression.csv | 15,acc15_park,knn4,OLS_HC3,log_pop_density,coef |
| h3_acc15_park_OLS_HC3_log_pop_density_lo | -1.6363 | -1.636335497 | 3.55e-05 | PASS | outputs/tables/h3_spatial_regression.csv | 15,acc15_park,knn4,OLS_HC3,log_pop_density,ci_low |
| h3_acc15_park_OLS_HC3_log_pop_density_hi | -0.1103 | -0.1103074029 | 7.4e-06 | PASS | outputs/tables/h3_spatial_regression.csv | 15,acc15_park,knn4,OLS_HC3,log_pop_density,ci_high |
| h3_acc15_park_OLS_HC3_log_pop_density_p | 0.0269 | 0.02694658183 | 4.66e-05 | PASS | outputs/tables/h3_spatial_regression.csv | 15,acc15_park,knn4,OLS_HC3,log_pop_density,p_value |
| h3_acc15_park_SEM_ML_elderly_share_coef | 2.0076 | 2.00757519 | 2.48e-05 | PASS | outputs/tables/h3_spatial_regression.csv | 15,acc15_park,knn4,SEM_ML,elderly_share,coef |
| h3_acc15_park_SEM_ML_elderly_share_lo | -13.3137 | -13.31369129 | 8.71e-06 | PASS | outputs/tables/h3_spatial_regression.csv | 15,acc15_park,knn4,SEM_ML,elderly_share,ci_low |
| h3_acc15_park_SEM_ML_elderly_share_hi | 17.3288 | 17.32884167 | 4.17e-05 | PASS | outputs/tables/h3_spatial_regression.csv | 15,acc15_park,knn4,SEM_ML,elderly_share,ci_high |
| h3_acc15_park_SEM_ML_elderly_share_p | 0.7973 | 0.797319372 | 1.94e-05 | PASS | outputs/tables/h3_spatial_regression.csv | 15,acc15_park,knn4,SEM_ML,elderly_share,p_value |
| h3_acc15_park_SEM_ML_log_pop_density_coef | -0.8801 | -0.8801465299 | 4.65e-05 | PASS | outputs/tables/h3_spatial_regression.csv | 15,acc15_park,knn4,SEM_ML,log_pop_density,coef |
| h3_acc15_park_SEM_ML_log_pop_density_lo | -1.3103 | -1.310281049 | 1.9e-05 | PASS | outputs/tables/h3_spatial_regression.csv | 15,acc15_park,knn4,SEM_ML,log_pop_density,ci_low |
| h3_acc15_park_SEM_ML_log_pop_density_hi | -0.4500 | -0.450012011 | 1.2e-05 | PASS | outputs/tables/h3_spatial_regression.csv | 15,acc15_park,knn4,SEM_ML,log_pop_density,ci_high |
| h3_acc15_park_SEM_ML_log_pop_density_p | 0.0001 | 6.058969681e-05 | 3.94e-05 | PASS | outputs/tables/h3_spatial_regression.csv | 15,acc15_park,knn4,SEM_ML,log_pop_density,p_value |
| h3_acc15_park_SLM_ML_elderly_share_coef | 3.2136 | 3.213639851 | 3.99e-05 | PASS | outputs/tables/h3_spatial_regression.csv | 15,acc15_park,knn4,SLM_ML,elderly_share,coef |
| h3_acc15_park_SLM_ML_elderly_share_lo | -12.3661 | -12.36614384 | 4.38e-05 | PASS | outputs/tables/h3_spatial_regression.csv | 15,acc15_park,knn4,SLM_ML,elderly_share,ci_low |
| h3_acc15_park_SLM_ML_elderly_share_hi | 18.7934 | 18.79342354 | 2.35e-05 | PASS | outputs/tables/h3_spatial_regression.csv | 15,acc15_park,knn4,SLM_ML,elderly_share,ci_high |
| h3_acc15_park_SLM_ML_elderly_share_p | 0.6860 | 0.6860057257 | 5.73e-06 | PASS | outputs/tables/h3_spatial_regression.csv | 15,acc15_park,knn4,SLM_ML,elderly_share,p_value |
| h3_acc15_park_SLM_ML_log_pop_density_coef | -0.8933 | -0.8933167668 | 1.68e-05 | PASS | outputs/tables/h3_spatial_regression.csv | 15,acc15_park,knn4,SLM_ML,log_pop_density,coef |
| h3_acc15_park_SLM_ML_log_pop_density_lo | -1.3387 | -1.338714634 | 1.46e-05 | PASS | outputs/tables/h3_spatial_regression.csv | 15,acc15_park,knn4,SLM_ML,log_pop_density,ci_low |
| h3_acc15_park_SLM_ML_log_pop_density_hi | -0.4479 | -0.4479188995 | 1.89e-05 | PASS | outputs/tables/h3_spatial_regression.csv | 15,acc15_park,knn4,SLM_ML,log_pop_density,ci_high |
| h3_acc15_park_SLM_ML_log_pop_density_p | 0.0001 | 8.458562954e-05 | 1.54e-05 | PASS | outputs/tables/h3_spatial_regression.csv | 15,acc15_park,knn4,SLM_ML,log_pop_density,p_value |
| h3_acc15_composite_OLS_HC3_elderly_share_coef | 1.9488 | 1.948838124 | 3.81e-05 | PASS | outputs/tables/h3_spatial_regression.csv | 15,acc15_composite,knn4,OLS_HC3,elderly_share,coef |
| h3_acc15_composite_OLS_HC3_elderly_share_lo | -0.6967 | -0.6966604875 | 3.95e-05 | PASS | outputs/tables/h3_spatial_regression.csv | 15,acc15_composite,knn4,OLS_HC3,elderly_share,ci_low |
| h3_acc15_composite_OLS_HC3_elderly_share_hi | 4.5943 | 4.594336735 | 3.67e-05 | PASS | outputs/tables/h3_spatial_regression.csv | 15,acc15_composite,knn4,OLS_HC3,elderly_share,ci_high |
| h3_acc15_composite_OLS_HC3_elderly_share_p | 0.1400 | 0.1400493655 | 4.94e-05 | PASS | outputs/tables/h3_spatial_regression.csv | 15,acc15_composite,knn4,OLS_HC3,elderly_share,p_value |
| h3_acc15_composite_OLS_HC3_log_pop_density_coef | -0.0580 | -0.05798301179 | 1.7e-05 | PASS | outputs/tables/h3_spatial_regression.csv | 15,acc15_composite,knn4,OLS_HC3,log_pop_density,coef |
| h3_acc15_composite_OLS_HC3_log_pop_density_lo | -0.1185 | -0.1185073757 | 7.38e-06 | PASS | outputs/tables/h3_spatial_regression.csv | 15,acc15_composite,knn4,OLS_HC3,log_pop_density,ci_low |
| h3_acc15_composite_OLS_HC3_log_pop_density_hi | 0.0025 | 0.002541352104 | 4.14e-05 | PASS | outputs/tables/h3_spatial_regression.csv | 15,acc15_composite,knn4,OLS_HC3,log_pop_density,ci_high |
| h3_acc15_composite_OLS_HC3_log_pop_density_p | 0.0595 | 0.05945445767 | 4.55e-05 | PASS | outputs/tables/h3_spatial_regression.csv | 15,acc15_composite,knn4,OLS_HC3,log_pop_density,p_value |
| h3_acc15_composite_SEM_ML_elderly_share_coef | 1.9366 | 1.93661168 | 1.17e-05 | PASS | outputs/tables/h3_spatial_regression.csv | 15,acc15_composite,knn4,SEM_ML,elderly_share,coef |
| h3_acc15_composite_SEM_ML_elderly_share_lo | 0.4098 | 0.4098372766 | 3.73e-05 | PASS | outputs/tables/h3_spatial_regression.csv | 15,acc15_composite,knn4,SEM_ML,elderly_share,ci_low |
| h3_acc15_composite_SEM_ML_elderly_share_hi | 3.4634 | 3.463386083 | 1.39e-05 | PASS | outputs/tables/h3_spatial_regression.csv | 15,acc15_composite,knn4,SEM_ML,elderly_share,ci_high |
| h3_acc15_composite_SEM_ML_elderly_share_p | 0.0129 | 0.01291575397 | 1.58e-05 | PASS | outputs/tables/h3_spatial_regression.csv | 15,acc15_composite,knn4,SEM_ML,elderly_share,p_value |
| h3_acc15_composite_SEM_ML_log_pop_density_coef | -0.0560 | -0.0560087951 | 8.8e-06 | PASS | outputs/tables/h3_spatial_regression.csv | 15,acc15_composite,knn4,SEM_ML,log_pop_density,coef |
| h3_acc15_composite_SEM_ML_log_pop_density_lo | -0.0989 | -0.09893654589 | 3.65e-05 | PASS | outputs/tables/h3_spatial_regression.csv | 15,acc15_composite,knn4,SEM_ML,log_pop_density,ci_low |
| h3_acc15_composite_SEM_ML_log_pop_density_hi | -0.0131 | -0.01308104431 | 1.9e-05 | PASS | outputs/tables/h3_spatial_regression.csv | 15,acc15_composite,knn4,SEM_ML,log_pop_density,ci_high |
| h3_acc15_composite_SEM_ML_log_pop_density_p | 0.0106 | 0.01055158671 | 4.84e-05 | PASS | outputs/tables/h3_spatial_regression.csv | 15,acc15_composite,knn4,SEM_ML,log_pop_density,p_value |
| h3_acc15_composite_SLM_ML_elderly_share_coef | 1.9635 | 1.963532869 | 3.29e-05 | PASS | outputs/tables/h3_spatial_regression.csv | 15,acc15_composite,knn4,SLM_ML,elderly_share,coef |
| h3_acc15_composite_SLM_ML_elderly_share_lo | 0.4482 | 0.4481501187 | 4.99e-05 | PASS | outputs/tables/h3_spatial_regression.csv | 15,acc15_composite,knn4,SLM_ML,elderly_share,ci_low |
| h3_acc15_composite_SLM_ML_elderly_share_hi | 3.4789 | 3.478915619 | 1.56e-05 | PASS | outputs/tables/h3_spatial_regression.csv | 15,acc15_composite,knn4,SLM_ML,elderly_share,ci_high |
| h3_acc15_composite_SLM_ML_elderly_share_p | 0.0111 | 0.01109818948 | 1.81e-06 | PASS | outputs/tables/h3_spatial_regression.csv | 15,acc15_composite,knn4,SLM_ML,elderly_share,p_value |
| h3_acc15_composite_SLM_ML_log_pop_density_coef | -0.0562 | -0.05619173125 | 8.27e-06 | PASS | outputs/tables/h3_spatial_regression.csv | 15,acc15_composite,knn4,SLM_ML,log_pop_density,coef |
| h3_acc15_composite_SLM_ML_log_pop_density_lo | -0.0976 | -0.09759267527 | 7.32e-06 | PASS | outputs/tables/h3_spatial_regression.csv | 15,acc15_composite,knn4,SLM_ML,log_pop_density,ci_low |
| h3_acc15_composite_SLM_ML_log_pop_density_hi | -0.0148 | -0.01479078723 | 9.21e-06 | PASS | outputs/tables/h3_spatial_regression.csv | 15,acc15_composite,knn4,SLM_ML,log_pop_density,ci_high |
| h3_acc15_composite_SLM_ML_log_pop_density_p | 0.0078 | 0.007809998641 | 1e-05 | PASS | outputs/tables/h3_spatial_regression.csv | 15,acc15_composite,knn4,SLM_ML,log_pop_density,p_value |
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
| h3_first_ols_elderly_share_coef | 3.3187 | 3.318688397 | 1.16e-05 | PASS | outputs/tables/h3_spatial_regression.csv | 15,acc15_park,knn4,OLS_HC3,elderly_share,coef |
| comp_within_15 | 0.0487 | 0.0487238315 | 2.38e-05 | PASS | outputs/tables/h1_inequality_lisa.csv | res9,15,composite,theil_within |
| comp_between_15 | 0.0185 | 0.01854262579 | 4.26e-05 | PASS | outputs/tables/h1_inequality_lisa.csv | res9,15,composite,theil_between |
| comp_moran_15 | 0.6061 | 0.6061409358 | 4.09e-05 | PASS | outputs/tables/h1_inequality_lisa.csv | res9,15,composite,moran_i |
| comp_moran_p_15 | 0.0010 | 0.001 | 0 | PASS | outputs/tables/h1_inequality_lisa.csv | res9,15,composite,moran_p |
| comp_n_ll | 3 | 3 | 0 | PASS | outputs/tables/h1_inequality_lisa.csv | res9,15,composite,n_ll_fdr |
| comp_n_hh | 9 | 9 | 0 | PASS | outputs/tables/h1_inequality_lisa.csv | res9,15,composite,n_hh_fdr |
| ev_moran_15 | 0.3450 | 0.3449961536 | 3.85e-06 | PASS | outputs/tables/h1_inequality_lisa.csv | res9,15,ev_charger,moran_i |
| ev_moran_p_15 | 0.0010 | 0.001 | 0 | PASS | outputs/tables/h1_inequality_lisa.csv | res9,15,ev_charger,moran_p |
| ev_n_ll | 47 | 47 | 0 | PASS | outputs/tables/h1_inequality_lisa.csv | res9,15,ev_charger,n_ll_fdr |
| ev_n_hh | 1 | 1 | 0 | PASS | outputs/tables/h1_inequality_lisa.csv | res9,15,ev_charger,n_hh_fdr |
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
| maup_r8 | 0.2293 | 0.2293227082 | 2.27e-05 | PASS | outputs/tables/maup_r8_vs_r9.csv | 15,composite,gini_r8 |
| maup_r9 | 0.1882 | 0.1881723798 | 2.76e-05 | PASS | outputs/tables/maup_r8_vs_r9.csv | 15,composite,gini_r9 |
| maup_delta | -0.0412 | -0.04115032843 | 4.97e-05 | PASS | outputs/tables/maup_r8_vs_r9.csv | 15,composite,gini_delta_r9_minus_r8 |
| vif_eld | 1.1585 | 1.158482627 | 1.74e-05 | PASS | outputs/tables/h3_regression_diagnostics.json | vif.elderly_share |
| vif_dens | 1.1585 | 1.158482627 | 1.74e-05 | PASS | outputs/tables/h3_regression_diagnostics.json | vif.log_pop_density |
| rf_mean_roc | 0.7728 | 0.77283316 | 3.32e-05 | PASS | outputs/metrics/phase_a_ml_baselines.json | models.random_forest.mean_roc_auc |
| lr_mean_roc | 0.7363 | 0.73630408 | 4.08e-06 | PASS | outputs/metrics/phase_a_ml_baselines.json | models.logistic_regression.mean_roc_auc |
| rf_std_roc | 0.1112 | 0.1112258 | 2.58e-05 | PASS | outputs/metrics/phase_a_ml_baselines.json | models.random_forest.std_roc_auc |
| imputed_cap | 12.0 | 12 | 0 | PASS | outputs/metrics/data_provenance.json | cem fallback_reason capacity=12.0 |
| ev_zero_pop_15 | 1.13 | 1.127965361 | 0.00203 | PASS | data/frozen/cells.gpkg | pop_pct(acc15_ev_charger==0) |
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
