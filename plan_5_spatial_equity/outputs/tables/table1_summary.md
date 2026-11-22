# Table 1. Phase A summary

Copied from `h1_inequality_lisa.csv`, `h3_spatial_regression.csv`, and `phase_a_ml_baselines.json`. H3 reports OLS_HC3, SEM_ML, and SLM_ML side by side for the same 15-minute outcomes on the k=4 nearest-neighbour weights. No model is selected from a diagnostic p-value.

| Panel | Item | Metric | Estimate | CI low | CI high | Aux metric | Aux value |
| --- | --- | --- | ---: | ---: | ---: | --- | ---: |
| H1_Gini_Theil_15min_res9 | park | gini | 0.6115 | 0.5288 | 0.6771 | theil_total | 1.2912 |
| H1_Gini_Theil_15min_res9 | park | theil_total | 1.2912 | 0.9090 | 1.6655 | gini | 0.6115 |
| H1_Gini_Theil_15min_res9 | health_centre | gini | 0.4923 | 0.4124 | 0.5655 | theil_total | 0.5383 |
| H1_Gini_Theil_15min_res9 | health_centre | theil_total | 0.5383 | 0.3384 | 0.7629 | gini | 0.4923 |
| H1_Gini_Theil_15min_res9 | library | gini | 0.3249 | 0.2624 | 0.3920 | theil_total | 0.2766 |
| H1_Gini_Theil_15min_res9 | library | theil_total | 0.2766 | 0.1486 | 0.4396 | gini | 0.3249 |
| H1_Gini_Theil_15min_res9 | bus_stop | gini | 0.3876 | 0.3326 | 0.4411 | theil_total | 0.5972 |
| H1_Gini_Theil_15min_res9 | bus_stop | theil_total | 0.5972 | 0.4553 | 0.7677 | gini | 0.3876 |
| H1_Gini_Theil_15min_res9 | ev_charger | gini | 0.4984 | 0.4315 | 0.5566 | theil_total | 0.6706 |
| H1_Gini_Theil_15min_res9 | ev_charger | theil_total | 0.6706 | 0.4408 | 0.9414 | gini | 0.4984 |
| H1_Gini_Theil_15min_res9 | composite | gini | 0.1882 | 0.1563 | 0.2202 | theil_total | 0.0673 |
| H1_Gini_Theil_15min_res9 | composite | theil_total | 0.0673 | 0.0479 | 0.0910 | gini | 0.1882 |
| H3_OLS_SEM_SLM_N23_knn4 | acc15_park|OLS_HC3|elderly_share | coef | 3.3187 | -20.4267 | 27.0641 | p_value | 0.7736 |
| H3_OLS_SEM_SLM_N23_knn4 | acc15_park|OLS_HC3|log_pop_density | coef | -0.8733 | -1.6363 | -0.1103 | p_value | 0.0269 |
| H3_OLS_SEM_SLM_N23_knn4 | acc15_park|SEM_ML|elderly_share | coef | 2.0076 | -13.3137 | 17.3288 | p_value | 0.7973 |
| H3_OLS_SEM_SLM_N23_knn4 | acc15_park|SEM_ML|log_pop_density | coef | -0.8801 | -1.3103 | -0.4500 | p_value | 0.0001 |
| H3_OLS_SEM_SLM_N23_knn4 | acc15_park|SLM_ML|elderly_share | coef | 3.2136 | -12.3661 | 18.7934 | p_value | 0.6860 |
| H3_OLS_SEM_SLM_N23_knn4 | acc15_park|SLM_ML|log_pop_density | coef | -0.8933 | -1.3387 | -0.4479 | p_value | 0.0001 |
| H3_OLS_SEM_SLM_N23_knn4 | acc15_composite|OLS_HC3|elderly_share | coef | 1.9488 | -0.6967 | 4.5943 | p_value | 0.1400 |
| H3_OLS_SEM_SLM_N23_knn4 | acc15_composite|OLS_HC3|log_pop_density | coef | -0.0580 | -0.1185 | 0.0025 | p_value | 0.0595 |
| H3_OLS_SEM_SLM_N23_knn4 | acc15_composite|SEM_ML|elderly_share | coef | 1.9366 | 0.4098 | 3.4634 | p_value | 0.0129 |
| H3_OLS_SEM_SLM_N23_knn4 | acc15_composite|SEM_ML|log_pop_density | coef | -0.0560 | -0.0989 | -0.0131 | p_value | 0.0106 |
| H3_OLS_SEM_SLM_N23_knn4 | acc15_composite|SLM_ML|elderly_share | coef | 1.9635 | 0.4482 | 3.4789 | p_value | 0.0111 |
| H3_OLS_SEM_SLM_N23_knn4 | acc15_composite|SLM_ML|log_pop_density | coef | -0.0562 | -0.0976 | -0.0148 | p_value | 0.0078 |
| ML_SpatialBlock_CV | naive_prior | oof_roc_auc | 0.2171 |  |  | oof_pr_auc | 0.1359 |
| ML_SpatialBlock_CV | naive_prior | oof_brier | 0.1756 |  |  | oof_top20_hit_rate | 0.1071 |
| ML_SpatialBlock_CV | logistic_regression | oof_roc_auc | 0.8527 |  |  | oof_pr_auc | 0.5320 |
| ML_SpatialBlock_CV | logistic_regression | oof_brier | 0.1160 |  |  | oof_top20_hit_rate | 0.5714 |
| ML_SpatialBlock_CV | random_forest | oof_roc_auc | 0.8581 |  |  | oof_pr_auc | 0.5780 |
| ML_SpatialBlock_CV | random_forest | oof_brier | 0.1132 |  |  | oof_top20_hit_rate | 0.5536 |
