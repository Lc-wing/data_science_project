# Exploratory tree comparison

These three models are outside the pre-specified plan.
The random-forest row is copied from Phase A and was not refit.
Tuning used only the four training spatial blocks.
Metrics are ROC-AUC, PR-AUC, and the top-20% hit rate.
± is the standard deviation across the five folds for seed 42.

| 模型 | 标签 | ROC-AUC | PR-AUC | 前 20% 命中率 |
| --- | --- | --- | --- | --- |
| random_forest | published_phase_a | 0.773 ± 0.111 | 0.632 ± 0.109 | 0.502 ± 0.251 |
| xgboost | exploratory | 0.741 ± 0.095 | 0.556 ± 0.120 | 0.469 ± 0.294 |
| lightgbm | exploratory | 0.720 ± 0.070 | 0.582 ± 0.107 | 0.420 ± 0.148 |
| hist_gradient_boosting | exploratory | 0.718 ± 0.064 | 0.499 ± 0.143 | 0.384 ± 0.157 |
