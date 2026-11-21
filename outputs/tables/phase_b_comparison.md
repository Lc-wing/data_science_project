# Phase B comparison

Primary metrics are ROC-AUC, PR-AUC, and the top-20% hit rate.
Empty Route B cells are blocked runs, not omitted models.
Scores are associations under the frozen spatial blocks.
Phase A ± is the standard deviation across the five folds for seed 42.
Route A ± is the standard deviation across the ten seeds of that fold-mean.
Those two dispersions are not interchangeable.

| 模型 | 协议 | ROC-AUC | PR-AUC | 前 20% 命中率 | 状态 |
| --- | --- | --- | --- | --- | --- |
| naive_prior | spatial_block_cv | 0.500 ± 0.000 | 0.304 ± 0.195 | 0.190 ± 0.167 | copied_from_phase_a |
| logistic_regression | spatial_block_cv | 0.736 ± 0.094 | 0.520 ± 0.114 | 0.388 ± 0.189 | copied_from_phase_a |
| random_forest | spatial_block_cv | 0.773 ± 0.111 | 0.632 ± 0.109 | 0.502 ± 0.251 | copied_from_phase_a |
| gatv2 | inductive | 0.712 ± 0.016 | 0.532 ± 0.019 | 0.367 ± 0.034 | ran_on_cpu |
| gatv2 | transductive | 0.656 ± 0.024 | 0.460 ± 0.029 | 0.325 ± 0.033 | ran_on_cpu |
| graph_transformer | inductive | 0.740 ± 0.026 | 0.582 ± 0.034 | 0.425 ± 0.046 | ran_on_cpu |
| graph_transformer | transductive | 0.729 ± 0.013 | 0.540 ± 0.034 | 0.412 ± 0.038 | ran_on_cpu |
| graphsage_5k | inductive | 0.664 ± 0.043 | 0.466 ± 0.050 | 0.333 ± 0.066 | ran_on_cpu |
| graphsage_5k | transductive | 0.663 ± 0.024 | 0.466 ± 0.054 | 0.334 ± 0.040 | ran_on_cpu |
| bge_m3_gatv2 | inductive | 0.718 ± 0.022 | 0.541 ± 0.026 | 0.436 ± 0.041 | ran_on_colab |
| bge_m3_gatv2 | transductive | 0.673 ± 0.012 | 0.478 ± 0.023 | 0.365 ± 0.034 | ran_on_colab |
| qwen3_embedding_4b_frozen_gatv2 | inductive | 0.708 ± 0.016 | 0.494 ± 0.041 | 0.398 ± 0.049 | ran_on_colab |
| qwen3_embedding_4b_frozen_gatv2 | transductive | 0.667 ± 0.030 | 0.442 ± 0.035 | 0.320 ± 0.037 | ran_on_colab |
| qwen3_embedding_4b_lora_gatv2 | inductive | 0.708 ± 0.011 | 0.504 ± 0.022 | 0.403 ± 0.023 | ran_on_colab |
| qwen3_embedding_4b_lora_gatv2 | transductive | 0.718 ± 0.014 | 0.502 ± 0.010 | 0.425 ± 0.029 | ran_on_colab |
