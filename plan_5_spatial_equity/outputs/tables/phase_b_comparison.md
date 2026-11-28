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
| gatv2 | inductive | 0.679 ± 0.019 | 0.500 ± 0.026 | 0.353 ± 0.034 | ran_on_cpu |
| gatv2 | transductive | 0.689 ± 0.013 | 0.490 ± 0.017 | 0.354 ± 0.030 | ran_on_cpu |
| graph_transformer | inductive | 0.705 ± 0.032 | 0.523 ± 0.041 | 0.386 ± 0.053 | ran_on_cpu |
| graph_transformer | transductive | 0.717 ± 0.031 | 0.519 ± 0.040 | 0.407 ± 0.050 | ran_on_cpu |
| graphsage_5k | inductive | 0.652 ± 0.056 | 0.474 ± 0.067 | 0.336 ± 0.066 | ran_on_cpu |
| graphsage_5k | transductive | 0.658 ± 0.020 | 0.497 ± 0.037 | 0.362 ± 0.041 | ran_on_cpu |
| bge_m3_gatv2 | inductive |  |  |  | not_run |
| bge_m3_gatv2 | transductive |  |  |  | not_run |
| bge_m3_graph_transformer | inductive |  |  |  | not_run |
| bge_m3_graph_transformer | transductive |  |  |  | not_run |
| qwen3_embedding_4b_frozen_gatv2 | inductive |  |  |  | not_run |
| qwen3_embedding_4b_frozen_gatv2 | transductive |  |  |  | not_run |
| qwen3_embedding_4b_frozen_graph_transformer | inductive |  |  |  | not_run |
| qwen3_embedding_4b_frozen_graph_transformer | transductive |  |  |  | not_run |
| qwen3_embedding_4b_lora_gatv2 | inductive |  |  |  | not_run |
| qwen3_embedding_4b_lora_gatv2 | transductive |  |  |  | not_run |
| qwen3_embedding_4b_lora_graph_transformer | inductive |  |  |  | not_run |
| qwen3_embedding_4b_lora_graph_transformer | transductive |  |  |  | not_run |
