# Table 2: 2025 Holdout Deep Time-Series & Event-Anchored LLM Ablation (H3)

| Model | 2025 Full QLIKE | Event-Day QLIKE | Tail PR-AUC (Event) | ECE | HLN-DM Holm p (Q / B vs LGBM) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Global LightGBM (Price-Only) | `0.2306` | `0.1980` | `0.1231 (0.1076)` | `0.0102` | `Ref.` |
| PatchTST (1M, 5-seed; 80% cov=63.1%) | `0.3142` | `0.2800` | `0.0964 (0.0887)` | `0.4407` | `<0.0001 / <0.0001` |
| PatchTST (15M, 5-seed; 80% cov=68.3%) | `0.2819` | `0.2473` | `0.1129 (0.1027)` | `0.4349` | `<0.001 / <0.0001` |
| chronos-2 (Zero-Shot; 80% cov=89.2%) | `0.3505` | `0.3181` | `—` | `—` | `0.006 / —` |
| chronos-t5-large (Zero-Shot; 80% cov=22.3%) | `0.5848` | `0.5448` | `—` | `—` | `0.004 / —` |
| LightGBM + FinBERT (110M) | `0.2251` | `0.2005` | `0.1249 (0.1398)` | `0.0101` | `0.020 / 0.928` |
| LightGBM + gemma-4-12B-it (Zero-Shot) | `0.2252` | `0.2009` | `0.1230 (0.1108)` | `0.0102` | `0.020 / 0.681` |
| LightGBM + gemma-4-12B-it (4-bit QLoRA) | `0.2252` | `0.2008` | `0.1224 (0.1076)` | `0.0102` | `0.020 / 0.759` |
