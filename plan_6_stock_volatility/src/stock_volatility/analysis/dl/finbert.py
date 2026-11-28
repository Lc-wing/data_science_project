"""Route B FinBERT (110M) Event-Anchored Baseline Risk Feature Extractor (Colab GPU-Gated)."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd
import torch

from stock_volatility.analysis.dl.qlora_llm import (
    require_cuda_and_download_flag,
    resolve_hf_model_revision,
)

DEFAULT_FINBERT_MODEL: str = "ProsusAI/finbert"


def extract_finbert_event_scores(
    *,
    event_splits: dict[str, pd.DataFrame],
    model_id: str = DEFAULT_FINBERT_MODEL,
    allow_weight_download: bool = False,
    max_seq_len: int = 256,
    batch_size: int = 64,
) -> dict[str, Any]:
    """Score pre-close event text with frozen `ProsusAI/finbert` on Val & Test event days.

    Produces `finbert_neg_logit = log((p_neg + eps) / (1 - p_neg + eps))` (higher = higher
    downside sentiment risk) alongside `finbert_p_neg` and `finbert_polarity = p_pos - p_neg`.
    Non-event trading days receive deterministic neutral `0.0` during Stage-2 fusion.
    """
    require_cuda_and_download_flag(
        allow_weight_download=allow_weight_download, component=f"finbert:{model_id}"
    )

    from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

    t0 = time.perf_counter()
    device = torch.device("cuda")
    hf_cfg = AutoConfig.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id).to(device)
    model.eval()

    # Map label indices robustly from id2label (ProsusAI/finbert: 0=positive, 1=negative, 2=neutral)
    id2label = {int(k): str(v).lower() for k, v in hf_cfg.id2label.items()}
    pos_idx = next(i for i, lbl in id2label.items() if "pos" in lbl)
    neg_idx = next(i for i, lbl in id2label.items() if "neg" in lbl)

    def _score_df(df: pd.DataFrame, mdl: Any) -> pd.DataFrame:
        texts = df["text_concat_capped_5"].fillna("").astype(str).tolist()
        probs_all: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(texts), batch_size):
                batch = texts[start : start + batch_size]
                enc = tokenizer(
                    batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=max_seq_len,
                ).to(device)
                logits = mdl(**enc).logits.float()
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
                probs_all.append(probs)
        mat = np.concatenate(probs_all, axis=0) if probs_all else np.zeros((0, 3))
        p_pos = mat[:, pos_idx] if len(mat) else np.array([], dtype=float)
        p_neg = mat[:, neg_idx] if len(mat) else np.array([], dtype=float)
        eps = 1e-5
        neg_logit = np.log(np.clip(p_neg, eps, 1.0 - eps) / np.clip(1.0 - p_neg, eps, 1.0 - eps))

        out = df[["ticker", "date"]].copy()
        out["finbert_p_neg"] = p_neg
        out["finbert_p_pos"] = p_pos
        out["finbert_polarity"] = p_pos - p_neg
        out["finbert_neg_logit"] = neg_logit
        return out

    val_scores = _score_df(event_splits["val"], model)
    test_scores = _score_df(event_splits["test"], model)
    param_count = int(sum(p.numel() for p in model.parameters()))
    resolved_revision = resolve_hf_model_revision(model_id, hf_cfg, model)
    elapsed = round(float(time.perf_counter() - t0), 2)
    print(
        f"[FinBERT] Scored Val={len(val_scores)}, Test={len(test_scores)} in {elapsed:.1f}s",
        flush=True,
    )
    del model
    torch.cuda.empty_cache()

    return {
        "model_id": model_id,
        "revision": resolved_revision,
        "param_count": param_count,
        "val_event_days": int(len(val_scores)),
        "test_event_days": int(len(test_scores)),
        "elapsed_sec": elapsed,
        "val_scores_df": val_scores,
        "test_scores_df": test_scores,
    }
