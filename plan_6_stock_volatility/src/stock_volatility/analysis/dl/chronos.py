"""Route B Time-Series Foundation Models: amazon/chronos-2 (120M) & chronos-t5-large (710M)."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd
import torch

from stock_volatility import config
from stock_volatility.analysis.dl.patchtst import nominal_interval_coverage
from stock_volatility.analysis.dl.qlora_llm import require_cuda_and_download_flag
from stock_volatility.evaluation import regression_metrics

DEFAULT_CHRONOS_MODELS: tuple[str, ...] = (
    "amazon/chronos-2",
    "amazon/chronos-t5-large",
)


def run_chronos_volatility_forecasts(
    *,
    panel_df: pd.DataFrame,
    model_id: str = "amazon/chronos-t5-large",
    allow_weight_download: bool = False,
    context_length: int = 128,
    batch_size: int = 64,
    num_samples: int = 20,
) -> dict[str, Any]:
    """Generate strictly causal 5-day forward RV forecasts via Chronos on 2025 Holdout Test.

    At every scorable 2025 test date `t`, feeds strictly past daily Parkinson volatility
    `rv_d[t - context_length + 1 : t + 1]` into Chronos to forecast the next 5 trading days
    `t+1..t+5`, aggregating the 5-day RMS into annualized 5-day RV + nominal `[q10, q90]` bounds.
    """
    require_cuda_and_download_flag(
        allow_weight_download=allow_weight_download, component=f"chronos:{model_id}"
    )

    t0 = time.perf_counter()
    use_v2 = "chronos-2" in model_id.lower()
    if use_v2:
        from chronos import Chronos2Pipeline

        pipeline: Any = Chronos2Pipeline.from_pretrained(model_id, device_map="cuda")
    else:
        from chronos import ChronosPipeline

        try:
            pipeline = ChronosPipeline.from_pretrained(
                model_id,
                device_map="cuda",
                dtype=torch.bfloat16,
            )
        except TypeError:
            pipeline = ChronosPipeline.from_pretrained(
                model_id,
                device_map="cuda",
                torch_dtype=torch.bfloat16,
            )

    # Collect strictly <= t context windows for every scorable 2025 test (ticker, date)
    contexts: list[np.ndarray] = []
    meta_rows: list[dict[str, Any]] = []
    for ticker, grp in panel_df.groupby("ticker", sort=True):
        g = grp.sort_values("date", kind="mergesort").reset_index(drop=True)
        rv_d = g["rv_d"].to_numpy(dtype=np.float32)
        splits = g["split"].to_numpy()
        has_f = g["has_full_features"].to_numpy(dtype=bool)
        has_t = g["has_full_target"].to_numpy(dtype=bool)
        dates = g["date"].astype(str).to_numpy()
        rv_fwd = g["rv_5d_fwd"].to_numpy(dtype=np.float32)

        for idx in range(len(g)):
            if splits[idx] != "test" or not (has_f[idx] and has_t[idx]):
                continue
            lo = max(0, idx - context_length + 1)
            seq = rv_d[lo : idx + 1]
            seq = np.where(np.isfinite(seq), seq, float(np.nanmedian(seq)))
            contexts.append(seq)
            meta_rows.append(
                {"ticker": str(ticker), "date": str(dates[idx]), "rv_5d_fwd": float(rv_fwd[idx])}
            )

    q10_all, q50_all, q90_all = [], [], []
    for start in range(0, len(contexts), batch_size):
        batch_ctx = [
            torch.tensor(c, dtype=torch.float32) for c in contexts[start : start + batch_size]
        ]
        if use_v2:
            # Chronos2Pipeline.predict_quantiles returns list of (1, H, Q) or (B, 1, H, Q) tensors
            q_out, _ = pipeline.predict_quantiles(
                batch_ctx,
                prediction_length=config.HORIZON_DAYS,
                quantile_levels=[0.10, 0.50, 0.90],
            )
            arr = (
                q_out.detach().cpu().numpy()
                if isinstance(q_out, torch.Tensor)
                else np.stack([q.detach().cpu().numpy() for q in q_out], axis=0)
            )
            arr = arr.reshape(len(batch_ctx), config.HORIZON_DAYS, 3)
            # RMS across 5 forecast steps for each quantile (shape: [B, 5, 3])
            rms = np.sqrt(np.mean(np.maximum(arr, 1e-4) ** 2, axis=1))
            q10_all.append(rms[:, 0])
            q50_all.append(rms[:, 1])
            q90_all.append(rms[:, 2])
        else:
            samples = pipeline.predict(
                batch_ctx,
                prediction_length=config.HORIZON_DAYS,
                num_samples=num_samples,
            )
            # samples shape: [B, num_samples, 5] -> 5-day RMS per sample -> empirical quantiles
            samp_np = np.maximum(samples.detach().cpu().numpy(), 1e-4)
            rms_samp = np.sqrt(np.mean(samp_np**2, axis=-1))
            q10_all.append(np.quantile(rms_samp, 0.10, axis=-1))
            q50_all.append(np.quantile(rms_samp, 0.50, axis=-1))
            q90_all.append(np.quantile(rms_samp, 0.90, axis=-1))

    q10 = np.concatenate(q10_all, axis=0)
    q50 = np.concatenate(q50_all, axis=0)
    q90 = np.concatenate(q90_all, axis=0)

    pred_df = pd.DataFrame(meta_rows)
    pred_df["rv_hat"] = np.maximum(q50, 1e-4)
    pred_df["q10_rv"] = np.maximum(q10, 1e-4)
    pred_df["q50_rv"] = np.maximum(q50, 1e-4)
    pred_df["q90_rv"] = np.maximum(q90, 1e-4)

    rm = regression_metrics(pred_df["rv_5d_fwd"].to_numpy(), pred_df["rv_hat"].to_numpy())
    cov80 = nominal_interval_coverage(
        pred_df["rv_5d_fwd"].to_numpy(),
        pred_df["q10_rv"].to_numpy(),
        pred_df["q90_rv"].to_numpy(),
    )
    elapsed = round(float(time.perf_counter() - t0), 2)
    print(
        f"[Chronos] {model_id} | rows={len(pred_df)} | QLIKE={rm['qlike']:.4f} "
        f"| 80% cov={100.0 * cov80:.1f}% | elapsed={elapsed:.1f}s",
        flush=True,
    )
    del pipeline
    torch.cuda.empty_cache()

    return {
        "model_id": model_id,
        "context_length": int(context_length),
        "test_rows": int(len(pred_df)),
        "test_metrics": rm,
        "test_nominal_80_coverage": float(cov80),
        "elapsed_sec": elapsed,
        "test_pred_df": pred_df,
    }
