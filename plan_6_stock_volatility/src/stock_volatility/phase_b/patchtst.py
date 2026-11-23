"""Route A Multivariate PatchTST: Ticker/Sector Embeddings, Pinball Quantiles & 1M/15M Ablation."""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from stock_volatility import config
from stock_volatility.evaluation import classification_metrics, qlike_loss_array, regression_metrics
from stock_volatility.phase_b.data_contract import PHASE_B_CONT_FEATURES

QUANTILES: tuple[float, float, float] = (0.10, 0.50, 0.90)


@dataclass(frozen=True)
class PatchTSTConfig:
    """Architecture configuration for Multivariate PatchTST volatility & tail-risk forecaster."""

    name: str
    lookback: int = 60
    n_features: int = len(PHASE_B_CONT_FEATURES)
    patch_len: int = 5
    patch_stride: int = 5
    n_tickers: int = 60
    n_sectors: int = 5
    d_model: int = 192
    n_heads: int = 4
    n_layers: int = 3
    d_ff: int = 512
    dropout: float = 0.15

    def with_overrides(self, **kwargs: Any) -> PatchTSTConfig:
        """Return a shallow copy with selected hyperparameters overridden."""
        return replace(self, **kwargs)


# Capacity ablation presets (~1.1M vs ~14.7M parameters per Plan 6 Route A)
PATCHTST_1M_CONFIG = PatchTSTConfig(
    name="patchtst_1m",
    d_model=192,
    n_heads=4,
    n_layers=3,
    d_ff=512,
    dropout=0.15,
)

PATCHTST_15M_CONFIG = PatchTSTConfig(
    name="patchtst_15m",
    d_model=448,
    n_heads=8,
    n_layers=6,
    d_ff=1792,
    dropout=0.15,
)


class MultivariatePatchTST(nn.Module):
    """PatchTST encoder with ticker/sector embeddings and non-crossing q10/q50/q90 + tail heads."""

    def __init__(self, cfg: PatchTSTConfig) -> None:
        super().__init__()
        if (cfg.lookback - cfg.patch_len) % cfg.patch_stride != 0:
            raise ValueError(
                f"lookback={cfg.lookback} incompatible with patch_len={cfg.patch_len} "
                f"and stride={cfg.patch_stride}"
            )
        self.cfg = cfg
        self.n_patches = (cfg.lookback - cfg.patch_len) // cfg.patch_stride + 1
        patch_dim = cfg.patch_len * cfg.n_features

        self.patch_proj = nn.Linear(patch_dim, cfg.d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, self.n_patches, cfg.d_model))
        self.ticker_emb = nn.Embedding(cfg.n_tickers, cfg.d_model)
        self.sector_emb = nn.Embedding(cfg.n_sectors, cfg.d_model)
        self.in_drop = nn.Dropout(cfg.dropout)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.d_ff,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            enc_layer,
            num_layers=cfg.n_layers,
            enable_nested_tensor=False,
        )
        self.final_norm = nn.LayerNorm(cfg.d_model)

        # Dual head: point log-RV, non-crossing (q50, delta_lo, delta_hi), and tail logit
        self.head = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model // 2),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model // 2, 5),
        )
        nn.init.normal_(self.pos_emb, mean=0.0, std=0.02)

    def forward(
        self,
        x_seq: torch.Tensor,
        ticker_id: torch.Tensor,
        sector_id: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Forward pass returning point log-RV, ordered (q10, q50, q90), and tail-risk logit."""
        # Unfold time axis into patches: (B, n_patches, patch_len * n_features)
        patches = x_seq.unfold(dimension=1, size=self.cfg.patch_len, step=self.cfg.patch_stride)
        patches = patches.contiguous().view(x_seq.size(0), self.n_patches, -1)

        cond = self.ticker_emb(ticker_id) + self.sector_emb(sector_id)
        tokens = self.patch_proj(patches) + self.pos_emb + cond.unsqueeze(1)
        tokens = self.in_drop(tokens)
        encoded = self.final_norm(self.encoder(tokens))
        pooled = encoded.mean(dim=1)

        raw = self.head(pooled)
        point = raw[:, 0]
        q50 = raw[:, 1]
        # Enforce non-crossing quantiles q10 <= q50 <= q90 via softplus deltas
        q10 = q50 - F.softplus(raw[:, 2])
        q90 = q50 + F.softplus(raw[:, 3])
        quantiles = torch.stack([q10, q50, q90], dim=-1)
        tail_logit = raw[:, 4]
        return {"point": point, "quantiles": quantiles, "tail_logit": tail_logit}


def count_trainable_parameters(model: nn.Module) -> int:
    """Return total trainable parameter count of a PyTorch module."""
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def pinball_loss(
    pred_quantiles: torch.Tensor,
    target: torch.Tensor,
    quantiles: tuple[float, ...] = QUANTILES,
) -> torch.Tensor:
    """Compute mean quantile Pinball loss across `(q10, q50, q90)`."""
    q_t = torch.tensor(quantiles, dtype=pred_quantiles.dtype, device=pred_quantiles.device)
    err = target.unsqueeze(-1) - pred_quantiles
    loss = torch.maximum(q_t * err, (q_t - 1.0) * err)
    return loss.mean()


def nominal_interval_coverage(
    y_true: np.ndarray,
    q_low: np.ndarray,
    q_high: np.ndarray,
) -> float:
    """Compute empirical coverage rate for the nominal 80% `[q10, q90]` interval."""
    y = np.asarray(y_true, dtype=float)
    lo = np.asarray(q_low, dtype=float)
    hi = np.asarray(q_high, dtype=float)
    return float(np.mean((y >= lo) & (y <= hi)))


def _seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _predict_split_tensors(
    model: MultivariatePatchTST,
    x_seq: torch.Tensor,
    ticker_id: torch.Tensor,
    sector_id: torch.Tensor,
    *,
    batch_size: int,
    use_bf16: bool,
) -> dict[str, np.ndarray]:
    """Run eval-mode inference directly on device-resident tensors (zero DataLoader overhead)."""
    model.eval()
    n = x_seq.size(0)
    pts, qs, tlogs = [], [], []
    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            xb = x_seq[start:end]
            tb = ticker_id[start:end]
            sb = sector_id[start:end]
            if use_bf16:
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    out = model(xb, tb, sb)
            else:
                out = model(xb, tb, sb)
            pts.append(out["point"].float().cpu().numpy())
            qs.append(out["quantiles"].float().cpu().numpy())
            tlogs.append(out["tail_logit"].float().cpu().numpy())
    return {
        "point": np.concatenate(pts, axis=0),
        "quantiles": np.concatenate(qs, axis=0),
        "tail_logit": np.concatenate(tlogs, axis=0),
    }


def train_and_predict_patchtst(
    *,
    windows: dict[str, dict[str, np.ndarray]],
    cfg: PatchTSTConfig,
    seeds: Sequence[int] = (42, 43, 44, 45, 46),
    epochs: int = 25,
    batch_size: int = 256,
    lr: float = 5e-4,
    weight_decay: float = 1e-2,
    patience: int = 5,
    device: str | None = None,
) -> dict[str, Any]:
    """Train multi-seed PatchTST strictly on Train, early-stop on 2024 Val QLIKE, score Test."""
    t_all = time.perf_counter()
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    use_bf16 = dev.type == "cuda" and torch.cuda.is_bf16_supported()
    tr = windows["train"]
    va = windows["val"]
    te = windows["test"]

    # Keep compact window tensors (~335 MB total) resident on device to eliminate DataLoader copy
    x_tr = torch.from_numpy(tr["x_seq"]).to(dev)
    t_tr = torch.from_numpy(tr["ticker_id"]).to(dev)
    s_tr = torch.from_numpy(tr["sector_id"]).to(dev)
    y_log_tr = torch.from_numpy(tr["y_log_rv"]).to(dev)
    y_tail_tr = torch.from_numpy(tr["y_tail"].astype(np.float32)).to(dev)

    x_va = torch.from_numpy(va["x_seq"]).to(dev)
    t_va = torch.from_numpy(va["ticker_id"]).to(dev)
    s_va = torch.from_numpy(va["sector_id"]).to(dev)

    x_te = torch.from_numpy(te["x_seq"]).to(dev)
    t_te = torch.from_numpy(te["ticker_id"]).to(dev)
    s_te = torch.from_numpy(te["sector_id"]).to(dev)

    n_tr = x_tr.size(0)
    eval_bs = max(batch_size * 2, 512)

    pos_cnt = max(float(np.sum(tr["y_tail"] == 1)), 1.0)
    neg_cnt = max(float(np.sum(tr["y_tail"] == 0)), 1.0)
    pos_weight = torch.tensor([neg_cnt / pos_cnt], dtype=torch.float32, device=dev)

    seed_summaries: list[dict[str, Any]] = []
    val_rv_list, test_rv_list = [], []
    val_q_list, test_q_list = [], []
    val_logit_list, test_logit_list = [], []
    param_count = 0

    for seed_i, seed in enumerate(seeds, start=1):
        t_seed = time.perf_counter()
        _seed_everything(int(seed))
        model = MultivariatePatchTST(cfg).to(dev)
        param_count = count_trainable_parameters(model)

        g_gen = torch.Generator(device=dev)
        g_gen.manual_seed(int(seed))

        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))

        best_val_qlike = float("inf")
        best_state: dict[str, torch.Tensor] | None = None
        best_epoch = 0
        no_improve = 0

        for ep in range(1, epochs + 1):
            model.train()
            perm = torch.randperm(n_tr, generator=g_gen, device=dev)
            resid_sum = 0.0
            for start in range(0, n_tr, batch_size):
                idx = perm[start : start + batch_size]
                xb = x_tr[idx]
                tb = t_tr[idx]
                sb = s_tr[idx]
                y_log_b = y_log_tr[idx]
                y_tail_b = y_tail_tr[idx]

                opt.zero_grad(set_to_none=True)
                if use_bf16:
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        out = model(xb, tb, sb)
                else:
                    out = model(xb, tb, sb)
                pt_f = out["point"].float()
                q_f = out["quantiles"].float()
                tl_f = out["tail_logit"].float()

                loss_reg = F.mse_loss(pt_f, y_log_b)
                loss_q = pinball_loss(q_f, y_log_b)
                loss_cls = F.binary_cross_entropy_with_logits(tl_f, y_tail_b, pos_weight=pos_weight)
                loss = loss_reg + loss_q + 0.35 * loss_cls
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                opt.step()

                with torch.no_grad():
                    resid_sum += float(torch.exp(torch.clamp(y_log_b - pt_f, -5.0, 5.0)).sum())
            sched.step()

            # Early stop strictly on 2024 Val QLIKE using Train Duan smearing
            smear_ep = resid_sum / max(n_tr, 1)
            va_pred_ep = _predict_split_tensors(
                model, x_va, t_va, s_va, batch_size=eval_bs, use_bf16=use_bf16
            )
            va_rv_hat = np.maximum(np.exp(va_pred_ep["point"]) * smear_ep - config.RV_EPS, 1e-4)
            va_qlike = float(np.mean(qlike_loss_array(va["rv_5d_fwd"], va_rv_hat)))

            if va_qlike < best_val_qlike - 1e-6:
                best_val_qlike = va_qlike
                best_epoch = ep
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    break

        if best_state is not None:
            model.load_state_dict(best_state)

        tr_out = _predict_split_tensors(
            model, x_tr, t_tr, s_tr, batch_size=eval_bs, use_bf16=use_bf16
        )
        smearing = float(np.mean(np.exp(np.clip(tr["y_log_rv"] - tr_out["point"], -5, 5))))
        va_out = _predict_split_tensors(
            model, x_va, t_va, s_va, batch_size=eval_bs, use_bf16=use_bf16
        )
        te_out = _predict_split_tensors(
            model, x_te, t_te, s_te, batch_size=eval_bs, use_bf16=use_bf16
        )

        va_rv_hat = np.maximum(np.exp(va_out["point"]) * smearing - config.RV_EPS, 1e-4)
        te_rv_hat = np.maximum(np.exp(te_out["point"]) * smearing - config.RV_EPS, 1e-4)
        va_q_rv = np.maximum(np.exp(va_out["quantiles"]) - config.RV_EPS, 1e-4)
        te_q_rv = np.maximum(np.exp(te_out["quantiles"]) - config.RV_EPS, 1e-4)

        val_rv_list.append(va_rv_hat)
        test_rv_list.append(te_rv_hat)
        val_q_list.append(va_q_rv)
        test_q_list.append(te_q_rv)
        val_logit_list.append(va_out["tail_logit"])
        test_logit_list.append(te_out["tail_logit"])

        rm = regression_metrics(te["rv_5d_fwd"], te_rv_hat)
        cov80 = nominal_interval_coverage(te["rv_5d_fwd"], te_q_rv[:, 0], te_q_rv[:, 2])
        seed_summaries.append(
            {
                "seed": int(seed),
                "best_epoch": int(best_epoch),
                "best_val_qlike": float(best_val_qlike),
                "duan_smearing": float(smearing),
                "test_qlike": float(rm["qlike"]),
                "test_rmse": float(rm["rmse"]),
                "test_nominal_80_coverage": float(cov80),
            }
        )
        print(
            f"[PatchTST {param_count / 1e6:.2f}M] seed {seed_i}/{len(seeds)} (s={seed}) "
            f"| best_ep={best_epoch}/{epochs} | val_QLIKE={best_val_qlike:.4f} "
            f"| test_QLIKE={rm['qlike']:.4f} | 80%cov={100.0 * cov80:.1f}% "
            f"| {time.perf_counter() - t_seed:.1f}s",
            flush=True,
        )

    del x_tr, t_tr, s_tr, y_log_tr, y_tail_tr, x_va, t_va, s_va, x_te, t_te, s_te
    if dev.type == "cuda":
        torch.cuda.empty_cache()

    print(
        f"[PatchTST {param_count / 1e6:.2f}M] Ensemble finished in "
        f"{time.perf_counter() - t_all:.1f}s (bf16={use_bf16})",
        flush=True,
    )

    # Multi-seed ensemble predictions
    ens_val_rv = np.mean(val_rv_list, axis=0)
    ens_test_rv = np.mean(test_rv_list, axis=0)
    ens_val_q = np.mean(val_q_list, axis=0)
    ens_test_q = np.mean(test_q_list, axis=0)
    ens_val_logit = np.mean(val_logit_list, axis=0)
    ens_test_logit = np.mean(test_logit_list, axis=0)

    # Fit 1-parameter Temperature Scaling strictly on 2024 Val logits
    from stock_volatility.phase_b.fusion_and_eval import (
        apply_temperature_scaling,
        fit_temperature_scaler,
    )

    temp = fit_temperature_scaler(ens_val_logit, va["y_tail"])
    ens_val_prob = apply_temperature_scaling(ens_val_logit, temp)
    ens_test_prob = apply_temperature_scaling(ens_test_logit, temp)

    val_pred_df = pd.DataFrame(
        {
            "ticker": va["tickers"],
            "date": va["dates"],
            "rv_hat": ens_val_rv,
            "q10_rv": ens_val_q[:, 0],
            "q50_rv": ens_val_q[:, 1],
            "q90_rv": ens_val_q[:, 2],
            "tail_logit": ens_val_logit,
            "p_tail": ens_val_prob,
        }
    )
    test_pred_df = pd.DataFrame(
        {
            "ticker": te["tickers"],
            "date": te["dates"],
            "rv_hat": ens_test_rv,
            "q10_rv": ens_test_q[:, 0],
            "q50_rv": ens_test_q[:, 1],
            "q90_rv": ens_test_q[:, 2],
            "tail_logit": ens_test_logit,
            "p_tail": ens_test_prob,
        }
    )

    ens_rm = regression_metrics(te["rv_5d_fwd"], ens_test_rv)
    ens_cm = classification_metrics(te["y_tail"], ens_test_prob)
    ens_cov80 = nominal_interval_coverage(te["rv_5d_fwd"], ens_test_q[:, 0], ens_test_q[:, 2])

    return {
        "config": asdict(cfg),
        "param_count": param_count,
        "seeds": [int(s) for s in seeds],
        "val_temperature": float(temp),
        "seed_summaries": seed_summaries,
        "test_metrics_ensemble": {**ens_rm, **ens_cm},
        "test_nominal_80_coverage": float(ens_cov80),
        "val_pred_df": val_pred_df,
        "test_pred_df": test_pred_df,
    }
