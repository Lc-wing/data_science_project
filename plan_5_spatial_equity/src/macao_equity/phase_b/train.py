"""One spatial-block fit.

The scaler, edge-weight normalization, and loss see the loss mask only. The
outer test block is not a stopping set. Early stopping reads the pre-specified
deterministic peninsula inner block (block 2, or block 1 when fold == 2).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.nn import BCEWithLogitsLoss

from macao_equity.phase_b.metrics import score_binary
from macao_equity.phase_b.models import build_model
from macao_equity.phase_b.spec import (
    DROPEDGE_P,
    LEARNING_RATE,
    LEARNING_RATE_MIN,
    SPEC_BUDGET,
    WEIGHT_DECAY,
    EpochBudget,
)
from macao_equity.phase_b.splits import FoldSplit, make_fold_split


@dataclass
class FoldFit:
    """Test-block probabilities aligned to every node. Non-test entries are NaN."""

    fold: int
    protocol: str
    model_name: str
    seed: int
    probabilities: np.ndarray
    metrics: dict[str, float]
    best_epoch: int
    n_train_edges: int
    n_eval_edges: int
    n_loss: int
    n_test: int
    scaler_mean: np.ndarray
    state_dict: dict[str, torch.Tensor]
    attention_edges: np.ndarray | None = None
    attention_weights: np.ndarray | None = None


def _seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def _edge_tensor(src: np.ndarray, dst: np.ndarray) -> torch.Tensor:
    if src.size == 0:
        return torch.zeros((2, 0), dtype=torch.long)
    stacked = np.vstack([src, dst]).astype(np.int64, copy=False)
    return torch.from_numpy(stacked)


def drop_undirected_edges(
    edge_index: torch.Tensor,
    drop_p: float,
    generator: torch.Generator,
    edge_weight: torch.Tensor | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Drop both directions of an undirected pair with probability ``drop_p``."""
    if drop_p <= 0.0 or edge_index.numel() == 0:
        return (edge_index, edge_weight) if edge_weight is not None else edge_index
    src = edge_index[0].tolist()
    dst = edge_index[1].tolist()
    groups: dict[tuple[int, int], list[int]] = {}
    for index, (node_u, node_v) in enumerate(zip(src, dst, strict=True)):
        key = (node_u, node_v) if node_u < node_v else (node_v, node_u)
        groups.setdefault(key, []).append(index)
    keys = sorted(groups)
    draws = torch.rand(len(keys), generator=generator)
    keep: list[int] = []
    for draw, key in zip(draws.tolist(), keys, strict=True):
        if float(draw) >= drop_p:
            keep.extend(groups[key])
    if not keep:
        keep = list(groups[keys[0]])
    kept = sorted(keep)
    if edge_weight is not None:
        return edge_index[:, kept], edge_weight[kept]
    return edge_index[:, kept]


def _adamw_groups(model: nn.Module) -> list[dict[str, object]]:
    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim == 1 or name.endswith(".bias"):
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {"params": decay, "weight_decay": WEIGHT_DECAY},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def _subgraph(
    x: torch.Tensor,
    y: torch.Tensor,
    edge_index: torch.Tensor,
    node_mask: np.ndarray,
    edge_weight: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Keep ``node_mask`` nodes and reindex edges. Used by inductive training."""
    keep = torch.from_numpy(np.asarray(node_mask, dtype=bool))
    nodes = torch.where(keep)[0]
    mapping = torch.full((x.shape[0],), -1, dtype=torch.long)
    mapping[nodes] = torch.arange(nodes.numel())
    src = mapping[edge_index[0]]
    dst = mapping[edge_index[1]]
    valid = (src >= 0) & (dst >= 0)
    if not bool(valid.all()):
        raise RuntimeError("inductive train edges touch a removed node")
    if edge_weight is None:
        weights_out = torch.ones((edge_index.shape[1],), dtype=torch.float32)
    else:
        weights_out = edge_weight
    return x[nodes], y[nodes], torch.stack([src, dst], dim=0), weights_out, nodes


def _restore(model: nn.Module, state: dict[str, torch.Tensor]) -> None:
    model.load_state_dict({key: value.clone() for key, value in state.items()})


def train_one_fold(
    x: np.ndarray,
    y: np.ndarray,
    blocks: np.ndarray,
    src: np.ndarray,
    dst: np.ndarray,
    *,
    model_name: str,
    protocol: str,
    fold: int,
    seed: int,
    budget: EpochBudget = SPEC_BUDGET,
    capture_attention: bool = False,
    edge_weight: np.ndarray | None = None,
) -> FoldFit:
    """Fit one outer fold. Test labels are not in the loss or the stopping rule."""
    split: FoldSplit = make_fold_split(
        blocks,
        src,
        dst,
        fold=fold,
        protocol=protocol,
        edge_weight=edge_weight,
    )
    y_np = np.asarray(y, dtype=np.int64)
    loss_y = y_np[split.loss_mask]
    if int(loss_y.min()) == int(loss_y.max()):
        raise ValueError("loss mask must contain both classes")
    val_y = y_np[split.val_mask]
    if int(val_y.min()) == int(val_y.max()):
        raise ValueError("inner validation block must contain both classes")

    scaler = StandardScaler()
    scaler.fit(np.asarray(x, dtype=np.float64)[split.loss_mask])
    x_scaled = scaler.transform(np.asarray(x, dtype=np.float64))
    if not np.isfinite(x_scaled).all():
        raise ValueError("scaled features contain NaN or Inf")

    _seed_everything(int(seed) + 1009 * (int(fold) + 1) + (0 if protocol == "inductive" else 17))
    x_all = torch.tensor(x_scaled, dtype=torch.float32)
    y_all = torch.tensor(y_np, dtype=torch.float32)
    train_edges = _edge_tensor(split.train_src, split.train_dst)
    eval_edges = _edge_tensor(split.eval_src, split.eval_dst)
    # Normalize walk-time inverse weights strictly on training edges so test
    # edges never influence scaling and self-loop fill_value=1.0 sits at mean.
    weight_scale = (
        float(np.mean(split.train_weight))
        if split.train_weight.size > 0 and float(np.mean(split.train_weight)) > 0.0
        else 1.0
    )
    train_weights = torch.tensor(split.train_weight / weight_scale, dtype=torch.float32)
    eval_weights = torch.tensor(split.eval_weight / weight_scale, dtype=torch.float32)

    if protocol == "inductive":
        train_nodes = ~split.test_mask
        x_fit, y_fit, edges_fit, weights_fit, _node_index = _subgraph(
            x_all,
            y_all,
            train_edges,
            train_nodes,
            edge_weight=train_weights,
        )
        loss_fit = torch.from_numpy(split.loss_mask[train_nodes])
        val_fit = torch.from_numpy(split.val_mask[train_nodes])
    else:
        x_fit, y_fit, edges_fit, weights_fit = x_all, y_all, train_edges, train_weights
        loss_fit = torch.from_numpy(split.loss_mask)
        val_fit = torch.from_numpy(split.val_mask)

    model = build_model(model_name, int(x_fit.shape[1]))
    optimizer = torch.optim.AdamW(_adamw_groups(model), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=budget.max_epochs,
        eta_min=LEARNING_RATE_MIN,
    )
    n_pos = y_fit[loss_fit].sum().clamp(min=1.0)
    n_neg = (loss_fit.sum().to(dtype=n_pos.dtype) - n_pos).clamp(min=1.0)
    criterion = BCEWithLogitsLoss(pos_weight=(n_neg / n_pos).detach().view(1))

    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    best_val = -1.0
    best_epoch = 0
    stale = 0
    for epoch in range(budget.max_epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        generator = torch.Generator()
        generator.manual_seed(int(seed) + 10007 * (int(fold) + 1) + epoch)
        dropped_pair = drop_undirected_edges(
            edges_fit,
            DROPEDGE_P,
            generator,
            edge_weight=weights_fit,
        )
        assert isinstance(dropped_pair, tuple)
        dropped_edges, dropped_weights = dropped_pair
        logits = model(x_fit, dropped_edges, edge_attr=dropped_weights.unsqueeze(-1))
        loss = criterion(logits[loss_fit], y_fit[loss_fit])
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite training loss")
        loss.backward()
        optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(x_fit, edges_fit, edge_attr=weights_fit.unsqueeze(-1))
            val_prob = torch.sigmoid(val_logits[val_fit]).detach().cpu().numpy()
        val_auc = float(roc_auc_score(val_y, val_prob))
        if val_auc > best_val:
            best_val = val_auc
            best_epoch = epoch + 1
            stale = 0
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
        else:
            stale += 1
        if epoch + 1 >= budget.min_epochs and stale >= budget.patience:
            break

    _restore(model, best_state)
    model.eval()
    attention_edges = None
    attention_weights = None
    with torch.no_grad():
        if capture_attention and model_name == "gatv2":
            full_logits, att_edge, alpha = model(
                x_all,
                eval_edges,
                edge_attr=eval_weights.unsqueeze(-1),
                return_attention=True,
            )
            attention_edges = att_edge.detach().cpu().numpy().astype(np.int64)
            weights = alpha.detach().cpu().numpy()
            if weights.ndim == 1:
                weights = weights[:, None]
            attention_weights = weights.mean(axis=1).astype(np.float64)
        else:
            full_logits = model(x_all, eval_edges, edge_attr=eval_weights.unsqueeze(-1))
        test_prob = torch.sigmoid(full_logits).detach().cpu().numpy()

    probabilities = np.full(y_np.shape[0], np.nan, dtype=np.float64)
    test_mask = split.test_mask
    probabilities[test_mask] = test_prob[test_mask]
    metrics = score_binary(y_np[test_mask], probabilities[test_mask])
    return FoldFit(
        fold=int(fold),
        protocol=protocol,
        model_name=model_name,
        seed=int(seed),
        probabilities=probabilities,
        metrics=metrics,
        best_epoch=int(best_epoch),
        n_train_edges=int(split.train_src.shape[0]),
        n_eval_edges=int(split.eval_src.shape[0]),
        n_loss=int(split.n_loss),
        n_test=int(split.n_test),
        scaler_mean=np.asarray(scaler.mean_, dtype=np.float64),
        state_dict=best_state,
        attention_edges=attention_edges,
        attention_weights=attention_weights,
    )
