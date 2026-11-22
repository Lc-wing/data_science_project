"""From-scratch Phase B networks.

Architectures are the constants in ``spec``. Parameter counts are checked
against the plan's caps and are not adjusted from a validation score.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GATv2Conv, SAGEConv, TransformerConv

from macao_equity.phase_b.spec import (
    DROPOUT,
    GATV2_CONCAT,
    GATV2_HEADS,
    GATV2_HIDDEN,
    SAGE_HIDDEN,
    TRANSFORMER_CONCAT,
    TRANSFORMER_HEADS,
    TRANSFORMER_HIDDEN,
)


def parameter_count(module: nn.Module) -> int:
    return int(sum(param.numel() for param in module.parameters()))


def effective_parameter_count(module: nn.Module) -> int:
    """Logical parameter count, unpacking bitsandbytes 4-bit packed tensors."""
    total = 0
    for param in module.parameters():
        qstate = getattr(param, "quant_state", None)
        if qstate is not None and getattr(qstate, "shape", None) is not None:
            total += int(math.prod(int(dim) for dim in qstate.shape))
        elif type(param).__name__ == "Params4bit" and param.dtype == torch.uint8:
            total += int(param.numel()) * 2
        else:
            total += int(param.numel())
    return int(total)


def trainable_parameter_count(module: nn.Module) -> int:
    """Parameters that receive gradients. Frozen base weights are excluded."""
    return int(sum(param.numel() for param in module.parameters() if param.requires_grad))


class _AttentionBlock(nn.Module):
    """Input projection, two residual attention layers with edge weights, and a logit head."""

    def __init__(
        self,
        in_dim: int,
        hidden: int,
        heads: int,
        *,
        concat: bool,
        conv_cls: type[nn.Module],
    ) -> None:
        super().__init__()
        width = hidden * heads if concat else hidden
        self.input_proj = nn.Linear(in_dim, width)
        self.norm0 = nn.LayerNorm(width)
        conv_kwargs: dict[str, object] = {
            "heads": heads,
            "concat": concat,
            "dropout": DROPOUT,
            "edge_dim": 1,
        }
        if conv_cls is GATv2Conv:
            conv_kwargs["fill_value"] = 1.0
        self.conv1 = conv_cls(width, hidden, **conv_kwargs)
        self.norm1 = nn.LayerNorm(width)
        self.conv2 = conv_cls(width, hidden, **conv_kwargs)
        self.norm2 = nn.LayerNorm(width)
        self.drop = nn.Dropout(DROPOUT)
        self.head = nn.Linear(width, 1)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
        *,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if edge_attr is None:
            edge_attr = x.new_ones((edge_index.shape[1], 1))
        elif edge_attr.ndim == 1:
            edge_attr = edge_attr.unsqueeze(-1)
        hidden = self.drop(F.elu(self.norm0(self.input_proj(x))))
        hidden = F.elu(self.norm1(self.conv1(hidden, edge_index, edge_attr=edge_attr) + hidden))
        hidden = self.drop(hidden)
        if return_attention:
            conv_out, (att_edge, alpha) = self.conv2(
                hidden,
                edge_index,
                edge_attr=edge_attr,
                return_attention_weights=True,
            )
        else:
            conv_out = self.conv2(hidden, edge_index, edge_attr=edge_attr)
            att_edge, alpha = None, None
        hidden = F.elu(self.norm2(conv_out + hidden))
        logits = self.head(self.drop(hidden)).view(-1)
        if return_attention:
            if att_edge is None or alpha is None:
                raise RuntimeError("attention weights were not returned")
            return logits, att_edge, alpha
        return logits


class GATv2Classifier(_AttentionBlock):
    def __init__(self, in_dim: int) -> None:
        super().__init__(
            in_dim,
            GATV2_HIDDEN,
            GATV2_HEADS,
            concat=GATV2_CONCAT,
            conv_cls=GATv2Conv,
        )


class GraphTransformerClassifier(_AttentionBlock):
    def __init__(self, in_dim: int) -> None:
        super().__init__(
            in_dim,
            TRANSFORMER_HIDDEN,
            TRANSFORMER_HEADS,
            concat=TRANSFORMER_CONCAT,
            conv_cls=TransformerConv,
        )


class GraphSAGEClassifier(nn.Module):
    """Two-layer SAGE ablation near 5k parameters (unweighted)."""

    def __init__(self, in_dim: int) -> None:
        super().__init__()
        hidden = SAGE_HIDDEN
        self.conv1 = SAGEConv(in_dim, hidden)
        self.norm1 = nn.LayerNorm(hidden)
        self.conv2 = SAGEConv(hidden, hidden)
        self.norm2 = nn.LayerNorm(hidden)
        self.drop = nn.Dropout(DROPOUT)
        self.head = nn.Linear(hidden, 1)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
        *,
        return_attention: bool = False,
    ) -> torch.Tensor:
        del edge_attr
        if return_attention:
            raise ValueError("GraphSAGE has no attention weights")
        hidden = self.drop(torch.relu(self.norm1(self.conv1(x, edge_index))))
        residual = self.conv2(hidden, edge_index)
        hidden = self.drop(torch.relu(self.norm2(residual + hidden)))
        return self.head(hidden).view(-1)


def build_model(name: str, in_dim: int) -> nn.Module:
    if name == "gatv2":
        return GATv2Classifier(in_dim)
    if name == "graph_transformer":
        return GraphTransformerClassifier(in_dim)
    if name == "graphsage_5k":
        return GraphSAGEClassifier(in_dim)
    raise ValueError(f"unknown model: {name}")
