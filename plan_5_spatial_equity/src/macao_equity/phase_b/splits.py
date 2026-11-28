"""Inductive and transductive edges for one frozen spatial block.

Inductive training removes every test node and every edge that touches a test
node. Inductive evaluation keeps train->train, train->test, and test->test
edges while dropping test->train edges, so a test node aggregates features from
both adjacent training cells and neighboring test cells without sending any
message back into the training subgraph. Transductive training keeps every
edge, including cross-block edges, and the loss still ignores test labels.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from macao_equity.phase_b.spec import PROTOCOLS


@dataclass(frozen=True)
class FoldSplit:
    """Node masks, edge endpoints, and aligned weights for one outer block."""

    fold: int
    protocol: str
    test_mask: np.ndarray
    val_mask: np.ndarray
    loss_mask: np.ndarray
    train_src: np.ndarray
    train_dst: np.ndarray
    eval_src: np.ndarray
    eval_dst: np.ndarray
    train_weight: np.ndarray
    eval_weight: np.ndarray

    @property
    def n_test(self) -> int:
        return int(np.sum(self.test_mask))

    @property
    def n_loss(self) -> int:
        return int(np.sum(self.loss_mask))


def training_block_mask(blocks: np.ndarray, test_block: int) -> np.ndarray:
    """Rows in every block except the outer test block (includes inner val)."""
    if test_block not in range(5):
        raise ValueError(f"test block must be 0..4, got {test_block}")
    return np.asarray(blocks, dtype=int) != int(test_block)


def inner_val_block(test_block: int) -> int:
    """Deterministic peninsula inner validation block (2, or 1 when fold==2).

    Fixed rule, not chosen by a score. Blocks 0..2 are Peninsula (29-32 cells,
    11-16 positives) and blocks 3..4 are Islands (98-99 cells, 6-9 positives).
    Validating on block 2 (or 1 for fold 2) keeps both Peninsula and Island
    blocks inside every fold's training loss mask and provides 11-14 validation
    positives for stable early stopping.
    """
    if test_block not in range(5):
        raise ValueError(f"test block must be 0..4, got {test_block}")
    return 2 if int(test_block) != 2 else 1


def loss_block_mask(blocks: np.ndarray, test_block: int) -> np.ndarray:
    """Strict training mask excluding both the outer test block and inner val block.

    Used for GNN loss, feature scaling, PCA projection, and QLoRA fine-tuning
    so early stopping on the inner validation block has zero label leakage.
    """
    block_ids = np.asarray(blocks, dtype=int)
    val_id = inner_val_block(int(test_block))
    return (block_ids != int(test_block)) & (block_ids != val_id)


def make_fold_split(
    blocks: np.ndarray,
    src: np.ndarray,
    dst: np.ndarray,
    *,
    fold: int,
    protocol: str,
    edge_weight: np.ndarray | None = None,
) -> FoldSplit:
    """Build masks, edges, and weights. ``src`` and ``dst`` are node indices."""
    if protocol not in PROTOCOLS:
        raise ValueError(f"unknown protocol: {protocol}")
    block_ids = np.asarray(blocks, dtype=int)
    src_arr = np.asarray(src, dtype=np.int64)
    dst_arr = np.asarray(dst, dtype=np.int64)
    if src_arr.shape != dst_arr.shape:
        raise ValueError("src and dst must have the same shape")
    if edge_weight is None:
        weight_arr = np.ones(src_arr.shape, dtype=np.float64)
    else:
        weight_arr = np.asarray(edge_weight, dtype=np.float64)
        if weight_arr.shape != src_arr.shape:
            raise ValueError("edge_weight must have the same shape as src and dst")
        if not np.isfinite(weight_arr).all() or np.any(weight_arr <= 0.0):
            raise ValueError("edge_weight must be finite and strictly positive")
    test_mask = block_ids == int(fold)
    if not np.any(test_mask) or np.all(test_mask):
        raise ValueError(f"fold {fold} does not split the nodes")
    val_id = inner_val_block(int(fold))
    val_mask = block_ids == val_id
    loss_mask = ~test_mask & ~val_mask
    if not np.any(loss_mask) or not np.any(val_mask):
        raise ValueError("loss and validation masks must both be non-empty")

    src_is_test = test_mask[src_arr]
    dst_is_test = test_mask[dst_arr]
    if protocol == "transductive":
        train_src, train_dst, train_w = src_arr, dst_arr, weight_arr
        eval_src, eval_dst, eval_w = src_arr, dst_arr, weight_arr
    else:
        train_keep = ~src_is_test & ~dst_is_test
        # Allow train->train, train->test, and test->test at evaluation while
        # forbidding test->train so test features never flow into train nodes.
        eval_keep = ~src_is_test | dst_is_test
        train_src = src_arr[train_keep]
        train_dst = dst_arr[train_keep]
        train_w = weight_arr[train_keep]
        eval_src = src_arr[eval_keep]
        eval_dst = dst_arr[eval_keep]
        eval_w = weight_arr[eval_keep]
    return FoldSplit(
        fold=int(fold),
        protocol=protocol,
        test_mask=test_mask,
        val_mask=val_mask,
        loss_mask=loss_mask,
        train_src=np.asarray(train_src, dtype=np.int64),
        train_dst=np.asarray(train_dst, dtype=np.int64),
        eval_src=np.asarray(eval_src, dtype=np.int64),
        eval_dst=np.asarray(eval_dst, dtype=np.int64),
        train_weight=np.asarray(train_w, dtype=np.float64),
        eval_weight=np.asarray(eval_w, dtype=np.float64),
    )
