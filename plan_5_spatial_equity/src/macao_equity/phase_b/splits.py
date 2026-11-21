"""Inductive and transductive edges for one frozen spatial block.

Inductive training removes every test node and every edge that touches a test
node. Inductive evaluation adds edges from training nodes to test nodes only,
so a test node can read training features and cannot send its features back.
Transductive training keeps every edge, including cross-block edges, and the
loss still ignores test labels. That is the feature-leakage contrast named in
the plan.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from macao_equity.phase_b.spec import PROTOCOLS


@dataclass(frozen=True)
class FoldSplit:
    """Node masks and edge endpoints for one outer block."""

    fold: int
    protocol: str
    test_mask: np.ndarray
    val_mask: np.ndarray
    loss_mask: np.ndarray
    train_src: np.ndarray
    train_dst: np.ndarray
    eval_src: np.ndarray
    eval_dst: np.ndarray

    @property
    def n_test(self) -> int:
        return int(np.sum(self.test_mask))

    @property
    def n_loss(self) -> int:
        return int(np.sum(self.loss_mask))


def training_block_mask(blocks: np.ndarray, test_block: int) -> np.ndarray:
    """Rows in every block except the outer test block.

    This is the LoRA fit set. It keeps the inner validation block. The GNN loss
    still drops that block through ``FoldSplit.loss_mask``.
    """
    if test_block not in range(5):
        raise ValueError(f"test block must be 0..4, got {test_block}")
    return np.asarray(blocks, dtype=int) != int(test_block)


def inner_val_block(test_block: int) -> int:
    """Largest remaining block id. Fixed rule, not chosen by a score."""
    if test_block not in range(5):
        raise ValueError(f"test block must be 0..4, got {test_block}")
    return max(block for block in range(5) if block != test_block)


def make_fold_split(
    blocks: np.ndarray,
    src: np.ndarray,
    dst: np.ndarray,
    *,
    fold: int,
    protocol: str,
) -> FoldSplit:
    """Build masks and edges. ``src`` and ``dst`` are node indices."""
    if protocol not in PROTOCOLS:
        raise ValueError(f"unknown protocol: {protocol}")
    block_ids = np.asarray(blocks, dtype=int)
    src_arr = np.asarray(src, dtype=np.int64)
    dst_arr = np.asarray(dst, dtype=np.int64)
    if src_arr.shape != dst_arr.shape:
        raise ValueError("src and dst must have the same shape")
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
        train_src, train_dst = src_arr, dst_arr
        eval_src, eval_dst = src_arr, dst_arr
    else:
        train_keep = ~src_is_test & ~dst_is_test
        eval_keep = train_keep | (~src_is_test & dst_is_test)
        train_src, train_dst = src_arr[train_keep], dst_arr[train_keep]
        eval_src, eval_dst = src_arr[eval_keep], dst_arr[eval_keep]
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
    )
