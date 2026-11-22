"""Route B text-attributed graph.

Weight files for Qwen3-Embedding-4B and BGE-M3 are loaded only after a CUDA
check. CPU tests cover the projection, the fuse step, and the refusal path.
The numeric GATv2 arm is Route A; this module does not retrain it.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA

from macao_equity.phase_b.comparison import (
    build_comparison_rows,
    comparison_frame,
    load_json,
    to_markdown,
)
from macao_equity.phase_b.data import PhaseBBundle
from macao_equity.phase_b.metrics import mean_std
from macao_equity.phase_b.models import (
    build_model,
    effective_parameter_count,
    parameter_count,
    trainable_parameter_count,
)
from macao_equity.phase_b.spec import (
    BGE_MODEL_ID,
    PARAM_BUDGET_MAX,
    PARAM_BUDGET_MIN,
    PHASE_B_SEEDS,
    PRIMARY_METRICS,
    PROTOCOLS,
    QWEN_LORA_ALPHA,
    QWEN_LORA_BATCH_SIZE,
    QWEN_LORA_DROPOUT,
    QWEN_LORA_EPOCHS,
    QWEN_LORA_GRAD_CLIP,
    QWEN_LORA_HEAD_LR,
    QWEN_LORA_LR,
    QWEN_LORA_R,
    QWEN_LORA_TARGET_MODULES,
    QWEN_MODEL_ID,
    QWEN_TASK_INSTRUCTION,
    ROUTE_B_ARMS,
    ROUTE_B_EMBEDDINGS,
    ROUTE_B_GNN_SEEDS,
    ROUTE_B_HEADS,
    ROUTE_B_LORA_SEEDS,
    SPEC_BUDGET,
    TEXT_DIM,
    TEXT_ENCODE_BATCH_SIZE,
    TEXT_MAX_LENGTH,
    TOTAL_PARAM_LIMIT,
    WEIGHT_DECAY,
    EpochBudget,
)
from macao_equity.phase_b.splits import loss_block_mask, make_fold_split
from macao_equity.phase_b.train import train_one_fold

REFUSAL = (
    "Refusing to download Qwen3-Embedding-4B or BGE-M3. Route B runs on a "
    "Colab L4 or A100. This CPU environment does not fetch those weights."
)


def ensure_route_b_runtime(allow_weight_download: bool) -> None:
    """Fail before any hub call when the flag is off or CUDA is missing."""
    if not allow_weight_download:
        raise RuntimeError(REFUSAL)
    if not torch.cuda.is_available():
        raise RuntimeError(REFUSAL)


def mrl_truncate(embeddings: np.ndarray, dim: int) -> np.ndarray:
    """Keep the leading coordinates and L2-normalize each row."""
    matrix = np.asarray(embeddings, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] < dim:
        raise ValueError(f"MRL truncate needs at least {dim} columns")
    cut = matrix[:, :dim]
    norms = np.linalg.norm(cut, axis=1, keepdims=True)
    return cut / np.maximum(norms, 1e-12)


def pca_project(embeddings: np.ndarray, fit_mask: np.ndarray, dim: int, seed: int) -> np.ndarray:
    """PCA basis from ``fit_mask`` rows only, then applied to every row."""
    matrix = np.asarray(embeddings, dtype=np.float64)
    mask = np.asarray(fit_mask, dtype=bool)
    if int(np.sum(mask)) <= dim:
        raise ValueError("PCA fit set must be larger than the projection dimension")
    basis = PCA(n_components=dim, svd_solver="full", random_state=int(seed))
    basis.fit(matrix[mask])
    return np.asarray(basis.transform(matrix), dtype=np.float64)


def project_text(
    embeddings: np.ndarray,
    *,
    method: str,
    fit_mask: np.ndarray,
    dim: int = TEXT_DIM,
    seed: int = 42,
) -> np.ndarray:
    if method == "mrl":
        return mrl_truncate(embeddings, dim)
    if method == "pca":
        return pca_project(embeddings, fit_mask, dim, seed)
    if method == "as_is":
        matrix = np.asarray(embeddings, dtype=np.float64)
        if matrix.shape[1] != dim:
            raise ValueError(f"as_is text width must be {dim}")
        return matrix
    raise ValueError(f"unknown text projection: {method}")


def fuse_features(numeric: np.ndarray, text: np.ndarray) -> np.ndarray:
    """Concatenate leak-free numeric columns with a text projection."""
    left = np.asarray(numeric, dtype=np.float64)
    right = np.asarray(text, dtype=np.float64)
    if left.shape[0] != right.shape[0]:
        raise ValueError("numeric and text rows differ")
    return np.concatenate([left, right], axis=1)


def last_token_pool(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Qwen3-Embedding pool: last non-padding token. Left padding uses column -1."""
    left_padding = bool(torch.all(attention_mask[:, -1] == 1))
    if left_padding:
        return hidden[:, -1]
    lengths = attention_mask.sum(dim=1) - 1
    batch = torch.arange(hidden.shape[0], device=hidden.device)
    return hidden[batch, lengths]


def cls_pool(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """BGE-M3 CLS is the first non-pad token.

    Right padding leaves that token at position 0. Left padding puts pad tokens
    there, so position 0 is not read unless it is a real token.
    """
    if attention_mask.shape[0] != hidden.shape[0]:
        raise ValueError("attention mask and hidden batch sizes differ")
    first_real = attention_mask.to(dtype=torch.long).argmax(dim=1)
    batch = torch.arange(hidden.shape[0], device=hidden.device)
    return hidden[batch, first_real]


def qwen_prompt(text: str) -> str:
    return f"Instruct: {QWEN_TASK_INSTRUCTION}\nQuery:{text}"


def _lora_config_dict() -> dict[str, Any]:
    return {
        "base_model": QWEN_MODEL_ID,
        "r": QWEN_LORA_R,
        "lora_alpha": QWEN_LORA_ALPHA,
        "lora_dropout": QWEN_LORA_DROPOUT,
        "target_modules": list(QWEN_LORA_TARGET_MODULES),
        "task_type": "FEATURE_EXTRACTION",
        "epochs": QWEN_LORA_EPOCHS,
        "learning_rate": QWEN_LORA_LR,
        "head_learning_rate": QWEN_LORA_HEAD_LR,
        "grad_clip": QWEN_LORA_GRAD_CLIP,
        "batch_size": QWEN_LORA_BATCH_SIZE,
        "seeds": list(ROUTE_B_LORA_SEEDS),
        "mrl_dim": TEXT_DIM,
        "fit_rows": "loss_mask_only",
    }


def _assert_param_cap(model: torch.nn.Module, label: str) -> int:
    total = effective_parameter_count(model)
    if total > TOTAL_PARAM_LIMIT:
        raise RuntimeError(f"{label} has {total} logical parameters, above 5B")
    return total


def _compute_dtype() -> torch.dtype:
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def _encode_hidden(
    model: torch.nn.Module,
    tokenizer: Any,
    texts: list[str],
    pool: str,
) -> np.ndarray:
    device = next(model.parameters()).device
    rows: list[np.ndarray] = []
    model.eval()
    for start in range(0, len(texts), TEXT_ENCODE_BATCH_SIZE):
        batch = texts[start : start + TEXT_ENCODE_BATCH_SIZE]
        tokens = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=TEXT_MAX_LENGTH,
            return_tensors="pt",
        )
        tokens = {key: value.to(device) for key, value in tokens.items()}
        with torch.no_grad():
            hidden = model(**tokens).last_hidden_state
            if pool == "last":
                pooled = last_token_pool(hidden, tokens["attention_mask"])
            elif pool == "cls":
                pooled = cls_pool(hidden, tokens["attention_mask"])
            else:
                raise ValueError(pool)
        rows.append(pooled.float().cpu().numpy())
    return np.vstack(rows)


def checkpoint_revision(model_id: str, model: Any) -> str:
    """Hub commit if the lookup works, otherwise the config hash, otherwise unavailable."""
    config = getattr(model, "config", None)
    from_config = str(getattr(config, "_commit_hash", "") or "")
    from_hub = ""
    try:
        from huggingface_hub import model_info

        from_hub = str(model_info(model_id).sha or "")
    except Exception:
        from_hub = ""
    return from_hub or from_config or "unavailable"


def _load_encoder(
    model_id: str,
    *,
    quantized: bool,
    padding_side: str,
) -> tuple[Any, Any, str]:
    from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig

    dtype = _compute_dtype()
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        padding_side=padding_side,
        trust_remote_code=False,
    )
    kwargs: dict[str, Any] = {"trust_remote_code": False}
    if quantized:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        )
        kwargs["device_map"] = "auto"
    else:
        kwargs["torch_dtype"] = dtype
        kwargs["device_map"] = "auto"
    model = AutoModel.from_pretrained(model_id, **kwargs)
    revision = checkpoint_revision(model_id, model)
    return model, tokenizer, revision


def encode_bge_m3(
    texts: list[str],
    *,
    allow_weight_download: bool,
) -> tuple[np.ndarray, dict[str, Any]]:
    ensure_route_b_runtime(allow_weight_download)
    model, tokenizer, revision = _load_encoder(BGE_MODEL_ID, quantized=False, padding_side="right")
    total = _assert_param_cap(model, BGE_MODEL_ID)
    embeddings = _encode_hidden(model, tokenizer, texts, "cls")
    meta = {
        "model": BGE_MODEL_ID,
        "revision": revision,
        "total_parameters": total,
        "trainable": False,
    }
    del model
    return embeddings, meta


def encode_qwen_frozen(
    texts: list[str],
    *,
    allow_weight_download: bool,
) -> tuple[np.ndarray, dict[str, Any]]:
    ensure_route_b_runtime(allow_weight_download)
    prompted = [qwen_prompt(text) for text in texts]
    model, tokenizer, revision = _load_encoder(
        QWEN_MODEL_ID,
        quantized=False,
        padding_side="left",
    )
    total = _assert_param_cap(model, QWEN_MODEL_ID)
    embeddings = _encode_hidden(model, tokenizer, prompted, "last")
    meta = {
        "model": QWEN_MODEL_ID,
        "revision": revision,
        "total_parameters": total,
        "trainable": False,
        "instruction": QWEN_TASK_INSTRUCTION,
    }
    del model
    return embeddings, meta


def finetune_qwen_lora(
    texts: list[str],
    fit_mask: np.ndarray,
    y: np.ndarray,
    *,
    allow_weight_download: bool,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """QLoRA on attention projections. ``fit_mask`` is strictly ``loss_mask``."""
    ensure_route_b_runtime(allow_weight_download)
    if int(seed) not in ROUTE_B_LORA_SEEDS:
        raise ValueError("LoRA seed is not in the pre-specified list")
    from peft import LoraConfig, TaskType, get_peft_model

    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))

    prompted = [qwen_prompt(text) for text in texts]
    model, tokenizer, revision = _load_encoder(
        QWEN_MODEL_ID,
        quantized=True,
        padding_side="left",
    )
    lora = LoraConfig(
        r=QWEN_LORA_R,
        lora_alpha=QWEN_LORA_ALPHA,
        lora_dropout=QWEN_LORA_DROPOUT,
        target_modules=list(QWEN_LORA_TARGET_MODULES),
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION,
    )
    model = get_peft_model(model, lora)
    total = _assert_param_cap(model, "Qwen3-Embedding-4B LoRA")
    packed = parameter_count(model)
    trainable = trainable_parameter_count(model)
    device = next(model.parameters()).device
    head = torch.nn.Linear(TEXT_DIM, 1).to(device=device, dtype=torch.float32)
    mask = np.asarray(fit_mask, dtype=bool)
    labels = np.asarray(y, dtype=np.float64)
    fit_index = np.flatnonzero(mask)
    if fit_index.size == 0 or np.unique(labels[mask]).size < 2:
        raise ValueError("LoRA fit mask must contain both classes")
    n_pos = max(float(labels[mask].sum()), 1.0)
    n_neg = max(float(mask.sum() - labels[mask].sum()), 1.0)
    criterion = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([n_neg / n_pos], dtype=torch.float32, device=device)
    )
    lora_params = [param for param in model.parameters() if param.requires_grad]
    head_params = list(head.parameters())
    trainable_all = [*lora_params, *head_params]
    optimizer = torch.optim.AdamW(
        [
            {"params": lora_params, "lr": QWEN_LORA_LR, "weight_decay": WEIGHT_DECAY},
            {"params": head_params, "lr": QWEN_LORA_HEAD_LR, "weight_decay": WEIGHT_DECAY},
        ]
    )
    steps_per_epoch = int(np.ceil(fit_index.size / float(QWEN_LORA_BATCH_SIZE)))
    total_steps = max(1, steps_per_epoch * int(QWEN_LORA_EPOCHS))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_steps,
        eta_min=QWEN_LORA_LR * 0.1,
    )
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    model.train()
    losses: list[float] = []
    epoch_losses: list[float] = []
    for _epoch in range(int(QWEN_LORA_EPOCHS)):
        perm = torch.randperm(fit_index.size, generator=generator).numpy()
        shuffled = fit_index[perm]
        epoch_batch_losses: list[float] = []
        for start in range(0, shuffled.size, QWEN_LORA_BATCH_SIZE):
            batch_ids = shuffled[start : start + QWEN_LORA_BATCH_SIZE]
            batch_text = [prompted[int(index)] for index in batch_ids]
            tokens = tokenizer(
                batch_text,
                padding=True,
                truncation=True,
                max_length=TEXT_MAX_LENGTH,
                return_tensors="pt",
            )
            tokens = {key: value.to(device) for key, value in tokens.items()}
            hidden = model(**tokens).last_hidden_state
            pooled = last_token_pool(hidden, tokens["attention_mask"]).float()
            truncated = pooled[:, :TEXT_DIM]
            truncated = torch.nn.functional.normalize(truncated, dim=-1)
            logits = head(truncated).view(-1)
            target = torch.tensor(labels[batch_ids], dtype=torch.float32, device=device)
            loss = criterion(logits, target)
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite LoRA loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_all, max_norm=QWEN_LORA_GRAD_CLIP)
            optimizer.step()
            scheduler.step()
            loss_val = float(loss.detach().cpu())
            losses.append(loss_val)
            epoch_batch_losses.append(loss_val)
        epoch_losses.append(float(np.mean(epoch_batch_losses)))
    first_loss = float(losses[0])
    last_loss = float(losses[-1])
    epoch_first = float(epoch_losses[0])
    epoch_last = float(epoch_losses[-1])
    embeddings = mrl_truncate(_encode_hidden(model, tokenizer, prompted, "last"), TEXT_DIM)
    meta = {
        **_lora_config_dict(),
        "revision": revision,
        "total_parameters": total,
        "packed_parameters": int(packed),
        "trainable_parameters": int(trainable),
        "seed": int(seed),
        "n_fit": int(mask.sum()),
        "n_steps": len(losses),
        "n_epochs": int(QWEN_LORA_EPOCHS),
        "fit_rows": "loss_mask_only",
        "loss_first": float(np.round(first_loss, 8)),
        "loss_last": float(np.round(last_loss, 8)),
        "epoch_loss_first": float(np.round(epoch_first, 8)),
        "epoch_loss_last": float(np.round(epoch_last, 8)),
        "loss_fell": bool(epoch_last < epoch_first),
    }
    del model
    return embeddings, meta


def _summary_from_folds(fold_frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model in ROUTE_B_ARMS:
        for protocol in PROTOCOLS:
            part = fold_frame[(fold_frame["model"] == model) & (fold_frame["protocol"] == protocol)]
            row: dict[str, Any] = {
                "model": model,
                "protocol": protocol,
                "n_seeds": int(part["seed"].nunique()),
                "status": "ran_on_colab",
                "std_definition": (
                    "mean and ddof=1 std across GNN seeds of the ddof=0 mean across 5 folds"
                ),
                "note": (
                    "LoRA is fit strictly on loss_mask rows (both outer test and inner "
                    "validation blocks excluded). The ten seeds belong to the GATv2 or "
                    "TransformerConv head with walk-time edge weights."
                ),
            }
            for metric in PRIMARY_METRICS:
                per_seed = [
                    float(seed_part[metric].mean()) for _seed, seed_part in part.groupby("seed")
                ]
                mean, std = mean_std(per_seed, ddof=1)
                row[f"{metric}_mean"] = mean
                row[f"{metric}_std"] = std
            rows.append(row)
    return rows


def run_fused_heads(
    bundle: PhaseBBundle,
    arm_embeddings: dict[str, Any],
    arm_methods: dict[str, str],
    *,
    seeds: tuple[int, ...] = ROUTE_B_GNN_SEEDS,
    budget: EpochBudget = SPEC_BUDGET,
) -> dict[str, Any]:
    """Train GATv2 and TransformerConv on fused features. Embeddings are already computed."""
    fused_dim = bundle.x.shape[1] + TEXT_DIM
    head_params = {
        head: parameter_count(build_model(head, fused_dim)) for head in ROUTE_B_HEADS
    }
    for head, count in head_params.items():
        if count < PARAM_BUDGET_MIN or count > PARAM_BUDGET_MAX:
            raise RuntimeError(f"fused {head} has {count} parameters, outside 0.2M–1.5M")
    fold_rows: list[dict[str, Any]] = []
    for embedding in ROUTE_B_EMBEDDINGS:
        method = arm_methods[embedding]
        stored = arm_embeddings[embedding]
        for head in ROUTE_B_HEADS:
            model_name = f"{embedding}_{head}"
            for protocol in PROTOCOLS:
                for seed in seeds:
                    for fold in range(5):
                        split = make_fold_split(
                            bundle.blocks,
                            bundle.src,
                            bundle.dst,
                            fold=fold,
                            protocol=protocol,
                            edge_weight=bundle.edge_weight,
                        )
                        text = stored[fold] if isinstance(stored, dict) else stored
                        projected = project_text(
                            text,
                            method=method,
                            fit_mask=split.loss_mask,
                            dim=TEXT_DIM,
                            seed=42,
                        )
                        fused = fuse_features(bundle.x, projected)
                        fit = train_one_fold(
                            fused,
                            bundle.y,
                            bundle.blocks,
                            bundle.src,
                            bundle.dst,
                            model_name=head,
                            protocol=protocol,
                            fold=fold,
                            seed=int(seed),
                            budget=budget,
                            edge_weight=bundle.edge_weight,
                        )
                        fold_rows.append(
                            {
                                "model": model_name,
                                "protocol": protocol,
                                "seed": int(seed),
                                "fold": fold,
                                "roc_auc": fit.metrics["roc_auc"],
                                "pr_auc": fit.metrics["pr_auc"],
                                "top20_hit_rate": fit.metrics["top20_hit_rate"],
                                "best_epoch": fit.best_epoch,
                            }
                        )
    fold_frame = pd.DataFrame(fold_rows)
    return {
        "parameter_count_gatv2": head_params["gatv2"],
        "parameter_count_graph_transformer": head_params["graph_transformer"],
        "gnn_seeds": [int(seed) for seed in seeds],
        "lora_seeds": list(ROUTE_B_LORA_SEEDS),
        "lora_config": _lora_config_dict(),
        "rows": _summary_from_folds(fold_frame),
        "fold_frame": fold_frame,
    }


def run_route_b(
    bundle: PhaseBBundle,
    *,
    allow_weight_download: bool,
    budget: EpochBudget = SPEC_BUDGET,
) -> dict[str, Any]:
    """Download, encode, LoRA-fit strictly on loss_mask, then train both graph heads."""
    ensure_route_b_runtime(allow_weight_download)
    texts = bundle.poi_text.tolist()
    bge, bge_meta = encode_bge_m3(texts, allow_weight_download=True)
    frozen, frozen_meta = encode_qwen_frozen(texts, allow_weight_download=True)
    lora_by_fold: dict[int, np.ndarray] = {}
    lora_meta: list[dict[str, Any]] = []
    lora_seed = ROUTE_B_LORA_SEEDS[0]
    for fold in range(5):
        encoded, meta = finetune_qwen_lora(
            texts,
            loss_block_mask(bundle.blocks, fold),
            bundle.y,
            allow_weight_download=True,
            seed=lora_seed,
        )
        lora_by_fold[fold] = encoded
        meta = dict(meta)
        meta["fold"] = fold
        lora_meta.append(meta)
    heads = run_fused_heads(
        bundle,
        {
            "bge_m3": bge,
            "qwen3_embedding_4b_frozen": frozen,
            "qwen3_embedding_4b_lora": lora_by_fold,
        },
        {
            "bge_m3": "pca",
            "qwen3_embedding_4b_frozen": "mrl",
            "qwen3_embedding_4b_lora": "as_is",
        },
        seeds=tuple(PHASE_B_SEEDS),
        budget=budget,
    )
    payload = {
        "task": "has_ev_charger text-attributed node classification",
        "feature_columns": list(bundle.feature_names),
        "text_dim": TEXT_DIM,
        "encoders": {"bge_m3": bge_meta, "qwen_frozen": frozen_meta, "qwen_lora": lora_meta},
        "rows": heads["rows"],
        "parameter_count_gatv2": heads["parameter_count_gatv2"],
        "parameter_count_graph_transformer": heads["parameter_count_graph_transformer"],
        "gnn_seeds": heads["gnn_seeds"],
        "lora_seeds": heads["lora_seeds"],
        "lora_config": heads["lora_config"],
        "language": "correlational",
    }
    return {"payload": payload, "fold_frame": heads["fold_frame"]}


RUN1_LIVE_NAMES = frozenset(
    {
        "phase_b_route_b.json",
        "phase_b_route_b_folds.csv",
        "phase_b_comparison.csv",
        "phase_b_comparison.md",
    }
)

RUN2_LIVE_NAMES = frozenset(
    {
        "phase_b_route_b_run2.json",
        "phase_b_route_b_folds_run2.csv",
        "phase_b_comparison_run2.csv",
        "phase_b_comparison_run2.md",
    }
)


def run2_output_paths(project_root: Path) -> dict[str, Path]:
    """Paths for the second Colab run. Run 1's files are not in this map."""
    root = Path(project_root)
    tables = root / "outputs" / "tables"
    paths = {
        "metrics": root / "outputs" / "metrics" / "phase_b_route_b_run2.json",
        "folds": tables / "phase_b_route_b_folds_run2.csv",
        "comparison_csv": tables / "phase_b_comparison_run2.csv",
        "comparison_md": tables / "phase_b_comparison_run2.md",
        "archive": root / "outputs" / "colab_runs" / "run2",
    }
    for path in paths.values():
        _refuse_protected(path)
    return paths


def run3_output_paths(project_root: Path) -> dict[str, Path]:
    """Paths for the third (leak-free QLoRA + edge-weighted) Colab run.

    Run 1 and Run 2 live and archived files are strictly protected.
    """
    root = Path(project_root)
    tables = root / "outputs" / "tables"
    paths = {
        "metrics": root / "outputs" / "metrics" / "phase_b_route_b_run3.json",
        "folds": tables / "phase_b_route_b_folds_run3.csv",
        "comparison_csv": tables / "phase_b_comparison_run3.csv",
        "comparison_md": tables / "phase_b_comparison_run3.md",
        "archive": root / "outputs" / "colab_runs" / "run3",
    }
    for path in paths.values():
        _refuse_protected_run3(path)
    return paths


def _refuse_protected(path: Path) -> None:
    if path.name in RUN1_LIVE_NAMES:
        raise RuntimeError(f"refusing to overwrite the first Colab outputs: {path}")
    if "colab_runs" in path.parts and "run1" in path.parts:
        raise RuntimeError(f"refusing to overwrite archived run 1: {path}")


def _refuse_protected_run3(path: Path) -> None:
    _refuse_protected(path)
    if path.name in RUN2_LIVE_NAMES:
        raise RuntimeError(f"refusing to overwrite the second Colab outputs: {path}")
    if "colab_runs" in path.parts and "run2" in path.parts:
        raise RuntimeError(f"refusing to overwrite archived run 2: {path}")


def _write_run_artifacts(
    result: dict[str, Any],
    *,
    paths: dict[str, Path],
    phase_a_metrics: Path,
    route_a_metrics: Path,
    guard_fn: Any,
) -> dict[str, Path]:
    paths["metrics"].parent.mkdir(parents=True, exist_ok=True)
    paths["folds"].parent.mkdir(parents=True, exist_ok=True)
    paths["archive"].mkdir(parents=True, exist_ok=True)
    payload_text = json.dumps(result["payload"], indent=2, allow_nan=False) + "\n"
    paths["metrics"].write_text(payload_text, encoding="utf-8")
    route_a_payload = load_json(route_a_metrics)
    phase_a_payload = load_json(phase_a_metrics)
    rows = build_comparison_rows(
        phase_a_payload,
        route_a_payload["summary_rows"],
        result["payload"],
    )
    frame = comparison_frame(rows)
    frame.to_csv(paths["comparison_csv"], index=False)
    paths["comparison_md"].write_text(to_markdown(frame), encoding="utf-8")
    result["fold_frame"].to_csv(paths["folds"], index=False)
    for key in ("metrics", "folds", "comparison_csv", "comparison_md"):
        archived = paths["archive"] / paths[key].name
        guard_fn(archived)
        shutil.copyfile(paths[key], archived)
    return paths


def write_route_b_run2(
    result: dict[str, Any],
    *,
    project_root: Path,
    phase_a_metrics: Path,
    route_a_metrics: Path,
) -> dict[str, Path]:
    """Write run 2 beside the archived first Colab comparison."""
    paths = run2_output_paths(project_root)
    return _write_run_artifacts(
        result,
        paths=paths,
        phase_a_metrics=phase_a_metrics,
        route_a_metrics=route_a_metrics,
        guard_fn=_refuse_protected,
    )


def write_route_b_run3(
    result: dict[str, Any],
    *,
    project_root: Path,
    phase_a_metrics: Path,
    route_a_metrics: Path,
) -> dict[str, Path]:
    """Write run 3 beside archived run 1 and run 2 comparisons."""
    paths = run3_output_paths(project_root)
    return _write_run_artifacts(
        result,
        paths=paths,
        phase_a_metrics=phase_a_metrics,
        route_a_metrics=route_a_metrics,
        guard_fn=_refuse_protected_run3,
    )
