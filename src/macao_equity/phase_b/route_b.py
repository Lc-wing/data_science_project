"""Route B text-attributed graph.

Weight files for Qwen3-Embedding-4B and BGE-M3 are loaded only after a CUDA
check. CPU tests cover the projection, the fuse step, and the refusal path.
The numeric GATv2 arm is Route A; this module does not retrain it.
"""

from __future__ import annotations

import json
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
from macao_equity.phase_b.models import build_model, parameter_count
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
    QWEN_LORA_LR,
    QWEN_LORA_MAX_STEPS,
    QWEN_LORA_R,
    QWEN_LORA_TARGET_MODULES,
    QWEN_MODEL_ID,
    QWEN_TASK_INSTRUCTION,
    ROUTE_B_ARMS,
    ROUTE_B_GNN_SEEDS,
    ROUTE_B_LORA_SEEDS,
    SPEC_BUDGET,
    TEXT_DIM,
    TEXT_ENCODE_BATCH_SIZE,
    TEXT_MAX_LENGTH,
    TOTAL_PARAM_LIMIT,
    WEIGHT_DECAY,
    EpochBudget,
)
from macao_equity.phase_b.splits import make_fold_split
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


def cls_pool(hidden: torch.Tensor) -> torch.Tensor:
    """BGE-M3 dense vector is the CLS state."""
    return hidden[:, 0]


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
        "max_steps": QWEN_LORA_MAX_STEPS,
        "learning_rate": QWEN_LORA_LR,
        "batch_size": QWEN_LORA_BATCH_SIZE,
        "seeds": list(ROUTE_B_LORA_SEEDS),
        "mrl_dim": TEXT_DIM,
    }


def _assert_param_cap(model: torch.nn.Module, label: str) -> int:
    total = parameter_count(model)
    if total > TOTAL_PARAM_LIMIT:
        raise RuntimeError(f"{label} has {total} parameters, above 5B")
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
                pooled = cls_pool(hidden)
            else:
                raise ValueError(pool)
        rows.append(pooled.float().cpu().numpy())
    return np.vstack(rows)


def _load_encoder(model_id: str, *, quantized: bool) -> tuple[Any, Any, str]:
    from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig

    dtype = _compute_dtype()
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        padding_side="left",
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
    revision = str(getattr(model.config, "_commit_hash", "") or "")
    return model, tokenizer, revision


def encode_bge_m3(
    texts: list[str],
    *,
    allow_weight_download: bool,
) -> tuple[np.ndarray, dict[str, Any]]:
    ensure_route_b_runtime(allow_weight_download)
    model, tokenizer, revision = _load_encoder(BGE_MODEL_ID, quantized=False)
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
    model, tokenizer, revision = _load_encoder(QWEN_MODEL_ID, quantized=False)
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
    """QLoRA on attention projections. The fit mask excludes the outer test block."""
    ensure_route_b_runtime(allow_weight_download)
    if int(seed) not in ROUTE_B_LORA_SEEDS:
        raise ValueError("LoRA seed is not in the pre-specified list")
    from peft import LoraConfig, TaskType, get_peft_model

    prompted = [qwen_prompt(text) for text in texts]
    model, tokenizer, revision = _load_encoder(QWEN_MODEL_ID, quantized=True)
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
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(head.parameters()),
        lr=QWEN_LORA_LR,
        weight_decay=WEIGHT_DECAY,
    )
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    model.train()
    steps = 0
    while steps < QWEN_LORA_MAX_STEPS:
        draw = torch.randint(0, fit_index.size, (QWEN_LORA_BATCH_SIZE,), generator=generator)
        batch_ids = fit_index[draw.numpy()]
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
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        steps += 1
    embeddings = mrl_truncate(_encode_hidden(model, tokenizer, prompted, "last"), TEXT_DIM)
    meta = {
        **_lora_config_dict(),
        "revision": revision,
        "total_parameters": total,
        "seed": int(seed),
        "n_fit": int(mask.sum()),
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
                    "Text projection is fit on the loss mask only. "
                    "LoRA uses one pre-specified seed; the ten seeds are the GATv2 head."
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
    """Train GATv2 on fused features. Embeddings are already computed."""
    head_params = parameter_count(build_model("gatv2", bundle.x.shape[1] + TEXT_DIM))
    if head_params < PARAM_BUDGET_MIN or head_params > PARAM_BUDGET_MAX:
        raise RuntimeError(f"fused GATv2 has {head_params} parameters, outside 0.2M–1.5M")
    fold_rows: list[dict[str, Any]] = []
    for arm in ROUTE_B_ARMS:
        method = arm_methods[arm]
        stored = arm_embeddings[arm]
        for protocol in PROTOCOLS:
            for seed in seeds:
                for fold in range(5):
                    split = make_fold_split(
                        bundle.blocks,
                        bundle.src,
                        bundle.dst,
                        fold=fold,
                        protocol=protocol,
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
                        model_name="gatv2",
                        protocol=protocol,
                        fold=fold,
                        seed=int(seed),
                        budget=budget,
                    )
                    fold_rows.append(
                        {
                            "model": arm,
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
        "parameter_count_gatv2": head_params,
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
    """Download, encode, optionally LoRA-fit, then train the fused GATv2 heads."""
    ensure_route_b_runtime(allow_weight_download)
    texts = bundle.poi_text.tolist()
    bge, bge_meta = encode_bge_m3(texts, allow_weight_download=True)
    frozen, frozen_meta = encode_qwen_frozen(texts, allow_weight_download=True)
    lora_by_fold: dict[int, np.ndarray] = {}
    lora_meta: list[dict[str, Any]] = []
    lora_seed = ROUTE_B_LORA_SEEDS[0]
    for fold in range(5):
        split = make_fold_split(
            bundle.blocks,
            bundle.src,
            bundle.dst,
            fold=fold,
            protocol="inductive",
        )
        encoded, meta = finetune_qwen_lora(
            texts,
            split.loss_mask,
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
            "bge_m3_gatv2": bge,
            "qwen3_embedding_4b_frozen_gatv2": frozen,
            "qwen3_embedding_4b_lora_gatv2": lora_by_fold,
        },
        {
            "bge_m3_gatv2": "pca",
            "qwen3_embedding_4b_frozen_gatv2": "mrl",
            "qwen3_embedding_4b_lora_gatv2": "as_is",
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
        "gnn_seeds": heads["gnn_seeds"],
        "lora_seeds": heads["lora_seeds"],
        "lora_config": heads["lora_config"],
        "language": "correlational",
    }
    return {"payload": payload, "fold_frame": heads["fold_frame"]}


def write_route_b_and_comparison(
    result: dict[str, Any],
    *,
    metrics_path: Path,
    tables_dir: Path,
    phase_a_metrics: Path,
    route_a_metrics: Path,
) -> dict[str, Path]:
    """Write Route B JSON and rebuild the comparison table around it."""
    metrics_path = Path(metrics_path)
    tables_dir = Path(tables_dir)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps(result["payload"], indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    route_a_payload = load_json(route_a_metrics)
    phase_a_payload = load_json(phase_a_metrics)
    rows = build_comparison_rows(
        phase_a_payload,
        route_a_payload["summary_rows"],
        result["payload"],
    )
    frame = comparison_frame(rows)
    csv_path = tables_dir / "phase_b_comparison.csv"
    md_path = tables_dir / "phase_b_comparison.md"
    fold_path = tables_dir / "phase_b_route_b_folds.csv"
    frame.to_csv(csv_path, index=False)
    md_path.write_text(to_markdown(frame), encoding="utf-8")
    result["fold_frame"].to_csv(fold_path, index=False)
    return {
        "metrics": metrics_path,
        "comparison_csv": csv_path,
        "comparison_md": md_path,
        "folds": fold_path,
    }
