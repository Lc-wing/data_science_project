"""Route B News/Event LLM Zero-Shot & 4-bit NF4 QLoRA (L4: Gemma-4-12B-it; A100: Qwen3.8-27B)."""

from __future__ import annotations

import gc
import math
import re
import time
from collections.abc import Sequence
from typing import Any, Literal

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from stock_volatility import config
from stock_volatility.analysis.dl.datasets import build_event_prompt

DEFAULT_LLM_MODEL: str = "Qwen/Qwen3.8-27B"
DEFAULT_L4_LLM_MODEL: str = "google/gemma-4-12B-it"
SUPPORTED_LLM_MODELS: tuple[str, ...] = (
    "google/gemma-4-12B-it",
    "google/gemma-4-12B",
    "Qwen/Qwen3-14B",
    "Qwen/Qwen3.8-27B",
    "Qwen/Qwen3-32B",
    "Qwen/Qwen3-30B-A3B",
    "google/gemma-4-31B-it",
    "google/gemma-4-31B",
    "meta-models/Muse-Glimmer-30B",
)

TEXT_LORA_SUFFIXES: frozenset[str] = frozenset(
    {
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "in_proj_qkv",
        "out_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    }
)
EXCLUDED_LORA_PREFIXES: tuple[str, ...] = (
    "visual",
    "vision",
    "vision_tower",
    "vision_embedder",
    "embed_vision",
    "audio",
    "audio_tower",
    "embed_audio",
    "multi_modal_projector",
    "merger",
    "mtp",
    "lm_head",
)


def resolve_hf_model_revision(model_id: str, *objs: Any) -> str:
    """Resolve exact Hugging Face commit SHA from loaded config/model or local HF cache path."""
    for obj in objs:
        if obj is None:
            continue
        for cand in (obj, getattr(obj, "config", None)):
            if cand is None:
                continue
            sha = getattr(cand, "_commit_hash", None)
            if isinstance(sha, str) and sha.strip() and sha.strip().lower() != "unknown":
                return sha.strip()
    try:
        from huggingface_hub import try_to_load_from_cache

        cached_cfg = try_to_load_from_cache(model_id, "config.json")
        if isinstance(cached_cfg, str):
            m = re.search(r"/snapshots/([0-9a-f]{7,40})/", cached_cfg.replace("\\", "/"))
            if m:
                return m.group(1)
    except Exception:
        pass
    return "unknown"


def select_default_llm_for_vram(vram_gb: float) -> str:
    """Choose the optimal ungated 4-bit QLoRA LLM based on single-GPU VRAM capacity.

    - >= 35 GB (e.g., A100-40GB / A100-80GB): `Qwen/Qwen3.8-27B` (~22-26 GB peak)
    - <  35 GB (e.g., L4-24GB): `google/gemma-4-12B-it` (11.96B, ~11-13 GB peak, zero OOM)
    """
    return DEFAULT_LLM_MODEL if float(vram_gb) >= 35.0 else DEFAULT_L4_LLM_MODEL


def require_cuda_and_download_flag(
    *,
    allow_weight_download: bool,
    component: str,
) -> None:
    """Hard gate preventing accidental GPU weight downloads or CUDA calls on CPU-only VMs."""
    if not allow_weight_download:
        raise RuntimeError(
            f"Refusing to download or run GPU weights for {component!r}: "
            "pass --allow-weight-download explicitly on a CUDA L4/A100 runtime."
        )
    if not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA GPU required for {component!r}, but torch.cuda.is_available() is False. "
            "Run notebooks/phase_b_colab.ipynb on a Colab L4 (24GB) or A100 (40GB/80GB) runtime."
        )


def detect_model_loader_kind(
    model_type: str,
    architectures: Sequence[str],
) -> Literal["image_text_to_text", "causal_lm"]:
    """Choose AutoModelForImageTextToText vs AutoModelForCausalLM cleanly from HF config.

    Critical for multimodal conditional-generation wrappers including `google/gemma-4-12B-it`
    (`model_type='gemma4_unified'`, `Gemma4UnifiedForConditionalGeneration`) and
    `Qwen/Qwen3.8-27B` (`model_type='qwen3_5'`, `Qwen3_5ForConditionalGeneration`).
    """
    mt = str(model_type or "").lower()
    arch_str = " ".join(str(a) for a in architectures).lower()
    if mt in {"qwen3_5", "gemma3", "gemma4", "gemma4_unified", "qwen2_5_vl", "qwen3_vl"}:
        return "image_text_to_text"
    if "forconditionalgeneration" in arch_str and "t5" not in arch_str:
        return "image_text_to_text"
    return "causal_lm"


def filter_text_lora_target_modules(module_names: Sequence[str]) -> list[str]:
    """Select strictly text-backbone linear projections for LoRA, excluding vision/audio/MTP."""
    selected: list[str] = []
    for raw_name in module_names:
        name = str(raw_name).strip()
        if not name:
            continue
        parts = name.split(".")
        if any(p in EXCLUDED_LORA_PREFIXES for p in parts):
            continue
        if parts[-1] in TEXT_LORA_SUFFIXES:
            selected.append(name)
    return sorted(set(selected))


def compute_yes_no_log_odds(
    last_token_logits: torch.Tensor,
    *,
    yes_token_id: int,
    no_token_id: int,
) -> torch.Tensor:
    """Compute exact next-token log-odds `logit(YES) - logit(NO)` for tail-risk scoring."""
    return last_token_logits[:, yes_token_id] - last_token_logits[:, no_token_id]


def _resolve_single_token_id(tokenizer: Any, token_text: str) -> int:
    """Resolve a single vocabulary token ID for YES/NO scoring."""
    for candidate in (f" {token_text}", token_text):
        ids = tokenizer.encode(candidate, add_special_tokens=False)
        if len(ids) == 1:
            return int(ids[0])
    ids = tokenizer.encode(token_text, add_special_tokens=False)
    if not ids:
        raise ValueError(f"Tokenizer returned empty IDs for {token_text!r}")
    return int(ids[0])


def _forward_last_logits(
    model: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    use_logits_to_keep: bool,
) -> tuple[torch.Tensor, bool]:
    """Forward pass returning last-token float32 logits `[B, V]` with optional `logits_to_keep=1`.

    Assumes `tokenizer.padding_side == 'left'`, so index `-1` is always each row's final token.
    """
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        if use_logits_to_keep:
            try:
                out = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    logits_to_keep=1,
                )
            except TypeError:
                use_logits_to_keep = False
                out = model(input_ids=input_ids, attention_mask=attention_mask)
        else:
            out = model(input_ids=input_ids, attention_mask=attention_mask)
    return out.logits[:, -1, :].float(), use_logits_to_keep


def _score_prompts_log_odds(
    model: Any,
    tokenizer: Any,
    prompts: Sequence[str],
    *,
    yes_id: int,
    no_id: int,
    max_seq_len: int,
    batch_size: int,
) -> np.ndarray:
    """Run eval-mode batched forward passes returning raw `logit(YES) - logit(NO)` scores."""
    model.eval()
    device = next(model.parameters()).device
    scores: list[np.ndarray] = []
    use_ltk = True
    with torch.no_grad():
        for start in range(0, len(prompts), batch_size):
            batch_txt = list(prompts[start : start + batch_size])
            enc = tokenizer(
                batch_txt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_seq_len,
            ).to(device)
            last_logits, use_ltk = _forward_last_logits(
                model,
                enc["input_ids"],
                enc["attention_mask"],
                use_logits_to_keep=use_ltk,
            )
            z = compute_yes_no_log_odds(last_logits, yes_token_id=yes_id, no_token_id=no_id)
            scores.append(z.cpu().numpy())
    if not scores:
        return np.empty((0,), dtype=np.float32)
    return np.concatenate(scores, axis=0).astype(np.float32)


def run_llm_zero_shot_and_qlora(
    *,
    event_splits: dict[str, pd.DataFrame],
    model_id: str = DEFAULT_LLM_MODEL,
    allow_weight_download: bool = False,
    max_seq_len: int = 512,
    epochs: int = 1,
    train_batch_size: int = 8,
    grad_accum_steps: int = 1,
    eval_batch_size: int = 16,
    lr: float = 1e-4,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    max_train_events: int | None = None,
    seed: int = config.RANDOM_SEED,
) -> dict[str, Any]:
    """Run Zero-Shot + batched 4-bit NF4 QLoRA on Colab L4/A100 for event-anchored days (H3)."""
    require_cuda_and_download_flag(
        allow_weight_download=allow_weight_download, component=f"llm:{model_id}"
    )

    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoConfig,
        AutoModelForCausalLM,
        AutoModelForImageTextToText,
        AutoTokenizer,
        BitsAndBytesConfig,
    )

    t0 = time.perf_counter()
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.reset_peak_memory_stats()

    hf_cfg = AutoConfig.from_pretrained(model_id)
    loader_kind = detect_model_loader_kind(
        getattr(hf_cfg, "model_type", ""),
        getattr(hf_cfg, "architectures", []) or [],
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Left-padding keeps each sequence's final real token at index -1, enabling logits_to_keep=1
    tokenizer.padding_side = "left"

    yes_id = _resolve_single_token_id(tokenizer, "YES")
    no_id = _resolve_single_token_id(tokenizer, "NO")

    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model_cls = (
        AutoModelForImageTextToText if loader_kind == "image_text_to_text" else AutoModelForCausalLM
    )
    try:
        base_model = model_cls.from_pretrained(
            model_id,
            quantization_config=bnb_cfg,
            device_map="auto",
            dtype=torch.bfloat16,
        )
    except TypeError:
        base_model = model_cls.from_pretrained(
            model_id,
            quantization_config=bnb_cfg,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
    resolved_revision = resolve_hf_model_revision(model_id, hf_cfg, base_model)
    base_model.config.use_cache = False
    if hasattr(base_model.config, "text_config") and base_model.config.text_config is not None:
        base_model.config.text_config.use_cache = False

    alloc_gb = torch.cuda.memory_allocated() / 1e9
    print(
        f"[LLM] Loaded {model_id} ({loader_kind}) | allocated={alloc_gb:.2f} GB "
        f"| train_bs={train_batch_size}x{grad_accum_steps} | eval_bs={eval_batch_size}",
        flush=True,
    )

    train_ev = event_splits["train"].copy()
    val_ev = event_splits["val"].copy()
    test_ev = event_splits["test"].copy()
    if max_train_events is not None and len(train_ev) > max_train_events:
        train_ev = train_ev.sample(n=max_train_events, random_state=seed).sort_values(
            ["date", "ticker"]
        )

    val_prompts = [build_event_prompt(r) for r in val_ev.to_dict(orient="records")]
    test_prompts = [build_event_prompt(r) for r in test_ev.to_dict(orient="records")]
    train_prompts = [build_event_prompt(r) for r in train_ev.to_dict(orient="records")]

    # 1. Zero-shot event-anchored inference on 2024 Val & 2025 Test
    t_zs = time.perf_counter()
    print(
        f"[LLM] Zero-shot inference: Val={len(val_prompts)}, Test={len(test_prompts)} "
        f"(batch_size={eval_batch_size})...",
        flush=True,
    )
    zs_val_logits = _score_prompts_log_odds(
        base_model,
        tokenizer,
        val_prompts,
        yes_id=yes_id,
        no_id=no_id,
        max_seq_len=max_seq_len,
        batch_size=eval_batch_size,
    )
    zs_test_logits = _score_prompts_log_odds(
        base_model,
        tokenizer,
        test_prompts,
        yes_id=yes_id,
        no_id=no_id,
        max_seq_len=max_seq_len,
        batch_size=eval_batch_size,
    )
    print(
        f"[LLM] Zero-shot done in {time.perf_counter() - t_zs:.1f}s "
        f"| peak_vram={torch.cuda.max_memory_allocated() / 1e9:.2f} GB",
        flush=True,
    )

    # 2. Attach text-only LoRA adapters (explicitly excluding vision / audio / mtp modules)
    linear_names = [
        name
        for name, mod in base_model.named_modules()
        if isinstance(mod, torch.nn.Linear) or mod.__class__.__name__.startswith("Linear4bit")
    ]
    target_modules = filter_text_lora_target_modules(linear_names)
    if not target_modules:
        raise RuntimeError(f"No text LoRA target modules matched for {model_id}")

    base_model = prepare_model_for_kbit_training(
        base_model,
        use_gradient_checkpointing=False,
    )
    if hasattr(base_model, "enable_input_require_grads"):
        base_model.enable_input_require_grads()
    if hasattr(base_model, "gradient_checkpointing_enable"):
        base_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    lora_cfg = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        target_modules=target_modules,
    )
    peft_model = get_peft_model(base_model, lora_cfg)
    trainable_params = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in peft_model.parameters())
    print(
        f"[LLM] LoRA attached: {len(target_modules)} text modules | "
        f"trainable={trainable_params:,} | vram={torch.cuda.memory_allocated() / 1e9:.2f} GB",
        flush=True,
    )

    # 3. Supervised class-weighted BCE fine-tuning strictly on Train event-anchored days
    y_train = train_ev["y_tail"].astype(float).to_numpy()
    pos_w = float(max(np.sum(y_train == 0), 1.0) / max(np.sum(y_train == 1), 1.0))
    device = next(peft_model.parameters()).device
    pos_weight_t = torch.tensor([pos_w], dtype=torch.float32, device=device)

    try:
        import bitsandbytes as bnb

        optimizer: torch.optim.Optimizer = bnb.optim.PagedAdamW8bit(
            [p for p in peft_model.parameters() if p.requires_grad],
            lr=lr,
            weight_decay=0.01,
        )
    except Exception:
        optimizer = torch.optim.AdamW(
            [p for p in peft_model.parameters() if p.requires_grad],
            lr=lr,
            weight_decay=0.01,
        )

    rng = np.random.default_rng(seed)
    n_train = len(train_prompts)
    n_batches = max(1, math.ceil(n_train / train_batch_size))
    total_opt_steps = max(1, math.ceil((n_batches * epochs) / grad_accum_steps))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_opt_steps)

    epoch_losses: list[float] = []
    use_ltk = True
    peft_model.train()
    for ep_idx in range(1, epochs + 1):
        t_ep = time.perf_counter()
        order = rng.permutation(n_train)
        running_loss = 0.0
        seen_rows = 0
        optimizer.zero_grad(set_to_none=True)
        for batch_idx, start in enumerate(range(0, n_train, train_batch_size), start=1):
            idx_slice = order[start : start + train_batch_size]
            batch_txt = [train_prompts[int(i)] for i in idx_slice]
            enc = tokenizer(
                batch_txt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_seq_len,
            ).to(device)
            last_logits, use_ltk = _forward_last_logits(
                peft_model,
                enc["input_ids"],
                enc["attention_mask"],
                use_logits_to_keep=use_ltk,
            )
            z = compute_yes_no_log_odds(last_logits, yes_token_id=yes_id, no_token_id=no_id)
            y_t = torch.tensor(y_train[idx_slice], dtype=torch.float32, device=device)
            loss = F.binary_cross_entropy_with_logits(z, y_t, pos_weight=pos_weight_t)
            (loss / grad_accum_steps).backward()

            bs_cur = len(idx_slice)
            running_loss += float(loss.detach().cpu().item()) * bs_cur
            seen_rows += bs_cur

            if batch_idx % grad_accum_steps == 0 or batch_idx == n_batches:
                torch.nn.utils.clip_grad_norm_(peft_model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if batch_idx == 1 or batch_idx % 25 == 0 or batch_idx == n_batches:
                el = time.perf_counter() - t_ep
                eta = (el / batch_idx) * (n_batches - batch_idx)
                print(
                    f"[LLM] Epoch {ep_idx}/{epochs} batch {batch_idx}/{n_batches} "
                    f"({seen_rows}/{n_train}) | loss={running_loss / seen_rows:.4f} "
                    f"| peak_vram={torch.cuda.max_memory_allocated() / 1e9:.2f} GB "
                    f"| elapsed={el:.0f}s | ETA={eta:.0f}s",
                    flush=True,
                )
        epoch_losses.append(running_loss / max(seen_rows, 1))

    # 4. QLoRA post-fine-tuning inference on 2024 Val & 2025 Test event-anchored days
    print(
        f"[LLM] Post-QLoRA inference on Val & Test (batch_size={eval_batch_size})...",
        flush=True,
    )
    qlora_val_logits = _score_prompts_log_odds(
        peft_model,
        tokenizer,
        val_prompts,
        yes_id=yes_id,
        no_id=no_id,
        max_seq_len=max_seq_len,
        batch_size=eval_batch_size,
    )
    qlora_test_logits = _score_prompts_log_odds(
        peft_model,
        tokenizer,
        test_prompts,
        yes_id=yes_id,
        no_id=no_id,
        max_seq_len=max_seq_len,
        batch_size=eval_batch_size,
    )

    peak_vram_gb = float(torch.cuda.max_memory_allocated() / 1e9)
    elapsed_sec = float(time.perf_counter() - t0)
    gpu_name = str(torch.cuda.get_device_name(0))
    print(
        f"[LLM] Complete ({model_id}) | peak_vram={peak_vram_gb:.2f} GB "
        f"| elapsed={elapsed_sec:.1f}s",
        flush=True,
    )

    # Free GPU memory cleanly after LLM run
    del optimizer, scheduler, peft_model, base_model
    gc.collect()
    torch.cuda.empty_cache()

    val_scores_df = val_ev[["ticker", "date"]].copy()
    val_scores_df["llm_zs_logit"] = zs_val_logits
    val_scores_df["llm_qlora_logit"] = qlora_val_logits

    test_scores_df = test_ev[["ticker", "date"]].copy()
    test_scores_df["llm_zs_logit"] = zs_test_logits
    test_scores_df["llm_qlora_logit"] = qlora_test_logits

    return {
        "model_id": model_id,
        "revision": resolved_revision,
        "loader_kind": loader_kind,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "lora_target_module_count": len(target_modules),
        "trainable_params": int(trainable_params),
        "total_loaded_params": int(total_params),
        "train_event_days_used": int(n_train),
        "train_event_days_total": int(len(event_splits["train"])),
        "val_event_days": int(len(val_ev)),
        "test_event_days": int(len(test_ev)),
        "epochs": int(epochs),
        "epoch_losses": epoch_losses,
        "gpu_name": gpu_name,
        "peak_vram_gb": round(peak_vram_gb, 3),
        "elapsed_sec": round(elapsed_sec, 2),
        "val_scores_df": val_scores_df,
        "test_scores_df": test_scores_df,
    }
