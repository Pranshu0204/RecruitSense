"""QLoRA fine-tuning with TRL SFTTrainer and automatic device detection.

Works on CUDA (4-bit NF4 + LoRA adapters), MPS (fp16 + LoRA), and CPU (fp32 + LoRA)
without code changes. The bitsandbytes import is lazy so the script runs cleanly on
macOS where bnb has no working build.
"""

import argparse
import importlib.util
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

from backend.core.config import get_settings
from backend.utils.logger import get_logger
from finetune import mlflow_utils

logger = get_logger(__name__)

DATA_DIR = Path("finetune/dataset/data")
RUNS_DIR = Path("finetune/runs")

# LoRA target modules for the Llama / Mistral / Qwen families. Most modern
# decoder-only models share these names; if a target is missing, PEFT silently
# skips it.
DEFAULT_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def _bnb_available() -> bool:
    """True iff ``bitsandbytes`` can be imported AND CUDA is available."""
    if not torch.cuda.is_available():
        return False
    return importlib.util.find_spec("bitsandbytes") is not None


def _detect_device() -> str:
    """Return ``cuda`` / ``mps`` / ``cpu`` based on what's actually usable."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _select_dtype(device: str) -> torch.dtype:
    """Pick the most efficient dtype for the active device."""
    if device == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if device == "mps":
        return torch.float16  # bf16 on MPS is shaky across torch versions
    return torch.float32


def build_model_and_tokenizer(base_model: str, device: str, dtype: torch.dtype):
    """Load the base model + tokenizer, applying 4-bit quant only when usable."""
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model_kwargs: dict = {"trust_remote_code": False}

    if device == "cuda" and _bnb_available():
        # Lazy import: only touch bnb when it's both installed and CUDA-backed.
        from transformers import BitsAndBytesConfig

        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs["device_map"] = "auto"
    elif device == "cuda":
        model_kwargs["torch_dtype"] = dtype
        model_kwargs["device_map"] = "auto"
    else:
        # MPS / CPU: load in target dtype and place manually.
        model_kwargs["torch_dtype"] = dtype

    logger.info("loading_base_model", model=base_model, device=device, dtype=str(dtype))
    model = AutoModelForCausalLM.from_pretrained(base_model, **model_kwargs)

    if device != "cuda" or not _bnb_available():
        model = model.to(device if device != "cpu" else "cpu")
    else:
        model = prepare_model_for_kbit_training(model)

    model.config.use_cache = False
    return model, tokenizer


def build_lora_config(rank: int, alpha: int, dropout: float) -> LoraConfig:
    """Standard causal-LM LoRA config tuned for instruction fine-tuning."""
    return LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=DEFAULT_TARGET_MODULES,
    )


def main() -> None:
    """Train a LoRA adapter and save it to ``finetune/runs/<timestamp>/``."""
    settings = get_settings()

    parser = argparse.ArgumentParser(description="LoRA fine-tune with TRL SFTTrainer.")
    parser.add_argument("--base-model", default=settings.finetune_base_model)
    parser.add_argument("--train-file", default=str(DATA_DIR / "train.jsonl"))
    parser.add_argument("--val-file", default=str(DATA_DIR / "val.jsonl"))
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--num-epochs", type=float, default=1.0)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=-1,
        help="Hard cap on training steps; overrides --num-epochs when > 0.",
    )
    parser.add_argument("--per-device-batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--save-steps", type=int, default=200)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not Path(args.train_file).exists():
        raise FileNotFoundError(
            f"{args.train_file} missing. Run `python -m finetune.prepare_dataset` first."
        )

    device = _detect_device()
    dtype = _select_dtype(device)
    bnb = _bnb_available()
    logger.info("device_detected", device=device, dtype=str(dtype), bitsandbytes=bnb)

    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir or RUNS_DIR / timestamp)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("output_dir", path=str(output_dir))

    # --- Datasets ---
    data_files = {"train": args.train_file}
    if Path(args.val_file).exists():
        data_files["validation"] = args.val_file
    raw = load_dataset("json", data_files=data_files)
    logger.info("dataset_loaded", **{k: len(v) for k, v in raw.items()})

    # --- Model + tokenizer + LoRA ---
    model, tokenizer = build_model_and_tokenizer(args.base_model, device, dtype)
    lora_cfg = build_lora_config(args.lora_rank, args.lora_alpha, args.lora_dropout)
    model = get_peft_model(model, lora_cfg)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(
        "lora_attached",
        trainable_params=trainable,
        total_params=total,
        trainable_pct=round(100.0 * trainable / total, 4),
    )

    # --- Trainer config ---
    sft_args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=args.num_epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        eval_strategy="steps" if "validation" in raw else "no",
        eval_steps=args.save_steps if "validation" in raw else None,
        bf16=(device == "cuda" and dtype == torch.bfloat16),
        fp16=(device != "cpu" and dtype == torch.float16),
        optim="paged_adamw_8bit" if bnb else "adamw_torch",
        report_to=[],  # we log to MLflow directly to avoid trainer-side coupling.
        seed=args.seed,
        max_seq_length=args.max_seq_length,
        packing=False,
        gradient_checkpointing=(device == "cuda"),
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=raw["train"],
        eval_dataset=raw.get("validation"),
        processing_class=tokenizer,
    )

    run_name = f"{Path(args.base_model).name}_{timestamp}"
    tags = {"device": device, "bitsandbytes": str(bnb), "base_model": args.base_model}

    with mlflow_utils.active_run(run_name=run_name, tags=tags):
        mlflow_utils.log_params(
            {
                **vars(args),
                "device": device,
                "dtype": str(dtype),
                "bitsandbytes": bnb,
                "trainable_params": trainable,
                "total_params": total,
            }
        )

        start = time.time()
        train_result = trainer.train()
        duration = time.time() - start

        metrics = dict(train_result.metrics)
        metrics["train_duration_s"] = duration
        mlflow_utils.log_metrics({k: v for k, v in metrics.items() if isinstance(v, int | float)})

        # Save the LoRA adapter + tokenizer + a small run manifest.
        adapter_dir = output_dir / "adapter"
        trainer.model.save_pretrained(str(adapter_dir))
        tokenizer.save_pretrained(str(adapter_dir))

        manifest = {
            "base_model": args.base_model,
            "device": device,
            "dtype": str(dtype),
            "bitsandbytes": bnb,
            "lora": {"r": args.lora_rank, "alpha": args.lora_alpha, "dropout": args.lora_dropout},
            "metrics": metrics,
            "completed_at": datetime.now(UTC).isoformat(),
        }
        (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
        mlflow_utils.log_artifact(adapter_dir)
        mlflow_utils.log_artifact(output_dir / "manifest.json")

    logger.info(
        "training_complete",
        adapter_dir=str(adapter_dir),
        duration_s=round(duration, 1),
        loss=metrics.get("train_loss"),
    )


if __name__ == "__main__":
    main()
