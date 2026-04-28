"""Evaluate a fine-tuned LoRA adapter against the held-out validation split.

Computes four families of metrics so you can read both **format quality**
and **scoring fidelity** at a glance:

- **JSON-parse rate** — % of generations that survive ``json.loads`` and a
  Pydantic ``ScoreOutput`` round-trip. The single most important number for
  a structured-output fine-tune; if this is low, nothing else matters.
- **Per-dimension MAE** — mean absolute error of each 0–10 dimension score
  vs. the reference target.
- **Composite MAE & tier accuracy** — error on the aggregate 0–100 score and
  how often the predicted tier (A/B/C/D) matches the reference.
- **ROUGE-L & BLEU** — surface-level text overlap between generated and
  reference rationales (a sanity check on style).

Usage::

    python -m finetune.evaluate \\
        --base-model meta-llama/Llama-3.2-1B-Instruct \\
        --adapter-dir finetune/runs/<timestamp>/adapter \\
        --val-file finetune/dataset/data/val.jsonl \\
        --max-samples 100
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from peft import PeftModel
from pydantic import ValidationError
from transformers import AutoModelForCausalLM, AutoTokenizer

from backend.core.schemas import (
    DIMENSION_NAMES,
    ScoreOutput,
    composite_from_dimensions,
    tier_from_composite,
)
from backend.utils.logger import get_logger
from finetune import mlflow_utils

logger = get_logger(__name__)


def _detect_device() -> str:
    """Return ``cuda`` / ``mps`` / ``cpu`` based on what's actually usable."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _select_dtype(device: str) -> torch.dtype:
    if device == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if device == "mps":
        return torch.float16
    return torch.float32


def _extract_first_json(text: str) -> dict[str, Any] | None:
    """Find the first balanced ``{...}`` block and ``json.loads`` it. Returns ``None`` on failure."""
    depth = 0
    start = -1
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start >= 0:
                try:
                    return json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    start = -1
                    continue
    return None


def _parses_as_scoreoutput(payload: dict[str, Any]) -> bool:
    """True iff ``payload`` round-trips through the production ``ScoreOutput`` schema."""
    try:
        ScoreOutput.model_validate(payload)
        return True
    except (ValidationError, KeyError, TypeError):
        return False


def _rougeL(pred: str, ref: str) -> float:
    """Lightweight ROUGE-L (LCS-based F1). No external metric server required."""
    pred_tokens = pred.split()
    ref_tokens = ref.split()
    if not pred_tokens or not ref_tokens:
        return 0.0
    m, n = len(pred_tokens), len(ref_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if pred_tokens[i - 1] == ref_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs = dp[m][n]
    if lcs == 0:
        return 0.0
    prec = lcs / m
    rec = lcs / n
    return 2 * prec * rec / (prec + rec)


def _bleu1(pred: str, ref: str) -> float:
    """Unigram precision with brevity penalty — quick BLEU-1 stand-in."""
    pred_tokens = pred.split()
    ref_tokens = ref.split()
    if not pred_tokens:
        return 0.0
    overlap = sum(1 for t in pred_tokens if t in ref_tokens) / len(pred_tokens)
    bp = min(1.0, len(pred_tokens) / max(1, len(ref_tokens)))
    return overlap * bp


def load_finetuned(base_model: str, adapter_dir: str, device: str, dtype: torch.dtype):
    """Load the base model, attach the trained LoRA adapter, return ``(model, tokenizer)``."""
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=dtype, trust_remote_code=False
    )
    model = PeftModel.from_pretrained(model, adapter_dir)
    model = model.to(device)
    model.eval()
    return model, tokenizer


def generate(
    model, tokenizer, messages: list[dict[str, str]], device: str, max_new_tokens: int
) -> str:
    """Apply the chat template and generate one response. Returns the assistant text."""
    prompt = tokenizer.apply_chat_template(
        messages[:-1], tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096).to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.pad_token_id,
        )
    new_tokens = out[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def main() -> None:
    """Run evaluation, print a metrics table, and log to MLflow."""
    parser = argparse.ArgumentParser(description="Evaluate a fine-tuned LoRA adapter.")
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter-dir", required=True)
    parser.add_argument("--val-file", default="finetune/dataset/data/val.jsonl")
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not Path(args.val_file).exists():
        raise FileNotFoundError(f"{args.val_file} missing.")
    if not Path(args.adapter_dir).exists():
        raise FileNotFoundError(f"{args.adapter_dir} missing — train.py first.")

    device = _detect_device()
    dtype = _select_dtype(device)
    logger.info("device_detected", device=device, dtype=str(dtype))

    ds = load_dataset("json", data_files={"val": args.val_file})["val"]
    if args.max_samples and args.max_samples < len(ds):
        ds = ds.select(range(args.max_samples))
    logger.info("eval_samples", n=len(ds))

    model, tokenizer = load_finetuned(args.base_model, args.adapter_dir, device, dtype)

    parsed_count = 0
    schema_ok_count = 0
    composite_errors: list[float] = []
    tier_correct = 0
    dim_errors: dict[str, list[float]] = {d: [] for d in DIMENSION_NAMES}
    rouge_scores: list[float] = []
    bleu_scores: list[float] = []

    for i, row in enumerate(ds):
        messages = row["messages"]
        ref_text = messages[-1]["content"]
        gen_text = generate(model, tokenizer, messages, device, args.max_new_tokens)

        rouge_scores.append(_rougeL(gen_text, ref_text))
        bleu_scores.append(_bleu1(gen_text, ref_text))

        gen_json = _extract_first_json(gen_text)
        if gen_json is None:
            logger.debug("eval_unparseable", idx=i)
            continue
        parsed_count += 1

        if _parses_as_scoreoutput(gen_json):
            schema_ok_count += 1

        try:
            ref_json = json.loads(ref_text)
        except json.JSONDecodeError:
            continue

        for dim in DIMENSION_NAMES:
            try:
                gen_score = float(gen_json["dimension_scores"][dim]["score"])
                ref_score = float(ref_json["dimension_scores"][dim]["score"])
                dim_errors[dim].append(abs(gen_score - ref_score))
            except (KeyError, TypeError, ValueError):
                continue

        try:
            from backend.core.schemas import DimensionScore
            gen_typed = {
                d: DimensionScore(
                    score=float(gen_json["dimension_scores"][d]["score"]),
                    rationale=str(gen_json["dimension_scores"][d].get("rationale", "—")),
                )
                for d in DIMENSION_NAMES
            }
            gen_composite = composite_from_dimensions(gen_typed)
            ref_composite = float(ref_json.get("composite_score", 0.0))
            composite_errors.append(abs(gen_composite - ref_composite))
            gen_tier = tier_from_composite(gen_composite).value
            ref_tier = ref_json.get("tier", "")
            if gen_tier == ref_tier:
                tier_correct += 1
        except (KeyError, TypeError, ValueError):
            continue

        if (i + 1) % 10 == 0:
            logger.info("eval_progress", done=i + 1, total=len(ds))

    n = len(ds)
    metrics = {
        "samples": n,
        "json_parse_rate": parsed_count / n if n else 0.0,
        "schema_valid_rate": schema_ok_count / n if n else 0.0,
        "composite_mae": (sum(composite_errors) / len(composite_errors)) if composite_errors else float("nan"),
        "tier_accuracy": tier_correct / n if n else 0.0,
        "rouge_l": sum(rouge_scores) / len(rouge_scores) if rouge_scores else 0.0,
        "bleu_1": sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0.0,
    }
    for dim, errs in dim_errors.items():
        metrics[f"mae_{dim}"] = sum(errs) / len(errs) if errs else float("nan")

    print("\n=== Evaluation results ===")
    for k, v in metrics.items():
        print(f"  {k:30s} {v:.4f}" if isinstance(v, float) else f"  {k:30s} {v}")

    run_name = f"eval_{Path(args.adapter_dir).parent.name}"
    with mlflow_utils.active_run(run_name=run_name, tags={"phase": "evaluate"}):
        mlflow_utils.log_params(vars(args))
        mlflow_utils.log_metrics(
            {k: float(v) for k, v in metrics.items() if isinstance(v, float) and v == v}  # drop NaN
        )

    out_path = Path(args.adapter_dir).parent / "eval_metrics.json"
    out_path.write_text(json.dumps(metrics, indent=2))
    logger.info("eval_complete", metrics_path=str(out_path), **{k: round(v, 4) if isinstance(v, float) else v for k, v in metrics.items()})


if __name__ == "__main__":
    main()
