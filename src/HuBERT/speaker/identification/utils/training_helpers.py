"""
Helpers for creating Hugging Face Trainer, handling fp16/bf16, evaluation, and saving artifacts.
"""

import logging
import os

import torch
import json
from typing import Any, Dict, Optional
from transformers import Trainer, TrainingArguments

logger = logging.getLogger(__name__)


def _bf16_supported() -> bool:
    """Check if GPU hardware likely supports bf16."""
    if not torch.cuda.is_available():
        return False
    try:
        prop = torch.cuda.get_device_properties(0)
        return prop.major >= 8
    except Exception:
        return False

def make_training_args(
    output_dir,
    args,
    fp16: bool = False,
    extra: Optional[Dict[str, Any]] = None,
) -> TrainingArguments:
    """Build HF TrainingArguments handling bf16/fp16 conflicts and gradient clipping."""
    requested_bf16 = bool(getattr(args, "bf16", False))
    hw_bf16_ok = _bf16_supported()

    effective_bf16 = requested_bf16
    effective_fp16 = bool(fp16)

    if requested_bf16 and effective_fp16:
        if hw_bf16_ok:
            logger.warning("Both bf16 and fp16 requested: using bf16 (fp16 disabled).")
            effective_fp16 = False
        else:
            logger.warning(
                "bf16 requested but hardware may not support it; falling back to fp16 (if requested)."
            )
            effective_bf16 = False

    if requested_bf16 and not hw_bf16_ok and not effective_fp16:
        logger.warning("bf16 requested but hardware does not support it; disabling bf16.")
        effective_bf16 = False

    eval_strategy = getattr(args, "evaluation_strategy", "epoch")
    save_strategy = getattr(args, "save_strategy", "epoch")
    load_best_model = getattr(args, "load_best_model_at_end", True)
    if eval_strategy == "no":
        logger.info(
            "No validation available: disabling load_best_model_at_end and save_strategy"
        )
        load_best_model = False
        save_strategy = "epoch"

    # build kwargs to avoid passing warmup_ratio=None (which triggers TF/TR error)
    kwargs = {
        # I/O
        "output_dir": output_dir,

        # Core training
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "num_train_epochs": args.num_train_epochs,
        "max_grad_norm": getattr(args, "max_grad_norm", 1.0),
        "seed": args.seed,
        "data_seed": args.seed,

        # Optimization
        "learning_rate": getattr(args, "learning_rate", 1e-5),
        "weight_decay": getattr(args, "weight_decay", 0.0),
        "lr_scheduler_type": getattr(args, "lr_scheduler_type", "cosine"),

        # Evaluation / checkpointing
        "eval_strategy": eval_strategy,
        "save_strategy": save_strategy,
        "load_best_model_at_end": load_best_model,
        "metric_for_best_model": getattr(args, "metric_for_best_model", "f1_macro"),
        "greater_is_better": getattr(args, "greater_is_better", True),

        # Precision / performance
        "bf16": effective_bf16,
        "fp16": effective_fp16,
        "deepspeed": getattr(args, "deepspeed", None),

        # Logging / reporting
        "logging_steps": 50,
        "report_to": None if getattr(args, "report_to", "none") == "none" else [args.report_to],
    }

    # Warmup: prefer warmup_ratio if provided, otherwise use warmup_steps (if any)
    warmup_ratio = getattr(args, "warmup_ratio", None)
    warmup_steps = getattr(args, "warmup_steps", None)

    if warmup_ratio is not None:
        # basic validation
        if not (0.0 < float(warmup_ratio) < 1.0):
            raise ValueError("warmup_ratio must be in (0, 1) if provided")
        kwargs["warmup_ratio"] = float(warmup_ratio)
    elif warmup_steps is not None:
        # accept zero as valid
        kwargs["warmup_steps"] = int(warmup_steps)

    # Finally construct TrainingArguments
    ta = TrainingArguments(**kwargs)

    if extra:
        for k, v in extra.items():
            setattr(ta, k, v)

    logger.info(
        f"TrainingArguments initialized (max_grad_norm={ta.max_grad_norm}, "
        f"fp16={ta.fp16}, bf16={ta.bf16})"
    )

    return ta

def build_trainer(
    model,
    training_args: TrainingArguments,
    train_dataset,
    eval_dataset,
    compute_metrics,
    data_collator,
) -> Trainer:
    """Create HF Trainer with clean signature."""
    return Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
    )


def evaluate_and_save_test(
    trainer: Trainer,
    test_ds,
    output_dir: str,
    variant_name: str,
    save_json_fn,
) -> Optional[dict]:
    """Evaluate on test set, save JSON, return metrics dict."""
    if test_ds is None:
        return None
    logger.info("Running test evaluation for %s", variant_name)
    tm = trainer.evaluate(test_ds)
    save_json_fn(tm, f"{output_dir}/test_metrics_{variant_name}.json")
    return tm


def save_common_artifacts(
    trainer: Trainer,
    args,
    output_dir: str,
    param_counts: dict,
    variant_name: str,
    save_json_fn,
    summarize_fn,
    timings: dict,
    test_metrics: Optional[dict],
):
    """Save label maps, model, and summary report."""
    save_json_fn(param_counts, f"{output_dir}/param_counts_{variant_name}.json")
    save_json_fn(args.label2id, f"{output_dir}/label2id.json")
    save_json_fn(args.id2label, f"{output_dir}/id2label.json")
    trainer.save_model(output_dir)

    summarize_fn(
        output_dir=output_dir,
        variant_name=variant_name,
        args_dict=vars(args),
        param_counts=param_counts,
        timings=timings,
        test_metrics=test_metrics,
    )

def process_test_metrics(trainer, test_metrics, output_dir, filename="eval_f1_macro.txt"):
    """
    Minimal and robust version:
    - Uses trainer.state.best_global_step or trainer.state.best_model_checkpoint to obtain the epoch
    - Avoids confusing the test evaluation (added after train) with evaluations performed during training
    - Writes eval_f1_macro.txt and best_info.json
    """
    if not test_metrics:
        return None, None

    # extract test f1 and print it
    f1_val = test_metrics.get("eval_f1_macro", test_metrics.get("f1_macro", None))
    try:
        f1_float = float(f1_val)
        print(f"{f1_float:.6f}", flush=True)
    except Exception:
        f1_float = None
        print("nan", flush=True)

    best_epoch = None
    best_ckpt = getattr(trainer.state, "best_model_checkpoint", None)
    best_step = getattr(trainer.state, "best_global_step", None)

    # 1) prefer best_global_step (map step -> entry that contains 'epoch')
    if best_step is not None:
        for entry in reversed(trainer.state.log_history):
            if entry.get("step") == best_step and "epoch" in entry:
                best_epoch = entry.get("epoch")
                break

    # 2) if there is no best_global_step, try to extract the step from the checkpoint name
    if best_epoch is None and best_ckpt:
        import re
        m = re.search(r"checkpoint[-_]?(\d+)", best_ckpt)
        if m:
            ck_step = int(m.group(1))
            for entry in reversed(trainer.state.log_history):
                if entry.get("step") == ck_step and "epoch" in entry:
                    best_epoch = entry.get("epoch")
                    break

    # write minimal output files
    try:
        with open(os.path.join(output_dir, filename), "w") as fh:
            fh.write(f"f1_macro: {f1_val}\n")
            fh.write(f"epoch: {best_epoch if best_epoch is not None else 'unknown'}\n")
        import json
        info = {
            "best_model_checkpoint": best_ckpt,
            "best_global_step": best_step,
            "best_epoch": best_epoch,
            "reported_test_f1": f1_val,
        }
        with open(os.path.join(output_dir, "best_info.json"), "w") as jh:
            json.dump(info, jh, indent=2)
    except Exception as e:
        logger.warning("Could not write best info files: %s", e)

    return f1_float, best_epoch
