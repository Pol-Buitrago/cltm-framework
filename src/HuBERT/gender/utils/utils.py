"""
Utility functions for metrics, parameter counting, and freezing model layers.
"""

import logging
import warnings
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import evaluate
from sklearn.metrics import f1_score

logger = logging.getLogger(__name__)

# Ignore specific warning from tensor operations
warnings.filterwarnings(
    "ignore",
    message="Was asked to gather along dimension 0, but all input tensors were scalars"
)

def count_parameters(model: nn.Module) -> Dict[str, float]:
    """Return total, trainable, and trainable percentage of model parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total": int(total),
        "trainable": int(trainable),
        "trainable_pct": 100.0 * trainable / total,
    }

def compute_metrics(eval_pred):
    """Compute accuracy and macro F1 from logits and labels, computed locally (no HF evaluate)."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)

    acc_value = float((preds == labels).mean())
    f1m = f1_score(labels, preds, average="macro", zero_division=0)

    return {"accuracy": acc_value, "f1_macro": float(f1m)}

def freeze_all_except_head(model: nn.Module):
    """
    Freeze all parameters except the final classifier head.
    Logs kept and frozen parameter names.
    """
    kept = []
    frozen = []

    for name, param in model.named_parameters():
        if name.startswith("classifier."):
            param.requires_grad = True
            kept.append(name)
        else:
            param.requires_grad = False
            frozen.append(name)

    logger.info(
        "freeze_all_except_head: kept %d params trainable, frozen %d params",
        len(kept),
        len(frozen),
    )
    logger.info("Trainable params: %s", kept)
    logger.info("Frozen params: %s", frozen)

    return kept, frozen


def freeze_encoder_generic(model: nn.Module, freeze_first_n: int = 0):
    """
    Freeze the first N transformer/encoder layers heuristically.
    Supports common patterns in names: encoder.layer, hubert.encoder.layers, etc.
    """
    if freeze_first_n <= 0:
        return

    layer_patterns = [
        "encoder.layer",
        "encoder.layers",
        "transformer.layer",
        "hubert.encoder.layers",
        "feature_extractor.conv_layers",
    ]
    frozen = 0

    for name, p in model.named_parameters():
        for patt in layer_patterns:
            if patt in name:
                parts = name.split(patt)[-1].split(".")
                layer_idx = None
                for tok in parts:
                    if tok.isdigit():
                        layer_idx = int(tok)
                        break
                if layer_idx is not None and layer_idx < freeze_first_n:
                    p.requires_grad = False
                    frozen += 1
                break

    logger.info("freeze_first_n heuristic frozen %d parameters (heuristic)", frozen)
