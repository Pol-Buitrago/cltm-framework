"""
Model builders for continuous and quantized architectures.
"""

import json
import logging
import os
import torch
import torch.nn as nn
import random
import numpy as np
from transformers import AutoModelForAudioClassification

from utils.utils import count_parameters, freeze_all_except_head, freeze_encoder_generic

logger = logging.getLogger(__name__)

SEED = getattr(args, "seed", 42) if 'args' in globals() else 42

def set_global_seed(seed: int):
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    try:
        torch.use_deterministic_algorithms(True)
    except Exception as e:
        print("torch.use_deterministic_algorithms not available:", e)

set_global_seed(SEED)

# ------------------------------
# Continuous (HubERT/Wav2Vec2) model
# ------------------------------
def build_continuous_model(
    hf_model_name,
    num_labels,
    label2id,
    id2label,
    gradient_checkpointing=False,
    freeze_encoder=False,
    freeze_first_n=0,
):
    """Build continuous (non-quantized) transformer model."""
    model = AutoModelForAudioClassification.from_pretrained(
        hf_model_name,
        num_labels=num_labels,
        label2id={str(v): k for k, v in label2id.items()},
        id2label={str(k): v for k, v in id2label.items()},
    )

    if gradient_checkpointing:
        try:
            model.gradient_checkpointing_enable()
        except Exception:
            logger.warning("Could not enable gradient checkpointing")

    model.projector = nn.Identity()
    model.classifier = nn.Linear(model.config.hidden_size, num_labels)

    if freeze_encoder:
        logger.info("Freezing entire encoder")
        freeze_all_except_head(model)
    if freeze_first_n > 0:
        logger.info("Freezing first %d encoder layers", freeze_first_n)
        freeze_encoder_generic(model, freeze_first_n=freeze_first_n)

    params = count_parameters(model)
    return model, params


# ------------------------------
# Quantized model
# ------------------------------
def build_quantized_model(num_clusters, dim, num_labels, centroids=None, embedding_trainable=False):
    """Build quantized model with optional trainable centroids."""
    from model import QuantizedClassifier
    model_q = QuantizedClassifier(
        num_clusters=num_clusters,
        dim=dim,
        num_labels=num_labels,
        centroids=centroids,
        embedding_trainable=embedding_trainable,
    )
    params = count_parameters(model_q)
    return model_q, params


# ------------------------------
# JSON utility
# ------------------------------
def save_json(obj, path):
    """Save dictionary as formatted JSON."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
