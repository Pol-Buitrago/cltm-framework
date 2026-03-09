"""
Precompute and cache tokenized sequences for quantized training.
"""

import gc
import logging
import os
import time
import traceback

import numpy as np
import torch

from utils.data_utils import precompute_token_ids_for_split

logger = logging.getLogger(__name__)


def precompute_and_cache(encoded_ds, args, feature_extractor, model_hidden, centroids, device):
    """
    Precompute token IDs for train/validation/test splits and cache to disk.
    Loads cache if it exists.
    """
    os.makedirs(args.quantizer_cache_dir, exist_ok=True)
    outs = {"train": None, "validation": None, "test": None}

    for split in ["train", "validation", "test"]:
        if split not in encoded_ds:
            logger.info("Split %s not present", split)
            continue

        cache_file = os.path.join(args.quantizer_cache_dir, f"{args.lang_prefix}.{split}.token_ids.npy")

        if os.path.exists(cache_file):
            # Load existing cache
            try:
                t0 = time.time()
                token_lists = np.load(cache_file, allow_pickle=True)
                logger.info("Loaded token cache %s in %.2f s", cache_file, time.time() - t0)
                outs[split] = token_lists
                gc.collect()
            except Exception as e:
                logger.error("Error loading cache %s: %s", cache_file, e)
                logger.error(traceback.format_exc())
        else:
            # Compute and save cache
            try:
                t0 = time.time()
                token_lists = precompute_token_ids_for_split(
                    encoded_ds[split],
                    feature_extractor,
                    model_hidden,
                    torch.from_numpy(centroids).to(device),
                    layer_idx=args.kmeans_layer,
                    batch_size=args.per_device_train_batch_size,
                    device=device,
                )
                np.save(cache_file, np.array(token_lists, dtype=object), allow_pickle=True)
                logger.info("Saved token cache %s (%.2f s)", cache_file, time.time() - t0)
                outs[split] = token_lists
                gc.collect()
            except Exception as e:
                logger.error("Error computing cache for split %s: %s", split, e)
                logger.error(traceback.format_exc())

    return outs
