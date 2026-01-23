import logging
import os

import numpy as np
import torch
from datasets import Audio, load_dataset
from transformers import AutoFeatureExtractor

from args import parse_args
from model import QuantizedClassifier
from train import run_variant_continuous, run_variant_quantized
from utils.data_utils import (
    load_and_preprocess_dataset,
    load_feature_extractor,
    load_kmeans_centroids,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ---------- feature extractor ----------
    feature_extractor = load_feature_extractor()

    # ---------- dataset ----------
    encoded = load_and_preprocess_dataset(args, feature_extractor)

    # ---------- prepare centroids if quantized mode requested ----------
    centroids = None
    if args.mode in ("quantized", "both"):
        logger.info("Loading kmeans centroids from %s", args.kmeans_path)
        centroids = load_kmeans_centroids(args.kmeans_path, args.n_clusters)
        logger.info("Loaded centroids shape: %s", centroids.shape)

    # ---------- run requested variants ----------
    if args.mode in ("continuous", "both"):
        out_dir_c = args.output_dir if args.mode == "continuous" else os.path.join(args.output_dir, "continuous")
        os.makedirs(out_dir_c, exist_ok=True)
        run_variant_continuous(encoded, args, feature_extractor, out_dir_c)

    if args.mode in ("quantized", "both"):
        out_dir_q = args.output_dir if args.mode == "quantized" else os.path.join(args.output_dir, "quantized")
        os.makedirs(out_dir_q, exist_ok=True)
        if args.precompute_quantized:
            os.makedirs(args.quantizer_cache_dir, exist_ok=True)
        run_variant_quantized(encoded, args, feature_extractor, centroids, out_dir_q)


if __name__ == "__main__":
    main()
