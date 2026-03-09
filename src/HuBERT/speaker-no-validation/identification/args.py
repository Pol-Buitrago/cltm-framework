"""
Argument parser for training/evaluation runs.
"""
import argparse


def parse_args():
    p = argparse.ArgumentParser(
        description="Train continuous and/or quantized mHuBERT gender classifier"
    )

    # ---- Core / model ----
    core = p.add_argument_group("Core / model")
    core.add_argument(
        "--mode",
        choices=["continuous", "quantized", "both"],
        default="both",
        help="Which variant(s) to run",
    )
    core.add_argument(
        "--task",
        type=str,
        default="gender",
        help="Task to run, e.g. 'gender', 'age', 'emotion'."
    )
    core.add_argument(
        "--hf_model",
        type=str,
        default="slprl/mhubert-base-25hz",
        help="Hugging Face model id for mHuBERT 25Hz",
    )
    core.add_argument(
        "--kmeans_path",
        type=str,
        default=(
            "/gpfs/projects/bsc88/speech/mm_s2st/scripts/paraling/"
            "gender_id_hubert/mhubert_base_25hz_cp_mls_cv_sp_fisher_L11_km500.bin"
        ),
        help="Path to kmeans .bin file (float32 centroids row-major)",
    )
    core.add_argument(
        "--kmeans_layer",
        type=int,
        default=11,
        help="Layer index used by kmeans (e.g. 11)",
    )
    core.add_argument(
        "--n_clusters",
        type=int,
        default=500,
        help="Number of clusters in kmeans file",
    )
    core.add_argument(
        "--precompute_quantized",
        action="store_true",
        help="Precompute token ids for all splits to speed up training",
    )
    core.add_argument(
        "--quantizer_cache_dir",
        type=str,
        default="./quant_cache",
        help="Directory to save token id caches if precomputing",
    )

    # ---- Data & I/O ----
    data = p.add_argument_group("Data & I/O")
    data.add_argument(
        "--data_dir",
        type=str,
        default=(
            "/gpfs/projects/bsc88/speech/data/processed_data/"
            "speech_salamandra/cv-corpus-21.0-2025-03-14__gender_id/02_filtered_cv_gender/tsv"
        ),
        help="Directory containing <lang>.train.tsv, <lang>.dev.tsv, <lang>.test.tsv",
    )
    data.add_argument(
        "--lang_prefix",
        type=str,
        default="ca",
        help="Language prefix used in filenames, e.g. 'ca' for ca.train.tsv",
    )
    data.add_argument(
        "--output_dir",
        type=str,
        default="./outputs/mhubert_gender",
        help="Base output directory; subfolders will be created for modes",
    )
    data.add_argument(
        "--num_proc",
        type=int,
        default=1,
        help="Number of processes to use in dataset.map (default: 1)",
    )

    # ---- Training hyperparameters ----
    training = p.add_argument_group("Training")
    training.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=16,
        help="Training batch size per device",
    )
    training.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=16,
        help="Evaluation batch size per device",
    )
    training.add_argument(
        "--learning_rate",
        type=float,
        default=3e-5,
        help="Optimizer learning rate",
    )
    training.add_argument(
        "--num_train_epochs",
        type=int,
        default=3,
        help="Number of training epochs",
    )
    training.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="Maximum gradient norm for gradient clipping (default: 1.0). Set to 0 to disable clipping.",
    )
    training.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="Weight decay for optimizer (default: 0.01)",
    )
    training.add_argument(
        "--lr_scheduler_type",
        type=str,
        default="cosine",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
        help="LR scheduler type for HF Trainer",
    )
    training.add_argument(
        "--warmup_steps",
        type=int,
        default=50,
        help="Number of warmup steps for the LR scheduler (absolute)",
    )
    training.add_argument(
        "--warmup_ratio",
        type=float,
        default=None,
        help="Optional warmup ratio (0-1). If set, overrides warmup_steps calculation.",
    )
    training.add_argument(
        "--evaluation_strategy",
        choices=["no", "steps", "epoch"],
        default="epoch",
        help="Evaluation strategy to adopt during training",
    )
    training.add_argument(
        "--save_strategy",
        choices=["no", "steps", "epoch"],
        default="epoch",
        help="Checkpoint save strategy",
    )
    training.add_argument(
        "--train_fraction",
        type=float,
        default=1.0,
        help="Fraction of train/validation to use for quick runs",
    )
    training.add_argument(
        "--report_to",
        choices=["tensorboard", "wandb", "none"],
        default="tensorboard",
        help="Where to report training metrics",
    )
    training.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    # ---- Optimization & precision ----
    opt = p.add_argument_group("Optimization & precision")
    opt.add_argument(
        "--bf16",
        action="store_true",
        help="Use bf16 if hardware supports it",
    )
    opt.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Enable gradient checkpointing to save memory",
    )
    opt.add_argument(
        "--deepspeed",
        type=str,
        default=None,
        help="Path to deepspeed config or None",
    )

    # ---- Freezing & quantized specifics ----
    freeze = p.add_argument_group("Freezing & quantized")
    freeze.add_argument(
        "--freeze_encoder",
        action="store_true",
        help="Freeze entire encoder (only classification head trains) for continuous mode",
    )
    freeze.add_argument(
        "--freeze_first_n",
        type=int,
        default=0,
        help="Freeze first N transformer layers in encoder (0 = none)",
    )
    freeze.add_argument(
        "--quantized_embedding_trainable",
        action="store_true",
        help="Allow token->vector embeddings to be trainable in quantized mode",
    )

    return p.parse_args()
