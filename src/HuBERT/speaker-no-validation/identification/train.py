import logging
import os

import torch    
from transformers import AutoModelForAudioClassification, Trainer

from utils.collators import HFDatasetWrapper, collate_fn_cont_factory, make_collate_fn_quant
from utils.model_builders import build_continuous_model, build_quantized_model, save_json
from utils.quantizer import precompute_and_cache
from utils.reporting import summarize_run_variant
from utils.utils import compute_metrics
from utils.timing import now, time_block, elapsed
from utils.training_helpers import (
    make_training_args,
    build_trainer,
    evaluate_and_save_test,
    save_common_artifacts,
    process_test_metrics,
)

logger = logging.getLogger(__name__)

# ---------------------
# run_variant_continuous
# ---------------------
def run_variant_continuous(encoded_ds, args, feature_extractor, output_dir):
    logger.info("Running continuous variant -> %s", output_dir)
    t_total = now()

    # build model
    with time_block("build_continuous_model"):
        model, params = build_continuous_model(
            args.hf_model, args.num_labels, args.label2id, args.id2label,
            gradient_checkpointing=args.gradient_checkpointing,
            freeze_encoder=args.freeze_encoder,
            freeze_first_n=args.freeze_first_n
        )
    
    save_json(params, os.path.join(output_dir, "param_counts_continuous.json"))

    # prepare trainer
    collate = collate_fn_cont_factory(args)
    training_args = make_training_args(output_dir, args, fp16=False)
    trainer = build_trainer(
        model,
        training_args,
        encoded_ds["train"],
        encoded_ds.get("validation", None),
        compute_metrics,
        collate,
    )

    # train
    t_train = now()
    trainer.train()
    train_elapsed = elapsed(t_train)

    # optional test evaluation
    if "test" in encoded_ds:
        test_metrics = evaluate_and_save_test(trainer, encoded_ds["test"], output_dir, "continuous", save_json)

    if test_metrics is not None:
        process_test_metrics(trainer, test_metrics, output_dir)

    # save artifacts and summarize
    timings = {"total": elapsed(t_total), "train": train_elapsed}
    save_common_artifacts(
        trainer,
        args,
        output_dir,
        params,
        "continuous",
        save_json,
        summarize_run_variant,
        timings,
        test_metrics,
    )

    logger.info("Continuous finished")

# ---------------------
# run_variant_quantized
# ---------------------
def run_variant_quantized(encoded_ds, args, feature_extractor, centroids, output_dir):
    logger.info("Running quantized variant -> %s", output_dir)
    t_total = now()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load hidden model for feature extraction
    with time_block("load model_hidden"):
        model_hidden = AutoModelForAudioClassification.from_pretrained(args.hf_model).to(device)
        model_hidden.eval()

    # optional precompute and wrap datasets
    train_wr = val_wr = test_wr = None
    if args.precompute_quantized:
        with time_block("precompute_and_cache"):
            caches = precompute_and_cache(encoded_ds, args, feature_extractor, model_hidden, centroids, device)
        if caches.get("train") is not None:
            train_wr = HFDatasetWrapper(encoded_ds["train"], list(map(list, caches["train"])))
        if caches.get("validation") is not None and "validation" in encoded_ds:
            val_wr = HFDatasetWrapper(encoded_ds["validation"], list(map(list, caches["validation"])))
        if caches.get("test") is not None and "test" in encoded_ds:
            test_wr = HFDatasetWrapper(encoded_ds["test"], list(map(list, caches["test"])))

    # build quantized model
    model_q, params_q = build_quantized_model(
        num_clusters=args.n_clusters,
        dim=centroids.shape[1],
        num_labels=args.num_labels,
        centroids=centroids,
        embedding_trainable=args.quantized_embedding_trainable,
    )

    save_json(params_q, os.path.join(output_dir, "param_counts_quantized.json"))

    # prepare trainer (fp16 recommended for quantized)
    collate = make_collate_fn_quant(feature_extractor, model_hidden, centroids, args, device)
    training_args = make_training_args(
        output_dir, args, fp16=True, extra={"dataloader_num_workers": 8, "dataloader_pin_memory": True}
    )

    train_ds = train_wr if train_wr is not None else encoded_ds["train"]
    eval_ds = val_wr if val_wr is not None else encoded_ds.get("validation", None)

    trainer = build_trainer(
        model_q, 
        training_args, 
        train_ds, 
        eval_ds, 
        compute_metrics, 
        collate
    )

    # train
    t_train = now()
    trainer.train()
    train_elapsed = elapsed(t_train)

    # optional test evaluation
    test_ds = test_wr if test_wr is not None else (encoded_ds["test"] if "test" in encoded_ds else None)
    test_metrics = evaluate_and_save_test(trainer, test_ds, output_dir, "quantized", save_json)

    if test_metrics is not None:
        process_test_metrics(trainer, test_metrics, output_dir)

   # save artifacts and summarize
    timings = {"total": elapsed(t_total), "train": train_elapsed}
    save_common_artifacts(
        trainer,
        args,
        output_dir,
        params_q,
        "quantized",
        save_json,
        summarize_run_variant,
        timings,
        test_metrics,
    )

    logger.info("Quantized finished")
