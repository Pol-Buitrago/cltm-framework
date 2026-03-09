"""
Helpers for loading datasets, extracting features, computing k-means tokens,
and simple pooling utilities.
"""

import logging
import os
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from datasets import Audio, load_dataset
from transformers import AutoFeatureExtractor, Wav2Vec2FeatureExtractor

logger = logging.getLogger(__name__)

# Map task name to actual TSV column
task_to_column = {
    "speaker": "client_id"  
}

# ------------------------------
# K-means centroids
# ------------------------------
def load_kmeans_centroids(kmeans_path: str, n_clusters: int) -> np.ndarray:
    """Load k-means centroids from .npy file and validate shape."""
    if not os.path.exists(kmeans_path):
        raise FileNotFoundError(f"kmeans file not found: {kmeans_path}")
    if not kmeans_path.endswith(".npy"):
        raise ValueError(f"Only .npy centroid files are supported: {kmeans_path}")

    cent = np.load(kmeans_path)
    if cent.shape[0] != n_clusters:
        raise ValueError(f"Centroid count mismatch: {cent.shape[0]} != {n_clusters}")
    return cent


# ------------------------------
# Simple padding utilities
# ------------------------------
def pad_input_values_list(arrays: List[List[float]]) -> Dict[str, torch.Tensor]:
    """Pad list of 1D audio arrays -> (input_values, attention_mask)."""
    tensors = [torch.tensor(a, dtype=torch.float32) for a in arrays]
    padded = pad_sequence(tensors, batch_first=True, padding_value=0.0)
    lengths = torch.tensor([t.shape[0] for t in tensors], dtype=torch.long)
    attn = torch.arange(padded.shape[1], device=padded.device).unsqueeze(0) < lengths.unsqueeze(1)
    return {"input_values": padded, "attention_mask": attn.to(dtype=torch.long)}


# ------------------------------
# Quantized precomputation
# ------------------------------
def precompute_token_ids_for_split(
    split_dataset,
    feature_extractor,
    model_for_hidden,
    centroids_torch,
    layer_idx: int,
    batch_size: int,
    device: torch.device,
) -> List[List[int]]:
    """
    Precompute quantized token IDs for all samples in a dataset split.
    Uses feature_extractor -> model_hidden -> nearest centroid per frame.
    """
    model_for_hidden.to(device).eval()
    n_samples = len(split_dataset)
    logger.info("Precomputing token ids (%d samples, batch_size=%d)", n_samples, batch_size)

    token_id_lists = []
    with torch.no_grad():
        for start in range(0, n_samples, batch_size):
            end = min(n_samples, start + batch_size)
            batch_ds = split_dataset.select(range(start, end))
            arrays = [ex["input_values"] for ex in batch_ds]

            fed = feature_extractor(arrays, sampling_rate=feature_extractor.sampling_rate, padding=True, return_tensors="pt")
            input_values = fed["input_values"].to(device)
            attn = fed.get("attention_mask", None)
            if attn is not None:
                attn = attn.to(device)

            out = model_for_hidden(input_values=input_values, attention_mask=attn, output_hidden_states=True, return_dict=True)
            hiddens = out.hidden_states
            if layer_idx >= len(hiddens):
                raise ValueError(f"Layer {layer_idx} out of range (model has {len(hiddens)} layers)")
            layer_hidden = hiddens[layer_idx]  # (B, T, D)
            B, T, D = layer_hidden.shape

            # distance computation
            frames = layer_hidden.view(-1, D)
            x2 = (frames * frames).sum(dim=1, keepdim=True)
            c2 = (centroids_torch * centroids_torch).sum(dim=1).unsqueeze(0)
            xc = frames @ centroids_torch.t()
            dists = x2 + c2 - 2.0 * xc
            nearest = dists.argmin(dim=1).cpu().numpy()

            # regroup into sequences
            if attn is not None:
                am = attn.cpu().numpy()
                idx = 0
                for bi in range(B):
                    valid_len = int(am[bi].sum())
                    seq_ids = nearest[idx : idx + T][:valid_len].tolist()
                    idx += T
                    token_id_lists.append(seq_ids)
            else:
                idx = 0
                for bi in range(B):
                    seq_ids = nearest[idx : idx + T].tolist()
                    idx += T
                    token_id_lists.append(seq_ids)

            if (start // batch_size) % 50 == 0:
                logger.info("Progress: %d/%d samples", end, n_samples)

    assert len(token_id_lists) == n_samples
    return token_id_lists


# ------------------------------
# Pooling helper
# ------------------------------
def mean_pool_hidden_states(hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
    """Mean-pool hidden states along time dimension (mask-aware if provided)."""
    if attention_mask is None:
        return hidden_states.mean(dim=1)
    mask = attention_mask.to(dtype=hidden_states.dtype).unsqueeze(-1)
    summed = (hidden_states * mask).sum(dim=1)
    lens = mask.sum(dim=1).clamp(min=1.0)
    return summed / lens


# ------------------------------
# Dataset loading
# ------------------------------
def load_and_preprocess_dataset(args, feature_extractor):
    """Load train/dev/test TSVs, cast audio, map labels, compute input_values.
    
    Behavior:
    - train and validation use args.lang_prefix as-is (e.g. "en" or "en_ca")
    - test uses only the first component of args.lang_prefix if it contains '_' (e.g. "en_ca" -> test uses "en")
    """
    column_name = task_to_column.get(args.task, args.task) 
    # Determine test prefix: first part before '_' if present, otherwise same as lang_prefix
    if getattr(args, "lang_prefix", None) is None:
        raise RuntimeError("args.lang_prefix is required")
    lang_prefix = args.lang_prefix
    if "_" in lang_prefix:
        test_prefix = lang_prefix.split("_", 1)[0]
    else:
        test_prefix = lang_prefix

    # Build file paths: train/validation use lang_prefix, test uses test_prefix
    data_files = {
        "train": os.path.join(args.data_dir, f"{lang_prefix}.train.tsv"),
        "validation": os.path.join(args.data_dir, f"{lang_prefix}.dev.tsv"),
        "test": os.path.join(args.data_dir, f"{test_prefix}.test.tsv"),
    }

    logger.info("Attempting to load dataset files, train/val use lang_prefix='%s', test uses test_prefix='%s'", lang_prefix, test_prefix)
    logger.info("Loading dataset from: %s", data_files)

    def file_has_data(path):
        """Return True if file exists and contains at least one non-header data line."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                # Contar líneas no vacías
                non_empty_lines = [ln for ln in (l.rstrip("\n\r") for l in f) if ln.strip() != ""]
                # Si solo hay 0 líneas, no hay nada, si hay 1 línea puede ser solo header
                # Consideramos que hay datos útiles si hay más de 1 línea
                return len(non_empty_lines) > 1
        except Exception as e:
            logger.debug("Could not read file %s: %s", path, e)
            return False

    # Construir dict sólo con ficheros existentes y con al menos una fila de datos (aparte del header)
    present_data_files = {}
    for k, v in data_files.items():
        if v is None:
            continue
        if os.path.exists(v) and file_has_data(v):
            present_data_files[k] = v
        else:
            logger.info("Ignoring split '%s' because file not found, empty, or contains no data rows: %s", k, v)

    # ---------------------------------------------------------
    # Si estamos en modo 'speaker', NO cargar el test original,
    # porque queremos usar train como evaluación interna. Evitar
    # cargar el fichero test evita que alguna fase previa lo lea
    # y cause KeyError por etiquetas no presentes en train.
    # ---------------------------------------------------------
    if getattr(args, "task", None) == "speaker":
        # keep only train and (optionally) validation, drop test entirely
        if "train" in present_data_files:
            pf = {"train": present_data_files["train"]}
            if "validation" in present_data_files:
                pf["validation"] = present_data_files["validation"]
                logger.info("Modo 'speaker': cargaremos 'train' y 'validation' (si existe), se IGNORARÁ 'test' original")
            else:
                logger.info("Modo 'speaker': cargaremos sólo 'train' (se IGNORARÁ 'test' original)")
            present_data_files = pf
        else:
            # no hay train -> mantener comportamiento previo para que el error se detecte más arriba
            logger.warning("Modo 'speaker' pero no se encontró split 'train' en %s; manteniendo present_data_files tal cual", args.data_dir)

    if len(present_data_files) == 0:
        raise RuntimeError("No data files found for any split in %s" % args.data_dir)

    # Ahora safe: load_dataset recibirá solo splits con datos reales
    ds = load_dataset("csv", data_files=present_data_files, delimiter="\t")

    # ------------------------------
    # Si estamos en modo 'speaker' forzamos test = train para ignorar el test original
    # y evitar KeyError por labels de test no presentes en train.
    # ------------------------------
    if getattr(args, "task", None) == "speaker":
        logger.info("Modo 'speaker': ignorando test original y usando train como test para evitar labels desconocidos")
        ds["test"] = ds["train"]

    # Si después de todo no hay validation, desactivar evaluación para evitar que Trainer falle
    has_validation = "validation" in ds and len(ds["validation"]) > 0
    if not has_validation:
        logger.info("No validation split available after loading, disabling evaluation during training")
        # Forzamos argumento, si args es mutable esto lo propagará al TrainingArguments que crees
        args.evaluation_strategy = "no"

    if "train" not in ds:
        raise RuntimeError("Train split not found")

    # optional train subset
    if args.train_fraction < 1.0:
        n = int(len(ds["train"]) * args.train_fraction)
        ds["train"] = ds["train"].shuffle(seed=args.seed).select(range(n))

    # label encoding
    labels = sorted(set(ds["train"][column_name]))
    args.label2id = {lab: i for i, lab in enumerate(labels)}
    args.id2label = {i: lab for lab, i in args.label2id.items()}
    args.num_labels = len(labels)
    logger.info("Labels: %s", labels)

    # cast audio column and extract input_values
    ds = ds.cast_column("audio", Audio(sampling_rate=16_000))

    def preprocess(batch):
        arrays = [x["array"] for x in batch["audio"]]
        outs = feature_extractor(arrays, sampling_rate=feature_extractor.sampling_rate, padding=False)
        return {"input_values": outs["input_values"]}

    encoded = ds.map(preprocess, batched=True, num_proc=args.num_proc)
    encoded = encoded.rename_column(column_name, "label")
    return encoded

# ------------------------------
# Feature extractor loading
# ------------------------------
def load_feature_extractor():
    """Load pretrained feature extractor or fallback to basic Wav2Vec2 extractor."""
    try:
        fe = AutoFeatureExtractor.from_pretrained(
            "/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/utils/mhubert-base-25hz"
        )
        logger.info("Loaded feature extractor from local cache")
    except Exception as e:
        fe = Wav2Vec2FeatureExtractor(
            sampling_rate=16000, feature_size=1, padding_value=0.0, do_normalize=True
        )
        logger.warning("Falling back to minimal feature extractor: %s", e)
    return fe
