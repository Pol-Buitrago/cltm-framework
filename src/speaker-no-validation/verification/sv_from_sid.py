#!/usr/bin/env python3
"""
sv_from_sid.py

Speaker verification a partir de un modelo de speaker identification preentrenado.
Entrada: TSV/CSV de pares con columnas (u, v, label) donde u y v son rutas a ficheros de audio
(o índices). Salida: embeddings por sample, pairs_scores.csv con columnas (u, v, label, score),
y métricas sencillas (EER si sklearn está disponible).

Mejoras añadidas:
 - trimming/VAD (librosa si disponible, fallback simple por RMS)
 - torch.inference_mode() + autocast (si CUDA)
 - layer-wise pooling (media de últimas K capas)
 - test-time augmentation (speed perturb) opcional
 - canonicalización de rutas y lookup robusto
"""

from __future__ import annotations
import argparse
import os
import re
import math
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
from tqdm import tqdm
import pandas as pd

from datasets import Dataset, Audio
from transformers import AutoFeatureExtractor, Wav2Vec2FeatureExtractor, AutoModel
from torch.nn.functional import normalize

# Optional metrics
try:
    from sklearn.metrics import roc_curve, auc
except Exception:
    roc_curve = None
    auc = None

# Optional audio libs
try:
    import librosa
    HAS_LIBROSA = True
except Exception:
    HAS_LIBROSA = False

# -------------------------
# Utilidades
# -------------------------
def get_device(prefer_cuda: bool = True) -> torch.device:
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def load_feature_extractor_safe(model_dir: str):
    try:
        fe = AutoFeatureExtractor.from_pretrained(model_dir)
        return fe
    except Exception:
        return Wav2Vec2FeatureExtractor(sampling_rate=16000, feature_size=1, padding_value=0.0, do_normalize=True)

def resolve_base_model(model):
    """
    Devuelve el submódulo base que produce hidden states:
    intenta hubert, wav2vec2, base_model, model.
    """
    for attr in ("hubert", "wav2vec2", "base_model", "model"):
        if hasattr(model, attr):
            return getattr(model, attr)
    return model

def mean_pool(hidden: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Pooling temporal, devuelve (B, D).
    """
    if attention_mask is None:
        return hidden.mean(dim=1)
    mask = attention_mask.to(dtype=hidden.dtype).unsqueeze(-1)
    summed = (hidden * mask).sum(dim=1)
    lengths = mask.sum(dim=1).clamp(min=1.0)
    return summed / lengths

def compute_eer(y_true: np.ndarray, scores: np.ndarray) -> Tuple[float, float]:
    """
    Calcula EER y threshold aproximado, requiere sklearn.
    Devuelve (eer, threshold).
    """
    if roc_curve is None:
        raise RuntimeError("sklearn no disponible, no puedo calcular EER")
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    fnr = 1 - tpr
    abs_diffs = np.abs(fpr - fnr)
    idx = np.argmin(abs_diffs)
    eer = (fpr[idx] + fnr[idx]) / 2.0
    thr = thresholds[idx]
    return float(eer), float(thr)

# -------------------------
# Trimming / VAD (librosa fallback)
# -------------------------
def trim_silence_or_vad(wave: np.ndarray, sr: int, top_db: int = 20) -> np.ndarray:
    """
    Recorta silencios. Si librosa está disponible usa librosa.effects.trim (RMS-based).
    Sino aplica trimming simple por energía.
    """
    if wave is None:
        return wave
    if HAS_LIBROSA:
        try:
            y_trimmed, _ = librosa.effects.trim(wave, top_db=top_db)
            return y_trimmed
        except Exception:
            pass
    # fallback energy-based trim:
    try:
        energy = np.sqrt(np.mean(np.square(wave), axis=-1)) if wave.ndim > 1 else np.sqrt(np.mean(np.square(wave)))
        # if stereo or weird shape, flatten
        w = wave.flatten() if wave.ndim > 1 else wave
        abs_w = np.abs(w)
        thresh = np.percentile(abs_w, 5) + 1e-6
        mask = abs_w > thresh
        if not np.any(mask):
            return w
        idx = np.where(mask)[0]
        start, end = idx[0], idx[-1] + 1
        return w[start:end]
    except Exception:
        return wave

# -------------------------
# TTA helpers (speed perturb using librosa if available)
# -------------------------
def speed_perturb(wave: np.ndarray, sr: int, speed: float) -> np.ndarray:
    if speed == 1.0 or not HAS_LIBROSA:
        return wave
    try:
        # resample to new rate then back to original rate to change speed
        new_sr = int(sr * speed)
        y_sp = librosa.resample(wave, orig_sr=sr, target_sr=new_sr)
        # keep the sampling rate label consistent (feature_extractor will be given original sr)
        return y_sp
    except Exception:
        return wave

# -------------------------
# Extraccion de embeddings
# -------------------------
def extract_embeddings_from_dataset(
    model,
    feature_extractor,
    dataset,
    audio_column: str = "audio",
    batch_size: int = 16,
    device: Optional[torch.device] = None,
    l2_norm: bool = True,
    last_k_layers: int = 4,
    tta_speeds: Optional[List[float]] = None,
    trim_top_db: int = 20,
):
    """
    Recorre dataset (datasets.Dataset) y extrae un embedding por fila.
    Retorna: embeddings (N, D) numpy, meta list de dicts {'idx', 'audio_path'}.
    - last_k_layers: combinar las últimas K capas (requiere output_hidden_states=True).
    - tta_speeds: lista de speed factors para TTA (ej: [1.0, 0.98, 1.02]).
    """
    if device is None:
        device = get_device()

    model = model.to(device)
    model.eval()
    base = resolve_base_model(model)

    embeddings: List[np.ndarray] = []
    meta: List[Dict] = []

    n = len(dataset)
    sr = getattr(feature_extractor, "sampling_rate", 16000)

    # choose amp context manager
    amp_ctx = torch.cuda.amp.autocast if (device.type == "cuda") else lambda: torch.cpu.amp.autocast(enabled=False)
    inference_ctx = torch.inference_mode

    for start in tqdm(range(0, n, batch_size), desc="Extracción batches"):
        end = min(n, start + batch_size)
        batch = dataset.select(range(start, end))
        arrays = []
        audio_paths = []
        raw_arrays = []
        for ex in batch:
            a = ex.get(audio_column)
            if a is None:
                raise RuntimeError(f"No se encontró columna de audio '{audio_column}' en el dataset")
            # datasets.cast_column('audio', Audio(...)) produce dict with 'array' and 'path'
            if isinstance(a, dict) and "array" in a:
                arr = a["array"]
                raw_arrays.append(arr)
                arrays.append(arr)  # default version (may be trimmed below)
                audio_paths.append(a.get("path", None))
            else:
                raise RuntimeError("Formato inesperado en la columna audio, use cast_column(Audio) al crear el dataset")

        # apply trimming per sample, and optionally TTA (we'll handle TTA per sample below)
        # feature_extractor expects list of 1D numpy arrays
        embeddings_batch = []
        for i, wav in enumerate(arrays):
            if wav is None:
                raise RuntimeError(f"Audio vacío en índice {start + i}")
            wav_proc = trim_silence_or_vad(wav, sr, top_db=trim_top_db)
            # TTA: generate embeddings for variants and average
            variants = tta_speeds if tta_speeds and len(tta_speeds) > 0 else [1.0]
            variant_embs = []
            for sp in variants:
                wav_var = speed_perturb(wav_proc, sr, sp) if sp != 1.0 else wav_proc
                fed = feature_extractor([wav_var], sampling_rate=sr, padding=True, return_tensors="pt")
                input_values = fed["input_values"].to(device)
                attn = fed.get("attention_mask", None)
                if attn is not None:
                    attn = attn.to(device)
                with inference_ctx():
                    with amp_ctx():
                        out = base(input_values=input_values, attention_mask=attn, output_hidden_states=True, return_dict=True)
                        # hidden states: tuple (layer0,...,layerN) each (B,T,D)
                        if not hasattr(out, "hidden_states") or out.hidden_states is None:
                            # fallback to last_hidden_state
                            if hasattr(out, "last_hidden_state"):
                                last_h = out.last_hidden_state
                            else:
                                last_h = out[0]
                            pooled = mean_pool(last_h, attn)
                        else:
                            hs = out.hidden_states
                            K = min(last_k_layers, len(hs))
                            layers = hs[-K:]  # last K
                            stack = torch.stack(layers, dim=0)  # (K, B, T, D)
                            mean_layers = stack.mean(dim=0)     # (B, T, D)
                            pooled = mean_pool(mean_layers, attn)
                        if l2_norm:
                            pooled = normalize(pooled, p=2, dim=1)
                        variant_embs.append(pooled.cpu().numpy()[0])
            # average variants
            emb_avg = np.mean(np.stack(variant_embs, axis=0), axis=0)
            embeddings_batch.append(emb_avg)

        # append batch embeddings and meta
        for i, apath in enumerate(audio_paths):
            embeddings.append(embeddings_batch[i])
            meta.append({"idx": start + i, "audio_path": apath})

    embeddings_arr = np.stack(embeddings, axis=0) if len(embeddings) > 0 else np.zeros((0, 0))
    return embeddings_arr, meta

# -------------------------
# Scoring de pares
# -------------------------
def canonicalize_key(k: Any) -> str:
    """
    Normaliza claves de ruta: si es un número, lo devolvemos como int-string; si es ruta, devolver abspath.
    """
    if isinstance(k, (int, np.integer)):
        return str(int(k))
    try:
        ks = str(k)
    except Exception:
        ks = repr(k)
    # si parece índice numérico, devolver su forma entera string
    if re.fullmatch(r"\d+", ks):
        return ks
    try:
        return os.path.abspath(ks)
    except Exception:
        return ks

def score_pairs_and_evaluate(
    pair_df: pd.DataFrame,
    embeddings_lookup: Dict,
    key_u: str,
    key_v: str,
    label_col: str = "label",
):
    """
    Espera pair_df con columnas key_u, key_v, label.
    embeddings_lookup: mapping audio_path (canonicalized str) o index-> embedding numpy (L2 normalizados).
    Devuelve df con columnas u, v, label, score; también (eer, thr, auc) si sklearn disponible.
    """
    rows = []

    # build a set of possible lookup keys for fast access
    # embeddings_lookup keys are expected to be canonicalized already
    for _, r in pair_df.iterrows():
        raw_u = r[key_u]
        raw_v = r[key_v]
        lab = r[label_col] if label_col in pair_df.columns else None

        def resolve(k):
            cand = canonicalize_key(k)
            # try exact match
            if cand in embeddings_lookup:
                return embeddings_lookup[cand]
            # if cand is numeric string, try int key
            try:
                ik = int(cand)
                if ik in embeddings_lookup:
                    return embeddings_lookup[ik]
            except Exception:
                pass
            # try fallback: original string as given
            if str(k) in embeddings_lookup:
                return embeddings_lookup[str(k)]
            raise KeyError(f"No pude resolver key '{k}' (cand: '{cand}') a un embedding. Keys disponibles: {list(embeddings_lookup.keys())[:8]}...")

        emb_u = resolve(raw_u)
        emb_v = resolve(raw_v)
        # embeddings are expected normalized if using cosine via dot
        score = float(np.dot(emb_u, emb_v))
        rows.append({"u": raw_u, "v": raw_v, "label": lab, "score": score})

    df_out = pd.DataFrame(rows)

    eer, thr, auc_val = None, None, None
    if "label" in df_out.columns and roc_curve is not None:
        y_true = df_out["label"].astype(float).values
        scores = df_out["score"].values
        try:
            eer, thr = compute_eer(y_true, scores)
            fpr, tpr, _ = roc_curve(y_true, scores)
            auc_val = auc(fpr, tpr) if auc is not None else None
        except Exception:
            eer, thr, auc_val = None, None, None

    return df_out, eer, thr, auc_val

# -------------------------
# Main
# -------------------------
def main(argv=None):
    p = argparse.ArgumentParser(description="SV: extrae embeddings de un SID y puntúa pares (u, v, label).")
    p.add_argument("--model_dir", required=True, help="Carpeta con el modelo entrenado (from_pretrained directory).")
    p.add_argument("--pairs_file", required=True, help="TSV/CSV con pares: columnas u, v, label. u,v deben ser rutas de audio o índices.")
    p.add_argument("--out_dir", required=True, help="Directorio de salida para embeddings y resultados.")
    p.add_argument("--audio_column", default="audio", help="Nombre de la columna temporal en el dataset de audios (por defecto 'audio').")
    p.add_argument("--batch_size", type=int, default=8, help="Batch size para extracción de embeddings.")
    p.add_argument("--l2_norm", action="store_true", help="Aplicar L2 normalización a embeddings (recomendado).")
    p.add_argument("--device", default=None, help="Device torch opcional, p. ej. 'cuda' o 'cpu'.")
    p.add_argument("--pairs_cols", nargs=3, default=["u", "v", "label"], help="Nombres de columnas en pairs_file: u v label")
    p.add_argument("--last_k_layers", type=int, default=4, help="Número de últimas capas a combinar para pooling.")
    p.add_argument("--tta_speeds", nargs="*", type=float, default=[1.0], help="Lista de speed factors para TTA. Ej: --tta_speeds 1.0 0.98 1.02")
    p.add_argument("--trim_top_db", type=int, default=20, help="top_db para trimming (librosa) o umbral implícito en fallback.")
    args = p.parse_args(argv)

    os.makedirs(args.out_dir, exist_ok=True)

    # leer fichero de pares y extraer rutas únicas
    delim_pairs = "\t" if args.pairs_file.endswith(".tsv") or args.pairs_file.endswith(".tsv.gz") else ","
    pairs_df = pd.read_csv(args.pairs_file, delimiter=delim_pairs)
    key_u, key_v, key_label = args.pairs_cols
    if key_u not in pairs_df.columns or key_v not in pairs_df.columns:
        raise RuntimeError(f"Pairs file {args.pairs_file} no contiene columnas {key_u}/{key_v}")

    # obtener rutas únicas (todas como strings)
    u_list = pairs_df[key_u].astype(str).tolist()
    v_list = pairs_df[key_v].astype(str).tolist()
    unique_paths = sorted(set(u_list + v_list))

    # crear dataset temporal con una fila por path usando datasets.Dataset.from_pandas
    df_audio = pd.DataFrame({"audio": unique_paths})
    ds = Dataset.from_pandas(df_audio)
    # cast column to Audio to allow loading waveforms
    ds = ds.cast_column("audio", Audio(sampling_rate=16_000))

    # cargar extractor y modelo
    fe = load_feature_extractor_safe(args.model_dir)
    try:
        model = AutoModel.from_pretrained(args.model_dir)
    except Exception:
        # fallback razonable: si el checkpoint contiene head de clasificación, AutoModel puede fallar
        from transformers import HubertForSequenceClassification
        model = HubertForSequenceClassification.from_pretrained(args.model_dir)

    device = torch.device(args.device) if args.device else get_device()

    # extraer embeddings
    print("Extrayendo embeddings para", len(ds), "ficheros de audio")
    emb, meta = extract_embeddings_from_dataset(
        model=model,
        feature_extractor=fe,
        dataset=ds,
        audio_column="audio",
        batch_size=args.batch_size,
        device=device,
        l2_norm=args.l2_norm,
        last_k_layers=args.last_k_layers,
        tta_speeds=args.tta_speeds,
        trim_top_db=args.trim_top_db,
    )
    if emb.size == 0:
        raise RuntimeError("No se extrajeron embeddings, revisa las rutas de audio")

    # asegurar normalización L2 en numpy (por si)
    if args.l2_norm and emb.size > 0:
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        emb = emb / norms

    # guardar embeddings y metadata
    emb_path = os.path.join(args.out_dir, "embeddings.npy")
    meta_path = os.path.join(args.out_dir, "embeddings_meta.csv")
    np.save(emb_path, emb)
    pd.DataFrame(meta).to_csv(meta_path, index=False)
    print(f"Guardado embeddings: {emb_path}")
    print(f"Guardado meta: {meta_path}")

    # construir lookup: por path (canonicalizado) y por índice (0-based)
    lookup: Dict = {}
    for i, m in enumerate(meta):
        apath = m.get("audio_path")
        # si audio_path es None, usar unique_paths[i]
        raw_path = apath if apath is not None else unique_paths[i]
        try:
            key_path = os.path.abspath(raw_path)
        except Exception:
            key_path = str(raw_path)
        lookup[key_path] = emb[i]
        lookup[str(i)] = emb[i]   # índice como string
        lookup[i] = emb[i]        # índice como int también usable

    # ahora puntuar pares
    print("Puntuando pares en:", args.pairs_file)
    out_df, eer, thr, auc_val = score_pairs_and_evaluate(pairs_df, lookup, key_u, key_v, key_label)
    pairs_out = os.path.join(args.out_dir, "pairs_scores.csv")
    out_df.to_csv(pairs_out, index=False)
    print(f"Pares puntuados guardados en: {pairs_out}")

    if eer is not None:
        print(f"EER: {eer:.4f}, threshold: {thr:.6f}")
    if auc_val is not None:
        print(f"AUC (ROC): {auc_val:.4f}")
    
    # -------------------------
    # Guardar resumen en CSV por idioma y seed (no crítico)
    # -------------------------
    try:
        # idioma: extraer de pairs_file basename, p.ej. en.test.tsv -> en
        lang = os.path.basename(args.pairs_file).split('.')[0]

        # seed: tomar de variable de entorno si existe (exportada por el launcher)
        seed = os.environ.get("SEED", "NA")

        env_data_dir = os.environ.get("DATA_DIR")  # e.g. /.../subset_002_n6
        if env_data_dir:
            subset_base = os.path.basename(env_data_dir.rstrip("/"))
        else:
            subset_base = os.path.basename(os.path.dirname(args.pairs_file))

        # intentar extraer 'n<digits>'
        m = re.findall(r'n(\d+)', subset_base)
        num_samples = int(m[-1]) if m else float("nan")
        
        # métricas: usar valores calculados (eer, auc_val, thr). Si no existen, NaN
        def fmt(x):
            if x is None:
                return "NaN"
            try:
                return f"{float(x):.6f}"
            except Exception:
                return str(x)

        eer_str = fmt(eer)
        auc_str = fmt(auc_val)
        thr_str = fmt(thr)

        # CSV path (outputs folder consistente con tu repo)
        out_dir_base = "/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs"
        os.makedirs(out_dir_base, exist_ok=True)
        csv_path = os.path.join(out_dir_base, f"{lang}.eer_by_samples_{seed}.csv")

        header = "eer,auc,thr,num_samples\n"
        row = f"{eer_str},{auc_str},{thr_str},{int(num_samples)}\n"

        # crear el fichero con header si no existe, luego anexar una línea
        if not os.path.exists(csv_path):
            with open(csv_path, "w", encoding="utf-8") as fh:
                fh.write(header)
                fh.write(row)
        else:
            with open(csv_path, "a", encoding="utf-8") as fh:
                fh.write(row)

        print(f"[SV] Appended summary to {csv_path}: {row.strip()}")
    except Exception as e:
        # no queremos que un fallo en el logging pare el proceso; solo lo registramos
        print(f"[SV] Warning: no se pudo escribir summary CSV: {e}")

    print("Proceso completado.")

if __name__ == "__main__":
    main()
