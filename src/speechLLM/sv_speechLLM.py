#!/usr/bin/env python3
"""
"""
import os
import argparse
import random
from pathlib import Path
from collections import defaultdict
import math
import time
import warnings
import itertools
import csv
import sys
import traceback
import json

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
import librosa

# ------------------------------------
# Rutas fijas por defecto (ajustables)
# ------------------------------------
DOWNLOAD_ROOT = os.environ.get('SPEECHLLM_DOWNLOAD_ROOT', "/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/models")
SPEECHLLM_LOCAL_DIR = os.path.join(DOWNLOAD_ROOT, "speechllm-2B")

# ------------------------------------
# Utilities (seed, audio)
# ------------------------------------
def set_seed(seed: int = 1234, cudnn_deterministic: bool = True):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass
    try:
        torch.backends.cudnn.deterministic = bool(cudnn_deterministic)
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass
    try:
        if hasattr(torch, "use_deterministic_algorithms"):
            torch.use_deterministic_algorithms(bool(cudnn_deterministic))
    except Exception:
        pass


def load_audio(path, sample_rate=16000, max_seconds=3.0):
    """Carga un fichero de audio y lo normaliza a 1D torch tensor de longitud
    sample_rate*max_seconds (recortando o paddeando con ceros).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Audio not found: {path}")
    try:
        waveform, sr = torchaudio.load(path)
        if waveform.ndim > 1 and waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != sample_rate:
            waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sample_rate)(waveform)
        waveform = waveform.squeeze(0)
        max_len = int(sample_rate * max_seconds)
        if waveform.shape[0] < max_len:
            pad = max_len - waveform.shape[0]
            waveform = torch.nn.functional.pad(waveform, (0, pad))
        else:
            waveform = waveform[:max_len]
        return waveform.float()
    except Exception as e:
        try:
            y, sr = librosa.load(path, sr=sample_rate, mono=True)
            max_len = int(sample_rate * max_seconds)
            if y.shape[0] < max_len:
                y = np.pad(y, (0, max_len - y.shape[0]))
            else:
                y = y[:max_len]
            return torch.from_numpy(y).float()
        except Exception as e2:
            raise RuntimeError(f"Failed to load audio {path}: {e} / {e2}")


# ------------------------
# Dataset for single utterances (raw waveform)
# ------------------------
class SingleUtteranceDataset(torch.utils.data.Dataset):
    """
    Expects TSV with columns: lang	client_id	audio	(optional ...)
    Returns raw waveform (1D torch tensor) and integer label.
    """
    def __init__(self, tsv_path, max_seconds=3.0, sample_rate=16000):
        self.tsv_path = tsv_path
        self.max_seconds = max_seconds
        self.sample_rate = sample_rate
        client2files = defaultdict(list)
        with open(tsv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='	')
            first = next(reader, None)
            if first is None:
                raise RuntimeError(f"TSV {tsv_path} vacío")
            if any(c.strip().lower() in ('lang', 'client_id', 'client', 'speaker', 'audio') for c in first):
                pass
            else:
                reader = itertools.chain([first], reader)
            for parts in reader:
                if not parts or all(not p.strip() for p in parts):
                    continue
                if len(parts) < 3:
                    continue
                client = parts[1].strip()
                audio = parts[2].strip()
                if client == '' or audio == '':
                    continue
                client2files[client].append(audio)
        self.clients = sorted([c for c, files in client2files.items() if len(files) >= 1])
        if len(self.clients) < 2:
            raise ValueError("Need at least two clients with >=1 utterance each")
        self.label_map = {c: i for i, c in enumerate(self.clients)}
        self.samples = []
        for c, files in client2files.items():
            if c not in self.label_map:
                continue
            lab = self.label_map[c]
            for p in files:
                self.samples.append((p, lab))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        wav = load_audio(path, sample_rate=self.sample_rate, max_seconds=self.max_seconds)
        return wav, torch.tensor(label, dtype=torch.long)


def collate_single(batch):
    waves = [b[0] for b in batch]
    labs = torch.stack([b[1] for b in batch], dim=0)
    max_len = max([w.shape[0] for w in waves])
    padded = []
    for w in waves:
        if w.shape[0] < max_len:
            pad = max_len - w.shape[0]
            w = torch.nn.functional.pad(w, (0, pad))
        padded.append(w.unsqueeze(0))
    waves_tensor = torch.cat(padded, dim=0)  # (B, T)
    return waves_tensor, labs


# ------------------------
# LLM embedding extraction utilities (SpeechLLM ONLY)
# ------------------------
class LLMEmbedder:
    """
    Wrapper that USES EXCLUSIVELY SpeechLLM (local repository or local path).
    Si no puede cargar SpeechLLM, aborta con explicaciones para el usuario.

    Parche clave: separa tokenizer de texto (LLM base) del repo SpeechLLM. Requiere
    --text_tokenizer para poder usar prompting/instrucciones.
    """
    def __init__(self, device, model_name=None, sample_rate=16000, local_speechllm_dir=None, text_tokenizer: str = None,
                 pre_instruction: str = None, post_instruction: str = None, out_prefix: str = None, allow_remote: bool = False):
        self.device = device
        self.sample_rate = sample_rate
        self.use_speechllm = False
        self.speechllm_tokenizer = None  # deprecated: we do not load tokenizer from speechllm repo
        self.speechllm_model = None
        self.text_tokenizer = None  # tokenizer used for textual prompts (LLM base tokenizer)
        self.out_dim = None

        model_name = model_name or ''
        self.local_speechllm_dir = local_speechllm_dir
        self.text_tokenizer_name = text_tokenizer
        self.allow_remote = allow_remote

        # Default instruction templates (puedes sobreescribir pasando args)
        self.pre_instruction = pre_instruction or (
            "Instruction:Give me the following information about the audio "
            "[SpeechActivity, Transcript, Gender, Age, Emotion, Accent]Input:<speech>"
        )
        self.post_instruction = post_instruction or "</speech>Output:"
        self.out_prefix = out_prefix or "<s>"

        # Decide whether to enforce offline mode: only force offline if user DID NOT pass --allow_remote
        local_files_only = False
        if self.local_speechllm_dir is not None and os.path.isdir(self.local_speechllm_dir) and not self.allow_remote:
            os.environ['TRANSFORMERS_OFFLINE'] = '1'
            local_files_only = True
            print("[LLMEmbedder] TRANSFORMERS_OFFLINE forced (local repo provided and --allow_remote not used).")
        else:
            # don't forcibly set TRANSFORMERS_OFFLINE; respect environment if pre-set
            if os.environ.get('TRANSFORMERS_OFFLINE') == '1':
                local_files_only = True
                print("[LLMEmbedder] TRANSFORMERS_OFFLINE is set in environment.")

        tried_paths = []
        load_errors = []

        # Helper: make local repo importable by adding candidate paths to sys.path
        def _ensure_repo_importable(repo_dir):
            inserted = []
            candidates = []
            if repo_dir:
                candidates.append(repo_dir)
                candidates.append(os.path.dirname(repo_dir))
                # also check typical subfolder names used by pretrained repos
                candidates.append(os.path.join(repo_dir, 'transformers_modules'))
                candidates.append(os.path.join(repo_dir, 'src'))
            for c in candidates:
                if c and os.path.isdir(c) and c not in sys.path:
                    sys.path.insert(0, c)
                    inserted.append(c)
            return inserted

        # Helper: try resolve HF id -> local path
        def _resolve_local_hf_ref(name_or_path):
            """
            If name_or_path is already a valid local path, return it.
            If it looks like an HF id (contains '/'), try to find a local folder with
            the same basename under a set of candidate roots (local_speechllm_dir, DOWNLOAD_ROOT, parent).
            If not found and allow_remote==True, return original name_or_path (will allow remote fetch).
            Otherwise raise FileNotFoundError.
            """
            # if already a local path
            if name_or_path is None:
                return name_or_path
            if os.path.isdir(name_or_path) or os.path.isfile(name_or_path):
                return name_or_path
            # common HF id check
            if '/' not in str(name_or_path):
                # not clearly HF id; return as-is
                return name_or_path
            basename = os.path.basename(name_or_path)
            candidates = []
            # direct sibling inside local speechllm dir
            if self.local_speechllm_dir:
                candidates.append(os.path.join(self.local_speechllm_dir, basename))
            # inside download root
            candidates.append(os.path.join(DOWNLOAD_ROOT, basename))
            # parent of speechllm dir
            if self.local_speechllm_dir:
                candidates.append(os.path.join(os.path.dirname(self.local_speechllm_dir), basename))
            # fallback: search DOWNLOAD_ROOT shallow
            try:
                for root in [self.local_speechllm_dir, DOWNLOAD_ROOT, os.path.dirname(self.local_speechllm_dir) if self.local_speechllm_dir else None]:
                    if not root or not os.path.isdir(root):
                        continue
                    for entry in os.listdir(root):
                        if entry == basename:
                            candidates.append(os.path.join(root, entry))
            except Exception:
                pass

            for c in candidates:
                if c and os.path.exists(c):
                    return c
            # If we get here: not found locally
            if self.allow_remote:
                return name_or_path
            raise FileNotFoundError(f"No local copy found for HF id '{name_or_path}'. Tried: {candidates}." +
                                    "Download it on a machine with internet and place it under one of those paths, or run the script with --allow_remote.")

        # Helper: monkeypatch transformers *before* importing model code that may call AutoProcessor.from_pretrained
        def _monkeypatch_transformers_for_local_resolution():
            try:
                import transformers
                # Safety patch: some custom model classes (like SpeechLLMModel) may not
                # define attributes expected by current transformers (e.g. all_tied_weights_keys).
                # Patch PreTrainedModel.mark_tied_weights_as_initialized to be robust.
                try:
                    from transformers.modeling_utils import PreTrainedModel
                    if not hasattr(PreTrainedModel, '_orig_mark_tied_weights'):
                        PreTrainedModel._orig_mark_tied_weights = PreTrainedModel.mark_tied_weights_as_initialized
                        def _safe_mark_tied_weights(self):
                            # Ensure attribute exists to avoid AttributeError in older custom classes
                            if not hasattr(self, 'all_tied_weights_keys'):
                                try:
                                    self.all_tied_weights_keys = {}
                                except Exception:
                                    # best-effort: if we cannot set, continue silently
                                    pass
                            try:
                                return PreTrainedModel._orig_mark_tied_weights(self)
                            except Exception:
                                # If original method fails for any reason, ignore to allow loading to continue
                                return None
                        PreTrainedModel.mark_tied_weights_as_initialized = _safe_mark_tied_weights
                except Exception:
                    # If transformers internals are unavailable, we skip this safety patch.
                    pass
            except Exception:
                return None

            patched = {}

            def _patch_attr(attr_name):
                if not hasattr(transformers, attr_name):
                    return
                cls = getattr(transformers, attr_name)
                if not hasattr(cls, 'from_pretrained'):
                    return
                orig = cls.from_pretrained

                def wrapper(name_or_path, *args, **kwargs):
                    # resolve possible mapping
                    try:
                        resolved = _resolve_local_hf_ref(name_or_path)
                    except FileNotFoundError:
                        # try calling original only if allow_remote
                        if self.allow_remote:
                            resolved = name_or_path
                        else:
                            raise
                    # enforce local files only unless allow_remote
                    if not self.allow_remote:
                        kwargs['local_files_only'] = True
                    # call original with resolved
                    return orig(resolved, *args, **kwargs)

                setattr(cls, 'from_pretrained', staticmethod(wrapper))
                patched[attr_name] = orig

            # Patch likely used classes
            for an in ('AutoProcessor', 'AutoTokenizer', 'AutoFeatureExtractor', 'AutoConfig'):
                _patch_attr(an)

            return patched

        # 1) Intentar cargar SpeechLLM desde directorio local explícito (preferido)
        if self.local_speechllm_dir and os.path.isdir(self.local_speechllm_dir):
            tried_paths.append(self.local_speechllm_dir)
            # add to sys.path to make package importable (fix for Unrecognized configuration class)
            inserted = _ensure_repo_importable(self.local_speechllm_dir)
            if inserted:
                print(f"[LLMEmbedder] Añadidos a sys.path para import dinámico: {inserted}")

            # monkeypatch transformers so internal AutoProcessor.from_pretrained resolves HF ids to local folders
            patched = _monkeypatch_transformers_for_local_resolution()

            try:
                print(f"[LLMEmbedder] intentando cargar SpeechLLM desde directorio local '{self.local_speechllm_dir}' (local_files_only={local_files_only})...")
                from transformers import AutoModel
                # trust_remote_code True para permitir clases custom en repo
                self.speechllm_model = AutoModel.from_pretrained(self.local_speechllm_dir, trust_remote_code=True, local_files_only=local_files_only)
                self.speechllm_model = self.speechllm_model.to(self.device)
                self.speechllm_model.eval()
                self.use_speechllm = True
                # intenta extraer hidden_size desde config
                self.out_dim = getattr(self.speechllm_model.config, 'hidden_size', None) or getattr(self.speechllm_model.config, 'd_model', None) or 1024
                print(f"[LLMEmbedder] SpeechLLM cargado desde local, out_dim ≈ {self.out_dim}")
            except Exception as e:
                load_errors.append((self.local_speechllm_dir, str(e)))
                warnings.warn(f"Fallo cargando SpeechLLM desde local '{self.local_speechllm_dir}': {e}{traceback.format_exc()}")
            finally:
                # restore patched methods to avoid side effects
                try:
                    import transformers
                    if patched:
                        for k, orig in patched.items():
                            if hasattr(transformers, k) and hasattr(getattr(transformers, k), 'from_pretrained'):
                                setattr(getattr(transformers, k), 'from_pretrained', orig)
                except Exception:
                    pass

        # 2) Si model_name se pasó y puede apuntar a una carpeta local, intentar cargar desde model_name con local_files_only
        if (not self.use_speechllm) and model_name:
            tried_paths.append(model_name)
            inserted2 = _ensure_repo_importable(model_name)
            if inserted2:
                print(f"[LLMEmbedder] Añadidos a sys.path para import dinámico: {inserted2}")

            patched2 = _monkeypatch_transformers_for_local_resolution()
            try:
                print(f"[LLMEmbedder] intentando cargar SpeechLLM desde '{model_name}' (local_files_only={local_files_only})...")
                from transformers import AutoModel
                self.speechllm_model = AutoModel.from_pretrained(model_name, trust_remote_code=True, local_files_only=local_files_only)
                self.speechllm_model = self.speechllm_model.to(self.device)
                self.speechllm_model.eval()
                self.use_speechllm = True
                self.out_dim = getattr(self.speechllm_model.config, 'hidden_size', None) or getattr(self.speechllm_model.config, 'd_model', None) or 1024
                print(f"[LLMEmbedder] SpeechLLM cargado desde '{model_name}', out_dim ≈ {self.out_dim}")
            except Exception as e:
                load_errors.append((model_name, str(e)))
                warnings.warn(f"Fallo cargando SpeechLLM '{model_name}' con local_files_only: {e}{traceback.format_exc()}")
            finally:
                try:
                    import transformers
                    if patched2:
                        for k, orig in patched2.items():
                            if hasattr(transformers, k) and hasattr(getattr(transformers, k), 'from_pretrained'):
                                setattr(getattr(transformers, k), 'from_pretrained', orig)
                except Exception:
                    pass

        # 3) Tokenizer de texto (para prompts/instrucciones): cargar desde self.text_tokenizer_name
        # Nota: forzamos un fallback explícito a la ruta en el cluster indicada por el usuario
        # para evitar fallos en entornos sin red si el flag --text_tokenizer no fue pasado.
        try:
            from transformers import AutoTokenizer
            # If given a HF id or local path, try to resolve to local path; otherwise fall back to the explicit cluster path
            tk_name = self.text_tokenizer_name
            # Ruta explícita facilitada por el usuario (fallback seguro)
            fallback_tk = "/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/models/TinyLlama-1.1B-Chat-v1.0"

            if tk_name and os.path.isdir(tk_name):
                tk_name_resolved = tk_name
            else:
                # try resolve via helper if a HF id was provided
                try:
                    tk_name_resolved = _resolve_local_hf_ref(tk_name) if tk_name else fallback_tk
                except FileNotFoundError:
                    if self.allow_remote:
                        tk_name_resolved = tk_name or fallback_tk
                    else:
                        # last-resort: use the explicit fallback path provided by the user
                        tk_name_resolved = fallback_tk

            # respectar TRANSFORMERS_OFFLINE si fue activado
            local_files_only_tk = True if (os.path.isdir(tk_name_resolved) or os.environ.get('TRANSFORMERS_OFFLINE') == '1') else False
            print(f"[LLMEmbedder] cargando tokenizer de texto '{tk_name_resolved}' (local_files_only={local_files_only_tk})...")
            self.text_tokenizer = AutoTokenizer.from_pretrained(tk_name_resolved, use_fast=False, local_files_only=local_files_only_tk)
            print("[LLMEmbedder] Tokenizer de texto cargado.")
        except Exception as e:
            load_errors.append((f"tokenizer:{self.text_tokenizer_name}", str(e)))
            warnings.warn(f"Fallo cargando tokenizer de texto '{self.text_tokenizer_name}': {e}{traceback.format_exc()}")

        # Si no cargó el modelo o el tokenizer (cuando se requieren), abortar con mensaje claro
        if not self.use_speechllm:
            msg_lines = [
                "No se pudo cargar SpeechLLM (el script está configurado para USAR SOLO SpeechLLM).",
                "Intentos:",
            ]
            for p in tried_paths:
                msg_lines.append(f" - {p}")
            msg_lines.append("")
            msg_lines.append("Errores recogidos durante los intentos:")
            for p, err in load_errors:
                msg_lines.append(f" * {p} : {err}")
            msg_lines.append("")
            msg_lines.append("Soluciones posibles:")
            msg_lines.append("  - Asegúrate de que la carpeta del repo de SpeechLLM (ej: 'speechllm-2B') está disponible localmente.")
            msg_lines.append("  - Añade la carpeta del repo (o su padre) al PYTHONPATH antes de ejecutar, p.e.:")
            msg_lines.append("      export PYTHONPATH=/ruta/al/speechllm-2B:$PYTHONPATH")
            msg_lines.append("  - O ejecuta `python sv_siamese_speechllm_offline.py --prepare_models` en una máquina con Internet para descargar el repo y luego copia la carpeta al servidor sin red.")
            msg_lines.append("  - Si sigues teniendo el error, comprueba que dentro de la carpeta local existe la estructura de paquete (p.e. 'transformers_modules/...') y que la carpeta se puede importar desde Python.")
            msg_lines.append("  - Si tienes red y quieres permitir descargas remotas, ejecuta con --allow_remote o quita TRANSFORMERS_OFFLINE=1 del entorno.")
            msg = "".join(msg_lines)
            raise RuntimeError(msg)

        # Si el usuario quiere instrucciones pero no hay tokenizer de texto, lanzar error
        if (self.pre_instruction or self.post_instruction or self.out_prefix) and (self.text_tokenizer is None):
            raise RuntimeError("Se requiere --text_tokenizer (ruta local o HF id) para usar prompting/instrucciones. Proporciona --text_tokenizer /ruta/al/tokenizer_local o descarga uno en la máquina con Internet y copia la carpeta al servidor.")

    def embed_batch(self, waveforms: list):
        """waveforms: list of 1D torch tensors (float), sampling_rate self.sample_rate
        returns: numpy array (B, D), L2-normalized

        Nota: este método asume que SpeechLLM se cargó correctamente y que, si se usan instrucciones,
        self.text_tokenizer también fue cargado.
        """
        if not self.use_speechllm:
            raise RuntimeError("SpeechLLM no está disponible en LLMEmbedder.embed_batch (carga fallida al iniciar).")

        if len(waveforms) == 0:
            return np.zeros((0, self.out_dim if self.out_dim is not None else 0), dtype=np.float32)

        try:
            # pad/stack waveforms to same length
            waves_list = [w.cpu() if isinstance(w, torch.Tensor) else torch.tensor(w, dtype=torch.float32) for w in waveforms]
            max_len = max([w.shape[0] for w in waves_list])
            padded = []
            for w in waves_list:
                if w.shape[0] < max_len:
                    pad = max_len - w.shape[0]
                    w = torch.nn.functional.pad(w, (0, pad))
                padded.append(w)
            waves = torch.stack(padded, dim=0).to(self.device)  # (B, T)

            # Si tenemos tokenizer de texto, preparar prompts tokenizados (instrucción)
            if self.text_tokenizer is not None:
                pre = self.pre_instruction
                post = self.post_instruction
                outp = self.out_prefix
                pre_ids = self.text_tokenizer(pre, padding="do_not_pad", return_tensors='pt',
                                             truncation=False, add_special_tokens=False).get("input_ids")
                post_ids = self.text_tokenizer(post, padding="do_not_pad", return_tensors='pt',
                                              truncation=False, add_special_tokens=False).get("input_ids")
                out_ids = self.text_tokenizer(outp, padding="do_not_pad", return_tensors='pt',
                                             truncation=False, add_special_tokens=False).get("input_ids")

                # move to device and repeat to batch size if needed
                pre_ids = pre_ids.to(self.device)
                post_ids = post_ids.to(self.device)
                out_ids = out_ids.to(self.device)

                bsz = waves.shape[0]
                if pre_ids.shape[0] != bsz:
                    pre_ids = pre_ids.repeat(bsz, 1)
                if post_ids.shape[0] != bsz:
                    post_ids = post_ids.repeat(bsz, 1)
                if out_ids.shape[0] != bsz:
                    out_ids = out_ids.repeat(bsz, 1)

                # call the model.encode as in the SpeechLLM repo (returns combined_embeds, atts, label_ids)
                with torch.no_grad():
                    enc_out = self.speechllm_model.encode(waves, pre_ids, post_ids, out_ids)
            else:
                # Sin tokenizer: llamar a encode solo con audio (algunos backends pueden soportarlo)
                with torch.no_grad():
                    enc_out = self.speechllm_model.encode(waves)

            if isinstance(enc_out, (tuple, list)) and len(enc_out) >= 1:
                combined = enc_out[0]
            else:
                combined = enc_out
            # combined is usually (B, T, H) or (B, H)
            if torch.is_tensor(combined):
                if combined.dim() == 3:
                    emb_t = combined.mean(dim=1).cpu()  # mean pool over time -> (B, H)
                elif combined.dim() == 2:
                    emb_t = combined.cpu()
                else:
                    # flatten
                    emb_t = combined.reshape(combined.shape[0], -1).cpu()
            else:
                emb_t = torch.tensor(np.array(combined)).float()
                if emb_t.dim() == 3:
                    emb_t = emb_t.mean(dim=1)
            emb = emb_t.numpy()
            # L2-normalize
            norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12
            emb = emb / norms
            return emb.astype(np.float32)
        except Exception as e:
            raise RuntimeError(f"Error extrayendo embeddings via SpeechLLM: {e}{traceback.format_exc()}")


# ------------------------
# Small pair-classifier and pair generation helper
# ------------------------
class PairClassifier(nn.Module):
    """Simple MLP that takes feature vector constructed from two embeddings.
    We'll use features = [abs(e1-e2), e1*e2] (dimension 2*D)
    """
    def __init__(self, emb_dim, hidden=512, dropout=0.1):
        super().__init__()
        in_dim = emb_dim * 2
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, max(hidden//2, 32)),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(max(hidden//2, 32), 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def build_train_pair_dataset(train_dataset, embed_cache, pairs_per_epoch=20000, rng_seed=12345):
    """Genera pares (a,b,label) a partir del train_dataset.samples y embeddings precomputadas.
    Filtra pares para los que existan embeddings en embed_cache.
    """
    label2paths = defaultdict(list)
    for p, lab in train_dataset.samples:
        label2paths[int(lab)].append(p)
    labels = list(label2paths.keys())
    rng = random.Random(rng_seed)
    pairs = []
    if len(labels) < 2:
        return pairs
    for _ in range(pairs_per_epoch):
        if rng.random() < 0.5:
            # positive pair (try)
            choice = rng.choice(labels)
            files = label2paths[choice]
            if len(files) >= 2:
                a, b = rng.sample(files, 2)
                pairs.append((a, b, 1))
            else:
                # fallback to negative
                la, lb = rng.sample(labels, 2)
                a = rng.choice(label2paths[la])
                b = rng.choice(label2paths[lb])
                pairs.append((a, b, 0))
        else:
            # negative
            la, lb = rng.sample(labels, 2)
            a = rng.choice(label2paths[la])
            b = rng.choice(label2paths[lb])
            pairs.append((a, b, 0))
    # filter pairs with embeddings
    filtered = [(a,b,l) for (a,b,l) in pairs if (a in embed_cache and b in embed_cache)]
    return filtered


# ------------------------
# Evaluation helpers
# ------------------------
def eer_from_scores(labels, scores):
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores).astype(float)
    order = np.argsort(scores)
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    if sorted_scores.size == 0:
        return 0.0, 0.0
    thresholds = np.concatenate(([sorted_scores[0] - 1e-6],
                                 (sorted_scores[:-1] + sorted_scores[1:]) / 2.0,
                                 [sorted_scores[-1] + 1e-6]))
    FP = []
    FN = []
    P = int((labels == 1).sum())
    N = int((labels == 0).sum())
    for th in thresholds:
        preds = (scores >= th).astype(int)
        fp = int(((preds == 1) & (labels == 0)).sum())
        fn = int(((preds == 0) & (labels == 1)).sum())
        FP.append(fp / max(1, N))
        FN.append(fn / max(1, P))
    FP = np.array(FP)
    FN = np.array(FN)
    idx = np.argmin(np.abs(FP - FN))
    eer = float((FP[idx] + FN[idx]) / 2.0)
    return eer, float(thresholds[idx])


# ------------------------
# Compute embeddings and main flow
# ------------------------
def compute_embeddings_llm(paths, embedder: LLMEmbedder, device, sample_rate=16000, max_seconds=3.0, batch_size=16):
    emb_cache = {}
    keys = list(dict.fromkeys(paths))
    for i in range(0, len(keys), batch_size):
        batch_paths = keys[i:i+batch_size]
        waves = []
        for p in batch_paths:
            try:
                wav = load_audio(p, sample_rate=sample_rate, max_seconds=max_seconds)
            except Exception as e:
                warnings.warn(f"Failed loading audio {p}: {e}")
                wav = torch.zeros(int(sample_rate * max_seconds), dtype=torch.float32)
            waves.append(wav)
        try:
            embs = embedder.embed_batch(waves)  # (B, D)
            if embs is None:
                raise RuntimeError("embed_batch devolvió None")
            if embs.shape[0] != len(batch_paths):
                if embs.shape[0] < len(batch_paths):
                    padrows = len(batch_paths) - embs.shape[0]
                    embs = np.vstack([embs, np.zeros((padrows, embs.shape[1]), dtype=np.float32)])
                else:
                    embs = embs[:len(batch_paths)]
        except Exception as e:
            warnings.warn(f"embed_batch falló para lote {batch_paths}: {e}{traceback.format_exc()}")
            dim = embedder.out_dim or 1024
            embs = np.zeros((len(batch_paths), dim), dtype=np.float32)

        for k, e in zip(batch_paths, embs):
            emb_cache[k] = e
    return emb_cache


def train_and_eval(args):
    set_seed(args.seed)
    device_str = getattr(args, "device", None)
    device = torch.device(device_str if device_str else ('cuda' if torch.cuda.is_available() else 'cpu'))
    print("Device:", device)

    # Cargar dataset de entrenamiento (si se proporciona)
    train_dataset = None
    if args.train_tsv:
        train_dataset = SingleUtteranceDataset(args.train_tsv, max_seconds=args.max_seconds, sample_rate=args.sample_rate)
        num_classes = len(train_dataset.label_map)
        print(f"Train dataset: {len(train_dataset)} samples, {num_classes} speakers. Se entrenará una cabeza sobre embeddings (si hay suficientes datos).")
    else:
        print("No se proporcionó train_tsv; se usará scoring directo por similitud coseno (sin entrenamiento).")

    model_name = getattr(args, 'llm_model', None)
    _local_speechllm_dir = getattr(args, 'local_speechllm_dir', SPEECHLLM_LOCAL_DIR)

    embedder = LLMEmbedder(
        device=device,
        model_name=model_name,
        sample_rate=getattr(args, "sample_rate", 16000),
        local_speechllm_dir=_local_speechllm_dir,
        text_tokenizer=getattr(args, 'text_tokenizer', None),
        pre_instruction=getattr(args, 'pre_instruction', None),
        post_instruction=getattr(args, 'post_instruction', None),
        out_prefix=getattr(args, 'out_prefix', None),
        allow_remote=getattr(args, 'allow_remote', False)
    )

    # Si existe train_dataset, precomputar embeddings de train y entrenar una cabeza ligera
    clf = None
    if train_dataset is not None:
        print("Computing embeddings for train set using SpeechLLM... (this may take a while)")
        train_paths = [p for p, _ in train_dataset.samples]
        train_paths_unique = list(dict.fromkeys(train_paths))
        train_emb_cache = compute_embeddings_llm(train_paths_unique, embedder, device,
                                                 sample_rate=args.sample_rate, max_seconds=args.max_seconds, batch_size=args.batch_size)

        # Construir pares de entrenamiento
        print(f"Generating up to {args.pairs_per_epoch} training pairs...")
        train_pairs = build_train_pair_dataset(train_dataset, train_emb_cache, pairs_per_epoch=args.pairs_per_epoch, rng_seed=args.seed)
        print(f"Train pairs generated: {len(train_pairs)} (after filtering pairs without embeddings)")

        if len(train_pairs) > 0:
            emb_dim = next(iter(train_emb_cache.values())).shape[0]
            clf = PairClassifier(emb_dim, hidden=512, dropout=0.1).to(device)
            opt = optim.Adam(clf.parameters(), lr=args.lr)
            criterion = nn.BCEWithLogitsLoss()

            # training loop
            bsz = 256
            epochs = max(1, args.epochs)
            for epoch in range(epochs):
                random.shuffle(train_pairs)
                losses = []
                for i in range(0, len(train_pairs), bsz):
                    batch = train_pairs[i:i+bsz]
                    xs = []
                    ys = []
                    for a,b,l in batch:
                        ea = train_emb_cache[a]
                        eb = train_emb_cache[b]
                        # feature: abs diff and elementwise product
                        feat = np.concatenate([np.abs(ea - eb), ea * eb], axis=0)
                        xs.append(feat)
                        ys.append(l)
                    x = torch.tensor(np.stack(xs), dtype=torch.float32, device=device)
                    y = torch.tensor(ys, dtype=torch.float32, device=device)
                    logits = clf(x)
                    loss = criterion(logits, y)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    losses.append(loss.item())
                print(f"[classifier train] epoch {epoch+1}/{epochs}, loss={np.mean(losses):.4f}")

            # guardar clasificador
            out_dir = Path(args.out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            clf_path = out_dir / "pair_classifier.pt"
            try:
                torch.save({'model_state_dict': clf.state_dict(), 'emb_dim': emb_dim}, str(clf_path))
                print(f"Saved trained classifier to {clf_path}")
            except Exception as e:
                warnings.warn(f"No se pudo guardar clasificador: {e}")
        else:
            print("No se generaron pares de entrenamiento útiles; se omitirá entrenamiento y se usará scoring por similitud coseno.")
    else:
        train_emb_cache = {}

    # ------------------------
    # Preparar pares de test y compute embeddings
    # ------------------------
    if not args.test_tsv:
        raise RuntimeError("Se requiere --test_tsv con pares u\tv\tlabel para evaluar.")

    print("Loading test pairs...")
    pairs = []
    bad_lines = 0

    def looks_like_header(row):
        if len(row) < 3:
            return False
        candidates = [row[0].strip().lower(), row[1].strip().lower(), row[2].strip().lower()]
        return ('u' in candidates[0] or 'v' in candidates[1] or 'label' in candidates[2]
                or 'path' in candidates[0] or 'path' in candidates[1] or 'audio' in candidates[2])

    def parse_label(s):
        s = s.strip().lower()
        if s == '':
            raise ValueError("Etiqueta vacía")
        try:
            v = int(float(s))
            return 1 if v != 0 else 0
        except Exception:
            pass
        if s in ('1', 'true', 'yes', 'y', 'same', 'target', 'positive'):
            return 1
        if s in ('0', 'false', 'no', 'n', 'different', 'diff', 'non-target', 'negative'):
            return 0
        raise ValueError(f"Unrecognized label string: '{s}'")

    with open(args.test_tsv, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='	')
        first_row = None
        for row in reader:
            if row and any(cell.strip() for cell in row):
                first_row = row
                break
        if first_row is None:
            raise RuntimeError(f"Test TSV {args.test_tsv} está vacío o no tiene líneas válidas")
        if looks_like_header(first_row):
            rows_iter = reader
        else:
            rows_iter = itertools.chain([first_row], reader)
        for row_idx, row in enumerate(rows_iter, start=1):
            if not row or not any(cell.strip() for cell in row):
                continue
            if len(row) < 3:
                bad_lines += 1
                if bad_lines <= 10:
                    print(f"[Warning] fila test tsv malformada (menos de 3 columnas): {row_idx}, contenido={row}")
                continue
            u = row[0].strip()
            v = row[1].strip()
            lab_str = row[2].strip()
            try:
                lab = parse_label(lab_str)
            except Exception as e:
                bad_lines += 1
                if bad_lines <= 20:
                    print(f"[Warning] no se pudo parsear etiqueta en línea {row_idx}: '{lab_str}', error: {e}. Línea omitida.")
                continue
            pairs.append((u, v, lab))
    if len(pairs) == 0:
        raise RuntimeError(f"No se cargó ningún par válido desde {args.test_tsv}")
    print(f"Cargados {len(pairs)} pares de test (skipped {bad_lines} líneas problemáticas).")

    all_paths = [p for p, q, _ in pairs] + [q for p, q, _ in pairs]
    unique_paths = []
    seen = set()
    for p in all_paths:
        if p not in seen:
            unique_paths.append(p)
            seen.add(p)

    print("Computing embeddings with SpeechLLM (batching)...")
    emb_cache = compute_embeddings_llm(unique_paths, embedder, device, sample_rate=args.sample_rate, max_seconds=args.max_seconds, batch_size=args.batch_size)

    # Merge train embeddings cache (if any) to avoid recomputing
    if train_dataset is not None and 'train_emb_cache' in locals():
        for k, v in train_emb_cache.items():
            if k not in emb_cache:
                emb_cache[k] = v

    scores = []
    labels = []

    # If we trained a classifier, use it to score; otherwise use cosine similarity
    use_clf = (clf is not None)
    if use_clf:
        clf.eval()

    for u, v, lab in pairs:
        eu = emb_cache.get(u)
        ev = emb_cache.get(v)
        if eu is None or ev is None:
            # missing embedding
            scores.append(-10.0)
            labels.append(lab)
            continue
        if use_clf:
            feat = np.concatenate([np.abs(eu - ev), eu * ev], axis=0)
            x = torch.tensor(feat, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                logit = clf(x).item()
                prob = 1.0 / (1.0 + math.exp(-logit))
            s = float(prob)
        else:
            s = float(np.dot(eu, ev) / (np.linalg.norm(eu) * np.linalg.norm(ev) + 1e-12))
        scores.append(s)
        labels.append(lab)

    eer, thr = eer_from_scores(labels, scores)
    print(f"EER = {eer*100:.3f} %, threshold = {thr:.6f}")

    auc = None
    roc_fpr = roc_tpr = None
    try:
        from sklearn.metrics import roc_auc_score, roc_curve
        labels_arr = np.asarray(labels, dtype=int)
        if labels_arr.min() == labels_arr.max():
            warnings.warn("Todas las etiquetas en el conjunto de test son idénticas, no es posible calcular AUC.")
            auc = None
        else:
            auc = float(roc_auc_score(labels_arr, np.asarray(scores, dtype=float)))
            roc_fpr, roc_tpr, roc_thresholds = roc_curve(labels_arr, np.asarray(scores, dtype=float))
    except Exception as e:
        if isinstance(e, ModuleNotFoundError) or isinstance(e, ImportError):
            warnings.warn("sklearn no disponible, calculando AUC y curva ROC manualmente.")
        else:
            warnings.warn(f"Error calculando AUC con sklearn ({e}), intentando método manual.")
        auc, roc_fpr, roc_tpr = compute_roc_auc_manual(labels, scores)

    if auc is None:
        print("AUC no está definida (etiquetas mono-clase o error).")
    else:
        print(f"AUC = {auc:.4f}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_scores = out_dir / "test_scores.tsv"
    with open(out_scores, 'w', encoding='utf-8') as fo:
        fo.write("u	v	label	score")
        for (u, v, lab), s in zip(pairs, scores):
            fo.write(f"{u}	{v}	{lab}	{s:.6f}")
    print("Saved scores to", out_scores)

    if roc_fpr is not None and roc_tpr is not None:
        out_roc = out_dir / "roc_curve.tsv"
        with open(out_roc, 'w', encoding='utf-8') as fo:
            fo.write("fpr	tpr")
            for fpr_v, tpr_v in zip(roc_fpr, roc_tpr):
                fo.write(f"{fpr_v:.8f}	{tpr_v:.8f}")
        print("Saved ROC curve to", out_roc)
    else:
        out_roc = None

    return {
        "eer": eer,
        "eer_threshold": thr,
        "auc": auc,
        "scores_file": str(out_scores),
        "roc_file": str(out_roc) if out_roc is not None else None
    }


# ------------------------
# CLI
# ------------------------
def compute_roc_auc_manual(lbls, scs):
    lbls = np.asarray(lbls, dtype=int)
    scs = np.asarray(scs, dtype=float)
    if lbls.size == 0:
        return None, None, None
    if lbls.min() == lbls.max():
        return None, None, None
    thresholds = np.unique(scs)[::-1]
    tprs = []
    fprs = []
    P = float((lbls == 1).sum())
    N = float((lbls == 0).sum())
    for thr_ in thresholds:
        preds = scs >= thr_
        TP = float(((preds == 1) & (lbls == 1)).sum())
        FP = float(((preds == 1) & (lbls == 0)).sum())
        tpr = TP / P if P > 0 else 0.0
        fpr = FP / N if N > 0 else 0.0
        tprs.append(tpr)
        fprs.append(fpr)
    fprs = np.asarray(fprs)
    tprs = np.asarray(tprs)
    fprs_full = np.concatenate(([0.0], fprs, [1.0]))
    tprs_full = np.concatenate(([0.0], tprs, [1.0]))
    auc_val = float(np.trapz(tprs_full, fprs_full))
    return auc_val, fprs_full, tprs_full


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train_tsv", required=False, help="Train TSV: lang\tclient_id\taudio\tgender")
    p.add_argument("--test_tsv", required=False, help="Test pairs TSV: u\tv\tlabel (1 same, 0 different)")
    p.add_argument("--out_dir", default="./out", help="Output directory")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--emb_dim", type=int, default=192)
    p.add_argument("--n_mels", type=int, default=64)
    p.add_argument("--sample_rate", type=int, default=16000)
    p.add_argument("--max_seconds", type=float, default=3.0)
    p.add_argument("--margin", type=float, default=1.0)
    p.add_argument("--pairs_per_epoch", type=int, default=20000)
    p.add_argument("--seed", type=int, default=42, help="Random seed (python, numpy, torch, etc.)")
    p.add_argument("--num_workers", type=int, default=4, help="Number of DataLoader workers")
    p.add_argument("--llm_model", type=str, default=None, help="Local path to SpeechLLM repo (required). If omitted, script will try default local path.")
    p.add_argument("--local_speechllm_dir", type=str, default=SPEECHLLM_LOCAL_DIR, help=f"Directorio local para SpeechLLM (por defecto: {SPEECHLLM_LOCAL_DIR})")
    p.add_argument("--text_tokenizer", type=str, default=None, help="Tokenizer de texto (ruta local o HF id) usado para tokenizar instrucciones/prompt. Requerido para prompting.")
    p.add_argument("--pre_instruction", type=str, default=None, help="Texto pre-instrucción (antes del <speech>)")
    p.add_argument("--post_instruction", type=str, default=None, help="Texto post-instrucción (después del </speech>)")
    p.add_argument("--out_prefix", type=str, default=None, help="Prefijo de salida (ej. \"\n<s>\")")
    p.add_argument("--device", type=str, default=None, help="Device to use (cuda or cpu). Default auto.")
    p.add_argument("--prepare_models", action="store_true", help="Descargar modelos HF a la ruta local (usar en máquina con Internet antes de ejecutar en SLURM sin red).")
    p.add_argument("--speechllm_repo", type=str, default="skit-ai/speechllm-2B", help="Repo HF a descargar para SpeechLLM cuando --prepare_models")
    p.add_argument("--allow_remote", action="store_true", help="Permitir descargas remotas incluso si --local_speechllm_dir está presente (útil si la máquina tiene Internet).")
    return p.parse_args()


# ------------------------
# Helpers para descargar modelos (usar en máquina con Internet)
# ------------------------
def download_models_to_local(speechllm_repo: str, speechllm_dir: str):
    """
    Descarga el repositorio de HF SpeechLLM a carpeta local usando huggingface_hub.snapshot_download.
    Ejecutar esto en una máquina con internet y luego copiar la carpeta al entorno sin red.
    """
    try:
        from huggingface_hub import snapshot_download
    except Exception as e:
        raise RuntimeError("Para usar --prepare_models necesitas instalar 'huggingface_hub'. "
                           "Instálalo con 'pip install huggingface_hub' y vuelve a intentarlo.") from e

    print("=== Preparando descarga del repo SpeechLLM a carpeta local ===")
    os.makedirs(speechllm_dir, exist_ok=True)

    # Descargar SpeechLLM repo (todo el repo: código + pesos)
    try:
        print(f"[download] descargando '{speechllm_repo}' a '{speechllm_dir}' ... (esto puede tardar)")
        snapshot_download(repo_id=speechllm_repo, local_dir=speechllm_dir, local_dir_use_symlinks=False)
        print(f"[download] terminado: {speechllm_dir}")
    except Exception as e:
        print(f"[download] fallo al descargar SpeechLLM '{speechllm_repo}': {e}{traceback.format_exc()}")
        raise

    print("=== Descarga completada. Copia la carpeta al servidor SLURM sin red y ejecuta el script sin --prepare_models ===")


if __name__ == "__main__":
    args = parse_args()

    # Si piden preparar modelos: descarga y salir (usar en máquina con Internet).
    if args.prepare_models:
        try:
            download_models_to_local(args.speechllm_repo, args.local_speechllm_dir)
        except Exception as e:
            print("Error durante la descarga de modelos:", e)
            sys.exit(2)
        print("Preparación completada. Salida.")
        sys.exit(0)

    # Comportamiento normal de evaluación
    result = train_and_eval(args)
    try:
        print(json.dumps(result))
    except Exception:
        print(result)
