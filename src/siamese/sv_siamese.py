#!/usr/bin/env python3
"""
sv_siamese.py
Compact speaker verification siamese training + evaluation script.

Usage:
    python sv_siamese.py --train_tsv train.tsv --test_tsv test_pairs.tsv --out_dir ./out \
        --epochs 20 --batch_size 128 --max_seconds 3.0

    python sv_siamese.py --train_tsv /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/ca.train.tsv \
        --test_tsv /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/08_test_val_pairs/tsv/ca.test.tsv --out_dir ./out \
        --epochs 20 --batch_size 128 --max_seconds 3.0

Requirements:
    torch, torchaudio, numpy, librosa
"""

import os
import argparse
import random
from pathlib import Path
from collections import defaultdict
import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio

import librosa

# ------------------------
# Utilities
# ------------------------
def set_seed(seed: int = 1234, cudnn_deterministic: bool = True):
    """
    Set global random seeds for python, numpy, torch (cpu + cuda) and PYTHONHASHSEED.
    If cudnn_deterministic is True, set cudnn to deterministic mode (may slow training).
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass

    # cudnn determinism (may slow or break some ops); optional fallback if not supported
    try:
        torch.backends.cudnn.deterministic = bool(cudnn_deterministic)
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass

    # Newer pytorch: optionally enforce deterministic algorithms (may raise on unsupported ops)
    try:
        torch.use_deterministic_algorithms(bool(cudnn_deterministic))
    except Exception:
        # ignore if not supported or would break execution
        pass

# Audio utils: load, resample, log-mel
def load_audio(path, sample_rate=16000, max_seconds=3.0):
    """
    Intenta cargar con torchaudio; si falla por falta de torchcodec u otro backend,
    cae en librosa (soundfile/audioread). Devuelve 1D tensor torch.FloatTensor.
    """
    # primera opción: torchaudio.load (rápido si backend soporta mp3/wav)
    try:
        waveform, sr = torchaudio.load(path)  # (channels, samples)
        if waveform.ndim > 1 and waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != sample_rate:
            waveform = torchaudio.transforms.Resample(sr, sample_rate)(waveform)
        waveform = waveform.squeeze(0)
        # pad/truncate
        max_len = int(sample_rate * max_seconds)
        if waveform.shape[0] < max_len:
            pad = max_len - waveform.shape[0]
            waveform = torch.nn.functional.pad(waveform, (0, pad))
        else:
            waveform = waveform[:max_len]
        return waveform.float()
    except Exception as e:
        # fallo en torchaudio (p. ej. torchcodec no instalado). Fallback a librosa.
        # print optional for debugging (evita muchos prints en producción)
        # print(f"[Warning] torchaudio.load failed ({e}), falling back to librosa")
        try:
            y, sr = librosa.load(path, sr=sample_rate, mono=True)
            max_len = int(sample_rate * max_seconds)
            if y.shape[0] < max_len:
                y = np.pad(y, (0, max_len - y.shape[0]))
            else:
                y = y[:max_len]
            return torch.from_numpy(y).float()
        except Exception as e2:
            # si librosa falla, propaga con mensaje claro
            raise RuntimeError(f"Failed to load audio with torchaudio ({e}) and librosa ({e2}) for file {path}")

class LogMelExtractor:
    def __init__(self, sample_rate=16000, n_mels=64, n_fft=400, hop=160, f_min=20, f_max=None):
        self.sample_rate = sample_rate
        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, n_fft=n_fft, hop_length=hop, n_mels=n_mels,
            f_min=f_min, f_max=f_max or sample_rate/2
        )
        self.db = torchaudio.transforms.AmplitudeToDB()
    def __call__(self, waveform):
        # waveform: 1D tensor (samples,)
        with torch.no_grad():
            S = self.melspec(waveform.unsqueeze(0))  # (1, n_mels, frames)
            S = self.db(S)
            # normalize per-utterance
            S = (S - S.mean()) / (S.std() + 1e-6)
        return S.squeeze(0)  # (n_mels, frames)

# ------------------------
# Dataset (pair sampler)
# ------------------------
class PairDataset(torch.utils.data.Dataset):
    """
    Expects a TSV with header: lang\tclient_id\taudio\tgender (tab separated).
    Builds a dict client_id -> list[audio_path], then samples positive and negative pairs.
    """
    def __init__(self, tsv_path, max_seconds=3.0, sample_rate=16000, transform=None, pairs_per_epoch=100000):
        self.tsv_path = tsv_path
        self.max_seconds = max_seconds
        self.sample_rate = sample_rate
        self.transform = transform or LogMelExtractor(sample_rate=sample_rate)
        self.pairs_per_epoch = pairs_per_epoch

        # read TSV
        import csv
        self.client2files = defaultdict(list)
        with open(tsv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='\t')
            # intentar detectar y saltar cabecera si existe
            first = next(reader, None)
            if first is None:
                raise RuntimeError(f"TSV {tsv_path} vacío")
            # si la primera fila parece cabecera (contiene 'lang' o 'client'), saltarla
            if any(c.strip().lower() in ('lang', 'client_id', 'client', 'speaker', 'audio') for c in first):
                pass  # ya consumida la cabecera
            else:
                # no era cabecera, procesarla como fila de datos
                # reusar 'first' como la primera línea de datos
                reader = itertools.chain([first], reader)

            for parts in reader:
                # saltar líneas vacías
                if not parts or all(not p.strip() for p in parts):
                    continue
                # aceptar al menos 3 columnas: lang, client_id, audio
                if len(parts) < 3:
                    continue
                lang = parts[0].strip()
                client_id = parts[1].strip()
                audio_rel = parts[2].strip()
                # optional gender
                gender = parts[3].strip() if len(parts) > 3 else ''
                # only keep rows with a non-empty client_id and audio path
                if client_id == '' or audio_rel == '':
                    continue
                self.client2files[client_id].append(audio_rel)

        # remove clients with <2 utts (can't form positive pair)
        self.clients = [c for c, files in self.client2files.items() if len(files) >= 2]
        if len(self.clients) < 2:
            raise ValueError("Need at least two clients with >=2 utterances each")

        # flatten list of file paths for caching optional
        self.all_files = []
        for files in self.client2files.values():
            self.all_files.extend(files)

        # cache for embeddings/preprocessed features if needed
        self.feature_cache = {}

    def __len__(self):
        return self.pairs_per_epoch

    def _load_feature(self, path):
        # caching to avoid repeated disk I/O in epoch
        if path in self.feature_cache:
            return self.feature_cache[path]
        waveform = load_audio(path, sample_rate=self.sample_rate, max_seconds=self.max_seconds)
        feat = self.transform(waveform)  # (n_mels, frames)
        self.feature_cache[path] = feat
        return feat

    def __getitem__(self, idx):
        # choose positive or negative with equal probability
        same = random.random() < 0.5
        if same:
            # sample a client with >=2 utts
            client = random.choice(self.clients)
            a, b = random.sample(self.client2files[client], 2)
            label = 1.0
        else:
            c1, c2 = random.sample(self.clients, 2)
            a = random.choice(self.client2files[c1])
            b = random.choice(self.client2files[c2])
            label = 0.0

        A = self._load_feature(a)  # (n_mels, frames)
        B = self._load_feature(b)

        return A.unsqueeze(0).float(), B.unsqueeze(0).float(), torch.tensor(label, dtype=torch.float32)  # add channel dim

def collate_pairs(batch):
    # batch: list of (A, B, label)
    A = torch.stack([x[0] for x in batch], dim=0)
    B = torch.stack([x[1] for x in batch], dim=0)
    L = torch.stack([x[2] for x in batch], dim=0)
    return A, B, L

# ------------------------
# Model
# ------------------------
class ConvEncoder(nn.Module):
    def __init__(self, n_mels=64, emb_dim=192, dummy_frames=200):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(5,5), padding=(2,2)),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d((1,2)),
            nn.Conv2d(32, 64, kernel_size=(5,5), padding=(2,2)),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d((1,2)),
            nn.Conv2d(64, 128, kernel_size=(3,3), padding=(1,1)),
            nn.ReLU(),
            nn.BatchNorm2d(128),
        )

        # inferir la dimensión de entrada del fc con un dummy
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_mels, dummy_frames)  # (B,1,F,T)
            out = self.conv(dummy)  # (1, C, F_after, T_after)
            C, F_after, T_after = out.shape[1], out.shape[2], out.shape[3]
            in_features = 2 * C * F_after  # mean+std

        self.fc = nn.Linear(in_features, emb_dim)

    def forward(self, x):
        # x: (B,1,n_mels,frames)
        out = self.conv(x)  # (B, C, F, T)
        B, C, F, T = out.shape
        out = out.permute(0, 3, 1, 2).contiguous().view(B, T, C * F)  # (B, T, C*F)
        mean = out.mean(dim=1)
        std = out.std(dim=1)
        stats = torch.cat([mean, std], dim=1)  # (B, 2*C*F)
        emb = self.fc(stats)
        emb = nn.functional.normalize(emb, p=2, dim=1)
        return emb

# ------------------------
# Loss
# ------------------------
def contrastive_loss(emb1, emb2, label, margin=1.0):
    # label: 1 = same, 0 = different
    # Euclidean distance
    dist = torch.norm(emb1 - emb2, p=2, dim=1)
    loss_pos = label * (dist ** 2)
    loss_neg = (1.0 - label) * (torch.clamp(margin - dist, min=0.0) ** 2)
    loss = loss_pos + loss_neg
    return loss.mean()

# ------------------------
# Evaluation utilities
# ------------------------
def compute_embeddings(paths, model, device, extractor, max_seconds=3.0, batch_size=64):
    model.eval()
    emb_cache = {}
    with torch.no_grad():
        keys = list(set(paths))
        for i in range(0, len(keys), batch_size):
            batch_paths = keys[i:i+batch_size]
            feats = []
            for p in batch_paths:
                wav = load_audio(p, max_seconds=max_seconds)  # 1D
                f = extractor(wav).unsqueeze(0).unsqueeze(0)  # (1,1,n_mels,frames)
                feats.append(f)
            feats = torch.cat(feats, dim=0).to(device)
            embs = model(feats).cpu().numpy()
            for k, e in zip(batch_paths, embs):
                emb_cache[k] = e
    return emb_cache

def eer_from_scores(labels, scores):
    # labels: 0/1 where 1 = target (same)
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores)
    # higher score -> more likely same speaker. We will search thresholds between min and max.
    # Compute FPR (false positive rate) and FNR for many thresholds and find EER
    desc = np.argsort(scores)
    sorted_scores = scores[desc]
    sorted_labels = labels[desc]
    # thresholds include just below each unique score
    thresholds = np.concatenate(([sorted_scores[0]-1e-6], (sorted_scores[:-1] + sorted_scores[1:]) / 2, [sorted_scores[-1]+1e-6]))
    FP = []
    FN = []
    P = sum(sorted_labels == 1)
    N = len(sorted_labels) - P
    for th in thresholds:
        # predict 1 if score >= th
        preds = (scores >= th).astype(int)
        fp = ((preds == 1) & (labels == 0)).sum()
        fn = ((preds == 0) & (labels == 1)).sum()
        FP.append(fp / max(1, N))
        FN.append(fn / max(1, P))
    FP = np.array(FP)
    FN = np.array(FN)
    # find threshold where abs(FP - FN) minimal
    idx = np.argmin(np.abs(FP - FN))
    eer = (FP[idx] + FN[idx]) / 2.0
    return eer, thresholds[idx]

# ------------------------
# Main training and evaluation
# ------------------------
def train_and_eval(args):
    import csv
    import itertools
    import time
    import warnings
    from pathlib import Path

    import numpy as np
    import torch
    import torch.optim as optim

    # reproducibility: set global seeds before creating dataset/loader
    set_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform = LogMelExtractor(sample_rate=args.sample_rate, n_mels=args.n_mels)
    dataset = PairDataset(args.train_tsv, max_seconds=args.max_seconds, sample_rate=args.sample_rate,
                          transform=transform, pairs_per_epoch=args.pairs_per_epoch)

    # worker init fn para que cada worker tenga una semilla derivada de la principal
    def worker_init_fn(worker_id):
        worker_seed = args.seed + worker_id + 1  # evitar seed == base
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        try:
            torch.manual_seed(worker_seed)
        except Exception:
            pass

    # generator para que shuffle sea reproducible
    g = torch.Generator()
    try:
        g.manual_seed(args.seed)
    except Exception:
        # en caso de versiones antiguas que no soporten Generator.manual_seed
        g = None

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_pairs,
        num_workers=args.num_workers,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
        generator=g
    )

    model = ConvEncoder(n_mels=args.n_mels, emb_dim=args.emb_dim).to(device)
    opt = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.5)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Training on device:", device)
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        t0 = time.time()
        for i, (A, B, L) in enumerate(loader):
            A = A.to(device)  # (B,1,n_mels,frames)
            B = B.to(device)
            L = L.to(device)
            opt.zero_grad()
            embA = model(A)
            embB = model(B)
            loss = contrastive_loss(embA, embB, L, margin=args.margin)
            loss.backward()
            opt.step()
            running_loss += loss.item()
            # optionally clear feature cache each few iterations to limit memory
            if i % 200 == 0:
                try:
                    dataset.feature_cache.clear()
                except Exception:
                    # silenciar si dataset no tiene cache
                    pass
        scheduler.step()
        dt = time.time() - t0
        lr_now = scheduler.get_last_lr()[0] if hasattr(scheduler, "get_last_lr") else opt.param_groups[0]['lr']
        print(f"Epoch {epoch}/{args.epochs}, loss={running_loss/(i+1):.4f}, time={dt:.1f}s, lr={lr_now:.2e}")

        # checkpoint
        torch.save(model.state_dict(), out_dir / f"model_epoch{epoch}.pt")

    # Evaluate on test TSV (robusto a cabeceras y formatos)
    print("Loading test pairs...")
    pairs = []
    bad_lines = 0

    def looks_like_header(row):
        if len(row) < 3:
            return False
        candidates = [row[0].strip().lower(), row[1].strip().lower(), row[2].strip().lower()]
        # detect typical header tokens
        return ('u' in candidates[0] or 'v' in candidates[1] or 'label' in candidates[2]
                or 'path' in candidates[0] or 'path' in candidates[1] or 'audio' in candidates[2])

    def parse_label(s):
        s = s.strip().lower()
        if s == '':
            raise ValueError("Etiqueta vacía")
        # intentar parse numérico
        try:
            v = int(float(s))
            return 1 if v != 0 else 0
        except Exception:
            pass
        # mappings comunes
        if s in ('1', 'true', 'yes', 'y', 'same', 'target', 'positive'):
            return 1
        if s in ('0', 'false', 'no', 'n', 'different', 'diff', 'non-target', 'negative'):
            return 0
        raise ValueError(f"Unrecognized label string: '{s}'")

    with open(args.test_tsv, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        # localizar primera fila no vacía
        first_row = None
        for row in reader:
            if row and any(cell.strip() for cell in row):
                first_row = row
                break
        if first_row is None:
            raise RuntimeError(f"Test TSV {args.test_tsv} está vacío o no tiene líneas válidas")

        if looks_like_header(first_row):
            rows_iter = reader  # ya hemos consumido la cabecera
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

    # compute embeddings for all unique files in test set
    all_paths = [p for p, q, _ in pairs] + [q for p, q, _ in pairs]
    # mantener orden único y estable
    unique_paths = []
    seen = set()
    for p in all_paths:
        if p not in seen:
            unique_paths.append(p)
            seen.add(p)

    emb_cache = compute_embeddings(unique_paths, model, device, transform, max_seconds=args.max_seconds, batch_size=64)

    # compute cosine similarity scores and compute EER
    scores = []
    labels = []
    for u, v, lab in pairs:
        eu = emb_cache.get(u)
        ev = emb_cache.get(v)
        if eu is None or ev is None:
            # fallback: zero score (se registra para revisión)
            scores.append(-10.0)
            labels.append(lab)
            continue
        # cosine similarity
        s = float(np.dot(eu, ev) / (np.linalg.norm(eu) * np.linalg.norm(ev) + 1e-12))
        scores.append(s)
        labels.append(lab)

    # EER (función auxiliar externa)
    eer, thr = eer_from_scores(labels, scores)
    print(f"EER = {eer*100:.3f} %, threshold = {thr:.4f}")

    # Compute AUC and ROC curve, con fallback manual si sklearn no está disponible
    def compute_roc_auc_manual(lbls, scs):
        # lbls: list/array de 0/1, scs: list/array de scores (may be floats)
        lbls = np.asarray(lbls, dtype=int)
        scs = np.asarray(scs, dtype=float)
        if lbls.size == 0:
            return None, None, None
        if lbls.min() == lbls.max():
            # solo una clase presente
            return None, None, None
        # thresholds: todos los valores únicos de score, ordenados descendente
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
        # añadir punto (0,0) y (1,1) si no están presentes para una integración estable
        fprs = np.asarray(fprs)
        tprs = np.asarray(tprs)
        # asegurar que fpr empieza en 0 y termina en 1
        fprs_full = np.concatenate(([0.0], fprs, [1.0]))
        tprs_full = np.concatenate(([0.0], tprs, [1.0]))
        auc_val = float(np.trapz(tprs_full, fprs_full))
        return auc_val, fprs_full, tprs_full

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
        # fallback manual
        if isinstance(e, ModuleNotFoundError) or isinstance(e, ImportError):
            warnings.warn("sklearn no disponible, calculando AUC y curva ROC manualmente.")
        else:
            warnings.warn(f"Error calculando AUC con sklearn ({e}), intentando método manual.")
        auc, roc_fpr, roc_tpr = compute_roc_auc_manual(labels, scores)

    if auc is None:
        print("AUC no está definida (etiquetas mono-clase o error).")
    else:
        print(f"AUC = {auc:.4f}")

    # save scores
    out_scores = out_dir / "test_scores.tsv"
    with open(out_scores, 'w', encoding='utf-8') as fo:
        fo.write("u\tv\tlabel\tscore\n")
        for (u, v, lab), s in zip(pairs, scores):
            fo.write(f"{u}\t{v}\t{lab}\t{s:.6f}\n")
    print("Saved scores to", out_scores)

    # save ROC curve if disponible
    if roc_fpr is not None and roc_tpr is not None:
        out_roc = out_dir / "roc_curve.tsv"
        with open(out_roc, 'w', encoding='utf-8') as fo:
            fo.write("fpr\ttpr\n")
            for fpr_v, tpr_v in zip(roc_fpr, roc_tpr):
                fo.write(f"{fpr_v:.8f}\t{tpr_v:.8f}\n")
        print("Saved ROC curve to", out_roc)

    return {
        "eer": eer,
        "eer_threshold": thr,
        "auc": auc,
        "scores_file": str(out_scores),
        "roc_file": str(out_roc) if (roc_fpr is not None and roc_tpr is not None) else None
    }

# ------------------------
# CLI
# ------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train_tsv", required=True, help="Train TSV: lang\\tclient_id\\taudio\\tgender")
    p.add_argument("--test_tsv", required=True, help="Test pairs TSV: u\\tv\\tlabel (1 same, 0 different)")
    p.add_argument("--out_dir", default="./out", help="Output directory")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--emb_dim", type=int, default=192)
    p.add_argument("--n_mels", type=int, default=64)
    p.add_argument("--sample_rate", type=int, default=16000)
    p.add_argument("--max_seconds", type=float, default=3.0)
    p.add_argument("--margin", type=float, default=1.0)
    p.add_argument("--pairs_per_epoch", type=int, default=20000)
    p.add_argument("--seed", type=int, default=42, help="Random seed (python, numpy, torch, etc.)")
    p.add_argument("--num_workers", type=int, default=4, help="Number of DataLoader workers")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    train_and_eval(args)
