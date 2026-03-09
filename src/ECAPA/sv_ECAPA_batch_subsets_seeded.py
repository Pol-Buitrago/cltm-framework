#!/usr/bin/env python3
"""
sv_ecapa_batch_subsets_seeded.py

Versión batch para entrenar un ECAPA-like TDNN por-subset y por-seed.
Mantiene la misma interfaz y comportamiento que sv_siamese_batch_subsets_seeded.py:
- Recorrer <subsets_root>/<lang>/subset_XXX_.../
- Por cada subset (usa train/test TSV dentro del subset) y por cada seed, entrena + evalúa ECAPA
- Resultados añadidos a <lang>.eer_by_samples_<seed>.csv en out_root
- CSV columns: eer,auc,thr,num_samples
- Si existe ya un registro con num_samples, se salta (resumible)
- Maneja errores y continúa

Uso ejemplo:
    python sv_ecapa_batch_subsets_seeded.py \
        --subsets_root /path/to/subsets_root \
        --out_root /path/to/out_root \
        --langs ca --seeds 41,42,43 --epochs 1 --num_workers 4
"""
import argparse
import csv
import io
import json
import os
import random
import sys
from pathlib import Path
from collections import defaultdict
import time
import warnings
import itertools

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
import librosa
import math

# ------------------------
# Utilities (seed, audio, log-mel)
# ------------------------
def set_seed(seed: int = 1234, cudnn_deterministic: bool = True):
    """Set global random seeds for python, numpy, torch (cpu + cuda) and PYTHONHASHSEED."""
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
        torch.use_deterministic_algorithms(bool(cudnn_deterministic))
    except Exception:
        pass

def load_audio(path, sample_rate=16000, max_seconds=3.0):
    try:
        waveform, sr = torchaudio.load(path)
        if waveform.ndim > 1 and waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != sample_rate:
            waveform = torchaudio.transforms.Resample(sr, sample_rate)(waveform)
        waveform = waveform.squeeze(0)
        max_len = int(sample_rate * max_seconds)
        if waveform.shape[0] < max_len:
            pad = max_len - waveform.shape[0]
            waveform = torch.nn.functional.pad(waveform, (0, pad))
        else:
            waveform = waveform[:max_len]
        return waveform.float()
    except Exception:
        try:
            y, sr = librosa.load(path, sr=sample_rate, mono=True)
            max_len = int(sample_rate * max_seconds)
            if y.shape[0] < max_len:
                y = np.pad(y, (0, max_len - y.shape[0]))
            else:
                y = y[:max_len]
            return torch.from_numpy(y).float()
        except Exception as e2:
            raise RuntimeError(f"Failed to load audio {path}: {e2}")

class LogMelExtractor:
    def __init__(self, sample_rate=16000, n_mels=64, n_fft=400, hop=160, f_min=20, f_max=None):
        self.sample_rate = sample_rate
        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, n_fft=n_fft, hop_length=hop, n_mels=n_mels,
            f_min=f_min, f_max=f_max or sample_rate/2
        )
        self.db = torchaudio.transforms.AmplitudeToDB()
    def __call__(self, waveform):
        with torch.no_grad():
            S = self.melspec(waveform.unsqueeze(0))  # (1, n_mels, frames)
            S = self.db(S)
            S = (S - S.mean()) / (S.std() + 1e-6)
        return S.squeeze(0)  # (n_mels, frames)

# ------------------------
# Dataset for SID training (single utterance -> label)
# ------------------------
class SingleUtteranceDataset(torch.utils.data.Dataset):
    """
    Expects TSV with columns: lang\tclient_id\taudio\t( optional: gender ... )
    Builds (audio_path, label_index) pairs.
    """
    def __init__(self, tsv_path, max_seconds=3.0, sample_rate=16000, transform=None):
        self.tsv_path = tsv_path
        self.max_seconds = max_seconds
        self.sample_rate = sample_rate
        self.transform = transform or LogMelExtractor(sample_rate=sample_rate)
        client2files = defaultdict(list)
        with open(tsv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='\t')
            first = next(reader, None)
            if first is None:
                raise RuntimeError(f"TSV {tsv_path} vacío")
            # detect header-like row
            if any(c.strip().lower() in ('lang', 'client_id', 'client', 'speaker', 'audio') for c in first):
                pass
            else:
                # reuse first as data
                reader = itertools.chain([first], reader)
            for parts in reader:
                if not parts or all(not p.strip() for p in parts):
                    continue
                if len(parts) < 3:
                    continue
                lang = parts[0].strip()
                client = parts[1].strip()
                audio = parts[2].strip()
                if client == '' or audio == '':
                    continue
                client2files[client].append(audio)
        # remove clients with no utts
        self.clients = sorted([c for c, files in client2files.items() if len(files) >= 1])
        if len(self.clients) < 2:
            raise ValueError("Need at least two clients with >=1 utterance each")
        self.label_map = {c:i for i,c in enumerate(self.clients)}
        self.samples = []
        for c, files in client2files.items():
            if c not in self.label_map:
                continue
            lab = self.label_map[c]
            for p in files:
                self.samples.append((p, lab))
        # feature cache optional
        self.feature_cache = {}

    def __len__(self):
        return len(self.samples)

    def _load_feature(self, path):
        if path in self.feature_cache:
            return self.feature_cache[path]
        waveform = load_audio(path, sample_rate=self.sample_rate, max_seconds=self.max_seconds)
        feat = self.transform(waveform)  # (n_mels, frames)
        self.feature_cache[path] = feat
        return feat

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        feat = self._load_feature(path)  # (n_mels, frames)
        return feat.unsqueeze(0).float(), torch.tensor(label, dtype=torch.long)  # (1, n_mels, frames), label

def collate_single(batch):
    feats = [x[0] for x in batch]  # cada f: (1, n_mels, frames)
    labs = [x[1] for x in batch]
    feats = torch.stack(feats, dim=0)
    labs = torch.stack(labs, dim=0)
    return feats, labs

# ------------------------
# ECAPA-like model (compact, robust)
# ------------------------
class SEBlock1d(nn.Module):
    def __init__(self, channels, r=8):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Conv1d(channels, channels // r, kernel_size=1)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv1d(channels // r, channels, kernel_size=1)
        self.sig = nn.Sigmoid()
    def forward(self, x):
        s = self.pool(x)
        s = self.fc1(s)
        s = self.relu(s)
        s = self.fc2(s)
        s = self.sig(s)
        return x * s

class TDNNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, dilation=1, relu=True):
        super().__init__()
        padding = (kernel_size - 1) // 2 * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, dilation=dilation, padding=padding)
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.ReLU() if relu else nn.Identity()
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class ECAPA_TDNN(nn.Module):
    def __init__(self, in_feats=64, channels=512, emb_dim=192):
        super().__init__()
        # input: (B, 1, n_mels, T)
        self.in_feats = in_feats
        self.bn_in = nn.BatchNorm2d(1)
        self.conv2d = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(3,3), padding=(1,1)),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=(3,3), padding=(1,1)),
            nn.ReLU(),
            nn.BatchNorm2d(128),
        )
        # TDNN stack
        self.tdnn1 = TDNNBlock(128, channels, kernel_size=5, dilation=1)
        self.tdnn2 = TDNNBlock(channels, channels, kernel_size=3, dilation=2)
        self.tdnn3 = TDNNBlock(channels, channels, kernel_size=3, dilation=3)
        # squeeze-excitation
        self.se = SEBlock1d(channels)
        self.res_conv = nn.Conv1d(channels*3, channels, kernel_size=1)
        # pooling
        self.pool_mean = nn.AdaptiveAvgPool1d(1)
        self.pool_std = lambda x: torch.sqrt(torch.mean((x - torch.mean(x, dim=2, keepdim=True))**2, dim=2, keepdim=True) + 1e-12)
        self.fc = nn.Linear(channels * 2, emb_dim)
        self.bn_out = nn.BatchNorm1d(emb_dim, affine=False)

    def forward(self, x):
        # x: (B, 1, n_mels, T)
        B = x.size(0)
        out = self.conv2d(x)  # (B, C2, n_mels, T)
        out = out.mean(dim=2)  # collapse freq -> (B, C2, T)
        t1 = self.tdnn1(out)
        t2 = self.tdnn2(t1)
        t3 = self.tdnn3(t2)
        cat = torch.cat([t1, t2, t3], dim=1)  # (B, 3C, T)
        cat = self.res_conv(cat)  # (B, C, T)
        cat = self.se(cat)
        mean = torch.mean(cat, dim=2)  # (B, C)
        std = torch.std(cat, dim=2)   # (B, C)
        stats = torch.cat([mean, std], dim=1)  # (B, 2C)
        emb = self.fc(stats)
        emb = self.bn_out(emb)
        emb = nn.functional.normalize(emb, p=2, dim=1)
        return emb

    def embed(self, feats):
        return self.forward(feats)

# ------------------------
# AAM-Softmax head
# ------------------------
class AAMSoftmaxLoss(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.2, eps=1e-7):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m
        self.eps = eps
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x, labels):
        x_norm = nn.functional.normalize(x, p=2, dim=1)
        w_norm = nn.functional.normalize(self.weight, p=2, dim=1)
        cos_theta = torch.matmul(x_norm, w_norm.t())  # (B, out)
        cos_theta = cos_theta.clamp(-1 + self.eps, 1 - self.eps)
        sin_theta = torch.sqrt(1.0 - torch.clamp(cos_theta**2, 0.0, 1.0))
        cos_theta_m = cos_theta * self.cos_m - sin_theta * self.sin_m
        one_hot = torch.zeros_like(cos_theta)
        one_hot.scatter_(1, labels.view(-1,1), 1.0)
        logits = cos_theta.clone()
        logits = logits * (1 - one_hot) + cos_theta_m * one_hot
        logits = logits * self.s
        return logits

# ------------------------
# Evaluation helpers
# ------------------------
def eer_from_scores(labels, scores):
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores)
    desc = np.argsort(scores)
    sorted_scores = scores[desc]
    sorted_labels = labels[desc]
    thresholds = np.concatenate(([sorted_scores[0]-1e-6], (sorted_scores[:-1] + sorted_scores[1:]) / 2, [sorted_scores[-1]+1e-6]))
    FP = []
    FN = []
    P = sum(sorted_labels == 1)
    N = len(sorted_labels) - P
    for th in thresholds:
        preds = (scores >= th).astype(int)
        fp = ((preds == 1) & (labels == 0)).sum()
        fn = ((preds == 0) & (labels == 1)).sum()
        FP.append(fp / max(1, N))
        FN.append(fn / max(1, P))
    FP = np.array(FP)
    FN = np.array(FN)
    idx = np.argmin(np.abs(FP - FN))
    eer = (FP[idx] + FN[idx]) / 2.0
    return eer, thresholds[idx]

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

def compute_embeddings(paths, model, device, extractor, max_seconds=3.0, batch_size=64):
    model.eval()
    emb_cache = {}
    with torch.no_grad():
        keys = list(dict.fromkeys(paths))  # stable unique
        for i in range(0, len(keys), batch_size):
            batch_paths = keys[i:i+batch_size]
            feats = []
            for p in batch_paths:
                wav = load_audio(p, max_seconds=max_seconds)
                f = extractor(wav).unsqueeze(0).unsqueeze(0)  # (1,1,n_mels,frames)
                feats.append(f)
            feats = torch.cat(feats, dim=0).to(device)
            embs = model.embed(feats).cpu().numpy()
            for k, e in zip(batch_paths, embs):
                emb_cache[k] = e
    return emb_cache

# ------------------------
# Main training + evaluation function (ECAPA)
# ------------------------
def train_and_eval(args):
    """
    args must include: train_tsv, test_tsv, out_dir, epochs, batch_size, lr, emb_dim, n_mels,
    sample_rate, max_seconds, seed, num_workers (extra args are ignored).
    Returns dict with keys: {'eer': float, 'auc': float or None, 'thr': float}
    """
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device:", device)

    extractor = LogMelExtractor(sample_rate=args.sample_rate, n_mels=args.n_mels)

    # build single-utterance dataset for SID training
    train_dataset = SingleUtteranceDataset(args.train_tsv, max_seconds=args.max_seconds,
                                          sample_rate=args.sample_rate, transform=extractor)
    num_classes = len(train_dataset.label_map)
    print(f"Train dataset: {len(train_dataset)} samples, {num_classes} speakers")

    # dataloader
    def worker_init_fn(worker_id):
        worker_seed = args.seed + worker_id + 1
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        try:
            torch.manual_seed(worker_seed)
        except Exception:
            pass
    g = torch.Generator()
    try:
        g.manual_seed(args.seed)
    except Exception:
        g = None

    loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_single,
        num_workers=args.num_workers,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
        generator=g
    )

    # model + head + optimizer
    model = ECAPA_TDNN(in_feats=args.n_mels, emb_dim=args.emb_dim).to(device)
    loss_head = AAMSoftmaxLoss(in_features=args.emb_dim, out_features=num_classes, s=30.0, m=0.2).to(device)
    opt = optim.AdamW(list(model.parameters()) + list(loss_head.parameters()), lr=args.lr, weight_decay=2e-6)
    scheduler = optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.5)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    criterion_ce = nn.CrossEntropyLoss()

    print("Training ECAPA as SID (AAM-Softmax).")
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_head.train()
        running_loss = 0.0
        t0 = time.time()
        for i, (X, Y) in enumerate(loader):
            X = X.to(device)  # (B,1,n_mels,frames)
            Y = Y.to(device, dtype=torch.long)
            opt.zero_grad()
            emb = model.embed(X)  # (B, emb_dim)
            logits = loss_head(emb, Y)  # (B, num_classes)
            loss = criterion_ce(logits, Y)
            loss.backward()
            opt.step()
            running_loss += float(loss.item())
            if i % 200 == 0:
                try:
                    train_dataset.feature_cache.clear()
                except Exception:
                    pass
        scheduler.step()
        dt = time.time() - t0
        lr_now = scheduler.get_last_lr()[0] if hasattr(scheduler, "get_last_lr") else opt.param_groups[0]['lr']
        print(f"Epoch {epoch}/{args.epochs}, loss={running_loss/(i+1):.4f}, time={dt:.1f}s, lr={lr_now:.2e}")
        torch.save({
            'epoch': epoch,
            'model_state': model.state_dict(),
            'head_state': loss_head.state_dict(),
            'opt_state': opt.state_dict()
        }, out_dir / f"model_epoch{epoch}.pt")

    # Evaluate on test pairs
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
        reader = csv.reader(f, delimiter='\t')
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

    # compute embeddings for unique files
    all_paths = [p for p, q, _ in pairs] + [q for p, q, _ in pairs]
    unique_paths = []
    seen = set()
    for p in all_paths:
        if p not in seen:
            unique_paths.append(p)
            seen.add(p)

    emb_cache = compute_embeddings(unique_paths, model, device, extractor, max_seconds=args.max_seconds, batch_size=64)

    scores = []
    labels = []
    for u, v, lab in pairs:
        eu = emb_cache.get(u)
        ev = emb_cache.get(v)
        if eu is None or ev is None:
            scores.append(-10.0)
            labels.append(lab)
            continue
        s = float(np.dot(eu, ev) / (np.linalg.norm(eu) * np.linalg.norm(ev) + 1e-12))
        scores.append(s)
        labels.append(lab)

    eer, thr = eer_from_scores(labels, scores)
    print(f"EER = {eer*100:.3f} %, threshold = {thr:.4f}")

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

    # Save scores & ROC if possible
    out_scores = out_dir / "test_scores.tsv"
    with open(out_scores, 'w', encoding='utf-8') as fo:
        fo.write("u\tv\tlabel\tscore\n")
        for (u, v, lab), s in zip(pairs, scores):
            fo.write(f"{u}\t{v}\t{lab}\t{s:.6f}\n")
    print("Saved scores to", out_scores)

    if roc_fpr is not None and roc_tpr is not None:
        out_roc = out_dir / "roc_curve.tsv"
        with open(out_roc, 'w', encoding='utf-8') as fo:
            fo.write("fpr\ttpr\n")
            for fpr_v, tpr_v in zip(roc_fpr, roc_tpr):
                fo.write(f"{fpr_v:.8f}\t{tpr_v:.8f}\n")
        print("Saved ROC curve to", out_roc)
    else:
        out_roc = None

    # Return standardized keys expected by the batch runner
    return {"eer": float(eer), "auc": (None if auc is None else float(auc)), "thr": float(thr)}

# ------------------------
# Batch orchestration (misma interfaz que tu script original)
# ------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--subsets_root", required=True, help="Root path containing language subdirs with subsets")
    p.add_argument("--out_root", required=True, help="Directory to store per-language, per-seed CSV results")
    p.add_argument("--langs", default="all", help="Comma-separated language codes to process (default all)")
    p.add_argument("--seeds", default="42", help="Comma-separated seeds to run (e.g. 42,2025)")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--emb_dim", type=int, default=192)
    p.add_argument("--n_mels", type=int, default=64)
    p.add_argument("--sample_rate", type=int, default=16000)
    p.add_argument("--max_seconds", type=float, default=3.0)
    p.add_argument("--margin", type=float, default=1.0)
    p.add_argument("--pairs_per_epoch", type=int, default=20000)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--dry_run", action='store_true', help="Do not actually train, only show planned runs")
    return p.parse_args()

def read_existing_nums(csv_path: Path):
    nums = set()
    if not csv_path.exists():
        return nums
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if 'num_samples' in row and row['num_samples'].strip() != '':
                    try:
                        nums.add(int(row['num_samples']))
                    except Exception:
                        pass
    except Exception:
        pass
    return nums

def count_train_samples(train_tsv: Path):
    c = 0
    with open(train_tsv, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            if line.strip():
                c += 1
    return c

def find_train_test_files(subset_dir: Path, lang: str):
    candidates = list(subset_dir.glob('*.tsv'))
    train_path = None
    test_path = None
    for p in candidates:
        name = p.name.lower()
        if name.endswith('.train.tsv') and lang in name:
            train_path = p
        if name.endswith('.test.tsv') and lang in name:
            test_path = p
    if train_path is None:
        for p in candidates:
            if p.name.lower().endswith('.train.tsv'):
                train_path = p
                break
    if test_path is None:
        for p in candidates:
            if p.name.lower().endswith('.test.tsv'):
                test_path = p
                break
    return train_path, test_path

def ensure_csv_header(csv_path: Path):
    if not csv_path.exists():
        with open(csv_path, 'w', encoding='utf-8') as fo:
            fo.write('eer,auc,thr,num_samples\n')

def main():
    args = parse_args()
    subsets_root = Path(args.subsets_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    if args.langs.strip().lower() == 'all':
        lang_dirs = [p for p in subsets_root.iterdir() if p.is_dir()]
    else:
        req = [x.strip() for x in args.langs.split(',') if x.strip()]
        lang_dirs = [subsets_root / l for l in req]

    seeds = [int(s) for s in args.seeds.split(',') if s.strip()]

    for lang_dir in sorted(lang_dirs):
        if not lang_dir.exists():
            print(f"Warning: language dir {lang_dir} does not exist, skipping.")
            continue
        lang = lang_dir.name
        subset_dirs = [p for p in sorted(lang_dir.iterdir()) if p.is_dir() and p.name.startswith('subset_')]
        if len(subset_dirs) == 0:
            print(f"No subset directories found under {lang_dir}, skipping.")
            continue
        for seed in seeds:
            csv_path = out_root / f"{lang}.eer_by_samples_{seed}.csv"
            ensure_csv_header(csv_path)
            existing_nums = read_existing_nums(csv_path)
            print(f"Processing language={lang}, seed={seed}. Existing entries: {sorted(existing_nums)}")

            for subset_dir in subset_dirs:
                train_tsv, test_tsv = find_train_test_files(subset_dir, lang)
                if train_tsv is None or test_tsv is None:
                    print(f"  Skipping {subset_dir.name}: missing train or test TSV (found: train={train_tsv}, test={test_tsv})")
                    continue
                try:
                    num_samples = count_train_samples(train_tsv)
                except Exception as e:
                    print(f"  Could not count samples in {train_tsv}: {e}. Skipping")
                    continue
                if num_samples in existing_nums:
                    print(f"  Skipping {subset_dir.name} (num_samples={num_samples}) already recorded in {csv_path.name}")
                    continue

                print(f"  Running subset {subset_dir.name} (train={train_tsv.name}, test={test_tsv.name}, samples={num_samples})")
                if args.dry_run:
                    print("    dry-run: would train here")
                    continue

                # prepare per-run args for train_and_eval
                run_args = argparse.Namespace(
                    train_tsv=str(train_tsv),
                    test_tsv=str(test_tsv),
                    out_dir=str(out_root / f"runs/run_{lang}_{subset_dir.name}_seed{seed}"),
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    lr=args.lr,
                    emb_dim=args.emb_dim,
                    n_mels=args.n_mels,
                    sample_rate=args.sample_rate,
                    max_seconds=args.max_seconds,
                    margin=args.margin,
                    pairs_per_epoch=args.pairs_per_epoch,
                    seed=seed,
                    num_workers=args.num_workers
                )

                try:
                    metrics = train_and_eval(run_args)
                except Exception as e:
                    print(f"    ERROR training on {subset_dir.name} seed={seed}: {e}")
                    continue

                # append metrics to CSV
                try:
                    with open(csv_path, 'a', encoding='utf-8') as fo:
                        eer_s = '' if metrics.get('eer') is None else f"{metrics['eer']:.6f}"
                        auc_s = '' if metrics.get('auc') is None else f"{metrics['auc']:.6f}"
                        thr_s = '' if metrics.get('thr') is None else f"{metrics['thr']:.6f}"
                        fo.write(f"{eer_s},{auc_s},{thr_s},{num_samples}\n")
                    print(f"    Recorded results to {csv_path}")
                except Exception as e:
                    print(f"    Could not write results to {csv_path}: {e}")

    print("All done.")

if __name__ == '__main__':
    main()