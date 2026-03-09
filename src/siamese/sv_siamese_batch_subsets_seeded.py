#!/usr/bin/env python3
"""
sv_siamese_batch_subsets_seeded.py

Batch runner for training a siamese speaker verification model per-subset and per-seed.

Features:
- Walks a root "subsets" directory structured as: <root>/<lang>/subset_XXX_<...>/
  Each subset directory is expected to contain a train TSV and a test TSV (e.g. ca.train.tsv / ca.test.tsv).
- For each language, each subset and each seed, trains+evaluates the siamese model using the training TSV inside the subset
  and the test TSV inside the subset.
- Results are appended to CSV files named: <lang>.eer_by_samples_<seed>.csv in an outputs directory.
  CSV columns: eer,auc,thr,num_samples
- If a CSV for that (lang,seed) already contains an entry with the same num_samples, the run is skipped.
- Resumable and robust: skips missing test/train, logs warnings and continues.

Usage example:
    python sv_siamese_batch_subsets_seeded.py \
        --subsets_root /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/experimental/subsets_speaker \
        --out_root /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/subsets/siamese/speaker/seeds \
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

import numpy as np
import torch

# The full siamese training/eval code is included below so this file is self-contained.
# (Based on the prior script you provided, extended to run over many subsets & seeds.)

import torchaudio
import librosa
import torch.nn as nn
import torch.optim as optim

# ------------------------
# Utilities
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

# Audio utils

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
        with torch.no_grad():
            S = self.melspec(waveform.unsqueeze(0))
            S = self.db(S)
            S = (S - S.mean()) / (S.std() + 1e-6)
        return S.squeeze(0)

# ------------------------
# Dataset
# ------------------------
class PairDataset(torch.utils.data.Dataset):
    def __init__(self, tsv_path, max_seconds=3.0, sample_rate=16000, transform=None, pairs_per_epoch=100000):
        self.tsv_path = tsv_path
        self.max_seconds = max_seconds
        self.sample_rate = sample_rate
        self.transform = transform or LogMelExtractor(sample_rate=sample_rate)
        self.pairs_per_epoch = pairs_per_epoch

        self.client2files = defaultdict(list)
        with open(tsv_path, 'r', encoding='utf-8') as f:
            header = f.readline()
            for line in f:
                if not line.strip():
                    continue
                parts = line.strip().split('\t')
                if len(parts) < 4:
                    continue
                _, client_id, audio_rel, _ = parts[:4]
                audio_path = audio_rel
                self.client2files[client_id].append(audio_path)

        self.clients = [c for c, files in self.client2files.items() if len(files) >= 2]
        if len(self.clients) < 2:
            raise ValueError("Need at least two clients with >=2 utterances each")

        self.all_files = []
        for files in self.client2files.values():
            self.all_files.extend(files)

        self.feature_cache = {}

    def __len__(self):
        return self.pairs_per_epoch

    def _load_feature(self, path):
        if path in self.feature_cache:
            return self.feature_cache[path]
        waveform = load_audio(path, sample_rate=self.sample_rate, max_seconds=self.max_seconds)
        feat = self.transform(waveform)
        self.feature_cache[path] = feat
        return feat

    def __getitem__(self, idx):
        same = random.random() < 0.5
        if same:
            client = random.choice(self.clients)
            a, b = random.sample(self.client2files[client], 2)
            label = 1.0
        else:
            c1, c2 = random.sample(self.clients, 2)
            a = random.choice(self.client2files[c1])
            b = random.choice(self.client2files[c2])
            label = 0.0

        A = self._load_feature(a)
        B = self._load_feature(b)
        return A.unsqueeze(0).float(), B.unsqueeze(0).float(), torch.tensor(label, dtype=torch.float32)

def collate_pairs(batch):
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
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_mels, dummy_frames)
            out = self.conv(dummy)
            C, F_after, T_after = out.shape[1], out.shape[2], out.shape[3]
            in_features = 2 * C * F_after
        self.fc = nn.Linear(in_features, emb_dim)

    def forward(self, x):
        out = self.conv(x)
        B, C, F, T = out.shape
        out = out.permute(0, 3, 1, 2).contiguous().view(B, T, C * F)
        mean = out.mean(dim=1)
        std = out.std(dim=1)
        stats = torch.cat([mean, std], dim=1)
        emb = self.fc(stats)
        emb = nn.functional.normalize(emb, p=2, dim=1)
        return emb

# ------------------------
# Loss
# ------------------------

def contrastive_loss(emb1, emb2, label, margin=1.0):
    dist = torch.norm(emb1 - emb2, p=2, dim=1)
    loss_pos = label * (dist ** 2)
    loss_neg = (1.0 - label) * (torch.clamp(margin - dist, min=0.0) ** 2)
    loss = loss_pos + loss_neg
    return loss.mean()

# ------------------------
# Evaluation utils
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
                wav = load_audio(p, max_seconds=max_seconds)
                f = extractor(wav).unsqueeze(0).unsqueeze(0)
                feats.append(f)
            feats = torch.cat(feats, dim=0).to(device)
            embs = model(feats).cpu().numpy()
            for k, e in zip(batch_paths, embs):
                emb_cache[k] = e
    return emb_cache


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

# ------------------------
# Train & eval (single run)
# ------------------------

def train_and_eval(args):
    """Train the model given args and evaluate on args.test_tsv. Returns dict with eer, auc, thr."""
    import itertools
    import warnings
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # reproducibility
    set_seed(args.seed)

    transform = LogMelExtractor(sample_rate=args.sample_rate, n_mels=args.n_mels)
    dataset = PairDataset(args.train_tsv, max_seconds=args.max_seconds, sample_rate=args.sample_rate,
                          transform=transform, pairs_per_epoch=args.pairs_per_epoch)

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

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        t0 = time.time()
        for i, (A, B, L) in enumerate(loader):
            A = A.to(device)
            B = B.to(device)
            L = L.to(device)
            opt.zero_grad()
            embA = model(A)
            embB = model(B)
            loss = contrastive_loss(embA, embB, L, margin=args.margin)
            loss.backward()
            opt.step()
            running_loss += loss.item()
            if i % 200 == 0:
                try:
                    dataset.feature_cache.clear()
                except Exception:
                    pass
        scheduler.step()
        dt = time.time() - t0
        lr_now = scheduler.get_last_lr()[0] if hasattr(scheduler, "get_last_lr") else opt.param_groups[0]['lr']
        print(f"Epoch {epoch}/{args.epochs}, loss={running_loss/(i+1):.4f}, time={dt:.1f}s, lr={lr_now:.2e}")
        torch.save(model.state_dict(), out_dir / f"model_epoch{epoch}.pt")

    # Evaluate
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

    all_paths = [p for p, q, _ in pairs] + [q for p, q, _ in pairs]
    unique_paths = []
    seen = set()
    for p in all_paths:
        if p not in seen:
            unique_paths.append(p)
            seen.add(p)

    emb_cache = compute_embeddings(unique_paths, model, device, transform, max_seconds=args.max_seconds, batch_size=64)

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

    # AUC
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

    return {"eer": float(eer), "auc": (None if auc is None else float(auc)), "thr": float(thr)}

# ------------------------
# Batch orchestration
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
    """Return set of num_samples already recorded in csv_path. If file missing, return empty set."""
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
    """Try to find train and test tsvs in subset_dir. Returns (train_path, test_path) or (None, None)."""
    # prefer explicit names like <lang>.train.tsv, <lang>.test.tsv
    candidates = list(subset_dir.glob('*.tsv'))
    train_path = None
    test_path = None
    for p in candidates:
        name = p.name.lower()
        if name.endswith('.train.tsv') and lang in name:
            train_path = p
        if name.endswith('.test.tsv') and lang in name:
            test_path = p
    # fallback: any *.train.tsv / *.test.tsv
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
    # final fallback: look for <lang>.tsv pairs
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
                    # append placeholder? no
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
