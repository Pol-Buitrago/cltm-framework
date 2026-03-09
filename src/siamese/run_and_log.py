#!/usr/bin/env python3
"""
run_and_log.py

Llama a sv_siamese.train_and_eval(...) con los argumentos apropiados y
apendea el resultado al csv indicado (crea cabecera si no existe).
"""

import argparse
import os
import sys
import csv
import json
import time
import fcntl
from pathlib import Path
from types import SimpleNamespace

# Asegúrate que el path a sv_siamese.py es accesible (modifica si es necesario)
# Si sv_siamese.py está en el mismo directorio que este script basta.
# Ejemplo: export PYTHONPATH=/gpfs/.../repos/paraling_speech/src:$PYTHONPATH
# o colocarlo en el mismo dir.
import importlib

def append_row_atomic(csv_path, header, row):
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    first = not csv_path.exists()
    with open(csv_path, "a+", encoding="utf-8", newline='') as f:
        # bloqueo POSIX
        try:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        except Exception:
            # Si flock no está disponible, seguimos de todas formas (cluster POSIX tendrá flock)
            pass
        if first:
            writer = csv.writer(f)
            writer.writerow(header)
        f.seek(0, os.SEEK_END)
        writer = csv.writer(f)
        writer.writerow(row)
        try:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        except Exception:
            pass

def build_args_namespace(cli_args):
    # Construye un namespace compatible con train_and_eval
    ns = SimpleNamespace()
    # Copia argumentos conocidos (puedes añadir los que necesites)
    ns.train_tsv = cli_args.train_tsv
    ns.test_tsv = cli_args.test_tsv
    ns.out_dir = cli_args.out_dir
    ns.epochs = cli_args.epochs
    ns.batch_size = cli_args.batch_size
    ns.lr = cli_args.lr
    ns.emb_dim = cli_args.emb_dim
    ns.n_mels = cli_args.n_mels
    ns.sample_rate = cli_args.sample_rate
    ns.max_seconds = cli_args.max_seconds
    ns.margin = cli_args.margin
    ns.pairs_per_epoch = cli_args.pairs_per_epoch
    ns.seed = cli_args.seed
    ns.num_workers = cli_args.num_workers
    return ns

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train_tsv", required=True)
    p.add_argument("--test_tsv", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--csv_path", required=True, help="CSV destino donde apendear resultados")
    p.add_argument("--model_id", required=True)
    p.add_argument("--type", required=True, choices=["single","dual"])
    p.add_argument("--lang_src", required=True)
    p.add_argument("--lang_tgt", default="")
    p.add_argument("--seed", type=int, default=42)
    # hyperparams (pasar explicitamente si quieres otros valores)
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
    args = p.parse_args()

    # IMPORT dinamico para evitar errores si el path no es correcto
    try:
        sv = importlib.import_module("sv_siamese")
    except Exception as e:
        print("ERROR: no se pudo importar sv_siamese. Asegúrate que sv_siamese.py está en PYTHONPATH.", file=sys.stderr)
        raise

    ns = build_args_namespace(args)

    # Llamada a train_and_eval (el training + eval se ejecuta aquí)
    print(f"=== RUN model_id={args.model_id}, seed={args.seed}, train={args.train_tsv}, test={args.test_tsv}")
    start = time.time()
    result = sv.train_and_eval(ns)
    dt = time.time() - start
    print(f"=== DONE in {dt:.1f}s. result: {result}")

    # Escribir CSV (cabecera: model_id,type,lang_src,lang_tgt,seed,eer,auc,threshold)
    header = ["model_id","type","lang_src","lang_tgt","seed","eer","auc","threshold"]
    eer = result.get("eer")
    auc = result.get("auc")
    thr = result.get("eer_threshold")
    row = [args.model_id, args.type, args.lang_src, args.lang_tgt, args.seed,
           f"{float(eer):.12f}" if eer is not None else "",
           f"{float(auc):.12f}" if auc is not None else "",
           f"{float(thr):.12f}" if thr is not None else ""]
    append_row_atomic(args.csv_path, header, row)
    print(f"Appended to {args.csv_path}: {row}")

if __name__ == "__main__":
    main()
