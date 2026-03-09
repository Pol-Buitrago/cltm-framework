#!/usr/bin/env python3
"""
make_batches.py

Orquestador que descubre *.train.tsv en las carpetas indicadas, evita
experimentos ya registrados en los CSV destino, y genera archivos
batch_{i:04d}.sh. Cada batch script incluirá la cabecera sbatch que
me pediste.

Usage: ver --help
"""

import argparse
from pathlib import Path
import csv

def load_existing_entries(csv_path):
    s = set()
    p = Path(csv_path)
    if not p.exists():
        return s
    with open(p, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            if len(row) < 5:
                continue
            model_id = row[0].strip()
            typ = row[1].strip()
            lang_src = row[2].strip()
            lang_tgt = row[3].strip()
            seed = row[4].strip()
            s.add((model_id, typ, lang_src, lang_tgt, seed))
    return s

def parse_langs_from_fname(fname):
    base = Path(fname).name
    if "_" in base:
        main = base.split(".")[0]
        parts = main.split("_")
        if len(parts) >= 2:
            return parts[0], parts[1]
    main = base.split(".")[0]
    return main, ""

def make_command(run_script, train_tsv, test_tsv, out_dir, csv_path, model_id, typ, lang_src, lang_tgt, seed, extra_args=""):
    cmd = (
        f"python {run_script} "
        f"--train_tsv '{train_tsv}' --test_tsv '{test_tsv}' --out_dir '{out_dir}' "
        f"--csv_path '{csv_path}' --model_id '{model_id}' --type '{typ}' --lang_src '{lang_src}' "
        f"--lang_tgt '{lang_tgt}' --seed {seed} {extra_args}"
    )
    return cmd

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pairs_dir", required=True)
    p.add_argument("--single1000_dir", required=True)
    p.add_argument("--single2000_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--run_script", default="./run_and_log.py")
    p.add_argument("--bilingual_csv", required=True)
    p.add_argument("--single_csv", required=True)
    p.add_argument("--single2000_csv", required=True)
    p.add_argument("--seeds", default="42-51")
    p.add_argument("--batch_size", type=int, default=1000)
    p.add_argument("--extra_args", default="")
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "logs").mkdir(parents=True, exist_ok=True)

    smin, smax = map(int, args.seeds.split("-", 1))
    seeds = list(range(smin, smax + 1))

    existing_bi = load_existing_entries(args.bilingual_csv)
    existing_s1 = load_existing_entries(args.single_csv)
    existing_s2 = load_existing_entries(args.single2000_csv)

    tasks = []

    # 1) bilingual pairs
    pairs_dir = Path(args.pairs_dir)
    for train_path in sorted(pairs_dir.glob("*.train.tsv")):
        lang_src, lang_tgt = parse_langs_from_fname(train_path.name)
        model_id = f"{lang_src}_{lang_tgt}"
        test_path = pairs_dir / f"{lang_src}.test.tsv"
        if not test_path.exists():
            print(f"[WARN] test {test_path} missing for {train_path}, skipping")
            continue
        typ = "dual"
        csv_target = args.bilingual_csv
        for seed in seeds:
            key = (model_id, typ, lang_src, lang_tgt, str(seed))
            if key in existing_bi:
                continue
            cmd = make_command(args.run_script, train_path, test_path, "./out", csv_target, model_id, typ, lang_src, lang_tgt, seed, args.extra_args)
            tasks.append((cmd, model_id, typ, lang_src, lang_tgt, seed))

    # 2) single 1000
    s1000_dir = Path(args.single1000_dir)
    for train_path in sorted(s1000_dir.glob("*.train.tsv")):
        lang_src, _ = parse_langs_from_fname(train_path.name)
        model_id = lang_src
        test_path = s1000_dir / f"{lang_src}.test.tsv"
        if not test_path.exists():
            print(f"[WARN] test {test_path} missing for {train_path}, skipping")
            continue
        typ = "single"
        csv_target = args.single_csv
        for seed in seeds:
            key = (model_id, typ, lang_src, "", str(seed))
            if key in existing_s1:
                continue
            cmd = make_command(args.run_script, train_path, test_path, "./out", csv_target, model_id, typ, lang_src, "", seed, args.extra_args)
            tasks.append((cmd, model_id, typ, lang_src, "", seed))

    # 3) single 2000 (guardado en single2000_csv)
    s2000_dir = Path(args.single2000_dir)
    for train_path in sorted(s2000_dir.glob("*.train.tsv")):
        lang_src, _ = parse_langs_from_fname(train_path.name)
        model_id = lang_src
        test_path = s2000_dir / f"{lang_src}.test.tsv"
        if not test_path.exists():
            print(f"[WARN] test {test_path} missing for {train_path}, skipping")
            continue
        typ = "single"
        csv_target = args.single2000_csv
        for seed in seeds:
            key = (model_id, typ, lang_src, "", str(seed))
            if key in existing_s2:
                continue
            cmd = make_command(args.run_script, train_path, test_path, "./out", csv_target, model_id, typ, lang_src, "", seed, args.extra_args)
            tasks.append((cmd, model_id, typ, lang_src, "", seed))

    batch_size = args.batch_size
    n_batches = (len(tasks) + batch_size - 1) // batch_size
    print(f"Found {len(tasks)} pending tasks, generating {n_batches} batch scripts (batch_size={batch_size})")

    # Cabecera sbatch solicitada (exacta)
    sbatch_header = """#!/bin/bash
#SBATCH --account bsc88
#SBATCH --qos acc_bscls
#SBATCH --nodes 1
#SBATCH --gres gpu:1
#SBATCH --cpus-per-task 20
#SBATCH --time 0-12:00:00
#SBATCH --job-name=speechLLM_train
#SBATCH --output=/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/logs/speechLLM_train_%j.log
#SBATCH --error=/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/logs/speechLLM_train_%j.err

set -e

echo "=== Activating conda ==="

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /gpfs/projects/bsc88/speech/mm_s2st/scripts/paraling/gender_id_hubert/env/

echo "=== Running speechLLM training ==="
"""

    for i in range(n_batches):
        batch_tasks = tasks[i*batch_size : (i+1)*batch_size]
        batch_fname = out_dir / f"batch_{i:04d}.sh"
        with open(batch_fname, "w", encoding="utf-8") as fo:
            fo.write(sbatch_header + "\n")
            fo.write("set -euo pipefail\n\n")
            fo.write("module purge\n")
            fo.write("# Carga tus módulos / activa conda aquí si hace falta. Ej:\n")
            fo.write("# module load cuda/xx\n")
            fo.write("# source /path/to/conda.sh && conda activate tu_env\n\n")
            for cmd, model_id, typ, lang_src, lang_tgt, seed in batch_tasks:
                fo.write(f"echo '--- TASK start: model={model_id} seed={seed}'\n")
                fo.write(cmd + "\n")
                fo.write(f"echo '--- TASK end: model={model_id} seed={seed}'\n\n")
        batch_fname.chmod(0o755)
        print(f"Wrote {batch_fname}")

    submit_all = out_dir / "submit_all.sh"
    with open(submit_all, "w", encoding="utf-8") as fo:
        fo.write("#!/bin/bash\n")
        fo.write("set -euo pipefail\n")
        fo.write("mkdir -p logs\n")
        for i in range(n_batches):
            fo.write(f"sbatch {out_dir}/batch_{i:04d}.sh\n")
    submit_all.chmod(0o755)
    print(f"Wrote {submit_all} (usa este script para enviar todos los batches)")

if __name__ == "__main__":
    main()
