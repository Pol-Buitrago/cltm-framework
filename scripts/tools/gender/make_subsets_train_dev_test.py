#!/usr/bin/env python3
# utils/make_subsets_train_dev_test.py
"""
Crear subsets partiendo de train 50/50 con control flexible de la reducción por clase.

Modos:
 - geometric (default): --n-subsets K, genera K niveles geométricos entre start y min-per-class
 - linear: --n-subsets K, genera K niveles lineales (valores absolutos)
 - divisors: --divisors 2 2 3, aplica sucesivamente divisores enteros sobre el valor actual (per_class // d)

Ejemplos:
  # modo geométrico (suave), 5 subsets
  python make_subsets_train_dev_test.py --train-tsv train.tsv --dev-tsv dev.tsv --test-tsv test.tsv --out-dir out --mode geometric --n-subsets 5

  # modo lineal (pasos iguales)
  python ... --mode linear --n-subsets 6

  # modo divisors (control por divisores enteros sucesivos)
  python ... --mode divisors --divisors 2 2 3 2

"""
#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import json
from datetime import datetime, timezone
import hashlib
import math
import sys


def parse_args():
    p = argparse.ArgumentParser(description="Crear subsets simples partiendo de train 50/50")
    p.add_argument("--train-tsv", required=True, help="Path a train.tsv (DEBE estar 50/50)")
    p.add_argument("--dev-tsv", required=False, default=None, help="Path a dev.tsv (opcional). Si no se pasa, se IGNORA.")
    p.add_argument("--test-tsv", required=True, help="Path a test.tsv (se copia igual)")
    p.add_argument("--out-dir", required=True, help="Directorio raíz donde crear subset_{i}")
    p.add_argument("--label-col", default="gender", help="Nombre de la columna de label (default: gender)")
    p.add_argument("--min-per-class", type=int, default=20, help="Umbral mínimo por clase para dejar de generar subsets")
    p.add_argument("--seed", type=int, default=42, help="Seed para muestreo reproducible")
    # new params
    p.add_argument("--mode", choices=["geometric","linear","divisors"], default="geometric",
                   help="Estrategia para generar pasos de reducción (default: geometric)")
    p.add_argument("--n-subsets", type=int, default=5,
                   help="(geometric|linear) Número aproximado de subsets a generar (incluye el full-train).")
    p.add_argument("--divisors", type=int, nargs="*", default=[],
                   help="(divisors) Secuencia de divisores enteros aplicados sucesivamente al valor actual (ej. 2 2 3 2)")

    # NEW: explicit per-class range support
    p.add_argument("--per-class-max", type=int, default=None,
                   help="(optional) If set, generate subsets for per-class = 1..PER_CLASS_MAX (step via --per-class-step).")
    p.add_argument("--per-class-step", type=int, default=1,
                   help="Step for the explicit per-class range (default 1).")
    return p.parse_args()


def read_tsv(path):
    return pd.read_csv(path, sep="\t", dtype=str)


def md5_of_file(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def compute_counts(df, label_col):
    vc = df[label_col].value_counts().to_dict()
    keys = sorted(df[label_col].unique())
    return {k: int(vc.get(k, 0)) for k in keys}


def check_balanced_train(train_df, label_col):
    vals = sorted(train_df[label_col].unique().tolist())
    if len(vals) != 2:
        raise ValueError(f"train must have exactly 2 label values, found: {vals}")
    counts = train_df[label_col].value_counts().to_dict()
    vals_counts = list(counts.values())
    if len(set(vals_counts)) != 1:
        raise ValueError(f"train.tsv is not balanced 50/50 by '{label_col}'. Counts: {counts}")
    per_class = vals_counts[0]
    return vals, per_class


def stratified_sample_per_class(df, label_col, per_class, rng):
    parts = []
    for v in sorted(df[label_col].unique()):
        group = df[df[label_col] == v]
        if len(group) < per_class:
            raise ValueError(f"No hay suficientes muestras en clase '{v}': {len(group)} < {per_class}")
        sampled = group.sample(n=per_class, random_state=rng, replace=False)
        parts.append(sampled)
    out = pd.concat(parts).sample(frac=1, random_state=rng).reset_index(drop=True)
    return out

# ------------------ new: generate per_class sequence ------------------
def generate_per_class_sequence(start, min_val, mode="geometric", n_subsets=5, divisors=None):
    """
    Devuelve lista de enteros decrecientes >= min_val, empezando por start.
    - geometric: geomspace start -> min_val (n_subsets samples)
    - linear: linspace start -> min_val (n_subsets samples)
    - divisors: aplicar sucesivamente divisores enteros sobre el valor actual (dividir por d: //d)
    Se garantiza unicidad y orden decreciente. Siempre se incluye 'start' y 'min_val' (si no redundante).
    """
    divisors = divisors or []
    if start <= 0 or min_val <= 0:
        raise ValueError("start and min_val must be positive integers")
    if min_val > start:
        raise ValueError("min_val must be <= start")

    seq = []
    if mode == "divisors":
        # include start
        seq.append(int(start))
        cur = int(start)
        for d in divisors:
            if d <= 1:
                continue
            nxt = cur // d
            if nxt < min_val:
                nxt = min_val
            if nxt >= cur:
                # avoid infinite loop
                break
            if nxt not in seq:
                seq.append(int(nxt))
            cur = int(nxt)
            if cur <= min_val:
                break
        # ensure min_val is present
        if seq[-1] != min_val:
            seq.append(int(min_val))
    else:
        # mode geometric or linear
        n = max(2, int(n_subsets))
        if mode == "geometric":
            # use floating geomspace and round down
            floats = np.geomspace(start, max(min_val,1), num=n)
            ints = [max(int(math.floor(x)), 1) for x in floats]
        else:
            floats = np.linspace(start, min_val, num=n)
            ints = [max(int(round(x)), 1) for x in floats]

        # ensure decreasing unique ints
        uniq = []
        for v in ints:
            if not uniq or v != uniq[-1]:
                uniq.append(v)
        uniq = sorted(set(uniq), reverse=True)
        if uniq[0] != start:
            uniq.insert(0, int(start))
        if uniq[-1] != min_val:
            uniq.append(int(min_val))
        seq = [int(x) for x in uniq]

    # final cleanup: remove values < min_val and duplicates, ensure monotonic decreasing
    seq = [int(x) for x in seq if int(x) >= min_val]
    cleaned = []
    prev = None
    for x in seq:
        if prev is None or x < prev:
            cleaned.append(x)
            prev = x
    if cleaned[-1] != min_val:
        cleaned.append(min_val)
    if cleaned[0] != start:
        cleaned.insert(0, start)
    final = []
    seen = set()
    for x in cleaned:
        if x not in seen:
            final.append(x)
            seen.add(x)
    return final

# ---------------------------------------------------------------------

def main():
    args = parse_args()
    rng_global = np.random.RandomState(args.seed)

    train_df = read_tsv(args.train_tsv)

    # dev is optional: only read if provided and exists
    dev_df = None
    dev_present = False
    if args.dev_tsv:
        dev_path = Path(args.dev_tsv)
        if dev_path.exists():
            dev_df = read_tsv(args.dev_tsv)
            dev_present = True
        else:
            print(f"[WARN] dev-tsv provided but file not found: {args.dev_tsv}. Continuing without dev.", file=sys.stderr)
            dev_present = False

    test_df  = read_tsv(args.test_tsv)

    # Derivar prefijo a partir del nombre del fichero train (texto antes del primer punto)
    train_name = Path(args.train_tsv).name
    if "." in train_name:
        prefix = train_name.split('.', 1)[0]
    else:
        prefix = ""

    # check train is exactly 50/50
    vals, per_class_start = check_balanced_train(train_df, args.label_col)

    # compute sequence of per_class values
    per_class_list = []

    # If user requests explicit per-class range (1..per_class_max with step)
    if args.per_class_max is not None:
        upper = min(int(args.per_class_max), int(per_class_start))
        lower = max(int(args.min_per_class), 1)
        step = max(1, int(args.per_class_step))
        if lower > upper:
            raise ValueError(f"After clamping, lower ({lower}) > upper ({upper}). Check min-per-class and per-class-max.")
        explicit = list(range(lower, upper + 1, step))
        per_class_list = explicit
    else:
        if args.mode in ("geometric","linear"):
            per_class_list = generate_per_class_sequence(per_class_start, args.min_per_class,
                                                         mode=args.mode, n_subsets=args.n_subsets)
        else:
            per_class_list = generate_per_class_sequence(per_class_start, args.min_per_class,
                                                         mode="divisors", divisors=args.divisors)

    # Always ensure we include the full-train per_class_start as the first element (if not already present)
    if per_class_list is None:
        per_class_list = [int(per_class_start)]
    else:
        per_class_list = [int(x) for x in per_class_list if int(x) >= args.min_per_class and int(x) <= per_class_start]
        per_class_list = sorted(list(dict.fromkeys(per_class_list)), reverse=True)
        if per_class_start not in per_class_list:
            per_class_list.insert(0, int(per_class_start))

    if per_class_list[0] != per_class_start:
        per_class_list.insert(0, per_class_start)

    per_class_list = [int(x) for x in per_class_list if int(x) >= args.min_per_class]
    per_class_list = sorted(list(dict.fromkeys(per_class_list)), reverse=True)

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    base_stats = {
        "source_train": str(Path(args.train_tsv).resolve()),
        "source_dev": str(Path(args.dev_tsv).resolve()) if dev_present else None,
        "source_test": str(Path(args.test_tsv).resolve()),
        "total_train_rows": int(train_df.shape[0]),
        "per_class_train_start": int(per_class_start),
        "total_dev_rows": int(dev_df.shape[0]) if dev_present else None,
        "dev_class_counts": compute_counts(dev_df, args.label_col) if dev_present else None,
        "total_test_rows": int(test_df.shape[0]),
        "test_class_counts": compute_counts(test_df, args.label_col),
        "label_col": args.label_col,
        "seed_global": int(args.seed),
        "mode": args.mode,
        "n_subsets": args.n_subsets,
        "divisors": args.divisors,
        "input_prefix": prefix,
        "dev_present": bool(dev_present)
    }

    summary_rows = []
    idx = 1

    for per_class in per_class_list:
        subset_name = f"subset_{idx:02d}_perclass_{per_class}"
        if prefix:
            subset_dir = out_root / f"{prefix}.{subset_name}"
        else:
            subset_dir = out_root / subset_name
        subset_dir.mkdir(parents=True, exist_ok=True)

        rng_train = rng_global.randint(0, 2**31-1)
        if per_class == per_class_start:
            train_sub = train_df.sample(frac=1, random_state=rng_train).reset_index(drop=True)
        else:
            train_sub = stratified_sample_per_class(train_df, args.label_col, per_class, rng_train)

        # handle dev only if present
        dev_sub_usable = None
        dev_sub = None
        rng_dev = None
        dev_counts = None
        if dev_present:
            rng_dev = rng_global.randint(0, 2**31-1)
            dev_counts = compute_counts(dev_df, args.label_col)
            if any(c < per_class for c in dev_counts.values()):
                dev_sub = dev_df.copy()
                dev_sub_usable = False
            else:
                dev_sub = stratified_sample_per_class(dev_df, args.label_col, per_class, rng_dev)
                dev_sub_usable = True

        # write files: train always, test always, dev only if present
        train_filename = f"{prefix}.train.tsv" if prefix else "train.tsv"
        test_filename = f"{prefix}.test.tsv" if prefix else "test.tsv"
        train_out = subset_dir / train_filename
        test_out  = subset_dir / test_filename
        train_sub.to_csv(train_out, sep="\t", index=False)
        test_df.to_csv(test_out, sep="\t", index=False)
        dev_out = None
        if dev_present:
            dev_filename = f"{prefix}.dev.tsv" if prefix else "dev.tsv"
            dev_out = subset_dir / dev_filename
            dev_sub.to_csv(dev_out, sep="\t", index=False)

        # md5 and counts
        train_md5 = md5_of_file(train_out)
        test_md5  = md5_of_file(test_out)
        dev_md5 = None
        if dev_present:
            dev_md5 = md5_of_file(dev_out)
        train_rows = int(train_sub.shape[0])
        test_rows = int(test_df.shape[0])
        dev_rows = int(dev_sub.shape[0]) if (dev_present and dev_sub is not None) else None
        train_counts_sub = compute_counts(train_sub, args.label_col)
        dev_counts_sub = compute_counts(dev_sub, args.label_col) if (dev_present and dev_sub is not None) else None

        meta = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "subset_name": subset_name,
            "per_class_train": int(per_class),
            "total_train": train_rows,
            "per_class_dev_requested": int(per_class) if dev_present else None,
            "per_class_dev_actual": {k: int(dev_counts_sub.get(k, 0)) for k in dev_counts_sub} if dev_counts_sub is not None else None,
            "dev_subsampled": bool(dev_sub_usable) if dev_present else None,
            "source_train": str(Path(args.train_tsv).resolve()),
            "source_dev": str(Path(args.dev_tsv).resolve()) if dev_present else None,
            "source_test": str(Path(args.test_tsv).resolve()),
            "seed_global": int(args.seed),
            "rng_train_seed": int(rng_train),
            "rng_dev_seed": int(rng_dev) if dev_present else None,
            "label_col": args.label_col,
            "train_md5": train_md5,
            "dev_md5": dev_md5,
            "test_md5": test_md5,
            "train_rows": train_rows,
            "dev_rows": dev_rows,
            "test_rows": test_rows,
            "train_class_counts": train_counts_sub,
            "dev_class_counts": dev_counts_sub,
            "output_filenames": {
                "train": train_filename,
                "dev": dev_filename if dev_present else None,
                "test": test_filename
            },
            "subset_dir": str(subset_dir.resolve())
        }
        write_json(subset_dir / "metadata.json", meta)

        summary_rows.append({
            "subset_name": subset_name,
            "subset_dir": str(subset_dir.resolve()),
            "per_class_train": per_class,
            "total_train": train_rows,
            "train_md5": train_md5,
            "per_class_dev_requested": int(per_class) if dev_present else None,
            "per_class_dev_actual": dev_counts_sub if dev_present else None,
            "dev_subsampled": bool(dev_sub_usable) if dev_present else None,
            "total_dev": dev_rows,
            "dev_md5": dev_md5,
            "total_test": test_rows,
            "test_md5": test_md5
        })

        if dev_present:
            print(f"[INFO] {subset_name}: train {train_rows} {train_counts_sub}, dev {dev_rows} {dev_counts_sub} (dev_subsampled={dev_sub_usable})")
            print(f"[INFO] Written files: {train_out}, {dev_out}, {test_out}")
        else:
            print(f"[INFO] {subset_name}: train {train_rows} {train_counts_sub}, dev NOT PROVIDED, test {test_rows}")
            print(f"[INFO] Written files: {train_out}, {test_out}")

        idx += 1

    # write summary files
    summary_csv = out_root / "subsets_summary.csv"
    rows_for_csv = []
    for r in summary_rows:
        rows_for_csv.append({
            "subset_name": r["subset_name"],
            "subset_dir": r["subset_dir"],
            "per_class_train": r["per_class_train"],
            "total_train": r["total_train"],
            "per_class_dev_requested": r["per_class_dev_requested"],
            "dev_subsampled": r["dev_subsampled"],
            "total_dev": r["total_dev"],
            "total_test": r["total_test"],
            "train_md5": r["train_md5"],
            "dev_md5": r["dev_md5"],
            "test_md5": r["test_md5"]
        })
    pd.DataFrame(rows_for_csv).to_csv(summary_csv, index=False)

    summary_json = out_root / "subsets_summary.json"
    write_json(summary_json, {"base_stats": base_stats, "subsets": summary_rows})

    print(f"[DONE] Generados {len(summary_rows)} subsets en {out_root}")
    print(f"[INFO] Summary CSV: {summary_csv}")
    print(f"[INFO] Summary JSON: {summary_json}")


if __name__ == "__main__":
    main()



"""
python make_subsets_train_dev_test.py \
  --train-tsv "/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__gender_id/03_balanced_cv_gender/tsv/ca.train.tsv" \
  --dev-tsv   "/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__gender_id/03_balanced_cv_gender/tsv/ca.dev.tsv" \
  --test-tsv  "/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__gender_id/03_balanced_cv_gender/tsv/ca.test.tsv" \
  --out-dir "/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/experimental/subsets" \
  --label-col "gender" \
  --mode linear --n-subsets 250 \
  --min-per-class 1 \
  --seed 42

python make_subsets_train_dev_test.py \
  --train-tsv "/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__gender_id/03_balanced_cv_gender/tsv/zh-CN.train.tsv" \
  --test-tsv  "/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__gender_id/03_balanced_cv_gender/tsv/zh-CN.test.tsv" \
  --out-dir "/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/experimental/subsets" \
  --label-col "gender" \
  --min-per-class 1 \
  --per-class-max 250 \
  --per-class-step 1 \
  --seed 42
"""