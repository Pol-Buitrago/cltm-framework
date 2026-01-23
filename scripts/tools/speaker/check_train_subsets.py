#!/usr/bin/env python3
import pandas as pd
from pathlib import Path
import argparse

def main():
    parser = argparse.ArgumentParser(description="Listar los train.tsv con al menos 500 muestras por clase.")
    parser.add_argument("--root", required=True, help="Directorio raíz donde buscar (por ejemplo: ./out)")
    parser.add_argument("--label-col", default="gender", help="Columna de etiqueta (default: gender)")
    parser.add_argument("--min-per-class", type=int, default=500, help="Umbral mínimo por clase (default: 500)")
    args = parser.parse_args()

    root = Path(args.root)
    train_files = list(root.rglob("*.train.tsv"))

    if not train_files:
        print(f"No se encontraron archivos *.train.tsv dentro de {root}")
        return

    print(f"Analizando {len(train_files)} ficheros encontrados en {root}...\n")
    valid_files = []

    for path in sorted(train_files):
        try:
            df = pd.read_csv(path, sep="\t", dtype=str)
            if args.label_col not in df.columns:
                print(f"[WARN] {path}: columna '{args.label_col}' no encontrada, se omite.")
                continue

            counts = df[args.label_col].value_counts().to_dict()
            min_class = min(counts.values())
            total = len(df)

            if all(v >= args.min_per_class for v in counts.values()):
                valid_files.append((path, total, counts))
                print(f"[OK] {path} → {counts} (total {total})")
            else:
                print(f"[SKIP] {path} → {counts} (alguna clase < {args.min_per_class})")

        except Exception as e:
            print(f"[ERROR] {path}: {e}")

    print("\nResumen final:")
    if valid_files:
        print(f"Total con ≥{args.min_per_class} por clase: {len(valid_files)}\n")
        for path, total, counts in valid_files:
            print(f" - {path}: {counts} (total {total})")
    else:
        print(f"Ningún train.tsv cumple el umbral de {args.min_per_class} por clase.")

if __name__ == "__main__":
    main()
