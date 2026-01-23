#!/usr/bin/env python3
import pandas as pd
import numpy as np
import glob
import os
import argparse

def main():
    parser = argparse.ArgumentParser(description="Combine multiple seed CSVs into a single averaged CSV")
    parser.add_argument("--indir", required=True, help="Directory containing CSVs")
    parser.add_argument("--lang", required=True, help="Language or prefix (e.g., 'en', 'es', 'de')")
    parser.add_argument("--pattern", default="{lang}.f1_by_samples_*.csv", help="Pattern to match input CSVs")
    parser.add_argument("--outdir", default=None, help="Directory to save the combined CSV (default: same as indir)")
    args = parser.parse_args()

    indir = args.indir
    lang = args.lang
    pattern = args.pattern.format(lang=lang)
    outdir = args.outdir or indir
    out_path = os.path.join(outdir, f"{lang}.f1_by_samples.csv")

    csv_files = sorted(glob.glob(os.path.join(indir, pattern)))
    if not csv_files:
        raise ValueError(f"No CSV files found in {indir} matching {pattern}")

    dfs = []
    for f in csv_files:
        df = pd.read_csv(f)
        if not {'f1', 'num_samples'}.issubset(df.columns):
            raise ValueError(f"Missing required columns in {f}")
        dfs.append(df)

    df_all = pd.concat(dfs, keys=range(len(dfs)), names=["seed"]).reset_index(level=0)
    df_summary = (
        df_all.groupby("num_samples")
              .agg(f1_mean=("f1", "mean"), f1_std=("f1", "std"))
              .reset_index()
    )

    # Rename for plotting compatibility
    df_summary.rename(columns={"f1_mean": "f1"}, inplace=True)
    df_summary.to_csv(out_path, index=False)
    print(f"✅ Combined CSV saved to: {out_path}")
    print(f"   (Averaged over {len(csv_files)} seeds)")

if __name__ == "__main__":
    main()

"""
python combine_seeds.py \
    --indir /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs \
    --lang en
"""