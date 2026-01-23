#!/usr/bin/env python3
import pandas as pd
import numpy as np
import glob
import os
import argparse

def combine_lang(indir, lang, outdir=None):
    """Combina CSVs de un idioma específico en un único CSV promedio."""
    pattern = f"{lang}.wer_by_samples_*.csv"
    outdir = outdir or indir
    out_path = os.path.join(outdir, f"{lang}.wer_by_samples.csv")

    csv_files = sorted(glob.glob(os.path.join(indir, pattern)))
    if not csv_files:
        print(f"⚠️  No CSV files found for {lang} in {indir} matching {pattern}")
        return

    dfs = []
    for f in csv_files:
        df = pd.read_csv(f)
        if not {'wer', 'num_samples'}.issubset(df.columns):
            raise ValueError(f"Missing required columns in {f}")
        dfs.append(df)

    df_all = pd.concat(dfs, keys=range(len(dfs)), names=["seed"]).reset_index(level=0)
    df_summary = (
        df_all.groupby("num_samples")
              .agg(wer_mean=("wer", "mean"), wer_std=("wer", "std"))
              .reset_index()
    )

    df_summary.rename(columns={"wer_mean": "wer"}, inplace=True)
    df_summary.to_csv(out_path, index=False)
    print(f"✅ Combined CSV saved to: {out_path} (Averaged over {len(csv_files)} seeds)")

def main():
    parser = argparse.ArgumentParser(description="Combine multiple seed CSVs into averaged CSVs for multiple languages (WER)")
    parser.add_argument("--indir", required=True, help="Directory containing CSVs")
    parser.add_argument("--langs", required=True, nargs="+", help="Languages or prefixes (e.g., en es de)")
    parser.add_argument("--outdir", default=None, help="Directory to save the combined CSVs (default: same as indir)")
    args = parser.parse_args()

    for lang in args.langs:
        combine_lang(args.indir, lang, args.outdir)

if __name__ == "__main__":
    main()

"""
python combine_seeds.py \
    --indir /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/asr \
    --langs en es eu eo ca hu ja ka ru sw th
"""