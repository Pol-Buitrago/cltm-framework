#!/usr/bin/env python3
import pandas as pd
import numpy as np
import glob
import os
import argparse

def main():
    parser = argparse.ArgumentParser(description="Combine multiple seed CSVs (eer, auc, thr, num_samples) into a single averaged CSV")
    parser.add_argument("--indir", required=True, help="Directory containing CSVs")
    parser.add_argument("--lang", required=True, help="Language or prefix (e.g., 'en', 'es', 'de')")
    parser.add_argument("--pattern", default="{lang}.eer_by_samples_*.csv", help="Pattern to match input CSVs")
    parser.add_argument("--outdir", default=None, help="Directory to save the combined CSV (default: same as indir)")
    args = parser.parse_args()

    indir = args.indir
    lang = args.lang
    pattern = args.pattern.format(lang=lang)
    outdir = args.outdir or indir
    out_path = os.path.join(outdir, f"{lang}.eer_by_samples.csv")

    csv_files = sorted(glob.glob(os.path.join(indir, pattern)))
    if not csv_files:
        raise ValueError(f"No CSV files found in {indir} matching {pattern}")

    required_cols = {'eer', 'auc', 'thr', 'num_samples'}
    dfs = []
    for f in csv_files:
        df = pd.read_csv(f)
        if not required_cols.issubset(df.columns):
            raise ValueError(f"Missing required columns in {f}. Required columns: {sorted(required_cols)}")
        dfs.append(df)

    # Concatenate with a seed key (seed index will be used, not required to be present in files)
    df_all = pd.concat(dfs, keys=range(len(dfs)), names=["seed"]).reset_index(level=0)

    # Ensure num_samples is treated consistently (try to cast to int if possible)
    if not np.issubdtype(df_all["num_samples"].dtype, np.number):
        try:
            df_all["num_samples"] = df_all["num_samples"].astype(int)
        except Exception:
            # leave as is (groupby will still work for strings)
            pass

    # Aggregate: mean and std for eer, auc, thr
    df_summary = (
        df_all.groupby("num_samples")
              .agg(
                  eer_mean=("eer", "mean"),
                  eer_std=("eer", "std"),
                  auc_mean=("auc", "mean"),
                  auc_std=("auc", "std"),
                  thr_mean=("thr", "mean"),
                  thr_std=("thr", "std"),
              )
              .reset_index()
    )

    # Rename mean columns to original short names for compatibility, keep std columns explicit
    df_summary = df_summary.rename(columns={
        "eer_mean": "eer",
        "auc_mean": "auc",
        "thr_mean": "thr",
    })

    # Optional: number of seeds used per num_samples (useful to check missing rows)
    seeds_per_group = df_all.groupby("num_samples")["seed"].nunique().reset_index(name="n_seeds")
    df_summary = df_summary.merge(seeds_per_group, on="num_samples", how="left")

    # Save
    df_summary.to_csv(out_path, index=False)
    print(f"✅ Combined CSV saved to: {out_path}")
    print(f"   (Averaged over {len(csv_files)} seed files)")

if __name__ == "__main__":
    main()

"""
python combine_seeds_eer_auc.py --indir /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs --lang ca
python combine_seeds_eer_auc.py --indir /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs --lang en
python combine_seeds_eer_auc.py --indir /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs --lang eo
python combine_seeds_eer_auc.py --indir /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs --lang es
python combine_seeds_eer_auc.py --indir /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs --lang eu
python combine_seeds_eer_auc.py --indir /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs --lang hu
python combine_seeds_eer_auc.py --indir /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs --lang ja
python combine_seeds_eer_auc.py --indir /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs --lang ka
python combine_seeds_eer_auc.py --indir /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs --lang ru
python combine_seeds_eer_auc.py --indir /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs --lang sw
python combine_seeds_eer_auc.py --indir /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs --lang th
python combine_seeds_eer_auc.py --indir /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs --lang zh-CN
"""