#!/usr/bin/env python3
# aggregate_all_speaker_matrices_robust.py
"""
Aggregate speaker matrix CSVs (single, bilingual, dual),
compute per-group robust mean and variance for eer, auc, threshold,
removing outliers using Z-score method (default threshold 2.0),
and combine all results into a single CSV.
"""

import pandas as pd
from pathlib import Path
import numpy as np

# --- Configurable paths ---
SINGLE_CSV = Path("/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speaker_matrix/seeds/speaker_matrix_single.csv")
BILINGUAL_CSV = Path("/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speaker_matrix/seeds/speaker_matrix_bilingual.csv")
DUAL_CSV = Path("/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speaker_matrix/seeds/speaker_matrix_dual.csv")
OUTPUT_COMBINED_CSV = Path("/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speaker_matrix/speaker_matrix.csv")

group_cols = ["model_id", "type", "lang_src", "lang_tgt"]
metrics = ["eer", "auc", "threshold"]
Z_THRESHOLD = 2.0  # Seeds with Z-score > this are considered outliers

def remove_outliers_zscore(group: pd.DataFrame, metrics: list, z_thresh: float = 2.0):
    """
    Remove rows in group where any metric exceeds z_thresh in absolute value.
    Returns filtered group and number of seeds removed.
    """
    filtered = group.copy()
    removed_count = 0
    for m in metrics:
        if len(filtered) < 2:
            continue  # Cannot compute z-score
        mean = filtered[m].mean()
        std = filtered[m].std(ddof=1)
        if std == 0:
            continue
        z_scores = (filtered[m] - mean) / std
        mask = z_scores.abs() <= z_thresh
        removed_count += (~mask).sum()
        filtered = filtered[mask]
    return filtered, removed_count

def aggregate_csv_robust(df: pd.DataFrame, transform_single_to_dual=False):
    """
    Aggregate a DataFrame by model_id, type, lang_src, lang_tgt.
    Optionally transform type='single' -> 'dual' with model_id X -> X_X.
    Applies Z-score outlier removal per group.
    """
    df = df.copy()
    df["lang_tgt"] = df["lang_tgt"].fillna("")
    
    # Convert metrics to numeric
    for m in metrics:
        df[m] = pd.to_numeric(df[m], errors="coerce")
    
    if transform_single_to_dual:
        mask_single = df["type"].astype(str).str.lower() == "single"
        df.loc[mask_single, "type"] = "dual"
        df.loc[mask_single, "lang_tgt"] = df.loc[mask_single, "lang_src"].astype(str)
        df.loc[mask_single, "model_id"] = df.loc[mask_single, "model_id"].astype(str).apply(
            lambda x: f"{x}_{x}" if "_" not in x else x
        )
    
    aggregated_rows = []
    for _, group in df.groupby(group_cols, sort=False, dropna=False):
        filtered_group, removed_count = remove_outliers_zscore(group, metrics, Z_THRESHOLD)
        if removed_count > 0:
            print(f"Removed {removed_count} seed(s) for model {group.iloc[0]['model_id']} ({group.iloc[0]['type']}) lang_src={group.iloc[0]['lang_src']}, lang_tgt={group.iloc[0]['lang_tgt']}")
        row = {
            "model_id": group.iloc[0]["model_id"],
            "type": group.iloc[0]["type"],
            "lang_src": group.iloc[0]["lang_src"],
            "lang_tgt": group.iloc[0]["lang_tgt"],
            "seed_count": len(filtered_group),
            "eer_mean": filtered_group["eer"].mean(),
            "eer_var": filtered_group["eer"].var(ddof=1) if len(filtered_group) > 1 else np.nan,
            "auc_mean": filtered_group["auc"].mean(),
            "auc_var": filtered_group["auc"].var(ddof=1) if len(filtered_group) > 1 else np.nan,
            "threshold_mean": filtered_group["threshold"].mean(),
            "threshold_var": filtered_group["threshold"].var(ddof=1) if len(filtered_group) > 1 else np.nan,
        }
        aggregated_rows.append(row)
    
    return pd.DataFrame(aggregated_rows)

# --- Read CSVs ---
df_single = pd.read_csv(SINGLE_CSV, dtype=str)
df_bilingual = pd.read_csv(BILINGUAL_CSV, dtype=str)
df_dual = pd.read_csv(DUAL_CSV, dtype=str)

# --- Aggregate each ---
agg_single = aggregate_csv_robust(df_single)
agg_bilingual = aggregate_csv_robust(df_bilingual)
agg_dual = aggregate_csv_robust(df_dual, transform_single_to_dual=True)

# --- Combine all ---
combined_df = pd.concat([agg_single, agg_bilingual, agg_dual], ignore_index=True)
combined_df = combined_df.sort_values(by=["model_id", "type", "lang_src", "lang_tgt"]).reset_index(drop=True)

# --- Save combined CSV ---
combined_df.to_csv(OUTPUT_COMBINED_CSV, index=False, float_format="%.12f")

print(f"Combined {len(combined_df)} aggregated rows into {OUTPUT_COMBINED_CSV}")
