#!/usr/bin/env python3
# aggregate_all_asr_matrices.py
"""
Aggregate ASR matrix CSVs (single, bilingual, dual),
compute per-group mean and variance for WER across seeds,
and combine all results into a single CSV.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# --- Configurable paths ---
SINGLE_CSV = Path(
    "/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/asr_matrix/seeds/asr_matrix_single.csv"
)
BILINGUAL_CSV = Path(
    "/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/asr_matrix/seeds/asr_matrix_bilingual.csv"
)
DUAL_CSV = Path(
    "/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/asr_matrix/seeds/asr_matrix_dual.csv"
)
OUTPUT_COMBINED_CSV = Path(
    "/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/asr_matrix/asr_matrix.csv"
)

group_cols = ["model_id", "type", "lang_src", "lang_tgt"]
metric = "wer"


def aggregate_csv(df: pd.DataFrame, transform_single_to_dual: bool = False):
    """
    Aggregate a DataFrame by model_id, type, lang_src, lang_tgt.
    Optionally transform type='single' -> 'dual' with model_id X -> X_X.
    Computes mean and variance of WER across seeds.
    """
    df = df.copy()
    df["lang_tgt"] = df["lang_tgt"].fillna("")

    # Convert WER to numeric
    df[metric] = pd.to_numeric(df[metric], errors="coerce")

    if transform_single_to_dual:
        mask_single = df["type"].astype(str).str.lower() == "single"
        df.loc[mask_single, "type"] = "dual"
        df.loc[mask_single, "lang_tgt"] = df.loc[mask_single, "lang_src"].astype(str)
        df.loc[mask_single, "model_id"] = (
            df.loc[mask_single, "model_id"]
            .astype(str)
            .apply(lambda x: f"{x}_{x}" if "_" not in x else x)
        )

    aggregated_rows = []
    for _, group in df.groupby(group_cols, sort=False, dropna=False):
        row = {
            "model_id": group.iloc[0]["model_id"],
            "type": group.iloc[0]["type"],
            "lang_src": group.iloc[0]["lang_src"],
            "lang_tgt": group.iloc[0]["lang_tgt"],
            "seed_count": len(group),
            "wer_mean": group[metric].mean(),
            "wer_var": group[metric].var(ddof=1) if len(group) > 1 else np.nan,
        }
        aggregated_rows.append(row)

    return pd.DataFrame(aggregated_rows)


# --- Read CSVs ---
df_single = pd.read_csv(SINGLE_CSV, dtype=str)
df_bilingual = pd.read_csv(BILINGUAL_CSV, dtype=str)
df_dual = pd.read_csv(DUAL_CSV, dtype=str)

# --- Aggregate each ---
agg_single = aggregate_csv(df_single)
agg_bilingual = aggregate_csv(df_bilingual)
agg_dual = aggregate_csv(df_dual, transform_single_to_dual=True)

# --- Combine all ---
combined_df = pd.concat(
    [agg_single, agg_bilingual, agg_dual], ignore_index=True
)
combined_df = (
    combined_df
    .sort_values(by=["model_id", "type", "lang_src", "lang_tgt"])
    .reset_index(drop=True)
)

# --- Save combined CSV ---
combined_df.to_csv(
    OUTPUT_COMBINED_CSV,
    index=False,
    float_format="%.12f"
)

print(
    f"Combined {len(combined_df)} aggregated rows into {OUTPUT_COMBINED_CSV}"
)
