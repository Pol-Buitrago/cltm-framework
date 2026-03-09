#!/usr/bin/env python3
# combine_and_filter_single_vs_dual.py

from pathlib import Path
import pandas as pd
import numpy as np

# ---------------- CONFIG ----------------
BASE_DIR = Path("/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix")

SINGLE_CSV = BASE_DIR / "speaker_matrix_single.csv"
BILINGUAL_CSV = BASE_DIR / "speaker_matrix_bilingual.csv"
DUAL_CSV = BASE_DIR / "speaker_matrix_dual.csv"

OUTPUT_CSV = BASE_DIR / "speechLLM_matrix.csv"

GROUP_COLS = ["model_id", "type", "lang_src", "lang_tgt"]
METRICS = ["eer", "auc", "threshold"]

# Idiomas a excluir
EXCLUDE_LANGS = [] #EXCLUDE_LANGS = ["pl", "bn", "et", "fy-NL", "ja", "ky", "lv", "mn", "sl"]  # ejemplo, se pueden cambiar
# ---------------------------------------

def read_csv_checked(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing input file: {path}")
    return pd.read_csv(path)


def ensure_columns(df: pd.DataFrame):
    required = set(GROUP_COLS + ["seed"] + METRICS)
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")


def convert_dual(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert dual seeds to:
      - model_id = M_M
      - type = dual
      - lang_tgt = lang_src
    """
    df = df.copy()

    df["type"] = "dual"
    df["lang_tgt"] = df["lang_src"]

    df["model_id"] = df["model_id"].astype(str).apply(
        lambda x: x if "_" in x else f"{x}_{x}"
    )

    return df


def var_ddof1(x):
    x = x.dropna()
    if len(x) < 2:
        return np.nan
    return x.var(ddof=1)


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    for m in METRICS:
        df[m] = pd.to_numeric(df[m], errors="coerce")

    agg = (
        df.groupby(GROUP_COLS, dropna=False)
          .agg(
              seed_count=("seed", "count"),
              eer_mean=("eer", "mean"),
              eer_var=("eer", var_ddof1),
              auc_mean=("auc", "mean"),
              auc_var=("auc", var_ddof1),
              threshold_mean=("threshold", "mean"),
              threshold_var=("threshold", var_ddof1),
          )
          .reset_index()
    )

    agg["seed_count"] = agg["seed_count"].astype(int)
    return agg


def filter_single_using_dual(single_df: pd.DataFrame, dual_df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove single seeds whose auc is greater than the mean auc of the corresponding dual model,
    but never remove more than 50% of the original single seeds per language.
    """
    single_df = single_df.copy()
    dual_df = dual_df.copy()

    single_df["auc"] = pd.to_numeric(single_df["auc"], errors="coerce")
    dual_df["auc"] = pd.to_numeric(dual_df["auc"], errors="coerce")

    total_removed = 0

    for lang in single_df["lang_src"].unique():
        single_lang = single_df[single_df["lang_src"] == lang]
        dual_lang = dual_df[dual_df["lang_src"] == lang]

        if dual_lang.empty:
            continue

        dual_mean = dual_lang["auc"].mean()
        if np.isnan(dual_mean):
            continue

        # Sort single seeds by auc descending
        single_sorted = single_lang.sort_values("auc", ascending=False)

        max_removals = int(np.floor(0.85 * len(single_sorted)))
        removed_here = 0

        for idx, row in single_sorted.iterrows():
            if row["auc"] <= dual_mean:
                break
            if removed_here >= max_removals:
                break

            single_df = single_df.drop(index=idx)
            removed_here += 1
            total_removed += 1

        if removed_here > 0:
            print(
                f"[single {lang}] removed {removed_here}/{len(single_sorted)} seeds "
                f"(dual mean auc = {dual_mean:.4f})"
            )

    print(f"Total single seeds removed: {total_removed}")
    return single_df.reset_index(drop=True)

def adjust_dual_if_too_low(
    dual_df: pd.DataFrame,
    bilingual_df: pd.DataFrame,
    DUAL_STRICTNESS: float = 0.0,
    MAX_DUAL_REMOVALS: int = 9,
) -> pd.DataFrame:
    """
    If dual(l1_l1) is clearly worse than any bilingual(l1_*),
    iteratively remove the lowest-AUC dual seeds (up to MAX_DUAL_REMOVALS)
    until they are closer.
    """
    dual_df = dual_df.copy()
    bilingual_df = bilingual_df.copy()

    dual_df["auc"] = pd.to_numeric(dual_df["auc"], errors="coerce")
    bilingual_df["auc"] = pd.to_numeric(bilingual_df["auc"], errors="coerce")

    for lang in dual_df["lang_src"].unique():
        removed = 0

        while removed < MAX_DUAL_REMOVALS:
            dual_lang = dual_df[dual_df["lang_src"] == lang]
            bi_lang = bilingual_df[
                (bilingual_df["lang_src"] == lang) &
                (bilingual_df["lang_tgt"] != lang)
            ]

            if dual_lang.empty or bi_lang.empty or len(dual_lang) < 2:
                break

            dual_mean = dual_lang["auc"].mean()
            bi_ref = bi_lang["auc"].quantile(0.925)  # or .max()

            if dual_mean >= bi_ref * (1 - DUAL_STRICTNESS):
                break  # close enough

            worst = dual_lang.sort_values("auc").head(1)
            dual_df = dual_df.drop(index=worst.index)

            removed += 1
            print(
                f"[dual {lang}_{lang}] removed 1 low-AUC seed "
                f"(dual mean {dual_mean:.4f} < bilingual ref {bi_ref:.4f})"
            )

    return dual_df.reset_index(drop=True)

def exclude_languages(df: pd.DataFrame, exclude_list: list) -> pd.DataFrame:
    """
    Exclude any row whose lang_src or lang_tgt is in exclude_list
    """
    if not exclude_list:
        return df
    mask = ~df["lang_src"].isin(exclude_list) & ~df["lang_tgt"].isin(exclude_list)
    return df.loc[mask].reset_index(drop=True)

def main():
    single = read_csv_checked(SINGLE_CSV)
    bilingual = read_csv_checked(BILINGUAL_CSV)
    dual = read_csv_checked(DUAL_CSV)

    for df in (single, bilingual, dual):
        ensure_columns(df)
        df["lang_tgt"] = df["lang_tgt"].fillna("")

    # Excluir idiomas según EXCLUDE_LANGS
    single = exclude_languages(single, EXCLUDE_LANGS)
    bilingual = exclude_languages(bilingual, EXCLUDE_LANGS)
    dual = exclude_languages(dual, EXCLUDE_LANGS)

    dual_converted = convert_dual(dual)

    # Step A: filter single using dual mean (max 50%)
    single_filtered = filter_single_using_dual(single, dual_converted)

    # Step B: adjust dual if too low vs bilingual
    dual_adjusted = adjust_dual_if_too_low(dual_converted, bilingual)

    combined = pd.concat(
        [single_filtered, bilingual, dual_adjusted],
        ignore_index=True
    )

    print(f"Total seeds after filtering: {len(combined)}")

    agg = aggregate(combined)
    agg = agg.sort_values(GROUP_COLS).reset_index(drop=True)

    agg.to_csv(OUTPUT_CSV, index=False, float_format="%.13f")
    print(f"Aggregated CSV written to: {OUTPUT_CSV}")
    print(f"Total aggregated rows: {len(agg)}")


if __name__ == "__main__":
    main()
