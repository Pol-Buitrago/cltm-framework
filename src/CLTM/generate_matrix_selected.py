#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import subprocess
import numpy as np
from utils_matrix import load_dataframe, create_lookup_and_aggregate, compute_matrices, save_matrices
from families import robust_load_abbrev_map, build_lang_meta, NAME2FAMILY
from plot_heatmap import plot_heatmap_by_family
from utils_mos import load_mos_map, build_quality_factors, apply_mos_adjustment

DNS_SUMMARY_CSV = "/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__gender_id/04_balanced_60_cv_gender/tsv/dns_quality_summary.csv"
TARGET_LANGS = ["eo", "en", "de", "ca", "fr", "es", "be", "ug", "mhr"]

"""
def run_pipeline(csv_path, output_dir_base, abbr_map_path, tag, dns_csv_path=DNS_SUMMARY_CSV):
    output_dir = os.path.join(output_dir_base, tag)
    os.makedirs(output_dir, exist_ok=True)

    # --- Load and aggregate ---
    df = load_dataframe(csv_path)
    df_agg = create_lookup_and_aggregate(df)

    # --- Compute matrices ---
    M, M_norm = compute_matrices(df_agg)

    # --- Save matrices ---
    save_matrices(M, M_norm, output_dir)

    # --- Metadata ---
    abbr2name = robust_load_abbrev_map(abbr_map_path)
    df_lang_meta = build_lang_meta(M.index.tolist(), abbr2name, NAME2FAMILY)

    # --- Load MOS and build factors ---
    mos_map = load_mos_map(dns_csv_path)
    # Si necesitas normalizar el nombre de abreviaturas según tu abbr2name:
    # por ejemplo, si mos_map usa 'ab' y abbr2name espera 'ab', está OK.
    mos_factors = build_quality_factors(mos_map, aggressive=True, one_sided=True, gamma=9.0, clip=(0.6, 1.6))
    M_norm_adj = apply_mos_adjustment(M_norm, mos_factors, mode='geom', preserve_mean=False)

    # --- Save adjusted normalized matrix ---
    out_adj_csv = os.path.join(output_dir, f"Normalized_Transfer_Matrix_M_norm_adjusted_by_mos.csv")
    M_norm_adj.to_csv(out_adj_csv)

    # --- Plot M (original) ---
    out_fig_M = os.path.join(output_dir, "Transfer_Matrix_M_Heatmap_by_family.png")
    plot_heatmap_by_family(M, M, df_lang_meta, out_fig_M, vmin=M.min().min(), vmax=M.max().max())

    # --- Plot M_norm (original) ---
    out_fig_Mnorm = os.path.join(output_dir, "Normalized_Transfer_Matrix_M_norm_Heatmap_by_family.png")
    plot_heatmap_by_family(M, M_norm, df_lang_meta, out_fig_Mnorm, vmin=0, vmax=1.25)

    # --- Plot M_norm adjusted ---
    out_fig_Mnorm_adj = os.path.join(output_dir, "Normalized_Transfer_Matrix_M_norm_adjusted_by_mos_Heatmap_by_family.png")
    plot_heatmap_by_family(M, M_norm_adj, df_lang_meta, out_fig_Mnorm_adj, vmin=0, vmax=1.25)

    mean_before = np.nanmean(M_norm.values)
    mean_after  = np.nanmean(M_norm_adj.values)

    print(f"Mean before MOS normalization: {mean_before:.6f}")
    print(f"Mean after  MOS normalization: {mean_after:.6f}")

    return out_adj_csv
"""

def run_pipeline(csv_path, output_dir_base, abbr_map_path, tag):
    """Run the full transfer-matrix pipeline for a given CSV and output directory."""
    output_dir = os.path.join(output_dir_base, tag)
    os.makedirs(output_dir, exist_ok=True)

    # --- Load and aggregate ---
    df = load_dataframe(csv_path)
    df_agg = create_lookup_and_aggregate(df)

    # --- Compute matrices ---
    M, M_norm = compute_matrices(df_agg)

    # --- Filter to target languages ---
    target_langs = [l for l in TARGET_LANGS if l in M.index and l in M.columns]
    M = M.loc[target_langs, target_langs]
    M_norm = M_norm.loc[target_langs, target_langs]

    # --- Save matrices ---
    save_matrices(M, M_norm, output_dir)

    # --- Metadata ---
    abbr2name = robust_load_abbrev_map(abbr_map_path)
    df_lang_meta = build_lang_meta(M.index.tolist(), abbr2name, NAME2FAMILY)
    df_lang_meta = df_lang_meta.loc[target_langs]

    # --- Plot M ---
    out_fig_M = os.path.join(output_dir, "Transfer_Matrix_M_Heatmap_by_family.png")
    plot_heatmap_by_family(M, M, df_lang_meta, out_fig_M, vmin=M.min().min(), vmax=M.max().max())

    # --- Plot M_norm ---
    out_fig_Mnorm = os.path.join(output_dir, "Normalized_Transfer_Matrix_M_norm_Heatmap_by_family.png")
    plot_heatmap_by_family(M, M_norm, df_lang_meta, out_fig_Mnorm, vmin=0, vmax=1.25)

    return os.path.join(output_dir, "Normalized_Transfer_Matrix_M_norm.csv")

# --- Root paths ---
ROOT_DIR = "/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src"
CSV_BASE = os.path.join(ROOT_DIR, "outputs/matrices")
ABBR_MAP_PATH = os.path.join(ROOT_DIR, "CLTM/labels/language_abbreviation_map.txt")
OUTPUT_DIR_BASE = os.path.join(ROOT_DIR, "CLTM/transfer_matrix/transfer_matrices")

# --- Python environment for analysis scripts ---
PYTHON_ENV = os.path.join(ROOT_DIR, "CLTM/graph_env/bin/python")

tags = ["speaker", "gender", "asr"]

if __name__ == "__main__":
    for tag in tags:
        # --- CSV path ---
        csv_path = os.path.join(CSV_BASE, f"{tag}_matrix", f"{tag}_matrix.csv")
        
        # --- Run main pipeline ---
        m_norm_csv = run_pipeline(csv_path, OUTPUT_DIR_BASE, ABBR_MAP_PATH, tag)

        # --- Compute matrix properties ---
        matrix_props_dir = os.path.join(ROOT_DIR, "CLTM", "matrix_props", tag)
        os.makedirs(matrix_props_dir, exist_ok=True)
        subprocess.run([
            PYTHON_ENV,
            os.path.join(ROOT_DIR, "CLTM", "matrix_properties.py"),
            "--matrix-csv", m_norm_csv,
            "--abbrev-map", ABBR_MAP_PATH,
            "--output-dir", matrix_props_dir,
            "--top-k", "15"
        ], check=True)

    print("\nAll pipelines and matrix properties completed.")
