#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import subprocess
import numpy as np
from utils_matrix import load_dataframe, create_lookup_and_aggregate, compute_matrices, save_matrices, summarize_and_print_CLTM_stats
from families import robust_load_abbrev_map, build_lang_meta, NAME2FAMILY
from plot_heatmap import plot_heatmap_by_family
from utils_mos import load_mos_map, build_quality_factors, apply_mos_adjustment



def run_pipeline(csv_path, output_dir_base, abbr_map_path, tag, selected_langs=None):
    """Run the full transfer-matrix pipeline for a given CSV and output directory."""
    output_dir = os.path.join(output_dir_base, tag)
    os.makedirs(output_dir, exist_ok=True)

    # --- Load and aggregate ---
    df = load_dataframe(csv_path)
    df_agg = create_lookup_and_aggregate(df)

    # --- Compute matrices ---
    M, M_norm, CLTM_se = compute_matrices(df_agg)

    # --- Filter to selected languages (si se pasó selected_langs) ---
    preserve_order = False
    if selected_langs:
        # Asegurarse de que solo usamos los que realmente existen en la matriz
        valid_langs = [l for l in selected_langs if l in M.index]
        if len(valid_langs) > 0:
            M = M.loc[valid_langs, valid_langs]
            M_norm = M_norm.loc[valid_langs, valid_langs]
            preserve_order = True  # indicar al plotting que respete este orden

    # --- Save matrices ---
    save_matrices(M, M_norm, output_dir)
    summarize_and_print_CLTM_stats(M_norm, CLTM_se, tag, output_dir)

    # --- Metadata ---
    abbr2name = robust_load_abbrev_map(abbr_map_path)
    df_lang_meta = build_lang_meta(M.index.tolist(), abbr2name, NAME2FAMILY)

    # --- Plot M ---
    out_fig_M = os.path.join(output_dir, "Transfer_Matrix_M_Heatmap_by_family.png")
    plot_heatmap_by_family(M, M, df_lang_meta, out_fig_M,
                           vmin=M.min().min(), vmax=M.max().max(), task=tag,
                           preserve_order=preserve_order)

    # --- Plot M_norm ---
    out_fig_Mnorm = os.path.join(output_dir, "Normalized_Transfer_Matrix_M_norm_Heatmap_by_family.png")
    plot_heatmap_by_family(M, M_norm, df_lang_meta, out_fig_Mnorm,
                           vmin=-1.5, vmax=1.5, task=tag,
                           preserve_order=preserve_order)

    return os.path.join(output_dir, "Normalized_Transfer_Matrix_M_norm.csv")

# --- Root paths ---
ROOT_DIR = "/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src"
CSV_BASE = os.path.join(ROOT_DIR, "outputs/matrices")
ABBR_MAP_PATH = os.path.join(ROOT_DIR, "CLTM/labels/language_abbreviation_map.txt")
OUTPUT_DIR_BASE = os.path.join(ROOT_DIR, "CLTM/transfer_matrix/transfer_matrices")

# --- Python environment for analysis scripts ---
PYTHON_ENV = os.path.join(ROOT_DIR, "CLTM/graph_env/bin/python")

tags = ["speechLLM", "ECAPA", "siamese", "speaker", "gender", "asr"]

# --- Opcional: lista de idiomas a forzar en ese orden ---
#SELECTED_LANGS = ["ar", "en", "de", "ckb", "kmr", "fr", "ca", "es", "be", "ru", "uk", "eu", "zh-CN", "zh-TW", "ky", "tr"]

if __name__ == "__main__":
    for tag in tags:
        csv_path = os.path.join(CSV_BASE, f"{tag}_matrix", f"{tag}_matrix.csv")
        # ahora pasamos SELECTED_LANGS (o None si no queremos filtrar)
        m_norm_csv = run_pipeline(csv_path, OUTPUT_DIR_BASE, ABBR_MAP_PATH, tag, #selected_langs=SELECTED_LANGS
        )

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
