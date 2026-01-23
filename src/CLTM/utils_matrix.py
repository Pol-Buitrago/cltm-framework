#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Funciones para cargar datos y construir las matrices M y M_norm.
"""

import os
import pandas as pd
import numpy as np


def load_dataframe(path):
    """Carga el CSV en un DataFrame pandas.
    Mantiene el formato original y lanza excepcion si no existe.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV file not found: {path}")
    df = pd.read_csv(path)
    return df


def create_lookup_and_aggregate(df):
    """Crea la columna lookup_key y devuelve un DataFrame con el promedio de la métrica por clave.

    Detecta automáticamente la métrica a usar:
      - 'auc_mean' (higher is better)
      - 'f1_mean'  (higher is better)
      - 'wer_mean' (lower is better, converted to 1 - WER)
    lookup_key: 'lang' para 'single' y 'lang_src_lang_tgt' para 'dual'.
    """

    df = df.copy()
    if 'auc_mean' in df.columns:
        df['metric'] = df['auc_mean']
    elif 'f1_mean' in df.columns:
        df['metric'] = df['f1_mean']
    elif 'wer_mean' in df.columns:
        # Convert WER to accuracy-like score
        df['metric'] = 1.0 - df['wer_mean']
    else:
        raise ValueError(
            "No recognized metric column found. "
            "Expected 'auc_mean', 'f1_mean' or 'wer_mean'."
        )

    def create_key(row):
        if row['type'] == 'single':
            return row['lang_src']
        elif row['type'] == 'dual':
            return f"{row['lang_src']}_{row['lang_tgt']}"
        return np.nan

    df['lookup_key'] = df.apply(create_key, axis=1)

    df_agg = (
        df.groupby('lookup_key', dropna=True)['metric']
          .mean()
          .reset_index()
    )

    return df_agg



def compute_matrices(df_agg):
    """Construye M y M_norm a partir de df_agg.

    df_agg debe contener columnas ['lookup_key', 'metric'].
    """
    singles = sorted([k for k in df_agg['lookup_key'].values if isinstance(k, str) and '_' not in k])

    Perf_i = {
        lang: df_agg.loc[df_agg['lookup_key'] == lang, 'metric'].values[0] 
        if lang in df_agg['lookup_key'].values else np.nan 
        for lang in singles
    }

    M = pd.DataFrame(index=singles, columns=singles, dtype=float)
    M_norm = pd.DataFrame(index=singles, columns=singles, dtype=object)

    for i in singles:
        for j in singles:
            Perf_i_val = Perf_i.get(i)
            if pd.isna(Perf_i_val):
                M.loc[i, j] = np.nan
                continue
            key_i_j = f"{i}_{j}"
            if key_i_j in df_agg['lookup_key'].values:
                Perf_i_j_val = df_agg.loc[df_agg['lookup_key'] == key_i_j, 'metric'].values[0]
            else:
                Perf_i_j_val = np.nan
            if not pd.isna(Perf_i_j_val) and Perf_i_val != 0:
                M.loc[i, j] = (Perf_i_j_val - Perf_i_val) / Perf_i_val
            else:
                M.loc[i, j] = np.nan

    for i in singles:
        for j in singles:
            M_ij = M.loc[i, j]
            M_ii = M.loc[i, i]
            if pd.isna(M_ij):
                M_norm.loc[i, j] = 'TBD'
            elif pd.isna(M_ii) or M_ii == 0:
                M_norm.loc[i, j] = 'NA/0'
            else:
                M_norm.loc[i, j] = M_ij / M_ii

    return M, M_norm


def save_matrices(M, M_norm, out_dir):
    M.to_csv(os.path.join(out_dir, 'Transfer_Matrix_M.csv'))
    M_norm.to_csv(os.path.join(out_dir, 'Normalized_Transfer_Matrix_M_norm.csv'))
