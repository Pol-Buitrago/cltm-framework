#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Funciones para cargar datos y construir las matrices M y M_norm.
Ahora además calcula y guarda los deltas self: Delta_{i<-i} = Perf(i+i) - Perf(i)
y emite un resumen (mean, std, count <= 0).

Parcheado: summarize_and_print_CLTM_stats ahora calcula mean/median SE
y versiones trimmed (eliminando el 25% de celdas con SE más alto).
"""

import os
import math
import datetime
import pandas as pd
import numpy as np
import scipy.stats as st

# -------------------------
# I/O helpers
# -------------------------
def load_dataframe(path):
    """Carga el CSV en un DataFrame pandas.
    Mantiene el formato original y lanza excepcion si no existe.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV file not found: {path}")
    df = pd.read_csv(path)
    return df


# -------------------------
# Aggregation
# -------------------------
def create_lookup_and_aggregate(df):
    """
    Crea la columna lookup_key y devuelve un DataFrame con:
      - metric (mean)
      - metric_var (mean of reported var between seeds / aggregated)
      - seed_count (first non-null or default 10)

    Mantiene la correspondencia entre 'auc_mean' <-> 'auc_var', etc.
    """
    df = df.copy()

    # Detect metric column and corresponding var column
    if 'auc_mean' in df.columns:
        metric_col = 'auc_mean'
        var_col = 'auc_var' if 'auc_var' in df.columns else None
    elif 'f1_mean' in df.columns:
        metric_col = 'f1_mean'
        var_col = 'f1_var' if 'f1_var' in df.columns else None
    elif 'wer_mean' in df.columns:
        metric_col = 'wer_mean'
        var_col = 'wer_var' if 'wer_var' in df.columns else None
    else:
        raise ValueError(
            "No recognized metric column found. "
            "Expected 'auc_mean', 'f1_mean' or 'wer_mean'."
        )

    # Build lookup_key
    def create_key(row):
        if row.get('type') == 'single':
            return row.get('lang_src')
        elif row.get('type') == 'dual':
            return f"{row.get('lang_src')}_{row.get('lang_tgt')}"
        return np.nan

    df['lookup_key'] = df.apply(create_key, axis=1)

    # Prepare columns: metric (possibly transformed), metric_var (if available), seed_count
    if metric_col == 'wer_mean':
        df['metric_raw'] = 1.0 - df[metric_col]  # transform WER -> accuracy-like
    else:
        df['metric_raw'] = df[metric_col]

    if var_col is not None:
        df['metric_var_raw'] = df[var_col]
    else:
        df['metric_var_raw'] = np.nan

    # seed_count: prefer column if exists, else default 10
    if 'seed_count' in df.columns:
        df['seed_count_raw'] = df['seed_count']
    else:
        df['seed_count_raw'] = 10

    # Aggregate: for metric -> mean; for var -> mean (approx), seed_count -> first non-null
    agg_dict = {
        'metric_raw': 'mean',
        'metric_var_raw': 'mean',
        'seed_count_raw': 'first'
    }

    df_agg = (
        df.groupby('lookup_key', dropna=True)
          .agg(agg_dict)
          .reset_index()
          .rename(columns={
              'metric_raw': 'metric',
              'metric_var_raw': 'metric_var',
              'seed_count_raw': 'seed_count'
          })
    )

    # Ensure numeric types where sensible
    df_agg['metric'] = pd.to_numeric(df_agg['metric'], errors='coerce')
    # metric_var may be all-NaN; keep as float with NaN allowed
    df_agg['metric_var'] = pd.to_numeric(df_agg['metric_var'], errors='coerce')
    # seed_count may be non-numeric if missing; coerce and fill default
    df_agg['seed_count'] = pd.to_numeric(df_agg['seed_count'], errors='coerce').fillna(10).astype(int)

    return df_agg


# -------------------------
# Compute matrices and SEs (delta-method)
# -------------------------
def compute_matrices(df_agg):
    """
    Construye M, CLTM y CLTM_se (matriz de SE aproximadas) usando el df_agg
    que contiene: lookup_key, metric, metric_var (var between seeds), seed_count.

    Además calcula Delta_{i<-i} = Perf(i+i) - Perf(i) para cada idioma,
    guarda un CSV 'delta_self_stats' en el directorio de trabajo y
    emite un resumen en stdout (mean, std, n_nonpositive).
    """
    # Map lookup_key -> mean metric, var(mean) and seed_count
    means = {}
    var_means = {}
    n_seeds_map = {}

    for _, row in df_agg.iterrows():
        key = row['lookup_key']
        # skip NaN keys if any
        if pd.isna(key):
            continue
        # metric may be NaN, keep as NaN
        means[key] = float(row['metric']) if not pd.isna(row['metric']) else np.nan
        # metric_var is var between seeds; var of mean = var_between / n_seeds
        if not pd.isna(row.get('metric_var', np.nan)):
            n_seeds = int(row.get('seed_count', 10)) if not pd.isna(row.get('seed_count', np.nan)) else 10
            var_means[key] = float(row['metric_var']) / max(1, n_seeds)
            n_seeds_map[key] = n_seeds
        else:
            var_means[key] = np.nan
            n_seeds_map[key] = None

    singles = sorted([k for k in df_agg['lookup_key'].values if isinstance(k, str) and '_' not in k])

    # Predefine DataFrames
    M = pd.DataFrame(index=singles, columns=singles, dtype=float)
    CLTM = pd.DataFrame(index=singles, columns=singles, dtype=float)
    CLTM_se = pd.DataFrame(index=singles, columns=singles, dtype=float)

    tau = 1e-9

    # 1) Fill M = (Perf(i+j) - Perf(i)) / Perf(i)
    for i in singles:
        mu_i = means.get(i, np.nan)
        for j in singles:
            if pd.isna(mu_i):
                M.loc[i, j] = np.nan
                continue
            key_ij = f"{i}_{j}"
            mu_ij = means.get(key_ij, np.nan)
            if pd.isna(mu_ij):
                M.loc[i, j] = np.nan
            else:
                if abs(mu_i) < tau:
                    M.loc[i, j] = np.nan
                else:
                    M.loc[i, j] = (mu_ij - mu_i) / mu_i

    # 2) Compute CLTM and CLTM_se via delta-method
    for i in singles:
        for j in singles:
            mu_i = means.get(i, np.nan)
            mu_ij = means.get(f"{i}_{j}", np.nan)
            mu_ii = means.get(f"{i}_{i}", np.nan)

            if any(pd.isna(x) for x in [mu_i, mu_ij, mu_ii]):
                CLTM.loc[i, j] = np.nan
                CLTM_se.loc[i, j] = np.nan
                continue

            num = mu_ij - mu_i
            den = mu_ii - mu_i
            if abs(den) < tau:
                CLTM.loc[i, j] = np.nan
                CLTM_se.loc[i, j] = np.nan
                continue

            r = num / den
            CLTM.loc[i, j] = r

            # Variances of sample means (of Perf means)
            var_a = var_means.get(f"{i}_{j}", np.nan)   # var of Perf(i+j) mean
            var_b = var_means.get(i, np.nan)            # var of Perf(i) mean
            var_c = var_means.get(f"{i}_{i}", np.nan)   # var of Perf(i+i) mean

            if any(pd.isna(x) for x in [var_a, var_b, var_c]):
                CLTM_se.loc[i, j] = np.nan
                continue

            # var(num) = var(a) + var(b)  (assuming independence a and b)
            var_num = var_a + var_b
            # var(den) = var(c) + var(b)
            var_den = var_c + var_b
            # cov(num, den) = cov(a-b, c-b) ≈ var_b  (approximation)
            cov_num_den = var_b

            # Delta-method for R = num / den:
            # Var(R) ≈ (1/den^2) * Var(num) + (num^2 / den^4) * Var(den) - 2 * num / den^3 * Cov(num,den)
            var_R = (var_num / (den**2)) + ((num**2) * var_den / (den**4)) - (2.0 * num * cov_num_den / (den**3))

            # Numerical guard
            if var_R < 0:
                var_R = 0.0

            CLTM_se.loc[i, j] = math.sqrt(var_R)

    # -------------------------
    # Compute self-deltas: Delta_{i<-i} = Perf(i+i) - Perf(i)
    # -------------------------
    deltas = {}
    for i in singles:
        mu_i = means.get(i, np.nan)
        mu_ii = means.get(f"{i}_{i}", np.nan)
        if pd.isna(mu_i) or pd.isna(mu_ii):
            deltas[i] = np.nan
        else:
            deltas[i] = mu_ii - mu_i

    # Convert to Series for easier handling
    deltas_series = pd.Series(deltas, name='delta_self')
    # Filter valid deltas
    valid_mask = ~deltas_series.isna()
    valid_deltas = deltas_series[valid_mask].astype(float)

    # Summary stats: mean, std (sample), count non-positive
    if valid_deltas.size > 0:
        mean_delta = float(valid_deltas.mean())
        std_delta = float(valid_deltas.std(ddof=1)) if valid_deltas.size > 1 else float(0.0)
        n_nonpos = int((valid_deltas <= 0.0).sum())
        n_valid = int(valid_deltas.size)
    else:
        mean_delta = float('nan')
        std_delta = float('nan')
        n_nonpos = 0
        n_valid = 0

    # Print summary to stdout (compact)
    print(f"[compute_matrices] Self-delta summary: n_valid={n_valid}, mean_delta={mean_delta:.6f}, std_delta={std_delta:.6f}, n_nonpositive={n_nonpos}")

    # Save per-language deltas CSV with timestamp to avoid accidental overwrite
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = f"delta_self_stats_{ts}.csv"
    try:
        deltas_series.to_csv(out_csv, header=True)
        print(f"[compute_matrices] Saved per-language self-deltas to: {out_csv}")
    except Exception as e:
        print(f"[compute_matrices] Warning: could not save deltas CSV: {e}")

    # Return matrices (backwards-compatible)
    return M, CLTM, CLTM_se


# -------------------------
# Summarize function (patched)
# -------------------------
def summarize_and_print_CLTM_stats(CLTM, CLTM_se, tag, out_dir, tau=1e-6):
    """
    # only off-diagonal entries
    n = CLTM.shape[0]
    mask = (~CLTM.isna()) & (~CLTM_se.isna())
    # ignore diagonal
    for k in CLTM.index:
        mask.loc[k,k] = False

    vals = CLTM.where(mask)
    ses = CLTM_se.where(mask)

    # flatten
    flat_vals = vals.values.flatten()
    flat_ses = ses.values.flatten()
    valid = ~np.isnan(flat_vals) & ~np.isnan(flat_ses)
    flat_vals = flat_vals[valid]
    flat_ses = flat_ses[valid]

    if flat_vals.size == 0:
        print(f"[{tag}] No valid off-diagonal CLTM entries to summarize.")
        return

    # Untrimmed metrics
    mean_se = np.mean(flat_ses)
    median_se = np.median(flat_ses)
    rel_mask = np.abs(flat_vals) > 1e-8
    avg_rel_se = np.mean(flat_ses[rel_mask] / np.abs(flat_vals[rel_mask])) if rel_mask.any() else np.nan

    # Sign probabilities (untrimmed)
    z = (0.0 - flat_vals) / flat_ses
    p_pos = 1.0 - st.norm.cdf(z)
    prop_high_conf_pos = np.mean(p_pos >= 0.95)
    prop_high_conf_neg = np.mean(p_pos <= 0.05)
    sign_consistency = np.mean((p_pos >= 0.95) | (p_pos <= 0.05))

    # approximate Frobenius STD: sqrt(sum var_R) / n
    sum_var = np.sum(flat_ses**2)
    sigma_F = math.sqrt(sum_var) / n

    # ----- Trim top 25% highest-SE entries and recompute -----
    pct = 85.0
    cutoff = np.percentile(flat_ses, pct)
    keep_mask = flat_ses <= cutoff
    trimmed_ses = flat_ses[keep_mask]
    trimmed_vals = flat_vals[keep_mask]

    if trimmed_ses.size > 0:
        trimmed_mean_se = float(np.mean(trimmed_ses))
        trimmed_median_se = float(np.median(trimmed_ses))
        # relative se trimmed
        rel_mask_trim = np.abs(trimmed_vals) > 1e-8
        trimmed_avg_rel_se = float(np.mean(trimmed_ses[rel_mask_trim] / np.abs(trimmed_vals[rel_mask_trim])) ) if rel_mask_trim.any() else np.nan

        # sign probabilities trimmed
        z_trim = (0.0 - trimmed_vals) / trimmed_ses
        p_pos_trim = 1.0 - st.norm.cdf(z_trim)
        prop_high_conf_pos_trim = float(np.mean(p_pos_trim >= 0.95))
        prop_high_conf_neg_trim = float(np.mean(p_pos_trim <= 0.05))
        sign_consistency_trim = float(np.mean((p_pos_trim >= 0.95) | (p_pos_trim <= 0.05)))

        # trimmed sigma_F
        sum_var_trim = np.sum(trimmed_ses**2)
        sigma_F_trim = float(math.sqrt(sum_var_trim) / n)
    else:
        trimmed_mean_se = float('nan')
        trimmed_median_se = float('nan')
        trimmed_avg_rel_se = float('nan')
        prop_high_conf_pos_trim = float('nan')
        prop_high_conf_neg_trim = float('nan')
        sign_consistency_trim = float('nan')
        sigma_F_trim = float('nan')

    prop_removed = 1.0 - (trimmed_ses.size / flat_ses.size)

    # print summary (compact + trimmed)
    print(f"[{tag}] n_langs={n}, n_entries={flat_vals.size}, mean_se={mean_se:.4f}, median_se={median_se:.4f}, "
          f"avg_rel_se={avg_rel_se:.3f}, prop_high_pos={prop_high_conf_pos:.3f}, prop_high_neg={prop_high_conf_neg:.3f}, "
          f"sign_consistency={sign_consistency:.3f}, sigma_F={sigma_F:.4f}")
    print(f"[{tag}] TRIMMED (remove top 25% SE): trimmed_mean_se={trimmed_mean_se:.4f}, trimmed_median_se={trimmed_median_se:.4f}, "
          f"trimmed_avg_rel_se={trimmed_avg_rel_se:.3f}, prop_high_pos_trim={prop_high_conf_pos_trim:.3f}, "
          f"prop_high_neg_trim={prop_high_conf_neg_trim:.3f}, sign_consistency_trim={sign_consistency_trim:.3f}, "
          f"sigma_F_trim={sigma_F_trim:.4f}, prop_removed={prop_removed:.3f}")

    # save a CSV with per-cell se if desired
    out_csv = os.path.join(out_dir, f"{tag}_cltm_cell_se.csv")
    try:
        CLTM_se.to_csv(out_csv, float_format="%.6f")
        print(f"[summarize] Saved per-cell SEs to: {out_csv}")
    except Exception as e:
        print(f"[summarize] Warning: could not save CLTM_se CSV: {e}")

    # Additionally save the trimmed index list if desired for inspection
    try:
        # build DataFrame of cells and their SEs/values
        rows = []
        idxs = vals.index.tolist()
        cols = vals.columns.tolist()
        for ii, i in enumerate(idxs):
            for jj, j in enumerate(cols):
                if i == j:
                    continue
                se_val = CLTM_se.loc[i, j]
                cltm_val = CLTM.loc[i, j]
                if pd.isna(se_val) or pd.isna(cltm_val):
                    continue
                rows.append((i, j, cltm_val, se_val))
        df_cells = pd.DataFrame(rows, columns=['lang_i','lang_j','cltm','se'])
        # mark trimmed
        df_cells['is_trimmed'] = df_cells['se'] > cutoff
        trimmed_csv = os.path.join(out_dir, f"{tag}_trimmed_cells_{pct}pct_cutoff.csv")
        df_cells.to_csv(trimmed_csv, index=False)
        print(f"[summarize] Saved trimmed cells table to: {trimmed_csv} (cutoff={cutoff:.6f})")
    except Exception as e:
        print(f"[summarize] Warning: could not save trimmed cells table: {e}")
    """


# -------------------------
# Save base matrices (unchanged)
# -------------------------
def save_matrices(M, M_norm, out_dir):
    try:
        M.to_csv(os.path.join(out_dir, 'Transfer_Matrix_M.csv'))
        M_norm.to_csv(os.path.join(out_dir, 'Normalized_Transfer_Matrix_M_norm.csv'))
        print(f"[save_matrices] Saved Transfer_Matrix_M.csv and Normalized_Transfer_Matrix_M_norm.csv to {out_dir}")
    except Exception as e:
        print(f"[save_matrices] Warning: could not save matrices to {out_dir}: {e}")