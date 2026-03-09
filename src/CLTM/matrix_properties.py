#!/usr/bin/env python3
"""
matrix_properties.py (patched)

Updated to:
 - treat off-diagonal entries as the canonical "donor->target" interactions
 - compute prop_positive/prop_negative/prop_gt1 excluding the diagonal
 - replace magnitude-weighted reciprocity with a simple count-based reciprocity
   (fraction of unordered off-diagonal pairs where both directions are positive)
 - exclude diagonal contributions from incoming/outgoing per-node aggregates
 - compute intra-family positive mass fraction excluding diagonal self-mass
 - list top inter-lingual pairs (exclude diagonal) for positives/negatives
 - expose an explicit `rfd1` variable (relative Frobenius deviation)

Minimal, logical changes: semantics align with paper wording (off-diagonal, unweighted reciprocity).
"""

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.linalg import norm, svd
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA


# Optional family mapping if not provided
NAME2FAMILY = {
    "Arabic": "Afro-Asiatic (Semitic)",
    "Kabyle": "Afro-Asiatic (Berber)",
    "Maltese": "Afro-Asiatic (Semitic)",
    "Indonesian": "Austronesian",
    "Esperanto": "Constructed",
    "Tamil": "Dravidian",
    "Armenian": "Indo-European",
    "Latgalian": "Indo-European (Baltic)",
    "Latvian": "Indo-European (Baltic)",
    "Irish": "Indo-European (Celtic)",
    "Welsh": "Indo-European (Celtic)",
    "Dutch": "Indo-European (Germanic)",
    "English": "Indo-European (Germanic)",
    "German": "Indo-European (Germanic)",
    "Swedish": "Indo-European (Germanic)",
    "Western Frisian": "Indo-European (Germanic)",
    "Bangla": "Indo-European (Indo-Aryan)",
    "Divehi": "Indo-European (Indo-Aryan)",
    "Urdu": "Indo-European (Indo-Aryan)",
    "Central Kurdish": "Indo-European (Iranian)",
    "Kurmanji Kurdish": "Indo-European (Iranian)",
    "Persian": "Indo-European (Iranian)",
    "Catalan": "Indo-European (Romance)",
    "French": "Indo-European (Romance)",
    "Galician": "Indo-European (Romance)",
    "Italian": "Indo-European (Romance)",
    "Portuguese": "Indo-European (Romance)",
    "Romanian": "Indo-European (Romance)",
    "Spanish": "Indo-European (Romance)",
    "Belarusian": "Indo-European (Slavic)",
    "Czech": "Indo-European (Slavic)",
    "Polish": "Indo-European (Slavic)",
    "Russian": "Indo-European (Slavic)",
    "Slovenian": "Indo-European (Slavic)",
    "Ukrainian": "Indo-European (Slavic)",
    "Japanese": "Japonic",
    "Georgian": "Kartvelian",
    "Thai": "Kra-Dai",
    "Basque": "Language isolate",
    "Mongolian": "Mongolic",
    "Ganda": "Niger-Congo",
    "Kinyarwanda": "Niger-Congo",
    "Swahili": "Niger-Congo",
    "Cantonese": "Sino-Tibetan",
    "Chinese (China)": "Sino-Tibetan",
    "Chinese (Hong Kong)": "Sino-Tibetan",
    "Chinese (Taiwan)": "Sino-Tibetan",
    "Hakha Chin": "Sino-Tibetan",
    "Kyrgyz": "Turkic",
    "Turkish": "Turkic",
    "Uyghur": "Turkic",
    "Uzbek": "Turkic",
    "Estonian": "Uralic (Finnic)",
    "Hungarian": "Uralic",
    "Meadow Mari": "Uralic",
    "Abkhazian": "Northwest Caucasian",
}


# ------------------------- utilities -------------------------

def ensure_dir(path):
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def load_json_or_dict_like(path):
    if path is None:
        return None
    with open(path, 'r', encoding='utf-8') as f:
        txt = f.read()
    try:
        return json.loads(txt)
    except Exception:
        # permissive fallback for python dict literal
        return eval(txt, {"__builtins__": {}}, {})


def gini(x):
    # Gini coefficient for 1d array (non-negative expected)
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return np.nan
    if np.all(x == 0):
        return 0.0
    x = np.sort(x)
    n = x.size
    cumx = np.cumsum(x)
    return (2.0 * np.sum((np.arange(1, n+1) * x))) / (n * cumx[-1]) - (n + 1) / n


# ------------------------- core diagnostics -------------------------

def compute_properties(M, labels=None, abbrev_map=None, name2family=None, outdir=None, top_k=10):
    """Compute a battery of diagnostics for normalized matrix M.

    Returns a dict of summary metrics and DataFrames for per-node metrics
    and pairwise similarity.
    """
    n = M.shape[0]
    ones = np.ones_like(M)

    # masks
    mask_offdiag = ~np.eye(n, dtype=bool)
    n_total = n * n
    n_offdiag = n * (n - 1)
    n_pairs_unordered = n * (n - 1) // 2

    # Basic sizes
    diag = np.diag(M)
    mean_diag = np.mean(diag)
    std_diag = np.std(diag)

    # Norms
    frob = norm(M, 'fro')
    frob_diff_ones = norm(M - ones, 'fro')
    norm_ones = norm(ones, 'fro')  # equals n
    rfd1 = frob_diff_ones / (norm_ones + 1e-12)  # Relative Frobenius Deviation (explicit)

    # Asymmetry measure (relative)
    asym_frob = norm(M - M.T, 'fro')
    asym_rel = asym_frob / (frob + 1e-12)

    # Positive/negative statistics (off-diagonal)
    n_pos_offdiag = np.sum((M > 0) & mask_offdiag)
    n_neg_offdiag = np.sum((M < 0) & mask_offdiag)
    prop_pos = n_pos_offdiag / (n_offdiag + 1e-12)
    prop_neg = n_neg_offdiag / (n_offdiag + 1e-12)
    prop_gt1 = np.sum((M > 1.0) & mask_offdiag) / (n_offdiag + 1e-12)

    # Incoming/outgoing positive strengths (per-language) -- exclude diagonal
    pos_M = np.maximum(M, 0.0)
    diag_pos = np.diag(pos_M).copy()
    incoming_pos = pos_M.sum(axis=1) - diag_pos  # how much each target benefits from other languages
    outgoing_pos = pos_M.sum(axis=0) - diag_pos  # how much each source donates to other languages

    # Reciprocity (count-based, unweighted): fraction of unordered off-diagonal pairs where both directions are positive
    reciprocal_count = 0
    for i in range(n):
        for j in range(i + 1, n):
            if (M[i, j] > 0) and (M[j, i] > 0):
                reciprocal_count += 1
    reciprocity_pos = reciprocal_count / (n_pairs_unordered + 1e-12)

    # Pairwise row similarity (cosine)
    # compute cosine similarity matrix between rows
    row_norms = np.linalg.norm(M, axis=1)
    row_norms[row_norms == 0] = 1e-12
    row_dot = M.dot(M.T)
    cos_sim = row_dot / (row_norms[:, None] * row_norms[None, :])
    # clamp numerical noise
    cos_sim = np.clip(cos_sim, -1.0, 1.0)
    avg_row_cos = (np.sum(cos_sim) - np.trace(cos_sim)) / (n * (n - 1) + 1e-12)

    # PCA on rows to estimate intrinsic dimensionality
    pca = PCA(n_components=min(n, M.shape[1]))
    pca.fit(M)
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    n_components_80 = int(np.searchsorted(cumvar, 0.80)) + 1

    # Spectral properties
    try:
        eigvals = np.linalg.eigvals(M)
        spectral_radius = max(abs(eigvals))
    except Exception:
        eigvals = np.array([])
        spectral_radius = np.nan
    # singular values and condition number
    try:
        s = svd(M, compute_uv=False)
        cond = s[0] / (s[-1] + 1e-12)
    except Exception:
        s = np.array([])
        cond = np.nan

    # row-wise concentration metrics (entropy on positive part, gini) -- off-diagonal
    row_entropy = np.zeros(n)
    row_gini = np.zeros(n)
    for i in range(n):
        pos = pos_M[i, :].copy()
        pos[i] = 0.0  # exclude diagonal self-mass
        ssum = pos.sum()
        if ssum <= 0:
            row_entropy[i] = np.nan
            row_gini[i] = np.nan
        else:
            p = pos / ssum
            row_entropy[i] = -np.nansum(p * np.log(p + 1e-12))
            row_gini[i] = gini(pos)

    # column-wise equivalents
    col_entropy = np.zeros(n)
    col_gini = np.zeros(n)
    for j in range(n):
        pos = pos_M[:, j].copy()
        pos[j] = 0.0
        ssum = pos.sum()
        if ssum <= 0:
            col_entropy[j] = np.nan
            col_gini[j] = np.nan
        else:
            p = pos / ssum
            col_entropy[j] = -np.nansum(p * np.log(p + 1e-12))
            col_gini[j] = gini(pos)

    # Top donors and receivers (based on off-diagonal sums)
    donors_idx = np.argsort(-outgoing_pos)[:top_k]
    receivers_idx = np.argsort(-incoming_pos)[:top_k]

    # Per-family aggregates (if name mapping / family mapping provided)
    family_stats = None
    if labels is not None:
        abbrevs = list(labels)
        full_names = None
        if abbrev_map is not None:
            full_names = [abbrev_map.get(a, a) for a in abbrevs]
        else:
            full_names = abbrevs
        if name2family is None:
            name2family = NAME2FAMILY
        families = [name2family.get(fn, 'Unknown') for fn in full_names]
        df_meta = pd.DataFrame({'abbrev': abbrevs, 'full_name': full_names, 'family': families,
                                'incoming_pos_offdiag': incoming_pos, 'outgoing_pos_offdiag': outgoing_pos,
                                'diag': diag})
        family_stats = df_meta.groupby('family').agg({
            'incoming_pos_offdiag': ['mean', 'sum', 'count'],
            'outgoing_pos_offdiag': ['mean', 'sum'],
            'diag': ['mean']
        })
    else:
        df_meta = None

    # Assortativity-like measure: fraction of positive mass that stays within same family (off-diagonal)
    intra_family_mass_frac = None
    if family_stats is not None:
        total_pos_mass_offdiag = pos_M[mask_offdiag].sum()
        intra = 0.0
        for fam in df_meta['family'].unique():
            members = df_meta[df_meta['family'] == fam]['abbrev'].values
            if len(members) <= 1:
                continue
            idx = [abbrevs.index(m) for m in members]
            sub = pos_M[np.ix_(idx, idx)].sum()
            # subtract diagonal self-mass for these members
            sub -= np.sum(np.diag(pos_M)[idx])
            intra += sub
        intra_family_mass_frac = intra / (total_pos_mass_offdiag + 1e-12)

    # Pairwise extreme pairs: strongest positive and strongest negative interactions (exclude diagonal)
    flat_idx_off = np.where(mask_offdiag.ravel())[0]
    off_values = M.ravel()[flat_idx_off]
    # strongest positive
    order_pos = np.argsort(-off_values)
    top_pairs_pos = []
    for k in range(min(top_k, len(order_pos))):
        flat = flat_idx_off[order_pos[k]]
        i, j = divmod(flat, n)
        top_pairs_pos.append((i, j, float(M[i, j])))
    # strongest negative
    order_neg = np.argsort(off_values)[:min(top_k, len(off_values))]
    top_pairs_neg = []
    for k in range(len(order_neg)):
        flat = flat_idx_off[order_neg[k]]
        i, j = divmod(flat, n)
        top_pairs_neg.append((i, j, float(M[i, j])))

    # Compose summary dict
    summary = {
        'n_languages': n,
        'mean_diag': float(mean_diag),
        'std_diag': float(std_diag),
        'frob_norm': float(frob),
        'frob_diff_to_ones': float(frob_diff_ones),
        'rfd1': float(rfd1),
        'asym_frob': float(asym_frob),
        'asym_rel': float(asym_rel),
        'prop_positive_offdiag': float(prop_pos),
        'prop_negative_offdiag': float(prop_neg),
        'prop_gt1_offdiag': float(prop_gt1),
        'reciprocity_pos_unweighted': float(reciprocity_pos),
        'avg_row_cosine': float(avg_row_cos),
        'n_components_80pct_var': int(n_components_80),
        'spectral_radius': float(spectral_radius),
        'svd_cond': float(cond),
        'intra_family_pos_mass_frac_offdiag': None if intra_family_mass_frac is None else float(intra_family_mass_frac),
    }

    # Per-node table
    per_node = pd.DataFrame({
        'abbrev': labels if labels is not None else [f'N{i}' for i in range(n)],
        'diag': diag,
        'incoming_pos_offdiag': incoming_pos,
        'outgoing_pos_offdiag': outgoing_pos,
        'row_entropy_pos_offdiag': row_entropy,
        'row_gini_pos_offdiag': row_gini,
        'col_entropy_pos_offdiag': col_entropy,
        'col_gini_pos_offdiag': col_gini,
    })
    if df_meta is not None:
        per_node = per_node.set_index('abbrev').join(
            df_meta.set_index('abbrev'),
            rsuffix='_meta'
        )

    # Pairwise similarity matrix (cosine) as DataFrame
    sim_df = pd.DataFrame(cos_sim, index=labels if labels is not None else None, columns=labels if labels is not None else None)

    # Top pairs formatting
    def idx_to_abbrev(pair_list, labels):
        out = []
        for i, j, val in pair_list:
            a = labels[i] if labels is not None else str(i)
            b = labels[j] if labels is not None else str(j)
            out.append((a, b, float(val)))
        return out

    top_pos_pairs = idx_to_abbrev(top_pairs_pos, labels)
    top_neg_pairs = idx_to_abbrev(top_pairs_neg, labels)

    results = {
        'summary': summary,
        'per_node': per_node,
        'pairwise_similarity': sim_df,
 'top_pos_pairs': top_pos_pairs,
        'top_neg_pairs': top_neg_pairs,
        'family_stats': family_stats,
    }

    # Save outputs if requested
    if outdir is not None:
        ensure_dir(outdir)
        per_node.to_csv(os.path.join(outdir, 'per_node_metrics.csv'))
        sim_df.to_csv(os.path.join(outdir, 'pairwise_cosine_similarity.csv'))
        # summary as JSON-like
        with open(os.path.join(outdir, 'summary_metrics.json'), 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        # top pairs
        pd.DataFrame(results['top_pos_pairs'], columns=['source', 'target', 'value']).to_csv(os.path.join(outdir, 'top_pos_pairs.csv'), index=False)
        pd.DataFrame(results['top_neg_pairs'], columns=['source', 'target', 'value']).to_csv(os.path.join(outdir, 'top_neg_pairs.csv'), index=False)
        if family_stats is not None:
            family_stats.to_csv(os.path.join(outdir, 'family_stats.csv'))

    return results


# ------------------------- printing helpers -------------------------

def pretty_print_summary(summary):
    print('\n=== MATRIX SUMMARY ===')
    for k, v in summary.items():
        print(f'{k:30s}: {v}')


def pretty_print_top_pairs(top_pairs, title):
    print(f'\n--- {title} ---')
    for a, b, val in top_pairs:
        print(f'{a:8s} -> {b:8s} : {val:.4f}')


# ------------------------- CLI -------------------------

def main():
    parser = argparse.ArgumentParser(description='Compute diagnostics on normalized transfer matrix (M_norm)')
    parser.add_argument('--matrix-csv', type=str, required=True, help='Path to normalized matrix CSV (index column present)')
    parser.add_argument('--abbrev-map', type=str, default=None, help='Optional abbrev->full name map (JSON or python dict literal)')
    parser.add_argument('--family-map', type=str, default=None, help='Optional mapping full_name->family (JSON)')
    parser.add_argument('--output-dir', type=str, default='./matrix_props', help='Directory to save CSV/JSON outputs')
    parser.add_argument('--top-k', type=int, default=10, help='Top-k pairs to report')
    args = parser.parse_args()

    df = pd.read_csv(args.matrix_csv, index_col=0)
    labels = list(df.index)
    M = df.values.astype(float)

    abbrev_map = None
    if args.abbrev_map is not None:
        abbrev_map = load_json_or_dict_like(args.abbrev_map)

    name2family = None
    if args.family_map is not None:
        name2family = load_json_or_dict_like(args.family_map)

    results = compute_properties(M, labels=labels, abbrev_map=abbrev_map, name2family=name2family, outdir=args.output_dir, top_k=args.top_k)

    pretty_print_summary(results['summary'])
    print('\nPer-node top 10 receivers by incoming_pos_offdiag:')
    rcv = results['per_node'].sort_values('incoming_pos_offdiag', ascending=False).head(10)
    print(rcv[['full_name' if 'full_name' in rcv.columns else 'diag', 'incoming_pos_offdiag']])

    print('\nPer-node top 10 donors by outgoing_pos_offdiag:')
    dnr = results['per_node'].sort_values('outgoing_pos_offdiag', ascending=False).head(10)
    print(dnr[['full_name' if 'full_name' in dnr.columns else 'diag', 'outgoing_pos_offdiag']])

    pretty_print_top_pairs(results['top_pos_pairs'], 'Top positive pairs (value desc)')
    pretty_print_top_pairs(results['top_neg_pairs'], 'Top negative pairs (value asc)')

    if results['family_stats'] is not None:
        print('\nFamily-level aggregates saved to output directory and a sample:')
        print(results['family_stats'].head(10))

    print('\nSaved outputs to', args.output_dir)


if __name__ == '__main__':
    main()

"""
python3 matrix_properties.py \
  --matrix-csv /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speaker/verification/generate_transfer_matrix/transfer_matrices/Normalized_Transfer_Matrix_M_norm.csv \
  --abbrev-map /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/scripts/labels/language_abbreviation_map.txt \
  --output-dir ./matrix_props \
  --top-k 15
"""
