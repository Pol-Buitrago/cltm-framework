#!/usr/bin/env python3
"""
enhanced_transfer_map_v5_nogrid.py

Mapa 2D avanzado desde la matriz de transferencia normalizada sin
búsqueda exhaustiva: ejecuta DBSCAN una sola vez con los parámetros
proporcionados por el usuario.

Uso:
python3 enhanced_transfer_map_v5_nogrid.py --matrix-csv /ruta/transfer_matrix.csv --outdir ./maps \
    --eps 0.5 --min-samples 3 --sim-th 0.75 --mut-th 0.25
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
from matplotlib.patches import Ellipse, Patch
from sklearn.manifold import MDS
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.utils.extmath import randomized_svd
from sklearn.cluster import DBSCAN
from scipy.stats import spearmanr
import math
from time import time

# -----------------------------
# Helpers (I/O, color, etc.)
# -----------------------------
def ensure_dir(path):
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_abbrev_map(path):
    if path is None:
        return None
    with open(path, 'r', encoding='utf-8') as f:
        txt = f.read()
    try:
        return json.loads(txt)
    except Exception:
        return eval(txt, {'__builtins__': {}}, {})

def map_abbrevs_to_names(abbrev_list, abbrev_map):
    if abbrev_map is None:
        return abbrev_list
    return [abbrev_map.get(ab, abbrev_map.get(ab.lower(), ab)) for ab in abbrev_list]

def assign_families(full_names, name2family):
    return [name2family.get(nm, 'Unknown') if nm is not None else 'Unknown' for nm in full_names]

def assign_family_colors(families):
    unique = sorted(set(families), key=lambda x: (x != 'Unknown', x))
    cmaps = [plt.get_cmap(c) for c in ['tab20', 'tab20b', 'tab20c']]
    colors = []
    for cmap in cmaps:
        colors.extend([to_hex(cmap(i)) for i in range(cmap.N)])
    colors = colors[:len(unique)]
    fam2color = {f: colors[i] for i, f in enumerate(unique)}
    cols = [fam2color[f] for f in families]
    return fam2color, cols

# -----------------------------
# Covariance ellipse drawer
# -----------------------------
def draw_cov_ellipse(ax, points, color, n_std=2.0, alpha=0.18, edgecolor=None, linewidth=0.8):
    if points.shape[0] == 1:
        x, y = points[0]
        circ = plt.Circle((x, y), radius=0.08 * max(1.0, n_std), facecolor=color, alpha=alpha, edgecolor=edgecolor)
        ax.add_patch(circ)
        return
    mean = points.mean(axis=0)
    cov = np.cov(points, rowvar=False)
    cov = cov + 1e-8 * np.eye(2)
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    width, height = 2.0 * n_std * np.sqrt(np.maximum(vals, 0.0))
    angle = np.degrees(np.arctan2(vecs[1,0], vecs[0,0]))
    ell = Ellipse(xy=mean, width=width, height=height, angle=angle,
                  facecolor=color, alpha=alpha, edgecolor=edgecolor if edgecolor else color, linewidth=linewidth)
    ax.add_patch(ell)

# -----------------------------
# Prepare bidirectional vectors
# -----------------------------
def normalize_rows(X, eps=1e-10):
    s = X.sum(axis=1, keepdims=True)
    s[s == 0] = 1.0
    return X / (s + eps)

def normalize_cols(X, eps=1e-10):
    s = X.sum(axis=0, keepdims=True)
    s[s == 0] = 1.0
    return X / (s + eps)

def prepare_bidirectional_vectors(M, include_neg=True):
    M_pos = np.maximum(M, 0.0)
    M_neg = np.maximum(-M, 0.0) if include_neg else np.zeros_like(M)
    Fpos = normalize_rows(M_pos)
    Fneg = normalize_rows(M_neg)
    Cpos = normalize_cols(M_pos)
    Cneg = normalize_cols(M_neg)
    F = np.hstack([Fpos, Fneg])
    C = np.hstack([Cpos.T, Cneg.T])
    V = np.hstack([F, C])
    return V, Fpos, Fneg, Cpos, Cneg

# -----------------------------
# SVD optional (asymmetric)
# -----------------------------
def svd_embeddings(M, n_components=10, random_state=42):
    U, S, Vt = randomized_svd(M, n_components=n_components, random_state=random_state)
    emb_emisor = U * np.sqrt(S[np.newaxis, :])
    emb_receptor = Vt.T * np.sqrt(S[np.newaxis, :])
    return emb_emisor, emb_receptor

# -----------------------------
# Mutuality
# -----------------------------
def compute_mutuality(M, method='sym_pos'):
    Mpos = np.maximum(M, 0.0)
    if method == 'sym_pos':
        mutual_raw = 0.5 * (Mpos + Mpos.T)
        maxv = mutual_raw.max() if mutual_raw.max() > 0 else 1.0
        mutual = mutual_raw / maxv
    elif method == 'harmonic_pos':
        eps = 1e-10
        a = Mpos
        b = Mpos.T
        mutual = 2.0 * np.minimum(a, b) / (a + b + eps)
        mutual[np.isnan(mutual)] = 0.0
    else:
        mutual_raw = 0.5 * (Mpos + Mpos.T)
        maxv = mutual_raw.max() if mutual_raw.max() > 0 else 1.0
        mutual = mutual_raw / maxv
    return mutual

# -----------------------------
# Composite similarity and distance
# -----------------------------
def composite_similarity(V, mutual, alpha=0.7, beta=0.3):
    cos = cosine_similarity(V)
    cos01 = (cos + 1.0) / 2.0
    S = alpha * cos01 + beta * mutual
    Smin, Smax = S.min(), S.max()
    if Smax > Smin:
        S = (S - Smin) / (Smax - Smin)
    return S

def similarity_to_distance(S):
    D = 1.0 - S
    D[D < 0] = 0.0
    return D

# -----------------------------
# Projection MDS
# -----------------------------
def project_mds_from_distance(D, random_state=42):
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=random_state)
    Y = mds.fit_transform(D)
    stress = mds.stress_
    return Y, stress

# -----------------------------
# Cluster validation helpers
# -----------------------------
def mean_pairwise_in_matrix(indices, M):
    if len(indices) < 2:
        return 0.0
    sub = M[np.ix_(indices, indices)]
    n = sub.shape[0]
    iu = np.triu_indices(n, k=1)
    vals = sub[iu]
    if vals.size == 0:
        return 0.0
    return float(np.nanmean(vals))

def validate_clusters(cluster_labels, mutual, composite_S, sim_threshold, mutual_threshold, min_cluster_size):
    """
    Devuelve: list of valid clusters (each como lista de índices), list of all clusters (listas)
    """
    labels = np.array(cluster_labels)
    unique = [c for c in np.unique(labels) if c != -1]
    all_clusters = []
    valid_clusters = []
    for cl in unique:
        idxs = np.where(labels == cl)[0].tolist()
        all_clusters.append(idxs)
        if len(idxs) < min_cluster_size:
            continue
        mean_mut = mean_pairwise_in_matrix(idxs, mutual)
        mean_sim = mean_pairwise_in_matrix(idxs, composite_S)
        if (mean_mut >= mutual_threshold) or (mean_sim >= sim_threshold):
            valid_clusters.append(idxs)
    return valid_clusters, all_clusters

# -----------------------------
# Scoring function
# -----------------------------
def score_configuration(valid_clusters, all_clusters, mutual, composite_S):
    n_valid = len(valid_clusters)
    n_all = len(all_clusters)
    if n_valid == 0:
        return 0.0
    cohesions = []
    for idxs in valid_clusters:
        mean_mut = mean_pairwise_in_matrix(idxs, mutual)
        mean_sim = mean_pairwise_in_matrix(idxs, composite_S)
        cohesions.append(0.5 * (mean_mut + mean_sim))
    mean_cohesion = float(np.nanmean(cohesions)) if len(cohesions) > 0 else 0.0
    score = (n_valid / (n_all + 1.0)) * mean_cohesion
    return float(score)

# -----------------------------
# Plot final: clusters elípticos y puntos
# -----------------------------
def plot_final(Y, labels_nodes, families, fam2color, cluster_labels, valid_clusters,
               mutual, composite_S, outpath, ellipse_n_std=2.0, ellipse_alpha=0.18, topk_arrows=0, legend_ncols=4):
    """
    Dibuja el mapa y coloca la leyenda *debajo* con `legend_ncols` columnas.
    """
    # colores por familia para puntos
    _, cols = assign_family_colors(families)

    fig, ax = plt.subplots(figsize=(12,10))
    ax.scatter(Y[:,0], Y[:,1], c=cols, s=100, alpha=0.95, edgecolor='k', linewidth=0.3)
    for i, lbl in enumerate(labels_nodes):
        ax.annotate(lbl, (Y[i,0], Y[i,1]), fontsize=8, alpha=0.95)

    ax.set_title('Composite MDS Map')

    # draw valid clusters as ellipses
    for comp in valid_clusters:
        pts = Y[comp]
        draw_cov_ellipse(ax, pts, color='#444444', n_std=ellipse_n_std, alpha=ellipse_alpha, edgecolor='#222222', linewidth=1.0)

    # topk arrows optional (mutual strongest links)
    if mutual is not None and topk_arrows > 0:
        n = mutual.shape[0]
        iu = np.triu_indices(n, k=1)
        vals = mutual[iu]
        top_idxs = np.argsort(vals)[-topk_arrows:][::-1]
        for idx in top_idxs:
            i = iu[0][idx]
            j = iu[1][idx]
            val = mutual[i,j]
            x1,y1 = Y[i,0], Y[i,1]
            x2,y2 = Y[j,0], Y[j,1]
            linewidth = 1.0 + 4.0 * float(val)
            ax.annotate('', xy=(x2,y2), xytext=(x1,y1),
                        arrowprops=dict(arrowstyle='->', lw=linewidth, alpha=0.6*val, color='gray'))

    # Preparar handles para la leyenda (Patch mantiene cuadrados de color)
    handles = [Patch(facecolor=c, edgecolor='none', label=f) for f, c in fam2color.items()]

    # Ajustar el espacio inferior para la leyenda y colocarla centrada debajo (n columnas)
    fig.subplots_adjust(bottom=0.16)  # espacio para la leyenda
    ncols = max(1, int(legend_ncols))
    fig.legend(handles=handles, loc='lower center', ncol=ncols, fontsize=8, frameon=False, bbox_to_anchor=(0.5, 0.02))

    # Guardar la figura, tight bbox para incluir la leyenda externa
    fig.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close(fig)

# -----------------------------
# Single-run DBSCAN routine
# -----------------------------
def run_dbscan_once(Y, mutual, composite_S, eps, min_samples, sim_threshold, mutual_threshold, min_cluster_size):
    cl = DBSCAN(eps=float(eps), min_samples=int(min_samples), metric='euclidean')
    labels = cl.fit_predict(Y)
    valid_clusters, all_clusters = validate_clusters(labels, mutual, composite_S, sim_threshold, mutual_threshold, min_cluster_size)
    score = score_configuration(valid_clusters, all_clusters, mutual, composite_S)
    return {
        'cluster_labels': labels,
        'valid_clusters': valid_clusters,
        'all_clusters': all_clusters,
        'score': score,
        'config': {
            'method': 'dbscan',
            'eps': float(eps),
            'min_samples': int(min_samples),
            'sim_threshold': float(sim_threshold),
            'mutual_threshold': float(mutual_threshold),
            'min_cluster_size': int(min_cluster_size)
        }
    }

# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description='Mapa 2D sin grid search (v5) — ejecuta DBSCAN una vez')
    parser.add_argument('--matrix-csv', required=True)
    parser.add_argument('--abbrev-map',
                        default='/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/scripts/labels/language_abbreviation_map.txt')
    parser.add_argument('--languages-json',
                        default='/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/CLTM/languages.json')
    parser.add_argument('--outdir', default='./maps')
    parser.add_argument('--alpha', type=float, default=0.4, help='peso similitud coseno (0..1)')
    parser.add_argument('--beta', type=float, default=0.6, help='peso mutualidad (0..1)')
    parser.add_argument('--svd-components', type=int, default=0, help='>0 para incluir embedding SVD en vectores')
    parser.add_argument('--mutual-method', type=str, default='sym_pos', choices=['sym_pos','harmonic_pos'])
    # single-run DBSCAN params (no grid)
    parser.add_argument('--eps', type=float, default=0.5, help='eps para DBSCAN (float)')
    parser.add_argument('--min-samples', type=int, default=3, help='min_samples para DBSCAN (int)')
    parser.add_argument('--sim-th', type=float, default=0.75, help='sim_threshold para validar clusters')
    parser.add_argument('--mut-th', type=float, default=0.25, help='mutual_threshold para validar clusters')
    parser.add_argument('--min-cluster-size', type=int, default=2)
    parser.add_argument('--ellipse-n-std', type=float, default=2.0)
    parser.add_argument('--ellipse-alpha', type=float, default=0.18)
    parser.add_argument('--topk-arrows', type=int, default=0)
    parser.add_argument('--legend-ncols', type=int, default=4, help='Número de columnas en la leyenda (ordenada en filas de 4 por defecto)')
    args = parser.parse_args()

    ensure_dir(args.outdir)
    df = pd.read_csv(args.matrix_csv, index_col=0)
    labels_abbr = list(df.index)
    M = df.values.astype(float)
    abbrev_map = load_abbrev_map(args.abbrev_map)
    labels_full = map_abbrevs_to_names(labels_abbr, abbrev_map)
    NAME2FAMILY = load_json(args.languages_json)
    families = assign_families(labels_full, NAME2FAMILY)
    fam2color, _ = assign_family_colors(families)

    # Prepare vectors
    V_base, Fpos, Fneg, Cpos, Cneg = prepare_bidirectional_vectors(M, include_neg=True)
    if args.svd_components > 0:
        emb_e, emb_r = svd_embeddings(M, n_components=args.svd_components)
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()
        emb_e_s = sc.fit_transform(emb_e)
        emb_r_s = sc.fit_transform(emb_r)
        V = np.hstack([V_base, emb_e_s, emb_r_s])
    else:
        V = V_base

    # Compute mutual and composite similarity
    mutual = compute_mutuality(M, method=args.mutual_method)
    alpha = float(args.alpha)
    beta = float(args.beta)
    if alpha + beta <= 0:
        raise ValueError("alpha + beta must ser > 0")
    s = alpha + beta
    alpha /= s
    beta /= s
    composite_S = composite_similarity(V, mutual, alpha=alpha, beta=beta)
    D = similarity_to_distance(composite_S)

    # Project with MDS
    Y, stress = project_mds_from_distance(D)
    print(f"MDS stress: {stress:.6f}")

    # Spearman with base
    cos_base = cosine_similarity(V_base)
    cos01 = (cos_base + 1.0) / 2.0
    D0 = 1.0 - cos01
    a = D0[np.triu_indices(D0.shape[0], k=1)]
    b = D[np.triu_indices(D.shape[0], k=1)]
    spearman_corr, spearman_p = spearmanr(a, b)
    print(f"Spearman corr base_vs_composite: {spearman_corr:.4f} (p={spearman_p:.2g})")

    # Run single DBSCAN
    t0 = time()
    result = run_dbscan_once(Y, mutual, composite_S, args.eps, args.min_samples, args.sim_th, args.mut_th, args.min_cluster_size)
    elapsed = time() - t0

    labels = result['cluster_labels']
    valid_clusters = result['valid_clusters']
    all_clusters = result['all_clusters']
    score = result['score']
    config = result['config']

    print("DBSCAN config used:", config)
    print(f"Found {len(all_clusters)} clusters (excluding noise); {len(valid_clusters)} validated clusters; score={score:.6f}; clustering time={elapsed:.3f}s")

    # If no valid clusters, save a simple point map and metrics
    final_out = os.path.join(args.outdir, 'composite_mds_map_v5_nogrid.png')
    if len(valid_clusters) == 0:
        print("No clusters passed validation con los umbrales proporcionados. Se guarda mapa sin elipses.")
        _, cols = assign_family_colors(families)
        fig, ax = plt.subplots(figsize=(12,10))
        ax.scatter(Y[:,0], Y[:,1], c=cols, s=100, alpha=0.95, edgecolor='k', linewidth=0.3)
        for i, lbl in enumerate(labels_abbr):
            ax.annotate(lbl, (Y[i,0], Y[i,1]), fontsize=8, alpha=0.95)

        # leyenda inferior también en el caso sin clusters
        handles = [Patch(facecolor=c, edgecolor='none', label=f) for f, c in fam2color.items()]
        fig.subplots_adjust(bottom=0.16)
        fig.legend(handles=handles, loc='lower center', ncol=max(1, args.legend_ncols), fontsize=8, frameon=False, bbox_to_anchor=(0.5, 0.02))

        fig.savefig(final_out, dpi=300, bbox_inches='tight')
        plt.close(fig)

        metrics = {
            'stress': float(stress),
            'spearman_corr_base_vs_composite': float(spearman_corr),
            'clustering_time_s': float(elapsed),
            'best_score': float(score),
            'config': config,
            'n_valid_clusters': int(len(valid_clusters)),
            'n_all_clusters': int(len(all_clusters))
        }
        metrics_path = os.path.join(args.outdir, 'metrics_nogrid_v5.json')
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2)
        clusters_detail = {
            'valid_clusters': valid_clusters,
            'all_clusters': all_clusters,
            'labels': labels.tolist() if hasattr(labels, "tolist") else list(labels)
        }
        clusters_path = os.path.join(args.outdir, 'clusters_nogrid_v5.json')
        with open(clusters_path, 'w', encoding='utf-8') as f:
            json.dump(clusters_detail, f, indent=2)

        print(f"Mapa generado en {final_out}")
        print(f"Métricas guardadas en {metrics_path}")
        print(f"Clusters guardados en {clusters_path}")
        return

    # Draw final with ellipses for valid clusters
    plot_final(Y, labels_abbr, families, fam2color, labels, valid_clusters,
               mutual, composite_S, final_out,
               ellipse_n_std=args.ellipse_n_std, ellipse_alpha=args.ellipse_alpha, topk_arrows=args.topk_arrows,
               legend_ncols=args.legend_ncols)

    # Save metrics and config
    metrics = {
        'stress': float(stress),
        'spearman_corr_base_vs_composite': float(spearman_corr),
        'clustering_time_s': float(elapsed),
        'best_score': float(score),
        'config': config,
        'n_valid_clusters': int(len(valid_clusters)),
        'n_all_clusters': int(len(all_clusters))
    }
    metrics_path = os.path.join(args.outdir, 'metrics_nogrid_v5.json')
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)

    clusters_detail = {
        'valid_clusters': valid_clusters,
        'all_clusters': all_clusters,
        'labels': labels.tolist() if hasattr(labels, "tolist") else list(labels)
    }
    clusters_path = os.path.join(args.outdir, 'clusters_nogrid_v5.json')
    with open(clusters_path, 'w', encoding='utf-8') as f:
        json.dump(clusters_detail, f, indent=2)

    print(f"Mapa final generado en {final_out}")
    print(f"Métricas guardadas en {metrics_path}")
    print(f"Clusters guardados en {clusters_path}")

if __name__ == '__main__':
    main()
