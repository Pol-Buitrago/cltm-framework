#!/usr/bin/env python3
"""
transfer_map_options_refined.py

Versión refinada: métodos seleccionados y nuevas variantes orientadas a
visualizar clusters basados en contribuciones relativas (per-row donor profiles).

Opciones disponibles: js_mds, umap_affinity, tsne_profiles, isomap,
nonmetric_mds, radial_dendrogram, topk_graph_map, all

Uso ejemplo:
python3 transfer_map_options_refined.py --matrix-csv /ruta/transfer_matrix.csv \
    --abbrev-map /ruta/abbrev.json --outdir ./maps --option all
"""
import os
import json
import argparse
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex

from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE, MDS, Isomap
from sklearn.decomposition import PCA
from sklearn.neighbors import kneighbors_graph
from sklearn.cluster import AgglomerativeClustering

import networkx as nx
import scipy
from scipy.spatial import ConvexHull
from scipy.cluster import hierarchy
from scipy.spatial.distance import jensenshannon

# umap optional
try:
    import umap
except Exception:
    umap = None

# -----------------------------
# Config: mapping nombre->familia
# -----------------------------
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
    "Kyrgyz": "Turkic",
    "Turkish": "Turkic",
    "Uyghur": "Turkic",
    "Uzbek": "Turkic",
    "Estonian": "Uralic (Finnic)",
    "Hungarian": "Uralic",
    "Meadow Mari": "Uralic",
    "Abkhazian": "Northwest Caucasian",
}

# -----------------------------
# Helpers
# -----------------------------
def ensure_dir(path):
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

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
    out = []
    for ab in abbrev_list:
        name = abbrev_map.get(ab, None)
        if name is None:
            name = abbrev_map.get(ab.lower(), ab)
        out.append(name)
    return out

def assign_families(full_names):
    return [NAME2FAMILY.get(nm, 'Unknown') if nm is not None else 'Unknown' for nm in full_names]

def assign_family_colors(families):
    """
    Input: families (list aligned with labels)
    Returns: fam2color (dict family->hex), cols (list of hex per label)
    """
    unique = sorted(set(families), key=lambda x: (x != 'Unknown', x))
    cmaps = [plt.get_cmap(c) for c in ['tab20', 'tab20b', 'tab20c']]
    colors = []
    for cmap in cmaps:
        colors.extend([to_hex(cmap(i)) for i in range(cmap.N)])
    colors = colors[:len(unique)]
    fam2color = {f: colors[i] for i, f in enumerate(unique)}
    cols = [fam2color[f] for f in families]
    return fam2color, cols

def draw_family_hulls(ax, coords, families, fam2color, alpha=0.12):
    for fam in sorted(set(families)):
        idxs = [i for i, f in enumerate(families) if f == fam]
        if len(idxs) >= 3:
            pts = coords[idxs]
            try:
                hull = ConvexHull(pts)
                poly = plt.Polygon(pts[hull.vertices], facecolor=fam2color[fam], alpha=alpha, edgecolor=fam2color[fam])
                ax.add_patch(poly)
            except Exception:
                pass

# ------------------------------------
# Per-row donor-normalized profiles
# ------------------------------------
def compute_row_profiles(M, clip_negative=True, smoothing=1e-12, exclude_diag=False):
    """
    Normaliza cada fila de M para obtener una distribución sobre donantes.
    - clip_negative: si True, valores negativos se ponen a 0 antes de normalizar.
    - exclude_diag: si True, se ignora la diagonal (auto-transfer) en la normalización.
    """
    P = M.copy().astype(float)
    if clip_negative:
        P = np.maximum(P, 0.0)
    if exclude_diag:
        P = P.copy()
        np.fill_diagonal(P, 0.0)
    row_sums = P.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    Pn = P / (row_sums + smoothing)
    return Pn

def js_distance_matrix_from_profiles(P):
    n = P.shape[0]
    D = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i+1, n):
            d = jensenshannon(P[i], P[j], base=2.0)
            if np.isnan(d):
                d = 0.0
            D[i, j] = d
            D[j, i] = d
    return D

# ------------------------------------
# Map methods
# ------------------------------------
def map_js_mds(M, labels, families, outpath):
    ensure_dir(os.path.dirname(outpath))
    P = compute_row_profiles(M, clip_negative=True, exclude_diag=False)
    D = js_distance_matrix_from_profiles(P)
    np.fill_diagonal(D, 0.0)
    mds = MDS(n_components=2, dissimilarity='precomputed', metric=False, random_state=42, n_init=4, max_iter=500)
    Y = mds.fit_transform(D)
    fam2color, cols = assign_family_colors(families)
    fig, ax = plt.subplots(figsize=(12,10))
    ax.scatter(Y[:,0], Y[:,1], c=cols, s=100, alpha=0.95)
    draw_family_hulls(ax, Y, families, fam2color)
    for i, lbl in enumerate(labels):
        ax.annotate(lbl, (Y[i,0], Y[i,1]), fontsize=8)
    for f in fam2color:
        ax.scatter([], [], color=fam2color[f], label=f)
    ax.set_title('JS-distance MDS (perfiles de donantes)')
    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close(fig)

def map_umap_affinity_profiles(M, labels, families, outpath, n_neighbors=15, min_dist=0.1):
    if umap is None:
        raise RuntimeError('umap no está instalado')
    ensure_dir(os.path.dirname(outpath))
    P = compute_row_profiles(M, clip_negative=True, exclude_diag=False)
    A = (P + P.T) / 2.0
    Xp = PCA(n_components=min(50, A.shape[0]-1), random_state=42).fit_transform(A)
    Xs = StandardScaler().fit_transform(Xp)
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, metric='euclidean', random_state=42)
    Y = reducer.fit_transform(Xs)
    fam2color, cols = assign_family_colors(families)
    fig, ax = plt.subplots(figsize=(12,10))
    ax.scatter(Y[:,0], Y[:,1], c=cols, s=100, alpha=0.95)
    draw_family_hulls(ax, Y, families, fam2color)
    for i, lbl in enumerate(labels):
        ax.annotate(lbl, (Y[i,0], Y[i,1]), fontsize=8)
    for f in fam2color:
        ax.scatter([], [], color=fam2color[f], label=f)
    ax.set_title('UMAP (afinidad de perfiles de donantes)')
    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close(fig)

def map_tsne_profiles(M, labels, families, outpath, perplexity=30):
    ensure_dir(os.path.dirname(outpath))
    P = compute_row_profiles(M, clip_negative=True, exclude_diag=False)
    X = StandardScaler().fit_transform(P)
    pca = PCA(n_components=min(50, X.shape[1]-1), random_state=42)
    Xp = pca.fit_transform(X)
    per = min(perplexity, max(5, Xp.shape[0]//3))
    tsne = TSNE(n_components=2, perplexity=per, init='pca', random_state=42)
    Y = tsne.fit_transform(Xp)
    fam2color, cols = assign_family_colors(families)
    fig, ax = plt.subplots(figsize=(12,10))
    ax.scatter(Y[:,0], Y[:,1], c=cols, s=90, alpha=0.95)
    draw_family_hulls(ax, Y, families, fam2color)
    for i, lbl in enumerate(labels):
        ax.annotate(lbl, (Y[i,0], Y[i,1]), fontsize=8)
    for f in fam2color:
        ax.scatter([], [], color=fam2color[f], label=f)
    ax.set_title('t-SNE (perfiles de donantes)')
    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close(fig)

def map_isomap(M, labels, families, outpath, n_neighbors=10):
    ensure_dir(os.path.dirname(outpath))
    P = compute_row_profiles(M, clip_negative=True, exclude_diag=False)
    A = (P + P.T) / 2.0
    Xp = PCA(n_components=min(50, A.shape[0]-1), random_state=42).fit_transform(A)
    iso = Isomap(n_neighbors=min(n_neighbors, A.shape[0]-1), n_components=2)
    Y = iso.fit_transform(Xp)
    fam2color, cols = assign_family_colors(families)
    fig, ax = plt.subplots(figsize=(12,10))
    ax.scatter(Y[:,0], Y[:,1], c=cols, s=90, alpha=0.95)
    draw_family_hulls(ax, Y, families, fam2color)
    for i, lbl in enumerate(labels):
        ax.annotate(lbl, (Y[i,0], Y[i,1]), fontsize=8)
    for f in fam2color:
        ax.scatter([], [], color=fam2color[f], label=f)
    ax.set_title('Isomap (perfiles de donantes)')
    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close(fig)

def map_nonmetric_mds(M, labels, families, outpath):
    ensure_dir(os.path.dirname(outpath))
    P = compute_row_profiles(M, clip_negative=True, exclude_diag=False)
    A = (P + P.T) / 2.0
    D = 1.0 - A
    D[D < 0] = 0.0
    np.fill_diagonal(D, 0.0)
    mds = MDS(n_components=2, dissimilarity='precomputed', metric=False, random_state=42, n_init=4, max_iter=500)
    Y = mds.fit_transform(D)
    fam2color, cols = assign_family_colors(families)
    fig, ax = plt.subplots(figsize=(12,10))
    ax.scatter(Y[:,0], Y[:,1], c=cols, s=100, alpha=0.95)
    draw_family_hulls(ax, Y, families, fam2color)
    for i, lbl in enumerate(labels):
        ax.annotate(lbl, (Y[i,0], Y[i,1]), fontsize=8)
    for f in fam2color:
        ax.scatter([], [], color=fam2color[f], label=f)
    ax.set_title('Non-metric MDS (perfiles simetrizados)')
    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close(fig)

def map_hierarchical_radial(M, labels, families, outpath, method='ward', k_radius=5, rmin=0.25, rmax=1.0):
    """
    Radial dendrogram refined:
    - Normaliza filas (perfiles de donantes).
    - Calcula distancia simétrica A = (P + P^T)/2 y D = 1 - A (0..1).
    - Clustering jerárquico con linkage (method).
    - Ordena hojas (leaves_list) para asignar ángulos.
    - Calcula por cada hoja la media de distancia a sus k vecinos más cercanos y
      usa esa métrica para mapear la distancia radial: idiomas con vecinos cercanos
      (baja media) reciben radio pequeño (más centrados).
    - rmin/rmax controlan el rango radial (0 es centro).
    """
    ensure_dir(os.path.dirname(outpath))

    # 1. perfiles por fila
    P = compute_row_profiles(M, clip_negative=True, exclude_diag=False)

    # 2. afinidad y distancia
    A = (P + P.T) / 2.0
    D = 1.0 - A
    D[D < 0] = 0.0
    np.fill_diagonal(D, 0.0)  # imprescindible para squareform

    # 3. linkage y orden de hojas
    Z = hierarchy.linkage(scipy.spatial.distance.squareform(D), method=method)
    leaf_order = hierarchy.leaves_list(Z)
    n = len(leaf_order)

    # 4. calcular medida local de proximidad (mean dist a k vecinos)
    k = min(max(1, k_radius), max(1, n-1))
    mean_knn = np.zeros(n, dtype=float)
    for i in range(n):
        drow = np.copy(D[i])
        drow[i] = np.inf
        nn_idx = np.argsort(drow)[:k]
        mean_knn[i] = drow[nn_idx].mean()

    # 5. normalizar mean_knn a rango radial inverso: menores distancias -> menor r
    mn = mean_knn.min()
    mx = mean_knn.max()
    if mx - mn <= 0:
        norm = np.zeros_like(mean_knn)
    else:
        norm = (mean_knn - mn) / (mx - mn)  # 0..1 where 0 = most central-worthy
    # invertimos para que valores pequeños (vecinos cercanos) sean r pequeños
    r_vals = rmin + (1.0 - norm) * (rmax - rmin)

    # 6. asignar ángulos segun leaf_order y coordenadas polares
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    coords = np.zeros((n, 2))
    # coords indexed by original label index
    for pos_idx, leaf in enumerate(leaf_order):
        r = float(r_vals[leaf])
        theta = angles[pos_idx]
        coords[leaf] = np.array([r * np.cos(theta), r * np.sin(theta)])

    # 7. colores por familia
    fam2color, cols = assign_family_colors(families)

    # 8. plot
    fig, ax = plt.subplots(figsize=(10,10))
    ax.scatter(coords[:,0], coords[:,1], c=cols, s=90, edgecolor='k', linewidth=0.2)
    # dibujar hulls (usando coords)
    draw_family_hulls(ax, coords, families, fam2color, alpha=0.12)
    for i, lbl in enumerate(labels):
        ax.annotate(lbl, (coords[i,0], coords[i,1]), fontsize=8, alpha=0.9)
    for f in fam2color:
        ax.scatter([], [], color=fam2color[f], label=f)
    ax.set_title('Radial dendrogram (leaves posicionadas por proximidad local)')
    ax.axis('off')
    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close(fig)

def map_topk_graph_map(M, labels, families, outpath, top_k=3):
    ensure_dir(os.path.dirname(outpath))
    P = compute_row_profiles(M, clip_negative=True, exclude_diag=False)
    n = P.shape[0]
    G = nx.DiGraph()
    for i in range(n):
        G.add_node(labels[i], family=families[i])
    for i in range(n):
        idx = np.argsort(P[i])[::-1]
        cnt = 0
        for j in idx:
            if i == j:
                continue
            if P[i,j] <= 0:
                continue
            G.add_edge(labels[j], labels[i], weight=float(P[i,j]))
            cnt += 1
            if cnt >= top_k:
                break
    A = nx.to_numpy_array(G, nodelist=labels)
    A_sym = (A + A.T) / 2.0
    from sklearn.manifold import SpectralEmbedding
    emb = SpectralEmbedding(n_components=2, affinity='precomputed', random_state=42)
    Y = emb.fit_transform(A_sym)
    pos_init = {labels[i]: (float(Y[i,0]), float(Y[i,1])) for i in range(len(labels))}
    H = nx.Graph()
    H.add_nodes_from(G.nodes(data=True))
    for u, v, d in G.edges(data=True):
        w = d.get('weight', 0.0)
        if w > 0:
            H.add_edge(u, v, weight=w)
    pos = nx.spring_layout(H, pos=pos_init, weight='weight', seed=42, iterations=300)
    fam2color, cols = assign_family_colors(families)
    fig, ax = plt.subplots(figsize=(12,10))
    nx.draw_networkx_edges(H, pos, alpha=0.25, width=0.8, edge_color='gray')
    node_colors = [fam2color[families[labels.index(n)]] for n in H.nodes()]
    nx.draw_networkx_nodes(H, pos, node_color=node_colors, node_size=160)
    nx.draw_networkx_labels(H, pos, font_size=8)
    for u, v, d in G.edges(data=True):
        w = d.get('weight', 0.0)
        if w > 0.0:
            x1, y1 = pos[u]
            x2, y2 = pos[v]
            ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                        arrowprops=dict(arrowstyle='->', lw=0.8, alpha=0.6, color='k'))
    for f in fam2color:
        ax.scatter([], [], color=fam2color[f], label=f)
    ax.set_title(f'Top-{top_k} donor graph map (spectral init + spring layout)')
    ax.axis('off')
    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close(fig)

# ------------------------------------
# Runner
# ------------------------------------
def run_option(opt, M, labels, families, outdir):
    if opt == 'js_mds':
        map_js_mds(M, labels, families, os.path.join(outdir, 'map_js_mds.png'))
    elif opt == 'umap_affinity':
        map_umap_affinity_profiles(M, labels, families, os.path.join(outdir, 'map_umap_affinity.png'))
    elif opt == 'tsne_profiles':
        map_tsne_profiles(M, labels, families, os.path.join(outdir, 'map_tsne_profiles.png'))
    elif opt == 'isomap':
        map_isomap(M, labels, families, os.path.join(outdir, 'map_isomap.png'))
    elif opt == 'nonmetric_mds':
        map_nonmetric_mds(M, labels, families, os.path.join(outdir, 'map_nonmetric_mds.png'))
    elif opt == 'radial_dendrogram':
        map_hierarchical_radial(M, labels, families, os.path.join(outdir, 'map_radial_dendrogram.png'))
    elif opt == 'topk_graph_map':
        map_topk_graph_map(M, labels, families, os.path.join(outdir, 'map_topk_graph_map.png'))
    else:
        raise ValueError('Opcion desconocida: ' + str(opt))

def main():
    parser = argparse.ArgumentParser(description='Generar mapas 2D refinados desde matriz de transferencia')
    parser.add_argument('--matrix-csv', required=True)
    parser.add_argument('--abbrev-map', default=None)
    parser.add_argument('--outdir', default='./transfer_maps')
    parser.add_argument('--option', default='all', help='js_mds, umap_affinity, tsne_profiles, isomap, nonmetric_mds, radial_dendrogram, topk_graph_map, all')
    args = parser.parse_args()

    ensure_dir(args.outdir)
    df = pd.read_csv(args.matrix_csv, index_col=0)
    labels_abbr = list(df.index)
    M = df.values.astype(float)
    abbrev_map = load_abbrev_map(args.abbrev_map) if args.abbrev_map else None
    labels_full = map_abbrevs_to_names(labels_abbr, abbrev_map)
    families = assign_families(labels_full)

    opts = ['js_mds', 'umap_affinity', 'tsne_profiles', 'isomap', 'nonmetric_mds', 'radial_dendrogram', 'topk_graph_map']
    if args.option == 'all':
        for opt in opts:
            print('Generando', opt)
            run_option(opt, M, labels_abbr, families, args.outdir)
    else:
        if args.option not in opts:
            raise ValueError('option must be one of: ' + ','.join(opts))
        run_option(args.option, M, labels_abbr, families, args.outdir)

    print('Mapas generados en', args.outdir)

if __name__ == '__main__':
    main()
