#!/usr/bin/env python3
"""
transfer_map_options.py

Diez opciones de mapas 2D para visualizar una matriz de transferencia entre idiomas.
Cada función genera un mapa distinto, justificado metodológicamente en su docstring.

Dependencias mínimas: numpy, pandas, matplotlib, sklearn, scipy, networkx.
Umap es opcional, se usa si está instalado.

Uso ejemplo:
python3 transfer_map_options.py --matrix-csv /ruta/transfer_matrix.csv --abbrev-map /ruta/abbrev.json --outdir ./maps --option all

Opciones: 1..10, all

Nota: la implementación prioriza claridad y robustez; ajusta parámetros según el tamaño del dataset.
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
    unique = sorted(set(families), key=lambda x: (x != 'Unknown', x))
    cmaps = [plt.get_cmap(c) for c in ['tab20', 'tab20b', 'tab20c']]
    colors = []
    for cmap in cmaps:
        colors.extend([to_hex(cmap(i)) for i in range(cmap.N)])
    colors = colors[:len(unique)]
    fam2color = {f: colors[i] for i, f in enumerate(unique)}
    cols = [fam2color[f] for f in families]
    return fam2color, cols


def symmetrize_affinity_from_M(M, clip_negative=True):
    # S = (M + M.T)/2, option to clip negative entries to zero
    S = (M + M.T) / 2.0
    if clip_negative:
        S = np.maximum(S, 0.0)
    return S


def affinity_to_distance(A):
    # distance in [0, inf) note: assumes A in [0,1] roughly
    D = 1.0 - A
    D[D < 0] = 0.0
    np.fill_diagonal(D, 0.0)
    return D


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
                # si falla convex hull, ignorar
                pass

# -----------------------------
# METODOS: 10 mapas distintos
# -----------------------------

# 1) Force-directed on thresholded graph
def map_force_threshold(M, labels, families, outpath, thresh_pct=90):
    """
    1) Force-directed layout on thresholded weighted graph.
    Razon: conserva la estructura de red, los nodos muy conectados se agrupan,
    y eliminando aristas pequeñas se reduce el ruido. Es intuitivo y revelador
    para flujos de transferencia entre familias.
    """
    ensure_dir(os.path.dirname(outpath))
    A = symmetrize_affinity_from_M(M)
    # Umbral absoluto como percentil
    thresh = np.percentile(A, thresh_pct)
    G = nx.Graph()
    n = A.shape[0]
    for i in range(n):
        G.add_node(labels[i], family=families[i])
    for i in range(n):
        for j in range(i+1, n):
            w = float(A[i, j])
            if w >= thresh:
                G.add_edge(labels[i], labels[j], weight=w)
    # spring layout with weights
    weights = [d['weight'] for u, v, d in G.edges(data=True)]
    pos = nx.spring_layout(G, weight='weight', seed=42, iterations=300)

    fam2color, cols = assign_family_colors(families)
    fig, ax = plt.subplots(figsize=(12, 10))
    # draw edges faint
    nx.draw_networkx_edges(G, pos, alpha=0.45, width=[max(0.4, w*6) for w in weights], ax=ax)
    # nodes colored by family
    node_colors = [fam2color[families[labels.index(n)]] for n in G.nodes()]
    sizes = [300 for _ in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_size=sizes, node_color=node_colors, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=8, ax=ax)
    # legend
    for f in fam2color:
        ax.scatter([], [], color=fam2color[f], label=f)
    ax.legend(fontsize=8, bbox_to_anchor=(1.02, 1.0), loc='upper left')
    ax.set_title('Force-directed map (thresholded edges)')
    ax.axis('off')
    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close(fig)

# 2) Force-directed with family anchors (contraction)
def map_force_family_anchored(M, labels, families, outpath, anchor_strength=0.2):
    """
    2) Force-directed layout with soft anchoring toward family centroids.
    Razon: refuerza la cohesión intra-familiar manteniendo la información
    de vecinos inter-familiares, facilita ver supergrupos y outliers.
    """
    ensure_dir(os.path.dirname(outpath))
    A = symmetrize_affinity_from_M(M)
    n = A.shape[0]
    G = nx.Graph()
    for i in range(n):
        G.add_node(labels[i], family=families[i])
    # add weighted edges for top-k neighbors to avoid más ruido
    k = max(3, min(10, n//10))
    for i in range(n):
        idx = np.argsort(A[i])[::-1]
        for j in idx[:k]:
            if i == j:
                continue
            G.add_edge(labels[i], labels[j], weight=float(A[i, j]))
    # compute spring layout
    pos = nx.spring_layout(G, weight='weight', seed=42, iterations=300)
    # compute family centroids in layout space
    fam2nodes = defaultdict(list)
    for node in G.nodes():
        fam2nodes[G.nodes[node]['family']].append(node)
    fam_centroids = {}
    for fam, nodes in fam2nodes.items():
        pts = np.array([pos[n] for n in nodes])
        fam_centroids[fam] = pts.mean(axis=0)
    # apply soft anchoring by shifting node positions toward centroid
    for node in list(G.nodes()):
        fam = G.nodes[node]['family']
        pos[node] = (1-anchor_strength)*np.array(pos[node]) + anchor_strength*fam_centroids[fam]

    fam2color, _ = assign_family_colors(families)
    fig, ax = plt.subplots(figsize=(12,10))
    nx.draw_networkx_edges(G, pos, alpha=0.35, width=0.6, ax=ax)
    node_colors = [fam2color[G.nodes[n]['family']] for n in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_size=220, node_color=node_colors, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=8, ax=ax)
    for f in fam2color:
        ax.scatter([], [], color=fam2color[f], label=f)
    ax.legend(fontsize=8, bbox_to_anchor=(1.02,1.0), loc='upper left')
    ax.set_title('Force-directed map with family anchoring')
    ax.axis('off')
    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close(fig)

# 3) UMAP on symmetric affinity
def map_umap_affinity(M, labels, families, outpath, n_neighbors=15, min_dist=0.1):
    """
    3) UMAP sobre afinidad simétrica.
    Razon: UMAP preserva estructura local y cierta estructura global, suele separar
    clusters mejor que t-SNE en muchos datasets, es robusto a ruido si se usa
    la afinidad simétrica derivada de la matriz de transferencia.
    """
    ensure_dir(os.path.dirname(outpath))
    if umap is None:
        raise RuntimeError('umap no est\u00e1 instalado, instala umap-learn para usar esta opci\u00f3n')
    A = symmetrize_affinity_from_M(M)
    # convert affinity a "feature" aunque UMAP puede recibir X, usamos PCA primero
    pca = PCA(n_components=min(50, A.shape[0]-1), random_state=42)
    Xp = pca.fit_transform(A)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(Xp)
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, metric='euclidean', random_state=42)
    Y = reducer.fit_transform(Xs)
    fam2color, cols = assign_family_colors(families)
    fig, ax = plt.subplots(figsize=(12,10))
    ax.scatter(Y[:,0], Y[:,1], c=cols, s=110, alpha=0.9)
    draw_family_hulls(ax, Y, families, fam2color)
    for i, lbl in enumerate(labels):
        ax.annotate(lbl, (Y[i,0], Y[i,1]), fontsize=8, alpha=0.9)
    for f in fam2color:
        ax.scatter([], [], color=fam2color[f], label=f)
    ax.legend(fontsize=8, bbox_to_anchor=(1.02,1.0), loc='upper left')
    ax.set_title('UMAP map (affinity-based)')
    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close(fig)

# 4) t-SNE on transfer-profile rows
def map_tsne_profiles(M, labels, families, outpath, perplexity=30):
    """
    4) t-SNE sobre los vectores de perfil de transferencia (cada fila de M).
    Razon: optimiza la preservaci\u00f3n de vecinos locales, muy eficaz para detectar
    subgrupos y outliers locales; ajustar perplexity seg\u00fan N.
    """
    ensure_dir(os.path.dirname(outpath))
    X = StandardScaler().fit_transform(M)
    # PCA previo para estabilidad
    pca = PCA(n_components=min(50, X.shape[1]-1), random_state=42)
    Xp = pca.fit_transform(X)
    tsne = TSNE(n_components=2, perplexity=min(perplexity, max(5, Xp.shape[0]//3)), init='pca', random_state=42)
    Y = tsne.fit_transform(Xp)
    fam2color, cols = assign_family_colors(families)
    fig, ax = plt.subplots(figsize=(12,10))
    ax.scatter(Y[:,0], Y[:,1], c=cols, s=100, alpha=0.9)
    draw_family_hulls(ax, Y, families, fam2color)
    for i, lbl in enumerate(labels):
        ax.annotate(lbl, (Y[i,0], Y[i,1]), fontsize=8, alpha=0.9)
    for f in fam2color:
        ax.scatter([], [], color=fam2color[f], label=f)
    ax.legend(fontsize=8, bbox_to_anchor=(1.02,1.0), loc='upper left')
    ax.set_title('t-SNE of transfer profiles')
    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close(fig)

# 5) Isomap (geodesic distances)
def map_isomap(M, labels, families, outpath, n_neighbors=10):
    """
    5) Isomap aplicado a la distancia geod\u00e9sica sobre la afinidad sim\u00e9trica.
    Razon: si los puntos se distribuyen sobre una variedad no lineal, Isomap
    recupera la estructura global de la misma conservando distancias geod\u00e9sicas.
    """
    ensure_dir(os.path.dirname(outpath))
    A = symmetrize_affinity_from_M(M)
    D = affinity_to_distance(A)
    # Isomap necesita features, no dissimilarity precomputed en sklearn, usamos kNN graph
    iso = Isomap(n_neighbors=min(n_neighbors, A.shape[0]-1), n_components=2)
    # como Isomap espera X, damos PCA de A
    Xp = PCA(n_components=min(50, A.shape[0]-1), random_state=42).fit_transform(A)
    Y = iso.fit_transform(Xp)
    fam2color, cols = assign_family_colors(families)
    fig, ax = plt.subplots(figsize=(12,10))
    ax.scatter(Y[:,0], Y[:,1], c=cols, s=100, alpha=0.9)
    draw_family_hulls(ax, Y, families, fam2color)
    for i, lbl in enumerate(labels):
        ax.annotate(lbl, (Y[i,0], Y[i,1]), fontsize=8, alpha=0.9)
    for f in fam2color:
        ax.scatter([], [], color=fam2color[f], label=f)
    ax.legend(fontsize=8, bbox_to_anchor=(1.02,1.0), loc='upper left')
    ax.set_title('Isomap of transfer affinity')
    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close(fig)

# 6) Non-metric MDS on symmetric distance
def map_nonmetric_mds(M, labels, families, outpath):
    """
    6) MDS no metrico sobre la distancia sim\u00e9trica.
    Razon: MDS no m\u00e9trico optimiza la preservaci\u00f3n del orden de proximidades,
    es apropiado cuando la escala absoluta de distancias no es fiable.
    """
    ensure_dir(os.path.dirname(outpath))
    A = symmetrize_affinity_from_M(M)
    D = affinity_to_distance(A)
    mds = MDS(n_components=2, dissimilarity='precomputed', metric=False, random_state=42, n_init=4, max_iter=500)
    Y = mds.fit_transform(D)
    fam2color, cols = assign_family_colors(families)
    fig, ax = plt.subplots(figsize=(12,10))
    ax.scatter(Y[:,0], Y[:,1], c=cols, s=100, alpha=0.9)
    draw_family_hulls(ax, Y, families, fam2color)
    for i, lbl in enumerate(labels):
        ax.annotate(lbl, (Y[i,0], Y[i,1]), fontsize=8, alpha=0.9)
    for f in fam2color:
        ax.scatter([], [], color=fam2color[f], label=f)
    ax.legend(fontsize=8, bbox_to_anchor=(1.02,1.0), loc='upper left')
    ax.set_title('Non-metric MDS')
    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close(fig)

# 7) Diffusion maps (autovectores de la matriz de transici\u00f3n)
def map_diffusion(M, labels, families, outpath, alpha=0.5, t=1):
    """
    7) Diffusion map, autovectores de la matriz de transici\u00f3n normalizada.
    Razon: captura conectividad de largo alcance y estructura de clúster a varias escalas,
    es robusto al ruido, y las coordenadas difusivas representan tiempos de difusi\u00f3n.
    """
    ensure_dir(os.path.dirname(outpath))
    A = symmetrize_affinity_from_M(M)
    # Formar matriz de afinidad K, aplicar normalizaci\u00f3n (alpha) como en diffusion maps
    K = A.copy()
    # evitar ceros totales
    row_sums = K.sum(axis=1)
    row_sums[row_sums == 0] = 1.0
    D_alpha = np.diag(row_sums ** (-alpha))
    K_tilde = D_alpha @ K @ D_alpha
    q = K_tilde.sum(axis=1)
    q[q == 0] = 1.0
    P = (K_tilde / q[:, None])
    # calcular autovectores de P
    try:
        vals, vecs = np.linalg.eig(P)
    except Exception:
        vals, vecs = scipy.linalg.eig(P)
    # ordenar por valor propio real descendente
    idx = np.argsort(-np.real(vals))
    vals = np.real(vals[idx])
    vecs = np.real(vecs[:, idx])
    # coordenadas difusivas: lambdas^t * psi
    # saltar primer autovector (trivial), tomar siguientes dos
    if vecs.shape[1] < 3:
        raise RuntimeError('No hay suficientes autovectores para diffusion map')
    Y = (vals[1:3]**t) * vecs[:, 1:3]
    fam2color, cols = assign_family_colors(families)
    fig, ax = plt.subplots(figsize=(12,10))
    ax.scatter(Y[:,0], Y[:,1], c=cols, s=110, alpha=0.95)
    draw_family_hulls(ax, Y, families, fam2color)
    for i, lbl in enumerate(labels):
        ax.annotate(lbl, (Y[i,0], Y[i,1]), fontsize=8, alpha=0.9)
    for f in fam2color:
        ax.scatter([], [], color=fam2color[f], label=f)
    ax.legend(fontsize=8, bbox_to_anchor=(1.02,1.0), loc='upper left')
    ax.set_title('Diffusion Map (t=%d)' % t)
    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close(fig)

# 8) Radial dendrogram (clustering jerarquico proyectado a 2D)
def map_hierarchical_radial(M, labels, families, outpath, method='ward'):
    """
    8) Dendrograma radial: clustering jerarquico proyectado en coordenadas polares.
    Razon: muestra la jerarquia completa de relaciones entre idiomas, ideal para
    revelar subfamilias y la estructura anidada de transferencia.
    """
    ensure_dir(os.path.dirname(outpath))
    A = symmetrize_affinity_from_M(M)
    D = affinity_to_distance(A)
    # linkage
    Z = hierarchy.linkage(scipy.spatial.distance.squareform(D), method=method)
    # obtener orden de hojas del dendrograma
    leaf_order = hierarchy.leaves_list(Z)
    # asignar angulos equiespaciados a hojas
    n = len(leaf_order)
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    pos_leaf = {}
    for i, leaf in enumerate(leaf_order):
        r = 1.0
        theta = angles[i]
        pos_leaf[leaf] = np.array([r*np.cos(theta), r*np.sin(theta)])
    # para nodos internos, proyectar al radio segun altura del linkage
    # simplificaci\u00f3n: solo dibujar hojas como puntos y conectar segun linkage
    fig, ax = plt.subplots(figsize=(10,10))
    fam2color, cols = assign_family_colors(families)
    coords = np.zeros((n, 2))
    for idx_pos, leaf in enumerate(leaf_order):
        coords[leaf] = pos_leaf[leaf]
    ax.scatter(coords[:,0], coords[:,1], c=[cols[i] for i in range(n)], s=90)
    for i, lbl in enumerate(labels):
        ax.annotate(lbl, (coords[i,0], coords[i,1]), fontsize=8)
    for f in fam2color:
        ax.scatter([], [], color=fam2color[f], label=f)
    ax.legend(fontsize=8, bbox_to_anchor=(1.02,1.0), loc='upper left')
    ax.set_title('Radial dendrogram leaves (jerarquico)')
    ax.axis('off')
    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close(fig)

# 9) Family centroids map (centroides + dispersion)
def map_family_centroids(M, labels, families, outpath):
    """
    9) Map of family centroids: proyecta cada idioma en PCA 2D, dibuja centroides
    de familia y vectores dispersi\u00f3n. Razon: mantiene interpretabilidad de ejes,
    muestra claramente qu\u00e9 familias est\u00e1n pr\u00f3ximas y su variabilidad interna.
    """
    ensure_dir(os.path.dirname(outpath))
    A = symmetrize_affinity_from_M(M)
    # usar PCA de la matriz de afinidad para ejes interpretables
    pca = PCA(n_components=2, random_state=42)
    Y = pca.fit_transform(A)
    fam2color, cols = assign_family_colors(families)
    fig, ax = plt.subplots(figsize=(12,10))
    # puntos idioma
    ax.scatter(Y[:,0], Y[:,1], c=cols, s=60, alpha=0.9)
    # centroides
    fam2nodes = defaultdict(list)
    for i, f in enumerate(families):
        fam2nodes[f].append(i)
    for fam, nodes in fam2nodes.items():
        pts = Y[nodes]
        centroid = pts.mean(axis=0)
        ax.scatter(centroid[0], centroid[1], s=300, marker='X', color=fam2color[fam], edgecolor='k')
        # dibujar vectores de dispersion
        for p in pts:
            ax.plot([centroid[0], p[0]], [centroid[1], p[1]], color=fam2color[fam], alpha=0.2)
    for i, lbl in enumerate(labels):
        ax.annotate(lbl, (Y[i,0], Y[i,1]), fontsize=8)
    for f in fam2color:
        ax.scatter([], [], color=fam2color[f], label=f)
    ax.legend(fontsize=8, bbox_to_anchor=(1.02,1.0), loc='upper left')
    ax.set_title('Family centroids and dispersion (PCA)')
    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close(fig)

# 10) KNN graph + spring embedding (graph Laplacian initialisation)
def map_knn_graph_spring(M, labels, families, outpath, k=8):
    """
    10) Construir grafo kNN sobre perfiles de transferencia, usar spring layout
    inicializado con eigenvectors del Laplaciano, luego visualizar como mapa.
    Razon: combina poder de preservaci\u00f3n de vecinos locales (kNN) con estabilidad
    num\u00e9rica de la inicializaci\u00f3n espectral, produce mapas claros de clusters.
    """
    ensure_dir(os.path.dirname(outpath))
    X = StandardScaler().fit_transform(M)
    # matriz kNN (simetricizada)
    knn = kneighbors_graph(X, n_neighbors=min(k, X.shape[0]-1), mode='connectivity', include_self=False)
    A = 0.5 * (knn.toarray() + knn.toarray().T)
    G = nx.from_numpy_array(A)
    mapping = dict(enumerate(labels))
    G = nx.relabel_nodes(G, mapping)
    # spectral initial positions: usar los 2 primeros vectores propios del Laplaciano
    L = nx.normalized_laplacian_matrix(G).astype(float).todense()
    vals, vecs = np.linalg.eigh(np.array(L))
    # tomar eigenvectores con indices 1 y 2 (0 es triv)
    if vecs.shape[1] > 2:
        pos_init = {labels[i]: (vecs[i,1], vecs[i,2]) for i in range(len(labels))}
    else:
        pos_init = None
    pos = nx.spring_layout(G, pos=pos_init, seed=42, iterations=300)
    fam2color, _ = assign_family_colors(families)
    fig, ax = plt.subplots(figsize=(12,10))
    nx.draw_networkx_edges(G, pos, alpha=0.35)
    node_colors = [fam2color[families[labels.index(n)]] for n in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_size=220, node_color=node_colors)
    nx.draw_networkx_labels(G, pos, font_size=8)
    for f in fam2color:
        ax.scatter([], [], color=fam2color[f], label=f)
    ax.legend(fontsize=8, bbox_to_anchor=(1.02,1.0), loc='upper left')
    ax.set_title('kNN graph spring embedding (spectral init)')
    ax.axis('off')
    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close(fig)

# -----------------------------
# Runner: seleccionar opci\u00f3n o ejecutar todas
# -----------------------------

def run_option(opt, M, labels, families, outdir):
    if opt == 1:
        map_force_threshold(M, labels, families, os.path.join(outdir, 'map_01_force_threshold.png'))
    elif opt == 2:
        map_force_family_anchored(M, labels, families, os.path.join(outdir, 'map_02_force_family_anchored.png'))
    elif opt == 3:
        map_umap_affinity(M, labels, families, os.path.join(outdir, 'map_03_umap_affinity.png'))
    elif opt == 4:
        map_tsne_profiles(M, labels, families, os.path.join(outdir, 'map_04_tsne_profiles.png'))
    elif opt == 5:
        map_isomap(M, labels, families, os.path.join(outdir, 'map_05_isomap.png'))
    elif opt == 6:
        map_nonmetric_mds(M, labels, families, os.path.join(outdir, 'map_06_nonmetric_mds.png'))
    elif opt == 7:
        map_diffusion(M, labels, families, os.path.join(outdir, 'map_07_diffusion.png'))
    elif opt == 8:
        map_hierarchical_radial(M, labels, families, os.path.join(outdir, 'map_08_radial_dendrogram.png'))
    elif opt == 9:
        map_family_centroids(M, labels, families, os.path.join(outdir, 'map_09_family_centroids.png'))
    elif opt == 10:
        map_knn_graph_spring(M, labels, families, os.path.join(outdir, 'map_10_knn_spring.png'))
    else:
        raise ValueError('Opcion desconocida')


def main():
    parser = argparse.ArgumentParser(description='Generar varios mapas 2D alternativos desde matriz de transferencia')
    parser.add_argument('--matrix-csv', required=True)
    parser.add_argument('--abbrev-map', default=None)
    parser.add_argument('--outdir', default='./transfer_maps_options')
    parser.add_argument('--option', default='all', help='1..10 o "all"')
    args = parser.parse_args()

    ensure_dir(args.outdir)
    df = pd.read_csv(args.matrix_csv, index_col=0)
    labels_abbr = list(df.index)
    M = df.values.astype(float)
    abbrev_map = load_abbrev_map(args.abbrev_map) if args.abbrev_map else None
    labels_full = map_abbrevs_to_names(labels_abbr, abbrev_map)
    families = assign_families(labels_full)

    if args.option == 'all':
        for i in range(1, 11):
            print(f'Generando mapa {i}...')
            run_option(i, M, labels_abbr, families, args.outdir)
    else:
        opt = int(args.option)
        print(f'Generando mapa {opt}...')
        run_option(opt, M, labels_abbr, families, args.outdir)

    print('Mapas generados en', args.outdir)

if __name__ == '__main__':
    main()
