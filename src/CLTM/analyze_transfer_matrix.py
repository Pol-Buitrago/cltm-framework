#!/usr/bin/env python3
"""
analyze_transfer_matrix.py

Rewritten to increase label and annotation sizes, reduce excessive horizontal whitespace,
and make figures more compact and report-ready.

Key changes:
 - larger default font sizes via matplotlib rcParams
 - reduced width scaling for heatmap, tighter colorbar placement
 - rotated x tick labels and reduced figure widths to avoid large empty margins
 - compact directed graph layout and tightened bbox when saving
 - explicit removal of self-loops in graph construction
 - annotations in t-SNE use small offsets to reduce overlap and improve readability
 - consistent use of tight_layout and bbox_inches='tight' to avoid white margins

Dependencies: numpy, pandas, matplotlib, networkx, sklearn, scipy, python-louvain optional
"""

import argparse
import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.linalg import norm

try:
    import community as community_louvain
    _LOUVAIN_AVAILABLE = True
except Exception:
    _LOUVAIN_AVAILABLE = False


# -------------------------------
# Global plotting defaults
# -------------------------------
# Increase base font sizes and reduce default figure padding to avoid tiny text and large white space
plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'figure.constrained_layout.use': False,
})


# Optional built-in NAME2FAMILY mapping
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


# -------------------------------
# Utility functions
# -------------------------------

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def load_abbrev_map(path):
    if path is None:
        return None
    with open(path, 'r', encoding='utf-8') as f:
        txt = f.read()
    try:
        return json.loads(txt)
    except Exception:
        return eval(txt, {"__builtins__": {}}, {})


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


def assign_families(full_names, name2family):
    families = []
    for nm in full_names:
        if nm is None:
            families.append('Unknown')
        else:
            families.append(name2family.get(nm, 'Unknown'))
    return families


def assign_family_colors(df_meta):
    families = sorted(df_meta['family'].unique(), key=lambda x: (x != 'Unknown', x))
    n = len(families)
    cmaps = [plt.get_cmap(c) for c in ['tab20', 'tab20b', 'tab20c']]
    colors = []
    for cmap in cmaps:
        colors.extend([to_hex(cmap(i)) for i in range(cmap.N)])
    colors = colors[:n]
    fam2color = {fam: colors[i] for i, fam in enumerate(families)}
    row_colors = [fam2color[df_meta.loc[abbr, 'family']] for abbr in df_meta.index]
    col_colors = row_colors.copy()
    return fam2color, row_colors, col_colors


# -------------------------------
# Plotting routines
# -------------------------------

def plot_heatmap(M, labels, outpath, vmin=None, vmax=None, cmap='RdBu_r'):
    ensure_dir(os.path.dirname(outpath) or '.')
    n = M.shape[0]
    # reduce width scaling to avoid very wide figures with a lot of empty space
    width = max(6, min(18, n * 0.12))
    height = max(6, min(14, n * 0.10))
    fig, ax = plt.subplots(figsize=(width, height))

    # choose aspect: keep cells roughly square when possible
    aspect = 'auto'
    im = ax.imshow(M, aspect=aspect, interpolation='nearest', cmap=cmap, vmin=vmin, vmax=vmax)

    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))

    # rotate and reduce label size to avoid overlap
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=10)
    ax.set_yticklabels(labels, fontsize=10)

    ax.tick_params(axis='x', which='both', length=0)
    ax.tick_params(axis='y', which='both', length=0)

    # use a horizontal colorbar at the bottom to reduce vertical whitespace
    cbar = fig.colorbar(im, ax=ax, orientation='horizontal', fraction=0.06, pad=0.09)
    cbar.ax.tick_params(labelsize=10)
    cbar.set_label('Normalized transfer (fraction of self improvement)', fontsize=11)

    # tighten layout further and save with tight bbox
    fig.tight_layout(pad=1.0)
    fig.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_bar_strengths(str_in, str_out, labels, outpath):
    ensure_dir(os.path.dirname(outpath) or '.')
    n = len(labels)
    width_fig = max(8, min(18, n * 0.12))
    fig, ax = plt.subplots(figsize=(width_fig, 6))
    ind = np.arange(n)
    width = 0.35
    ax.bar(ind - width / 2, str_in, width, label='incoming strength')
    ax.bar(ind + width / 2, str_out, width, label='outgoing strength')

    ax.set_xticks(ind)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=10)
    ax.set_ylabel('Sum positive normalized transfer', fontsize=11)
    ax.set_title('Incoming and outgoing positive transfer strengths')
    ax.legend()
    fig.tight_layout(pad=1.0)
    fig.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close(fig)


def build_graph(M, labels, family_labels=None, pos_threshold=0.05):
    G = nx.DiGraph()
    n = M.shape[0]
    for i, lbl in enumerate(labels):
        G.add_node(lbl)
        if family_labels is not None:
            G.nodes[lbl]['family'] = family_labels[i]
    for i in range(n):
        for j in range(n):
            # skip self-edges explicitly
            if i == j:
                continue
            w = float(M[i, j])
            if abs(w) >= pos_threshold:
                # edge from source j to target i, consistent with previous code
                G.add_edge(labels[j], labels[i], weight=w)
    # safety: remove any self-loops that might remain
    G.remove_edges_from(nx.selfloop_edges(G))
    return G


def plot_directed_graph(G, outpath, max_edges=300, node_scale=900, edge_alpha_pos=0.7, edge_alpha_neg=0.25):
    ensure_dir(os.path.dirname(outpath) or '.')
    try:
        df_meta = pd.DataFrame({n: G.nodes[n] for n in G.nodes()}).T
    except Exception:
        df_meta = None

    families = [G.nodes[n].get('family', 'Unknown') for n in G.nodes()]

    if df_meta is not None and 'family' in df_meta.columns:
        fam2color, row_colors, col_colors = assign_family_colors(df_meta)
        node_colors = [fam2color.get(G.nodes[n].get('family', 'Unknown'), '#777777') for n in G.nodes()]
    else:
        cmap = plt.get_cmap('tab20')
        node_colors = [cmap(i % 20) for i in range(len(G.nodes()))]

    in_strength = defaultdict(float)
    for u, v, d in G.edges(data=True):
        w = d.get('weight', 0.0)
        if w > 0:
            in_strength[v] += w

    base = 160
    vals = np.array([in_strength.get(n, 0.0) for n in G.nodes()])
    if vals.max() > 0:
        scaled = base + (vals - vals.min()) / (vals.max() - vals.min() + 1e-12) * (node_scale - base)
    else:
        scaled = np.array([base] * len(vals))
    node_sizes = [float(s) for s in scaled]

    # use spring layout with a modest k to reduce overlap, then rescale bounding box tightly
    try:
        pos = nx.spring_layout(G, k=0.6, seed=42, iterations=300)
    except Exception:
        pos = nx.kamada_kawai_layout(G)

    edges = sorted(G.edges(data=True), key=lambda e: abs(e[2].get('weight', 0.0)), reverse=True)
    if len(edges) > max_edges:
        edges = edges[:max_edges]

    fig, ax = plt.subplots(figsize=(12, 9))
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, ax=ax)

    # draw edges with proportional widths
    for (u, v, d) in edges:
        w = d.get('weight', 0.0)
        lw = max(0.3, min(6.0, abs(w) * 4.0))
        alpha = edge_alpha_pos if w > 0 else edge_alpha_neg
        color = '#2b83ba' if w > 0 else '#d7191c'
        nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], width=lw, alpha=alpha, edge_color=color, arrowsize=8, ax=ax)

    nx.draw_networkx_labels(G, pos, font_size=10, ax=ax)

    if df_meta is not None and 'family' in df_meta.columns:
        for f in sorted(df_meta['family'].unique(), key=lambda x: (x != 'Unknown', x)):
            ax.scatter([], [], color=fam2color[f], label=f)
        ax.legend(scatterpoints=1, fontsize=9, loc='upper left', bbox_to_anchor=(1.02, 1.0))

    ax.set_title('Directed transfer graph, positive and negative edges shown')
    ax.axis('off')

    # tighten and save with tight bbox to remove empty margins
    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close(fig)


def compute_tsne(M, labels, families, outpath, fam2color=None, perplexity=10, random_state=42, marker_size=120):
    ensure_dir(os.path.dirname(outpath) or '.')
    pca = PCA(n_components=min(30, max(2, M.shape[1] - 1)), random_state=random_state)
    Xp = pca.fit_transform(M)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(Xp)

    tsne = TSNE(n_components=2, perplexity=perplexity, init='pca', random_state=random_state)
    Y = tsne.fit_transform(Xs)

    unique_fams = sorted(set(families))
    if fam2color is None:
        cmap = plt.get_cmap('tab20')
        fam2color = {f: cmap(i % 20) for i, f in enumerate(unique_fams)}

    colors = [fam2color.get(f, '#777777') for f in families]

    fig, ax = plt.subplots(figsize=(11, 8))
    ax.scatter(Y[:, 0], Y[:, 1], s=marker_size, c=colors)

    for i, lbl in enumerate(labels):
        # annotate with a small offset to reduce overlaps
        ax.annotate(lbl, (Y[i, 0], Y[i, 1]), fontsize=10, alpha=0.95, xytext=(4, 3), textcoords='offset points')

    for f in unique_fams:
        ax.scatter([], [], color=fam2color.get(f, '#777777'), label=f)
    ax.legend(fontsize=9, bbox_to_anchor=(1.02, 1.0), loc='upper left')
    ax.set_title('t-SNE of language transfer profiles')
    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close(fig)


# -------------------------------
# Main routine
# -------------------------------

def main():
    parser = argparse.ArgumentParser(description='Analyze and visualize a normalized cross-lingual transfer matrix')
    parser.add_argument('--matrix-csv', type=str, required=True, help='Path to Normalized_Transfer_Matrix_M_norm.csv')
    parser.add_argument('--abbrev-map', type=str, default=None, help='Path to abbreviation map JSON (abbr -> full name)')
    parser.add_argument('--output-dir', type=str, default='./figures_transfer', help='Directory to save figures and metrics')
    parser.add_argument('--edge-thresh', type=float, default=0.05, help='Minimum absolute edge weight to draw in graph')
    parser.add_argument('--tsne-perplexity', type=float, default=10.0, help='Perplexity for t-SNE')
    parser.add_argument('--tsne-marker-size', type=float, default=120.0, help='Marker size for t-SNE scatter')
    parser.add_argument('--graph-node-scale', type=float, default=900.0, help='Max node scale for graph node sizing')
    parser.add_argument('--edge-alpha-pos', type=float, default=0.7, help='Alpha for positive edges')
    parser.add_argument('--edge-alpha-neg', type=float, default=0.25, help='Alpha for negative edges')
    args = parser.parse_args()

    ensure_dir(args.output_dir)

    df = pd.read_csv(args.matrix_csv, index_col=0)
    labels_abbr = list(df.index)
    M = df.values.astype(float)

    abbrev_map = load_abbrev_map(args.abbrev_map) if args.abbrev_map is not None else None
    labels_full = map_abbrevs_to_names(labels_abbr, abbrev_map)
    families = assign_families(labels_full, NAME2FAMILY)

    diag = np.diag(M)
    frob_dist_to_ones = norm(M - np.ones_like(M), ord='fro')
    summary_lines = [
        f'Loaded matrix: {args.matrix_csv}',
        f'Number of languages: {M.shape[0]}',
        f'Frobenius distance to ones matrix: {frob_dist_to_ones:.4f}',
        f'Diagonal statistics: mean={diag.mean():.4f}, min={diag.min():.4f}, max={diag.max():.4f}',
    ]

    pos_M = np.maximum(M, 0.0)
    incoming = pos_M.sum(axis=1)
    outgoing = pos_M.sum(axis=0)

    sym_pos = (pos_M + pos_M.T) / 2.0
    Gsym = nx.from_numpy_array(sym_pos)
    try:
        pagerank_vals = nx.pagerank_numpy(nx.relabel_nodes(Gsym, dict(enumerate(labels_abbr))), alpha=0.85)
        pagerank_list = [pagerank_vals.get(lbl, 0.0) for lbl in labels_abbr]
    except Exception:
        pagerank_list = [0.0] * len(labels_abbr)

    if _LOUVAIN_AVAILABLE:
        G_louv = nx.from_numpy_array(sym_pos)
        mapping = dict(enumerate(labels_abbr))
        G_louv = nx.relabel_nodes(G_louv, mapping)
        partition = community_louvain.best_partition(G_louv)
        communities = [partition.get(lbl, -1) for lbl in labels_abbr]
    else:
        communities = [-1] * len(labels_abbr)

    metrics_df = pd.DataFrame({
        'abbrev': labels_abbr,
        'full_name': labels_full,
        'family': families,
        'incoming_pos_strength': incoming,
        'outgoing_pos_strength': outgoing,
        'pagerank_sympos': pagerank_list,
        'community_louvain': communities,
    })
    metrics_df.to_csv(os.path.join(args.output_dir, 'transfer_metrics.csv'), index=False)

    with open(os.path.join(args.output_dir, 'summary.txt'), 'w', encoding='utf-8') as f:
        for line in summary_lines:
            f.write(line + '')

    vmin = np.percentile(M, 1)
    vmax = np.percentile(M, 99)
    plot_heatmap(M, labels_abbr, os.path.join(args.output_dir, 'heatmap_matrix.png'), vmin=vmin, vmax=vmax)
    plot_heatmap(M, labels_abbr, os.path.join(args.output_dir, 'heatmap_matrix.svg'), vmin=vmin, vmax=vmax)

    plot_bar_strengths(incoming, outgoing, labels_abbr, os.path.join(args.output_dir, 'strengths_bar.png'))

    G = build_graph(M, labels_abbr, family_labels=families, pos_threshold=args.edge_thresh)
    plot_directed_graph(G, os.path.join(args.output_dir, 'directed_graph.png'), max_edges=300, node_scale=args.graph_node_scale, edge_alpha_pos=args.edge_alpha_pos, edge_alpha_neg=args.edge_alpha_neg)

    df_meta_indexed = metrics_df.set_index('abbrev')
    fam2color, row_colors, col_colors = assign_family_colors(df_meta_indexed)
    compute_tsne(M, labels_abbr, families, os.path.join(args.output_dir, 'tsne_transfer.png'), fam2color=fam2color, perplexity=args.tsne_perplexity, marker_size=args.tsne_marker_size)

    print('Analysis complete. Outputs saved to', args.output_dir)


if __name__ == '__main__':
    main()
