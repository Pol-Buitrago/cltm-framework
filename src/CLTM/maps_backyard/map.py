import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import MDS, TSNE
import umap
import networkx as nx
import os
from matplotlib.colors import to_hex

# -------------------------------
# Parámetros
# -------------------------------
ruta_csv = "/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/CLTM/transfer_matrix/transfer_matrices/speaker/Normalized_Transfer_Matrix_M_norm.csv"
output_dir = "./maps_output"
top_n = 3  # para grafo top-n
os.makedirs(output_dir, exist_ok=True)

# -------------------------------
# Colores y familias
# -------------------------------
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

# Funciones de colores por familia
def assign_families(labels, name2family):
    return [name2family.get(lbl, 'Unknown') for lbl in labels]

def assign_family_colors(families):
    unique_fams = sorted(set(families))
    cmaps = [plt.get_cmap(c) for c in ['tab20', 'tab20b', 'tab20c']]
    colors = []
    for cmap in cmaps:
        colors.extend([to_hex(cmap(i)) for i in range(cmap.N)])
    colors = colors[:len(unique_fams)]
    fam2color = {f: colors[i] for i, f in enumerate(unique_fams)}
    node_colors = [fam2color[f] for f in families]
    return fam2color, node_colors

# -------------------------------
# Cargar CSV
# -------------------------------
df = pd.read_csv(ruta_csv, index_col=0)
labels = df.index.tolist()
M = df.values.astype(float)
families = assign_families(labels, NAME2FAMILY)
fam2color, node_colors = assign_family_colors(families)

# -------------------------------
# 1. MDS
# -------------------------------
dist_matrix = 1 - (M + M.T)/2
mds = MDS(n_components=2, dissimilarity='precomputed', metric_mds=True, n_init=4, max_iter=300, random_state=42, init='random')
coords_mds = mds.fit_transform(dist_matrix)

plt.figure(figsize=(12,10))
plt.scatter(coords_mds[:,0], coords_mds[:,1], c=node_colors)
for i, lbl in enumerate(labels):
    plt.annotate(lbl, (coords_mds[i,0], coords_mds[i,1]))
plt.title("MDS - Idiomas")
plt.savefig(f"{output_dir}/MDS_map.png")
plt.close()

# -------------------------------
# 2. t-SNE
# -------------------------------
tsne = TSNE(n_components=2, random_state=42, perplexity=10, init='random')
coords_tsne = tsne.fit_transform(M)

plt.figure(figsize=(12,10))
plt.scatter(coords_tsne[:,0], coords_tsne[:,1], c=node_colors)
for i, lbl in enumerate(labels):
    plt.annotate(lbl, (coords_tsne[i,0], coords_tsne[i,1]))
plt.title("t-SNE - Idiomas")
plt.savefig(f"{output_dir}/tSNE_map.png")
plt.close()

# -------------------------------
# 3. UMAP
# -------------------------------
reducer = umap.UMAP(metric='euclidean', random_state=42)
coords_umap = reducer.fit_transform(M)

plt.figure(figsize=(12,10))
plt.scatter(coords_umap[:,0], coords_umap[:,1], c=node_colors)
for i, lbl in enumerate(labels):
    plt.annotate(lbl, (coords_umap[i,0], coords_umap[i,1]))
plt.title("UMAP - Idiomas")
plt.savefig(f"{output_dir}/UMAP_map.png")
plt.close()

# -------------------------------
# 4. Grafo top-n
# -------------------------------
top_donors = {}
for i, row in enumerate(M):
    top_indices = row.argsort()[-top_n-1:][::-1]
    top_indices = [idx for idx in top_indices if idx != i]
    top_donors[labels[i]] = [labels[idx] for idx in top_indices[:top_n]]

G = nx.DiGraph()
for src, dsts in top_donors.items():
    for dst in dsts:
        G.add_edge(src, dst)

plt.figure(figsize=(14,12))
pos = nx.spring_layout(G, seed=42)
nx.draw(G, pos, with_labels=True, node_color=node_colors, edge_color="gray", node_size=1200)
plt.title(f"Grafo top-{top_n}")
plt.savefig(f"{output_dir}/graph_top{top_n}.png")
plt.close()

print("Mapas generados en:", output_dir)
