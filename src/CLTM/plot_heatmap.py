#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Funciones de ploteo para la matriz de transferencia agrupada por familia.
Visualización ajustada:
 - filas: nombre completo en ejes con recuadro de abreviatura coloreado
 - columnas: solo abreviatura, recuadro de color arriba
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap, to_hex
from matplotlib.patches import Patch, Rectangle

def assign_family_colors(df_meta):
    families = sorted(df_meta['family'].unique(), key=lambda x: (x != 'Unknown', x))
    n = len(families)
    
    # Combinar tab20, tab20b, tab20c
    cmaps = [plt.get_cmap(c) for c in ['tab20', 'tab20b', 'tab20c']]
    colors = []
    for cmap in cmaps:
        colors.extend([to_hex(cmap(i)) for i in range(cmap.N)])
    
    # Tomar solo los necesarios
    colors = colors[:n]
    
    fam2color = {fam: colors[i] for i, fam in enumerate(families)}
    row_colors = [fam2color[df_meta.loc[abbr, 'family']] for abbr in df_meta.index]
    col_colors = row_colors.copy()
    return fam2color, row_colors, col_colors


def _text_contrast_color(hex_color):
    rgb = mcolors.to_rgb(hex_color)
    lum = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
    return 'black' if lum > 0.6 else 'white'

def plot_heatmap_by_family(M, M_norm, df_meta, out_path,
                           vmin=-1.5, vmax=1.5, figsize_per_item=0.35):
    df_meta_sorted = df_meta.reset_index().sort_values(['family', 'fullname'], key=lambda col: col)
    df_meta_sorted = df_meta_sorted.set_index('abbr')
    abbrs = [a for a in df_meta_sorted.index if a in M.index]

    M = M.reindex(index=abbrs, columns=abbrs)
    M_norm = M_norm.reindex(index=abbrs, columns=abbrs)

    fullnames = [df_meta_sorted.loc[a, 'fullname'] for a in abbrs]
    families = [df_meta_sorted.loc[a, 'family'] for a in abbrs]

    fam2color, row_colors, col_colors = assign_family_colors(df_meta_sorted)

    sns.set(style='white')
    n = len(abbrs)
    fig = plt.figure(figsize=(max(8, n * 0.5 + 4), max(8, n * 0.35 + 4)))

    left = 0.16
    bottom = 0.12
    width = 0.70
    height = 0.70

    ax_main = fig.add_axes([left, bottom, width, height])
    ax_cbar = fig.add_axes([left + width + 0.01, bottom, 0.015, height])

    M_norm_plot = M_norm.copy()
    M_norm_plot = M_norm_plot.mask(M_norm_plot.isin(['TBD', 'NA/0']), np.nan)
    M_norm_plot = M_norm_plot.astype(float)

    sns.heatmap(
        M_norm_plot,
        annot=True,
        fmt='.2f',
        cmap='coolwarm',
        linewidths=.5,
        linecolor='black',
        vmin=vmin,
        vmax=vmax,
        cbar=True,
        ax=ax_main,
        cbar_ax=ax_cbar,
        xticklabels=False,
        yticklabels=fullnames,
        annot_kws={"fontsize": 8} 
    )

    ax_main.set_title('Normalized Transfer Matrix ($M_{norm}$)', fontsize=14)
    ax_main.set_xlabel('Source Language ($L_j$)', fontsize=12)
    ax_main.set_ylabel('Target Language ($L_i$)', fontsize=12)
    ax_main.set_yticklabels(ax_main.get_yticklabels(), rotation=0, fontsize=8)

    # TBD / NA/0
    for i in range(M_norm.shape[0]):
        for j in range(M_norm.shape[1]):
            val = M_norm.iloc[i, j]
            if isinstance(val, str):
                ax_main.text(j + 0.5, i + 0.5, val, ha='center', va='center', color='black', fontsize=8)

    # --- Filas: recuadro abreviatura (más ancho) ---
    box_w = 1  
    box_h = 1
    x_left = -1
    for i, abbr in enumerate(abbrs):
        color = fam2color[families[i]]
        rect = Rectangle((x_left, i + 0.5 - box_h / 2), box_w, box_h,
                         transform=ax_main.transData, facecolor=color, edgecolor='k', linewidth=0.4)
        ax_main.add_patch(rect)
        text_color = _text_contrast_color(color)
        ax_main.text(x_left + box_w / 2, i + 0.5, abbr, ha='center', va='center',
                     fontsize=8, color=text_color, fontweight='bold', transform=ax_main.transData)

    # --- Columnas: recuadro arriba con abreviatura ---
    box_h_col = 1
    y_top = -1  # colocar recuadros arriba de la matriz
    for j, abbr in enumerate(abbrs):
        color = fam2color[families[j]]
        rect = Rectangle((j + 0.5 - box_w/2, y_top), box_w, box_h_col,
                         transform=ax_main.transData, facecolor=color, edgecolor='k', linewidth=0.4)
        ax_main.add_patch(rect)
        text_color = _text_contrast_color(color)
        ax_main.text(j + 0.5, y_top + box_h_col / 2, abbr, ha='center', va='center',
                     fontsize=7, color=text_color, fontweight='bold', rotation=0, transform=ax_main.transData)

    ax_main.set_xlim(-1.2, n)
    ax_main.set_ylim(n, -1.5)  # extendido para recuadros arriba

    # Leyenda
    # Número máximo de elementos por fila
    max_per_row = 12
    ncol = min(max_per_row, len(fam2color))

    patches = [Patch(facecolor=fam2color[f], edgecolor='k', label=f) for f in fam2color.keys()]
    ax_main.legend(
        handles=patches,
        loc='upper center',
        bbox_to_anchor=(0.5, -0.02),  # ajustar verticalmente según tamaño de figura
        #title='Family',
        fontsize=9,
        ncol=ncol,
        frameon=False
    )

    plt.savefig(out_path, bbox_inches='tight', dpi=200)
    plt.close()
    print(f"Heatmap guardado en: {out_path}")
