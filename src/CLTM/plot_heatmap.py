#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Funciones de visualización para la matriz de transferencia agrupada por familia.

Diseño:
 - Filas: recuadro con abreviatura coloreado a la izquierda.
 - Columnas: fullname arriba en diagonal; recuadro de color encima con la abreviatura dentro.

Añadido: opción para dibujar separadores entre familias.
"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import to_hex
from matplotlib.patches import Patch, Rectangle


def assign_family_colors(df_meta):
    """
    Asigna colores a familias y devuelve mapping + listas de colores para filas/columnas.
    """
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


def _text_contrast_color(hex_color):
    """
    Devuelve 'black' o 'white' según contraste relativo del color (legibilidad).
    """
    rgb = mcolors.to_rgb(hex_color)
    lum = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
    return 'black' if lum > 0.6 else 'white'


def split_first_space(s):
    """
    Inserta un salto de línea en el primer espacio del string (para etiquetas).
    """
    s = '' if s is None else str(s).strip()
    if ' ' in s:
        left, right = s.split(' ', 1)
        return f"{left}\n{right}"
    return s


def plot_heatmap_by_family(M, M_norm, df_meta, out_path,
                           vmin=-1.5, vmax=1.5, figsize_per_item=0.35, task=None,
                           preserve_order=False, border_style='lines', border_width=1.2,
                           border_alpha=1.0):
    """
    Dibuja y guarda un heatmap de matrices M / M_norm ordenado por familia.
    - La barra de color queda sin texto, adyacente al heatmap y con la misma altura.
    - Entre casillas se dibuja un borde blanco muy fino.
    - Las abreviaturas de filas y columnas usan el mismo tamaño de fuente.

    Nuevas opciones:
    - border_style: 'lines' (líneas entre familias) o 'boxes' (rectángulos alrededor de cada bloque familiar) o None.
    - border_width: grosor de las líneas/rectángulos.
    - border_alpha: opacidad de las líneas/rectángulos.
    """
    # Parámetros de estilo reutilizables
    abbr_fontsize = 8        # tamaño de fuente para abreviaturas (filas y columnas)
    abbr_fontweight = 'bold' # peso de la fuente para abreviaturas
    annot_fontsize = 8       # tamaño de las anotaciones dentro de las casillas

    # Ordenar metadata por familia y fullname
    df_meta_sorted = df_meta.reset_index().sort_values(['family', 'fullname'], key=lambda col: col)
    df_meta_sorted = df_meta_sorted.set_index('abbr')
    if preserve_order:
        # df_meta debe tener index = 'abbr'; reindexamos según el orden de M.index
        # (mantiene solo los que estén presentes)
        df_meta_sorted = df_meta.loc[[a for a in M.index if a in df_meta.index]]
    else:
        # Orden por familia y fullname (comportamiento anterior)
        df_meta_sorted = df_meta.reset_index().sort_values(['family', 'fullname'], key=lambda col: col)
        df_meta_sorted = df_meta_sorted.set_index('abbr')

    abbrs = [a for a in df_meta_sorted.index if a in M.index]

    # Reindexar matrices según orden obtenido (si preserve_order=True, esto mantiene el orden)
    M = M.reindex(index=abbrs, columns=abbrs)
    M_norm = M_norm.reindex(index=abbrs, columns=abbrs)

    # Preparar etiquetas y familias
    fullnames = [split_first_space(df_meta_sorted.loc[a, 'fullname']) for a in abbrs]
    families = [df_meta_sorted.loc[a, 'family'] for a in abbrs]

    fam2color, row_colors, col_colors = assign_family_colors(df_meta_sorted)

    sns.set(style='white')
    n = len(abbrs)

    # Tamaño de figura (ajustable)
    fig = plt.figure(figsize=(max(8, n * 0.5 + 4), max(8, n * 0.35 + 4)))

    # Coordenadas del área principal: calibradas para permitir leyendas y colorbar pegado
    left = 0.16
    bottom = 0.12
    width = 0.70
    height = 0.70

    ax_main = fig.add_axes([left, bottom, width, height])

    # Colorbar: pegada al heatmap y misma altura (bottom y height iguales a ax_main)
    cbar_gap = 0.005
    cbar_x = left + width + cbar_gap
    cbar_y = bottom
    cbar_w = 0.01
    cbar_h = height
    ax_cbar = fig.add_axes([cbar_x, cbar_y, cbar_w, cbar_h])

    # Limpiar valores no numéricos y convertir a float
    M_norm_plot = M_norm.copy()
    M_norm_plot = M_norm_plot.mask(M_norm_plot.isin(['TBD', 'NA/0']), np.nan)
    M_norm_plot = M_norm_plot.astype(float)

    # Dibujar heatmap: anotar, usar cmap y pasar cbar_ax
    h = sns.heatmap(
        M_norm_plot,
        annot=True,
        fmt='.2f',
        cmap='coolwarm',
        linewidths=0.35,      # borde fino entre casillas
        linecolor='white',    # color del borde entre casillas
        vmin=vmin,
        vmax=vmax,
        cbar=True,
        ax=ax_main,
        cbar_ax=ax_cbar,
        xticklabels=False,
        yticklabels=False,
        annot_kws={"fontsize": annot_fontsize}
    )

    # -----------------------------
    # Dibujar separadores entre familias (opciones 'lines' o 'boxes')
    # -----------------------------
    if border_style in ('lines', 'boxes') and len(families) > 0:
        # Calcular índices donde cambia la familia
        boundaries = []
        cur = families[0]
        for idx, fam in enumerate(families):
            if fam != cur:
                boundaries.append(idx)
                cur = fam

        xmin = -1.0
        xmax = n
        ymin = -1.0
        ymax = n

        if border_style == 'lines':
            # Dibujar líneas horizontales y verticales en cada frontera
            for b in boundaries:
                ax_main.hlines(b, xmin, xmax, colors='k', linewidth=border_width, alpha=border_alpha, zorder=8, linestyles='--')
                ax_main.vlines(b, ymin, ymax, colors='k', linewidth=border_width, alpha=border_alpha, zorder=8, linestyles='--')

        else:  # 'boxes'
            # encontrar intervalos [start, end) por familia
            intervals = []
            start = 0
            cur = families[0]
            for idx, fam in enumerate(families[1:], start=1):
                if fam != cur:
                    intervals.append((start, idx))
                    start = idx
                    cur = fam
            intervals.append((start, len(families)))  # último bloque

            # Dibujar un rectángulo por bloque (sin fill)
            for (s, e) in intervals:
                # Si el bloque tiene longitud 0, saltarlo
                if e - s <= 0:
                    continue
                pad = 0.0
                rect = Rectangle((s - pad, s - pad), (e - s) + 0.0, (e - s) + 0.0,
                                 facecolor='none', edgecolor='k', linewidth=border_width,
                                 linestyle='--', alpha=border_alpha, zorder=9, transform=ax_main.transData, clip_on=False)
                ax_main.add_patch(rect)

    # Ajustes del colorbar: eliminar texto/label y armonizar tamaños
    try:
        cbar = h.collections[0].colorbar
        cbar.ax.set_ylabel('')
        cbar.ax.tick_params(
            labelsize=10,
            length=2,
            width=0.5,
            direction='out',
            pad=2
        )
        cbar.outline.set_linewidth(0.4)
    except Exception:
        pass

    # Título (incluye información de tarea si se proporciona)
    title_task = f" (Task: {task})" if task is not None else ""
    ax_main.set_title(f"Cross-Lingual Transfer Matrix (CLTM){title_task}", fontsize=14)

    # Etiquetas de ejes: eje X (label abajo), ticks arriba (las etiquetas de columna se dibujan arriba)
    ax_main.set_ylabel('Target Language ($L_i$)', fontsize=12)
    ax_main.xaxis.set_ticks_position('top')
    ax_main.xaxis.set_label_position('bottom')

    # Anotaciones para valores de tipo string (por ejemplo 'TBD')
    for i in range(M_norm.shape[0]):
        for j in range(M_norm.shape[1]):
            val = M_norm.iloc[i, j]
            if isinstance(val, str):
                ax_main.text(j + 0.5, i + 0.5, val, ha='center', va='center',
                             color='black', fontsize=annot_fontsize, zorder=10)

    # Parámetros de recuadros (abreviaturas) a la izquierda de la matriz
    box_w = 1.0
    box_h = 1.0
    x_left = -1.0
    for i, abbr in enumerate(abbrs):
        color = fam2color[families[i]]
        rect = Rectangle((x_left, i + 0.5 - box_h / 2), box_w, box_h,
                         transform=ax_main.transData, facecolor=color,
                         edgecolor='#444444', linewidth=0.4, zorder=5, clip_on=False)
        ax_main.add_patch(rect)
        text_color = _text_contrast_color(color)
        ax_main.text(x_left + box_w / 2, i + 0.5, abbr, ha='center', va='center',
                     fontsize=abbr_fontsize, color=text_color, fontweight=abbr_fontweight,
                     transform=ax_main.transData, zorder=6, clip_on=False)

    # Recuadros superiores con abreviaturas (ahora usan el mismo tamaño que las filas)
    box_h_col = 1.0
    y_top = -1.0
    for j, abbr in enumerate(abbrs):
        color = fam2color[families[j]]
        rect = Rectangle((j, y_top), box_w, box_h_col,
                         transform=ax_main.transData, facecolor=color,
                         edgecolor='#444444', linewidth=0.4, zorder=5, clip_on=False)
        ax_main.add_patch(rect)
        text_color = _text_contrast_color(color)
        ax_main.text(j + 0.5, y_top + box_h_col / 2, abbr, ha='center', va='center',
                     fontsize=abbr_fontsize, color=text_color, fontweight=abbr_fontweight,
                     transform=ax_main.transData, zorder=6, clip_on=False)

    # Etiquetas de columna (fullnames) arriba de la matriz, en diagonal
    ax_main.set_xticks(np.arange(n) + 0.5)
    ax_main.set_xticklabels(fullnames, rotation=45, ha='left', va='bottom',
                           rotation_mode='anchor', fontsize=9)

    # Reducir separación entre etiquetas X y matriz (ajustable)
    pad_labels = 8
    ax_main.tick_params(axis='x', which='major', pad=pad_labels)

    # Eliminar marcas de ticks manteniendo etiquetas
    ax_main.tick_params(axis='x', which='both', length=0)
    ax_main.tick_params(axis='y', which='both', length=0)

    # Leyenda de familias: si hay pocas familias, ajustar ncol
    legend_ncol = min(4, max(1, len(fam2color)))

    # Etiqueta X situada debajo de la matriz, con labelpad para separar de la leyenda
    ax_main.set_xlabel('Source Language ($L_j$)', fontsize=12)

    # Ajuste de límites con pequeño margen derecho para evitar recorte
    ax_main.set_xlim(-1.0, n + 0.6)
    ax_main.set_ylim(n, -1.0)

    # Leyenda de familias: desplazada hacia abajo para evitar solapamiento con label X
    patches = [Patch(facecolor=fam2color[f], edgecolor='#444444', label=f) for f in fam2color.keys()]
    ax_main.legend(
        handles=patches,
        loc='upper center',
        bbox_to_anchor=(0.5, -0.04),
        fontsize=10,
        ncol=legend_ncol,
        frameon=False
    )

    # Guardar la figura con tight bbox para respetar los ajustes realizados
    plt.savefig(out_path, bbox_inches='tight', dpi=200)
    plt.close()
    print(f"Heatmap guardado en: {out_path}")
