#!/usr/bin/env python3
import pandas as pd
import numpy as np
from pathlib import Path

# Paths a los CSVs
dual_csv = Path("/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speaker_matrix/seeds/speaker_matrix_dual.csv")
single_csv = Path("/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speaker_matrix/seeds/speaker_matrix_single.csv")

# Idiomas a analizar
langs = {"it", "yue", "en", "cy", "ckb", "lv", "pl", "bn", "fa"}

# Leer CSVs
df_dual = pd.read_csv(dual_csv)
df_single = pd.read_csv(single_csv)

# Filtrar solo los idiomas deseados
df_dual = df_dual[df_dual['model_id'].isin(langs)]
df_single = df_single[df_single['model_id'].isin(langs)]

# Iterar por idioma
results = []

for lang in langs:
    dual_vals = df_dual[df_dual['model_id'] == lang]['auc'].values
    single_vals = df_single[df_single['model_id'] == lang]['auc'].values

    # Inicialmente calcular la diferencia de medias
    mean_dual = np.mean(dual_vals)
    mean_single = np.mean(single_vals)
    diff = mean_dual - mean_single

    # Preparar lista para iteraciones eliminando extremos
    dual_sorted = np.sort(dual_vals)
    single_sorted = np.sort(single_vals)

    iteration = 0
    diffs_iter = [(iteration, mean_dual, mean_single, diff)]

    # Eliminamos máximo de single y mínimo de dual sucesivamente hasta que queden al menos 1 valor
    while len(single_sorted) > 1 and len(dual_sorted) > 1:
        iteration += 1
        # eliminar extremos
        single_sorted = single_sorted[:-1]  # eliminar el mayor de single
        dual_sorted = dual_sorted[1:]       # eliminar el menor de dual

        mean_dual_iter = np.mean(dual_sorted)
        mean_single_iter = np.mean(single_sorted)
        diff_iter = mean_dual_iter - mean_single_iter
        diffs_iter.append((iteration, mean_dual_iter, mean_single_iter, diff_iter))

    results.append((lang, diffs_iter))

# Mostrar resultados
for lang, diffs_iter in results:
    print(f"\nLanguage: {lang}")
    print("Iter | mean_dual | mean_single | diff")
    for it, m_dual, m_single, diff in diffs_iter:
        print(f"{it:>4} | {m_dual:>9.6f} | {m_single:>11.6f} | {diff:>7.6f}")
