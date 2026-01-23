import os
import glob
import pandas as pd
import numpy as np

# Carpeta donde están los CSV
csv_dir = "/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/subsets_csv/gender"

# Umbral de error estándar deseado
epsilon = 0.1  # puedes cambiarlo según tu criterio

# Buscar todos los ficheros *_f1_by_samples.csv
csv_files = glob.glob(os.path.join(csv_dir, "*.f1_by_samples.csv"))

results = []

for f in csv_files:
    df = pd.read_csv(f)
    # Ignorar filas sin std (por si acaso)
    stds = df['f1_std'].dropna()
    if len(stds) == 0:
        continue
    # Tomamos la std máxima como la más conservadora
    max_std = stds.max()
    # Calcular número de seeds recomendado
    seeds_needed = int(np.ceil((max_std / epsilon) ** 2))
    results.append({
        'language': os.path.basename(f).split(".")[0],
        'max_std': max_std,
        'recommended_seeds': seeds_needed
    })

# Mostrar resultados ordenados
results_sorted = sorted(results, key=lambda x: x['recommended_seeds'], reverse=True)
for r in results_sorted:
    print(f"{r['language']:>6} | max_std={r['max_std']:.4f} | recommended_seeds={r['recommended_seeds']}")
