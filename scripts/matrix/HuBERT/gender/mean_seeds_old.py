#!/usr/bin/env python3
# aggregate_gender_matrix_with_var.py
"""
Agrupa gender_matrix_single.csv por model_id,type,lang_src,lang_tgt,
calcula la media de f1, cuenta el número de seeds y calcula la varianza (ddof=1).
Salida con columnas:
model_id,type,lang_src,lang_tgt,num_seeds,f1,f1_var
"""

import pandas as pd
from pathlib import Path
import numpy as np

# Rutas (cambia si hace falta)
INPUT_CSV = Path("/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/gender_matrix.csv")
OUTPUT_CSV = Path("/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/gender_matrix/gender_matrix_test_agg.csv")

# Leer CSV
df = pd.read_csv(INPUT_CSV)

# Comprobar columnas necesarias
required = {"model_id", "type", "lang_src", "lang_tgt", "seed", "f1"}
missing = required - set(df.columns)
if missing:
    raise SystemExit(f"Faltan columnas en el CSV de entrada: {missing}")

# Columnas de agrupación
group_cols = ["model_id", "type", "lang_src", "lang_tgt"]

# Agrupar y calcular estadísticas:
# - count() -> número de seeds
# - mean()  -> media de f1
# - var(ddof=1) -> varianza muestral de f1 (NaN si count < 2)
agg_df = (
    df.groupby(group_cols, sort=False, dropna=False)["f1"]
      .agg(num_seeds="count", f1="mean", f1_var=lambda x: x.var(ddof=1))
      .reset_index()
)

# Asegurar tipos numéricos
agg_df["num_seeds"] = agg_df["num_seeds"].astype(int)
agg_df["f1"] = agg_df["f1"].astype(float)
# f1_var puede contener NaN; mantener float

# Reordenar columnas (por claridad)
cols_order = ["model_id", "type", "lang_src", "lang_tgt", "num_seeds", "f1", "f1_var"]
agg_df = agg_df[cols_order]

# Guardar CSV (formato float con 12 decimales para f1 y var)
agg_df.to_csv(OUTPUT_CSV, index=False, float_format="%.12f")

print(f"Wrote {len(agg_df)} grouped rows to {OUTPUT_CSV}")
