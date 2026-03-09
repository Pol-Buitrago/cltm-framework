#!/usr/bin/env python3
# aggregate_and_convert_single_to_dual.py
"""
Lee gender_matrix_single.csv, transforma filas single -> dual (lang_tgt = lang_src,
model_id X -> X_X si procede), agrupa por model_id,type,lang_src,lang_tgt,
y calcula num_seeds, media f1 y varianza muestral f1_var.
Salida: gender_matrix_dual_agg.csv
"""

import pandas as pd
from pathlib import Path
import numpy as np

# --- Configurable: rutas de entrada/salida ---
INPUT_CSV = Path("/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/gender_matrix/seeds/gender_matrix_dual.csv")
OUTPUT_CSV = Path("/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/gender_matrix/gender_matrix_dual_agg.csv")
# ------------------------------------------------

# Leer CSV
df = pd.read_csv(INPUT_CSV, dtype={"model_id": str, "type": str, "lang_src": str, "lang_tgt": str, "seed": object, "f1": float})

# Comprobación mínima
required = {"model_id", "type", "lang_src", "lang_tgt", "seed", "f1"}
missing = required - set(df.columns)
if missing:
    raise SystemExit(f"Faltan columnas en el CSV de entrada: {missing}")

# Normalizar nulos en lang_tgt a cadena vacía (para facilidad)
df["lang_tgt"] = df["lang_tgt"].fillna("")

# Trabajar sobre copia para transformar
df_mod = df.copy()

# Función para transformar model_id según la regla: si no contiene '_' entonces X -> X_X
def expand_model_id(mid: str) -> str:
    if mid is None:
        return mid
    mid = str(mid)
    if "_" in mid:
        return mid
    # si es '*' también lo convertimos a '*_*' según pedido
    return f"{mid}_{mid}"

# Aplicar transformaciones solo a filas con type == 'single'
mask_single = df_mod["type"].astype(str) == "single"

# 1) cambiar type -> 'dual'
df_mod.loc[mask_single, "type"] = "dual"

# 2) poner lang_tgt = lang_src donde antes estaba vacío o en cualquier caso (según pedido)
df_mod.loc[mask_single, "lang_tgt"] = df_mod.loc[mask_single, "lang_src"].astype(str)

# 3) convertir model_id X -> X_X para estos casos (y '*'->'*_*')
df_mod.loc[mask_single, "model_id"] = df_mod.loc[mask_single, "model_id"].astype(str).apply(expand_model_id)

# Ahora agrupamos por las columnas transformadas
group_cols = ["model_id", "type", "lang_src", "lang_tgt"]

agg_df = (
    df_mod.groupby(group_cols, sort=False, dropna=False)["f1"]
          .agg(num_seeds="count", f1="mean", f1_var=lambda x: x.var(ddof=1))
          .reset_index()
)

# Asegurar tipos
agg_df["num_seeds"] = agg_df["num_seeds"].astype(int)
agg_df["f1"] = agg_df["f1"].astype(float)
# f1_var puede ser NaN si num_seeds < 2

# Reordenar columnas como pediste: model_id,type,lang_src,lang_tgt,num_seeds,f1,f1_var
cols_order = ["model_id", "type", "lang_src", "lang_tgt", "num_seeds", "f1", "f1_var"]
agg_df = agg_df[cols_order]

# Guardar CSV
agg_df.to_csv(OUTPUT_CSV, index=False, float_format="%.12f")

print(f"Wrote {len(agg_df)} rows to {OUTPUT_CSV}")
