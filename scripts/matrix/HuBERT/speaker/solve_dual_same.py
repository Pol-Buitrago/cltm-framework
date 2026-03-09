#!/usr/bin/env python3
# aggregate_and_convert_single_to_dual_three_metrics.py
"""
Lee un CSV con columnas que incluyen model_id,type,lang_src,lang_tgt,seed,eer,auc,threshold.
Transforma filas type == 'single' a 'dual' poniendo lang_tgt = lang_src y model_id X -> X_X
(also '*' -> '*_*' si procede). Después agrupa por model_id,type,lang_src,lang_tgt y calcula:
num_seeds, media y varianza muestral (ddof=1) para eer, auc y threshold.

Salida:
gender_matrix_dual_agg_three_metrics.csv
"""

import pandas as pd
from pathlib import Path
import numpy as np

# --- Configurable: rutas de entrada/salida ---
INPUT_CSV = Path("/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/speaker_matrix/seeds/speaker_matrix_dual.csv")
OUTPUT_CSV = Path("/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/speaker_matrix/speaker_matrix_dual.csv")
# ------------------------------------------------

# Leer CSV forzando tipos basicos para evitar sorpresas
df = pd.read_csv(INPUT_CSV, dtype={"model_id": str, "type": str, "lang_src": str, "lang_tgt": str, "seed": object})

# Comprobación mínima
required = {"model_id", "type", "lang_src", "lang_tgt", "seed", "eer", "auc", "threshold"}
missing = required - set(df.columns)
if missing:
    raise SystemExit(f"Faltan columnas en el CSV de entrada: {missing}")

# Normalizar nulos en lang_tgt a cadena vacía
df["lang_tgt"] = df["lang_tgt"].fillna("")

# Convertir métricas a numérico
for col in ["eer", "auc", "threshold"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Copia para transformar filas single -> dual
df_mod = df.copy()

def expand_model_id(mid: str) -> str:
    if mid is None:
        return mid
    mid = str(mid)
    if "_" in mid:
        return mid
    return f"{mid}_{mid}"

# Mascara de filas single (case-insensitive por si acaso)
mask_single = df_mod["type"].astype(str).str.lower() == "single"

# 1) cambiar type -> 'dual'
df_mod.loc[mask_single, "type"] = "dual"

# 2) lang_tgt = lang_src para esas filas
df_mod.loc[mask_single, "lang_tgt"] = df_mod.loc[mask_single, "lang_src"].astype(str)

# 3) convertir model_id X -> X_X para esas filas
df_mod.loc[mask_single, "model_id"] = df_mod.loc[mask_single, "model_id"].astype(str).apply(expand_model_id)

# Columnas de agrupación
group_cols = ["model_id", "type", "lang_src", "lang_tgt"]

# Función auxiliar para varianza muestral
def var_ddof1(x):
    return x.var(ddof=1)

# Agrupar y agregar las tres métricas
agg_df = (
    df_mod.groupby(group_cols, sort=False, dropna=False)
          .agg(
              num_seeds = ("seed", "count"),
              eer       = ("eer", "mean"),
              eer_var   = ("eer", var_ddof1),
              auc       = ("auc", "mean"),
              auc_var   = ("auc", var_ddof1),
              threshold = ("threshold", "mean"),
              threshold_var = ("threshold", var_ddof1),
          )
          .reset_index()
)

# Forzar tipos
agg_df["num_seeds"] = agg_df["num_seeds"].astype(int)
for col in ["eer", "eer_var", "auc", "auc_var", "threshold", "threshold_var"]:
    agg_df[col] = agg_df[col].astype(float)

# Reordenar columnas
cols_order = ["model_id", "type", "lang_src", "lang_tgt", "num_seeds",
              "eer", "eer_var", "auc", "auc_var", "threshold", "threshold_var"]
agg_df = agg_df[cols_order]

# Guardar CSV
agg_df.to_csv(OUTPUT_CSV, index=False, float_format="%.12f")

print(f"Wrote {len(agg_df)} rows to {OUTPUT_CSV}")
