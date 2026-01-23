#!/usr/bin/env python3
# reconstruct_csv_from_logs.py
import pandas as pd
import re
from pathlib import Path
import ast

# Carpeta de logs
LOG_DIR = Path("/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/logs/speaker")

# Patrón regex para extraer ruta CSV y diccionario
pattern = re.compile(r"\[SV\] Appended summary to (.*?): (.*)")

# Diccionarios para agrupar por CSV
csv_rows = {}

# Recorrer logs
for log_file in LOG_DIR.glob("*.log"):
    with open(log_file, "r") as f:
        for line in f:
            match = pattern.search(line)
            if match:
                csv_path_str, dict_str = match.groups()
                csv_path = Path(csv_path_str.strip())
                # Parsear el dict literal seguro con ast.literal_eval
                row_dict = ast.literal_eval(dict_str.strip())
                # Añadir a la lista del CSV correspondiente
                csv_rows.setdefault(csv_path, []).append(row_dict)

# Guardar cada CSV
for csv_path, rows in csv_rows.items():
    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False, float_format="%.12f")
    print(f"Reconstructed {len(df)} rows into {csv_path}")
