#!/usr/bin/env python3
import os
import random
from itertools import combinations
from collections import defaultdict
import pandas as pd

# ---------- Config ----------
INPUT_DIR = "/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/03_combined_train/tsv"
OUTPUT_DIR = "/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/04_test_pairs/tsv"

# Parámetros de muestreo
SEED = 42
MAX_POS_PAIRS_PER_SPEAKER = 200   # máximo de pares positivos por hablante (si hay más, se samplea)
NEG_TO_POS_RATIO = 1.0            # número de negativos por positivo objetivo (1.0 => igual número)
MAX_NEG_SAMPLING_ATTEMPTS_FACTOR = 10  # tope de intentos = factor * objetivo_negativos

VALID_GENDERS = ["male_masculine", "female_feminine"]

# ---------- Setup ----------
os.makedirs(OUTPUT_DIR, exist_ok=True)
random.seed(SEED)

def sample_pairs_from_combinations(items, max_pairs):
    """Devuelve lista de pares (a,b) muestreados uniformemente entre todas las combinaciones de items."""
    if len(items) < 2:
        return []
    all_combos = list(combinations(items, 2))
    if len(all_combos) <= max_pairs:
        return all_combos
    return random.sample(all_combos, max_pairs)

# Procesar todos los .test.tsv
files = sorted(f for f in os.listdir(INPUT_DIR) if f.endswith(".test.tsv"))
total_files = len(files)
if total_files == 0:
    print("No se encontraron archivos .test.tsv en", INPUT_DIR)

for idx, fname in enumerate(files, 1):
    lang = fname.split(".")[0]
    in_path = os.path.join(INPUT_DIR, fname)
    out_path = os.path.join(OUTPUT_DIR, fname)

    print(f"[{idx}/{total_files}] Procesando idioma: {lang}")

    try:
        df = pd.read_csv(in_path, sep="\t", engine="python", on_bad_lines="skip", encoding="utf-8-sig")
    except Exception as e:
        print(f"  Error leyendo {in_path}: {e}, se salta")
        continue

    # Filtrar géneros válidos
    df = df[df["gender"].isin(VALID_GENDERS)].reset_index(drop=True)
    if df.empty:
        print("  No hay filas con gender válido, se salta")
        continue

    # Agrupar por género
    rows_out = []
    for gender, df_g in df.groupby("gender"):
        # Agrupar por client_id dentro del género
        speaker_to_audios = defaultdict(list)
        for _, r in df_g.iterrows():
            sid = r["client_id"]
            audio = r["audio"]
            if pd.isna(sid) or pd.isna(audio):
                continue
            speaker_to_audios[sid].append(audio)

        # Positivos: por hablante
        pos_pairs = []
        for sid, audios in speaker_to_audios.items():
            if len(audios) < 2:
                continue
            # número objetivo de pares positivos para este hablante
            # heurística: min(combinaciones totales, MAX_POS_PAIRS_PER_SPEAKER)
            n_combos = (len(audios) * (len(audios) - 1)) // 2
            max_take = min(n_combos, MAX_POS_PAIRS_PER_SPEAKER)
            sampled = sample_pairs_from_combinations(audios, max_take)
            pos_pairs.extend(sampled)

        n_pos = len(pos_pairs)
        print(f"  {lang} / {gender}: {len(speaker_to_audios)} speakers, {n_pos} positive pairs")

        # Añadir positivos al output
        for u, v in pos_pairs:
            rows_out.append((u, v, 1))

        # Negativos: muestrear pares de distintos hablantes pero mismo género
        # objetivo: NEG_TO_POS_RATIO * n_pos
        n_neg_target = int(round(NEG_TO_POS_RATIO * n_pos))
        neg_pairs = set()
        speakers = list(speaker_to_audios.keys())
        if len(speakers) < 2:
            print(f"    No hay suficientes speakers para negativos en {gender}, se generan 0 negativos")
            continue

        attempts = 0
        max_attempts = max(1000, MAX_NEG_SAMPLING_ATTEMPTS_FACTOR * max(1, n_neg_target))
        while len(neg_pairs) < n_neg_target and attempts < max_attempts:
            attempts += 1
            # escoger dos hablantes distintos
            s1, s2 = random.sample(speakers, 2)
            a1 = random.choice(speaker_to_audios[s1])
            a2 = random.choice(speaker_to_audios[s2])
            pair = (a1, a2) if a1 <= a2 else (a2, a1)  # ordenar para evitar duplicados u/v y v/u
            if pair in neg_pairs:
                continue
            # prevenir accidentalmente usar un par idéntico a un positivo
            if (pair[0], pair[1], 1) in rows_out:
                continue
            neg_pairs.add(pair)

        n_neg = len(neg_pairs)
        print(f"    Muestreados {n_neg} negative pairs (target {n_neg_target}, attempts {attempts})")
        for u, v in neg_pairs:
            rows_out.append((u, v, 0))

    # Si no hay pares generados, saltar
    if len(rows_out) == 0:
        print(f"  No se generaron pares para {lang}, se salta guardado")
        continue

    # Desordenar y guardar
    random.shuffle(rows_out)
    df_out = pd.DataFrame(rows_out, columns=["u", "v", "label"])
    df_out.to_csv(out_path, sep="\t", index=False)
    print(f"  Guardado {len(df_out)} pares en {out_path}")
