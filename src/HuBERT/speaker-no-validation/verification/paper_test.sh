#!/bin/bash
# Script para combinar TSVs y crear archivos para paper tests

# Directorios
SRC_DIR="/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv"
DST_DIR="/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/08_paper_tests/tsv"

mkdir -p "$DST_DIR"

# ------------------------------------------------------------
# 1️⃣ eo_X.train.tsv (eo + de + ro + ru)
# ------------------------------------------------------------
TARGET_LANGS=("eo" "de" "ro" "ru")
OUT_FILE="$DST_DIR/eo_4N.train.tsv"

head -n 1 "$SRC_DIR/eo.train.tsv" > "$OUT_FILE"
for LANG in "${TARGET_LANGS[@]}"; do
    tail -n +2 "$SRC_DIR/${LANG}.train.tsv" >> "$OUT_FILE"
done
echo "Archivo combinado creado: $OUT_FILE"

# ------------------------------------------------------------
# 2️⃣ eo_ro_ru.train.tsv (eo + ro + ru)
# ------------------------------------------------------------
TARGET_LANGS2=("eo" "ro" "ru")
OUT_FILE2="$DST_DIR/eo_3N.train.tsv"

head -n 1 "$SRC_DIR/eo.train.tsv" > "$OUT_FILE2"
for LANG in "${TARGET_LANGS2[@]}"; do
    tail -n +2 "$SRC_DIR/${LANG}.train.tsv" >> "$OUT_FILE2"
done
echo "Archivo combinado eo+ro+ru creado: $OUT_FILE2"

# ------------------------------------------------------------
# 3️⃣ eo_ru.train.tsv (eo + ru)
# ------------------------------------------------------------
TARGET_LANGS3=("eo" "ru")
OUT_FILE3="$DST_DIR/eo_2N.train.tsv"

head -n 1 "$SRC_DIR/eo.train.tsv" > "$OUT_FILE3"
for LANG in "${TARGET_LANGS3[@]}"; do
    tail -n +2 "$SRC_DIR/${LANG}.train.tsv" >> "$OUT_FILE3"
done
echo "Archivo combinado eo+ru creado: $OUT_FILE3"

# ------------------------------------------------------------
# 4️⃣ eo_only.train.tsv (solo eo)
# ------------------------------------------------------------
cp "$SRC_DIR/eo.train.tsv" "$DST_DIR/eo_only.train.tsv"
echo "Archivo eo-only copiado: $DST_DIR/eo_only.train.tsv"