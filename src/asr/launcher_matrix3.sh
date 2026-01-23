#!/bin/bash
set -euo pipefail

# =========================
# CONFIG: editar según necesites
# =========================

SBATCH_SCRIPT="/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/asr/ft_asr_matrix3.sh"

# Directorio que contiene los .train.tsv (leer los model_id desde aqui)
TSV_DIR="/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__asr/03_pairs/tsv"

# Seeds: 42..51 inclusive (10 seeds)
SEEDS=({42..51})

JOB_NAME_PREFIX="ft_matrix"

SLEEP_BETWEEN_SUB=0.2

HF_CACHE_ROOT="/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/hf_cache/pairs"
#HF_CACHE_ROOT=""

DRY_RUN="${DRY_RUN:-false}"

# CSV de resultados (nuevo formato que usas ahora)
RESULTS_CSV="/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/asr_matrix/seeds/gender_matrix_bilingual.csv"

# =========================
# Comprobaciones iniciales
# =========================
if [ ! -d "${TSV_DIR}" ]; then
  printf 'ERROR: TSV directory not found: %s\n' "${TSV_DIR}"
  exit 2
fi

if [ ! -f "${SBATCH_SCRIPT}" ]; then
  printf 'ERROR: SBATCH script not found: %s\n' "${SBATCH_SCRIPT}"
  exit 2
fi

# =========================
# Construir lista única de model_id a partir de *.train.tsv
# =========================
declare -A seen_models=()

shopt -s nullglob
for f in "${TSV_DIR}"/*.train.tsv; do
  base=$(basename "${f}")
  # quitar sufijo .train.tsv
  model_id="${base%.train.tsv}"
  if [ -z "${model_id}" ]; then
    continue
  fi
  seen_models["${model_id}"]=1
done
shopt -u nullglob

# Si no hay modelos, salir
if [ "${#seen_models[@]}" -eq 0 ]; then
  printf 'No .train.tsv files found in %s. Nothing to submit.\n' "${TSV_DIR}"
  exit 0
fi

# =========================
# Loop principal: por model_id y por seed
# =========================
for model_id in "${!seen_models[@]}"; do

  # Comprobar que el .train.tsv existe (robustez)
  TSV_PATH="${TSV_DIR}/${model_id}.train.tsv"
  if [ ! -f "${TSV_PATH}" ]; then
    printf 'Skipping model_id %s: TSV not found: %s\n' "${model_id}" "${TSV_PATH}"
    continue
  fi

  for seed in "${SEEDS[@]}"; do

    # comprobar si ya existe entrada en RESULTS_CSV con model_id y seed
    SKIP_SUBMIT=false
    if [ -f "${RESULTS_CSV}" ]; then
      if awk -F',' -v m="${model_id}" -v s="${seed}" 'BEGIN{found=0} NR>1{
            gsub(/^[ \t]+|[ \t]+$/,"",$1);
            gsub(/^[ \t]+|[ \t]+$/,"",$5);
            if($1==m && $5==s){found=1; exit}
          }
          END{ if(found) exit 0; else exit 1 }' "${RESULTS_CSV}"; then
        printf 'Already present in results CSV: %s s%s -> skipping\n' "${model_id}" "${seed}"
        SKIP_SUBMIT=true
      fi
    fi

    if [ "${SKIP_SUBMIT}" = true ]; then
      continue
    fi

    # preparar HF cache por modelo y seed si se desea
    HF_CACHE=""
    if [ -n "${HF_CACHE_ROOT}" ]; then
      HF_CACHE="${HF_CACHE_ROOT}/${model_id}_s${seed}"
      mkdir -p -- "${HF_CACHE}"
    fi

    # Preparar variables a exportar al job
    if [ -n "${HF_CACHE}" ]; then
      EXPORT_STR="ALL,CV_BASE_DATA_DIR=${TSV_DIR},CV_LANG=${model_id},ASR_SEED=${seed},HF_DATASETS_CACHE=${HF_CACHE}"
    else
      EXPORT_STR="ALL,CV_BASE_DATA_DIR=${TSV_DIR},CV_LANG=${model_id},ASR_SEED=${seed}"
    fi

    JOB_NAME="${JOB_NAME_PREFIX}_${model_id}_s${seed}"

    # Mensaje conciso antes de enviar
    printf 'Submitting: model_id=%s seed=%s TSV=%s\n' "${model_id}" "${seed}" "${TSV_PATH}"

    if [ "${DRY_RUN}" = "true" ]; then
      printf "DRY RUN: sbatch --job-name='%s' --export='%s' '%s'\n" "${JOB_NAME}" "${EXPORT_STR}" "${SBATCH_SCRIPT}"
    else
      if SBATCH_OUT=$(sbatch --job-name="${JOB_NAME}" --export="${EXPORT_STR}" "${SBATCH_SCRIPT}" 2>&1); then
        printf 'Submitted: %s -> %s\n' "${JOB_NAME}" "${SBATCH_OUT}"
      else
        printf 'Error submitting %s: %s\n' "${JOB_NAME}" "${SBATCH_OUT}"
        continue
      fi
    fi

    sleep "${SLEEP_BETWEEN_SUB}"

  done # seeds

done # model_id

echo "All submissions done."
