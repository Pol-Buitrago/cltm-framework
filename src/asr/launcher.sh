#!/bin/bash
set -euo pipefail

# =========================
# CONFIG: edita según necesites
# =========================

SBATCH_SCRIPT="/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/asr/ft_asr.sh"

LANGS=("ca" "en" "eo" "es" "eu" "hu" "ja" "ka" "ru" "sw" "th") # "zh-CN"

SEEDS=(41 42 43)

BASE_ROOT="/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/experimental/subsets_asr"

JOB_NAME_PREFIX="ft_subset"

SLEEP_BETWEEN_SUB=0.2

HF_CACHE_ROOT="/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/hf_cache"
#HF_CACHE_ROOT=""

DRY_RUN="${DRY_RUN:-false}"

RESULTS_DIR_BASE="/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/asr"

# =========================
# Loop principal
# =========================
for lang in "${LANGS[@]}"; do
  BASE_SUBSETS_DIR="${BASE_ROOT}/${lang}"
  if [ ! -d "${BASE_SUBSETS_DIR}" ]; then
    printf 'Skipping lang %s: directory not found: %s\n' "${lang}" "${BASE_SUBSETS_DIR}"
    continue
  fi

  # --- construir lista ordenada numéricamente de subsets ---
  subset_entries=()
  for subset_dir in "${BASE_SUBSETS_DIR}"/train_*; do
    [ -d "${subset_dir}" ] || continue
    subset_name=$(basename "${subset_dir}")
    if [[ "${subset_name}" =~ ^train_([0-9]+)$ ]]; then
      num="${BASH_REMATCH[1]}"
    else
      # intentar extraer cualquier número en el nombre; si no hay, poner muy grande para que vaya al final
      if [[ "${subset_name}" =~ ([0-9]+) ]]; then
        num="${BASH_REMATCH[1]}"
      else
        num="9999999999"
      fi
    fi
    # guardamos como "num|path" para poder ordenar por num numericamente
    subset_entries+=("${num}|${subset_dir}")
  done

  # si no hay subsets, continuar
  if [ "${#subset_entries[@]}" -eq 0 ]; then
    printf 'No subsets found for lang %s (pattern %s/train_*). Skipping.\n' "${lang}" "${BASE_ROOT}"
    continue
  fi

  # ordenar numéricamente por la primera columna (num)
  IFS=$'\n' sorted_entries=($(printf '%s\n' "${subset_entries[@]}" | sort -t'|' -k1,1n))
  unset IFS

  for entry in "${sorted_entries[@]}"; do
    subset_dir="${entry#*|}"

    SUBSET_NAME=$(basename "${subset_dir}")
    SUBSET_NAME_CLEAN=${SUBSET_NAME// /_}
    TSV_PATH="${subset_dir}/${lang}.train.tsv"

    if [ ! -f "${TSV_PATH}" ]; then
      printf 'Skipping %s %s: TSV not found: %s\n' "${lang}" "${SUBSET_NAME_CLEAN}" "${TSV_PATH}"
      continue
    fi

    # intentar extraer NUM_SAMPLES desde el nombre, fallback a contar lines del TSV
    NUM_SAMPLES=0
    if [[ "${SUBSET_NAME_CLEAN}" =~ ^train_([0-9]+)$ ]]; then
      NUM_SAMPLES="${BASH_REMATCH[1]}"
    else
      # fallback: contar lineas en el TSV menos header
      if NUM_LINES=$(wc -l < "${TSV_PATH}" 2>/dev/null | tr -d ' '); then
        if [ -n "${NUM_LINES}" ] && [ "${NUM_LINES}" -gt 0 ]; then
          NUM_SAMPLES=$((NUM_LINES - 1))
          if [ "${NUM_SAMPLES}" -lt 0 ]; then NUM_SAMPLES=0; fi
        else
          NUM_SAMPLES=0
        fi
      else
        NUM_SAMPLES=0
      fi
    fi

    for seed in "${SEEDS[@]}"; do

      HF_CACHE=""
      if [ -n "${HF_CACHE_ROOT}" ]; then
        HF_CACHE="${HF_CACHE_ROOT}/${lang}_${SUBSET_NAME_CLEAN}_s${seed}"
        mkdir -p -- "${HF_CACHE}"
      fi

      if [ -n "${HF_CACHE}" ]; then
        EXPORT_STR="ALL,CV_BASE_DATA_DIR=${subset_dir},CV_LANG=${lang},ASR_SEED=${seed},SUBSET_NAME=${SUBSET_NAME_CLEAN},HF_DATASETS_CACHE=${HF_CACHE}"
      else
        EXPORT_STR="ALL,CV_BASE_DATA_DIR=${subset_dir},CV_LANG=${lang},ASR_SEED=${seed},SUBSET_NAME=${SUBSET_NAME_CLEAN}"
      fi

      JOB_NAME="${JOB_NAME_PREFIX}_${lang}_${SUBSET_NAME_CLEAN}_s${seed}"

      # comprobar CSV de resultados existente para evitar reenviar el mismo experimento
      OUTFILE="${RESULTS_DIR_BASE}/${lang}.wer_by_samples_${seed}.csv"
      SKIP_SUBMIT=false
      if [ -f "${OUTFILE}" ] && [ "${NUM_SAMPLES}" -gt 0 ]; then
        if awk -F, -v n="${NUM_SAMPLES}" '{
             gsub(/^[ \t]+|[ \t]+$/, "", $2);
             if ($2 == n) { found=1; exit }
           }
           END { exit (found ? 0 : 1) }' "${OUTFILE}"; then
          printf 'Already present: %s %s s%s (%d samples) -> %s\n' "${lang}" "${SUBSET_NAME_CLEAN}" "${seed}" "${NUM_SAMPLES}" "${OUTFILE}"
          SKIP_SUBMIT=true
        fi
      fi

      if [ "${SKIP_SUBMIT}" = true ]; then
        continue
      fi

      # mensaje conciso antes de enviar
      printf 'Submitting: lang=%s subset=%s seed=%s samples=%d\n' "${lang}" "${SUBSET_NAME_CLEAN}" "${seed}" "${NUM_SAMPLES}"

      if [ "${DRY_RUN}" = "true" ]; then
        printf "DRY RUN: sbatch --job-name='%s' --export='%s' '%s'\n" "${JOB_NAME}" "${EXPORT_STR}" "${SBATCH_SCRIPT}"
      else
        if SBATCH_OUT=$(sbatch --job-name="${JOB_NAME}" --export="${EXPORT_STR}" "${SBATCH_SCRIPT}" 2>&1); then
          printf 'Submitted: %s -> %s\n' "${JOB_NAME}" "${SBATCH_OUT}"
        else
          # sbatch falló; lo informamos y continuamos
          printf 'Error submitting %s: %s\n' "${JOB_NAME}" "${SBATCH_OUT}"
          continue
        fi
      fi

      sleep "${SLEEP_BETWEEN_SUB}"
    done # seeds

  done # subsets
done # langs

echo "All submissions done."
