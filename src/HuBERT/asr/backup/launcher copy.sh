#!/bin/bash
set -euo pipefail

# Script sbatch a lanzar
SBATCH_SCRIPT="/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/asr/ft_asr.sh"

# Variables comunes que quieres exportar
CV_LANG="rw"
ASR_SEED=41

# Directorio que contiene los subsets (ajusta si cambia)
BASE_SUBSETS_DIR="/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/experimental/subsets_asr/${CV_LANG}"


# Opcional: prefijo para los nombres de job
JOB_NAME_PREFIX="ft_subset"

# Si quieres limitar el número de jobs en cola simultáneamente (opcional)
# MAX_IN_QUEUE=10
# function queued_count() { squeue -u "$USER" -h | wc -l; }

for subset_dir in "${BASE_SUBSETS_DIR}"/train_*; do
  # comprobar que existe y es directorio
  if [ ! -d "${subset_dir}" ]; then
    echo "Skipping ${subset_dir}: not a directory"
    continue
  fi

  # nombre del subset (ej: train_1000)
  SUBSET_NAME=$(basename "${subset_dir}")

  # comprobar que existen los tsvs necesarios (ajusta el nombre del archivo si difiere)
  if [ ! -f "${subset_dir}/${CV_LANG}.train.tsv" ]; then
    echo "Skipping ${SUBSET_NAME}: ${CV_LANG}.train.tsv not found in ${subset_dir}"
    continue
  fi

  # opcional: limitar número de jobs en cola (descomenta sección si la quieres)
  # while [ "$(queued_count)" -ge "${MAX_IN_QUEUE}" ]; do
  #   echo "Queue full (>=${MAX_IN_QUEUE}). Waiting 30s..."
  #   sleep 30
  # done

  # lanzar sbatch exportando CV_BASE_DATA_DIR al directorio del subset, y pasando SUBSET_NAME
  echo "Submitting ${SUBSET_NAME} -> CV_BASE_DATA_DIR=${subset_dir}"
  sbatch \
    --job-name="${JOB_NAME_PREFIX}_${SUBSET_NAME}" \
    --export=ALL,CV_BASE_DATA_DIR="${subset_dir}",CV_LANG="${CV_LANG}",ASR_SEED="${ASR_SEED}",SUBSET_NAME="${SUBSET_NAME}" \
    "${SBATCH_SCRIPT}"

  # pequeño sleep opcional para no inundar el scheduler
  sleep 0.2
done

echo "All subset submissions done."
