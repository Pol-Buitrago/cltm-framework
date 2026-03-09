#!/bin/bash
#SBATCH --account bsc88
#SBATCH --qos acc_bscls
#SBATCH --nodes 1
#SBATCH --cpus-per-task 20
#SBATCH --gres gpu:1
#SBATCH --time 02-00:00:00
#SBATCH --exclusive
#SBATCH --exclude=as04r2b01,as02r3b04,as07r1b02
#SBATCH --job-name=sid_sv_launcher
#SBATCH --output=/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/logs/speaker/%x_%j.log
#SBATCH --error=/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/logs/speaker/%x_%j.err
#SBATCH --export=ALL,CUBLAS_WORKSPACE_CONFIG=:4096:8,PYTHONHASHSEED=42,OMP_NUM_THREADS=1,MKL_NUM_THREADS=1

set -euo pipefail

# ---------------- CONFIG ----------------
RUN_SCRIPT="./run_sid_sv.sh"   # script que requiere: <LANG> <SEED> <TIMESTAMP>

LANGS=("en")
SEEDS=(41)
SUBSETS_ROOT="/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/experimental/subsets_speaker"
SUMMARY_ROOT="/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs"

echo "[launcher] Starting submissions (one sbatch per run)"
echo "[launcher] Script to submit: $RUN_SCRIPT"
echo

for LANG in "${LANGS[@]}"; do
  SUBSET_DIR="${SUBSETS_ROOT%/}/${LANG}"
  if [ ! -d "${SUBSET_DIR}" ]; then
    echo "[launcher] No directory for LANG=${LANG} at ${SUBSET_DIR}, skipping."
    continue
  fi

  mapfile -t DATA_DIRS < <(find "${SUBSET_DIR}" -maxdepth 1 -type d -name "subset_*" -printf '%p\n' 2>/dev/null | sort -V)
  if [ ${#DATA_DIRS[@]} -eq 0 ]; then
    echo "[launcher] No subset_* directories found for LANG=${LANG}, skipping."
    continue
  fi

  echo "[launcher] Found ${#DATA_DIRS[@]} subsets for LANG=${LANG}"

  for SEED in "${SEEDS[@]}"; do
    CSV_FILE="${SUMMARY_ROOT}/${LANG}.eer_by_samples_${SEED}.csv"

    for DATA_DIR in "${DATA_DIRS[@]}"; do
      subset_base=$(basename "$DATA_DIR")
      if [[ "$subset_base" =~ n([0-9]+) ]]; then
        NUM_SAMPLES="${BASH_REMATCH[1]}"
      else
        NUM_SAMPLES="nan"
      fi

      # Saltar si ya existe la fila para ese num_samples
      if [ -f "$CSV_FILE" ] && grep -qE "^[^,]*,[^,]*,[^,]*,${NUM_SAMPLES}$" "$CSV_FILE"; then
        echo "[launcher] Skipping LANG=${LANG}, SEED=${SEED}, subset=${subset_base} (num_samples=${NUM_SAMPLES}) — already in ${CSV_FILE}"
        continue
      fi

      TIMESTAMP=$(date +"%y%m%d_%H%M%S")
      echo "--------------------------------------------------"
      echo "[launcher] SUBMIT LANG=${LANG}  SEED=${SEED}  DATA_DIR=${DATA_DIR}  TIMESTAMP=${TIMESTAMP}"
      echo "--------------------------------------------------"

      # Construir comando sbatch: IMPORTANTISSIMO, pasamos los tres argumentos
      SBATCH_CMD=(sbatch --export=ALL,DATA_DIR="${DATA_DIR}",LANGS_STR="${LANG}",SEED="${SEED}",TIMESTAMP="${TIMESTAMP}" "${RUN_SCRIPT}" "${LANG}" "${SEED}" "${TIMESTAMP}")

      printf "Submitting: "
      printf "%s " "${SBATCH_CMD[@]}"
      echo

      set +e
      submit_out=$("${SBATCH_CMD[@]}" 2>&1)
      ret=$?
      set -e

      if [ $ret -ne 0 ]; then
        echo "[launcher] Submission FAILED for subset=${subset_base}:"
        echo "${submit_out}"
      else
        echo "[launcher] Submitted: ${submit_out}"
      fi

      sleep 0.3
    done
  done
done

echo "[launcher] All submissions done."
