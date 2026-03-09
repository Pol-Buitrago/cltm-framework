#!/bin/bash
#SBATCH --account cns125
#SBATCH --qos acc_resa
#SBATCH --nodes 1
#SBATCH --cpus-per-task 20
#SBATCH --gres gpu:1
#SBATCH --time 00-00:20:00
#SBATCH --exclusive
#SBATCH --exclude=as04r2b01,as02r3b04,as07r1b02
#SBATCH --job-name=hubert_train_single
#SBATCH --output=/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/logs/%x_%A_%a.log
#SBATCH --error=/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/logs/%x_%A_%a.err

set -euo pipefail

SUBMIT_SCRIPT="submit_samples.sh"

if [ "$#" -ne 1 ]; then
  echo "Usage: $0 TASKS_FILE.tsv" >&2
  exit 2
fi

TASKS_FILE="$1"
TASK_ID=${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID not set}

# sed is 1-based; array ID is 0-based
LINE_NUM=$((TASK_ID + 1))

# Read the corresponding line: LANG<TAB>SEED<TAB>FULLPATH<TAB>NUM_SAMPLES
LINE=$(sed -n "${LINE_NUM}p" "${TASKS_FILE}" || true)

if [ -z "${LINE}" ]; then
  echo "No line ${LINE_NUM} in ${TASKS_FILE} (TASK_ID=${TASK_ID}); nothing to do."
  exit 0
fi

IFS=$'\t' read -r LANG SEED FULLPATH NUM_SAMPLES <<< "${LINE}"

echo "SLURM_ARRAY_JOB_ID = ${SLURM_ARRAY_JOB_ID:-?}"
echo "SLURM_ARRAY_TASK_ID = ${TASK_ID}"
echo "Using TASKS_FILE = ${TASKS_FILE}"
echo "Parsed line:"
echo "  LANG        = ${LANG}"
echo "  SEED        = ${SEED}"
echo "  FULLPATH    = ${FULLPATH}"
echo "  NUM_SAMPLES = ${NUM_SAMPLES}"
echo

# Derive CSV_PATH just like before (in case submit_samples.sh needs it)
CSV_PATH="/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/${LANG}.f1_by_samples_${SEED}.csv"
echo "CSV_PATH (for information) = ${CSV_PATH}"
echo

# Export environment for submit_samples.sh
export DATA_DIR="${FULLPATH}"
export NUM_SAMPLES="${NUM_SAMPLES}"
export LANGS_STR="${LANG}"
export SEED="${SEED}"

echo "Running ${SUBMIT_SCRIPT}..."
bash "${SUBMIT_SCRIPT}"
echo "Done."
