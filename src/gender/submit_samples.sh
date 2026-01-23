#!/bin/bash
#SBATCH --account cns125
#SBATCH --qos acc_resa
#SBATCH --nodes 1
#SBATCH --cpus-per-task 20
#SBATCH --gres gpu:1
#SBATCH --time 00-00:35:00
#SBATCH --exclusive
#SBATCH --exclude=as04r2b01,as02r3b04,as07r1b02
#SBATCH --job-name=hubert_train_single
#SBATCH --output=/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/logs/%x_%j.log
#SBATCH --error=/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/logs/%x_%j.err
#SBATCH --export=ALL,CUBLAS_WORKSPACE_CONFIG=:4096:8,PYTHONHASHSEED=43,OMP_NUM_THREADS=1,MKL_NUM_THREADS=1

set -euo pipefail

# -----------------------------------------------------------------------------
#  USER CONFIGURATION BLOCK — EDIT ONLY BELOW THIS LINE
# -----------------------------------------------------------------------------

# Environment / conda
CONDA_ENV_PATH="/gpfs/projects/bsc88/speech/mm_s2st/scripts/paraling/gender_id_hubert/env/"

# Paths and dataset (edit DATA_DIR aquí)
TASK="gender"
SCRIPT_PATH="main.py"
KMEANS_PATH="utils/kmeans/mhubert_base_25hz_cp_mls_cv_sp_fisher_L11_km500.bin.centroids.npy"
QUANT_CACHE_DIR="quant_cache/${TASK}"
HF_MODEL="utter-project/mHuBERT-147"

# Cambia aquí la ruta al dataset que quieras usar para este run single
DATA_ROOT="/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__"
: "${DATA_DIR:=${DATA_ROOT}${TASK}_id/03_balanced_cv_gender/tsv}"

# Lengua (una sola; single-run)
: "${LANGS_STR:=eo}"  # usa eo solo si no se pasa desde el entorno

# Toggles y seeds
MODE="continuous"
PRECOMPUTE=false
QUANT_TRAINABLE=false
FREEZE_ENCODER=false
FREEZE_FIRST_N=0
USE_SEED=true
: "${SEED:=42}"

# Construcción automática del path de salida con sufijo = LANGS_STR
CSV_PATH="/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/${LANGS_STR}.f1_by_samples_${SEED}.csv"

BF16=false
GRADIENT_CHECKPOINTING=false
DEEPSPEED=""
REPORT_TO="tensorboard"
NUM_PROC=1

# Hyperparámetros (single-run)
NUM_TRAIN_EPOCHS=1
TRAIN_FRACTION=1
EVALUATION_STRATEGY="epoch"
SAVE_STRATEGY="epoch"
WARMUP_STEPS=0

LEARNING_RATE=1e-05
PER_DEVICE_TRAIN_BATCH_SIZE=1
PER_DEVICE_EVAL_BATCH_SIZE=1
WEIGHT_DECAY=0
LR_SCHEDULER_TYPE="constant_with_warmup"
MAX_GRAD_NORM=1
WARMUP_RATIO=""

# Output root
MODEL_SLUG="${HF_MODEL##*/}"
OUTPUT_ROOT="outputs/single/${TASK}_id/${MODEL_SLUG}"

# -----------------------------------------------------------------------------
#  END OF USER CONFIGURATION BLOCK
# -----------------------------------------------------------------------------

# helper
timestamp() { date +"%y%m%d_%H%M%S"; }

print_header() {
  echo "------------------------------------------------------------"
  echo "SINGLE SUBMIT SCRIPT"
  echo "Timestamp: $(timestamp)"
  echo "TASK: ${TASK}"
  echo "DATA_DIR: ${DATA_DIR}"
  echo "OUTPUT_ROOT: ${OUTPUT_ROOT}"
  echo "CSV_PATH: ${CSV_PATH}"
  echo "------------------------------------------------------------"
}

# activate conda env if provided
if [ -n "${CONDA_ENV_PATH:-}" ]; then
  if [ -f "$(conda info --base)/etc/profile.d/conda.sh" ]; then
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV_PATH}"
  else
    echo "Warning: conda profile not found; skipping conda activation."
  fi
fi

# inferir NUM_SAMPLES desde DATA_DIR: último número en el path (ej: ..._3003 -> 3003)
NUM_SAMPLES=$(echo "${DATA_DIR}" | grep -oE '[0-9]+$' || true)
if [ -z "${NUM_SAMPLES}" ]; then
  echo "Warning: no se pudo inferir NUM_SAMPLES desde '${DATA_DIR}', asignando 0"
  NUM_SAMPLES=0
fi

# export variables para que las vea run_single.sh
export TASK SCRIPT_PATH KMEANS_PATH QUANT_CACHE_DIR HF_MODEL DATA_ROOT DATA_DIR
export LANGS_STR MODE PRECOMPUTE QUANT_TRAINABLE FREEZE_ENCODER FREEZE_FIRST_N
export USE_SEED SEED BF16 GRADIENT_CHECKPOINTING DEEPSPEED REPORT_TO NUM_PROC
export HUGGINGFACE_HUB_TOKEN=""  # si necesitas token ponlo aquí

export NUM_TRAIN_EPOCHS TRAIN_FRACTION EVALUATION_STRATEGY SAVE_STRATEGY WARMUP_STEPS

export PER_DEVICE_TRAIN_BATCH_SIZE PER_DEVICE_EVAL_BATCH_SIZE
export LEARNING_RATE WEIGHT_DECAY LR_SCHEDULER_TYPE MAX_GRAD_NORM WARMUP_RATIO

export MODEL_SLUG OUTPUT_ROOT
export NUM_SAMPLES CSV_PATH

# determinism-related exports (refuerzo)
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=43
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo "HOST: $(hostname)"
echo "DATE: $(date)"
nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv
echo "CUBLAS_WORKSPACE_CONFIG=$CUBLAS_WORKSPACE_CONFIG"
echo "PYTHONHASHSEED=$PYTHONHASHSEED"
echo "OMP_NUM_THREADS=$OMP_NUM_THREADS"
echo "MKL_NUM_THREADS=$MKL_NUM_THREADS"

# header
print_header

# asegurar output root
mkdir -p "${OUTPUT_ROOT}"

# lanzar worker single (utils/run_single.sh usa OUTPUT_ROOT y LANGS_STR y escribirá en CSV_PATH)
echo "Launching SINGLE worker: utils/run_samples.sh"
# lanza el wrapper con srun para que Slurm gestione correctamente el proceso
srun --export=ALL ./utils/run_samples.sh

echo "Submit finished."
