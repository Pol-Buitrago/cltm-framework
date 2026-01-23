#!/bin/bash
#SBATCH --account bsc88
#SBATCH --qos acc_bscls
#SBATCH --nodes 1
#SBATCH --cpus-per-task 20
#SBATCH --gres gpu:1
#SBATCH --time 00-01:00:00
#SBATCH --exclusive
#SBATCH --exclude=as04r2b01,as02r3b04,as07r1b02
#SBATCH --job-name=hubert_train
#SBATCH --output=/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/logs/%x_%j.log
#SBATCH --error=/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/logs/%x_%j.err
# Important: export environment vars from Slurm to children
#SBATCH --export=ALL,CUBLAS_WORKSPACE_CONFIG=:4096:8,PYTHONHASHSEED=42,OMP_NUM_THREADS=1,MKL_NUM_THREADS=1

set -euo pipefail

# -----------------------------------------------------------------------------
#  USER CONFIGURATION BLOCK — EDIT ONLY BELOW THIS LINE
# -----------------------------------------------------------------------------

# --- Mode: 'grid' or 'single'
RUN_MODE="single"   # options: "grid" or "single"

# --- Environment / conda
CONDA_ENV_PATH="/gpfs/projects/bsc88/speech/mm_s2st/scripts/paraling/gender_id_hubert/env/"

# --- Paths and dataset
TASK="gender"
SCRIPT_PATH="main.py"   # path to python training script (relative o absoluto)
KMEANS_PATH="utils/kmeans/mhubert_base_25hz_cp_mls_cv_sp_fisher_L11_km500.bin.centroids.npy"
QUANT_CACHE_DIR="quant_cache/${TASK}"
HF_MODEL="utter-project/mHuBERT-147"

DATA_ROOT="/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__"
DATA_DIR="${DATA_DIR:-${DATA_ROOT}${TASK}_id/05_balanced_120_cv_gender/tsv}"

# Si LANGS_B64 fue pasada (para evitar problemas con comas en sbatch --export), decodifícala
if [ -n "${LANGS_B64:-}" ]; then
  # decodificar base64, eliminar posibles saltos de línea
  LANGS_STR="$(printf '%s' "${LANGS_B64}" | base64 --decode | tr -d '\n')"
fi

# valor por defecto si no se pasó nada
LANGS_STR="${LANGS_STR:-eo}"


# --- Common toggles / flags
MODE="continuous"
PRECOMPUTE=false
QUANT_TRAINABLE=false
FREEZE_ENCODER=false
FREEZE_FIRST_N=0
USE_SEED=true
SEED="${SEED:-42}"

BF16=false
GRADIENT_CHECKPOINTING=false
DEEPSPEED=""
REPORT_TO="tensorboard"
NUM_PROC=1

# --- Common hyperparameters
NUM_TRAIN_EPOCHS=1
TRAIN_FRACTION=1
EVALUATION_STRATEGY="epoch"
SAVE_STRATEGY="epoch"
WARMUP_STEPS=0

# --- Single-run hyperparameters
LEARNING_RATE=1e-05
PER_DEVICE_TRAIN_BATCH_SIZE=1
PER_DEVICE_EVAL_BATCH_SIZE=1
WEIGHT_DECAY=0
LR_SCHEDULER_TYPE="constant_with_warmup"
MAX_GRAD_NORM=1 
WARMUP_RATIO=""

# --- Output
MODEL_SLUG="${HF_MODEL##*/}"
OUTPUT_ROOT="outputs/${RUN_MODE}/${TASK}_id/${MODEL_SLUG}"

HUGGINGFACE_HUB_TOKEN=""

# -----------------------------------------------------------------------------
#  END OF USER CONFIGURATION BLOCK
# -----------------------------------------------------------------------------

timestamp() { date +"%y%m%d_%H%M%S"; }

print_header() {
  echo "------------------------------------------------------------"
  echo "UNIFIED SUBMIT SCRIPT"
  echo "RUN_MODE: ${RUN_MODE}"
  echo "Timestamp: $(timestamp)"
  echo "------------------------------------------------------------"
}

case "${RUN_MODE}" in
  grid|single) ;;
  *)
    echo "ERROR: RUN_MODE must be 'grid' or 'single'. Found: ${RUN_MODE}"
    exit 1
    ;;
esac

# Activate conda env
if [ -n "${CONDA_ENV_PATH}" ]; then
  if [ -f "$(conda info --base)/etc/profile.d/conda.sh" ]; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV_PATH}"
  else
    echo "Warning: conda profile not found; skipping conda activation."
  fi
fi

# -------------------------
# Export to environment for workers (refuerzo)
# -------------------------
export TASK SCRIPT_PATH KMEANS_PATH QUANT_CACHE_DIR HF_MODEL DATA_ROOT DATA_DIR
export LANGS_STR MODE PRECOMPUTE QUANT_TRAINABLE FREEZE_ENCODER FREEZE_FIRST_N
export USE_SEED SEED BF16 GRADIENT_CHECKPOINTING DEEPSPEED REPORT_TO NUM_PROC
export HUGGINGFACE_HUB_TOKEN
# determinism-related exports
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=42
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

export NUM_TRAIN_EPOCHS TRAIN_FRACTION EVALUATION_STRATEGY SAVE_STRATEGY WARMUP_STEPS

export PER_DEVICE_TRAIN_BATCH_SIZE PER_DEVICE_EVAL_BATCH_SIZE
export LEARNING_RATE WEIGHT_DECAY LR_SCHEDULER_TYPE MAX_GRAD_NORM WARMUP_RATIO

export LEARNING_RATES_STR BATCH_SIZES_STR WEIGHT_DECAYS_STR LR_SCHEDULERS_STR
export MAX_GRAD_NORMS_STR WARMUP_RATIOS_STR

export MODEL_SLUG OUTPUT_ROOT

print_header

mkdir -p "${OUTPUT_ROOT}"

# Fix CPU affinity (reduce nondeterministic scheduling)
TASKSET_CPUS="0-3"
taskset -c ${TASKSET_CPUS} true || true

# Log basic environment for debugging
echo "HOST: $(hostname)"
echo "DATE: $(date)"
echo "CONDA ENV: ${CONDA_ENV_PATH}"
nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv
echo "Environment variables (determinism):"
echo "  CUBLAS_WORKSPACE_CONFIG=$CUBLAS_WORKSPACE_CONFIG"
echo "  PYTHONHASHSEED=$PYTHONHASHSEED"
echo "  OMP_NUM_THREADS=$OMP_NUM_THREADS"
echo "  MKL_NUM_THREADS=$MKL_NUM_THREADS"
echo "Launching on node: $(hostname)"

# -------------------------
# Launch wrapper via srun (wrapper runs under srun)
# -------------------------
case "${RUN_MODE}" in
  grid)
    echo "Launching GRID worker via srun: utils/run_grid.sh"
    srun ./utils/run_grid.sh
    ;;
  single)
    echo "Launching SINGLE worker via srun: utils/run_single.sh"
    srun ./utils/run_single.sh
    ;;
esac

echo "Submit finished."

