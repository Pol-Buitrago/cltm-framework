#!/bin/bash
#SBATCH --account bsc88
#SBATCH --qos acc_bscls
#SBATCH --nodes 1
#SBATCH --cpus-per-task 20
#SBATCH --gres gpu:1
#SBATCH --time 00-03:00:00
#SBATCH --exclusive
#SBATCH --exclude=as04r2b01,as02r3b04,as07r1b02
#SBATCH --job-name=hubert_train
#SBATCH --output=/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/logs/speaker/%x_%j.log
#SBATCH --error=/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/logs/speaker/%x_%j.err
# Important: export environment vars from Slurm to children
#SBATCH --export=ALL,CUBLAS_WORKSPACE_CONFIG=:4096:8,PYTHONHASHSEED=42,OMP_NUM_THREADS=1,MKL_NUM_THREADS=1

set -euo pipefail

# ================================================================
#  UNIFIED PIPELINE (SEQUENTIAL):
#       IDENTIFICATION → VERIFICATION
#
#  Usage:
#      ./run_sid_sv.sh <LANG> <SEED> <TIMESTAMP>
# ================================================================

if [ $# -ne 3 ]; then
  echo "Usage: $0 <LANG> <SEED> <TIMESTAMP>"
  exit 1
fi

LANG="$1"
SEED="$2"
TIMESTAMP="$3"

# -------------------------------
#  Paths
# -------------------------------
ROOT="/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech"

SID_SUBMIT="${ROOT}/src/speaker/identification/submit_slurm.sh"
SV_SCRIPT="${ROOT}/src/speaker/verification/sv_from_sid.py"

CONDA_ENV_PATH="/gpfs/projects/bsc88/speech/mm_s2st/scripts/paraling/gender_id_hubert/env/"

HF_MODEL="utter-project/mHuBERT-147"
MODEL_SLUG="mHuBERT-147"

SID_OUTPUT_ROOT="${ROOT}/src/speaker/verification/outputs/single/speaker_id/${MODEL_SLUG}"

# Deterministic path to the model
MODEL_DIR="${SID_OUTPUT_ROOT}/${LANG}_${TIMESTAMP}_${SLURM_JOB_ID}"

echo "=============================================================="
echo " RUNNING UNIFIED PIPELINE (SPEAKER)"
echo "  LANG      = ${LANG}"
echo "  SEED      = ${SEED}"
echo "  TIMESTAMP = ${TIMESTAMP}"
echo "  MODEL_DIR = ${MODEL_DIR}"
echo "=============================================================="

# ----------------------------------------------------------
# 1) IDENTIFICATION (SIF)
# ----------------------------------------------------------

echo "[1/2] Running identification training..."

export LANGS_STR="${LANG}"
export SEED="${SEED}"
export TIMESTAMP="${TIMESTAMP}"

bash "${SID_SUBMIT}"

echo "[OK] Identification completed."
echo

# ----------------------------------------------------------
# 2) VERIFICATION (SV)
# ----------------------------------------------------------

echo "Using externally provided TIMESTAMP=${TIMESTAMP}   LANG=${LANGS_STR}   SEED=${SEED} and SLURM_JOB_ID=${SLURM_JOB_ID}"

PAIRS_FILE="/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/04_test_pairs/tsv/${LANG}.test.tsv"
SV_OUT_DIR="${ROOT}/src/speaker/verification/outputs/embeddings_verif/${LANG}/${TIMESTAMP}"
mkdir -p "${SV_OUT_DIR}"

echo "[2/2] Running verification scoring..."
echo "  MODEL_DIR  = ${MODEL_DIR}"
echo "  PAIRS_FILE = ${PAIRS_FILE}"
echo "  OUT_DIR    = ${SV_OUT_DIR}"
echo

# Activate environment
if [ -f "$(conda info --base)/etc/profile.d/conda.sh" ]; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV_PATH}"
fi

python "${SV_SCRIPT}" \
  --model_dir "${MODEL_DIR}" \
  --pairs_file "${PAIRS_FILE}" \
  --out_dir "${SV_OUT_DIR}" \
  --batch_size 8 \
  --l2_norm \
  --device cuda \
  --pairs_cols u v label \
  --tta_speeds 1.0 0.98 1.02

echo
echo "=============================================================="
echo "Unified sequential pipeline finished."
echo "SID model stored in: ${MODEL_DIR}"
echo "SV results stored in: ${SV_OUT_DIR}"
echo "=============================================================="
