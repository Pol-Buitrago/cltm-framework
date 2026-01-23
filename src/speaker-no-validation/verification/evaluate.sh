#!/usr/bin/env bash
set -euo pipefail

# ---------------- CONFIG ----------------
MODEL_DIR="/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speaker/identification/outputs/single/speaker_id/mHuBERT-147/az_251127_1630_32840918"
PAIRS_FILE="/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/04_test_pairs/tsv/az.test.tsv"
OUT_DIR="/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speaker/verification/outputs/embeddings_verif/az/my_pairs"

BATCH_SIZE=8
DEVICE="cuda"
L2_NORM_FLAG="--l2_norm"
SCRIPT_PATH="sv_from_sid.py"

# ---------------- ENVIRONMENT ----------------
CONDA_ENV_PATH="/gpfs/projects/bsc88/speech/mm_s2st/scripts/paraling/gender_id_hubert/env/"

echo "Activating virtual environment at: $CONDA_ENV_PATH"
# Activate conda env
if [ -n "${CONDA_ENV_PATH}" ]; then
  if [ -f "$(conda info --base)/etc/profile.d/conda.sh" ]; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV_PATH}"
  else
    echo "Warning: conda profile not found; skipping conda activation."
  fi
fi

PYTHON_BIN="python"

# Crear directorio de salida
mkdir -p "$OUT_DIR"

echo "Running SV scoring"
echo "MODEL_DIR: $MODEL_DIR"
echo "PAIRS_FILE: $PAIRS_FILE"
echo "OUT_DIR: $OUT_DIR"
echo "BATCH_SIZE: $BATCH_SIZE"
echo "DEVICE: $DEVICE"

(
  date
  echo "Command:"
  echo "$PYTHON_BIN $SCRIPT_PATH --model_dir \"$MODEL_DIR\" --pairs_file \"$PAIRS_FILE\" --out_dir \"$OUT_DIR\" --batch_size $BATCH_SIZE $L2_NORM_FLAG --device $DEVICE --pairs_cols u v label"
  echo "----- Output -----"

  $PYTHON_BIN "$SCRIPT_PATH" \
    --model_dir "$MODEL_DIR" \
    --pairs_file "$PAIRS_FILE" \
    --out_dir "$OUT_DIR" \
    --batch_size "$BATCH_SIZE" \
    $L2_NORM_FLAG \
    --device "$DEVICE" \
    --pairs_cols u v label

  echo "Finished at: $(date)"
) 2>&1 | tee "$OUT_DIR/run.log"
