#!/bin/bash
set -euo pipefail

# Single-run worker.
# Expects variables exported by the submit script:
# SCRIPT_PATH, HF_MODEL, KMEANS_PATH, DATA_DIR, TASK, LANGS_STR, OUTPUT_ROOT, etc.

# -------------------------
# Ensure required env vars
# -------------------------
: "${SCRIPT_PATH:?}"
: "${HF_MODEL:?}"
: "${KMEANS_PATH:?}"
: "${DATA_DIR:?}"
: "${TASK:?}"
: "${LANGS_STR:?}"
: "${OUTPUT_ROOT:?}"

# -------------------------
# Parse LANGS_STR (comma-separated)
# -------------------------
IFS=',' read -ra LANGS <<< "${LANGS_STR}"

# helper for boolean strings
is_true() {
  case "${1}" in
    "true"|"True"|"1") return 0 ;;
    *) return 1 ;;
  esac
}

# -------------------------
# Loop languages sequentially
# -------------------------
for LANG_PREFIX in "${LANGS[@]}"; do
  # usar TIMESTAMP exportado por el submit si existe, si no generarlo localmente
  TIMESTAMP="${TIMESTAMP:-$(date +"%y%m%d_%H%M")}"
  echo "Worker using TIMESTAMP=${TIMESTAMP} for LANG_PREFIX=${LANG_PREFIX} and SLURM_JOB_ID=${SLURM_JOB_ID}"

  OUTPUT_DIR="${OUTPUT_ROOT}/${LANG_PREFIX}_${TIMESTAMP}_${SLURM_JOB_ID}"
  mkdir -p "${OUTPUT_DIR}" "${OUTPUT_DIR}/logs" "${OUTPUT_DIR}/checkpoints"

  echo "-----------------------------------------------"
  echo "Starting single run for language: ${LANG_PREFIX}"
  echo "Outputs: ${OUTPUT_DIR}"
  echo "-----------------------------------------------"

  # --- Construir FLAGS para reproducir la invocación directa ---
  FLAGS=()
  FLAGS+=( --seed "${SEED:-42}" )
  FLAGS+=( --hf_model "${HF_MODEL}" )
  FLAGS+=( --data_dir "${DATA_DIR}" )
  FLAGS+=( --lang_prefix "${LANG_PREFIX}" )
  FLAGS+=( --output_dir "${OUTPUT_DIR}" )
  FLAGS+=( --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE:-1}" )
  FLAGS+=( --per_device_eval_batch_size  "${PER_DEVICE_EVAL_BATCH_SIZE:-1}" )
  FLAGS+=( --num_train_epochs "${NUM_TRAIN_EPOCHS:-1}" )
  FLAGS+=( --evaluation_strategy "${EVALUATION_STRATEGY:-epoch}" )
  FLAGS+=( --save_strategy "${SAVE_STRATEGY:-epoch}" )
  FLAGS+=( --learning_rate "${LEARNING_RATE:-1e-05}" )
  FLAGS+=( --weight_decay "${WEIGHT_DECAY:-1e-03}" )
  FLAGS+=( --lr_scheduler_type "${LR_SCHEDULER_TYPE:-constant_with_warmup}" )
  FLAGS+=( --max_grad_norm "${MAX_GRAD_NORM:-1.0}" )
  FLAGS+=( --warmup_steps "${WARMUP_STEPS:-0}" )

  if [ -n "${DEEPSPEED:-}" ]; then
    FLAGS+=( --deepspeed "${DEEPSPEED}" )
  fi

  FLAGS+=( --report_to "${REPORT_TO:-tensorboard}" )
  FLAGS+=( --mode "${MODE:-continuous}" )
  FLAGS+=( --task "${TASK}" )

  if is_true "${BF16:-false}"; then FLAGS+=( --bf16 ); fi
  if is_true "${GRADIENT_CHECKPOINTING:-false}"; then FLAGS+=( --gradient_checkpointing ); fi

  # Export para compatibilidad con código que pueda leer env vars
  export LANG_PREFIX="${LANG_PREFIX}"
  export SEED="${SEED:-42}"

  # Guardar run_info para trazabilidad
  {
    echo "LANG_PREFIX=${LANG_PREFIX}"
    echo "SEED=${SEED:-42}"
    echo "FLAGS=${FLAGS[*]}"
  } > "${OUTPUT_DIR}/run_info.txt"

  # debug: print command
  echo "Running command:"
  printf "python -u %s " "${SCRIPT_PATH}"
  printf "%s " "${FLAGS[@]}"
  echo
  echo

  # synchronous run using the same interpreter flags que la ejecución directa
  python -u "${SCRIPT_PATH}" "${FLAGS[@]}"

  echo "Finished run for ${LANG_PREFIX}"
  echo
done

echo "All single runs finished."
