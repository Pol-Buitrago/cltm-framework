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
  TIMESTAMP=$(date +"%y%m%d_%H%M")
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

  # <-- AÑADIR LANG_PREFIX explícito para que Python reciba el idioma
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

  # Evitar pasar --deepspeed con argumento vacío
  if [ -n "${DEEPSPEED:-}" ]; then
    FLAGS+=( --deepspeed "${DEEPSPEED}" )
  fi

  FLAGS+=( --report_to "${REPORT_TO:-tensorboard}" )
  FLAGS+=( --mode "${MODE:-continuous}" )
  FLAGS+=( --task "${TASK}" )

  # optional switches preserved but only if verdaderos
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

  # synchronous run using the same interpreter flags as la ejecución directa
  python -u "${SCRIPT_PATH}" "${FLAGS[@]}"

  echo "Finished run for ${LANG_PREFIX}"

  # -------------------------
  # Post-train: extract F1 and append to CSV (formatted)
  # -------------------------

  CSV_PATH="/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/gender_matrix.csv"
  LOCKFILE="${CSV_PATH}.lock"

  F1_FILE="${OUTPUT_DIR}/eval_f1_macro.txt"
  BEST_INFO="${OUTPUT_DIR}/best_info.json"

  # Extract F1 robustly
  RUN_F1="NaN"
  if [ -f "${F1_FILE}" ]; then
    RUN_F1=$(awk -F': ' '/f1_macro/ {print $2; exit}' "${F1_FILE}" | tr -d '[:space:]')
  elif [ -f "${BEST_INFO}" ]; then
    if command -v jq >/dev/null 2>&1; then
      RUN_F1=$(jq -r '.reported_test_f1 // .eval_f1_macro // .f1_macro // "NaN"' "${BEST_INFO}")
    else
      RUN_F1=$(python3 - <<PY
import json
try:
    d = json.load(open("${BEST_INFO}"))
    v = d.get("reported_test_f1") or d.get("eval_f1_macro") or d.get("f1_macro") or "NaN"
    print(v)
except:
    print("NaN")
PY
)
    fi
  fi
  RUN_F1="${RUN_F1:-NaN}"

  # Derivar fields a partir de LANG_PREFIX
  # Si LANG_PREFIX tiene formato "src_tgt" -> dual; si no -> single
  model_id="${LANG_PREFIX}"
  seed_val="${SEED:-NA}"

  if [[ "${LANG_PREFIX}" == *"_"* ]]; then
    # p. ej. en_ca
    lang_src="${LANG_PREFIX%%_*}"
    lang_tgt="${LANG_PREFIX#*_}"
    type_val="dual"
  else
    lang_src="${LANG_PREFIX}"
    lang_tgt=""
    type_val="single"
  fi

  # Append to CSV safely with flock, header exacto: model_id,type,lang_src,lang_tgt,seed,f1
  mkdir -p "$(dirname "${CSV_PATH}")"
  exec {LOCKFD}>"${LOCKFILE}"
  flock -x "${LOCKFD}"

  if [ ! -f "${CSV_PATH}" ]; then
    printf 'model_id,type,lang_src,lang_tgt,seed,f1\n' > "${CSV_PATH}"
  fi

  # Escapar o normalizar valores simples (no se esperan comas en estos campos)
  printf '%s,%s,%s,%s,%s,%s\n' "${model_id}" "${type_val}" "${lang_src}" "${lang_tgt}" "${seed_val}" "${RUN_F1}" >> "${CSV_PATH}"

  flock -u "${LOCKFD}"
  exec {LOCKFD}>&-

  echo
done

echo "All single runs finished."
