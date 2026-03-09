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

# New optional env vars (can be passed via sbatch --export=NUM_SAMPLES=...,CSV_PATH=...)
: "${NUM_SAMPLES:=0}"
: "${CSV_PATH:=/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/f1_by_samples.csv}"

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

  # write run info for traceability
  {
    echo "OUTPUT_DIR=${OUTPUT_DIR}"
    echo "LANG_PREFIX=${LANG_PREFIX}"
    echo "NUM_SAMPLES=${NUM_SAMPLES}"
    echo "TIMESTAMP=${TIMESTAMP}"
  } > "${OUTPUT_DIR}/run_info.txt"

  echo "-----------------------------------------------"
  echo "Starting single run for language: ${LANG_PREFIX}"
  echo "Outputs: ${OUTPUT_DIR}"
  echo "-----------------------------------------------"

  FLAGS=()
  FLAGS+=( --mode "${MODE:-quantized}" )
  FLAGS+=( --hf_model "${HF_MODEL}" )
  FLAGS+=( --kmeans_path "${KMEANS_PATH}" )
  FLAGS+=( --kmeans_layer 11 )
  FLAGS+=( --n_clusters 500 )
  FLAGS+=( --quantizer_cache_dir "${QUANT_CACHE_DIR:-quant_cache}" )
  FLAGS+=( --data_dir "${DATA_DIR}" )
  FLAGS+=( --task "${TASK}" )
  FLAGS+=( --lang_prefix "${LANG_PREFIX}" )
  FLAGS+=( --output_dir "${OUTPUT_DIR}" )

  # boolean flags
  if is_true "${PRECOMPUTE:-false}"; then
    FLAGS+=( --precompute_quantized )
  fi
  if is_true "${QUANT_TRAINABLE:-false}"; then
    FLAGS+=( --quantized_embedding_trainable )
  fi
  if is_true "${FREEZE_ENCODER:-false}"; then
    FLAGS+=( --freeze_encoder )
  fi
  if [ -n "${FREEZE_FIRST_N:-}" ] && [ "${FREEZE_FIRST_N}" -gt 0 ] 2>/dev/null; then
    FLAGS+=( --freeze_first_n "${FREEZE_FIRST_N}" )
  fi
  if is_true "${USE_SEED:-false}"; then
    FLAGS+=( --seed "${SEED:-42}" )
  fi
  if is_true "${BF16:-false}"; then
    FLAGS+=( --bf16 )
  fi
  if is_true "${GRADIENT_CHECKPOINTING:-false}"; then
    FLAGS+=( --gradient_checkpointing )
  fi
  if [ -n "${DEEPSPEED:-}" ]; then
    FLAGS+=( --deepspeed "${DEEPSPEED}" )
  fi
  if [ -n "${REPORT_TO:-}" ]; then
    FLAGS+=( --report_to "${REPORT_TO}" )
  else
    FLAGS+=( --report_to "none" )
  fi

  # misc and training hyperparams
  FLAGS+=( --num_proc "${NUM_PROC:-1}" )
  FLAGS+=( --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE:-1}" )
  FLAGS+=( --per_device_eval_batch_size  "${PER_DEVICE_EVAL_BATCH_SIZE:-1}" )
  FLAGS+=( --num_train_epochs "${NUM_TRAIN_EPOCHS:-1}" )
  FLAGS+=( --train_fraction "${TRAIN_FRACTION:-1}" )
  FLAGS+=( --max_grad_norm "${MAX_GRAD_NORM:-1.0}" )
  FLAGS+=( --learning_rate "${LEARNING_RATE:-1e-05}" )
  FLAGS+=( --weight_decay "${WEIGHT_DECAY:-1e-03}" )
  FLAGS+=( --evaluation_strategy "${EVALUATION_STRATEGY:-epoch}" )
  FLAGS+=( --save_strategy "${SAVE_STRATEGY:-epoch}" )
  FLAGS+=( --lr_scheduler_type "${LR_SCHEDULER_TYPE:-constant_with_warmup}" )

  if [ -n "${WARMUP_RATIO:-}" ]; then
    FLAGS+=( --warmup_ratio "${WARMUP_RATIO}" )
  else
    FLAGS+=( --warmup_steps "${WARMUP_STEPS}" )
  fi

  # debug: print command
  echo "Running command:"
  printf "python3 %s " "${SCRIPT_PATH}"
  printf "%s " "${FLAGS[@]}"
  echo
  echo

  # synchronous run
  python3 "${SCRIPT_PATH}" "${FLAGS[@]}"

  echo "Finished run for ${LANG_PREFIX}"
  echo

  # -------------------------
  # Post-train: extract F1 from this job's OUTPUT_DIR and append to CSV (safe for concurrent jobs)
  # -------------------------

  # Paths expected in OUTPUT_DIR
  F1_FILE="${OUTPUT_DIR%/}/eval_f1_macro.txt"
  BEST_INFO="${OUTPUT_DIR%/}/best_info.json"

  # Determine RUN_F1 robustly: prefer eval_f1_macro.txt, fallback to best_info.json
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
    d=json.load(open("${BEST_INFO}"))
    v = d.get("reported_test_f1") or d.get("eval_f1_macro") or d.get("f1_macro") or "NaN"
    print(v)
except:
    print("NaN")
PY
)
    fi
  fi

  RUN_F1="${RUN_F1:-NaN}"
  echo "Job ${SLURM_JOB_ID:-$$}: Extracted RUN_F1=${RUN_F1} from ${OUTPUT_DIR}"

  # write local summary for traceability
  {
    echo "f1=${RUN_F1}"
    echo "num_samples=${NUM_SAMPLES}"
  } > "${OUTPUT_DIR}/result_summary.txt"

  # CSV append with flock to avoid races
  CSV_DIR=$(dirname "${CSV_PATH}")
  mkdir -p "${CSV_DIR}"

  LOCKFILE="${CSV_PATH}.lock"

  # open file descriptor for lock
  exec {LOCKFD}>"${LOCKFILE}"

  # acquire exclusive lock (blocking until available)
  flock -x "${LOCKFD}"

  # create CSV with header if not exists (header EXACT: f1,num_samples)
  if [ ! -f "${CSV_PATH}" ]; then
    printf 'f1,num_samples\n' > "${CSV_PATH}"
  fi

  # append the line (no quoting necessary for numeric/simple values)
  printf '%s,%s\n' "${RUN_F1}" "${NUM_SAMPLES}" >> "${CSV_PATH}"

  # release lock and close fd
  flock -u "${LOCKFD}"
  exec {LOCKFD}>&-

  echo "Appended to ${CSV_PATH}: ${RUN_F1},${NUM_SAMPLES}"
done

echo "All single runs finished."
