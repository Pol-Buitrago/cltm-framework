#!/bin/bash
set -euo pipefail

# Worker script that parses exported env vars and runs the grid loop.
# It expects the main slurm script to have exported variables like LEARNING_RATES_STR, etc.

# -------------------------
# Helper: convert space/string to array
# -------------------------
read -ra LEARNING_RATES <<< "${LEARNING_RATES_STR}"
read -ra BATCH_SIZES     <<< "${BATCH_SIZES_STR}"
read -ra WEIGHT_DECAYS   <<< "${WEIGHT_DECAYS_STR}"
read -ra LR_SCHEDULERS   <<< "${LR_SCHEDULERS_STR}"

# languages: split comma-separated LANGS_STR
IFS=',' read -ra LANGS <<< "${LANGS_STR}"

# --- Warmup handling:
# If WARMUP_RATIOS_STR is set (non-empty) use it as warmup_ratio sweep.
# Otherwise fall back to WARMUP_STEPS or WARMUP_STEPS_STR if provided.
WARMUP_USE_RATIO=false
if [ -n "${WARMUP_RATIOS_STR:-}" ]; then
  WARMUP_USE_RATIO=true
  read -ra WARMUP_RATIOS <<< "${WARMUP_RATIOS_STR}"
else
  # allow WARMUP_STEPS_STR sweep, else single value from WARMUP_STEPS
  if [ -n "${WARMUP_STEPS_STR:-}" ]; then
    read -ra WARMUP_STEPS_ARR <<< "${WARMUP_STEPS_STR}"
  else
    # ensure we have a single-element array with the scalar WARMUP_STEPS
    WARMUP_STEPS_ARR=("${WARMUP_STEPS}")
  fi
fi

# --- Max grad norm handling:
# If MAX_GRAD_NORMS_STR is set, read it; otherwise use scalar MAX_GRAD_NORM
if [ -n "${MAX_GRAD_NORMS_STR:-}" ]; then
  read -ra MAX_GRAD_NORMS <<< "${MAX_GRAD_NORMS_STR}"
else
  MAX_GRAD_NORMS=("${MAX_GRAD_NORM}")
fi

# ensure required vars are set
: "${SCRIPT_PATH:?}"
: "${OUTPUT_ROOT:?}"

# -------------------------
# Helper functions
# -------------------------
# is_valid_ratio <value>
# returns 0 (success) if value is a number and 0 < value < 1
# returns 1 otherwise
is_valid_ratio() {
  local val="${1:-}"
  if [ -z "${val}" ]; then
    return 1
  fi
  # use awk to evaluate numericness and range (portable)
  awk -v x="${val}" 'BEGIN{
    if (x+0==x && x>0 && x<1) exit 0; else exit 1
  }'
}

# is_numeric (allows integers >= 0)
is_nonnegative_integer_or_float() {
  local val="${1:-}"
  if [ -z "${val}" ]; then
    return 1
  fi
  # allow floats or integers (e.g., 0, 0.0, 10, 3.5)
  awk -v x="${val}" 'BEGIN{
    if (x+0==x && x+0>=0) exit 0; else exit 1
  }'
}

# -------------------------
# Loop per language and grid
# -------------------------
for LANG_PREFIX in "${LANGS[@]}"; do
  TIMESTAMP=$(date +"%y%m%d_%H%M")
  BASE_OUTPUT_DIR="${OUTPUT_ROOT}/${LANG_PREFIX}_${TIMESTAMP}"
  mkdir -p "${BASE_OUTPUT_DIR}"

  RESULTS_FILE="${BASE_OUTPUT_DIR}/lr_grid_results.csv"
  echo "lr,bs,wd,scheduler,warmup,max_grad_norm,f1,epoch,output_dir" > "${RESULTS_FILE}"

  BEST_F1=-1
  BEST_CONFIG=""

  for LR in "${LEARNING_RATES[@]}"; do
    for BS in "${BATCH_SIZES[@]}"; do
      for WD in "${WEIGHT_DECAYS[@]}"; do
        for SCHED in "${LR_SCHEDULERS[@]}"; do

          # choose warmup loop depending on ratio vs steps
          if [ "${WARMUP_USE_RATIO}" = true ]; then
            WU_LOOP=("${WARMUP_RATIOS[@]}")
          else
            WU_LOOP=("${WARMUP_STEPS_ARR[@]}")
          fi

          for WU in "${WU_LOOP[@]}"; do
            for MG in "${MAX_GRAD_NORMS[@]}"; do

              # sanitize for safe filenames (dots -> p)
              WU_SAFE="${WU//./p}"
              MG_SAFE="${MG//./p}"

              # output dir per combination
              LR_SAFE="${LR}"
              WD_SAFE="${WD}"
              OUTPUT_DIR="${BASE_OUTPUT_DIR}/lr_${LR_SAFE}_bs${BS}_wd${WD_SAFE}_sched_${SCHED}_wu${WU_SAFE}_mg${MG_SAFE}"
              mkdir -p "${OUTPUT_DIR}" "${OUTPUT_DIR}/logs" "${OUTPUT_DIR}/checkpoints"

              # build flags
              FLAGS=()
              FLAGS+=( --mode "${MODE}" )
              FLAGS+=( --hf_model "${HF_MODEL}" )
              FLAGS+=( --kmeans_path "${KMEANS_PATH}" )
              FLAGS+=( --kmeans_layer 11 )
              FLAGS+=( --n_clusters 500 )
              FLAGS+=( --quantizer_cache_dir "${QUANT_CACHE_DIR}" )
              FLAGS+=( --data_dir "${DATA_DIR}" )
              FLAGS+=( --task "${TASK}" )
              FLAGS+=( --lang_prefix "${LANG_PREFIX}" )
              FLAGS+=( --output_dir "${OUTPUT_DIR}" )

              # common flags
              if [ "${PRECOMPUTE}" = true ]; then
                FLAGS+=( --precompute_quantized )
              fi
              if [ "${QUANT_TRAINABLE}" = true ]; then
                FLAGS+=( --quantized_embedding_trainable )
              fi
              if [ "${USE_SEED}" = true ]; then
                FLAGS+=( --seed "${SEED}" )
              fi
              if [ "${BF16}" = true ]; then
                FLAGS+=( --bf16 )
              fi
              if [ "${GRADIENT_CHECKPOINTING}" = true ]; then
                FLAGS+=( --gradient_checkpointing )
              fi
              if [ -n "${DEEPSPEED}" ]; then
                FLAGS+=( --deepspeed "${DEEPSPEED}" )
              fi
              if [ "${REPORT_TO}" != "none" ]; then
                FLAGS+=( --report_to "${REPORT_TO}" )
              else
                FLAGS+=( --report_to "none" )
              fi
              if [ "${FREEZE_ENCODER}" = true ]; then
                FLAGS+=( --freeze_encoder )
              fi

              # combination-specific flags
              FLAGS+=( --num_proc "${NUM_PROC}" )
              FLAGS+=( --per_device_train_batch_size "${BS}" )
              FLAGS+=( --per_device_eval_batch_size  "${BS}" )
              FLAGS+=( --num_train_epochs "${NUM_TRAIN_EPOCHS}" )
              FLAGS+=( --train_fraction "${TRAIN_FRACTION}" )
              FLAGS+=( --max_grad_norm "${MG}" )
              FLAGS+=( --learning_rate "${LR}" )
              FLAGS+=( --weight_decay "${WD}" )
              FLAGS+=( --evaluation_strategy "${EVALUATION_STRATEGY}" )
              FLAGS+=( --save_strategy "${SAVE_STRATEGY}" )
              FLAGS+=( --lr_scheduler_type "${SCHED}" )

              # ---------- Warmup handling (improved) ----------
              if [ "${WARMUP_USE_RATIO}" = true ]; then
                # only add --warmup_ratio if it's a valid ratio 0 < x < 1
                if is_valid_ratio "${WU}"; then
                  FLAGS+=( --warmup_ratio "${WU}" )
                else
                  # If not valid (e.g., "0", "0.0", negative, or non-numeric), do NOT pass any warmup_ratio
                  # This means warmup will be handled by defaults or warmup_steps if your code chooses so.
                  echo "Note: skipping --warmup_ratio for value '${WU}' (not in (0,1))."
                fi
              else
                # using warmup steps: pass the provided WU even if 0
                if is_nonnegative_integer_or_float "${WU}"; then
                  FLAGS+=( --warmup_steps "${WU}" )
                else
                  echo "Warning: skipping --warmup_steps for invalid value '${WU}'."
                fi
              fi
              # -------------------------------------------------

              echo "Running: LR=${LR} BS=${BS} WD=${WD} SCHED=${SCHED} WU=${WU} MG=${MG}"
              echo              
              # debug: print command
              echo "Running command:"
              printf "python3 %s " "${SCRIPT_PATH}"
              printf "%s " "${FLAGS[@]}"
              echo
              echo
              # run python synchronously
              python3 "${SCRIPT_PATH}" "${FLAGS[@]}"

              # then read the file written by Python and extract f1 + epoch
              if [ -f "${OUTPUT_DIR}/eval_f1_macro.txt" ]; then
                # Extract value after "f1_macro:" and trim whitespace
                f1=$(awk -F': *' '/^f1_macro/ { gsub(/^[ \t]+|[ \t]+$/, "", $2); print $2; exit }' "${OUTPUT_DIR}/eval_f1_macro.txt" || true)
                # Extract value after "epoch:" and trim whitespace
                epoch=$(awk -F': *' '/^epoch/ { gsub(/^[ \t]+|[ \t]+$/, "", $2); print $2; exit }' "${OUTPUT_DIR}/eval_f1_macro.txt" || true)

                # fallbacks if empty
                if [ -z "${f1}" ]; then
                  f1="nan"
                fi
                if [ -z "${epoch}" ]; then
                  epoch="unknown"
                fi
              else
                echo "Warning: ${OUTPUT_DIR}/eval_f1_macro.txt not found"
                f1="nan"
                epoch="unknown"
              fi

              echo " -> f1=${f1} epoch=${epoch}"

              # save to CSV including epoch, warmup and max grad norm
              echo "${LR},${BS},${WD},${SCHED},${WU},${MG},${f1},${epoch},${OUTPUT_DIR}" >> "${RESULTS_FILE}"

              # update best (only if f1 is a valid number)
              if [ "${f1}" != "nan" ] && echo "${f1}" | grep -Eq '^[+-]?[0-9]*\.?[0-9]+$'; then
                # use bc for reliable float comparison; set LC_NUMERIC=C for safety
                cmp=$(LC_NUMERIC=C echo "${f1} > ${BEST_F1}" | bc -l 2>/dev/null || echo 0)
                if [ "${cmp}" = "1" ]; then
                  BEST_F1="${f1}"
                  BEST_CONFIG="lr=${LR},bs=${BS},wd=${WD},sched=${SCHED},warmup=${WU},max_grad=${MG},epoch=${epoch},out=${OUTPUT_DIR}"
                fi
              else
                echo "Warning: F1 not numeric for LR=${LR},BS=${BS},WD=${WD},SCHED=${SCHED},WU=${WU},MG=${MG}"
              fi

              echo

            done
          done

        done
      done
    done
  done

  echo "Grid finished for ${LANG_PREFIX}"
  echo "Best F1: ${BEST_F1}"
  echo "Best config: ${BEST_CONFIG}"
  echo "All results: ${RESULTS_FILE}"
  echo
done

echo "All language runs finished."
