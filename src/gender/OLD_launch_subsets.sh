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
#SBATCH --output=/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/logs/%x_%j.log
#SBATCH --error=/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/logs/%x_%j.err

set -euo pipefail

# -------------------------
# Config / defaults (editar aquí)
# -------------------------
# Lista de idiomas (definida dentro del script; comma-separated)
LANGS_LIST="ru,hu,sw,th,ja,eu,es,ka"

# Lista de seeds (definida dentro del script; comma-separated)
SEEDS_LIST="41,42,43"

# Root de los subsets (definido aquí; no se modifica desde la línea de comando)
SUBSETS_ROOT="/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/experimental/subsets"

# Otras opciones
DRY_RUN=false
SUBMIT_SCRIPT="submit_samples.sh"

# -------------------------
# Parse options (solo flags, no posicionales)
# -------------------------
while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) DRY_RUN=true; shift ;;
    --submit-script) SUBMIT_SCRIPT="$2"; shift 2 ;;
    -h|--help)
      cat <<EOF
Usage: $0 [--dry-run] [--submit-script submit_samples.sh]
All configuration (SUBSETS_ROOT, LANGS_LIST, SEEDS_LIST) is inside the script.
EOF
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

# -------------------------
# Helpers
# -------------------------
extract_num_samples() {
  local path="$1"
  if [[ "$path" =~ perclass_([0-9]+)$ ]]; then
    echo "${BASH_REMATCH[1]}"
    return 0
  fi
  if num=$(echo "$path" | grep -oE '[0-9]+$' || true); then
    if [ -n "$num" ]; then
      echo "$num"
      return 0
    fi
  fi
  echo "0"
}

csv_contains_num() {
  local csv="$1"
  local num="$2"

  [ -f "$csv" ] || return 1

  if awk -F',' -v n="$num" 'NR>1 {
       gsub(/\r/,"",$2);
       gsub(/^[ \t]+|[ \t]+$/,"",$2);
       if ($2 == n) { found=1; exit 0 }
     }
     END { exit !found }' "$csv"
  then
    return 0
  else
    return 1
  fi
}

# -------------------------
# Parse lists into arrays
# -------------------------
read -r -a LANGS <<< "$(tr ',' ' ' <<< "${LANGS_LIST}")"
read -r -a SEEDS <<< "$(tr ',' ' ' <<< "${SEEDS_LIST}")"

# -------------------------
# Main
# -------------------------
echo "Launch script"
echo "SUBSETS_ROOT = ${SUBSETS_ROOT}"
echo "SUBMIT_SCRIPT = ${SUBMIT_SCRIPT}"
echo "LANGS_LIST = ${LANGS_LIST}"
echo "SEEDS_LIST = ${SEEDS_LIST}"
echo "DRY_RUN = ${DRY_RUN}"
echo

for LANG in "${LANGS[@]}"; do
  # encontrar subsets para este idioma (se soportan guiones en LANG)
  mapfile -t SUBSET_DIRS < <(find "${SUBSETS_ROOT}" -maxdepth 1 -type d -name "${LANG}.subset_*" -printf '%f\n' 2>/dev/null | sort -V)

  if [ ${#SUBSET_DIRS[@]} -eq 0 ]; then
    echo "Warning: No subset directories found for language '${LANG}' under ${SUBSETS_ROOT}"
    continue
  fi

  echo "Found ${#SUBSET_DIRS[@]} subset(s) for lang='${LANG}'."

  for SEED in "${SEEDS[@]}"; do
    for dname in "${SUBSET_DIRS[@]}"; do
      fullpath="${SUBSETS_ROOT%/}/${dname}"
      NUM_SAMPLES=$(extract_num_samples "${dname}")

      echo "----------------------------------------"
      echo "Lang: ${LANG}  Seed: ${SEED}"
      echo "Subset: ${dname}"
      echo " Full path: ${fullpath}"
      echo " Num samples inferred: ${NUM_SAMPLES}"

      # CSV path según convención en submit_samples.sh
      CSV_PATH="/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/${LANG}.f1_by_samples_${SEED}.csv"
      echo "CSV_PATH (used to detect duplicates) = ${CSV_PATH}"

      if csv_contains_num "${CSV_PATH}" "${NUM_SAMPLES}"; then
        echo "ACTION: Skipping submit for num_samples=${NUM_SAMPLES} (detected in ${CSV_PATH})."
        continue
      fi

      SBATCH_CMD=(sbatch --export=DATA_DIR="${fullpath}",NUM_SAMPLES="${NUM_SAMPLES}",LANGS_STR="${LANG}",SEED="${SEED}" "${SUBMIT_SCRIPT}")

      echo "Command:"
      printf ' %s' "${SBATCH_CMD[@]}"
      echo
      if [ "${DRY_RUN}" = true ]; then
        echo "(dry-run) not submitting"
      else
        set +e
        submit_out=$("${SBATCH_CMD[@]}" 2>&1)
        ret=$?
        set -e
        if [ $ret -ne 0 ]; then
          echo "Submission failed for ${dname} (lang=${LANG}, seed=${SEED}):"
          echo "${submit_out}"
        else
          echo "Submitted: ${submit_out}"
        fi
        sleep 0.2
      fi
    done
  done
done

echo "Done."
