#!/bin/bash
set -euo pipefail

# -------------------------
# Same config as before
# -------------------------
LANGS_LIST="ru,hu,sw,th,ja,eu,es,ka"
SEEDS_LIST="41,42,43"
SUBSETS_ROOT="/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/experimental/subsets"

# -------------------------
# Helpers (same as before)
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
# Enumerate combos and output ONLY missing ones
# Output format (TSV):
#   LANG<TAB>SEED<TAB>FULLPATH<TAB>NUM_SAMPLES
# -------------------------
for LANG in "${LANGS[@]}"; do
  mapfile -t SUBSET_DIRS < <(find "${SUBSETS_ROOT}" -maxdepth 1 -type d -name "${LANG}.subset_*" -printf '%f\n' 2>/dev/null | sort -V)

  if [ ${#SUBSET_DIRS[@]} -eq 0 ]; then
    >&2 echo "Warning: No subset directories found for language '${LANG}' under ${SUBSETS_ROOT}"
    continue
  fi

  for SEED in "${SEEDS[@]}"; do
    CSV_PATH="/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/${LANG}.f1_by_samples_${SEED}.csv"

    for dname in "${SUBSET_DIRS[@]}"; do
      fullpath="${SUBSETS_ROOT%/}/${dname}"
      NUM_SAMPLES=$(extract_num_samples "${dname}")

      # Skip if CSV already has this NUM_SAMPLES
      if csv_contains_num "${CSV_PATH}" "${NUM_SAMPLES}"; then
        continue
      fi

      # This combination is NOT done yet -> print one line (TSV)
      printf '%s\t%s\t%s\t%s\n' "${LANG}" "${SEED}" "${fullpath}" "${NUM_SAMPLES}"
    done
  done
done
