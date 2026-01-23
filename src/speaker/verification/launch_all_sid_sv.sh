#!/usr/bin/env bash
set -euo pipefail

# ---------------- CONFIG ----------------
RUN_SCRIPT="./run_sid_sv.sh"

# Lista de idiomas
LANGS=("az")

# Lista de seeds
SEEDS=(41)

# ---------------- MAIN LOOP ----------------
echo "[launcher] Starting sequential SID+SV executions"
echo "[launcher] Script to call: $RUN_SCRIPT"
echo

for LANG in "${LANGS[@]}"; do
  for SEED in "${SEEDS[@]}"; do

    # Timestamp único
    TIMESTAMP=$(date +"%y%m%d_%H%M%S")

    echo "====================================================="
    echo "[launcher] LANG=${LANG}  SEED=${SEED}  TIMESTAMP=${TIMESTAMP}"
    echo "====================================================="

    # Llamada correcta usando argumentos
    bash "$RUN_SCRIPT" "$LANG" "$SEED" "$TIMESTAMP"

    echo
    echo "[launcher] Finished LANG=${LANG}  SEED=${SEED}"
    echo
  done
done

echo "[launcher] All done."
