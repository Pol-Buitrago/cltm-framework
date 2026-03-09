#!/bin/bash
set -euo pipefail

RUN_SCRIPT="./run_sid_sv.sh"
LANGS=("ca" "en" "eo" "es" "eu" "hu" "ja" "ka" "ru" "sw" "th" "zh-CN")
SEEDS=(41 42 43)
SUBSETS_ROOT="/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/experimental/subsets_speaker"
SUMMARY_ROOT="/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs"

echo "[launcher] Starting submissions (one sbatch per seed+subset, all langs inside)"

for SEED in "${SEEDS[@]}"; do
    SUBSET_DIR="${SUBSETS_ROOT}/en"
    mapfile -t DATA_DIRS < <(find "${SUBSET_DIR}" -maxdepth 1 -type d -name "subset_*" -printf '%p\n' 2>/dev/null | sort -V)

    for DATA_DIR in "${DATA_DIRS[@]}"; do
        subset_base=$(basename "$DATA_DIR")
        if [[ "$subset_base" =~ n([0-9]+) ]]; then
            NUM_SAMPLES="${BASH_REMATCH[1]}"
        else
            NUM_SAMPLES="nan"
        fi

        TIMESTAMP=$(date +"%y%m%d_%H%M%S")
        TMP_SBATCH=$(mktemp /tmp/sbatch_${SEED}_${subset_base}_XXXX.sh)

        # Cabecera
        cat <<EOT > "$TMP_SBATCH"
#!/bin/bash
#SBATCH --account bsc88
#SBATCH --qos acc_bscls
#SBATCH --nodes 1
#SBATCH --cpus-per-task 20
#SBATCH --gres gpu:1
#SBATCH --time 00-10:00:00
#SBATCH --exclude=as04r2b01,as02r3b04,as07r1b02
#SBATCH --job-name=sid_sv_${SEED}_${subset_base}
#SBATCH --output=/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/logs/speaker/subsets2/%x_%j.log
#SBATCH --error=/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/logs/speaker/subsets2/%x_%j.err
#SBATCH --export=ALL,CUBLAS_WORKSPACE_CONFIG=:4096:8,PYTHONHASHSEED=42,OMP_NUM_THREADS=1,MKL_NUM_THREADS=1

set -euo pipefail
EOT

        # Contador para saber si hay algo que ejecutar
        commands_added=0

        # Bucle por idiomas
        for LANG in "${LANGS[@]}"; do
            DATA_DIR_LANG="${SUBSETS_ROOT}/${LANG}/${subset_base}"
            if [ ! -d "$DATA_DIR_LANG" ]; then
                echo "[launcher] Skipping LANG=${LANG}, subset=${subset_base} (folder does not exist)"
                continue
            fi

            CSV_FILE="${SUMMARY_ROOT}/${LANG}.eer_by_samples_${SEED}.csv"
            if [ -f "$CSV_FILE" ] && grep -qE "^[^,]*,[^,]*,[^,]*,${NUM_SAMPLES}$" "$CSV_FILE"; then
                echo "[launcher] Skipping LANG=${LANG}, subset=${subset_base} (already in CSV)"
                continue
            fi

            # Si llegamos aquí, se ejecuta
            commands_added=$((commands_added + 1))
            echo "echo 'Running ${LANG} for SEED=${SEED}, subset=${subset_base}'" >> "$TMP_SBATCH"
            echo "export DATA_DIR='${DATA_DIR_LANG}'" >> "$TMP_SBATCH"
            echo "$RUN_SCRIPT ${LANG} ${SEED} ${TIMESTAMP}" >> "$TMP_SBATCH"
        done

        # ---- SOLO SUBMIT SI HAY TRABAJO ----
        if (( commands_added > 0 )); then
            sbatch_out=$(sbatch "$TMP_SBATCH")
            echo "[launcher] Submitted sbatch for SEED=${SEED}, subset=${subset_base}: $sbatch_out"
        else
            echo "[launcher] No languages pending for SEED=${SEED}, subset=${subset_base}. Not submitting."
            rm -f "$TMP_SBATCH"
        fi

        sleep 0.3
    done
done

echo "[launcher] All submissions done."
