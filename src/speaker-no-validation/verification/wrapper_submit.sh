#!/bin/bash
set -euo pipefail

# Paths
BASE_DATA_DIR="/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv"
SUMMARY_CSV="/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/speaker_matrix_bilingual.csv"
RUN_SCRIPT="./run_sid_sv2.sh"

# Seeds
SEEDS=(42 43 44 45 46 47 48 49 50 51)

# Tamaño del paquete de idiomas por sbatch
BUNDLE_LANGS="${BUNDLE_LANGS:-1}"

# --- Extraer idiomas de los ficheros *.train.tsv ---
LANGS=()
for f in "$BASE_DATA_DIR"/*.train.tsv; do
    fname=$(basename "$f")
    lang=${fname%%.*}   # nombre antes del primer punto
    LANGS+=("$lang")
done

echo "[launcher] Found ${#LANGS[@]} languages: ${LANGS[*]}"
echo "[launcher] Bundle size per sbatch: ${BUNDLE_LANGS}"

# --- Función robusta para verificar si ya existe la entrada en CSV ---
is_already_in_csv() {
    local model_id="$1"
    local seed="$2"

    if [ ! -f "$SUMMARY_CSV" ]; then
        echo "0"
        return
    fi

    local exists
    exists=$(awk -F',' -v id="$model_id" -v s="$seed" '
        NR>1 && $1==id && $5==s {print 1; exit}
    ' "$SUMMARY_CSV" || echo 0)

    if [ "$exists" == "1" ]; then
        echo "1"
    else
        echo "0"
    fi
}


# --- Función para dividir array en chunks ---
chunk_and_emit() {
    local chunk_size="$1"; shift
    local -a arr=("$@")
    local total=${#arr[@]}
    local i=0
    while [ $i -lt $total ]; do
        local end=$(( i + chunk_size - 1 ))
        if [ $end -ge $(( total - 1 )) ]; then
            end=$(( total - 1 ))
        fi
        local j
        local first=true
        local out=""
        for (( j=i; j<=end; j++ )); do
            if [ "$first" = true ]; then
                out="${arr[j]}"
                first=false
            else
                out="${out},${arr[j]}"
            fi
        done
        echo "$out"
        i=$(( end + 1 ))
    done
}

# --- Loop sobre seeds ---
for SEED in "${SEEDS[@]}"; do
    echo
    echo "=== Seed ${SEED} ==="

    # Construir lista de idiomas pendientes
    REMAIN_LANGS=()
    for lang in "${LANGS[@]}"; do
        if [ "$(is_already_in_csv "$lang" "$SEED")" == "1" ]; then
            echo "[launcher] Skipping ${lang} for seed ${SEED}: already in CSV"
            continue
        fi
        REMAIN_LANGS+=("$lang")
    done

    if [ ${#REMAIN_LANGS[@]} -eq 0 ]; then
        echo "[launcher] No remaining languages for seed ${SEED}, skipping."
        continue
    fi

    echo "[launcher] Remaining ${#REMAIN_LANGS[@]} languages for seed ${SEED}: ${REMAIN_LANGS[*]}"

    # --- Dividir en chunks y lanzar sbatch por chunk ---
    while read -r chunk; do
        [ -z "$chunk" ] && continue

        TIMESTAMP=$(date +"%y%m%d_%H%M%S")
        TMP_SBATCH=$(mktemp /tmp/sbatch_${SEED}_chunk_XXXX.sh)

        # Cabecera sbatch
        cat <<EOT > "$TMP_SBATCH"
#!/bin/bash
#SBATCH --account bsc88
#SBATCH --qos acc_bscls
#SBATCH --nodes 1
#SBATCH --cpus-per-task 20
#SBATCH --gres gpu:1
#SBATCH --time 00-01:00:00
#SBATCH --exclusive
#SBATCH --exclude=as04r2b01,as02r3b04,as07r1b02
#SBATCH --job-name=sid_sv_${SEED}_batch
#SBATCH --output=/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/logs/speaker/%x_%j.log
#SBATCH --error=/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/logs/speaker/%x_%j.err
#SBATCH --export=ALL,CUBLAS_WORKSPACE_CONFIG=:4096:8,PYTHONHASHSEED=42,OMP_NUM_THREADS=1,MKL_NUM_THREADS=1

set -euo pipefail
EOT

        # Añadir comandos para cada idioma del chunk
        IFS=',' read -ra LANGS_IN_CHUNK <<< "$chunk"
        for LANG in "${LANGS_IN_CHUNK[@]}"; do
            DATA_DIR_LANG="${BASE_DATA_DIR}"
            echo "echo 'Running ${LANG} for SEED=${SEED}'" >> "$TMP_SBATCH"
            echo "export DATA_DIR='${DATA_DIR_LANG}'" >> "$TMP_SBATCH"
            echo "$RUN_SCRIPT ${LANG} ${SEED} ${TIMESTAMP}" >> "$TMP_SBATCH"
        done

        sbatch_out=$(sbatch "$TMP_SBATCH")
        echo "[launcher] Submitted sbatch for seed=${SEED}, chunk=[${chunk}]: $sbatch_out"

        sleep 0.3
    done < <(chunk_and_emit "$BUNDLE_LANGS" "${REMAIN_LANGS[@]}")
done

echo
echo "[launcher] All submissions done."
