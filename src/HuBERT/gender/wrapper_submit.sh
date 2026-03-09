#!/bin/bash
set -euo pipefail

# Paths
BASE_DATA_DIR="/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__gender_id/06_pairs_gender/tsv"
MATRIX_CSV="/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/gender_matrix.csv"
SUBMIT_SCRIPT="/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/submit_slurm.sh"

# Seeds
SEEDS=(42 43 44 45 46 47 48 49 50 51)

# Tamaño del paquete de idiomas por sbatch (se puede exportar BUNDLE_LANGS antes de llamar al script)
BUNDLE_LANGS="${BUNDLE_LANGS:-5}"

# Extraer idiomas de los ficheros train
LANGS=()
for f in "$BASE_DATA_DIR"/*.train.tsv; do
    fname=$(basename "$f")
    lang=${fname%%.*}   # antes del primer punto
    LANGS+=("$lang")
done

echo "Found ${#LANGS[@]} languages: ${LANGS[*]}"
echo "Bundle size (idiomas por sbatch): ${BUNDLE_LANGS}"

# Función auxiliar: comprueba en MATRIX_CSV si existe entrada con model_id==lang, seed==seed y type=="dual"
is_already_dual_for_seed() {
    local lang="$1"
    local seed="$2"
    if [ ! -f "$MATRIX_CSV" ]; then
        echo "0"
        return
    fi
    # NR>1 para saltar header, $1=model_id, $2=type, $5=seed en el formato esperado: model_id,type,lang_src,lang_tgt,seed,f1
    local exists
    exists=$(awk -F',' -v l="$lang" -v s="$seed" 'NR>1 && $1==l && $2=="dual" && $5==s {print 1; exit}' "$MATRIX_CSV" || echo 0)
    if [ "$exists" == "1" ]; then
        echo "1"
    else
        echo "0"
    fi
}

# Divide array en chunks de tamaño n, devuelve cada chunk como línea con comas
# args: chunk_size, array...
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

# Loop sobre seeds
for seed in "${SEEDS[@]}"; do
    echo
    echo "=== Seed ${seed} ==="

    # Construir la lista de idiomas que faltan para esta seed
    REMAIN_LANGS=()
    for lang in "${LANGS[@]}"; do
        # Si existe ya un registro tipo "dual" para este model_id y seed, lo saltamos
        if [ "$(is_already_dual_for_seed "$lang" "$seed")" == "1" ]; then
            echo "Skipping ${lang} for seed ${seed}: already present as dual in ${MATRIX_CSV}"
            continue
        fi
        # Si no existe, añadimos a la lista de pendientes
        REMAIN_LANGS+=("$lang")
    done

    if [ ${#REMAIN_LANGS[@]} -eq 0 ]; then
        echo "No remaining languages for seed ${seed}, skipping."
        continue
    fi

    echo "Remaining ${#REMAIN_LANGS[@]} languages for seed ${seed}: ${REMAIN_LANGS[*]}"

    # Dividir en chunks y lanzar un sbatch por chunk
    while read -r chunk; do
        # chunk es una cadena "lang1,lang2,..."
        # Protección: si chunk vacío, saltar
        if [ -z "${chunk}" ]; then
            continue
        fi

        # codificar en base64 sin saltos de línea (portable)
        chunk_b64=$(printf '%s' "$chunk" | base64 | tr -d '\n')

        echo "Submitting seed ${seed} batch with LANGS_STR='${chunk}' (encoded)"
        sbatch --export=ALL,DATA_DIR="$BASE_DATA_DIR",LANGS_B64="$chunk_b64",SEED="$seed" \
            "$SUBMIT_SCRIPT"

    done < <(chunk_and_emit "$BUNDLE_LANGS" "${REMAIN_LANGS[@]}")
done

echo
echo "All submissions finished."
