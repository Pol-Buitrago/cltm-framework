#!/bin/bash
#SBATCH --account=bsc88
#SBATCH --qos=acc_bscls

############# Obligatorias #######################
#SBATCH --time=01-00:00:00

################# HOST ###########################
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=80          # 4 GPUs x 20 CPUs
#SBATCH --gres=gpu:4

################ Logging #########################
#SBATCH --job-name=hubert_ft_asr
#SBATCH --verbose
#SBATCH --output=/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/asr/logs/%x/%j.txt
#SBATCH --error=/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/asr/logs/%x/%j.err

################ Job notifications #########################
#SBATCH --mail-type=none
#SBATCH --mail-user=pol.buitrago@bsc.es

date +%Y-%m-%d_%H:%M:%S

module load intel mkl impi hdf5
module load cuda/12.1
module load cudnn/9.1.0-cuda12
module load nccl/2.20.5
module load cusparselt/0.6.2-cuda12

# Activate environment
source /gpfs/projects/bsc88/speech/speech_representations/environments/hubert_eval_asr/bin/activate
cd /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/asr

export GPUS_PER_NODE=4
export SRUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}
export LD_LIBRARY_PATH=/apps/ACC/CUDNN/9.1.0/cuda12/lib/:$LD_LIBRARY_PATH
export TRANSFORMERS_OFFLINE=1

################ Dataset paths ####################
export CV_BASE_DATA_DIR="${CV_BASE_DATA_DIR:-/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__asr/01_preprocessed/tsv}"
export CV_LANG="${CV_LANG:-sl}"

export CV_TRAIN_TSV=${CV_BASE_DATA_DIR}/${CV_LANG}.train.tsv
export CV_DEV_TSV=${CV_BASE_DATA_DIR}/${CV_LANG}.dev.tsv
export CV_TEST_TSV=${CV_BASE_DATA_DIR}/${CV_LANG}.test.tsv

######################
#### Set network #####
######################
head_node_ip=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)

export LAUNCHER="accelerate launch \
    --num_processes ${GPUS_PER_NODE} \
    --num_machines 1 \
    --machine_rank 0 \
    --multi-gpu \
    --main_process_ip ${head_node_ip} \
    --main_process_port 29500 \
    --rdzv_backend c10d \
"

export SCRIPT="/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/asr/ft_asr.py"

########################################
# Deterministic output directory (shell)
########################################
TIMESTAMP=$(date +"%Y_%m_%d_%H_%M_%S")
MODEL_NAME="mHuBERT-147"

FT_BASE="/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/asr/finetuned"
FT_OUTPUT_DIR="${FT_BASE}/${MODEL_NAME}_ft_for_asr/${TIMESTAMP}_job${SLURM_JOB_ID}"

mkdir -p "${FT_OUTPUT_DIR}"
echo "FT_OUTPUT_DIR=${FT_OUTPUT_DIR}"

########################################
# Script arguments
########################################
export SCRIPT_ARGS=" \
    --pre_trained_hf_model /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/asr/${MODEL_NAME} \
    --fine_tuned_output_folder ${FT_OUTPUT_DIR} \
    --dataloader_folder_dir /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/asr \
    --num_train_epochs 15 \
    --learning_rate 1e-5 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.01 \
    --weight_decay 0.0001 \
    --feature_extractor_do_normalize original_config \
    --eval_accumulation_steps 16 \
    --per_device_eval_batch_size 4 \
    --create_vocab_file \
    --vocab_file_path ${FT_OUTPUT_DIR}/vocab.txt \
    --model_freezing feature_encoder \
    --set_seed \
    --seed ${ASR_SEED:-42} \
"
#--eval_accumulation_steps 16

export CMD="${LAUNCHER} ${SCRIPT} ${SCRIPT_ARGS}"

########################################
# Run training
########################################
srun bash -c "${CMD}"

date +%Y-%m-%d_%H:%M:%S

########################################
# Read final TEST eval_wer (simple & strict)
########################################
CSV="${FT_OUTPUT_DIR}/df_full_evaluation_metrics.csv"

if [ ! -f "${CSV}" ]; then
    echo "ERROR: metrics CSV not found: ${CSV}"
    exit 1
fi

# Column order is known; eval_wer is column 4 (tab-separated)
TEST_EVAL_WER=$(awk -F '\t' 'NR==2 {print $4}' "${CSV}")

echo "FINAL_EVAL_WER=${TEST_EVAL_WER}"

# ------------------------------
# Compute number of train samples (extract from path, fallback to wc -l)
# ------------------------------
NUM_SAMPLES=0

# Try 1: extract from directory name containing the train TSV
if [ -n "${CV_TRAIN_TSV:-}" ]; then
    TSV_DIR=$(dirname "${CV_TRAIN_TSV}")
    BASE_DIR=$(basename "${TSV_DIR}")
    echo "DEBUG: TSV_DIR=${TSV_DIR}, BASE_DIR=${BASE_DIR}"

    if [[ "${BASE_DIR}" =~ ^train_([0-9]+)$ ]]; then
        NUM_SAMPLES=${BASH_REMATCH[1]}
        echo "DEBUG: extracted NUM_SAMPLES from base dir: ${NUM_SAMPLES}"
    else
        # Try to extract digits after 'train_' anywhere in the path (more permissive)
        if [[ "${CV_TRAIN_TSV}" =~ train_([0-9]+) ]]; then
            NUM_SAMPLES=${BASH_REMATCH[1]}
            echo "DEBUG: extracted NUM_SAMPLES from CV_TRAIN_TSV path: ${NUM_SAMPLES}"
        fi
    fi
fi

# Try 2: if SUBSET_NAME env var provided (wrapper may set it), try using it
if [ "${NUM_SAMPLES}" -eq 0 ] && [ -n "${SUBSET_NAME:-}" ]; then
    echo "DEBUG: SUBSET_NAME present: ${SUBSET_NAME}"
    if [[ "${SUBSET_NAME}" =~ ^train_([0-9]+)$ ]]; then
        NUM_SAMPLES=${BASH_REMATCH[1]}
        echo "DEBUG: extracted NUM_SAMPLES from SUBSET_NAME: ${NUM_SAMPLES}"
    elif [[ "${SUBSET_NAME}" =~ ([0-9]+) ]]; then
        NUM_SAMPLES=${BASH_REMATCH[1]}
        echo "DEBUG: extracted NUM_SAMPLES digits from SUBSET_NAME: ${NUM_SAMPLES}"
    fi
fi

# Fallback: if still zero, count lines in the TSV (minus header)
if [ "${NUM_SAMPLES}" -eq 0 ]; then
    if [ -f "${CV_TRAIN_TSV}" ]; then
        NUM_LINES=$(wc -l < "${CV_TRAIN_TSV}" | tr -d ' ')
        if [ -n "${NUM_LINES}" ] && [ "${NUM_LINES}" -gt 0 ]; then
            NUM_SAMPLES=$((NUM_LINES - 1))
            if [ "${NUM_SAMPLES}" -lt 0 ]; then NUM_SAMPLES=0; fi
            echo "DEBUG: fallback NUM_SAMPLES from wc -l: ${NUM_SAMPLES}"
        else
            echo "WARNING: couldn't count lines in ${CV_TRAIN_TSV}"
            NUM_SAMPLES=0
        fi
    else
        echo "WARNING: CV_TRAIN_TSV not found: ${CV_TRAIN_TSV}"
        NUM_SAMPLES=0
    fi
fi

# Ensure NUM_SAMPLES is a non-negative integer
if ! [[ "${NUM_SAMPLES}" =~ ^[0-9]+$ ]]; then
    echo "WARNING: NUM_SAMPLES is not numeric (${NUM_SAMPLES}), setting to 0"
    NUM_SAMPLES=0
fi

echo "NUM_SAMPLES=${NUM_SAMPLES}"

# ------------------------------
# Write result to central CSV
# ------------------------------
# Directorio de resultados central (por idioma y seed)
RESULTS_DIR="${FT_BASE}/results"
mkdir -p "${RESULTS_DIR}"

# Nombre de fichero: ej. en.wer_by_samples_42.cv
OUTFILE="/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/asr/${CV_LANG}.wer_by_samples_${ASR_SEED}.csv"
LOCKFILE="${OUTFILE}.lock"

# Si no existe, escribir cabecera
if [ ! -f "${OUTFILE}" ]; then
    echo "wer,num_samples" > "${OUTFILE}"
fi

# Append de manera segura usando flock (crea lockfile si necesario)
# Usamos descriptor 200 para el bloqueo
(
  # abrir descriptor para flock
  flock -x 200

  # escribir la línea con wer y num_samples
  # si TEST_EVAL_WER está vacío, escribe "NA"
  if [ -z "${TEST_EVAL_WER}" ]; then
      printf "%s,%s\n" "NA" "${NUM_SAMPLES}" >> "${OUTFILE}"
  else
      printf "%s,%s\n" "${TEST_EVAL_WER}" "${NUM_SAMPLES}" >> "${OUTFILE}"
  fi

) 200>"${LOCKFILE}"

echo "Appended result to ${OUTFILE}"

date +%Y-%m-%d_%H:%M:%S

