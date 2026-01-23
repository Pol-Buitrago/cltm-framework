#!/bin/bash
#SBATCH --account=bsc88
#SBATCH --qos=acc_bscls

############# Obligatorias #######################
#SBATCH --time=00-01:00:00

################# HOST ###########################
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=80          # 4 GPUs x 20 CPUs
#SBATCH --gres=gpu:4

################ Logging #########################
#SBATCH --job-name=hubert_ft_asr
#SBATCH --verbose
#SBATCH --output=/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/asr/logs/%x/%j.txt
#SBATCH --error=/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/asr/logs/%x_%j.err

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
export CV_LANG="${CV_LANG:-zh-CN}"

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
    --num_train_epochs 3 \
    --learning_rate 1e-5 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.01 \
    --weight_decay 0.0001 \
    --feature_extractor_do_normalize original_config \
    --eval_accumulation_steps 16 \
    --per_device_eval_batch_size 1 \
    --create_vocab_file \
    --vocab_file_path ${FT_OUTPUT_DIR}/vocab.txt \
    --model_freezing feature_encoder \
    --set_seed \
    --seed ${ASR_SEED:-42} \
"

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
