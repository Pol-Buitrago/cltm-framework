#!/bin/bash
#SBATCH --account bsc88
#SBATCH --qos acc_bscls
#SBATCH --nodes 1
#SBATCH --gres gpu:1
#SBATCH --cpus-per-task 20
#SBATCH --time 1-00:00:00
#SBATCH --job-name=speechLLM_subsets
#SBATCH --output=/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/logs/speechLLM_%j.log
#SBATCH --error=/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/logs/speechLLM_%j.err

set -e

echo "=== Activating conda ==="

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /gpfs/projects/bsc88/speech/mm_s2st/scripts/paraling/gender_id_hubert/env/

echo "=== Running speechLLM subset training ==="

python sv_speechLLM_batch_subsets_seeded.py \
    --subsets_root /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/experimental/subsets_speaker \
    --out_root /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/subsets/speechLLM/speaker/seeds \
    --langs ja,ka,ru,sw,th,zh-CN \
    --seeds 42 \
    --epochs 1 \
    --num_workers 4

echo "=== Done ==="
