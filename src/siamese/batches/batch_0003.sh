#!/bin/bash
#SBATCH --account bsc88
#SBATCH --qos acc_bscls
#SBATCH --nodes 1
#SBATCH --gres gpu:1
#SBATCH --cpus-per-task 20
#SBATCH --time 0-10:00:00
#SBATCH --job-name=siamese_train
#SBATCH --output=/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/logs/siamese_train_%j.log
#SBATCH --error=/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/logs/siamese_train_%j.err

set -e

echo "=== Activating conda ==="

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /gpfs/projects/bsc88/speech/mm_s2st/scripts/paraling/gender_id_hubert/env/

echo "=== Running siamese training ==="

set -euo pipefail

module purge
# Carga tus módulos / activa conda aquí si hace falta. Ej:
# module load cuda/xx
# source /path/to/conda.sh && conda activate tu_env

echo '--- TASK start: model=yue_ta seed=42'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue_ta.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'yue_ta' --type 'dual' --lang_src 'yue' --lang_tgt 'ta' --seed 42 --epochs 1 --batch_size 128
echo '--- TASK end: model=yue_ta seed=42'

echo '--- TASK start: model=yue_ta seed=43'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue_ta.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'yue_ta' --type 'dual' --lang_src 'yue' --lang_tgt 'ta' --seed 43 --epochs 1 --batch_size 128
echo '--- TASK end: model=yue_ta seed=43'

echo '--- TASK start: model=yue_ur seed=43'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue_ur.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'yue_ur' --type 'dual' --lang_src 'yue' --lang_tgt 'ur' --seed 43 --epochs 1 --batch_size 128
echo '--- TASK end: model=yue_ur seed=43'

echo '--- TASK start: model=yue_ur seed=44'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue_ur.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'yue_ur' --type 'dual' --lang_src 'yue' --lang_tgt 'ur' --seed 44 --epochs 1 --batch_size 128
echo '--- TASK end: model=yue_ur seed=44'

echo '--- TASK start: model=yue_ur seed=45'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue_ur.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'yue_ur' --type 'dual' --lang_src 'yue' --lang_tgt 'ur' --seed 45 --epochs 1 --batch_size 128
echo '--- TASK end: model=yue_ur seed=45'

echo '--- TASK start: model=yue_ur seed=46'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue_ur.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'yue_ur' --type 'dual' --lang_src 'yue' --lang_tgt 'ur' --seed 46 --epochs 1 --batch_size 128
echo '--- TASK end: model=yue_ur seed=46'

echo '--- TASK start: model=yue_ur seed=47'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue_ur.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'yue_ur' --type 'dual' --lang_src 'yue' --lang_tgt 'ur' --seed 47 --epochs 1 --batch_size 128
echo '--- TASK end: model=yue_ur seed=47'

echo '--- TASK start: model=yue_ur seed=48'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue_ur.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'yue_ur' --type 'dual' --lang_src 'yue' --lang_tgt 'ur' --seed 48 --epochs 1 --batch_size 128
echo '--- TASK end: model=yue_ur seed=48'

echo '--- TASK start: model=yue_ur seed=49'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue_ur.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'yue_ur' --type 'dual' --lang_src 'yue' --lang_tgt 'ur' --seed 49 --epochs 1 --batch_size 128
echo '--- TASK end: model=yue_ur seed=49'

echo '--- TASK start: model=yue_ur seed=50'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue_ur.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'yue_ur' --type 'dual' --lang_src 'yue' --lang_tgt 'ur' --seed 50 --epochs 1 --batch_size 128
echo '--- TASK end: model=yue_ur seed=50'

echo '--- TASK start: model=yue_ur seed=51'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue_ur.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'yue_ur' --type 'dual' --lang_src 'yue' --lang_tgt 'ur' --seed 51 --epochs 1 --batch_size 128
echo '--- TASK end: model=yue_ur seed=51'

echo '--- TASK start: model=yue_uz seed=42'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue_uz.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'yue_uz' --type 'dual' --lang_src 'yue' --lang_tgt 'uz' --seed 42 --epochs 1 --batch_size 128
echo '--- TASK end: model=yue_uz seed=42'

echo '--- TASK start: model=yue_uz seed=43'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue_uz.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'yue_uz' --type 'dual' --lang_src 'yue' --lang_tgt 'uz' --seed 43 --epochs 1 --batch_size 128
echo '--- TASK end: model=yue_uz seed=43'

echo '--- TASK start: model=yue_uz seed=44'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue_uz.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'yue_uz' --type 'dual' --lang_src 'yue' --lang_tgt 'uz' --seed 44 --epochs 1 --batch_size 128
echo '--- TASK end: model=yue_uz seed=44'

echo '--- TASK start: model=yue_uz seed=45'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue_uz.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'yue_uz' --type 'dual' --lang_src 'yue' --lang_tgt 'uz' --seed 45 --epochs 1 --batch_size 128
echo '--- TASK end: model=yue_uz seed=45'

echo '--- TASK start: model=yue_uz seed=46'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue_uz.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'yue_uz' --type 'dual' --lang_src 'yue' --lang_tgt 'uz' --seed 46 --epochs 1 --batch_size 128
echo '--- TASK end: model=yue_uz seed=46'

echo '--- TASK start: model=yue_uz seed=47'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue_uz.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'yue_uz' --type 'dual' --lang_src 'yue' --lang_tgt 'uz' --seed 47 --epochs 1 --batch_size 128
echo '--- TASK end: model=yue_uz seed=47'

echo '--- TASK start: model=yue_uz seed=48'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue_uz.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'yue_uz' --type 'dual' --lang_src 'yue' --lang_tgt 'uz' --seed 48 --epochs 1 --batch_size 128
echo '--- TASK end: model=yue_uz seed=48'

echo '--- TASK start: model=yue_uz seed=49'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue_uz.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'yue_uz' --type 'dual' --lang_src 'yue' --lang_tgt 'uz' --seed 49 --epochs 1 --batch_size 128
echo '--- TASK end: model=yue_uz seed=49'

echo '--- TASK start: model=yue_zh-TW seed=46'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue_zh-TW.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'yue_zh-TW' --type 'dual' --lang_src 'yue' --lang_tgt 'zh-TW' --seed 46 --epochs 1 --batch_size 128
echo '--- TASK end: model=yue_zh-TW seed=46'

echo '--- TASK start: model=yue_zh-TW seed=47'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue_zh-TW.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'yue_zh-TW' --type 'dual' --lang_src 'yue' --lang_tgt 'zh-TW' --seed 47 --epochs 1 --batch_size 128
echo '--- TASK end: model=yue_zh-TW seed=47'

echo '--- TASK start: model=yue_zh-TW seed=48'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue_zh-TW.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'yue_zh-TW' --type 'dual' --lang_src 'yue' --lang_tgt 'zh-TW' --seed 48 --epochs 1 --batch_size 128
echo '--- TASK end: model=yue_zh-TW seed=48'

echo '--- TASK start: model=yue_zh-TW seed=49'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue_zh-TW.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'yue_zh-TW' --type 'dual' --lang_src 'yue' --lang_tgt 'zh-TW' --seed 49 --epochs 1 --batch_size 128
echo '--- TASK end: model=yue_zh-TW seed=49'

echo '--- TASK start: model=yue_zh-TW seed=50'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue_zh-TW.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'yue_zh-TW' --type 'dual' --lang_src 'yue' --lang_tgt 'zh-TW' --seed 50 --epochs 1 --batch_size 128
echo '--- TASK end: model=yue_zh-TW seed=50'

echo '--- TASK start: model=yue_zh-TW seed=51'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue_zh-TW.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'yue_zh-TW' --type 'dual' --lang_src 'yue' --lang_tgt 'zh-TW' --seed 51 --epochs 1 --batch_size 128
echo '--- TASK end: model=yue_zh-TW seed=51'

echo '--- TASK start: model=zh-CN_ab seed=42'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN_ab.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'zh-CN_ab' --type 'dual' --lang_src 'zh-CN' --lang_tgt 'ab' --seed 42 --epochs 1 --batch_size 128
echo '--- TASK end: model=zh-CN_ab seed=42'

echo '--- TASK start: model=zh-CN_ab seed=43'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN_ab.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'zh-CN_ab' --type 'dual' --lang_src 'zh-CN' --lang_tgt 'ab' --seed 43 --epochs 1 --batch_size 128
echo '--- TASK end: model=zh-CN_ab seed=43'

echo '--- TASK start: model=zh-CN_ab seed=44'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN_ab.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'zh-CN_ab' --type 'dual' --lang_src 'zh-CN' --lang_tgt 'ab' --seed 44 --epochs 1 --batch_size 128
echo '--- TASK end: model=zh-CN_ab seed=44'

echo '--- TASK start: model=zh-CN_ab seed=45'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN_ab.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'zh-CN_ab' --type 'dual' --lang_src 'zh-CN' --lang_tgt 'ab' --seed 45 --epochs 1 --batch_size 128
echo '--- TASK end: model=zh-CN_ab seed=45'

echo '--- TASK start: model=zh-CN_ab seed=46'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN_ab.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'zh-CN_ab' --type 'dual' --lang_src 'zh-CN' --lang_tgt 'ab' --seed 46 --epochs 1 --batch_size 128
echo '--- TASK end: model=zh-CN_ab seed=46'

echo '--- TASK start: model=zh-CN_ab seed=47'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN_ab.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'zh-CN_ab' --type 'dual' --lang_src 'zh-CN' --lang_tgt 'ab' --seed 47 --epochs 1 --batch_size 128
echo '--- TASK end: model=zh-CN_ab seed=47'

echo '--- TASK start: model=zh-CN_ab seed=48'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN_ab.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'zh-CN_ab' --type 'dual' --lang_src 'zh-CN' --lang_tgt 'ab' --seed 48 --epochs 1 --batch_size 128
echo '--- TASK end: model=zh-CN_ab seed=48'

echo '--- TASK start: model=zh-CN_be seed=46'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN_be.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'zh-CN_be' --type 'dual' --lang_src 'zh-CN' --lang_tgt 'be' --seed 46 --epochs 1 --batch_size 128
echo '--- TASK end: model=zh-CN_be seed=46'

echo '--- TASK start: model=zh-CN_ca seed=45'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN_ca.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'zh-CN_ca' --type 'dual' --lang_src 'zh-CN' --lang_tgt 'ca' --seed 45 --epochs 1 --batch_size 128
echo '--- TASK end: model=zh-CN_ca seed=45'

echo '--- TASK start: model=zh-CN_ca seed=46'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN_ca.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'zh-CN_ca' --type 'dual' --lang_src 'zh-CN' --lang_tgt 'ca' --seed 46 --epochs 1 --batch_size 128
echo '--- TASK end: model=zh-CN_ca seed=46'

echo '--- TASK start: model=zh-CN_ca seed=47'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN_ca.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'zh-CN_ca' --type 'dual' --lang_src 'zh-CN' --lang_tgt 'ca' --seed 47 --epochs 1 --batch_size 128
echo '--- TASK end: model=zh-CN_ca seed=47'

echo '--- TASK start: model=zh-CN_ca seed=48'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN_ca.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'zh-CN_ca' --type 'dual' --lang_src 'zh-CN' --lang_tgt 'ca' --seed 48 --epochs 1 --batch_size 128
echo '--- TASK end: model=zh-CN_ca seed=48'

echo '--- TASK start: model=zh-CN_ca seed=49'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN_ca.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'zh-CN_ca' --type 'dual' --lang_src 'zh-CN' --lang_tgt 'ca' --seed 49 --epochs 1 --batch_size 128
echo '--- TASK end: model=zh-CN_ca seed=49'

echo '--- TASK start: model=zh-CN_ca seed=50'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN_ca.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'zh-CN_ca' --type 'dual' --lang_src 'zh-CN' --lang_tgt 'ca' --seed 50 --epochs 1 --batch_size 128
echo '--- TASK end: model=zh-CN_ca seed=50'

echo '--- TASK start: model=zh-CN_ca seed=51'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN_ca.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'zh-CN_ca' --type 'dual' --lang_src 'zh-CN' --lang_tgt 'ca' --seed 51 --epochs 1 --batch_size 128
echo '--- TASK end: model=zh-CN_ca seed=51'

echo '--- TASK start: model=zh-CN_ckb seed=42'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN_ckb.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'zh-CN_ckb' --type 'dual' --lang_src 'zh-CN' --lang_tgt 'ckb' --seed 42 --epochs 1 --batch_size 128
echo '--- TASK end: model=zh-CN_ckb seed=42'

echo '--- TASK start: model=zh-CN_ckb seed=43'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN_ckb.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'zh-CN_ckb' --type 'dual' --lang_src 'zh-CN' --lang_tgt 'ckb' --seed 43 --epochs 1 --batch_size 128
echo '--- TASK end: model=zh-CN_ckb seed=43'

echo '--- TASK start: model=zh-CN_ckb seed=44'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN_ckb.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'zh-CN_ckb' --type 'dual' --lang_src 'zh-CN' --lang_tgt 'ckb' --seed 44 --epochs 1 --batch_size 128
echo '--- TASK end: model=zh-CN_ckb seed=44'

echo '--- TASK start: model=zh-CN_ckb seed=45'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN_ckb.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'zh-CN_ckb' --type 'dual' --lang_src 'zh-CN' --lang_tgt 'ckb' --seed 45 --epochs 1 --batch_size 128
echo '--- TASK end: model=zh-CN_ckb seed=45'

echo '--- TASK start: model=zh-CN_ckb seed=46'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN_ckb.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'zh-CN_ckb' --type 'dual' --lang_src 'zh-CN' --lang_tgt 'ckb' --seed 46 --epochs 1 --batch_size 128
echo '--- TASK end: model=zh-CN_ckb seed=46'

echo '--- TASK start: model=zh-CN_ckb seed=47'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN_ckb.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'zh-CN_ckb' --type 'dual' --lang_src 'zh-CN' --lang_tgt 'ckb' --seed 47 --epochs 1 --batch_size 128
echo '--- TASK end: model=zh-CN_ckb seed=47'

echo '--- TASK start: model=zh-CN_ckb seed=48'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN_ckb.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'zh-CN_ckb' --type 'dual' --lang_src 'zh-CN' --lang_tgt 'ckb' --seed 48 --epochs 1 --batch_size 128
echo '--- TASK end: model=zh-CN_ckb seed=48'

echo '--- TASK start: model=zh-CN_ckb seed=49'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN_ckb.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'zh-CN_ckb' --type 'dual' --lang_src 'zh-CN' --lang_tgt 'ckb' --seed 49 --epochs 1 --batch_size 128
echo '--- TASK end: model=zh-CN_ckb seed=49'

echo '--- TASK start: model=zh-CN_ckb seed=50'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN_ckb.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'zh-CN_ckb' --type 'dual' --lang_src 'zh-CN' --lang_tgt 'ckb' --seed 50 --epochs 1 --batch_size 128
echo '--- TASK end: model=zh-CN_ckb seed=50'

echo '--- TASK start: model=zh-CN_ckb seed=51'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN_ckb.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'zh-CN_ckb' --type 'dual' --lang_src 'zh-CN' --lang_tgt 'ckb' --seed 51 --epochs 1 --batch_size 128
echo '--- TASK end: model=zh-CN_ckb seed=51'

echo '--- TASK start: model=zh-CN_cs seed=42'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN_cs.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'zh-CN_cs' --type 'dual' --lang_src 'zh-CN' --lang_tgt 'cs' --seed 42 --epochs 1 --batch_size 128
echo '--- TASK end: model=zh-CN_cs seed=42'

echo '--- TASK start: model=zh-CN_cs seed=43'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN_cs.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'zh-CN_cs' --type 'dual' --lang_src 'zh-CN' --lang_tgt 'cs' --seed 43 --epochs 1 --batch_size 128
echo '--- TASK end: model=zh-CN_cs seed=43'

echo '--- TASK start: model=zh-CN_cs seed=44'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN_cs.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'zh-CN_cs' --type 'dual' --lang_src 'zh-CN' --lang_tgt 'cs' --seed 44 --epochs 1 --batch_size 128
echo '--- TASK end: model=zh-CN_cs seed=44'

echo '--- TASK start: model=zh-CN_cs seed=45'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN_cs.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'zh-CN_cs' --type 'dual' --lang_src 'zh-CN' --lang_tgt 'cs' --seed 45 --epochs 1 --batch_size 128
echo '--- TASK end: model=zh-CN_cs seed=45'

echo '--- TASK start: model=zh-CN_cs seed=46'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN_cs.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'zh-CN_cs' --type 'dual' --lang_src 'zh-CN' --lang_tgt 'cs' --seed 46 --epochs 1 --batch_size 128
echo '--- TASK end: model=zh-CN_cs seed=46'

echo '--- TASK start: model=zh-CN_de seed=43'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN_de.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'zh-CN_de' --type 'dual' --lang_src 'zh-CN' --lang_tgt 'de' --seed 43 --epochs 1 --batch_size 128
echo '--- TASK end: model=zh-CN_de seed=43'

echo '--- TASK start: model=zh-CN_de seed=44'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN_de.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'zh-CN_de' --type 'dual' --lang_src 'zh-CN' --lang_tgt 'de' --seed 44 --epochs 1 --batch_size 128
echo '--- TASK end: model=zh-CN_de seed=44'

echo '--- TASK start: model=zh-CN_de seed=45'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN_de.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'zh-CN_de' --type 'dual' --lang_src 'zh-CN' --lang_tgt 'de' --seed 45 --epochs 1 --batch_size 128
echo '--- TASK end: model=zh-CN_de seed=45'

echo '--- TASK start: model=zh-CN_de seed=46'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN_de.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'zh-CN_de' --type 'dual' --lang_src 'zh-CN' --lang_tgt 'de' --seed 46 --epochs 1 --batch_size 128
echo '--- TASK end: model=zh-CN_de seed=46'

echo '--- TASK start: model=zh-CN_de seed=47'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN_de.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'zh-CN_de' --type 'dual' --lang_src 'zh-CN' --lang_tgt 'de' --seed 47 --epochs 1 --batch_size 128
echo '--- TASK end: model=zh-CN_de seed=47'

echo '--- TASK start: model=zh-CN_de seed=48'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN_de.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'zh-CN_de' --type 'dual' --lang_src 'zh-CN' --lang_tgt 'de' --seed 48 --epochs 1 --batch_size 128
echo '--- TASK end: model=zh-CN_de seed=48'

echo '--- TASK start: model=zh-CN_de seed=49'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN_de.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'zh-CN_de' --type 'dual' --lang_src 'zh-CN' --lang_tgt 'de' --seed 49 --epochs 1 --batch_size 128
echo '--- TASK end: model=zh-CN_de seed=49'

echo '--- TASK start: model=zh-CN_de seed=50'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN_de.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'zh-CN_de' --type 'dual' --lang_src 'zh-CN' --lang_tgt 'de' --seed 50 --epochs 1 --batch_size 128
echo '--- TASK end: model=zh-CN_de seed=50'

echo '--- TASK start: model=zh-CN_de seed=51'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN_de.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'zh-CN_de' --type 'dual' --lang_src 'zh-CN' --lang_tgt 'de' --seed 51 --epochs 1 --batch_size 128
echo '--- TASK end: model=zh-CN_de seed=51'

echo '--- TASK start: model=zh-CN_en seed=42'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN_en.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'zh-CN_en' --type 'dual' --lang_src 'zh-CN' --lang_tgt 'en' --seed 42 --epochs 1 --batch_size 128
echo '--- TASK end: model=zh-CN_en seed=42'

echo '--- TASK start: model=zh-CN_en seed=43'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN_en.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'zh-CN_en' --type 'dual' --lang_src 'zh-CN' --lang_tgt 'en' --seed 43 --epochs 1 --batch_size 128
echo '--- TASK end: model=zh-CN_en seed=43'

echo '--- TASK start: model=zh-CN_en seed=44'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN_en.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'zh-CN_en' --type 'dual' --lang_src 'zh-CN' --lang_tgt 'en' --seed 44 --epochs 1 --batch_size 128
echo '--- TASK end: model=zh-CN_en seed=44'

echo '--- TASK start: model=zh-CN_en seed=45'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN_en.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'zh-CN_en' --type 'dual' --lang_src 'zh-CN' --lang_tgt 'en' --seed 45 --epochs 1 --batch_size 128
echo '--- TASK end: model=zh-CN_en seed=45'

echo '--- TASK start: model=zh-CN_en seed=46'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN_en.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'zh-CN_en' --type 'dual' --lang_src 'zh-CN' --lang_tgt 'en' --seed 46 --epochs 1 --batch_size 128
echo '--- TASK end: model=zh-CN_en seed=46'

echo '--- TASK start: model=zh-CN_en seed=47'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN_en.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'zh-CN_en' --type 'dual' --lang_src 'zh-CN' --lang_tgt 'en' --seed 47 --epochs 1 --batch_size 128
echo '--- TASK end: model=zh-CN_en seed=47'

echo '--- TASK start: model=zh-CN_en seed=48'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN_en.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'zh-CN_en' --type 'dual' --lang_src 'zh-CN' --lang_tgt 'en' --seed 48 --epochs 1 --batch_size 128
echo '--- TASK end: model=zh-CN_en seed=48'

echo '--- TASK start: model=zh-CN_en seed=49'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN_en.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'zh-CN_en' --type 'dual' --lang_src 'zh-CN' --lang_tgt 'en' --seed 49 --epochs 1 --batch_size 128
echo '--- TASK end: model=zh-CN_en seed=49'

echo '--- TASK start: model=zh-CN_en seed=50'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN_en.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'zh-CN_en' --type 'dual' --lang_src 'zh-CN' --lang_tgt 'en' --seed 50 --epochs 1 --batch_size 128
echo '--- TASK end: model=zh-CN_en seed=50'

echo '--- TASK start: model=zh-CN_en seed=51'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN_en.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'zh-CN_en' --type 'dual' --lang_src 'zh-CN' --lang_tgt 'en' --seed 51 --epochs 1 --batch_size 128
echo '--- TASK end: model=zh-CN_en seed=51'

echo '--- TASK start: model=zh-CN_eo seed=42'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN_eo.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'zh-CN_eo' --type 'dual' --lang_src 'zh-CN' --lang_tgt 'eo' --seed 42 --epochs 1 --batch_size 128
echo '--- TASK end: model=zh-CN_eo seed=42'

echo '--- TASK start: model=zh-CN_eo seed=43'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN_eo.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'zh-CN_eo' --type 'dual' --lang_src 'zh-CN' --lang_tgt 'eo' --seed 43 --epochs 1 --batch_size 128
echo '--- TASK end: model=zh-CN_eo seed=43'

echo '--- TASK start: model=zh-CN_eo seed=44'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN_eo.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'zh-CN_eo' --type 'dual' --lang_src 'zh-CN' --lang_tgt 'eo' --seed 44 --epochs 1 --batch_size 128
echo '--- TASK end: model=zh-CN_eo seed=44'

echo '--- TASK start: model=zh-CN_eo seed=45'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN_eo.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'zh-CN_eo' --type 'dual' --lang_src 'zh-CN' --lang_tgt 'eo' --seed 45 --epochs 1 --batch_size 128
echo '--- TASK end: model=zh-CN_eo seed=45'

echo '--- TASK start: model=zh-CN_eo seed=46'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN_eo.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'zh-CN_eo' --type 'dual' --lang_src 'zh-CN' --lang_tgt 'eo' --seed 46 --epochs 1 --batch_size 128
echo '--- TASK end: model=zh-CN_eo seed=46'

echo '--- TASK start: model=zh-CN_fa seed=51'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN_fa.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'zh-CN_fa' --type 'dual' --lang_src 'zh-CN' --lang_tgt 'fa' --seed 51 --epochs 1 --batch_size 128
echo '--- TASK end: model=zh-CN_fa seed=51'

echo '--- TASK start: model=zh-CN_fr seed=42'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN_fr.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'zh-CN_fr' --type 'dual' --lang_src 'zh-CN' --lang_tgt 'fr' --seed 42 --epochs 1 --batch_size 128
echo '--- TASK end: model=zh-CN_fr seed=42'

echo '--- TASK start: model=zh-CN_fr seed=43'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN_fr.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'zh-CN_fr' --type 'dual' --lang_src 'zh-CN' --lang_tgt 'fr' --seed 43 --epochs 1 --batch_size 128
echo '--- TASK end: model=zh-CN_fr seed=43'

echo '--- TASK start: model=zh-CN_fr seed=44'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN_fr.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'zh-CN_fr' --type 'dual' --lang_src 'zh-CN' --lang_tgt 'fr' --seed 44 --epochs 1 --batch_size 128
echo '--- TASK end: model=zh-CN_fr seed=44'

echo '--- TASK start: model=zh-CN_fr seed=45'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN_fr.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'zh-CN_fr' --type 'dual' --lang_src 'zh-CN' --lang_tgt 'fr' --seed 45 --epochs 1 --batch_size 128
echo '--- TASK end: model=zh-CN_fr seed=45'

echo '--- TASK start: model=zh-CN_fr seed=46'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN_fr.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'zh-CN_fr' --type 'dual' --lang_src 'zh-CN' --lang_tgt 'fr' --seed 46 --epochs 1 --batch_size 128
echo '--- TASK end: model=zh-CN_fr seed=46'

echo '--- TASK start: model=zh-CN_fr seed=47'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN_fr.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'zh-CN_fr' --type 'dual' --lang_src 'zh-CN' --lang_tgt 'fr' --seed 47 --epochs 1 --batch_size 128
echo '--- TASK end: model=zh-CN_fr seed=47'

echo '--- TASK start: model=zh-CN_fr seed=48'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN_fr.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'zh-CN_fr' --type 'dual' --lang_src 'zh-CN' --lang_tgt 'fr' --seed 48 --epochs 1 --batch_size 128
echo '--- TASK end: model=zh-CN_fr seed=48'

echo '--- TASK start: model=zh-CN_fr seed=49'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN_fr.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'zh-CN_fr' --type 'dual' --lang_src 'zh-CN' --lang_tgt 'fr' --seed 49 --epochs 1 --batch_size 128
echo '--- TASK end: model=zh-CN_fr seed=49'

echo '--- TASK start: model=zh-CN_fr seed=50'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN_fr.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'zh-CN_fr' --type 'dual' --lang_src 'zh-CN' --lang_tgt 'fr' --seed 50 --epochs 1 --batch_size 128
echo '--- TASK end: model=zh-CN_fr seed=50'

echo '--- TASK start: model=zh-CN_fr seed=51'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN_fr.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'zh-CN_fr' --type 'dual' --lang_src 'zh-CN' --lang_tgt 'fr' --seed 51 --epochs 1 --batch_size 128
echo '--- TASK end: model=zh-CN_fr seed=51'

echo '--- TASK start: model=zh-CN_gl seed=42'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN_gl.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'zh-CN_gl' --type 'dual' --lang_src 'zh-CN' --lang_tgt 'gl' --seed 42 --epochs 1 --batch_size 128
echo '--- TASK end: model=zh-CN_gl seed=42'

echo '--- TASK start: model=zh-CN_gl seed=43'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN_gl.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'zh-CN_gl' --type 'dual' --lang_src 'zh-CN' --lang_tgt 'gl' --seed 43 --epochs 1 --batch_size 128
echo '--- TASK end: model=zh-CN_gl seed=43'

echo '--- TASK start: model=zh-CN_gl seed=44'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN_gl.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'zh-CN_gl' --type 'dual' --lang_src 'zh-CN' --lang_tgt 'gl' --seed 44 --epochs 1 --batch_size 128
echo '--- TASK end: model=zh-CN_gl seed=44'

echo '--- TASK start: model=zh-CN_gl seed=45'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN_gl.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'zh-CN_gl' --type 'dual' --lang_src 'zh-CN' --lang_tgt 'gl' --seed 45 --epochs 1 --batch_size 128
echo '--- TASK end: model=zh-CN_gl seed=45'

echo '--- TASK start: model=zh-CN_gl seed=46'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN_gl.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'zh-CN_gl' --type 'dual' --lang_src 'zh-CN' --lang_tgt 'gl' --seed 46 --epochs 1 --batch_size 128
echo '--- TASK end: model=zh-CN_gl seed=46'

echo '--- TASK start: model=zh-CN_gl seed=47'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN_gl.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'zh-CN_gl' --type 'dual' --lang_src 'zh-CN' --lang_tgt 'gl' --seed 47 --epochs 1 --batch_size 128
echo '--- TASK end: model=zh-CN_gl seed=47'

echo '--- TASK start: model=zh-CN_gl seed=48'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN_gl.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'zh-CN_gl' --type 'dual' --lang_src 'zh-CN' --lang_tgt 'gl' --seed 48 --epochs 1 --batch_size 128
echo '--- TASK end: model=zh-CN_gl seed=48'

echo '--- TASK start: model=zh-CN_gl seed=49'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN_gl.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'zh-CN_gl' --type 'dual' --lang_src 'zh-CN' --lang_tgt 'gl' --seed 49 --epochs 1 --batch_size 128
echo '--- TASK end: model=zh-CN_gl seed=49'

echo '--- TASK start: model=zh-CN_gl seed=50'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN_gl.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'zh-CN_gl' --type 'dual' --lang_src 'zh-CN' --lang_tgt 'gl' --seed 50 --epochs 1 --batch_size 128
echo '--- TASK end: model=zh-CN_gl seed=50'

echo '--- TASK start: model=zh-CN_gl seed=51'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN_gl.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'zh-CN_gl' --type 'dual' --lang_src 'zh-CN' --lang_tgt 'gl' --seed 51 --epochs 1 --batch_size 128
echo '--- TASK end: model=zh-CN_gl seed=51'

