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

echo '--- TASK start: model=zh-CN_hu seed=42'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN_hu.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'zh-CN_hu' --type 'dual' --lang_src 'zh-CN' --lang_tgt 'hu' --seed 42 --epochs 1 --batch_size 128
echo '--- TASK end: model=zh-CN_hu seed=42'

echo '--- TASK start: model=zh-CN_hu seed=43'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN_hu.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'zh-CN_hu' --type 'dual' --lang_src 'zh-CN' --lang_tgt 'hu' --seed 43 --epochs 1 --batch_size 128
echo '--- TASK end: model=zh-CN_hu seed=43'

echo '--- TASK start: model=zh-CN_it seed=43'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN_it.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'zh-CN_it' --type 'dual' --lang_src 'zh-CN' --lang_tgt 'it' --seed 43 --epochs 1 --batch_size 128
echo '--- TASK end: model=zh-CN_it seed=43'

echo '--- TASK start: model=zh-CN_it seed=44'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN_it.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'zh-CN_it' --type 'dual' --lang_src 'zh-CN' --lang_tgt 'it' --seed 44 --epochs 1 --batch_size 128
echo '--- TASK end: model=zh-CN_it seed=44'

echo '--- TASK start: model=zh-CN_it seed=45'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN_it.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'zh-CN_it' --type 'dual' --lang_src 'zh-CN' --lang_tgt 'it' --seed 45 --epochs 1 --batch_size 128
echo '--- TASK end: model=zh-CN_it seed=45'

echo '--- TASK start: model=zh-CN_it seed=46'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN_it.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'zh-CN_it' --type 'dual' --lang_src 'zh-CN' --lang_tgt 'it' --seed 46 --epochs 1 --batch_size 128
echo '--- TASK end: model=zh-CN_it seed=46'

echo '--- TASK start: model=zh-CN_it seed=47'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN_it.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'zh-CN_it' --type 'dual' --lang_src 'zh-CN' --lang_tgt 'it' --seed 47 --epochs 1 --batch_size 128
echo '--- TASK end: model=zh-CN_it seed=47'

echo '--- TASK start: model=zh-CN_it seed=48'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN_it.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'zh-CN_it' --type 'dual' --lang_src 'zh-CN' --lang_tgt 'it' --seed 48 --epochs 1 --batch_size 128
echo '--- TASK end: model=zh-CN_it seed=48'

echo '--- TASK start: model=zh-CN_it seed=49'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN_it.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'zh-CN_it' --type 'dual' --lang_src 'zh-CN' --lang_tgt 'it' --seed 49 --epochs 1 --batch_size 128
echo '--- TASK end: model=zh-CN_it seed=49'

echo '--- TASK start: model=zh-CN_it seed=50'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN_it.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'zh-CN_it' --type 'dual' --lang_src 'zh-CN' --lang_tgt 'it' --seed 50 --epochs 1 --batch_size 128
echo '--- TASK end: model=zh-CN_it seed=50'

echo '--- TASK start: model=zh-CN_it seed=51'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN_it.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'zh-CN_it' --type 'dual' --lang_src 'zh-CN' --lang_tgt 'it' --seed 51 --epochs 1 --batch_size 128
echo '--- TASK end: model=zh-CN_it seed=51'

echo '--- TASK start: model=zh-CN_ka seed=42'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN_ka.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'zh-CN_ka' --type 'dual' --lang_src 'zh-CN' --lang_tgt 'ka' --seed 42 --epochs 1 --batch_size 128
echo '--- TASK end: model=zh-CN_ka seed=42'

echo '--- TASK start: model=zh-CN_ka seed=43'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN_ka.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'zh-CN_ka' --type 'dual' --lang_src 'zh-CN' --lang_tgt 'ka' --seed 43 --epochs 1 --batch_size 128
echo '--- TASK end: model=zh-CN_ka seed=43'

echo '--- TASK start: model=zh-CN_ka seed=44'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN_ka.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'zh-CN_ka' --type 'dual' --lang_src 'zh-CN' --lang_tgt 'ka' --seed 44 --epochs 1 --batch_size 128
echo '--- TASK end: model=zh-CN_ka seed=44'

echo '--- TASK start: model=zh-CN_ka seed=45'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN_ka.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'zh-CN_ka' --type 'dual' --lang_src 'zh-CN' --lang_tgt 'ka' --seed 45 --epochs 1 --batch_size 128
echo '--- TASK end: model=zh-CN_ka seed=45'

echo '--- TASK start: model=zh-CN_ka seed=46'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN_ka.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'zh-CN_ka' --type 'dual' --lang_src 'zh-CN' --lang_tgt 'ka' --seed 46 --epochs 1 --batch_size 128
echo '--- TASK end: model=zh-CN_ka seed=46'

echo '--- TASK start: model=zh-CN_ka seed=47'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN_ka.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'zh-CN_ka' --type 'dual' --lang_src 'zh-CN' --lang_tgt 'ka' --seed 47 --epochs 1 --batch_size 128
echo '--- TASK end: model=zh-CN_ka seed=47'

echo '--- TASK start: model=zh-CN_ka seed=48'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN_ka.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'zh-CN_ka' --type 'dual' --lang_src 'zh-CN' --lang_tgt 'ka' --seed 48 --epochs 1 --batch_size 128
echo '--- TASK end: model=zh-CN_ka seed=48'

echo '--- TASK start: model=zh-CN_ka seed=49'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN_ka.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'zh-CN_ka' --type 'dual' --lang_src 'zh-CN' --lang_tgt 'ka' --seed 49 --epochs 1 --batch_size 128
echo '--- TASK end: model=zh-CN_ka seed=49'

echo '--- TASK start: model=zh-CN_ka seed=50'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN_ka.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'zh-CN_ka' --type 'dual' --lang_src 'zh-CN' --lang_tgt 'ka' --seed 50 --epochs 1 --batch_size 128
echo '--- TASK end: model=zh-CN_ka seed=50'

echo '--- TASK start: model=zh-CN_ka seed=51'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN_ka.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'zh-CN_ka' --type 'dual' --lang_src 'zh-CN' --lang_tgt 'ka' --seed 51 --epochs 1 --batch_size 128
echo '--- TASK end: model=zh-CN_ka seed=51'

echo '--- TASK start: model=zh-CN_kab seed=42'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN_kab.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'zh-CN_kab' --type 'dual' --lang_src 'zh-CN' --lang_tgt 'kab' --seed 42 --epochs 1 --batch_size 128
echo '--- TASK end: model=zh-CN_kab seed=42'

echo '--- TASK start: model=zh-CN_kab seed=43'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN_kab.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'zh-CN_kab' --type 'dual' --lang_src 'zh-CN' --lang_tgt 'kab' --seed 43 --epochs 1 --batch_size 128
echo '--- TASK end: model=zh-CN_kab seed=43'

echo '--- TASK start: model=zh-CN_lg seed=50'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN_lg.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'zh-CN_lg' --type 'dual' --lang_src 'zh-CN' --lang_tgt 'lg' --seed 50 --epochs 1 --batch_size 128
echo '--- TASK end: model=zh-CN_lg seed=50'

echo '--- TASK start: model=zh-CN_lg seed=51'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN_lg.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'zh-CN_lg' --type 'dual' --lang_src 'zh-CN' --lang_tgt 'lg' --seed 51 --epochs 1 --batch_size 128
echo '--- TASK end: model=zh-CN_lg seed=51'

echo '--- TASK start: model=zh-CN_mhr seed=42'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN_mhr.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'zh-CN_mhr' --type 'dual' --lang_src 'zh-CN' --lang_tgt 'mhr' --seed 42 --epochs 1 --batch_size 128
echo '--- TASK end: model=zh-CN_mhr seed=42'

echo '--- TASK start: model=zh-CN_mhr seed=43'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN_mhr.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'zh-CN_mhr' --type 'dual' --lang_src 'zh-CN' --lang_tgt 'mhr' --seed 43 --epochs 1 --batch_size 128
echo '--- TASK end: model=zh-CN_mhr seed=43'

echo '--- TASK start: model=zh-CN_mhr seed=44'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN_mhr.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'zh-CN_mhr' --type 'dual' --lang_src 'zh-CN' --lang_tgt 'mhr' --seed 44 --epochs 1 --batch_size 128
echo '--- TASK end: model=zh-CN_mhr seed=44'

echo '--- TASK start: model=zh-CN_mhr seed=45'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN_mhr.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'zh-CN_mhr' --type 'dual' --lang_src 'zh-CN' --lang_tgt 'mhr' --seed 45 --epochs 1 --batch_size 128
echo '--- TASK end: model=zh-CN_mhr seed=45'

echo '--- TASK start: model=zh-CN_mhr seed=46'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN_mhr.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'zh-CN_mhr' --type 'dual' --lang_src 'zh-CN' --lang_tgt 'mhr' --seed 46 --epochs 1 --batch_size 128
echo '--- TASK end: model=zh-CN_mhr seed=46'

echo '--- TASK start: model=zh-CN_mhr seed=47'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN_mhr.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'zh-CN_mhr' --type 'dual' --lang_src 'zh-CN' --lang_tgt 'mhr' --seed 47 --epochs 1 --batch_size 128
echo '--- TASK end: model=zh-CN_mhr seed=47'

echo '--- TASK start: model=zh-CN_mhr seed=48'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN_mhr.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'zh-CN_mhr' --type 'dual' --lang_src 'zh-CN' --lang_tgt 'mhr' --seed 48 --epochs 1 --batch_size 128
echo '--- TASK end: model=zh-CN_mhr seed=48'

echo '--- TASK start: model=zh-CN_mhr seed=49'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN_mhr.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'zh-CN_mhr' --type 'dual' --lang_src 'zh-CN' --lang_tgt 'mhr' --seed 49 --epochs 1 --batch_size 128
echo '--- TASK end: model=zh-CN_mhr seed=49'

echo '--- TASK start: model=zh-CN_mhr seed=50'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN_mhr.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'zh-CN_mhr' --type 'dual' --lang_src 'zh-CN' --lang_tgt 'mhr' --seed 50 --epochs 1 --batch_size 128
echo '--- TASK end: model=zh-CN_mhr seed=50'

echo '--- TASK start: model=zh-CN_mhr seed=51'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN_mhr.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'zh-CN_mhr' --type 'dual' --lang_src 'zh-CN' --lang_tgt 'mhr' --seed 51 --epochs 1 --batch_size 128
echo '--- TASK end: model=zh-CN_mhr seed=51'

echo '--- TASK start: model=zh-CN_nl seed=42'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN_nl.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'zh-CN_nl' --type 'dual' --lang_src 'zh-CN' --lang_tgt 'nl' --seed 42 --epochs 1 --batch_size 128
echo '--- TASK end: model=zh-CN_nl seed=42'

echo '--- TASK start: model=zh-CN_nl seed=43'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN_nl.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'zh-CN_nl' --type 'dual' --lang_src 'zh-CN' --lang_tgt 'nl' --seed 43 --epochs 1 --batch_size 128
echo '--- TASK end: model=zh-CN_nl seed=43'

echo '--- TASK start: model=zh-CN_nl seed=44'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN_nl.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'zh-CN_nl' --type 'dual' --lang_src 'zh-CN' --lang_tgt 'nl' --seed 44 --epochs 1 --batch_size 128
echo '--- TASK end: model=zh-CN_nl seed=44'

echo '--- TASK start: model=zh-CN_nl seed=45'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN_nl.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'zh-CN_nl' --type 'dual' --lang_src 'zh-CN' --lang_tgt 'nl' --seed 45 --epochs 1 --batch_size 128
echo '--- TASK end: model=zh-CN_nl seed=45'

echo '--- TASK start: model=zh-CN_nl seed=46'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN_nl.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'zh-CN_nl' --type 'dual' --lang_src 'zh-CN' --lang_tgt 'nl' --seed 46 --epochs 1 --batch_size 128
echo '--- TASK end: model=zh-CN_nl seed=46'

echo '--- TASK start: model=zh-CN_nl seed=47'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN_nl.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'zh-CN_nl' --type 'dual' --lang_src 'zh-CN' --lang_tgt 'nl' --seed 47 --epochs 1 --batch_size 128
echo '--- TASK end: model=zh-CN_nl seed=47'

echo '--- TASK start: model=zh-CN_nl seed=48'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN_nl.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'zh-CN_nl' --type 'dual' --lang_src 'zh-CN' --lang_tgt 'nl' --seed 48 --epochs 1 --batch_size 128
echo '--- TASK end: model=zh-CN_nl seed=48'

echo '--- TASK start: model=zh-CN_nl seed=49'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN_nl.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'zh-CN_nl' --type 'dual' --lang_src 'zh-CN' --lang_tgt 'nl' --seed 49 --epochs 1 --batch_size 128
echo '--- TASK end: model=zh-CN_nl seed=49'

echo '--- TASK start: model=zh-CN_nl seed=50'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN_nl.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'zh-CN_nl' --type 'dual' --lang_src 'zh-CN' --lang_tgt 'nl' --seed 50 --epochs 1 --batch_size 128
echo '--- TASK end: model=zh-CN_nl seed=50'

echo '--- TASK start: model=zh-CN_pl seed=51'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN_pl.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'zh-CN_pl' --type 'dual' --lang_src 'zh-CN' --lang_tgt 'pl' --seed 51 --epochs 1 --batch_size 128
echo '--- TASK end: model=zh-CN_pl seed=51'

echo '--- TASK start: model=zh-CN_pt seed=42'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN_pt.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'zh-CN_pt' --type 'dual' --lang_src 'zh-CN' --lang_tgt 'pt' --seed 42 --epochs 1 --batch_size 128
echo '--- TASK end: model=zh-CN_pt seed=42'

echo '--- TASK start: model=zh-CN_pt seed=43'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN_pt.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'zh-CN_pt' --type 'dual' --lang_src 'zh-CN' --lang_tgt 'pt' --seed 43 --epochs 1 --batch_size 128
echo '--- TASK end: model=zh-CN_pt seed=43'

echo '--- TASK start: model=zh-CN_pt seed=44'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN_pt.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'zh-CN_pt' --type 'dual' --lang_src 'zh-CN' --lang_tgt 'pt' --seed 44 --epochs 1 --batch_size 128
echo '--- TASK end: model=zh-CN_pt seed=44'

echo '--- TASK start: model=zh-CN_pt seed=45'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN_pt.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'zh-CN_pt' --type 'dual' --lang_src 'zh-CN' --lang_tgt 'pt' --seed 45 --epochs 1 --batch_size 128
echo '--- TASK end: model=zh-CN_pt seed=45'

echo '--- TASK start: model=zh-CN_pt seed=46'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN_pt.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'zh-CN_pt' --type 'dual' --lang_src 'zh-CN' --lang_tgt 'pt' --seed 46 --epochs 1 --batch_size 128
echo '--- TASK end: model=zh-CN_pt seed=46'

echo '--- TASK start: model=zh-CN_pt seed=47'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN_pt.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'zh-CN_pt' --type 'dual' --lang_src 'zh-CN' --lang_tgt 'pt' --seed 47 --epochs 1 --batch_size 128
echo '--- TASK end: model=zh-CN_pt seed=47'

echo '--- TASK start: model=zh-CN_pt seed=48'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN_pt.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'zh-CN_pt' --type 'dual' --lang_src 'zh-CN' --lang_tgt 'pt' --seed 48 --epochs 1 --batch_size 128
echo '--- TASK end: model=zh-CN_pt seed=48'

echo '--- TASK start: model=zh-CN_pt seed=49'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN_pt.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'zh-CN_pt' --type 'dual' --lang_src 'zh-CN' --lang_tgt 'pt' --seed 49 --epochs 1 --batch_size 128
echo '--- TASK end: model=zh-CN_pt seed=49'

echo '--- TASK start: model=zh-CN_pt seed=50'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN_pt.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'zh-CN_pt' --type 'dual' --lang_src 'zh-CN' --lang_tgt 'pt' --seed 50 --epochs 1 --batch_size 128
echo '--- TASK end: model=zh-CN_pt seed=50'

echo '--- TASK start: model=zh-CN_pt seed=51'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN_pt.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'zh-CN_pt' --type 'dual' --lang_src 'zh-CN' --lang_tgt 'pt' --seed 51 --epochs 1 --batch_size 128
echo '--- TASK end: model=zh-CN_pt seed=51'

echo '--- TASK start: model=zh-CN_ro seed=42'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN_ro.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'zh-CN_ro' --type 'dual' --lang_src 'zh-CN' --lang_tgt 'ro' --seed 42 --epochs 1 --batch_size 128
echo '--- TASK end: model=zh-CN_ro seed=42'

echo '--- TASK start: model=zh-CN_ro seed=43'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN_ro.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'zh-CN_ro' --type 'dual' --lang_src 'zh-CN' --lang_tgt 'ro' --seed 43 --epochs 1 --batch_size 128
echo '--- TASK end: model=zh-CN_ro seed=43'

echo '--- TASK start: model=zh-CN_ro seed=44'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN_ro.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'zh-CN_ro' --type 'dual' --lang_src 'zh-CN' --lang_tgt 'ro' --seed 44 --epochs 1 --batch_size 128
echo '--- TASK end: model=zh-CN_ro seed=44'

echo '--- TASK start: model=zh-CN_ro seed=45'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN_ro.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'zh-CN_ro' --type 'dual' --lang_src 'zh-CN' --lang_tgt 'ro' --seed 45 --epochs 1 --batch_size 128
echo '--- TASK end: model=zh-CN_ro seed=45'

echo '--- TASK start: model=zh-CN_ro seed=46'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN_ro.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'zh-CN_ro' --type 'dual' --lang_src 'zh-CN' --lang_tgt 'ro' --seed 46 --epochs 1 --batch_size 128
echo '--- TASK end: model=zh-CN_ro seed=46'

echo '--- TASK start: model=zh-CN_ro seed=47'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN_ro.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'zh-CN_ro' --type 'dual' --lang_src 'zh-CN' --lang_tgt 'ro' --seed 47 --epochs 1 --batch_size 128
echo '--- TASK end: model=zh-CN_ro seed=47'

echo '--- TASK start: model=zh-CN_ro seed=48'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN_ro.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'zh-CN_ro' --type 'dual' --lang_src 'zh-CN' --lang_tgt 'ro' --seed 48 --epochs 1 --batch_size 128
echo '--- TASK end: model=zh-CN_ro seed=48'

echo '--- TASK start: model=zh-CN_ro seed=49'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN_ro.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'zh-CN_ro' --type 'dual' --lang_src 'zh-CN' --lang_tgt 'ro' --seed 49 --epochs 1 --batch_size 128
echo '--- TASK end: model=zh-CN_ro seed=49'

echo '--- TASK start: model=zh-CN_ro seed=50'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN_ro.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'zh-CN_ro' --type 'dual' --lang_src 'zh-CN' --lang_tgt 'ro' --seed 50 --epochs 1 --batch_size 128
echo '--- TASK end: model=zh-CN_ro seed=50'

echo '--- TASK start: model=zh-CN_ro seed=51'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN_ro.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'zh-CN_ro' --type 'dual' --lang_src 'zh-CN' --lang_tgt 'ro' --seed 51 --epochs 1 --batch_size 128
echo '--- TASK end: model=zh-CN_ro seed=51'

echo '--- TASK start: model=zh-CN_ru seed=42'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN_ru.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'zh-CN_ru' --type 'dual' --lang_src 'zh-CN' --lang_tgt 'ru' --seed 42 --epochs 1 --batch_size 128
echo '--- TASK end: model=zh-CN_ru seed=42'

echo '--- TASK start: model=zh-CN_ru seed=43'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN_ru.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'zh-CN_ru' --type 'dual' --lang_src 'zh-CN' --lang_tgt 'ru' --seed 43 --epochs 1 --batch_size 128
echo '--- TASK end: model=zh-CN_ru seed=43'

echo '--- TASK start: model=zh-CN_ru seed=44'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN_ru.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'zh-CN_ru' --type 'dual' --lang_src 'zh-CN' --lang_tgt 'ru' --seed 44 --epochs 1 --batch_size 128
echo '--- TASK end: model=zh-CN_ru seed=44'

echo '--- TASK start: model=zh-CN_ru seed=45'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN_ru.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'zh-CN_ru' --type 'dual' --lang_src 'zh-CN' --lang_tgt 'ru' --seed 45 --epochs 1 --batch_size 128
echo '--- TASK end: model=zh-CN_ru seed=45'

echo '--- TASK start: model=zh-CN_ru seed=46'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN_ru.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'zh-CN_ru' --type 'dual' --lang_src 'zh-CN' --lang_tgt 'ru' --seed 46 --epochs 1 --batch_size 128
echo '--- TASK end: model=zh-CN_ru seed=46'

echo '--- TASK start: model=zh-CN_ru seed=47'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN_ru.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'zh-CN_ru' --type 'dual' --lang_src 'zh-CN' --lang_tgt 'ru' --seed 47 --epochs 1 --batch_size 128
echo '--- TASK end: model=zh-CN_ru seed=47'

echo '--- TASK start: model=zh-CN_ru seed=48'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN_ru.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'zh-CN_ru' --type 'dual' --lang_src 'zh-CN' --lang_tgt 'ru' --seed 48 --epochs 1 --batch_size 128
echo '--- TASK end: model=zh-CN_ru seed=48'

echo '--- TASK start: model=zh-CN_ru seed=49'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN_ru.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'zh-CN_ru' --type 'dual' --lang_src 'zh-CN' --lang_tgt 'ru' --seed 49 --epochs 1 --batch_size 128
echo '--- TASK end: model=zh-CN_ru seed=49'

echo '--- TASK start: model=zh-CN_ru seed=50'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN_ru.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'zh-CN_ru' --type 'dual' --lang_src 'zh-CN' --lang_tgt 'ru' --seed 50 --epochs 1 --batch_size 128
echo '--- TASK end: model=zh-CN_ru seed=50'

echo '--- TASK start: model=zh-CN_ta seed=44'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN_ta.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'zh-CN_ta' --type 'dual' --lang_src 'zh-CN' --lang_tgt 'ta' --seed 44 --epochs 1 --batch_size 128
echo '--- TASK end: model=zh-CN_ta seed=44'

echo '--- TASK start: model=zh-CN_ta seed=45'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN_ta.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'zh-CN_ta' --type 'dual' --lang_src 'zh-CN' --lang_tgt 'ta' --seed 45 --epochs 1 --batch_size 128
echo '--- TASK end: model=zh-CN_ta seed=45'

echo '--- TASK start: model=zh-CN_ta seed=46'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN_ta.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'zh-CN_ta' --type 'dual' --lang_src 'zh-CN' --lang_tgt 'ta' --seed 46 --epochs 1 --batch_size 128
echo '--- TASK end: model=zh-CN_ta seed=46'

echo '--- TASK start: model=zh-CN_ta seed=47'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN_ta.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'zh-CN_ta' --type 'dual' --lang_src 'zh-CN' --lang_tgt 'ta' --seed 47 --epochs 1 --batch_size 128
echo '--- TASK end: model=zh-CN_ta seed=47'

echo '--- TASK start: model=zh-CN_ta seed=48'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN_ta.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'zh-CN_ta' --type 'dual' --lang_src 'zh-CN' --lang_tgt 'ta' --seed 48 --epochs 1 --batch_size 128
echo '--- TASK end: model=zh-CN_ta seed=48'

echo '--- TASK start: model=zh-CN_ta seed=49'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN_ta.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'zh-CN_ta' --type 'dual' --lang_src 'zh-CN' --lang_tgt 'ta' --seed 49 --epochs 1 --batch_size 128
echo '--- TASK end: model=zh-CN_ta seed=49'

echo '--- TASK start: model=zh-CN_ta seed=50'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN_ta.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'zh-CN_ta' --type 'dual' --lang_src 'zh-CN' --lang_tgt 'ta' --seed 50 --epochs 1 --batch_size 128
echo '--- TASK end: model=zh-CN_ta seed=50'

echo '--- TASK start: model=zh-CN_ta seed=51'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN_ta.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'zh-CN_ta' --type 'dual' --lang_src 'zh-CN' --lang_tgt 'ta' --seed 51 --epochs 1 --batch_size 128
echo '--- TASK end: model=zh-CN_ta seed=51'

echo '--- TASK start: model=zh-CN_th seed=42'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN_th.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'zh-CN_th' --type 'dual' --lang_src 'zh-CN' --lang_tgt 'th' --seed 42 --epochs 1 --batch_size 128
echo '--- TASK end: model=zh-CN_th seed=42'

echo '--- TASK start: model=zh-CN_th seed=43'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN_th.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'zh-CN_th' --type 'dual' --lang_src 'zh-CN' --lang_tgt 'th' --seed 43 --epochs 1 --batch_size 128
echo '--- TASK end: model=zh-CN_th seed=43'

echo '--- TASK start: model=zh-CN_th seed=44'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN_th.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'zh-CN_th' --type 'dual' --lang_src 'zh-CN' --lang_tgt 'th' --seed 44 --epochs 1 --batch_size 128
echo '--- TASK end: model=zh-CN_th seed=44'

echo '--- TASK start: model=zh-CN_th seed=45'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN_th.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'zh-CN_th' --type 'dual' --lang_src 'zh-CN' --lang_tgt 'th' --seed 45 --epochs 1 --batch_size 128
echo '--- TASK end: model=zh-CN_th seed=45'

echo '--- TASK start: model=zh-CN_th seed=46'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN_th.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'zh-CN_th' --type 'dual' --lang_src 'zh-CN' --lang_tgt 'th' --seed 46 --epochs 1 --batch_size 128
echo '--- TASK end: model=zh-CN_th seed=46'

echo '--- TASK start: model=zh-CN_th seed=47'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN_th.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'zh-CN_th' --type 'dual' --lang_src 'zh-CN' --lang_tgt 'th' --seed 47 --epochs 1 --batch_size 128
echo '--- TASK end: model=zh-CN_th seed=47'

echo '--- TASK start: model=zh-CN_th seed=48'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN_th.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'zh-CN_th' --type 'dual' --lang_src 'zh-CN' --lang_tgt 'th' --seed 48 --epochs 1 --batch_size 128
echo '--- TASK end: model=zh-CN_th seed=48'

echo '--- TASK start: model=zh-CN_th seed=49'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN_th.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'zh-CN_th' --type 'dual' --lang_src 'zh-CN' --lang_tgt 'th' --seed 49 --epochs 1 --batch_size 128
echo '--- TASK end: model=zh-CN_th seed=49'

echo '--- TASK start: model=zh-CN_th seed=50'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN_th.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'zh-CN_th' --type 'dual' --lang_src 'zh-CN' --lang_tgt 'th' --seed 50 --epochs 1 --batch_size 128
echo '--- TASK end: model=zh-CN_th seed=50'

echo '--- TASK start: model=zh-CN_th seed=51'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN_th.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'zh-CN_th' --type 'dual' --lang_src 'zh-CN' --lang_tgt 'th' --seed 51 --epochs 1 --batch_size 128
echo '--- TASK end: model=zh-CN_th seed=51'

echo '--- TASK start: model=zh-CN_tr seed=42'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN_tr.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'zh-CN_tr' --type 'dual' --lang_src 'zh-CN' --lang_tgt 'tr' --seed 42 --epochs 1 --batch_size 128
echo '--- TASK end: model=zh-CN_tr seed=42'

echo '--- TASK start: model=zh-CN_tr seed=43'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN_tr.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'zh-CN_tr' --type 'dual' --lang_src 'zh-CN' --lang_tgt 'tr' --seed 43 --epochs 1 --batch_size 128
echo '--- TASK end: model=zh-CN_tr seed=43'

echo '--- TASK start: model=zh-CN_tr seed=44'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN_tr.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'zh-CN_tr' --type 'dual' --lang_src 'zh-CN' --lang_tgt 'tr' --seed 44 --epochs 1 --batch_size 128
echo '--- TASK end: model=zh-CN_tr seed=44'

echo '--- TASK start: model=zh-CN_tr seed=45'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN_tr.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'zh-CN_tr' --type 'dual' --lang_src 'zh-CN' --lang_tgt 'tr' --seed 45 --epochs 1 --batch_size 128
echo '--- TASK end: model=zh-CN_tr seed=45'

echo '--- TASK start: model=zh-CN_tr seed=46'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN_tr.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'zh-CN_tr' --type 'dual' --lang_src 'zh-CN' --lang_tgt 'tr' --seed 46 --epochs 1 --batch_size 128
echo '--- TASK end: model=zh-CN_tr seed=46'

echo '--- TASK start: model=zh-CN_tr seed=47'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN_tr.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'zh-CN_tr' --type 'dual' --lang_src 'zh-CN' --lang_tgt 'tr' --seed 47 --epochs 1 --batch_size 128
echo '--- TASK end: model=zh-CN_tr seed=47'

echo '--- TASK start: model=zh-CN_tr seed=48'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN_tr.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'zh-CN_tr' --type 'dual' --lang_src 'zh-CN' --lang_tgt 'tr' --seed 48 --epochs 1 --batch_size 128
echo '--- TASK end: model=zh-CN_tr seed=48'

echo '--- TASK start: model=zh-CN_uk seed=47'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN_uk.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-CN.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'zh-CN_uk' --type 'dual' --lang_src 'zh-CN' --lang_tgt 'uk' --seed 47 --epochs 1 --batch_size 128
echo '--- TASK end: model=zh-CN_uk seed=47'

