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

echo '--- TASK start: model=uz_ru seed=45'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz_ru.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'uz_ru' --type 'dual' --lang_src 'uz' --lang_tgt 'ru' --seed 45 --epochs 1 --batch_size 128
echo '--- TASK end: model=uz_ru seed=45'

echo '--- TASK start: model=uz_ru seed=46'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz_ru.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'uz_ru' --type 'dual' --lang_src 'uz' --lang_tgt 'ru' --seed 46 --epochs 1 --batch_size 128
echo '--- TASK end: model=uz_ru seed=46'

echo '--- TASK start: model=uz_ru seed=47'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz_ru.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'uz_ru' --type 'dual' --lang_src 'uz' --lang_tgt 'ru' --seed 47 --epochs 1 --batch_size 128
echo '--- TASK end: model=uz_ru seed=47'

echo '--- TASK start: model=uz_ru seed=48'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz_ru.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'uz_ru' --type 'dual' --lang_src 'uz' --lang_tgt 'ru' --seed 48 --epochs 1 --batch_size 128
echo '--- TASK end: model=uz_ru seed=48'

echo '--- TASK start: model=uz_ru seed=49'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz_ru.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'uz_ru' --type 'dual' --lang_src 'uz' --lang_tgt 'ru' --seed 49 --epochs 1 --batch_size 128
echo '--- TASK end: model=uz_ru seed=49'

echo '--- TASK start: model=uz_ru seed=50'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz_ru.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'uz_ru' --type 'dual' --lang_src 'uz' --lang_tgt 'ru' --seed 50 --epochs 1 --batch_size 128
echo '--- TASK end: model=uz_ru seed=50'

echo '--- TASK start: model=uz_ru seed=51'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz_ru.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'uz_ru' --type 'dual' --lang_src 'uz' --lang_tgt 'ru' --seed 51 --epochs 1 --batch_size 128
echo '--- TASK end: model=uz_ru seed=51'

echo '--- TASK start: model=uz_rw seed=42'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz_rw.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'uz_rw' --type 'dual' --lang_src 'uz' --lang_tgt 'rw' --seed 42 --epochs 1 --batch_size 128
echo '--- TASK end: model=uz_rw seed=42'

echo '--- TASK start: model=uz_rw seed=43'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz_rw.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'uz_rw' --type 'dual' --lang_src 'uz' --lang_tgt 'rw' --seed 43 --epochs 1 --batch_size 128
echo '--- TASK end: model=uz_rw seed=43'

echo '--- TASK start: model=uz_rw seed=44'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz_rw.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'uz_rw' --type 'dual' --lang_src 'uz' --lang_tgt 'rw' --seed 44 --epochs 1 --batch_size 128
echo '--- TASK end: model=uz_rw seed=44'

echo '--- TASK start: model=uz_rw seed=45'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz_rw.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'uz_rw' --type 'dual' --lang_src 'uz' --lang_tgt 'rw' --seed 45 --epochs 1 --batch_size 128
echo '--- TASK end: model=uz_rw seed=45'

echo '--- TASK start: model=uz_rw seed=46'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz_rw.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'uz_rw' --type 'dual' --lang_src 'uz' --lang_tgt 'rw' --seed 46 --epochs 1 --batch_size 128
echo '--- TASK end: model=uz_rw seed=46'

echo '--- TASK start: model=uz_rw seed=47'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz_rw.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'uz_rw' --type 'dual' --lang_src 'uz' --lang_tgt 'rw' --seed 47 --epochs 1 --batch_size 128
echo '--- TASK end: model=uz_rw seed=47'

echo '--- TASK start: model=uz_rw seed=48'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz_rw.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'uz_rw' --type 'dual' --lang_src 'uz' --lang_tgt 'rw' --seed 48 --epochs 1 --batch_size 128
echo '--- TASK end: model=uz_rw seed=48'

echo '--- TASK start: model=uz_rw seed=49'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz_rw.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'uz_rw' --type 'dual' --lang_src 'uz' --lang_tgt 'rw' --seed 49 --epochs 1 --batch_size 128
echo '--- TASK end: model=uz_rw seed=49'

echo '--- TASK start: model=uz_rw seed=50'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz_rw.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'uz_rw' --type 'dual' --lang_src 'uz' --lang_tgt 'rw' --seed 50 --epochs 1 --batch_size 128
echo '--- TASK end: model=uz_rw seed=50'

echo '--- TASK start: model=uz_rw seed=51'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz_rw.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'uz_rw' --type 'dual' --lang_src 'uz' --lang_tgt 'rw' --seed 51 --epochs 1 --batch_size 128
echo '--- TASK end: model=uz_rw seed=51'

echo '--- TASK start: model=uz_sv-SE seed=42'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz_sv-SE.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'uz_sv-SE' --type 'dual' --lang_src 'uz' --lang_tgt 'sv-SE' --seed 42 --epochs 1 --batch_size 128
echo '--- TASK end: model=uz_sv-SE seed=42'

echo '--- TASK start: model=uz_sv-SE seed=43'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz_sv-SE.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'uz_sv-SE' --type 'dual' --lang_src 'uz' --lang_tgt 'sv-SE' --seed 43 --epochs 1 --batch_size 128
echo '--- TASK end: model=uz_sv-SE seed=43'

echo '--- TASK start: model=uz_sv-SE seed=44'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz_sv-SE.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'uz_sv-SE' --type 'dual' --lang_src 'uz' --lang_tgt 'sv-SE' --seed 44 --epochs 1 --batch_size 128
echo '--- TASK end: model=uz_sv-SE seed=44'

echo '--- TASK start: model=uz_sv-SE seed=45'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz_sv-SE.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'uz_sv-SE' --type 'dual' --lang_src 'uz' --lang_tgt 'sv-SE' --seed 45 --epochs 1 --batch_size 128
echo '--- TASK end: model=uz_sv-SE seed=45'

echo '--- TASK start: model=uz_sv-SE seed=46'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz_sv-SE.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'uz_sv-SE' --type 'dual' --lang_src 'uz' --lang_tgt 'sv-SE' --seed 46 --epochs 1 --batch_size 128
echo '--- TASK end: model=uz_sv-SE seed=46'

echo '--- TASK start: model=uz_sv-SE seed=47'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz_sv-SE.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'uz_sv-SE' --type 'dual' --lang_src 'uz' --lang_tgt 'sv-SE' --seed 47 --epochs 1 --batch_size 128
echo '--- TASK end: model=uz_sv-SE seed=47'

echo '--- TASK start: model=uz_sv-SE seed=48'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz_sv-SE.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'uz_sv-SE' --type 'dual' --lang_src 'uz' --lang_tgt 'sv-SE' --seed 48 --epochs 1 --batch_size 128
echo '--- TASK end: model=uz_sv-SE seed=48'

echo '--- TASK start: model=uz_sv-SE seed=49'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz_sv-SE.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'uz_sv-SE' --type 'dual' --lang_src 'uz' --lang_tgt 'sv-SE' --seed 49 --epochs 1 --batch_size 128
echo '--- TASK end: model=uz_sv-SE seed=49'

echo '--- TASK start: model=uz_sv-SE seed=50'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz_sv-SE.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'uz_sv-SE' --type 'dual' --lang_src 'uz' --lang_tgt 'sv-SE' --seed 50 --epochs 1 --batch_size 128
echo '--- TASK end: model=uz_sv-SE seed=50'

echo '--- TASK start: model=uz_ta seed=42'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz_ta.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'uz_ta' --type 'dual' --lang_src 'uz' --lang_tgt 'ta' --seed 42 --epochs 1 --batch_size 128
echo '--- TASK end: model=uz_ta seed=42'

echo '--- TASK start: model=uz_ta seed=43'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz_ta.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'uz_ta' --type 'dual' --lang_src 'uz' --lang_tgt 'ta' --seed 43 --epochs 1 --batch_size 128
echo '--- TASK end: model=uz_ta seed=43'

echo '--- TASK start: model=uz_ta seed=44'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz_ta.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'uz_ta' --type 'dual' --lang_src 'uz' --lang_tgt 'ta' --seed 44 --epochs 1 --batch_size 128
echo '--- TASK end: model=uz_ta seed=44'

echo '--- TASK start: model=uz_ta seed=45'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz_ta.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'uz_ta' --type 'dual' --lang_src 'uz' --lang_tgt 'ta' --seed 45 --epochs 1 --batch_size 128
echo '--- TASK end: model=uz_ta seed=45'

echo '--- TASK start: model=uz_ta seed=46'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz_ta.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'uz_ta' --type 'dual' --lang_src 'uz' --lang_tgt 'ta' --seed 46 --epochs 1 --batch_size 128
echo '--- TASK end: model=uz_ta seed=46'

echo '--- TASK start: model=uz_ta seed=47'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz_ta.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'uz_ta' --type 'dual' --lang_src 'uz' --lang_tgt 'ta' --seed 47 --epochs 1 --batch_size 128
echo '--- TASK end: model=uz_ta seed=47'

echo '--- TASK start: model=uz_ta seed=48'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz_ta.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'uz_ta' --type 'dual' --lang_src 'uz' --lang_tgt 'ta' --seed 48 --epochs 1 --batch_size 128
echo '--- TASK end: model=uz_ta seed=48'

echo '--- TASK start: model=uz_ta seed=49'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz_ta.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'uz_ta' --type 'dual' --lang_src 'uz' --lang_tgt 'ta' --seed 49 --epochs 1 --batch_size 128
echo '--- TASK end: model=uz_ta seed=49'

echo '--- TASK start: model=uz_ta seed=50'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz_ta.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'uz_ta' --type 'dual' --lang_src 'uz' --lang_tgt 'ta' --seed 50 --epochs 1 --batch_size 128
echo '--- TASK end: model=uz_ta seed=50'

echo '--- TASK start: model=uz_ta seed=51'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz_ta.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'uz_ta' --type 'dual' --lang_src 'uz' --lang_tgt 'ta' --seed 51 --epochs 1 --batch_size 128
echo '--- TASK end: model=uz_ta seed=51'

echo '--- TASK start: model=uz_th seed=42'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz_th.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'uz_th' --type 'dual' --lang_src 'uz' --lang_tgt 'th' --seed 42 --epochs 1 --batch_size 128
echo '--- TASK end: model=uz_th seed=42'

echo '--- TASK start: model=uz_th seed=43'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz_th.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'uz_th' --type 'dual' --lang_src 'uz' --lang_tgt 'th' --seed 43 --epochs 1 --batch_size 128
echo '--- TASK end: model=uz_th seed=43'

echo '--- TASK start: model=uz_th seed=44'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz_th.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'uz_th' --type 'dual' --lang_src 'uz' --lang_tgt 'th' --seed 44 --epochs 1 --batch_size 128
echo '--- TASK end: model=uz_th seed=44'

echo '--- TASK start: model=uz_th seed=45'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz_th.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'uz_th' --type 'dual' --lang_src 'uz' --lang_tgt 'th' --seed 45 --epochs 1 --batch_size 128
echo '--- TASK end: model=uz_th seed=45'

echo '--- TASK start: model=uz_th seed=46'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz_th.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'uz_th' --type 'dual' --lang_src 'uz' --lang_tgt 'th' --seed 46 --epochs 1 --batch_size 128
echo '--- TASK end: model=uz_th seed=46'

echo '--- TASK start: model=uz_th seed=47'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz_th.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'uz_th' --type 'dual' --lang_src 'uz' --lang_tgt 'th' --seed 47 --epochs 1 --batch_size 128
echo '--- TASK end: model=uz_th seed=47'

echo '--- TASK start: model=uz_th seed=48'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz_th.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'uz_th' --type 'dual' --lang_src 'uz' --lang_tgt 'th' --seed 48 --epochs 1 --batch_size 128
echo '--- TASK end: model=uz_th seed=48'

echo '--- TASK start: model=uz_th seed=49'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz_th.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'uz_th' --type 'dual' --lang_src 'uz' --lang_tgt 'th' --seed 49 --epochs 1 --batch_size 128
echo '--- TASK end: model=uz_th seed=49'

echo '--- TASK start: model=uz_th seed=50'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz_th.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'uz_th' --type 'dual' --lang_src 'uz' --lang_tgt 'th' --seed 50 --epochs 1 --batch_size 128
echo '--- TASK end: model=uz_th seed=50'

echo '--- TASK start: model=uz_th seed=51'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz_th.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'uz_th' --type 'dual' --lang_src 'uz' --lang_tgt 'th' --seed 51 --epochs 1 --batch_size 128
echo '--- TASK end: model=uz_th seed=51'

echo '--- TASK start: model=uz_tr seed=42'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz_tr.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'uz_tr' --type 'dual' --lang_src 'uz' --lang_tgt 'tr' --seed 42 --epochs 1 --batch_size 128
echo '--- TASK end: model=uz_tr seed=42'

echo '--- TASK start: model=uz_tr seed=43'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz_tr.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'uz_tr' --type 'dual' --lang_src 'uz' --lang_tgt 'tr' --seed 43 --epochs 1 --batch_size 128
echo '--- TASK end: model=uz_tr seed=43'

echo '--- TASK start: model=uz_tr seed=44'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz_tr.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'uz_tr' --type 'dual' --lang_src 'uz' --lang_tgt 'tr' --seed 44 --epochs 1 --batch_size 128
echo '--- TASK end: model=uz_tr seed=44'

echo '--- TASK start: model=uz_tr seed=45'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz_tr.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'uz_tr' --type 'dual' --lang_src 'uz' --lang_tgt 'tr' --seed 45 --epochs 1 --batch_size 128
echo '--- TASK end: model=uz_tr seed=45'

echo '--- TASK start: model=uz_tr seed=46'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz_tr.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'uz_tr' --type 'dual' --lang_src 'uz' --lang_tgt 'tr' --seed 46 --epochs 1 --batch_size 128
echo '--- TASK end: model=uz_tr seed=46'

echo '--- TASK start: model=uz_tr seed=47'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz_tr.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'uz_tr' --type 'dual' --lang_src 'uz' --lang_tgt 'tr' --seed 47 --epochs 1 --batch_size 128
echo '--- TASK end: model=uz_tr seed=47'

echo '--- TASK start: model=uz_tr seed=48'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz_tr.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'uz_tr' --type 'dual' --lang_src 'uz' --lang_tgt 'tr' --seed 48 --epochs 1 --batch_size 128
echo '--- TASK end: model=uz_tr seed=48'

echo '--- TASK start: model=uz_tr seed=49'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz_tr.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'uz_tr' --type 'dual' --lang_src 'uz' --lang_tgt 'tr' --seed 49 --epochs 1 --batch_size 128
echo '--- TASK end: model=uz_tr seed=49'

echo '--- TASK start: model=uz_tr seed=50'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz_tr.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'uz_tr' --type 'dual' --lang_src 'uz' --lang_tgt 'tr' --seed 50 --epochs 1 --batch_size 128
echo '--- TASK end: model=uz_tr seed=50'

echo '--- TASK start: model=uz_ur seed=42'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz_ur.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'uz_ur' --type 'dual' --lang_src 'uz' --lang_tgt 'ur' --seed 42 --epochs 1 --batch_size 128
echo '--- TASK end: model=uz_ur seed=42'

echo '--- TASK start: model=uz_ur seed=43'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz_ur.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'uz_ur' --type 'dual' --lang_src 'uz' --lang_tgt 'ur' --seed 43 --epochs 1 --batch_size 128
echo '--- TASK end: model=uz_ur seed=43'

echo '--- TASK start: model=uz_ur seed=44'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz_ur.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'uz_ur' --type 'dual' --lang_src 'uz' --lang_tgt 'ur' --seed 44 --epochs 1 --batch_size 128
echo '--- TASK end: model=uz_ur seed=44'

echo '--- TASK start: model=uz_ur seed=45'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz_ur.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'uz_ur' --type 'dual' --lang_src 'uz' --lang_tgt 'ur' --seed 45 --epochs 1 --batch_size 128
echo '--- TASK end: model=uz_ur seed=45'

echo '--- TASK start: model=uz_ur seed=46'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz_ur.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'uz_ur' --type 'dual' --lang_src 'uz' --lang_tgt 'ur' --seed 46 --epochs 1 --batch_size 128
echo '--- TASK end: model=uz_ur seed=46'

echo '--- TASK start: model=uz_ur seed=47'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz_ur.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'uz_ur' --type 'dual' --lang_src 'uz' --lang_tgt 'ur' --seed 47 --epochs 1 --batch_size 128
echo '--- TASK end: model=uz_ur seed=47'

echo '--- TASK start: model=uz_ur seed=48'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz_ur.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'uz_ur' --type 'dual' --lang_src 'uz' --lang_tgt 'ur' --seed 48 --epochs 1 --batch_size 128
echo '--- TASK end: model=uz_ur seed=48'

echo '--- TASK start: model=uz_ur seed=49'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz_ur.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'uz_ur' --type 'dual' --lang_src 'uz' --lang_tgt 'ur' --seed 49 --epochs 1 --batch_size 128
echo '--- TASK end: model=uz_ur seed=49'

echo '--- TASK start: model=uz_ur seed=50'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz_ur.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'uz_ur' --type 'dual' --lang_src 'uz' --lang_tgt 'ur' --seed 50 --epochs 1 --batch_size 128
echo '--- TASK end: model=uz_ur seed=50'

echo '--- TASK start: model=uz_ur seed=51'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz_ur.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'uz_ur' --type 'dual' --lang_src 'uz' --lang_tgt 'ur' --seed 51 --epochs 1 --batch_size 128
echo '--- TASK end: model=uz_ur seed=51'

echo '--- TASK start: model=uz_yue seed=42'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz_yue.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'uz_yue' --type 'dual' --lang_src 'uz' --lang_tgt 'yue' --seed 42 --epochs 1 --batch_size 128
echo '--- TASK end: model=uz_yue seed=42'

echo '--- TASK start: model=uz_yue seed=43'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz_yue.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'uz_yue' --type 'dual' --lang_src 'uz' --lang_tgt 'yue' --seed 43 --epochs 1 --batch_size 128
echo '--- TASK end: model=uz_yue seed=43'

echo '--- TASK start: model=uz_yue seed=44'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz_yue.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'uz_yue' --type 'dual' --lang_src 'uz' --lang_tgt 'yue' --seed 44 --epochs 1 --batch_size 128
echo '--- TASK end: model=uz_yue seed=44'

echo '--- TASK start: model=uz_yue seed=45'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz_yue.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'uz_yue' --type 'dual' --lang_src 'uz' --lang_tgt 'yue' --seed 45 --epochs 1 --batch_size 128
echo '--- TASK end: model=uz_yue seed=45'

echo '--- TASK start: model=uz_yue seed=46'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz_yue.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'uz_yue' --type 'dual' --lang_src 'uz' --lang_tgt 'yue' --seed 46 --epochs 1 --batch_size 128
echo '--- TASK end: model=uz_yue seed=46'

echo '--- TASK start: model=uz_yue seed=47'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz_yue.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'uz_yue' --type 'dual' --lang_src 'uz' --lang_tgt 'yue' --seed 47 --epochs 1 --batch_size 128
echo '--- TASK end: model=uz_yue seed=47'

echo '--- TASK start: model=uz_yue seed=48'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz_yue.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'uz_yue' --type 'dual' --lang_src 'uz' --lang_tgt 'yue' --seed 48 --epochs 1 --batch_size 128
echo '--- TASK end: model=uz_yue seed=48'

echo '--- TASK start: model=uz_yue seed=49'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz_yue.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'uz_yue' --type 'dual' --lang_src 'uz' --lang_tgt 'yue' --seed 49 --epochs 1 --batch_size 128
echo '--- TASK end: model=uz_yue seed=49'

echo '--- TASK start: model=uz_yue seed=50'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz_yue.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'uz_yue' --type 'dual' --lang_src 'uz' --lang_tgt 'yue' --seed 50 --epochs 1 --batch_size 128
echo '--- TASK end: model=uz_yue seed=50'

echo '--- TASK start: model=uz_yue seed=51'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz_yue.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'uz_yue' --type 'dual' --lang_src 'uz' --lang_tgt 'yue' --seed 51 --epochs 1 --batch_size 128
echo '--- TASK end: model=uz_yue seed=51'

echo '--- TASK start: model=uz_zh-CN seed=42'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz_zh-CN.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'uz_zh-CN' --type 'dual' --lang_src 'uz' --lang_tgt 'zh-CN' --seed 42 --epochs 1 --batch_size 128
echo '--- TASK end: model=uz_zh-CN seed=42'

echo '--- TASK start: model=uz_zh-CN seed=43'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz_zh-CN.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'uz_zh-CN' --type 'dual' --lang_src 'uz' --lang_tgt 'zh-CN' --seed 43 --epochs 1 --batch_size 128
echo '--- TASK end: model=uz_zh-CN seed=43'

echo '--- TASK start: model=uz_zh-CN seed=44'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz_zh-CN.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'uz_zh-CN' --type 'dual' --lang_src 'uz' --lang_tgt 'zh-CN' --seed 44 --epochs 1 --batch_size 128
echo '--- TASK end: model=uz_zh-CN seed=44'

echo '--- TASK start: model=uz_zh-CN seed=45'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz_zh-CN.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'uz_zh-CN' --type 'dual' --lang_src 'uz' --lang_tgt 'zh-CN' --seed 45 --epochs 1 --batch_size 128
echo '--- TASK end: model=uz_zh-CN seed=45'

echo '--- TASK start: model=uz_zh-CN seed=46'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz_zh-CN.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'uz_zh-CN' --type 'dual' --lang_src 'uz' --lang_tgt 'zh-CN' --seed 46 --epochs 1 --batch_size 128
echo '--- TASK end: model=uz_zh-CN seed=46'

echo '--- TASK start: model=uz_zh-CN seed=47'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz_zh-CN.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'uz_zh-CN' --type 'dual' --lang_src 'uz' --lang_tgt 'zh-CN' --seed 47 --epochs 1 --batch_size 128
echo '--- TASK end: model=uz_zh-CN seed=47'

echo '--- TASK start: model=uz_zh-CN seed=48'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz_zh-CN.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'uz_zh-CN' --type 'dual' --lang_src 'uz' --lang_tgt 'zh-CN' --seed 48 --epochs 1 --batch_size 128
echo '--- TASK end: model=uz_zh-CN seed=48'

echo '--- TASK start: model=uz_zh-CN seed=49'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz_zh-CN.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'uz_zh-CN' --type 'dual' --lang_src 'uz' --lang_tgt 'zh-CN' --seed 49 --epochs 1 --batch_size 128
echo '--- TASK end: model=uz_zh-CN seed=49'

echo '--- TASK start: model=uz_zh-CN seed=50'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz_zh-CN.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'uz_zh-CN' --type 'dual' --lang_src 'uz' --lang_tgt 'zh-CN' --seed 50 --epochs 1 --batch_size 128
echo '--- TASK end: model=uz_zh-CN seed=50'

echo '--- TASK start: model=uz_zh-TW seed=42'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz_zh-TW.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'uz_zh-TW' --type 'dual' --lang_src 'uz' --lang_tgt 'zh-TW' --seed 42 --epochs 1 --batch_size 128
echo '--- TASK end: model=uz_zh-TW seed=42'

echo '--- TASK start: model=uz_zh-TW seed=43'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz_zh-TW.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'uz_zh-TW' --type 'dual' --lang_src 'uz' --lang_tgt 'zh-TW' --seed 43 --epochs 1 --batch_size 128
echo '--- TASK end: model=uz_zh-TW seed=43'

echo '--- TASK start: model=uz_zh-TW seed=44'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz_zh-TW.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'uz_zh-TW' --type 'dual' --lang_src 'uz' --lang_tgt 'zh-TW' --seed 44 --epochs 1 --batch_size 128
echo '--- TASK end: model=uz_zh-TW seed=44'

echo '--- TASK start: model=uz_zh-TW seed=45'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz_zh-TW.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'uz_zh-TW' --type 'dual' --lang_src 'uz' --lang_tgt 'zh-TW' --seed 45 --epochs 1 --batch_size 128
echo '--- TASK end: model=uz_zh-TW seed=45'

echo '--- TASK start: model=uz_zh-TW seed=46'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz_zh-TW.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'uz_zh-TW' --type 'dual' --lang_src 'uz' --lang_tgt 'zh-TW' --seed 46 --epochs 1 --batch_size 128
echo '--- TASK end: model=uz_zh-TW seed=46'

echo '--- TASK start: model=uz_zh-TW seed=47'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz_zh-TW.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'uz_zh-TW' --type 'dual' --lang_src 'uz' --lang_tgt 'zh-TW' --seed 47 --epochs 1 --batch_size 128
echo '--- TASK end: model=uz_zh-TW seed=47'

echo '--- TASK start: model=uz_zh-TW seed=48'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz_zh-TW.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'uz_zh-TW' --type 'dual' --lang_src 'uz' --lang_tgt 'zh-TW' --seed 48 --epochs 1 --batch_size 128
echo '--- TASK end: model=uz_zh-TW seed=48'

echo '--- TASK start: model=uz_zh-TW seed=49'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz_zh-TW.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'uz_zh-TW' --type 'dual' --lang_src 'uz' --lang_tgt 'zh-TW' --seed 49 --epochs 1 --batch_size 128
echo '--- TASK end: model=uz_zh-TW seed=49'

echo '--- TASK start: model=uz_zh-TW seed=50'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz_zh-TW.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'uz_zh-TW' --type 'dual' --lang_src 'uz' --lang_tgt 'zh-TW' --seed 50 --epochs 1 --batch_size 128
echo '--- TASK end: model=uz_zh-TW seed=50'

echo '--- TASK start: model=uz_zh-TW seed=51'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz_zh-TW.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'uz_zh-TW' --type 'dual' --lang_src 'uz' --lang_tgt 'zh-TW' --seed 51 --epochs 1 --batch_size 128
echo '--- TASK end: model=uz_zh-TW seed=51'

echo '--- TASK start: model=yue_ab seed=42'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue_ab.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'yue_ab' --type 'dual' --lang_src 'yue' --lang_tgt 'ab' --seed 42 --epochs 1 --batch_size 128
echo '--- TASK end: model=yue_ab seed=42'

echo '--- TASK start: model=yue_ab seed=43'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue_ab.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'yue_ab' --type 'dual' --lang_src 'yue' --lang_tgt 'ab' --seed 43 --epochs 1 --batch_size 128
echo '--- TASK end: model=yue_ab seed=43'

echo '--- TASK start: model=yue_ab seed=44'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue_ab.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'yue_ab' --type 'dual' --lang_src 'yue' --lang_tgt 'ab' --seed 44 --epochs 1 --batch_size 128
echo '--- TASK end: model=yue_ab seed=44'

echo '--- TASK start: model=yue_ab seed=45'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue_ab.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'yue_ab' --type 'dual' --lang_src 'yue' --lang_tgt 'ab' --seed 45 --epochs 1 --batch_size 128
echo '--- TASK end: model=yue_ab seed=45'

echo '--- TASK start: model=yue_ab seed=46'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue_ab.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'yue_ab' --type 'dual' --lang_src 'yue' --lang_tgt 'ab' --seed 46 --epochs 1 --batch_size 128
echo '--- TASK end: model=yue_ab seed=46'

echo '--- TASK start: model=yue_ab seed=47'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue_ab.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'yue_ab' --type 'dual' --lang_src 'yue' --lang_tgt 'ab' --seed 47 --epochs 1 --batch_size 128
echo '--- TASK end: model=yue_ab seed=47'

