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

echo '--- TASK start: model=yue_ab seed=48'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue_ab.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'yue_ab' --type 'dual' --lang_src 'yue' --lang_tgt 'ab' --seed 48 --epochs 1 --batch_size 128
echo '--- TASK end: model=yue_ab seed=48'

echo '--- TASK start: model=yue_ab seed=49'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue_ab.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'yue_ab' --type 'dual' --lang_src 'yue' --lang_tgt 'ab' --seed 49 --epochs 1 --batch_size 128
echo '--- TASK end: model=yue_ab seed=49'

echo '--- TASK start: model=yue_ab seed=50'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue_ab.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'yue_ab' --type 'dual' --lang_src 'yue' --lang_tgt 'ab' --seed 50 --epochs 1 --batch_size 128
echo '--- TASK end: model=yue_ab seed=50'

echo '--- TASK start: model=yue_ab seed=51'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue_ab.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'yue_ab' --type 'dual' --lang_src 'yue' --lang_tgt 'ab' --seed 51 --epochs 1 --batch_size 128
echo '--- TASK end: model=yue_ab seed=51'

echo '--- TASK start: model=yue_ar seed=42'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue_ar.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'yue_ar' --type 'dual' --lang_src 'yue' --lang_tgt 'ar' --seed 42 --epochs 1 --batch_size 128
echo '--- TASK end: model=yue_ar seed=42'

echo '--- TASK start: model=yue_ar seed=43'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue_ar.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'yue_ar' --type 'dual' --lang_src 'yue' --lang_tgt 'ar' --seed 43 --epochs 1 --batch_size 128
echo '--- TASK end: model=yue_ar seed=43'

echo '--- TASK start: model=yue_ar seed=44'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue_ar.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'yue_ar' --type 'dual' --lang_src 'yue' --lang_tgt 'ar' --seed 44 --epochs 1 --batch_size 128
echo '--- TASK end: model=yue_ar seed=44'

echo '--- TASK start: model=yue_ar seed=45'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue_ar.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'yue_ar' --type 'dual' --lang_src 'yue' --lang_tgt 'ar' --seed 45 --epochs 1 --batch_size 128
echo '--- TASK end: model=yue_ar seed=45'

echo '--- TASK start: model=yue_ar seed=46'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue_ar.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'yue_ar' --type 'dual' --lang_src 'yue' --lang_tgt 'ar' --seed 46 --epochs 1 --batch_size 128
echo '--- TASK end: model=yue_ar seed=46'

echo '--- TASK start: model=yue_ar seed=47'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue_ar.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'yue_ar' --type 'dual' --lang_src 'yue' --lang_tgt 'ar' --seed 47 --epochs 1 --batch_size 128
echo '--- TASK end: model=yue_ar seed=47'

echo '--- TASK start: model=yue_ar seed=48'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue_ar.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'yue_ar' --type 'dual' --lang_src 'yue' --lang_tgt 'ar' --seed 48 --epochs 1 --batch_size 128
echo '--- TASK end: model=yue_ar seed=48'

echo '--- TASK start: model=yue_ar seed=49'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue_ar.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'yue_ar' --type 'dual' --lang_src 'yue' --lang_tgt 'ar' --seed 49 --epochs 1 --batch_size 128
echo '--- TASK end: model=yue_ar seed=49'

echo '--- TASK start: model=yue_ar seed=50'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue_ar.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'yue_ar' --type 'dual' --lang_src 'yue' --lang_tgt 'ar' --seed 50 --epochs 1 --batch_size 128
echo '--- TASK end: model=yue_ar seed=50'

echo '--- TASK start: model=yue_cs seed=50'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue_cs.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'yue_cs' --type 'dual' --lang_src 'yue' --lang_tgt 'cs' --seed 50 --epochs 1 --batch_size 128
echo '--- TASK end: model=yue_cs seed=50'

echo '--- TASK start: model=yue_cs seed=51'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue_cs.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'yue_cs' --type 'dual' --lang_src 'yue' --lang_tgt 'cs' --seed 51 --epochs 1 --batch_size 128
echo '--- TASK end: model=yue_cs seed=51'

echo '--- TASK start: model=yue_cy seed=42'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue_cy.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'yue_cy' --type 'dual' --lang_src 'yue' --lang_tgt 'cy' --seed 42 --epochs 1 --batch_size 128
echo '--- TASK end: model=yue_cy seed=42'

echo '--- TASK start: model=yue_cy seed=43'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue_cy.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'yue_cy' --type 'dual' --lang_src 'yue' --lang_tgt 'cy' --seed 43 --epochs 1 --batch_size 128
echo '--- TASK end: model=yue_cy seed=43'

echo '--- TASK start: model=yue_cy seed=44'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue_cy.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'yue_cy' --type 'dual' --lang_src 'yue' --lang_tgt 'cy' --seed 44 --epochs 1 --batch_size 128
echo '--- TASK end: model=yue_cy seed=44'

echo '--- TASK start: model=yue_cy seed=45'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue_cy.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'yue_cy' --type 'dual' --lang_src 'yue' --lang_tgt 'cy' --seed 45 --epochs 1 --batch_size 128
echo '--- TASK end: model=yue_cy seed=45'

echo '--- TASK start: model=yue_cy seed=46'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue_cy.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'yue_cy' --type 'dual' --lang_src 'yue' --lang_tgt 'cy' --seed 46 --epochs 1 --batch_size 128
echo '--- TASK end: model=yue_cy seed=46'

echo '--- TASK start: model=yue_cy seed=47'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue_cy.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'yue_cy' --type 'dual' --lang_src 'yue' --lang_tgt 'cy' --seed 47 --epochs 1 --batch_size 128
echo '--- TASK end: model=yue_cy seed=47'

echo '--- TASK start: model=yue_cy seed=48'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue_cy.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'yue_cy' --type 'dual' --lang_src 'yue' --lang_tgt 'cy' --seed 48 --epochs 1 --batch_size 128
echo '--- TASK end: model=yue_cy seed=48'

echo '--- TASK start: model=yue_cy seed=49'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue_cy.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'yue_cy' --type 'dual' --lang_src 'yue' --lang_tgt 'cy' --seed 49 --epochs 1 --batch_size 128
echo '--- TASK end: model=yue_cy seed=49'

echo '--- TASK start: model=yue_cy seed=50'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue_cy.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'yue_cy' --type 'dual' --lang_src 'yue' --lang_tgt 'cy' --seed 50 --epochs 1 --batch_size 128
echo '--- TASK end: model=yue_cy seed=50'

echo '--- TASK start: model=yue_cy seed=51'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue_cy.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'yue_cy' --type 'dual' --lang_src 'yue' --lang_tgt 'cy' --seed 51 --epochs 1 --batch_size 128
echo '--- TASK end: model=yue_cy seed=51'

echo '--- TASK start: model=yue_de seed=42'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue_de.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'yue_de' --type 'dual' --lang_src 'yue' --lang_tgt 'de' --seed 42 --epochs 1 --batch_size 128
echo '--- TASK end: model=yue_de seed=42'

echo '--- TASK start: model=yue_de seed=43'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue_de.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'yue_de' --type 'dual' --lang_src 'yue' --lang_tgt 'de' --seed 43 --epochs 1 --batch_size 128
echo '--- TASK end: model=yue_de seed=43'

echo '--- TASK start: model=yue_de seed=44'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue_de.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'yue_de' --type 'dual' --lang_src 'yue' --lang_tgt 'de' --seed 44 --epochs 1 --batch_size 128
echo '--- TASK end: model=yue_de seed=44'

echo '--- TASK start: model=yue_de seed=45'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue_de.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'yue_de' --type 'dual' --lang_src 'yue' --lang_tgt 'de' --seed 45 --epochs 1 --batch_size 128
echo '--- TASK end: model=yue_de seed=45'

echo '--- TASK start: model=yue_es seed=42'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue_es.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'yue_es' --type 'dual' --lang_src 'yue' --lang_tgt 'es' --seed 42 --epochs 1 --batch_size 128
echo '--- TASK end: model=yue_es seed=42'

echo '--- TASK start: model=yue_es seed=43'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue_es.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'yue_es' --type 'dual' --lang_src 'yue' --lang_tgt 'es' --seed 43 --epochs 1 --batch_size 128
echo '--- TASK end: model=yue_es seed=43'

echo '--- TASK start: model=yue_es seed=44'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue_es.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'yue_es' --type 'dual' --lang_src 'yue' --lang_tgt 'es' --seed 44 --epochs 1 --batch_size 128
echo '--- TASK end: model=yue_es seed=44'

echo '--- TASK start: model=yue_es seed=45'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue_es.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'yue_es' --type 'dual' --lang_src 'yue' --lang_tgt 'es' --seed 45 --epochs 1 --batch_size 128
echo '--- TASK end: model=yue_es seed=45'

echo '--- TASK start: model=yue_es seed=46'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue_es.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'yue_es' --type 'dual' --lang_src 'yue' --lang_tgt 'es' --seed 46 --epochs 1 --batch_size 128
echo '--- TASK end: model=yue_es seed=46'

echo '--- TASK start: model=yue_es seed=47'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue_es.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'yue_es' --type 'dual' --lang_src 'yue' --lang_tgt 'es' --seed 47 --epochs 1 --batch_size 128
echo '--- TASK end: model=yue_es seed=47'

echo '--- TASK start: model=yue_es seed=48'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue_es.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'yue_es' --type 'dual' --lang_src 'yue' --lang_tgt 'es' --seed 48 --epochs 1 --batch_size 128
echo '--- TASK end: model=yue_es seed=48'

echo '--- TASK start: model=yue_es seed=49'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue_es.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'yue_es' --type 'dual' --lang_src 'yue' --lang_tgt 'es' --seed 49 --epochs 1 --batch_size 128
echo '--- TASK end: model=yue_es seed=49'

echo '--- TASK start: model=yue_es seed=50'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue_es.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'yue_es' --type 'dual' --lang_src 'yue' --lang_tgt 'es' --seed 50 --epochs 1 --batch_size 128
echo '--- TASK end: model=yue_es seed=50'

echo '--- TASK start: model=yue_es seed=51'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue_es.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'yue_es' --type 'dual' --lang_src 'yue' --lang_tgt 'es' --seed 51 --epochs 1 --batch_size 128
echo '--- TASK end: model=yue_es seed=51'

echo '--- TASK start: model=yue_eu seed=42'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue_eu.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'yue_eu' --type 'dual' --lang_src 'yue' --lang_tgt 'eu' --seed 42 --epochs 1 --batch_size 128
echo '--- TASK end: model=yue_eu seed=42'

echo '--- TASK start: model=yue_eu seed=43'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue_eu.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'yue_eu' --type 'dual' --lang_src 'yue' --lang_tgt 'eu' --seed 43 --epochs 1 --batch_size 128
echo '--- TASK end: model=yue_eu seed=43'

echo '--- TASK start: model=yue_eu seed=44'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue_eu.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'yue_eu' --type 'dual' --lang_src 'yue' --lang_tgt 'eu' --seed 44 --epochs 1 --batch_size 128
echo '--- TASK end: model=yue_eu seed=44'

echo '--- TASK start: model=yue_eu seed=45'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue_eu.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'yue_eu' --type 'dual' --lang_src 'yue' --lang_tgt 'eu' --seed 45 --epochs 1 --batch_size 128
echo '--- TASK end: model=yue_eu seed=45'

echo '--- TASK start: model=yue_hy-AM seed=46'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue_hy-AM.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'yue_hy-AM' --type 'dual' --lang_src 'yue' --lang_tgt 'hy-AM' --seed 46 --epochs 1 --batch_size 128
echo '--- TASK end: model=yue_hy-AM seed=46'

echo '--- TASK start: model=yue_hy-AM seed=47'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue_hy-AM.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'yue_hy-AM' --type 'dual' --lang_src 'yue' --lang_tgt 'hy-AM' --seed 47 --epochs 1 --batch_size 128
echo '--- TASK end: model=yue_hy-AM seed=47'

echo '--- TASK start: model=yue_hy-AM seed=48'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue_hy-AM.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'yue_hy-AM' --type 'dual' --lang_src 'yue' --lang_tgt 'hy-AM' --seed 48 --epochs 1 --batch_size 128
echo '--- TASK end: model=yue_hy-AM seed=48'

echo '--- TASK start: model=yue_hy-AM seed=49'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue_hy-AM.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'yue_hy-AM' --type 'dual' --lang_src 'yue' --lang_tgt 'hy-AM' --seed 49 --epochs 1 --batch_size 128
echo '--- TASK end: model=yue_hy-AM seed=49'

echo '--- TASK start: model=yue_hy-AM seed=50'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue_hy-AM.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'yue_hy-AM' --type 'dual' --lang_src 'yue' --lang_tgt 'hy-AM' --seed 50 --epochs 1 --batch_size 128
echo '--- TASK end: model=yue_hy-AM seed=50'

echo '--- TASK start: model=yue_hy-AM seed=51'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue_hy-AM.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'yue_hy-AM' --type 'dual' --lang_src 'yue' --lang_tgt 'hy-AM' --seed 51 --epochs 1 --batch_size 128
echo '--- TASK end: model=yue_hy-AM seed=51'

echo '--- TASK start: model=yue_it seed=42'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue_it.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'yue_it' --type 'dual' --lang_src 'yue' --lang_tgt 'it' --seed 42 --epochs 1 --batch_size 128
echo '--- TASK end: model=yue_it seed=42'

echo '--- TASK start: model=yue_it seed=43'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue_it.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'yue_it' --type 'dual' --lang_src 'yue' --lang_tgt 'it' --seed 43 --epochs 1 --batch_size 128
echo '--- TASK end: model=yue_it seed=43'

echo '--- TASK start: model=yue_it seed=44'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue_it.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'yue_it' --type 'dual' --lang_src 'yue' --lang_tgt 'it' --seed 44 --epochs 1 --batch_size 128
echo '--- TASK end: model=yue_it seed=44'

echo '--- TASK start: model=yue_it seed=45'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue_it.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'yue_it' --type 'dual' --lang_src 'yue' --lang_tgt 'it' --seed 45 --epochs 1 --batch_size 128
echo '--- TASK end: model=yue_it seed=45'

echo '--- TASK start: model=yue_it seed=46'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue_it.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'yue_it' --type 'dual' --lang_src 'yue' --lang_tgt 'it' --seed 46 --epochs 1 --batch_size 128
echo '--- TASK end: model=yue_it seed=46'

echo '--- TASK start: model=yue_it seed=47'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue_it.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'yue_it' --type 'dual' --lang_src 'yue' --lang_tgt 'it' --seed 47 --epochs 1 --batch_size 128
echo '--- TASK end: model=yue_it seed=47'

echo '--- TASK start: model=yue_it seed=48'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue_it.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'yue_it' --type 'dual' --lang_src 'yue' --lang_tgt 'it' --seed 48 --epochs 1 --batch_size 128
echo '--- TASK end: model=yue_it seed=48'

echo '--- TASK start: model=yue_it seed=49'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue_it.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'yue_it' --type 'dual' --lang_src 'yue' --lang_tgt 'it' --seed 49 --epochs 1 --batch_size 128
echo '--- TASK end: model=yue_it seed=49'

echo '--- TASK start: model=yue_kmr seed=44'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue_kmr.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'yue_kmr' --type 'dual' --lang_src 'yue' --lang_tgt 'kmr' --seed 44 --epochs 1 --batch_size 128
echo '--- TASK end: model=yue_kmr seed=44'

echo '--- TASK start: model=yue_kmr seed=45'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue_kmr.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'yue_kmr' --type 'dual' --lang_src 'yue' --lang_tgt 'kmr' --seed 45 --epochs 1 --batch_size 128
echo '--- TASK end: model=yue_kmr seed=45'

echo '--- TASK start: model=yue_kmr seed=46'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue_kmr.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'yue_kmr' --type 'dual' --lang_src 'yue' --lang_tgt 'kmr' --seed 46 --epochs 1 --batch_size 128
echo '--- TASK end: model=yue_kmr seed=46'

echo '--- TASK start: model=yue_kmr seed=47'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue_kmr.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'yue_kmr' --type 'dual' --lang_src 'yue' --lang_tgt 'kmr' --seed 47 --epochs 1 --batch_size 128
echo '--- TASK end: model=yue_kmr seed=47'

echo '--- TASK start: model=yue_kmr seed=48'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue_kmr.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'yue_kmr' --type 'dual' --lang_src 'yue' --lang_tgt 'kmr' --seed 48 --epochs 1 --batch_size 128
echo '--- TASK end: model=yue_kmr seed=48'

echo '--- TASK start: model=yue_kmr seed=49'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue_kmr.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'yue_kmr' --type 'dual' --lang_src 'yue' --lang_tgt 'kmr' --seed 49 --epochs 1 --batch_size 128
echo '--- TASK end: model=yue_kmr seed=49'

echo '--- TASK start: model=yue_kmr seed=50'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue_kmr.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'yue_kmr' --type 'dual' --lang_src 'yue' --lang_tgt 'kmr' --seed 50 --epochs 1 --batch_size 128
echo '--- TASK end: model=yue_kmr seed=50'

echo '--- TASK start: model=yue_kmr seed=51'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue_kmr.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'yue_kmr' --type 'dual' --lang_src 'yue' --lang_tgt 'kmr' --seed 51 --epochs 1 --batch_size 128
echo '--- TASK end: model=yue_kmr seed=51'

echo '--- TASK start: model=yue_ky seed=42'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue_ky.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'yue_ky' --type 'dual' --lang_src 'yue' --lang_tgt 'ky' --seed 42 --epochs 1 --batch_size 128
echo '--- TASK end: model=yue_ky seed=42'

echo '--- TASK start: model=yue_ky seed=43'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue_ky.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'yue_ky' --type 'dual' --lang_src 'yue' --lang_tgt 'ky' --seed 43 --epochs 1 --batch_size 128
echo '--- TASK end: model=yue_ky seed=43'

echo '--- TASK start: model=yue_ky seed=44'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue_ky.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'yue_ky' --type 'dual' --lang_src 'yue' --lang_tgt 'ky' --seed 44 --epochs 1 --batch_size 128
echo '--- TASK end: model=yue_ky seed=44'

echo '--- TASK start: model=yue_ky seed=45'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue_ky.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'yue_ky' --type 'dual' --lang_src 'yue' --lang_tgt 'ky' --seed 45 --epochs 1 --batch_size 128
echo '--- TASK end: model=yue_ky seed=45'

echo '--- TASK start: model=yue_ky seed=46'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue_ky.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'yue_ky' --type 'dual' --lang_src 'yue' --lang_tgt 'ky' --seed 46 --epochs 1 --batch_size 128
echo '--- TASK end: model=yue_ky seed=46'

echo '--- TASK start: model=yue_ky seed=47'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue_ky.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'yue_ky' --type 'dual' --lang_src 'yue' --lang_tgt 'ky' --seed 47 --epochs 1 --batch_size 128
echo '--- TASK end: model=yue_ky seed=47'

echo '--- TASK start: model=yue_ky seed=48'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue_ky.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'yue_ky' --type 'dual' --lang_src 'yue' --lang_tgt 'ky' --seed 48 --epochs 1 --batch_size 128
echo '--- TASK end: model=yue_ky seed=48'

echo '--- TASK start: model=yue_ky seed=49'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue_ky.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'yue_ky' --type 'dual' --lang_src 'yue' --lang_tgt 'ky' --seed 49 --epochs 1 --batch_size 128
echo '--- TASK end: model=yue_ky seed=49'

echo '--- TASK start: model=yue_pt seed=50'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue_pt.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'yue_pt' --type 'dual' --lang_src 'yue' --lang_tgt 'pt' --seed 50 --epochs 1 --batch_size 128
echo '--- TASK end: model=yue_pt seed=50'

echo '--- TASK start: model=yue_pt seed=51'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue_pt.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'yue_pt' --type 'dual' --lang_src 'yue' --lang_tgt 'pt' --seed 51 --epochs 1 --batch_size 128
echo '--- TASK end: model=yue_pt seed=51'

echo '--- TASK start: model=yue_ro seed=42'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue_ro.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'yue_ro' --type 'dual' --lang_src 'yue' --lang_tgt 'ro' --seed 42 --epochs 1 --batch_size 128
echo '--- TASK end: model=yue_ro seed=42'

echo '--- TASK start: model=yue_ro seed=43'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue_ro.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'yue_ro' --type 'dual' --lang_src 'yue' --lang_tgt 'ro' --seed 43 --epochs 1 --batch_size 128
echo '--- TASK end: model=yue_ro seed=43'

echo '--- TASK start: model=yue_ro seed=44'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue_ro.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'yue_ro' --type 'dual' --lang_src 'yue' --lang_tgt 'ro' --seed 44 --epochs 1 --batch_size 128
echo '--- TASK end: model=yue_ro seed=44'

echo '--- TASK start: model=yue_ro seed=45'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue_ro.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'yue_ro' --type 'dual' --lang_src 'yue' --lang_tgt 'ro' --seed 45 --epochs 1 --batch_size 128
echo '--- TASK end: model=yue_ro seed=45'

echo '--- TASK start: model=yue_ro seed=46'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue_ro.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'yue_ro' --type 'dual' --lang_src 'yue' --lang_tgt 'ro' --seed 46 --epochs 1 --batch_size 128
echo '--- TASK end: model=yue_ro seed=46'

echo '--- TASK start: model=yue_ro seed=47'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue_ro.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'yue_ro' --type 'dual' --lang_src 'yue' --lang_tgt 'ro' --seed 47 --epochs 1 --batch_size 128
echo '--- TASK end: model=yue_ro seed=47'

echo '--- TASK start: model=yue_ro seed=48'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue_ro.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'yue_ro' --type 'dual' --lang_src 'yue' --lang_tgt 'ro' --seed 48 --epochs 1 --batch_size 128
echo '--- TASK end: model=yue_ro seed=48'

echo '--- TASK start: model=yue_ro seed=49'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue_ro.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'yue_ro' --type 'dual' --lang_src 'yue' --lang_tgt 'ro' --seed 49 --epochs 1 --batch_size 128
echo '--- TASK end: model=yue_ro seed=49'

echo '--- TASK start: model=yue_ro seed=50'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue_ro.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'yue_ro' --type 'dual' --lang_src 'yue' --lang_tgt 'ro' --seed 50 --epochs 1 --batch_size 128
echo '--- TASK end: model=yue_ro seed=50'

echo '--- TASK start: model=yue_ro seed=51'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue_ro.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'yue_ro' --type 'dual' --lang_src 'yue' --lang_tgt 'ro' --seed 51 --epochs 1 --batch_size 128
echo '--- TASK end: model=yue_ro seed=51'

echo '--- TASK start: model=yue_ru seed=42'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue_ru.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'yue_ru' --type 'dual' --lang_src 'yue' --lang_tgt 'ru' --seed 42 --epochs 1 --batch_size 128
echo '--- TASK end: model=yue_ru seed=42'

echo '--- TASK start: model=yue_ru seed=43'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue_ru.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'yue_ru' --type 'dual' --lang_src 'yue' --lang_tgt 'ru' --seed 43 --epochs 1 --batch_size 128
echo '--- TASK end: model=yue_ru seed=43'

echo '--- TASK start: model=yue_sv-SE seed=49'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue_sv-SE.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'yue_sv-SE' --type 'dual' --lang_src 'yue' --lang_tgt 'sv-SE' --seed 49 --epochs 1 --batch_size 128
echo '--- TASK end: model=yue_sv-SE seed=49'

echo '--- TASK start: model=yue_sv-SE seed=50'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue_sv-SE.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'yue_sv-SE' --type 'dual' --lang_src 'yue' --lang_tgt 'sv-SE' --seed 50 --epochs 1 --batch_size 128
echo '--- TASK end: model=yue_sv-SE seed=50'

echo '--- TASK start: model=yue_sv-SE seed=51'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue_sv-SE.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'yue_sv-SE' --type 'dual' --lang_src 'yue' --lang_tgt 'sv-SE' --seed 51 --epochs 1 --batch_size 128
echo '--- TASK end: model=yue_sv-SE seed=51'

echo '--- TASK start: model=yue_sw seed=42'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue_sw.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'yue_sw' --type 'dual' --lang_src 'yue' --lang_tgt 'sw' --seed 42 --epochs 1 --batch_size 128
echo '--- TASK end: model=yue_sw seed=42'

echo '--- TASK start: model=yue_sw seed=43'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue_sw.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'yue_sw' --type 'dual' --lang_src 'yue' --lang_tgt 'sw' --seed 43 --epochs 1 --batch_size 128
echo '--- TASK end: model=yue_sw seed=43'

echo '--- TASK start: model=yue_sw seed=44'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue_sw.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'yue_sw' --type 'dual' --lang_src 'yue' --lang_tgt 'sw' --seed 44 --epochs 1 --batch_size 128
echo '--- TASK end: model=yue_sw seed=44'

echo '--- TASK start: model=yue_sw seed=45'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue_sw.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'yue_sw' --type 'dual' --lang_src 'yue' --lang_tgt 'sw' --seed 45 --epochs 1 --batch_size 128
echo '--- TASK end: model=yue_sw seed=45'

echo '--- TASK start: model=yue_sw seed=46'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue_sw.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'yue_sw' --type 'dual' --lang_src 'yue' --lang_tgt 'sw' --seed 46 --epochs 1 --batch_size 128
echo '--- TASK end: model=yue_sw seed=46'

echo '--- TASK start: model=yue_sw seed=47'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue_sw.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'yue_sw' --type 'dual' --lang_src 'yue' --lang_tgt 'sw' --seed 47 --epochs 1 --batch_size 128
echo '--- TASK end: model=yue_sw seed=47'

echo '--- TASK start: model=yue_sw seed=48'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue_sw.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'yue_sw' --type 'dual' --lang_src 'yue' --lang_tgt 'sw' --seed 48 --epochs 1 --batch_size 128
echo '--- TASK end: model=yue_sw seed=48'

echo '--- TASK start: model=yue_sw seed=49'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue_sw.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'yue_sw' --type 'dual' --lang_src 'yue' --lang_tgt 'sw' --seed 49 --epochs 1 --batch_size 128
echo '--- TASK end: model=yue_sw seed=49'

echo '--- TASK start: model=yue_sw seed=50'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue_sw.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'yue_sw' --type 'dual' --lang_src 'yue' --lang_tgt 'sw' --seed 50 --epochs 1 --batch_size 128
echo '--- TASK end: model=yue_sw seed=50'

echo '--- TASK start: model=yue_sw seed=51'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue_sw.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/yue.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'yue_sw' --type 'dual' --lang_src 'yue' --lang_tgt 'sw' --seed 51 --epochs 1 --batch_size 128
echo '--- TASK end: model=yue_sw seed=51'

