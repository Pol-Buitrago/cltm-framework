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

echo '--- TASK start: model=mhr_cy seed=42'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/mhr_cy.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/mhr.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'mhr_cy' --type 'dual' --lang_src 'mhr' --lang_tgt 'cy' --seed 42 --epochs 1 --batch_size 128
echo '--- TASK end: model=mhr_cy seed=42'

echo '--- TASK start: model=mhr_cy seed=43'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/mhr_cy.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/mhr.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'mhr_cy' --type 'dual' --lang_src 'mhr' --lang_tgt 'cy' --seed 43 --epochs 1 --batch_size 128
echo '--- TASK end: model=mhr_cy seed=43'

echo '--- TASK start: model=mhr_cy seed=44'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/mhr_cy.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/mhr.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'mhr_cy' --type 'dual' --lang_src 'mhr' --lang_tgt 'cy' --seed 44 --epochs 1 --batch_size 128
echo '--- TASK end: model=mhr_cy seed=44'

echo '--- TASK start: model=mhr_eo seed=50'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/mhr_eo.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/mhr.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'mhr_eo' --type 'dual' --lang_src 'mhr' --lang_tgt 'eo' --seed 50 --epochs 1 --batch_size 128
echo '--- TASK end: model=mhr_eo seed=50'

echo '--- TASK start: model=mhr_eo seed=51'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/mhr_eo.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/mhr.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'mhr_eo' --type 'dual' --lang_src 'mhr' --lang_tgt 'eo' --seed 51 --epochs 1 --batch_size 128
echo '--- TASK end: model=mhr_eo seed=51'

echo '--- TASK start: model=mhr_fa seed=43'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/mhr_fa.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/mhr.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'mhr_fa' --type 'dual' --lang_src 'mhr' --lang_tgt 'fa' --seed 43 --epochs 1 --batch_size 128
echo '--- TASK end: model=mhr_fa seed=43'

echo '--- TASK start: model=mhr_gl seed=45'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/mhr_gl.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/mhr.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'mhr_gl' --type 'dual' --lang_src 'mhr' --lang_tgt 'gl' --seed 45 --epochs 1 --batch_size 128
echo '--- TASK end: model=mhr_gl seed=45'

echo '--- TASK start: model=mhr_hu seed=43'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/mhr_hu.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/mhr.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'mhr_hu' --type 'dual' --lang_src 'mhr' --lang_tgt 'hu' --seed 43 --epochs 1 --batch_size 128
echo '--- TASK end: model=mhr_hu seed=43'

echo '--- TASK start: model=mhr_hu seed=44'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/mhr_hu.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/mhr.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'mhr_hu' --type 'dual' --lang_src 'mhr' --lang_tgt 'hu' --seed 44 --epochs 1 --batch_size 128
echo '--- TASK end: model=mhr_hu seed=44'

echo '--- TASK start: model=mhr_it seed=50'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/mhr_it.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/mhr.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'mhr_it' --type 'dual' --lang_src 'mhr' --lang_tgt 'it' --seed 50 --epochs 1 --batch_size 128
echo '--- TASK end: model=mhr_it seed=50'

echo '--- TASK start: model=mhr_it seed=51'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/mhr_it.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/mhr.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'mhr_it' --type 'dual' --lang_src 'mhr' --lang_tgt 'it' --seed 51 --epochs 1 --batch_size 128
echo '--- TASK end: model=mhr_it seed=51'

echo '--- TASK start: model=mhr_ka seed=42'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/mhr_ka.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/mhr.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'mhr_ka' --type 'dual' --lang_src 'mhr' --lang_tgt 'ka' --seed 42 --epochs 1 --batch_size 128
echo '--- TASK end: model=mhr_ka seed=42'

echo '--- TASK start: model=mhr_ka seed=50'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/mhr_ka.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/mhr.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'mhr_ka' --type 'dual' --lang_src 'mhr' --lang_tgt 'ka' --seed 50 --epochs 1 --batch_size 128
echo '--- TASK end: model=mhr_ka seed=50'

echo '--- TASK start: model=mhr_ka seed=51'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/mhr_ka.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/mhr.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'mhr_ka' --type 'dual' --lang_src 'mhr' --lang_tgt 'ka' --seed 51 --epochs 1 --batch_size 128
echo '--- TASK end: model=mhr_ka seed=51'

echo '--- TASK start: model=mhr_kab seed=42'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/mhr_kab.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/mhr.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'mhr_kab' --type 'dual' --lang_src 'mhr' --lang_tgt 'kab' --seed 42 --epochs 1 --batch_size 128
echo '--- TASK end: model=mhr_kab seed=42'

echo '--- TASK start: model=mhr_nl seed=44'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/mhr_nl.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/mhr.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'mhr_nl' --type 'dual' --lang_src 'mhr' --lang_tgt 'nl' --seed 44 --epochs 1 --batch_size 128
echo '--- TASK end: model=mhr_nl seed=44'

echo '--- TASK start: model=mhr_nl seed=45'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/mhr_nl.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/mhr.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'mhr_nl' --type 'dual' --lang_src 'mhr' --lang_tgt 'nl' --seed 45 --epochs 1 --batch_size 128
echo '--- TASK end: model=mhr_nl seed=45'

echo '--- TASK start: model=mhr_nl seed=46'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/mhr_nl.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/mhr.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'mhr_nl' --type 'dual' --lang_src 'mhr' --lang_tgt 'nl' --seed 46 --epochs 1 --batch_size 128
echo '--- TASK end: model=mhr_nl seed=46'

echo '--- TASK start: model=mhr_pl seed=51'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/mhr_pl.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/mhr.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'mhr_pl' --type 'dual' --lang_src 'mhr' --lang_tgt 'pl' --seed 51 --epochs 1 --batch_size 128
echo '--- TASK end: model=mhr_pl seed=51'

echo '--- TASK start: model=mhr_pt seed=42'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/mhr_pt.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/mhr.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'mhr_pt' --type 'dual' --lang_src 'mhr' --lang_tgt 'pt' --seed 42 --epochs 1 --batch_size 128
echo '--- TASK end: model=mhr_pt seed=42'

echo '--- TASK start: model=mhr_pt seed=43'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/mhr_pt.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/mhr.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'mhr_pt' --type 'dual' --lang_src 'mhr' --lang_tgt 'pt' --seed 43 --epochs 1 --batch_size 128
echo '--- TASK end: model=mhr_pt seed=43'

echo '--- TASK start: model=mhr_pt seed=51'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/mhr_pt.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/mhr.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'mhr_pt' --type 'dual' --lang_src 'mhr' --lang_tgt 'pt' --seed 51 --epochs 1 --batch_size 128
echo '--- TASK end: model=mhr_pt seed=51'

echo '--- TASK start: model=mhr_ro seed=42'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/mhr_ro.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/mhr.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'mhr_ro' --type 'dual' --lang_src 'mhr' --lang_tgt 'ro' --seed 42 --epochs 1 --batch_size 128
echo '--- TASK end: model=mhr_ro seed=42'

echo '--- TASK start: model=mhr_ro seed=43'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/mhr_ro.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/mhr.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'mhr_ro' --type 'dual' --lang_src 'mhr' --lang_tgt 'ro' --seed 43 --epochs 1 --batch_size 128
echo '--- TASK end: model=mhr_ro seed=43'

echo '--- TASK start: model=mhr_ro seed=51'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/mhr_ro.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/mhr.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'mhr_ro' --type 'dual' --lang_src 'mhr' --lang_tgt 'ro' --seed 51 --epochs 1 --batch_size 128
echo '--- TASK end: model=mhr_ro seed=51'

echo '--- TASK start: model=mhr_ru seed=42'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/mhr_ru.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/mhr.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'mhr_ru' --type 'dual' --lang_src 'mhr' --lang_tgt 'ru' --seed 42 --epochs 1 --batch_size 128
echo '--- TASK end: model=mhr_ru seed=42'

echo '--- TASK start: model=mhr_ru seed=43'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/mhr_ru.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/mhr.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'mhr_ru' --type 'dual' --lang_src 'mhr' --lang_tgt 'ru' --seed 43 --epochs 1 --batch_size 128
echo '--- TASK end: model=mhr_ru seed=43'

echo '--- TASK start: model=mhr_ru seed=51'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/mhr_ru.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/mhr.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'mhr_ru' --type 'dual' --lang_src 'mhr' --lang_tgt 'ru' --seed 51 --epochs 1 --batch_size 128
echo '--- TASK end: model=mhr_ru seed=51'

echo '--- TASK start: model=mhr_rw seed=42'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/mhr_rw.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/mhr.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'mhr_rw' --type 'dual' --lang_src 'mhr' --lang_tgt 'rw' --seed 42 --epochs 1 --batch_size 128
echo '--- TASK end: model=mhr_rw seed=42'

echo '--- TASK start: model=mhr_rw seed=50'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/mhr_rw.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/mhr.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'mhr_rw' --type 'dual' --lang_src 'mhr' --lang_tgt 'rw' --seed 50 --epochs 1 --batch_size 128
echo '--- TASK end: model=mhr_rw seed=50'

echo '--- TASK start: model=mhr_sv-SE seed=48'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/mhr_sv-SE.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/mhr.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'mhr_sv-SE' --type 'dual' --lang_src 'mhr' --lang_tgt 'sv-SE' --seed 48 --epochs 1 --batch_size 128
echo '--- TASK end: model=mhr_sv-SE seed=48'

echo '--- TASK start: model=mhr_sv-SE seed=49'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/mhr_sv-SE.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/mhr.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'mhr_sv-SE' --type 'dual' --lang_src 'mhr' --lang_tgt 'sv-SE' --seed 49 --epochs 1 --batch_size 128
echo '--- TASK end: model=mhr_sv-SE seed=49'

echo '--- TASK start: model=mhr_sv-SE seed=50'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/mhr_sv-SE.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/mhr.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'mhr_sv-SE' --type 'dual' --lang_src 'mhr' --lang_tgt 'sv-SE' --seed 50 --epochs 1 --batch_size 128
echo '--- TASK end: model=mhr_sv-SE seed=50'

echo '--- TASK start: model=mhr_sw seed=48'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/mhr_sw.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/mhr.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'mhr_sw' --type 'dual' --lang_src 'mhr' --lang_tgt 'sw' --seed 48 --epochs 1 --batch_size 128
echo '--- TASK end: model=mhr_sw seed=48'

echo '--- TASK start: model=mhr_ta seed=46'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/mhr_ta.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/mhr.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'mhr_ta' --type 'dual' --lang_src 'mhr' --lang_tgt 'ta' --seed 46 --epochs 1 --batch_size 128
echo '--- TASK end: model=mhr_ta seed=46'

echo '--- TASK start: model=mhr_ta seed=47'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/mhr_ta.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/mhr.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'mhr_ta' --type 'dual' --lang_src 'mhr' --lang_tgt 'ta' --seed 47 --epochs 1 --batch_size 128
echo '--- TASK end: model=mhr_ta seed=47'

echo '--- TASK start: model=mhr_th seed=45'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/mhr_th.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/mhr.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'mhr_th' --type 'dual' --lang_src 'mhr' --lang_tgt 'th' --seed 45 --epochs 1 --batch_size 128
echo '--- TASK end: model=mhr_th seed=45'

echo '--- TASK start: model=mhr_th seed=46'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/mhr_th.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/mhr.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'mhr_th' --type 'dual' --lang_src 'mhr' --lang_tgt 'th' --seed 46 --epochs 1 --batch_size 128
echo '--- TASK end: model=mhr_th seed=46'

echo '--- TASK start: model=mhr_th seed=47'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/mhr_th.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/mhr.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'mhr_th' --type 'dual' --lang_src 'mhr' --lang_tgt 'th' --seed 47 --epochs 1 --batch_size 128
echo '--- TASK end: model=mhr_th seed=47'

echo '--- TASK start: model=mhr_ug seed=42'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/mhr_ug.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/mhr.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'mhr_ug' --type 'dual' --lang_src 'mhr' --lang_tgt 'ug' --seed 42 --epochs 1 --batch_size 128
echo '--- TASK end: model=mhr_ug seed=42'

echo '--- TASK start: model=mhr_ug seed=43'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/mhr_ug.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/mhr.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'mhr_ug' --type 'dual' --lang_src 'mhr' --lang_tgt 'ug' --seed 43 --epochs 1 --batch_size 128
echo '--- TASK end: model=mhr_ug seed=43'

echo '--- TASK start: model=mhr_ug seed=44'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/mhr_ug.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/mhr.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'mhr_ug' --type 'dual' --lang_src 'mhr' --lang_tgt 'ug' --seed 44 --epochs 1 --batch_size 128
echo '--- TASK end: model=mhr_ug seed=44'

echo '--- TASK start: model=mhr_uk seed=42'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/mhr_uk.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/mhr.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'mhr_uk' --type 'dual' --lang_src 'mhr' --lang_tgt 'uk' --seed 42 --epochs 1 --batch_size 128
echo '--- TASK end: model=mhr_uk seed=42'

echo '--- TASK start: model=mhr_uk seed=43'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/mhr_uk.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/mhr.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'mhr_uk' --type 'dual' --lang_src 'mhr' --lang_tgt 'uk' --seed 43 --epochs 1 --batch_size 128
echo '--- TASK end: model=mhr_uk seed=43'

echo '--- TASK start: model=mhr_uk seed=44'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/mhr_uk.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/mhr.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'mhr_uk' --type 'dual' --lang_src 'mhr' --lang_tgt 'uk' --seed 44 --epochs 1 --batch_size 128
echo '--- TASK end: model=mhr_uk seed=44'

echo '--- TASK start: model=mhr_ur seed=42'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/mhr_ur.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/mhr.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'mhr_ur' --type 'dual' --lang_src 'mhr' --lang_tgt 'ur' --seed 42 --epochs 1 --batch_size 128
echo '--- TASK end: model=mhr_ur seed=42'

echo '--- TASK start: model=mhr_ur seed=43'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/mhr_ur.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/mhr.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'mhr_ur' --type 'dual' --lang_src 'mhr' --lang_tgt 'ur' --seed 43 --epochs 1 --batch_size 128
echo '--- TASK end: model=mhr_ur seed=43'

echo '--- TASK start: model=mhr_ur seed=44'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/mhr_ur.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/mhr.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'mhr_ur' --type 'dual' --lang_src 'mhr' --lang_tgt 'ur' --seed 44 --epochs 1 --batch_size 128
echo '--- TASK end: model=mhr_ur seed=44'

echo '--- TASK start: model=mhr_uz seed=49'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/mhr_uz.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/mhr.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'mhr_uz' --type 'dual' --lang_src 'mhr' --lang_tgt 'uz' --seed 49 --epochs 1 --batch_size 128
echo '--- TASK end: model=mhr_uz seed=49'

echo '--- TASK start: model=mhr_yue seed=49'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/mhr_yue.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/mhr.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'mhr_yue' --type 'dual' --lang_src 'mhr' --lang_tgt 'yue' --seed 49 --epochs 1 --batch_size 128
echo '--- TASK end: model=mhr_yue seed=49'

echo '--- TASK start: model=mhr_yue seed=50'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/mhr_yue.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/mhr.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'mhr_yue' --type 'dual' --lang_src 'mhr' --lang_tgt 'yue' --seed 50 --epochs 1 --batch_size 128
echo '--- TASK end: model=mhr_yue seed=50'

echo '--- TASK start: model=mhr_yue seed=51'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/mhr_yue.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/mhr.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'mhr_yue' --type 'dual' --lang_src 'mhr' --lang_tgt 'yue' --seed 51 --epochs 1 --batch_size 128
echo '--- TASK end: model=mhr_yue seed=51'

echo '--- TASK start: model=mhr_zh-CN seed=49'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/mhr_zh-CN.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/mhr.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'mhr_zh-CN' --type 'dual' --lang_src 'mhr' --lang_tgt 'zh-CN' --seed 49 --epochs 1 --batch_size 128
echo '--- TASK end: model=mhr_zh-CN seed=49'

echo '--- TASK start: model=mhr_zh-CN seed=50'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/mhr_zh-CN.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/mhr.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'mhr_zh-CN' --type 'dual' --lang_src 'mhr' --lang_tgt 'zh-CN' --seed 50 --epochs 1 --batch_size 128
echo '--- TASK end: model=mhr_zh-CN seed=50'

echo '--- TASK start: model=mhr_zh-CN seed=51'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/mhr_zh-CN.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/mhr.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'mhr_zh-CN' --type 'dual' --lang_src 'mhr' --lang_tgt 'zh-CN' --seed 51 --epochs 1 --batch_size 128
echo '--- TASK end: model=mhr_zh-CN seed=51'

echo '--- TASK start: model=mhr_zh-TW seed=47'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/mhr_zh-TW.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/mhr.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'mhr_zh-TW' --type 'dual' --lang_src 'mhr' --lang_tgt 'zh-TW' --seed 47 --epochs 1 --batch_size 128
echo '--- TASK end: model=mhr_zh-TW seed=47'

echo '--- TASK start: model=nl_ab seed=47'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/nl_ab.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/nl.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'nl_ab' --type 'dual' --lang_src 'nl' --lang_tgt 'ab' --seed 47 --epochs 1 --batch_size 128
echo '--- TASK end: model=nl_ab seed=47'

echo '--- TASK start: model=nl_ab seed=48'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/nl_ab.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/nl.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'nl_ab' --type 'dual' --lang_src 'nl' --lang_tgt 'ab' --seed 48 --epochs 1 --batch_size 128
echo '--- TASK end: model=nl_ab seed=48'

echo '--- TASK start: model=ta_hu seed=47'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/ta_hu.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/ta.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'ta_hu' --type 'dual' --lang_src 'ta' --lang_tgt 'hu' --seed 47 --epochs 1 --batch_size 128
echo '--- TASK end: model=ta_hu seed=47'

echo '--- TASK start: model=ta_hy-AM seed=47'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/ta_hy-AM.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/ta.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'ta_hy-AM' --type 'dual' --lang_src 'ta' --lang_tgt 'hy-AM' --seed 47 --epochs 1 --batch_size 128
echo '--- TASK end: model=ta_hy-AM seed=47'

echo '--- TASK start: model=uk_cy seed=42'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uk_cy.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uk.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'uk_cy' --type 'dual' --lang_src 'uk' --lang_tgt 'cy' --seed 42 --epochs 1 --batch_size 128
echo '--- TASK end: model=uk_cy seed=42'

echo '--- TASK start: model=uk_de seed=42'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uk_de.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uk.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'uk_de' --type 'dual' --lang_src 'uk' --lang_tgt 'de' --seed 42 --epochs 1 --batch_size 128
echo '--- TASK end: model=uk_de seed=42'

echo '--- TASK start: model=uk_pl seed=43'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uk_pl.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uk.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'uk_pl' --type 'dual' --lang_src 'uk' --lang_tgt 'pl' --seed 43 --epochs 1 --batch_size 128
echo '--- TASK end: model=uk_pl seed=43'

echo '--- TASK start: model=uk_ru seed=44'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uk_ru.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uk.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'uk_ru' --type 'dual' --lang_src 'uk' --lang_tgt 'ru' --seed 44 --epochs 1 --batch_size 128
echo '--- TASK end: model=uk_ru seed=44'

echo '--- TASK start: model=uk_rw seed=44'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uk_rw.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uk.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'uk_rw' --type 'dual' --lang_src 'uk' --lang_tgt 'rw' --seed 44 --epochs 1 --batch_size 128
echo '--- TASK end: model=uk_rw seed=44'

echo '--- TASK start: model=uk_ur seed=45'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uk_ur.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uk.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'uk_ur' --type 'dual' --lang_src 'uk' --lang_tgt 'ur' --seed 45 --epochs 1 --batch_size 128
echo '--- TASK end: model=uk_ur seed=45'

echo '--- TASK start: model=uk_uz seed=45'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uk_uz.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uk.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'uk_uz' --type 'dual' --lang_src 'uk' --lang_tgt 'uz' --seed 45 --epochs 1 --batch_size 128
echo '--- TASK end: model=uk_uz seed=45'

echo '--- TASK start: model=uk_yue seed=45'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uk_yue.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uk.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'uk_yue' --type 'dual' --lang_src 'uk' --lang_tgt 'yue' --seed 45 --epochs 1 --batch_size 128
echo '--- TASK end: model=uk_yue seed=45'

echo '--- TASK start: model=uk_zh-HK seed=45'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uk_zh-HK.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uk.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'uk_zh-HK' --type 'dual' --lang_src 'uk' --lang_tgt 'zh-HK' --seed 45 --epochs 1 --batch_size 128
echo '--- TASK end: model=uk_zh-HK seed=45'

echo '--- TASK start: model=uz_ky seed=50'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz_ky.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'uz_ky' --type 'dual' --lang_src 'uz' --lang_tgt 'ky' --seed 50 --epochs 1 --batch_size 128
echo '--- TASK end: model=uz_ky seed=50'

echo '--- TASK start: model=uz_lg seed=51'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz_lg.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'uz_lg' --type 'dual' --lang_src 'uz' --lang_tgt 'lg' --seed 51 --epochs 1 --batch_size 128
echo '--- TASK end: model=uz_lg seed=51'

echo '--- TASK start: model=uz_mhr seed=42'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz_mhr.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'uz_mhr' --type 'dual' --lang_src 'uz' --lang_tgt 'mhr' --seed 42 --epochs 1 --batch_size 128
echo '--- TASK end: model=uz_mhr seed=42'

echo '--- TASK start: model=uz_mhr seed=43'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz_mhr.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'uz_mhr' --type 'dual' --lang_src 'uz' --lang_tgt 'mhr' --seed 43 --epochs 1 --batch_size 128
echo '--- TASK end: model=uz_mhr seed=43'

echo '--- TASK start: model=uz_mhr seed=44'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz_mhr.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'uz_mhr' --type 'dual' --lang_src 'uz' --lang_tgt 'mhr' --seed 44 --epochs 1 --batch_size 128
echo '--- TASK end: model=uz_mhr seed=44'

echo '--- TASK start: model=uz_mhr seed=45'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz_mhr.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'uz_mhr' --type 'dual' --lang_src 'uz' --lang_tgt 'mhr' --seed 45 --epochs 1 --batch_size 128
echo '--- TASK end: model=uz_mhr seed=45'

echo '--- TASK start: model=uz_mhr seed=46'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz_mhr.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'uz_mhr' --type 'dual' --lang_src 'uz' --lang_tgt 'mhr' --seed 46 --epochs 1 --batch_size 128
echo '--- TASK end: model=uz_mhr seed=46'

echo '--- TASK start: model=uz_mhr seed=47'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz_mhr.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'uz_mhr' --type 'dual' --lang_src 'uz' --lang_tgt 'mhr' --seed 47 --epochs 1 --batch_size 128
echo '--- TASK end: model=uz_mhr seed=47'

echo '--- TASK start: model=uz_mhr seed=48'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz_mhr.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'uz_mhr' --type 'dual' --lang_src 'uz' --lang_tgt 'mhr' --seed 48 --epochs 1 --batch_size 128
echo '--- TASK end: model=uz_mhr seed=48'

echo '--- TASK start: model=uz_mhr seed=49'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz_mhr.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'uz_mhr' --type 'dual' --lang_src 'uz' --lang_tgt 'mhr' --seed 49 --epochs 1 --batch_size 128
echo '--- TASK end: model=uz_mhr seed=49'

echo '--- TASK start: model=uz_mhr seed=50'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz_mhr.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'uz_mhr' --type 'dual' --lang_src 'uz' --lang_tgt 'mhr' --seed 50 --epochs 1 --batch_size 128
echo '--- TASK end: model=uz_mhr seed=50'

echo '--- TASK start: model=uz_mhr seed=51'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz_mhr.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'uz_mhr' --type 'dual' --lang_src 'uz' --lang_tgt 'mhr' --seed 51 --epochs 1 --batch_size 128
echo '--- TASK end: model=uz_mhr seed=51'

echo '--- TASK start: model=uz_nl seed=42'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz_nl.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'uz_nl' --type 'dual' --lang_src 'uz' --lang_tgt 'nl' --seed 42 --epochs 1 --batch_size 128
echo '--- TASK end: model=uz_nl seed=42'

echo '--- TASK start: model=uz_nl seed=43'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz_nl.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'uz_nl' --type 'dual' --lang_src 'uz' --lang_tgt 'nl' --seed 43 --epochs 1 --batch_size 128
echo '--- TASK end: model=uz_nl seed=43'

echo '--- TASK start: model=uz_nl seed=44'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz_nl.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'uz_nl' --type 'dual' --lang_src 'uz' --lang_tgt 'nl' --seed 44 --epochs 1 --batch_size 128
echo '--- TASK end: model=uz_nl seed=44'

echo '--- TASK start: model=uz_nl seed=45'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz_nl.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'uz_nl' --type 'dual' --lang_src 'uz' --lang_tgt 'nl' --seed 45 --epochs 1 --batch_size 128
echo '--- TASK end: model=uz_nl seed=45'

echo '--- TASK start: model=uz_nl seed=46'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz_nl.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'uz_nl' --type 'dual' --lang_src 'uz' --lang_tgt 'nl' --seed 46 --epochs 1 --batch_size 128
echo '--- TASK end: model=uz_nl seed=46'

echo '--- TASK start: model=uz_nl seed=47'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz_nl.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'uz_nl' --type 'dual' --lang_src 'uz' --lang_tgt 'nl' --seed 47 --epochs 1 --batch_size 128
echo '--- TASK end: model=uz_nl seed=47'

echo '--- TASK start: model=uz_nl seed=48'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz_nl.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'uz_nl' --type 'dual' --lang_src 'uz' --lang_tgt 'nl' --seed 48 --epochs 1 --batch_size 128
echo '--- TASK end: model=uz_nl seed=48'

echo '--- TASK start: model=uz_nl seed=49'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz_nl.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'uz_nl' --type 'dual' --lang_src 'uz' --lang_tgt 'nl' --seed 49 --epochs 1 --batch_size 128
echo '--- TASK end: model=uz_nl seed=49'

echo '--- TASK start: model=uz_nl seed=50'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz_nl.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'uz_nl' --type 'dual' --lang_src 'uz' --lang_tgt 'nl' --seed 50 --epochs 1 --batch_size 128
echo '--- TASK end: model=uz_nl seed=50'

echo '--- TASK start: model=uz_nl seed=51'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz_nl.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'uz_nl' --type 'dual' --lang_src 'uz' --lang_tgt 'nl' --seed 51 --epochs 1 --batch_size 128
echo '--- TASK end: model=uz_nl seed=51'

echo '--- TASK start: model=uz_pl seed=42'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz_pl.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'uz_pl' --type 'dual' --lang_src 'uz' --lang_tgt 'pl' --seed 42 --epochs 1 --batch_size 128
echo '--- TASK end: model=uz_pl seed=42'

echo '--- TASK start: model=uz_pl seed=43'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz_pl.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'uz_pl' --type 'dual' --lang_src 'uz' --lang_tgt 'pl' --seed 43 --epochs 1 --batch_size 128
echo '--- TASK end: model=uz_pl seed=43'

echo '--- TASK start: model=uz_pl seed=44'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz_pl.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'uz_pl' --type 'dual' --lang_src 'uz' --lang_tgt 'pl' --seed 44 --epochs 1 --batch_size 128
echo '--- TASK end: model=uz_pl seed=44'

echo '--- TASK start: model=uz_pl seed=45'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz_pl.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'uz_pl' --type 'dual' --lang_src 'uz' --lang_tgt 'pl' --seed 45 --epochs 1 --batch_size 128
echo '--- TASK end: model=uz_pl seed=45'

echo '--- TASK start: model=uz_pl seed=46'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz_pl.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'uz_pl' --type 'dual' --lang_src 'uz' --lang_tgt 'pl' --seed 46 --epochs 1 --batch_size 128
echo '--- TASK end: model=uz_pl seed=46'

echo '--- TASK start: model=uz_pl seed=47'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz_pl.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'uz_pl' --type 'dual' --lang_src 'uz' --lang_tgt 'pl' --seed 47 --epochs 1 --batch_size 128
echo '--- TASK end: model=uz_pl seed=47'

echo '--- TASK start: model=uz_pl seed=48'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz_pl.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'uz_pl' --type 'dual' --lang_src 'uz' --lang_tgt 'pl' --seed 48 --epochs 1 --batch_size 128
echo '--- TASK end: model=uz_pl seed=48'

echo '--- TASK start: model=uz_pt seed=49'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz_pt.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'uz_pt' --type 'dual' --lang_src 'uz' --lang_tgt 'pt' --seed 49 --epochs 1 --batch_size 128
echo '--- TASK end: model=uz_pt seed=49'

echo '--- TASK start: model=uz_pt seed=50'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz_pt.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/uz.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/speaker_matrix_bilingual.csv' --model_id 'uz_pt' --type 'dual' --lang_src 'uz' --lang_tgt 'pt' --seed 50 --epochs 1 --batch_size 128
echo '--- TASK end: model=uz_pt seed=50'

