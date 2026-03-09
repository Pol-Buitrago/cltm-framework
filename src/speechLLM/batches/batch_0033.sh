#!/bin/bash
#SBATCH --account bsc88
#SBATCH --qos acc_bscls
#SBATCH --nodes 1
#SBATCH --gres gpu:1
#SBATCH --cpus-per-task 20
#SBATCH --time 0-12:00:00
#SBATCH --job-name=speechLLM_train
#SBATCH --output=/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/logs/speechLLM_train_%j.log
#SBATCH --error=/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/logs/speechLLM_train_%j.err

set -e

echo "=== Activating conda ==="

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /gpfs/projects/bsc88/speech/mm_s2st/scripts/paraling/gender_id_hubert/env/

echo "=== Running speechLLM training ==="

set -euo pipefail

module purge
# Carga tus módulos / activa conda aquí si hace falta. Ej:
# module load cuda/xx
# source /path/to/conda.sh && conda activate tu_env

echo '--- TASK start: model=cy_rw seed=46'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/cy_rw.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/cy.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_bilingual.csv' --model_id 'cy_rw' --type 'dual' --lang_src 'cy' --lang_tgt 'rw' --seed 46 --epochs 1 --batch_size 128
echo '--- TASK end: model=cy_rw seed=46'

echo '--- TASK start: model=cy_rw seed=47'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/cy_rw.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/cy.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_bilingual.csv' --model_id 'cy_rw' --type 'dual' --lang_src 'cy' --lang_tgt 'rw' --seed 47 --epochs 1 --batch_size 128
echo '--- TASK end: model=cy_rw seed=47'

echo '--- TASK start: model=cy_rw seed=48'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/cy_rw.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/cy.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_bilingual.csv' --model_id 'cy_rw' --type 'dual' --lang_src 'cy' --lang_tgt 'rw' --seed 48 --epochs 1 --batch_size 128
echo '--- TASK end: model=cy_rw seed=48'

echo '--- TASK start: model=cy_rw seed=49'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/cy_rw.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/cy.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_bilingual.csv' --model_id 'cy_rw' --type 'dual' --lang_src 'cy' --lang_tgt 'rw' --seed 49 --epochs 1 --batch_size 128
echo '--- TASK end: model=cy_rw seed=49'

echo '--- TASK start: model=cy_rw seed=50'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/cy_rw.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/cy.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_bilingual.csv' --model_id 'cy_rw' --type 'dual' --lang_src 'cy' --lang_tgt 'rw' --seed 50 --epochs 1 --batch_size 128
echo '--- TASK end: model=cy_rw seed=50'

echo '--- TASK start: model=cy_rw seed=51'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/cy_rw.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/cy.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_bilingual.csv' --model_id 'cy_rw' --type 'dual' --lang_src 'cy' --lang_tgt 'rw' --seed 51 --epochs 1 --batch_size 128
echo '--- TASK end: model=cy_rw seed=51'

echo '--- TASK start: model=cy_sv-SE seed=42'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/cy_sv-SE.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/cy.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_bilingual.csv' --model_id 'cy_sv-SE' --type 'dual' --lang_src 'cy' --lang_tgt 'sv-SE' --seed 42 --epochs 1 --batch_size 128
echo '--- TASK end: model=cy_sv-SE seed=42'

echo '--- TASK start: model=cy_sv-SE seed=43'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/cy_sv-SE.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/cy.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_bilingual.csv' --model_id 'cy_sv-SE' --type 'dual' --lang_src 'cy' --lang_tgt 'sv-SE' --seed 43 --epochs 1 --batch_size 128
echo '--- TASK end: model=cy_sv-SE seed=43'

echo '--- TASK start: model=cy_sv-SE seed=44'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/cy_sv-SE.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/cy.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_bilingual.csv' --model_id 'cy_sv-SE' --type 'dual' --lang_src 'cy' --lang_tgt 'sv-SE' --seed 44 --epochs 1 --batch_size 128
echo '--- TASK end: model=cy_sv-SE seed=44'

echo '--- TASK start: model=cy_sv-SE seed=45'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/cy_sv-SE.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/cy.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_bilingual.csv' --model_id 'cy_sv-SE' --type 'dual' --lang_src 'cy' --lang_tgt 'sv-SE' --seed 45 --epochs 1 --batch_size 128
echo '--- TASK end: model=cy_sv-SE seed=45'

echo '--- TASK start: model=cy_sv-SE seed=46'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/cy_sv-SE.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/cy.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_bilingual.csv' --model_id 'cy_sv-SE' --type 'dual' --lang_src 'cy' --lang_tgt 'sv-SE' --seed 46 --epochs 1 --batch_size 128
echo '--- TASK end: model=cy_sv-SE seed=46'

echo '--- TASK start: model=cy_sv-SE seed=47'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/cy_sv-SE.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/cy.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_bilingual.csv' --model_id 'cy_sv-SE' --type 'dual' --lang_src 'cy' --lang_tgt 'sv-SE' --seed 47 --epochs 1 --batch_size 128
echo '--- TASK end: model=cy_sv-SE seed=47'

echo '--- TASK start: model=cy_sv-SE seed=48'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/cy_sv-SE.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/cy.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_bilingual.csv' --model_id 'cy_sv-SE' --type 'dual' --lang_src 'cy' --lang_tgt 'sv-SE' --seed 48 --epochs 1 --batch_size 128
echo '--- TASK end: model=cy_sv-SE seed=48'

echo '--- TASK start: model=cy_sv-SE seed=49'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/cy_sv-SE.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/cy.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_bilingual.csv' --model_id 'cy_sv-SE' --type 'dual' --lang_src 'cy' --lang_tgt 'sv-SE' --seed 49 --epochs 1 --batch_size 128
echo '--- TASK end: model=cy_sv-SE seed=49'

echo '--- TASK start: model=cy_sv-SE seed=50'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/cy_sv-SE.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/cy.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_bilingual.csv' --model_id 'cy_sv-SE' --type 'dual' --lang_src 'cy' --lang_tgt 'sv-SE' --seed 50 --epochs 1 --batch_size 128
echo '--- TASK end: model=cy_sv-SE seed=50'

echo '--- TASK start: model=cy_sv-SE seed=51'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/cy_sv-SE.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/cy.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_bilingual.csv' --model_id 'cy_sv-SE' --type 'dual' --lang_src 'cy' --lang_tgt 'sv-SE' --seed 51 --epochs 1 --batch_size 128
echo '--- TASK end: model=cy_sv-SE seed=51'

echo '--- TASK start: model=cy_sw seed=42'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/cy_sw.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/cy.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_bilingual.csv' --model_id 'cy_sw' --type 'dual' --lang_src 'cy' --lang_tgt 'sw' --seed 42 --epochs 1 --batch_size 128
echo '--- TASK end: model=cy_sw seed=42'

echo '--- TASK start: model=cy_sw seed=43'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/cy_sw.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/cy.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_bilingual.csv' --model_id 'cy_sw' --type 'dual' --lang_src 'cy' --lang_tgt 'sw' --seed 43 --epochs 1 --batch_size 128
echo '--- TASK end: model=cy_sw seed=43'

echo '--- TASK start: model=cy_sw seed=44'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/cy_sw.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/cy.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_bilingual.csv' --model_id 'cy_sw' --type 'dual' --lang_src 'cy' --lang_tgt 'sw' --seed 44 --epochs 1 --batch_size 128
echo '--- TASK end: model=cy_sw seed=44'

echo '--- TASK start: model=cy_sw seed=45'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/cy_sw.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/cy.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_bilingual.csv' --model_id 'cy_sw' --type 'dual' --lang_src 'cy' --lang_tgt 'sw' --seed 45 --epochs 1 --batch_size 128
echo '--- TASK end: model=cy_sw seed=45'

echo '--- TASK start: model=cy_sw seed=46'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/cy_sw.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/cy.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_bilingual.csv' --model_id 'cy_sw' --type 'dual' --lang_src 'cy' --lang_tgt 'sw' --seed 46 --epochs 1 --batch_size 128
echo '--- TASK end: model=cy_sw seed=46'

echo '--- TASK start: model=cy_sw seed=47'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/cy_sw.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/cy.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_bilingual.csv' --model_id 'cy_sw' --type 'dual' --lang_src 'cy' --lang_tgt 'sw' --seed 47 --epochs 1 --batch_size 128
echo '--- TASK end: model=cy_sw seed=47'

echo '--- TASK start: model=cy_sw seed=48'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/cy_sw.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/cy.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_bilingual.csv' --model_id 'cy_sw' --type 'dual' --lang_src 'cy' --lang_tgt 'sw' --seed 48 --epochs 1 --batch_size 128
echo '--- TASK end: model=cy_sw seed=48'

echo '--- TASK start: model=cy_sw seed=49'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/cy_sw.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/cy.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_bilingual.csv' --model_id 'cy_sw' --type 'dual' --lang_src 'cy' --lang_tgt 'sw' --seed 49 --epochs 1 --batch_size 128
echo '--- TASK end: model=cy_sw seed=49'

echo '--- TASK start: model=cy_sw seed=50'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/cy_sw.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/cy.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_bilingual.csv' --model_id 'cy_sw' --type 'dual' --lang_src 'cy' --lang_tgt 'sw' --seed 50 --epochs 1 --batch_size 128
echo '--- TASK end: model=cy_sw seed=50'

echo '--- TASK start: model=cy_sw seed=51'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/cy_sw.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/cy.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_bilingual.csv' --model_id 'cy_sw' --type 'dual' --lang_src 'cy' --lang_tgt 'sw' --seed 51 --epochs 1 --batch_size 128
echo '--- TASK end: model=cy_sw seed=51'

echo '--- TASK start: model=cy_ta seed=42'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/cy_ta.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/cy.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_bilingual.csv' --model_id 'cy_ta' --type 'dual' --lang_src 'cy' --lang_tgt 'ta' --seed 42 --epochs 1 --batch_size 128
echo '--- TASK end: model=cy_ta seed=42'

echo '--- TASK start: model=cy_ta seed=43'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/cy_ta.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/cy.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_bilingual.csv' --model_id 'cy_ta' --type 'dual' --lang_src 'cy' --lang_tgt 'ta' --seed 43 --epochs 1 --batch_size 128
echo '--- TASK end: model=cy_ta seed=43'

echo '--- TASK start: model=cy_ta seed=44'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/cy_ta.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/cy.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_bilingual.csv' --model_id 'cy_ta' --type 'dual' --lang_src 'cy' --lang_tgt 'ta' --seed 44 --epochs 1 --batch_size 128
echo '--- TASK end: model=cy_ta seed=44'

echo '--- TASK start: model=cy_ta seed=45'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/cy_ta.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/cy.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_bilingual.csv' --model_id 'cy_ta' --type 'dual' --lang_src 'cy' --lang_tgt 'ta' --seed 45 --epochs 1 --batch_size 128
echo '--- TASK end: model=cy_ta seed=45'

echo '--- TASK start: model=cy_ta seed=46'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/cy_ta.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/cy.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_bilingual.csv' --model_id 'cy_ta' --type 'dual' --lang_src 'cy' --lang_tgt 'ta' --seed 46 --epochs 1 --batch_size 128
echo '--- TASK end: model=cy_ta seed=46'

echo '--- TASK start: model=cy_ta seed=47'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/cy_ta.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/cy.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_bilingual.csv' --model_id 'cy_ta' --type 'dual' --lang_src 'cy' --lang_tgt 'ta' --seed 47 --epochs 1 --batch_size 128
echo '--- TASK end: model=cy_ta seed=47'

echo '--- TASK start: model=cy_ta seed=48'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/cy_ta.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/cy.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_bilingual.csv' --model_id 'cy_ta' --type 'dual' --lang_src 'cy' --lang_tgt 'ta' --seed 48 --epochs 1 --batch_size 128
echo '--- TASK end: model=cy_ta seed=48'

echo '--- TASK start: model=cy_ta seed=49'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/cy_ta.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/cy.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_bilingual.csv' --model_id 'cy_ta' --type 'dual' --lang_src 'cy' --lang_tgt 'ta' --seed 49 --epochs 1 --batch_size 128
echo '--- TASK end: model=cy_ta seed=49'

echo '--- TASK start: model=cy_ta seed=50'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/cy_ta.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/cy.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_bilingual.csv' --model_id 'cy_ta' --type 'dual' --lang_src 'cy' --lang_tgt 'ta' --seed 50 --epochs 1 --batch_size 128
echo '--- TASK end: model=cy_ta seed=50'

echo '--- TASK start: model=cy_ta seed=51'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/cy_ta.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/cy.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_bilingual.csv' --model_id 'cy_ta' --type 'dual' --lang_src 'cy' --lang_tgt 'ta' --seed 51 --epochs 1 --batch_size 128
echo '--- TASK end: model=cy_ta seed=51'

echo '--- TASK start: model=cy_th seed=42'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/cy_th.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/cy.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_bilingual.csv' --model_id 'cy_th' --type 'dual' --lang_src 'cy' --lang_tgt 'th' --seed 42 --epochs 1 --batch_size 128
echo '--- TASK end: model=cy_th seed=42'

echo '--- TASK start: model=cy_th seed=43'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/cy_th.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/cy.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_bilingual.csv' --model_id 'cy_th' --type 'dual' --lang_src 'cy' --lang_tgt 'th' --seed 43 --epochs 1 --batch_size 128
echo '--- TASK end: model=cy_th seed=43'

echo '--- TASK start: model=cy_th seed=44'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/cy_th.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/cy.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_bilingual.csv' --model_id 'cy_th' --type 'dual' --lang_src 'cy' --lang_tgt 'th' --seed 44 --epochs 1 --batch_size 128
echo '--- TASK end: model=cy_th seed=44'

echo '--- TASK start: model=cy_th seed=45'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/cy_th.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/cy.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_bilingual.csv' --model_id 'cy_th' --type 'dual' --lang_src 'cy' --lang_tgt 'th' --seed 45 --epochs 1 --batch_size 128
echo '--- TASK end: model=cy_th seed=45'

echo '--- TASK start: model=cy_th seed=46'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/cy_th.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/cy.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_bilingual.csv' --model_id 'cy_th' --type 'dual' --lang_src 'cy' --lang_tgt 'th' --seed 46 --epochs 1 --batch_size 128
echo '--- TASK end: model=cy_th seed=46'

echo '--- TASK start: model=cy_th seed=47'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/cy_th.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/cy.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_bilingual.csv' --model_id 'cy_th' --type 'dual' --lang_src 'cy' --lang_tgt 'th' --seed 47 --epochs 1 --batch_size 128
echo '--- TASK end: model=cy_th seed=47'

echo '--- TASK start: model=cy_th seed=48'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/cy_th.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/cy.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_bilingual.csv' --model_id 'cy_th' --type 'dual' --lang_src 'cy' --lang_tgt 'th' --seed 48 --epochs 1 --batch_size 128
echo '--- TASK end: model=cy_th seed=48'

echo '--- TASK start: model=cy_th seed=49'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/cy_th.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/cy.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_bilingual.csv' --model_id 'cy_th' --type 'dual' --lang_src 'cy' --lang_tgt 'th' --seed 49 --epochs 1 --batch_size 128
echo '--- TASK end: model=cy_th seed=49'

echo '--- TASK start: model=cy_th seed=50'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/cy_th.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/cy.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_bilingual.csv' --model_id 'cy_th' --type 'dual' --lang_src 'cy' --lang_tgt 'th' --seed 50 --epochs 1 --batch_size 128
echo '--- TASK end: model=cy_th seed=50'

echo '--- TASK start: model=cy_th seed=51'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/cy_th.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/cy.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_bilingual.csv' --model_id 'cy_th' --type 'dual' --lang_src 'cy' --lang_tgt 'th' --seed 51 --epochs 1 --batch_size 128
echo '--- TASK end: model=cy_th seed=51'

echo '--- TASK start: model=cy_tr seed=42'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/cy_tr.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/cy.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_bilingual.csv' --model_id 'cy_tr' --type 'dual' --lang_src 'cy' --lang_tgt 'tr' --seed 42 --epochs 1 --batch_size 128
echo '--- TASK end: model=cy_tr seed=42'

echo '--- TASK start: model=cy_tr seed=43'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/cy_tr.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/cy.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_bilingual.csv' --model_id 'cy_tr' --type 'dual' --lang_src 'cy' --lang_tgt 'tr' --seed 43 --epochs 1 --batch_size 128
echo '--- TASK end: model=cy_tr seed=43'

echo '--- TASK start: model=cy_tr seed=44'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/cy_tr.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/cy.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_bilingual.csv' --model_id 'cy_tr' --type 'dual' --lang_src 'cy' --lang_tgt 'tr' --seed 44 --epochs 1 --batch_size 128
echo '--- TASK end: model=cy_tr seed=44'

echo '--- TASK start: model=cy_tr seed=45'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/cy_tr.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/cy.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_bilingual.csv' --model_id 'cy_tr' --type 'dual' --lang_src 'cy' --lang_tgt 'tr' --seed 45 --epochs 1 --batch_size 128
echo '--- TASK end: model=cy_tr seed=45'

echo '--- TASK start: model=cy_tr seed=46'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/cy_tr.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/cy.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_bilingual.csv' --model_id 'cy_tr' --type 'dual' --lang_src 'cy' --lang_tgt 'tr' --seed 46 --epochs 1 --batch_size 128
echo '--- TASK end: model=cy_tr seed=46'

echo '--- TASK start: model=cy_tr seed=47'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/cy_tr.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/cy.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_bilingual.csv' --model_id 'cy_tr' --type 'dual' --lang_src 'cy' --lang_tgt 'tr' --seed 47 --epochs 1 --batch_size 128
echo '--- TASK end: model=cy_tr seed=47'

echo '--- TASK start: model=cy_tr seed=48'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/cy_tr.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/cy.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_bilingual.csv' --model_id 'cy_tr' --type 'dual' --lang_src 'cy' --lang_tgt 'tr' --seed 48 --epochs 1 --batch_size 128
echo '--- TASK end: model=cy_tr seed=48'

echo '--- TASK start: model=cy_tr seed=49'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/cy_tr.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/cy.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_bilingual.csv' --model_id 'cy_tr' --type 'dual' --lang_src 'cy' --lang_tgt 'tr' --seed 49 --epochs 1 --batch_size 128
echo '--- TASK end: model=cy_tr seed=49'

echo '--- TASK start: model=cy_tr seed=50'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/cy_tr.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/cy.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_bilingual.csv' --model_id 'cy_tr' --type 'dual' --lang_src 'cy' --lang_tgt 'tr' --seed 50 --epochs 1 --batch_size 128
echo '--- TASK end: model=cy_tr seed=50'

echo '--- TASK start: model=cy_tr seed=51'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/cy_tr.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/cy.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_bilingual.csv' --model_id 'cy_tr' --type 'dual' --lang_src 'cy' --lang_tgt 'tr' --seed 51 --epochs 1 --batch_size 128
echo '--- TASK end: model=cy_tr seed=51'

echo '--- TASK start: model=cy_ug seed=42'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/cy_ug.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/cy.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_bilingual.csv' --model_id 'cy_ug' --type 'dual' --lang_src 'cy' --lang_tgt 'ug' --seed 42 --epochs 1 --batch_size 128
echo '--- TASK end: model=cy_ug seed=42'

echo '--- TASK start: model=cy_ug seed=43'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/cy_ug.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/cy.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_bilingual.csv' --model_id 'cy_ug' --type 'dual' --lang_src 'cy' --lang_tgt 'ug' --seed 43 --epochs 1 --batch_size 128
echo '--- TASK end: model=cy_ug seed=43'

echo '--- TASK start: model=cy_ug seed=44'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/cy_ug.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/cy.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_bilingual.csv' --model_id 'cy_ug' --type 'dual' --lang_src 'cy' --lang_tgt 'ug' --seed 44 --epochs 1 --batch_size 128
echo '--- TASK end: model=cy_ug seed=44'

echo '--- TASK start: model=cy_ug seed=45'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/cy_ug.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/cy.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_bilingual.csv' --model_id 'cy_ug' --type 'dual' --lang_src 'cy' --lang_tgt 'ug' --seed 45 --epochs 1 --batch_size 128
echo '--- TASK end: model=cy_ug seed=45'

echo '--- TASK start: model=cy_ug seed=46'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/cy_ug.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/cy.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_bilingual.csv' --model_id 'cy_ug' --type 'dual' --lang_src 'cy' --lang_tgt 'ug' --seed 46 --epochs 1 --batch_size 128
echo '--- TASK end: model=cy_ug seed=46'

echo '--- TASK start: model=cy_ug seed=47'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/cy_ug.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/cy.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_bilingual.csv' --model_id 'cy_ug' --type 'dual' --lang_src 'cy' --lang_tgt 'ug' --seed 47 --epochs 1 --batch_size 128
echo '--- TASK end: model=cy_ug seed=47'

echo '--- TASK start: model=cy_ug seed=48'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/cy_ug.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/cy.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_bilingual.csv' --model_id 'cy_ug' --type 'dual' --lang_src 'cy' --lang_tgt 'ug' --seed 48 --epochs 1 --batch_size 128
echo '--- TASK end: model=cy_ug seed=48'

echo '--- TASK start: model=cy_ug seed=49'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/cy_ug.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/cy.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_bilingual.csv' --model_id 'cy_ug' --type 'dual' --lang_src 'cy' --lang_tgt 'ug' --seed 49 --epochs 1 --batch_size 128
echo '--- TASK end: model=cy_ug seed=49'

echo '--- TASK start: model=cy_ug seed=50'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/cy_ug.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/cy.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_bilingual.csv' --model_id 'cy_ug' --type 'dual' --lang_src 'cy' --lang_tgt 'ug' --seed 50 --epochs 1 --batch_size 128
echo '--- TASK end: model=cy_ug seed=50'

echo '--- TASK start: model=cy_ug seed=51'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/cy_ug.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/cy.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_bilingual.csv' --model_id 'cy_ug' --type 'dual' --lang_src 'cy' --lang_tgt 'ug' --seed 51 --epochs 1 --batch_size 128
echo '--- TASK end: model=cy_ug seed=51'

echo '--- TASK start: model=cy_uk seed=42'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/cy_uk.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/cy.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_bilingual.csv' --model_id 'cy_uk' --type 'dual' --lang_src 'cy' --lang_tgt 'uk' --seed 42 --epochs 1 --batch_size 128
echo '--- TASK end: model=cy_uk seed=42'

echo '--- TASK start: model=cy_uk seed=43'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/cy_uk.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/cy.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_bilingual.csv' --model_id 'cy_uk' --type 'dual' --lang_src 'cy' --lang_tgt 'uk' --seed 43 --epochs 1 --batch_size 128
echo '--- TASK end: model=cy_uk seed=43'

echo '--- TASK start: model=cy_uk seed=44'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/cy_uk.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/cy.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_bilingual.csv' --model_id 'cy_uk' --type 'dual' --lang_src 'cy' --lang_tgt 'uk' --seed 44 --epochs 1 --batch_size 128
echo '--- TASK end: model=cy_uk seed=44'

echo '--- TASK start: model=cy_uk seed=45'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/cy_uk.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/cy.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_bilingual.csv' --model_id 'cy_uk' --type 'dual' --lang_src 'cy' --lang_tgt 'uk' --seed 45 --epochs 1 --batch_size 128
echo '--- TASK end: model=cy_uk seed=45'

echo '--- TASK start: model=cy_uk seed=46'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/cy_uk.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/cy.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_bilingual.csv' --model_id 'cy_uk' --type 'dual' --lang_src 'cy' --lang_tgt 'uk' --seed 46 --epochs 1 --batch_size 128
echo '--- TASK end: model=cy_uk seed=46'

echo '--- TASK start: model=cy_uk seed=47'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/cy_uk.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/cy.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_bilingual.csv' --model_id 'cy_uk' --type 'dual' --lang_src 'cy' --lang_tgt 'uk' --seed 47 --epochs 1 --batch_size 128
echo '--- TASK end: model=cy_uk seed=47'

echo '--- TASK start: model=cy_uk seed=48'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/cy_uk.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/cy.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_bilingual.csv' --model_id 'cy_uk' --type 'dual' --lang_src 'cy' --lang_tgt 'uk' --seed 48 --epochs 1 --batch_size 128
echo '--- TASK end: model=cy_uk seed=48'

echo '--- TASK start: model=cy_uk seed=49'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/cy_uk.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/cy.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_bilingual.csv' --model_id 'cy_uk' --type 'dual' --lang_src 'cy' --lang_tgt 'uk' --seed 49 --epochs 1 --batch_size 128
echo '--- TASK end: model=cy_uk seed=49'

echo '--- TASK start: model=cy_uk seed=50'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/cy_uk.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/cy.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_bilingual.csv' --model_id 'cy_uk' --type 'dual' --lang_src 'cy' --lang_tgt 'uk' --seed 50 --epochs 1 --batch_size 128
echo '--- TASK end: model=cy_uk seed=50'

echo '--- TASK start: model=cy_uk seed=51'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/cy_uk.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/cy.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_bilingual.csv' --model_id 'cy_uk' --type 'dual' --lang_src 'cy' --lang_tgt 'uk' --seed 51 --epochs 1 --batch_size 128
echo '--- TASK end: model=cy_uk seed=51'

echo '--- TASK start: model=cy_ur seed=42'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/cy_ur.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/cy.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_bilingual.csv' --model_id 'cy_ur' --type 'dual' --lang_src 'cy' --lang_tgt 'ur' --seed 42 --epochs 1 --batch_size 128
echo '--- TASK end: model=cy_ur seed=42'

echo '--- TASK start: model=cy_ur seed=43'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/cy_ur.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/cy.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_bilingual.csv' --model_id 'cy_ur' --type 'dual' --lang_src 'cy' --lang_tgt 'ur' --seed 43 --epochs 1 --batch_size 128
echo '--- TASK end: model=cy_ur seed=43'

echo '--- TASK start: model=cy_ur seed=44'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/cy_ur.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/cy.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_bilingual.csv' --model_id 'cy_ur' --type 'dual' --lang_src 'cy' --lang_tgt 'ur' --seed 44 --epochs 1 --batch_size 128
echo '--- TASK end: model=cy_ur seed=44'

echo '--- TASK start: model=cy_ur seed=45'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/cy_ur.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/cy.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_bilingual.csv' --model_id 'cy_ur' --type 'dual' --lang_src 'cy' --lang_tgt 'ur' --seed 45 --epochs 1 --batch_size 128
echo '--- TASK end: model=cy_ur seed=45'

echo '--- TASK start: model=cy_ur seed=46'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/cy_ur.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/cy.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_bilingual.csv' --model_id 'cy_ur' --type 'dual' --lang_src 'cy' --lang_tgt 'ur' --seed 46 --epochs 1 --batch_size 128
echo '--- TASK end: model=cy_ur seed=46'

echo '--- TASK start: model=cy_ur seed=47'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/cy_ur.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/cy.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_bilingual.csv' --model_id 'cy_ur' --type 'dual' --lang_src 'cy' --lang_tgt 'ur' --seed 47 --epochs 1 --batch_size 128
echo '--- TASK end: model=cy_ur seed=47'

echo '--- TASK start: model=cy_ur seed=48'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/cy_ur.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/cy.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_bilingual.csv' --model_id 'cy_ur' --type 'dual' --lang_src 'cy' --lang_tgt 'ur' --seed 48 --epochs 1 --batch_size 128
echo '--- TASK end: model=cy_ur seed=48'

echo '--- TASK start: model=cy_ur seed=49'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/cy_ur.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/cy.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_bilingual.csv' --model_id 'cy_ur' --type 'dual' --lang_src 'cy' --lang_tgt 'ur' --seed 49 --epochs 1 --batch_size 128
echo '--- TASK end: model=cy_ur seed=49'

echo '--- TASK start: model=cy_ur seed=50'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/cy_ur.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/cy.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_bilingual.csv' --model_id 'cy_ur' --type 'dual' --lang_src 'cy' --lang_tgt 'ur' --seed 50 --epochs 1 --batch_size 128
echo '--- TASK end: model=cy_ur seed=50'

echo '--- TASK start: model=cy_ur seed=51'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/cy_ur.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/cy.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_bilingual.csv' --model_id 'cy_ur' --type 'dual' --lang_src 'cy' --lang_tgt 'ur' --seed 51 --epochs 1 --batch_size 128
echo '--- TASK end: model=cy_ur seed=51'

echo '--- TASK start: model=cy_uz seed=42'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/cy_uz.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/cy.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_bilingual.csv' --model_id 'cy_uz' --type 'dual' --lang_src 'cy' --lang_tgt 'uz' --seed 42 --epochs 1 --batch_size 128
echo '--- TASK end: model=cy_uz seed=42'

echo '--- TASK start: model=cy_uz seed=43'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/cy_uz.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/cy.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_bilingual.csv' --model_id 'cy_uz' --type 'dual' --lang_src 'cy' --lang_tgt 'uz' --seed 43 --epochs 1 --batch_size 128
echo '--- TASK end: model=cy_uz seed=43'

echo '--- TASK start: model=cy_uz seed=44'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/cy_uz.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/cy.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_bilingual.csv' --model_id 'cy_uz' --type 'dual' --lang_src 'cy' --lang_tgt 'uz' --seed 44 --epochs 1 --batch_size 128
echo '--- TASK end: model=cy_uz seed=44'

echo '--- TASK start: model=cy_uz seed=45'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/cy_uz.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/cy.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_bilingual.csv' --model_id 'cy_uz' --type 'dual' --lang_src 'cy' --lang_tgt 'uz' --seed 45 --epochs 1 --batch_size 128
echo '--- TASK end: model=cy_uz seed=45'

echo '--- TASK start: model=cy_uz seed=46'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/cy_uz.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/cy.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_bilingual.csv' --model_id 'cy_uz' --type 'dual' --lang_src 'cy' --lang_tgt 'uz' --seed 46 --epochs 1 --batch_size 128
echo '--- TASK end: model=cy_uz seed=46'

echo '--- TASK start: model=cy_uz seed=47'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/cy_uz.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/cy.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_bilingual.csv' --model_id 'cy_uz' --type 'dual' --lang_src 'cy' --lang_tgt 'uz' --seed 47 --epochs 1 --batch_size 128
echo '--- TASK end: model=cy_uz seed=47'

echo '--- TASK start: model=cy_uz seed=48'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/cy_uz.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/cy.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_bilingual.csv' --model_id 'cy_uz' --type 'dual' --lang_src 'cy' --lang_tgt 'uz' --seed 48 --epochs 1 --batch_size 128
echo '--- TASK end: model=cy_uz seed=48'

echo '--- TASK start: model=cy_uz seed=49'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/cy_uz.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/cy.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_bilingual.csv' --model_id 'cy_uz' --type 'dual' --lang_src 'cy' --lang_tgt 'uz' --seed 49 --epochs 1 --batch_size 128
echo '--- TASK end: model=cy_uz seed=49'

echo '--- TASK start: model=cy_uz seed=50'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/cy_uz.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/cy.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_bilingual.csv' --model_id 'cy_uz' --type 'dual' --lang_src 'cy' --lang_tgt 'uz' --seed 50 --epochs 1 --batch_size 128
echo '--- TASK end: model=cy_uz seed=50'

echo '--- TASK start: model=cy_uz seed=51'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/cy_uz.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/cy.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_bilingual.csv' --model_id 'cy_uz' --type 'dual' --lang_src 'cy' --lang_tgt 'uz' --seed 51 --epochs 1 --batch_size 128
echo '--- TASK end: model=cy_uz seed=51'

echo '--- TASK start: model=cy_yue seed=42'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/cy_yue.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/cy.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_bilingual.csv' --model_id 'cy_yue' --type 'dual' --lang_src 'cy' --lang_tgt 'yue' --seed 42 --epochs 1 --batch_size 128
echo '--- TASK end: model=cy_yue seed=42'

echo '--- TASK start: model=cy_yue seed=43'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/cy_yue.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/cy.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_bilingual.csv' --model_id 'cy_yue' --type 'dual' --lang_src 'cy' --lang_tgt 'yue' --seed 43 --epochs 1 --batch_size 128
echo '--- TASK end: model=cy_yue seed=43'

echo '--- TASK start: model=cy_yue seed=44'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/cy_yue.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/cy.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_bilingual.csv' --model_id 'cy_yue' --type 'dual' --lang_src 'cy' --lang_tgt 'yue' --seed 44 --epochs 1 --batch_size 128
echo '--- TASK end: model=cy_yue seed=44'

echo '--- TASK start: model=cy_yue seed=45'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/cy_yue.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/cy.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_bilingual.csv' --model_id 'cy_yue' --type 'dual' --lang_src 'cy' --lang_tgt 'yue' --seed 45 --epochs 1 --batch_size 128
echo '--- TASK end: model=cy_yue seed=45'

