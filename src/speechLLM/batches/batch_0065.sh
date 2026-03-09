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

echo '--- TASK start: model=gl_ckb seed=46'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/gl_ckb.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/gl.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_bilingual.csv' --model_id 'gl_ckb' --type 'dual' --lang_src 'gl' --lang_tgt 'ckb' --seed 46 --epochs 1 --batch_size 128
echo '--- TASK end: model=gl_ckb seed=46'

echo '--- TASK start: model=gl_ckb seed=47'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/gl_ckb.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/gl.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_bilingual.csv' --model_id 'gl_ckb' --type 'dual' --lang_src 'gl' --lang_tgt 'ckb' --seed 47 --epochs 1 --batch_size 128
echo '--- TASK end: model=gl_ckb seed=47'

echo '--- TASK start: model=gl_ckb seed=48'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/gl_ckb.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/gl.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_bilingual.csv' --model_id 'gl_ckb' --type 'dual' --lang_src 'gl' --lang_tgt 'ckb' --seed 48 --epochs 1 --batch_size 128
echo '--- TASK end: model=gl_ckb seed=48'

echo '--- TASK start: model=gl_ckb seed=49'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/gl_ckb.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/gl.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_bilingual.csv' --model_id 'gl_ckb' --type 'dual' --lang_src 'gl' --lang_tgt 'ckb' --seed 49 --epochs 1 --batch_size 128
echo '--- TASK end: model=gl_ckb seed=49'

echo '--- TASK start: model=gl_ckb seed=50'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/gl_ckb.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/gl.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_bilingual.csv' --model_id 'gl_ckb' --type 'dual' --lang_src 'gl' --lang_tgt 'ckb' --seed 50 --epochs 1 --batch_size 128
echo '--- TASK end: model=gl_ckb seed=50'

echo '--- TASK start: model=gl_ckb seed=51'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/gl_ckb.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/gl.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_bilingual.csv' --model_id 'gl_ckb' --type 'dual' --lang_src 'gl' --lang_tgt 'ckb' --seed 51 --epochs 1 --batch_size 128
echo '--- TASK end: model=gl_ckb seed=51'

echo '--- TASK start: model=gl_cs seed=42'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/gl_cs.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/gl.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_bilingual.csv' --model_id 'gl_cs' --type 'dual' --lang_src 'gl' --lang_tgt 'cs' --seed 42 --epochs 1 --batch_size 128
echo '--- TASK end: model=gl_cs seed=42'

echo '--- TASK start: model=gl_cs seed=43'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/gl_cs.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/gl.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_bilingual.csv' --model_id 'gl_cs' --type 'dual' --lang_src 'gl' --lang_tgt 'cs' --seed 43 --epochs 1 --batch_size 128
echo '--- TASK end: model=gl_cs seed=43'

echo '--- TASK start: model=gl_cs seed=44'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/gl_cs.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/gl.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_bilingual.csv' --model_id 'gl_cs' --type 'dual' --lang_src 'gl' --lang_tgt 'cs' --seed 44 --epochs 1 --batch_size 128
echo '--- TASK end: model=gl_cs seed=44'

echo '--- TASK start: model=gl_cs seed=45'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/gl_cs.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/gl.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_bilingual.csv' --model_id 'gl_cs' --type 'dual' --lang_src 'gl' --lang_tgt 'cs' --seed 45 --epochs 1 --batch_size 128
echo '--- TASK end: model=gl_cs seed=45'

echo '--- TASK start: model=gl_cs seed=46'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/gl_cs.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/gl.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_bilingual.csv' --model_id 'gl_cs' --type 'dual' --lang_src 'gl' --lang_tgt 'cs' --seed 46 --epochs 1 --batch_size 128
echo '--- TASK end: model=gl_cs seed=46'

echo '--- TASK start: model=gl_cs seed=47'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/gl_cs.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/gl.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_bilingual.csv' --model_id 'gl_cs' --type 'dual' --lang_src 'gl' --lang_tgt 'cs' --seed 47 --epochs 1 --batch_size 128
echo '--- TASK end: model=gl_cs seed=47'

echo '--- TASK start: model=gl_cs seed=48'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/gl_cs.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/gl.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_bilingual.csv' --model_id 'gl_cs' --type 'dual' --lang_src 'gl' --lang_tgt 'cs' --seed 48 --epochs 1 --batch_size 128
echo '--- TASK end: model=gl_cs seed=48'

echo '--- TASK start: model=gl_cs seed=49'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/gl_cs.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/gl.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_bilingual.csv' --model_id 'gl_cs' --type 'dual' --lang_src 'gl' --lang_tgt 'cs' --seed 49 --epochs 1 --batch_size 128
echo '--- TASK end: model=gl_cs seed=49'

echo '--- TASK start: model=gl_cs seed=50'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/gl_cs.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/gl.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_bilingual.csv' --model_id 'gl_cs' --type 'dual' --lang_src 'gl' --lang_tgt 'cs' --seed 50 --epochs 1 --batch_size 128
echo '--- TASK end: model=gl_cs seed=50'

echo '--- TASK start: model=gl_cs seed=51'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/gl_cs.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/gl.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_bilingual.csv' --model_id 'gl_cs' --type 'dual' --lang_src 'gl' --lang_tgt 'cs' --seed 51 --epochs 1 --batch_size 128
echo '--- TASK end: model=gl_cs seed=51'

echo '--- TASK start: model=gl_cy seed=42'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/gl_cy.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/gl.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_bilingual.csv' --model_id 'gl_cy' --type 'dual' --lang_src 'gl' --lang_tgt 'cy' --seed 42 --epochs 1 --batch_size 128
echo '--- TASK end: model=gl_cy seed=42'

echo '--- TASK start: model=gl_cy seed=43'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/gl_cy.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/gl.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_bilingual.csv' --model_id 'gl_cy' --type 'dual' --lang_src 'gl' --lang_tgt 'cy' --seed 43 --epochs 1 --batch_size 128
echo '--- TASK end: model=gl_cy seed=43'

echo '--- TASK start: model=gl_cy seed=44'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/gl_cy.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/gl.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_bilingual.csv' --model_id 'gl_cy' --type 'dual' --lang_src 'gl' --lang_tgt 'cy' --seed 44 --epochs 1 --batch_size 128
echo '--- TASK end: model=gl_cy seed=44'

echo '--- TASK start: model=gl_cy seed=45'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/gl_cy.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/gl.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_bilingual.csv' --model_id 'gl_cy' --type 'dual' --lang_src 'gl' --lang_tgt 'cy' --seed 45 --epochs 1 --batch_size 128
echo '--- TASK end: model=gl_cy seed=45'

echo '--- TASK start: model=gl_cy seed=46'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/gl_cy.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/gl.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_bilingual.csv' --model_id 'gl_cy' --type 'dual' --lang_src 'gl' --lang_tgt 'cy' --seed 46 --epochs 1 --batch_size 128
echo '--- TASK end: model=gl_cy seed=46'

echo '--- TASK start: model=gl_cy seed=47'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/gl_cy.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/gl.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_bilingual.csv' --model_id 'gl_cy' --type 'dual' --lang_src 'gl' --lang_tgt 'cy' --seed 47 --epochs 1 --batch_size 128
echo '--- TASK end: model=gl_cy seed=47'

echo '--- TASK start: model=gl_cy seed=48'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/gl_cy.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/gl.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_bilingual.csv' --model_id 'gl_cy' --type 'dual' --lang_src 'gl' --lang_tgt 'cy' --seed 48 --epochs 1 --batch_size 128
echo '--- TASK end: model=gl_cy seed=48'

echo '--- TASK start: model=gl_cy seed=49'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/gl_cy.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/gl.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_bilingual.csv' --model_id 'gl_cy' --type 'dual' --lang_src 'gl' --lang_tgt 'cy' --seed 49 --epochs 1 --batch_size 128
echo '--- TASK end: model=gl_cy seed=49'

echo '--- TASK start: model=gl_cy seed=50'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/gl_cy.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/gl.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_bilingual.csv' --model_id 'gl_cy' --type 'dual' --lang_src 'gl' --lang_tgt 'cy' --seed 50 --epochs 1 --batch_size 128
echo '--- TASK end: model=gl_cy seed=50'

echo '--- TASK start: model=gl_cy seed=51'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/gl_cy.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/gl.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_bilingual.csv' --model_id 'gl_cy' --type 'dual' --lang_src 'gl' --lang_tgt 'cy' --seed 51 --epochs 1 --batch_size 128
echo '--- TASK end: model=gl_cy seed=51'

echo '--- TASK start: model=gl_de seed=42'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/gl_de.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/gl.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_bilingual.csv' --model_id 'gl_de' --type 'dual' --lang_src 'gl' --lang_tgt 'de' --seed 42 --epochs 1 --batch_size 128
echo '--- TASK end: model=gl_de seed=42'

echo '--- TASK start: model=gl_de seed=43'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/gl_de.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/gl.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_bilingual.csv' --model_id 'gl_de' --type 'dual' --lang_src 'gl' --lang_tgt 'de' --seed 43 --epochs 1 --batch_size 128
echo '--- TASK end: model=gl_de seed=43'

echo '--- TASK start: model=gl_de seed=44'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/gl_de.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/gl.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_bilingual.csv' --model_id 'gl_de' --type 'dual' --lang_src 'gl' --lang_tgt 'de' --seed 44 --epochs 1 --batch_size 128
echo '--- TASK end: model=gl_de seed=44'

echo '--- TASK start: model=gl_de seed=45'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/gl_de.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/gl.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_bilingual.csv' --model_id 'gl_de' --type 'dual' --lang_src 'gl' --lang_tgt 'de' --seed 45 --epochs 1 --batch_size 128
echo '--- TASK end: model=gl_de seed=45'

echo '--- TASK start: model=gl_de seed=46'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/gl_de.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/gl.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_bilingual.csv' --model_id 'gl_de' --type 'dual' --lang_src 'gl' --lang_tgt 'de' --seed 46 --epochs 1 --batch_size 128
echo '--- TASK end: model=gl_de seed=46'

echo '--- TASK start: model=gl_de seed=47'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/gl_de.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/gl.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_bilingual.csv' --model_id 'gl_de' --type 'dual' --lang_src 'gl' --lang_tgt 'de' --seed 47 --epochs 1 --batch_size 128
echo '--- TASK end: model=gl_de seed=47'

echo '--- TASK start: model=gl_de seed=48'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/gl_de.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/gl.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_bilingual.csv' --model_id 'gl_de' --type 'dual' --lang_src 'gl' --lang_tgt 'de' --seed 48 --epochs 1 --batch_size 128
echo '--- TASK end: model=gl_de seed=48'

echo '--- TASK start: model=gl_de seed=49'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/gl_de.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/gl.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_bilingual.csv' --model_id 'gl_de' --type 'dual' --lang_src 'gl' --lang_tgt 'de' --seed 49 --epochs 1 --batch_size 128
echo '--- TASK end: model=gl_de seed=49'

echo '--- TASK start: model=gl_de seed=50'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/gl_de.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/gl.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_bilingual.csv' --model_id 'gl_de' --type 'dual' --lang_src 'gl' --lang_tgt 'de' --seed 50 --epochs 1 --batch_size 128
echo '--- TASK end: model=gl_de seed=50'

echo '--- TASK start: model=gl_de seed=51'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/gl_de.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/gl.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_bilingual.csv' --model_id 'gl_de' --type 'dual' --lang_src 'gl' --lang_tgt 'de' --seed 51 --epochs 1 --batch_size 128
echo '--- TASK end: model=gl_de seed=51'

echo '--- TASK start: model=gl_en seed=42'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/gl_en.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/gl.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_bilingual.csv' --model_id 'gl_en' --type 'dual' --lang_src 'gl' --lang_tgt 'en' --seed 42 --epochs 1 --batch_size 128
echo '--- TASK end: model=gl_en seed=42'

echo '--- TASK start: model=gl_en seed=43'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/gl_en.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/gl.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_bilingual.csv' --model_id 'gl_en' --type 'dual' --lang_src 'gl' --lang_tgt 'en' --seed 43 --epochs 1 --batch_size 128
echo '--- TASK end: model=gl_en seed=43'

echo '--- TASK start: model=gl_en seed=44'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/gl_en.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/gl.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_bilingual.csv' --model_id 'gl_en' --type 'dual' --lang_src 'gl' --lang_tgt 'en' --seed 44 --epochs 1 --batch_size 128
echo '--- TASK end: model=gl_en seed=44'

echo '--- TASK start: model=gl_en seed=45'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/gl_en.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/gl.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_bilingual.csv' --model_id 'gl_en' --type 'dual' --lang_src 'gl' --lang_tgt 'en' --seed 45 --epochs 1 --batch_size 128
echo '--- TASK end: model=gl_en seed=45'

echo '--- TASK start: model=gl_en seed=46'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/gl_en.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/gl.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_bilingual.csv' --model_id 'gl_en' --type 'dual' --lang_src 'gl' --lang_tgt 'en' --seed 46 --epochs 1 --batch_size 128
echo '--- TASK end: model=gl_en seed=46'

echo '--- TASK start: model=gl_en seed=47'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/gl_en.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/gl.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_bilingual.csv' --model_id 'gl_en' --type 'dual' --lang_src 'gl' --lang_tgt 'en' --seed 47 --epochs 1 --batch_size 128
echo '--- TASK end: model=gl_en seed=47'

echo '--- TASK start: model=gl_en seed=48'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/gl_en.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/gl.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_bilingual.csv' --model_id 'gl_en' --type 'dual' --lang_src 'gl' --lang_tgt 'en' --seed 48 --epochs 1 --batch_size 128
echo '--- TASK end: model=gl_en seed=48'

echo '--- TASK start: model=gl_en seed=49'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/gl_en.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/gl.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_bilingual.csv' --model_id 'gl_en' --type 'dual' --lang_src 'gl' --lang_tgt 'en' --seed 49 --epochs 1 --batch_size 128
echo '--- TASK end: model=gl_en seed=49'

echo '--- TASK start: model=gl_en seed=50'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/gl_en.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/gl.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_bilingual.csv' --model_id 'gl_en' --type 'dual' --lang_src 'gl' --lang_tgt 'en' --seed 50 --epochs 1 --batch_size 128
echo '--- TASK end: model=gl_en seed=50'

echo '--- TASK start: model=gl_en seed=51'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/gl_en.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/gl.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_bilingual.csv' --model_id 'gl_en' --type 'dual' --lang_src 'gl' --lang_tgt 'en' --seed 51 --epochs 1 --batch_size 128
echo '--- TASK end: model=gl_en seed=51'

echo '--- TASK start: model=gl_eo seed=42'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/gl_eo.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/gl.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_bilingual.csv' --model_id 'gl_eo' --type 'dual' --lang_src 'gl' --lang_tgt 'eo' --seed 42 --epochs 1 --batch_size 128
echo '--- TASK end: model=gl_eo seed=42'

echo '--- TASK start: model=gl_eo seed=43'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/gl_eo.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/gl.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_bilingual.csv' --model_id 'gl_eo' --type 'dual' --lang_src 'gl' --lang_tgt 'eo' --seed 43 --epochs 1 --batch_size 128
echo '--- TASK end: model=gl_eo seed=43'

echo '--- TASK start: model=gl_eo seed=44'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/gl_eo.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/gl.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_bilingual.csv' --model_id 'gl_eo' --type 'dual' --lang_src 'gl' --lang_tgt 'eo' --seed 44 --epochs 1 --batch_size 128
echo '--- TASK end: model=gl_eo seed=44'

echo '--- TASK start: model=gl_eo seed=45'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/gl_eo.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/gl.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_bilingual.csv' --model_id 'gl_eo' --type 'dual' --lang_src 'gl' --lang_tgt 'eo' --seed 45 --epochs 1 --batch_size 128
echo '--- TASK end: model=gl_eo seed=45'

echo '--- TASK start: model=gl_eo seed=46'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/gl_eo.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/gl.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_bilingual.csv' --model_id 'gl_eo' --type 'dual' --lang_src 'gl' --lang_tgt 'eo' --seed 46 --epochs 1 --batch_size 128
echo '--- TASK end: model=gl_eo seed=46'

echo '--- TASK start: model=gl_eo seed=47'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/gl_eo.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/gl.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_bilingual.csv' --model_id 'gl_eo' --type 'dual' --lang_src 'gl' --lang_tgt 'eo' --seed 47 --epochs 1 --batch_size 128
echo '--- TASK end: model=gl_eo seed=47'

echo '--- TASK start: model=gl_eo seed=48'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/gl_eo.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/gl.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_bilingual.csv' --model_id 'gl_eo' --type 'dual' --lang_src 'gl' --lang_tgt 'eo' --seed 48 --epochs 1 --batch_size 128
echo '--- TASK end: model=gl_eo seed=48'

echo '--- TASK start: model=gl_eo seed=49'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/gl_eo.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/gl.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_bilingual.csv' --model_id 'gl_eo' --type 'dual' --lang_src 'gl' --lang_tgt 'eo' --seed 49 --epochs 1 --batch_size 128
echo '--- TASK end: model=gl_eo seed=49'

echo '--- TASK start: model=gl_eo seed=50'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/gl_eo.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/gl.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_bilingual.csv' --model_id 'gl_eo' --type 'dual' --lang_src 'gl' --lang_tgt 'eo' --seed 50 --epochs 1 --batch_size 128
echo '--- TASK end: model=gl_eo seed=50'

echo '--- TASK start: model=gl_eo seed=51'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/gl_eo.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/gl.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_bilingual.csv' --model_id 'gl_eo' --type 'dual' --lang_src 'gl' --lang_tgt 'eo' --seed 51 --epochs 1 --batch_size 128
echo '--- TASK end: model=gl_eo seed=51'

echo '--- TASK start: model=gl_es seed=42'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/gl_es.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/gl.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_bilingual.csv' --model_id 'gl_es' --type 'dual' --lang_src 'gl' --lang_tgt 'es' --seed 42 --epochs 1 --batch_size 128
echo '--- TASK end: model=gl_es seed=42'

echo '--- TASK start: model=gl_es seed=43'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/gl_es.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/gl.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_bilingual.csv' --model_id 'gl_es' --type 'dual' --lang_src 'gl' --lang_tgt 'es' --seed 43 --epochs 1 --batch_size 128
echo '--- TASK end: model=gl_es seed=43'

echo '--- TASK start: model=gl_es seed=44'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/gl_es.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/gl.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_bilingual.csv' --model_id 'gl_es' --type 'dual' --lang_src 'gl' --lang_tgt 'es' --seed 44 --epochs 1 --batch_size 128
echo '--- TASK end: model=gl_es seed=44'

echo '--- TASK start: model=gl_es seed=45'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/gl_es.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/gl.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_bilingual.csv' --model_id 'gl_es' --type 'dual' --lang_src 'gl' --lang_tgt 'es' --seed 45 --epochs 1 --batch_size 128
echo '--- TASK end: model=gl_es seed=45'

echo '--- TASK start: model=gl_es seed=46'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/gl_es.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/gl.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_bilingual.csv' --model_id 'gl_es' --type 'dual' --lang_src 'gl' --lang_tgt 'es' --seed 46 --epochs 1 --batch_size 128
echo '--- TASK end: model=gl_es seed=46'

echo '--- TASK start: model=gl_es seed=47'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/gl_es.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/gl.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_bilingual.csv' --model_id 'gl_es' --type 'dual' --lang_src 'gl' --lang_tgt 'es' --seed 47 --epochs 1 --batch_size 128
echo '--- TASK end: model=gl_es seed=47'

echo '--- TASK start: model=gl_es seed=48'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/gl_es.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/gl.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_bilingual.csv' --model_id 'gl_es' --type 'dual' --lang_src 'gl' --lang_tgt 'es' --seed 48 --epochs 1 --batch_size 128
echo '--- TASK end: model=gl_es seed=48'

echo '--- TASK start: model=gl_es seed=49'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/gl_es.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/gl.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_bilingual.csv' --model_id 'gl_es' --type 'dual' --lang_src 'gl' --lang_tgt 'es' --seed 49 --epochs 1 --batch_size 128
echo '--- TASK end: model=gl_es seed=49'

echo '--- TASK start: model=gl_es seed=50'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/gl_es.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/gl.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_bilingual.csv' --model_id 'gl_es' --type 'dual' --lang_src 'gl' --lang_tgt 'es' --seed 50 --epochs 1 --batch_size 128
echo '--- TASK end: model=gl_es seed=50'

echo '--- TASK start: model=gl_es seed=51'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/gl_es.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/gl.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_bilingual.csv' --model_id 'gl_es' --type 'dual' --lang_src 'gl' --lang_tgt 'es' --seed 51 --epochs 1 --batch_size 128
echo '--- TASK end: model=gl_es seed=51'

echo '--- TASK start: model=gl_eu seed=42'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/gl_eu.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/gl.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_bilingual.csv' --model_id 'gl_eu' --type 'dual' --lang_src 'gl' --lang_tgt 'eu' --seed 42 --epochs 1 --batch_size 128
echo '--- TASK end: model=gl_eu seed=42'

echo '--- TASK start: model=gl_eu seed=43'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/gl_eu.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/gl.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_bilingual.csv' --model_id 'gl_eu' --type 'dual' --lang_src 'gl' --lang_tgt 'eu' --seed 43 --epochs 1 --batch_size 128
echo '--- TASK end: model=gl_eu seed=43'

echo '--- TASK start: model=gl_eu seed=44'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/gl_eu.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/gl.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_bilingual.csv' --model_id 'gl_eu' --type 'dual' --lang_src 'gl' --lang_tgt 'eu' --seed 44 --epochs 1 --batch_size 128
echo '--- TASK end: model=gl_eu seed=44'

echo '--- TASK start: model=gl_eu seed=45'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/gl_eu.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/gl.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_bilingual.csv' --model_id 'gl_eu' --type 'dual' --lang_src 'gl' --lang_tgt 'eu' --seed 45 --epochs 1 --batch_size 128
echo '--- TASK end: model=gl_eu seed=45'

echo '--- TASK start: model=gl_eu seed=46'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/gl_eu.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/gl.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_bilingual.csv' --model_id 'gl_eu' --type 'dual' --lang_src 'gl' --lang_tgt 'eu' --seed 46 --epochs 1 --batch_size 128
echo '--- TASK end: model=gl_eu seed=46'

echo '--- TASK start: model=gl_eu seed=47'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/gl_eu.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/gl.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_bilingual.csv' --model_id 'gl_eu' --type 'dual' --lang_src 'gl' --lang_tgt 'eu' --seed 47 --epochs 1 --batch_size 128
echo '--- TASK end: model=gl_eu seed=47'

echo '--- TASK start: model=gl_eu seed=48'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/gl_eu.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/gl.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_bilingual.csv' --model_id 'gl_eu' --type 'dual' --lang_src 'gl' --lang_tgt 'eu' --seed 48 --epochs 1 --batch_size 128
echo '--- TASK end: model=gl_eu seed=48'

echo '--- TASK start: model=gl_eu seed=49'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/gl_eu.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/gl.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_bilingual.csv' --model_id 'gl_eu' --type 'dual' --lang_src 'gl' --lang_tgt 'eu' --seed 49 --epochs 1 --batch_size 128
echo '--- TASK end: model=gl_eu seed=49'

echo '--- TASK start: model=gl_eu seed=50'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/gl_eu.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/gl.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_bilingual.csv' --model_id 'gl_eu' --type 'dual' --lang_src 'gl' --lang_tgt 'eu' --seed 50 --epochs 1 --batch_size 128
echo '--- TASK end: model=gl_eu seed=50'

echo '--- TASK start: model=gl_eu seed=51'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/gl_eu.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/gl.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_bilingual.csv' --model_id 'gl_eu' --type 'dual' --lang_src 'gl' --lang_tgt 'eu' --seed 51 --epochs 1 --batch_size 128
echo '--- TASK end: model=gl_eu seed=51'

echo '--- TASK start: model=gl_fa seed=42'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/gl_fa.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/gl.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_bilingual.csv' --model_id 'gl_fa' --type 'dual' --lang_src 'gl' --lang_tgt 'fa' --seed 42 --epochs 1 --batch_size 128
echo '--- TASK end: model=gl_fa seed=42'

echo '--- TASK start: model=gl_fa seed=43'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/gl_fa.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/gl.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_bilingual.csv' --model_id 'gl_fa' --type 'dual' --lang_src 'gl' --lang_tgt 'fa' --seed 43 --epochs 1 --batch_size 128
echo '--- TASK end: model=gl_fa seed=43'

echo '--- TASK start: model=gl_fa seed=44'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/gl_fa.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/gl.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_bilingual.csv' --model_id 'gl_fa' --type 'dual' --lang_src 'gl' --lang_tgt 'fa' --seed 44 --epochs 1 --batch_size 128
echo '--- TASK end: model=gl_fa seed=44'

echo '--- TASK start: model=gl_fa seed=45'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/gl_fa.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/gl.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_bilingual.csv' --model_id 'gl_fa' --type 'dual' --lang_src 'gl' --lang_tgt 'fa' --seed 45 --epochs 1 --batch_size 128
echo '--- TASK end: model=gl_fa seed=45'

echo '--- TASK start: model=gl_fa seed=46'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/gl_fa.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/gl.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_bilingual.csv' --model_id 'gl_fa' --type 'dual' --lang_src 'gl' --lang_tgt 'fa' --seed 46 --epochs 1 --batch_size 128
echo '--- TASK end: model=gl_fa seed=46'

echo '--- TASK start: model=gl_fa seed=47'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/gl_fa.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/gl.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_bilingual.csv' --model_id 'gl_fa' --type 'dual' --lang_src 'gl' --lang_tgt 'fa' --seed 47 --epochs 1 --batch_size 128
echo '--- TASK end: model=gl_fa seed=47'

echo '--- TASK start: model=gl_fa seed=48'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/gl_fa.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/gl.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_bilingual.csv' --model_id 'gl_fa' --type 'dual' --lang_src 'gl' --lang_tgt 'fa' --seed 48 --epochs 1 --batch_size 128
echo '--- TASK end: model=gl_fa seed=48'

echo '--- TASK start: model=gl_fa seed=49'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/gl_fa.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/gl.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_bilingual.csv' --model_id 'gl_fa' --type 'dual' --lang_src 'gl' --lang_tgt 'fa' --seed 49 --epochs 1 --batch_size 128
echo '--- TASK end: model=gl_fa seed=49'

echo '--- TASK start: model=gl_fa seed=50'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/gl_fa.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/gl.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_bilingual.csv' --model_id 'gl_fa' --type 'dual' --lang_src 'gl' --lang_tgt 'fa' --seed 50 --epochs 1 --batch_size 128
echo '--- TASK end: model=gl_fa seed=50'

echo '--- TASK start: model=gl_fa seed=51'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/gl_fa.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/gl.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_bilingual.csv' --model_id 'gl_fa' --type 'dual' --lang_src 'gl' --lang_tgt 'fa' --seed 51 --epochs 1 --batch_size 128
echo '--- TASK end: model=gl_fa seed=51'

echo '--- TASK start: model=gl_fr seed=42'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/gl_fr.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/gl.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_bilingual.csv' --model_id 'gl_fr' --type 'dual' --lang_src 'gl' --lang_tgt 'fr' --seed 42 --epochs 1 --batch_size 128
echo '--- TASK end: model=gl_fr seed=42'

echo '--- TASK start: model=gl_fr seed=43'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/gl_fr.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/gl.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_bilingual.csv' --model_id 'gl_fr' --type 'dual' --lang_src 'gl' --lang_tgt 'fr' --seed 43 --epochs 1 --batch_size 128
echo '--- TASK end: model=gl_fr seed=43'

echo '--- TASK start: model=gl_fr seed=44'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/gl_fr.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/gl.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_bilingual.csv' --model_id 'gl_fr' --type 'dual' --lang_src 'gl' --lang_tgt 'fr' --seed 44 --epochs 1 --batch_size 128
echo '--- TASK end: model=gl_fr seed=44'

echo '--- TASK start: model=gl_fr seed=45'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/gl_fr.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/gl.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_bilingual.csv' --model_id 'gl_fr' --type 'dual' --lang_src 'gl' --lang_tgt 'fr' --seed 45 --epochs 1 --batch_size 128
echo '--- TASK end: model=gl_fr seed=45'

echo '--- TASK start: model=gl_fr seed=46'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/gl_fr.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/gl.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_bilingual.csv' --model_id 'gl_fr' --type 'dual' --lang_src 'gl' --lang_tgt 'fr' --seed 46 --epochs 1 --batch_size 128
echo '--- TASK end: model=gl_fr seed=46'

echo '--- TASK start: model=gl_fr seed=47'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/gl_fr.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/gl.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_bilingual.csv' --model_id 'gl_fr' --type 'dual' --lang_src 'gl' --lang_tgt 'fr' --seed 47 --epochs 1 --batch_size 128
echo '--- TASK end: model=gl_fr seed=47'

echo '--- TASK start: model=gl_fr seed=48'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/gl_fr.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/gl.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_bilingual.csv' --model_id 'gl_fr' --type 'dual' --lang_src 'gl' --lang_tgt 'fr' --seed 48 --epochs 1 --batch_size 128
echo '--- TASK end: model=gl_fr seed=48'

echo '--- TASK start: model=gl_fr seed=49'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/gl_fr.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/gl.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_bilingual.csv' --model_id 'gl_fr' --type 'dual' --lang_src 'gl' --lang_tgt 'fr' --seed 49 --epochs 1 --batch_size 128
echo '--- TASK end: model=gl_fr seed=49'

echo '--- TASK start: model=gl_fr seed=50'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/gl_fr.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/gl.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_bilingual.csv' --model_id 'gl_fr' --type 'dual' --lang_src 'gl' --lang_tgt 'fr' --seed 50 --epochs 1 --batch_size 128
echo '--- TASK end: model=gl_fr seed=50'

echo '--- TASK start: model=gl_fr seed=51'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/gl_fr.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/gl.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_bilingual.csv' --model_id 'gl_fr' --type 'dual' --lang_src 'gl' --lang_tgt 'fr' --seed 51 --epochs 1 --batch_size 128
echo '--- TASK end: model=gl_fr seed=51'

echo '--- TASK start: model=gl_hu seed=42'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/gl_hu.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/gl.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_bilingual.csv' --model_id 'gl_hu' --type 'dual' --lang_src 'gl' --lang_tgt 'hu' --seed 42 --epochs 1 --batch_size 128
echo '--- TASK end: model=gl_hu seed=42'

echo '--- TASK start: model=gl_hu seed=43'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/gl_hu.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/gl.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_bilingual.csv' --model_id 'gl_hu' --type 'dual' --lang_src 'gl' --lang_tgt 'hu' --seed 43 --epochs 1 --batch_size 128
echo '--- TASK end: model=gl_hu seed=43'

echo '--- TASK start: model=gl_hu seed=44'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/gl_hu.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/gl.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_bilingual.csv' --model_id 'gl_hu' --type 'dual' --lang_src 'gl' --lang_tgt 'hu' --seed 44 --epochs 1 --batch_size 128
echo '--- TASK end: model=gl_hu seed=44'

echo '--- TASK start: model=gl_hu seed=45'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/gl_hu.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/gl.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_bilingual.csv' --model_id 'gl_hu' --type 'dual' --lang_src 'gl' --lang_tgt 'hu' --seed 45 --epochs 1 --batch_size 128
echo '--- TASK end: model=gl_hu seed=45'

