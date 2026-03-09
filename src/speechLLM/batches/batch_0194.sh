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

echo '--- TASK start: model=ca seed=46'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/ca.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/ca.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_dual.csv' --model_id 'ca' --type 'single' --lang_src 'ca' --lang_tgt '' --seed 46 --epochs 1 --batch_size 128
echo '--- TASK end: model=ca seed=46'

echo '--- TASK start: model=ca seed=47'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/ca.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/ca.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_dual.csv' --model_id 'ca' --type 'single' --lang_src 'ca' --lang_tgt '' --seed 47 --epochs 1 --batch_size 128
echo '--- TASK end: model=ca seed=47'

echo '--- TASK start: model=ca seed=48'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/ca.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/ca.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_dual.csv' --model_id 'ca' --type 'single' --lang_src 'ca' --lang_tgt '' --seed 48 --epochs 1 --batch_size 128
echo '--- TASK end: model=ca seed=48'

echo '--- TASK start: model=ca seed=49'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/ca.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/ca.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_dual.csv' --model_id 'ca' --type 'single' --lang_src 'ca' --lang_tgt '' --seed 49 --epochs 1 --batch_size 128
echo '--- TASK end: model=ca seed=49'

echo '--- TASK start: model=ca seed=50'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/ca.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/ca.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_dual.csv' --model_id 'ca' --type 'single' --lang_src 'ca' --lang_tgt '' --seed 50 --epochs 1 --batch_size 128
echo '--- TASK end: model=ca seed=50'

echo '--- TASK start: model=ca seed=51'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/ca.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/ca.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_dual.csv' --model_id 'ca' --type 'single' --lang_src 'ca' --lang_tgt '' --seed 51 --epochs 1 --batch_size 128
echo '--- TASK end: model=ca seed=51'

echo '--- TASK start: model=ckb seed=42'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/ckb.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/ckb.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_dual.csv' --model_id 'ckb' --type 'single' --lang_src 'ckb' --lang_tgt '' --seed 42 --epochs 1 --batch_size 128
echo '--- TASK end: model=ckb seed=42'

echo '--- TASK start: model=ckb seed=43'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/ckb.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/ckb.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_dual.csv' --model_id 'ckb' --type 'single' --lang_src 'ckb' --lang_tgt '' --seed 43 --epochs 1 --batch_size 128
echo '--- TASK end: model=ckb seed=43'

echo '--- TASK start: model=ckb seed=44'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/ckb.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/ckb.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_dual.csv' --model_id 'ckb' --type 'single' --lang_src 'ckb' --lang_tgt '' --seed 44 --epochs 1 --batch_size 128
echo '--- TASK end: model=ckb seed=44'

echo '--- TASK start: model=ckb seed=45'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/ckb.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/ckb.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_dual.csv' --model_id 'ckb' --type 'single' --lang_src 'ckb' --lang_tgt '' --seed 45 --epochs 1 --batch_size 128
echo '--- TASK end: model=ckb seed=45'

echo '--- TASK start: model=ckb seed=46'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/ckb.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/ckb.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_dual.csv' --model_id 'ckb' --type 'single' --lang_src 'ckb' --lang_tgt '' --seed 46 --epochs 1 --batch_size 128
echo '--- TASK end: model=ckb seed=46'

echo '--- TASK start: model=ckb seed=47'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/ckb.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/ckb.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_dual.csv' --model_id 'ckb' --type 'single' --lang_src 'ckb' --lang_tgt '' --seed 47 --epochs 1 --batch_size 128
echo '--- TASK end: model=ckb seed=47'

echo '--- TASK start: model=ckb seed=48'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/ckb.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/ckb.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_dual.csv' --model_id 'ckb' --type 'single' --lang_src 'ckb' --lang_tgt '' --seed 48 --epochs 1 --batch_size 128
echo '--- TASK end: model=ckb seed=48'

echo '--- TASK start: model=ckb seed=49'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/ckb.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/ckb.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_dual.csv' --model_id 'ckb' --type 'single' --lang_src 'ckb' --lang_tgt '' --seed 49 --epochs 1 --batch_size 128
echo '--- TASK end: model=ckb seed=49'

echo '--- TASK start: model=ckb seed=50'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/ckb.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/ckb.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_dual.csv' --model_id 'ckb' --type 'single' --lang_src 'ckb' --lang_tgt '' --seed 50 --epochs 1 --batch_size 128
echo '--- TASK end: model=ckb seed=50'

echo '--- TASK start: model=ckb seed=51'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/ckb.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/ckb.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_dual.csv' --model_id 'ckb' --type 'single' --lang_src 'ckb' --lang_tgt '' --seed 51 --epochs 1 --batch_size 128
echo '--- TASK end: model=ckb seed=51'

echo '--- TASK start: model=cs seed=42'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/cs.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/cs.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_dual.csv' --model_id 'cs' --type 'single' --lang_src 'cs' --lang_tgt '' --seed 42 --epochs 1 --batch_size 128
echo '--- TASK end: model=cs seed=42'

echo '--- TASK start: model=cs seed=43'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/cs.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/cs.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_dual.csv' --model_id 'cs' --type 'single' --lang_src 'cs' --lang_tgt '' --seed 43 --epochs 1 --batch_size 128
echo '--- TASK end: model=cs seed=43'

echo '--- TASK start: model=cs seed=44'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/cs.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/cs.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_dual.csv' --model_id 'cs' --type 'single' --lang_src 'cs' --lang_tgt '' --seed 44 --epochs 1 --batch_size 128
echo '--- TASK end: model=cs seed=44'

echo '--- TASK start: model=cs seed=45'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/cs.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/cs.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_dual.csv' --model_id 'cs' --type 'single' --lang_src 'cs' --lang_tgt '' --seed 45 --epochs 1 --batch_size 128
echo '--- TASK end: model=cs seed=45'

echo '--- TASK start: model=cs seed=46'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/cs.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/cs.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_dual.csv' --model_id 'cs' --type 'single' --lang_src 'cs' --lang_tgt '' --seed 46 --epochs 1 --batch_size 128
echo '--- TASK end: model=cs seed=46'

echo '--- TASK start: model=cs seed=47'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/cs.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/cs.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_dual.csv' --model_id 'cs' --type 'single' --lang_src 'cs' --lang_tgt '' --seed 47 --epochs 1 --batch_size 128
echo '--- TASK end: model=cs seed=47'

echo '--- TASK start: model=cs seed=48'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/cs.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/cs.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_dual.csv' --model_id 'cs' --type 'single' --lang_src 'cs' --lang_tgt '' --seed 48 --epochs 1 --batch_size 128
echo '--- TASK end: model=cs seed=48'

echo '--- TASK start: model=cs seed=49'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/cs.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/cs.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_dual.csv' --model_id 'cs' --type 'single' --lang_src 'cs' --lang_tgt '' --seed 49 --epochs 1 --batch_size 128
echo '--- TASK end: model=cs seed=49'

echo '--- TASK start: model=cs seed=50'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/cs.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/cs.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_dual.csv' --model_id 'cs' --type 'single' --lang_src 'cs' --lang_tgt '' --seed 50 --epochs 1 --batch_size 128
echo '--- TASK end: model=cs seed=50'

echo '--- TASK start: model=cs seed=51'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/cs.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/cs.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_dual.csv' --model_id 'cs' --type 'single' --lang_src 'cs' --lang_tgt '' --seed 51 --epochs 1 --batch_size 128
echo '--- TASK end: model=cs seed=51'

echo '--- TASK start: model=cy seed=42'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/cy.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/cy.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_dual.csv' --model_id 'cy' --type 'single' --lang_src 'cy' --lang_tgt '' --seed 42 --epochs 1 --batch_size 128
echo '--- TASK end: model=cy seed=42'

echo '--- TASK start: model=cy seed=43'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/cy.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/cy.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_dual.csv' --model_id 'cy' --type 'single' --lang_src 'cy' --lang_tgt '' --seed 43 --epochs 1 --batch_size 128
echo '--- TASK end: model=cy seed=43'

echo '--- TASK start: model=cy seed=44'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/cy.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/cy.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_dual.csv' --model_id 'cy' --type 'single' --lang_src 'cy' --lang_tgt '' --seed 44 --epochs 1 --batch_size 128
echo '--- TASK end: model=cy seed=44'

echo '--- TASK start: model=cy seed=45'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/cy.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/cy.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_dual.csv' --model_id 'cy' --type 'single' --lang_src 'cy' --lang_tgt '' --seed 45 --epochs 1 --batch_size 128
echo '--- TASK end: model=cy seed=45'

echo '--- TASK start: model=cy seed=46'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/cy.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/cy.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_dual.csv' --model_id 'cy' --type 'single' --lang_src 'cy' --lang_tgt '' --seed 46 --epochs 1 --batch_size 128
echo '--- TASK end: model=cy seed=46'

echo '--- TASK start: model=cy seed=47'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/cy.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/cy.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_dual.csv' --model_id 'cy' --type 'single' --lang_src 'cy' --lang_tgt '' --seed 47 --epochs 1 --batch_size 128
echo '--- TASK end: model=cy seed=47'

echo '--- TASK start: model=cy seed=48'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/cy.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/cy.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_dual.csv' --model_id 'cy' --type 'single' --lang_src 'cy' --lang_tgt '' --seed 48 --epochs 1 --batch_size 128
echo '--- TASK end: model=cy seed=48'

echo '--- TASK start: model=cy seed=49'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/cy.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/cy.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_dual.csv' --model_id 'cy' --type 'single' --lang_src 'cy' --lang_tgt '' --seed 49 --epochs 1 --batch_size 128
echo '--- TASK end: model=cy seed=49'

echo '--- TASK start: model=cy seed=50'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/cy.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/cy.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_dual.csv' --model_id 'cy' --type 'single' --lang_src 'cy' --lang_tgt '' --seed 50 --epochs 1 --batch_size 128
echo '--- TASK end: model=cy seed=50'

echo '--- TASK start: model=cy seed=51'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/cy.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/cy.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_dual.csv' --model_id 'cy' --type 'single' --lang_src 'cy' --lang_tgt '' --seed 51 --epochs 1 --batch_size 128
echo '--- TASK end: model=cy seed=51'

echo '--- TASK start: model=de seed=42'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/de.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/de.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_dual.csv' --model_id 'de' --type 'single' --lang_src 'de' --lang_tgt '' --seed 42 --epochs 1 --batch_size 128
echo '--- TASK end: model=de seed=42'

echo '--- TASK start: model=de seed=43'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/de.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/de.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_dual.csv' --model_id 'de' --type 'single' --lang_src 'de' --lang_tgt '' --seed 43 --epochs 1 --batch_size 128
echo '--- TASK end: model=de seed=43'

echo '--- TASK start: model=de seed=44'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/de.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/de.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_dual.csv' --model_id 'de' --type 'single' --lang_src 'de' --lang_tgt '' --seed 44 --epochs 1 --batch_size 128
echo '--- TASK end: model=de seed=44'

echo '--- TASK start: model=de seed=45'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/de.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/de.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_dual.csv' --model_id 'de' --type 'single' --lang_src 'de' --lang_tgt '' --seed 45 --epochs 1 --batch_size 128
echo '--- TASK end: model=de seed=45'

echo '--- TASK start: model=de seed=46'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/de.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/de.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_dual.csv' --model_id 'de' --type 'single' --lang_src 'de' --lang_tgt '' --seed 46 --epochs 1 --batch_size 128
echo '--- TASK end: model=de seed=46'

echo '--- TASK start: model=de seed=47'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/de.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/de.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_dual.csv' --model_id 'de' --type 'single' --lang_src 'de' --lang_tgt '' --seed 47 --epochs 1 --batch_size 128
echo '--- TASK end: model=de seed=47'

echo '--- TASK start: model=de seed=48'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/de.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/de.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_dual.csv' --model_id 'de' --type 'single' --lang_src 'de' --lang_tgt '' --seed 48 --epochs 1 --batch_size 128
echo '--- TASK end: model=de seed=48'

echo '--- TASK start: model=de seed=49'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/de.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/de.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_dual.csv' --model_id 'de' --type 'single' --lang_src 'de' --lang_tgt '' --seed 49 --epochs 1 --batch_size 128
echo '--- TASK end: model=de seed=49'

echo '--- TASK start: model=de seed=50'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/de.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/de.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_dual.csv' --model_id 'de' --type 'single' --lang_src 'de' --lang_tgt '' --seed 50 --epochs 1 --batch_size 128
echo '--- TASK end: model=de seed=50'

echo '--- TASK start: model=de seed=51'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/de.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/de.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_dual.csv' --model_id 'de' --type 'single' --lang_src 'de' --lang_tgt '' --seed 51 --epochs 1 --batch_size 128
echo '--- TASK end: model=de seed=51'

echo '--- TASK start: model=en seed=42'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/en.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/en.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_dual.csv' --model_id 'en' --type 'single' --lang_src 'en' --lang_tgt '' --seed 42 --epochs 1 --batch_size 128
echo '--- TASK end: model=en seed=42'

echo '--- TASK start: model=en seed=43'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/en.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/en.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_dual.csv' --model_id 'en' --type 'single' --lang_src 'en' --lang_tgt '' --seed 43 --epochs 1 --batch_size 128
echo '--- TASK end: model=en seed=43'

echo '--- TASK start: model=en seed=44'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/en.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/en.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_dual.csv' --model_id 'en' --type 'single' --lang_src 'en' --lang_tgt '' --seed 44 --epochs 1 --batch_size 128
echo '--- TASK end: model=en seed=44'

echo '--- TASK start: model=en seed=45'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/en.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/en.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_dual.csv' --model_id 'en' --type 'single' --lang_src 'en' --lang_tgt '' --seed 45 --epochs 1 --batch_size 128
echo '--- TASK end: model=en seed=45'

echo '--- TASK start: model=en seed=46'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/en.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/en.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_dual.csv' --model_id 'en' --type 'single' --lang_src 'en' --lang_tgt '' --seed 46 --epochs 1 --batch_size 128
echo '--- TASK end: model=en seed=46'

echo '--- TASK start: model=en seed=47'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/en.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/en.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_dual.csv' --model_id 'en' --type 'single' --lang_src 'en' --lang_tgt '' --seed 47 --epochs 1 --batch_size 128
echo '--- TASK end: model=en seed=47'

echo '--- TASK start: model=en seed=48'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/en.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/en.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_dual.csv' --model_id 'en' --type 'single' --lang_src 'en' --lang_tgt '' --seed 48 --epochs 1 --batch_size 128
echo '--- TASK end: model=en seed=48'

echo '--- TASK start: model=en seed=49'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/en.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/en.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_dual.csv' --model_id 'en' --type 'single' --lang_src 'en' --lang_tgt '' --seed 49 --epochs 1 --batch_size 128
echo '--- TASK end: model=en seed=49'

echo '--- TASK start: model=en seed=50'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/en.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/en.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_dual.csv' --model_id 'en' --type 'single' --lang_src 'en' --lang_tgt '' --seed 50 --epochs 1 --batch_size 128
echo '--- TASK end: model=en seed=50'

echo '--- TASK start: model=en seed=51'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/en.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/en.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_dual.csv' --model_id 'en' --type 'single' --lang_src 'en' --lang_tgt '' --seed 51 --epochs 1 --batch_size 128
echo '--- TASK end: model=en seed=51'

echo '--- TASK start: model=eo seed=42'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/eo.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/eo.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_dual.csv' --model_id 'eo' --type 'single' --lang_src 'eo' --lang_tgt '' --seed 42 --epochs 1 --batch_size 128
echo '--- TASK end: model=eo seed=42'

echo '--- TASK start: model=eo seed=43'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/eo.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/eo.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_dual.csv' --model_id 'eo' --type 'single' --lang_src 'eo' --lang_tgt '' --seed 43 --epochs 1 --batch_size 128
echo '--- TASK end: model=eo seed=43'

echo '--- TASK start: model=eo seed=44'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/eo.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/eo.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_dual.csv' --model_id 'eo' --type 'single' --lang_src 'eo' --lang_tgt '' --seed 44 --epochs 1 --batch_size 128
echo '--- TASK end: model=eo seed=44'

echo '--- TASK start: model=eo seed=45'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/eo.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/eo.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_dual.csv' --model_id 'eo' --type 'single' --lang_src 'eo' --lang_tgt '' --seed 45 --epochs 1 --batch_size 128
echo '--- TASK end: model=eo seed=45'

echo '--- TASK start: model=eo seed=46'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/eo.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/eo.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_dual.csv' --model_id 'eo' --type 'single' --lang_src 'eo' --lang_tgt '' --seed 46 --epochs 1 --batch_size 128
echo '--- TASK end: model=eo seed=46'

echo '--- TASK start: model=eo seed=47'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/eo.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/eo.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_dual.csv' --model_id 'eo' --type 'single' --lang_src 'eo' --lang_tgt '' --seed 47 --epochs 1 --batch_size 128
echo '--- TASK end: model=eo seed=47'

echo '--- TASK start: model=eo seed=48'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/eo.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/eo.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_dual.csv' --model_id 'eo' --type 'single' --lang_src 'eo' --lang_tgt '' --seed 48 --epochs 1 --batch_size 128
echo '--- TASK end: model=eo seed=48'

echo '--- TASK start: model=eo seed=49'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/eo.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/eo.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_dual.csv' --model_id 'eo' --type 'single' --lang_src 'eo' --lang_tgt '' --seed 49 --epochs 1 --batch_size 128
echo '--- TASK end: model=eo seed=49'

echo '--- TASK start: model=eo seed=50'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/eo.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/eo.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_dual.csv' --model_id 'eo' --type 'single' --lang_src 'eo' --lang_tgt '' --seed 50 --epochs 1 --batch_size 128
echo '--- TASK end: model=eo seed=50'

echo '--- TASK start: model=eo seed=51'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/eo.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/eo.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_dual.csv' --model_id 'eo' --type 'single' --lang_src 'eo' --lang_tgt '' --seed 51 --epochs 1 --batch_size 128
echo '--- TASK end: model=eo seed=51'

echo '--- TASK start: model=es seed=42'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/es.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/es.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_dual.csv' --model_id 'es' --type 'single' --lang_src 'es' --lang_tgt '' --seed 42 --epochs 1 --batch_size 128
echo '--- TASK end: model=es seed=42'

echo '--- TASK start: model=es seed=43'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/es.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/es.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_dual.csv' --model_id 'es' --type 'single' --lang_src 'es' --lang_tgt '' --seed 43 --epochs 1 --batch_size 128
echo '--- TASK end: model=es seed=43'

echo '--- TASK start: model=es seed=44'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/es.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/es.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_dual.csv' --model_id 'es' --type 'single' --lang_src 'es' --lang_tgt '' --seed 44 --epochs 1 --batch_size 128
echo '--- TASK end: model=es seed=44'

echo '--- TASK start: model=es seed=45'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/es.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/es.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_dual.csv' --model_id 'es' --type 'single' --lang_src 'es' --lang_tgt '' --seed 45 --epochs 1 --batch_size 128
echo '--- TASK end: model=es seed=45'

echo '--- TASK start: model=es seed=46'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/es.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/es.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_dual.csv' --model_id 'es' --type 'single' --lang_src 'es' --lang_tgt '' --seed 46 --epochs 1 --batch_size 128
echo '--- TASK end: model=es seed=46'

echo '--- TASK start: model=es seed=47'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/es.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/es.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_dual.csv' --model_id 'es' --type 'single' --lang_src 'es' --lang_tgt '' --seed 47 --epochs 1 --batch_size 128
echo '--- TASK end: model=es seed=47'

echo '--- TASK start: model=es seed=48'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/es.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/es.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_dual.csv' --model_id 'es' --type 'single' --lang_src 'es' --lang_tgt '' --seed 48 --epochs 1 --batch_size 128
echo '--- TASK end: model=es seed=48'

echo '--- TASK start: model=es seed=49'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/es.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/es.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_dual.csv' --model_id 'es' --type 'single' --lang_src 'es' --lang_tgt '' --seed 49 --epochs 1 --batch_size 128
echo '--- TASK end: model=es seed=49'

echo '--- TASK start: model=es seed=50'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/es.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/es.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_dual.csv' --model_id 'es' --type 'single' --lang_src 'es' --lang_tgt '' --seed 50 --epochs 1 --batch_size 128
echo '--- TASK end: model=es seed=50'

echo '--- TASK start: model=es seed=51'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/es.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/es.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_dual.csv' --model_id 'es' --type 'single' --lang_src 'es' --lang_tgt '' --seed 51 --epochs 1 --batch_size 128
echo '--- TASK end: model=es seed=51'

echo '--- TASK start: model=eu seed=42'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/eu.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/eu.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_dual.csv' --model_id 'eu' --type 'single' --lang_src 'eu' --lang_tgt '' --seed 42 --epochs 1 --batch_size 128
echo '--- TASK end: model=eu seed=42'

echo '--- TASK start: model=eu seed=43'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/eu.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/eu.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_dual.csv' --model_id 'eu' --type 'single' --lang_src 'eu' --lang_tgt '' --seed 43 --epochs 1 --batch_size 128
echo '--- TASK end: model=eu seed=43'

echo '--- TASK start: model=eu seed=44'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/eu.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/eu.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_dual.csv' --model_id 'eu' --type 'single' --lang_src 'eu' --lang_tgt '' --seed 44 --epochs 1 --batch_size 128
echo '--- TASK end: model=eu seed=44'

echo '--- TASK start: model=eu seed=45'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/eu.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/eu.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_dual.csv' --model_id 'eu' --type 'single' --lang_src 'eu' --lang_tgt '' --seed 45 --epochs 1 --batch_size 128
echo '--- TASK end: model=eu seed=45'

echo '--- TASK start: model=eu seed=46'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/eu.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/eu.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_dual.csv' --model_id 'eu' --type 'single' --lang_src 'eu' --lang_tgt '' --seed 46 --epochs 1 --batch_size 128
echo '--- TASK end: model=eu seed=46'

echo '--- TASK start: model=eu seed=47'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/eu.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/eu.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_dual.csv' --model_id 'eu' --type 'single' --lang_src 'eu' --lang_tgt '' --seed 47 --epochs 1 --batch_size 128
echo '--- TASK end: model=eu seed=47'

echo '--- TASK start: model=eu seed=48'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/eu.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/eu.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_dual.csv' --model_id 'eu' --type 'single' --lang_src 'eu' --lang_tgt '' --seed 48 --epochs 1 --batch_size 128
echo '--- TASK end: model=eu seed=48'

echo '--- TASK start: model=eu seed=49'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/eu.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/eu.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_dual.csv' --model_id 'eu' --type 'single' --lang_src 'eu' --lang_tgt '' --seed 49 --epochs 1 --batch_size 128
echo '--- TASK end: model=eu seed=49'

echo '--- TASK start: model=eu seed=50'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/eu.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/eu.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_dual.csv' --model_id 'eu' --type 'single' --lang_src 'eu' --lang_tgt '' --seed 50 --epochs 1 --batch_size 128
echo '--- TASK end: model=eu seed=50'

echo '--- TASK start: model=eu seed=51'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/eu.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/eu.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_dual.csv' --model_id 'eu' --type 'single' --lang_src 'eu' --lang_tgt '' --seed 51 --epochs 1 --batch_size 128
echo '--- TASK end: model=eu seed=51'

echo '--- TASK start: model=fa seed=42'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/fa.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/fa.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_dual.csv' --model_id 'fa' --type 'single' --lang_src 'fa' --lang_tgt '' --seed 42 --epochs 1 --batch_size 128
echo '--- TASK end: model=fa seed=42'

echo '--- TASK start: model=fa seed=43'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/fa.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/fa.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_dual.csv' --model_id 'fa' --type 'single' --lang_src 'fa' --lang_tgt '' --seed 43 --epochs 1 --batch_size 128
echo '--- TASK end: model=fa seed=43'

echo '--- TASK start: model=fa seed=44'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/fa.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/fa.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_dual.csv' --model_id 'fa' --type 'single' --lang_src 'fa' --lang_tgt '' --seed 44 --epochs 1 --batch_size 128
echo '--- TASK end: model=fa seed=44'

echo '--- TASK start: model=fa seed=45'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/fa.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/fa.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_dual.csv' --model_id 'fa' --type 'single' --lang_src 'fa' --lang_tgt '' --seed 45 --epochs 1 --batch_size 128
echo '--- TASK end: model=fa seed=45'

echo '--- TASK start: model=fa seed=46'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/fa.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/fa.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_dual.csv' --model_id 'fa' --type 'single' --lang_src 'fa' --lang_tgt '' --seed 46 --epochs 1 --batch_size 128
echo '--- TASK end: model=fa seed=46'

echo '--- TASK start: model=fa seed=47'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/fa.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/fa.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_dual.csv' --model_id 'fa' --type 'single' --lang_src 'fa' --lang_tgt '' --seed 47 --epochs 1 --batch_size 128
echo '--- TASK end: model=fa seed=47'

echo '--- TASK start: model=fa seed=48'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/fa.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/fa.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_dual.csv' --model_id 'fa' --type 'single' --lang_src 'fa' --lang_tgt '' --seed 48 --epochs 1 --batch_size 128
echo '--- TASK end: model=fa seed=48'

echo '--- TASK start: model=fa seed=49'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/fa.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/fa.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_dual.csv' --model_id 'fa' --type 'single' --lang_src 'fa' --lang_tgt '' --seed 49 --epochs 1 --batch_size 128
echo '--- TASK end: model=fa seed=49'

echo '--- TASK start: model=fa seed=50'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/fa.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/fa.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_dual.csv' --model_id 'fa' --type 'single' --lang_src 'fa' --lang_tgt '' --seed 50 --epochs 1 --batch_size 128
echo '--- TASK end: model=fa seed=50'

echo '--- TASK start: model=fa seed=51'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/fa.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/fa.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_dual.csv' --model_id 'fa' --type 'single' --lang_src 'fa' --lang_tgt '' --seed 51 --epochs 1 --batch_size 128
echo '--- TASK end: model=fa seed=51'

echo '--- TASK start: model=fr seed=42'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/fr.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/fr.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_dual.csv' --model_id 'fr' --type 'single' --lang_src 'fr' --lang_tgt '' --seed 42 --epochs 1 --batch_size 128
echo '--- TASK end: model=fr seed=42'

echo '--- TASK start: model=fr seed=43'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/fr.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/fr.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_dual.csv' --model_id 'fr' --type 'single' --lang_src 'fr' --lang_tgt '' --seed 43 --epochs 1 --batch_size 128
echo '--- TASK end: model=fr seed=43'

echo '--- TASK start: model=fr seed=44'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/fr.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/fr.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_dual.csv' --model_id 'fr' --type 'single' --lang_src 'fr' --lang_tgt '' --seed 44 --epochs 1 --batch_size 128
echo '--- TASK end: model=fr seed=44'

echo '--- TASK start: model=fr seed=45'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/fr.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/fr.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_dual.csv' --model_id 'fr' --type 'single' --lang_src 'fr' --lang_tgt '' --seed 45 --epochs 1 --batch_size 128
echo '--- TASK end: model=fr seed=45'

