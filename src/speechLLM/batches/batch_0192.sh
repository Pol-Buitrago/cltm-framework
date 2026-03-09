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

echo '--- TASK start: model=ro seed=46'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/ro.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/ro.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'ro' --type 'single' --lang_src 'ro' --lang_tgt '' --seed 46 --epochs 1 --batch_size 128
echo '--- TASK end: model=ro seed=46'

echo '--- TASK start: model=ro seed=47'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/ro.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/ro.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'ro' --type 'single' --lang_src 'ro' --lang_tgt '' --seed 47 --epochs 1 --batch_size 128
echo '--- TASK end: model=ro seed=47'

echo '--- TASK start: model=ro seed=48'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/ro.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/ro.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'ro' --type 'single' --lang_src 'ro' --lang_tgt '' --seed 48 --epochs 1 --batch_size 128
echo '--- TASK end: model=ro seed=48'

echo '--- TASK start: model=ro seed=49'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/ro.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/ro.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'ro' --type 'single' --lang_src 'ro' --lang_tgt '' --seed 49 --epochs 1 --batch_size 128
echo '--- TASK end: model=ro seed=49'

echo '--- TASK start: model=ro seed=50'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/ro.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/ro.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'ro' --type 'single' --lang_src 'ro' --lang_tgt '' --seed 50 --epochs 1 --batch_size 128
echo '--- TASK end: model=ro seed=50'

echo '--- TASK start: model=ro seed=51'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/ro.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/ro.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'ro' --type 'single' --lang_src 'ro' --lang_tgt '' --seed 51 --epochs 1 --batch_size 128
echo '--- TASK end: model=ro seed=51'

echo '--- TASK start: model=ru seed=42'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/ru.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/ru.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'ru' --type 'single' --lang_src 'ru' --lang_tgt '' --seed 42 --epochs 1 --batch_size 128
echo '--- TASK end: model=ru seed=42'

echo '--- TASK start: model=ru seed=43'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/ru.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/ru.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'ru' --type 'single' --lang_src 'ru' --lang_tgt '' --seed 43 --epochs 1 --batch_size 128
echo '--- TASK end: model=ru seed=43'

echo '--- TASK start: model=ru seed=44'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/ru.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/ru.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'ru' --type 'single' --lang_src 'ru' --lang_tgt '' --seed 44 --epochs 1 --batch_size 128
echo '--- TASK end: model=ru seed=44'

echo '--- TASK start: model=ru seed=45'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/ru.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/ru.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'ru' --type 'single' --lang_src 'ru' --lang_tgt '' --seed 45 --epochs 1 --batch_size 128
echo '--- TASK end: model=ru seed=45'

echo '--- TASK start: model=ru seed=46'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/ru.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/ru.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'ru' --type 'single' --lang_src 'ru' --lang_tgt '' --seed 46 --epochs 1 --batch_size 128
echo '--- TASK end: model=ru seed=46'

echo '--- TASK start: model=ru seed=47'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/ru.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/ru.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'ru' --type 'single' --lang_src 'ru' --lang_tgt '' --seed 47 --epochs 1 --batch_size 128
echo '--- TASK end: model=ru seed=47'

echo '--- TASK start: model=ru seed=48'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/ru.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/ru.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'ru' --type 'single' --lang_src 'ru' --lang_tgt '' --seed 48 --epochs 1 --batch_size 128
echo '--- TASK end: model=ru seed=48'

echo '--- TASK start: model=ru seed=49'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/ru.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/ru.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'ru' --type 'single' --lang_src 'ru' --lang_tgt '' --seed 49 --epochs 1 --batch_size 128
echo '--- TASK end: model=ru seed=49'

echo '--- TASK start: model=ru seed=50'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/ru.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/ru.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'ru' --type 'single' --lang_src 'ru' --lang_tgt '' --seed 50 --epochs 1 --batch_size 128
echo '--- TASK end: model=ru seed=50'

echo '--- TASK start: model=ru seed=51'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/ru.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/ru.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'ru' --type 'single' --lang_src 'ru' --lang_tgt '' --seed 51 --epochs 1 --batch_size 128
echo '--- TASK end: model=ru seed=51'

echo '--- TASK start: model=rw seed=42'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/rw.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/rw.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'rw' --type 'single' --lang_src 'rw' --lang_tgt '' --seed 42 --epochs 1 --batch_size 128
echo '--- TASK end: model=rw seed=42'

echo '--- TASK start: model=rw seed=43'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/rw.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/rw.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'rw' --type 'single' --lang_src 'rw' --lang_tgt '' --seed 43 --epochs 1 --batch_size 128
echo '--- TASK end: model=rw seed=43'

echo '--- TASK start: model=rw seed=44'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/rw.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/rw.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'rw' --type 'single' --lang_src 'rw' --lang_tgt '' --seed 44 --epochs 1 --batch_size 128
echo '--- TASK end: model=rw seed=44'

echo '--- TASK start: model=rw seed=45'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/rw.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/rw.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'rw' --type 'single' --lang_src 'rw' --lang_tgt '' --seed 45 --epochs 1 --batch_size 128
echo '--- TASK end: model=rw seed=45'

echo '--- TASK start: model=rw seed=46'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/rw.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/rw.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'rw' --type 'single' --lang_src 'rw' --lang_tgt '' --seed 46 --epochs 1 --batch_size 128
echo '--- TASK end: model=rw seed=46'

echo '--- TASK start: model=rw seed=47'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/rw.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/rw.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'rw' --type 'single' --lang_src 'rw' --lang_tgt '' --seed 47 --epochs 1 --batch_size 128
echo '--- TASK end: model=rw seed=47'

echo '--- TASK start: model=rw seed=48'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/rw.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/rw.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'rw' --type 'single' --lang_src 'rw' --lang_tgt '' --seed 48 --epochs 1 --batch_size 128
echo '--- TASK end: model=rw seed=48'

echo '--- TASK start: model=rw seed=49'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/rw.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/rw.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'rw' --type 'single' --lang_src 'rw' --lang_tgt '' --seed 49 --epochs 1 --batch_size 128
echo '--- TASK end: model=rw seed=49'

echo '--- TASK start: model=rw seed=50'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/rw.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/rw.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'rw' --type 'single' --lang_src 'rw' --lang_tgt '' --seed 50 --epochs 1 --batch_size 128
echo '--- TASK end: model=rw seed=50'

echo '--- TASK start: model=rw seed=51'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/rw.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/rw.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'rw' --type 'single' --lang_src 'rw' --lang_tgt '' --seed 51 --epochs 1 --batch_size 128
echo '--- TASK end: model=rw seed=51'

echo '--- TASK start: model=sv-SE seed=42'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/sv-SE.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/sv-SE.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'sv-SE' --type 'single' --lang_src 'sv-SE' --lang_tgt '' --seed 42 --epochs 1 --batch_size 128
echo '--- TASK end: model=sv-SE seed=42'

echo '--- TASK start: model=sv-SE seed=43'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/sv-SE.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/sv-SE.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'sv-SE' --type 'single' --lang_src 'sv-SE' --lang_tgt '' --seed 43 --epochs 1 --batch_size 128
echo '--- TASK end: model=sv-SE seed=43'

echo '--- TASK start: model=sv-SE seed=44'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/sv-SE.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/sv-SE.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'sv-SE' --type 'single' --lang_src 'sv-SE' --lang_tgt '' --seed 44 --epochs 1 --batch_size 128
echo '--- TASK end: model=sv-SE seed=44'

echo '--- TASK start: model=sv-SE seed=45'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/sv-SE.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/sv-SE.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'sv-SE' --type 'single' --lang_src 'sv-SE' --lang_tgt '' --seed 45 --epochs 1 --batch_size 128
echo '--- TASK end: model=sv-SE seed=45'

echo '--- TASK start: model=sv-SE seed=46'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/sv-SE.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/sv-SE.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'sv-SE' --type 'single' --lang_src 'sv-SE' --lang_tgt '' --seed 46 --epochs 1 --batch_size 128
echo '--- TASK end: model=sv-SE seed=46'

echo '--- TASK start: model=sv-SE seed=47'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/sv-SE.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/sv-SE.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'sv-SE' --type 'single' --lang_src 'sv-SE' --lang_tgt '' --seed 47 --epochs 1 --batch_size 128
echo '--- TASK end: model=sv-SE seed=47'

echo '--- TASK start: model=sv-SE seed=48'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/sv-SE.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/sv-SE.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'sv-SE' --type 'single' --lang_src 'sv-SE' --lang_tgt '' --seed 48 --epochs 1 --batch_size 128
echo '--- TASK end: model=sv-SE seed=48'

echo '--- TASK start: model=sv-SE seed=49'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/sv-SE.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/sv-SE.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'sv-SE' --type 'single' --lang_src 'sv-SE' --lang_tgt '' --seed 49 --epochs 1 --batch_size 128
echo '--- TASK end: model=sv-SE seed=49'

echo '--- TASK start: model=sv-SE seed=50'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/sv-SE.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/sv-SE.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'sv-SE' --type 'single' --lang_src 'sv-SE' --lang_tgt '' --seed 50 --epochs 1 --batch_size 128
echo '--- TASK end: model=sv-SE seed=50'

echo '--- TASK start: model=sv-SE seed=51'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/sv-SE.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/sv-SE.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'sv-SE' --type 'single' --lang_src 'sv-SE' --lang_tgt '' --seed 51 --epochs 1 --batch_size 128
echo '--- TASK end: model=sv-SE seed=51'

echo '--- TASK start: model=sw seed=42'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/sw.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/sw.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'sw' --type 'single' --lang_src 'sw' --lang_tgt '' --seed 42 --epochs 1 --batch_size 128
echo '--- TASK end: model=sw seed=42'

echo '--- TASK start: model=sw seed=43'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/sw.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/sw.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'sw' --type 'single' --lang_src 'sw' --lang_tgt '' --seed 43 --epochs 1 --batch_size 128
echo '--- TASK end: model=sw seed=43'

echo '--- TASK start: model=sw seed=44'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/sw.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/sw.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'sw' --type 'single' --lang_src 'sw' --lang_tgt '' --seed 44 --epochs 1 --batch_size 128
echo '--- TASK end: model=sw seed=44'

echo '--- TASK start: model=sw seed=45'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/sw.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/sw.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'sw' --type 'single' --lang_src 'sw' --lang_tgt '' --seed 45 --epochs 1 --batch_size 128
echo '--- TASK end: model=sw seed=45'

echo '--- TASK start: model=sw seed=46'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/sw.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/sw.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'sw' --type 'single' --lang_src 'sw' --lang_tgt '' --seed 46 --epochs 1 --batch_size 128
echo '--- TASK end: model=sw seed=46'

echo '--- TASK start: model=sw seed=47'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/sw.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/sw.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'sw' --type 'single' --lang_src 'sw' --lang_tgt '' --seed 47 --epochs 1 --batch_size 128
echo '--- TASK end: model=sw seed=47'

echo '--- TASK start: model=sw seed=48'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/sw.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/sw.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'sw' --type 'single' --lang_src 'sw' --lang_tgt '' --seed 48 --epochs 1 --batch_size 128
echo '--- TASK end: model=sw seed=48'

echo '--- TASK start: model=sw seed=49'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/sw.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/sw.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'sw' --type 'single' --lang_src 'sw' --lang_tgt '' --seed 49 --epochs 1 --batch_size 128
echo '--- TASK end: model=sw seed=49'

echo '--- TASK start: model=sw seed=50'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/sw.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/sw.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'sw' --type 'single' --lang_src 'sw' --lang_tgt '' --seed 50 --epochs 1 --batch_size 128
echo '--- TASK end: model=sw seed=50'

echo '--- TASK start: model=sw seed=51'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/sw.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/sw.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'sw' --type 'single' --lang_src 'sw' --lang_tgt '' --seed 51 --epochs 1 --batch_size 128
echo '--- TASK end: model=sw seed=51'

echo '--- TASK start: model=ta seed=42'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/ta.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/ta.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'ta' --type 'single' --lang_src 'ta' --lang_tgt '' --seed 42 --epochs 1 --batch_size 128
echo '--- TASK end: model=ta seed=42'

echo '--- TASK start: model=ta seed=43'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/ta.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/ta.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'ta' --type 'single' --lang_src 'ta' --lang_tgt '' --seed 43 --epochs 1 --batch_size 128
echo '--- TASK end: model=ta seed=43'

echo '--- TASK start: model=ta seed=44'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/ta.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/ta.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'ta' --type 'single' --lang_src 'ta' --lang_tgt '' --seed 44 --epochs 1 --batch_size 128
echo '--- TASK end: model=ta seed=44'

echo '--- TASK start: model=ta seed=45'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/ta.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/ta.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'ta' --type 'single' --lang_src 'ta' --lang_tgt '' --seed 45 --epochs 1 --batch_size 128
echo '--- TASK end: model=ta seed=45'

echo '--- TASK start: model=ta seed=46'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/ta.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/ta.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'ta' --type 'single' --lang_src 'ta' --lang_tgt '' --seed 46 --epochs 1 --batch_size 128
echo '--- TASK end: model=ta seed=46'

echo '--- TASK start: model=ta seed=47'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/ta.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/ta.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'ta' --type 'single' --lang_src 'ta' --lang_tgt '' --seed 47 --epochs 1 --batch_size 128
echo '--- TASK end: model=ta seed=47'

echo '--- TASK start: model=ta seed=48'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/ta.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/ta.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'ta' --type 'single' --lang_src 'ta' --lang_tgt '' --seed 48 --epochs 1 --batch_size 128
echo '--- TASK end: model=ta seed=48'

echo '--- TASK start: model=ta seed=49'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/ta.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/ta.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'ta' --type 'single' --lang_src 'ta' --lang_tgt '' --seed 49 --epochs 1 --batch_size 128
echo '--- TASK end: model=ta seed=49'

echo '--- TASK start: model=ta seed=50'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/ta.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/ta.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'ta' --type 'single' --lang_src 'ta' --lang_tgt '' --seed 50 --epochs 1 --batch_size 128
echo '--- TASK end: model=ta seed=50'

echo '--- TASK start: model=ta seed=51'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/ta.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/ta.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'ta' --type 'single' --lang_src 'ta' --lang_tgt '' --seed 51 --epochs 1 --batch_size 128
echo '--- TASK end: model=ta seed=51'

echo '--- TASK start: model=th seed=42'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/th.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/th.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'th' --type 'single' --lang_src 'th' --lang_tgt '' --seed 42 --epochs 1 --batch_size 128
echo '--- TASK end: model=th seed=42'

echo '--- TASK start: model=th seed=43'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/th.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/th.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'th' --type 'single' --lang_src 'th' --lang_tgt '' --seed 43 --epochs 1 --batch_size 128
echo '--- TASK end: model=th seed=43'

echo '--- TASK start: model=th seed=44'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/th.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/th.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'th' --type 'single' --lang_src 'th' --lang_tgt '' --seed 44 --epochs 1 --batch_size 128
echo '--- TASK end: model=th seed=44'

echo '--- TASK start: model=th seed=45'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/th.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/th.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'th' --type 'single' --lang_src 'th' --lang_tgt '' --seed 45 --epochs 1 --batch_size 128
echo '--- TASK end: model=th seed=45'

echo '--- TASK start: model=th seed=46'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/th.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/th.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'th' --type 'single' --lang_src 'th' --lang_tgt '' --seed 46 --epochs 1 --batch_size 128
echo '--- TASK end: model=th seed=46'

echo '--- TASK start: model=th seed=47'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/th.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/th.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'th' --type 'single' --lang_src 'th' --lang_tgt '' --seed 47 --epochs 1 --batch_size 128
echo '--- TASK end: model=th seed=47'

echo '--- TASK start: model=th seed=48'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/th.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/th.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'th' --type 'single' --lang_src 'th' --lang_tgt '' --seed 48 --epochs 1 --batch_size 128
echo '--- TASK end: model=th seed=48'

echo '--- TASK start: model=th seed=49'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/th.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/th.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'th' --type 'single' --lang_src 'th' --lang_tgt '' --seed 49 --epochs 1 --batch_size 128
echo '--- TASK end: model=th seed=49'

echo '--- TASK start: model=th seed=50'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/th.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/th.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'th' --type 'single' --lang_src 'th' --lang_tgt '' --seed 50 --epochs 1 --batch_size 128
echo '--- TASK end: model=th seed=50'

echo '--- TASK start: model=th seed=51'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/th.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/th.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'th' --type 'single' --lang_src 'th' --lang_tgt '' --seed 51 --epochs 1 --batch_size 128
echo '--- TASK end: model=th seed=51'

echo '--- TASK start: model=tr seed=42'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/tr.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/tr.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'tr' --type 'single' --lang_src 'tr' --lang_tgt '' --seed 42 --epochs 1 --batch_size 128
echo '--- TASK end: model=tr seed=42'

echo '--- TASK start: model=tr seed=43'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/tr.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/tr.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'tr' --type 'single' --lang_src 'tr' --lang_tgt '' --seed 43 --epochs 1 --batch_size 128
echo '--- TASK end: model=tr seed=43'

echo '--- TASK start: model=tr seed=44'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/tr.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/tr.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'tr' --type 'single' --lang_src 'tr' --lang_tgt '' --seed 44 --epochs 1 --batch_size 128
echo '--- TASK end: model=tr seed=44'

echo '--- TASK start: model=tr seed=45'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/tr.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/tr.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'tr' --type 'single' --lang_src 'tr' --lang_tgt '' --seed 45 --epochs 1 --batch_size 128
echo '--- TASK end: model=tr seed=45'

echo '--- TASK start: model=tr seed=46'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/tr.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/tr.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'tr' --type 'single' --lang_src 'tr' --lang_tgt '' --seed 46 --epochs 1 --batch_size 128
echo '--- TASK end: model=tr seed=46'

echo '--- TASK start: model=tr seed=47'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/tr.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/tr.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'tr' --type 'single' --lang_src 'tr' --lang_tgt '' --seed 47 --epochs 1 --batch_size 128
echo '--- TASK end: model=tr seed=47'

echo '--- TASK start: model=tr seed=48'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/tr.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/tr.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'tr' --type 'single' --lang_src 'tr' --lang_tgt '' --seed 48 --epochs 1 --batch_size 128
echo '--- TASK end: model=tr seed=48'

echo '--- TASK start: model=tr seed=49'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/tr.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/tr.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'tr' --type 'single' --lang_src 'tr' --lang_tgt '' --seed 49 --epochs 1 --batch_size 128
echo '--- TASK end: model=tr seed=49'

echo '--- TASK start: model=tr seed=50'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/tr.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/tr.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'tr' --type 'single' --lang_src 'tr' --lang_tgt '' --seed 50 --epochs 1 --batch_size 128
echo '--- TASK end: model=tr seed=50'

echo '--- TASK start: model=tr seed=51'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/tr.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/tr.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'tr' --type 'single' --lang_src 'tr' --lang_tgt '' --seed 51 --epochs 1 --batch_size 128
echo '--- TASK end: model=tr seed=51'

echo '--- TASK start: model=ug seed=42'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/ug.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/ug.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'ug' --type 'single' --lang_src 'ug' --lang_tgt '' --seed 42 --epochs 1 --batch_size 128
echo '--- TASK end: model=ug seed=42'

echo '--- TASK start: model=ug seed=43'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/ug.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/ug.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'ug' --type 'single' --lang_src 'ug' --lang_tgt '' --seed 43 --epochs 1 --batch_size 128
echo '--- TASK end: model=ug seed=43'

echo '--- TASK start: model=ug seed=44'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/ug.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/ug.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'ug' --type 'single' --lang_src 'ug' --lang_tgt '' --seed 44 --epochs 1 --batch_size 128
echo '--- TASK end: model=ug seed=44'

echo '--- TASK start: model=ug seed=45'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/ug.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/ug.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'ug' --type 'single' --lang_src 'ug' --lang_tgt '' --seed 45 --epochs 1 --batch_size 128
echo '--- TASK end: model=ug seed=45'

echo '--- TASK start: model=ug seed=46'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/ug.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/ug.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'ug' --type 'single' --lang_src 'ug' --lang_tgt '' --seed 46 --epochs 1 --batch_size 128
echo '--- TASK end: model=ug seed=46'

echo '--- TASK start: model=ug seed=47'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/ug.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/ug.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'ug' --type 'single' --lang_src 'ug' --lang_tgt '' --seed 47 --epochs 1 --batch_size 128
echo '--- TASK end: model=ug seed=47'

echo '--- TASK start: model=ug seed=48'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/ug.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/ug.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'ug' --type 'single' --lang_src 'ug' --lang_tgt '' --seed 48 --epochs 1 --batch_size 128
echo '--- TASK end: model=ug seed=48'

echo '--- TASK start: model=ug seed=49'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/ug.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/ug.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'ug' --type 'single' --lang_src 'ug' --lang_tgt '' --seed 49 --epochs 1 --batch_size 128
echo '--- TASK end: model=ug seed=49'

echo '--- TASK start: model=ug seed=50'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/ug.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/ug.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'ug' --type 'single' --lang_src 'ug' --lang_tgt '' --seed 50 --epochs 1 --batch_size 128
echo '--- TASK end: model=ug seed=50'

echo '--- TASK start: model=ug seed=51'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/ug.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/ug.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'ug' --type 'single' --lang_src 'ug' --lang_tgt '' --seed 51 --epochs 1 --batch_size 128
echo '--- TASK end: model=ug seed=51'

echo '--- TASK start: model=uk seed=42'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/uk.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/uk.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'uk' --type 'single' --lang_src 'uk' --lang_tgt '' --seed 42 --epochs 1 --batch_size 128
echo '--- TASK end: model=uk seed=42'

echo '--- TASK start: model=uk seed=43'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/uk.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/uk.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'uk' --type 'single' --lang_src 'uk' --lang_tgt '' --seed 43 --epochs 1 --batch_size 128
echo '--- TASK end: model=uk seed=43'

echo '--- TASK start: model=uk seed=44'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/uk.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/uk.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'uk' --type 'single' --lang_src 'uk' --lang_tgt '' --seed 44 --epochs 1 --batch_size 128
echo '--- TASK end: model=uk seed=44'

echo '--- TASK start: model=uk seed=45'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/uk.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/uk.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'uk' --type 'single' --lang_src 'uk' --lang_tgt '' --seed 45 --epochs 1 --batch_size 128
echo '--- TASK end: model=uk seed=45'

echo '--- TASK start: model=uk seed=46'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/uk.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/uk.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'uk' --type 'single' --lang_src 'uk' --lang_tgt '' --seed 46 --epochs 1 --batch_size 128
echo '--- TASK end: model=uk seed=46'

echo '--- TASK start: model=uk seed=47'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/uk.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/uk.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'uk' --type 'single' --lang_src 'uk' --lang_tgt '' --seed 47 --epochs 1 --batch_size 128
echo '--- TASK end: model=uk seed=47'

echo '--- TASK start: model=uk seed=48'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/uk.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/uk.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'uk' --type 'single' --lang_src 'uk' --lang_tgt '' --seed 48 --epochs 1 --batch_size 128
echo '--- TASK end: model=uk seed=48'

echo '--- TASK start: model=uk seed=49'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/uk.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/uk.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'uk' --type 'single' --lang_src 'uk' --lang_tgt '' --seed 49 --epochs 1 --batch_size 128
echo '--- TASK end: model=uk seed=49'

echo '--- TASK start: model=uk seed=50'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/uk.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/uk.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'uk' --type 'single' --lang_src 'uk' --lang_tgt '' --seed 50 --epochs 1 --batch_size 128
echo '--- TASK end: model=uk seed=50'

echo '--- TASK start: model=uk seed=51'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/uk.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/uk.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'uk' --type 'single' --lang_src 'uk' --lang_tgt '' --seed 51 --epochs 1 --batch_size 128
echo '--- TASK end: model=uk seed=51'

echo '--- TASK start: model=ur seed=42'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/ur.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/ur.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'ur' --type 'single' --lang_src 'ur' --lang_tgt '' --seed 42 --epochs 1 --batch_size 128
echo '--- TASK end: model=ur seed=42'

echo '--- TASK start: model=ur seed=43'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/ur.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/ur.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'ur' --type 'single' --lang_src 'ur' --lang_tgt '' --seed 43 --epochs 1 --batch_size 128
echo '--- TASK end: model=ur seed=43'

echo '--- TASK start: model=ur seed=44'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/ur.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/ur.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'ur' --type 'single' --lang_src 'ur' --lang_tgt '' --seed 44 --epochs 1 --batch_size 128
echo '--- TASK end: model=ur seed=44'

echo '--- TASK start: model=ur seed=45'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/ur.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/ur.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'ur' --type 'single' --lang_src 'ur' --lang_tgt '' --seed 45 --epochs 1 --batch_size 128
echo '--- TASK end: model=ur seed=45'

