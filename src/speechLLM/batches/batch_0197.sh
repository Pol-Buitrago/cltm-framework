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

echo '--- TASK start: model=th seed=46'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/th.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/th.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_dual.csv' --model_id 'th' --type 'single' --lang_src 'th' --lang_tgt '' --seed 46 --epochs 1 --batch_size 128
echo '--- TASK end: model=th seed=46'

echo '--- TASK start: model=th seed=47'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/th.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/th.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_dual.csv' --model_id 'th' --type 'single' --lang_src 'th' --lang_tgt '' --seed 47 --epochs 1 --batch_size 128
echo '--- TASK end: model=th seed=47'

echo '--- TASK start: model=th seed=48'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/th.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/th.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_dual.csv' --model_id 'th' --type 'single' --lang_src 'th' --lang_tgt '' --seed 48 --epochs 1 --batch_size 128
echo '--- TASK end: model=th seed=48'

echo '--- TASK start: model=th seed=49'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/th.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/th.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_dual.csv' --model_id 'th' --type 'single' --lang_src 'th' --lang_tgt '' --seed 49 --epochs 1 --batch_size 128
echo '--- TASK end: model=th seed=49'

echo '--- TASK start: model=th seed=50'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/th.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/th.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_dual.csv' --model_id 'th' --type 'single' --lang_src 'th' --lang_tgt '' --seed 50 --epochs 1 --batch_size 128
echo '--- TASK end: model=th seed=50'

echo '--- TASK start: model=th seed=51'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/th.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/th.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_dual.csv' --model_id 'th' --type 'single' --lang_src 'th' --lang_tgt '' --seed 51 --epochs 1 --batch_size 128
echo '--- TASK end: model=th seed=51'

echo '--- TASK start: model=tr seed=42'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/tr.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/tr.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_dual.csv' --model_id 'tr' --type 'single' --lang_src 'tr' --lang_tgt '' --seed 42 --epochs 1 --batch_size 128
echo '--- TASK end: model=tr seed=42'

echo '--- TASK start: model=tr seed=43'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/tr.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/tr.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_dual.csv' --model_id 'tr' --type 'single' --lang_src 'tr' --lang_tgt '' --seed 43 --epochs 1 --batch_size 128
echo '--- TASK end: model=tr seed=43'

echo '--- TASK start: model=tr seed=44'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/tr.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/tr.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_dual.csv' --model_id 'tr' --type 'single' --lang_src 'tr' --lang_tgt '' --seed 44 --epochs 1 --batch_size 128
echo '--- TASK end: model=tr seed=44'

echo '--- TASK start: model=tr seed=45'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/tr.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/tr.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_dual.csv' --model_id 'tr' --type 'single' --lang_src 'tr' --lang_tgt '' --seed 45 --epochs 1 --batch_size 128
echo '--- TASK end: model=tr seed=45'

echo '--- TASK start: model=tr seed=46'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/tr.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/tr.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_dual.csv' --model_id 'tr' --type 'single' --lang_src 'tr' --lang_tgt '' --seed 46 --epochs 1 --batch_size 128
echo '--- TASK end: model=tr seed=46'

echo '--- TASK start: model=tr seed=47'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/tr.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/tr.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_dual.csv' --model_id 'tr' --type 'single' --lang_src 'tr' --lang_tgt '' --seed 47 --epochs 1 --batch_size 128
echo '--- TASK end: model=tr seed=47'

echo '--- TASK start: model=tr seed=48'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/tr.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/tr.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_dual.csv' --model_id 'tr' --type 'single' --lang_src 'tr' --lang_tgt '' --seed 48 --epochs 1 --batch_size 128
echo '--- TASK end: model=tr seed=48'

echo '--- TASK start: model=tr seed=49'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/tr.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/tr.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_dual.csv' --model_id 'tr' --type 'single' --lang_src 'tr' --lang_tgt '' --seed 49 --epochs 1 --batch_size 128
echo '--- TASK end: model=tr seed=49'

echo '--- TASK start: model=tr seed=50'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/tr.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/tr.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_dual.csv' --model_id 'tr' --type 'single' --lang_src 'tr' --lang_tgt '' --seed 50 --epochs 1 --batch_size 128
echo '--- TASK end: model=tr seed=50'

echo '--- TASK start: model=tr seed=51'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/tr.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/tr.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_dual.csv' --model_id 'tr' --type 'single' --lang_src 'tr' --lang_tgt '' --seed 51 --epochs 1 --batch_size 128
echo '--- TASK end: model=tr seed=51'

echo '--- TASK start: model=ug seed=42'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/ug.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/ug.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_dual.csv' --model_id 'ug' --type 'single' --lang_src 'ug' --lang_tgt '' --seed 42 --epochs 1 --batch_size 128
echo '--- TASK end: model=ug seed=42'

echo '--- TASK start: model=ug seed=43'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/ug.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/ug.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_dual.csv' --model_id 'ug' --type 'single' --lang_src 'ug' --lang_tgt '' --seed 43 --epochs 1 --batch_size 128
echo '--- TASK end: model=ug seed=43'

echo '--- TASK start: model=ug seed=44'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/ug.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/ug.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_dual.csv' --model_id 'ug' --type 'single' --lang_src 'ug' --lang_tgt '' --seed 44 --epochs 1 --batch_size 128
echo '--- TASK end: model=ug seed=44'

echo '--- TASK start: model=ug seed=45'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/ug.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/ug.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_dual.csv' --model_id 'ug' --type 'single' --lang_src 'ug' --lang_tgt '' --seed 45 --epochs 1 --batch_size 128
echo '--- TASK end: model=ug seed=45'

echo '--- TASK start: model=ug seed=46'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/ug.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/ug.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_dual.csv' --model_id 'ug' --type 'single' --lang_src 'ug' --lang_tgt '' --seed 46 --epochs 1 --batch_size 128
echo '--- TASK end: model=ug seed=46'

echo '--- TASK start: model=ug seed=47'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/ug.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/ug.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_dual.csv' --model_id 'ug' --type 'single' --lang_src 'ug' --lang_tgt '' --seed 47 --epochs 1 --batch_size 128
echo '--- TASK end: model=ug seed=47'

echo '--- TASK start: model=ug seed=48'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/ug.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/ug.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_dual.csv' --model_id 'ug' --type 'single' --lang_src 'ug' --lang_tgt '' --seed 48 --epochs 1 --batch_size 128
echo '--- TASK end: model=ug seed=48'

echo '--- TASK start: model=ug seed=49'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/ug.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/ug.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_dual.csv' --model_id 'ug' --type 'single' --lang_src 'ug' --lang_tgt '' --seed 49 --epochs 1 --batch_size 128
echo '--- TASK end: model=ug seed=49'

echo '--- TASK start: model=ug seed=50'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/ug.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/ug.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_dual.csv' --model_id 'ug' --type 'single' --lang_src 'ug' --lang_tgt '' --seed 50 --epochs 1 --batch_size 128
echo '--- TASK end: model=ug seed=50'

echo '--- TASK start: model=ug seed=51'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/ug.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/ug.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_dual.csv' --model_id 'ug' --type 'single' --lang_src 'ug' --lang_tgt '' --seed 51 --epochs 1 --batch_size 128
echo '--- TASK end: model=ug seed=51'

echo '--- TASK start: model=uk seed=42'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/uk.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/uk.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_dual.csv' --model_id 'uk' --type 'single' --lang_src 'uk' --lang_tgt '' --seed 42 --epochs 1 --batch_size 128
echo '--- TASK end: model=uk seed=42'

echo '--- TASK start: model=uk seed=43'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/uk.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/uk.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_dual.csv' --model_id 'uk' --type 'single' --lang_src 'uk' --lang_tgt '' --seed 43 --epochs 1 --batch_size 128
echo '--- TASK end: model=uk seed=43'

echo '--- TASK start: model=uk seed=44'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/uk.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/uk.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_dual.csv' --model_id 'uk' --type 'single' --lang_src 'uk' --lang_tgt '' --seed 44 --epochs 1 --batch_size 128
echo '--- TASK end: model=uk seed=44'

echo '--- TASK start: model=uk seed=45'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/uk.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/uk.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_dual.csv' --model_id 'uk' --type 'single' --lang_src 'uk' --lang_tgt '' --seed 45 --epochs 1 --batch_size 128
echo '--- TASK end: model=uk seed=45'

echo '--- TASK start: model=uk seed=46'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/uk.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/uk.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_dual.csv' --model_id 'uk' --type 'single' --lang_src 'uk' --lang_tgt '' --seed 46 --epochs 1 --batch_size 128
echo '--- TASK end: model=uk seed=46'

echo '--- TASK start: model=uk seed=47'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/uk.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/uk.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_dual.csv' --model_id 'uk' --type 'single' --lang_src 'uk' --lang_tgt '' --seed 47 --epochs 1 --batch_size 128
echo '--- TASK end: model=uk seed=47'

echo '--- TASK start: model=uk seed=48'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/uk.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/uk.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_dual.csv' --model_id 'uk' --type 'single' --lang_src 'uk' --lang_tgt '' --seed 48 --epochs 1 --batch_size 128
echo '--- TASK end: model=uk seed=48'

echo '--- TASK start: model=uk seed=49'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/uk.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/uk.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_dual.csv' --model_id 'uk' --type 'single' --lang_src 'uk' --lang_tgt '' --seed 49 --epochs 1 --batch_size 128
echo '--- TASK end: model=uk seed=49'

echo '--- TASK start: model=uk seed=50'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/uk.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/uk.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_dual.csv' --model_id 'uk' --type 'single' --lang_src 'uk' --lang_tgt '' --seed 50 --epochs 1 --batch_size 128
echo '--- TASK end: model=uk seed=50'

echo '--- TASK start: model=uk seed=51'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/uk.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/uk.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_dual.csv' --model_id 'uk' --type 'single' --lang_src 'uk' --lang_tgt '' --seed 51 --epochs 1 --batch_size 128
echo '--- TASK end: model=uk seed=51'

echo '--- TASK start: model=ur seed=42'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/ur.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/ur.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_dual.csv' --model_id 'ur' --type 'single' --lang_src 'ur' --lang_tgt '' --seed 42 --epochs 1 --batch_size 128
echo '--- TASK end: model=ur seed=42'

echo '--- TASK start: model=ur seed=43'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/ur.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/ur.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_dual.csv' --model_id 'ur' --type 'single' --lang_src 'ur' --lang_tgt '' --seed 43 --epochs 1 --batch_size 128
echo '--- TASK end: model=ur seed=43'

echo '--- TASK start: model=ur seed=44'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/ur.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/ur.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_dual.csv' --model_id 'ur' --type 'single' --lang_src 'ur' --lang_tgt '' --seed 44 --epochs 1 --batch_size 128
echo '--- TASK end: model=ur seed=44'

echo '--- TASK start: model=ur seed=45'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/ur.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/ur.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_dual.csv' --model_id 'ur' --type 'single' --lang_src 'ur' --lang_tgt '' --seed 45 --epochs 1 --batch_size 128
echo '--- TASK end: model=ur seed=45'

echo '--- TASK start: model=ur seed=46'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/ur.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/ur.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_dual.csv' --model_id 'ur' --type 'single' --lang_src 'ur' --lang_tgt '' --seed 46 --epochs 1 --batch_size 128
echo '--- TASK end: model=ur seed=46'

echo '--- TASK start: model=ur seed=47'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/ur.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/ur.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_dual.csv' --model_id 'ur' --type 'single' --lang_src 'ur' --lang_tgt '' --seed 47 --epochs 1 --batch_size 128
echo '--- TASK end: model=ur seed=47'

echo '--- TASK start: model=ur seed=48'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/ur.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/ur.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_dual.csv' --model_id 'ur' --type 'single' --lang_src 'ur' --lang_tgt '' --seed 48 --epochs 1 --batch_size 128
echo '--- TASK end: model=ur seed=48'

echo '--- TASK start: model=ur seed=49'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/ur.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/ur.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_dual.csv' --model_id 'ur' --type 'single' --lang_src 'ur' --lang_tgt '' --seed 49 --epochs 1 --batch_size 128
echo '--- TASK end: model=ur seed=49'

echo '--- TASK start: model=ur seed=50'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/ur.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/ur.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_dual.csv' --model_id 'ur' --type 'single' --lang_src 'ur' --lang_tgt '' --seed 50 --epochs 1 --batch_size 128
echo '--- TASK end: model=ur seed=50'

echo '--- TASK start: model=ur seed=51'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/ur.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/ur.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_dual.csv' --model_id 'ur' --type 'single' --lang_src 'ur' --lang_tgt '' --seed 51 --epochs 1 --batch_size 128
echo '--- TASK end: model=ur seed=51'

echo '--- TASK start: model=uz seed=42'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/uz.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/uz.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_dual.csv' --model_id 'uz' --type 'single' --lang_src 'uz' --lang_tgt '' --seed 42 --epochs 1 --batch_size 128
echo '--- TASK end: model=uz seed=42'

echo '--- TASK start: model=uz seed=43'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/uz.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/uz.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_dual.csv' --model_id 'uz' --type 'single' --lang_src 'uz' --lang_tgt '' --seed 43 --epochs 1 --batch_size 128
echo '--- TASK end: model=uz seed=43'

echo '--- TASK start: model=uz seed=44'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/uz.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/uz.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_dual.csv' --model_id 'uz' --type 'single' --lang_src 'uz' --lang_tgt '' --seed 44 --epochs 1 --batch_size 128
echo '--- TASK end: model=uz seed=44'

echo '--- TASK start: model=uz seed=45'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/uz.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/uz.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_dual.csv' --model_id 'uz' --type 'single' --lang_src 'uz' --lang_tgt '' --seed 45 --epochs 1 --batch_size 128
echo '--- TASK end: model=uz seed=45'

echo '--- TASK start: model=uz seed=46'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/uz.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/uz.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_dual.csv' --model_id 'uz' --type 'single' --lang_src 'uz' --lang_tgt '' --seed 46 --epochs 1 --batch_size 128
echo '--- TASK end: model=uz seed=46'

echo '--- TASK start: model=uz seed=47'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/uz.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/uz.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_dual.csv' --model_id 'uz' --type 'single' --lang_src 'uz' --lang_tgt '' --seed 47 --epochs 1 --batch_size 128
echo '--- TASK end: model=uz seed=47'

echo '--- TASK start: model=uz seed=48'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/uz.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/uz.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_dual.csv' --model_id 'uz' --type 'single' --lang_src 'uz' --lang_tgt '' --seed 48 --epochs 1 --batch_size 128
echo '--- TASK end: model=uz seed=48'

echo '--- TASK start: model=uz seed=49'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/uz.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/uz.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_dual.csv' --model_id 'uz' --type 'single' --lang_src 'uz' --lang_tgt '' --seed 49 --epochs 1 --batch_size 128
echo '--- TASK end: model=uz seed=49'

echo '--- TASK start: model=uz seed=50'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/uz.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/uz.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_dual.csv' --model_id 'uz' --type 'single' --lang_src 'uz' --lang_tgt '' --seed 50 --epochs 1 --batch_size 128
echo '--- TASK end: model=uz seed=50'

echo '--- TASK start: model=uz seed=51'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/uz.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/uz.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_dual.csv' --model_id 'uz' --type 'single' --lang_src 'uz' --lang_tgt '' --seed 51 --epochs 1 --batch_size 128
echo '--- TASK end: model=uz seed=51'

echo '--- TASK start: model=yue seed=42'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/yue.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/yue.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_dual.csv' --model_id 'yue' --type 'single' --lang_src 'yue' --lang_tgt '' --seed 42 --epochs 1 --batch_size 128
echo '--- TASK end: model=yue seed=42'

echo '--- TASK start: model=yue seed=43'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/yue.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/yue.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_dual.csv' --model_id 'yue' --type 'single' --lang_src 'yue' --lang_tgt '' --seed 43 --epochs 1 --batch_size 128
echo '--- TASK end: model=yue seed=43'

echo '--- TASK start: model=yue seed=44'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/yue.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/yue.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_dual.csv' --model_id 'yue' --type 'single' --lang_src 'yue' --lang_tgt '' --seed 44 --epochs 1 --batch_size 128
echo '--- TASK end: model=yue seed=44'

echo '--- TASK start: model=yue seed=45'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/yue.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/yue.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_dual.csv' --model_id 'yue' --type 'single' --lang_src 'yue' --lang_tgt '' --seed 45 --epochs 1 --batch_size 128
echo '--- TASK end: model=yue seed=45'

echo '--- TASK start: model=yue seed=46'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/yue.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/yue.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_dual.csv' --model_id 'yue' --type 'single' --lang_src 'yue' --lang_tgt '' --seed 46 --epochs 1 --batch_size 128
echo '--- TASK end: model=yue seed=46'

echo '--- TASK start: model=yue seed=47'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/yue.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/yue.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_dual.csv' --model_id 'yue' --type 'single' --lang_src 'yue' --lang_tgt '' --seed 47 --epochs 1 --batch_size 128
echo '--- TASK end: model=yue seed=47'

echo '--- TASK start: model=yue seed=48'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/yue.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/yue.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_dual.csv' --model_id 'yue' --type 'single' --lang_src 'yue' --lang_tgt '' --seed 48 --epochs 1 --batch_size 128
echo '--- TASK end: model=yue seed=48'

echo '--- TASK start: model=yue seed=49'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/yue.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/yue.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_dual.csv' --model_id 'yue' --type 'single' --lang_src 'yue' --lang_tgt '' --seed 49 --epochs 1 --batch_size 128
echo '--- TASK end: model=yue seed=49'

echo '--- TASK start: model=yue seed=50'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/yue.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/yue.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_dual.csv' --model_id 'yue' --type 'single' --lang_src 'yue' --lang_tgt '' --seed 50 --epochs 1 --batch_size 128
echo '--- TASK end: model=yue seed=50'

echo '--- TASK start: model=yue seed=51'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/yue.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/yue.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_dual.csv' --model_id 'yue' --type 'single' --lang_src 'yue' --lang_tgt '' --seed 51 --epochs 1 --batch_size 128
echo '--- TASK end: model=yue seed=51'

echo '--- TASK start: model=zh-CN seed=42'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/zh-CN.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/zh-CN.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_dual.csv' --model_id 'zh-CN' --type 'single' --lang_src 'zh-CN' --lang_tgt '' --seed 42 --epochs 1 --batch_size 128
echo '--- TASK end: model=zh-CN seed=42'

echo '--- TASK start: model=zh-CN seed=43'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/zh-CN.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/zh-CN.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_dual.csv' --model_id 'zh-CN' --type 'single' --lang_src 'zh-CN' --lang_tgt '' --seed 43 --epochs 1 --batch_size 128
echo '--- TASK end: model=zh-CN seed=43'

echo '--- TASK start: model=zh-CN seed=44'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/zh-CN.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/zh-CN.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_dual.csv' --model_id 'zh-CN' --type 'single' --lang_src 'zh-CN' --lang_tgt '' --seed 44 --epochs 1 --batch_size 128
echo '--- TASK end: model=zh-CN seed=44'

echo '--- TASK start: model=zh-CN seed=45'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/zh-CN.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/zh-CN.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_dual.csv' --model_id 'zh-CN' --type 'single' --lang_src 'zh-CN' --lang_tgt '' --seed 45 --epochs 1 --batch_size 128
echo '--- TASK end: model=zh-CN seed=45'

echo '--- TASK start: model=zh-CN seed=46'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/zh-CN.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/zh-CN.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_dual.csv' --model_id 'zh-CN' --type 'single' --lang_src 'zh-CN' --lang_tgt '' --seed 46 --epochs 1 --batch_size 128
echo '--- TASK end: model=zh-CN seed=46'

echo '--- TASK start: model=zh-CN seed=47'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/zh-CN.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/zh-CN.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_dual.csv' --model_id 'zh-CN' --type 'single' --lang_src 'zh-CN' --lang_tgt '' --seed 47 --epochs 1 --batch_size 128
echo '--- TASK end: model=zh-CN seed=47'

echo '--- TASK start: model=zh-CN seed=48'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/zh-CN.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/zh-CN.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_dual.csv' --model_id 'zh-CN' --type 'single' --lang_src 'zh-CN' --lang_tgt '' --seed 48 --epochs 1 --batch_size 128
echo '--- TASK end: model=zh-CN seed=48'

echo '--- TASK start: model=zh-CN seed=49'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/zh-CN.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/zh-CN.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_dual.csv' --model_id 'zh-CN' --type 'single' --lang_src 'zh-CN' --lang_tgt '' --seed 49 --epochs 1 --batch_size 128
echo '--- TASK end: model=zh-CN seed=49'

echo '--- TASK start: model=zh-CN seed=50'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/zh-CN.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/zh-CN.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_dual.csv' --model_id 'zh-CN' --type 'single' --lang_src 'zh-CN' --lang_tgt '' --seed 50 --epochs 1 --batch_size 128
echo '--- TASK end: model=zh-CN seed=50'

echo '--- TASK start: model=zh-CN seed=51'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/zh-CN.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/zh-CN.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_dual.csv' --model_id 'zh-CN' --type 'single' --lang_src 'zh-CN' --lang_tgt '' --seed 51 --epochs 1 --batch_size 128
echo '--- TASK end: model=zh-CN seed=51'

echo '--- TASK start: model=zh-HK seed=42'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/zh-HK.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/zh-HK.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_dual.csv' --model_id 'zh-HK' --type 'single' --lang_src 'zh-HK' --lang_tgt '' --seed 42 --epochs 1 --batch_size 128
echo '--- TASK end: model=zh-HK seed=42'

echo '--- TASK start: model=zh-HK seed=43'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/zh-HK.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/zh-HK.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_dual.csv' --model_id 'zh-HK' --type 'single' --lang_src 'zh-HK' --lang_tgt '' --seed 43 --epochs 1 --batch_size 128
echo '--- TASK end: model=zh-HK seed=43'

echo '--- TASK start: model=zh-HK seed=44'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/zh-HK.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/zh-HK.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_dual.csv' --model_id 'zh-HK' --type 'single' --lang_src 'zh-HK' --lang_tgt '' --seed 44 --epochs 1 --batch_size 128
echo '--- TASK end: model=zh-HK seed=44'

echo '--- TASK start: model=zh-HK seed=45'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/zh-HK.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/zh-HK.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_dual.csv' --model_id 'zh-HK' --type 'single' --lang_src 'zh-HK' --lang_tgt '' --seed 45 --epochs 1 --batch_size 128
echo '--- TASK end: model=zh-HK seed=45'

echo '--- TASK start: model=zh-HK seed=46'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/zh-HK.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/zh-HK.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_dual.csv' --model_id 'zh-HK' --type 'single' --lang_src 'zh-HK' --lang_tgt '' --seed 46 --epochs 1 --batch_size 128
echo '--- TASK end: model=zh-HK seed=46'

echo '--- TASK start: model=zh-HK seed=47'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/zh-HK.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/zh-HK.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_dual.csv' --model_id 'zh-HK' --type 'single' --lang_src 'zh-HK' --lang_tgt '' --seed 47 --epochs 1 --batch_size 128
echo '--- TASK end: model=zh-HK seed=47'

echo '--- TASK start: model=zh-HK seed=48'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/zh-HK.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/zh-HK.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_dual.csv' --model_id 'zh-HK' --type 'single' --lang_src 'zh-HK' --lang_tgt '' --seed 48 --epochs 1 --batch_size 128
echo '--- TASK end: model=zh-HK seed=48'

echo '--- TASK start: model=zh-HK seed=49'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/zh-HK.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/zh-HK.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_dual.csv' --model_id 'zh-HK' --type 'single' --lang_src 'zh-HK' --lang_tgt '' --seed 49 --epochs 1 --batch_size 128
echo '--- TASK end: model=zh-HK seed=49'

echo '--- TASK start: model=zh-HK seed=50'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/zh-HK.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/zh-HK.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_dual.csv' --model_id 'zh-HK' --type 'single' --lang_src 'zh-HK' --lang_tgt '' --seed 50 --epochs 1 --batch_size 128
echo '--- TASK end: model=zh-HK seed=50'

echo '--- TASK start: model=zh-HK seed=51'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/zh-HK.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/zh-HK.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_dual.csv' --model_id 'zh-HK' --type 'single' --lang_src 'zh-HK' --lang_tgt '' --seed 51 --epochs 1 --batch_size 128
echo '--- TASK end: model=zh-HK seed=51'

echo '--- TASK start: model=zh-TW seed=42'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/zh-TW.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/zh-TW.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_dual.csv' --model_id 'zh-TW' --type 'single' --lang_src 'zh-TW' --lang_tgt '' --seed 42 --epochs 1 --batch_size 128
echo '--- TASK end: model=zh-TW seed=42'

echo '--- TASK start: model=zh-TW seed=43'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/zh-TW.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/zh-TW.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_dual.csv' --model_id 'zh-TW' --type 'single' --lang_src 'zh-TW' --lang_tgt '' --seed 43 --epochs 1 --batch_size 128
echo '--- TASK end: model=zh-TW seed=43'

echo '--- TASK start: model=zh-TW seed=44'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/zh-TW.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/zh-TW.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_dual.csv' --model_id 'zh-TW' --type 'single' --lang_src 'zh-TW' --lang_tgt '' --seed 44 --epochs 1 --batch_size 128
echo '--- TASK end: model=zh-TW seed=44'

echo '--- TASK start: model=zh-TW seed=45'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/zh-TW.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/zh-TW.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_dual.csv' --model_id 'zh-TW' --type 'single' --lang_src 'zh-TW' --lang_tgt '' --seed 45 --epochs 1 --batch_size 128
echo '--- TASK end: model=zh-TW seed=45'

echo '--- TASK start: model=zh-TW seed=46'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/zh-TW.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/zh-TW.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_dual.csv' --model_id 'zh-TW' --type 'single' --lang_src 'zh-TW' --lang_tgt '' --seed 46 --epochs 1 --batch_size 128
echo '--- TASK end: model=zh-TW seed=46'

echo '--- TASK start: model=zh-TW seed=47'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/zh-TW.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/zh-TW.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_dual.csv' --model_id 'zh-TW' --type 'single' --lang_src 'zh-TW' --lang_tgt '' --seed 47 --epochs 1 --batch_size 128
echo '--- TASK end: model=zh-TW seed=47'

echo '--- TASK start: model=zh-TW seed=48'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/zh-TW.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/zh-TW.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_dual.csv' --model_id 'zh-TW' --type 'single' --lang_src 'zh-TW' --lang_tgt '' --seed 48 --epochs 1 --batch_size 128
echo '--- TASK end: model=zh-TW seed=48'

echo '--- TASK start: model=zh-TW seed=49'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/zh-TW.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/zh-TW.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_dual.csv' --model_id 'zh-TW' --type 'single' --lang_src 'zh-TW' --lang_tgt '' --seed 49 --epochs 1 --batch_size 128
echo '--- TASK end: model=zh-TW seed=49'

echo '--- TASK start: model=zh-TW seed=50'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/zh-TW.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/zh-TW.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_dual.csv' --model_id 'zh-TW' --type 'single' --lang_src 'zh-TW' --lang_tgt '' --seed 50 --epochs 1 --batch_size 128
echo '--- TASK end: model=zh-TW seed=50'

echo '--- TASK start: model=zh-TW seed=51'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/zh-TW.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv/zh-TW.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_dual.csv' --model_id 'zh-TW' --type 'single' --lang_src 'zh-TW' --lang_tgt '' --seed 51 --epochs 1 --batch_size 128
echo '--- TASK end: model=zh-TW seed=51'

