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

echo '--- TASK start: model=it seed=46'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/it.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/it.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'it' --type 'single' --lang_src 'it' --lang_tgt '' --seed 46 --epochs 1 --batch_size 128
echo '--- TASK end: model=it seed=46'

echo '--- TASK start: model=it seed=47'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/it.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/it.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'it' --type 'single' --lang_src 'it' --lang_tgt '' --seed 47 --epochs 1 --batch_size 128
echo '--- TASK end: model=it seed=47'

echo '--- TASK start: model=it seed=48'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/it.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/it.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'it' --type 'single' --lang_src 'it' --lang_tgt '' --seed 48 --epochs 1 --batch_size 128
echo '--- TASK end: model=it seed=48'

echo '--- TASK start: model=it seed=49'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/it.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/it.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'it' --type 'single' --lang_src 'it' --lang_tgt '' --seed 49 --epochs 1 --batch_size 128
echo '--- TASK end: model=it seed=49'

echo '--- TASK start: model=it seed=50'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/it.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/it.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'it' --type 'single' --lang_src 'it' --lang_tgt '' --seed 50 --epochs 1 --batch_size 128
echo '--- TASK end: model=it seed=50'

echo '--- TASK start: model=it seed=51'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/it.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/it.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'it' --type 'single' --lang_src 'it' --lang_tgt '' --seed 51 --epochs 1 --batch_size 128
echo '--- TASK end: model=it seed=51'

echo '--- TASK start: model=ka seed=42'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/ka.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/ka.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'ka' --type 'single' --lang_src 'ka' --lang_tgt '' --seed 42 --epochs 1 --batch_size 128
echo '--- TASK end: model=ka seed=42'

echo '--- TASK start: model=ka seed=43'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/ka.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/ka.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'ka' --type 'single' --lang_src 'ka' --lang_tgt '' --seed 43 --epochs 1 --batch_size 128
echo '--- TASK end: model=ka seed=43'

echo '--- TASK start: model=ka seed=44'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/ka.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/ka.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'ka' --type 'single' --lang_src 'ka' --lang_tgt '' --seed 44 --epochs 1 --batch_size 128
echo '--- TASK end: model=ka seed=44'

echo '--- TASK start: model=ka seed=45'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/ka.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/ka.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'ka' --type 'single' --lang_src 'ka' --lang_tgt '' --seed 45 --epochs 1 --batch_size 128
echo '--- TASK end: model=ka seed=45'

echo '--- TASK start: model=ka seed=46'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/ka.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/ka.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'ka' --type 'single' --lang_src 'ka' --lang_tgt '' --seed 46 --epochs 1 --batch_size 128
echo '--- TASK end: model=ka seed=46'

echo '--- TASK start: model=ka seed=47'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/ka.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/ka.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'ka' --type 'single' --lang_src 'ka' --lang_tgt '' --seed 47 --epochs 1 --batch_size 128
echo '--- TASK end: model=ka seed=47'

echo '--- TASK start: model=ka seed=48'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/ka.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/ka.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'ka' --type 'single' --lang_src 'ka' --lang_tgt '' --seed 48 --epochs 1 --batch_size 128
echo '--- TASK end: model=ka seed=48'

echo '--- TASK start: model=ka seed=49'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/ka.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/ka.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'ka' --type 'single' --lang_src 'ka' --lang_tgt '' --seed 49 --epochs 1 --batch_size 128
echo '--- TASK end: model=ka seed=49'

echo '--- TASK start: model=ka seed=50'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/ka.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/ka.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'ka' --type 'single' --lang_src 'ka' --lang_tgt '' --seed 50 --epochs 1 --batch_size 128
echo '--- TASK end: model=ka seed=50'

echo '--- TASK start: model=ka seed=51'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/ka.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/ka.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'ka' --type 'single' --lang_src 'ka' --lang_tgt '' --seed 51 --epochs 1 --batch_size 128
echo '--- TASK end: model=ka seed=51'

echo '--- TASK start: model=kab seed=42'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/kab.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/kab.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'kab' --type 'single' --lang_src 'kab' --lang_tgt '' --seed 42 --epochs 1 --batch_size 128
echo '--- TASK end: model=kab seed=42'

echo '--- TASK start: model=kab seed=43'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/kab.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/kab.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'kab' --type 'single' --lang_src 'kab' --lang_tgt '' --seed 43 --epochs 1 --batch_size 128
echo '--- TASK end: model=kab seed=43'

echo '--- TASK start: model=kab seed=44'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/kab.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/kab.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'kab' --type 'single' --lang_src 'kab' --lang_tgt '' --seed 44 --epochs 1 --batch_size 128
echo '--- TASK end: model=kab seed=44'

echo '--- TASK start: model=kab seed=45'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/kab.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/kab.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'kab' --type 'single' --lang_src 'kab' --lang_tgt '' --seed 45 --epochs 1 --batch_size 128
echo '--- TASK end: model=kab seed=45'

echo '--- TASK start: model=kab seed=46'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/kab.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/kab.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'kab' --type 'single' --lang_src 'kab' --lang_tgt '' --seed 46 --epochs 1 --batch_size 128
echo '--- TASK end: model=kab seed=46'

echo '--- TASK start: model=kab seed=47'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/kab.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/kab.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'kab' --type 'single' --lang_src 'kab' --lang_tgt '' --seed 47 --epochs 1 --batch_size 128
echo '--- TASK end: model=kab seed=47'

echo '--- TASK start: model=kab seed=48'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/kab.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/kab.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'kab' --type 'single' --lang_src 'kab' --lang_tgt '' --seed 48 --epochs 1 --batch_size 128
echo '--- TASK end: model=kab seed=48'

echo '--- TASK start: model=kab seed=49'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/kab.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/kab.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'kab' --type 'single' --lang_src 'kab' --lang_tgt '' --seed 49 --epochs 1 --batch_size 128
echo '--- TASK end: model=kab seed=49'

echo '--- TASK start: model=kab seed=50'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/kab.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/kab.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'kab' --type 'single' --lang_src 'kab' --lang_tgt '' --seed 50 --epochs 1 --batch_size 128
echo '--- TASK end: model=kab seed=50'

echo '--- TASK start: model=kab seed=51'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/kab.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/kab.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'kab' --type 'single' --lang_src 'kab' --lang_tgt '' --seed 51 --epochs 1 --batch_size 128
echo '--- TASK end: model=kab seed=51'

echo '--- TASK start: model=kmr seed=42'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/kmr.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/kmr.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'kmr' --type 'single' --lang_src 'kmr' --lang_tgt '' --seed 42 --epochs 1 --batch_size 128
echo '--- TASK end: model=kmr seed=42'

echo '--- TASK start: model=kmr seed=43'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/kmr.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/kmr.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'kmr' --type 'single' --lang_src 'kmr' --lang_tgt '' --seed 43 --epochs 1 --batch_size 128
echo '--- TASK end: model=kmr seed=43'

echo '--- TASK start: model=kmr seed=44'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/kmr.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/kmr.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'kmr' --type 'single' --lang_src 'kmr' --lang_tgt '' --seed 44 --epochs 1 --batch_size 128
echo '--- TASK end: model=kmr seed=44'

echo '--- TASK start: model=kmr seed=45'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/kmr.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/kmr.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'kmr' --type 'single' --lang_src 'kmr' --lang_tgt '' --seed 45 --epochs 1 --batch_size 128
echo '--- TASK end: model=kmr seed=45'

echo '--- TASK start: model=kmr seed=46'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/kmr.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/kmr.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'kmr' --type 'single' --lang_src 'kmr' --lang_tgt '' --seed 46 --epochs 1 --batch_size 128
echo '--- TASK end: model=kmr seed=46'

echo '--- TASK start: model=kmr seed=47'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/kmr.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/kmr.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'kmr' --type 'single' --lang_src 'kmr' --lang_tgt '' --seed 47 --epochs 1 --batch_size 128
echo '--- TASK end: model=kmr seed=47'

echo '--- TASK start: model=kmr seed=48'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/kmr.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/kmr.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'kmr' --type 'single' --lang_src 'kmr' --lang_tgt '' --seed 48 --epochs 1 --batch_size 128
echo '--- TASK end: model=kmr seed=48'

echo '--- TASK start: model=kmr seed=49'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/kmr.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/kmr.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'kmr' --type 'single' --lang_src 'kmr' --lang_tgt '' --seed 49 --epochs 1 --batch_size 128
echo '--- TASK end: model=kmr seed=49'

echo '--- TASK start: model=kmr seed=50'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/kmr.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/kmr.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'kmr' --type 'single' --lang_src 'kmr' --lang_tgt '' --seed 50 --epochs 1 --batch_size 128
echo '--- TASK end: model=kmr seed=50'

echo '--- TASK start: model=kmr seed=51'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/kmr.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/kmr.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'kmr' --type 'single' --lang_src 'kmr' --lang_tgt '' --seed 51 --epochs 1 --batch_size 128
echo '--- TASK end: model=kmr seed=51'

echo '--- TASK start: model=ky seed=42'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/ky.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/ky.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'ky' --type 'single' --lang_src 'ky' --lang_tgt '' --seed 42 --epochs 1 --batch_size 128
echo '--- TASK end: model=ky seed=42'

echo '--- TASK start: model=ky seed=43'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/ky.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/ky.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'ky' --type 'single' --lang_src 'ky' --lang_tgt '' --seed 43 --epochs 1 --batch_size 128
echo '--- TASK end: model=ky seed=43'

echo '--- TASK start: model=ky seed=44'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/ky.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/ky.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'ky' --type 'single' --lang_src 'ky' --lang_tgt '' --seed 44 --epochs 1 --batch_size 128
echo '--- TASK end: model=ky seed=44'

echo '--- TASK start: model=ky seed=45'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/ky.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/ky.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'ky' --type 'single' --lang_src 'ky' --lang_tgt '' --seed 45 --epochs 1 --batch_size 128
echo '--- TASK end: model=ky seed=45'

echo '--- TASK start: model=ky seed=46'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/ky.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/ky.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'ky' --type 'single' --lang_src 'ky' --lang_tgt '' --seed 46 --epochs 1 --batch_size 128
echo '--- TASK end: model=ky seed=46'

echo '--- TASK start: model=ky seed=47'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/ky.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/ky.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'ky' --type 'single' --lang_src 'ky' --lang_tgt '' --seed 47 --epochs 1 --batch_size 128
echo '--- TASK end: model=ky seed=47'

echo '--- TASK start: model=ky seed=48'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/ky.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/ky.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'ky' --type 'single' --lang_src 'ky' --lang_tgt '' --seed 48 --epochs 1 --batch_size 128
echo '--- TASK end: model=ky seed=48'

echo '--- TASK start: model=ky seed=49'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/ky.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/ky.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'ky' --type 'single' --lang_src 'ky' --lang_tgt '' --seed 49 --epochs 1 --batch_size 128
echo '--- TASK end: model=ky seed=49'

echo '--- TASK start: model=ky seed=50'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/ky.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/ky.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'ky' --type 'single' --lang_src 'ky' --lang_tgt '' --seed 50 --epochs 1 --batch_size 128
echo '--- TASK end: model=ky seed=50'

echo '--- TASK start: model=ky seed=51'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/ky.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/ky.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'ky' --type 'single' --lang_src 'ky' --lang_tgt '' --seed 51 --epochs 1 --batch_size 128
echo '--- TASK end: model=ky seed=51'

echo '--- TASK start: model=lg seed=42'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/lg.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/lg.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'lg' --type 'single' --lang_src 'lg' --lang_tgt '' --seed 42 --epochs 1 --batch_size 128
echo '--- TASK end: model=lg seed=42'

echo '--- TASK start: model=lg seed=43'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/lg.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/lg.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'lg' --type 'single' --lang_src 'lg' --lang_tgt '' --seed 43 --epochs 1 --batch_size 128
echo '--- TASK end: model=lg seed=43'

echo '--- TASK start: model=lg seed=44'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/lg.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/lg.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'lg' --type 'single' --lang_src 'lg' --lang_tgt '' --seed 44 --epochs 1 --batch_size 128
echo '--- TASK end: model=lg seed=44'

echo '--- TASK start: model=lg seed=45'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/lg.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/lg.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'lg' --type 'single' --lang_src 'lg' --lang_tgt '' --seed 45 --epochs 1 --batch_size 128
echo '--- TASK end: model=lg seed=45'

echo '--- TASK start: model=lg seed=46'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/lg.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/lg.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'lg' --type 'single' --lang_src 'lg' --lang_tgt '' --seed 46 --epochs 1 --batch_size 128
echo '--- TASK end: model=lg seed=46'

echo '--- TASK start: model=lg seed=47'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/lg.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/lg.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'lg' --type 'single' --lang_src 'lg' --lang_tgt '' --seed 47 --epochs 1 --batch_size 128
echo '--- TASK end: model=lg seed=47'

echo '--- TASK start: model=lg seed=48'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/lg.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/lg.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'lg' --type 'single' --lang_src 'lg' --lang_tgt '' --seed 48 --epochs 1 --batch_size 128
echo '--- TASK end: model=lg seed=48'

echo '--- TASK start: model=lg seed=49'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/lg.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/lg.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'lg' --type 'single' --lang_src 'lg' --lang_tgt '' --seed 49 --epochs 1 --batch_size 128
echo '--- TASK end: model=lg seed=49'

echo '--- TASK start: model=lg seed=50'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/lg.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/lg.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'lg' --type 'single' --lang_src 'lg' --lang_tgt '' --seed 50 --epochs 1 --batch_size 128
echo '--- TASK end: model=lg seed=50'

echo '--- TASK start: model=lg seed=51'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/lg.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/lg.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'lg' --type 'single' --lang_src 'lg' --lang_tgt '' --seed 51 --epochs 1 --batch_size 128
echo '--- TASK end: model=lg seed=51'

echo '--- TASK start: model=mhr seed=42'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/mhr.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/mhr.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'mhr' --type 'single' --lang_src 'mhr' --lang_tgt '' --seed 42 --epochs 1 --batch_size 128
echo '--- TASK end: model=mhr seed=42'

echo '--- TASK start: model=mhr seed=43'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/mhr.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/mhr.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'mhr' --type 'single' --lang_src 'mhr' --lang_tgt '' --seed 43 --epochs 1 --batch_size 128
echo '--- TASK end: model=mhr seed=43'

echo '--- TASK start: model=mhr seed=44'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/mhr.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/mhr.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'mhr' --type 'single' --lang_src 'mhr' --lang_tgt '' --seed 44 --epochs 1 --batch_size 128
echo '--- TASK end: model=mhr seed=44'

echo '--- TASK start: model=mhr seed=45'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/mhr.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/mhr.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'mhr' --type 'single' --lang_src 'mhr' --lang_tgt '' --seed 45 --epochs 1 --batch_size 128
echo '--- TASK end: model=mhr seed=45'

echo '--- TASK start: model=mhr seed=46'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/mhr.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/mhr.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'mhr' --type 'single' --lang_src 'mhr' --lang_tgt '' --seed 46 --epochs 1 --batch_size 128
echo '--- TASK end: model=mhr seed=46'

echo '--- TASK start: model=mhr seed=47'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/mhr.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/mhr.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'mhr' --type 'single' --lang_src 'mhr' --lang_tgt '' --seed 47 --epochs 1 --batch_size 128
echo '--- TASK end: model=mhr seed=47'

echo '--- TASK start: model=mhr seed=48'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/mhr.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/mhr.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'mhr' --type 'single' --lang_src 'mhr' --lang_tgt '' --seed 48 --epochs 1 --batch_size 128
echo '--- TASK end: model=mhr seed=48'

echo '--- TASK start: model=mhr seed=49'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/mhr.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/mhr.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'mhr' --type 'single' --lang_src 'mhr' --lang_tgt '' --seed 49 --epochs 1 --batch_size 128
echo '--- TASK end: model=mhr seed=49'

echo '--- TASK start: model=mhr seed=50'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/mhr.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/mhr.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'mhr' --type 'single' --lang_src 'mhr' --lang_tgt '' --seed 50 --epochs 1 --batch_size 128
echo '--- TASK end: model=mhr seed=50'

echo '--- TASK start: model=mhr seed=51'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/mhr.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/mhr.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'mhr' --type 'single' --lang_src 'mhr' --lang_tgt '' --seed 51 --epochs 1 --batch_size 128
echo '--- TASK end: model=mhr seed=51'

echo '--- TASK start: model=nl seed=42'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/nl.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/nl.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'nl' --type 'single' --lang_src 'nl' --lang_tgt '' --seed 42 --epochs 1 --batch_size 128
echo '--- TASK end: model=nl seed=42'

echo '--- TASK start: model=nl seed=43'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/nl.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/nl.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'nl' --type 'single' --lang_src 'nl' --lang_tgt '' --seed 43 --epochs 1 --batch_size 128
echo '--- TASK end: model=nl seed=43'

echo '--- TASK start: model=nl seed=44'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/nl.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/nl.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'nl' --type 'single' --lang_src 'nl' --lang_tgt '' --seed 44 --epochs 1 --batch_size 128
echo '--- TASK end: model=nl seed=44'

echo '--- TASK start: model=nl seed=45'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/nl.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/nl.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'nl' --type 'single' --lang_src 'nl' --lang_tgt '' --seed 45 --epochs 1 --batch_size 128
echo '--- TASK end: model=nl seed=45'

echo '--- TASK start: model=nl seed=46'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/nl.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/nl.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'nl' --type 'single' --lang_src 'nl' --lang_tgt '' --seed 46 --epochs 1 --batch_size 128
echo '--- TASK end: model=nl seed=46'

echo '--- TASK start: model=nl seed=47'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/nl.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/nl.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'nl' --type 'single' --lang_src 'nl' --lang_tgt '' --seed 47 --epochs 1 --batch_size 128
echo '--- TASK end: model=nl seed=47'

echo '--- TASK start: model=nl seed=48'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/nl.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/nl.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'nl' --type 'single' --lang_src 'nl' --lang_tgt '' --seed 48 --epochs 1 --batch_size 128
echo '--- TASK end: model=nl seed=48'

echo '--- TASK start: model=nl seed=49'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/nl.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/nl.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'nl' --type 'single' --lang_src 'nl' --lang_tgt '' --seed 49 --epochs 1 --batch_size 128
echo '--- TASK end: model=nl seed=49'

echo '--- TASK start: model=nl seed=50'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/nl.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/nl.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'nl' --type 'single' --lang_src 'nl' --lang_tgt '' --seed 50 --epochs 1 --batch_size 128
echo '--- TASK end: model=nl seed=50'

echo '--- TASK start: model=nl seed=51'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/nl.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/nl.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'nl' --type 'single' --lang_src 'nl' --lang_tgt '' --seed 51 --epochs 1 --batch_size 128
echo '--- TASK end: model=nl seed=51'

echo '--- TASK start: model=pl seed=42'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/pl.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/pl.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'pl' --type 'single' --lang_src 'pl' --lang_tgt '' --seed 42 --epochs 1 --batch_size 128
echo '--- TASK end: model=pl seed=42'

echo '--- TASK start: model=pl seed=43'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/pl.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/pl.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'pl' --type 'single' --lang_src 'pl' --lang_tgt '' --seed 43 --epochs 1 --batch_size 128
echo '--- TASK end: model=pl seed=43'

echo '--- TASK start: model=pl seed=44'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/pl.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/pl.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'pl' --type 'single' --lang_src 'pl' --lang_tgt '' --seed 44 --epochs 1 --batch_size 128
echo '--- TASK end: model=pl seed=44'

echo '--- TASK start: model=pl seed=45'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/pl.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/pl.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'pl' --type 'single' --lang_src 'pl' --lang_tgt '' --seed 45 --epochs 1 --batch_size 128
echo '--- TASK end: model=pl seed=45'

echo '--- TASK start: model=pl seed=46'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/pl.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/pl.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'pl' --type 'single' --lang_src 'pl' --lang_tgt '' --seed 46 --epochs 1 --batch_size 128
echo '--- TASK end: model=pl seed=46'

echo '--- TASK start: model=pl seed=47'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/pl.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/pl.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'pl' --type 'single' --lang_src 'pl' --lang_tgt '' --seed 47 --epochs 1 --batch_size 128
echo '--- TASK end: model=pl seed=47'

echo '--- TASK start: model=pl seed=48'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/pl.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/pl.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'pl' --type 'single' --lang_src 'pl' --lang_tgt '' --seed 48 --epochs 1 --batch_size 128
echo '--- TASK end: model=pl seed=48'

echo '--- TASK start: model=pl seed=49'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/pl.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/pl.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'pl' --type 'single' --lang_src 'pl' --lang_tgt '' --seed 49 --epochs 1 --batch_size 128
echo '--- TASK end: model=pl seed=49'

echo '--- TASK start: model=pl seed=50'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/pl.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/pl.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'pl' --type 'single' --lang_src 'pl' --lang_tgt '' --seed 50 --epochs 1 --batch_size 128
echo '--- TASK end: model=pl seed=50'

echo '--- TASK start: model=pl seed=51'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/pl.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/pl.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'pl' --type 'single' --lang_src 'pl' --lang_tgt '' --seed 51 --epochs 1 --batch_size 128
echo '--- TASK end: model=pl seed=51'

echo '--- TASK start: model=pt seed=42'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/pt.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/pt.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'pt' --type 'single' --lang_src 'pt' --lang_tgt '' --seed 42 --epochs 1 --batch_size 128
echo '--- TASK end: model=pt seed=42'

echo '--- TASK start: model=pt seed=43'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/pt.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/pt.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'pt' --type 'single' --lang_src 'pt' --lang_tgt '' --seed 43 --epochs 1 --batch_size 128
echo '--- TASK end: model=pt seed=43'

echo '--- TASK start: model=pt seed=44'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/pt.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/pt.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'pt' --type 'single' --lang_src 'pt' --lang_tgt '' --seed 44 --epochs 1 --batch_size 128
echo '--- TASK end: model=pt seed=44'

echo '--- TASK start: model=pt seed=45'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/pt.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/pt.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'pt' --type 'single' --lang_src 'pt' --lang_tgt '' --seed 45 --epochs 1 --batch_size 128
echo '--- TASK end: model=pt seed=45'

echo '--- TASK start: model=pt seed=46'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/pt.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/pt.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'pt' --type 'single' --lang_src 'pt' --lang_tgt '' --seed 46 --epochs 1 --batch_size 128
echo '--- TASK end: model=pt seed=46'

echo '--- TASK start: model=pt seed=47'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/pt.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/pt.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'pt' --type 'single' --lang_src 'pt' --lang_tgt '' --seed 47 --epochs 1 --batch_size 128
echo '--- TASK end: model=pt seed=47'

echo '--- TASK start: model=pt seed=48'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/pt.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/pt.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'pt' --type 'single' --lang_src 'pt' --lang_tgt '' --seed 48 --epochs 1 --batch_size 128
echo '--- TASK end: model=pt seed=48'

echo '--- TASK start: model=pt seed=49'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/pt.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/pt.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'pt' --type 'single' --lang_src 'pt' --lang_tgt '' --seed 49 --epochs 1 --batch_size 128
echo '--- TASK end: model=pt seed=49'

echo '--- TASK start: model=pt seed=50'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/pt.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/pt.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'pt' --type 'single' --lang_src 'pt' --lang_tgt '' --seed 50 --epochs 1 --batch_size 128
echo '--- TASK end: model=pt seed=50'

echo '--- TASK start: model=pt seed=51'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/pt.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/pt.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'pt' --type 'single' --lang_src 'pt' --lang_tgt '' --seed 51 --epochs 1 --batch_size 128
echo '--- TASK end: model=pt seed=51'

echo '--- TASK start: model=ro seed=42'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/ro.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/ro.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'ro' --type 'single' --lang_src 'ro' --lang_tgt '' --seed 42 --epochs 1 --batch_size 128
echo '--- TASK end: model=ro seed=42'

echo '--- TASK start: model=ro seed=43'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/ro.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/ro.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'ro' --type 'single' --lang_src 'ro' --lang_tgt '' --seed 43 --epochs 1 --batch_size 128
echo '--- TASK end: model=ro seed=43'

echo '--- TASK start: model=ro seed=44'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/ro.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/ro.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'ro' --type 'single' --lang_src 'ro' --lang_tgt '' --seed 44 --epochs 1 --batch_size 128
echo '--- TASK end: model=ro seed=44'

echo '--- TASK start: model=ro seed=45'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/ro.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/ro.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'ro' --type 'single' --lang_src 'ro' --lang_tgt '' --seed 45 --epochs 1 --batch_size 128
echo '--- TASK end: model=ro seed=45'

