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

echo '--- TASK start: model=zh-TW_zh-CN seed=46'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-TW_zh-CN.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-TW.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_bilingual.csv' --model_id 'zh-TW_zh-CN' --type 'dual' --lang_src 'zh-TW' --lang_tgt 'zh-CN' --seed 46 --epochs 1 --batch_size 128
echo '--- TASK end: model=zh-TW_zh-CN seed=46'

echo '--- TASK start: model=zh-TW_zh-CN seed=47'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-TW_zh-CN.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-TW.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_bilingual.csv' --model_id 'zh-TW_zh-CN' --type 'dual' --lang_src 'zh-TW' --lang_tgt 'zh-CN' --seed 47 --epochs 1 --batch_size 128
echo '--- TASK end: model=zh-TW_zh-CN seed=47'

echo '--- TASK start: model=zh-TW_zh-CN seed=48'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-TW_zh-CN.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-TW.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_bilingual.csv' --model_id 'zh-TW_zh-CN' --type 'dual' --lang_src 'zh-TW' --lang_tgt 'zh-CN' --seed 48 --epochs 1 --batch_size 128
echo '--- TASK end: model=zh-TW_zh-CN seed=48'

echo '--- TASK start: model=zh-TW_zh-CN seed=49'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-TW_zh-CN.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-TW.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_bilingual.csv' --model_id 'zh-TW_zh-CN' --type 'dual' --lang_src 'zh-TW' --lang_tgt 'zh-CN' --seed 49 --epochs 1 --batch_size 128
echo '--- TASK end: model=zh-TW_zh-CN seed=49'

echo '--- TASK start: model=zh-TW_zh-CN seed=50'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-TW_zh-CN.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-TW.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_bilingual.csv' --model_id 'zh-TW_zh-CN' --type 'dual' --lang_src 'zh-TW' --lang_tgt 'zh-CN' --seed 50 --epochs 1 --batch_size 128
echo '--- TASK end: model=zh-TW_zh-CN seed=50'

echo '--- TASK start: model=zh-TW_zh-CN seed=51'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-TW_zh-CN.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-TW.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_bilingual.csv' --model_id 'zh-TW_zh-CN' --type 'dual' --lang_src 'zh-TW' --lang_tgt 'zh-CN' --seed 51 --epochs 1 --batch_size 128
echo '--- TASK end: model=zh-TW_zh-CN seed=51'

echo '--- TASK start: model=zh-TW_zh-HK seed=42'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-TW_zh-HK.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-TW.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_bilingual.csv' --model_id 'zh-TW_zh-HK' --type 'dual' --lang_src 'zh-TW' --lang_tgt 'zh-HK' --seed 42 --epochs 1 --batch_size 128
echo '--- TASK end: model=zh-TW_zh-HK seed=42'

echo '--- TASK start: model=zh-TW_zh-HK seed=43'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-TW_zh-HK.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-TW.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_bilingual.csv' --model_id 'zh-TW_zh-HK' --type 'dual' --lang_src 'zh-TW' --lang_tgt 'zh-HK' --seed 43 --epochs 1 --batch_size 128
echo '--- TASK end: model=zh-TW_zh-HK seed=43'

echo '--- TASK start: model=zh-TW_zh-HK seed=44'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-TW_zh-HK.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-TW.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_bilingual.csv' --model_id 'zh-TW_zh-HK' --type 'dual' --lang_src 'zh-TW' --lang_tgt 'zh-HK' --seed 44 --epochs 1 --batch_size 128
echo '--- TASK end: model=zh-TW_zh-HK seed=44'

echo '--- TASK start: model=zh-TW_zh-HK seed=45'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-TW_zh-HK.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-TW.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_bilingual.csv' --model_id 'zh-TW_zh-HK' --type 'dual' --lang_src 'zh-TW' --lang_tgt 'zh-HK' --seed 45 --epochs 1 --batch_size 128
echo '--- TASK end: model=zh-TW_zh-HK seed=45'

echo '--- TASK start: model=zh-TW_zh-HK seed=46'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-TW_zh-HK.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-TW.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_bilingual.csv' --model_id 'zh-TW_zh-HK' --type 'dual' --lang_src 'zh-TW' --lang_tgt 'zh-HK' --seed 46 --epochs 1 --batch_size 128
echo '--- TASK end: model=zh-TW_zh-HK seed=46'

echo '--- TASK start: model=zh-TW_zh-HK seed=47'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-TW_zh-HK.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-TW.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_bilingual.csv' --model_id 'zh-TW_zh-HK' --type 'dual' --lang_src 'zh-TW' --lang_tgt 'zh-HK' --seed 47 --epochs 1 --batch_size 128
echo '--- TASK end: model=zh-TW_zh-HK seed=47'

echo '--- TASK start: model=zh-TW_zh-HK seed=48'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-TW_zh-HK.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-TW.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_bilingual.csv' --model_id 'zh-TW_zh-HK' --type 'dual' --lang_src 'zh-TW' --lang_tgt 'zh-HK' --seed 48 --epochs 1 --batch_size 128
echo '--- TASK end: model=zh-TW_zh-HK seed=48'

echo '--- TASK start: model=zh-TW_zh-HK seed=49'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-TW_zh-HK.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-TW.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_bilingual.csv' --model_id 'zh-TW_zh-HK' --type 'dual' --lang_src 'zh-TW' --lang_tgt 'zh-HK' --seed 49 --epochs 1 --batch_size 128
echo '--- TASK end: model=zh-TW_zh-HK seed=49'

echo '--- TASK start: model=zh-TW_zh-HK seed=50'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-TW_zh-HK.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-TW.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_bilingual.csv' --model_id 'zh-TW_zh-HK' --type 'dual' --lang_src 'zh-TW' --lang_tgt 'zh-HK' --seed 50 --epochs 1 --batch_size 128
echo '--- TASK end: model=zh-TW_zh-HK seed=50'

echo '--- TASK start: model=zh-TW_zh-HK seed=51'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-TW_zh-HK.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv/zh-TW.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_bilingual.csv' --model_id 'zh-TW_zh-HK' --type 'dual' --lang_src 'zh-TW' --lang_tgt 'zh-HK' --seed 51 --epochs 1 --batch_size 128
echo '--- TASK end: model=zh-TW_zh-HK seed=51'

echo '--- TASK start: model=ab seed=42'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/ab.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/ab.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'ab' --type 'single' --lang_src 'ab' --lang_tgt '' --seed 42 --epochs 1 --batch_size 128
echo '--- TASK end: model=ab seed=42'

echo '--- TASK start: model=ab seed=43'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/ab.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/ab.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'ab' --type 'single' --lang_src 'ab' --lang_tgt '' --seed 43 --epochs 1 --batch_size 128
echo '--- TASK end: model=ab seed=43'

echo '--- TASK start: model=ab seed=44'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/ab.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/ab.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'ab' --type 'single' --lang_src 'ab' --lang_tgt '' --seed 44 --epochs 1 --batch_size 128
echo '--- TASK end: model=ab seed=44'

echo '--- TASK start: model=ab seed=45'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/ab.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/ab.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'ab' --type 'single' --lang_src 'ab' --lang_tgt '' --seed 45 --epochs 1 --batch_size 128
echo '--- TASK end: model=ab seed=45'

echo '--- TASK start: model=ab seed=46'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/ab.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/ab.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'ab' --type 'single' --lang_src 'ab' --lang_tgt '' --seed 46 --epochs 1 --batch_size 128
echo '--- TASK end: model=ab seed=46'

echo '--- TASK start: model=ab seed=47'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/ab.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/ab.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'ab' --type 'single' --lang_src 'ab' --lang_tgt '' --seed 47 --epochs 1 --batch_size 128
echo '--- TASK end: model=ab seed=47'

echo '--- TASK start: model=ab seed=48'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/ab.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/ab.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'ab' --type 'single' --lang_src 'ab' --lang_tgt '' --seed 48 --epochs 1 --batch_size 128
echo '--- TASK end: model=ab seed=48'

echo '--- TASK start: model=ab seed=49'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/ab.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/ab.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'ab' --type 'single' --lang_src 'ab' --lang_tgt '' --seed 49 --epochs 1 --batch_size 128
echo '--- TASK end: model=ab seed=49'

echo '--- TASK start: model=ab seed=50'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/ab.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/ab.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'ab' --type 'single' --lang_src 'ab' --lang_tgt '' --seed 50 --epochs 1 --batch_size 128
echo '--- TASK end: model=ab seed=50'

echo '--- TASK start: model=ab seed=51'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/ab.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/ab.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'ab' --type 'single' --lang_src 'ab' --lang_tgt '' --seed 51 --epochs 1 --batch_size 128
echo '--- TASK end: model=ab seed=51'

echo '--- TASK start: model=ar seed=42'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/ar.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/ar.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'ar' --type 'single' --lang_src 'ar' --lang_tgt '' --seed 42 --epochs 1 --batch_size 128
echo '--- TASK end: model=ar seed=42'

echo '--- TASK start: model=ar seed=43'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/ar.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/ar.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'ar' --type 'single' --lang_src 'ar' --lang_tgt '' --seed 43 --epochs 1 --batch_size 128
echo '--- TASK end: model=ar seed=43'

echo '--- TASK start: model=ar seed=44'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/ar.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/ar.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'ar' --type 'single' --lang_src 'ar' --lang_tgt '' --seed 44 --epochs 1 --batch_size 128
echo '--- TASK end: model=ar seed=44'

echo '--- TASK start: model=ar seed=45'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/ar.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/ar.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'ar' --type 'single' --lang_src 'ar' --lang_tgt '' --seed 45 --epochs 1 --batch_size 128
echo '--- TASK end: model=ar seed=45'

echo '--- TASK start: model=ar seed=46'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/ar.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/ar.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'ar' --type 'single' --lang_src 'ar' --lang_tgt '' --seed 46 --epochs 1 --batch_size 128
echo '--- TASK end: model=ar seed=46'

echo '--- TASK start: model=ar seed=47'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/ar.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/ar.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'ar' --type 'single' --lang_src 'ar' --lang_tgt '' --seed 47 --epochs 1 --batch_size 128
echo '--- TASK end: model=ar seed=47'

echo '--- TASK start: model=ar seed=48'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/ar.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/ar.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'ar' --type 'single' --lang_src 'ar' --lang_tgt '' --seed 48 --epochs 1 --batch_size 128
echo '--- TASK end: model=ar seed=48'

echo '--- TASK start: model=ar seed=49'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/ar.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/ar.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'ar' --type 'single' --lang_src 'ar' --lang_tgt '' --seed 49 --epochs 1 --batch_size 128
echo '--- TASK end: model=ar seed=49'

echo '--- TASK start: model=ar seed=50'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/ar.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/ar.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'ar' --type 'single' --lang_src 'ar' --lang_tgt '' --seed 50 --epochs 1 --batch_size 128
echo '--- TASK end: model=ar seed=50'

echo '--- TASK start: model=ar seed=51'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/ar.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/ar.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'ar' --type 'single' --lang_src 'ar' --lang_tgt '' --seed 51 --epochs 1 --batch_size 128
echo '--- TASK end: model=ar seed=51'

echo '--- TASK start: model=be seed=42'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/be.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/be.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'be' --type 'single' --lang_src 'be' --lang_tgt '' --seed 42 --epochs 1 --batch_size 128
echo '--- TASK end: model=be seed=42'

echo '--- TASK start: model=be seed=43'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/be.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/be.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'be' --type 'single' --lang_src 'be' --lang_tgt '' --seed 43 --epochs 1 --batch_size 128
echo '--- TASK end: model=be seed=43'

echo '--- TASK start: model=be seed=44'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/be.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/be.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'be' --type 'single' --lang_src 'be' --lang_tgt '' --seed 44 --epochs 1 --batch_size 128
echo '--- TASK end: model=be seed=44'

echo '--- TASK start: model=be seed=45'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/be.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/be.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'be' --type 'single' --lang_src 'be' --lang_tgt '' --seed 45 --epochs 1 --batch_size 128
echo '--- TASK end: model=be seed=45'

echo '--- TASK start: model=be seed=46'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/be.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/be.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'be' --type 'single' --lang_src 'be' --lang_tgt '' --seed 46 --epochs 1 --batch_size 128
echo '--- TASK end: model=be seed=46'

echo '--- TASK start: model=be seed=47'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/be.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/be.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'be' --type 'single' --lang_src 'be' --lang_tgt '' --seed 47 --epochs 1 --batch_size 128
echo '--- TASK end: model=be seed=47'

echo '--- TASK start: model=be seed=48'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/be.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/be.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'be' --type 'single' --lang_src 'be' --lang_tgt '' --seed 48 --epochs 1 --batch_size 128
echo '--- TASK end: model=be seed=48'

echo '--- TASK start: model=be seed=49'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/be.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/be.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'be' --type 'single' --lang_src 'be' --lang_tgt '' --seed 49 --epochs 1 --batch_size 128
echo '--- TASK end: model=be seed=49'

echo '--- TASK start: model=be seed=50'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/be.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/be.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'be' --type 'single' --lang_src 'be' --lang_tgt '' --seed 50 --epochs 1 --batch_size 128
echo '--- TASK end: model=be seed=50'

echo '--- TASK start: model=be seed=51'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/be.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/be.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'be' --type 'single' --lang_src 'be' --lang_tgt '' --seed 51 --epochs 1 --batch_size 128
echo '--- TASK end: model=be seed=51'

echo '--- TASK start: model=bn seed=42'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/bn.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/bn.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'bn' --type 'single' --lang_src 'bn' --lang_tgt '' --seed 42 --epochs 1 --batch_size 128
echo '--- TASK end: model=bn seed=42'

echo '--- TASK start: model=bn seed=43'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/bn.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/bn.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'bn' --type 'single' --lang_src 'bn' --lang_tgt '' --seed 43 --epochs 1 --batch_size 128
echo '--- TASK end: model=bn seed=43'

echo '--- TASK start: model=bn seed=44'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/bn.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/bn.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'bn' --type 'single' --lang_src 'bn' --lang_tgt '' --seed 44 --epochs 1 --batch_size 128
echo '--- TASK end: model=bn seed=44'

echo '--- TASK start: model=bn seed=45'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/bn.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/bn.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'bn' --type 'single' --lang_src 'bn' --lang_tgt '' --seed 45 --epochs 1 --batch_size 128
echo '--- TASK end: model=bn seed=45'

echo '--- TASK start: model=bn seed=46'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/bn.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/bn.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'bn' --type 'single' --lang_src 'bn' --lang_tgt '' --seed 46 --epochs 1 --batch_size 128
echo '--- TASK end: model=bn seed=46'

echo '--- TASK start: model=bn seed=47'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/bn.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/bn.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'bn' --type 'single' --lang_src 'bn' --lang_tgt '' --seed 47 --epochs 1 --batch_size 128
echo '--- TASK end: model=bn seed=47'

echo '--- TASK start: model=bn seed=48'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/bn.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/bn.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'bn' --type 'single' --lang_src 'bn' --lang_tgt '' --seed 48 --epochs 1 --batch_size 128
echo '--- TASK end: model=bn seed=48'

echo '--- TASK start: model=bn seed=49'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/bn.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/bn.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'bn' --type 'single' --lang_src 'bn' --lang_tgt '' --seed 49 --epochs 1 --batch_size 128
echo '--- TASK end: model=bn seed=49'

echo '--- TASK start: model=bn seed=50'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/bn.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/bn.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'bn' --type 'single' --lang_src 'bn' --lang_tgt '' --seed 50 --epochs 1 --batch_size 128
echo '--- TASK end: model=bn seed=50'

echo '--- TASK start: model=bn seed=51'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/bn.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/bn.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'bn' --type 'single' --lang_src 'bn' --lang_tgt '' --seed 51 --epochs 1 --batch_size 128
echo '--- TASK end: model=bn seed=51'

echo '--- TASK start: model=ca seed=42'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/ca.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/ca.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'ca' --type 'single' --lang_src 'ca' --lang_tgt '' --seed 42 --epochs 1 --batch_size 128
echo '--- TASK end: model=ca seed=42'

echo '--- TASK start: model=ca seed=43'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/ca.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/ca.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'ca' --type 'single' --lang_src 'ca' --lang_tgt '' --seed 43 --epochs 1 --batch_size 128
echo '--- TASK end: model=ca seed=43'

echo '--- TASK start: model=ca seed=44'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/ca.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/ca.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'ca' --type 'single' --lang_src 'ca' --lang_tgt '' --seed 44 --epochs 1 --batch_size 128
echo '--- TASK end: model=ca seed=44'

echo '--- TASK start: model=ca seed=45'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/ca.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/ca.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'ca' --type 'single' --lang_src 'ca' --lang_tgt '' --seed 45 --epochs 1 --batch_size 128
echo '--- TASK end: model=ca seed=45'

echo '--- TASK start: model=ca seed=46'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/ca.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/ca.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'ca' --type 'single' --lang_src 'ca' --lang_tgt '' --seed 46 --epochs 1 --batch_size 128
echo '--- TASK end: model=ca seed=46'

echo '--- TASK start: model=ca seed=47'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/ca.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/ca.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'ca' --type 'single' --lang_src 'ca' --lang_tgt '' --seed 47 --epochs 1 --batch_size 128
echo '--- TASK end: model=ca seed=47'

echo '--- TASK start: model=ca seed=48'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/ca.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/ca.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'ca' --type 'single' --lang_src 'ca' --lang_tgt '' --seed 48 --epochs 1 --batch_size 128
echo '--- TASK end: model=ca seed=48'

echo '--- TASK start: model=ca seed=49'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/ca.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/ca.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'ca' --type 'single' --lang_src 'ca' --lang_tgt '' --seed 49 --epochs 1 --batch_size 128
echo '--- TASK end: model=ca seed=49'

echo '--- TASK start: model=ca seed=50'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/ca.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/ca.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'ca' --type 'single' --lang_src 'ca' --lang_tgt '' --seed 50 --epochs 1 --batch_size 128
echo '--- TASK end: model=ca seed=50'

echo '--- TASK start: model=ca seed=51'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/ca.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/ca.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'ca' --type 'single' --lang_src 'ca' --lang_tgt '' --seed 51 --epochs 1 --batch_size 128
echo '--- TASK end: model=ca seed=51'

echo '--- TASK start: model=ckb seed=42'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/ckb.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/ckb.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'ckb' --type 'single' --lang_src 'ckb' --lang_tgt '' --seed 42 --epochs 1 --batch_size 128
echo '--- TASK end: model=ckb seed=42'

echo '--- TASK start: model=ckb seed=43'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/ckb.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/ckb.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'ckb' --type 'single' --lang_src 'ckb' --lang_tgt '' --seed 43 --epochs 1 --batch_size 128
echo '--- TASK end: model=ckb seed=43'

echo '--- TASK start: model=ckb seed=44'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/ckb.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/ckb.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'ckb' --type 'single' --lang_src 'ckb' --lang_tgt '' --seed 44 --epochs 1 --batch_size 128
echo '--- TASK end: model=ckb seed=44'

echo '--- TASK start: model=ckb seed=45'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/ckb.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/ckb.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'ckb' --type 'single' --lang_src 'ckb' --lang_tgt '' --seed 45 --epochs 1 --batch_size 128
echo '--- TASK end: model=ckb seed=45'

echo '--- TASK start: model=ckb seed=46'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/ckb.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/ckb.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'ckb' --type 'single' --lang_src 'ckb' --lang_tgt '' --seed 46 --epochs 1 --batch_size 128
echo '--- TASK end: model=ckb seed=46'

echo '--- TASK start: model=ckb seed=47'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/ckb.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/ckb.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'ckb' --type 'single' --lang_src 'ckb' --lang_tgt '' --seed 47 --epochs 1 --batch_size 128
echo '--- TASK end: model=ckb seed=47'

echo '--- TASK start: model=ckb seed=48'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/ckb.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/ckb.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'ckb' --type 'single' --lang_src 'ckb' --lang_tgt '' --seed 48 --epochs 1 --batch_size 128
echo '--- TASK end: model=ckb seed=48'

echo '--- TASK start: model=ckb seed=49'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/ckb.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/ckb.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'ckb' --type 'single' --lang_src 'ckb' --lang_tgt '' --seed 49 --epochs 1 --batch_size 128
echo '--- TASK end: model=ckb seed=49'

echo '--- TASK start: model=ckb seed=50'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/ckb.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/ckb.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'ckb' --type 'single' --lang_src 'ckb' --lang_tgt '' --seed 50 --epochs 1 --batch_size 128
echo '--- TASK end: model=ckb seed=50'

echo '--- TASK start: model=ckb seed=51'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/ckb.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/ckb.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'ckb' --type 'single' --lang_src 'ckb' --lang_tgt '' --seed 51 --epochs 1 --batch_size 128
echo '--- TASK end: model=ckb seed=51'

echo '--- TASK start: model=cs seed=42'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/cs.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/cs.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'cs' --type 'single' --lang_src 'cs' --lang_tgt '' --seed 42 --epochs 1 --batch_size 128
echo '--- TASK end: model=cs seed=42'

echo '--- TASK start: model=cs seed=43'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/cs.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/cs.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'cs' --type 'single' --lang_src 'cs' --lang_tgt '' --seed 43 --epochs 1 --batch_size 128
echo '--- TASK end: model=cs seed=43'

echo '--- TASK start: model=cs seed=44'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/cs.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/cs.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'cs' --type 'single' --lang_src 'cs' --lang_tgt '' --seed 44 --epochs 1 --batch_size 128
echo '--- TASK end: model=cs seed=44'

echo '--- TASK start: model=cs seed=45'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/cs.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/cs.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'cs' --type 'single' --lang_src 'cs' --lang_tgt '' --seed 45 --epochs 1 --batch_size 128
echo '--- TASK end: model=cs seed=45'

echo '--- TASK start: model=cs seed=46'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/cs.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/cs.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'cs' --type 'single' --lang_src 'cs' --lang_tgt '' --seed 46 --epochs 1 --batch_size 128
echo '--- TASK end: model=cs seed=46'

echo '--- TASK start: model=cs seed=47'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/cs.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/cs.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'cs' --type 'single' --lang_src 'cs' --lang_tgt '' --seed 47 --epochs 1 --batch_size 128
echo '--- TASK end: model=cs seed=47'

echo '--- TASK start: model=cs seed=48'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/cs.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/cs.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'cs' --type 'single' --lang_src 'cs' --lang_tgt '' --seed 48 --epochs 1 --batch_size 128
echo '--- TASK end: model=cs seed=48'

echo '--- TASK start: model=cs seed=49'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/cs.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/cs.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'cs' --type 'single' --lang_src 'cs' --lang_tgt '' --seed 49 --epochs 1 --batch_size 128
echo '--- TASK end: model=cs seed=49'

echo '--- TASK start: model=cs seed=50'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/cs.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/cs.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'cs' --type 'single' --lang_src 'cs' --lang_tgt '' --seed 50 --epochs 1 --batch_size 128
echo '--- TASK end: model=cs seed=50'

echo '--- TASK start: model=cs seed=51'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/cs.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/cs.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'cs' --type 'single' --lang_src 'cs' --lang_tgt '' --seed 51 --epochs 1 --batch_size 128
echo '--- TASK end: model=cs seed=51'

echo '--- TASK start: model=cy seed=42'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/cy.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/cy.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'cy' --type 'single' --lang_src 'cy' --lang_tgt '' --seed 42 --epochs 1 --batch_size 128
echo '--- TASK end: model=cy seed=42'

echo '--- TASK start: model=cy seed=43'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/cy.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/cy.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'cy' --type 'single' --lang_src 'cy' --lang_tgt '' --seed 43 --epochs 1 --batch_size 128
echo '--- TASK end: model=cy seed=43'

echo '--- TASK start: model=cy seed=44'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/cy.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/cy.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'cy' --type 'single' --lang_src 'cy' --lang_tgt '' --seed 44 --epochs 1 --batch_size 128
echo '--- TASK end: model=cy seed=44'

echo '--- TASK start: model=cy seed=45'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/cy.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/cy.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'cy' --type 'single' --lang_src 'cy' --lang_tgt '' --seed 45 --epochs 1 --batch_size 128
echo '--- TASK end: model=cy seed=45'

echo '--- TASK start: model=cy seed=46'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/cy.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/cy.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'cy' --type 'single' --lang_src 'cy' --lang_tgt '' --seed 46 --epochs 1 --batch_size 128
echo '--- TASK end: model=cy seed=46'

echo '--- TASK start: model=cy seed=47'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/cy.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/cy.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'cy' --type 'single' --lang_src 'cy' --lang_tgt '' --seed 47 --epochs 1 --batch_size 128
echo '--- TASK end: model=cy seed=47'

echo '--- TASK start: model=cy seed=48'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/cy.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/cy.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'cy' --type 'single' --lang_src 'cy' --lang_tgt '' --seed 48 --epochs 1 --batch_size 128
echo '--- TASK end: model=cy seed=48'

echo '--- TASK start: model=cy seed=49'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/cy.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/cy.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'cy' --type 'single' --lang_src 'cy' --lang_tgt '' --seed 49 --epochs 1 --batch_size 128
echo '--- TASK end: model=cy seed=49'

echo '--- TASK start: model=cy seed=50'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/cy.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/cy.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'cy' --type 'single' --lang_src 'cy' --lang_tgt '' --seed 50 --epochs 1 --batch_size 128
echo '--- TASK end: model=cy seed=50'

echo '--- TASK start: model=cy seed=51'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/cy.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/cy.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'cy' --type 'single' --lang_src 'cy' --lang_tgt '' --seed 51 --epochs 1 --batch_size 128
echo '--- TASK end: model=cy seed=51'

echo '--- TASK start: model=de seed=42'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/de.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/de.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'de' --type 'single' --lang_src 'de' --lang_tgt '' --seed 42 --epochs 1 --batch_size 128
echo '--- TASK end: model=de seed=42'

echo '--- TASK start: model=de seed=43'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/de.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/de.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'de' --type 'single' --lang_src 'de' --lang_tgt '' --seed 43 --epochs 1 --batch_size 128
echo '--- TASK end: model=de seed=43'

echo '--- TASK start: model=de seed=44'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/de.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/de.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'de' --type 'single' --lang_src 'de' --lang_tgt '' --seed 44 --epochs 1 --batch_size 128
echo '--- TASK end: model=de seed=44'

echo '--- TASK start: model=de seed=45'
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/speechLLM/run_and_log.py --train_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/de.train.tsv' --test_tsv '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv/de.test.tsv' --out_dir './out' --csv_path '/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/speechLLM_matrix/speaker_matrix_single.csv' --model_id 'de' --type 'single' --lang_src 'de' --lang_tgt '' --seed 45 --epochs 1 --batch_size 128
echo '--- TASK end: model=de seed=45'

