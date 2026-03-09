#!/bin/bash
# Script de lanzamiento del orquestador (SIN cabecera sbatch)
# Hazlo ejecutable: chmod +x run_make_batches.sh
python /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/ECAPA/make_batches.py \
  --pairs_dir /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/07_pairs_speaker/tsv \
  --single1000_dir /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/05_balanced_1000_cv_gender/tsv \
  --single2000_dir /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/06_balanced_2000_cv_gender/tsv \
  --output_dir /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/ECAPA/batches \
  --run_script /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/ECAPA/run_and_log.py \
  --bilingual_csv /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/ECAPA_matrix/speaker_matrix_bilingual.csv \
  --single_csv /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/ECAPA_matrix/speaker_matrix_single.csv \
  --single2000_csv /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/ECAPA_matrix/speaker_matrix_dual.csv \
  --seeds 42-51 \
  --batch_size 100 \
  --extra_args "--epochs 1 --batch_size 128"

/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/ECAPA/batches/submit_all.sh