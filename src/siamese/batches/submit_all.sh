#!/bin/bash
set -euo pipefail
mkdir -p logs
sbatch /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/batches/batch_0000.sh
sbatch /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/batches/batch_0001.sh
sbatch /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/batches/batch_0002.sh
sbatch /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/batches/batch_0003.sh
sbatch /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/batches/batch_0004.sh
sbatch /gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/siamese/batches/batch_0005.sh
