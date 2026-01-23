#!/bin/bash
#SBATCH --account bsc88
#SBATCH --qos acc_bscls
#SBATCH --nodes 1
#SBATCH --cpus-per-task 20
#SBATCH --gres gpu:1
#SBATCH --time 00-03:00:00
#SBATCH --exclusive
#SBATCH --exclude=as04r2b01,as02r3b04,as07r1b02
#SBATCH --job-name=hubert_train
#SBATCH --output=/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/logs/speaker/%x_%j.log
#SBATCH --error=/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/logs/speaker/%x_%j.err
# Important: export environment vars from Slurm to children
#SBATCH --export=ALL,CUBLAS_WORKSPACE_CONFIG=:4096:8,PYTHONHASHSEED=42,OMP_NUM_THREADS=1,MKL_NUM_THREADS=1

set -euo pipefail

# ================================================================
#  UNIFIED PIPELINE (SEQUENTIAL):
#       IDENTIFICATION → VERIFICATION (post-hoc validation over checkpoints)
#
#  Usage:
#      ./run_sid_sv.sh <LANG> <SEED> <TIMESTAMP>
# ================================================================

if [ $# -ne 3 ]; then
  echo "Usage: $0 <LANG> <SEED> <TIMESTAMP>"
  exit 1
fi

LANG="$1"
SEED="$2"
TIMESTAMP="$3"

# -------------------------------
# Config (ajustables)
# -------------------------------
# Número máximo de checkpoints esperados (no obliga a que haya tantos)
MAX_EPOCHS=${MAX_EPOCHS:-15}
METRIC_KEY=${METRIC_KEY:-EER}   # métrica para comparar, por defecto EER (menor mejor)

# Rutas de pares val/test (usa tu carpeta de split creada previamente)
PAIRS_VAL="/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/08_test_val_pairs/tsv/${LANG}.val.tsv"
PAIRS_TEST="/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__speaker_verif/08_test_val_pairs/tsv/${LANG}.test.tsv"

# -------------------------------
# Paths
# -------------------------------
ROOT="/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech"

SID_SUBMIT="${ROOT}/src/speaker/identification/submit_slurm.sh"
SV_SCRIPT="${ROOT}/src/speaker/verification/sv_from_sid.py"

CONDA_ENV_PATH="/gpfs/projects/bsc88/speech/mm_s2st/scripts/paraling/gender_id_hubert/env/"

HF_MODEL="utter-project/mHuBERT-147"
MODEL_SLUG="mHuBERT-147"

# Original convention usada en tu repo
SID_OUTPUT_ROOT="${ROOT}/src/speaker/verification/outputs/single/speaker_id/${MODEL_SLUG}"

# Deterministic path al experimento (igual que antes)
if [ -z "${SLURM_JOB_ID:-}" ]; then
  echo "Warning: SLURM_JOB_ID not set, using manual run"
  JOB_ID="manual"
else
  JOB_ID="${SLURM_JOB_ID}"
fi

MODEL_DIR="${SID_OUTPUT_ROOT}/${LANG}_${TIMESTAMP}_${JOB_ID}"

echo "=============================================================="
echo " RUNNING UNIFIED PIPELINE (SPEAKER)"
echo "  LANG      = ${LANG}"
echo "  SEED      = ${SEED}"
echo "  TIMESTAMP = ${TIMESTAMP}"
echo "  MODEL_DIR = ${MODEL_DIR}"
echo "  MAX_EPOCHS= ${MAX_EPOCHS}"
echo "  METRIC_KEY= ${METRIC_KEY}"
echo "=============================================================="

# ----------------------------------------------------------
# 1) IDENTIFICATION (SID)  -- dejamos idéntico al original
# ----------------------------------------------------------

echo "[1/2] Running identification training..."
export LANGS_STR="${LANG}"
export SEED="${SEED}"
export TIMESTAMP="${TIMESTAMP}"

bash "${SID_SUBMIT}"

echo "[OK] Identification completed."
echo

# ----------------------------------------------------------
# 2) POST-HOC VALIDATION: validar checkpoints HF correctamente
# ----------------------------------------------------------

SV_VAL_ROOT="${ROOT}/src/speaker/verification/outputs/embeddings_verif/${LANG}/${TIMESTAMP}/val_by_ckpt"
SV_TEST_ROOT="${ROOT}/src/speaker/verification/outputs/embeddings_verif/${LANG}/${TIMESTAMP}/test_final"
mkdir -p "${SV_VAL_ROOT}" "${SV_TEST_ROOT}"

# Activate environment for SV
if [ -f "$(conda info --base)/etc/profile.d/conda.sh" ]; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV_PATH}"
fi

# ----------------------------------------------------------
# Helper: find HF checkpoints (directories only)
# ----------------------------------------------------------
find_hf_checkpoints() {
  local base="${MODEL_DIR}"
  find "${base}" -maxdepth 1 -type d -name "checkpoint-*" | sort -V
}

# ----------------------------------------------------------
# Helper: read metric, with fallback computing from pairs_scores.csv
# ----------------------------------------------------------
read_metric() {
  local metrics_json="$1"
  local key="$2"

  # If metrics.json already exists, read it (jq if available)
  if [ -f "${metrics_json}" ]; then
    if command -v jq >/dev/null 2>&1; then
      jq -r --arg k "${key}" '.[$k] // "nan"' "${metrics_json}" 2>/dev/null || echo "nan"
    else
      grep -oP "\"${key}\"\s*:\s*\K[0-9.]+" "${metrics_json}" | head -n1 || echo "nan"
    fi
    return
  fi

  # No metrics.json: try to compute from pairs_scores.csv in same out dir
  local out_dir
  out_dir="$(dirname "${metrics_json}")"
  local pairs_csv="${out_dir}/pairs_scores.csv"

  if [ ! -f "${pairs_csv}" ]; then
    echo "nan"
    return
  fi

  # Run embedded Python to compute EER and AUC and write metrics.json
  python - "${pairs_csv}" "${out_dir}" > "${out_dir}/metrics_compute.stdout" 2> "${out_dir}/metrics_compute.log" <<'PY'
import sys, json, csv, math
from pathlib import Path
try:
    pairs_csv = Path(sys.argv[1])
    out_dir = Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "metrics.json"

    labels = []
    scores = []
    with pairs_csv.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for r in reader:
            # score
            if 'score' in r and r['score'] != '':
                sc = r['score']
            else:
                # fallback to last column
                sc = list(r.values())[-1]
            # label
            lab = r.get('label') or r.get('y') or r.get('target') or r.get('gt')
            try:
                scores.append(float(sc))
                labels.append(int(float(lab)))
            except Exception:
                continue

    if len(scores) == 0:
        json.dump({}, out_json.open("w", encoding="utf-8"), indent=2)
        print(json.dumps({}))
        sys.exit(0)

    # Use numpy for numeric ops
    try:
        import numpy as np
    except Exception:
        # fallback naive computation: produce empty metrics
        json.dump({}, out_json.open("w", encoding="utf-8"), indent=2)
        print(json.dumps({}))
        sys.exit(0)

    s = np.array(scores)
    t = np.array(labels)
    P = float((t == 1).sum())
    N = float((t == 0).sum())

    # AUC
    if P == 0 or N == 0:
        auc = float('nan')
    else:
        order = np.argsort(-s)
        t_sorted = t[order]
        tps = np.cumsum(t_sorted == 1).astype(float)
        fps = np.cumsum(t_sorted == 0).astype(float)
        tpr = tps / P
        fpr = fps / N
        fpr_full = np.concatenate(([0.0], fpr, [1.0]))
        tpr_full = np.concatenate(([0.0], tpr, [1.0]))
        auc = float(np.trapz(tpr_full, fpr_full))

    # EER
    if P == 0 or N == 0:
        eer = float('nan')
    else:
        thresholds = np.unique(s)
        desc = np.sort(thresholds)[::-1]
        fpr_list = []
        fnr_list = []
        for thr in desc:
            preds = (s >= thr).astype(int)
            tp = float(((preds == 1) & (t == 1)).sum())
            fp = float(((preds == 1) & (t == 0)).sum())
            fn = float(((preds == 0) & (t == 1)).sum())
            tn = float(((preds == 0) & (t == 0)).sum())
            fpr_v = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            fnr_v = fn / (fn + tp) if (fn + tp) > 0 else 0.0
            fpr_list.append(fpr_v)
            fnr_list.append(fnr_v)
        fpr_arr = np.array(fpr_list)
        fnr_arr = np.array(fnr_list)
        idx = int(np.argmin(np.abs(fpr_arr - fnr_arr)))
        eer = float((fpr_arr[idx] + fnr_arr[idx]) / 2.0)

    def maybe_round(x):
        if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
            return None
        return float(np.round(x, 6))

    out = {
        "EER": maybe_round(eer),
        "AUC": maybe_round(auc),
        "n_pairs": int(len(scores)),
        "n_pos": int(int((t == 1).sum())) if len(t) > 0 else 0,
        "n_neg": int(int((t == 0).sum())) if len(t) > 0 else 0
    }

    with out_json.open("w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)

    print(json.dumps(out))
except Exception as e:
    # best effort: write empty metrics.json and exit
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        with (out_dir / "metrics.json").open("w", encoding="utf-8") as fh:
            json.dump({}, fh)
    except Exception:
        pass
    print(json.dumps({}))
    sys.exit(0)
PY

  # After python run, read generated metrics.json
  if [ -f "${metrics_json}" ]; then
    if command -v jq >/dev/null 2>&1; then
      jq -r --arg k "${key}" '.[$k] // "nan"' "${metrics_json}" 2>/dev/null || echo "nan"
    else
      grep -oP "\"${key}\"\s*:\s*\K[0-9.]+" "${metrics_json}" | head -n1 || echo "nan"
    fi
  else
    echo "nan"
  fi
}

echo "[2/2] Searching for HF checkpoints produced by SID..."

mapfile -t ALL_CKPTS < <(find_hf_checkpoints)

if [ ${#ALL_CKPTS[@]} -eq 0 ]; then
  echo "ERROR: No HF checkpoints found in ${MODEL_DIR}"
  exit 1
fi

echo "Found ${#ALL_CKPTS[@]} checkpoints."

# limitar a MAX_EPOCHS (últimos)
if [ ${#ALL_CKPTS[@]} -gt "${MAX_EPOCHS}" ]; then
  START=$((${#ALL_CKPTS[@]} - MAX_EPOCHS))
  CKPTS=( "${ALL_CKPTS[@]:${START}:${MAX_EPOCHS}}" )
else
  CKPTS=( "${ALL_CKPTS[@]}" )
fi

echo "Checkpoints to validate:"
for ck in "${CKPTS[@]}"; do
  echo "  ${ck}"
done

BEST_METRIC=9999
BEST_CKPT=""

i=0
for ckpt_dir in "${CKPTS[@]}"; do
  i=$((i+1))
  name=$(basename "${ckpt_dir}")

  echo
  echo "--------------------------------------------------------------"
  echo " VALIDATING checkpoint ${i} / ${#CKPTS[@]}"
  echo "  CKPT_DIR = ${ckpt_dir}"
  echo "--------------------------------------------------------------"

  SV_OUT="${SV_VAL_ROOT}/${name}"
  mkdir -p "${SV_OUT}"

  python "${SV_SCRIPT}" \
    --model_dir "${ckpt_dir}" \
    --pairs_file "${PAIRS_VAL}" \
    --out_dir "${SV_OUT}" \
    --batch_size 8 \
    --l2_norm \
    --device cuda \
    --pairs_cols u v label \
    --tta_speeds 1.0 0.98 1.02 \
    --no_summary

  METR_JSON="${SV_OUT}/metrics.json"
  VAL_METRIC="$(read_metric "${METR_JSON}" "${METRIC_KEY}")"

  echo " Validation ${METRIC_KEY} = ${VAL_METRIC}"

  if [[ "${VAL_METRIC}" == "nan" || -z "${VAL_METRIC}" ]]; then
    echo " Warning: invalid metric, skipping"
    continue
  fi

  is_better=$(awk -v a="${VAL_METRIC}" -v b="${BEST_METRIC}" 'BEGIN{print (a+0 < b+0) ? 1 : 0}')
  if [ "${is_better}" -eq 1 ]; then
    BEST_METRIC="${VAL_METRIC}"
    BEST_CKPT="${ckpt_dir}"
    echo " New best checkpoint: ${BEST_CKPT}"
  fi
done

if [ -z "${BEST_CKPT}" ]; then
  echo "ERROR: No valid checkpoint found during validation."
  exit 1
fi

echo
echo "=============================================================="
echo "Best checkpoint selected:"
echo "  ${BEST_CKPT}"
echo "  ${METRIC_KEY} = ${BEST_METRIC}"
echo "=============================================================="

# ----------------------------------------------------------
# Final test evaluation
# ----------------------------------------------------------
SV_TEST_OUT="${SV_TEST_ROOT}/final_best"
mkdir -p "${SV_TEST_OUT}"

python "${SV_SCRIPT}" \
  --model_dir "${BEST_CKPT}" \
  --pairs_file "${PAIRS_TEST}" \
  --out_dir "${SV_TEST_OUT}" \
  --batch_size 8 \
  --l2_norm \
  --device cuda \
  --pairs_cols u v label \
  --tta_speeds 1.0 0.98 1.02
