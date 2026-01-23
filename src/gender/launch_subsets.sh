#!/bin/bash
set -euo pipefail

# ---------------------------------------
# Config
# ---------------------------------------
PLAN_SCRIPT="./launch_subsets_plan_tasks.sh"
WORKER_SCRIPT="./launch_subsets_array_worker.sh"

# Where to write the global tasks list
TASKS_FILE="pending_tasks.tsv"

# Max tasks per array
MAX_TASKS_PER_ARRAY=100

# Max number of array tasks running in parallel per array
ARRAY_MAX_CONCURRENT=100

# Sleep time between array submissions (in seconds)
SLEEP_BETWEEN=1800   # 30 minutes

# ---------------------------------------
# Preconditions
# ---------------------------------------
if [ ! -x "${PLAN_SCRIPT}" ]; then
  echo "Error: plan script not found or not executable: ${PLAN_SCRIPT}" >&2
  exit 1
fi

if [ ! -x "${WORKER_SCRIPT}" ]; then
  echo "Error: worker script not found or not executable: ${WORKER_SCRIPT}" >&2
  exit 1
fi

# ---------------------------------------
# 1) Generate pending_tasks.tsv
# ---------------------------------------
echo "Generating task plan with ${PLAN_SCRIPT} ..."
"${PLAN_SCRIPT}" > "${TASKS_FILE}"

TOTAL_LINES=$(wc -l < "${TASKS_FILE}")
echo "Total tasks in ${TASKS_FILE}: ${TOTAL_LINES}"

if [ "${TOTAL_LINES}" -eq 0 ]; then
  echo "Nothing to do: task plan is empty."
  exit 0
fi

# ---------------------------------------
# 2) Split TSV into chunks
# ---------------------------------------
echo "Splitting into chunks of at most ${MAX_TASKS_PER_ARRAY} tasks per array."

# Clean old chunks if any
rm -f pending_tasks_part_*.tsv pending_tasks_part_*

# split creates files: pending_tasks_part_00, pending_tasks_part_01, ...
split -d -l "${MAX_TASKS_PER_ARRAY}" "${TASKS_FILE}" pending_tasks_part_

# Rename to .tsv for clarity
for f in pending_tasks_part_*; do
  mv "${f}" "${f}.tsv"
done

PARTS=(pending_tasks_part_*.tsv)
NUM_PARTS=${#PARTS[@]}

echo "Created ${NUM_PARTS} chunk file(s)."
echo

# ---------------------------------------
# 3) Submit one array per chunk, with pause
# ---------------------------------------
for ((i=0; i<NUM_PARTS; i++)); do
  part="${PARTS[$i]}"
  LINES_IN_PART=$(wc -l < "${part}")

  if [ "${LINES_IN_PART}" -eq 0 ]; then
    echo "Skipping empty chunk ${part}"
    continue
  fi

  echo "Submitting array for chunk $((i+1))/${NUM_PARTS}: ${part}"
  echo "  Lines in chunk: ${LINES_IN_PART}"

  # Array indices are 0..LINES_IN_PART-1
  CMD=(sbatch "--array=0-$((LINES_IN_PART-1))%${ARRAY_MAX_CONCURRENT}" "${WORKER_SCRIPT}" "${part}")
  echo "  Command: ${CMD[*]}"

  job_out=$("${CMD[@]}")
  echo "  sbatch response: ${job_out}"
  echo

  # Sleep between arrays, except after the last one
  if [ $((i+1)) -lt "${NUM_PARTS}" ]; then
    echo "Sleeping ${SLEEP_BETWEEN} seconds before next array..."
    sleep "${SLEEP_BETWEEN}"
    echo
  fi
done

echo "All arrays submitted."
