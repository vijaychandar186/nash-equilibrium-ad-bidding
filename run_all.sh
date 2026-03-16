#!/usr/bin/env bash
# run_all.sh — run pipeline and write logs to logs/run_all.log
set -euo pipefail

# Activate virtual environment if it exists
if [ -f "env/bin/activate" ]; then
    source env/bin/activate
fi

# Ensure python prints unbuffered output (helps real-time logging)
export PYTHONUNBUFFERED=1

LOGDIR="./logs"
LOGFILE="${LOGDIR}/run_all.log"

mkdir -p "${LOGDIR}"
# Redirect all stdout/stderr to logfile AND to console (tee -a)
exec > >(tee -a "${LOGFILE}") 2>&1

echo "================================================================="
echo "Pipeline run started: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Log file: ${LOGFILE}"
echo "================================================================="
echo ""

run_step() {
  local name="$1"
  shift
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] START: ${name}"
  # run command passed in args, exit on failure (set -e takes care)
  "$@"
  local rc=$?
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] END:   ${name} (exit ${rc})"
  echo "-----------------------------------------------------------------"
  echo ""
}

run_step "data_loading" python src/data_loading.py
run_step "feature_engineering" python src/feature_engineering.py
run_step "eda" python src/eda.py
run_step "preprocessing" python src/preprocessing.py
run_step "train_test_split_and_imputation" python src/train_test_split_and_imputation.py
run_step "models_and_training" python src/models_and_training.py
run_step "results_and_statistics" python src/results_and_statistics.py

echo ""
echo "================================================================="
echo "Pipeline finished: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Logs at: ${LOGFILE}"
echo "================================================================="
