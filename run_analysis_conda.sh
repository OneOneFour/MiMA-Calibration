#!/bin/bash

# Usage: ./run_analysis_conda.sh [env] [backend] [model_dir] [warmup] [main_iter] [sample_step] [n_chain]
# Example: ./run_analysis_conda.sh py311noipython mici models/T21 100 100 3 2

ENV=${1:-"py311noipython"}
BACKEND=${2:-mici}   # mici | blackjax
MODEL_DIR=${3:-models/T21}
WARMUP=${4:-60}
MAIN=${5:-60}
SAMPLE_STEP=${6:-3}
N_CHAIN=${7:-2}
# N_PROCESSES=${8:-1}


# Ensure we are in the script's directory
cd "$(dirname "$0")"

echo "----------------------------------------------------------------"
echo "Starting Analysis Pipeline"
echo "Backend: $BACKEND"
echo "Model Dir: $MODEL_DIR"
echo "Warmup: $WARMUP"
echo "Main Iterations: $MAIN"
echo "Environment: $ENV"
echo "Sample Step: $SAMPLE_STEP"
echo "Chains: $N_CHAIN"
# echo "Processes: $N_PROCESSES"
echo "----------------------------------------------------------------"

CONFIG_PATH="$MODEL_DIR/config.py"
EXP_NAME=$(basename "$MODEL_DIR")

echo "[1/6] Running MCMC Sampling ($BACKEND)..."
# Stream output from conda-run and run Python unbuffered for live progress
# Note: we use python main.py run ...
PYTHONUNBUFFERED=1 conda run --no-capture-output -n $ENV python -u main.py run "$MODEL_DIR" -W "$WARMUP" -N "$MAIN" --n_chain "$N_CHAIN" --backend "$BACKEND" #--n_processes "$N_PROCESSES"
if [ $? -ne 0 ]; then
    echo "Error: MCMC sampling failed."
    exit 1
fi

# Extract experiment name from model_dir
EXP_NAME=$(basename "$MODEL_DIR")

# Find the latest run directory in experiments/$EXP_NAME
# We look for directories starting with 20 (year)
LATEST_RUN_DIR=$(ls -td experiments/"$EXP_NAME"/20* 2>/dev/null | head -n 1)

if [ -z "$LATEST_RUN_DIR" ]; then
    echo "Error: No run directory found in experiments/$EXP_NAME after MCMC."
    exit 1
fi

echo "Latest Run Directory: $LATEST_RUN_DIR"


# Archive model and config files and generate separate hashes
ARCHIVE_DIR="$LATEST_RUN_DIR/model"
mkdir -p "$ARCHIVE_DIR"

MODEL_SRC="$MODEL_DIR/model.py"
CONFIG_SRC="$MODEL_DIR/config.py"

if [ -f "$MODEL_SRC" ]; then
    cp "$MODEL_SRC" "$ARCHIVE_DIR/model.py"
    shasum -a 256 "$MODEL_SRC" | awk '{print $1}' > "$ARCHIVE_DIR/model-hash.txt"
else
    echo "Warning: Missing $MODEL_SRC for archiving."
fi

if [ -f "$CONFIG_SRC" ]; then
    cp "$CONFIG_SRC" "$ARCHIVE_DIR/config.py"
    shasum -a 256 "$CONFIG_SRC" | awk '{print $1}' > "$ARCHIVE_DIR/config-hash.txt"
else
    echo "Warning: Missing $CONFIG_SRC for archiving."
fi

echo "[2/6] Plotting Data Samples..."
conda run -n $ENV python main.py plot sim_sample --model_dir "$LATEST_RUN_DIR/model" --output_dir "$LATEST_RUN_DIR" --sample_step "$SAMPLE_STEP" --alpha 0.9
if [ $? -ne 0 ]; then
    echo "Error: Data plotting failed."
    exit 1
fi

echo "[3/6] Generating Diagnostic Plots..."
conda run -n $ENV python main.py plot trace --model_dir "$LATEST_RUN_DIR/model" --output_dir "$LATEST_RUN_DIR"
if [ $? -ne 0 ]; then
    echo "Error: Diagnostics plotting failed."
    exit 1
fi

echo "[4/6] Running Posterior Predictive Checks..."

# Find the NetCDF file in the latest run directory
CHAIN_FILE=$(ls "$LATEST_RUN_DIR"/*.nc 2>/dev/null | head -n 1)

if [ -z "$CHAIN_FILE" ]; then
    echo "Error: No chain file found in $LATEST_RUN_DIR to run PPC."
    exit 1
fi

echo "Using chain file: $CHAIN_FILE"
conda run -n $ENV python main.py plot ppc --model_dir "$LATEST_RUN_DIR/model" --file_path "$CHAIN_FILE" --output_dir "$LATEST_RUN_DIR/ppc"
if [ $? -ne 0 ]; then
    echo "Error: PPC plotting failed."
    exit 1
fi

# 5. Plot Predictions (may be experimental)
echo "[5/6] Plotting Predictions (experimental)..."
conda run -n $ENV python main.py plot predictions --model_dir "$LATEST_RUN_DIR/model" --output_dir "$LATEST_RUN_DIR" --plot_observations True --chain_path "$CHAIN_FILE"
if [ $? -ne 0 ]; then
    echo "Warning: plot_predictions failed (continuing)."
fi

echo "[6/6] Analyzing Posterior..."
conda run -n $ENV python main.py analyze "$LATEST_RUN_DIR"
if [ $? -ne 0 ]; then
    echo "Error: Posterior analysis failed."
    exit 1
fi

echo "----------------------------------------------------------------"
echo "Pipeline Completed Successfully."
echo "Output Directory: $LATEST_RUN_DIR"
echo "----------------------------------------------------------------"
