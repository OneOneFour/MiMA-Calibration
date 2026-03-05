#!/bin/bash

# Usage: ./rerun_plots.sh [experiment_dir] [env]
# Example: ./rerun_plots.sh experiments/T21_land_2D/20251223_203536_W1000_N4000

EXPERIMENT_DIR=$1
ENV=${2:-"py311noipython"}

# Ensure we are in the script's directory (MiMA/)
cd "$(dirname "$0")"

if [ -z "$EXPERIMENT_DIR" ]; then
    echo "Usage: ./rerun_plots.sh <experiment_dir> [env]"
    exit 1
fi

if [ ! -d "$EXPERIMENT_DIR" ]; then
    echo "Error: Directory $EXPERIMENT_DIR does not exist."
    exit 1
fi

echo "================================================================"
echo "Re-running Analysis Pipeline"
echo "Experiment Directory: $EXPERIMENT_DIR"
echo "Environment: $ENV"
echo "================================================================"

MODEL_DIR="$EXPERIMENT_DIR/model"
SAMPLE_STEP=3 # Default sample step

if [ ! -d "$MODEL_DIR" ]; then
    echo "Error: Model directory $MODEL_DIR does not exist."
    exit 1
fi

echo "[1/5] Plotting Data Samples..."
conda run -n $ENV python main.py plot sim_sample --model_dir "$MODEL_DIR" --output_dir "$EXPERIMENT_DIR" --sample_step "$SAMPLE_STEP" --alpha 0.9
if [ $? -ne 0 ]; then
    echo "Error: Data plotting failed."
    exit 1
fi

echo "[2/5] Generating Diagnostic Plots..."
conda run -n $ENV python main.py plot trace --model_dir "$MODEL_DIR" --output_dir "$EXPERIMENT_DIR"
if [ $? -ne 0 ]; then
    echo "Error: Diagnostics plotting failed."
    exit 1
fi

echo "[3/5] Running Posterior Predictive Checks..."
CHAIN_FILE=$(ls "$EXPERIMENT_DIR"/*.nc 2>/dev/null | head -n 1)

if [ -z "$CHAIN_FILE" ]; then
    echo "Error: No chain file found in $EXPERIMENT_DIR to run PPC."
    exit 1
fi

echo "Using chain file: $CHAIN_FILE"
conda run -n $ENV python main.py plot ppc --model_dir "$MODEL_DIR" --file_path "$CHAIN_FILE" --output_dir "$EXPERIMENT_DIR/ppc"
if [ $? -ne 0 ]; then
    echo "Error: PPC plotting failed."
    exit 1
fi

echo "[4/5] Plotting Predictions..."
conda run -n $ENV python main.py plot predictions --model_dir "$MODEL_DIR" --output_dir "$EXPERIMENT_DIR" --plot_observations True --chain_path "$CHAIN_FILE"
if [ $? -ne 0 ]; then
    echo "Warning: plot_predictions failed (continuing)."
fi

echo "[5/5] Analyzing Posterior..."
conda run -n $ENV python main.py analyze "$EXPERIMENT_DIR"
if [ $? -ne 0 ]; then
    echo "Error: Posterior analysis failed."
    exit 1
fi

echo "================================================================"
echo "Analysis Completed Successfully."
echo "Output Directory: $EXPERIMENT_DIR"
echo "================================================================"
