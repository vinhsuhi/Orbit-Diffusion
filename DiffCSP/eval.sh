#!/bin/bash
#SBATCH -t 1-00:00        # Runtime in D-HH:MM format
#SBATCH --gres=gpu:1      # Number of GPUs
#SBATCH -o %j.out         # Name of stdout output file with job ID
#SBATCH -e %j.err         # Name of stderr error file with job ID

# eval "$(conda shell.bash hook)"
# conda activate diffcsp_test

# Set environment variables
export PROJECT_ROOT=$(pwd)
export HYDRA_JOBS=$PROJECT_ROOT
export WABDB_DIR=$PROJECT_ROOT
export HYDRA_FULL_ERROR=1
export PYTHONPATH=$PROJECT_ROOT

# Evaluation loop
root_path=$1
dataset=$2
label=$3
python3 scripts/evaluate.py --model_path ${root_path} --dataset ${dataset} --label ${label}
python3 scripts/compute_metrics.py --root_path ${root_path} --label ${label} --tasks csp | tee "${root_path}/${label}.out" 2>&1

