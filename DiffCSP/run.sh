#!/bin/bash
#SBATCH -t 1-00:00        # Runtime in D-HH:MM format
#SBATCH --gres=gpu:1      # Number of GPUs
#SBATCH -o %j.out         # Name of stdout output file with job ID
#SBATCH -e %j.err         # Name of stderr error file with job ID

# eval "$(conda shell.bash hook)"
# conda activate diffcsp

# Set environment variables
export PROJECT_ROOT=$(pwd)
export HYDRA_JOBS=$PROJECT_ROOT
export WABDB_DIR=$PROJECT_ROOT
export HYDRA_FULL_ERROR=1
export PYTHONPATH=$PROJECT_ROOT

# Retrieve arguments
dataset=$1
n_trans=$2
wn_noise=$3

# Convert new_noise to lowercase for case-insensitive comparison
wn_noise_lower=$(echo "$wn_noise" | tr '[:upper:]' '[:lower:]')

# Determine shift_C and construct expname based on new_noise
if [[ "$wn_noise_lower" == "true" ]]; then
    expname="${dataset}_trans${n_trans}_WN_$(date "+%y-%m-%d-%H-%M-%S")"
else
    expname="${dataset}_trans${n_trans}_U_$(date "+%y-%m-%d-%H-%M-%S")"
fi


echo ${expname}
today=$(date "+%Y-%m-%d")
# Run the Python script
python diffcsp/run.py data=${dataset} \
    expname=${expname} \
    data.n_perm=0 data.n_pos=1 +model.num_trans=${n_trans} \
    data.include_identity=True +model.include_identity=True \
    +model.new_noise=${wn_noise}
