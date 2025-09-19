#!/bin/bash
#SBATCH --time=1-00:00:00       # Runtime in D-HH:MM:SS
#SBATCH --gres=gpu:1            # Request 1 GPU
#SBATCH --output=%j.out         # Standard output log with Job ID
#SBATCH --error=%j.err          # Standard error log with Job ID

# Activate the Conda environment
eval "$(conda shell.bash hook)"
conda activate tgdmat

# Navigate to the task directory
cd csp_task/

dataset=$1
prompt_type=$2
n_trans=$3
wn_noise=$4
datetime_str=$(date +"%d%m%Y%H%M%S")

# Convert new_noise to lowercase for case-insensitive comparison
wn_noise_lower=$(echo "$wn_noise" | tr '[:upper:]' '[:lower:]')

# Determine exp_name
if [[ "$n_trans" -eq 0 ]]; then
    exp_name="${dataset}_${prompt_type}_baseline_${datetime_str}"
else
    if [[ "$wn_noise_lower" == "true" ]]; then
        exp_name="${dataset}_${prompt_type}_trans${n_trans}_WN_${datetime_str}"
    else
        exp_name="${dataset}_trans${n_trans}_U_${datetime_str}"
    fi
fi

echo ${exp_name}


noise_arg=""
if [ "${wn_noise}" = "True" ]; then
    noise_arg="--new_noise"
fi
python -W ignore train.py --dataset ${dataset} \
                        --prompt_type ${prompt_type} \
                        --num_trans ${n_trans} \
                        --exp_name ${exp_name} ${noise_arg} --epochs 10
