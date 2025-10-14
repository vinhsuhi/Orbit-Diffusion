#!/bin/bash
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --nodes=1
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --output=%x_%u_%j.out
#SBATCH --job-name=qm9_g_ief
#SBATCH --partition=gpu,dgx

eval "$(conda shell.bash hook)"
conda activate etflow

export DATA_DIR="/data/work/ac141281/IEF_processed_data"

config_path=$1  # Pass checkpoint path as argument
checkpoint_path=$2  # Pass checkpoint path as argument
wandb offline

if [[ "$config_path" == *"gaussian"* ]]; then
    prior="gaussian"
else
    prior="harmonic"
    # exit 1
fi

sampler_type="ode"

# Extract experiment details from checkpoint path
experiment_name=$(echo "$checkpoint_path" | awk -F'/' '{print $(NF-5)}')
timestamp=$(echo "$checkpoint_path" | awk -F'/' '{print $(NF-2)}')
checkpoint_name=$(basename "$checkpoint_path" .ckpt)

for nsteps in 50 100; do
    test_name="${sampler_type}_qm9_${experiment_name}_${timestamp}_${prior}_nstep${nsteps}_sigma0.1_${checkpoint_name}"
    generated_path="$(pwd)/logs/samples/${test_name}/flow_nsteps_${nsteps}/generated_files.pkl"
    log_file="results/${test_name}"

    python etflow/eval.py \
    --config ${config_path} \
    --sampler_type ${sampler_type} \
    --test_name ${test_name} \
    --nsteps ${nsteps} --dataset_type qm9 \
    --checkpoint ${checkpoint_path} 

    python etflow/eval_cov_mat.py --path ${generated_path} --num_workers 10 --log_file ${log_file}
    echo ${log_file}
    echo ${generated_path}
done