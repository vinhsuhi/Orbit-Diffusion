#!/bin/sh
#SBATCH -t 1-00:00        # Runtime in D-HH:MM format
#SBATCH --gres=gpu:1      # Number of GPUs
#SBATCH -o %j.out  # Name of stdout output file with job ID
#SBATCH -e %j.err   # Name of stderr error file with job ID

eval "$(conda shell.bash hook)"
conda activate tgdmat
model_path=$1
label=$2
# Detect dataset (checks for known patterns)
if [[ "$model_path" == *"perov_5"* ]]; then
    dataset="perov_5"
elif [[ "$model_path" == *"mp_20"* ]]; then
    dataset="mp_20"
else
    echo "Unknown dataset in path!"
    exit 1
fi

# Detect prompt_type
if [[ "$model_path" == *"long"* ]]; then
    prompt_type="long"
elif [[ "$model_path" == *"short"* ]]; then
    prompt_type="short"
else
    echo "Unknown prompt_type in path!"
    exit 1
fi

cd csp_task/

python -W ignore evaluate.py --model_path ${model_path} \
                            --chkpt_path  ${model_path}/model_final.pt \
                            --tasks csp --num_evals 1 --dataset ${dataset}  \
                            --batch_size 512 --timesteps 1000 --prompt_type ${prompt_type} \
                            --label ${label}
python compute_metrics.py --root_path ${model_path} --tasks csp --label ${label}
