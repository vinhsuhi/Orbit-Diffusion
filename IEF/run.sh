#!/bin/bash
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --nodes=1
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --output=%x_%u_%j.out
#SBATCH --job-name=qm9_g_ief
#SBATCH --partition=gpu,dgx
#SBATCH --time=1-00:00:00

# eval "$(conda shell.bash hook)"
# conda activate etflow

export DATA_DIR="/data/work/ac141281/IEF_processed_data"

# print the current conda environment
echo "Current conda environment: $CONDA_DEFAULT_ENV"

if [ "$2" == "baseline" ]; then
    python3 etflow/train.py -c $1 --train_mode baseline \
    --pretrained_ckpt /data/work/ac141281/IEF/pretrained/qm9-o3.ckpt
else
    python3 etflow/train.py -c $1 --train_mode ief \
    --num_rotations $3 \
    --max_perms $4 \
    --z 0.4  --sigma 0.1 --pretrained_ckpt /data/work/ac141281/IEF/pretrained/qm9-o3.ckpt
fi
