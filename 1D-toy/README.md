
# 1D Toy Experiment

This repository contains code for training and evaluating diffusion models on a simple **1D toy problem**.

We compare three experimental settings:
- **Ours**: proposed method
- **Baseline**: standard baseline
- **Baseline + Augmentation**: baseline with data augmentation
---
## 🚀 Training
All models are trained for **100,000 epochs** with **3 random seeds**.  
### Baseline
```bash
python3 main.py --num_epochs 100000 --num_seeds 3 --equi --baseline --save_dir ./your-save-dir
```
### Baseline + Augmentation
```bash
python3 main.py --num_epochs 100000 --num_seeds 3 --equi --baseline --augment --save_dir ./your-save-dir
```
### Ours
```bash
python3 main.py --num_epochs 100000 --num_seeds 3 --equi --save_dir ./your-save-dir
```
After training, you should see the following directories: `./your-save-dir/baseline_equi/``./trained_models/baseline_equi_augment/`, `./your-save-dir/ours_equi/`
Each directory contains: `model_0.pt`, `model_1.pt`, `model_2.pt` for 3 seeds and `training_loss.npy`,  `validation_error.npy` to save the learning dynamic.
We provide the pretrained models in `./trained_models/` which are ready to evaluate. 

## 📊 Evaluation (RMSD and W2)
To evaluate the trained models, run:
```
bash
python report_dist.py trained_models/baseline_equi_augment/
python report_dist.py trained_models/baseline_equi/
python report_dist.py trained_models/ours_equi/
```
This will compute the mean and standard deviation of RMSD and W2 distance as reported in the paper.

## 📉 Visualization
To visualize training losses, run:
```
bash
python draw.py \
  trained_models/baseline_equi/training_loss.npy \
  trained_models/baseline_equi_augment/training_loss.npy \
  trained_models/ours_equi/training_loss.npy
 ```
This will generate plots comparing training dynamics across different methods.