
# OrbDiff for DiffCSP
Implementation of **OrbDiff** adapted for [Crystal Structure Prediction by Joint Equivariant Diffusion (NeurIPS 2023)](https://arxiv.org/abs/2309.04475).
---
### 🚀 Dependencies and Setup
We provide a convenient setup script:
```bash
conda create --name diffcsp python=3.8.13 -y
conda activate diffcsp
bash setup.sh
```
For more details, you can refer to the original [DiffCSP repository](https://github.com/jiaor17/DiffCSP).

### 🔧 Training
To train the model, run:
```
bash run.sh <dataset> <number-of-transition> <wn_noise>
```
-  `dataset`: `perov_5` or `mp_20`
-  `number-of-transition`: 1000 - the number of transitions
-  `wn_noise`: `True` for Wrapped Normal (WN), `False` for Uniform (U)

The model, logs, and outputs will be saved in `./singlerun/YYYY-MM-DD/${dataset}_trans${num_trans}_[U or WN]_YY-MM-DD-HH-MM-SS`
Pre-trained checkpoints are provided [here]().

### 📊 Evaluation
To evaluate a trained model, run:
```
bash eval.sh <root_path> <dataset> <label>
```
- `root_path`: absolute path to trained model directory
- `dataset`: `perov_5` or `mp_20`
- `label`: string to differentiate different evaluations
The result will be saved in `<root_path>/<label>.out`