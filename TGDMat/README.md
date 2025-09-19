# OrbDiff for TGDMat
Implementation of **TGDMat** adapted for [Periodic Materials Generation using Text-Guided Joint Diffusion Model (ICLR 2025)](https://arxiv.org/pdf/2503.00522).
---
### 🚀 Dependencies and Setup
Please refer to the original [TGDMat repository](https://github.com/kdmsit/TGDMat/).

### 🔧 Training
To train the model, run:
```
bash run.sh <dataset> <prompt_type> <number-of-transition> <wn_noise>
```
-  `dataset`: `perov_5` or `mp_20`
-  `number-of-transition`: the number of transitions
-  `prompt_type`: `long` or `short`
-  `wn_noise`: `True` for Wrapped Normal (WN), `False` for Uniform (U)
    - For baseline runs, set `number-of-transition`=0 and `wn_noise` is ignored.
The model, logs, and outputs will be saved in `./csp_task/out/${dataset}_${prompt_type}_trans${num_trans}_[U or WN]_ddmmyyyyHHMMSS` for OrbDiff and `./csp_task/out/${dataset}_baseline_ddmmyyyyHHMMSS` for baseline.

#### Example runs
- OrbDiff:
```
# Wrapped Normal noise
bash run.sh perov_5 short 1000 True
# Uniform noise
bash run.sh perov_5 short 1000 False
```
- Baseline:
```
bash run.sh perov_5 short 0
``` 
Our pre-trained OrbDiff checkpoints are provided [here]().

### 📊 Evaluation
To evaluate a trained model, run:
```
bash eval.sh <root_path> <dataset> <label>
```
- `root_path`: absolute path to trained model directory
- `label`: string to differentiate different evaluations
The output will be saved in `<root_path>/eval_recon_<label>.pt` and metríc will be saved in `<root_path>/eval_recon_<label>.txt`