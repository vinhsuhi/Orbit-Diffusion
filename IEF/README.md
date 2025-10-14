# OrbDiff for ETFlow
Implementation of **OrbDiff** adapted for [Equivariant Flow Matching for Molecule Conformer Generation](https://arxiv.org/abs/2410.22388).

---

### 🚀 Dependencies and Setup
Please refer to the original [ETFlow repository](https://github.com/shenoynikhil/ETFlow).

### 📦 Training Data
Training data preparation follows the same process as described in the [ETFlow repository](https://github.com/shenoynikhil/ETFlow).

### 🔧 Training
Please make sure you have exported `DATA_DIR` in the previous step. 
This implementation fine-tunes the pre-trained ETFlow model.  
Please download the pre-trained checkpoint `qm9-o3.ckpt` from [Zenodo](https://zenodo.org/records/14226681) and place it in the `./pretrained` directory.
To train the model, run:

```
bash run.sh configs/qm9-o3.yaml <training-mode> [<number-of-rotation> <number-of-permutation>]
```

- `training-mode`: `baseline` (without OrbDiff) or `ief` (with OrbDiff)
-  `number-of-rotation`: the number of rotations
-  `number-of-permutation`: the number of permutations
If `training-mode` is `baseline`, `<number-of-rotation>` and `<number-of-permutation>` will be ignored.
The model will be saved in 
- `./logs/baseline-flow/qm9-o3-harmonic-sigma0.1harmonic/runs/yyyy-mm-dd_HH-MM-SS/checkpoint/*ckpt` for baseline 
- `./logs/ief-z0.4-${number-of-permutation}perms-${number-of-rotation}rots-flow/qm9-o3-harmonic-sigma0.1harmonic/runs/yyyy-mm-dd_HH-MM-SScheckpoint/*ckpt` for OrbDiff.

#### Example runs

- OrbDiff:

```
bash run.sh configs/qm9-o3.yaml ief 200 50
```

- Baseline:

```
bash run.sh configs/qm9-o3.yaml baseline
```

Our pre-trained OrbDiff checkpoints are provided [here](https://drive.google.com/drive/folders/1OLgCh1zfIvp94M8NCUo6RKO6VTUZfn0_?usp=sharing).

  

### 📊 Evaluation

To evaluate a trained model, run:

```
bash eval.sh qm9-o3.yaml <path-to-checkpoint-file>
```
The script will generate molecular conformers using ODE with 50 and 100 steps.
The generated data will be saved in `./logs/samples/` and evaluation metrics will be saved in `./results`