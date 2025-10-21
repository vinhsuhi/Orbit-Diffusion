# Orbit-Diffusion  
**Accepted at NeurIPS 2025**  

Implementation of *“Rao-Blackwell Gradient Estimators for Equivariant Denoising Diffusion”*  
Vinh Tong, Trung-Dung Hoang, Anji Liu, Guy Van den Broeck, Mathias Niepert  
[arXiv:2502.09890v4](https://arxiv.org/abs/2502.09890)


## Overview  
Generative modelling in domains such as molecules, crystals and proteins often involves symmetries (e.g., rotations, reflections). Two common strategies to handle them are:  
- designing equivariant network architectures, or  
- using data augmentation to approximate symmetry invariance.  
This work reframes augmentation-based training as a Monte Carlo gradient estimator, then applies **Rao-Blackwellization** to analytically integrate out symmetry transformations. The result: a lower-variance gradient estimator, requiring only a single forward/backward pass per sample, with provable equivariant minimizers. (See the paper for full theory.)  

## Core Contribution  
- A novel gradient estimator for equivariant denoising diffusion models that reduces variance via Rao-Blackwellization.  
- A practical algorithmic framework (“Orbit Diffusion”) that incorporates this estimator into diffusion training and sampling.  
- Empirical results showing improved performance on tasks such as molecular conformation generation (GEOM-QM9), crystal structure prediction (Perov-5, MP-20), and protein generation.  

## Repository Structure  

```bash
Orbit-Diffusion/
├── 1D-toy/              # Toy experiments
├── DiffCSP/             # Crystal structure prediction experiments
├── IEF/                 # Core estimator code
├── TGDMat/              # Dataset / tools
├── requirements.txt
└── README.md            # This file
```

Each subfolder contains its own **README** and **environment setup**.  
➡️ **Please go into each subfolder and follow its specific installation instructions** before running experiments.

## Citation 

```
@inproceedings{tong2024learning,
  title     = {Rao-Blackwell Gradient Estimators for Equivariant Denoising Diffusion},
  author    = {Tong, Vinh and Hoang, Trung-Dung and Liu, Anji and Van den Broeck, Guy and Niepert, Mathias},
  booktitle = {Proceedings of the 39th Conference on Neural Information Processing Systems},
  year      = {2025}
}
```

```
@article{tong2025raoblackwell,
  title   = {Rao-Blackwell Gradient Estimators for Equivariant Denoising Diffusion},
  author  = {Tong, Vinh and Hoang, Trung-Dung and Liu, Anji and Van den Broeck, Guy and Niepert, Mathias},
  journal = {arXiv preprint arXiv:2502.09890 v4},
  year    = {2025}
}
```
