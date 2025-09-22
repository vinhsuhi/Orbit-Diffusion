import re
import sys
import random
import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path

from main import *


# -------------------------------
# Helpers
# -------------------------------

def compute_w2_distance(samples: np.ndarray, n_samples: int = 100000) -> float:
    """Compute the Wasserstein-2 distance between samples and balanced ±1 target."""
    sorted_target = np.concatenate([-np.ones(n_samples // 2), np.ones(n_samples // 2)])
    sorted_samples = np.sort(samples)
    return np.sqrt(np.mean((sorted_target - sorted_samples) ** 2))


def find_model_files(model_dir: str):
    pattern = re.compile(r"model_\d+\.pt$")
    dir_path = Path(model_dir)
    
    return [
        str(file)
        for file in dir_path.rglob("*.pt")  # search recursively for .pt files
        if pattern.search(file.name)
    ]


# -------------------------------
# Main
# -------------------------------

def main(model_dir: str):
    NUM_TIMESTEPS = 100
    N_SAMPLES = 100000
    BATCH_SIZE = 1000
    SEED = 42

    # diffusion params
    sigmas, alphas = initialize_diffusion_params(NUM_TIMESTEPS)

    # checkpoints
    model_paths = find_model_files(model_dir)
    if not model_paths:
        print("No valid checkpoints found.")
        return

    w2_distances, mean_abss = [], []

    for model_path in model_paths:
        print(f"Evaluating {model_path}...")
        model = torch.load(model_path)

        # fix seeds for reproducibility
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        random.seed(SEED)

        generated_samples_list = []

        for _ in tqdm(range(N_SAMPLES // BATCH_SIZE)):
            init = torch.cat([
                torch.abs(torch.randn(BATCH_SIZE // 2, 1)),
                -torch.abs(torch.randn(BATCH_SIZE // 2, 1)),
            ]).cuda()

            gen = sample(model, init, alphas, sigmas, NUM_TIMESTEPS)
            generated_samples_list.append(gen.cpu().flatten().numpy())

        generated_samples = np.concatenate(generated_samples_list)

        mean_abs = np.mean(np.abs(generated_samples - np.sign(generated_samples)))
        w2_distance = compute_w2_distance(generated_samples, N_SAMPLES)

        mean_abss.append(mean_abs)
        w2_distances.append(w2_distance)

    # results
    mean_abs_mean, mean_abs_std = np.mean(mean_abss), np.std(mean_abss)
    w2_mean, w2_std = np.mean(w2_distances), np.std(w2_distances)

    print("RMSD:", f"{mean_abs_mean * 1e5:.2f} ± {mean_abs_std * 1e5:.2f}", "×10e-5")
    print("W2 distance:", f"{w2_mean * 1e3:.3f} ± {w2_std * 1e3:.3f}", "×10e-3")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} <model_dir>")
        sys.exit(1)
    main(sys.argv[1])
