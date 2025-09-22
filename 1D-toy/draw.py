import sys
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# Utility Functions
# -------------------------------

def moving_average_same_length(data, window_size: int):
    """Compute a moving average with same length as input data."""
    pad_size = window_size // 2
    padded_data = np.pad(data, pad_size, mode="edge")
    smoothed = np.convolve(padded_data, np.ones(window_size) / window_size, mode="valid")
    return smoothed[:len(data)]

def format_to_k(x, _):
    """Format x-axis ticks in 'k' notation."""
    return "Iter=0k" if x == 0 else f"{int(x // 1000)}k"

def plot_with_log(ylabel, output_path, ours, baseline, augment, log_scale=False):
    """Plot comparison with optional log scaling."""
    plt.figure()
    plt.plot(epochs, baseline, label="EquiNet", color="#81b29a")
    plt.plot(epochs, augment, label="+ [Aug]", color="#3d405b")
    plt.plot(epochs, ours, label="+ [OrbDiff]", color="#e07a5f")

    if log_scale:
        plt.yscale("log")

    ax = plt.gca()
    ax.xaxis.set_major_formatter(plt.FuncFormatter(format_to_k))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.xlabel("Training iterations")
    plt.ylabel(ylabel)
    plt.legend(frameon=False)  # no border
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

# -------------------------------
# Plot Settings
# -------------------------------

W = 0.35 * 13.968 / 2.54  # figure width scaling
plt.rcParams.update({
    "font.size": 7,
    "axes.titlesize": 7,
    "axes.labelsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "figure.figsize": (W, W * 0.8),
    "text.usetex": False,
    "font.family": "DejaVu Serif",   # closest to LaTeX CM
    "mathtext.fontset": "stix"
})

# -------------------------------
# Load Data
# -------------------------------

(
    baseline_loss_path,
    augment_loss_path,
    ours_loss_path,
) = sys.argv[1:4]

baseline_losses = np.load(baseline_loss_path)[:, 1::2]
augment_losses = np.load(augment_loss_path)[:, 1::2]
ours_losses     = np.load(ours_loss_path)[:, 1::2]

# -------------------------------
# Compute Metrics
# -------------------------------

def compute_stats(arr):
    return (
        moving_average_same_length(arr.mean(axis=0), 2),
        moving_average_same_length(arr.std(axis=0), 2),
    )

losses_mean, _           = compute_stats(ours_losses)
baseline_losses_mean, _  = compute_stats(baseline_losses)
augment_losses_mean, _   = compute_stats(augment_losses)
epochs = np.arange(len(losses_mean)) * 500

# -------------------------------
# Generate Plots
# -------------------------------

plot_with_log(
    "Training Loss",
    "viz.pdf",
    losses_mean,
    baseline_losses_mean,
    augment_losses_mean,
)

# python3 draw.py trained_models/baseline_equi/validation_error.npy trained_models/baseline_equi/training_loss.npy trained_models/baseline_equi_augment/validation_error.npy trained_models/baseline_equi_augment/training_loss.npy trained_models/ours_equi/validation_error.npy trained_models/ours_equi/training_loss.npy 