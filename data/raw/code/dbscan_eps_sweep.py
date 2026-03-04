"""
dbscan_eps_sweep.py

Sweeps over multiple eps values and visualises DBSCAN behaviour on each.
Demonstrates how eps controls cluster formation, merging, and noise levels.
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler


# ── Parameters ────────────────────────────────────────────────────────────────
N_SAMPLES    = 300
NOISE        = 0.08
MIN_SAMPLES  = 5      # Held constant during the eps sweep
RANDOM_STATE = 42

# eps sweep values — increasing eps progressively merges clusters
# and reduces noise, eventually collapsing everything into one cluster.
EPS_VALUES = [0.10, 0.20, 0.30, 0.50, 0.80, 1.20]


# ── Data ──────────────────────────────────────────────────────────────────────
def load_data(n_samples: int, noise: float, random_state: int):
    """
    Generate and standardise a two-moon dataset.

    Returns:
        X: Standardised array of shape (n_samples, 2).
    """
    X, _ = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
    return StandardScaler().fit_transform(X)


# ── Clustering ────────────────────────────────────────────────────────────────
def run_dbscan(X, eps: float, min_samples: int):
    """
    Fit DBSCAN and return labels.

    Args:
        X:           Feature matrix.
        eps:         Neighbourhood radius.
        min_samples: Core-point density threshold.

    Returns:
        labels: Integer cluster labels; -1 = noise.
    """
    return DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X)


# ── Summary helper ────────────────────────────────────────────────────────────
def cluster_summary(labels) -> dict:
    """
    Compute quick statistics for a set of cluster labels.

    Returns:
        dict with keys 'n_clusters' and 'n_noise'.
    """
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise    = int((labels == -1).sum())
    return {"n_clusters": n_clusters, "n_noise": n_noise}


# ── Sweep plot ────────────────────────────────────────────────────────────────
def plot_eps_sweep(X, eps_values: list, min_samples: int):
    """
    Create a grid of scatter plots, one per eps value.

    Each subplot title shows the eps value, number of clusters found,
    and number of noise points, allowing direct visual comparison of
    how neighbourhood radius affects cluster detection.

    Args:
        X:           Standardised feature matrix.
        eps_values:  List of eps values to evaluate.
        min_samples: Shared density threshold across all subplots.
    """
    n_cols = 3
    n_rows = int(np.ceil(len(eps_values) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, n_rows * 4))
    axes = axes.flatten()

    for i, eps in enumerate(eps_values):
        labels  = run_dbscan(X, eps, min_samples)
        summary = cluster_summary(labels)

        ax = axes[i]
        mask_noise = labels == -1

        # Grey noise points distinguish unassigned areas clearly
        ax.scatter(X[mask_noise, 0], X[mask_noise, 1],
                   c="lightgrey", s=12, label="Noise")
        ax.scatter(X[~mask_noise, 0], X[~mask_noise, 1],
                   c=labels[~mask_noise], cmap="tab10", s=12, alpha=0.85)

        ax.set_title(
            f"eps={eps}\n"
            f"clusters={summary['n_clusters']}  noise={summary['n_noise']}",
            fontsize=9
        )
        ax.set_xticks([])
        ax.set_yticks([])

    # Hide any unused subplot panels in the grid
    for j in range(len(eps_values), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        f"DBSCAN eps sweep  |  min_samples={min_samples}",
        fontsize=13, y=1.02
    )
    plt.tight_layout()
    plt.show()


# ── Numeric summary table ─────────────────────────────────────────────────────
def print_sweep_table(X, eps_values: list, min_samples: int):
    """
    Print a formatted table of clusters and noise count for each eps value.

    Args:
        X:           Feature matrix.
        eps_values:  List of eps values.
        min_samples: Shared density threshold.
    """
    print(f"\n{'eps':>8}  {'clusters':>10}  {'noise':>8}")
    print("-" * 32)
    for eps in eps_values:
        labels  = run_dbscan(X, eps, min_samples)
        summary = cluster_summary(labels)
        print(f"{eps:>8.2f}  {summary['n_clusters']:>10}  {summary['n_noise']:>8}")
    print()


# ── Entry point ────────────────────────────────────────────────────────────────
def main():
    X = load_data(N_SAMPLES, NOISE, RANDOM_STATE)
    print_sweep_table(X, EPS_VALUES, MIN_SAMPLES)
    plot_eps_sweep(X, EPS_VALUES, MIN_SAMPLES)


if __name__ == "__main__":
    main()
