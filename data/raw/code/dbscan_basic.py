"""
dbscan_basic.py

Demonstrates DBSCAN clustering on a two-moon dataset.
Shows how eps and min_samples define cluster shapes and noise detection.
"""

import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler


# ── Parameters ──────────────────────────────────────────────────────────────
N_SAMPLES = 300       # Number of data points to generate
NOISE     = 0.08      # Gaussian noise added to the moons (higher = more overlap)
EPS       = 0.3       # Neighbourhood radius — critical for cluster detection
MIN_SAMPLES = 5       # Minimum neighbours to qualify as a core point
RANDOM_STATE = 42


# ── Data generation ──────────────────────────────────────────────────────────
def load_data(n_samples: int, noise: float, random_state: int):
    """
    Generate a two-moon synthetic dataset and standardise it.

    Args:
        n_samples:    Total number of points across both moons.
        noise:        Standard deviation of Gaussian noise applied to each point.
        random_state: Seed for reproducibility.

    Returns:
        X_scaled: Standardised feature matrix of shape (n_samples, 2).
    """
    X, _ = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
    X_scaled = StandardScaler().fit_transform(X)
    return X_scaled


# ── Clustering ───────────────────────────────────────────────────────────────
def run_dbscan(X, eps: float, min_samples: int):
    """
    Fit DBSCAN and return cluster labels.

    Label -1 indicates noise points that belong to no cluster.

    Args:
        X:           Feature matrix (n_samples, n_features).
        eps:         Neighbourhood radius.  Too small → most points become noise.
                     Too large → distinct clusters merge.
        min_samples: Density threshold.  High values → only the densest cores
                     are retained.

    Returns:
        labels: Array of integer cluster labels, -1 for noise.
    """
    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(X)
    return labels


# ── Reporting ────────────────────────────────────────────────────────────────
def report(labels):
    """Print a brief cluster summary to stdout."""
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise    = (labels == -1).sum()
    print(f"Clusters found : {n_clusters}")
    print(f"Noise points   : {n_noise}")


# ── Visualisation ────────────────────────────────────────────────────────────
def plot_clusters(X, labels, eps: float, min_samples: int):
    """
    Scatter plot of DBSCAN results.

    Noise points are rendered in light grey to distinguish them visually
    from labelled cluster members.

    Args:
        X:           Standardised feature matrix.
        labels:      DBSCAN output labels.
        eps:         Used in the plot title for reference.
        min_samples: Used in the plot title for reference.
    """
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise    = (labels == -1).sum()

    fig, ax = plt.subplots(figsize=(7, 5))

    # Plot noise in grey, clusters in colour
    mask_noise = labels == -1
    ax.scatter(X[mask_noise, 0], X[mask_noise, 1],
               c="lightgrey", s=18, label="Noise", zorder=2)
    ax.scatter(X[~mask_noise, 0], X[~mask_noise, 1],
               c=labels[~mask_noise], cmap="tab10", s=18,
               alpha=0.85, label="Cluster", zorder=3)

    ax.set_title(
        f"DBSCAN  eps={eps}  min_samples={min_samples}\n"
        f"Clusters: {n_clusters}  |  Noise: {n_noise}"
    )
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.legend()
    plt.tight_layout()
    plt.show()


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    X      = load_data(N_SAMPLES, NOISE, RANDOM_STATE)
    labels = run_dbscan(X, EPS, MIN_SAMPLES)
    report(labels)
    plot_clusters(X, labels, EPS, MIN_SAMPLES)


if __name__ == "__main__":
    main()
