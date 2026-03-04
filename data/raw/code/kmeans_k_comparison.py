"""
kmeans_k_comparison.py

Sweeps over multiple k values to demonstrate the elbow method and silhouette
analysis for selecting the optimal number of KMeans clusters.
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


# ── Parameters ────────────────────────────────────────────────────────────────
N_SAMPLES    = 400
N_CENTERS    = 4      # True blob count — optimal k should surface near this value
CLUSTER_STD  = 0.85
K_RANGE      = range(2, 10)   # Range of k values to evaluate
N_INIT       = 10
RANDOM_STATE = 42


# ── Data ──────────────────────────────────────────────────────────────────────
def load_data(n_samples: int, n_centers: int, cluster_std: float, random_state: int):
    """
    Generate and standardise a blob dataset.

    Returns:
        X: Standardised feature matrix (n_samples, 2).
    """
    X, _ = make_blobs(
        n_samples=n_samples,
        centers=n_centers,
        cluster_std=cluster_std,
        random_state=random_state,
    )
    return StandardScaler().fit_transform(X)


# ── K sweep ───────────────────────────────────────────────────────────────────
def k_sweep(X, k_range, n_init: int, random_state: int) -> dict:
    """
    Fit KMeans for each k; record inertia and silhouette score.

    Returns:
        dict with keys 'k', 'inertia', 'silhouette'.
    """
    ks, inertias, silhouettes = [], [], []

    for k in k_range:
        km     = KMeans(n_clusters=k, n_init=n_init,
                        random_state=random_state)
        labels = km.fit_predict(X)

        ks.append(k)
        inertias.append(km.inertia_)
        # Silhouette requires at least 2 clusters and mixed labels
        silhouettes.append(silhouette_score(X, labels))

    return {"k": ks, "inertia": inertias, "silhouette": silhouettes}


# ── Scatter grid ──────────────────────────────────────────────────────────────
def plot_cluster_grid(X, k_values, n_init: int, random_state: int):
    """Grid of scatter plots for each k. Centroids marked with black crosses."""
    n_cols = 4
    n_rows = int(np.ceil(len(k_values) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, n_rows * 3.5))
    axes = axes.flatten()

    for i, k in enumerate(k_values):
        km     = KMeans(n_clusters=k, n_init=n_init, random_state=random_state)
        labels = km.fit_predict(X)
        centers = km.cluster_centers_

        ax = axes[i]
        ax.scatter(X[:, 0], X[:, 1], c=labels, cmap="tab10",
                   s=10, alpha=0.75)
        ax.scatter(centers[:, 0], centers[:, 1],
                   c="black", marker="X", s=80, zorder=5)
        ax.set_title(f"k={k}  inertia={km.inertia_:.1f}", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])

    for j in range(len(k_values), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("KMeans cluster comparison", fontsize=12)
    plt.tight_layout()
    plt.show()


# ── Elbow + silhouette plots ───────────────────────────────────────────────────
def plot_elbow_and_silhouette(results: dict, true_k: int):
    """
    Side-by-side elbow (inertia) and silhouette score plots.

    Red dashed line marks the known true cluster count for reference.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Elbow plot — look for the kink where inertia stops falling steeply
    ax1.plot(results["k"], results["inertia"], marker="o", color="steelblue")
    ax1.axvline(true_k, color="red", linestyle="--", label=f"True k={true_k}")
    ax1.set_title("Elbow Method — Inertia vs k")
    ax1.set_xlabel("Number of clusters k")
    ax1.set_ylabel("Inertia")
    ax1.legend()

    # Silhouette — peak marks the k with best internal cohesion vs separation
    ax2.plot(results["k"], results["silhouette"], marker="o", color="darkorange")
    ax2.axvline(true_k, color="red", linestyle="--", label=f"True k={true_k}")
    ax2.set_title("Silhouette Score vs k")
    ax2.set_xlabel("Number of clusters k")
    ax2.set_ylabel("Silhouette score")
    ax2.legend()

    plt.tight_layout()
    plt.show()


# ── Entry point ────────────────────────────────────────────────────────────────
def main():
    X       = load_data(N_SAMPLES, N_CENTERS, CLUSTER_STD, RANDOM_STATE)
    results = k_sweep(X, K_RANGE, N_INIT, RANDOM_STATE)
    plot_cluster_grid(X, list(K_RANGE), N_INIT, RANDOM_STATE)
    plot_elbow_and_silhouette(results, true_k=N_CENTERS)


if __name__ == "__main__":
    main()
