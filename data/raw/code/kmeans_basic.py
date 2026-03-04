"""
kmeans_basic.py

Demonstrates KMeans clustering on a synthetic blob dataset.
Shows centroid positions, cluster assignments, and inertia.
"""

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler


# ── Parameters ────────────────────────────────────────────────────────────────
N_SAMPLES    = 400
N_FEATURES   = 2
N_CENTERS    = 4     # True number of blobs in the data
CLUSTER_STD  = 0.85  # Spread of each blob. Higher values cause overlap.
K            = 4     # Number of clusters to fit — should match N_CENTERS ideally
MAX_ITER     = 300   # Maximum optimisation iterations before forced stop
N_INIT       = 10    # Times the algorithm runs from different random seeds;
                     # best result by inertia is kept
RANDOM_STATE = 42


# ── Data ──────────────────────────────────────────────────────────────────────
def load_data(n_samples: int, n_centers: int, cluster_std: float, random_state: int):
    """
    Generate a labelled blob dataset and standardise it.

    Args:
        n_samples:   Total data points across all blobs.
        n_centers:   Number of distinct blob centres.
        cluster_std: Standard deviation of each blob.
        random_state: Seed for reproducibility.

    Returns:
        X_scaled: Standardised feature array (n_samples, 2).
        y_true:   Ground-truth blob labels (for comparison only).
    """
    X, y = make_blobs(
        n_samples=n_samples,
        centers=n_centers,
        cluster_std=cluster_std,
        random_state=random_state,
    )
    X_scaled = StandardScaler().fit_transform(X)
    return X_scaled, y


# ── Clustering ────────────────────────────────────────────────────────────────
def run_kmeans(X, k: int, max_iter: int, n_init: int, random_state: int):
    """
    Fit KMeans and return the fitted model.

    Args:
        X:           Feature matrix.
        k:           Number of cluster centroids.
        max_iter:    Hard stop after this many iterations if not converged.
        n_init:      Runs with different random centroid seeds;
                     keeps run with lowest inertia.
        random_state: Seed for reproducible centroid initialisation.

    Returns:
        km: Fitted KMeans instance with attributes labels_, cluster_centers_,
            and inertia_.
    """
    km = KMeans(
        n_clusters=k,
        max_iter=max_iter,
        n_init=n_init,
        random_state=random_state,
    )
    km.fit(X)
    return km


# ── Reporting ─────────────────────────────────────────────────────────────────
def report(km):
    """Print inertia and iteration count to stdout."""
    print(f"Inertia (within-cluster SSE) : {km.inertia_:.4f}")
    print(f"Iterations to convergence    : {km.n_iter_}")


# ── Visualisation ──────────────────────────────────────────────────────────────
def plot_clusters(X, km):
    """
    Plot KMeans cluster assignments and centroid markers.

    Centroids are drawn as black crosses so they stand out against the
    coloured point cloud.  Inertia is shown in the title for quick reference.

    Args:
        X:  Standardised feature matrix.
        km: Fitted KMeans model.
    """
    labels   = km.labels_
    centers  = km.cluster_centers_

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(X[:, 0], X[:, 1], c=labels, cmap="tab10",
               s=20, alpha=0.8, label="Data points")
    ax.scatter(centers[:, 0], centers[:, 1],
               c="black", marker="X", s=150, linewidths=1.5,
               zorder=5, label="Centroids")

    ax.set_title(
        f"KMeans  k={km.n_clusters}  |  inertia={km.inertia_:.2f}"
    )
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.legend()
    plt.tight_layout()
    plt.show()


# ── Entry point ────────────────────────────────────────────────────────────────
def main():
    X, _ = load_data(N_SAMPLES, N_CENTERS, CLUSTER_STD, RANDOM_STATE)
    km    = run_kmeans(X, K, MAX_ITER, N_INIT, RANDOM_STATE)
    report(km)
    plot_clusters(X, km)


if __name__ == "__main__":
    main()
