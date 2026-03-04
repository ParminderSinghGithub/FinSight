"""
isolation_forest_basic.py

Demonstrates Isolation Forest anomaly detection on a mixed dataset containing
dense inlier clusters plus randomly scattered outlier points.
Visualises detection results and anomaly score distribution.
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler


# ── Parameters ────────────────────────────────────────────────────────────────
N_INLIERS      = 350   # Structured blob inlier points
N_OUTLIERS     = 35    # Uniform random outlier points
N_CENTERS      = 3     # Inlier cluster count
CONTAMINATION  = 0.09  # Expected outlier fraction. Too low → missed outliers;
                       # too high → inliers falsely flagged at cluster edges.
N_ESTIMATORS   = 100   # Isolation trees. More → stable scores (diminishing >200).
MAX_SAMPLES    = "auto"  # Sub-sample per tree; "auto" = min(256, n).
RANDOM_STATE   = 42


# ── Data ──────────────────────────────────────────────────────────────────────
def load_data(n_inliers: int, n_outliers: int, n_centers: int,
              random_state: int):
    """
    Combine structured blob inliers with uniform random outliers.

    Returns:
        X:      Standardised feature matrix (n_inliers+n_outliers, 2).
        y_true: Ground-truth labels: 1=inlier, -1=outlier.
    """
    X_in, _ = make_blobs(n_samples=n_inliers, centers=n_centers,
                         cluster_std=0.5, random_state=random_state)
    rng = np.random.default_rng(random_state)
    X_out = rng.uniform(low=-8, high=8, size=(n_outliers, 2))

    X = np.vstack([X_in, X_out])
    y_true = np.array([1] * n_inliers + [-1] * n_outliers)
    X = StandardScaler().fit_transform(X)
    return X, y_true


# ── Model ─────────────────────────────────────────────────────────────────────
def run_isolation_forest(X, contamination, n_estimators,
                         max_samples, random_state):
    """
    Fit Isolation Forest and return labels (1/-1) and anomaly scores.

    Lower decision-function score = shorter average isolation path = more anomalous.
    """
    iso = IsolationForest(
        contamination=contamination,
        n_estimators=n_estimators,
        max_samples=max_samples,
        random_state=random_state,
    )
    labels = iso.fit_predict(X)
    scores = iso.decision_function(X)
    return labels, scores


# ── Reporting ─────────────────────────────────────────────────────────────────
def report(labels, y_true):
    """Print detection counts and simple precision/recall stats."""
    detected    = (labels == -1).sum()
    true_pos    = ((labels == -1) & (y_true == -1)).sum()
    false_pos   = ((labels == -1) & (y_true == 1)).sum()
    false_neg   = ((labels == 1)  & (y_true == -1)).sum()
    precision   = true_pos / detected         if detected   > 0 else 0.0
    recall      = true_pos / (y_true == -1).sum()

    print(f"Outliers detected  : {detected}")
    print(f"True positives     : {true_pos}")
    print(f"False positives    : {false_pos}")
    print(f"False negatives    : {false_neg}")
    print(f"Precision          : {precision:.3f}")
    print(f"Recall             : {recall:.3f}")


# ── Visualisation ──────────────────────────────────────────────────────────────
def plot_detections(X, labels, contamination: float):
    """Scatter plot of inliers (blue) and detected outliers (red)."""
    inliers  = X[labels == 1]
    outliers = X[labels == -1]
    n_out    = len(outliers)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(inliers[:, 0],  inliers[:, 1],
               c="steelblue", s=18, alpha=0.75, label=f"Inliers ({len(inliers)})")
    ax.scatter(outliers[:, 0], outliers[:, 1],
               c="crimson", marker="x", s=60, linewidths=1.5,
               label=f"Outliers ({n_out})")
    ax.set_title(f"Isolation Forest  |  contamination={contamination}")
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_score_distribution(scores, labels):
    """
    Histogram of anomaly scores by class.

    Well-separated distributions indicate clean inlier/outlier distinction.
    Overlap near zero suggests boundary-sensitive points.
    """
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(scores[labels == 1],  bins=30, alpha=0.65,
            color="steelblue", label="Inliers")
    ax.hist(scores[labels == -1], bins=15, alpha=0.65,
            color="crimson",   label="Outliers")
    ax.axvline(0, color="black", linestyle="--", lw=1, label="Decision threshold")
    ax.set_xlabel("Anomaly score (decision function)")
    ax.set_ylabel("Count")
    ax.set_title("Isolation Forest — Score Distribution")
    ax.legend()
    plt.tight_layout()
    plt.show()


# ── Entry point ────────────────────────────────────────────────────────────────
def main():
    X, y_true       = load_data(N_INLIERS, N_OUTLIERS, N_CENTERS, RANDOM_STATE)
    labels, scores  = run_isolation_forest(
        X, CONTAMINATION, N_ESTIMATORS, MAX_SAMPLES, RANDOM_STATE
    )
    report(labels, y_true)
    plot_detections(X, labels, CONTAMINATION)
    plot_score_distribution(scores, labels)


if __name__ == "__main__":
    main()
