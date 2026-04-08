"""
data_generation/generate_ml_visualizations.py

Generates synthetic ML visualisation plots and saves them to data_main/raw/images/.
Metadata for every saved image is appended to data_main/processed/image_metadata.json.

Usage (from project root):
    python data_generation/generate_ml_visualizations.py
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — safe on Windows without a display
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from sklearn.datasets import make_blobs, make_circles, make_moons
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Path setup (all relative to project root — no hardcoded absolute paths)
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
IMAGES_DIR = PROJECT_ROOT / "data_main" / "raw" / "images"
METADATA_FILE = PROJECT_ROOT / "data_main" / "processed" / "image_metadata.json"

# Add project root to sys.path so utils can be imported regardless of cwd
sys.path.insert(0, str(PROJECT_ROOT))
from utils.metadata_schema import generate_unique_id, image_metadata  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def ensure_dirs() -> None:
    """Create output directories if they do not exist."""
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_FILE.parent.mkdir(parents=True, exist_ok=True)


def load_metadata() -> list:
    """Load existing metadata list from JSON, or return empty list."""
    if METADATA_FILE.exists() and METADATA_FILE.stat().st_size > 2:
        with open(METADATA_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, list) else []
    return []


def save_metadata(records: list) -> None:
    """Write the full metadata list back to JSON."""
    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)


def register_image(
    records: list,
    index: int,
    filename: str,
    caption: str,
    width: int,
    height: int,
) -> None:
    """
    Build an image_metadata record and append it to records in-place.

    Args:
        records:  Existing metadata list (mutated in-place).
        index:    Global image index used for ID generation.
        filename: PNG filename (basename only).
        caption:  Human-readable description of the plot.
        width:    Image width in pixels.
        height:   Image height in pixels.
    """
    doc_id = generate_unique_id("image", index)
    record = image_metadata(
        doc_id=doc_id,
        source_file=str(IMAGES_DIR / filename),
        format="PNG",
        width=width,
        height=height,
        caption=caption,
    )
    records.append(record)
    print(f"  [+] {filename}  ({doc_id})")


def save_figure(filename: str) -> tuple[int, int]:
    """
    Save the current matplotlib figure to IMAGES_DIR and close it.

    Returns:
        (width, height) in pixels.
    """
    fig = plt.gcf()
    dpi = fig.get_dpi()
    w_px = int(fig.get_figwidth() * dpi)
    h_px = int(fig.get_figheight() * dpi)
    filepath = IMAGES_DIR / filename
    fig.savefig(filepath, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return w_px, h_px


# ---------------------------------------------------------------------------
# Plot generators
# ---------------------------------------------------------------------------

def generate_dbscan_plots(records: list, start_index: int) -> int:
    """
    Generate one DBSCAN scatter plot per eps value using make_moons data.

    Args:
        records:     Metadata list (mutated in-place).
        start_index: Starting global image index.

    Returns:
        Next available image index after this batch.
    """
    print("\n[DBSCAN]")
    X, _ = make_moons(n_samples=300, noise=0.1, random_state=42)
    X = StandardScaler().fit_transform(X)

    eps_values = [0.2, 0.3, 0.5, 0.8]
    idx = start_index

    for eps in eps_values:
        labels = DBSCAN(eps=eps, min_samples=5).fit_predict(X)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = (labels == -1).sum()

        filename = f"dbscan_eps_{str(eps).replace('.', '_')}.png"
        caption = (
            f"DBSCAN clustering (eps={eps}): {n_clusters} cluster(s) found, "
            f"{n_noise} noise point(s). Dataset: make_moons(n=300, noise=0.1)."
        )

        fig, ax = plt.subplots(figsize=(6, 5))
        scatter = ax.scatter(X[:, 0], X[:, 1], c=labels, cmap="tab10", s=20, alpha=0.8)
        ax.set_title(f"DBSCAN  eps={eps}  |  clusters={n_clusters}  noise={n_noise}")
        ax.set_xlabel("Feature 1")
        ax.set_ylabel("Feature 2")
        plt.colorbar(scatter, ax=ax, label="Cluster label (-1 = noise)")
        plt.tight_layout()

        w, h = save_figure(filename)
        register_image(records, idx, filename, caption, w, h)
        idx += 1

    return idx


def generate_kmeans_plots(records: list, start_index: int) -> int:
    """
    Generate one KMeans scatter plot per k value using make_blobs data.

    Args:
        records:     Metadata list (mutated in-place).
        start_index: Starting global image index.

    Returns:
        Next available image index after this batch.
    """
    print("\n[KMeans]")
    X, _ = make_blobs(n_samples=400, centers=4, cluster_std=0.9, random_state=42)
    X = StandardScaler().fit_transform(X)

    k_values = [2, 3, 4, 5]
    idx = start_index

    for k in k_values:
        km = KMeans(n_clusters=k, random_state=42, n_init="auto")
        labels = km.fit_predict(X)
        centers = km.cluster_centers_

        filename = f"kmeans_{k}_clusters.png"
        caption = (
            f"KMeans clustering with k={k} on make_blobs(n=400, centers=4). "
            f"Centroids are marked with black crosses."
        )

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(X[:, 0], X[:, 1], c=labels, cmap="tab10", s=20, alpha=0.8)
        ax.scatter(
            centers[:, 0], centers[:, 1],
            c="black", marker="x", s=120, linewidths=2, label="Centroids",
        )
        ax.set_title(f"KMeans  k={k}")
        ax.set_xlabel("Feature 1")
        ax.set_ylabel("Feature 2")
        ax.legend()
        plt.tight_layout()

        w, h = save_figure(filename)
        register_image(records, idx, filename, caption, w, h)
        idx += 1

    return idx


def generate_roc_curve(records: list, start_index: int) -> int:
    """
    Generate a ROC curve plot using logistic regression on make_circles data.

    Args:
        records:     Metadata list (mutated in-place).
        start_index: Starting global image index.

    Returns:
        Next available image index after this batch.
    """
    print("\n[ROC Curve]")
    X, y = make_circles(n_samples=500, noise=0.1, factor=0.5, random_state=42)
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    clf = LogisticRegression(random_state=42)
    clf.fit(X_train, y_train)
    y_prob = clf.predict_proba(X_test)[:, 1]

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)

    filename = "roc_curve_example.png"
    caption = (
        f"ROC curve for LogisticRegression on make_circles(n=500, noise=0.1). "
        f"AUC = {auc:.3f}."
    )

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="steelblue", lw=2, label=f"ROC (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--", label="Random baseline")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — Logistic Regression")
    ax.legend(loc="lower right")
    plt.tight_layout()

    w, h = save_figure(filename)
    register_image(records, start_index, filename, caption, w, h)
    return start_index + 1


def generate_isolation_forest_plot(records: list, start_index: int) -> int:
    """
    Generate an Isolation Forest anomaly detection scatter plot on make_blobs data.

    Args:
        records:     Metadata list (mutated in-place).
        start_index: Starting global image index.

    Returns:
        Next available image index after this batch.
    """
    print("\n[Isolation Forest]")
    X, _ = make_blobs(n_samples=300, centers=2, cluster_std=0.5, random_state=42)
    rng = np.random.default_rng(0)
    outliers = rng.uniform(low=-6, high=6, size=(30, 2))
    X_all = np.vstack([X, outliers])

    iso = IsolationForest(contamination=0.09, random_state=42)
    preds = iso.fit_predict(X_all)  # 1 = inlier, -1 = outlier

    inliers = X_all[preds == 1]
    detected_outliers = X_all[preds == -1]

    filename = "isolation_forest_outliers.png"
    caption = (
        f"Isolation Forest anomaly detection: {len(detected_outliers)} outliers detected "
        f"out of {len(X_all)} points (contamination=0.09). "
        f"Dataset: make_blobs(n=300) + 30 uniform random outliers."
    )

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(inliers[:, 0], inliers[:, 1], c="steelblue", s=20, alpha=0.7, label="Inliers")
    ax.scatter(
        detected_outliers[:, 0], detected_outliers[:, 1],
        c="crimson", s=60, marker="x", linewidths=1.5, label="Outliers",
    )
    ax.set_title(f"Isolation Forest  |  {len(detected_outliers)} outliers detected")
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.legend()
    plt.tight_layout()

    w, h = save_figure(filename)
    register_image(records, start_index, filename, caption, w, h)
    return start_index + 1


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    ensure_dirs()
    records = load_metadata()
    start = len(records)  # continue indexing from wherever we left off

    print(f"Output directory : {IMAGES_DIR}")
    print(f"Metadata file    : {METADATA_FILE}")
    print(f"Existing records : {start}")

    idx = start
    idx = generate_dbscan_plots(records, idx)
    idx = generate_kmeans_plots(records, idx)
    idx = generate_roc_curve(records, idx)
    idx = generate_isolation_forest_plot(records, idx)

    save_metadata(records)

    generated = idx - start
    print(f"\n{'=' * 50}")
    print(f"Images generated : {generated}")
    print(f"Total in metadata: {len(records)}")
    print(f"Metadata saved to: {METADATA_FILE}")
    print("Done.")


if __name__ == "__main__":
    main()
