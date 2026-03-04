"""
logistic_regression_basic.py

Trains a logistic regression classifier on a synthetic binary dataset and
visualises the decision boundary alongside class probability contours.
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ── Parameters ────────────────────────────────────────────────────────────────
N_SAMPLES      = 500
N_FEATURES     = 2
N_INFORMATIVE  = 2    # All features carry signal (no redundant features)
N_REDUNDANT    = 0
CLASS_SEP      = 1.2  # Higher = more linearly separable classes
C              = 1.0  # Inverse regularisation strength.
                      # Low C → strong regularisation → smaller coefficients.
                      # High C → weak regularisation → follows data more tightly.
SOLVER         = "lbfgs"   # Efficient quasi-Newton solver for small/medium data
MAX_ITER       = 200
TEST_SIZE      = 0.25
RANDOM_STATE   = 42


# ── Data ──────────────────────────────────────────────────────────────────────
def load_data(n_samples, n_features, n_informative, n_redundant,
              class_sep, test_size, random_state):
    """
    Generate a binary classification dataset, standardise, and split it.

    Returns:
        X_train, X_test, y_train, y_test: Train/test splits.
        scaler: Fitted StandardScaler (for transforming the decision grid later).
    """
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=n_redundant,
        class_sep=class_sep,
        random_state=random_state,
    )
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test, scaler


# ── Training ──────────────────────────────────────────────────────────────────
def train(X_train, y_train, C: float, solver: str, max_iter: int, random_state: int):
    """
    Fit a logistic regression model.

    Args:
        X_train:     Training features.
        y_train:     Training labels.
        C:           Regularisation inverse. Smaller = stronger penalty on weights.
        solver:      Optimisation algorithm (lbfgs, saga, liblinear, etc.).
        max_iter:    Convergence iteration cap.
        random_state: Seed for reproducibility.

    Returns:
        clf: Fitted LogisticRegression instance.
    """
    clf = LogisticRegression(C=C, solver=solver,
                             max_iter=max_iter, random_state=random_state)
    clf.fit(X_train, y_train)
    return clf


# ── Reporting ─────────────────────────────────────────────────────────────────
def report(clf, X_test, y_test):
    """Print coefficient magnitudes and a full classification report."""
    print("Intercept  :", clf.intercept_)
    print("Coefficients:", clf.coef_)
    print()
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred, digits=3))


# ── Visualisation ──────────────────────────────────────────────────────────────
def plot_decision_boundary(clf, X, y, C: float):
    """
    Plot class assignments, the decision boundary, and probability contours.

    The boundary is the set of points where the model is exactly 50 %
    confident.  The soft colour gradient shows how probability varies
    across the feature space, helping assess calibration and margin width.

    Args:
        clf: Fitted LogisticRegression.
        X:   Full standardised feature matrix.
        y:   True labels.
        C:   Shown in title for reference.
    """
    h = 0.02   # Step size for the decision grid
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    grid   = np.c_[xx.ravel(), yy.ravel()]
    proba  = clf.predict_proba(grid)[:, 1].reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(7, 5))
    # Probability contour background
    cf = ax.contourf(xx, yy, proba, levels=50, cmap="RdBu_r", alpha=0.65)
    plt.colorbar(cf, ax=ax, label="P(class=1)")
    # Decision boundary at probability = 0.5
    ax.contour(xx, yy, proba, levels=[0.5], colors="black",
               linewidths=1.5, linestyles="--")
    # Data points
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap="bwr",
               edgecolors="k", linewidths=0.3, s=20, alpha=0.9)
    ax.set_title(f"Logistic Regression  C={C}")
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    plt.tight_layout()
    plt.show()


# ── Entry point ────────────────────────────────────────────────────────────────
def main():
    X_train, X_test, y_train, y_test, scaler = load_data(
        N_SAMPLES, N_FEATURES, N_INFORMATIVE, N_REDUNDANT,
        CLASS_SEP, TEST_SIZE, RANDOM_STATE
    )
    clf = train(X_train, y_train, C, SOLVER, MAX_ITER, RANDOM_STATE)
    report(clf, X_test, y_test)
    X_all = np.vstack([X_train, X_test])
    y_all = np.concatenate([y_train, y_test])
    plot_decision_boundary(clf, X_all, y_all, C)


if __name__ == "__main__":
    main()
