"""
logistic_roc_curve.py

Trains logistic regression on a binary classification dataset and generates
a full ROC curve with AUC, illustrating threshold-independent evaluation.
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler


# ── Parameters ────────────────────────────────────────────────────────────────
N_SAMPLES      = 600
N_FEATURES     = 6      # Richer space → smoother probability estimates
N_INFORMATIVE  = 4
N_REDUNDANT    = 1
CLASS_SEP      = 0.9    # Moderate separation keeps AUC in an informative range
C              = 1.0    # L2 regularisation inverse
SOLVER         = "lbfgs"
MAX_ITER       = 300
TEST_SIZE      = 0.3
N_CV_FOLDS     = 5
RANDOM_STATE   = 42


# ── Data ──────────────────────────────────────────────────────────────────────
def load_data(n_samples, n_features, n_informative, n_redundant,
              class_sep, test_size, random_state):
    """Generate a binary dataset; standardise and return a train/test split."""
    X, y = make_classification(
        n_samples=n_samples, n_features=n_features,
        n_informative=n_informative, n_redundant=n_redundant,
        class_sep=class_sep, random_state=random_state,
    )
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    return X_train, X_test, y_train, y_test, X, y


# ── Model ─────────────────────────────────────────────────────────────────────
def train(X_train, y_train, C, solver, max_iter, random_state):
    """Fit logistic regression. Small C = strong L2 regularisation."""
    clf = LogisticRegression(C=C, solver=solver,
                             max_iter=max_iter, random_state=random_state)
    clf.fit(X_train, y_train)
    return clf


# ── Single ROC plot ───────────────────────────────────────────────────────────
def plot_single_roc(clf, X_test, y_test):
    """
    Plot ROC curve with AUC and average precision annotations.

    Curve bowing toward the upper-left reflects stronger ranking ability.
    """
    y_prob = clf.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)
    ap  = average_precision_score(y_test, y_prob)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, lw=2, color="steelblue",
            label=f"Logistic Reg (AUC={auc:.3f}, AP={ap:.3f})")
    ax.plot([0, 1], [0, 1], lw=1, linestyle="--",
            color="grey", label="Random baseline")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate (Recall)")
    ax.set_title("ROC Curve — Logistic Regression")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.show()
    return auc, ap


# ── Cross-validated mean ROC ──────────────────────────────────────────────────
def plot_cv_roc(X, y, C, solver, max_iter, n_folds, random_state):
    """
    Overlay per-fold ROC curves and the mean curve from stratified CV.

    AUC variance across folds reveals generalisation stability.
    """
    cv   = StratifiedKFold(n_splits=n_folds, shuffle=True,
                           random_state=random_state)
    mean_fpr = np.linspace(0, 1, 100)
    tprs, aucs = [], []

    fig, ax = plt.subplots(figsize=(7, 5))

    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        clf = LogisticRegression(C=C, solver=solver,
                                 max_iter=max_iter, random_state=random_state)
        clf.fit(X[train_idx], y[train_idx])
        y_prob = clf.predict_proba(X[test_idx])[:, 1]
        fpr, tpr, _ = roc_curve(y[test_idx], y_prob)
        auc_fold = roc_auc_score(y[test_idx], y_prob)
        aucs.append(auc_fold)

        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        ax.plot(fpr, tpr, lw=0.8, alpha=0.4,
                label=f"Fold {fold+1} (AUC={auc_fold:.2f})")

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = np.mean(aucs)
    std_auc  = np.std(aucs)

    ax.plot(mean_fpr, mean_tpr, color="navy", lw=2,
            label=f"Mean ROC (AUC={mean_auc:.2f} ± {std_auc:.2f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"Cross-validated ROC  ({n_folds} folds)")
    ax.legend(loc="lower right", fontsize=7)
    plt.tight_layout()
    plt.show()


# ── Entry point ────────────────────────────────────────────────────────────────
def main():
    X_train, X_test, y_train, y_test, X_all, y_all = load_data(
        N_SAMPLES, N_FEATURES, N_INFORMATIVE, N_REDUNDANT,
        CLASS_SEP, TEST_SIZE, RANDOM_STATE
    )
    clf = train(X_train, y_train, C, SOLVER, MAX_ITER, RANDOM_STATE)
    auc, ap = plot_single_roc(clf, X_test, y_test)
    print(f"Hold-out AUC : {auc:.4f}   |   AP : {ap:.4f}")
    plot_cv_roc(X_all, y_all, C, SOLVER, MAX_ITER, N_CV_FOLDS, RANDOM_STATE)


if __name__ == "__main__":
    main()
