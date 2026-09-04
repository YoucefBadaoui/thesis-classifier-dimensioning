"""Seed and fold dispersion of the ISCX anchor classifiers, clean condition.

Reproduces the clean-condition protocol of notebooks/02_classifiers.ipynb and scripts/iscx/retrain_mlp_weighted.py on data/processed/iscx_5class_15s_clean.csv: stratified five-fold cross-validation with the scaler fit inside each training fold, XGBoost with balanced per-sample weights, and the MLP on a class-balanced random oversample of the training fold. Across seeds, the split is held at random_state 42 and only the XGBoost model seed varies over SEEDS; balanced accuracy is the diagonal mean of the fold-averaged row-normalised matrix. Across folds, per-fold balanced accuracy at the anchor seed is recorded for both models.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "src").is_dir())
sys.path.insert(0, str(ROOT))

from src.analytical.constants import CLASS_ORDER_OTT
from src.iscx.config import FEATURE_COLS
from src.analytical.kaufman_roberts import row_normalise

CSV = ROOT / "data" / "processed" / "iscx_5class_15s_clean.csv"
OUT = ROOT / "data" / "processed" / "iscx_seed_sweep.npz"

ANCHOR_SEED = 42
SEEDS = [42, 7, 123, 256, 1024]
N_SPLITS = 5

CLASS_ORDER = list(CLASS_ORDER_OTT)
K = len(CLASS_ORDER)

def make_xgb(seed: int, device: str) -> XGBClassifier:
    return XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8, eval_metric="mlogloss",
        device=device, random_state=seed, n_jobs=-1, verbosity=0,
    )


def make_mlp(seed: int) -> MLPClassifier:
    return MLPClassifier(
        hidden_layer_sizes=(256, 128, 64), activation="relu", solver="adam",
        alpha=1e-4, batch_size=256, learning_rate="adaptive",
        learning_rate_init=1e-3, max_iter=200, random_state=seed,
        early_stopping=True, validation_fraction=0.1, n_iter_no_change=20,
    )


def oversample(X: np.ndarray, y: np.ndarray, seed: int):
    """Class-balanced random oversampling with replacement to the majority count."""
    rng = np.random.RandomState(seed)
    counts = np.bincount(y, minlength=K)
    target = counts.max()
    idx = []
    for c in range(K):
        members = np.flatnonzero(y == c)
        idx.append(members)
        extra = target - len(members)
        if extra > 0:
            idx.append(rng.choice(members, size=extra, replace=True))
    idx = np.concatenate(idx)
    return X[idx], y[idx]


def fold_recalls(cm: np.ndarray) -> np.ndarray:
    return np.diag(row_normalise(cm))


def _fit_xgb(Xtr, ytr, seed, fold_idx, device):
    m = make_xgb(seed, device)
    m.fit(Xtr, ytr, sample_weight=compute_sample_weight("balanced", ytr))
    return m


def _fit_mlp(Xtr, ytr, seed, fold_idx, device):
    # the MLP trains on a class-balanced oversample; the fold index keeps the resampling streams distinct across folds
    Xb, yb = oversample(Xtr, ytr, seed + fold_idx)
    m = make_mlp(seed)
    m.fit(Xb, yb)
    return m


def run_cv(X, y, split_seed: int, model_seed: int, fit_fold, device: str):
    """Fold-averaged per-class recall and the per-fold balanced accuracies."""
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=split_seed)
    cms, fold_bacc = [], []
    for fold_idx, (tr, te) in enumerate(skf.split(X, y)):
        sc = StandardScaler().fit(X[tr])
        m = fit_fold(sc.transform(X[tr]), y[tr], model_seed, fold_idx, device)
        cm = confusion_matrix(y[te], m.predict(sc.transform(X[te])), labels=np.arange(K))
        cms.append(cm)
        fold_bacc.append(fold_recalls(cm).mean())
    return fold_recalls(np.mean(cms, axis=0)), np.array(fold_bacc)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda",
                    help="XGBoost device; pass cpu on a machine without CUDA")
    args = ap.parse_args()

    df = pd.read_csv(CSV)
    X = df[FEATURE_COLS].to_numpy(dtype=np.float64)
    y = LabelEncoder().fit(CLASS_ORDER).transform(df["class1"].to_numpy())

    xgb_recall = np.zeros((len(SEEDS), K))
    xgb_fold = np.zeros((len(SEEDS), N_SPLITS))
    for i, s in enumerate(SEEDS):
        xgb_recall[i], xgb_fold[i] = run_cv(X, y, ANCHOR_SEED, s, _fit_xgb,
                                            args.device)
        print(f"[xgb seed {s:5d}] balanced accuracy {xgb_recall[i].mean():.4f}  "
              f"recall {np.round(xgb_recall[i], 4)}")
    xgb_bacc = xgb_recall.mean(axis=1)

    mlp_recall, mlp_fold = run_cv(X, y, ANCHOR_SEED, ANCHOR_SEED, _fit_mlp,
                                  args.device)
    print(f"[mlp seed {ANCHOR_SEED:5d}] balanced accuracy {mlp_recall.mean():.4f}  "
          f"recall {np.round(mlp_recall, 4)}")

    anchor = SEEDS.index(ANCHOR_SEED)
    out = dict(
        seeds=np.array(SEEDS), anchor_seed=np.array(ANCHOR_SEED),
        class_order=np.array(CLASS_ORDER),
        xgb_bacc_by_seed=xgb_bacc,
        xgb_recall_by_seed=xgb_recall,
        xgb_bacc_seed_std_pp=np.array(100 * xgb_bacc.std(ddof=1)),
        xgb_recall_seed_std_pp=100 * xgb_recall.std(axis=0, ddof=1),
        xgb_fold_bacc_anchor=xgb_fold[anchor],
        xgb_fold_bacc_std_pp=np.array(100 * xgb_fold[anchor].std(ddof=1)),
        mlp_recall_anchor=mlp_recall,
        mlp_fold_bacc_anchor=mlp_fold,
        mlp_fold_bacc_std_pp=np.array(100 * mlp_fold.std(ddof=1)),
    )
    np.savez(OUT, **out)
    print(f"\nacross-seed std of XGBoost balanced accuracy: {out['xgb_bacc_seed_std_pp']:.2f} pp")
    print(f"largest across-seed std of a single-class recall: "
          f"{out['xgb_recall_seed_std_pp'].max():.2f} pp "
          f"({CLASS_ORDER[int(out['xgb_recall_seed_std_pp'].argmax())]})")
    print(f"across-fold std of balanced accuracy at the anchor seed: "
          f"XGBoost {out['xgb_fold_bacc_std_pp']:.2f} pp, MLP {out['mlp_fold_bacc_std_pp']:.2f} pp")
    print(f"[done] {OUT}")


if __name__ == "__main__":
    main()
