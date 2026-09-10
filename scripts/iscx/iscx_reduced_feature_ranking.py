"""Per-fold reduced-feature cross-validation on the ISCX 5-class subset.

The per-fold protocol ranks features inside each fold on the training partition only; the global protocol ranks once on the full subset. Both are run, so the per-class recall difference gives the ranking-leakage bias. Refreshes the xgb_reduced_feat and mlp_reduced_feat slots of data/processed/confusion_matrices.npz with the per-fold results and writes both sets of per-class recalls with their difference to data/processed/reduced_feature_ranking.json.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "src").is_dir())
sys.path.insert(0, str(ROOT))
from src.analytical.constants import CLASS_ORDER_OTT
from src.iscx.config import RANDOM_STATE, FEATURE_COLS, MLP_KWARGS
from src.analytical.kaufman_roberts import row_normalise
PROCESSED_DIR = ROOT / "data" / "processed"

CLASS_ORDER = list(CLASS_ORDER_OTT)
N_CLASSES = len(CLASS_ORDER)

XGB_KWARGS = dict(
    n_estimators=200, max_depth=6, learning_rate=0.1,
    subsample=0.8, colsample_bytree=0.8, eval_metric="mlogloss",
    tree_method="hist",
    random_state=RANDOM_STATE, n_jobs=-1, verbosity=0,
    objective="multi:softmax", num_class=N_CLASSES,
)

def rank_features(X_train: np.ndarray, y_train: np.ndarray) -> list[str]:
    """Feature names ordered by the gain importance of a balanced-weight XGBoost fitted on the given rows only."""
    X_s = StandardScaler().fit_transform(X_train)
    xgb = XGBClassifier(**XGB_KWARGS)
    xgb.fit(X_s, y_train, sample_weight=compute_sample_weight("balanced", y_train))
    order = np.argsort(xgb.feature_importances_)[::-1]
    return [FEATURE_COLS[i] for i in order]


def fit_xgb(X: np.ndarray, y: np.ndarray, _fold_idx: int) -> XGBClassifier:
    xgb = XGBClassifier(**XGB_KWARGS)
    xgb.fit(X, y, sample_weight=compute_sample_weight("balanced", y))
    return xgb


def fit_mlp(X: np.ndarray, y: np.ndarray, fold_idx: int) -> MLPClassifier:
    ros = RandomOverSampler(random_state=RANDOM_STATE + fold_idx)
    X_bal, y_bal = ros.fit_resample(X, y)
    mlp = MLPClassifier(**MLP_KWARGS)
    mlp.fit(X_bal, y_bal)
    return mlp


def cv_reduced(X: np.ndarray, y: np.ndarray, fit, reduced_global: list[str] | None
               ) -> tuple[np.ndarray, list[list[str]]]:
    """Five-fold cross-validation on the reduced feature set.

    With reduced_global given, the same columns are used in every fold (global ranking). Otherwise the five most important features are ranked on the training partition of each fold and removed there (per-fold ranking).
    """
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cms_raw, removed = [], []
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        y_tr, y_te = y[train_idx], y[test_idx]
        if reduced_global is None:
            top5 = rank_features(X[train_idx], y_tr)[:5]
            reduced = [f for f in FEATURE_COLS if f not in top5]
        else:
            # global branch: no per-fold ranking, so the removed list is padded
            top5, reduced = None, reduced_global
        removed.append(top5)
        cols = [FEATURE_COLS.index(f) for f in reduced]
        scaler = StandardScaler().fit(X[train_idx][:, cols])
        X_tr = scaler.transform(X[train_idx][:, cols])
        X_te = scaler.transform(X[test_idx][:, cols])
        y_pred = fit(X_tr, y_tr, fold_idx).predict(X_te)
        cms_raw.append(confusion_matrix(y_te, y_pred, labels=np.arange(N_CLASSES)))
    return row_normalise(np.mean(cms_raw, axis=0)), removed


def show_diag(label: str, cm: np.ndarray) -> None:
    print(f"  {label} diag: " + "  ".join(
        f"{c}={cm[i, i]:.4f}" for i, c in enumerate(CLASS_ORDER)))


def main() -> None:
    print("Per-fold reduced-feature ranking")

    df = pd.read_csv(PROCESSED_DIR / "iscx_5class_15s_clean.csv")
    X = df[FEATURE_COLS].values.astype(np.float64)
    y_raw = df["class1"].values
    le = LabelEncoder().fit(CLASS_ORDER)
    y = le.transform(y_raw)
    print(f"Data: X={X.shape}, y={y.shape}, classes={CLASS_ORDER}")

    print(f"\nGlobal ranking on the full {len(df):,}-flow subset")
    top5_global = rank_features(X, y)[:5]
    reduced_global = [f for f in FEATURE_COLS if f not in top5_global]
    print(f"Global top-5 (global): {top5_global}")

    cm_xgb_a, _ = cv_reduced(X, y, fit_xgb, reduced_global)
    cm_mlp_a, _ = cv_reduced(X, y, fit_mlp, reduced_global)
    show_diag("xgb_reduced (global)", cm_xgb_a)
    show_diag("mlp_reduced (global)", cm_mlp_a)

    print("\nPer-fold ranking on the training partition only")
    cm_xgb_b, top5_xgb_folds = cv_reduced(X, y, fit_xgb, None)
    cm_mlp_b, _ = cv_reduced(X, y, fit_mlp, None)
    show_diag("xgb_reduced (per-fold)", cm_xgb_b)
    show_diag("mlp_reduced (per-fold)", cm_mlp_b)

    # positive bias means the global ranking over-reports recall
    print("\nPer-class bias (global recall minus per-fold recall), percentage points")
    bias_xgb_pp = (np.diag(cm_xgb_a) - np.diag(cm_xgb_b)) * 100.0
    bias_mlp_pp = (np.diag(cm_mlp_a) - np.diag(cm_mlp_b)) * 100.0
    print(f"{'Class':14s}  {'XGB bias (pp)':>14s}  {'MLP bias (pp)':>14s}")
    for i, cls in enumerate(CLASS_ORDER):
        print(f"{cls:14s}  {bias_xgb_pp[i]:14.2f}  {bias_mlp_pp[i]:14.2f}")
    print()
    for tag, bias in (("xgb", bias_xgb_pp), ("mlp", bias_mlp_pp)):
        print(f"max|bias_{tag}| = {np.max(np.abs(bias)):.2f} pp, "
              f"mean (signed) = {np.mean(bias):+.2f} pp")

    archive_path = PROCESSED_DIR / "confusion_matrices.npz"
    existing = dict(np.load(archive_path, allow_pickle=True))
    existing["xgb_reduced_feat"] = cm_xgb_b
    existing["mlp_reduced_feat"] = cm_mlp_b
    np.savez(archive_path, **existing)
    print()
    print(f"Saved per-fold slots to {archive_path}")

    comparison = {
        "global_top5": top5_global,
        "per_fold_top5": top5_xgb_folds,
        "class_order": CLASS_ORDER,
    }
    for protocol, cm_x, cm_m in (("global", cm_xgb_a, cm_mlp_a),
                                 ("per_fold", cm_xgb_b, cm_mlp_b)):
        comparison[protocol] = {
            "xgb_diag": np.diag(cm_x).tolist(),
            "mlp_diag": np.diag(cm_m).tolist(),
            "xgb_balanced": float(np.mean(np.diag(cm_x))),
            "mlp_balanced": float(np.mean(np.diag(cm_m))),
        }
    comparison["bias_pp"] = {
        "xgb": bias_xgb_pp.tolist(),
        "mlp": bias_mlp_pp.tolist(),
        "xgb_max_abs": float(np.max(np.abs(bias_xgb_pp))),
        "mlp_max_abs": float(np.max(np.abs(bias_mlp_pp))),
    }
    out_path = PROCESSED_DIR / "reduced_feature_ranking.json"
    out_path.write_text(json.dumps(comparison, indent=2))
    print(f"Saved ranking comparison to {out_path}")


if __name__ == "__main__":
    main()
