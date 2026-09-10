"""Retrain the MLP classifiers with class-balanced oversampling.

Applies RandomOverSampler to each training fold before fit(). For stochastic gradient methods that matches a class-weighted cross-entropy loss up to minibatch sampling noise: every class contributes the same number of effective gradient updates per epoch. Refreshes only the mlp_clean and mlp_vpn_shift slots of data/processed/confusion_matrices.npz; the empirical and synthetic matrices are loaded from the archive and rewritten unchanged. The mlp_reduced_feat slot is owned by scripts/iscx/iscx_reduced_feature_ranking.py, which ranks inside each training fold instead of on the full subset.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from scipy.io import arff
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

# matches notebooks/02_classifiers.ipynb
ROOT = next(p for p in Path(__file__).resolve().parents if (p / "src").is_dir())
sys.path.insert(0, str(ROOT))
from src.analytical.constants import CLASS_ORDER_OTT
from src.iscx.config import RANDOM_STATE, FEATURE_COLS, MLP_KWARGS
from src.analytical.kaufman_roberts import row_normalise
DATA_DIR = ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
VPN_ARFF = DATA_DIR / "Scenario A2-ARFF" / "TimeBasedFeatures-Dataset-15s-VPN.arff"

CLASS_ORDER = list(CLASS_ORDER_OTT)
N_CLASSES = len(CLASS_ORDER)

def make_mlp() -> MLPClassifier:
    return MLPClassifier(**MLP_KWARGS)


def show_recalls(label: str, cm: np.ndarray, extra: str = "") -> None:
    row = "  ".join(f"{c}={cm[i, i]:.1%}" for i, c in enumerate(CLASS_ORDER))
    print(f"[{label}] {extra}\n          diagonal recalls: {row}")


def cv_mlp_balanced(X: np.ndarray, y: np.ndarray, label: str) -> np.ndarray:
    """Stratified 5-fold MLP with RandomOverSampler on each training fold."""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cms_raw = []
    accs = []
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        scaler = StandardScaler().fit(X_tr)
        X_tr_s = scaler.transform(X_tr)
        X_te_s = scaler.transform(X_te)

        ros = RandomOverSampler(random_state=RANDOM_STATE + fold_idx)
        X_bal, y_bal = ros.fit_resample(X_tr_s, y_tr)

        mlp = make_mlp()
        mlp.fit(X_bal, y_bal)
        y_pred = mlp.predict(X_te_s)

        cms_raw.append(confusion_matrix(y_te, y_pred, labels=np.arange(N_CLASSES)))
        accs.append(accuracy_score(y_te, y_pred))

    cm_avg_norm = row_normalise(np.mean(cms_raw, axis=0))
    show_recalls(label, cm_avg_norm,
                 f"balanced-MLP 5-fold: accuracy = {np.mean(accs):.4f} "
                 f"+/- {np.std(accs):.4f}")
    return cm_avg_norm


def load_vpn_features() -> tuple[np.ndarray, np.ndarray]:
    """Load VPN ARFF (Scenario A2), filter to the 5 thesis classes."""
    data, _ = arff.loadarff(VPN_ARFF)
    df = pd.DataFrame(data)
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].str.decode("utf-8")

    # labels arrive as VPN-<CLASS>; strip the prefix, then map the bare names
    class_map = {
        "BROWSING": "Browsing", "CHAT": "Chat",
        "FT": "FileTransfer", "FILETRANSFER": "FileTransfer",
        "STREAMING": "Streaming", "VOIP": "VoIP",
    }
    df["class1_std"] = (df["class1"].str.upper()
                        .str.removeprefix("VPN-").map(class_map))
    df = df.dropna(subset=["class1_std"]).copy()

    X_vpn = df[FEATURE_COLS].values.astype(np.float64)
    le = LabelEncoder().fit(CLASS_ORDER)
    y_vpn = le.transform(df["class1_std"].values)
    return X_vpn, y_vpn


def vpn_shift_mlp_balanced(X_nonvpn: np.ndarray, y_nonvpn: np.ndarray) -> np.ndarray:
    """Train on non-VPN with RandomOverSampler, test on VPN."""
    X_vpn, y_vpn = load_vpn_features()

    scaler = StandardScaler().fit(X_nonvpn)
    X_tr_s = scaler.transform(X_nonvpn)
    X_te_s = scaler.transform(X_vpn)

    ros = RandomOverSampler(random_state=RANDOM_STATE)
    X_bal, y_bal = ros.fit_resample(X_tr_s, y_nonvpn)

    mlp = make_mlp()
    mlp.fit(X_bal, y_bal)
    y_pred = mlp.predict(X_te_s)

    cm_raw = confusion_matrix(y_vpn, y_pred, labels=np.arange(N_CLASSES))
    cm_norm = row_normalise(cm_raw)
    acc = accuracy_score(y_vpn, y_pred)

    show_recalls("mlp_vpn_shift", cm_norm,
                 f"balanced-MLP train-on-nonVPN, test-on-VPN: accuracy = {acc:.4f}")
    return cm_norm


def main() -> None:
    print("MLP retraining with class-proportional weighting (RandomOverSampler)")

    df = pd.read_csv(PROCESSED_DIR / "iscx_5class_15s_clean.csv")
    X = df[FEATURE_COLS].values.astype(np.float64)
    y_raw = df["class1"].values
    le = LabelEncoder().fit(CLASS_ORDER)
    y = le.transform(y_raw)
    print(f"Data: X={X.shape}, y={y.shape}, classes={CLASS_ORDER}")
    print()

    cm_mlp_clean = cv_mlp_balanced(X, y, label="mlp_clean")
    cm_mlp_vpn = vpn_shift_mlp_balanced(X, y)

    archive_path = PROCESSED_DIR / "confusion_matrices.npz"
    existing = dict(np.load(archive_path, allow_pickle=True))
    existing["mlp_clean"] = cm_mlp_clean
    existing["mlp_vpn_shift"] = cm_mlp_vpn
    np.savez(archive_path, **existing)
    print(f"\nSaved refreshed MLP CMs to {archive_path}")

    print("\nRefreshed MLP diagonal recalls (post-reweighting)")
    print(f"{'Class':15s}  {'Clean':>10s}  {'VPN shift':>10s}")
    for i, cls in enumerate(CLASS_ORDER):
        print(f"{cls:15s}  {cm_mlp_clean[i, i]:10.1%}  {cm_mlp_vpn[i, i]:10.1%}")


if __name__ == "__main__":
    main()
