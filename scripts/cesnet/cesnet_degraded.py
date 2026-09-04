"""Deliberately degraded CESNET classifiers, spanning good to near-chance.

Trains crippled XGBoost variants (one feature, three flow statistics, and full features on a 1500-flow sample) so the H2 condition set covers low-accuracy 23-category matrices measured on the CESNET corpus itself.
"""

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "src").is_dir())
sys.path.insert(0, str(ROOT))
from src.cesnet.training import DR, app_cat, build
OUT = ROOT / "data" / "processed" / "cesnet_degraded.npz"


def fit_eval(Xtr, ytr, Xva, yva, n, est=150):
    # max_depth 6 without subsampling, unlike the src/cesnet/training.py anchor
    c = XGBClassifier(n_estimators=est, max_depth=6, learning_rate=0.1, tree_method="hist",
                      objective="multi:softprob", num_class=n, n_jobs=-1, random_state=42,
                      eval_metric="mlogloss")
    c.fit(Xtr, ytr, sample_weight=compute_sample_weight("balanced", ytr))
    p = c.predict(Xva)
    return confusion_matrix(yva, p, labels=list(range(n))).astype(float), balanced_accuracy_score(yva, p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--size", default="M")
    a = ap.parse_args()
    tr, va = (6000, 3000) if a.smoke else (500000, 1500000)

    d, cfg = build(a.size, tr, va, va, 42)
    sm = pd.read_csv(DR / a.size / "servicemap.csv", index_col="Tag")
    known = d.get_known_apps()
    ac, cats = app_cat(known, sm)
    n = len(cats)
    feat = cfg.get_feature_names(flatten_ppi=True)
    tr_df = d.get_train_df(flatten_ppi=True)
    va_df = d.get_val_df(flatten_ppi=True)
    ytr = ac[tr_df["APP"].to_numpy()]
    yva = ac[va_df["APP"].to_numpy()]

    def F(df, cols):
        return df[cols].to_numpy(np.float32)

    # degradation strategies spanning toward chance (1/23 ~ 0.043 balanced acc)
    flow_stats = [c for c in ["BYTES", "BYTES_REV", "DURATION", "PACKETS", "PACKETS_REV"] if c in tr_df.columns]
    strategies = {
        "degraded_dur1": ["DURATION"],
        "degraded_flow3": flow_stats[:3],
        "degraded_full_tiny": feat,   # full features, 1500-row sample below
    }
    save = {"category_names": np.array(cats), "size": np.array(a.size)}
    print(f"size={a.size} categories={n} train={tr_df.shape[0]}")
    for name, cols in strategies.items():
        cols = [c for c in cols if c in tr_df.columns]
        if not cols:
            continue
        if name == "degraded_full_tiny":
            idx = np.random.default_rng(0).choice(len(ytr), size=min(1500, len(ytr)), replace=False)
            cm, b = fit_eval(F(tr_df, cols)[idx], ytr[idx], F(va_df, cols), yva, n, est=60)
        else:
            cm, b = fit_eval(F(tr_df, cols), ytr, F(va_df, cols), yva, n)
        save[f"{name}_cat_counts"] = cm
        save[f"bacc_{name}"] = np.array(b)
        print(f"  {name:22s} feats={len(cols):3d}  balanced_acc={b:.4f}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez(OUT, **save)
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
