"""CESNET-TLS-Year22 loading and classifier fitting shared by the CESNET scripts."""

import re
from pathlib import Path

import numpy as np
from cesnet_datazoo.config import AppSelection, DatasetConfig, ValidationApproach
from cesnet_datazoo.datasets import CESNET_TLS_Year22
from lightgbm import LGBMClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[2]
DR = ROOT / "data" / "cesnet"
# per-packet information (PPI) feature names of the DataZoo tables
PPI_RE = re.compile(r"^(IPT|DIR|SIZE|PUSH)_\d+$")


def build(size, tr, va, te, seed):
    """Configured CESNET-TLS-Year22 dataset and its DatasetConfig, for the given split sizes and seed."""
    d = CESNET_TLS_Year22(str(DR), size=size, silent=True)
    cfg = DatasetConfig(
        dataset=d, apps_selection=AppSelection.ALL_KNOWN,
        train_period_name="M-2022-9", test_period_name="M-2022-10",
        need_val_set=True, val_approach=ValidationApproach.SPLIT_FROM_TRAIN,
        train_size=tr, val_known_size=va, test_known_size=te,
        use_packet_histograms=True, use_tcp_features=True, random_state=seed,
    )
    d.set_dataset_config_and_initialize(cfg)
    return d, cfg


def app_cat(known, sm):
    co = sm["Service Category"].to_dict()
    cats = sorted({co[a] for a in known})
    idx = {c: i for i, c in enumerate(cats)}
    return np.array([idx[co[a]] for a in known], dtype=int), cats


def bsub(X, y, per, seed):
    rng = np.random.default_rng(seed)
    out = []
    for c in np.unique(y):
        ix = np.where(y == c)[0]
        out.append(rng.choice(ix, per, replace=len(ix) < per))
    o = np.concatenate(out)
    rng.shuffle(o)
    return X[o], y[o]


def fit_xgb(X, y, n, seed, est):
    c = XGBClassifier(n_estimators=est, max_depth=8, learning_rate=0.1, subsample=0.8,
                      colsample_bytree=0.8, tree_method="hist", objective="multi:softprob",
                      num_class=n, n_jobs=-1, random_state=seed, eval_metric="mlogloss")
    c.fit(X, y, sample_weight=compute_sample_weight("balanced", y))
    return c


def fit_lgbm(X, y, n, seed, _est=None):
    # _est is the XGBoost tree budget, accepted so the fitters share one signature and ignored here: LightGBM uses a fixed 500 trees, which is what built the archives. Regularised; the library default leaf-wise configuration is not used.
    c = LGBMClassifier(n_estimators=500, max_depth=8, num_leaves=63, learning_rate=0.05,
                       min_child_samples=500, subsample=0.8, colsample_bytree=0.8,
                       objective="multiclass", num_class=n,
                       n_jobs=-1, random_state=seed, verbose=-1)
    c.fit(X, y, sample_weight=compute_sample_weight("balanced", y))
    return c


def fit_mlp(X, y, per, seed):
    sc = StandardScaler().fit(X)
    Xb, yb = bsub(sc.transform(X), y, per, seed)
    c = MLPClassifier(hidden_layer_sizes=(256, 128, 64), activation="relu", solver="adam",
                      alpha=1e-4, batch_size=256, learning_rate="adaptive", learning_rate_init=1e-3,
                      max_iter=200, random_state=seed, early_stopping=True, validation_fraction=0.1,
                      n_iter_no_change=20)
    c.fit(Xb, yb)
    return c, sc
