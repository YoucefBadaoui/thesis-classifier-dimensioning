"""Multi-seed CESNET-TLS-Year22 ensemble and per-category exploration.

One data load per run, at the 23 service-category granularity. Produces XGBoost, MLP and LightGBM confusion matrices under clean, drift and reduced conditions across model seeds; cross-validated balanced-accuracy intervals for XGBoost at seed 42; and per-category support, mean and median holding time (DURATION), downstream throughput and a packet-sequence active-rate proxy, which feed the offered-load prior and the AU ladder.
"""

import argparse
import sys
import json
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "src").is_dir())
sys.path.insert(0, str(ROOT))
from src.cesnet.training import DR, PPI_RE, app_cat, build, fit_lgbm, fit_mlp, fit_xgb

# line-buffered so a piped log follows a long run
sys.stdout.reconfigure(line_buffering=True)
OUT = ROOT / "data" / "processed" / "cesnet_definitive.npz"
OUT_J = ROOT / "data" / "processed" / "cesnet_definitive_eda.json"
SIZE_RE = re.compile(r"^SIZE_\d+$")

# per-category rate and holding-time statistics collected on the train split
CAT_STAT_KEYS = ("hold_med", "hold_mean", "down_p99", "down_mean",
                 "ppi_rate_p99", "ppi_rate_mean")

# (clean condition, drift condition or None, fitter, feature set)
FITS = (("xgb_clean", "xgb_drift", fit_xgb, "full"),
        ("lgbm_clean", "lgbm_drift", fit_lgbm, "full"),
        ("mlp_clean", "mlp_drift", fit_mlp, "full"),
        ("xgb_reduced", None, fit_xgb, "reduced"),
        ("lgbm_reduced", None, fit_lgbm, "reduced"))


def cmb(clf, X, y, n, sc=None):
    p = clf.predict(sc.transform(X) if sc is not None else X)
    return confusion_matrix(y, p, labels=list(range(n))).astype(float), balanced_accuracy_score(y, p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="M")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        tr, va, te, per, est, seeds, folds = 6000, 3000, 3000, 200, 40, [42, 7], 2
    else:
        tr, va, te, per, est, seeds, folds = 1_000_000, 1_000_000, 3_000_000, 8000, 300, [42, 7, 123], 5

    print(f"[init] size={a.size} train={tr} seeds={seeds} smoke={a.smoke}")
    d, cfg = build(a.size, tr, va, te, 42)
    sm = pd.read_csv(DR / a.size / "servicemap.csv", index_col="Tag")
    known = d.get_known_apps()
    ac, cats = app_cat(known, sm)
    n = len(cats)
    feat = cfg.get_feature_names(flatten_ppi=True)
    red = [c for c in feat if not PPI_RE.match(c)]

    tr_df = d.get_train_df(flatten_ppi=True)
    va_df = d.get_val_df(flatten_ppi=True)
    te_df = d.get_test_df(flatten_ppi=True)
    print(f"[load] train={tr_df.shape} val={va_df.shape} test={te_df.shape}")

    def F(df, cols):
        return df[cols].to_numpy(np.float32)

    def CAT(df):
        return ac[df["APP"].to_numpy()]

    Xtr, Xva, Xte = F(tr_df, feat), F(va_df, feat), F(te_df, feat)
    Xtr_r, Xva_r = F(tr_df, red), F(va_df, red)
    ytr, yva, yte = CAT(tr_df), CAT(va_df), CAT(te_df)

    save = {
        "category_names": np.array(cats), "size": np.array(a.size),
        "train_support": np.bincount(ytr, minlength=n),
        "val_support": np.bincount(yva, minlength=n),
        "test_support": np.bincount(yte, minlength=n),
    }

    # holding time and throughput (train df)
    dur = tr_df["DURATION"].to_numpy(float)
    brv = tr_df["BYTES_REV"].to_numpy(float)
    valid = np.isfinite(dur) & (dur > 0)
    rated = np.where(valid, brv * 8 / np.where(valid, dur, 1) / 1e6, np.nan)
    size_cols = [c for c in tr_df.columns if SIZE_RE.match(c)]
    ppi_rate = np.full(len(tr_df), np.nan)
    if size_cols and "PPI_DURATION" in tr_df.columns:
        first_bytes = np.abs(tr_df[size_cols].to_numpy(float)).sum(axis=1)
        pdur = tr_df["PPI_DURATION"].to_numpy(float)
        pv = np.isfinite(pdur) & (pdur > 0)
        ppi_rate = np.where(pv, first_bytes * 8 / np.where(pv, pdur, 1) / 1e6, np.nan)

    cs = {k: np.zeros(n) for k in CAT_STAT_KEYS}
    for c in range(n):
        m = (ytr == c) & valid
        if m.sum() > 0:
            cs["hold_med"][c] = float(np.median(dur[m]))
            cs["hold_mean"][c] = float(np.mean(dur[m]))
            cs["down_p99"][c] = float(np.nanpercentile(rated[m], 99))
            cs["down_mean"][c] = float(np.nanmean(rated[m]))
            pr = ppi_rate[ytr == c]  # PPI validity is independent of DURATION validity
            pr = pr[np.isfinite(pr)]
            if len(pr):
                cs["ppi_rate_p99"][c] = float(np.percentile(pr, 99))
                cs["ppi_rate_mean"][c] = float(np.mean(pr))
    save.update(cs)

    # Jerabek et al. 2025 redundancy: share of val/test flows whose packet sequence also appears in train. loose = SIZE+DIR only, which shared handshakes overcount; strict = full PPI including inter-packet times.
    loose_cols = [c for c in feat if re.match(r"^(SIZE|DIR)_\d+$", c)]
    strict_cols = [c for c in feat if PPI_RE.match(c)]

    def _hf(df, cols):
        return pd.util.hash_pandas_object(df[cols], index=False).to_numpy()

    if strict_cols:
        tl, vl, el = _hf(tr_df, loose_cols), _hf(va_df, loose_cols), _hf(te_df, loose_cols)
        ts, vs, es = _hf(tr_df, strict_cols), _hf(va_df, strict_cols), _hf(te_df, strict_cols)
        save["ppi_leak_val_loose"] = np.array(float(np.isin(vl, tl).mean()))
        save["ppi_leak_test_loose"] = np.array(float(np.isin(el, tl).mean()))
        save["ppi_leak_val_strict"] = np.array(float(np.isin(vs, ts).mean()))
        save["ppi_leak_test_strict"] = np.array(float(np.isin(es, ts).mean()))
        print("[leak] share of val/test flows seen in train: " + "  ".join(
            f"{tag}_{sp}={float(save[f'ppi_leak_{sp}_{tag}']):.4f}"
            for tag in ("loose", "strict") for sp in ("val", "test")))

    # per-category support gate (target >= 1500 test flows for stable recall CIs)
    order = np.argsort(save["test_support"])
    print("[gate] six smallest per-category test supports:")
    for i in order[:6]:
        sup_i = int(save["test_support"][i])
        print(f"   {cats[i]:<26} test={sup_i}{'' if sup_i >= 1500 else '  (below 1500)'}")

    n_train = int(tr_df.shape[0])
    del tr_df, va_df, te_df

    for seed in seeds:
        print(f"[seed {seed}] training xgb/lgbm/mlp...")
        for clean, drift, fit, featset in FITS:
            Xt, Xv = (Xtr, Xva) if featset == "full" else (Xtr_r, Xva_r)
            model, sc = (fit(Xt, ytr, per, seed) if fit is fit_mlp
                         else (fit(Xt, ytr, n, seed, est), None))
            for cond, X, y in ((clean, Xv, yva), (drift, Xte, yte)):
                if cond is None:
                    continue
                save[f"{cond}_s{seed}"], b = cmb(model, X, y, n, sc)
                save[f"bacc_{cond}_s{seed}"] = np.array(b)
            del model
        print(f"[seed {seed}] xgb_clean={float(save[f'bacc_xgb_clean_s{seed}']):.4f} "
              f"lgbm_clean={float(save[f'bacc_lgbm_clean_s{seed}']):.4f} "
              f"mlp_clean={float(save[f'bacc_mlp_clean_s{seed}']):.4f}")

    print("[cv] xgb stratified k-fold (seed 42)...")
    skf = StratifiedKFold(folds, shuffle=True, random_state=42)
    cv = []
    for k, (i, j) in enumerate(skf.split(Xtr, ytr), 1):
        c = fit_xgb(Xtr[i], ytr[i], n, 42, est)
        _, bb = cmb(c, Xtr[j], ytr[j], n)
        cv.append(bb)
        print(f"  fold {k}/{folds}={bb:.4f}")
    save["xgb_cv"] = np.array(cv)

    eda = {
        "size": a.size, "n_categories": n, "category_names": list(cats),
        "seeds": seeds, "train_rows": n_train,
    }
    for key, vec, cast in (("train_support", save["train_support"], int),
                           ("hold_med_s", cs["hold_med"], float),
                           ("down_p99_mbps", cs["down_p99"], float),
                           ("ppi_rate_p99_mbps", cs["ppi_rate_p99"], float)):
        eda[key] = {c: cast(v) for c, v in zip(cats, vec)}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez(OUT, **save)
    OUT_J.write_text(json.dumps(eda, indent=2))
    print(f"[done] {OUT}  xgb_cv={np.mean(cv):.4f} +/- {np.std(cv):.4f}")


if __name__ == "__main__":
    main()
