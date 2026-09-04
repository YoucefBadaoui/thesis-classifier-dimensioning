"""Duration-weighted confusion matrices on CESNET-TLS-Year22.

C^dur_ij = sum{T_l : true_l=i, pred_l=j} / sum{T_l : true_l=i} is row-stochastic and makes a_hat = (C^dur)^T a an identity; a_hat = C^T a with the flow-count matrix holds only when holding time is conditionally independent of the predicted label given the true class. Emits, per classifier condition, the raw duration-sum matrix, and the ratio matrix R_ij = E[T | true=i, pred=j] / E[T | true=i], which measures that assumption.
"""

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from sklearn.metrics import balanced_accuracy_score

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "src").is_dir())
sys.path.insert(0, str(ROOT))
from src.cesnet.training import DR, PPI_RE, app_cat, build, fit_lgbm, fit_mlp, fit_xgb

# line-buffered so a piped log follows a long run
sys.stdout.reconfigure(line_buffering=True)
OUT = ROOT / "data" / "processed" / "cesnet_duration_weighted.npz"

CONDITIONS = ("xgb_clean", "lgbm_clean", "mlp_clean", "xgb_reduced", "lgbm_reduced",
              "dur1", "flow3")

# (clean condition, drift condition or None, fitter, feature set)
FITTERS = (("xgb_clean", "xgb_drift", fit_xgb, "full"),
           ("lgbm_clean", "lgbm_drift", fit_lgbm, "full"),
           ("mlp_clean", "mlp_drift", fit_mlp, "full"),
           ("xgb_reduced", None, fit_xgb, "reduced"),
           ("lgbm_reduced", None, fit_lgbm, "reduced"))


def _safe_div(num, den):
    """Elementwise num/den, nan where den is non-positive."""
    return np.where(den > 0, num / np.where(den > 0, den, 1), np.nan)


def matrices(y_true, y_pred, dur, n):
    """Flow-count matrix, duration-sum matrix, and conditional-mean ratio matrix."""
    cnt = np.zeros((n, n))
    dsum = np.zeros((n, n))
    np.add.at(cnt, (y_true, y_pred), 1.0)
    np.add.at(dsum, (y_true, y_pred), dur)
    with np.errstate(invalid="ignore", divide="ignore"):
        cell_mean = _safe_div(dsum, cnt)
        row_mean = _safe_div(dsum.sum(1, keepdims=True), cnt.sum(1, keepdims=True))
        ratio = cell_mean / row_mean
    return cnt, dsum, ratio


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="M", help="CESNET-TLS-Year22 partition size")
    ap.add_argument("--seed", type=int, default=42, help="model seed")
    ap.add_argument("--smoke", action="store_true", help="tiny splits for a wiring check")
    ap.add_argument("--with-drift", action="store_true",
                    help="also load the 3M October test partition for the drift twins")
    ap.add_argument("--conditions", default=",".join(CONDITIONS),
                    help="comma-separated subset of the conditions to fit")
    a = ap.parse_args()
    if a.smoke:
        tr, va, te, per, est = 6000, 3000, 3000, 200, 40
    else:
        tr, va, te, per, est = 1_000_000, 1_000_000, 3_000_000, 8000, 300

    print(f"[init] size={a.size} seed={a.seed} train={tr} val={va} test={te}")
    d, cfg = build(a.size, tr, va, te, 42)
    sm = pd.read_csv(DR / a.size / "servicemap.csv", index_col="Tag")
    known = d.get_known_apps()
    ac, cats = app_cat(known, sm)
    n = len(cats)
    feat = cfg.get_feature_names(flatten_ppi=True)
    red = [c for c in feat if not PPI_RE.match(c)]

    drift = a.with_drift
    tr_df = d.get_train_df(flatten_ppi=True)
    va_df = d.get_val_df(flatten_ppi=True)
    te_df = d.get_test_df(flatten_ppi=True) if drift else None
    print(f"[load] train={tr_df.shape} val={va_df.shape} "
          f"test={te_df.shape if drift else 'skipped'}")
    flow3_cols = [c for c in ["BYTES", "BYTES_REV", "DURATION", "PACKETS", "PACKETS_REV"]
                  if c in tr_df.columns][:3]
    probe_cols = {"dur1": ["DURATION"], "flow3": flow3_cols}

    F = lambda df, cols: df[cols].to_numpy(np.float32)
    CAT = lambda df: ac[df["APP"].to_numpy()]

    Xtr, Xva = F(tr_df, feat), F(va_df, feat)
    Xtr_r, Xva_r = F(tr_df, red), F(va_df, red)
    ytr, yva = CAT(tr_df), CAT(va_df)
    Xte = F(te_df, feat) if drift else None
    yte = CAT(te_df) if drift else None

    # non-positive or non-finite durations carry no Erlang mass; dropping them from both matrices keeps the count and duration forms on one flow set
    dur_va = va_df["DURATION"].to_numpy(float)
    ok_va = np.isfinite(dur_va) & (dur_va > 0)
    dur_te = te_df["DURATION"].to_numpy(float) if drift else None
    ok_te = (np.isfinite(dur_te) & (dur_te > 0)) if drift else None
    print(f"[dur] val usable {ok_va.sum()}/{len(ok_va)}"
          + (f"  test usable {ok_te.sum()}/{len(ok_te)}" if drift else ""))

    # per-category mean holding time on the train split, the quantity scripts/cesnet/cesnet_dimension.py uses to build a_erlang
    dur_tr = tr_df["DURATION"].to_numpy(float)
    ok_tr = np.isfinite(dur_tr) & (dur_tr > 0)
    hold_mean_tr = np.zeros(n)
    for c in range(n):
        m = (ytr == c) & ok_tr
        if m.any():
            hold_mean_tr[c] = float(np.mean(dur_tr[m]))

    save = {
        "category_names": np.array(cats),
        "size": np.array(a.size),
        "seed": np.array(a.seed),
        "hold_mean_train": hold_mean_tr,
        "val_usable_frac": np.array(float(ok_va.mean())),
        "test_usable_frac": np.array(float(ok_te.mean()) if drift else np.nan),
    }

    probes = {
        tag: (F(tr_df, cols), F(va_df, cols), list(cols))
        for tag, cols in probe_cols.items()
        if all(c in tr_df.columns for c in cols) and cols
    }

    del tr_df, va_df, te_df

    seed = a.seed
    want = set(a.conditions.split(","))
    jobs = []

    for clean, drift_name, fit, featset in FITTERS:
        if clean not in want:
            continue
        Xt, Xv = (Xtr, Xva) if featset == "full" else (Xtr_r, Xva_r)
        want_drift = drift and drift_name is not None
        print(f"[fit] {clean} ({featset} features)")
        if fit is fit_mlp:
            model, sc = fit(Xt, ytr, per, seed)
            pv = model.predict(sc.transform(Xv))
            pt = model.predict(sc.transform(Xte)) if want_drift else None
            del sc
        else:
            model = fit(Xt, ytr, n, seed, est)
            pv = model.predict(Xv)
            pt = model.predict(Xte) if want_drift else None
        del model
        jobs.append((clean, pv, yva, dur_va, ok_va))
        if pt is not None:
            jobs.append((drift_name, pt, yte, dur_te, ok_te))

    # degraded probes; dur1 keys on duration alone, so the predicted label becomes a function of the quantity the Erlang load carries
    for tag, (Ptr, Pva, cols) in probes.items():
        if tag not in want:
            continue
        print(f"[fit] degraded probe {tag} on {cols}")
        gp = fit_xgb(Ptr, ytr, n, seed, min(est, 150))
        jobs.append((f"degraded_{tag}", gp.predict(Pva), yva, dur_va, ok_va))
        del gp

    for cond, pred, ytrue, dur, ok in jobs:
        cnt, dsum, ratio = matrices(ytrue[ok], pred[ok], dur[ok], n)
        bacc = float(balanced_accuracy_score(ytrue[ok], pred[ok]))
        save[f"{cond}_count"] = cnt
        save[f"{cond}_dursum"] = dsum
        save[f"{cond}_ratio"] = ratio
        save[f"bacc_{cond}"] = np.array(bacc)
        rc = cnt.sum(1, keepdims=True)
        rd = dsum.sum(1, keepdims=True)
        Cc = cnt / np.where(rc == 0, 1, rc)
        Cd = dsum / np.where(rd == 0, 1, rd)
        save[f"{cond}_C_count"] = Cc
        save[f"{cond}_C_dur"] = Cd
        dev = np.abs(Cd - Cc)
        print(f"  {cond:13s} bacc={bacc:.4f}  max|C_dur-C_count|={dev.max():.4f} "
              f"mean={dev.mean():.5f}  diag shift={np.mean(np.diag(Cd)-np.diag(Cc)):+.5f}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez(OUT, **save)
    print(f"[done] {OUT}")


if __name__ == "__main__":
    main()
