"""Per-service-category throughput from CESNET-TLS-Year22 flow records.

Derives, per service category, the distribution of measured per-flow throughput (total and downstream) from BYTES, BYTES_REV and DURATION (Hynek et al. 2024, Scientific Data 11:1156), so the AU tier demands rest on the corpus's own measured rates.
"""

import argparse
import sys
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from cesnet_datazoo.config import AppSelection, DatasetConfig
from cesnet_datazoo.datasets import CESNET_TLS_Year22

RANDOM_STATE = 42
ROOT = next(p for p in Path(__file__).resolve().parents if (p / "src").is_dir())
sys.path.insert(0, str(ROOT))
from src.cesnet.training import DR, app_cat

# line-buffered so a piped log follows a long run
sys.stdout.reconfigure(line_buffering=True)
OUT_JSON = ROOT / "data" / "processed" / "cesnet_category_throughput.json"


def build_dataset(size, train_size):
    d = CESNET_TLS_Year22(str(DR), size=size, silent=True)
    cfg = DatasetConfig(
        dataset=d,
        apps_selection=AppSelection.ALL_KNOWN,
        train_period_name="M-2022-9",
        test_period_name="M-2022-10",
        need_val_set=False,
        train_size=train_size,
        use_packet_histograms=True,
        use_tcp_features=True,
        random_state=RANDOM_STATE,
    )
    d.set_dataset_config_and_initialize(cfg)
    return d


def pct(a, q):
    return float(np.percentile(a, q))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="M")
    ap.add_argument("--train-size", type=int, default=400_000)
    args = ap.parse_args()

    print(f"[init] size={args.size} train_size={args.train_size}")
    d = build_dataset(args.size, args.train_size)
    servicemap = pd.read_csv(DR / args.size / "servicemap.csv", index_col="Tag")
    known = d.get_known_apps()
    cat_vec, cat_names = app_cat(known, servicemap)
    n_cat = len(cat_names)

    df = d.get_train_df(flatten_ppi=True)
    print(f"[load] train df shape={df.shape}")

    cat = cat_vec[df["APP"].to_numpy()]
    bytes_fwd = df["BYTES"].to_numpy(dtype=np.float64)
    bytes_rev = df["BYTES_REV"].to_numpy(dtype=np.float64)
    dur = df["DURATION"].to_numpy(dtype=np.float64)

    valid = np.isfinite(dur) & (dur > 0)
    total_bits = (bytes_fwd + bytes_rev) * 8.0
    down_bits = bytes_rev * 8.0
    safe = np.where(valid, dur, 1.0)
    rate_total_mbps = np.where(valid, total_bits / safe / 1e6, np.nan)
    rate_down_mbps = np.where(valid, down_bits / safe / 1e6, np.nan)

    stats = {}
    for c in range(n_cat):
        m = (cat == c) & valid
        rt = rate_total_mbps[m]
        rd = rate_down_mbps[m]
        rt = rt[np.isfinite(rt)]
        rd = rd[np.isfinite(rd)]
        if len(rt) == 0:
            continue
        s = {
            "n_flows": int(m.sum()),
            "total_mbps_median": float(np.median(rt)),
            "total_mbps_mean": float(np.mean(rt)),
            "total_mbps_p90": pct(rt, 90),
            "total_mbps_p99": pct(rt, 99),
            "down_mbps_median": float(np.median(rd)),
            "down_mbps_mean": float(np.mean(rd)),
            "down_mbps_p90": pct(rd, 90),
            "down_mbps_p99": pct(rd, 99),
            "bytes_total_median": float(np.median((bytes_fwd + bytes_rev)[m])),
            "duration_median_s": float(np.median(dur[m])),
        }
        stats[cat_names[c]] = s

    print("\n=== per-category measured throughput (sorted by downstream median Mbps) ===")
    print(f"{'category':24s} {'n':>7s} {'down_med':>9s} {'down_mean':>9s} "
          f"{'down_p90':>9s} {'tot_med':>9s}")
    for name, s in sorted(stats.items(), key=lambda kv: -kv[1]["down_mbps_median"]):
        print(f"{name:24s} {s['n_flows']:7d} {s['down_mbps_median']:9.3f} "
              f"{s['down_mbps_mean']:9.3f} {s['down_mbps_p90']:9.3f} "
              f"{s['total_mbps_median']:9.3f}")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps({
        "size": args.size,
        "train_period": "M-2022-9",
        "n_categories": n_cat,
        "duration_unit_assumed": "seconds",
        "rate_definition": "bits over flow duration; total=(BYTES+BYTES_REV)*8/DURATION, down=BYTES_REV*8/DURATION",
        "per_category": stats,
    }, indent=2))
    print(f"\n[done] wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
