"""Tier grouping and dimensioning helpers for the CESNET ensemble archives.

The Recommended OTT grouping maps the 23 CESNET service categories onto the six-tier AU ladder and drops the eight machine-to-machine background categories; the all-inclusive control grouping re-enters them at tier 0.
"""

import numpy as np

from src.analytical.constants import A_TOTAL_CESNET, B_TARGET_DEFAULT, TIER_AU_CESNET
from src.analytical.kaufman_roberts import blocking_deviation, capacity_overhead, row_normalise
from src.analytical.scenarios import compute_wbd

B_TARGET, A_TOTAL = B_TARGET_DEFAULT, A_TOTAL_CESNET
TIER_AU = [int(x) for x in TIER_AU_CESNET]
SEEDS = [42, 7, 123]
CONDS = ["xgb_clean", "xgb_drift", "lgbm_clean", "lgbm_drift", "mlp_clean", "mlp_drift"]

GROUP_REC = {
    "Search": 0, "Mail": 0,
    "Other APIs": 1, "Other services": 1, "Information systems": 1, "Internet banking": 1,
    "Instant messaging": 2, "Social": 2,
    "Music": 3, "Videoconferencing": 3, "Remote desktop": 3,
    "File sharing": 4, "Software updates": 4,
    "Media": 5, "Games": 5,
    "Advertising": -1, "Analytics & Telemetry": -1, "Antivirus": -1, "Authentication": -1,
    "Location": -1, "Notifications": -1, "Weather": -1, "Virtual assistant": -1,
}
# all-inclusive control: the 8 excluded categories re-enter at their low-rate tier 0
GROUP_ALL = dict(GROUP_REC)
for k, v in list(GROUP_ALL.items()):
    if v == -1:
        GROUP_ALL[k] = 0


def grouping_arrays(names, mapping):
    """Category-to-tier index array, the sorted tier list, and the AU vector."""
    ct = np.array([mapping.get(n, -1) for n in names])
    tiers = sorted(set(ct[ct >= 0].tolist()))
    t = np.array([TIER_AU[i] for i in tiers])
    return ct, tiers, t


def tier_vec(per_cat_w, ct, tiers):
    pos = {t: i for i, t in enumerate(tiers)}
    v = np.zeros(len(tiers))
    for c in range(len(ct)):
        if ct[c] >= 0:
            v[pos[ct[c]]] += per_cat_w[c]
    return v


def agg(cm, ct, tiers):
    pos = {t: i for i, t in enumerate(tiers)}
    K = len(tiers)
    o = np.zeros((K, K))
    for i in range(cm.shape[0]):
        if ct[i] < 0:
            continue
        for j in range(cm.shape[1]):
            if ct[j] < 0:
                continue
            o[pos[ct[i]], pos[ct[j]]] += cm[i, j]
    return row_normalise(o)


def pick_matrix(z, cond, seeds=SEEDS):
    """Return (matrix, seed) for the median-balanced-accuracy seed, so a collapsed seed cannot drive the result."""
    cands = []
    for s in seeds:
        k = f"{cond}_s{s}"
        if k not in z.files:
            continue
        bk = f"bacc_{cond}_s{s}"
        bacc = float(z[bk])
        cands.append((bacc, s, z[k]))
    if not cands:
        return None, None
    cands.sort(key=lambda x: x[0])
    _, s, cm = cands[len(cands) // 2]
    return cm, s


def tier_load(weights, ct, tiers):
    """Per-tier offered load from per-category weights, normalised to A_TOTAL."""
    a = tier_vec(weights, ct, tiers)
    return a / a.sum() * A_TOTAL


def tier_loads(sup, holdm, ct, tiers):
    """Flow-count and Erlang-corrected tier loads from supports and mean holding times."""
    return tier_load(sup, ct, tiers), tier_load(sup * holdm, ct, tiers)


def updown_flows(C, a, t):
    """Load moved up the AU ladder and down it by the misclassification C."""
    K = len(t)
    up = sum(a[i] * C[i, j] * (t[j] - t[i]) for i in range(K) for j in range(K) if t[j] > t[i])
    dn = sum(a[i] * C[i, j] * (t[i] - t[j]) for i in range(K) for j in range(K) if t[i] > t[j])
    return up, dn


def dimension(C, a, t):
    """Nominal V, relative overhead under C, the weighted bandwidth deficit, and which direction of misclassification dominates."""
    V = capacity_overhead(a, t, B_TARGET, V_start=1)
    dev = blocking_deviation(V, a, C, t)
    Vp = capacity_overhead(dev["a_hat"], t, B_TARGET, V_start=V)
    up, dn = updown_flows(C, a, t)
    return V, (Vp - V) / V, compute_wbd(C, a, t), ("lo->hi" if up > dn else "hi->lo")
