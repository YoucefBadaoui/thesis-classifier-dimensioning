"""Blocking-deviation pipelines for the 4-class VPN and 5G empirical CMs.

FP-VPN runs as a native 4-class scenario, dropping Browsing instead of padding the 5x5 archive entry with a zero row and column. The 12 published 5G CMs (3 Malkoc, 9 Islam) are renormalised before the bridge equation because they are read off published heatmaps.
"""

import numpy as np

from .kaufman_roberts import (
    uniform_spillover_cm,
    blocking_deviation, capacity_overhead, kaufman_roberts,
)
from .constants import A_OTT, T_OTT, A_5G, T_5G, B_TARGET_DEFAULT
from .published_cms import (
    FLOWPIC_CM_VPN_4x4, ISLAM_CLASSES_3, MALKOC_CLASSES_3, PUBLISHED_CMS,
)

# Permutation from thesis-alphabetical 4-class [Chat, FileTransfer, Streaming, VoIP] to FlowPic-native VPN order [VoIP, Video, FileTransfer, Chat]. Streaming <-> Video is a label remap, not a re-projection of probability mass.
PERM_VPN4_THESIS_TO_NATIVE = [3, 2, 1, 0]
CLASS_ORDER_VPN4 = ("Chat", "FileTransfer", "Streaming", "VoIP")
A_OTT_VPN4 = A_OTT[[1, 2, 3, 4]]   # drop Browsing (alphabetical index 0)
T_OTT_VPN4 = T_OTT[[1, 2, 3, 4]]

# The published 5G CMs, already in thesis-alphabetical [eMBB, mMTC, URLLC]
EMPIRICAL_5G_CMS: dict[str, np.ndarray] = {
    k: v["cm"] for k, v in PUBLISHED_CMS.items()
    if v["classes"] is MALKOC_CLASSES_3 or v["classes"] is ISLAM_CLASSES_3
}


def flowpic_vpn_4class_cm() -> tuple[np.ndarray, tuple[str, ...]]:
    """Return the FlowPic VPN matrix in thesis-alphabetical 4-class order.

    Browsing is dropped because the VPN portion of ISCX holds no Browsing flows (Shapira & Shavitt 2021, Table I, p. 1220). Both axes are permuted from FlowPic-native [VoIP, Video, FileTransfer, Chat] to thesis-alphabetical [Chat, FileTransfer, Streaming, VoIP].
    """
    perm = PERM_VPN4_THESIS_TO_NATIVE
    cm_thesis = FLOWPIC_CM_VPN_4x4[np.ix_(perm, perm)]
    return cm_thesis, CLASS_ORDER_VPN4


def compute_wbd(C: np.ndarray, a: np.ndarray, t: np.ndarray) -> float:
    """Weighted bandwidth deficit: sum over high-to-low misclassifications.

    WBD = sum_{i,j: t_i > t_j} a_i * C_{ij} * (t_i - t_j)

    Returns AU-erlangs (offered-load * AU-gap, per the dual-impact framing in Chapter 3).
    """
    K = len(a)
    total = 0.0
    for i in range(K):
        for j in range(K):
            if t[i] > t[j]:
                total += a[i] * C[i, j] * (t[i] - t[j])
    return float(total)


def make_5g_cm(r: float, spill: str) -> np.ndarray:
    """Row-stochastic 5G CM for the canonical spillover scenarios, diagonal recall r, in [eMBB, mMTC, URLLC] order.

    "uniform" spreads (1 - r) evenly over the off-diagonal cells of each row; "worst" sends it all to the highest-t class (eMBB) and "best" to the lowest-t class (mMTC). In the target class's own row the mass is spread over the other two classes so the row still sums to one.
    """
    K = len(T_5G)
    if spill == "uniform":
        return uniform_spillover_cm(K, r)
    if spill == "worst":
        target = int(np.argmax(T_5G))
    elif spill == "best":
        target = int(np.argmin(T_5G))
    else:
        raise ValueError(f"unknown spill mode {spill!r}; "
                         "expected 'uniform', 'worst', or 'best'")
    off = 1.0 - r
    C = np.zeros((K, K))
    for i in range(K):
        C[i, i] = r
        if i == target:
            C[i, [j for j in range(K) if j != i]] = off / (K - 1)
        else:
            C[i, target] = off
    return C


def _overhead(a_hat: np.ndarray, t: np.ndarray, B_target: float, V_nom: int):
    """Minimum capacity restoring B_target under a_hat, and the overhead in percent. Returns (-1, nan) when no V' below 2000 AU works."""
    try:
        V_prime = capacity_overhead(a_hat, t, B_target, V_start=V_nom, V_max=2000)
    except ValueError:
        return np.int64(-1), np.float64("nan")
    return np.int64(V_prime), np.float64((V_prime - V_nom) / V_nom * 100.0)


def flowpic_vpn_4class_results(B_target: float = B_TARGET_DEFAULT) -> dict[str, np.ndarray]:
    """End-to-end FP-VPN 4-class pipeline.

    Returns the same dict shape as the fp_vpn_4x4_* keys in data/processed/analytical_results.npz.
    """
    cm, classes = flowpic_vpn_4class_cm()
    a = A_OTT_VPN4
    t = T_OTT_VPN4
    V_nom = capacity_overhead(a, t, B_target, V_start=1, V_max=500)
    _, B_base = kaufman_roberts(V_nom, a, t)
    res = blocking_deviation(V_nom, a, cm, t)
    V_prime, overhead = _overhead(res["a_hat"], t, B_target, V_nom)
    return {
        "classes": np.array(classes),
        "cm": cm,
        "a_true": a,
        "t": t,
        "V_nominal": np.int64(V_nom),
        "B_baseline": B_base,
        "a_hat": res["a_hat"],
        "B_distorted": res["B_distorted"],
        "delta_B": res["delta_B"],
        "V_prime": V_prime,
        "overhead_pct": overhead,
        "WBD": np.float64(compute_wbd(cm, a, t)),
    }


def empirical_5g_results(B_target: float = B_TARGET_DEFAULT) -> dict[str, dict[str, np.ndarray]]:
    """Route all 12 published 5G CMs through the pipeline.

    Matrices read off heatmaps at three-decimal precision drift from row sum one by about 1e-4, so rows are renormalised here before the bridge equation. Returns a dict keyed by CM name, each entry shaped like the result of flowpic_vpn_4class_results.
    """
    V_nom = capacity_overhead(A_5G, T_5G, B_target, V_start=1, V_max=500)
    out: dict[str, dict[str, np.ndarray]] = {}
    for name, cm in EMPIRICAL_5G_CMS.items():
        rs = cm.sum(axis=1)
        cm_norm = cm / rs[:, None] if not np.allclose(rs, 1.0, atol=1e-9) else cm
        res = blocking_deviation(V_nom, A_5G, cm_norm, T_5G)
        V_prime, overhead = _overhead(res["a_hat"], T_5G, B_target, V_nom)
        out[name] = {
            "delta_B": res["delta_B"],
            "a_hat": res["a_hat"],
            "B_distorted": res["B_distorted"],
            "V_prime": V_prime,
            "overhead_pct": overhead,
            "WBD": np.float64(compute_wbd(cm_norm, A_5G, T_5G)),
            "balanced_acc": np.float64(np.mean(np.diag(cm_norm))),
        }
    return out

