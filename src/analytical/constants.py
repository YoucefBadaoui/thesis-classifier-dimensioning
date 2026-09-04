"""Frozen numerical constants of the analytical scenarios."""

import numpy as np

# blocking-probability target, Ch.3 Sec. 3.2
B_TARGET_DEFAULT: float = 0.01

# OTT/IPTV scenario (Ch.3 Table 3.1; Ch.4 Table 4.2). CLASS_ORDER_OTT is alphabetical, while Ch.3 Table 3.1 lists the same five classes in ascending-t order; permute by class name, not by position.
CLASS_ORDER_OTT: tuple[str, ...] = ("Browsing", "Chat", "FileTransfer",
                                    "Streaming", "VoIP")
A_OTT: np.ndarray = np.array([25.0, 15.0, 20.0, 8.0, 20.0])  # Erlangs
T_OTT: np.ndarray = np.array([2, 1, 8, 15, 1])                # AUs
V_NOMINAL_OTT: int = 499                                       # AUs (B_target = 0.01)

# 5G slicing scenario parameters (Ch.5 Table 5.1).
CLASS_ORDER_5G: tuple[str, ...] = ("eMBB", "mMTC", "URLLC")
A_5G: np.ndarray = np.array([15.0, 10.0, 12.0])  # Erlangs
T_5G: np.ndarray = np.array([10, 1, 2])           # AUs
V_NOMINAL_5G: int = 277                            # AUs (B_target = 0.01)

# CESNET-TLS-Year22 scenario (Ch.4 / Ch.5). Two separate objects share the corpus: the 6-tier dimensioning object, from data/processed/cesnet_dimension.npz keys rec_cov_* and rec_V_*, and the K=23 Finding-F1 anchor, from data/processed/cesnet_highk_real.npz. Their V and cov(a, t) values are different quantities and must not be cross-wired. A_TOTAL_CESNET is numerically equal to the ISCX offered load (88 Erl), so tag the scenario (ISCX-OTT vs CESNET-6tier) wherever either appears.
A_TOTAL_CESNET: float = 88.0                       # Erlangs (CESNET 6-tier)
TIER_AU_CESNET: np.ndarray = np.array([1, 2, 4, 6, 10, 15])   # AUs per tier
V_NOMINAL_CESNET_COUNT: int = 679                  # AUs, flow-count prior, B_target = 0.01
V_NOMINAL_CESNET_ERLANG: int = 658                 # AUs, Erlang prior (headline, decision d)
# 6-tier offered loads (Erlangs), ascending AU tier [1, 2, 4, 6, 10, 15], copied from cesnet_dimension.npz keys rec_a_count / rec_a_erlang. The validator recomputes them against that NPZ.
A_CESNET_TIER_COUNT: np.ndarray = np.array(
    [11.3501, 23.2661, 17.3842, 10.8393, 9.9786, 15.1816])
A_CESNET_TIER_ERLANG: np.ndarray = np.array(
    [13.8649, 17.7760, 20.7457, 14.7409, 6.2662, 14.6063])

# Capacity-search upper bound. The largest V' any scenario needs is about 600 (OTT/IPTV at epsilon = 15%), so 2000 leaves a wide margin.
V_SEARCH_MAX: int = 2000

R_STEPS_SYSTEM: int = 100      # uniform-spillover whole-system search
R_STEPS_PER_CLASS: int = 200   # per-class isolated-class-k search

# Multi-link EFPA cascade (Model A: linear network with cross traffic). The OTT/IPTV classes traverse every link and carry the distortion a_hat = C^T a. Each link also carries a local copy of the same mix scaled by CASCADE_BG_FACTOR; without that local load an identical-route tandem collapses to a single FAG.
CASCADE_LINKS: int = 3
CASCADE_LINK_NAMES: tuple[str, ...] = ("access", "aggregation", "core")
CASCADE_BG_FACTOR: np.ndarray = np.array([0.40, 0.90, 0.60])  # local load / monitored load
