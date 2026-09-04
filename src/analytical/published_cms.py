"""Confusion matrices transcribed from published papers.

Rows are true classes and columns predicted classes, so every matrix is row-stochastic and feeds the bridge equation in kaufman_roberts. PUBLISHED_CMS collects them with their paper, figure and dataset metadata. Matrices read off a published heatmap rather than a printed table are accurate to three decimals and are renormalised before use.
"""

import numpy as np


# FlowPic confusion matrices, Shapira & Shavitt (2021), Fig. 5, p. 1226. Dataset ISCX VPN-nonVPN (Draper-Gil et al., 2016); LeNet-5 style CNN on 1500x1500 FlowPic images; 90/10 split per encryption technique.

# Native FlowPic order, so cells read against Fig. 5. The thesis order is alphabetical, the convention used in data/processed/*.npz; perm = [4, 3, 2, 1, 0] maps thesis to native and FlowPic "Video" to thesis "Streaming". Apply as cm_thesis = FLOWPIC_CM[np.ix_(perm, perm)]; notebook 02 does this.
FLOWPIC_CLASSES_5 = ["VoIP", "Video", "File Transfer", "Chat", "Browsing"]

# VPN portion of ISCX has no Browsing traffic, so its native matrix is 4x4
FLOWPIC_CLASSES_VPN = ["VoIP", "Video", "File Transfer", "Chat"]

# Non-VPN, Fig. 5(a). Table III: balanced accuracy 85.0%, imbalanced 93.8%. This is the imbalanced evaluation, which is prevalence-weighted, so the diagonal mean (83.1%) matches neither figure. Chat recall is 27.8%, with 72.2% leaking to Browsing (p. 1227).
FLOWPIC_CM_NONVPN = np.array([
    [0.984, 0.000, 0.016, 0.000, 0.000],  # VoIP
    [0.000, 0.986, 0.000, 0.000, 0.014],  # Video
    [0.000, 0.000, 1.000, 0.000, 0.000],  # File Transfer
    [0.000, 0.000, 0.000, 0.278, 0.722],  # Chat
    [0.000, 0.000, 0.000, 0.091, 0.909],  # Browsing
], dtype=np.float64)

# VPN, Fig. 5(b), native 4x4 (no Browsing in the VPN portion, Table I, p. 1220). Table III: balanced accuracy 98.4%, imbalanced 97.6%.
FLOWPIC_CM_VPN_4x4 = np.array([
    [1.000, 0.000, 0.000, 0.000],  # VoIP
    [0.000, 1.000, 0.000, 0.000],  # Video
    [0.000, 0.000, 0.933, 0.067],  # File Transfer
    [0.000, 0.000, 0.000, 1.000],  # Chat
], dtype=np.float64)

# Same matrix zero-padded to 5x5 for the 5-class framework. The zero Browsing row and column mean Browsing is neither offered nor predicted.
FLOWPIC_CM_VPN = np.array([
    [1.000, 0.000, 0.000, 0.000, 0.000],  # VoIP
    [0.000, 1.000, 0.000, 0.000, 0.000],  # Video
    [0.000, 0.000, 0.933, 0.067, 0.000],  # File Transfer
    [0.000, 0.000, 0.000, 1.000, 0.000],  # Chat
    [0.000, 0.000, 0.000, 0.000, 0.000],  # Browsing (absent)
], dtype=np.float64)

# Tor, Fig. 5(c). Table III: balanced accuracy 67.8%, imbalanced 86.9%. Read at 400 dpi the Video diagonal is 100.0% and the File Transfer diagonal is 0.0%, which matches the paper's remark that the model failed on file transfer over Tor (p. 1227). The diagonal mean is 70.6% against the reported 67.8% balanced accuracy; the Ch.4 conventions note reconciles the two.
FLOWPIC_CM_TOR = np.array([
    [1.000, 0.000, 0.000, 0.000, 0.000],  # VoIP: 100% recall
    [0.000, 1.000, 0.000, 0.000, 0.000],  # Video: 100% recall
    [0.708, 0.000, 0.000, 0.125, 0.167],  # File Transfer: 0% recall (70.8% -> VoIP, 12.5% -> Chat, 16.7% -> Browsing)
    [0.000, 0.000, 0.000, 0.911, 0.089],  # Chat: 91.1% recall
    [0.000, 0.000, 0.024, 0.357, 0.619],  # Browsing: 61.9% recall
], dtype=np.float64)

# Merged non-VPN + VPN + Tor, Fig. 5(d). Table III: balanced accuracy 83.0%, no separate imbalanced figure. the VoIP and File Transfer rows read as identical values, which is implausible for two distinct classes.
FLOWPIC_CM_MERGED = np.array([
    [0.304, 0.009, 0.627, 0.017, 0.043],  # VoIP
    [0.000, 1.000, 0.000, 0.000, 0.000],  # Video
    [0.304, 0.009, 0.627, 0.017, 0.043],  # File Transfer
    [0.009, 0.000, 0.000, 0.735, 0.256],  # Chat
    [0.000, 0.000, 0.019, 0.212, 0.769],  # Browsing
], dtype=np.float64)


# Malkoc & Kholidy (2023), arXiv:2310.01747. Kaggle/DeepSlice synthetic 5G dataset (Thantharate et al., 2019), 31,584 train / 31,585 test rows, test support eMBB=3380, URLLC=1494, mMTC=1443 (total 6317). Task: 3-class slice identification. The paper labels classes numerically (1=eMBB, 2=URLLC, 3=mMTC); every matrix below is reordered to alphabetical (eMBB, mMTC, URLLC). The dataset is synthetic, so the perfect scores do not transfer to real 5G traffic.
MALKOC_CLASSES_3 = ["eMBB", "mMTC", "URLLC"]

# BNB/GNB, 94.19% overall accuracy. The only error is 367 of 1443 mMTC samples predicted as URLLC; the high accuracy comes from eMBB holding 53.5% of the test set.
MALKOC_CM_BNB = np.array([
    [1.000, 0.000, 0.000],  # eMBB:  100.0% recall
    [0.000, 0.746, 0.254],  # mMTC:   74.6% recall, 25.4% -> URLLC
    [0.000, 0.000, 1.000],  # URLLC: 100.0% recall
], dtype=np.float64)

# LDA, 76.35% overall accuracy. Worst case for the URLLC slice: all 1494 URLLC samples are predicted as mMTC, so URLLC is never dimensioned for.
MALKOC_CM_LDA = np.array([
    [1.000, 0.000, 0.000],  # eMBB:  100.0% recall
    [0.000, 1.000, 0.000],  # mMTC:  100.0% recall
    [0.000, 1.000, 0.000],  # URLLC:   0.0% recall (ALL predicted as mMTC)
], dtype=np.float64)

# LR, KNN, DT, RF and SVC all reach 100% on the synthetic test set, so their shared CM is the identity. The thesis uses it as the zero-error reference.
MALKOC_CM_PERFECT = np.eye(3, dtype=np.float64)


# Islam et al. (2025), PLOS ONE 20(10) e0333286. Same CRAWDAD/DeepSlice synthetic 5G dataset as Malkoc above, but a 70:30 split with VAE preprocessing. ML test support eMBB=37685, mMTC=39652, URLLC=62685 (total 140022); the DL models use far smaller test sets (DNF=594, MLP=445, NADAM-CNN=467). The paper's matrices are in axis order [URLLC, eMBB, mMTC] on both axes; every matrix below is reordered to alphabetical [eMBB, mMTC, URLLC]. For the five non-GaussianNB traditional models the eMBB-mMTC cross-cells are zero, so all error flows through URLLC.
ISLAM_CLASSES_3 = ["eMBB", "mMTC", "URLLC"]

# KNN, Fig. 11(a), p. 22. Table 4 reports 76% accuracy, recomputed 75.8%. Best of the traditional models.
ISLAM_CM_KNN = np.array([
    [0.7792, 0.0000, 0.2208],  # eMBB:  77.9% recall
    [0.0000, 0.7839, 0.2161],  # mMTC:  78.4% recall
    [0.1345, 0.1369, 0.7286],  # URLLC: 72.9% recall
], dtype=np.float64)

# Random Forest, Fig. 11(b), p. 22. Table 4 reports 69% accuracy, recomputed 69.5%.
ISLAM_CM_RF = np.array([
    [0.716704, 0.000000, 0.283296],  # eMBB:  71.7% recall
    [0.000000, 0.728791, 0.271209],  # mMTC:  72.9% recall
    [0.168860, 0.170583, 0.660557],  # URLLC: 66.1% recall
], dtype=np.float64)

# Decision Tree, Fig. 11(c), p. 22. Table 4 reports 69% accuracy, recomputed 69.5%. Most even per-class recall of the traditional models.
ISLAM_CM_DT = np.array([
    [0.6944, 0.0000, 0.3056],  # eMBB:  69.4% recall
    [0.0000, 0.7054, 0.2946],  # mMTC:  70.5% recall
    [0.1554, 0.1558, 0.6888],  # URLLC: 68.9% recall
], dtype=np.float64)

# GaussianNB, Fig. 11(d), p. 22. Table 4 reports 55% accuracy, recomputed 55.2%, the lowest of the nine models. URLLC recall is 0%: all 62685 URLLC samples go to eMBB (20%) or mMTC (80%), as in the Malkoc LDA case.
ISLAM_CM_GNB = np.array([
    [1.0000, 0.0000, 0.0000],  # eMBB:  100.0% recall
    [0.0000, 1.0000, 0.0000],  # mMTC:  100.0% recall
    [0.1999, 0.8001, 0.0000],  # URLLC:   0.0% recall (ALL misclassified)
], dtype=np.float64)

# BaggingClassifier, Fig. 11(e), p. 22. Table 4 reports 70% accuracy, recomputed 69.8%. The mMTC diagonal is read as 28832 rather than 28632, because only 28832 gives the row sum 39652 shared by all six ML models.
ISLAM_CM_BAG = np.array([
    [0.718987, 0.000000, 0.281013],  # eMBB:  71.9% recall
    [0.000000, 0.727126, 0.272874],  # mMTC:  72.7% recall
    [0.167058, 0.166068, 0.666874],  # URLLC: 66.7% recall
], dtype=np.float64)

# AdaBoost, Fig. 11(f), p. 22. Table 4 reports 69% accuracy, recomputed 69.5%.
ISLAM_CM_ADA = np.array([
    [0.7240, 0.0000, 0.2760],  # eMBB:  72.4% recall
    [0.0000, 0.7076, 0.2924],  # mMTC:  70.8% recall
    [0.1732, 0.1572, 0.6696],  # URLLC: 67.0% recall
], dtype=np.float64)

# Deep Neural Forest, Fig. 13(a), p. 24. recomputed accuracy 64.3% against 65% in Table 4, a 0.7pp gap. mMTC recall is 36.9%.
ISLAM_CM_DNF = np.array([
    [1.000000, 0.000000, 0.000000],  # eMBB:  100.0% recall
    [0.000000, 0.369048, 0.630952],  # mMTC:   36.9% recall (63.1% -> URLLC)
    [0.199248, 0.199248, 0.601504],  # URLLC:  60.2% recall
], dtype=np.float64)

# MLP, Fig. 13(b), p. 24. Table 4 reports 72% accuracy, recomputed 72.8%. mMTC recall falls below 50%.
ISLAM_CM_MLP = np.array([
    [0.8047, 0.0000, 0.1953],  # eMBB:  80.5% recall
    [0.0000, 0.4880, 0.5120],  # mMTC:  48.8% recall (51.2% -> URLLC)
    [0.0729, 0.0938, 0.8333],  # URLLC: 83.3% recall
], dtype=np.float64)

# NADAM-optimized CNN, the paper's proposed model, Fig. 13(c), p. 24. Table 4 reports 84% accuracy, recomputed 83.9%, the best of the nine. Off-diagonal mass is sparse: URLLC errors go only to eMBB, mMTC errors only to URLLC.
ISLAM_CM_CNN = np.array([
    [1.0000, 0.0000, 0.0000],  # eMBB:  100.0% recall
    [0.0000, 0.7328, 0.2672],  # mMTC:   73.3% recall (26.7% -> URLLC)
    [0.1980, 0.0000, 0.8020],  # URLLC:  80.2% recall (19.8% -> eMBB)
], dtype=np.float64)


_MALKOC = {
    "paper": "Malkoc & Kholidy, 2023",
    "doi": "10.48550/arXiv.2310.01747",
    "arxiv": "2310.01747",
    "dataset": (
        "Kaggle/DeepSlice synthetic 5G dataset (Thantharate et al., 2019). "
        "Test set: eMBB=3380, URLLC=1494, mMTC=1443 (total 6317)."
    ),
    "evaluation": "Single train/test split (31584 train / 31585 test)",
    "classes": MALKOC_CLASSES_3,
}

_ISLAM = {
    "paper": "Islam et al., 2025 (INBSI/NADAM-CNN)",
    "doi": "10.1371/journal.pone.0333286",
    "evaluation": "Single 70/30 train/test split with VAE preprocessing",
    "classes": ISLAM_CLASSES_3,
}

# the six traditional ML models share the large test split; the three DL models each report their own much smaller one
_ISLAM_ML = _ISLAM | {
    "dataset": (
        "CRAWDAD/DeepSlice synthetic 5G dataset (Thantharate et al., 2019). "
        "70:30 split. ML test set: eMBB=37685, mMTC=39652, URLLC=62685 "
        "(total 140022)."
    ),
}

PUBLISHED_CMS: dict[str, dict] = {
    "flowpic_nonvpn": {
        "paper": "Shapira & Shavitt, 2021 (FlowPic)",
        "doi": "10.1109/TNSM.2021.3071441",
        "source": "Figure 5(a), p. 1226",
        "dataset": "ISCX VPN-nonVPN (non-VPN portion)",
        "classifier": "LeNet-5 CNN on 1500x1500 FlowPic images",
        "evaluation": "Imbalanced dataset, 90/10 train/test split",
        "classes": FLOWPIC_CLASSES_5,
        "cm": FLOWPIC_CM_NONVPN,
        "reported_accuracy_balanced": 0.850,
        "reported_accuracy_imbalanced": 0.938,
    },
    "flowpic_vpn": {
        "paper": "Shapira & Shavitt, 2021 (FlowPic)",
        "doi": "10.1109/TNSM.2021.3071441",
        "source": "Figure 5(b), p. 1226",
        "dataset": "ISCX VPN-nonVPN (VPN portion)",
        "classifier": "LeNet-5 CNN on 1500x1500 FlowPic images",
        "evaluation": "Imbalanced dataset, 90/10 train/test split",
        "classes": FLOWPIC_CLASSES_5,
        "cm": FLOWPIC_CM_VPN,
        "cm_native_4x4": FLOWPIC_CM_VPN_4x4,
        "classes_native": FLOWPIC_CLASSES_VPN,
        "reported_accuracy_balanced": 0.984,
        "reported_accuracy_imbalanced": 0.976,
    },
    "flowpic_tor": {
        "paper": "Shapira & Shavitt, 2021 (FlowPic)",
        "doi": "10.1109/TNSM.2021.3071441",
        "source": "Figure 5(c), p. 1226",
        "dataset": "ISCX VPN-nonVPN (Tor portion)",
        "classifier": "LeNet-5 CNN on 1500x1500 FlowPic images",
        "evaluation": "Imbalanced dataset, 90/10 train/test split",
        "classes": FLOWPIC_CLASSES_5,
        "cm": FLOWPIC_CM_TOR,
        "reported_accuracy_balanced": 0.678,
        "reported_accuracy_imbalanced": 0.869,
    },
    "flowpic_merged": {
        "paper": "Shapira & Shavitt, 2021 (FlowPic)",
        "doi": "10.1109/TNSM.2021.3071441",
        "source": "Figure 5(d), p. 1226",
        "dataset": "ISCX VPN-nonVPN (merged: Non-VPN + VPN + Tor)",
        "classifier": "LeNet-5 CNN on 1500x1500 FlowPic images",
        "evaluation": "Balanced dataset (equal samples per class), "
                      "90/10 train/test split",
        "classes": FLOWPIC_CLASSES_5,
        "cm": FLOWPIC_CM_MERGED,
        "reported_accuracy_balanced": 0.830,
        "reliable": False,
    },
    "malkoc_bnb": _MALKOC | {
        "source": "Confusion matrix figure, BNB/GNB classifier result",
        "classifier": "Bernoulli/Gaussian Naive Bayes (BNB/GNB)",
        "cm": MALKOC_CM_BNB,
        "reported_accuracy": 0.9419,
    },
    "malkoc_lda": _MALKOC | {
        "source": "Confusion matrix figure, LDA classifier result",
        "classifier": "Linear Discriminant Analysis (LDA)",
        "cm": MALKOC_CM_LDA,
        "reported_accuracy": 0.7635,
    },
    "malkoc_perfect": _MALKOC | {
        "source": "Classifier results for LR, KNN, DT, RF, SVC",
        "classifier": (
            "Logistic Regression (LR), K-Nearest Neighbours (KNN), "
            "Decision Tree (DT), Random Forest (RF), Support Vector "
            "Classifier (SVC); all five achieve identical 100% accuracy"
        ),
        "cm": MALKOC_CM_PERFECT,
        "reported_accuracy": 1.0,
    },
    "islam_knn": _ISLAM_ML | {
        "source": "Figure 11(a), p. 22",
        "classifier": "K-Nearest Neighbours (KNN)",
        "cm": ISLAM_CM_KNN,
        "reported_accuracy": 0.76,
        "computed_accuracy": 0.7579,
    },
    "islam_rf": _ISLAM_ML | {
        "source": "Figure 11(b), p. 22",
        "classifier": "Random Forest (RF)",
        "cm": ISLAM_CM_RF,
        "reported_accuracy": 0.69,
        "computed_accuracy": 0.6950,
    },
    "islam_dt": _ISLAM_ML | {
        "source": "Figure 11(c), p. 22",
        "classifier": "Decision Tree (DT)",
        "cm": ISLAM_CM_DT,
        "reported_accuracy": 0.69,
        "computed_accuracy": 0.6950,
    },
    "islam_gnb": _ISLAM_ML | {
        "source": "Figure 11(d), p. 22",
        "classifier": "Gaussian Naive Bayes (GaussianNB)",
        "cm": ISLAM_CM_GNB,
        "reported_accuracy": 0.55,
        "computed_accuracy": 0.5523,
    },
    "islam_bag": _ISLAM_ML | {
        "source": "Figure 11(e), p. 22",
        "classifier": "BaggingClassifier",
        "cm": ISLAM_CM_BAG,
        "reported_accuracy": 0.70,
        "computed_accuracy": 0.6980,
    },
    "islam_ada": _ISLAM_ML | {
        "source": "Figure 11(f), p. 22",
        "classifier": "AdaBoost",
        "cm": ISLAM_CM_ADA,
        "reported_accuracy": 0.69,
        "computed_accuracy": 0.6950,
    },
    "islam_dnf": _ISLAM | {
        "source": "Figure 13(a), p. 24",
        "dataset": (
            "CRAWDAD/DeepSlice synthetic 5G dataset (Thantharate et al., 2019). "
            "70:30 split. DL test set: eMBB=160, mMTC=168, URLLC=266 "
            "(total 594)."
        ),
        "classifier": "Deep Neural Forest (DNF)",
        "cm": ISLAM_CM_DNF,
        "reported_accuracy": 0.65,
        "computed_accuracy": 0.6431,
    },
    "islam_mlp": _ISLAM | {
        "source": "Figure 13(b), p. 24",
        "dataset": (
            "CRAWDAD/DeepSlice synthetic 5G dataset (Thantharate et al., 2019). "
            "70:30 split. DL test set: eMBB=128, mMTC=125, URLLC=192 "
            "(total 445)."
        ),
        "classifier": "Multilayer Perceptron (MLP)",
        "cm": ISLAM_CM_MLP,
        "reported_accuracy": 0.72,
        "computed_accuracy": 0.7281,
    },
    "islam_cnn": _ISLAM | {
        "source": "Figure 13(c), p. 24",
        "dataset": (
            "CRAWDAD/DeepSlice synthetic 5G dataset (Thantharate et al., 2019). "
            "70:30 split. DL test set: eMBB=134, mMTC=131, URLLC=202 "
            "(total 467)."
        ),
        "classifier": "NADAM-optimized CNN (proposed INBSI model)",
        "cm": ISLAM_CM_CNN,
        "reported_accuracy": 0.84,
        "computed_accuracy": 0.8394,
    },
}


def validate_cm(cm: np.ndarray, name: str) -> list[str]:
    """Check row sums, non-negativity and per-class recall below 0.5; return the issues."""
    issues = []

    # this 1e-6 is looser than the atol=1e-9 bridge_equation enforces, so a CM read off a heatmap at 3-digit precision can pass here and still need row-normalising before use
    row_sums = cm.sum(axis=1)
    for i, rs in enumerate(row_sums):
        if rs > 0 and not np.isclose(rs, 1.0, atol=1e-6):
            issues.append(f"{name} row {i}: sum = {rs:.6f}, expected 1.0")

    if np.any(cm < 0):
        issues.append(f"{name}: contains negative values")

    # a zero row means the class is absent from the dataset, as Browsing is from the VPN portion of ISCX
    for i in range(cm.shape[0]):
        if row_sums[i] > 0 and cm[i, i] < 0.5:
            issues.append(
                f"{name} row {i}: diagonal ({cm[i,i]:.3f}) < 0.5 "
                f"(class has < 50% recall)"
            )

    return issues
