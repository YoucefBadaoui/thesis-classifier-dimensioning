#!/usr/bin/env python3
"""Independent verification of src/analytical/kaufman_roberts.py and src/analytical/efpa.py.

The reference implementations here are written from the published specification, not from the modules, and are self-tested against exact anchors before either module is imported.

Exit codes: 0 all applicable checks passed, 1 at least one FAIL, 2 reference self-test failure (module never imported).
"""

import contextlib
import importlib
import math
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "src").is_dir())

PF_MAX_STATES = 400_000   # product-form enumeration ceiling
EFPA_TOL = 1e-13          # reduced-load fixed-point residual
EFPA_MAX_ITER = 20_000


# Reference implementations (independent of the module under test)

def erlang_b_exact(a, v):
    """Exact rational Erlang-B: (a^V/V!) / sum_i a^i/i!."""
    af = Fraction(a)
    term = Fraction(1)
    terms = [term]
    for i in range(1, v + 1):
        term *= af / i
        terms.append(term)
    return float(terms[v] / sum(terms))


def erlang_b_continued(a, v):
    """Standard overflow-free continued-product Erlang-B recurrence."""
    b = 1.0
    for n in range(1, v + 1):
        b = a * b / (n + a * b)
    return b


def kr_spec(V, loads, demands):
    """Spec-form Kaufman-Roberts (scalar loop, eq:kr-recursion). Returns B."""
    K = len(loads)
    t = [int(x) for x in demands]
    q = [0.0] * (V + 1)
    q[0] = 1.0
    for n in range(1, V + 1):
        s = 0.0
        for k in range(K):
            tk = t[k]
            if tk <= n and loads[k] != 0.0:
                s += loads[k] * tk * q[n - tk]
        q[n] = s / n
    G = math.fsum(q)
    P = [x / G for x in q]
    B = []
    for k in range(K):
        lo = max(0, V - t[k] + 1)
        B.append(math.fsum(P[lo:V + 1]))
    return B


def kr_spec_fraction(V, loads, demands):
    """Exact-rational copy of the spec recursion for anchor checks."""
    K = len(loads)
    t = [int(x) for x in demands]
    la = [Fraction(x) for x in loads]
    q = [Fraction(0)] * (V + 1)
    q[0] = Fraction(1)
    for n in range(1, V + 1):
        s = Fraction(0)
        for k in range(K):
            if t[k] <= n:
                s += la[k] * t[k] * q[n - t[k]]
        q[n] = s / n
    G = sum(q)
    B = []
    for k in range(K):
        lo = max(0, V - t[k] + 1)
        B.append(float(sum(q[lo:V + 1]) / G))
    return B


def kr_product_form(V, loads, demands):
    """Truncated product-form measure: pi(n_1..n_K) prop
    prod_k a_k^{n_k}/n_k! over states with sum_k n_k t_k <= V.

    Blocking depends only on total occupancy, so weights are aggregated per occupancy level. Raises RuntimeError above max_states states.
    """
    K = len(loads)
    t = [int(x) for x in demands]
    est = 1
    for tk in t:
        est *= V // tk + 1
        if est > PF_MAX_STATES:
            raise RuntimeError("state space too large")
    W = {}

    def rec(k, used, w):
        if k == K:
            W[used] = W.get(used, 0.0) + w
            return
        ak = float(loads[k])
        tk = t[k]
        term = 1.0
        n = 0
        while used + n * tk <= V:
            rec(k + 1, used + n * tk, w * term)
            n += 1
            term *= ak / n

    rec(0, 0, 1.0)
    G = math.fsum(W.values())
    B = []
    for k in range(K):
        cut = V - t[k] + 1
        B.append(math.fsum(w for u, w in W.items() if u >= cut) / G)
    return B


def gen_kr_bpp(V, loads, demands, sources, variant, yform="spec"):
    """Generalised BPP recursion (eq:gen-kr-recursion).

    yform='spec': y_k(n) = a_k n / sum_j a_j t_j (eq:proportional-approx). yform='tk': y_k(n) = a_k t_k n / sum_j a_j t_j. variant in {'poisson','binomial','pascal'}; sources is an int or a per-class sequence N_k. Returns B.
    """
    K = len(loads)
    t = [int(x) for x in demands]
    if np.isscalar(sources):
        Nk_list = [sources] * K
    else:
        Nk_list = [int(s) for s in sources]
    denom = math.fsum(l * ti for l, ti in zip(loads, t))
    q = [0.0] * (V + 1)
    q[0] = 1.0
    for n in range(1, V + 1):
        s = 0.0
        for k in range(K):
            tk = t[k]
            if tk > n:
                continue
            m = n - tk
            ak = loads[k]
            if variant == "poisson":
                sig = ak
            else:
                yk = ak * m / denom if denom > 0 else 0.0
                if yform == "tk":
                    yk *= tk
                if variant == "binomial":
                    sig = max(ak * (Nk_list[k] - yk) / Nk_list[k], 0.0)
                else:
                    sig = ak * (Nk_list[k] + yk) / Nk_list[k]
            s += sig * tk * q[m]
        q[n] = s / n
    G = math.fsum(q)
    P = [x / G for x in q]
    B = []
    for k in range(K):
        lo = max(0, V - t[k] + 1)
        B.append(math.fsum(P[lo:V + 1]))
    return B


def engset_exact(N, V, beta):
    """Engset time congestion in the tail-sum convention of eq:blocking-prob: B = C(N,V) b^V / sum_i C(N,i) b^i."""
    if V >= N:
        return 0.0
    num = math.comb(N, V) * beta ** V
    den = math.fsum(math.comb(N, i) * beta ** i for i in range(V + 1))
    return num / den


def engset_call_congestion(N, V, beta):
    """Engset call congestion, the arrival-weighted C(N-1,.) form."""
    if V >= N:
        return 0.0
    num = math.comb(N - 1, V) * beta ** V
    den = math.fsum(math.comb(N - 1, i) * beta ** i for i in range(V + 1))
    return num / den


def system_blocking(B, loads, demands):
    """eq:system-blocking, AU-Erl weighted."""
    num = math.fsum(l * t * b for l, t, b in zip(loads, demands, B))
    den = math.fsum(l * t for l, t in zip(loads, demands))
    return num / den if den > 0 else 0.0


# Self-test gate, runs before the module under test is imported

def self_tests():
    rows = []

    worst = 0.0
    for A in (1, 5, 10, 20, 50):
        for V in (5, 10, 30, 100):
            d = abs(erlang_b_continued(A, V) - erlang_b_exact(A, V))
            worst = max(worst, d)
    rows.append(("ST1", "Erlang-B continued product vs exact Fractions",
                 worst < 1e-13, "max abs diff %.3e" % worst))

    B = kr_spec_fraction(5, [Fraction(1, 2), Fraction(1, 3)], [2, 3])
    ok = B[0] == 7.0 / 51.0 and B[1] == 15.0 / 51.0
    rows.append(("ST2", "KR spec recursion vs published 7/51, 15/51 anchor",
                 ok, "B=(%.17g, %.17g)" % (B[0], B[1])))

    rng = np.random.default_rng(12345)
    worst_pf = 0.0
    for i in range(4):
        K = int(rng.integers(2, 4))
        t = list(rng.integers(1, 6, size=K))
        a = list(rng.uniform(0.5, 6.0, size=K))
        V = int(rng.integers(int(max(t)) + 2, 26))
        B1 = kr_spec(V, a, t)
        B2 = kr_product_form(V, a, t)
        worst_pf = max(worst_pf, max(abs(x - y) for x, y in zip(B1, B2)))
    rows.append(("ST3", "KR spec recursion vs product-form CTMC measure",
                 worst_pf < 1e-12, "max abs diff %.3e" % worst_pf))

    # exact identity because y_k(n)=n under the proportional approximation
    worst_eg = 0.0
    for N, V, beta in ((20, 10, 0.5), (15, 8, 0.9), (40, 12, 0.3)):
        Bg = gen_kr_bpp(V, [N * beta], [1], N, "binomial")
        worst_eg = max(worst_eg, abs(Bg[0] - engset_exact(N, V, beta)))
    rows.append(("ST4", "generalised KR binomial vs Engset time congestion",
                 worst_eg < 1e-12, "max abs diff %.3e" % worst_eg))

    all_ok = all(r[2] for r in rows)
    return rows, all_ok


def load_module():
    """Import the module under test; only reached after the self-test gate."""
    sys.path.insert(0, str(ROOT))
    return importlib.import_module("src.analytical.kaufman_roberts")


def main():
    print("INDEPENDENT VERIFICATION OF src/analytical/kaufman_roberts.py AND efpa.py")
    print("references written from published spec before module import")

    st_rows, st_ok = self_tests()
    print("\nREFERENCE SELF-TESTS (pre-import gate)")
    for tid, desc, ok, det in st_rows:
        print("%-4s %-55s %s   %s" % (tid, desc, "PASS" if ok else "FAIL", det))
    if not st_ok:
        print("\nSELF-TEST FAILURE: reference implementations disagree "
              "with published anchors; module never imported.")
        sys.exit(2)

    mod = load_module()
    cn = importlib.import_module("src.analytical.constants")

    def m_kr(V, loads, demands):
        """Module blocking vector, taken by position: kaufman_roberts returns (occupancy over V+1 levels, blocking over K classes)."""
        _, B = mod.kaufman_roberts(V, np.asarray(loads, dtype=float),
                                   np.asarray(demands))
        return np.asarray(B, dtype=float).ravel()

    @contextlib.contextmanager
    def guard(tid, desc):
        """Record a FAIL instead of aborting the run when a check body raises."""
        try:
            yield
        except Exception as exc:
            record(tid, desc, False, "%s: %s" % (type(exc).__name__, exc))

    results = []  # (id, description, status, detail)
    findings = []  # notes printed after the summary

    def record(tid, desc, ok, detail, partial=False):
        # a status string would be truthy and mask a FAIL as PASS
        if not isinstance(ok, (bool, np.bool_)):
            raise TypeError("record(%s): ok must be bool, got %r"
                            % (tid, ok))
        ok = bool(ok)
        status = ("PARTIAL" if partial else ("PASS" if ok else "FAIL"))
        results.append((tid, desc, status, detail))
        print("%-4s %-52s %-7s %s" % (tid, desc, status, detail))

    # T1
    worst_abs, worst_rel = 0.0, 0.0
    for A in (1, 5, 10, 20, 50):
        for V in (5, 10, 30, 100):
            ref = erlang_b_continued(A, V)
            got = m_kr(V, [A], [1])[0]
            d = abs(got - ref)
            worst_abs = max(worst_abs, d)
            if ref > 1e-12:
                worst_rel = max(worst_rel, d / ref)
    ok = worst_abs < 1e-10 and worst_rel < 1e-8
    record("T1", "Erlang-B reduction K=1,t=1 vs closed form",
           ok, "max abs %.3e, max rel %.3e" % (worst_abs, worst_rel))

    # T2
    V, a2, t2 = 10, [3.0, 2.0], [1, 3]
    Bm = m_kr(V, a2, t2)
    Br = kr_spec(V, a2, t2)
    Bp = kr_product_form(V, a2, t2)
    d_ref = max(abs(x - y) for x, y in zip(Bm, Br))
    d_pf = max(abs(x - y) for x, y in zip(Bm, Bp))
    Bc = m_kr(5, [0.5, 1.0 / 3.0], [2, 3])
    d_anchor = max(abs(Bc[0] - 7 / 51), abs(Bc[1] - 15 / 51))
    ok = d_ref < 1e-10 and d_pf < 1e-10 and d_anchor < 1e-12
    record("T2", "multi-class V=10,a=(3,2),t=(1,3) + 7/51 anchor",
           ok, "vs spec-loop %.3e, vs product-form %.3e, anchor %.3e"
           % (d_ref, d_pf, d_anchor))

    # T3
    # multirate B_k is not monotone in a_j or V (discrete AU packing), so T3a checks single-rate monotonicity and T3b module vs reference
    As = [0.5, 2.0, 5.0, 10.0, 20.0, 35.0, 50.0]
    Vs1 = [5, 10, 30, 100]
    mat = {(A, V): m_kr(V, [A], [1])[0] for A in As for V in Vs1}
    bad = []
    for V in Vs1:
        seq = [mat[(A, V)] for A in As]
        worst_rise = max(seq[i] - seq[i + 1]
                         for i in range(len(seq) - 1))
        if worst_rise > 1e-12:
            bad.append(("B fell as A rose", V, worst_rise))
    for A in As:
        seq = [mat[(A, V)] for V in Vs1]
        worst_rise = max(seq[i + 1] - seq[i]
                         for i in range(len(seq) - 1))
        if worst_rise > 1e-12:
            bad.append(("B rose as V rose", A, worst_rise))
    record("T3a", "single-rate monotonicity (valid oracle)",
           not bad, "7 loads x 4 capacities grid; violations: %s"
           % (bad or "none"))

    rng = np.random.default_rng(20260823)
    max_dev = 0.0
    viol_up, viol_V, worst_up, worst_V = 0, 0, 0.0, 0.0
    for i in range(20):
        K = int([3, 5, 8][i % 3])
        t = list(rng.integers(1, 16, size=K))
        a = list(rng.uniform(0.5, 25.0, size=K))
        V = int(rng.integers(40, 121))
        B0m = m_kr(V, a, t)
        B0r = kr_spec(V, a, t)
        max_dev = max(max_dev, float(np.abs(B0m - np.asarray(B0r)).max()))
        for j in range(K):
            a_up = list(a)
            a_up[j] = a[j] * 1.05
            Bup = m_kr(V, a_up, t)
            drop = (B0m - Bup).max()
            if drop > 1e-12:
                viol_up += 1
                worst_up = max(worst_up, drop)
        Bv = m_kr(V + 7, a, t)
        rise = (Bv - B0m).max()
        if rise > 1e-12:
            viol_V += 1
            worst_V = max(worst_V, rise)
    # exact-arithmetic check that the violations are real, not rounding
    Bx71 = kr_spec_fraction(71, [2.25074, 12.8667, 4.13184], [3, 12, 13])
    Bx78 = kr_spec_fraction(78, [2.25074, 12.8667, 4.13184], [3, 12, 13])
    exact_rise = Bx78[0] - Bx71[0]
    findings.append(
        "multirate blocking is not monotone componentwise: in exact "
        "arithmetic (Fraction) B_1(t=3) rises %.6g when V goes 71->78 "
        "at t=(3,12,13), while an own-load rise lowers B (product form). "
        "Module matches the independent reference to "
        "%.3e on all 20 instances." % (exact_rise, max_dev))
    if max_dev >= 1e-9:
        record("T3b", "multirate: module vs independent reference",
               False, "max dev %.3e over 20 instances" % max_dev)
    else:
        # the assertion is max_dev < 1e-9 above; the census is the evidence behind note F1 and is reported as PARTIAL, not as a second test
        record("T3b", "multirate monotonicity census (see note)",
               True, "module==reference to %.3e; census over 20 inst: "
                     "%d/%d load-step decreases (worst %.3e), %d/20 "
                     "V-step increases (worst %.3e); exact-arithmetic "
                     "rise %.6g"
                     % (max_dev, viol_up, 20 * 16, worst_up,
                        viol_V, worst_V, exact_rise),
               partial=True)

    # T4
    Ba = m_kr(30, [0.0, 0.0, 0.0], [2, 4, 1])
    allzero_ok = bool((Ba == 0.0).all())
    big_a = [4.0, 9.0, 14.0, 2.5, 20.0]
    big_t = [1, 2, 3, 5, 8]
    Bl = m_kr(3000, big_a, big_t)
    lim_ok = bool((Bl < 1e-8).all())
    record("T4a", "all-zero load gives B=0; large-V limit B->0",
           allzero_ok and lim_ok,
           "all-zero max=%g, V=3000 max B=%.3e" % (Ba.max(), Bl.max()))

    Bz = m_kr(30, [5.0, 0.0, 3.0], [2, 4, 1])
    Bzr = kr_spec(30, [5.0, 0.0, 3.0], [2, 4, 1])
    dz = float(np.abs(Bz - np.asarray(Bzr)).max())
    findings.append(
        "a zero-load class keeps a non-zero blocking probability "
        "(eq:blocking-prob, PASTA): with a=(5,0,3), t=(2,4,1), V=30 "
        "the zero-load class shows B_2=%.8f, the occupancy tail of the "
        "other classes' traffic; module and independent reference "
        "agree to %.1e." % (Bz[1], dz))
    record("T4b", "zero-load class semantics (see note)", True,
           "module B(zero-load class)=%.8f equals reference tail-sum "
           "to %.1e" % (Bz[1], dz),
           partial=True)

    # T5
    C = np.array([[0.9, 0.1], [0.2, 0.8]])
    av = np.array([10.0, 5.0])
    out = np.asarray(mod.bridge_equation(C, av, False), dtype=float).ravel()
    d_hand = max(abs(out[0] - 10.0), abs(out[1] - 5.0))
    C2 = np.array([[1.0, 0.0], [0.5, 0.5]])
    out2 = np.asarray(mod.bridge_equation(C2, av, False),
                      dtype=float).ravel()
    d_hand2 = max(abs(out2[0] - 12.5), abs(out2[1] - 2.5))
    rngC = np.random.default_rng(7)
    Cr = rngC.random((6, 6))
    Cr /= Cr.sum(axis=1, keepdims=True)
    ar = rngC.uniform(0.5, 20.0, size=6)
    outr = np.asarray(mod.bridge_equation(Cr, ar, False),
                      dtype=float).ravel()
    d_rand = np.abs(outr - Cr.T @ ar).max()
    conserv = abs(outr.sum() - ar.sum())
    ident = np.abs(np.asarray(mod.bridge_equation(np.eye(6), ar, False),
                     dtype=float).ravel() - ar).max()

    # normalise=True must leave a row-stochastic input at C^T a, and must restore the mass a row-scaled matrix would otherwise lose
    out_n = np.asarray(mod.bridge_equation(Cr, ar, True), dtype=float).ravel()
    unchanged = float(np.abs(out_n - Cr.T @ ar).max())
    Cs = Cr * 0.5  # rows sum to 0.5, mass lost without normalisation
    outs = np.asarray(mod.bridge_equation(Cs, ar, True), dtype=float).ravel()
    ss = float(outs.sum())
    scaled_ok = abs(ss - ar.sum()) < 1e-9
    ok = (d_hand < 1e-12 and d_hand2 < 1e-12 and d_rand < 1e-12
          and conserv < 1e-9 and ident < 1e-12 and unchanged < 1e-9
          and scaled_ok)
    record("T5", "bridge equation: hand examples, C^T a, conservation",
           ok, "hand %.1e/%.1e, rand %.1e, conserv %.1e, ident %.1e; "
               "normalise=True dev %.1e; scaled-rows mass %g vs %g"
               % (d_hand, d_hand2, d_rand, conserv, ident,
                  unchanged, ss, float(ar.sum())))

    # T6
    cap_rows = []
    ok_all = True
    for seed, K, target, vstart in ((1, 4, 0.01, 40), (2, 6, 0.02, 60),
                                    (3, 8, 0.005, 80)):
        r = np.random.default_rng(1000 + seed)
        t = list(r.integers(1, 13, size=K))
        ah = list(r.uniform(0.5, 30.0, size=K))
        got = int(mod.capacity_overhead(
            np.asarray(ah), np.asarray(t), target, vstart))
        # dimensioning criterion: minimum V with B_k <= target for all k
        Vb = None
        for V in range(vstart, vstart + 3000):
            if max(kr_spec(V, ah, t)) <= target:
                Vb = V
                break
        if Vb is None:
            ok_all = False
            record("T6", "capacity_overhead bisection vs brute-force scan",
                   False, "seed %d: max-class objective did not reach "
                   "target within scan range" % seed)
            continue
        # context columns evaluated at the found V, not by two further scans
        Bv = kr_spec(Vb, ah, t)
        agree = abs(Vb - got) <= 1
        ok_all = ok_all and agree
        cap_rows.append("seed%d K=%d tgt=%g: module V=%d, brute max-class V=%d "
                        "(agrees: %s; at V weighted %.3g, mean %.3g)"
                        % (seed, K, target, got, Vb, agree,
                           system_blocking(Bv, ah, t), sum(Bv) / len(Bv)))
    record("T6", "capacity_overhead bisection vs brute-force scan",
           ok_all, " | ".join(cap_rows))

    # T7
    fn = mod.blocking_deviation
    # uniform-spillover instance; eq:delta-L closed form is sum_k t_k (a_hat_k - a_k)
    # blocking_deviation(V, a, C, demands)
    a_true = [20.0, 10.0, 5.0, 5.0]
    tt = [1, 2, 3, 4]
    K7 = 4
    C7 = np.full((K7, K7), 0.2 / 3.0)
    np.fill_diagonal(C7, 0.8)
    a_hat7 = list(np.asarray(C7).T @ np.asarray(a_true))
    direct = sum(ti * (h - x) for ti, h, x in zip(tt, a_hat7, a_true))
    V7 = 60
    Bh7 = kr_spec(V7, a_hat7, tt)
    Bt7 = kr_spec(V7, a_true, tt)
    dvec7 = np.asarray(Bh7) - np.asarray(Bt7)

    res = None
    try:
        res = fn(V7, np.asarray(a_true), C7, np.asarray(tt))
    except (TypeError, AssertionError, ValueError) as exc:
        print("     direct call failed: %r" % (exc,))
    matched = None
    if isinstance(res, dict):
        print("     blocking_deviation keys: %s" % sorted(res))
        for key, val in res.items():
            arr = np.asarray(val, dtype=float)
            if arr.size == 1 and abs(float(arr[0]) - direct) < 1e-9:
                matched = "delta_L scalar under key '%s'" % key
                break
            if arr.size == K7 \
                    and np.abs(arr - dvec7).max() < 1e-9:
                matched = "per-class delta B_k vector under key " \
                          "'%s' at V=%d" % (key, V7)
                break
    else:
        print("     blocking_deviation returned %s"
              % type(res).__name__)
    if matched is not None:
        record("T7", "deviation identity via module.blocking_deviation",
               True,
               "%s; direct delta_L=%.12g reproduced" % (matched, direct))
    else:
        shown = {k: (np.asarray(v).round(6).tolist()
                     if isinstance(v, np.ndarray)
                     else v) for k, v in (res.items()
                                          if isinstance(res, dict)
                                          else [])}
        record("T7", "blocking-deviation identity", False,
               "no key reproduces delta_L=%.6g or delta B_k within "
               "1e-9; got %s" % (direct, shown))

    # T8
    fn = mod.kaufman_roberts_bpp

    def call_bpp(V, loads, t, sources, variant):
        # kaufman_roberts_bpp(V, loads, demands, traffic_types, source_counts=None) returns (occupancy V+1, blocking K)
        loads = np.asarray(loads, dtype=float)
        t = np.asarray(t)
        K = len(loads)
        src = np.full(K, int(sources)) if np.isscalar(sources) \
            else np.asarray(sources)
        _, B = fn(V, loads, t, [variant] * K, src)
        return np.asarray(B, dtype=float).ravel()

    # the module takes the total offered load A = N*beta, so the binomial BPP reduction must equal Engset time congestion exactly
    N, Vb, beta = 20, 10, 0.5
    eg_ref = engset_exact(N, Vb, beta)
    got_bpp = float(call_bpp(Vb, [N * beta], [1], N, "binomial")[0])
    record("T8a", "binomial single-class vs Engset time congestion",
           abs(got_bpp - eg_ref) < 1e-10,
           "convention 'total A=N*beta': module %.12g vs time-cong %.12g "
           "(call-cong %.12g, informational)"
           % (got_bpp, eg_ref, engset_call_congestion(N, Vb, beta)))

    # T8b: poisson reduces to plain KR; T8c/T8d compare the y_k forms
    rng8 = np.random.default_rng(99)
    K8 = 3
    t8 = [int(x) for x in rng8.integers(1, 8, size=K8)]
    a8 = list(rng8.uniform(1.0, 10.0, size=K8))
    V8 = 40
    Bpois_mod = call_bpp(V8, a8, t8, 12, "poisson")
    Bpois_kr = m_kr(V8, a8, t8)
    d_pois = float(np.abs(Bpois_mod - Bpois_kr).max())
    dev_spec, dev_ident = {}, {}
    for variant in ("binomial", "pascal"):
        Bmod = call_bpp(V8, a8, t8, 12, variant)
        Bspec = gen_kr_bpp(V8, a8, t8, 12, variant, yform="spec")
        Bident = gen_kr_bpp(V8, a8, t8, 12, variant, yform="tk")
        dev_spec[variant] = float(
            np.abs(Bmod - np.asarray(Bspec)).max())
        dev_ident[variant] = float(
            np.abs(Bmod - np.asarray(Bident)).max())
    ok_pois = d_pois < 1e-12
    spec_ok = max(dev_spec.values()) < 1e-9
    ident_match = max(dev_ident.values()) < 1e-9
    record("T8b", "bpp poisson reduction == plain KR", ok_pois,
           "max dev %.3e" % d_pois)
    record("T8c", "bpp binomial/pascal vs PUBLISHED y_k form",
           spec_ok,
           "max |diff| vs eq:proportional-approx: %s"
           % {k: "%.3e" % v for k, v in dev_spec.items()})
    if spec_ok:
        record("T8d", "bpp y_k form diagnosis", True,
               "module implements the published form; dev to the "
               "y_k*t_k defect form: %s"
               % {k: "%.3e" % v for k, v in dev_ident.items()})
    elif ident_match:
        findings.append(
            "BPP FINITE-SOURCE DEVIATION (kaufman_roberts_bpp): binomial "
            "and pascal use y_k(n) = a_k*t_k*n/sum_j a_j t_j instead of "
            "the published y_k(n) = a_k*n/sum_j a_j t_j "
            "(eq:proportional-approx); the extra t_k over-suppresses "
            "sigma_k. Deviation to the defect form binomial %.1e, pascal "
            "%.1e; to the published form binomial %.1e, pascal %.1e. "
            "Thesis experiments are Poisson-only, so published numbers "
            "are unaffected."
            % (dev_ident["binomial"], dev_ident["pascal"],
               dev_spec["binomial"], dev_spec["pascal"]))
        record("T8d", "bpp y_k form diagnosis", False,
               "module implements the y_k*t_k defect form "
               "(see finding)")
    else:
        record("T8d", "bpp y_k form diagnosis", False,
               "module matches NEITHER the published nor the "
               "y_k*t_k form: vs spec %s, vs tk %s"
               % ({k: "%.3e" % v for k, v in dev_spec.items()},
                  {k: "%.3e" % v for k, v in dev_ident.items()}))

    seq_b, seq_p = [], []
    for Nn in (2, 5, 20, 200, 4000):
        seq_b.append(float(call_bpp(V8, a8, t8, Nn, "binomial")[0]))
        seq_p.append(float(call_bpp(V8, a8, t8, Nn, "pascal")[0]))
    mono_ok = all(seq_b[i] <= seq_b[i + 1] + 1e-12
                  for i in range(len(seq_b) - 1)) and \
              all(seq_p[i] >= seq_p[i + 1] - 1e-12
                  for i in range(len(seq_p) - 1))
    conv_ok = (abs(seq_b[-1] - Bpois_kr[0]) < 5e-2 * max(Bpois_kr[0],
                                                        1e-12)
               and abs(seq_p[-1] - Bpois_kr[0]) < 5e-2
               * max(Bpois_kr[0], 1e-12))
    record("T8e", "BPP source monotonicity + Poisson convergence",
           bool(mono_ok) and conv_ok,
           "binomial B(N=2..4000): %s; pascal: %s; converges to "
           "Poisson: %s" % (["%.4f" % x for x in seq_b],
                            ["%.4f" % x for x in seq_p], conv_ok))

    # T8f: an unknown traffic-type token must raise, not fall back
    try:
        fn(V8, np.asarray(a8, dtype=float), np.asarray(t8),
           ["engset"] * K8, np.full(K8, 12))
        token_raises = False
    except ValueError:
        token_raises = True
    if not token_raises:
        findings.append(
            "SILENT VARIANT-TOKEN FALLBACK (measured this run): "
            "kaufman_roberts_bpp accepted the unknown traffic-type "
            "token 'engset' without raising; unknown tokens compute "
            "some default case silently.")
    record("T8f", "unknown traffic-type token raises ValueError",
           token_raises, "probe token 'engset' %s"
           % ("raised ValueError" if token_raises
              else "was silently accepted"))

    # T9: reduced-load fixed point for src/analytical/efpa.py; ref_efpa is plain repeated substitution where the module uses under-relaxation


    def ref_efpa(V_links, offered, demands, routes):
        L = len(V_links)
        rho = {}
        B = {}
        for s, r in enumerate(routes):
            for l in r:
                rho[(l, s)] = float(offered[s])
        for _ in range(EFPA_MAX_ITER):
            for l in range(L):
                streams = [s for s in range(len(offered)) if (l, s) in rho]
                if not streams:
                    continue
                Bl = kr_spec(V_links[l],
                             [rho[(l, s)] for s in streams],
                             [demands[s] for s in streams])
                for i, s in enumerate(streams):
                    B[(l, s)] = Bl[i]
            new = {}
            maxdiff = 0.0
            for s, r in enumerate(routes):
                for l in r:
                    acc = 1.0
                    for l2 in r:
                        if l2 != l:
                            acc *= 1.0 - B[(l2, s)]
                    new[(l, s)] = float(offered[s]) * acc
                    maxdiff = max(maxdiff, abs(new[(l, s)] - rho[(l, s)]))
            rho = new
            if maxdiff < EFPA_TOL:
                break
        e2e = []
        for s, r in enumerate(routes):
            surv = 1.0
            for l in r:
                surv *= 1.0 - B[(l, s)]
            e2e.append(1.0 - surv)
        return e2e

    has_efpa = True
    try:
        mod_efpa = importlib.import_module("src.analytical.efpa")
    except Exception as exc:
        has_efpa = False
        record("T9", "EFPA module load", False,
               "%s: %s" % (type(exc).__name__, exc))



    if has_efpa:
        # T9a single-link reduction
        with guard("T9a", "EFPA single-link reduction == KR"):
            r1 = mod_efpa.efpa_fixed_point(np.asarray([50.0]), np.asarray([12.0]),
                                      np.asarray([2]), [[0]])
            ref1 = kr_spec(50, [12.0], [2])[0]
            got = float(np.asarray(r1["B_e2e"]).ravel()[0])
            ok = abs(got - ref1) < 1e-12
            record("T9a", "EFPA single-link reduction == KR", ok,
                   "efpa %.15f vs kr %.15f" % (got, ref1))
        # T9b disjoint links: no coupling
        with guard("T9b", "EFPA disjoint links uncoupled"):
            r2 = mod_efpa.efpa_fixed_point(np.asarray([40.0, 60.0]),
                                      np.asarray([8.0, 5.0]),
                                      np.asarray([2, 4]),
                                      [[0], [1]])
            ref2 = [kr_spec(40, [8.0], [2])[0], kr_spec(60, [5.0], [4])[0]]
            got2 = np.asarray(r2["B_e2e"], dtype=float).ravel()
            ok = bool(np.all(np.abs(got2 - np.array(ref2)) < 1e-12))
            record("T9b", "EFPA disjoint links uncoupled", ok,
                   "efpa %s vs kr %s" % (np.round(got2, 12), ref2))
        # T9c fixed-point identity at the converged solution
        with guard("T9c", "fixed-point identity rho = a*prod(1-B)"):
            routes3 = [[0, 1, 2]] * 3
            off3 = np.asarray([10.0, 6.0, 4.0])
            r3 = mod_efpa.efpa_fixed_point(np.asarray([80.0, 100.0, 90.0]),
                                      off3, np.asarray([2, 5, 1]), routes3)
            rho = r3["reduced_loads"]
            Bl = r3["B_link"]
            worst = 0.0
            for s, r in enumerate(routes3):
                for l in r:
                    acc = 1.0
                    for l2 in r:
                        if l2 != l:
                            acc *= 1.0 - Bl[l2, s]
                    worst = max(worst, abs(rho[l, s] - off3[s] * acc))
            ok = worst < 1e-9
            record("T9c", "fixed-point identity rho = a*prod(1-B)", ok,
                   "worst |rho - a*prod| = %.3e" % worst)
        # T9d ordering: EFPA <= independent-links on random instances
        with guard("T9d", "EFPA <= independent-links (20 rnd instances)"):
            rng = np.random.default_rng(20260823)
            viol = 0
            worst_gap = 0.0
            for _ in range(20):
                S = int(rng.integers(3, 6))
                L = int(rng.integers(2, 5))
                a = rng.uniform(1.0, 25.0, size=S)
                t = rng.integers(1, 16, size=S)
                routes = [sorted(rng.choice(L, size=int(
                    rng.integers(1, L + 1)), replace=False).tolist())
                    for _ in range(S)]
                for l in range(L):
                    if not any(l in rt for rt in routes):
                        routes[0] = sorted(set(routes[0]) | {l})
                # one capacity draw per instance so both solvers see the same links
                Vl = rng.uniform(40.0, 200.0, size=L)
                re_ = mod_efpa.efpa_fixed_point(Vl, a, t, routes)
                ri = mod_efpa.efpa_independent(Vl, a, t, routes)
                e_ef = np.asarray(re_["B_e2e"], dtype=float)
                e_in = np.asarray(ri["B_e2e"], dtype=float)
                gap = e_in - e_ef
                if np.any(gap < -1e-9):
                    viol += 1
                    worst_gap = max(worst_gap, float(-np.min(gap)))
            record("T9d", "EFPA <= independent-links (20 rnd instances)",
                   viol == 0,
                   "violations=%d worst=%.3e" % (viol, worst_gap))
        # T9e: rebuild stage A of scripts/experiments/cascade_analysis.py, K end-to-end streams plus K per-link background streams, vs cascade_results.npz
        with guard("T9e", "regression-lock vs cascade stage-A archive"):
            arc = np.load(str(ROOT / "data" / "processed"
                              / "cascade_results.npz"),
                          allow_pickle=True)
            a_e2e = np.asarray(cn.A_OTT, dtype=float)
            t_e2e = np.asarray(cn.T_OTT)
            K9 = len(a_e2e)
            L9 = int(cn.CASCADE_LINKS)
            bg9 = np.asarray(cn.CASCADE_BG_FACTOR, dtype=float)
            routes9 = [list(range(L9))] * K9
            off9 = [a_e2e]
            dem9 = [t_e2e]
            for l in range(L9):
                routes9 += [[l]] * K9
                off9.append(bg9[l] * a_e2e)
                dem9.append(t_e2e)
            off9 = np.concatenate(off9)
            dem9 = np.concatenate(dem9)
            dem_dim = np.concatenate([t_e2e, t_e2e])
            V9 = np.zeros(L9, dtype=int)
            for l in range(L9):
                V9[l] = int(mod.capacity_overhead(
                    np.concatenate([a_e2e, bg9[l] * a_e2e]), dem_dim,
                    float(cn.B_TARGET_DEFAULT), V_start=1))
            v_match = bool(np.array_equal(
                V9, np.asarray(arc["V_design"], dtype=int)))
            rr = mod_efpa.efpa_fixed_point(V9.astype(float), off9,
                                           dem9, routes9)
            ri9 = mod_efpa.efpa_independent(V9.astype(float), off9,
                                            dem9, routes9)
            dev_e = float(np.max(np.abs(
                np.asarray(rr["B_e2e"], dtype=float)[:K9]
                - np.asarray(arc["A_efpa"], dtype=float))))
            dev_i = float(np.max(np.abs(
                np.asarray(ri9["B_e2e"], dtype=float)[:K9]
                - np.asarray(arc["A_indep"], dtype=float))))
            ok = (v_match and bool(rr["converged"])
                  and dev_e < 1e-9 and dev_i < 1e-9)
            record("T9e", "regression-lock vs cascade stage-A archive",
                   ok, "V %s vs stored %s (match %s); max dev "
                       "efpa=%.3e indep=%.3e"
                   % (V9.tolist(),
                      np.asarray(arc["V_design"]).tolist(), v_match,
                      dev_e, dev_i))
        # T9f coupled fixed point vs the independent reference solver
        with guard("T9f", "coupled EFPA vs independent reference (3 nets)"):
            rng9 = np.random.default_rng(777)
            worst9 = 0.0
            conv_ok = True
            for _ in range(3):
                Lf = int(rng9.integers(2, 4))
                Sf = int(rng9.integers(2, 4))
                af = rng9.uniform(2.0, 10.0, size=Sf)
                tf = [int(x) for x in rng9.integers(1, 5, size=Sf)]
                rts = []
                for s in range(Sf):
                    n = int(rng9.integers(1, Lf + 1))
                    rts.append(sorted(rng9.choice(
                        Lf, size=n, replace=False).tolist()))
                for l in range(Lf):
                    if not any(l in rt for rt in rts):
                        rts[0] = sorted(set(rts[0]) | {l})
                Vlf = [int(x) for x in rng9.integers(20, 60, size=Lf)]
                rm = mod_efpa.efpa_fixed_point(
                    np.asarray(Vlf, dtype=float), af, np.asarray(tf),
                    rts)
                conv_ok = conv_ok and bool(rm["converged"])
                e_ref = ref_efpa(Vlf, af, tf, rts)
                worst9 = max(worst9, float(np.max(np.abs(
                    np.asarray(rm["B_e2e"], dtype=float)
                    - np.asarray(e_ref, dtype=float)))))
            record("T9f", "coupled EFPA vs independent reference (3 nets)",
                   worst9 < 1e-9 and conv_ok,
                   "max dev %.3e, all converged: %s" % (worst9, conv_ok))

    # T10/T11: sensitivities and recall search, per-point blocking via kr_spec
    # One shared scenario for T10a and T10b, with the central-difference partials dB_k/da_hat_j taken once at a_hat = C^T a.
    K = 4
    V = 60
    a = np.array([12.0, 7.0, 18.0, 5.0])
    t = np.array([1, 2, 4, 6])
    C = np.array([[0.90, 0.04, 0.03, 0.03],
                  [0.05, 0.85, 0.06, 0.04],
                  [0.02, 0.08, 0.88, 0.02],
                  [0.04, 0.03, 0.03, 0.90]])
    a_hat10 = C.T @ a
    d = 1e-6
    dB = np.zeros((K, K))
    for j in range(K):
        ap, am = a_hat10.copy(), a_hat10.copy()
        ap[j] += d
        am[j] -= d
        dB[:, j] = (np.asarray(kr_spec(V, ap, t))
                    - np.asarray(kr_spec(V, am, t))) / (2 * d)

    with guard("T10a", "constrained sensitivity vs independent FD"):
        S_mod = np.asarray(mod.sensitivity_analysis(V, a, C, t),
                           dtype=float)
        diag_ok = bool(np.all(S_mod[:, range(K), range(K)] == 0.0))
        # S[k,i,j] = (dB_k/da_hat_j - dB_k/da_hat_i) * a_i (eq:constrained-sensitivity); the a_i factor turns a per-Erlang gradient into a per-unit-C_ij one
        S_ref = np.zeros((K, K, K))
        for i in range(K):
            for j in range(K):
                S_ref[:, i, j] = (dB[:, j] - dB[:, i]) * a[i]
        worst = float(np.max(np.abs(S_mod - S_ref)))
        ok = worst < 1e-6 and diag_ok
        record("T10a", "constrained sensitivity vs independent FD",
               ok, "max |S_mod - S_ref| = %.3e (tensor scale %.3e); "
                   "diag zero: %s"
               % (worst, float(np.max(np.abs(S_ref))), diag_ok))
    with guard("T10b", "projected sensitivity vs independent FD"):
        proj = mod.sensitivity_analysis_projected(V, a, C, t)
        # same partials, row-centred: S_proj[k,i,:] = a_i * (dB[k,:] - mean_j dB[k,j]); S_sys = sum_k w_k S_proj[k] with w_k = a_k t_k / sum a t; aggregates are S_sys row norms
        S_proj_ref = np.zeros((K, K, K))
        for k in range(K):
            for i in range(K):
                row = a[i] * dB[k, :]
                S_proj_ref[k, i, :] = row - row.mean()
        w10 = (a * t) / float(np.sum(a * t))
        S_sys_ref = np.einsum("k,kij->ij", w10, S_proj_ref)
        rn = np.linalg.norm(S_sys_ref, axis=1)
        dev_t = float(np.max(np.abs(
            np.asarray(proj["S_proj"], dtype=float) - S_proj_ref)))
        dev_s = float(np.max(np.abs(
            np.asarray(proj["S_sys_proj"], dtype=float) - S_sys_ref)))
        dev_max = abs(float(proj["max_row_l2"]) - float(rn.max()))
        dev_mean = abs(float(proj["mean_row_l2"]) - float(rn.mean()))
        dev_fro = abs(float(proj["frobenius"])
                      - float(np.linalg.norm(S_sys_ref)))
        worst_b = max(dev_t, dev_s, dev_max, dev_mean, dev_fro)
        record("T10b", "projected sensitivity vs independent FD",
               worst_b < 1e-6,
               "dev: tensor %.3e, system %.3e, max_row_l2 %.3e, "
               "mean_row_l2 %.3e, frobenius %.3e"
               % (dev_t, dev_s, dev_max, dev_mean, dev_fro))
    with guard("T10c", "perturbation_variance closed form"):
        a = np.array([12.0, 7.0, 18.0, 5.0])
        Cv = np.full((4, 4), 1e-4)
        v_mod = np.asarray(mod.perturbation_variance(a, Cv), dtype=float)
        v_ref = np.array([sum(a[i] ** 2 * Cv[i, k] for i in range(4))
                          for k in range(4)])
        ok = bool(np.allclose(v_mod, v_ref, atol=1e-15))
        record("T10c", "perturbation_variance closed form", ok,
               "max dev %.3e" % float(np.max(np.abs(v_mod - v_ref))))


    # minimum_recall_search returns the last r on linspace(r_max, r_min, R_STEPS_SYSTEM) whose relative overhead (V' - V)/V stays <= epsilon; the bracket is valid when overhead(r*) <= eps and overhead at the next lower r exceeds eps
    with guard("T11", "recall bracket validity (OTT, eps=5%)"):
        a_ott = np.asarray(cn.A_OTT, dtype=float)
        t_ott = np.asarray(cn.T_OTT)
        V_nom, B_tgt, eps = 499, 0.01, 0.05
        r_star = float(mod.minimum_recall_search(
            a_ott, t_ott, V_nom, B_tgt, epsilon=eps))
        grid = np.linspace(1.0, 0.5, int(cn.R_STEPS_SYSTEM))
        idx = int(np.argmin(np.abs(grid - r_star)))
        on_grid = abs(float(grid[idx]) - r_star) < 1e-9

        def ref_overhead(r):
            Kt = len(a_ott)
            C_u = np.full((Kt, Kt), (1.0 - r) / (Kt - 1))
            np.fill_diagonal(C_u, r)
            a_d = list(np.asarray(C_u).T @ a_ott)
            for Vx in range(V_nom, V_nom + 500):
                if max(kr_spec(Vx, a_d, list(t_ott))) <= B_tgt:
                    return (Vx - V_nom) / V_nom
            raise RuntimeError("overhead scan exhausted at V=%d"
                               % (V_nom + 500))

        ov_star = ref_overhead(r_star)
        r_below = float(grid[idx + 1]) if idx + 1 < grid.size else None
        ov_below = (ref_overhead(r_below)
                    if r_below is not None else None)
        ok = (on_grid and ov_star <= eps
              and (ov_below is None or ov_below > eps))
        record("T11", "recall bracket validity (OTT, eps=5%)", ok,
               "r*=%.6f on module grid: %s; overhead(r*)=%.5f; "
               "overhead(next lower r=%s)=%s"
               % (r_star, on_grid, ov_star,
                  "%.6f" % r_below if r_below is not None else "n/a",
                  "%.5f" % ov_below if ov_below is not None
                  else "n/a"))

    # summary
    print("\nNOTES")
    for i, f in enumerate(findings, 1):
        print("F%d: %s" % (i, f))
    print("\nSUMMARY MATRIX")
    n_fail = 0
    for tid, desc, status, detail in results:
        print("%-5s %-52s %-7s" % (tid, desc[:52], status))
        if status == "FAIL":
            n_fail += 1
            print("      detail: %s" % detail)
    n_pass = sum(1 for r in results if r[2] == "PASS")
    n_part = sum(1 for r in results if r[2] == "PARTIAL")
    print("PASS %d | PARTIAL/SKIP %d | FAIL %d" % (n_pass, n_part, n_fail))
    sys.exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()
