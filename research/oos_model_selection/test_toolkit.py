"""Sanity checks with known answers. Run: python test_toolkit.py"""
import numpy as np
from scipy import stats

import oos_toolkit as tk

rng = np.random.default_rng(123)


def check(name, cond, info=""):
    info = str(info)
    print(("PASS " if cond else "FAIL ") + name + (f"  [{info}]" if info else ""))
    return cond


ok = True

# flat_bet_returns: hand-computed example
p = np.array([0.60, 0.40, 0.50])
oa = np.array([2.0, 2.0, 1.9])
ob = np.array([1.8, 1.8, 1.9])
y = np.array([1.0, 1.0, 0.0])
r, b = tk.flat_bet_returns(p, oa, ob, y)
# game0: EV_A = .2, EV_B = -.28 -> back A, won -> +1.0
# game1: EV_A = -.2, EV_B = .08 -> back B, lost -> -1.0
# game2: EV_A = EV_B = -.05 -> no bet
ok &= check("flat_bet_returns", np.allclose(r, [1.0, -1.0, 0.0]) and b.tolist() == [True, True, False], r)

# aggregate_by_game keeps first-appearance order
agg = tk.aggregate_by_game(np.array([1.0, 2.0, 3.0, 4.0]), np.array([7, 3, 7, 3]))
ok &= check("aggregate_by_game", np.allclose(agg, [4.0, 6.0]), agg)

# expected max of N standard normals vs Monte Carlo
mc = rng.standard_normal((20000, 200)).max(1).mean()
ok &= check("expected_max_sr(200, 1)", abs(tk.expected_max_sr(200, 1.0) - mc) < 0.05,
            f"formula={tk.expected_max_sr(200, 1.0):.3f} MC={mc:.3f}")

# PSR: matches hand formula; ~uniform under H0 (false-positive rate ~5%)
x = rng.normal(0.05, 1, 1000)
sr = x.mean() / x.std(ddof=1)
hand = stats.norm.cdf(sr * np.sqrt(999) / np.sqrt(1 - stats.skew(x) * sr
                      + (stats.kurtosis(x, fisher=False) - 1) / 4 * sr ** 2))
ok &= check("psr formula", abs(tk.psr(x) - hand) < 1e-12)
fp = np.mean([tk.psr(rng.normal(0, 1, 500)) > 0.95 for _ in range(4000)])
ok &= check("psr size under H0", 0.04 < fp < 0.06, f"{fp:.3f}")

# SPA / RC: single null model and 50 correlated null models -> ~5% rejections
rej1 = np.mean([tk.spa_vs_no_bet(rng.normal(0, 1, (400, 1)), reps=300, seed=i)["p_spa"] < 0.05
                for i in range(600)])
ok &= check("spa size K=1", 0.03 < rej1 < 0.075, f"{rej1:.3f}")
res = []
for i in range(400):
    Z = 0.8 * rng.normal(0, 1, (400, 1)) + 0.6 * rng.normal(0, 1, (400, 50))
    res.append(tk.spa_vs_no_bet(Z, reps=300, seed=10_000 + i))
rs = np.mean([r["p_spa"] < 0.05 for r in res])
rr = np.mean([r["p_rc"] < 0.05 for r in res])
ok &= check("spa/rc size, 50 correlated null models", rs < 0.08 and rr < 0.08, f"spa={rs:.3f} rc={rr:.3f}")
# power: one model with a large real mean shift (t ~ 8) among 49 nulls.
# (With a +0.25 shift an unlucky draw gave t = 3.16 and SPA p = 0.026, which is
#  correct: 3.16 is barely above the 95% quantile of the max of 50 normals.)
X = rng.normal(0, 1, (400, 50))
X[:, 0] += 0.4
ok &= check("spa detects strong model", tk.spa_vs_no_bet(X, reps=500, seed=1)["p_spa"] < 0.01)

# SPA / RC agree with the reference implementation (arch) on the same data.
# Different bootstrap draws -> Monte Carlo noise of ~0.01-0.02 per p-value.
from arch.bootstrap import SPA

diffs_spa, diffs_rc = [], []
for i in range(15):
    X = 0.7 * rng.normal(0, 1, (1000, 1)) + 0.7 * rng.normal(0, 1, (1000, 40))
    X[:, :3] += 0.02 * (i % 5)                          # from no edge to a clear edge
    mine = tk.spa_vs_no_bet(X, reps=2000, seed=i)
    a = SPA(np.zeros(1000), -X, block_size=1, reps=2000, seed=i)
    a.compute()
    b = SPA(np.zeros(1000), -X, block_size=1, reps=2000, studentize=False, seed=i)
    b.compute()
    diffs_spa.append(abs(mine["p_spa"] - a.pvalues["consistent"]))
    diffs_rc.append(abs(mine["p_rc"] - b.pvalues["upper"]))
ok &= check("spa matches arch (consistent)", np.mean(diffs_spa) < 0.03 and max(diffs_spa) < 0.08,
            f"mean|diff|={np.mean(diffs_spa):.3f} max={max(diffs_spa):.3f}")
ok &= check("rc matches arch (non-studentized, upper)", np.mean(diffs_rc) < 0.03 and max(diffs_rc) < 0.08,
            f"mean|diff|={np.mean(diffs_rc):.3f} max={max(diffs_rc):.3f}")

# MCS (arch): one model clearly best. Range statistic isolates it; T_max cannot.
L = rng.normal(0, 1, (500, 30))
L[:, 5] -= 0.5
m_r = tk.model_confidence_set(L, alpha=0.10, reps=500, method="R", seed=2)
m_x = tk.model_confidence_set(L, alpha=0.10, reps=500, method="max", seed=2)
ok &= check("mcs R isolates the best", m_r[5] and m_r.sum() <= 3, f"size={m_r.sum()}")
print(f"INFO mcs 'max' in the same case keeps {m_x.sum()} of 30 (known low power of T_max here)")
Leq = rng.normal(0, 1, (500, 30))
m2 = tk.model_confidence_set(Leq, alpha=0.10, reps=500, method="R", seed=3)
ok &= check("mcs keeps equal models", m2.sum() >= 15, f"size={m2.sum()}")

# PBO: equal-skill models -> ~0.5; one dominant model -> ~0
pbo_eq = np.mean([tk.pbo_cscv(rng.normal(0.02, 1, (800, 50)), 8)["pbo"] for _ in range(20)])
ok &= check("pbo equal skill ~0.5", 0.35 < pbo_eq < 0.65, f"{pbo_eq:.3f}")
Xd = rng.normal(0, 1, (800, 50))
Xd[:, 0] += 0.3
ok &= check("pbo dominant model ~0", tk.pbo_cscv(Xd, 8)["pbo"] < 0.05)

# clustered SE: every game duplicated -> SE ~ sqrt(2) x naive
v = rng.normal(0, 1, 2000)
d = np.repeat(v, 2)
_, se_cl, _ = tk.clustered_mean_test(d, np.repeat(np.arange(2000), 2))
ratio = se_cl / (d.std(ddof=1) / np.sqrt(d.size))
ok &= check("clustered SE on duplicated games", abs(ratio - np.sqrt(2)) < 0.05, f"{ratio:.3f}")

# implied independent trials (DSR paper, Eq. 9): identical -> 1, independent -> ~M,
# average correlation 0.5 with M=10 -> 0.5 + 0.5 * 10 = 5.5
base = rng.normal(0, 1, (3000, 1))
n_same = tk.implied_independent_trials(np.repeat(base, 10, 1) + 1e-9 * rng.normal(size=(3000, 10)))
n_ind = tk.implied_independent_trials(rng.normal(0, 1, (3000, 10)))
n_half = tk.implied_independent_trials(np.sqrt(0.5) * base + np.sqrt(0.5) * rng.normal(0, 1, (3000, 10)))
ok &= check("implied independent trials", n_same < 1.01 and 9.7 < n_ind < 10.3 and 5.1 < n_half < 5.9,
            f"same={n_same:.2f} indep={n_ind:.2f} rho=.5 -> {n_half:.2f}")

# bets_needed: hand value at odds 1.9, edge 3%, z = 2
p_ = 1.03 / 1.9
hand_n = int(np.ceil((2 * 1.9 * np.sqrt(p_ * (1 - p_)) / 0.03) ** 2))
ok &= check("bets_needed", tk.bets_needed(0.03, 1.9, 2.0) == hand_n, f"{hand_n}")

print("\nALL PASS" if ok else "\nSOME CHECKS FAILED")
