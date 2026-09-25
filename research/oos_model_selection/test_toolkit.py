"""Sanity checks with known answers. Run: python test_toolkit.py (~2 min)."""
import numpy as np
from scipy import stats
from scipy.stats import norm

import oos_toolkit as tk

rng = np.random.default_rng(123)


def check(name, cond, info=""):
    info = str(info)
    print(("PASS " if cond else "FAIL ") + name + (f"  [{info}]" if info else ""))
    return bool(cond)


ok = True

# --------------------------------------------------------------------------- #
# returns / aggregation
# --------------------------------------------------------------------------- #
p = np.array([0.60, 0.40, 0.50, 0.15])
oa = np.array([2.0, 2.0, 1.9, 6.0])
ob = np.array([1.8, 1.8, 1.9, 1.15])
y = np.array([1.0, 1.0, 0.0, 1.0])
r, b = tk.flat_bet_returns(p, oa, ob, y)
# g0: EV_A=.2 -> back A, won: +1 | g1: EV_B=.08 -> back B, lost: -1 | g2: no bet
# g3: EV_A = .15*6-1 = -.10, EV_B = .85*1.15-1 = -.0225 -> no bet
ok &= check("flat_bet_returns", np.allclose(r, [1.0, -1.0, 0.0, 0.0]) and b.tolist() == [True, True, False, False], r)
r_cap, b_cap = tk.flat_bet_returns(np.array([0.30]), np.array([4.0]), np.array([1.3]), np.array([1.0]),
                                   max_odds=3.5)
ok &= check("max_odds blocks a +20% EV bet at odds 4.0", not b_cap[0] and r_cap[0] == 0.0)

agg = tk.aggregate_by_game(np.array([1.0, 2.0, 3.0, 4.0]), np.array([7, 3, 7, 3]))
ok &= check("aggregate_by_game", np.allclose(agg, [4.0, 6.0]), agg)

# --------------------------------------------------------------------------- #
# Sharpe family
# --------------------------------------------------------------------------- #
for N in (2, 10, 50, 200):
    mc = rng.standard_normal((40000, N)).max(1).mean()
    f = tk.expected_max_sr(N, 1.0)
    ok &= check(f"expected_max_sr(N={N}) vs Monte Carlo", abs(f - mc) < 0.06, f"formula={f:.3f} MC={mc:.3f}")
ok &= check("expected_max_sr floored at 0 for N=1.05", tk.expected_max_sr(1.05, 1.0) == 0.0)

x = rng.normal(0.05, 1, 1000)
sr = x.mean() / x.std(ddof=1)
hand = norm.cdf(sr * np.sqrt(999) / np.sqrt(1 - stats.skew(x) * sr
                + (stats.kurtosis(x, fisher=False) - 1) / 4 * sr ** 2))
ok &= check("psr formula", abs(tk.psr(x) - hand) < 1e-12)
fp = np.mean([tk.psr(rng.normal(0, 1, 500)) > 0.95 for _ in range(4000)])
ok &= check("psr size under H0", 0.04 < fp < 0.06, f"{fp:.3f}")
try:
    tk.psr(np.zeros(100))
    ok &= check("psr refuses a series with no variance", False)
except ValueError:
    ok &= check("psr refuses a series with no variance", True)
X = rng.normal(0.02, 1, (800, 1))
ok &= check("dsr with N=1 equals psr", abs(tk.deflated_sharpe(X[:, 0], np.array([0.1, 0.2]), 1)[0]
                                            - tk.psr(X[:, 0])) < 1e-12)

# implied independent trials (DSR paper, Eq. 9)
base = rng.normal(0, 1, (3000, 1))
n_same = tk.implied_independent_trials(np.repeat(base, 10, 1) + 1e-9 * rng.normal(size=(3000, 10)))
n_ind = tk.implied_independent_trials(rng.normal(0, 1, (3000, 10)))
n_half = tk.implied_independent_trials(np.sqrt(0.5) * base + np.sqrt(0.5) * rng.normal(0, 1, (3000, 10)))
n_one = tk.implied_independent_trials(np.column_stack([base[:, 0], np.zeros(3000)]))
ok &= check("implied independent trials", n_same < 1.01 and 9.7 < n_ind < 10.3 and 5.1 < n_half < 5.9
            and n_one == 1.0, f"same={n_same:.2f} indep={n_ind:.2f} rho=.5 -> {n_half:.2f} one col -> {n_one}")

# --------------------------------------------------------------------------- #
# SPA / RC
# --------------------------------------------------------------------------- #
def literal_spa(D, W):
    """Hansen (2005) written step by step with loops, sharing the bootstrap weights."""
    n, K = D.shape
    thr = np.sqrt(2 * np.log(np.log(n)))
    dbar = D.mean(0)
    dstar = np.array([[W[b_] @ D[:, k] / n for k in range(K)] for b_ in range(W.shape[0])])
    omega = np.array([np.sqrt(n * np.mean((dstar[:, k] - dbar[k]) ** 2)) for k in range(K)])
    t_obs = max(0.0, max(np.sqrt(n) * dbar[k] / omega[k] for k in range(K)))
    g = [dbar[k] if np.sqrt(n) * dbar[k] / omega[k] >= -thr else 0.0 for k in range(K)]
    t_star = [max(0.0, max(np.sqrt(n) * (dstar[b_, k] - g[k]) / omega[k] for k in range(K)))
              for b_ in range(W.shape[0])]
    return float(np.mean(np.array(t_star) >= t_obs))


# heteroskedastic candidates, some clearly bad (they must not drive the null)
sds = np.linspace(0.5, 3.0, 20)
means = np.r_[np.full(5, -0.3), np.zeros(12), np.full(3, 0.05)]
D = means + sds * rng.normal(size=(1500, 20))
for mb in (1.0, 8.0):
    seed = 7
    W = tk.bootstrap_weights(1500, 400, np.random.default_rng(seed), mb)
    mine = tk.spa_vs_no_bet(D, reps=400, seed=seed, mean_block=mb, min_bets=0)["p_spa"]
    lit = literal_spa(D, W)
    ok &= check(f"spa equals literal Hansen (mean_block={mb})", abs(mine - lit) < 1e-12, f"{mine:.4f} vs {lit:.4f}")

from arch.bootstrap import SPA  # arch 8.0: raw means, i.e. the (non-studentized) RC

diffs = []
for i in range(10):
    Xa = D[:, rng.permutation(20)] + 0.02 * (i % 5)
    mine = tk.spa_vs_no_bet(Xa, reps=2000, seed=i, min_bets=0)["p_rc"]
    a = SPA(np.zeros(1500), -Xa, block_size=1, reps=2000, studentize=False, seed=i)
    a.compute()
    diffs.append(abs(mine - a.pvalues["upper"]))
ok &= check("rc matches arch 'upper' (heteroskedastic data)", np.mean(diffs) < 0.03 and max(diffs) < 0.08,
            f"mean|diff|={np.mean(diffs):.3f} max={max(diffs):.3f}")

rej1 = np.mean([tk.spa_vs_no_bet(rng.normal(0, 1, (400, 1)), reps=300, seed=i)["p_spa"] < 0.05
                for i in range(600)])
ok &= check("spa size K=1", 0.03 < rej1 < 0.075, f"{rej1:.3f}")
res = []
for i in range(400):
    Z = 0.8 * rng.normal(0, 1, (400, 1)) + 0.6 * rng.normal(0, 1, (400, 50))
    res.append(tk.spa_vs_no_bet(Z, reps=300, seed=10_000 + i))
rs = np.mean([q["p_spa"] < 0.05 for q in res])
rr = np.mean([q["p_rc"] < 0.05 for q in res])
ok &= check("spa/rc size, 50 correlated null models", 0.02 < rs < 0.08 and 0.02 < rr < 0.08,
            f"spa={rs:.3f} rc={rr:.3f}")
Xs = rng.normal(0, 1, (400, 50))
Xs[:, 0] += 0.4   # t ~ 8
ok &= check("spa detects strong model", tk.spa_vs_no_bet(Xs, reps=500, seed=1)["p_spa"] < 0.01)
sparse = np.where(rng.random((2000, 3)) < np.array([0.01, 0.2, 0.3]), rng.normal(0, 1, (2000, 3)), 0.0)
ok &= check("min_bets drops the rare bettor", tk.spa_vs_no_bet(sparse, reps=100, seed=1)["n_tested"] == 2)

# --------------------------------------------------------------------------- #
# MCS
# --------------------------------------------------------------------------- #
L = rng.normal(0, 1, (500, 30))
L[:, 5] -= 0.5
m_r = tk.model_confidence_set(L, alpha=0.10, reps=500, method="R", seed=2)
m_x = tk.model_confidence_set(L, alpha=0.10, reps=500, method="max", seed=2)
ok &= check("mcs R isolates the best", m_r[5] and m_r.sum() <= 3, f"size={m_r.sum()}")
print(f"INFO mcs 'max' in the same case keeps {m_x.sum()} of 30 (known low power of T_max here)")
m2 = tk.model_confidence_set(rng.normal(0, 1, (500, 30)), alpha=0.10, reps=500, method="R", seed=3)
ok &= check("mcs keeps equal models", m2.sum() >= 15, f"size={m2.sum()}")

# --------------------------------------------------------------------------- #
# PBO / CSCV
# --------------------------------------------------------------------------- #
pbo_eq = np.mean([tk.pbo_cscv(rng.normal(0.02, 1, (800, 50)), 8)["pbo"] for _ in range(20)])
ok &= check("pbo equal skill ~0.5", 0.35 < pbo_eq < 0.65, f"{pbo_eq:.3f}")
Xd = rng.normal(0, 1, (800, 50))
Xd[:, 0] += 0.3
out = tk.pbo_cscv(Xd, 8, metric="mean")
ok &= check("pbo dominant model ~0; degradation slope -1 by construction",
            out["pbo"] < 0.05 and abs(out["degradation_slope"] + 1) < 1e-6, f"slope={out['degradation_slope']:.6f}")

# --------------------------------------------------------------------------- #
# clustering, sample size
# --------------------------------------------------------------------------- #
v = rng.normal(0, 1, 2000)
d = np.repeat(v, 2)
_, se_cl, _ = tk.clustered_mean_test(d, np.repeat(np.arange(2000), 2))
ratio = se_cl / (d.std(ddof=1) / np.sqrt(d.size))
ok &= check("clustered SE on duplicated games", abs(ratio - np.sqrt(2)) < 0.05, f"{ratio:.3f}")
try:
    tk.clustered_mean_test(np.ones(5), np.zeros(5))
    ok &= check("clustered SE refuses a single cluster", False)
except ValueError:
    ok &= check("clustered SE refuses a single cluster", True)

n_b = tk.bets_needed(0.03, 1.9, 2.0)
p_win = 1.03 / 1.9
wins = rng.random((3000, n_b)) < p_win
prof = np.where(wins, 0.9, -1.0)
tstat = prof.mean(1) / (prof.std(1, ddof=1) / np.sqrt(n_b))
ok &= check("bets_needed gives t ~ 2 on simulated bets", abs(tstat.mean() - 2.0) < 0.1, f"n={n_b}, mean t={tstat.mean():.3f}")

# --------------------------------------------------------------------------- #
# scoring against the market
# --------------------------------------------------------------------------- #
def market_world(n, info, under=1.0):
    z = rng.normal(0, 1.2, n)                  # what the price knows (logit scale)
    e = rng.normal(0, 0.5, n)                  # what only the model may know
    truth = 1 / (1 + np.exp(-(under * z + e)))
    yy = (rng.random(n) < truth).astype(float)
    q = 1 / (1 + np.exp(-z))
    pm = 1 / (1 + np.exp(-(under * z + info * e + rng.normal(0, 0.3, n))))
    return yy, q, pm


size = np.mean([tk.encompassing_test(*market_world(1500, 0.0), np.arange(1500))["p_one_sided"] < 0.05
                for _ in range(300)])
ok &= check("encompassing c: size ~5% when the model has no extra info", 0.02 < size < 0.09, f"{size:.3f}")
yy, q, pm = market_world(3000, 1.0)
ok &= check("encompassing c detects extra info", tk.encompassing_test(yy, q, pm, np.arange(3000))["p_one_sided"] < 0.01)
ok &= check("logloss_vs_market: better model has negative delta",
            tk.logloss_vs_market(yy, pm, q, np.arange(3000))["delta_logloss"] < 0)
# the price is under-confident (truth = 1.4 x its logit); the model is only that
# recalibration plus noise (exactly 1.4 x logit would be collinear with the price)
z = rng.normal(0, 1.2, 20000)
truth = 1 / (1 + np.exp(-1.4 * z))
yy = (rng.random(20000) < truth).astype(float)
recal = 1 / (1 + np.exp(-(1.4 * z + rng.normal(0, 0.3, 20000))))
enc = tk.encompassing_test(yy, 1 / (1 + np.exp(-z)), recal, np.arange(20000))
ok &= check("recalibration edge shows in the joint test, not in c", enc["p_joint"] < 1e-6 and enc["p_one_sided"] > 0.01,
            f"b={enc['b_market']:.2f} c={enc['c_model']:.2f} p_c={enc['p_one_sided']:.2f} p_joint={enc['p_joint']:.1e}")

# --------------------------------------------------------------------------- #
# family information gate, Benter blend
# --------------------------------------------------------------------------- #
def info_world(n, K, info_idx=(), bias=1.0, tilt=0.0):
    """bias: truth = bias * logit(q). tilt: models lean on logit(q) with no private info."""
    lq = rng.normal(0, 1.2, n)
    e = rng.normal(0, 0.5, n)
    yy = (rng.random(n) < 1 / (1 + np.exp(-(bias * lq + e)))).astype(float)
    d = 0.3 * rng.normal(size=(n, 1)) + 0.1 * rng.normal(size=(n, K)) + tilt * lq[:, None]
    for i in info_idx:
        d[:, i] += 0.5 * e
    return yy, 1 / (1 + np.exp(-lq)), 1 / (1 + np.exp(-(lq[:, None] + d)))


rej = np.mean([tk.information_gate(*info_world(2000, 50), reps=500, seed=s)["p_family"] < 0.05
               for s in range(300)])
ok &= check("information gate size, 50 models without information", 0.02 < rej < 0.08, f"{rej:.3f}")
rej_raw = np.mean([tk.information_gate(*info_world(2000, 50, bias=1.3, tilt=0.3), reps=500, seed=s,
                                       adjust_price_bias=False)["p_family"] < 0.05 for s in range(100)])
rej_orth = np.mean([tk.information_gate(*info_world(2000, 50, bias=1.3, tilt=0.3), reps=500, seed=s)
                    ["p_family"] < 0.05 for s in range(200)])
ok &= check("biased price: orthogonalised gate keeps its size, raw scores do not",
            rej_orth < 0.09 and rej_raw > 0.5, f"orthogonal={rej_orth:.3f} raw={rej_raw:.3f}")
g_out = tk.information_gate(*info_world(3000, 50, info_idx=(17,)), reps=1000, seed=1)
ok &= check("information gate finds the informative model", g_out["p_family"] < 0.01 and g_out["best"] == 17
            and g_out["significant"].sum() <= 3, f"p={g_out['p_family']:.3f} best={g_out['best']} n_sig={g_out['significant'].sum()}")

lq = rng.normal(0, 1.2, 50000)
e = rng.normal(0, 0.5, 50000)
yy = (rng.random(50000) < 1 / (1 + np.exp(-(lq + e)))).astype(float)
pm = 1 / (1 + np.exp(-(lq + e + rng.normal(0, 0.5, 50000))))      # informative but noisy
coef = tk.fit_blend(yy, 1 / (1 + np.exp(-lq)), pm)
ok &= check("blend recovers a~0, b~1, 0<c<1 (shrinks a noisy model)", abs(coef[0]) < 0.05 and abs(coef[1] - 1) < 0.05
            and 0.2 < coef[2] < 0.8, np.round(coef, 3))

# --------------------------------------------------------------------------- #
# live monitoring checked after EVERY bet
# --------------------------------------------------------------------------- #
n_bets, paths = 3000, 2000
odds_l = rng.uniform(1.5, 3.5, (paths, n_bets))
b_l = 1 / odds_l
claim = np.minimum(b_l * 1.05, 0.99)                    # claims a +5% edge on every bet
won0 = rng.random((paths, n_bets)) < b_l               # H0: break-even (no edge)
won1 = rng.random((paths, n_bets)) < claim             # H1: the claim is right
dec0 = [tk.sprt_break_even(won0[i], claim[i], odds_l[i])[1] for i in range(paths)]
dec1 = [tk.sprt_break_even(won1[i], claim[i], odds_l[i])[1] for i in range(paths)]
prof0 = np.where(won0, odds_l - 1, -1.0)
k = np.arange(1, n_bets + 1)
run_t = np.cumsum(prof0, 1) / k / (np.sqrt(np.maximum(np.cumsum(prof0 ** 2, 1) / k - (np.cumsum(prof0, 1) / k) ** 2, 1e-12)) / np.sqrt(k))
naive_fa = np.mean((run_t[:, 29:] > 1.645).any(1))
ok &= check("sprt: false scale-up <= 5% when checked after every bet", np.mean(np.array(dec0) == "scale") <= 0.055,
            f"false scale-up={np.mean(np.array(dec0) == 'scale'):.3f}; naive t>1.645 checked after every bet={naive_fa:.3f}")
ok &= check("sprt: false kill <= 5% when the claim is right", np.mean(np.array(dec1) == "kill") <= 0.055,
            f"false kill={np.mean(np.array(dec1) == 'kill'):.3f}; scaled={np.mean(np.array(dec1) == 'scale'):.3f}")

# --------------------------------------------------------------------------- #
# markout: unbiased when the later market has caught up, blind otherwise
# --------------------------------------------------------------------------- #
import simulate as S

for case, VB, VBp, want in (("catch-up", 0.02, 0.0, "zero"), ("persistent", 0.0, 0.02, "positive")):
    g = S.draw_games(400_000, rng, VB=VB, VBp=VBp)
    p15 = norm.cdf((g["A"] + g["B"] + g["Bp"]) / np.sqrt(1 - S.PARAMS["VA"] - VB - VBp))
    r15, bet = tk.flat_bet_returns(p15, g["oa15"], g["ob15"], g["y"], 0.0, 5.0)
    back_a = bet & (p15 * g["oa15"] > (1 - p15) * g["ob15"])
    odds = np.where(back_a, g["oa15"], g["ob15"])[bet]
    ql = np.where(back_a, g["q20"], 1 - g["q20"])[bet]
    ys = np.where(back_a, g["y"], 1 - g["y"])[bet]
    resid = odds * (ys - ql)
    zres = resid.mean() / (resid.std() / np.sqrt(resid.size))
    good = abs(zres) < 3 if want == "zero" else zres > 5
    ok &= check(f"markout residual is {want} ({case} edge)", good, f"z={zres:.1f}, profit/bet={r15[bet].mean():+.3f}")

# --------------------------------------------------------------------------- #
# stationary bootstrap recovers the long-run variance of an AR(1)
# --------------------------------------------------------------------------- #
phi, n = 0.5, 4000
e = rng.normal(size=n)
xa = np.empty(n)
xa[0] = e[0]
for t in range(1, n):
    xa[t] = phi * xa[t - 1] + e[t]
lrv = 1 / (1 - phi) ** 2
for mb, lo, hi in ((1.0, 0.0, 0.45), (25.0, 0.75, 1.25)):
    W = tk.bootstrap_weights(n, 400, np.random.default_rng(3), mb)
    ratio = n * ((W @ xa) / n).var() / lrv
    ok &= check(f"bootstrap variance / long-run variance (mean_block={mb})", lo < ratio < hi, f"{ratio:.2f}")

print("\nALL PASS" if ok else "\nSOME CHECKS FAILED")
