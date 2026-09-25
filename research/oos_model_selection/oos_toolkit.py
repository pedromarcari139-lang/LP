"""
Reference implementations of the model-selection statistics discussed in README.md.

Conventions (read before using on real data)
--------------------------------------------
* One row = one GAME (gameid unique), rows sorted by game start time.
  If a model bets at 15:00 AND 20:00 of the same game, SUM both bets into the
  game row first (see `aggregate_by_game`). Then no fold / block / bootstrap
  draw can ever split a game. Maps of one Bo3/Bo5 series are also dependent:
  cluster by series id where a function takes `groups`.
* R[g, k] = profit of model k in game g, in units staked (0 if no bet).
* Every input must already be OUT-OF-SAMPLE (walk-forward) predictions/returns.
  Nothing in this file can detect a model that was fitted on the rows it scores.
"""
from itertools import combinations

import numpy as np
from scipy import stats

EULER_GAMMA = 0.5772156649015329


# --------------------------------------------------------------------------- #
# Odds / returns
# --------------------------------------------------------------------------- #
def devig_multiplicative(odds_a, odds_b):
    """Fair probability of side A with the proportional (multiplicative) method.
    With two outcomes the additive and Shin methods coincide and differ from
    this one by up to ~2 pp on lopsided prices (README §8)."""
    ia, ib = 1.0 / np.asarray(odds_a, float), 1.0 / np.asarray(odds_b, float)
    return ia / (ia + ib)


def flat_bet_returns(p_model, odds_a, odds_b, y, min_ev=0.0, max_odds=np.inf):
    """Unit stake on the side whose EV (at the offered odds) exceeds `min_ev`,
    never on a side priced above `max_odds`.

    p_model: (n,) or (n, K) model probability that side A wins.
    odds_a, odds_b, y: (n,) or broadcastable; y = 1 if side A won.
    Returns (profit, bet_mask). With any overround at most one side has EV > 0.
    """
    ev_a = p_model * odds_a - 1.0
    ev_b = (1.0 - p_model) * odds_b - 1.0
    bet_a = (ev_a > min_ev) & (ev_a >= ev_b) & (odds_a <= max_odds)
    bet_b = (ev_b > min_ev) & (ev_b > ev_a) & (odds_b <= max_odds)
    profit = np.where(bet_a, y * odds_a - 1.0, 0.0)
    profit = np.where(bet_b, (1 - y) * odds_b - 1.0, profit)
    return profit, bet_a | bet_b


def aggregate_by_game(values, gameid):
    """Sum row-level values (n,) or (n, K) into one row per game, keeping the
    order of first appearance (so time order is preserved if rows were sorted)."""
    gameid = np.asarray(gameid)
    _, first, inv = np.unique(gameid, return_index=True, return_inverse=True)
    order = np.argsort(first)                 # games in order of first appearance
    rank = np.empty_like(order)
    rank[order] = np.arange(order.size)
    values = np.asarray(values, float)
    out = np.zeros((order.size,) + values.shape[1:])
    np.add.at(out, rank[inv], values)
    return out


# --------------------------------------------------------------------------- #
# Sharpe-ratio family (Bailey & López de Prado)
# --------------------------------------------------------------------------- #
def sharpe(x, axis=0):
    """Per-period (per-game) Sharpe ratio; 0 when the series has no variance
    (e.g. a model that never bets)."""
    x = np.asarray(x, float)
    m, s = x.mean(axis), x.std(axis, ddof=1)
    return np.divide(m, s, out=np.zeros_like(m, dtype=float), where=s > 0)


def psr(x, sr_benchmark=0.0):
    """Probabilistic Sharpe Ratio: P(true SR > sr_benchmark) given one track
    record x (per-game returns). Uses sample skewness and RAW kurtosis."""
    x = np.asarray(x, float)
    if x.std() == 0:
        raise ValueError("track record has no variance (a model that never bets?)")
    T = x.size
    sr = float(sharpe(x))
    g3 = stats.skew(x)
    g4 = stats.kurtosis(x, fisher=False)
    denom = np.sqrt(max(1.0 - g3 * sr + (g4 - 1.0) / 4.0 * sr ** 2, 1e-12))
    return float(stats.norm.cdf((sr - sr_benchmark) * np.sqrt(T - 1) / denom))


def expected_max_sr(n_trials, var_sr):
    """Expected maximum of n_trials SR estimates whose true SR is 0
    (approximation used by the Deflated Sharpe Ratio). The approximation goes
    negative for n_trials below ~1.3, so it is floored at 0."""
    if n_trials <= 1:
        return 0.0
    z1 = stats.norm.ppf(1.0 - 1.0 / n_trials)
    z2 = stats.norm.ppf(1.0 - 1.0 / (n_trials * np.e))
    return float(max(0.0, np.sqrt(var_sr) * ((1 - EULER_GAMMA) * z1 + EULER_GAMMA * z2)))


def deflated_sharpe(x_selected, sr_all_trials, n_trials=None):
    """DSR = PSR against the expected max SR of all trials. Returns (dsr, sr0).

    Pass n_trials = M, the number of ALL trials ever run (default: the number
    of SRs given), with the cross-sectional variance of their SRs (always used).
    That variance already shrinks when trials are correlated, so shrinking N
    too (e.g. with `implied_independent_trials`) counts the correlation twice:
    200 trials, correlation 0.9, true SR 0 -> 7.0% false positives at a nominal
    5% with N_hat vs 4.5% with N = M (EXP-10 in simulate.py)."""
    sr_all = np.asarray(sr_all_trials, float)
    n = sr_all.size if n_trials is None else n_trials
    sr0 = expected_max_sr(n, sr_all.var(ddof=1))
    return psr(x_selected, sr0), sr0


def implied_independent_trials(R):
    """Bailey & López de Prado (2014), Appendix A.3, Eq. 9:
    N_hat = rho + (1 - rho) * M, rho = average off-diagonal correlation of the
    M trials' per-game returns. Reported for reference only: do NOT feed it to
    `deflated_sharpe` together with the SR variance of the same trials."""
    R = np.asarray(R, float)
    R = R[:, R.std(0) > 0]
    M = R.shape[1]
    if M < 2:
        return 1.0
    C = np.corrcoef(R, rowvar=False)
    rho = (C.sum() - M) / (M * (M - 1))
    return float(rho + (1.0 - rho) * M)


# --------------------------------------------------------------------------- #
# CSCV / Probability of Backtest Overfitting
# --------------------------------------------------------------------------- #
def pbo_cscv(R, n_blocks=16, metric="sharpe"):
    """Bailey, Borwein, López de Prado & Zhu. R: (T games x N models), time order.

    Rows are cut into `n_blocks` contiguous blocks (a game is one row, so a
    game is never split). For every C(S, S/2) choice of IS blocks, the IS-best
    model is ranked inside the complementary OOS blocks.

    CAUTION
    * Half of the splits select on games that come AFTER the test games. CSCV
      measures how stable the SELECTION is, not whether the edge survives the
      passage of time; pair it with a walk-forward holdout.
    * PBO ~ 0.5 whenever candidates have equal skill, with or without an edge.
      Read it next to prob_oos_loss.
    * degradation_slope: IS and OOS are complementary halves, so for a model
      selected in every split OOS = 2 * full - IS and the slope is -1 by
      construction. It says little about overfitting.
    """
    R = np.asarray(R, float)
    T, N = R.shape
    if n_blocks % 2:
        raise ValueError("n_blocks must be even")
    edges = np.linspace(0, T, n_blocks + 1).astype(int)
    bounds = list(zip(edges[:-1], edges[1:]))
    s1 = np.array([R[a:b].sum(0) for a, b in bounds])          # (S, N)
    s2 = np.array([(R[a:b] ** 2).sum(0) for a, b in bounds])   # (S, N)
    cnt = np.diff(edges).astype(float)                          # (S,)

    combos = np.array(list(combinations(range(n_blocks), n_blocks // 2)))
    ind = np.zeros((len(combos), n_blocks))
    ind[np.arange(len(combos))[:, None], combos] = 1.0

    def perf(i):
        n = (i @ cnt)[:, None]
        m1 = (i @ s1) / n
        if metric == "mean":
            return m1
        var = ((i @ s2) / n - m1 ** 2) * n / (n - 1)
        sd = np.sqrt(np.maximum(var, 0.0))
        return np.divide(m1, sd, out=np.zeros_like(m1), where=sd > 1e-12)

    is_p, oos_p = perf(ind), perf(1.0 - ind)
    rows = np.arange(len(combos))
    best = is_p.argmax(1)
    oos_best = oos_p[rows, best]
    rank = stats.rankdata(oos_p, axis=1)[rows, best]    # 1 = worst ... N = best
    w = rank / (N + 1.0)
    lam = np.log(w / (1.0 - w))
    slope, intercept = np.polyfit(is_p[rows, best], oos_best, 1)
    return {
        "pbo": float((lam <= 0).mean()),
        "prob_oos_loss": float((oos_best < 0).mean()),
        "median_oos_of_is_best": float(np.median(oos_best)),
        "median_is_of_is_best": float(np.median(is_p[rows, best])),
        "degradation_slope": float(slope),
        "degradation_intercept": float(intercept),
        "n_splits": len(combos),
    }


# --------------------------------------------------------------------------- #
# Bootstrap-based multiple-testing procedures
# mean_block=1 resamples games iid; use mean_block > 1 (games sorted by time)
# when games on the same day / patch are dependent.
# --------------------------------------------------------------------------- #
def bootstrap_weights(n, B, rng, mean_block=1.0):
    """(B x n) resampling counts over rows (= games).
    mean_block <= 1: iid resampling of games.
    mean_block  > 1: stationary bootstrap (Politis & Romano 1994)."""
    if mean_block <= 1:
        return rng.multinomial(n, np.full(n, 1.0 / n), size=B).astype(float)
    idx = np.empty((B, n), dtype=np.int64)
    idx[:, 0] = rng.integers(0, n, B)
    jump = rng.random((B, n)) < 1.0 / mean_block
    start = rng.integers(0, n, (B, n))
    for t in range(1, n):
        idx[:, t] = np.where(jump[:, t], start[:, t], (idx[:, t - 1] + 1) % n)
    flat = (np.arange(B)[:, None] * n + idx).ravel()
    return np.bincount(flat, minlength=B * n).reshape(B, n).astype(float)


def spa_vs_no_bet(R, reps=1000, mean_block=1.0, seed=None, min_bets=100):
    """Hansen (2005) SPA (studentized, consistent p-value) and White (2000)
    Reality Check (not studentized).

    R[g, k] = profit per game. Benchmark = not betting (profit 0).
    H0: no model has a positive expected profit. The max statistic is
    bootstrapped jointly, so correlation between models is handled
    automatically (no effective-N guess needed).

    min_bets: candidates with fewer bets are dropped before testing.
    Studentizing the mean of a few skewed bets over-rejects, and the filter
    only partly helps. EXP-10 (zero EV, 200 candidates at fixed odds 1.4-5.0,
    2,000 games, nominal 5%): SPA 7.8% without the filter, 6.3% with it, 7.9%
    when every candidate bet on >= 5% of games; the RC stayed at 3.8-6.4%.

    test_toolkit.py checks the SPA against a literal re-implementation and the
    RC against arch. arch 8.0's SPA does not studentize, and its variance loop
    is O(n^2), which is why this version exists.
    """
    D = np.asarray(R, float)
    keep = (D != 0).sum(0) >= min_bets
    if not keep.any():
        return {"p_spa": 1.0, "p_rc": 1.0, "n_tested": 0}
    D = D[:, keep]
    rng = np.random.default_rng(seed)
    n = D.shape[0]
    W = bootstrap_weights(n, reps, rng, mean_block)
    dbar = D.mean(0)
    dstar = (W @ D) / n                                   # (reps, K) bootstrap means
    omega = np.sqrt(n * ((dstar - dbar) ** 2).mean(0))    # sd of sqrt(n) * dbar
    omega = np.where(omega > 1e-12, omega, np.inf)
    t = np.sqrt(n) * dbar / omega
    # SPA: clearly bad models (t below -sqrt(2 log log n)) keep their own mean
    mu = np.where(t <= -np.sqrt(2.0 * np.log(np.log(n))), dbar, 0.0)
    z = np.sqrt(n) * (dstar - dbar + mu) / omega
    p_spa = float((np.maximum(z.max(1), 0.0) >= max(t.max(), 0.0)).mean())
    # RC: not studentized, every model recentred at zero
    p_rc = float(((dstar - dbar).max(1) >= dbar.max()).mean())
    return {"p_spa": p_spa, "p_rc": p_rc, "n_tested": int(keep.sum())}


def model_confidence_set(L, alpha=0.10, reps=1000, block_size=1, method="R", seed=None):
    """Hansen, Lunde & Nason (2011) Model Confidence Set.

    L[g, k] = LOSS of model k in game g (lower = better): -profit, or log loss.
    Returns a boolean mask of the models that cannot be told apart from the
    best one at level alpha. method="R" (range statistic, the arch default) is
    slower but can be much more discriminating; method="max" has little power
    when ONE model dominates many equal ones (see test_toolkit.py)."""
    from arch.bootstrap import MCS

    L = np.asarray(L, float)
    mcs = MCS(L, size=alpha, reps=reps, block_size=block_size, method=method, seed=seed)
    mcs.compute()
    mask = np.zeros(L.shape[1], bool)
    mask[np.asarray(list(mcs.included), dtype=int)] = True
    return mask


# --------------------------------------------------------------------------- #
# Probability scoring against the market
# --------------------------------------------------------------------------- #
def log_loss(y, p, eps=1e-12):
    p = np.clip(p, eps, 1 - eps)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def clustered_mean_test(d, groups):
    """Mean of row-level d with a standard error clustered by `groups`
    (gameid, or series id to also cover maps of one series).
    Returns (mean, se, z)."""
    d = np.asarray(d, float)
    _, inv = np.unique(groups, return_inverse=True)
    G = inv.max() + 1
    if G < 2:
        raise ValueError("need at least two clusters")
    resid_sum = np.bincount(inv, weights=d - d.mean(), minlength=G)
    se = np.sqrt((resid_sum ** 2).sum() * G / (G - 1)) / d.size
    return float(d.mean()), float(se), float(d.mean() / se)


def logloss_vs_market(y, p_model, p_market, groups):
    """Paired log-loss difference (model - market). Negative = model better.
    One-sided p-value for 'model better than market'."""
    mean, se, z = clustered_mean_test(log_loss(y, p_model) - log_loss(y, p_market), groups)
    return {"delta_logloss": mean, "se": se, "z": z, "p_one_sided": float(stats.norm.cdf(z))}


def encompassing_test(y, p_market, p_model, groups):
    """Encompassing regression on the logit scale (Fair & Shiller 1990 idea;
    Benter 1994 combined his model with the public odds the same way):

        logit P(y=1) = a + b*logit(q) + c*(logit(p) - logit(q))

    * c > 0: the model carries information beyond ANY logit-linear
      recalibration of the price.
    * (a, b) != (0, 1): the price itself is miscalibrated (e.g. a
      favourite-longshot bias). An edge that is only a recalibration of the
      price shows up here, NOT in c.
    * p_joint: Wald test of (a, b, c) = (0, 1, 0), i.e. the fitted blend
      improves on the raw price.
    Neither implies profit after margin and execution. SEs clustered by
    `groups` (gameid, or series id)."""
    import statsmodels.api as sm

    eps = 1e-9
    q = np.clip(np.asarray(p_market, float), eps, 1 - eps)
    p = np.clip(np.asarray(p_model, float), eps, 1 - eps)
    lm, lp = np.log(q / (1 - q)), np.log(p / (1 - p))
    X = sm.add_constant(np.column_stack([lm, lp - lm]), has_constant="add")
    fit = sm.Logit(y, X).fit(disp=0, cov_type="cluster", cov_kwds={"groups": groups})
    a, b, c = (float(v) for v in fit.params)
    se_c = float(fit.bse[2])
    dev = np.array([a, b - 1.0, c])
    wald = float(dev @ np.linalg.solve(fit.cov_params(), dev))
    return {"a": a, "b_market": b, "c_model": c, "se_c": se_c,
            "p_one_sided": float(1 - stats.norm.cdf(c / se_c)),
            "p_joint": float(1 - stats.chi2.cdf(wald, 3))}


def _price_basis(lq, price_df, strata):
    """Columns that model the price's own calibration: a natural cubic spline
    in logit(q) with price_df columns (it spans the constant), one separate
    curve per stratum. price_df=None gives (1, logit q)."""
    if price_df is None:
        X = np.column_stack([np.ones_like(lq), lq])
    else:
        from patsy import dmatrix

        X = np.asarray(dmatrix(f"cr(x, df={price_df}) - 1", {"x": lq}))
    if strata is not None:
        _, s = np.unique(np.asarray(strata), return_inverse=True)
        X = (X[:, :, None] * np.eye(s.max() + 1)[s][:, None, :]).reshape(len(lq), -1)
    return X


def information_gate(y, p_market, P_models, groups, reps=2000, seed=None, alpha=0.05,
                     price_df=6, strata=None):
    """Family-wise test: does ANY of the K candidates carry information beyond
    the price? The multiple-testing-controlled version of the encompassing test,
    and much more powerful than testing P&L.

    For model k, d_k = logit(p_k) - logit(q); the statistic is the score for
    c = 0 in  logit P(y) = f(q) + c*d_k:  s_k = sum_g (y_g - q~_g) * d~_gk.
    * f is the price's own calibration curve, fitted by logistic regression: a
      natural cubic spline in logit(q) with `price_df` columns, one curve per
      stratum (e.g. season x checkpoint) if `strata` is given; q~ is the fit.
      A favourite-longshot bias, or a de-vig that does not match the book's
      margin, is absorbed here instead of being read as information.
    * d~_k is d_k minus its q~(1-q~)-weighted projection on the same columns
      (Neyman-orthogonal score), which absorbs the estimation of f.
    price_df=None uses (1, logit q) only: an independent review measured 9-36%
    false positives with it under a nonlinear favourite-longshot bias or a
    mismatched de-vig, and 24-40% under calibration drift between seasons (no
    strata); a spline with strata brought them to ~1-8% (EXP-11).
    `groups` (gameid, or series id) is required: rows of a group are summed
    first; stacking 15:00 and 20:00 rows without it gave 16.5% false positives.
    Gaussian multiplier bootstrap of the max studentized score (one multiplier
    per group, shared by all models). Single-step max-T adjusted p-values:
    `significant` controls the family-wise error; p_family = min(p_adjusted).
    """
    import statsmodels.api as sm

    y = np.asarray(y, float)
    q = np.clip(np.asarray(p_market, float), 1e-9, 1 - 1e-9)
    P = np.clip(np.asarray(P_models, float), 1e-9, 1 - 1e-9)
    if P.ndim == 1:
        P = P[:, None]
    lq = np.log(q / (1 - q))
    D = np.log(P / (1 - P)) - lq[:, None]
    X = _price_basis(lq, price_df, strata)
    qt = sm.Logit(y, X).fit(disp=0, method="newton", maxiter=100).predict(X)
    XtW = X.T * (qt * (1 - qt))
    D = D - X @ np.linalg.lstsq(XtW @ X, XtW @ D, rcond=None)[0]
    E = aggregate_by_game((y - qt)[:, None] * D, groups)
    den = np.sqrt((E ** 2).sum(0))
    den = np.where(den > 1e-12, den, np.inf)
    t = E.sum(0) / den
    xi = np.random.default_rng(seed).standard_normal((reps, E.shape[0]))
    mx = np.sort(((xi @ E) / den).max(1))
    p_adj = 1.0 - np.searchsorted(mx, t, side="left") / reps
    return {"p_family": float(p_adj.min()), "p_adjusted": p_adj, "t": t,
            "significant": p_adj < alpha, "best": int(t.argmax())}


def price_calibration_test(y, p_market, groups, price_df=6, strata=None):
    """Is the de-vigged price itself miscalibrated? Wald test, SEs clustered by
    `groups`, of theta = 0 in  logit P(y) = logit(q) + basis(q) @ theta  (same
    flexible basis as information_gate). If it rejects, the recalibrated price
    is an edge candidate on its own, with no model at all.
    Returns the p-value and the recalibrated probability."""
    import statsmodels.api as sm

    y = np.asarray(y, float)
    q = np.clip(np.asarray(p_market, float), 1e-9, 1 - 1e-9)
    lq = np.log(q / (1 - q))
    X = _price_basis(lq, price_df, strata)
    _, g = np.unique(np.asarray(groups), return_inverse=True)
    fit = sm.GLM(y, X, family=sm.families.Binomial(), offset=lq).fit(
        cov_type="cluster", cov_kwds={"groups": g})
    th, V = np.asarray(fit.params), np.asarray(fit.cov_params())
    wald = float(th @ np.linalg.pinv(V) @ th)
    return {"p_value": float(1 - stats.chi2.cdf(wald, np.linalg.matrix_rank(V))),
            "q_recalibrated": fit.predict(X, offset=lq)}


def fit_blend(y, p_market, p_model):
    """Benter-style blend fitted on PAST games only:
    logit P(y) = a + b*logit(q) + c*(logit(p) - logit(q)). Returns (a, b, c).
    Bet with blend_prob(...) on later games. It is a fitted model too: with a
    few thousand games its coefficients are noisy (EXP-7)."""
    import statsmodels.api as sm

    q = np.clip(np.asarray(p_market, float), 1e-9, 1 - 1e-9)
    p = np.clip(np.asarray(p_model, float), 1e-9, 1 - 1e-9)
    lq, lp = np.log(q / (1 - q)), np.log(p / (1 - p))
    X = sm.add_constant(np.column_stack([lq, lp - lq]), has_constant="add")
    return tuple(float(v) for v in sm.Logit(np.asarray(y, float), X).fit(disp=0).params)


def blend_prob(p_market, p_model, coef):
    a, b, c = coef
    q = np.clip(np.asarray(p_market, float), 1e-9, 1 - 1e-9)
    p = np.clip(np.asarray(p_model, float), 1e-9, 1 - 1e-9)
    lq, lp = np.log(q / (1 - q)), np.log(p / (1 - p))
    return 1.0 / (1.0 + np.exp(-(a + b * lq + c * (lp - lq))))


def pnl_eprocess(game_returns, alpha=0.05, lambdas=(0.005, 0.01, 0.02, 0.04, 0.08)):
    """Anytime-valid live monitoring of REALISED profit (testing by betting,
    Shafer 2021 / Ramdas et al. 2023; validity from Ville's inequality).

    game_returns: profit of each game you bet, in time order, in units of ONE
    fixed reference stake, at the odds you actually obtained, with all bets on
    the same game summed (aggregate_by_game).
    * up = mean over lambdas of prod(1 + lambda*r). If no bet has positive
      expected value at the odds obtained, this is a nonnegative supermartingale
      (sums of same-game bets included, even when the 20:00 bet depends on the
      20:00 state, because expectations add), so P(up EVER >= 1/alpha) <= alpha
      however often you look: 'scale'.
    * down = mean over lambdas of prod(1 - lambda*r): the same guarantee when no
      bet has negative expected value: 'kill' when down >= 1/alpha.
    It uses no claimed probabilities, so overconfident claims and slippage
    cannot fool it; the cost is less power than a test that trusts claims.
    Returns log paths, the decision and the 1-based game index.
    """
    from scipy.special import logsumexp

    r = np.asarray(game_returns, float)
    lam = np.asarray(lambdas, float)
    lr = np.outer(r, lam)
    if (np.abs(lr) >= 1).any():
        raise ValueError("|lambda * game return| must stay below 1: lower the lambdas")
    log_up = logsumexp(np.cumsum(np.log1p(lr), axis=0), axis=1) - np.log(lam.size)
    log_down = logsumexp(np.cumsum(np.log1p(-lr), axis=0), axis=1) - np.log(lam.size)
    thr = np.log(1.0 / alpha)
    hits = np.flatnonzero((log_up >= thr) | (log_down >= thr))
    if hits.size == 0:
        return {"log_up": log_up, "log_down": log_down, "decision": "continue", "game": None}
    i = int(hits[0])
    return {"log_up": log_up, "log_down": log_down,
            "decision": "scale" if log_up[i] >= thr else "kill", "game": i + 1}


# --------------------------------------------------------------------------- #
# Live-betting substitute for CLV
# --------------------------------------------------------------------------- #
def markout(odds_taken, fair_prob_later):
    """EV of a bet re-priced at a LATER fair (de-vigged) market probability of
    the side you backed: odds_taken * q_later - 1. With q_later = closing
    price this is the usual CLV-based EV. Unbiased for the realised EV only if
    the later market has absorbed all the information you had at bet time."""
    return np.asarray(odds_taken, float) * np.asarray(fair_prob_later, float) - 1.0


# --------------------------------------------------------------------------- #
# Sample size
# --------------------------------------------------------------------------- #
def bets_needed(edge, odds, z=2.0):
    """Independent flat bets needed so that a true ROI `edge` at decimal `odds`
    sits z standard errors from 0. Per-bet sd = odds * sqrt(p(1-p)),
    p = (1 + edge) / odds."""
    p = (1.0 + edge) / odds
    sd = odds * np.sqrt(p * (1.0 - p))
    return int(np.ceil((z * sd / edge) ** 2))
