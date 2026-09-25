"""
Synthetic "LoL live betting" world used to put numbers on the claims in README.md.

It is NOT a model of real LoL markets. Every magnitude (margin, size of the edge,
how much of the game is decided between 15:00 and 20:00, the odds cap) is an
assumption in PARAMS below. The point is to show MECHANISMS (selection bias, what
PBO / DSR / SPA / MCS react to, when markouts can replace CLV) with numbers.

Latent final score X = A + B + Bp + C + D, side A wins iff X > 0 (all Normal):
  A  : public at 15:00 (market and models see it)
  B  : seen by an informative model at 15:00, public by 20:00   ("catch-up" edge)
  Bp : seen by an informative model, never public before the end ("persistent" edge)
  C  : revealed between 15:00 and 20:00 (game events)
  D  : revealed after 20:00
Market fair prices are exact conditional probabilities given public info, so the
market is calibrated and a martingale; the bookmaker adds a proportional margin.
No rule ever backs a price above MAX_ODDS.

Models are fixed rules (not fitted), so the ONLY overfitting simulated here is
selection among 200 candidates. Real pipelines add fitting overfit and leakage.

Run:  python simulate.py            (full run, ~1.5 h on 4 cores)
      python simulate.py --quick    (smoke test)
"""
import argparse
import json
import time
from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

import oos_toolkit as tk

PARAMS = dict(VA=0.35, VC=0.15, margin=0.05, sig_common=0.08, sig_idio=0.03,
              V_EDGE=0.005, MAX_ODDS=5.0)


# --------------------------------------------------------------------------- #
# World
# --------------------------------------------------------------------------- #
def draw_games(n, rng, VB=0.0, VBp=0.0, margin=None):
    VA, VC = PARAMS["VA"], PARAMS["VC"]
    m = PARAMS["margin"] if margin is None else margin
    VD = 1.0 - VA - VB - VBp - VC
    A = rng.normal(0, np.sqrt(VA), n)
    B = rng.normal(0, np.sqrt(VB), n) if VB > 0 else np.zeros(n)
    Bp = rng.normal(0, np.sqrt(VBp), n) if VBp > 0 else np.zeros(n)
    C = rng.normal(0, np.sqrt(VC), n)
    D = rng.normal(0, np.sqrt(VD), n)
    y = (A + B + Bp + C + D > 0).astype(float)
    q15 = norm.cdf(A / np.sqrt(1.0 - VA))                    # market fair prob at 15:00
    q20 = norm.cdf((A + B + C) / np.sqrt(VBp + VD))          # market fair prob at 20:00
    return dict(A=A, B=B, Bp=Bp, C=C, y=y, q15=q15, q20=q20,
                oa15=1.0 / (q15 * (1 + m)), ob15=1.0 / ((1 - q15) * (1 + m)),
                oa20=1.0 / (q20 * (1 + m)), ob20=1.0 / ((1 - q20) * (1 + m)),
                VBp=VBp, VD=VD)


@dataclass
class Models:
    beta: np.ndarray          # weight each model puts on its signal
    thr: np.ndarray           # minimum EV required to bet
    informative: np.ndarray   # does the signal contain B + Bp?


GRID_BETAS = np.linspace(0.4, 0.8, 50)
GRID_THRS = (0.0, 0.02, 0.04, 0.06)
CLONE = (0.6, 0.02)


def grid_models(rng, n_informative, K=200):
    """50 signal weights x 4 EV thresholds: 'many very similar models'."""
    beta = np.repeat(GRID_BETAS, K // 50)
    thr = np.tile(GRID_THRS, K // 4)
    inf = np.zeros(K, bool)
    inf[rng.choice(K, n_informative, replace=False)] = True
    return Models(beta, thr, inf)


def clone_models(informative, K=200, beta=CLONE[0], thr=CLONE[1]):
    """200 copies of one rule that differ only by idiosyncratic feature noise:
    every candidate has EXACTLY the same true edge."""
    return Models(np.full(K, beta), np.full(K, thr), np.full(K, informative))


def model_signal(g, models, rng):
    n, K = g["A"].size, models.beta.size
    common = rng.normal(0, PARAMS["sig_common"], n)[:, None]       # shared feature noise
    idio = rng.normal(0, PARAMS["sig_idio"], (n, K))
    edge_info = (g["B"] + g["Bp"])[:, None] * models.informative[None, :]
    return edge_info + common + idio


def returns_15(g, models, rng):
    S = model_signal(g, models, rng)
    p = norm.cdf((g["A"][:, None] + models.beta[None, :] * S) / np.sqrt(1 - PARAMS["VA"]))
    R, bet = tk.flat_bet_returns(p, g["oa15"][:, None], g["ob15"][:, None],
                                 g["y"][:, None], models.thr[None, :], PARAMS["MAX_ODDS"])
    return R, bet, p


def roi_per_bet(R, bet):
    return R.sum(0) / np.maximum(bet.sum(0), 1)


# --------------------------------------------------------------------------- #
# True ROI of a rule
# * no information: every bet has EV = 1/(1+m) - 1 exactly (price is fair).
# * informative: depends only on (beta, thr); estimated once on 10M games.
# --------------------------------------------------------------------------- #
TRUE_ROI = {}


def build_true_roi_table(n_total=10_000_000, chunk=100_000, seed=99):
    pairs = sorted({(round(float(b), 10), t) for b in GRID_BETAS for t in GRID_THRS} | {CLONE})
    models = Models(np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs]),
                    np.ones(len(pairs), bool))
    rng = np.random.default_rng(seed)
    profit, bets = np.zeros(len(pairs)), np.zeros(len(pairs))
    for _ in range(n_total // chunk):
        R, bet, _ = returns_15(draw_games(chunk, rng, VB=PARAMS["V_EDGE"]), models, rng)
        profit += R.sum(0)
        bets += bet.sum(0)
    for i, pair in enumerate(pairs):
        TRUE_ROI[pair] = profit[i] / bets[i]
    return {"n_games": n_total, "min_bets_per_rule": int(bets.min()),
            "informative_roi_min": float(min(TRUE_ROI.values())),
            "informative_roi_max": float(max(TRUE_ROI.values())),
            "clone_roi": float(TRUE_ROI[CLONE])}


def true_roi(models, k, margin):
    if not models.informative[k]:
        return 1.0 / (1.0 + margin) - 1.0
    return TRUE_ROI[(round(float(models.beta[k]), 10), float(models.thr[k]))]


# --------------------------------------------------------------------------- #
# EXP-1  Sample size (analytic)
# --------------------------------------------------------------------------- #
def exp1_sample_size():
    z80 = norm.ppf(0.95) + norm.ppf(0.80)                         # 80% power, 5% one-sided
    z80_bonf = norm.ppf(1 - 0.05 / 200) + norm.ppf(0.80)          # same, Bonferroni K=200
    rows = []
    for odds in (1.5, 1.9, 2.5, 3.5):
        for edge in (0.01, 0.02, 0.03, 0.05):
            rows.append(dict(odds=odds, edge=edge,
                             t2=tk.bets_needed(edge, odds, 2.0),
                             t3=tk.bets_needed(edge, odds, 3.0),
                             power80=tk.bets_needed(edge, odds, z80),
                             power80_bonf200=tk.bets_needed(edge, odds, z80_bonf)))
    return dict(z80=z80, z80_bonf=z80_bonf, rows=rows)


# --------------------------------------------------------------------------- #
# EXP-2  Winner's curse: IS ROI of the best of 200 vs its true ROI
# --------------------------------------------------------------------------- #
def exp2_winners_curse(reps, ns, seed=2):
    rng = np.random.default_rng(seed)
    m = PARAMS["margin"]
    out = []
    for world, n_inf, VB in (("grid_null", 0, 0.0), ("grid_20_informative", 20, PARAMS["V_EDGE"])):
        for n in ns:
            is_roi, tr_roi, naive_sig, picked_inf = [], [], [], []
            for _ in range(reps):
                models = grid_models(rng, n_inf)
                R, bet, _ = returns_15(draw_games(n, rng, VB=VB), models, rng)
                sr = tk.sharpe(R)
                k = int(sr.argmax())
                is_roi.append(roi_per_bet(R, bet)[k])
                naive_sig.append(sr[k] * np.sqrt(n) > norm.ppf(0.95))
                tr_roi.append(true_roi(models, k, m))
                picked_inf.append(models.informative[k])
            out.append(dict(world=world, n_games=n,
                            median_is_roi=float(np.median(is_roi)),
                            median_true_roi=float(np.median(tr_roi)),
                            share_pick_informative=float(np.mean(picked_inf)),
                            share_true_roi_negative=float(np.mean(np.array(tr_roi) < 0)),
                            share_naive_p_lt_5pct=float(np.mean(naive_sig))))
    return out


# --------------------------------------------------------------------------- #
# EXP-3/4  What PBO, DSR, SPA, RC, MCS say in six worlds
# "minus margin" worlds: every rule without information loses the margin.
# "zero EV" worlds: no margin, so rules without information have EV exactly 0;
# rejection rates there are the tests' real false-positive rates (size).
# --------------------------------------------------------------------------- #
WORLDS = {
    "clones_zero_ev": dict(models=lambda rng: clone_models(False), VB=0.0, margin=0.0),
    "grid_zero_ev": dict(models=lambda rng: grid_models(rng, 0), VB=0.0, margin=0.0),
    "clones_no_edge": dict(models=lambda rng: clone_models(False), VB=0.0, margin=PARAMS["margin"]),
    "grid_no_edge": dict(models=lambda rng: grid_models(rng, 0), VB=0.0, margin=PARAMS["margin"]),
    "clones_same_real_edge": dict(models=lambda rng: clone_models(True), VB=PARAMS["V_EDGE"],
                                  margin=PARAMS["margin"]),
    "grid_20_of_200_real": dict(models=lambda rng: grid_models(rng, 20), VB=PARAMS["V_EDGE"],
                                margin=PARAMS["margin"]),
}


def one_replication(world, n, B, seed):
    rng = np.random.default_rng(seed)
    spec = WORLDS[world]
    models = spec["models"](rng)
    R, bet, _ = returns_15(draw_games(n, rng, VB=spec["VB"], margin=spec["margin"]), models, rng)
    sr = tk.sharpe(R)
    k = int(sr.argmax())
    K = R.shape[1]
    inf_k = bool(models.informative[k])
    naive = 1 - norm.cdf(sr[k] * np.sqrt(n)) < 0.05
    bonf = 1 - norm.cdf(sr[k] * np.sqrt(n)) < 0.05 / K
    dsr_m, _ = tk.deflated_sharpe(R[:, k], sr, K)
    n_hat = tk.implied_independent_trials(R)
    dsr_hat, _ = tk.deflated_sharpe(R[:, k], sr, max(n_hat, 1.0))
    spa = tk.spa_vs_no_bet(R, reps=B, seed=int(rng.integers(2 ** 31)))
    in_mcs = tk.model_confidence_set(-R, alpha=0.10, reps=min(B, 300), method="R",
                                     seed=int(rng.integers(2 ** 31)))
    pbo = tk.pbo_cscv(R, n_blocks=16)
    corr = np.corrcoef(R[:, R.std(0) > 0], rowvar=False)
    out = dict(
        is_roi_selected=float(roi_per_bet(R, bet)[k]),
        bets_selected=int(bet[:, k].sum()),
        median_bets_all_models=float(np.median(bet.sum(0))),
        true_roi_selected=float(true_roi(models, k, spec["margin"])),
        selected_is_informative=inf_k,
        n_hat=n_hat,
        p_spa=spa["p_spa"], spa_reject=bool(spa["p_spa"] < 0.05),
        p_rc=spa["p_rc"], rc_reject=bool(spa["p_rc"] < 0.05),
        spa_n_tested=spa["n_tested"],
        mcs_size=int(in_mcs.sum()),
        mcs_informative_share=float(models.informative[in_mcs].mean()),
        pbo=pbo["pbo"], prob_oos_loss=pbo["prob_oos_loss"],
        degradation_slope=pbo["degradation_slope"],
        mean_pairwise_corr=float(corr[np.triu_indices_from(corr, 1)].mean()),
    )
    # tests about the SELECTED model: a rejection is a true discovery only if
    # the selected model really has information
    for name, rej in (("naive", naive), ("bonferroni", bonf),
                      ("dsr_N_M", dsr_m > 0.95), ("dsr_N_hat", dsr_hat > 0.95)):
        out[f"{name}_reject"] = bool(rej)
        out[f"{name}_true_disc"] = bool(rej and inf_k)
        out[f"{name}_false_disc"] = bool(rej and not inf_k)
    return out


def _one_replication_star(args):
    return one_replication(*args)


def exp34_worlds(reps, n, B, seed=34, processes=4):
    from multiprocessing import Pool

    seeds = np.random.default_rng(seed).integers(2 ** 31, size=(len(WORLDS), reps))
    tasks = [(w, n, B, int(s)) for w, row in zip(WORLDS, seeds) for s in row]
    with Pool(processes) as pool:
        flat = pool.map(_one_replication_star, tasks)
    summary = {}
    for i, world in enumerate(WORLDS):
        rs = flat[i * reps:(i + 1) * reps]
        agg = {key: float(np.mean([r[key] for r in rs])) for key in rs[0]}
        for key in ("pbo", "true_roi_selected", "is_roi_selected", "bets_selected", "mcs_size"):
            agg[f"median_{key}"] = float(np.median([r[key] for r in rs]))
        agg["replications"] = reps
        summary[world] = agg
    return summary


# --------------------------------------------------------------------------- #
# EXP-5  Power: ROI t-test vs log loss vs market vs encompassing regression
# --------------------------------------------------------------------------- #
def exp5_power(reps, ns, seed=5):
    rng = np.random.default_rng(seed)
    one_inf, one_null = clone_models(True, K=1), clone_models(False, K=1)
    out = []
    for label, models, VB in (("real_edge", one_inf, PARAMS["V_EDGE"]), ("no_edge", one_null, 0.0)):
        for n in ns:
            rej = dict(roi_ttest=0, logloss_vs_market=0, encompassing_c=0, encompassing_joint=0)
            for _ in range(reps):
                g = draw_games(n, rng, VB=VB)
                R, _, p = returns_15(g, models, rng)
                r = R[:, 0]
                if r.std() > 0:
                    rej["roi_ttest"] += r.mean() / (r.std(ddof=1) / np.sqrt(n)) > norm.ppf(0.95)
                gid = np.arange(n)
                rej["logloss_vs_market"] += tk.logloss_vs_market(g["y"], p[:, 0], g["q15"], gid)["p_one_sided"] < 0.05
                enc = tk.encompassing_test(g["y"], g["q15"], p[:, 0], gid)
                rej["encompassing_c"] += enc["p_one_sided"] < 0.05
                rej["encompassing_joint"] += enc["p_joint"] < 0.05
            out.append(dict(world=label, n_games=n, **{k: v / reps for k, v in rej.items()}))
    return out


# --------------------------------------------------------------------------- #
# EXP-6  Live substitute for CLV (markouts at 20:00) and gameid clustering
# --------------------------------------------------------------------------- #
def exp6_markouts(n_big, reps, ns, seed=6):
    rng = np.random.default_rng(seed)
    beta, thr, cap = CLONE[0], CLONE[1], PARAMS["MAX_ODDS"]
    res = {}
    for case, VB, VBp in (("catch_up_edge", PARAMS["V_EDGE"], 0.0),
                          ("persistent_edge", 0.0, PARAMS["V_EDGE"]),
                          ("no_edge", 0.0, 0.0)):
        def one_sample(n):
            g = draw_games(n, rng, VB=VB, VBp=VBp)
            common = rng.normal(0, PARAMS["sig_common"], n)
            idio = rng.normal(0, PARAMS["sig_idio"], n)
            s15 = g["B"] + g["Bp"] + common + idio
            p15 = norm.cdf((g["A"] + beta * s15) / np.sqrt(1 - PARAMS["VA"]))
            r15, bet15 = tk.flat_bet_returns(p15, g["oa15"], g["ob15"], g["y"], thr, cap)
            back_a = bet15 & (p15 * g["oa15"] > (1 - p15) * g["ob15"])
            odds = np.where(back_a, g["oa15"], g["ob15"])
            q_later = np.where(back_a, g["q20"], 1 - g["q20"])
            y_side = np.where(back_a, g["y"], 1 - g["y"])
            mk = np.where(bet15, tk.markout(odds, q_later), 0.0)
            resid = np.where(bet15, odds * (y_side - q_later), 0.0)   # profit - markout
            # the same rule at 20:00: its only remaining private info is Bp
            s20 = g["Bp"] + common + idio if VBp > 0 else np.zeros(n)
            p20 = norm.cdf((g["A"] + g["B"] + g["C"] + beta * s20) / np.sqrt(g["VBp"] + g["VD"]))
            r20, bet20 = tk.flat_bet_returns(p20, g["oa20"], g["ob20"], g["y"], thr, cap)
            return r15, bet15, mk, resid, r20, bet20

        r15, bet15, mk, resid, r20, bet20 = one_sample(n_big)
        pr, m = r15[bet15], mk[bet15]
        both = bet15 & bet20
        # clustering: all bets (15:00 and 20:00) of the rule, naive vs by-game SE
        vals = np.concatenate([r15[bet15], r20[bet20]])
        gid = np.concatenate([np.flatnonzero(bet15), np.flatnonzero(bet20)])
        _, se_cl, _ = tk.clustered_mean_test(vals, gid)
        se_naive = vals.std(ddof=1) / np.sqrt(vals.size)
        d = dict(
            bets_15=int(bet15.sum()), mean_profit_per_bet=float(pr.mean()),
            se_profit_mean=float(pr.std() / np.sqrt(pr.size)),
            mean_markout_per_bet=float(m.mean()), sd_profit=float(pr.std()),
            sd_markout=float(m.std()), variance_ratio=float(pr.var() / m.var()),
            mean_residual_profit_minus_markout=float(resid[bet15].mean()),
            se_residual=float(resid[bet15].std() / np.sqrt(bet15.sum())),
            share_games_bet_at_both_times=float(both.sum() / max(bet15.sum(), 1)),
            corr_15_20_returns_same_game=(float(np.corrcoef(r15[both], r20[both])[0, 1])
                                          if both.sum() > 10 else None),
            se_ratio_clustered_over_naive=float(se_cl / se_naive),
        )
        power = []
        for n in ns:
            rej_p = rej_m = 0
            for _ in range(reps):
                r15s, b, mks, *_ = one_sample(n)
                x, z = r15s[b], mks[b]
                rej_p += x.mean() / (x.std(ddof=1) / np.sqrt(x.size)) > norm.ppf(0.95)
                rej_m += z.mean() / (z.std(ddof=1) / np.sqrt(z.size)) > norm.ppf(0.95)
            power.append(dict(n_games=n, reject_rate_profit_ttest=rej_p / reps,
                              reject_rate_markout_ttest=rej_m / reps))
        d["power"] = power
        res[case] = d
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", default="1,2,34,34big,5,6")
    ap.add_argument("--out", default="results.json")
    a = ap.parse_args()
    q = a.quick
    todo = set(a.only.split(","))
    results = {"params": PARAMS}
    t0 = time.time()
    results["true_roi_table"] = build_true_roi_table(n_total=500_000 if q else 10_000_000)
    print("true-ROI table done", round(time.time() - t0), "s", flush=True)
    if "1" in todo:
        results["exp1_sample_size"] = exp1_sample_size()
    if "2" in todo:
        results["exp2_winners_curse"] = exp2_winners_curse(5 if q else 200, (500, 1000, 2000, 5000))
        print("exp2 done", round(time.time() - t0), "s", flush=True)
    if "34" in todo:
        results["exp34_worlds_n2000"] = exp34_worlds(3 if q else 200, 2000, 200 if q else 500)
        print("exp34 n=2000 done", round(time.time() - t0), "s", flush=True)
    if "34big" in todo:
        results["exp34_worlds_n10000"] = exp34_worlds(3 if q else 100, 10000, 200 if q else 500, seed=3434)
        print("exp34 n=10000 done", round(time.time() - t0), "s", flush=True)
    if "5" in todo:
        results["exp5_power"] = exp5_power(20 if q else 1000, (500, 1000, 2000, 5000, 10000))
        print("exp5 done", round(time.time() - t0), "s", flush=True)
    if "6" in todo:
        results["exp6_markouts"] = exp6_markouts(50_000 if q else 1_000_000, 20 if q else 1000,
                                                 (500, 1000, 2000))
        print("exp6 done", round(time.time() - t0), "s", flush=True)
    with open(a.out, "w") as f:
        json.dump(results, f, indent=1, default=float)
    print(json.dumps(results, indent=1, default=float))


if __name__ == "__main__":
    main()
