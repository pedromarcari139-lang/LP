"""Builds research/oos_model_selection/README.md from results.json and
lol_resolution_check.out, so that no number in the README is typed by hand."""
import json
import re
from string import Template

import os
D = os.path.dirname(os.path.abspath(__file__))
import sys
RES = sys.argv[1] if len(sys.argv) > 1 else f"{D}/results.json"
OUT = sys.argv[2] if len(sys.argv) > 2 else f"{D}/README.md"
r = json.load(open(RES))
lol = open(f"{D}/lol_resolution_check.out").read()


def f(pattern, cast=float):
    m = re.search(pattern, lol)
    assert m, pattern
    return cast(m.group(1))


def pct(x, d=0):
    return f"{100 * x:.{d}f}%"


def spct(x, d=1):
    v = round(100 * x, d)
    return f"{0.0 if v == 0 else v:+.{d}f}%"


# ---------------------------------------------------------------- numbers
T = r["true_roi_table"]
E2 = r["exp2_winners_curse"]
A, B = r["exp34_worlds_n2000"], r["exp34_worlds_n10000"]
E5 = {(x["world"], x["n_games"]): x for x in r["exp5_power"]}
E6 = r["exp6_markouts"]
E7 = r["exp7_blend"]
G2, G10 = r["exp8_gates_n2000"], r["exp8_gates_n10000"]
E9 = r["exp9_live"]

names = {"clones_zero_ev": "clones, zero EV", "grid_zero_ev": "grid, zero EV",
         "clones_no_edge": "clones, no edge (−margin)", "grid_no_edge": "grid, no edge (−margin)",
         "clones_same_real_edge": "clones, same real edge", "grid_20_of_200_real": "grid, 20 of 200 real"}
null_worlds = ("clones_zero_ev", "grid_zero_ev", "clones_no_edge", "grid_no_edge")

# winner's curse
wc_rows = []
for x in E2:
    w = "no edge" if x["world"] == "grid_null" else "20 of 200 real"
    wc_rows.append(f"| {w} | {x['n_games']:,} | {spct(x['median_is_roi'])} | {spct(x['median_true_roi'])} | "
                   f"{pct(x['share_pick_informative'])} | {pct(x['share_naive_p_lt_5pct'], 1)} |")
tbl_wc = "\n".join(["| World | Games | In-sample ROI of the pick | True ROI of the pick | Pick has information | Naive p < 5% |",
                    "|---|---|---|---|---|---|", *wc_rows])
gaps = [x["median_is_roi"] - x["median_true_roi"] for x in E2]
gaps += [v["median_is_roi_selected"] - v["median_true_roi_selected"] for v in list(A.values()) + list(B.values())]
wc_range = f"+{100 * min(gaps):.0f} to +{100 * max(gaps):.0f}"


def rej(v, name):
    tot, fal = v[f"{name}_reject"], v[f"{name}_false_disc"]
    return pct(tot, 1) if fal in (0, tot) else f"{pct(tot, 1)} ({pct(fal, 1)} false)"


tool_rows = []
for tag, S in (("2,000", A), ("10,000", B)):
    for w, v in S.items():
        tool_rows.append(
            f"| {names[w]} | {tag} | {spct(v['median_true_roi_selected'])} | {rej(v, 'naive')} | {rej(v, 'bonferroni')} | "
            f"{rej(v, 'dsr_N_M')} | {rej(v, 'dsr_N_hat')} ({v['n_hat']:.0f}) | {pct(v['spa_reject'], 1)} | "
            f"{pct(v['rc_reject'], 1)} | {v['median_mcs_size']:.0f} | {v['pbo']:.2f} | {v['prob_oos_loss']:.2f} |")
tbl_tools = "\n".join(["| World | Games | True ROI of pick (median) | Naive | Bonferroni | DSR, N = M | DSR, N̂ (N̂) | SPA | RC | MCS size (median) | PBO | P(loss) |",
                       "|---|---|---|---|---|---|---|---|---|---|---|---|", *tool_rows])
size_worlds = [S[w] for S in (A, B) for w in null_worlds]
size_max = {k: max(v[k] for v in size_worlds) for k in ("spa_reject", "rc_reject", "dsr_N_M_reject", "dsr_N_hat_reject",
                                                        "bonferroni_reject", "naive_reject")}
zero_ev = [S[w] for S in (A, B) for w in ("clones_zero_ev", "grid_zero_ev")]
naive_zero = (min(v["naive_reject"] for v in zero_ev), max(v["naive_reject"] for v in zero_ev))
edge_c10 = B["clones_same_real_edge"]
mcs_all = [S[w]["median_mcs_size"] for S in (A, B) for w in S]

# power table
pw_rows = []
for n in (500, 1000, 2000, 5000, 10000):
    x = E5[("real_edge", n)]
    pw_rows.append(f"| {n:,} | {pct(x['roi_ttest'], 1)} | {pct(x['logloss_vs_market'], 1)} | {pct(x['encompassing_c'], 1)} | {pct(x['encompassing_joint'], 1)} |")
tbl_power = "\n".join(["| Games | ROI t-test | Δlog loss vs market | Encompassing, c > 0 | Encompassing, joint |",
                       "|---|---|---|---|---|", *pw_rows])
size5 = [E5[("no_edge", n)] for n in (500, 1000, 2000, 5000, 10000)]
enc_size = (min(x["encompassing_c"] for x in size5), max(x["encompassing_c"] for x in size5))
joint_size = (min(x["encompassing_joint"] for x in size5), max(x["encompassing_joint"] for x in size5))

# gates
gate_rows = []
for tag, S in (("2,000", G2), ("10,000", G10)):
    for w in ("clones_no_edge", "grid_no_edge", "clones_same_real_edge", "grid_20_of_200_real"):
        v = S[w]
        fwer = pct(v["any_uninformative_significant"], 1)
        gp = "—" if v["true_roi_gate_pick"] is None else spct(v["true_roi_gate_pick"])
        gate_rows.append(
            f"| {names[w]} | {tag} | {pct(v['info_gate_reject'], 1)} | {pct(v['spa_reject'], 1)} | {fwer} | "
            f"{pct(v['gate_go_with_info'], 1)} / {pct(v['gate_go_without_info'], 1)} / {gp} | "
            f"{pct(v['pick_sharpe_informative'])} / {spct(v['median_true_roi_pick_sharpe'])} |")
tbl_gates = "\n".join(["| World | Games | Information gate rejects | SPA on P&L rejects | A no-information model declared significant | Funnel (gate, then argmax of t): goes with an informative model / goes with a no-information model / true ROI of what goes | In-sample Sharpe argmax (always goes): has info / true ROI |",
                       "|---|---|---|---|---|---|---|", *gate_rows])

# blend
bl_rows = []
for x in E7:
    bl_rows.append(
        f"| {x['public_info_error_var']} | {x['n_train']:,} | {pct(x['encompassing_power_2000_games'], 1)} | "
        f"{spct(x['raw_roi_pooled'])} / {spct(x['raw_profit_per_game'], 2)} / {pct(x['raw_bet_rate'])} | "
        f"{spct(x['blend_roi_pooled'])} / {spct(x['blend_profit_per_game'], 2)} / {pct(x['blend_bet_rate'], 1)} "
        f"({pct(x['blend_bet_rate_min'], 1)}–{pct(x['blend_bet_rate_max'], 1)}) | {pct(x['blend_share_of_fits_losing'])} |")
tbl_blend = "\n".join(["| Error variance on public info | Training games | Encompassing detects the information (2,000 games) | Raw model: ROI per bet / profit per game / bet rate | Blend: ROI per bet / profit per game / bet rate (range over fits) | Losing blend fits |",
                       "|---|---|---|---|---|---|", *bl_rows])
B7 = {(x["public_info_error_var"], x["n_train"]): x for x in E7}

# markouts
lab = {"catch_up_edge": "catch-up edge (market learns it by 20:00)", "persistent_edge": "persistent edge (market never learns it)", "no_edge": "no edge"}
mk_rows = []
for k, d in E6.items():
    pw = [p for p in d["power"] if p["n_games"] == 2000][0]
    mk_rows.append(f"| {lab[k]} | {spct(d['mean_profit_per_bet'])} ± {100 * d['se_profit_mean']:.1f} | {spct(d['mean_markout_per_bet'])} | "
                   f"{spct(d['mean_residual_profit_minus_markout'])} ± {100 * d['se_residual']:.1f} | {d['variance_ratio']:.1f} | "
                   f"{pct(pw['reject_rate_profit_ttest'], 1)} vs {pct(pw['reject_rate_markout_ttest'], 1)} |")
tbl_mk = "\n".join(["| World (1M games, odds ≤ 5) | Profit/bet ± SE | Markout/bet | Profit − markout ± SE | Var(profit)/Var(markout) | Detection at 2,000 games: profit vs markout |",
                    "|---|---|---|---|---|---|", *mk_rows])
pe = E6["persistent_edge"]
resid_var_share = 1 - 1 / E6["catch_up_edge"]["variance_ratio"]

# live
lv_rows = []
for k, v in E9.items():
    if k.startswith("drop_top"):
        continue
    med = "—" if v["median_bets_to_decision_if_decided"] is None else f"{v['median_bets_to_decision_if_decided']:,.0f}"
    naive = pct(v["naive_t_above_1_645_at_any_check"], 1) if "naive_t_above_1_645_at_any_check" in v else "—"
    lv_rows.append(f"| {k} | {spct(v['true_roi_per_bet'])} | {pct(v['scale'], 1)} | {pct(v['kill'], 1)} | "
                   f"{pct(v['undecided'], 1)} | {med} | {naive} |")
tbl_live = "\n".join(["| Scenario | True ROI per bet | E-process scales up | E-process kills | Undecided after 5,000 | Median games to a decision (decided paths) | Naive t > 1.645 at any check |",
                      "|---|---|---|---|---|---|---|", *lv_rows])
trim = E9["drop_top_1pct_kills_real_3pct_edge_share"]
tbl_trim = " · ".join(f"odds {o}: {pct(v)}" for o, v in trim.items())
L9 = E9

# gate robustness (EXP-11)
E11 = r["exp11_gate_robustness"]
case_name = {"devig": "de-vig does not match the book's margin", "flb": "nonlinear favourite-longshot bias",
             "drift": "calibration drifts between two seasons"}
gr_rows, lin_all, flex_all = [], [], []
for key, v in E11.items():
    case, n = key.rsplit("_", 1)
    st = "—" if v["spline_with_strata"] is None else pct(v["spline_with_strata"], 1)
    gr_rows.append(f"| {case_name[case]} | {int(n):,} | {pct(v['logit_linear'], 1)} | {pct(v['spline'], 1)} | {st} |")
    lin_all.append(v["logit_linear"])
    flex_all.append(v["spline"] if v["spline_with_strata"] is None else v["spline_with_strata"])
tbl_gate_robust = "\n".join(["| Scenario (no model has information) | Games | Logit-linear correction | Spline (default) | Spline + strata (season) |",
                             "|---|---|---|---|---|", *gr_rows])

# sample size
ss_rows = []
for x in r["exp1_sample_size"]["rows"]:
    if x["odds"] in (1.5, 1.9, 2.5) and x["edge"] in (0.02, 0.03, 0.05):
        ss_rows.append(f"| {x['odds']} | {x['edge']:.0%} | {x['t2']:,} | {x['t3']:,} | {x['power80']:,} | {x['power80_bonf200']:,} |")
tbl_ss = "\n".join(["| Decimal odds | True ROI | t = 2 | t = 3 | 80% power (5%, one-sided) | Same, Bonferroni ×200 |",
                    "|---|---|---|---|---|---|", *ss_rows])
ss19 = [x for x in r["exp1_sample_size"]["rows"] if x["odds"] == 1.9 and x["edge"] == 0.03][0]

LOL = dict(
    test=f(r"test games \(2023-24\): (\d+)", int), short=f(r"ended before 20:00: (\d+)", int),
    acc15=f(r"at 15:00  accuracy=([\d.]+)"), acc20=f(r"at 20:00 \(live games\) accuracy=([\d.]+)"),
    meanp=f(r"mean p=([\d.]+)"), wr=f(r"blue winrate=([\d.]+)"),
    vout=f(r"E\[q15\(1-q15\)\]=([\d.]+)"), dq=f(r"E\[\(q20-q15\)\^2\]=([\d.]+)"),
    gap=f(r"= ([+-][\d.]+) \(SE"), gap_se=f(r"\(SE ([\d.]+)\)"), brier=f(r"Brier at 15:00 = ([\d.]+)"),
    all=f(r"all sides\s+:\s+([\d.]+)"), band=f(r"fair odds 1.5-3.0:\s+([\d.]+)"), le5=f(r"fair odds <= 5\s+:\s+([\d.]+)"),
    long_share=f(r"sides with q15 < 0.1: ([\d.]+)%"), long_var=f(r"of sides, ([\d.]+)% of the profit"),
)

E10 = r["exp10_sizes"]
tbl_dsr = "\n".join(["| Correlation between trials | DSR false positives, N = M | DSR false positives, N̂ (Eq. 9) |",
                     "     |---|---|---|",
                     *[f"     | {rho} | {pct(v['N_M'], 1)} | {pct(v['N_hat'], 1)} |" for rho, v in E10["dsr"].items()]])
tbl_spa_rare = "\n".join(["| Candidates | SPA false positives | RC false positives |", "     |---|---|---|",
                          *[f"     | {k} | {pct(v['spa'], 1)} | {pct(v['rc'], 1)} |" for k, v in E10["spa_rare"].items()]])

def rng_(ws, key):
    xs = [S[w][key] for S in (A, B) for w in ws]
    lo, hi = min(xs), max(xs)
    return pct(lo, 1) if abs(hi - lo) < 1e-12 else f"{pct(lo, 1)}–{pct(hi, 1)}"


def span(xs, d=1):
    lo, hi = min(xs), max(xs)
    return pct(lo, d) if abs(hi - lo) < 1e-12 else f"{pct(lo, d)}–{pct(hi, d)}"


zw, mw = ("clones_zero_ev", "grid_zero_ev"), ("clones_no_edge", "grid_no_edge")
no_info = ("clones_no_edge", "grid_no_edge")
b0, b2 = B7[(0.0, 10000)], B7[(0.02, 2000)]
vals = dict(
    tbl_dsr=tbl_dsr, tbl_spa_rare=tbl_spa_rare, tbl_gate_robust=tbl_gate_robust,
    e11_lin_rng=span(lin_all), e11_flex_rng=span(flex_all),
    z_spa=rng_(zw, "spa_reject"), m_spa=rng_(mw, "spa_reject"), z_rc=rng_(zw, "rc_reject"), m_rc=rng_(mw, "rc_reject"),
    z_dsr=rng_(zw, "dsr_N_M_reject"), m_dsr=rng_(mw, "dsr_N_M_reject"), z_dsrhat=rng_(zw, "dsr_N_hat_reject"),
    m_dsrhat=rng_(mw, "dsr_N_hat_reject"), z_naive=rng_(zw, "naive_reject"), m_naive=rng_(mw, "naive_reject"),
    gsize_rng=span([S[w]["info_gate_reject"] for S in (G2, G10) for w in no_info]),
    gsize_rng2=span([G2[w]["info_gate_reject"] for w in no_info]),
    gsize_rng10=span([G10[w]["info_gate_reject"] for w in no_info]),
    go_wrong_max=pct(max(S["grid_20_of_200_real"]["gate_go_without_info"] for S in (G2, G10)), 1),
    sharpe_wrong_2k=pct(1 - G2["grid_20_of_200_real"]["pick_sharpe_informative"]),
    gate2=pct(G2["clones_same_real_edge"]["info_gate_reject"]), gate10=pct(G10["clones_same_real_edge"]["info_gate_reject"]),
    spa2=pct(G2["clones_same_real_edge"]["spa_reject"]),
    raw_roi0=spct(b0["raw_roi_pooled"]), blend_roi0=spct(b0["blend_roi_pooled"]),
    raw_ppg0=spct(b0["raw_profit_per_game"], 2), blend_ppg0=spct(b0["blend_profit_per_game"], 2),
    raw_roi2=spct(b2["raw_roi_pooled"]), blend_roi2=spct(b2["blend_roi_pooled"]),
    blend_lose2=pct(b2["blend_share_of_fits_losing"]),
    raw_ppg2=spct(b2["raw_profit_per_game"], 2),
    blend_ppg2_10k=spct(B7[(0.02, 10000)]["blend_profit_per_game"], 2),
    blend_rate2_10k=pct(B7[(0.02, 10000)]["blend_bet_rate"], 1),
    ep_kill3=pct(L9["losing -3% on every bet"]["kill"]),
    ep_undec0=pct(L9["break-even (no edge)"]["undecided"]),
    naive_fa=pct(L9["break-even (no edge)"]["naive_t_above_1_645_at_any_check"]),
    ep_fa=pct(L9["break-even (no edge)"]["scale"], 1),
    ep_chase=pct(L9["chasing at 20:00 (zero EV, bets summed per game)"]["scale"], 1),
    ep_mixed=pct(L9["mixed: 40% of bets +5%, 60% -5% (overall -1%)"]["scale"], 1),
    ep_slip=pct(L9["+2% at requested odds, filled 0-8% worse"]["scale"], 1),
    ep_power5=pct(L9["real +5% on every bet"]["scale"]),
    clone_roi=spct(T["clone_roi"]), inf_lo=spct(T["informative_roi_min"]), inf_hi=spct(T["informative_roi_max"]),
    wc_range=wc_range, tbl_wc=tbl_wc, tbl_tools=tbl_tools, tbl_power=tbl_power, tbl_gates=tbl_gates,
    tbl_blend=tbl_blend, tbl_mk=tbl_mk, tbl_live=tbl_live, tbl_trim=tbl_trim, tbl_ss=tbl_ss,
    spa10=pct(edge_c10["spa_reject"]), rc10=pct(edge_c10["rc_reject"]), dsr10=pct(edge_c10["dsr_N_M_reject"]),
    mcs_lo=f"{min(mcs_all):.0f}", mcs_hi=f"{max(mcs_all):.0f}",
    enc_lo=pct(enc_size[0], 1), enc_hi=pct(enc_size[1], 1), joint_lo=pct(joint_size[0], 1), joint_hi=pct(joint_size[1], 1),
    enc2k=pct(E5[("real_edge", 2000)]["encompassing_c"]), roi2k=pct(E5[("real_edge", 2000)]["roi_ttest"]),
    vr_catch=f"{E6['catch_up_edge']['variance_ratio']:.1f}", resid_share=pct(resid_var_share),
    pe_both=pct(pe["share_games_bet_at_both_times"]), pe_corr=f"{pe['corr_15_20_returns_same_game']:.2f}",
    pe_se=f"{pe['se_ratio_clustered_over_naive']:.2f}",
    ss_t2=f"{ss19['t2']:,}", ss_bonf=f"{ss19['power80_bonf200']:,}",
    pbo_clone_edge=f"{B['clones_same_real_edge']['pbo']:.2f}", pbo_clone_noedge=f"{B['clones_no_edge']['pbo']:.2f}",
    pbo_grid_noedge=f"{B['grid_no_edge']['pbo']:.2f}", pbo_g20=f"{B['grid_20_of_200_real']['pbo']:.2f}",
    ploss_edge=f"{B['clones_same_real_edge']['prob_oos_loss']:.2f}", ploss_noedge=f"{B['clones_no_edge']['prob_oos_loss']:.2f}",
    nhat_lo=f"{min(S[w]['n_hat'] for S in (A, B) for w in S):.0f}", nhat_hi=f"{max(S[w]['n_hat'] for S in (A, B) for w in S):.0f}",
    lol_test=f"{LOL['test']:,}", lol_short=LOL["short"],
    lol_acc15=pct(LOL["acc15"], 1), lol_acc20=pct(LOL["acc20"], 1),
    lol_meanp=f"{LOL['meanp']:.3f}", lol_wr=f"{LOL['wr']:.3f}",
    lol_vout=f"{LOL['vout']:.3f}", lol_dq=f"{LOL['dq']:.3f}",
    lol_gap=f"{LOL['gap']:.3f}", lol_gap_se=f"{LOL['gap_se']:.3f}", lol_brier=f"{LOL['brier']:.3f}",
    lol_all=f"{LOL['all']:.1f}", lol_band=f"{LOL['band']:.1f}", lol_le5=f"{LOL['le5']:.1f}",
    lol_long_share=f"{LOL['long_share']:.1f}%", lol_long_var=f"{LOL['long_var']:.1f}%",
    runtime="~1.5 h",
)
tpl = Template(open(f"{D}/README.tpl").read())
out = tpl.substitute(vals)
open(OUT, "w").write(out)
print("README written,", len(out.split()), "words")
