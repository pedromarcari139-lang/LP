# Choosing betting models that survive out-of-sample (live binary markets, LoL)

Scope: ~200 similar models, live binary markets at fixed game-clock timestamps (15:00, 20:00),
no closing line available. What to compute, in what order, and when to kill a model or an edge.

**Evidence behind this file**

* **Code.** `oos_toolkit.py` holds the reference implementations. `test_toolkit.py` checks them
  against known answers:
  * SPA against a literal re-implementation of Hansen (2005);
  * RC and MCS against the `arch` package;
  * the sizes of the tests, and the live e-process under checking after every game.
* **Simulation.** `simulate.py` builds a synthetic world. Its numbers (sections 1, 3–8) come from
  `results.json`.
* **Real data.** `lol_resolution_check.py` uses pro-LoL games from Oracle's Elixir 2022–24.
  * The §6 numbers come from `lol_resolution_check.out`.
  * The §9 counts come from the same data.
* **Test outputs.** A few figures come from the output of `test_toolkit.py` and are marked as such.
* **Literature.** Checked in primary sources where reachable; see References.
* **Reviews.** Three independent hostile reviews (code, text, and the new methods) found errors in
  earlier versions. Their findings are fixed here, and their scenarios are reproduced as EXP-11.

**Tags:**
* **[confirmado]**: source read, or computed here.
* **[provável]**: well supported, not fully verified.
* **[especulação]**: my estimate or proposal.
* **[não sei]**: unknown.

"Confirmed in simulation" means true **in that synthetic world**, not necessarily in yours.

---

## 1. Bottom line

1. **No rule guarantees out-of-sample performance [confirmado].**
   * PBO, DSR, SPA and bootstraps only *estimate or penalise* selection bias.
   * The only real out-of-sample data is data nobody looked at while choosing: a lockbox used once,
     then live bets.
2. **With 200 similar models, P&L cannot tell you whether any of them has an edge
   [confirmado in simulation].**
   * A real ${clone_roi} edge shared by 200 clones, with 10,000 games, was detected only part of the
     time: SPA ${spa10}, RC ${rc10}, DSR ${dsr10}.
   * The Model Confidence Set kept ${mcs_lo}–${mcs_hi} of the 200 models (medians) in every world.
3. **Test *information* instead: does a model know something the price doesn't?**
   * [confirmado in simulation] The family **information gate** (stage 3) detected that same edge
     ${gate2} of the time with 2,000 games and ${gate10} with 10,000. False positives in the
     no-information worlds were ${gsize_rng}.
   * [confirmado in simulation] Its validity depends on modelling **the price's own calibration**
     flexibly.
     * Under a nonlinear favourite–longshot bias, a de-vig that does not match the book's margin, or
       a drift between seasons, a logit-linear correction let through ${e11_lin_rng} false
       positives (EXP-11).
     * A spline curve, fitted per season, brought them to ${e11_flex_rng}.
   * [provável] The gate should decide whether *any* model goes forward. P&L tests become reports.
4. **Choosing the argmax is fine; believing its backtest is not [confirmado in simulation].**
   * The in-sample best overstated its ROI by ${wc_range} pp.
   * Choose with a rule fixed in advance (the largest information statistic among the models that
     pass the gate). Then *measure* its edge on the lockbox.
5. **Blending model and price (Benter 1994) shrinks overconfidence, but it is not free
   [confirmado in simulation; provável in general].**
   * With perfect public information and 10,000 training games, the walk-forward blend bet fewer
     games.
     * ROI per bet: raw model ${raw_roi0}, blend ${blend_roi0}.
     * Profit per game: raw model ${raw_ppg0}, blend ${blend_ppg0}.
   * When the model misread public information, the raw model lost ${raw_roi2} per bet
     (${raw_ppg2} per game).
     * The blend fitted on 2,000 games returned ${blend_roi2} pooled, and ${blend_lose2} of its fits
       lost money.
     * Fitted on 10,000 games, it cut the loss to ${blend_ppg2_10k} per game by betting on only
       ${blend_rate2_10k} of games. That limits damage; it does not create an edge.
   * With 2,000 training games the blend's ROI per bet was below the raw model's in every scenario.
   * Judge the blend by realised walk-forward P&L, never by its claimed EV.
6. **A live CLV exists, the markout, but it accelerates; it does not judge
   [confirmado: the math; provável: the magnitude].**
   * On real LoL prices (proxies) it has ≈ ${lol_band}–${lol_le5}× less variance than P&L.
   * It is blind to edges the market never learns.
   * Finding out which kind of edge you have takes as many bets as P&L does.
7. **Monitor live bets with an e-process on realised P&L [confirmado in simulation].**
   * A t-test checked after every bet "found" an edge in ${naive_fa} of break-even bettors. The
     e-process did so in ${ep_fa}.
   * The e-process also held:
     * ${ep_chase} when the 20:00 re-bet on the same game depended on the price;
     * ${ep_mixed} on a losing mix of bets;
     * ${ep_slip} with fill slippage.
   * The cost is power.
     * A real +5% edge on every bet was scaled up ${ep_power5} of the time within 5,000 bets.
     * A strategy losing 3% per bet was killed only ${ep_kill3} of the time.
     * Most paths stay undecided for thousands of bets, and that is the honest state.
8. **PBO judges the selection step, not the edge [confirmado: paper and simulation].**
   * Among near-clones it is ≈ 0.5 with or without a real edge: ${pbo_clone_noedge} vs
     ${pbo_clone_edge} at 10,000 games.
   * It is not comparable across different candidate sets.

---

## 2. What is publicly known about how betting quants validate

* **[não sei]** How syndicates validate internally. None of them publish it.
* **[confirmado]** Benter (1994), horse racing, did not bet his model raw.
  * He fitted a logit that **combined his model with the public's odds**.
  * He judged the model by how much it improved on the odds alone.
  * **[provável]** Reported pseudo-R²: public 0.1218, model 0.1245, combined 0.1396. The numbers
    are from a secondary source.
* **[confirmado]** Pinnacle and Buchdahl treat CLV (beating the closing price) as the best indicator
  of skill.
  * **[provável]** Buchdahl: CLV evidence reaches significance in far fewer bets than P&L, because
    its variance is much lower (secondary source).
* **[não sei]** Whether an established "in-play CLV" metric exists. The literature search found none.
* **[confirmado]** Accuracy is not profitability:
  * Wunderlich & Memmert (2020).
  * Hubáček, Šourek & Železný (2019): a model that is accurate but correlated with the bookmaker
    loses.
  * Hubáček & Šír (2023): a model worse than the market can still profit if decorrelated from it.
  * Walsh & Joshi (2024): selecting by calibration beat selecting by accuracy.
    * A 2025 corrigendum changed all their numbers (a feature-engineering bug).
    * [provável] According to a secondary summary, the direction still holds.
* **[confirmado]** From quant finance, for selection among many backtests:
  * White's Reality Check (2000).
  * Hansen's SPA (2005).
  * Romano–Wolf StepM (2005).
  * The Model Confidence Set (2011).
  * PSR/DSR (2012/2014).
  * PBO/CSCV (2017).
  * The t > 3 hurdle (Harvey–Liu–Zhu 2016).
* **[não sei]** How many firms actually apply these.

---

## 3. The funnel

Run the stages in order. **Thresholds are conventions I propose [especulação].** Fix them before
looking at any result, or they become one more thing you overfit.

| # | Stage | What you compute | Rule (proposed defaults) |
|---|---|---|---|
| 0 | **Freeze the protocol** | The list of every trial ever run (models, feature sets, EV thresholds, discarded variants), metrics, bet rule. The strata of the price-calibration model (e.g. season × checkpoint). The lockbox: the most recent ~20–30% of games, used **once**, with its pass criterion and that criterion's power at the lockbox size. | Changing anything after seeing the lockbox burns it: it becomes in-sample |
| 1 | **Leakage audit** (§9) | Odds↔state timestamp alignment; as-of assertion on every feature; canary column; synthetic-outcome control | Any failure: fix the **pipeline**, restart at stage 0 |
| 2 | **Walk-forward predictions** | Train strictly before test. All rows of a game (15:00 and 20:00) and all maps of a series in one fold. Store the price at every checkpoint of **every** game, not only the games you bet. | Every later stage uses only these predictions |
| 3 | **Family information gate** (all candidates ever tried, not survivors) | `information_gate(y, q, P, groups=series, strata=season×checkpoint)`: max-t over all models of the encompassing score, with the price's calibration curve (spline in logit q, per stratum) partialled out, multiplier bootstrap by game or series. In parallel, `price_calibration_test`: a miscalibrated price is an edge candidate on its own. | Both p ≥ 0.05: **stop, nothing goes forward.** Otherwise the models with adjusted p < 0.05 (family-wise control), and/or the recalibrated price, go to stage 4. |
| 4 | **Blend and realised P&L** (per survivor) | Fit the blend `a + b·logit q + c·(logit p − logit q)` on past games only. Keep a = 0, b = 1 unless the calibration test rejected, and shrink c when fitted on a few thousand games [especulação]. Bet with the blend at obtainable prices (delays, rejections, suspensions, limits, odds cap). Record walk-forward **realised** P&L per game. | Kill if the kill side of `pnl_eprocess` fires on the walk-forward bets, or if execution costs exceed the edge. **Claimed EV is not evidence**: an EV rule only bets when claimed EV clears its threshold. |
| 5 | **Reports, not vetoes** | RC (preferred) or SPA on P&L, DSR with N = all trials, PBO with probability of loss, MCS | Veto only if the family's P&L is significantly **negative** after execution |
| 6 | **Choose** | Among the gate's significant models: the largest information t, or the simplest one within noise of it (fixed in stage 0). Average only near-duplicates of one idea. | Never use the pick's backtest ROI as its expected ROI |
| 7 | **Robustness** | Sign of the information statistic across time blocks, leagues, patches, 15:00 vs 20:00, favourite/underdog, odds bands; parameter plateau; audit the prices of the top-1% contributing bets | Edge lives in one slice you did not pre-specify. Pick is an isolated spike. A top bet sits on a wrong price (Clegg & Cartlidge 2025). |
| 8 | **Lockbox** | One run of the one chosen model, with the stage-0 criterion. Use an information criterion (encompassing c, or the blend's log-loss gain): in EXP-5 a P&L criterion on 2,000 games detected a real edge ${roi2k} of the time, against ${enc2k} for the information test. | Fails: stop |
| 9 | **Live** | Small stakes. `pnl_eprocess` on realised profit per game (bets on one game summed, at the odds obtained), checked as often as you like. Log every attempt: requested vs obtained odds, rejections, later prices for markouts. | Scale up when the "up" process reaches 1/α = 20. Kill when the "down" process reaches 20. Markouts are an early warning, **never** a kill rule. |

### Metrics and scorers, and what each one answers

| Question | Metric / test | Role |
|---|---|---|
| Is the probability good in absolute terms? | Log loss, Brier (= RPS for 2 outcomes), calibration curve | Diagnostic. Accuracy/AUC ignore calibration; don't select betting models on them [confirmado]. |
| Is the price itself calibrated? | `price_calibration_test`: flexible curve, clustered Wald | Stage 3. A miscalibrated price is an edge candidate. |
| Does a model know something the price doesn't? | Encompassing score for c, with a flexible price curve; family max-t gate | **Stage 3 gate** |
| Same question, stand-alone | Paired Δlog loss vs the de-vigged price, clustered | Diagnostic. A model worse than the market overall can still add information (Hubáček & Šír 2023) [confirmado]. |
| Is the edge estimate calibrated? | Realised ROI by bucket of predicted EV | Stage 4 [especulação: my rule] |
| Does it make money after margin and execution? | Walk-forward profit per game, ROI per bet, clustered t, drawdown, at obtainable prices | Stage 4, realised. Needs thousands of bets (§8). |
| Skill or luck, given how much I tried? | Information gate over all trials; RC/SPA, DSR (N = all trials), PBO with P(loss), MCS | Gate (stage 3); reports (stage 5) |
| Early evidence without waiting for outcomes | Markouts at t+Δ (§6) | Accelerator and alarm, **never** a veto |
| Is it robust? | Sign of the information statistic across slices; parameter plateau; audit of the top bets | Stage 7 |
| Live: scale up or kill? | `pnl_eprocess` on realised P&L | Stage 9 |

---

## 4. Why information and not P&L: power, sizes and the family gate

**Power of three tests for the SAME real edge [confirmado in simulation].** One model, true ROI
${clone_roi}, one-sided 5%, 1,000 replications (EXP-5).

${tbl_power}

* With no edge, the encompassing test for c rejected ${enc_lo}–${enc_hi} of the time and the joint
  test ${joint_lo}–${joint_hi} (nominal 5%).
* The encompassing test uses **every game**, not just the games you bet, and the full probability,
  not just win/lose.
* It proves *information*, not profit after margin and execution. That is why stage 4 exists.
* The single-model `encompassing_test` corrects the price only logit-linearly (a + b·logit q). Use it
  as a diagnostic.
  * c > 0 there means information beyond a logit-linear recalibration of the price.
  * A purely logit-linear bias of the price shows up in b, not in c. In `test_toolkit.py`:
    b = 1.40, c = 0.03 (p = 0.28), joint p ≈ 0.
  * A nonlinear bias or a wrong de-vig can create c > 0 without any information. That is why the
    gate models the price's calibration flexibly (below).

**Family gate on information vs family gate on P&L (EXP-8) [confirmado in simulation].**
* Same replications for both gates.
* "Declared significant" = any model *without* information passes the gate's family-wise control.
* Funnel column: a model goes forward only if the gate rejects, and then it is the largest information
  t; otherwise nothing is bet.
* Sharpe column: what you get by always betting the in-sample Sharpe argmax.

${tbl_gates}

* False positives of the information gate in the no-information worlds:
  * ${gsize_rng2} at 2,000 games;
  * ${gsize_rng10} at 10,000 games (100–200 replications; SE 1.5–2.7 pp).
  * Roughly nominal, possibly slightly liberal at 10k.
* In the no-information worlds, every rejection is a wrong "go"; those are the false positives above.
* In the world where only 20 of 200 models carry information, the gate almost never let a
  no-information model through: ≤ ${go_wrong_max} of replications.
  * The in-sample Sharpe argmax always bets, and picked a model with no information
    ${sharpe_wrong_2k} of the time at 2,000 games (grid, 20 of 200 real).
* It finds real information that P&L cannot: ${gate2} vs ${spa2} (clones, 2,000 games).
* The gate proves *some* model carries information. Whether that information survives the margin
  and execution is stage 4.

**The gate's size depends on how the price's own calibration is modelled (EXP-11)
[confirmado in simulation].**
* Scenarios designed by the third hostile review. No model has private information in any of them,
  so every rejection is a false positive (nominal 5%).
* "Logit-linear" corrects the price with a + b·logit q.
* "Spline" uses a natural cubic spline in logit q (the default).
* "+ strata" fits one curve per season.

${tbl_gate_robust}

* The spline alone does **not** fix drift between seasons; the per-season strata do.
* The synthetic-outcome control (§9.7) **cannot** see this problem: it makes the price calibrated by
  construction. The reviewer measured it at 5.0% in a world where the logit-linear gate rejected 36%.
* So the price's calibration curve must be flexible, and stratified by season and checkpoint.
* If your prices drift within a season (patches), stratify more finely. That was **not** tested.

---

## 5. Blend the model with the price, and judge the blend by realised P&L (EXP-7)

In the rest of the simulation every model reads the market's public information perfectly, which
flatters the models (a hostile review pointed this out). Here the informative model also misreads
public information; its error variance is in the first column.
* The blend is fitted on the previous 2,000 or 10,000 games, then bets 100,000 independent later
  games.
* 40 fits per row.
* ROI per bet is pooled over all fits.
* "Losing fits" = fits whose blend lost money.

${tbl_blend}

* **A model can carry real information and still lose money on its own probabilities.**
  * The encompassing test still detects it, but the model's errors on public information cost more
    than its private edge earns.
* **With 10,000 training games the blend shrinks the model toward the price.** It bets fewer games at
  a higher ROI per bet, but with flat stakes that did not mean more profit per game. Compare the
  profit-per-game columns.
* **With 2,000 training games its ROI per bet was below the raw model's in every row.**
* **The blend is itself a fitted model.**
  * On 2,000 training games its bet rate swings widely between fits, and a large share of fits lose.
  * Fit it on as many games as you can.
  * Keep a = 0 and b = 1 unless the price calibration test rejects.
  * Judge it by realised walk-forward P&L (stage 4), never by its claimed EV.
* Benter (1994) combined his model with the public's odds. The logic is the same; the magnitudes here
  are from the simulation.

---

## 6. A live substitute for CLV: markouts

Your premise was "I can't use CLV because I bet live". Half right. There is no *closing* line for a
15:00 bet, but CLV is one case of a general idea you **can** use: re-price the bet later.

**Definition.** Back side s at odds `o` at 15:00. Take the market's de-vigged probability of s at a
later moment t+Δ (+1 min, +5 min, 20:00). Then

```
markout = o · q_s(t+Δ) − 1        (profit = o · y_s − 1)
```

**Why it works [confirmado: the math, under its assumptions].** Suppose q(t+Δ) is the exact
conditional probability of a market that by then knows everything **you** knew at bet time.
* `E[profit − markout | your info] = 0`: the markout is **unbiased** for your EV.
* `Var(profit) = Var(markout) + E[o² · q(t+Δ)·(1 − q(t+Δ))]`: the markout has **lower variance**.

**How much faster, on real LoL data [provável for the magnitude; proxy prices].**
`lol_resolution_check.py` fits win probability at 15:00 and at 20:00 on 2022, and tests on
${lol_test} games from 2023–24.
* ${lol_short} of those games ended before 20:00. Their "20:00 price" is the result. Dropping them
  would use end-of-game information.
* Accuracy: ${lol_acc15} at 15:00, and ${lol_acc20} at 20:00 on the games still live.
* Mean predicted p at 15:00 is ${lol_meanp}, against a Blue win rate of ${lol_wr}.
* Variance: the 15:00→20:00 price move has E[(q20−q15)²] = ${lol_dq}, against
  E[q15(1−q15)] = ${lol_vout} left in the outcome.
* Var(profit)/Var(markout at 20:00) for a 15:00 bet at fair odds:

  | Fair odds | Ratio |
  |---|---|
  | 1.5–3.0 | **${lol_band}×** |
  | ≤ 5 | **${lol_le5}×** |
  | all sides | ${lol_all}×, driven by longshots |

  Sides priced q < 0.1 are ${lol_long_share} of sides but carry ${lol_long_var} of the profit
  variance.
* **The proxies are not perfect martingales.**
  * The identity E[q15(1−q15)] = E[dq²] + E[(y−q20)²] misses by ${lol_gap} (SE ${lol_gap_se}).
  * The 15:00 proxy is slightly under-confident: Brier ${lol_brier} vs ${lol_vout}.
  * Read ≈ 5–6× as an order of magnitude.

**When it fails (EXP-6) [confirmado in simulation].** The edge is built two ways with the same
information content.

${tbl_mk}

* **Persistent edge** (e.g. better team-strength priors that the live book never learns): the
  markout reads ≈ −margin on a real edge. The profit column is noise around the true ≈ ${clone_roi}.
  Killing on markouts would kill a winner.
* **The residual test cannot tell the two cases apart at LoL sample sizes.** The residual is
  `o·(y − q_later)`, the part of profit the markout does not see.
  * Its variance is Var(profit) − Var(markout), ≈ ${resid_share} of the profit variance here.
  * So it needs as many bets as P&L. The small SEs in the table come from a million games.
* **Use markouts:**
  * as an accelerator, when you have a structural reason to believe your edge is catch-up (you read
    the game state before the price does);
  * as an early warning.
  * **Never as a kill rule.**
* **Practicalities.**
  * Log both sides' prices at several horizons after every bet *and* every non-bet: a markout curve.
  * De-vig each snapshot. If the later price is 1.10/7.00, the choice of de-vig method moves the
    underdog's probability by 1.89 pp (§9). That moves the markout of a bet taken at 3.50 by
    3.5 × 1.89 ≈ 6.6 pp.
  * Compare markouts on bet vs non-bet games: on a thin book your own bet may move the price.
  * A game that ends before the horizon uses the result.
  * A market suspended at the horizon uses the next open price.
* **Clustering.** The rule re-bet the same game at 20:00 in ${pe_both} of cases, with a within-game
  return correlation of ${pe_corr}. The game-clustered SE was ${pe_se}× the naive one.

---

## 7. Choosing among 200 near-identical models: what each tool tells you (EXP-2/3/4)

**Setup.**
* 200 fixed rules bet flat stakes at 15:00, never at odds above 5.0, against a calibrated market.
* Four kinds of world:
  * **clones**: 200 copies of one rule;
  * **grid**: 50 signal weights × 4 EV thresholds;
  * **zero EV** (margin 0: no-information rules have EV exactly 0, so rejections are the tests'
    real false-positive rates);
  * **−margin** (margin 5%).
* The informative rules have true ROI ${inf_lo} to ${inf_hi}, and the clones ${clone_roi}.
* The "pick" is the in-sample best per-game Sharpe, as in the PBO paper.
* The true ROI of a rule is exact without information; with information it comes from a
  10-million-game table.

**Winner's curse (medians over 200 replications).**

${tbl_wc}

**What each tool said.**
* 200 replications at 2,000 games; 100 at 10,000.
* For tests about the pick, "(x% false)" is the share of rejections where the pick had no
  information.
* MCS uses the range statistic at 90%.

${tbl_tools}

1. **False positives.** The honest test is at the zero-EV boundary; the −margin worlds are easier.

   | Test | Zero-EV worlds | −margin worlds |
   |---|---|---|
   | SPA | ${z_spa} | ${m_spa} |
   | RC | ${z_rc} | ${m_rc} |
   | DSR, N = M | ${z_dsr} | ${m_dsr} |
   | DSR, N̂ | ${z_dsrhat} | ${m_dsrhat} |
   | Naive test of the pick | **${z_naive}** | ${m_naive} |

   * Each cell has 100–200 replications, so its sampling SE is 1.5–2.2 pp.
   * The corrected tests are therefore roughly at their nominal 5%, not clearly below it.
   * The naive test is far above it.

2. **DSR: use N = all trials together with the cross-sectional variance of their Sharpe ratios.**
   * That variance already shrinks when trials are correlated.
   * Also shrinking N with the paper's N̂ = ρ̄ + (1−ρ̄)M (here N̂ ≈ ${nhat_lo}–${nhat_hi}) counts the
     correlation twice.
   * EXP-10 checks this directly: 200 trials, true SR 0, 1,000 returns each.

     ${tbl_dsr}
   * DSR is very conservative for independent trials.
3. **Studentized SPA on P&L is somewhat liberal with skewed bets, and a minimum-bets filter only
   partly fixes it.**
   * EXP-10: zero EV, 200 candidates at fixed odds 1.4 / 1.9 / 2.8 / 5.0, 2,000 games.

     ${tbl_spa_rare}
   * Prefer the RC as the P&L family report, or read SPA p-values as slightly optimistic.
   * P&L tests are reports here, not gates.
   * `arch` 8.0's `SPA` does not studentize despite its `studentize` flag (source read). It is a
     Reality Check.
4. **PBO judges the selection step, not the edge.**
   * Among clones it is ≈ 0.5 with or without an edge. The paper says so itself (References).
   * It was ${pbo_grid_noedge} where every model loses. There Sharpe selection consistently picks the
     rules that bet least, because they lose least.
   * It was ${pbo_g20} where 20 of 200 rules are genuinely better, the one world where it behaved as
     intended.
   * The paper's rule "reject if PBO > 0.05" would have rejected the real edges here.
   * CSCV's probability of loss separated the worlds better (${ploss_edge} with an edge vs
     ${ploss_noedge} without), but it is not a formal test.
   * CSCV's "performance degradation" slope is −1 by construction when the same model is picked in
     every split. In-sample and out-of-sample are complementary halves, so OOS = 2·full − IS.
5. **The MCS kept ${mcs_lo}–${mcs_hi} of 200 models.** The data cannot rank them by P&L.
   * Averaging the MCS members is **not** a fix. Where only 20 of 200 carry information, the average
     is ~85–90% noise and loses (as the hostile review measured).
   * Choose by the pre-registered information rule (stage 6) and measure on the lockbox.
6. **The pick bets rarely and got lucky.**
   * In-sample Sharpe favours rules with few bets.
   * With probit/logit-symmetric errors, spurious EVs concentrate on underdogs, where per-bet variance
     is highest.
   * That is why the simulation caps odds at 5.0. Without the cap, bets above 5.0 were ~25–31% of bets
     but ~70% of the P&L variance (independent review of the first run).

---

## 8. Sample size and live monitoring

**Independent flat bets needed [confirmado: arithmetic, and checked by simulation in
`test_toolkit.py`].**

${tbl_ss}

* Two bets on the same game (15:00 and 20:00) are **not** two independent bets: count games, or
  cluster.
* A pro LoL year has ~10–12.5k games **in total** in Oracle's Elixir:
  * 12,549 games in 55 leagues (2022);
  * 11,010 in 51 (2023);
  * 9,804 in 51 (2024, file ends 2024-12-08).
* The games you can bet live at 15:00 are fewer.
* **P&L alone cannot validate the best of 200 candidates:**
  * a 3% edge at odds 1.9 needs ${ss_t2} bets for t = 2;
  * it needs ${ss_bonf} for 80% power after a Bonferroni correction.

**Live monitoring on realised P&L, checked after every game (EXP-9) [confirmado in simulation].**
* 2,000 paths of 5,000 bets at odds 1.5–3.5, flat stakes.
* `pnl_eprocess` scales up when "up" = mean over λ of ∏(1 + λr) reaches 20, and kills when "down" =
  mean over λ of ∏(1 − λr) reaches 20. λ ∈ {0.005, …, 0.08}, and r is the profit of a game at the
  odds obtained.

${tbl_live}

* **Why it holds (Ville's inequality; testing by betting, Shafer 2021).**
  * If no bet has positive expected value at the odds obtained, "up" is a nonnegative
    supermartingale. So it **ever** reaches 20 with probability ≤ 5%, however often you look.
  * "Down" has the same guarantee when no bet has negative expected value.
* **Bets on one game at 15:00 and 20:00: sum their profits into one step.**
  * Expectations add, so the guarantee survives even when the 20:00 bet depends on the 20:00 state
    (the "chasing" row).
  * An earlier version of this toolkit used a likelihood-ratio SPRT on claimed probabilities. It
    broke when it multiplied the factors of same-game bets, and again when it averaged them with
    weights chosen at 20:00 (third hostile review).
* **It uses realised profit at the odds you actually got, not your claimed probabilities.**
  * Claims are not used at all, so overconfident claims cannot inflate it.
  * Slippage lowers the realised returns it sees.
  * The "mixed" and "slippage" rows fall outside the formal guarantee, because some bets there have
    positive EV. It still did not scale them up.
* **The price is power.**
  * A real +5% edge was scaled up in ${ep_power5} of paths within 5,000 bets.
  * A −3% strategy was killed in only ${ep_kill3}.
  * Break-even bettors stayed undecided in ${ep_undec0} of paths.
  * Live evidence takes thousands of bets; until then "undecided" is the honest state. Keep stakes
    small.
* The naive rule ("scale up when t > 1.645") fired for ${naive_fa} of break-even bettors when checked
  after every bet. Armitage et al. (1969) describe this optional-stopping problem.
* **The drop-top-1% rule kills real edges.**
  * Shares of real +3% edges (2,000 bets) whose ROI turns ≤ 0 after dropping the best 1% of bets:
    ${tbl_trim}.
  * Audit the prices of the top bets instead. If you trim, compare against the same trim applied to
    bootstrap or null samples.

---

## 9. LoL-live leakage checklist (run before any metric)

1. **Odds ↔ state alignment (the most important live leak) [provável].**
   * The price you score at "15:00" must be the price on offer while your 15:00 features were known.
   * Game clock, data feed and broadcast are all out of sync.
   * Test: re-run with the odds snapshot deliberately taken 10–30 s later. If the edge collapses, it
     was a timing artefact.
2. **Whole-game columns used as 15:00 features [confirmado].**
   * Oracle's Elixir mixes time-stamped columns (`golddiffat15`, `killsat15`, …) with **whole-game**
     columns: `towers` (mean 6.06 in 2023), `dragons`, `firsttower`, `firstdragon`, `firstherald`,
     `firstbaron`, `firsttothreetowers`, `inhibitors`, `kills`.
   * Rebuild features from the event timeline at the timestamp.
3. **Train/serve skew [provável].**
   * `golddiffat15` in a post-game dataset is not necessarily what you can observe live, from the
     overlay or your feed, at 15:00. The definition and the timing can differ.
   * Train on what you will actually see.
4. **gameid and series splits [confirmado as a mechanism].**
   * The 15:00 and 20:00 rows of a game share one outcome.
   * Maps of a series share teams, day and form.
   * Keep them in one fold, and cluster by them.
5. **Pre-game features from the future [provável].**
   * Ratings, form, champion and patch statistics must use only earlier games.
   * A champion win rate computed over the whole patch is a leak.
6. **As-of assertions and a canary [especulação: my design].**
   * Every feature carries the timestamp of its newest input. Assert that it is ≤ the decision time.
   * Inject a known future column (a canary) and check that the audit flags it.
7. **Synthetic-outcome control [especulação: my design; the logic is exact].**
   * Replace outcomes with `y* ~ Bernoulli(de-vigged price)` and rerun the whole pipeline.
   * Nobody can beat the price on y*, so an edge on y* means label leakage or a bug.
   * It **cannot** catch features that encode the real outcome (items 2–3). Do not permute real
     outcomes instead: that breaks the odds↔outcome link.
   * It **cannot** check the information gate's calibration model either, because it makes the price
     calibrated by construction (§4, EXP-11).
8. **Survivorship [provável].**
   * Markets are suspended around fights and objectives.
   * Dropping unpriced games or timestamps biases the sample. Log suspensions.
9. **Partial data [confirmado].**
   * 15.1% / 15.8% / 14.2% of Oracle's Elixir games (2022/23/24) are flagged partial and have no
     at-10/15/20 stats.
   * 91–93% of those are LPL or LDL.
10. **Execution and adverse selection [confirmado as a mechanism].**
    * Live bets are delayed and can be rejected when the game state changes:
      * Pinnacle holds live bets and rejects them after material changes;
      * Betfair's in-play delay is 1–12 s.
    * The book's official data feed may run ahead of what you see, so the bets it accepts can be
      adversely selected.
    * Backtests at snapshot prices are an upper bound. Log every attempt.
11. **Data errors [confirmado].**
    * Clegg & Cartlidge (2025): most of a published strategy's out-of-sample profit came from one bet
      at erroneous long odds.
    * Check for stale prices, impossible overrounds and duplicated games.
12. **The de-vig method moves your benchmark [confirmado: arithmetic].**
    * With two outcomes, the additive and Shin methods coincide (`implied` R package docs); the
      multiplicative method differs.
    * How much the favourite's fair probability moves between the two methods:

      | Odds | Change |
      |---|---|
      | 1.87 / 1.95 | 0.05 pp |
      | 1.30 / 3.50 | 1.26 pp |
      | 1.10 / 7.00 | 1.89 pp |

    * Choose the method by the walk-forward log loss of the de-vigged price on your data, and fix it
      in stage 0 [provável].

---

## 10. Reproduce and use

```
pip install numpy scipy pandas statsmodels arch
python test_toolkit.py                                   # ~1 min, all checks must PASS
OMP_NUM_THREADS=1 python simulate.py --out results.json  # ${runtime} on 4 cores
python lol_resolution_check.py /path/to/oracles_elixir_csvs
python make_readme.py                                    # rebuilds README.md from README.tpl + results
```

**On your data**
* One row per game, sorted by time. The 15:00 and 20:00 bets are **summed** into the game row
  (`aggregate_by_game`).
* Walk-forward predictions of every candidate for every game (`P`), the de-vigged price (`q`), the
  outcome (`y`), and the P&L matrix (`R`).
* Then, in funnel order:

| Stage | Function |
|---|---|
| 3 | `information_gate(y, q, P, groups=series_id, strata=season_x_checkpoint)` and `price_calibration_test(y, q, groups=series_id)` |
| 4 | `fit_blend` / `blend_prob` (walk-forward), then `pnl_eprocess` on the walk-forward realised P&L per game |
| 5 | `spa_vs_no_bet(R)` (use `p_rc`), `deflated_sharpe`, `pbo_cscv(R)`, `model_confidence_set(-R)`, as reports |
| 9 | `pnl_eprocess(aggregate_by_game(profit, gameid))` live |

---

## 11. What was NOT verified / limitations

* **The simulation is not LoL.**
  * Every magnitude in it is an assumption: margin, edge size, how much of the game is decided
    between 15:00 and 20:00, correlation between models, the odds cap.
  * It shows mechanisms with numbers. It does not predict your results.
* **Only selection among fixed rules is simulated.**
  * Real pipelines also overfit through fitting and feature search.
  * Reality is likely worse [provável].
* **No non-stationarity.**
  * Patches, meta, roster changes and bookmaker model changes are absent.
  * The bootstraps here resample games iid. If games on the same day or patch are dependent, use
    blocks: `mean_block > 1` with games sorted by time, or `groups=`.
* **The information gate is new in this form** (my combination of standard parts).
  * Checked only in this simulation, in `test_toolkit.py` and in the third review's scenarios.
  * Its size relies on the flexible, stratified price-calibration curve capturing how the price is
    biased.
  * Drift inside a season (patches) and other shapes of bias were not tested.
  * The synthetic-outcome control cannot validate this (§4).
* **The prescribed pipeline was not simulated end to end.**
  * EXP-8 picks raw rules.
  * The blend (stage 4) and the live e-process (stage 9) were tested separately.
* **The live e-process gives up power for robustness.**
  * A test that trusts your claimed probabilities decides faster, but was fooled by overconfident
    claims, slippage and same-game bets (third review).
* **The live-market magnitudes use proxy prices, not real odds** (§6).
* **Literature access.**
  * Several sources were checked only through official abstracts or secondary summaries (tagged
    below).
  * The Walsh & Joshi corrigendum's corrected numbers were not checked against the corrigendum
    itself.
* **What professional syndicates do internally is unknown [não sei].**
* **"MSR" in the question is ambiguous.** It is not an acronym used in the PBO or DSR papers.
  Plausible meanings, and where each lives in this funnel:
  * SR₀, the expected maximum Sharpe ratio: the benchmark inside DSR.
  * The Minimum Track Record Length (PSR paper).
  * A "modified Sharpe ratio" (modified VaR).
  * The maximum-Sharpe portfolio.

---

## References (how each was checked)

How each reference was checked:
* **VP** = primary text read.
* **VP-idx** = official abstract, record or journal page, seen via search.
* **VS** = reliable secondary source.

The research agents could not open SSRN, arXiv or journal sites directly. They read author PDFs
mirrored on GitHub and search-index records.

**Backtest overfitting**
* Bailey, Borwein, López de Prado & Zhu, "The Probability of Backtest Overfitting",
  *J. Computational Finance* 20(4), 2017. **VP** (SSRN version).
  * Algorithm 2.3.
  * Four diagnostics.
  * "Reject models for which PBO is estimated to be greater than 0.05."
  * Verbatim: "it is entirely possible that all the N strategies have high but similar Sharpe ratios.
    Since none of the strategies is clearly better than the rest, PBO will be high."
  * C(16,8) = 12,870; the paper's "12,780" is a typo.
* Bailey & López de Prado, "The Deflated Sharpe Ratio", *J. Portfolio Management* 40(5), 2014. **VP**.
  * Eq. 2 (raw kurtosis).
  * Eq. 9, N̂ = ρ̂ + (1 − ρ̂)M, read on the page image.
* Bailey & López de Prado, "The Sharpe Ratio Efficient Frontier", *J. Risk* 15(2), 2012. **VP**.
  PSR, MinTRL.
* Bailey, Borwein, López de Prado & Zhu, "Pseudo-Mathematics and Financial Charlatanism",
  *Notices AMS* 61(5), 2014. **VP**. MinBTL.
* López de Prado & Bailey, "The False Strategy Theorem", *Amer. Math. Monthly* 128(9), 2021.
  **VP-idx**.
* López de Prado, *J. Financial Data Science* 1(1), 2019. **VP-idx**.
* López de Prado & Lewis, *Quantitative Finance* 19(9), 2019. **VP-idx**.

**Data snooping and multiple testing**
* White, *Econometrica* 68(5), 2000. **VP-idx**.
* Hansen, *JBES* 23(4), 2005. **VP-idx**.
* Romano & Wolf, *Econometrica* 73(4), 2005. **VP-idx**. Asymptotic FWER control.
* Hansen, Lunde & Nason, *Econometrica* 79(2), 2011. **VP-idx**.
* Politis & Romano, *JASA* 89(428), 1994. **VP-idx**.
* Westfall & Young, *Resampling-Based Multiple Testing*, Wiley 1993. **VS**. Single-step max-T.
* Chernozhukov, Chetverikov & Kato, *Ann. Statist.* 41(6), 2013. **VP-idx**. Multiplier bootstrap for
  maxima.
* Harvey, Liu & Zhu, *RFS* 29(1), 2016. **VP-idx**.
* Harvey & Liu, *JPM* 42(1), 2015. **VS**.

**Forecast comparison and orthogonal scores**
* Diebold & Mariano, *JBES* 13(3), 1995. **VP-idx**.
* Giacomini & White, *Econometrica* 74(6), 2006. **VP-idx**.
* Fair & Shiller, *AER* 80(3), 1990. **VP-idx**.
* Clements & Harvey, *J. Applied Econometrics* 25(6), 2010. **VP-idx**.
* Neyman, "Optimal asymptotic tests of composite statistical hypotheses", 1959. **VS**.
* Chernozhukov et al., "Double/debiased machine learning", *Econometrics Journal* 21(1), 2018.
  **VP-idx**.

**Sequential testing**
* Ville, *Étude critique de la notion de collectif*, 1939. **VS**.
* Wald, *Ann. Math. Statist.* 16(2), 1945. **VP-idx**.
* Shafer, "Testing by betting", *JRSS-A* 184(2), 2021. **VP-idx**.
* Ramdas, Grünwald, Vovk & Shafer, *Statistical Science* 38(4), 2023. **VP-idx**.
* Vovk & Wang, *Ann. Statist.* 49(3), 2021. **VP-idx**.
* Armitage, McPherson & Rowe, *JRSS-A* 132(2), 1969. **VP-idx**.

**Betting**
* Benter, 1994. **VP** for the method, **VS** for the numbers.
* Hubáček, Šourek & Železný, *IJF* 35(2), 2019. **VP**.
* Hubáček & Šír, *IJF* 39(2), 2023. **VP**.
* Walsh & Joshi, *MLWA* 16, 2024, with corrigendum in *MLWA* 19, 2025. **VP** that the corrigendum
  exists.
* Wunderlich & Memmert, *IJF* 36(2), 2020. **VP**.
* Kaunitz, Zhong & Kreiner, arXiv 1710.02824, 2017. **VP**.
* Clegg & Cartlidge, *IJF* 41(2), 2025. **VP**.

**Converting odds and staking**
* Štrumbelj, *IJF* 30(4), 2014. **VP**.
* Clarke, Kovalchik & Ingram, *AJSS* 5(6), 2017. **VP**.
* The `implied` R package documentation. **VP**.
* Baker & McHale, *Decision Analysis* 10(3), 2013. **VP**.

**In-play markets and execution**
* Pinnacle and Buchdahl on CLV. **VP**.
* Pinnacle's rules for accepting live bets, and Betfair's in-play delays. **VP**.
* Databento's definition of "markout". **VS**.
