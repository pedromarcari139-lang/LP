# Choosing betting models that survive out-of-sample (live binary markets, LoL)

Scope: ~200 similar models, live binary markets at fixed game-clock timestamps (15:00, 20:00),
no closing line available. What to compute, in what order, and when to kill a model or an edge.

**Evidence behind this file**

* **Code.** `oos_toolkit.py` holds the reference implementations. `test_toolkit.py` checks them
  against known answers:
  * SPA against a literal re-implementation of Hansen (2005);
  * RC and MCS against the `arch` package;
  * the sizes of the tests, and the SPRT under checking after every bet.
* **Simulation.** `simulate.py` builds a synthetic world. Its numbers (sections 1, 3–8) come from
  `results.json`.
* **Real data.** `lol_resolution_check.py` uses pro-LoL games from Oracle's Elixir 2022–24.
  * The §6 numbers come from `lol_resolution_check.out`.
  * The §9 counts come from the same data.
* **Test outputs.** A few figures come from the output of `test_toolkit.py` and are marked as such.
* **Literature.** Checked in primary sources where reachable; see References.
* **Reviews.** Two independent hostile reviews (one of the code, one of this text) found errors in
  an earlier version. Their findings are fixed here.

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
   * A real +2.8% edge shared by 200 clones, with 10,000 games, was detected only part of the
     time: SPA 29%, RC 26%, DSR 25%.
   * The Model Confidence Set kept 183–200 of the 200 models (medians) in every world.
3. **Test *information* instead: does a model know something the price doesn't?**
   * [confirmado in simulation] The family **information gate** (§3, stage 3) detected that same edge:
     * 54% of the time with 2,000 games;
     * 99% with 10,000 games;
     * with 2.5%–8.0% false positives in the no-information worlds (nominal 5%).
   * [provável] It should decide whether *any* model goes forward. P&L tests become reports.
4. **Choosing the argmax is fine; believing its backtest is not [confirmado in simulation].**
   * The in-sample best overstated its ROI by +2 to +44 pp.
   * Choose with a rule fixed in advance (the largest information statistic among the models that
     pass the gate). Then *measure* its edge on the lockbox.
5. **Bet the blend of model and price, not the raw model [confirmado in simulation; Benter 1994].**
   * Betting the walk-forward blend instead of the raw model raised ROI per bet:
     * from +2.9% to +4.7% (± 0.4) when the model read public information perfectly;
     * from -2.6% to +0.4% (± 2.0) when it misread it.
   * Mostly by betting less. In the second case the blend's profit is uncertain, but it avoided the
     loss.
6. **A live CLV exists, the markout, but it accelerates; it does not judge
   [confirmado: the math; provável: the magnitude].**
   * On real LoL prices (proxies) it has ≈ 5.5–5.9× less variance than P&L.
   * It is blind to edges the market never learns.
   * Finding out which kind of edge you have takes as many bets as P&L does.
7. **Monitor live bets with a test that survives being checked after every bet
   [confirmado in simulation].**
   * A t-test checked after every bet "found" an edge in 40% of break-even bettors.
   * Wald's SPRT between break-even and your claimed probabilities, valid by Ville's inequality,
     did so in 4.0%.
8. **PBO judges the selection step, not the edge [confirmado: paper and simulation].**
   * Among near-clones it is ≈ 0.5 with or without a real edge.
   * It was lower in a world where everything loses (0.35) than in one with a real edge
     (0.52).

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
| 0 | **Freeze the protocol** | The list of every trial ever run (models, feature sets, EV thresholds, discarded variants), metrics, bet rule. The lockbox: the most recent ~20–30% of games, used **once**. Its pass criterion and that criterion's power at the lockbox size. | Changing anything after seeing the lockbox burns it: it becomes in-sample |
| 1 | **Leakage audit** (§9) | Odds↔state timestamp alignment; as-of assertion on every feature; canary column; synthetic-outcome control | Any failure: fix the **pipeline**, restart at stage 0 |
| 2 | **Walk-forward predictions** | Train strictly before test. All rows of a game (15:00 and 20:00) and all maps of a series in one fold. Store the price at every checkpoint of **every** game, not only the games you bet. | Every later stage uses only these predictions |
| 3 | **Family information gate** (all candidates, not survivors) | `information_gate`: max-t over all models of the encompassing score for c, with the price's own bias (a, b) partialled out, multiplier bootstrap by game or series. Also fit (a, b): a miscalibrated price is an edge on its own. | Family p ≥ 0.05: **stop, nothing goes forward.** Otherwise keep only models significant with family-wise error control. |
| 4 | **Blend and executable EV** (per survivor) | Fit the blend `a + b·logit q + c·(logit p − logit q)` on past games only; compute EVs with the **blend**; apply delays, rejections, suspensions, limits and an odds cap; profit per game, clustered t, drawdown, fractional Kelly on the blend | Kill if the blend's average EV at the prices you can actually get is ≤ 0. Realised P&L is reported with its CI; at these sizes it cannot confirm an edge. |
| 5 | **Reports, not vetoes** | SPA/RC on P&L, DSR with N = all trials, PBO and probability of loss, MCS | Veto only if the family's P&L is significantly **negative** after execution |
| 6 | **Choose** | Among the gate's significant models: the largest information t, or the simplest one within noise of it (fixed in stage 0). Average only near-duplicates of one idea. | Never use the pick's backtest ROI as its expected ROI |
| 7 | **Robustness** | Sign of the information statistic across time blocks, leagues, patches, 15:00 vs 20:00, favourite/underdog, odds bands; parameter plateau; audit the prices of the top-1% contributing bets | Edge lives in one slice you did not pre-specify. Pick is an isolated spike. A top bet sits on a wrong price (Clegg & Cartlidge 2025). |
| 8 | **Lockbox** | One run of the one chosen model, with the stage-0 criterion. Use an information criterion (encompassing c, or the blend's log-loss gain): in EXP-5 a P&L criterion on 2,000 games detected a real edge 11% of the time, against 56% for the information test. | Fails: stop |
| 9 | **Live** | Small stakes. SPRT between break-even (1/odds) and the blend's claimed probability, checked as often as you like, with **one averaged factor per game** (`groups=gameid`). Log every attempt: requested vs obtained odds, rejections, later prices for markouts. | Scale up when the SPRT crosses 1/α = 20. Kill when it crosses β = 0.05. Markouts are an early warning, **never** a kill rule. |

---

## 4. Why information and not P&L: power, sizes and the family gate

**Power of three tests for the SAME real edge [confirmado in simulation].** One model, true ROI
+2.8%, one-sided 5%, 1,000 replications (EXP-5).

| Games | ROI t-test | Δlog loss vs market | Encompassing, c > 0 | Encompassing, joint |
|---|---|---|---|---|
| 500 | 6.3% | 8.5% | 24.0% | 11.1% |
| 1,000 | 8.9% | 8.9% | 37.9% | 15.8% |
| 2,000 | 10.7% | 11.0% | 56.3% | 28.6% |
| 5,000 | 15.7% | 18.7% | 87.9% | 66.8% |
| 10,000 | 26.1% | 31.2% | 99.6% | 95.1% |

* With no edge, the encompassing test for c rejected 4.0%–6.6% of the time and the joint
  test 4.6%–5.7% (nominal 5%).
* The encompassing test uses **every game**, not just the games you bet, and the full probability,
  not just win/lose.
* It proves *information*, not profit after margin and execution. That is why stage 4 exists.
* Encompassing coefficients:
  * c > 0 means information beyond any logit-linear recalibration of the price.
  * An edge that is only a recalibration of the price (e.g. a favourite–longshot bias) shows up in b,
    **not** in c. In `test_toolkit.py`: b = 1.40, c = 0.03 (p = 0.28), joint p ≈ 0.
  * A wrong de-vig can also create c > 0, so treat c > 0 as a signal to investigate, not as proof.

**Family gate on information vs family gate on P&L (EXP-8) [confirmado in simulation].**
* Same replications for both gates.
* "Declared significant" = any model *without* information passes the gate's family-wise control.
* Funnel column: a model goes forward only if the gate rejects, and then it is the largest information
  t; otherwise nothing is bet.
* Sharpe column: what you get by always betting the in-sample Sharpe argmax.

| World | Games | Information gate rejects | SPA on P&L rejects | A no-information model declared significant | Funnel (gate, then argmax of t): goes with an informative model / goes with a no-information model / true ROI of what goes | In-sample Sharpe argmax (always goes): has info / true ROI |
|---|---|---|---|---|---|---|
| clones, no edge (−margin) | 2,000 | 2.5% | 0.0% | 2.5% | 0.0% / 2.5% / -4.8% | 0% / -4.8% |
| grid, no edge (−margin) | 2,000 | 5.0% | 0.5% | 5.0% | 0.0% / 5.0% / -4.8% | 0% / -4.8% |
| clones, same real edge | 2,000 | 54.0% | 8.0% | 0.0% | 54.5% / 0.0% / +2.8% | 100% / +2.8% |
| grid, 20 of 200 real | 2,000 | 38.5% | 2.5% | 5.5% | 38.5% / 0.5% / +3.4% | 56% / +1.7% |
| clones, no edge (−margin) | 10,000 | 4.0% | 0.0% | 4.0% | 0.0% / 4.0% / -4.8% | 0% / -4.8% |
| grid, no edge (−margin) | 10,000 | 8.0% | 0.0% | 8.0% | 0.0% / 8.0% / -4.8% | 0% / -4.8% |
| clones, same real edge | 10,000 | 99.0% | 21.0% | 0.0% | 100.0% / 0.0% / +2.8% | 100% / +2.8% |
| grid, 20 of 200 real | 10,000 | 97.0% | 10.0% | 5.0% | 97.0% / 0.0% / +3.6% | 87% / +3.8% |

* False positives of the information gate in the no-information worlds:
  * 2.5%–5.0% at 2,000 games;
  * 4.0%–8.0% at 10,000 games (100–200 replications; SE 1.5–2.7 pp).
  * Roughly nominal, possibly slightly liberal at 10k.
* In the no-information worlds, every rejection is a wrong "go"; those are the false positives above.
* In the world where only 20 of 200 models carry information, the gate almost never let a
  no-information model through: ≤ 0.5% of replications.
  * The in-sample Sharpe argmax always bets, and picked a model with no information
    44% of the time at 2,000 games (grid, 20 of 200 real).
* It finds real information that P&L cannot: 54% vs 8% (clones, 2,000 games).
* **Its size depends on accounting for the price's own bias.**
  * In `test_toolkit.py`, with a biased price and models leaning on logit q, the unadjusted score
    rejected 90% of the time.
  * The orthogonalised score rejected 3.5%.
* The gate proves *some* model carries information. Whether that information survives the margin
  and execution is stage 4.

---

## 5. Bet the blend, not the raw model (EXP-7) [confirmado in simulation]

In the rest of the simulation every model reads the market's public information perfectly, which
flatters the models (a hostile review pointed this out). Here the informative model also misreads
public information, with the error variance in the first column. The blend is fitted on the 10,000
previous games, then used to bet the next ones; 20 fits.

| Error variance on public info | Encompassing detects it (2,000 games) | Raw model: ROI per bet ± SE | Walk-forward blend: ROI per bet ± SE |
|---|---|---|---|
| 0.0 | 57.4% | +2.9% ± 0.1 (bets on 21% of games) | +4.7% ± 0.4 (bets on 11.6% of games) |
| 0.005 | 34.2% | -0.7% ± 0.1 (bets on 33% of games) | +3.1% ± 1.7 (bets on 6.7% of games) |
| 0.02 | 18.6% | -2.6% ± 0.1 (bets on 48% of games) | +0.4% ± 2.0 (bets on 2.1% of games) |

* A model can carry real information (the gate still detects it) and still lose money on its own
  probabilities, because its errors on public information cost more than its private edge earns.
* The blend keeps the information and mostly discards the errors, betting only where both agree.
  * That is why it bets on far fewer games.
  * With poor public information, its profit estimate is uncertain (± ~2%), but it no longer loses
    the way the raw model does.
* This is Benter's (1994) practice, and it costs nothing to adopt.

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
17,677 games from 2023–24.
* 163 of those games ended before 20:00. Their "20:00 price" is the result. Dropping them
  would use end-of-game information.
* Accuracy: 74.3% at 15:00, and 79.6% at 20:00 on the games still live.
* Mean predicted p at 15:00 is 0.532, against a Blue win rate of 0.531.
* Variance: the 15:00→20:00 price move has E[(q20−q15)²] = 0.029, against
  E[q15(1−q15)] = 0.175 left in the outcome.
* Var(profit)/Var(markout at 20:00) for a 15:00 bet at fair odds:

  | Fair odds | Ratio |
  |---|---|
  | 1.5–3.0 | **5.5×** |
  | ≤ 5 | **5.9×** |
  | all sides | 15.7×, driven by longshots |

  Sides priced q < 0.1 are 7.9% of sides but carry 70.7% of the profit
  variance.
* **The proxies are not perfect martingales.**
  * The identity E[q15(1−q15)] = E[dq²] + E[(y−q20)²] misses by 0.006 (SE 0.001).
  * The 15:00 proxy is slightly under-confident: Brier 0.171 vs 0.175.
  * Read ≈ 5–6× as an order of magnitude.

**When it fails (EXP-6) [confirmado in simulation].** The edge is built two ways with the same
information content.

| World (1M games, odds ≤ 5) | Profit/bet ± SE | Markout/bet | Profit − markout ± SE | Var(profit)/Var(markout) | Detection at 2,000 games: profit vs markout |
|---|---|---|---|---|---|
| catch-up edge (market learns it by 20:00) | +3.0% ± 0.3 | +2.9% | +0.1% ± 0.3 | 7.0 | 10.1% vs 32.3% |
| persistent edge (market never learns it) | +3.2% ± 0.3 | -4.6% | +7.9% ± 0.3 | 7.6 | 9.7% vs 0.0% |
| no edge | -4.4% ± 0.4 | -4.7% | +0.3% ± 0.3 | 7.4 | 1.6% vs 0.0% |

* **Persistent edge** (e.g. better team-strength priors that the live book never learns): the
  markout reads ≈ −margin on a real edge. The profit column is noise around the true ≈ +2.8%.
  Killing on markouts would kill a winner.
* **The residual test cannot tell the two cases apart at LoL sample sizes.** The residual is
  `o·(y − q_later)`, the part of profit the markout does not see.
  * Its variance is Var(profit) − Var(markout), ≈ 86% of the profit variance here.
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
* **Clustering.** The rule re-bet the same game at 20:00 in 64% of cases, with a within-game
  return correlation of 0.90. The game-clustered SE was 1.26× the naive one.

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
* The informative rules have true ROI +0.7% to +8.5%, and the clones +2.8%.
* The "pick" is the in-sample best per-game Sharpe, as in the PBO paper.
* The true ROI of a rule is exact without information; with information it comes from a
  10-million-game table.

**Winner's curse (medians over 200 replications).**

| World | Games | In-sample ROI of the pick | True ROI of the pick | Pick has information | Naive p < 5% |
|---|---|---|---|---|---|
| no edge | 500 | +39.5% | -4.8% | 0% | 28.0% |
| no edge | 1,000 | +27.7% | -4.8% | 0% | 23.5% |
| no edge | 2,000 | +22.5% | -4.8% | 0% | 32.0% |
| no edge | 5,000 | +13.5% | -4.8% | 0% | 24.5% |
| 20 of 200 real | 500 | +37.7% | -4.8% | 38% | 47.0% |
| 20 of 200 real | 1,000 | +25.1% | -4.8% | 44% | 43.0% |
| 20 of 200 real | 2,000 | +21.7% | +1.5% | 55% | 47.5% |
| 20 of 200 real | 5,000 | +11.0% | +3.0% | 77% | 57.0% |

**What each tool said.**
* 200 replications at 2,000 games; 100 at 10,000.
* For tests about the pick, "(x% false)" is the share of rejections where the pick had no
  information.
* MCS uses the range statistic at 90%.

| World | Games | True ROI of pick (median) | Naive | Bonferroni | DSR, N = M | DSR, N̂ (N̂) | SPA | RC | MCS size (median) | PBO | P(loss) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| clones, zero EV | 2,000 | +0.0% | 34.5% | 1.0% | 4.0% | 7.0% (43) | 3.5% | 4.5% | 200 | 0.49 | 0.48 |
| grid, zero EV | 2,000 | +0.0% | 43.0% | 0.0% | 0.5% | 3.0% (65) | 0.5% | 4.0% | 200 | 0.50 | 0.51 |
| clones, no edge (−margin) | 2,000 | -4.8% | 17.5% | 0.0% | 0.5% | 0.5% (61) | 0.5% | 0.5% | 200 | 0.49 | 0.68 |
| grid, no edge (−margin) | 2,000 | -4.8% | 27.5% | 0.0% | 0.5% | 1.0% (85) | 0.0% | 0.0% | 200 | 0.44 | 0.61 |
| clones, same real edge | 2,000 | +2.8% | 50.5% | 0.5% | 9.5% | 16.0% (41) | 9.5% | 10.0% | 200 | 0.50 | 0.38 |
| grid, 20 of 200 real | 2,000 | +1.9% | 43.0% (15.5% false) | 0.0% | 0.0% | 1.0% (88) | 2.5% | 5.5% | 200 | 0.35 | 0.51 |
| clones, zero EV | 10,000 | +0.0% | 28.0% | 0.0% | 6.0% | 9.0% (43) | 9.0% | 9.0% | 200 | 0.48 | 0.50 |
| grid, zero EV | 10,000 | +0.0% | 40.0% | 2.0% | 3.0% | 5.0% (65) | 6.0% | 6.0% | 200 | 0.48 | 0.51 |
| clones, no edge (−margin) | 10,000 | -4.8% | 6.0% | 0.0% | 0.0% | 0.0% (61) | 0.0% | 0.0% | 200 | 0.49 | 0.81 |
| grid, no edge (−margin) | 10,000 | -4.8% | 17.0% | 0.0% | 0.0% | 0.0% (84) | 0.0% | 0.0% | 199 | 0.35 | 0.70 |
| clones, same real edge | 10,000 | +2.8% | 73.0% | 9.0% | 25.0% | 36.0% (42) | 29.0% | 26.0% | 200 | 0.52 | 0.22 |
| grid, 20 of 200 real | 10,000 | +3.9% | 61.0% (5.0% false) | 2.0% | 0.0% | 0.0% (89) | 6.0% | 6.0% | 183 | 0.12 | 0.38 |

1. **False positives.** The honest test is at the zero-EV boundary; the −margin worlds are easier.

   | Test | Zero-EV worlds | −margin worlds |
   |---|---|---|
   | SPA | 0.5%–9.0% | 0.0%–0.5% |
   | RC | 4.0%–9.0% | 0.0%–0.5% |
   | DSR, N = M | 0.5%–6.0% | 0.0%–0.5% |
   | DSR, N̂ | 3.0%–9.0% | 0.0%–1.0% |
   | Naive test of the pick | **28.0%–43.0%** | 6.0%–27.5% |

   * Each cell has 100–200 replications, so its sampling SE is 1.5–2.2 pp.
   * The corrected tests are therefore roughly at their nominal 5%, not clearly below it.
   * The naive test is far above it.

2. **DSR: use N = all trials together with the cross-sectional variance of their Sharpe ratios.**
   * That variance already shrinks when trials are correlated.
   * Also shrinking N with the paper's N̂ = ρ̄ + (1−ρ̄)M (here N̂ ≈ 41–89) counts the
     correlation twice.
   * EXP-10 checks this directly: 200 trials, true SR 0, 1,000 returns each.

     | Correlation between trials | DSR false positives, N = M | DSR false positives, N̂ (Eq. 9) |
     |---|---|---|
     | 0.0 | 0.0% | 0.0% |
     | 0.5 | 1.6% | 2.8% |
     | 0.7 | 2.6% | 4.9% |
     | 0.9 | 4.5% | 7.0% |
   * DSR is very conservative for independent trials.
3. **Studentized SPA on P&L is somewhat liberal with skewed bets, and a minimum-bets filter only
   partly fixes it.**
   * EXP-10: zero EV, 200 candidates at fixed odds 1.4 / 1.9 / 2.8 / 5.0, 2,000 games.

     | Candidates | SPA false positives | RC false positives |
     |---|---|---|
     | rates 0.5%-40%, no filter | 7.8% | 6.4% |
     | rates 0.5%-40%, min 100 bets | 6.2% | 3.8% |
     | rates 5%-40%, no filter | 7.9% | 3.9% |
   * Prefer the RC as the P&L family report, or read SPA p-values as slightly optimistic.
   * P&L tests are reports here, not gates.
   * `arch` 8.0's `SPA` does not studentize despite its `studentize` flag (source read). It is a
     Reality Check.
4. **PBO judges the selection step, not the edge.**
   * Among clones it is ≈ 0.5 with or without an edge. The paper says so itself (References).
   * It was 0.35 where every model loses. There Sharpe selection consistently picks the
     rules that bet least, because they lose least.
   * It was 0.12 where 20 of 200 rules are genuinely better, the one world where it behaved as
     intended.
   * The paper's rule "reject if PBO > 0.05" would have rejected the real edges here.
   * CSCV's probability of loss separated the worlds better (0.22 with an edge vs
     0.81 without), but it is not a formal test.
   * CSCV's "performance degradation" slope is −1 by construction when the same model is picked in
     every split. In-sample and out-of-sample are complementary halves, so OOS = 2·full − IS.
5. **The MCS kept 183–200 of 200 models.** The data cannot rank them by P&L.
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

| Decimal odds | True ROI | t = 2 | t = 3 | 80% power (5%, one-sided) | Same, Bonferroni ×200 |
|---|---|---|---|---|---|
| 1.5 | 2% | 4,896 | 11,016 | 7,568 | 22,868 |
| 1.5 | 3% | 2,152 | 4,841 | 3,326 | 10,050 |
| 1.5 | 5% | 756 | 1,701 | 1,169 | 3,532 |
| 1.9 | 2% | 8,976 | 20,196 | 13,874 | 41,925 |
| 1.9 | 3% | 3,983 | 8,961 | 6,156 | 18,602 |
| 1.9 | 5% | 1,428 | 3,213 | 2,208 | 6,670 |
| 2.5 | 2% | 15,096 | 33,966 | 23,333 | 70,510 |
| 2.5 | 3% | 6,730 | 15,142 | 10,402 | 31,431 |
| 2.5 | 5% | 2,437 | 5,482 | 3,766 | 11,378 |

* Two bets on the same game (15:00 and 20:00) are **not** two independent bets: count games, or
  cluster.
* A pro LoL year has ~10–12.5k games **in total** in Oracle's Elixir:
  * 12,549 games in 55 leagues (2022);
  * 11,010 in 51 (2023);
  * 9,804 in 51 (2024, file ends 2024-12-08).
* The games you can bet live at 15:00 are fewer.
* **P&L alone cannot validate the best of 200 candidates:**
  * a 3% edge at odds 1.9 needs 3,983 bets for t = 2;
  * it needs 18,602 for 80% power after a Bonferroni correction.

**Live monitoring when you look after every bet (EXP-9) [confirmado in simulation].**
* 2,000 paths.
* Odds are uniform on 1.5–3.5, and the bettor claims +5% on every bet.
* SPRT: scale up at 20, kill at 0.05.

| True state (the bettor always claims +5%) | SPRT scales up | SPRT kills | Undecided after 5,000 bets | Median bets to decision | Naive t > 1.645 at any check |
|---|---|---|---|---|---|
| break_even (no edge) | 4.0% | 85.5% | 10.6% | 2,174 | 39.8% |
| claim right (+5%) | 84.4% | 4.3% | 11.3% | 2,230 | 96.1% |
| overconfident (claims +5%, true +2%) | 25.1% | 43.6% | 31.2% | 3,358 | 67.7% |
| losing (true -3%) | 0.1% | 99.7% | 0.2% | 1,176 | 19.4% |

* The naive rule ("scale up when t > 1.645") fires for 40% of break-even bettors when checked
  after every bet. Armitage et al. (1969) describe this optional-stopping problem.
* The SPRT's false scale-up rate stays ≤ 5% by Ville's inequality, and its false kill rate when the
  claim is right stays ≤ 5%.
  * It is valid for bets where your claimed probability is ≥ 1/odds, which is what an EV rule bets on.
  * It tests "better than break-even", not "as good as claimed".
  * **An overconfident claim hurts.** Claiming +5% when the truth was +2%, the SPRT killed that real,
    smaller edge 44% of the time, and scaled up only 25%.
  * Claim what the lockbox measured, or shrink the claim toward break-even. A shrunk claim keeps the
    guarantee.
* **Bets on the same game at 15:00 and 20:00 settle on one outcome [confirmado: algebra and
  `test_toolkit.py`].**
  * Multiplying their factors breaks the guarantee. For two bets on one side at the break-even
    price, E[f₁f₂] = 1 + (p−b)²/(b(1−b)) > 1.
  * In the test, break-even bettors who bet every game twice were scaled up 16.2% of the time.
  * Averaging the factors within each game (`groups=gameid`) kept it at 3.4%.
* **The drop-top-1% rule kills real edges.**
  * Shares of real +3% edges (2,000 bets) whose ROI turns ≤ 0 after dropping the best 1% of bets:
    odds 1.9: 17% · odds 2.5: 31% · odds 3.5: 44% · odds 5.0: 59%.
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
OMP_NUM_THREADS=1 python simulate.py --out results.json  # ~1.5 h on 4 cores
python lol_resolution_check.py /path/to/oracles_elixir_csvs
```

**On your data**
* One row per game, sorted by time. The 15:00 and 20:00 bets are **summed** into the game row
  (`aggregate_by_game`).
* Walk-forward predictions of every candidate for every game (`P`), the de-vigged price (`q`), the
  outcome (`y`), and the P&L matrix (`R`).
* Then, in funnel order:

| Stage | Function |
|---|---|
| 3 | `information_gate(y, q, P, groups=series_id)` |
| 4 | `fit_blend` / `blend_prob` (walk-forward) |
| 5 | `spa_vs_no_bet(R)`, `deflated_sharpe`, `pbo_cscv(R)`, `model_confidence_set(-R)` as reports |
| 9 | `sprt_break_even(won, p_claimed, odds, groups=gameid)` live |

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
  * Checked only in this simulation and in `test_toolkit.py`.
  * Validate its size on your own data with the synthetic-outcome control (§9.7).
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
