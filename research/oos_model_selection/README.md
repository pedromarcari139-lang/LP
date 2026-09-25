# Choosing betting models that survive out-of-sample (live binary markets, LoL)

Scope: ~200 similar models, live binary markets at fixed game-clock timestamps (15:00, 20:00),
no closing line available. What to compute, in what order, and when to kill a model or an edge.

Evidence used here:
* `oos_toolkit.py`: reference implementations, checked by `test_toolkit.py` against known answers:
  * the SPA against a literal re-implementation of Hansen (2005);
  * the RC and the MCS against the `arch` package. `arch` 8.0's SPA does not studentize (source read),
    so it cannot check the SPA itself.
* `simulate.py`: synthetic world, which gives the numbers in sections 5–7 (`results.json`).
* `lol_resolution_check.py`: real pro-LoL data (Oracle's Elixir 2022–24), used in section 5.
* A literature check (primary sources where reachable; see References).

Tags: **[confirmado]** = source read or computed here · **[provável]** = well supported, not fully verified ·
**[especulação]** = my estimate or proposal · **[não sei]** = unknown.

---

## 1. Bottom line

1. **No rule guarantees out-of-sample performance [confirmado].**
   * PBO, DSR, SPA and the bootstrap only *estimate or penalise* selection bias.
   * The only real out-of-sample data is data nobody looked at while choosing: a lockbox used once,
     then a live forward test.
2. **With 200 near-identical models, "which one is best" is not answerable from the data.**
   * [confirmado in simulation] Even with 10,000 games, the Model Confidence Set kept 97–100% of the
     models, and the in-sample best overstated its ROI by +5 to +35 pp.
   * [provável, my inference] Test the *family*. Then choose by what you pre-specified, by simplicity
     and by robustness, never by the in-sample argmax.
   * Average only near-duplicates of the same idea: averaging a set that is 90% noise dilutes the signal.
3. **The first gate is information versus the price at bet time, not ROI and not accuracy.**
   * [confirmado in simulation] An encompassing regression detected a real +3.6%-ROI edge 56% of the
     time with 2,000 games. An ROI t-test detected it 7% of the time, and 21% even at 10,000.
   * [provável] Using it as the gate is my recommendation; Benter (1994) is the published precedent
     for combining the model with the public price.
4. **You can have a live CLV: the markout [confirmado: the math; provável: the magnitude].**
   * Re-price each bet at a later fair price, e.g. the 20:00 price for a 15:00 bet.
   * On real LoL data (proxy prices, odds 1.5–5) it cuts variance ≈ 5.5–5.9×.
   * It is blind to edges the market never learns. Use it as an accelerator and an alarm, never as
     the only kill rule.
5. **PBO judges the selection step, not the edge [confirmado: paper and simulation].**
   * Among near-clones it is ≈ 0.5 with or without a real edge.
   * It can be lower in a world where everything loses.
6. **P&L alone cannot validate 200 candidates at LoL sample sizes
   [confirmado: arithmetic and simulation].**
   * A 3% edge at odds 1.9 needs ~4,000 independent bets for t = 2, and ~18,600 for 80% power after a
     Bonferroni correction for 200.
   * The corrected tests (SPA, RC, DSR) kept false positives ≤ 3%, but detected a real +3.6% edge only
     16–42% of the time at 10,000 games.
7. **Therefore:**
   * cut the candidate list *before* testing: pre-specify, cluster, and average only near-duplicates;
   * gate on information vs the price;
   * use markouts for speed;
   * count every trial for DSR and SPA;
   * let the lockbox and a live forward test make the final call.

---

## 2. What is publicly known about how betting quants validate

* **[não sei]** How syndicates validate internally. None of them publish it.
* **[confirmado]** Benter (1994), horse racing. He did not bet his model raw. He fitted a logit that
  **combined his model with the public's odds**, and judged the model by how much it improved on the
  odds alone. Reported pseudo-R²: public 0.1218, model 0.1245, combined 0.1396 (the numbers are from a
  secondary source).
* **[confirmado]** Pinnacle and Buchdahl treat CLV (beating the closing price) as the best indicator of
  skill.
* **[provável]** Buchdahl: CLV evidence becomes significant in far fewer bets than P&L, because its
  variance is much lower (secondary source).
* **[confirmado]** I found no documented, established "in-play CLV" metric.
* **[confirmado]** Accuracy is not profitability:
  * Wunderlich & Memmert (2020).
  * Hubáček, Šourek & Železný (2019): a model that is accurate but correlated with the bookmaker loses.
  * Hubáček & Šír (2023): a model worse than the market can still profit if it is decorrelated from it.
  * Walsh & Joshi (2024): selecting by calibration beat selecting by accuracy.
    * **A 2025 corrigendum changed all their numbers** (a feature-engineering bug).
    * According to a secondary summary of the corrigendum, the direction still holds [provável].
      Cite the direction, not the figures.
* **[confirmado]** From quant finance, for selection among many backtests:
  * White's Reality Check (2000).
  * Hansen's SPA (2005).
  * Romano–Wolf StepM (2005).
  * Hansen–Lunde–Nason's Model Confidence Set (2011).
  * Bailey & López de Prado's PSR/DSR (2012/2014).
  * PBO/CSCV (Bailey, Borwein, López de Prado & Zhu, 2017).
  * Harvey–Liu–Zhu's t > 3 hurdle (2016).
* **[provável]** How many firms actually apply these, and how strictly. Nobody publishes it.

---

## 3. The funnel

Run the stages in order. A model that fails a stage never reaches the next one.

**Thresholds are conventions I propose [especulação]. Fix them BEFORE you look at the results**,
otherwise the thresholds themselves become another thing you overfit.

| # | Stage | What you compute | Kill rule (proposed defaults) |
|---|---|---|---|
| 0 | **Freeze the protocol** | The list of every trial (models, feature sets, EV thresholds, variants you discarded), the metric, the bet rule, and the lockbox (the most recent ~20–30% of games, used **once**) | Changing the protocol after seeing the lockbox burns the lockbox: it becomes in-sample |
| 1 | **Leakage audit** (§8) | Odds↔state timestamp check, whole-game-column check, gameid/series grouping, negative control | Any failure: fix the **pipeline**, restart from stage 0. Do not tweak the model. |
| 2 | **Walk-forward predictions** | Expanding or rolling window by date. Train strictly before test. All rows of a gameid (15:00 and 20:00) and all maps of a series in the same fold. | Every later stage sees **only** these out-of-sample predictions |
| 3 | **Information vs the price** (per model or per cluster average) | Encompassing regression (a, b, c + joint test); Δlog loss vs market; calibration of the edge. Needs the price at every checkpoint of **every** game, not only the games you bet. | Kill if c is not > 0 after a multiple-testing adjustment (Holm, or BHY for FDR); 200 tests at 5% let ~10 noise models through. Exception: c ≈ 0 but (a, b) ≠ (0, 1), i.e. the price is miscalibrated. Then the edge is a simple recalibration of the price; keep *that* as the candidate and drop the complex model. Also kill if realised ROI does not rise with predicted EV. |
| 4 | **Executable P&L** (per model) | Bets at obtainable prices (delay, rejections, suspensions, limits); profit per game; clustered t; drawdown; fractional-Kelly growth | Edge gone after realistic execution. Edge gone after dropping the top 1% most profitable bets. |
| 5 | **Family gate** (over **all** candidates ever tried, not only survivors) | SPA and RC vs "no bet"; DSR of the candidate with honest N (§6). Filtering on stages 3–4 with the same data is itself selection, so a test run only on the survivors is optimistic. | SPA p ≥ 0.05: **no model goes live**. The family has not shown any edge. |
| 6 | **Selection inside the family** | MCS (90%, range statistic) on log loss or −profit. PBO/CSCV as a diagnostic of the *selection step*. | Outside the MCS: drop. If the MCS is small, average its members. If it keeps nearly everything (as in §6), the data cannot choose: take the pre-specified or simplest model and treat it as **unvalidated** until stage 8. Never the in-sample argmax. |
| 7 | **Robustness** | Stability across time blocks, leagues, patches, timestamps, favourite/underdog, odds bands; parameter plateau | The edge lives in one slice that you did not pre-specify. The pick is an isolated spike among its neighbours. |
| 8 | **Lockbox, then live** | One run on the lockbox with the pre-registered criterion; then small-stake live with full logging (requested vs obtained odds, rejections, later prices for markouts) | Lockbox fails: stop. Live: cumulative P&L below the 5th percentile of the P&L you simulated from the claimed edge; markouts significantly negative; fill rate collapses. |

Scale up only when the **live sample on its own** supports the edge. Adding it to the backtest does not count.

---

## 4. Metrics and scorers: what each one answers

| Question | Metric / test | Use as | Notes |
|---|---|---|---|
| Is the probability good in absolute terms? | Log loss, Brier (= RPS for 2 outcomes), calibration slope/intercept, reliability curve | diagnostic | Accuracy/AUC ignore calibration; don't select betting models on them [confirmado]. |
| Does the model know something **the price doesn't**? | **Encompassing regression** (Fair–Shiller / Benter style): `logit P(y)=a+b·logit(q)+c·[logit(p)−logit(q)]`, SE clustered by game | **primary gate** | c > 0: information beyond any logit-linear recalibration of the price. (a, b) ≠ (0, 1): the price itself is miscalibrated (e.g. favourite–longshot bias). That edge shows in b, **not** in c [confirmado in `test_toolkit.py`: b = 1.40, c = 0.03]. |
| Same question, stand-alone | Paired Δlog loss (model − de-vigged market), SE clustered by game (Diebold–Mariano-type) | diagnostic | Not a kill rule: a model can be worse than the market overall and still add information (Hubáček & Šír 2023) [confirmado]. |
| Is the edge estimate itself calibrated? | Bucket bets by predicted EV; realised ROI should rise with predicted EV | kill / recalibrate | Tests the *edge*, not just the probability [especulação: my rule]. |
| Does the betting rule make money after margin and execution? | Profit per game, ROI per bet, t-stat clustered by game, max drawdown, log-growth at fractional Kelly | final judge | Needs thousands of bets (§7). |
| Early evidence without waiting for outcomes | Markout at t+Δ (live CLV analogue, §5) | accelerator, **not a veto** | Blind to "persistent" edges [confirmado in simulation]. |
| Skill or luck, given how much I tried? | SPA / RC (family vs no-bet), DSR with honest N, MCS (set of indistinguishable models), PBO + probability of loss + degradation | family gate / selection | Count every trial ever run [confirmado: PBO paper]. |
| Is it robust? | Sign/size stable across time blocks, leagues, patches, 15:00 vs 20:00, favourite/underdog, odds bands; parameter plateau; drop-top-1%-bets test | kill | [especulação: thresholds are mine]. |
| Can I actually get the price? | Fill rate, rejection rate, obtained vs requested odds, delay-adjusted backtest | kill | Live bets get delayed and rejected [confirmado]. |

---

## 5. A live substitute for CLV: markouts

Your premise was "I can't use CLV because I bet live". Half right. There is no *closing* line for a
15:00 bet, but CLV is one case of a general idea you **can** use: re-price the bet later.

**Definition.** Back side s at odds `o` at 15:00. Take the market's de-vigged probability of s at a
later moment t+Δ (20:00, or +1 min, +5 min). Then

```
markout = o · q_s(t+Δ) − 1        (profit = o · y_s − 1)
```

With the closing price as q_s, this is the usual CLV-based EV.

**Why it works [confirmado: the math, given its assumptions].** Suppose q(t+Δ) is the exact
conditional probability of the market at t+Δ, and that market already knows everything **you** knew
at bet time. By the tower property:

* `E[profit − markout | your info] = o · E[y − q(t+Δ) | your info] = 0`, so markout is **unbiased**
  for your true EV.
* `Var(profit) = Var(markout) + E[o² · q(t+Δ)·(1 − q(t+Δ))]`, so markout has **lower variance**.

That lower variance is why CLV is used at all.

**How much faster, on real LoL data [provável for the magnitude; proxy prices].**

`lol_resolution_check.py` fits win probability at 15:00 and at 20:00 on 2022 and tests on 17,677
games from 2023–24.
* 163 of them ended before 20:00. For those, the "20:00 price" is the result.
* Dropping them would have used end-of-game information.

| Measure | 15:00 | 20:00 (games still live) |
|---|---|---|
| Accuracy | 74.3% | 79.6% |
| Mean predicted p vs Blue win rate 0.531 | 0.532 | — |

* The 15:00→20:00 price move has E[(q20−q15)²] = 0.029. The variance left in the outcome at 15:00 is
  E[q15(1−q15)] = 0.175.
* Var(profit)/Var(markout at 20:00) for a 15:00 bet at fair odds:

  | Odds range | Ratio |
  |---|---|
  | 1.5–3.0 | **≈ 5.5×** |
  | ≤ 5 (82% of sides) | **≈ 5.9×** |
  | all sides | 15.7×, but misleading |

  * The "all sides" figure is driven by longshots. Sides priced q < 0.1 are 7.9% of sides but carry
    70.7% of the profit variance.
* So for the prices you actually bet, you need roughly **5–6× fewer bets** to reach the same t-stat,
  *if* the conditions below hold.
* **These proxies are not perfect martingales.**
  * The identity E[q15(1−q15)] = E[dq²] + E[(y−q20)²] is off by 0.006 (SE 0.001).
  * The 15:00 proxy is slightly under-confident: Brier 0.171 vs E[q(1−q)] 0.175.
  * Treat 5–6× as an order of magnitude.

**When it fails (simulation, EXP-6).**

The edge was built two ways in the simulation, each with the same information content.

| World (1M games) | Profit/bet | Markout/bet | Profit − markout (± SE) | Var(profit)/Var(markout) | Bets for t = 2: profit vs markout | Detection at 2,000 games: profit vs markout |
|---|---|---|---|---|---|---|
| catch-up edge (market learns it by 20:00) | +3.1% | +3.6% | -0.5% ± 0.4% | 10.8 | 19,606 vs 1,320 | 7.5% vs 36.5% |
| persistent edge (market never learns it) | +4.2% | -4.8% | +9.0% ± 0.4% | 11.7 | 9,975 vs never (mean < 0) | 8.1% vs 0.0% |
| no edge | -3.9% | -4.8% | +1.0% ± 0.6% | 16.4 | — vs never (mean < 0) | 1.4% vs 0.0% |

* **Catch-up edge.** The markout is unbiased: profit − markout ≈ 0.
  * The variance ratio is ≈ 11×, so it needs ~11× fewer bets.
  * The table's 19,606 vs 1,320 also reflects sampling noise in the profit mean (+3.1% ± 0.4%).
* **Persistent edge.** The markout reads ≈ −margin on a **real** +4.2% edge.
  * Used as a kill rule, it would have killed a winner.
  * The residual test flags it (+9.0% ± 0.4%).
* **Clustering, persistent world.** The same rule re-bet the same game at 20:00 in 88% of cases.
  * Within-game return correlation was 0.58.
  * The game-clustered SE was 1.19× the naive SE.

**How to use it.**
1. **Logging is required.** Store the price of both sides at +30 s, +1 min, +5 min, and at the next
   checkpoint (20:00 for a 15:00 bet), for every bet *and* every non-bet.
2. De-vig each snapshot.
3. Report the mean markout with a t-stat clustered by game.
4. **Also run the residual test:** the mean of `o·(y − q_later)` over your bets, with its SE.
   * ≈ 0: the later market absorbs your information, and markouts can speak for P&L.
   * Significantly > 0: your edge is "persistent" and markouts understate it.
   * < 0: markouts overstate the edge. Danger.
5. **Use markouts as an accelerator and an alarm, never as the only kill rule.** A persistent edge
   (for example, better team-strength priors that the live book never learns) shows ~−margin
   markouts while being real.
6. If a sharper live book exists for your markets, its later price is a better reference than your
   own book's [provável].

---

## 6. Choosing among 200 near-identical models: what each tool really tells you

**Setup (`simulate.py`, EXP-2/3/4).**
* 200 candidate rules bet flat stakes at 15:00 against a calibrated market with a 5% margin.
* Pairwise correlation of per-game P&L between candidates is 0.63–0.83.
* Four worlds:
  * **clones, no edge**: 200 copies of one rule with no information.
  * **clones, same real edge**: 200 copies, each with a true ROI of +3.6%.
  * **grid, no edge**: 50 signal weights × 4 EV thresholds, none with information.
  * **grid, 20 of 200 real**: the same grid, 20 of them with real information.
* The "pick" is the in-sample best per-game Sharpe, as in the PBO paper.
* True ROI is measured on 200k fresh games.
* 200 replications at 2,000 games; 100 at 10,000.

**Winner's curse (EXP-2, grid worlds; medians over 200 replications) [confirmado in simulation]**

| World | Games | In-sample ROI of the pick | True ROI of the pick | Naive p < 5% |
|---|---|---|---|---|
| no edge | 500 | +30.3% | −4.8% | 13% |
| no edge | 2,000 | +19.3% | −4.9% | 16% |
| no edge | 5,000 | +13.4% | −4.7% | 12.5% |
| 20 of 200 real | 500 | +31.7% | −3.4% | 20.5% |
| 20 of 200 real | 2,000 | +19.7% | +3.2% | 30% |
| 20 of 200 real | 5,000 | +12.0% | +4.6% | 41.5% |

The pick tends to be a model that **bets rarely** and got lucky. In "grid, no edge" at 2,000 games the
pick placed a median of 151 bets, against 310 for the typical candidate.

Noise concentrates on long odds. A model whose errors are symmetric on the probit/logit scale sees the
largest apparent EVs on underdogs, where per-bet variance is highest. Over 60 samples of 10,000 games,
the pooled ROI of the no-information models ranged from −11% to +10%, around a true −4.8%. (This was an
ad-hoc check on the "grid, 20 of 200 real" world; it is not in `results.json`.)

**What each tool said (EXP-3/4) [confirmado in simulation]**

* In the no-edge worlds, "rejects" = false positives; in the real-edge worlds = power.
* N̂ = the paper's Eq. 9.
* MCS = 90% confidence set, T_max statistic. A spot check with the range statistic gave the same sizes.
* P(loss) = CSCV probability that the pick loses out of sample.

| World | Games | True ROI of pick (median) | Naive | Bonferroni | DSR N=200 | DSR N̂ (N̂) | SPA | RC | MCS size | PBO | P(loss) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| clones, no edge | 2,000 | -4.7% | 12.5% | 0.5% | 2.5% | 3% (53) | 0.5% | 0.5% | 199.8 | 0.50 | 0.65 |
| clones, same real edge | 2,000 | +3.6% | 38% | 1% | 12% | 18% (37) | 5% | 7% | 199.9 | 0.51 | 0.42 |
| grid, no edge | 2,000 | -4.6% | 10% | 0% | 0.5% | 2% (72) | 0% | 0.5% | 199.2 | 0.49 | 0.66 |
| grid, 20 of 200 real | 2,000 | +2.4% | 27.5% | 0% | 2.5% | 3% (75) | 1% | 3.5% | 198.8 | 0.37 | 0.55 |
| clones, no edge | 10,000 | -4.8% | 2% | 0% | 0% | 0% (52) | 0% | 0% | 199.8 | 0.49 | 0.79 |
| clones, same real edge | 10,000 | +3.6% | 64% | 2% | 29% | 42% (35) | 16% | 23% | 199.8 | 0.51 | 0.27 |
| grid, no edge | 10,000 | -5.0% | 11% | 0% | 0% | 0% (67) | 0% | 0% | 196.7 | 0.39 | 0.65 |
| grid, 20 of 200 real | 10,000 | +4.2% | 47% | 0% | 0% | 1% (69) | 2% | 6% | 194.6 | 0.10 | 0.34 |

How to read it:
1. **The corrected tests control false positives.**
   * SPA, RC, DSR and Bonferroni had ≤ 3% false positives in every no-edge world.
   * Testing the pick naively gave 2–16%.
2. **Their power is poor with 200 candidates and a few-% edge.**
   * A real +3.6% edge with 10,000 games was detected by SPA 16%, RC 23%, DSR 29–42% and Bonferroni 2%
     of the time.
   * **P&L alone cannot validate 200 candidates at LoL sample sizes.**
3. **PBO judges the selection step, not the edge.**
   * Among clones it is ≈ 0.5 whether or not the edge exists. The PBO paper says as much (see
     References).
   * It was *lower* (0.39) in a world where every model loses than in the clones-with-edge world (0.51).
   * The paper's rule "reject if PBO > 0.05" would have rejected every world here, including those with
     a real edge.
   * PBO behaved as intended only when candidates genuinely differed (grid, 20 of 200 real: 0.10).
   * CSCV's probability of loss separated the worlds better (0.27 vs 0.79) but is not a formal test.
4. **The MCS keeps 97–100% of the models even at 10,000 games.** The data cannot rank them.
   * In "grid, 20 of 200 real" the set is ~90% noise models.
   * Averaging it would dilute the real signal, so averaging is only sensible for near-duplicates of one
     idea.
   * The choice has to come from what you pre-specified and from simplicity, and it stays unvalidated
     until the lockbox and live test.
5. **For DSR, the paper's N̂ = ρ̄ + (1−ρ̄)M gave N̂ ≈ 35–75 for these 200 models.**
   * It kept false positives ≤ 3% and had more power than N = 200.
   * Report both, and count **every** trial you ever ran in M, not just the final 200.

---

## 7. Sample-size reality check

**Independent flat bets needed [confirmado: arithmetic, `bets_needed`].**
Per-bet sd = odds·√(p(1−p)), with p = (1+ROI)/odds. Full grid in `results.json`.

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

* Two bets on the same game (15:00 and 20:00) are **not** two independent bets.
  * In EXP-6 the clustered SE was 1.19× the naive SE.
  * Count games, not bets, or cluster.
* The Bonferroni column is an upper bound for correlated models; SPA/DSR need less.
  * Still, the order of magnitude is **tens of thousands of bets to validate the best of 200
    candidates by P&L alone**. Ranking them against each other needs more.
* A professional LoL year has ~10–12.5k pro games **in total** in Oracle's Elixir.
  * Counted: 12,549 games in 55 leagues (2022); 11,010 in 51 (2023); 9,804 in 51 (2024, file ends
    2024-12-08).
  * The games you can actually bet live at 15:00 are fewer.

**Power of three tests for the SAME real edge (EXP-5; one model; true ROI ≈ +3.6%; one-sided 5%;
1,000 replications) [confirmado in simulation]**

| Games | ROI t-test | Δlog loss vs market | Encompassing (c > 0) |
|---|---|---|---|
| 500 | 4% | 8% | 24% |
| 1,000 | 5% | 9% | 38% |
| 2,000 | 7% | 11% | 56% |
| 5,000 | 12% | 19% | 88% |
| 10,000 | 21% | 31% | 100% |

* With no edge, the encompassing test rejected 4.0%–6.6% of the time
  (nominal 5%); the other two rejected ≤ 2%.
* The encompassing test uses **every game**, not just the games you bet, and the full probability, not
  just win/lose.
  * At 2,000 games it detected the edge 56% of the time. The ROI t-test was still at 21% with
    10,000 games.
  * So it needs **more than 5× fewer games** here. Extrapolating the ROI curve suggests ~20–35×, but
    that is an extrapolation [especulação].
* It proves *information*, not *profit after margin and execution*.
  * So it is the first gate (stage 3), not the last.

---

## 8. LoL-live leakage checklist (run this before any metric)

Each item is a way a live LoL backtest can look profitable without being so.

1. **Odds ↔ state alignment (the most important live leak) [provável].**
   * The odds you score at "15:00" must be the price that was actually on offer while your 15:00
     features were already known.
   * Game clock, data feed and broadcast are all out of sync. If the odds snapshot is a few seconds
     older than an event your features include (a kill, a dragon), the backtest shows an edge that
     does not exist.
   * Test: distribution of (odds timestamp − feature timestamp). Re-run with the odds snapshot
     deliberately taken 10–30 s *later*. If the edge collapses, it was a timing artefact.
2. **Whole-game columns used as 15:00 features [confirmado].**
   * Oracle's Elixir mixes time-stamped columns (`golddiffat15`, `killsat15`, …) with **whole-game**
     columns: `towers` (full-game total, mean 6.06 in 2023), `dragons`, `firsttower`, `firstdragon`,
     `firstherald`, `firstbaron`, `firsttothreetowers`, `inhibitors`, `kills`.
   * Any of these used at 15:00 leaks the future. Rebuild them from the event timeline at the timestamp.
3. **gameid split [confirmado as mechanism].**
   * The 15:00 and 20:00 rows of a game share one outcome.
   * If one row is in train and the other in test, the model can memorise the game.
   * Keep all rows of a game in the same fold, and all maps of a series (same teams, same day) too.
4. **Pre-game features from the future [provável].**
   * Team ratings, Elo, form, champion win rates and patch statistics must use only games *before*
     the game (for a same-day series, only earlier maps).
   * A champion win rate computed over the whole patch is a leak.
5. **Survivorship in the odds data [provável].**
   * Markets are suspended around fights and objectives.
   * If games or timestamps with no price are dropped, and suspension correlates with game state,
     the sample is biased.
   * Log suspensions; don't silently drop them.
6. **Partial data [confirmado].**
   * Counted here, in Oracle's Elixir, games flagged partial have no at-10/15/20 stats:

     | | 2022 | 2023 | 2024 |
     |---|---|---|---|
     | Share of games that are partial | 15.1% | 15.8% | 14.2% |

   * About 92–93% of the partial games are LPL or LDL.
   * If you bet those leagues live, a model trained on "complete" games has not seen them.
7. **Execution [confirmado as mechanism].**
   * Live bets are delayed and can be rejected when the game state changes:
     * Pinnacle holds live bets and rejects them after material changes.
     * Betfair in-play bet delay is 1–12 s.
   * A backtest at the snapshot price is an upper bound. Log every attempt (price requested vs
     obtained, accepted or rejected).
8. **Data errors [confirmado].**
   * Clegg & Cartlidge (2025) found most of a published strategy's OOS profit came from one bet at
     erroneous long odds.
   * Check for stale prices (unchanged while the game changes), impossible overrounds and duplicated
     games.
9. **The de-vig method moves your benchmark [confirmado: arithmetic].**
   * With two outcomes, additive = Shin (per the `implied` R package docs); multiplicative differs.
   * How much the favourite's fair probability changes between the two methods:

     | Odds | Change |
     |---|---|
     | 1.87 / 1.95 | 0.05 pp |
     | 1.30 / 3.50 | 1.26 pp |
     | 1.10 / 7.00 | 1.89 pp |

   * Against a 2–3% edge that is not small, and lopsided prices are common at 15:00.
   * Choose the method by the walk-forward log loss of the de-vigged market on your own data, and fix
     it in stage 0 [provável].
10. **Negative control with synthetic outcomes [especulação: my design; the logic is exact].**
    * Replace every outcome with `y* ~ Bernoulli(de-vigged market probability)`.
    * Rerun the **whole** pipeline on y*: feature engineering, training, walk-forward, evaluation.
    * By construction nobody can beat the market on y*, so the backtest must show ROI ≈ −margin.
    * An "edge" on y* means label leakage (for example, target-derived features computed with future
      rows) or an evaluation bug.
    * Do **not** simply permute the real outcomes. That breaks the odds↔outcome link: an underdog at
      3.50 "wins" ~50% of the time and looks hugely profitable with no leak at all.
    * This control cannot catch features that encode the real outcome (item 2). Only the timestamp
      audit catches those.

---

## 9. Reproduce

```
pip install numpy scipy pandas statsmodels arch
python test_toolkit.py                                   # ~40 s, all checks must PASS
OMP_NUM_THREADS=1 python simulate.py --out results.json  # ~40 min on 4 cores
python lol_resolution_check.py /path/to/oracles_elixir_csvs
```

To use the toolkit on your data, build:
* one row per game, sorted by time;
* `R[g, k]` = walk-forward out-of-sample profit of model k in game g, with the 15:00 and 20:00 bets
  **summed** into the game row (`aggregate_by_game`);
* `y`, the market's fair probability at bet time, and each model's probability.

---

## 10. What was NOT verified / limitations

* **The simulation is not LoL.**
  * Every magnitude in it is an assumption: margin 5%, the size of the edge, how much of the game is
    decided between 15:00 and 20:00, and the correlation between models.
  * It shows *mechanisms*, with numbers. It does not predict your results.
* **Only selection bias is simulated.**
  * The models are fixed rules, not fitted. Real pipelines also overfit through fitting and feature
    search, so reality is likely worse than these tables [provável].
* **Stationarity.**
  * Patches, meta, roster changes and bookmaker model changes are not simulated.
  * The bootstrap here resamples games iid. If games on the same day or patch are dependent, use
    `mean_block > 1` with games sorted by time.
* **Markout magnitude.**
  * The real-data ratio (≈ 5.5–5.9× at odds 1.5–5) uses *proxy* prices from a gold/xp/kills logistic
    model, not real odds. Those proxies are not perfect martingales (a 0.006 gap in the variance
    identity).
  * Real market prices contain more information (draft, team strength), so the true ratio can differ
    in either direction.
  * Measure `Var(profit) / Var(markout)` on your own logged odds.
* **Literature access.**
  * Several sources were checked only through official abstracts or secondary summaries (tagged in
    References).
  * The corrected numbers in the Walsh & Joshi corrigendum were not checked against the corrigendum
    itself.
* **What professional syndicates do internally is unknown [não sei].**
  * Section 2 lists only what is published.
* **"MSR"** in the question is ambiguous (maximum Sharpe ratio? expected-max SR / SR₀? modified
  Sharpe? minimum track record?).
  * It does not appear in the PBO or DSR papers.
  * I did not guess its meaning.

---

## References (how each was checked)

How each reference was checked:
* **VP** = primary text read.
* **VP-idx** = official abstract or journal page, seen via search.
* **VS** = reliable secondary source.

In this sandbox the research agents could not open SSRN, arXiv or journal sites directly. They read
author PDFs mirrored on GitHub and the search-engine extracts.

**Backtest overfitting**
* Bailey, Borwein, López de Prado & Zhu, "The Probability of Backtest Overfitting",
  *J. Computational Finance* 20(4), 2017. **VP** (SSRN version).
  * Algorithm 2.3.
  * Four diagnostics: PBO, performance degradation, probability of loss, stochastic dominance.
  * "Reject models for which PBO is estimated to be greater than 0.05."
  * Verbatim: "it is entirely possible that all the N strategies have high but similar Sharpe ratios.
    Since none of the strategies is clearly better than the rest, PBO will be high."
  * Note: C(16,8) = 12,870; the paper's "12,780" is a typo.
* Bailey & López de Prado, "The Deflated Sharpe Ratio", *J. Portfolio Management* 40(5), 2014. **VP**.
  * Eq. 2 (DSR, raw kurtosis).
  * Appendix A.3, Eq. 9: N̂ = ρ̂ + (1 − ρ̂)M. I read this on the page image.
* Bailey & López de Prado, "The Sharpe Ratio Efficient Frontier", *J. Risk* 15(2), 2012. **VP**.
  PSR and MinTRL.
* Bailey, Borwein, López de Prado & Zhu, "Pseudo-Mathematics and Financial Charlatanism",
  *Notices AMS* 61(5), 2014. **VP**. MinBTL.
* López de Prado & Bailey, "The False Strategy Theorem", *American Mathematical Monthly* 128(9), 2021.
  **VP-idx**.
* López de Prado, "A Data Science Solution to the Multiple-Testing Crisis in Financial Research",
  *J. Financial Data Science* 1(1), 2019. **VP-idx**.
* López de Prado & Lewis, "Detection of false investment strategies using unsupervised learning
  methods", *Quantitative Finance* 19(9), 2019. **VP-idx**.

**Data-snooping tests**
* White, "A Reality Check for Data Snooping", *Econometrica* 68(5), 2000. **VP-idx**.
* Hansen, "A Test for Superior Predictive Ability", *JBES* 23(4), 2005. **VP-idx**.
* Romano & Wolf, "Stepwise Multiple Testing as Formalized Data Snooping", *Econometrica* 73(4), 2005.
  **VP-idx**.
* Hansen, Lunde & Nason, "The Model Confidence Set", *Econometrica* 79(2), 2011. **VP-idx**.
  "Uninformative data yield a MCS with many models."
* Politis & Romano, "The Stationary Bootstrap", *JASA* 89(428), 1994. **VP-idx**.
* Harvey, Liu & Zhu, "…and the Cross-Section of Expected Returns", *RFS* 29(1), 2016. **VP-idx**.
  t > 3.0.
* Harvey & Liu, "Backtesting", *JPM* 42(1), 2015. **VS**.

**Forecast comparison**
* Diebold & Mariano, *JBES* 13(3), 1995. **VP-idx**.
* Giacomini & White, *Econometrica* 74(6), 2006. **VP-idx**.
* Fair & Shiller, "Comparing information in forecasts from econometric models", *AER* 80(3), 1990.
  **VP-idx**.
* Clements & Harvey, "Forecast encompassing tests and probability forecasts",
  *J. Applied Econometrics* 25(6), 2010. **VP-idx**.

**Betting**
* Benter, "Computer based horse race handicapping and wagering systems", in
  *Efficiency of Racetrack Betting Markets*, 1994. **VP** for the method, **VS** for the pseudo-R²
  numbers.
* Hubáček, Šourek & Železný, *IJF* 35(2), 2019. **VP**.
* Hubáček & Šír, *IJF* 39(2), 2023. **VP**.
* Walsh & Joshi, *Machine Learning with Applications* 16, 2024, with corrigendum in MLWA 19, 2025.
  **VP** that the corrigendum exists; its corrected numbers are **VS**.
* Wunderlich & Memmert, *IJF* 36(2), 2020. **VP**.
* Kaunitz, Zhong & Kreiner, arXiv 1710.02824, 2017. **VP**.
* Clegg & Cartlidge, *IJF* 41(2), 2025. **VP**.

**Converting odds and staking**
* Štrumbelj, *IJF* 30(4), 2014 (Shin's method). **VP**.
* Clarke, Kovalchik & Ingram, *AJSS* 5(6), 2017 (power method). **VP**.
* The documentation of the R package `implied` states that with two outcomes the additive and Shin
  methods coincide. **VP**.
* Baker & McHale, "Optimal betting under parameter uncertainty", *Decision Analysis* 10(3), 2013
  (shrink the Kelly stake). **VP**.

**In-play markets and execution**
* Pinnacle and Buchdahl on CLV. **VP**.
* Pinnacle rules on accepting live bets, and Betfair in-play bet delays. **VP**.
* Databento microstructure guide, definition of "markout". **VS**.
