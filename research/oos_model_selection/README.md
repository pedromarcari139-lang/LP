# Choosing betting models that survive out-of-sample (live binary markets, LoL)

Scope: ~200 similar models, live binary markets at fixed game-clock timestamps (15:00, 20:00),
no closing line available. What to compute, in what order, and when to kill a model or an edge.

Evidence used here:
* `oos_toolkit.py`: reference implementations (checked by `test_toolkit.py`, cross-checked against `arch`).
* `simulate.py`: synthetic world, which gives the numbers in sections 5–7 (`results.json`).
* `lol_resolution_check.py`: real pro-LoL data (Oracle's Elixir 2022–24), used in section 5.
* A literature check (primary sources where reachable; see References).

Tags: **[confirmado]** = source read or computed here · **[provável]** = well supported, not fully verified ·
**[especulação]** = my estimate or proposal · **[não sei]** = unknown.

---

## 1. Bottom line

@@BOTTOM_LINE@@

---

## 2. What is publicly known about how betting quants validate

* **[não sei]** How syndicates validate internally. None of them publish it.
* **[confirmado]** Benter (1994), horse racing. He did not bet his model raw. He fitted a logit that
  **combined his model with the public's odds**, and judged the model by how much it improved on the
  odds alone. Reported pseudo-R²: public 0.1218, model 0.1245, combined 0.1396 (the numbers are from a
  secondary source).
* **[confirmado]** Pinnacle and Buchdahl treat CLV (beating the closing price) as the best short-sample
  indicator of skill, because it has far less variance than P&L.
* **[confirmado]** I found no documented, established "in-play CLV" metric.
* **[confirmado]** Accuracy is not profitability:
  * Wunderlich & Memmert (2020).
  * Hubáček, Šourek & Železný (2019): a model that is accurate but correlated with the bookmaker loses.
  * Hubáček & Šír (2023): a model worse than the market can still profit if it is decorrelated from it.
  * Walsh & Joshi (2024): selecting by calibration beat selecting by accuracy. **A 2025 corrigendum
    changed all their numbers** (a feature-engineering bug), so cite the direction, not the figures.
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
| 3 | **Information vs the price** (per model) | Encompassing regression; Δlog loss vs market; calibration of the edge | c not > 0 (one-sided p ≥ 0.05). Realised ROI not rising with predicted EV. |
| 4 | **Executable P&L** (per model) | Bets at obtainable prices (delay, rejections, suspensions, limits); profit per game; clustered t; drawdown; fractional-Kelly growth | Edge gone after realistic execution. Edge gone after dropping the top 1% most profitable bets. |
| 5 | **Family gate** (all survivors together) | SPA and RC vs "no bet"; DSR of the candidate with honest N (§6) | SPA p ≥ 0.05: **no model goes live**. The family has not shown any edge. |
| 6 | **Selection inside the family** | MCS (90%, range statistic) on log loss or −profit. PBO/CSCV as a diagnostic of the *selection step*. | Outside the MCS: drop. Pick the **average of the MCS members** (or the simplest one), never the in-sample argmax. |
| 7 | **Robustness** | Stability across time blocks, leagues, patches, timestamps, favourite/underdog, odds bands; parameter plateau | The edge lives in one slice that you did not pre-specify. The pick is an isolated spike among its neighbours. |
| 8 | **Lockbox, then live** | One run on the lockbox with the pre-registered criterion; then small-stake live with full logging (requested vs obtained odds, rejections, later prices for markouts) | Lockbox fails: stop. Live: cumulative P&L below the 5th percentile of the P&L you simulated from the claimed edge; markouts significantly negative; fill rate collapses. |

Scale up only when the **live sample on its own** supports the edge. Adding it to the backtest does not count.

---

## 4. Metrics and scorers: what each one answers

| Question | Metric / test | Use as | Notes |
|---|---|---|---|
| Is the probability good in absolute terms? | Log loss, Brier (= RPS for 2 outcomes), calibration slope/intercept, reliability curve | diagnostic | Accuracy/AUC ignore calibration; don't select betting models on them [confirmado]. |
| Does the model know something **the price doesn't**? | **Encompassing regression** (Fair–Shiller / Benter style): `logit P(y)=a+b·logit(q)+c·[logit(p)−logit(q)]`, SE clustered by game | **primary kill rule** | c > 0 ⇔ information beyond the price. |
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
`lol_resolution_check.py` fits win probability at 15:00 and at 20:00 on 2022 and tests on 17,578
games from 2023–24.

| Measure | 15:00 | 20:00 |
|---|---|---|
| Accuracy | 74.1% | 79.6% |
| Mean predicted p vs Blue win rate 0.530 | 0.532 | 0.534 |
| E[q(1−q)] (variance left in the outcome) | 0.176 | — |

* The 15:00→20:00 price move has E[(q20−q15)²] = 0.029.
* At fair odds, Var(profit)/Var(markout at 20:00) for a 15:00 bet:
  * **≈ 5.6×** for balanced games (0.3 < q15 < 0.7);
  * **≈ 12.8×** over all games.
* So you would need roughly **5–6× fewer bets** to reach the same t-stat, *if* the conditions below
  hold.

**When it fails (simulation, EXP-6).**

@@EXP6@@

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

@@SELECTION@@

---

## 7. Sample-size reality check

@@SAMPLESIZE@@

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
9. **Negative control [especulação as a rule, standard practice in spirit].**
   * Permute the outcomes within (league, patch) blocks and rerun the full pipeline.
   * The edge must vanish (ROI ≈ −margin). If it does not, something leaks.

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
    search, so reality is worse than these tables.
* **Stationarity.**
  * Patches, meta, roster changes and bookmaker model changes are not simulated.
  * The bootstrap here resamples games iid. If games on the same day or patch are dependent, use
    `mean_block > 1` with games sorted by time.
* **Markout magnitude.**
  * The real-data ratio (5.6× for balanced games) uses *proxy* prices from a gold/xp/kills logistic
    model, not real odds.
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
