"""How much of a pro LoL game's outcome variance is resolved between 15:00 and 20:00?

Uses Oracle's Elixir match data (oracleselixir.com; yearly CSVs, NOT included in
this repo): put oe2022.csv, oe2023.csv, oe2024.csv in one folder and pass it.
Walk-forward: fit on 2022, evaluate on 2023+2024 (no test game is ever in the fit).
The two logistic models on gold/xp/kill differences are only PROXIES for live
market prices; measure the same ratio on your own logged odds before relying on it.
"""
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm

DATA = sys.argv[1] if len(sys.argv) > 1 else "."
STATE = {t: [f"golddiffat{t}", f"xpdiffat{t}", f"killsat{t}", f"opp_killsat{t}"] for t in (15, 20)}
cols = ["gameid", "datacompleteness", "side", "position", "gamelength", "result", *STATE[15], *STATE[20]]

frames = []
for year in (2022, 2023, 2024):
    d = pd.read_csv(f"{DATA}/oe{year}.csv", usecols=cols, low_memory=False)
    d = d[(d.position == "team") & (d.side == "Blue")].copy()
    d["year"] = year
    frames.append(d)
df = pd.concat(frames)
assert df.gameid.is_unique, "one Blue team row per game expected"
n_all = len(df)
df = df[df.datacompleteness == "complete"].dropna(subset=STATE[15])
# a game that ends before 20:00 has no 20:00 state; its "price at 20:00" is the result
df["reached20"] = df.gamelength >= 20 * 60
df = df[~df.reached20 | df[STATE[20]].notna().all(axis=1)]
print(f"blue-side rows {n_all}; complete with 15:00 state: {len(df)}; "
      f"ended before 20:00: {(~df.reached20).sum()} (kept: their 20:00 price is the result)")

for t in (15, 20):
    df[f"kd{t}"] = df[f"killsat{t}"] - df[f"opp_killsat{t}"]


def X(d, t):
    return sm.add_constant(np.column_stack([d[f"golddiffat{t}"] / 1000, d[f"xpdiffat{t}"] / 1000, d[f"kd{t}"]]))


train, test = df[df.year == 2022], df[df.year >= 2023]
print(f"train games (2022): {len(train)}, test games (2023-24): {len(test)}")
fit15 = sm.Logit(train.result.values, X(train, 15)).fit(disp=0)
tr20 = train[train.reached20]                         # the 20:00 model only runs on live games
fit20 = sm.Logit(tr20.result.values, X(tr20, 20)).fit(disp=0)

y = test.result.values.astype(float)
q15 = fit15.predict(X(test, 15))
live20 = test.reached20.values
q20 = y.copy()
q20[live20] = fit20.predict(X(test[test.reached20], 20))


def ll(yv, qv):
    qv = np.clip(qv, 1e-6, 1 - 1e-6)
    return -np.mean(yv * np.log(qv) + (1 - yv) * np.log(1 - qv))


print(f"at 15:00  accuracy={np.mean((q15 > .5) == y):.3f}  logloss={ll(y, q15):.4f}  mean p={q15.mean():.3f} vs blue winrate={y.mean():.3f}")
print(f"at 20:00 (live games) accuracy={np.mean((q20[live20] > .5) == y[live20]):.3f}  logloss={ll(y[live20], q20[live20]):.4f}")
dq = q20 - q15
gap = q15 * (1 - q15) - dq ** 2 - (y - q20) ** 2
print(f"E[q15(1-q15)]={np.mean(q15 * (1 - q15)):.4f}  E[(q20-q15)^2]={np.mean(dq ** 2):.4f}  E[(y-q20)^2]={np.mean((y - q20) ** 2):.4f}")
print(f"martingale identity gap E[q15(1-q15)] - E[dq^2] - E[(y-q20)^2] = {gap.mean():+.4f} (SE {gap.std() / np.sqrt(gap.size):.4f}); "
      f"mean(q20-q15) = {dq.mean():+.4f}; Brier at 15:00 = {np.mean((y - q15) ** 2):.4f}")

# per-bet variance at fair odds, backing either side of every game:
# Var(profit) = (1-q)/q, Var(markout at 20:00) = E[dq^2]/q^2
q_side = np.concatenate([q15, 1 - q15])
v_profit = (1 - q_side) / q_side
v_mark = np.concatenate([dq, -dq]) ** 2 / q_side ** 2
odds = 1 / q_side
for label, mask in (("all sides", np.ones(q_side.size, bool)),
                    ("fair odds 1.5-3.0", (odds >= 1.5) & (odds <= 3.0)),
                    ("fair odds <= 5", odds <= 5.0)):
    print(f"variance ratio profit/markout, {label:17s}: {v_profit[mask].mean() / v_mark[mask].mean():5.2f}  "
          f"({mask.mean():.0%} of sides)")
long_ = q_side < 0.1
print(f"sides with q15 < 0.1: {long_.mean():.1%} of sides, {v_profit[long_].sum() / v_profit.sum():.1%} of the profit variance")
