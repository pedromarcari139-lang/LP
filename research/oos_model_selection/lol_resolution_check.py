"""How much of a pro LoL game's outcome variance is resolved between 15:00 and 20:00?

Uses Oracle's Elixir match data (oracleselixir.com; yearly CSVs, NOT included in
this repo): put oe2022.csv, oe2023.csv, oe2024.csv next to this script.
Walk-forward: fit on 2022, evaluate on 2023+2024 (no test game is ever in the fit).
The two logistic models on gold/xp/kill differences are only PROXIES for live
market prices; measure the same ratio on your own logged odds before relying on it.
"""
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm

DATA = sys.argv[1] if len(sys.argv) > 1 else "."
cols = ["gameid", "datacompleteness", "league", "date", "side", "position", "gamelength", "result"]
for t in (15, 20):
    cols += [f"golddiffat{t}", f"xpdiffat{t}", f"killsat{t}", f"opp_killsat{t}"]
frames = []
for y in (2022, 2023, 2024):
    d = pd.read_csv(f"{DATA}/oe{y}.csv", usecols=cols, low_memory=False)
    d = d[(d.position == "team") & (d.side == "Blue")].copy()
    d["year"] = y
    frames.append(d)
df = pd.concat(frames)
n0 = len(df)
df = df[df.datacompleteness == "complete"]
n1 = len(df)
short = (df.gamelength < 20 * 60).sum()
df = df[df.gamelength >= 20 * 60].dropna(subset=[c for c in cols if "at" in c])
assert df.gameid.is_unique, "one Blue team row per game expected"
print(f"blue rows {n0}, complete {n1}, dropped games shorter than 20:00: {short}, used {len(df)}")
for t in (15, 20):
    df[f"kd{t}"] = df[f"killsat{t}"] - df[f"opp_killsat{t}"]
def X(d, t):
    return sm.add_constant(np.column_stack([d[f"golddiffat{t}"] / 1000, d[f"xpdiffat{t}"] / 1000, d[f"kd{t}"]]))
train, test = df[df.year == 2022], df[df.year >= 2023]
print(f"train games (2022): {len(train)}, test games (2023-24): {len(test)}")
p = {}
for t in (15, 20):
    fit = sm.Logit(train.result.values, X(train, t)).fit(disp=0)
    p[t] = fit.predict(X(test, t))
y = test.result.values.astype(float)
for t in (15, 20):
    q = np.clip(p[t], 1e-6, 1 - 1e-6)
    ll = -np.mean(y * np.log(q) + (1 - y) * np.log(1 - q))
    print(f"at {t}:00  accuracy={np.mean((q > .5) == y):.3f}  logloss={ll:.4f}  mean p={q.mean():.3f} vs blue winrate={y.mean():.3f}")
q15, q20 = p[15], p[20]
dq = q20 - q15
v_out, v_move, v_rest = np.mean(q15 * (1 - q15)), np.mean(dq ** 2), np.mean((y - q20) ** 2)
print(f"E[q15(1-q15)]={v_out:.4f}  E[(q20-q15)^2]={v_move:.4f}  E[(y-q20)^2]={v_rest:.4f}  (martingale => first ~ second + third: {v_move + v_rest:.4f})")
print(f"mean(q20-q15)={dq.mean():+.4f} (martingale => ~0)")
# per-bet variance at fair odds, averaged over backing either side of every game
num = np.concatenate([(1 - q15) / q15, q15 / (1 - q15)])                 # Var(profit) = (1-q)/q
den = np.concatenate([dq ** 2 / q15 ** 2, dq ** 2 / (1 - q15) ** 2])     # Var(markout) = E[dq^2]/q^2
print(f"variance ratio profit/markout (both sides, fair odds): {num.mean() / den.mean():.2f}")
mask = (q15 > .3) & (q15 < .7)
print(f"same ratio, only games with 0.3<q15<0.7: {num[np.tile(mask, 2)].mean() / den[np.tile(mask, 2)].mean():.2f}")
