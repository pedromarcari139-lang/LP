import numpy as np, pandas as pd
rng = np.random.default_rng(1); N = 760; T = [5,10,15,20,25,30,35,40,45]
dur = rng.uniform(25, 55, N); s = rng.normal(0, 1, N); res1 = (rng.random(N) < 1/(1+np.exp(-s))).astype(int)
gold = {tau: s*tau*60 + rng.normal(0, 400*np.sqrt(tau/5), N) for tau in T}
pre = {c: rng.uniform(30, 70, N) for c in ["SOLOQ","SOLOQ25","SOLOQ30","SOLOQ35","SOLOQ35MAIS","PRO","PRO25","PRO30","PRO35","PRO35MAIS"]}
pre.update({c: rng.uniform(-1, 1, N) for c in ["GOLDTOP","GOLDJG","GOLDMID","GOLDADC","GOLDSUP","IMPTOP","IMPJG","IMPMID","IMPADC","IMPSUP"]})
flags = ["bisolo","bisolo25","bisolo30","bisolo35","bisolo35mais","bipro","bipro25","bipro30","bipro35","bipro35mais","cWR","cWRgrande","cPROBS","cPROBSmenor","cPROBSgrande","cPROBSgrandemenor","roles"]
pre.update({c: rng.integers(0, 2, N).astype(float) for c in flags})
def pz(x): return np.clip(100/(1+np.exp(-x)), 3, 97)
for t in T:
    rows = []
    for g in range(N):
        if dur[g] < t: continue
        for side in (1, 2):
            sg = 1 if side == 1 else -1; r = {"gameid": g+1, "side": side, "ordem": 2*(g+1) - (2-side), "resultado": res1[g] if side == 1 else 1-res1[g], "PATCH": 1 + g//200, "timeA": "A", "timeB": "B"}
            z = pz(sg*0.8*s[g]); vz = pz(-sg*0.8*s[g]); r.update(Z=z*1.04, VSZ=vz*1.04, odd=100/(z*1.04), vsodd=100/(vz*1.04))
            for c, v in pre.items(): r[c] = (v[g] if side == 1 else (100-v[g] if c.startswith(("SOLOQ","PRO")) else v[g]))
            for tau in [x for x in T if x <= t]:
                gd = sg*gold[tau][g]; zt = pz(sg*(0.8*s[g] + gold[tau][g]/2500) + rng.normal(0, .2)); vzt = 100 - zt
                r.update({f"g{tau}": gd, f"k{tau}m": max(0, int(gd/600)+3), f"d{tau}m": max(0, int(-gd/600)+3), f"drake{tau}": int(tau>=10)*rng.integers(0,3), f"opdrake{tau}": int(tau>=10)*rng.integers(0,3),
                          f"tt{tau}": int(tau>=15)*rng.integers(0,4), f"optt{tau}": int(tau>=15)*rng.integers(0,4), f"Z{tau}": zt*1.04, f"VSZ{tau}": vzt*1.04, f"odd{tau}": 100/(zt*1.04), f"vsodd{tau}": 100/(vzt*1.04)})
            rows.append(r)
    pd.DataFrame(rows).to_excel(f"zz{t}.xlsx", index=False); print(t, len(rows))
