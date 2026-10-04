# -*- coding: utf-8 -*-
"""
comparar_opcoes.py — compara OPÇÕES FIXAS (modelo × flag) cabeça a cabeça, nos MESMOS jogos, lendo o lucro_por_jogo.csv.gz do funil.

Não recalcula previsões nem apostas: usa o lucro de cada opção em cada jogo executável (o mesmo que o funil usa) e o universo.csv.gz.
Para cada opção: lucro, PPG, nº de apostas, ROI, maxDD, calmar, sharpe/jogo, % de blocos de 150 positivos, PPG por período;
para cada par (opção vs a 1ª): Δ PPG PAREADO jogo a jogo com erro-padrão por cluster de 10 gameids (z).
ATENÇÃO: opção FIXA no período todo = olhar o resultado para escolher (retrospecto). Serve para entender, não para provar.

USO: python comparar_opcoes.py [PASTA_OUT] --opcoes "TUDO_MOM_CALR,TUDO_INI_CALR|bipro,TUDO_INI_CALR,TUDO_MOM_CALR|bipro" [--funil FIRST_10a35] [--desde 6115] [--ate N]
     (PASTA_OUT padrão: OUT_FUNIL_CONTINUACAO se existir, senão OUT_FUNIL) → imprime e grava PASTA_OUT/COMPARAR_OPCOES_<funil>.txt
"""
import os, sys, argparse
import numpy as np
import pandas as pd

ap = argparse.ArgumentParser(); ap.add_argument("pasta", nargs="?", default=None)
ap.add_argument("--opcoes", default="TUDO_MOM_CALR,TUDO_INI_CALR|bipro,TUDO_INI_CALR,TUDO_MOM_CALR|bipro")
ap.add_argument("--funil", default="FIRST_10a35"); ap.add_argument("--desde", type=int, default=6115); ap.add_argument("--ate", type=int, default=None)
ap.add_argument("--periodos", default="6115,6628,7729,8448")
a = ap.parse_args()
PASTA = a.pasta or ("OUT_FUNIL_CONTINUACAO" if os.path.isdir("OUT_FUNIL_CONTINUACAO") else "OUT_FUNIL")
LJ = pd.read_csv(os.path.join(PASTA, "lucro_por_jogo.csv.gz")); U = np.sort(pd.read_csv(os.path.join(PASTA, "universo.csv.gz")).gameid.values)
O = pd.read_csv(os.path.join(PASTA, "opcoes.csv")).set_index("nome")
ops = [o.strip() for o in a.opcoes.split(",") if o.strip()]
falta = [o for o in ops if o not in O.index]
if falta: sys.exit(f"opções inexistentes: {falta}. Exemplos válidos: {list(O.index[:8])}")
if a.funil not in set(LJ.funil): sys.exit(f"funil {a.funil} não está no lucro_por_jogo (há: {sorted(LJ.funil.unique())[:10]}…)")
ini = max([a.desde] + [int(O.loc[o, "w"]) for o in ops])                        # mesmos jogos: todas as opções já existem
js = U[(U >= ini) & ((U < a.ate) if a.ate else True)]
lj = LJ[(LJ.funil == a.funil) & LJ.opcao.isin(ops)]
L = {o: lj[lj.opcao == o].set_index("gameid").lucro.reindex(js).fillna(0.0) for o in ops}
N = {o: lj[lj.opcao == o].set_index("gameid").n_apostas.reindex(js).fillna(0.0) for o in ops}
cortes = sorted({ini} | {int(x) for x in a.periodos.split(",") if int(x) > ini} | {int(js.max()) + 1})
txt = []
def diz(s=""): print(s, flush=True); txt.append(s)
def media_se(x):
    x = pd.Series(x); c = (x.index.values - 1) // 10
    d = pd.DataFrame(dict(x=x.values, c=c)).groupby("c")["x"].agg(["sum", "size"]); m = d["sum"].sum() / d["size"].sum(); G = len(d)
    return float(m), float(np.sqrt(((d["sum"] - d["size"] * m) ** 2).sum() * G / max(G - 1, 1)) / d["size"].sum()) if G > 1 else np.nan
diz(f"COMPARAÇÃO DE OPÇÕES FIXAS · funil {a.funil} · {len(js)} jogos executáveis de {ini} a {int(js.max())} (todas as opções já existem) · {os.path.abspath(PASTA)}")
diz("RETROSPECTO: opção fixa no período inteiro = escolhida olhando o resultado. Δ PPG pareado = mesma série de jogos, z por cluster de 10 gameids.")
for o in ops:
    x = L[o]; n = N[o].sum(); c = np.cumsum(x.values); dd = float(np.max(np.maximum.accumulate(np.r_[0.0, c])[1:] - c)); m, se = media_se(x)
    bl = [x.values[i:i + 150].mean() for i in range(0, len(x) - 149, 150)]
    diz(f"\n=== {o}")
    diz(f"   PPG {m:+.4f} ± {se:.4f} · lucro {c[-1]:+.1f} u · apostas {int(n)} ({n / len(js):.2f} por jogo) · ROI {c[-1] / n:+.3f} · maxDD {dd:.1f} u · calmar {c[-1] / dd if dd > 0 else np.nan:.2f} · "
        f"sharpe/jogo {x.mean() / x.std(ddof=1):+.3f} · blocos de 150 positivos {np.mean(np.array(bl) > 0):.0%}" if n > 0 else "   sem apostas")
    diz("   PPG por período: " + " · ".join(f"{p0}–{p1 - 1} {media_se(x[(js >= p0) & (js < p1)])[0]:+.3f}" for p0, p1 in zip(cortes[:-1], cortes[1:]) if ((js >= p0) & (js < p1)).sum() > 1))
    if o != ops[0]:
        d_, s_ = media_se(x - L[ops[0]]); diz(f"   Δ PPG contra {ops[0]} (pareado): {d_:+.4f} ± {s_:.4f} · z {d_ / s_ if s_ and s_ > 0 else np.nan:+.2f}")
open(os.path.join(PASTA, f"COMPARAR_OPCOES_{a.funil}.txt"), "w", encoding="utf-8").write("\n".join(txt))
print(f"\ngravado {os.path.join(PASTA, f'COMPARAR_OPCOES_{a.funil}.txt')}")
