# -*- coding: utf-8 -*-
"""
ver_vencedores.py — FICHA COMPLETA dos vencedores de cada funil, lida do resumo_funil.csv + mcs_reality_check.csv (não recalcula nada).

USO: python ver_vencedores.py [PASTA_OUT] [--politica FIRST] [--top 3]
     (PASTA_OUT padrão: OUT_FUNIL_CONTINUACAO se existir, senão OUT_FUNIL) → imprime e grava PASTA_OUT/VENCEDORES_<POLITICA>.txt
Para cada nível de EV (EV>0 = hoje, EV>5%, EV>10%) × janela × conjunto: as TOP regras por PPG com lucro, ROI, IC, calmar, sharpe, sortino,
drawdown, blocos positivos, otimismo, percentil fora da amostra, p contra o acaso, MCS, Reality Check do bloco, CLV, Brier skill, PPG por
período e opções mais escolhidas; e o BASE (V6_MOM fixo) do mesmo bloco como referência.
"""
import os, sys, argparse
import numpy as np
import pandas as pd

ap = argparse.ArgumentParser(); ap.add_argument("pasta", nargs="?", default=None); ap.add_argument("--politica", default="FIRST"); ap.add_argument("--top", type=int, default=3)
a = ap.parse_args()
PASTA = a.pasta or ("OUT_FUNIL_CONTINUACAO" if os.path.isdir("OUT_FUNIL_CONTINUACAO") else "OUT_FUNIL")
R = pd.read_csv(os.path.join(PASTA, "resumo_funil.csv"), low_memory=False); MC = pd.read_csv(os.path.join(PASTA, "mcs_reality_check.csv"))
cad = sorted(c for c in R.cadencia.dropna().unique() if str(c).startswith("a_cada_"))[0]
funis = [f for f in [f"{a.politica}_10a35", f"{a.politica}_10a35_EV5", f"{a.politica}_10a35_EV10"] if f in set(R.funil)]
per = [c[4:] for c in R.columns if c.startswith("ppg_") and c[4:] not in ("prometido",) and f"se_{c[4:]}" in R.columns]
txt = []
def diz(s=""): print(s, flush=True); txt.append(s)
f_ = lambda v, fmt: "n/d" if v is None or (isinstance(v, float) and not np.isfinite(v)) or (isinstance(v, str)) else format(v, fmt)
def g(x, c): return x[c] if c in x.index else np.nan

def ficha(x, ref=False):
    lin = [f"PPG {f_(x.ppg, '+.4f')} [{f_(x.ic_lo, '+.4f')}; {f_(x.ic_hi, '+.4f')}] · lucro {f_(g(x, 'lucro'), '+.1f')} u em {int(x.n_jogos)} jogos · "
           f"apostas {f_(g(x, 'n_apostas'), '.0f')} · ROI {f_(g(x, 'roi'), '+.3f')}"
           + ("" if ref else f" [{f_(g(x, 'roi_ic_lo'), '+.3f')}; {f_(g(x, 'roi_ic_hi'), '+.3f')}] × esperado {f_(g(x, 'ev_medio_modelo'), '+.3f')} (calibrado {f_(g(x, 'ev_medio_calibrado'), '+.3f')})"),
           f"calmar {f_(g(x, 'calmar'), '.2f')} · sharpe/jogo {f_(g(x, 'sharpe_jogo'), '+.3f')} · sortino/jogo {f_(g(x, 'sortino_jogo'), '+.3f')} · maxDD jogo/aposta {f_(g(x, 'maxdd'), '.1f')}/"
           f"{f_(g(x, 'maxdd_por_aposta'), '.1f')} u · pior seq. jogos negativos {f_(g(x, 'pior_seq_jogos_negativos'), '.0f')} · pior seq. apostas perdidas {f_(g(x, 'pior_seq_apostas_perdidas'), '.0f')} · "
           f"blocos de 150 positivos {f_(g(x, 'pct_blocos_positivos'), '.0%')} (pior bloco {f_(g(x, 'pior_bloco_ppg'), '+.3f')})"]
    if not ref:
        lin.append(f"validação: otimismo {f_(g(x, 'otimismo'), '+.3f')} · percentil OOS {f_(g(x, 'percentil_oos_medio'), '.2f')} · acaso pct {f_(g(x, 'percentil_no_acaso'), '.2f')} "
                   f"p={f_(g(x, 'p_vs_acaso'), '.3f')} · no MCS {'sim' if g(x, 'no_MCS') is True or g(x, 'no_MCS') == 'True' else 'não'} · trocas {f_(g(x, 'trocas'), '.0f')} · "
                   f"IC unilateral p={f_(g(x, 'p_unilateral'), '.4f')}")
    lin.append(f"CLV {f_(g(x, 'clv_fech_medio'), '+.3f')} · mk_fech {f_(g(x, 'mk_fech_medio'), '+.4f')} · supera {f_(g(x, 'supera_fech_taxa'), '.0%')} · Brier skill {f_(g(x, 'bsskill_escolhida'), '+.5f')}"
               + (f" · {x.mais_escolhidas}" if isinstance(g(x, 'mais_escolhidas'), str) else ""))
    lin.append("PPG por período: " + " · ".join(f"{p} {f_(g(x, 'ppg_' + p), '+.3f')}±{f_(g(x, 'se_' + p), '.3f')}" for p in per))
    return lin

diz(f"VENCEDORES {a.politica} — {os.path.abspath(PASTA)} · top {a.top} por PPG em cada (nível de EV × janela × conjunto) · tudo lido do resumo_funil.csv")
diz("ATENÇÃO: escolher a regra (ou o nível de EV) por esta lista é seleção nos mesmos dados; use o Reality Check do bloco e o Δ pareado do RESUMO.")
for fu in funis:
    for (jan, cj), r in R[R.funil == fu].groupby(["janela", "conjunto"], sort=False):
        mc = MC[(MC.funil == fu) & (MC.janela == jan) & (MC.conjunto == cj)]
        diz(f"\n=== {fu} · janela {jan} · opções {cj} · Reality Check p={f_(mc.rc_p_reality_check.iloc[0], '.4f') if len(mc) else 'n/d'} · MCS: {mc.mcs_regras.iloc[0] if len(mc) else 'n/d'}")
        top = r[r.cadencia == cad].sort_values("ppg", ascending=False).head(a.top)
        for k, (_, x) in enumerate(top.iterrows(), 1):
            diz(f" {k}º {x.regra}"); [diz("     " + l) for l in ficha(x)]
        base = r[(r.cadencia == "-") & r.regra.str.startswith("BASE_")]
        if len(base): diz(f" [ref] {base.regra.iloc[0]}"); [diz("     " + l) for l in ficha(base.iloc[0], ref=True)]
open(os.path.join(PASTA, f"VENCEDORES_{a.politica}.txt"), "w", encoding="utf-8").write("\n".join(txt))
print(f"\ngravado {os.path.join(PASTA, f'VENCEDORES_{a.politica}.txt')}")
