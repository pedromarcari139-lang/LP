# Confere que V6/V6WR dão a MESMA previsão no motor limpo e no motor sujo (rode depois de uma rodada de teste, na mesma pasta)
import importlib.util, sys, numpy as np, pandas as pd
sys.argv = ["x"]; import validar_funil as VF
M = {t: (VF.carregar_motor(a, t), None) for t, a in (("sujo", VF.ARQ_SUJO), ("limpo", VF.ARQ_LIMPO))}
B_l = M["limpo"][0]; B_l.CFG.update(VF.CFG_EXTRA); B_l.CFG["MODO"] = "limpo"; B_l.CFG["FAMILIAS"] = ["V6", "V6WR"]
B_l.CFG.update(C_FIXO=1.0, RETREINO_DEV_JOGOS=2, LOCKBOX_GID_MIN=700, LOCKBOX_ETAPA="desenvolvimento", PASTA_OUT="OUT_V6L"); B_l._pastas()
br = {t: B_l.carregar_bruto(t) for t in [10, 25, 35]}; fr = {t: B_l.preparar(br[t], t) for t in br}; B_l.CFG["TEMPOS"] = [10, 25, 35]
bl = [b for b in B_l.blocos_teste(760) if b[2] <= 700]
p, _ = B_l.walk_forward_dev(fr, bl, B_l.candidatos(), banco=None)
s = pd.read_csv("OUT_r1/previsoes_funil.csv.gz"); s = s[(s.motor == "sujo") & s.cand.isin(p.cand.unique()) & s.t.isin([10, 25, 35]) & (s.gameid >= 500)]
m = p.merge(s, on=["t", "cand", "gameid", "side"], suffixes=("_limpo", "_sujo"))
print(f"V6/V6WR no motor LIMPO × no motor SUJO: {len(m)} previsões casadas de {len(p)} · max|Δp| = {float((m.p_limpo - m.p_sujo).abs().max()):.2e}")
