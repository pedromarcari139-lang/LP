# Testes unitários da v5.1 (baterias de ODD MÍNIMA): valores montados à mão, sem dados reais e sem motor.
# Uso: python teste_v51.py   (na pasta _testes_funil; carrega ../validar_funil.py como módulo)
import os, sys, importlib.util
import numpy as np
import pandas as pd

sp = importlib.util.spec_from_file_location("vf", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "validar_funil.py"))
VF = importlib.util.module_from_spec(sp); sys.argv = ["x"]; sp.loader.exec_module(VF)
falhas, feitas = [], []
def ok(nome, cond, det=""):
    print(("OK  " if cond else "FALHOU ") + nome + (f": {det}" if det else "")); feitas.append(nome); falhas.append(nome) if not cond else None

VF.CFG = dict(EDGE_MIN=0.0, ODD_MIN=1.01, ODD_MAX=7.0, STAKE=1.0, TEMPOS_DEPLOY=[10, 15, 20])
CONT, ABAN, MIN_ = VF.ODD_POR_SUF["_CONT150"], VF.ODD_POR_SUF["_ABAN150"], VF.ODD_POR_SUF["_MIN150"]
# jogo 1: T10 sinal no lado 1 com odd 1,40 (barato) · T15 sinal lado 1 odd 1,60 · T20 sinal lado 1 odd 1,45 (barato só se o mínimo vale em todos os minutos)
# jogo 2: T10 sinal odd 2,00 (não barato) · T15 sinal odd 1,20 (barato) · T20 sinal odd 1,80
# jogo 3: T10 SEM sinal (edge < 0 nos dois lados) · T15 sinal odd 1,35 (barato) · T20 sem sinal
linhas = []
def par(g, t, o1, p1, o2):                                                                  # lado 1: odd o1 e p1; lado 2: odd o2 e p = 1 − p1
    linhas.append(dict(cand="X", gameid=g, t=t, side=1, odd_t=o1, p=p1, y=1)); linhas.append(dict(cand="X", gameid=g, t=t, side=2, odd_t=o2, p=1 - p1, y=0))
par(1, 10, 1.40, 0.80, 3.00); par(1, 15, 1.60, 0.70, 2.40); par(1, 20, 1.45, 0.75, 2.80)
par(2, 10, 2.00, 0.55, 1.85); par(2, 15, 1.20, 0.90, 4.50); par(2, 20, 1.80, 0.60, 2.10)
par(3, 10, 1.90, 0.50, 1.90); par(3, 15, 1.35, 0.80, 3.20); par(3, 20, 1.90, 0.50, 1.90)
b = pd.DataFrame(linhas); b["q_raw_t"] = 1 / b.odd_t; b["edge"] = b.p - b.q_raw_t; b["ev"] = b.p * b.odd_t - 1
T = [10, 15, 20]
def cont_col(r):
    return (~VF.sinal_barato(b["t"].values, b["odd_t"].values, r)).astype(int)

ok("sinal barato (mínimo 1,50 em T10/T15): odd 1,40 no T10 e 1,20 no T15 → sim; 1,45 no T20 → não; 1,60 → não",
   list(VF.sinal_barato(np.array([10, 15, 20, 15]), np.array([1.40, 1.20, 1.45, 1.60]), CONT)) == [True, True, False, False])
ok("mínimo em TODOS os minutos: 1,45 no T20 → barato", list(VF.sinal_barato(np.array([20]), np.array([1.45]), MIN_)) == [True])

mp, _ = VF.apostas(b, "X", "MULTI", tempos=T); fp_, _ = VF.apostas(b, "X", "FIRST", tempos=T)
ok("PADRÃO MULTI: sinais em (1,10) (1,15) (1,20) (2,10) (2,15) (2,20) (3,15)", list(zip(mp.gameid, mp.t)) == [(1, 10), (1, 15), (1, 20), (2, 10), (2, 15), (2, 20), (3, 15)], f"{list(zip(mp.gameid, mp.t))}")
ok("PADRÃO FIRST: (1,10) (2,10) (3,15)", list(zip(fp_.gameid, fp_.t)) == [(1, 10), (2, 10), (3, 15)])

b["__c"] = cont_col(CONT); fc, _ = VF.apostas(b, "X", "FIRST", tempos=T, filtro_col="__c"); mc, _ = VF.apostas(b, "X", "MULTI", tempos=T, filtro_col="__c")
ok("CONTINUA 1,50 FIRST: jogo 1 pula o T10 barato e entra no T15 (1,60); jogo 2 igual (T10 2,00); jogo 3 sem aposta (T15 barato, T20 sem sinal)",
   list(zip(fc.gameid, fc.t)) == [(1, 15), (2, 10)], f"{list(zip(fc.gameid, fc.t))}")
ok("CONTINUA 1,50 MULTI: tira só os sinais baratos de T10/T15 (1,10) (2,15) (3,15); o T20 de 1,45 fica", list(zip(mc.gameid, mc.t)) == [(1, 15), (1, 20), (2, 10), (2, 20)], f"{list(zip(mc.gameid, mc.t))}")

fa = VF.abandonar_jogo(mp, ABAN, "FIRST"); ma = VF.abandonar_jogo(mp, ABAN, "MULTI")
ok("ABANDONA 1,50 FIRST: jogo 1 largado no T10 (1º sinal barato) → sem aposta; jogo 2 aposta no T10 (2,00); jogo 3 largado no T15",
   list(zip(fa.gameid, fa.t)) == [(2, 10)], f"{list(zip(fa.gameid, fa.t))}")
ok("ABANDONA 1,50 MULTI: jogo 1 nada; jogo 2 só o T10 (largado no T15 barato, o T20 não entra); jogo 3 nada", list(zip(ma.gameid, ma.t)) == [(2, 10)], f"{list(zip(ma.gameid, ma.t))}")
ok("lucro das apostas que ficam = o da padrão (mesmas linhas)", np.allclose(ma.lucro.values, mp[(mp.gameid == 2) & (mp.t == 10)].lucro.values))

b["__m"] = cont_col(MIN_); fm, _ = VF.apostas(b, "X", "FIRST", tempos=T, filtro_col="__m"); mm, _ = VF.apostas(b, "X", "MULTI", tempos=T, filtro_col="__m")
ok("MÍNIMO 1,50 EM TODOS: FIRST (1,15) (2,10); MULTI (1,15) (2,10) (2,20) — o T20 de 1,45 do jogo 1 também sai",
   list(zip(fm.gameid, fm.t)) == [(1, 15), (2, 10)] and list(zip(mm.gameid, mm.t)) == [(1, 15), (2, 10), (2, 20)], f"{list(zip(fm.gameid, fm.t))} {list(zip(mm.gameid, mm.t))}")

# o lado escolhido continua o de MAIOR edge: o filtro de odd não troca de lado (no jogo 1 / T10 o lado 2 tem edge < 0 e não vira aposta)
ok("filtro de odd não faz apostar no outro lado", set(mc.side) == {1} and set(ma.side) == {1})
# sem informação do futuro: tirar o T20 dos dados não muda as apostas de T10/T15 no ABANDONA
ma2 = VF.abandonar_jogo(mp[mp.t < 20], ABAN, "MULTI")
ok("ABANDONA: apagar os minutos depois (T20) não muda o que acontece em T10/T15", list(zip(ma2.gameid, ma2.t)) == [(g, t) for g, t in zip(ma.gameid, ma.t) if t < 20])
ok("funis da v5.1: 6 baterias × FIRST/MULTI = 12, nomes com sufixo, EV0", len(VF.FUNIS_ODD) == 12 and all(f["ev_min"] is None for f in VF.FUNIS_ODD)
   and {f["nome"] for f in VF.FUNIS_ODD} >= {"FIRST_10a35_CONT130", "MULTI_10a35_ABAN150", "FIRST_10a35_MIN150"})

print(f"{len(feitas)} verificações OK" if not falhas else f"FALHAS: {falhas}"); sys.exit(1 if falhas else 0)
