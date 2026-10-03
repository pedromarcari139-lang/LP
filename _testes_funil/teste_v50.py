# Testes unitários das funções NOVAS da v5.0 (sem dados reais, sem motor): valores calculados à mão.
# Uso: python teste_v50.py   (na pasta _testes_funil; carrega ../validar_funil.py como módulo)
import os, sys, importlib.util
import numpy as np
import pandas as pd

sp = importlib.util.spec_from_file_location("vf", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "validar_funil.py"))
VF = importlib.util.module_from_spec(sp); sys.argv = ["x"]; sp.loader.exec_module(VF)
falhas, feitas = [], []
def ok(nome, cond, det=""):
    print(("OK  " if cond else "FALHOU ") + nome + (f": {det}" if det else "")); feitas.append(nome); falhas.append(nome) if not cond else None

# ---------------- 1. CLV / markout ----------------
fr = lambda rows: pd.DataFrame(rows, columns=["gameid", "side", "odd_t", "q_devig_t"])
frames = {10: fr([(1, 1, 2.0, 0.48), (1, 2, 1.9, 0.52), (2, 1, 3.0, 0.32), (2, 2, 1.4, 0.68), (3, 1, 2.5, 0.40), (3, 2, 1.6, 0.60)]),
          15: fr([(1, 1, 1.8, 0.52), (1, 2, 2.1, 0.48)]),                                     # jogo 3 SEM o minuto 15 (falta dado)
          20: fr([(1, 1, 1.7, 0.55), (1, 2, 2.2, 0.45), (2, 1, 3.5, 0.28), (2, 2, 1.3, 0.72), (3, 1, 2.2, 0.44), (3, 2, 1.7, 0.56)])}
extras = {40: pd.DataFrame(dict(gameid=[1, 1], side=[1, 2], odd=[1.6, 2.4], q=[0.60, 0.40]))}
VF.ODDS = VF.montar_odds(frames, extras)
c = VF.clv_campos(np.array([1, 2, 3, 2]), np.array([1, 1, 1, 1]), np.array([10, 10, 10, 20]), np.array([2.0, 3.0, 2.5, 3.5]), np.array([0.48, 0.32, 0.40, 0.28]))
ok("markout +5 jogo 1: mk = 0,52 − 0,48; ROI markout = 2·0,52 − 1; green-up = 2/1,8 − 1", np.allclose([c["mk5"][0], c["ev5"][0], c["gr5"][0]], [0.04, 0.04, 2 / 1.8 - 1]), f"{c['mk5'][0]:.4f} {c['ev5'][0]:.4f} {c['gr5'][0]:.4f}")
ok("markout +5 jogo 3 (sem minuto 15) = NaN; +10 = 0,44 − 0,40", np.isnan(c["mk5"][2]) and np.isclose(c["mk10"][2], 0.04))
ok("fechamento jogo 1 = minuto 40 (só odds): q 0,60 → CLV = 2·0,60 − 1 = 0,20, supera", c["minuto_fech"][0] == 40 and np.isclose(c["clv"][0], 0.2) and c["supera"][0] == 1.0)
ok("fechamento jogo 2 (aposta no 10) = minuto 20: q 0,28 → CLV = 3·0,28 − 1 = −0,16, NÃO supera; bruto: 3,0 < 3,5 → não", c["minuto_fech"][1] == 20 and np.isclose(c["clv"][1], -0.16) and c["supera"][1] == 0.0 and c["supera_bruto"][1] == 0.0)
ok("aposta no minuto 20 do jogo 2 (último minuto do jogo) = SEM fechamento (NaN, minuto −1)", np.isnan(c["clv"][3]) and c["minuto_fech"][3] == -1)
ok("markout +25 do minuto 10 (= minuto 35, não existe) = NaN", np.isnan(c["mk25"]).all())

# ---------------- 2. modo sombra ----------------
G = np.arange(100, 140, dtype=np.int64)                                                    # 40 jogos executáveis
S = np.zeros((2, len(G) + 1)); N = np.zeros((2, len(G) + 1))
N[:, 1:] = np.arange(1, len(G) + 1)[None, :]                                               # 1 linha por jogo para cada opção
S[0, 1:] = np.cumsum(np.full(len(G), 0.01)); S[1, 1:] = np.cumsum(np.full(len(G), 0.02))   # opção 1 melhor (0,02 por jogo × 0,01)
AC = {"brier": (S, N)}
a = np.arange(100, 140, 2); idx = np.where(a < 110, 0, 1)                                  # a regra escolhe 0 até 108, 1 a partir de 110
prod, ev = VF.sombra(idx, a, a, G, AC, "brier", 6, 3)
ok("sombra: produção = 0 até a sombra (iniciada em 110) completar 6 jogos; promoção em 116 (jogos 110–115)", list(prod[a < 116]) == [0] * int((a < 116).sum()) and list(prod[a >= 116]) == [1] * int((a >= 116).sum()),
   f"{list(zip(a.tolist(), prod.tolist()))[3:10]}")
ok("sombra: eventos início_produção 100, início_sombra 110 (desde 110), promovida 116 com 6 jogos", [(e["evento"], e["a"]) for e in ev] == [("inicio_producao", 100), ("inicio_sombra", 110), ("promovida", 116)] and ev[2]["jogos_em_sombra"] == 6)
S2 = S.copy(); S2[1, 1:] = np.cumsum(np.full(len(G), 0.005))                               # agora a opção 1 é PIOR
prod2, ev2 = VF.sombra(idx, a, a, G, {"brier": (S2, N)}, "brier", 6, 3)
ok("sombra: nova pior → rejeitada em 116 e a sombra RECOMEÇA (rejeita de novo em 122, 128…); produção fica 0", (prod2 == 0).all() and [e["a"] for e in ev2 if e["evento"] == "rejeitada"][:3] == [116, 122, 128])
fut = S.copy(); fut[1, 1 + 30:] += 999                                                     # futuro (jogos >= 130) adulterado
prod3, _ = VF.sombra(idx, a, a, G, {"brier": (fut, N)}, "brier", 6, 3)
ok("sombra: adulterar jogos >= 130 não muda a produção dos pares <= 130", np.array_equal(prod3[a <= 130], prod[a <= 130]))

# ---------------- 3. EV calibrado (só apostas ANTERIORES ao par, da mesma opção) ----------------
VF.MIN_CALIB = 2
bt = dict(g=np.array([1, 2, 3, 4, 5]), t=np.array([10] * 5), side=np.array([1] * 5), odd=np.array([2.0] * 5), p=np.array([0.53] * 5), y=np.array([1, 0, 1, 1, 0]),
          q=np.array([0.5] * 5), L=np.array([1.0, -1.0, 1.0, 1.0, -1.0]), edge=np.array([0.03] * 5), clv=np.full(5, np.nan), supera=np.full(5, np.nan))
D = dict(bets=[bt]); js = np.array([4, 5]); A = VF.apostas_serie(D, np.array([0, 0]), js, np.array([4, 4]), {})
ok("EV calibrado das apostas do par 4: usa só os jogos 1–3 (lucro médio (1 − 1 + 1)/3); última usada = jogo 3 < 4", np.allclose(A["ev_cal"], 1 / 3) and (A["calib_ate_jogo"] == 3).all() and (A["calib_n"] == 3).all(),
   f"ev_cal {A['ev_cal']} · até {A['calib_ate_jogo']}")
A2 = VF.apostas_serie(D, np.array([0, 0]), js, np.array([2, 2]), {})
ok("par 2 com MIN_CALIB = 2: só 1 aposta antes → usa o EV do modelo (0,53·2 − 1 = 0,06)", np.allclose(A2["ev_cal"], 0.06) and (A2["calib_n"] == 1).all())

# ---------------- 4. janela rolante ----------------
r = VF.rolling_janela(np.arange(6.0), np.array([1, 2, 3, 4, 5, 6.0]), np.ones(6), 3)
ok("ROI rolante W=3: NaN até a janela encher; depois média dos 3 últimos", np.isnan(r[:2]).all() and np.allclose(r[2:], [2, 3, 4, 5]), f"{r}")

# ---------------- 5. limiar de EV pelo filtro do motor (FIRST) ----------------
VF.CFG = dict(EDGE_MIN=0.0, ODD_MIN=1.01, ODD_MAX=7.0, STAKE=1.0, TEMPOS_DEPLOY=[10, 15])
b = pd.DataFrame(dict(cand="X", t=[10, 10, 15, 15], gameid=[1, 1, 1, 1], side=[1, 2, 1, 2], p=[0.52, 0.48, 0.60, 0.40], odd_t=[2.0, 1.9, 2.0, 1.9], y=[1, 0, 1, 0]))
b["q_raw_t"] = 1 / b.odd_t; b["edge"] = b.p - b.q_raw_t; b["ev"] = b.p * b.odd_t - 1
b["__ev0.05_sem_flag"] = (b.ev > 0.05).astype(int)
x0, _ = VF.apostas(b, "X", "FIRST", tempos=[10, 15]); x5, _ = VF.apostas(b, "X", "FIRST", tempos=[10, 15], filtro_col="__ev0.05_sem_flag")
ok("FIRST EV>0 aposta no minuto 10 (EV 4%); FIRST EV>5% pula o 10 e aposta no 15 (EV 20%)", list(x0.t) == [10] and list(x5.t) == [15] and np.isclose(x5.ev.iloc[0], 0.2))

print(f"{len(feitas)} verificações OK" if not falhas else f"FALHAS: {falhas}"); sys.exit(1 if falhas else 0)
