# Teste unitário da camada de REGRAS do validar_funil.py (sem dados, sem motor): matrizes pequenas feitas à mão.
# Uso: python _testes_funil/teste_regras.py   (da pasta do repositório)
import os, sys, importlib.util
import numpy as np
import pandas as pd

sp = importlib.util.spec_from_file_location("vf", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "validar_funil.py"))
VF = importlib.util.module_from_spec(sp); sys.argv = ["x"]; sp.loader.exec_module(VF)
nan = np.nan
# 3 decisões × 4 opções (ordem alfabética A, B, C, D)
El = np.array([[1, 1, 1, 1], [1, 1, 1, 0], [0, 0, 0, 0]], bool)
M = {k: np.full((3, 4), nan) for k in VF.METRICAS}
M["brier_todas"][:] = [[0.20, 0.18, 0.18, 0.10], [0.25, 0.19, 0.30, 0.01], [0.1, 0.1, 0.1, 0.1]]   # menor é melhor; D é o melhor mas não é elegível na decisão 1
M["roi"][:] = [[0.05, -0.01, 0.03, -0.02], [-0.10, -0.05, -0.01, 0.50], [0, 0, 0, 0]]
M["n_apostas"][:] = [[200, 50, 300, 20], [200, 60, 300, 20], [0, 0, 0, 0]]
M["ppg"][:] = [[0.01, 0.02, 0.02, nan], [0.03, 0.01, 0.00, 0.05], [0, 0, 0, 0]]
ok = 0
def confere(nome, obtido, esperado):
    global ok
    assert list(obtido) == list(esperado), f"{nome}: {list(obtido)} ≠ {list(esperado)}"; ok += 1; print(f"OK  {nome}: {list(obtido)}")

confere("simples brier (menor), empate C/B → B (alfabética), D fora na dec 1", VF.aplicar_regra("brier_todas", M, El), [3, 1, -1])
confere("simples ppg (maior), empate B/C → B; D com NaN não conta", VF.aplicar_regra("ppg", M, El), [1, 0, -1])
r = dict(nome="brier_roi>=0", ordenar="brier_todas", filtros=[("roi", ">=", 0.0)])
confere("filtro roi>=0 → melhor brier entre A e C; dec 1 ninguém passa → melhor_sem_filtro (B)", VF.aplicar_regra(dict(r, sem_candidato="melhor_sem_filtro"), M, El), [2, 1, -1])
confere("mesmo filtro, ninguém passa → nao_apostar (−2)", VF.aplicar_regra(dict(r, sem_candidato="nao_apostar"), M, El), [2, -2, -1])
r2 = dict(nome="brier_roi>=0_e_n>=100", ordenar="brier_todas", filtros=[("n_apostas", ">=", 100), ("roi", ">=", 0.0)], sem_candidato="relaxar")
confere("relaxar: dec 0 passa nos 2 (A, C → C); dec 1 tira o roi e fica n>=100 (A, C → A)", VF.aplicar_regra(r2, M, El), [2, 0, -1])
confere("função: sempre a última elegível", VF.aplicar_regra(dict(nome="f", funcao=lambda M_, E_: np.where(E_.any(1), E_.shape[1] - 1 - np.argmax(E_[:, ::-1], 1), -1)), M, El), [3, 2, -1])
try:
    VF.aplicar_regra(dict(nome="ruim", funcao=lambda M_, E_: np.array([3, 3, -1])), M, El); raise SystemExit("deveria ter recusado opção não elegível")
except AssertionError as e:
    ok += 1; print("OK  função que escolhe opção NÃO elegível é recusada:", e)
try:
    VF.aplicar_regra(dict(nome="ruim2", ordenar="brier_todas", filtros=[("nao_existe", ">=", 0)]), M, El); raise SystemExit("deveria ter recusado filtro inválido")
except AssertionError as e:
    ok += 1; print("OK  filtro com métrica inexistente é recusado:", e)
# serie_procedimento: −2 (não apostar) vira lucro 0 e o jogo CONTA; −1 (sem decisão) não conta
G = np.array([10, 11, 12, 13, 14, 15]); D = dict(Lg=np.array([[1., 2, 3, 4, 5, 6], [-1, -1, -1, -1, -1, -1]]), E=np.ones((2, 6, 5)))
lu, nb, esc, seg = VF.serie_procedimento(G, D, [(10, 12), (12, 14), (14, 16)], np.array([1, -2, -1]))
confere("serie_procedimento: jogos avaliados", lu.index, [10, 11, 12, 13]); confere("serie_procedimento: lucro", lu.values, [-1, -1, 0, 0])
# risco_por_aposta: −1 e +1 no MESMO jogo → por jogo soma 0 (DD 0), por aposta o DD é 1
Dr = dict(bets=[dict(g=np.array([10, 10, 11]), t=np.array([5, 10, 5]), L=np.array([-1.0, 1.0, 0.5]))])
r_ = VF.risco_por_aposta(Dr, np.array([0, 0]), np.array([10, 11])); confere("DD por aposta (−1,+1 no jogo 10)", [r_["maxdd_por_aposta"], r_["pior_seq_apostas_perdidas"]], [1.0, 1])
r_ = VF.risco_serie(pd.Series([0.0, 0.5], index=[10, 11]), pd.Series([2, 1], index=[10, 11])); confere("DD por jogo do mesmo caso (esconde a perda)", [r_["maxdd"]], [0.0])
# benchmarks: VIG por dutching perde SEMPRE 1/S − 1; jogo sem nenhuma odd na faixa fica fora; FAVORITO/ZEBRA respeitam a faixa
info = pd.DataFrame(dict(t=[5] * 6, gameid=[10, 10, 11, 11, 12, 12], side=[1, 2] * 3, y=[1, 0, 0, 1, 1, 0], odd_t=[1.8, 2.1, 1.01, 30.0, 8.0, 9.0]))
Gb = np.array([10, 11, 12]); Db = dict(Lg=np.zeros((0, 3)), E=np.zeros((0, 3, 5)))
bn = VF.benchmarks(info, Gb, dict(politica="FIRST", tempos=[5]), dict(ODD_MIN=1.01, ODD_MAX=7.0), Db, [])
vig = bn["VIG_MERCADO (dutching: perde a margem 1/S − 1)"][0]; S10, S11 = 1 / 1.8 + 1 / 2.1, 1 / 1.01 + 1 / 30
confere("VIG dutching jogo 10, 11 = 1/S − 1; jogo 12 (8 e 9 fora da faixa) = 0", list(np.round(vig, 12)), list(np.round([1 / S10 - 1, 1 / S11 - 1, 0.0], 12)))
confere("FAVORITO: 10 ganha 0,8; 11 perde 1 (odd 1,01 na faixa); 12 fora", list(np.round(bn["FAVORITO (todos os jogos)"][0], 12)), [0.8, -1.0, 0.0])
confere("ZEBRA: 10 perde; 11 odd 30 FORA da faixa (sem aposta); 12 fora", list(np.round(bn["ZEBRA (todos os jogos)"][0], 12)), [-1.0, 0.0, 0.0])
print(f"{ok} verificações OK")
