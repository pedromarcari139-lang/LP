# -*- coding: utf-8 -*-
# Mostra TODAS as opções (modelo × flag) que cada regra escolheu ao longo do tempo, a partir de OUT_FUNIL/trilhas.csv.gz.
# Uso (na pasta do RODAR.bat):  .venv\Scripts\python.exe ver_escolhas.py [funil] [janela] [conjunto] [regra1 regra2 ...]
#   padrão: FIRST_10a35 propria todas bsskill_todas llskill_todas
# Grava também OUT_FUNIL/escolhas_<regra>.csv (uma linha por opção) e OUT_FUNIL/trocas_<regra>.csv (cada troca, com o jogo).
import sys, os, pandas as pd

arg = sys.argv[1:]
funil = arg[0] if len(arg) > 0 else "FIRST_10a35"
janela = arg[1] if len(arg) > 1 else "propria"
conj = arg[2] if len(arg) > 2 else "todas"
regras = arg[3:] or ["bsskill_todas", "llskill_todas"]
pasta = "OUT_FUNIL"
T = pd.read_csv(os.path.join(pasta, "trilhas.csv.gz"))
T = T[(T.funil == funil) & (T.janela == janela) & (T.conjunto == conj) & (T.cadencia.str.startswith("a_cada_"))].sort_values("a")
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500)
for rg in regras:
    t = T[T.regra == rg].copy()
    if not len(t): print(f"\n### {rg}: não encontrada em {funil}/{janela}/{conj}"); continue
    t["escolhida"] = t["escolhida"].fillna("(sem decisão)")
    n_jogos = t["jogos_no_par"].sum()
    g = t.groupby("escolhida").agg(pares=("a", "size"), jogos=("jogos_no_par", "sum"), apostas=("apostas_no_par", "sum"), lucro=("lucro_no_par", "sum"),
                                   primeiro_par=("a", "min"), ultimo_par=("a", "max"))
    g["pct_jogos"] = g["jogos"] / n_jogos; g["ppg_quando_escolhida"] = g["lucro"] / g["jogos"]
    g = g.sort_values("jogos", ascending=False)
    # blocos contínuos (cada vez que a escolha muda)
    t["bloco"] = (t["escolhida"] != t["escolhida"].shift()).cumsum()
    tr = t.groupby("bloco").agg(escolhida=("escolhida", "first"), de_jogo=("a", "min"), ate_jogo=("b", "max"), pares=("a", "size"),
                                jogos=("jogos_no_par", "sum"), lucro=("lucro_no_par", "sum")).reset_index(drop=True)
    tr["ate_jogo"] = tr["ate_jogo"] - 1
    print(f"\n### {rg} · {funil} / janela {janela} / opções {conj} · {len(t)} pares · {int(n_jogos)} jogos · {len(g)} opções usadas · {len(tr) - 1} trocas")
    print(g.assign(pct_jogos=(g.pct_jogos * 100).round(1), lucro=g.lucro.round(1), ppg_quando_escolhida=g.ppg_quando_escolhida.round(4)).to_string())
    print(f"\n  linha do tempo ({len(tr)} trechos; de_jogo–ate_jogo = jogos em que essa opção apostou):")
    print(tr.assign(lucro=tr.lucro.round(1)).to_string(index=False))
    print(f"\n  ÚLTIMA escolha: {t.iloc[-1]['escolhida']} (par {int(t.iloc[-1]['a'])}–{int(t.iloc[-1]['b']) - 1})")
    g.to_csv(os.path.join(pasta, f"escolhas_{rg}.csv")); tr.to_csv(os.path.join(pasta, f"trocas_{rg}.csv"), index=False)
