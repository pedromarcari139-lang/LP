# Refaz À MÃO, dos zz crus + previsoes_funil.csv.gz, as apostas que a opção escolhida em cada par teria feito (política FIRST/MULTI do motor:
# lado de maior edge = p − 1/odd, edge > 0, odd 1,01–7,00, flag do lado escolhido == 1 a partir do início do flag) e compara com o LIVRO
# (apostas_procedimento.csv.gz). Não usa nenhuma função do validar_funil.py nem do motor.
# Uso: python refazer_livro_a_mao.py PASTA_DOS_ZZ  (com OUT_FUNIL dentro dela)  [regra]  — padrão: ll_todas
import sys, json, numpy as np, pandas as pd
pasta = sys.argv[1]; regra = sys.argv[2] if len(sys.argv) > 2 else "ll_todas"; out = f"{pasta}/OUT_FUNIL"
cfg = json.load(open(f"{out}/config_funil.json", encoding="utf-8")); L0 = cfg["lockbox_serio"]; fl_ini = cfg["flag_inicio"]
LV = pd.read_csv(f"{out}/apostas_procedimento.csv.gz"); T = pd.read_csv(f"{out}/trilhas.csv.gz"); JC = pd.read_csv(f"{out}/jogos_comuns.csv.gz")
P = pd.read_csv(f"{out}/previsoes_funil.csv.gz"); P = P[P.gameid < L0]
zz = {t: pd.read_excel(f"{pasta}/zz{t}.xlsx") for t in cfg["tempos"]}; ok_tudo = True
for fu in cfg["funis"]:
    funil, pol, tempos = fu["nome"], fu["politica"], fu["tempos"]
    tr = T[(T.funil == funil) & (T.janela == "propria") & (T.conjunto == "todas") & (T.regra == regra) & (T.cadencia == f"a_cada_{cfg['PASSO_ESCOLHA']}")].sort_values("a")
    js = set(JC[(JC.funil == funil) & (JC.janela == "propria") & (JC.conjunto == "todas")].gameid); meu = []
    for _, r in tr.iterrows():
        if pd.isna(r.escolhida) or r.escolhida == "NAO_APOSTA": continue
        cand, flag = (r.escolhida.split("|") + [None])[:2]
        for g in range(int(r.a), int(r.b)):
            if g not in js: continue
            for t in tempos:
                d = zz[t][zz[t].gameid == g][["side", "resultado", f"odd{t}", f"vsodd{t}"] + ([flag] if flag else [])]
                if len(d) != 2 or not (np.isfinite(d[f"odd{t}"]).all() and (d[f"odd{t}"] > 1).all() and (d[f"vsodd{t}"] > 1).all()): continue
                pv = P[(P.cand == cand) & (P.t == t) & (P.gameid == g)].merge(d, on="side")
                if len(pv) != 2: continue
                pv = pv.assign(edge=pv.p - 1 / pv[f"odd{t}"]).sort_values("edge", ascending=False); x = pv.iloc[0]
                if x.edge > 0 and 1.01 <= x[f"odd{t}"] <= 7.0 and (flag is None or (g >= fl_ini[flag] and x[flag] == 1)):
                    meu.append((g, t, int(x.side), round(float(x[f"odd{t}"] - 1 if x.resultado == 1 else -1.0), 10)))
                    if pol == "FIRST": break
    lv = LV[(LV.funil == funil) & (LV.janela == "propria") & (LV.conjunto == "todas") & (LV.regra == regra)]
    dele = sorted((int(a), int(b), int(c), round(float(d), 10)) for a, b, c, d in lv[["gameid", "t", "side", "lucro"]].values)
    igual = sorted(meu) == dele; ok_tudo &= igual
    print(f"{funil} regra {regra}: refeitas à mão {len(meu)} · no livro {len(dele)} · IGUAIS: {igual} · trocas de modelo {int((tr.escolhida != tr.escolhida.shift()).sum() - 1)}")
print("RESULTADO:", "TUDO OK" if ok_tudo else "DIFERENTE"); sys.exit(0 if ok_tudo else 1)
