# -*- coding: utf-8 -*-
"""
analisar_odds_baixas.py — PASSO 0 da teoria "odd baixa no começo (T10/T15) não vale a pena" (v5.1, 04/10/2026).

Lê SÓ as saídas já gravadas pelo validar_funil.py (não roda modelo, não treina, não escolhe nada): o LIVRO de apostas do procedimento
(apostas_procedimento.csv.gz) e a tabela de odds por minuto (odds_por_minuto.csv.gz). É DESCRITIVO: as apostas são as que cada regra
fez com a regra de entrada de hoje; nada aqui decide a regra (quem testa a regra é o funil v5.1, baterias CONT/ABAN/MIN).

Para cada (funil, regra) × faixa de MINUTO × faixa de ODD da aposta:
  n · acerto real × acerto necessário (média de 1/odd) · ROI [IC 95% por cluster de 10 gameids] · EV médio previsto pelo modelo (p·odd − 1)
  · ROI − EV (z): muito negativo = o modelo é otimista nessa faixa ("falso positivo") · CLV de fechamento (odd·q_fech − 1)
  · markout +5/+10/+15 (q(t+h) − q(t), mesmo lado): NEGATIVO = o mercado foi CONTRA a aposta, i.e. a odd SUBIU depois
  · % das apostas em que a odd DO MESMO LADO ficou >= LIMITE em algum minuto POSTERIOR do jogo (até o último minuto apostável) e a mediana da
    maior odd posterior — "teria tido odd melhor esperando?" (usa o futuro do jogo: é só descritivo, NÃO é uma regra que dê para aplicar;
    o modelo pode nem dar sinal nesse momento — quem mede isso é a bateria CONT do funil).

USO: python analisar_odds_baixas.py [PASTA_OUT] [--janela propria] [--conjunto todas] [--regras bsskill_todas,sortino,...] [--todas-regras]
     (PASTA_OUT padrão: OUT_FUNIL_CONTINUACAO se existir, senão OUT_FUNIL) → imprime e grava PASTA_OUT/ODDS_BAIXAS.txt e odds_baixas_diagnostico.csv
"""
import os, sys, argparse
import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("pasta", nargs="?", default=None)
ap.add_argument("--janela", default="propria")
ap.add_argument("--conjunto", default="todas")
ap.add_argument("--regras", default="bsskill_todas,llskill_todas,sortino,roi,ppg,clv_medio")
ap.add_argument("--todas-regras", action="store_true")
ap.add_argument("--funis", default="FIRST_10a35,MULTI_10a35")
ap.add_argument("--limites", default="1.30,1.50")
a = ap.parse_args()
PASTA = a.pasta or ("OUT_FUNIL_CONTINUACAO" if os.path.isdir("OUT_FUNIL_CONTINUACAO") else "OUT_FUNIL")
FAIXAS_ODD = [1.0, 1.30, 1.50, 2.00, 3.00, np.inf]                       # decididas antes de olhar resultado
ROT_ODD = ["<1,30", "1,30–1,50", "1,50–2,00", "2,00–3,00", ">=3,00"]
FAIXAS_MIN = [("T10", [10]), ("T15", [15]), ("T20+", [20, 25, 30, 35])]
HZ = [5, 10, 15]
LIMITES = [float(x) for x in a.limites.split(",")]
MIN_N = 30                                                                 # células com menos apostas: marcadas "pouca amostra"
txt = []
def diz(s=""): print(s, flush=True); txt.append(s)


def media_se(x, g):
    """média e erro-padrão por cluster de 10 gameids (mesma conta do funil)"""
    x = np.asarray(x, float); c = (np.asarray(g) - 1) // 10
    if len(x) < 2: return (float(x.mean()) if len(x) else np.nan), np.nan
    d = pd.DataFrame(dict(x=x, c=c)).groupby("c")["x"].agg(["sum", "size"]); m = d["sum"].sum() / d["size"].sum(); G = len(d)
    return float(m), float(np.sqrt(((d["sum"] - d["size"] * m) ** 2).sum() * G / max(G - 1, 1)) / d["size"].sum())


fp = os.path.join(PASTA, "apostas_procedimento.csv.gz")
if not os.path.exists(fp): sys.exit(f"não achei {fp} — rode o validar_funil.py antes (ou passe a pasta de saída)")
LV = pd.read_csv(fp, low_memory=False)
LV = LV[LV.funil.isin(a.funis.split(",")) & (LV.janela == a.janela) & (LV.conjunto == a.conjunto)]
if not a.todas_regras: LV = LV[LV.regra.isin(a.regras.split(","))]
if not len(LV): sys.exit("nenhuma aposta com esses filtros (funis/janela/conjunto/regras)")
OD = pd.read_csv(os.path.join(PASTA, "odds_por_minuto.csv.gz")) if os.path.exists(os.path.join(PASTA, "odds_por_minuto.csv.gz")) else None
t_max_aposta = int(max(m for _, ms in FAIXAS_MIN for m in ms))
if OD is not None:                                                         # maior odd do MESMO lado em minutos POSTERIORES (até o último minuto apostável)
    lado = lambda s: pd.to_numeric(s, errors="coerce").astype("Int64").astype(str) if pd.to_numeric(s, errors="coerce").notna().all() else s.astype(str)
    OD["_s"] = lado(OD["side"]); LV["_s"] = lado(LV["side"]); OD = OD[OD.t <= t_max_aposta]
    u = LV[["gameid", "_s", "t"]].drop_duplicates().merge(OD[["gameid", "_s", "t", "odd"]].rename(columns={"t": "_t2", "odd": "_o2"}), on=["gameid", "_s"])
    u = u[u._t2 > u.t].groupby(["gameid", "_s", "t"])._o2.max().rename("odd_max_depois").reset_index()
    LV = LV.merge(u, on=["gameid", "_s", "t"], how="left")
LV["faixa_odd"] = pd.cut(LV.odd, FAIXAS_ODD, right=False, labels=ROT_ODD)
LV["faixa_min"] = LV.t.map({m: n for n, ms in FAIXAS_MIN for m in ms})

linhas = []
for (fu, rg), d0 in LV.groupby(["funil", "regra"], sort=False):
    for fm, _ms in FAIXAS_MIN:
        for fo in ROT_ODD:
            d = d0[(d0.faixa_min == fm) & (d0.faixa_odd == fo)]
            if not len(d): continue
            roi, se = media_se(d.lucro, d.gameid); dif, se_d = media_se(d.lucro - d.ev, d.gameid)
            r_ = dict(funil=fu, regra=rg, minuto=fm, faixa_odd=fo, n=len(d), odd_media=d.odd.mean(), acerto=d.y.mean(), acerto_necessario=(1 / d.odd).mean(),
                      roi=roi, roi_ic_lo=roi - 1.96 * se if np.isfinite(se) else np.nan, roi_ic_hi=roi + 1.96 * se if np.isfinite(se) else np.nan,
                      ev_modelo=d.ev.mean(), roi_menos_ev=dif, z_roi_menos_ev=dif / se_d if se_d and se_d > 0 else np.nan,
                      clv_fech=d.clv_fech.mean(), n_com_fech=int(d.clv_fech.notna().sum()), pouca_amostra=len(d) < MIN_N)
            for h in HZ:
                if f"q_mais{h}" in d:
                    mk = d[f"q_mais{h}"] - d.q_devig; r_[f"mk{h}"] = mk.mean(); r_[f"pct_odd_subiu_{h}"] = (mk[mk.notna()] < 0).mean() if mk.notna().any() else np.nan
                    r_[f"n_mk{h}"] = int(mk.notna().sum())
            if "odd_max_depois" in d:
                om = d.odd_max_depois; r_["mediana_maior_odd_depois"] = om.median()
                for L in LIMITES: r_[f"pct_odd_depois_ge_{L:.2f}"] = (om >= L).mean()               # NaN (sem minuto depois) conta como "não teve"
            linhas.append(r_)
RES = pd.DataFrame(linhas); RES.to_csv(os.path.join(PASTA, "odds_baixas_diagnostico.csv"), index=False)

f_ = lambda v, fmt: "n/d" if v is None or (isinstance(v, float) and not np.isfinite(v)) else format(v, fmt)
diz(f"ODDS BAIXAS NO COMEÇO — diagnóstico descritivo do livro de apostas · {os.path.abspath(PASTA)} · janela {a.janela} · opções {a.conjunto} · "
    f"{LV.regra.nunique()} regras · {len(LV)} apostas")
diz("Colunas: n · odd média · acerto real / necessário (1/odd) · ROI [IC 95%] · EV previsto · ROI−EV (z; muito < 0 = modelo otimista = 'falso positivo') · CLV fech · "
    "mk+5/+10/+15 (< 0 = odd SUBIU depois) e % com odd subindo em +5 · % com odd do mesmo lado >= " + "/".join(f"{L:.2f}" for L in LIMITES) +
    " em algum minuto posterior (<= T" + str(t_max_aposta) + ", usa o futuro: só descritivo) · (*) = menos de " + str(MIN_N) + " apostas")
for (fu, rg), x in RES.groupby(["funil", "regra"], sort=False):
    diz(f"\n=== {fu} · regra {rg}")
    for _, r in x.iterrows():
        diz(f"   {r.minuto:<4} odd {r.faixa_odd:<10} n {int(r.n):>5}{'(*)' if r.pouca_amostra else '   '} · odd {r.odd_media:.2f} · acerto {r.acerto:.1%}/{r.acerto_necessario:.1%} · "
            f"ROI {f_(r.roi, '+.3f')} [{f_(r.roi_ic_lo, '+.3f')}; {f_(r.roi_ic_hi, '+.3f')}] · EV {f_(r.ev_modelo, '+.3f')} · ROI−EV {f_(r.roi_menos_ev, '+.3f')} (z {f_(r.z_roi_menos_ev, '+.1f')}) · "
            f"CLV {f_(r.clv_fech, '+.3f')} · mk " + "/".join(f_(r.get(f"mk{h}"), "+.3f") for h in HZ) + f" · odd subiu +5: {f_(r.get('pct_odd_subiu_5'), '.0%')}"
            + (" · odd depois >= " + "/".join(f"{L:.2f}: {f_(r.get(f'pct_odd_depois_ge_{L:.2f}'), '.0%')}" for L in LIMITES) + f" (mediana maior {f_(r.get('mediana_maior_odd_depois'), '.2f')})"
               if "mediana_maior_odd_depois" in r else ""))
diz("\nCOMO LER (e o que NÃO concluir):")
diz(" - Se em T10/T15 com odd < 1,30 ou 1,30–1,50 o ROI−EV for bem negativo (z <= −2) e o ROI < 0, sustenta 'falso positivo' (modelo otimista nas odds baixas do começo).")
diz(" - Se o mk +5/+10 for negativo e o % com odd depois >= limite for alto, sustenta 'a odd sobe e aparece entrada melhor' — MAS só a bateria CONT do funil mostra se o")
diz("   modelo ainda manda entrar nessa odd melhor (o modelo pode deixar de dar sinal). Compare com as faixas de odd maiores e com T20+ (referência).")
diz(" - Muitas células × regras: alguma vai parecer ruim/boa por acaso. As regras compartilham muitas apostas (não são amostras independentes).")
diz(" - Quem DECIDE é a tabela 'ODD MÍNIMA' do RESUMO v5.1 (PPG pareado contra a padrão, nos mesmos jogos), não este diagnóstico.")
open(os.path.join(PASTA, "ODDS_BAIXAS.txt"), "w", encoding="utf-8").write("\n".join(txt))
print(f"\ngravado {os.path.join(PASTA, 'ODDS_BAIXAS.txt')} e odds_baixas_diagnostico.csv")
