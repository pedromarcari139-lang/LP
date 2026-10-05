# v5.1: REGRESSÃO — tudo o que já existia na v5.0 (funis padrão, EV5/EV10, por minuto, ensemble, sombra, monitor, próximas escolhas) tem de sair
# IDÊNTICO na v5.1 (as baterias de odd mínima são funis a mais, não mudam os outros). Uso: python comparar_v51_v50.py V50/OUT_FUNIL V51/OUT_FUNIL
import os, sys
import numpy as np
import pandas as pd

A, B = sys.argv[1], sys.argv[2]; falhas = []; ODD = ("_CONT", "_ABAN", "_MIN1")
def sem_odd(d):                                                                              # tira o que é NOVO: baterias de odd mínima e o conjunto so_INI_puro
    if "funil" in d.columns: d = d[~d["funil"].astype(str).str.contains("|".join(ODD))]
    if "conjunto" in d.columns: d = d[d["conjunto"].astype(str) != "so_INI_puro"]
    return d
arqs = ["resumo_funil.csv", "trilhas.csv.gz", "apostas_procedimento.csv.gz", "lucro_por_jogo.csv.gz", "comparar_regras.csv", "mcs_reality_check.csv", "sombra.csv",
        "sombra_eventos.csv.gz", "monitor_alarmes.csv", "proximas_escolhas.csv", "ensemble_minutos.csv", "apostas_ensemble.csv.gz", "ev_comparacao_pareada.csv",
        "clv_markout.csv", "distribuicao_acaso.csv", "jogos_comuns.csv.gz", "pareado_2_vs_bloco.csv", "metricas_previsao_resumo.csv", "odds_por_minuto.csv.gz"]
for f in arqs:
    pa, pb = os.path.join(A, f), os.path.join(B, f)
    if not (os.path.exists(pa) and os.path.exists(pb)): print(f"-- {f}: ausente em uma das pastas"); continue
    a, b = sem_odd(pd.read_csv(pa, low_memory=False)), sem_odd(pd.read_csv(pb, low_memory=False))
    cols = [c for c in a.columns if c in b.columns]; extra = sorted(set(b.columns) - set(a.columns))
    a, b = a[cols].reset_index(drop=True), b[cols].reset_index(drop=True)
    if len(a) != len(b): falhas.append(f"{f}: {len(a)} ≠ {len(b)} linhas"); print(f"FALHOU {f}: linhas {len(a)} ≠ {len(b)}"); continue
    dif = []
    for c in cols:
        x, y = a[c], b[c]
        if pd.api.types.is_numeric_dtype(x) and pd.api.types.is_numeric_dtype(y):
            xv, yv = x.values.astype(float), y.values.astype(float)
            if not (np.array_equal(np.isnan(xv), np.isnan(yv)) and np.allclose(np.nan_to_num(xv), np.nan_to_num(yv), rtol=0, atol=1e-12)): dif.append(c)
        elif not x.astype(str).equals(y.astype(str)): dif.append(c)
    print(("OK    " if not dif else "FALHOU") + f" {f}: {len(a)} linhas × {len(cols)} colunas comuns" + (f" · colunas diferentes: {dif[:8]}" if dif else "") + (f" · colunas novas na v5.1: {extra}" if extra else ""))
    if dif: falhas.append(f)
print("RESULTADO:", "IDÊNTICO (tudo da v5.0 igual na v5.1)" if not falhas else f"DIFERENÇAS: {falhas}"); sys.exit(1 if falhas else 0)
