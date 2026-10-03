# v5.0: compara a rodada BASE com a rodada SABOTADA (jogos >= CORTE com todas as colunas trocadas, 30% apagados, texto lixo no lockbox) nas saídas NOVAS:
# livro de apostas (CLV, fechamento, q em t+h, EV calibrado), eventos do modo sombra, tabela de odds — tudo o que é de jogo < CORTE (ou decisão <= CORTE)
# tem de ser IDÊNTICO. Uso: python comparar_v50_sabotagem.py BASE/OUT_FUNIL SABOTADA/OUT_FUNIL CORTE
import sys, numpy as np, pandas as pd
A, B, C = sys.argv[1], sys.argv[2], int(sys.argv[3]); falhas = []
def checa(nome, ok, det=""):
    print(f"[{'OK ' if ok else 'FALHOU'}] {nome}" + (f" — {det}" if det else "")); falhas.append(nome) if not ok else None
def igual(x, y, chave):
    x = x.sort_values(chave, kind="mergesort").reset_index(drop=True); y = y.sort_values(chave, kind="mergesort").reset_index(drop=True)
    if x.shape != y.shape: return False, f"forma {x.shape} × {y.shape}"
    num = [c for c in x.columns if pd.api.types.is_numeric_dtype(x[c]) and pd.api.types.is_numeric_dtype(y[c])]
    t_ok = x.drop(columns=num).fillna("<NA>").astype(str).equals(y.drop(columns=num).fillna("<NA>").astype(str))
    xv, yv = x[num].values.astype(float), y[num].values.astype(float)
    n_ok = np.array_equal(np.isnan(xv), np.isnan(yv)); d = float(np.nanmax(np.abs(xv - yv))) if np.isfinite(xv).any() else 0.0
    return t_ok and n_ok and d < 1e-9, f"{x.shape} · max|Δ| {d:.2e} · texto {'igual' if t_ok else 'DIFERENTE'} · vazios {'iguais' if n_ok else 'DIFERENTES'}"
la, lb = pd.read_csv(f"{A}/apostas_procedimento.csv.gz", low_memory=False), pd.read_csv(f"{B}/apostas_procedimento.csv.gz", low_memory=False)
la, lb = la[la.par_a < C].drop(columns=["lucro_acumulado"]), lb[lb.par_a < C].drop(columns=["lucro_acumulado"])
ok, det = igual(la, lb, ["funil", "janela", "conjunto", "regra", "gameid", "t"])
checa(f"livro de apostas dos pares < {C} idêntico (inclui CLV, fechamento, q em t+5…t+25, EV e EV calibrado)", ok, det)
print(f"      (controle) apostas nos pares >= {C}: {int((pd.read_csv(f'{A}/apostas_procedimento.csv.gz', usecols=['par_a']).par_a >= C).sum())} × {int((pd.read_csv(f'{B}/apostas_procedimento.csv.gz', usecols=['par_a']).par_a >= C).sum())}")
sa, sb = pd.read_csv(f"{A}/sombra_eventos.csv.gz"), pd.read_csv(f"{B}/sombra_eventos.csv.gz")
ok, det = igual(sa[sa.a <= C], sb[sb.a <= C], ["funil", "janela", "conjunto", "regra", "criterio", "a", "evento"])
checa(f"eventos do MODO SOMBRA com decisão <= {C} idênticos", ok, det)
oa, ob = pd.read_csv(f"{A}/odds_por_minuto.csv.gz"), pd.read_csv(f"{B}/odds_por_minuto.csv.gz")
ok, det = igual(oa[oa.gameid < C], ob[ob.gameid < C], ["gameid", "side", "t"])
checa(f"tabela de odds dos jogos < {C} idêntica", ok, det)
pa, pb = pd.read_csv(f"{A}/proximas_escolhas.csv"), pd.read_csv(f"{B}/proximas_escolhas.csv")
print(f"      (controle) próximas escolhas iguais: {int((pa.proxima_escolha.fillna('-').values == pb.proxima_escolha.fillna('-').values).sum())} de {len(pa)} (podem mudar: usam o histórico até o último jogo)")
print("RESULTADO:", "TUDO OK" if not falhas else "FALHAS: " + ", ".join(falhas)); sys.exit(1 if falhas else 0)
