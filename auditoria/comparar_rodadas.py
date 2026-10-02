# Compara as saídas de duas rodadas do validar_funil.py (pastas OUT_FUNIL), para os testes de sabotagem e de lockbox da auditoria.
# Uso: python comparar_rodadas.py BASE/OUT_FUNIL OUTRA/OUT_FUNIL CORTE [exato]
#   CORTE: só compara previsões com gameid < CORTE e decisões com a <= CORTE (o que vem depois PODE mudar: é o futuro sabotado)
#   exato: compara também TODOS os números do resumo, das métricas de Brier e das comparações (teste do lockbox: nada pode mudar)
import sys, numpy as np, pandas as pd
A, Bp, C = sys.argv[1], sys.argv[2], int(sys.argv[3]); exato = len(sys.argv) > 4 and sys.argv[4] == "exato"; falhas = []
def checa(nome, ok, det=""):
    print(f"[{'OK ' if ok else 'FALHOU'}] {nome}" + (f" — {det}" if det else "")); falhas.append(nome) if not ok else None
k = ["motor", "t", "cand", "gameid", "side"]
pa, pb = pd.read_csv(f"{A}/previsoes_funil.csv.gz"), pd.read_csv(f"{Bp}/previsoes_funil.csv.gz")
xa, xb = pa[pa.gameid < C].sort_values(k).reset_index(drop=True), pb[pb.gameid < C].sort_values(k).reset_index(drop=True)
mesmo = xa[k].equals(xb[k]); dmax = float((xa.p - xb.p).abs().max()) if mesmo else np.inf
checa(f"previsões com gameid < {C} idênticas (chaves e p)", mesmo and dmax < 1e-9, f"{len(xa)} linhas · max|Δp| = {dmax:.2e}")
ya, yb = pa[pa.gameid >= C], pb[pb.gameid >= C]
print(f"      (controle) previsões com gameid >= {C}: {len(ya)} × {len(yb)} linhas")
ta, tb = pd.read_csv(f"{A}/trilhas.csv.gz"), pd.read_csv(f"{Bp}/trilhas.csv.gz")
kk = ["funil", "janela", "conjunto", "regra", "cadencia", "a", "b"]
m = ta.merge(tb, on=kk, suffixes=("_1", "_2"), how="outer", indicator=True); an = m[m.a <= C]
iguais = (an._merge == "both") & (an.escolhida_1.fillna("-") == an.escolhida_2.fillna("-")) & (an.opcoes_elegiveis_1 == an.opcoes_elegiveis_2) \
         & np.isclose(an.valor_ordenacao_1.fillna(0), an.valor_ordenacao_2.fillna(0), atol=1e-9) & np.isclose(an.ppg_historico_escolhida_1.fillna(0), an.ppg_historico_escolhida_2.fillna(0), atol=1e-9)
checa(f"decisões com a <= {C}: escolha, nº de elegíveis e valores idênticos (todas as regras, funis, janelas, conjuntos, cadências)", bool(iguais.all()), f"{len(an)} decisões · {int((~iguais).sum())} diferentes")
dp = m[(m.a > C) & (m._merge == "both")]
print(f"      (controle) decisões com a > {C} cujo valor mudou: {int((~np.isclose(dp.ppg_historico_escolhida_1.fillna(0), dp.ppg_historico_escolhida_2.fillna(0))).sum())} de {len(dp)}")
if exato:
    for f, chave in (("resumo_funil.csv", ["funil", "janela", "conjunto", "regra", "cadencia"]), ("metricas_previsao_resumo.csv", ["modelo", "flag", "recorte", "periodo", "t"]),
                     ("comparar_regras.csv", ["funil", "janela", "conjunto", "regra_1", "regra_2"]), ("distribuicao_acaso.csv", ["funil", "janela", "conjunto"])):
        x, y = pd.read_csv(f"{A}/{f}").sort_values(chave).reset_index(drop=True), pd.read_csv(f"{Bp}/{f}").sort_values(chave).reset_index(drop=True)
        num = [c for c in x.columns if pd.api.types.is_numeric_dtype(x[c])]
        ok = x.shape == y.shape and x.drop(columns=num).fillna("").astype(str).equals(y.drop(columns=num).fillna("").astype(str))
        d = float(np.nanmax(np.abs(x[num].values.astype(float) - y[num].values.astype(float)))) if ok else np.inf
        checa(f"{f} idêntico", ok and d < 1e-9, f"{x.shape} · max|Δ| numérico {d:.2e}")
print("RESULTADO:", "TUDO OK" if not falhas else "FALHAS: " + ", ".join(falhas)); sys.exit(1 if falhas else 0)
