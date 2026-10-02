# Gera, a partir dos zz sintéticos (pasta ORIGEM), as variantes da auditoria:
#   sabotado/: jogos >= CORTE com TODAS as colunas trocadas (inclusive odds, flags e resultado), 30% desses jogos APAGADOS (os mesmos em
#              todos os minutos) e TEXTO lixo em células numéricas das linhas do lockbox (>= LOCKBOX)
#   truncado/: só as linhas com gameid < LOCKBOX
# Uso: python gerar_variantes.py ORIGEM DESTINO_SABOTADO DESTINO_TRUNCADO CORTE LOCKBOX
import sys, zlib, numpy as np, pandas as pd
ori, sab, tru, CORTE, L0 = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
apagar = {g for g in range(CORTE, 100000) if zlib.crc32(f"apaga{g}".encode()) % 10 < 3}       # 30% dos jogos >= CORTE, determinístico
manter = ("gameid", "side", "ordem", "timeA", "timeB")
for t in [5, 10, 15, 20, 25, 30, 35, 40, 45]:
    d = pd.read_excel(f"{ori}/zz{t}.xlsx")
    d[d.gameid < L0].to_excel(f"{tru}/zz{t}.xlsx", index=False)
    m = (d.gameid >= CORTE).values
    for c in d.columns:
        if c in manter or not pd.api.types.is_numeric_dtype(d[c]): continue
        h = zlib.crc32(c.encode()) % 997; v = np.sin(d.loc[m, "gameid"].values * 1.7 + d.loc[m, "side"].values * 0.3 + h)
        d[c] = d[c].astype(float)
        if c == "resultado": d.loc[m, c] = np.where(d.loc[m, "side"].values == 1, (v > 0).astype(int), (v <= 0).astype(int))       # resultado novo e coerente nos 2 lados
        elif c.startswith(("odd", "vsodd")): d.loc[m, c] = np.round(1.05 + 6 * np.abs(v), 3)                                           # odds novas (válidas)
        elif c.startswith(("Z", "VSZ")): d.loc[m, c] = np.round(5 + 90 * np.abs(v), 3)
        elif c in ("cWRgrande", "cPROBSgrande") or c.startswith(("bisolo", "bipro", "cWR", "cPROBS", "roles")): d.loc[m, c] = (v > 0).astype(float)
        else: d.loc[m, c] = np.round(50 * v, 3)
    d = d[~d.gameid.isin(apagar)].copy()
    d["Z"] = d["Z"].astype(object); d.loc[d.gameid >= L0, "Z"] = "lixo_texto"                        # texto numa coluna numérica, só no lockbox
    d.to_excel(f"{sab}/zz{t}.xlsx", index=False)
print("variantes geradas")
