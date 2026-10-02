# v4.6: compara TODOS os arquivos .csv/.csv.gz de duas pastas OUT_FUNIL (mesmas linhas na mesma ordem; texto igual; números com max|Δ|).
# Uso: python comparar_todas_saidas.py PASTA_A PASTA_B [tolerância=0]
# Sai com 1 se algum arquivo faltar, mudar de forma/texto ou passar da tolerância. Ignora config_funil.json, logs e o texto do RESUMO
# (têm data/hora, SHA dos arquivos e versão), que são comparados à parte quando preciso.
import os, sys, numpy as np, pandas as pd
A, B = sys.argv[1], sys.argv[2]; tol = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
arqs = sorted(f for f in os.listdir(A) if f.endswith((".csv", ".csv.gz")))
falhas = []; pior = 0.0
for f in arqs:
    if not os.path.exists(os.path.join(B, f)): print(f"[FALHOU] {f}: não existe em B"); falhas.append(f); continue
    x, y = pd.read_csv(os.path.join(A, f), low_memory=False), pd.read_csv(os.path.join(B, f), low_memory=False)
    if x.shape != y.shape or list(x.columns) != list(y.columns): print(f"[FALHOU] {f}: forma {x.shape} × {y.shape}"); falhas.append(f); continue
    num = [c for c in x.columns if pd.api.types.is_numeric_dtype(x[c]) and pd.api.types.is_numeric_dtype(y[c])]
    txt = [c for c in x.columns if c not in num]
    ok_txt = x[txt].fillna("<NA>").astype(str).equals(y[txt].fillna("<NA>").astype(str))
    xv, yv = x[num].values.astype(float), y[num].values.astype(float)
    nan_ok = bool(np.array_equal(np.isnan(xv), np.isnan(yv)))
    d = float(np.nanmax(np.abs(xv - yv))) if xv.size and np.isfinite(xv).any() else 0.0
    d = 0.0 if np.isnan(d) else d; pior = max(pior, d)
    ok = ok_txt and nan_ok and d <= tol
    print(f"[{'OK ' if ok else 'FALHOU'}] {f}: {x.shape} · texto {'igual' if ok_txt else 'DIFERENTE'} · vazios {'iguais' if nan_ok else 'DIFERENTES'} · max|Δ| numérico {d:.2e}")
    if not ok: falhas.append(f)
so_b = sorted(f for f in os.listdir(B) if f.endswith((".csv", ".csv.gz")) and f not in arqs)
if so_b: print(f"[FALHOU] só em B: {so_b}"); falhas += so_b
print(f"{len(arqs)} arquivos comparados · tolerância {tol:g} · maior |Δ| {pior:.2e}")
print("RESULTADO:", "TUDO OK" if not falhas else "FALHAS: " + ", ".join(falhas)); sys.exit(1 if falhas else 0)
