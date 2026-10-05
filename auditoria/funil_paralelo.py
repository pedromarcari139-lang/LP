# -*- coding: utf-8 -*-
"""
funil_paralelo.py — IMPLEMENTAÇÃO PARALELA E INDEPENDENTE do funil (v5.1), para conferir o validar_funil.py.

NÃO importa o validar_funil.py, o auditar_saidas.py nem o motor. Lê só:
  - os zz{t}.xlsx crus (gameid, side, resultado, odd{t}, vsodd{t}, flags) — e zz40/zz45 se existirem (só odds, para o fechamento);
  - previsoes_funil.csv.gz e ajustes_funil.csv.gz (as PREVISÕES walk-forward e o log de cada treino) e config_funil.json da pasta de saída.
Refaz TUDO o que vem depois das previsões, com código escrito de outro jeito (máscaras diretas, sem somas acumuladas):
  P1  walk-forward do TREINO: toda previsão (minuto, modelo, jogo) foi feita por um treino que terminou ANTES do par do jogo, par de <= 2 jogos
  P2  universo de jogos executáveis e opções (modelo × flag, início w) == universo.csv / opcoes.csv
  P3  APOSTAS de cada opção em TODOS os funis (FIRST/MULTI, EV 0/5/10, por minuto, baterias de odd mínima CONT/ABAN/MIN) == lucro_por_jogo.csv.gz
  P4  ESCOLHA de cada regra (16 simples + clv_medio, clv_mk, clv_mk_com_skill) em decisões sorteadas de cada funil × janela × conjunto, só com
      jogos < início do par − embargo, == trilhas.csv.gz (e a próxima escolha == proximas_escolhas.csv)
  P5  VAZAMENTO na própria implementação: as mesmas decisões com TUDO de jogo >= início do par trocado por lixo → mesma escolha
  P6  PPG de cada procedimento (histórico inteiro e só >= início do draft) refeito com as minhas apostas e as escolhas da trilha == resumo_funil.csv;
      Δ pareado das baterias de odd mínima e dos limiares de EV == odd_/ev_comparacao_pareada.csv
USO: python funil_paralelo.py PASTA_OUT PASTA_DOS_ZZ [--decisoes 25] [--semente 7]
"""
import os, sys, json, re, argparse
import numpy as np
import pandas as pd

ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("zz"); ap.add_argument("--decisoes", type=int, default=25); ap.add_argument("--semente", type=int, default=7)
ap.add_argument("--vazar", type=int, default=0, help="CONTROLE NEGATIVO: deixa a escolha ver N jogos DEPOIS do início do par (tem de FALHAR no P4)")
A = ap.parse_args(); OUT, ZZ = A.out, A.zz; rng = np.random.default_rng(A.semente)
cfg = json.load(open(os.path.join(OUT, "config_funil.json"), encoding="utf-8"))
L0, H0, EMB, MINH, MINL = int(cfg["lockbox_serio"]), int(cfg["hist_inicio"]), int(cfg["embargo_escolha"]), int(cfg["min_hist"]), int(cfg["min_linhas"])
TEMPOS = [int(t) for t in cfg["tempos"]]; FINI = {k: int(v) for k, v in cfg["flag_inicio"].items()}; FLAGS = list(FINI); IDRAFT = int(cfg["inicio_opcoes_draft"])
ODR = {r["suf"]: r for r in cfg.get("odd_regras") or []}; ODD_MIN, ODD_MAX = 1.01, 7.0
FUNIS = list(cfg["funis"]) + list(cfg.get("funis_minuto") or []) + list(cfg.get("funis_odd") or [])
SENT = {"bsskill_todas": 1, "llskill_todas": 1, "brier_todas": -1, "ll_todas": -1, "bsskill_entradas": 1, "llskill_entradas": 1, "brier_entradas": -1, "ll_entradas": -1,
        "ppg": 1, "lucro": 1, "roi": 1, "sharpe": 1, "sortino": 1, "calmar": 1, "maxdd": -1, "seq_ruim": -1, "clv_medio": 1, "clv_mk": 1}
REGRAS = list(SENT) + ["clv_mk_com_skill"]
falhas = []; txt = []
def diz(s): print(s, flush=True); txt.append(s)
def checa(nome, ok, det=""): diz(f"[{'OK ' if ok else 'FALHOU'}] {nome}" + (f" — {det}" if det else "")); falhas.append(nome) if not ok else None

# ---------------- dados crus ----------------
def ler_zz(t):
    d = pd.read_excel(os.path.join(ZZ, f"zz{t}.xlsx")); d = d.loc[:, [c for c in d.columns if not str(c).startswith("Unnamed")]]
    d["gameid"] = pd.to_numeric(d["gameid"], errors="coerce"); d = d[d["gameid"].notna() & (d["gameid"] < L0)].copy(); d["gameid"] = d["gameid"].astype(np.int64)
    return d
INFO = []
for t in TEMPOS:
    d = ler_zz(t); oa, ob = pd.to_numeric(d[f"odd{t}"], errors="coerce"), pd.to_numeric(d[f"vsodd{t}"], errors="coerce")
    boa = np.isfinite(oa) & np.isfinite(ob) & (oa > 1) & (ob > 1)
    x = pd.DataFrame(dict(t=t, gameid=d["gameid"].values, side=pd.to_numeric(d["side"]).astype(int).values, y=pd.to_numeric(d["resultado"]).astype(int).values,
                          odd=oa.values, q=np.where(boa, (1 / oa) / (1 / oa + 1 / ob), np.nan)))
    for F in FLAGS: x[F] = np.where(x["gameid"] >= FINI[F], pd.to_numeric(d[F], errors="coerce").values, np.nan)
    INFO.append(x)
INFO = pd.concat(INFO, ignore_index=True)
PR = pd.read_csv(os.path.join(OUT, "previsoes_funil.csv.gz")); PR = PR[PR.motor == "sujo"]
DRAFT_RE = r"^(TUDO|DRAFT)"                                                                  # modelos COM draft (famílias TUDO*/DRAFT*): só viram opção a partir do início do draft
PR_TODAS = PR.copy(); PR = PR[~(PR.cand.str.match(DRAFT_RE) & (PR.gameid < IDRAFT))]          # (premissa minha, conferida pelo P2: w desses modelos == opcoes.csv)
AJ = pd.read_csv(os.path.join(OUT, "ajustes_funil.csv.gz"), low_memory=False)

# ---------------- P1: walk-forward do treino ----------------
aj = AJ[(AJ.status == "OK") & (AJ.motor == "sujo")].copy()
for c in ("gid_treino_max", "gid_teste_min", "gid_teste_max"): aj[c] = pd.to_numeric(aj[c]).astype(np.int64)
pr = PR_TODAS[["t", "cand", "gameid"]].drop_duplicates().copy(); pr["cb"] = pr.cand.str.replace(r"_(CALR|CAL)$", "", regex=True)
m = pr.merge(aj[["t", "cand", "gid_treino_max", "gid_teste_min", "gid_teste_max"]].rename(columns={"cand": "cb"}), on=["t", "cb"])
m = m[(m.gameid >= m.gid_teste_min) & (m.gameid <= m.gid_teste_max)]
cnt = m.groupby(["t", "cand", "gameid"]).size()
sem = len(pr) - len(cnt); dup = int((cnt > 1).sum())
checa("P1 cada previsão (minuto, modelo, jogo) tem exatamente 1 treino que a cobre; o treino terminou ANTES do 1º jogo que ele prevê; prevê <= 2 jogos; nada >= lockbox",
      sem == 0 and dup == 0 and bool((m.gid_treino_max < m.gid_teste_min).all()) and bool(((m.gid_teste_max - m.gid_teste_min) <= 1).all()) and int(PR_TODAS.gameid.max()) < L0,
      f"{len(pr)} (minuto, modelo, jogo) · sem treino {sem} · com 2+ treinos {dup} · folga mínima treino→teste {int((m.gid_teste_min - m.gid_treino_max).min())} jogo(s)")

# ---------------- base de apostas (preds × odds × resultado) ----------------
B = PR.merge(INFO, on=["t", "gameid", "side"], how="left")
B = B[np.isfinite(B.p) & (B.q > 0) & (B.q < 1) & np.isfinite(B.odd) & (B.odd > 1)].copy()
B["edge"] = B.p - 1 / B.odd; B["ev"] = B.p * B.odd - 1
B["bsm"], B["bsk"] = (B.p - B.y) ** 2, (B.q - B.y) ** 2
B["llm"] = -(B.y * np.log(np.clip(B.p, 1e-6, 1)) + (1 - B.y) * np.log(np.clip(1 - B.p, 1e-6, 1)))
qc = B.q.clip(1e-6, 1 - 1e-6); B["llk"] = -(B.y * np.log(qc) + (1 - B.y) * np.log(1 - qc))

# ---------------- P2: universo e opções ----------------
U = set()
for t in TEMPOS:
    x = INFO[(INFO.t == t) & (INFO.gameid >= H0)]
    for g, d in x.groupby("gameid"):
        if d.side.nunique() == 2 and bool((np.isfinite(d.odd) & (d.odd > 1)).all()) and bool(d.odd.between(ODD_MIN, ODD_MAX).any()): U.add(int(g))
G = np.array(sorted(U), dtype=np.int64); Gs = set(U)
U_rep = np.sort(pd.read_csv(os.path.join(OUT, "universo.csv.gz")).gameid.values)
ini_mod = B[B.gameid.isin(Gs)].groupby("cand").gameid.min()
ops = []
for c, g0 in ini_mod.items():
    for F in [None] + FLAGS:
        w = max(int(g0), FINI[F] if F else 0, H0)
        if w < L0: ops.append(dict(nome=c if F is None else f"{c}|{F}", cand=c, flag=F, w=w))
ops = sorted(ops, key=lambda o: o["nome"]); NOMES = [o["nome"] for o in ops]
O_rep = pd.read_csv(os.path.join(OUT, "opcoes.csv")).sort_values("nome")
checa("P2 universo de jogos executáveis e opções (modelo × flag, início w) refeitos dos zz == universo.csv / opcoes.csv",
      np.array_equal(G, U_rep) and NOMES == O_rep.nome.tolist() and [o["w"] for o in ops] == O_rep.w.astype(int).tolist(),
      f"{len(G)} jogos · {len(ops)} opções")

# ---------------- fechamento (CLV) ----------------
OD = INFO[np.isfinite(INFO.q) & (INFO.q > 0) & (INFO.q < 1) & np.isfinite(INFO.odd) & (INFO.odd > 1)][["gameid", "side", "t", "q"]]
for t in (40, 45):
    fp = os.path.join(ZZ, f"zz{t}.xlsx")
    if os.path.exists(fp) and t not in TEMPOS:
        d = ler_zz(t); d = d[d.groupby("gameid")["side"].transform("nunique") == 2]
        oa, ob = pd.to_numeric(d[f"odd{t}"], errors="coerce"), pd.to_numeric(d[f"vsodd{t}"], errors="coerce"); boa = np.isfinite(oa) & np.isfinite(ob) & (oa > 1) & (ob > 1)
        x = pd.DataFrame(dict(gameid=d.gameid.values, side=pd.to_numeric(d.side).astype(int).values, t=t, q=np.where(boa, (1 / oa) / (1 / oa + 1 / ob), np.nan)))
        OD = pd.concat([OD, x[np.isfinite(x.q) & (x.q > 0) & (x.q < 1)]], ignore_index=True)
FECH = OD.sort_values("t").groupby(["gameid", "side"]).tail(1).set_index(["gameid", "side"])[["t", "q"]].rename(columns={"t": "tf", "q": "qf"})

# ---------------- P3: apostas de cada opção em cada funil ----------------
def sinais(c, F, tempos, ev_min):
    """todos os (jogo, minuto) em que a opção passa a regra de hoje (+ limiar de EV): lado de MAIOR edge entre os 2 lados (diferentes) previstos"""
    b = B[(B.cand == c) & B.t.isin(tempos)].sort_values(["gameid", "t", "edge"], ascending=[True, True, False], kind="mergesort")
    k = b.groupby(["gameid", "t"]).side.agg(["size", "nunique"]); ok2 = k[(k["size"] == 2) & (k["nunique"] == 2)].index
    b = b.set_index(["gameid", "t"]).loc[lambda d: d.index.isin(ok2)].reset_index()
    e = b.groupby(["gameid", "t"], sort=False).head(1)                                       # lado escolhido (maior edge)
    ok = e.odd.between(ODD_MIN, ODD_MAX) & (e.edge > 0)
    if F: ok &= (e[F] == 1)
    if ev_min is not None: ok &= (e.ev > float(ev_min))
    return e[ok].sort_values(["gameid", "t"], kind="mergesort")
def barato(d, r): return (d.odd.values < float(r["odd_min"])) & (d.t.isin(r["minutos"]).values if r["minutos"] else np.ones(len(d), bool))
def apostas_op(op, fu):
    s = sinais(op["cand"], op["flag"], fu["tempos"], fu.get("ev_min")); r = ODR.get(fu.get("odd_regra"))
    if r is not None and r["modo"] == "continua": s = s[~barato(s, r)]
    if r is not None and r["modo"] == "abandona":
        tb = s[barato(s, r)].groupby("gameid").t.min(); s = s[~(s.t.values >= s.gameid.map(tb).fillna(np.inf).values)]
    if fu["politica"] == "FIRST": s = s.groupby("gameid", sort=False).head(1)
    s = s[s.gameid.isin(Gs) & (s.gameid >= op["w"])].copy(); s["L"] = np.where(s.y == 1, s.odd - 1, -1.0)
    f_ = FECH.reindex(pd.MultiIndex.from_arrays([s.gameid.values, s.side.values])); tem = (f_.tf.values > s.t.values)
    s["clv"] = np.where(tem, s.odd.values * f_.qf.values - 1, np.nan); s["mkf"] = np.where(tem, f_.qf.values - s.q.values, np.nan)
    return s
FLGANY = INFO.groupby(["t", "gameid"])[FLAGS].max().eq(1)
def linhas_op(op, fu):
    b = B[(B.cand == op["cand"]) & B.t.isin(fu["minutos_criterio"]) & B.gameid.isin(Gs) & (B.gameid >= op["w"])]
    l = b.groupby(["t", "gameid"])[["bsm", "bsk", "llm", "llk"]].mean().reset_index()
    if op["flag"]: l = l[FLGANY.reindex(pd.MultiIndex.from_arrays([l.t.values, l.gameid.values]))[op["flag"]].fillna(False).values]
    return l
LJ = pd.read_csv(os.path.join(OUT, "lucro_por_jogo.csv.gz"))
DADOS, e3, n3 = {}, [], 0
for fu in FUNIS:
    lj = LJ[LJ.funil == fu["nome"]].set_index(["opcao", "gameid"])
    meu = []
    for op in ops:
        s = apostas_op(op, fu); DADOS[(fu["nome"], op["nome"])] = (s, linhas_op(op, fu))
        if len(s): meu.append(s.groupby("gameid").L.agg(["sum", "size"]).assign(opcao=op["nome"]).reset_index())
    meu = pd.concat(meu, ignore_index=True).set_index(["opcao", "gameid"]) if meu else pd.DataFrame(columns=["sum", "size"])
    n3 += len(meu); j = meu.join(lj, how="outer")
    if j[["sum", "lucro"]].isna().any(axis=None) or (j["sum"] - j["lucro"]).abs().max() > 1e-9 or (j["size"] != j["n_apostas"]).any():
        e3.append(f"{fu['nome']}: {int(j[['sum', 'lucro']].isna().any(axis=1).sum())} (opção, jogo) só de um lado; dif máx {(j['sum'] - j['lucro']).abs().max():.2e}")
checa("P3 apostas de CADA opção em TODOS os funis (FIRST/MULTI, EV 0/5/10, por minuto, odd mínima CONT/ABAN/MIN) refeitas dos zz + previsões == lucro_por_jogo.csv.gz",
      not e3, f"{len(FUNIS)} funis × {len(ops)} opções · {n3} (opção, jogo) com aposta" + ("; " + "; ".join(e3[:4]) if e3 else ""))

# ---------------- P4/P5: escolha de cada regra, só com o passado ----------------
def metricas(s, l, w, a_h):
    """métricas da opção na janela [w, a_h) — conta DIRETA por máscara (sem somas acumuladas)"""
    nu = int(((G >= w) & (G < a_h)).sum()); sb = s[(s.gameid >= w) & (s.gameid < a_h)]; sl = l[(l.gameid >= w) & (l.gameid < a_h)]
    L = sb.L.values; nb, nr = len(L), len(sl); V = {k: np.nan for k in SENT}
    V["lucro"] = float(L.sum()); V["ppg"] = L.sum() / nu if nu > 0 else np.nan
    if nr >= MINL:
        V.update(brier_todas=sl.bsm.mean(), ll_todas=sl.llm.mean(), bsskill_todas=(sl.bsk - sl.bsm).mean(), llskill_todas=(sl.llk - sl.llm).mean())
    if nb >= MINL:
        p, y, q = sb.p.values, sb.y.values, sb.q.values
        bm, bk = (p - y) ** 2, (q - y) ** 2; lm = -(y * np.log(np.clip(p, 1e-6, 1)) + (1 - y) * np.log(np.clip(1 - p, 1e-6, 1))); lk = -(y * np.log(q) + (1 - y) * np.log(1 - q))
        roi = L.mean(); sd = L.std(ddof=1); dn = np.sqrt(np.mean(np.minimum(L, 0) ** 2))
        c = np.cumsum(L); mdd = float(np.max(np.maximum.accumulate(np.r_[0.0, c])[1:] - c))
        run = best = 0
        for v in L: run = run + 1 if v < 0 else 0; best = max(best, run)
        V.update(brier_entradas=bm.mean(), ll_entradas=lm.mean(), bsskill_entradas=(bk - bm).mean(), llskill_entradas=(lk - lm).mean(), roi=roi,
                 sharpe=roi / sd if sd > 0 else np.nan, sortino=roi / dn if dn > 0 else np.nan, maxdd=mdd, seq_ruim=float(best), calmar=L.sum() / mdd if mdd > 0 else np.nan)
    cv = sb.clv.values; tc = np.isfinite(cv)
    if tc.sum() >= MINL: V["clv_medio"] = cv[tc].mean(); V["clv_mk"] = sb.mkf.values[tc].mean()
    return V
def escolher(fu_nome, jan, conj, a, lixo=False):
    a_h = a - EMB + A.vazar                                                                     # A.vazar > 0 só no controle negativo
    el = [i for i, o in enumerate(ops) if o["w"] < a_h and int(((G >= o["w"]) & (G < a_h)).sum()) >= MINH and (conj == "todas" or (conj == "so_INI" and re.search(r"_INI(_|$)", o["cand"])) or (conj == "so_INI_puro" and re.search(r"_INI$", o["cand"])))]
    if not el: return {r: None for r in REGRAS}
    wc = max(ops[i]["w"] for i in el)
    V = {}
    for i in el:
        s, l = DADOS[(fu_nome, ops[i]["nome"])]
        if lixo:                                                                                 # P5: tudo de jogo >= a_h vira lixo (resultado, lucro, previsão, CLV)
            s = s.copy(); l = l.copy(); fs, fl = s.gameid >= a_h, l.gameid >= a_h
            s.loc[fs, ["L", "p", "clv", "mkf"]] = rng.normal(0, 9, size=(int(fs.sum()), 4)); s.loc[fs, "y"] = 1 - s.loc[fs, "y"]
            l.loc[fl, ["bsm", "bsk", "llm", "llk"]] = rng.normal(0, 9, size=(int(fl.sum()), 4))
        V[i] = metricas(s, l, ops[i]["w"] if jan == "propria" else wc, a_h)
    out = {}
    for r in REGRAS:
        met = "clv_mk" if r == "clv_mk_com_skill" else r; sg = SENT[met]
        cand = [i for i in el if np.isfinite(V[i][met])]
        if r == "clv_mk_com_skill":
            f_ = [i for i in cand if np.isfinite(V[i]["bsskill_todas"]) and V[i]["bsskill_todas"] > 0]; cand = f_ or cand
        if not cand: out[r] = None; continue
        best = max(V[i][met] * sg for i in cand); out[r] = NOMES[min(i for i in cand if V[i][met] * sg >= best - 1e-12)]
    return out
T = pd.read_csv(os.path.join(OUT, "trilhas.csv.gz")); t2 = T[T.cadencia == f"a_cada_{cfg['PASSO_ESCOLHA']}"]
PX = pd.read_csv(os.path.join(OUT, "proximas_escolhas.csv"))
e4, e5, n4 = [], 0, 0
for fu in FUNIS:
    for (jan, conj), tt in t2[(t2.funil == fu["nome"]) & t2.regra.isin(REGRAS)].groupby(["janela", "conjunto"]):
        decs = np.sort(tt.a.unique()); pick = rng.choice(decs, size=min(A.decisoes, len(decs)), replace=False)
        for a in list(pick) + [None]:
            if a is None:                                                                        # a PRÓXIMA escolha (depois do último par)
                px = PX[(PX.funil == fu["nome"]) & (PX.janela == jan) & (PX.conjunto == conj) & PX.regra.isin(REGRAS)]
                if not len(px): continue
                a = int(px.decisao_para_jogo.iloc[0]); rep = px.set_index("regra").proxima_escolha
            else: rep = tt[tt.a == a].set_index("regra").escolhida
            meu = escolher(fu["nome"], jan, conj, int(a))
            for r in REGRAS:
                if r not in rep.index: continue
                g_ = None if pd.isna(rep[r]) else rep[r]; n4 += 1
                if meu[r] != g_: e4.append(f"{fu['nome']}/{jan}/{conj}/{r}/a={a}: paralelo {meu[r]} ≠ funil {g_}")
            if rng.random() < 0.25: e5 += int(escolher(fu["nome"], jan, conj, int(a), lixo=True) != meu)
checa("P4 escolha de cada regra refeita SÓ com jogos < início do par − embargo (conta direta, independente) == trilhas e próxima escolha", not e4,
      f"{n4} escolhas conferidas" + ("; " + "; ".join(e4[:5]) if e4 else ""))
checa("P5 (vazamento) as mesmas decisões com tudo de jogo >= início do par trocado por lixo dão a MESMA escolha", e5 == 0, f"{e5} decisões mudaram")

# ---------------- P6: PPG do procedimento e Δ pareados ----------------
R = pd.read_csv(os.path.join(OUT, "resumo_funil.csv")); JC = pd.read_csv(os.path.join(OUT, "jogos_comuns.csv.gz"))
cad = f"a_cada_{cfg['PASSO_ESCOLHA']}"; pd_ = f"desde_{IDRAFT}"
LUC = {k: v[0].groupby("gameid").L.sum() for k, v in DADOS.items()}
SER, e6, n6 = {}, [], 0
for (fn, jan, conj, rg), tt in t2[t2.regra.isin(REGRAS)].groupby(["funil", "janela", "conjunto", "regra"]):
    js = np.sort(JC[(JC.funil == fn) & (JC.janela == jan) & (JC.conjunto == conj)].gameid.values)
    if not len(js): continue
    tt = tt.sort_values("a"); ini = tt.a.values; sg = np.searchsorted(ini, js, side="right") - 1; esc = tt.escolhida.values[sg]
    lc = np.array([0.0 if (e is None or (isinstance(e, float) and np.isnan(e)) or e == "NAO_APOSTA") else float(LUC[(fn, e)].get(g, 0.0)) for g, e in zip(js, esc)])
    s = pd.Series(lc, index=js); SER[(fn, jan, conj, rg)] = s
    r = R[(R.funil == fn) & (R.janela == jan) & (R.conjunto == conj) & (R.regra == rg) & (R.cadencia == cad)]
    if not len(r): continue
    n6 += 1; r = r.iloc[0]
    if abs(s.mean() - r.ppg) > 1e-9: e6.append(f"{fn}/{jan}/{conj}/{rg}: PPG paralelo {s.mean():+.6f} ≠ {r.ppg:+.6f}")
    if f"ppg_{pd_}" in r.index and abs(s[s.index >= IDRAFT].mean() - r[f"ppg_{pd_}"]) > 1e-9: e6.append(f"{fn}/{jan}/{conj}/{rg}: PPG >= {IDRAFT} paralelo ≠ resumo")
def cmp_pareado(arq, c0, c1):
    if not os.path.exists(os.path.join(OUT, arq)): return 0
    X = pd.read_csv(os.path.join(OUT, arq)); k = 0
    for _, x in X[X.regra.isin(REGRAS)].iterrows():
        a_, b_ = SER.get((x[c0], x.janela, x.conjunto, x.regra)), SER.get((x[c1], x.janela, x.conjunto, x.regra))
        if a_ is None or b_ is None: continue
        k += 1; d_ = (b_ - a_).mean(); dd = (b_ - a_)[a_.index >= IDRAFT].mean()
        if abs(d_ - x.delta) > 1e-9: e6.append(f"{arq} {x[c1]}/{x.regra}: Δ paralelo {d_:+.6f} ≠ {x.delta:+.6f}")
        if f"delta_{pd_}" in x.index and abs(dd - x[f"delta_{pd_}"]) > 1e-9: e6.append(f"{arq} {x[c1]}/{x.regra}: Δ >= {IDRAFT} paralelo ≠ gravado")
    return k
k_odd, k_ev = cmp_pareado("odd_comparacao_pareada.csv", "funil_padrao", "funil_odd"), cmp_pareado("ev_comparacao_pareada.csv", "funil_ev0", "funil_ev")
checa(f"P6 PPG de cada procedimento (inteiro e só >= {IDRAFT}) e Δ pareados (odd mínima, EV) refeitos com as apostas do paralelo == resumo / *_comparacao_pareada.csv",
      not e6 and n6 > 0, f"{n6} procedimentos · {k_odd} Δ de odd mínima · {k_ev} Δ de EV" + ("; " + "; ".join(e6[:5]) if e6 else ""))
diz(f"RESULTADO: {'TUDO IGUAL — a implementação paralela reproduz o funil' if not falhas else 'DIFERENÇAS: ' + ', '.join(falhas)}")
open(os.path.join(OUT, "FUNIL_PARALELO.txt"), "w", encoding="utf-8").write("\n".join(txt)); sys.exit(1 if falhas else 0)
