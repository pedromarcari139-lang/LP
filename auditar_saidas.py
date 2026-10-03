# -*- coding: utf-8 -*-
"""
auditar_saidas.py — AUDITORIA INDEPENDENTE das saídas do validar_funil.py (pasta OUT_FUNIL).

NÃO importa o validar_funil.py nem o motor: lê só os arquivos gravados e refaz as contas com pandas, com código escrito de outro jeito.
Serve para a rodada sintética e para a rodada REAL (é só apontar para a OUT_FUNIL dela).

  A1  nenhum arquivo de saída tem jogo >= lockbox (gameid / a / gid_*), e b <= lockbox
  A2  log de ajustes: todo treino OK termina ANTES do par que prevê, prevê um único par de <= PASSO jogos, 1 treino por (minuto, candidato, par)
  A3  toda previsão (inclusive _CALR) tem o treino do seu par, e esse treino termina antes do par
  A4  trilhas: pares contíguos de <= PASSO_ESCOLHA jogos; nº de opções elegíveis recalculado (opcoes.csv + universo + MIN_HIST + embargo)
      == o gravado; a opção escolhida é elegível
  A5  as regras ppg / lucro / roi recalculadas DO ZERO (lucro_por_jogo + universo) em decisões sorteadas, nas janelas própria e comum,
      dão a MESMA escolha e o mesmo PPG histórico da escolhida que a trilha gravada
  A6  PPG do procedimento (regra ppg, a cada 2) recalculado das trilhas + lucro_por_jogo == o do resumo_funil.csv
  A8  livro de apostas do procedimento (apostas_procedimento.csv.gz) == lucro por par das trilhas == PPG e nº de apostas do resumo
  A10 (v4.8) próxima escolha das regras ppg/lucro/roi recalculada do zero
  A9  (v4.7) ensemble por minuto refeito do livro dos funis T10…T35 == ensemble_minutos.csv e apostas_ensemble.csv.gz
  A11–A15 (v5.0) CLV/markout refeitos da tabela de odds; EV calibrado só com jogos < par; modo sombra refeito dos eventos; ROI/ROI esperado/CLV do resumo
      == livro; limiar de EV respeitado
  A7  tabela de índices de decisões sorteadas: janela do histórico, último jogo do histórico, jogos do par, último jogo de treino da
      previsão usada — para conferir a olho (auditoria_indices.csv)
USO: python auditar_saidas.py [pasta OUT_FUNIL]   → imprime e grava OUT_FUNIL/AUDITORIA_SAIDAS.txt; termina com erro se algo falhar.
"""
import os, sys, re, json, glob
import numpy as np
import pandas as pd

PASTA = sys.argv[1] if len(sys.argv) > 1 else "OUT_FUNIL"
N_AMOSTRA = 300            # decisões sorteadas por (funil, janela, conjunto) no A5
SEMENTE = 20261002
linhas_txt, falhas = [], []


def diz(msg): print(msg, flush=True); linhas_txt.append(msg)


def checa(nome, ok, detalhe=""):
    diz(f"[{'OK ' if ok else 'FALHOU'}] {nome}" + (f" — {detalhe}" if detalhe else ""))
    if not ok: falhas.append(nome)


cfg = json.load(open(os.path.join(PASTA, "config_funil.json"), encoding="utf-8"))
L0, PASSO, PASSO_E, MIN_HIST = int(cfg["lockbox_serio"]), int(cfg["PASSO"]), int(cfg["PASSO_ESCOLHA"]), int(cfg["min_hist"])
EMB = int(cfg.get("embargo_escolha", 0)); MIN_LINHAS = int(cfg["min_linhas"])
diz(f"AUDITORIA INDEPENDENTE de {os.path.abspath(PASTA)} · lockbox {L0} · PASSO {PASSO} · PASSO_ESCOLHA {PASSO_E} · MIN_HIST {MIN_HIST} · embargo {EMB}")

# ---------- A1 ----------
viol = []
for fp in sorted(glob.glob(os.path.join(PASTA, "*.csv")) + glob.glob(os.path.join(PASTA, "*.csv.gz"))):
    try: d = pd.read_csv(fp, low_memory=False)
    except Exception as e: viol.append(f"{os.path.basename(fp)}: ilegível ({e})"); continue
    for c in d.columns:
        if c in ("gameid", "a") or c.startswith("gid_"):
            v = pd.to_numeric(d[c], errors="coerce")
            if (v >= L0).any(): viol.append(f"{os.path.basename(fp)}:{c} máx {int(v.max())}")
        if c == "b":
            v = pd.to_numeric(d[c], errors="coerce")
            if (v > L0).any(): viol.append(f"{os.path.basename(fp)}:b máx {int(v.max())}")
checa("A1 nenhum jogo do lockbox nas saídas", not viol, "; ".join(viol) or f"{len(glob.glob(os.path.join(PASTA, '*.csv*')))} arquivos lidos")

# ---------- blocos / cortes dos pares (das trilhas: segmentos 'bloco_motor' = blocos de avaliação do motor) ----------
T = pd.read_csv(os.path.join(PASTA, "trilhas.csv.gz"))
blk = T[T.cadencia == "bloco_motor"][["a", "b"]].drop_duplicates().sort_values("a").values
cortes = set()
for a, b in blk: cortes |= {int(a), int(b)} | set(range((int(a) // PASSO + 1) * PASSO, int(b), PASSO))
cortes = np.array(sorted(cortes))
par_ini = lambda g: cortes[np.searchsorted(cortes, np.asarray(g), side="right") - 1]

# ---------- A2 / A3 ----------
F = pd.read_csv(os.path.join(PASTA, "ajustes_funil.csv.gz"), low_memory=False); F = F[F.status == "OK"].copy()
for c in ("gid_treino_max", "gid_teste_min", "gid_teste_max"): F[c] = pd.to_numeric(F[c]).astype(np.int64)
F["pa"] = par_ini(F.gid_teste_min.values); F["pb"] = par_ini(F.gid_teste_max.values)
checa("A2a treino termina antes do par (gid_treino_max < início do par)", bool((F.gid_treino_max < F.pa).all()), f"{len(F)} treinos · folga mínima {int((F.pa - F.gid_treino_max).min())}")
checa("A2b cada treino prevê um único par", bool((F.pa == F.pb).all() and ((F.gid_teste_max - F.gid_teste_min) < PASSO).all()))
checa("A2c um treino por (motor, minuto, candidato, par)", not F.duplicated(["motor", "t", "cand", "pa"]).any())
Pv = pd.read_csv(os.path.join(PASTA, "previsoes_funil.csv.gz"), usecols=["motor", "t", "cand", "gameid"])
Pv["cand_b"] = Pv.cand.str.replace(r"_(CALR|CAL)$", "", regex=True); Pv["pa"] = par_ini(Pv.gameid.values)
k = Pv[["motor", "t", "cand_b", "pa"]].drop_duplicates().merge(F[["motor", "t", "cand", "pa", "gid_treino_max"]].rename(columns={"cand": "cand_b"}), how="left")
checa("A3 toda previsão tem o treino do seu par, terminado antes do par", bool(k.gid_treino_max.notna().all() and (k.gid_treino_max < k.pa).all()),
      f"{len(Pv)} previsões · {len(k)} (minuto, candidato, par)")

# ---------- A4 ----------
O = pd.read_csv(os.path.join(PASTA, "opcoes.csv")); U = np.sort(pd.read_csv(os.path.join(PASTA, "universo.csv.gz")).gameid.values)
O = O.sort_values("nome").reset_index(drop=True); W = O.w.values.astype(np.int64); nomes = O.nome.values
conj_f = {"todas": lambda n: True, "so_INI": lambda n: bool(re.search(r"_INI(_|$)", n.split("|")[0]))}
def elegiveis_em(a, conj):
    fim = a - EMB; n_hist = np.searchsorted(U, fim, "left") - np.searchsorted(U, W, "left")
    return (W < fim) & (n_hist >= MIN_HIST) & np.array([conj_f[conj](n) for n in nomes])
t2 = T[T.cadencia == f"a_cada_{PASSO_E}"]
seg = t2[["a", "b"]].drop_duplicates().sort_values("a").values
checa("A4a pares contíguos de <= PASSO_ESCOLHA jogos", bool(((seg[:, 1] - seg[:, 0]) <= PASSO_E).all() and (seg[1:, 0] == seg[:-1, 1]).all()), f"{len(seg)} pares")
amostra_a = np.random.default_rng(SEMENTE).choice(seg[:, 0], size=min(400, len(seg)), replace=False)
erro_el, erro_esc = 0, 0
for a in amostra_a:
    for conj in t2.conjunto.unique():
        el = elegiveis_em(int(a), conj); rows = t2[(t2.a == a) & (t2.conjunto == conj)]
        erro_el += int((rows.opcoes_elegiveis != int(el.sum())).sum())
        esc = rows.escolhida.dropna(); esc = esc[esc != "NAO_APOSTA"]
        erro_esc += int((~esc.isin(set(nomes[el]))).sum())
checa("A4b nº de opções elegíveis recalculado == gravado", erro_el == 0, f"{len(amostra_a)} decisões sorteadas × conjuntos · {erro_el} divergências")
checa("A4c a opção escolhida era elegível na decisão", erro_esc == 0, f"{erro_esc} escolhas não elegíveis")

# ---------- A5 / A6 ----------
LJ = pd.read_csv(os.path.join(PASTA, "lucro_por_jogo.csv.gz"))
R = pd.read_csv(os.path.join(PASTA, "resumo_funil.csv")); JC = pd.read_csv(os.path.join(PASTA, "jogos_comuns.csv.gz"))
pos_u = {g: i for i, g in enumerate(U)}; rng = np.random.default_rng(SEMENTE + 1); res5, res6 = [], []
for fu in LJ.funil.unique():
    lj = LJ[LJ.funil == fu]; Lm = np.zeros((len(nomes), len(U))); Nm = np.zeros((len(nomes), len(U))); idx_o = {n: i for i, n in enumerate(nomes)}
    Lm[lj.opcao.map(idx_o).values, lj.gameid.map(pos_u).values] = lj.lucro.values; Nm[lj.opcao.map(idx_o).values, lj.gameid.map(pos_u).values] = lj.n_apostas.values
    for (jan, conj), tt in t2[t2.funil == fu].groupby(["janela", "conjunto"]):
        decs = tt[["a", "b"]].drop_duplicates().values; pick = decs[rng.choice(len(decs), size=min(N_AMOSTRA, len(decs)), replace=False)]
        for regra in ("ppg", "lucro", "roi"):
            tr = tt[tt.regra == regra].set_index("a")
            if not len(tr): continue
            dif = 0; n_ok = 0
            for a, _b in pick:
                el = elegiveis_em(int(a), conj)
                if not el.any(): continue
                fim = int(a) - EMB; w_ = W.copy() if jan == "propria" else np.full(len(W), W[el].max())
                vals = np.full(len(nomes), np.nan)
                for j in np.where(el)[0]:                                                   # conta DIRETA, opção por opção
                    m = (U >= w_[j]) & (U < fim); s, n_ap, n_j = Lm[j, m].sum(), Nm[j, m].sum(), int(m.sum())
                    vals[j] = s / n_j if regra == "ppg" else (s if regra == "lucro" else (s / n_ap if n_ap >= MIN_LINHAS else np.nan))
                ok_ = el & np.isfinite(vals)
                esc_ind = None
                if ok_.any():
                    best = np.nanmax(np.where(ok_, vals, -np.inf)); esc_ind = nomes[np.where(ok_ & (vals >= best - 1e-12))[0][0]]
                esc_rep = tr.loc[a, "escolhida"]; esc_rep = None if pd.isna(esc_rep) else esc_rep
                if esc_ind != esc_rep: dif += 1
                elif esc_ind is not None and regra == "ppg":
                    j = idx_o[esc_ind]; m = (U >= w_[j]) & (U < fim)
                    if abs(Lm[j, m].sum() / m.sum() - tr.loc[a, "ppg_historico_escolhida"]) > 1e-9: dif += 1
                n_ok += 1
            res5.append((fu, jan, conj, regra, n_ok, dif))
        # A6: PPG do procedimento 'ppg' a cada 2 nos jogos comuns
        tr = tt[tt.regra == "ppg"]; js = JC[(JC.funil == fu) & (JC.janela == jan) & (JC.conjunto == conj)].gameid.values
        ini = tr.a.values; ordem = np.argsort(ini); ini, esc = ini[ordem], tr.escolhida.values[ordem]
        sg = np.searchsorted(ini, js, side="right") - 1
        luc = np.array([0.0 if (e is None or (isinstance(e, float) and np.isnan(e)) or e == "NAO_APOSTA") else Lm[idx_o[e], pos_u[g]] for g, e in zip(js, esc[sg])])
        rep = R[(R.funil == fu) & (R.janela == jan) & (R.conjunto == conj) & (R.regra == "ppg") & (R.cadencia == f"a_cada_{PASSO_E}")].ppg.iloc[0]
        res6.append((fu, jan, conj, float(luc.mean()), float(rep)))
d5 = sum(r[5] for r in res5); n5 = sum(r[4] for r in res5)
checa("A5 regras ppg/lucro/roi recalculadas do zero == trilhas (escolha e PPG histórico)", d5 == 0, f"{n5} decisões conferidas em {len(res5)} combinações · {d5} divergências")
d6 = max(abs(a - b) for (*_, a, b) in res6)
checa("A6 PPG do procedimento 'ppg' recalculado == resumo_funil", d6 < 1e-9, f"{len(res6)} combinações · dif máx {d6:.2e}")

# ---------- A10 (v4.8): PRÓXIMA escolha das regras ppg / lucro / roi recalculada do zero (histórico < decisão; decisão <= lockbox) ----------
fp_px = os.path.join(PASTA, "proximas_escolhas.csv")
if os.path.exists(fp_px):
    PXE = pd.read_csv(fp_px); d10, n10 = [], 0
    if len(PXE) and int(PXE.decisao_para_jogo.max()) > L0: d10.append("decisão depois do lockbox")
    for fu in LJ.funil.unique():
        lj = LJ[LJ.funil == fu]; Lm = np.zeros((len(nomes), len(U))); Nm = np.zeros((len(nomes), len(U))); idx_o = {n: i for i, n in enumerate(nomes)}
        Lm[lj.opcao.map(idx_o).values, lj.gameid.map(pos_u).values] = lj.lucro.values; Nm[lj.opcao.map(idx_o).values, lj.gameid.map(pos_u).values] = lj.n_apostas.values
        for _, r in PXE[(PXE.funil == fu) & PXE.regra.isin(["ppg", "lucro", "roi"])].iterrows():
            a = int(r.decisao_para_jogo); el = elegiveis_em(a, r.conjunto); fim = a - EMB; n10 += 1
            if int(el.sum()) != int(r.opcoes_elegiveis): d10.append(f"{fu}/{r.janela}/{r.conjunto}/{r.regra}: elegíveis {int(el.sum())} ≠ {int(r.opcoes_elegiveis)}"); continue
            w_ = W.copy() if r.janela == "propria" else np.full(len(W), W[el].max() if el.any() else 0)
            vals = np.full(len(nomes), np.nan)
            for j in np.where(el)[0]:
                m = (U >= w_[j]) & (U < fim); s_, n_ap, n_j = Lm[j, m].sum(), Nm[j, m].sum(), int(m.sum())
                vals[j] = s_ / n_j if r.regra == "ppg" else (s_ if r.regra == "lucro" else (s_ / n_ap if n_ap >= MIN_LINHAS else np.nan))
            ok_ = el & np.isfinite(vals); esc = None
            if ok_.any():
                best = np.nanmax(np.where(ok_, vals, -np.inf)); esc = nomes[np.where(ok_ & (vals >= best - 1e-12))[0][0]]
            rep_ = None if pd.isna(r.proxima_escolha) else r.proxima_escolha
            if esc != rep_: d10.append(f"{fu}/{r.janela}/{r.conjunto}/{r.regra}: refeita {esc} ≠ gravada {rep_}")
    checa("A10 próxima escolha (regras ppg/lucro/roi) recalculada do zero == proximas_escolhas.csv; decisão <= lockbox", not d10 and n10 > 0,
          f"{n10} próximas escolhas conferidas" + ("; " + "; ".join(d10[:5]) if d10 else ""))

# ---------- A8: livro de apostas do procedimento == lucro por par das trilhas == resumo final ----------
fp_lv = os.path.join(PASTA, "apostas_procedimento.csv.gz")
if os.path.exists(fp_lv):
    LV = pd.read_csv(fp_lv); erros8, n8 = [], 0
    for (fu, jan, conj, regra), lv in LV.groupby(["funil", "janela", "conjunto", "regra"]):
        n8 += 1; js = JC[(JC.funil == fu) & (JC.janela == jan) & (JC.conjunto == conj)].gameid.values
        r = R[(R.funil == fu) & (R.janela == jan) & (R.conjunto == conj) & (R.regra == regra) & (R.cadencia == f"a_cada_{PASSO_E}")].iloc[0]
        if not set(lv.gameid) <= set(js): erros8.append(f"{fu}/{jan}/{conj}/{regra}: aposta fora dos jogos avaliados")
        if abs(lv.lucro.sum() / len(js) - r.ppg) > 1e-9: erros8.append(f"{fu}/{jan}/{conj}/{regra}: livro {lv.lucro.sum() / len(js):+.6f} ≠ resumo {r.ppg:+.6f}")
        if int(len(lv)) != int(r.n_apostas): erros8.append(f"{fu}/{jan}/{conj}/{regra}: {len(lv)} apostas no livro ≠ {int(r.n_apostas)} no resumo")
        if abs(lv.lucro_acumulado.iloc[-1] - lv.lucro.sum()) > 1e-9: erros8.append(f"{fu}/{jan}/{conj}/{regra}: acumulado final ≠ soma")
        tr = t2[(t2.funil == fu) & (t2.janela == jan) & (t2.conjunto == conj) & (t2.regra == regra)].set_index("a")
        por_par = lv.groupby("par_a").lucro.sum(); dif = (tr.lucro_no_par.reindex(por_par.index) - por_par).abs().max()
        if not (dif < 1e-9 and abs(tr.lucro_no_par.sum() - lv.lucro.sum()) < 1e-9): erros8.append(f"{fu}/{jan}/{conj}/{regra}: lucro por par das trilhas ≠ livro (dif {dif:.2e})")
        esc_lv = lv.groupby("par_a").opcao.first(); esc_tr = tr.escolhida.reindex(esc_lv.index)
        if not (esc_lv == esc_tr).all(): erros8.append(f"{fu}/{jan}/{conj}/{regra}: opção do livro ≠ escolhida na trilha")
    checa("A8 livro de apostas == lucro por par das trilhas == PPG e nº de apostas do resumo; aposta só da opção escolhida no par", not erros8,
          f"{n8} séries (regra × funil × janela × conjunto) · {len(LV)} apostas" + ("; " + "; ".join(erros8[:5]) if erros8 else ""))
else:
    checa("A8 livro de apostas presente", False, "apostas_procedimento.csv.gz não existe")

# ---------- A9 (v4.7): ENSEMBLE por minuto refeito do LIVRO dos funis T10…T35 ----------
fp_en = os.path.join(PASTA, "ensemble_minutos.csv")
if os.path.exists(fp_en) and os.path.exists(fp_lv):
    EN = pd.read_csv(fp_en); LV = pd.read_csv(fp_lv); erros9, n9 = [], 0
    lv_e = pd.read_csv(os.path.join(PASTA, "apostas_ensemble.csv.gz")) if os.path.exists(os.path.join(PASTA, "apostas_ensemble.csv.gz")) else None
    todos_min = [f for f in LV.funil.unique() if re.fullmatch(r"T\d+(_EV\d+)?", str(f))]
    for suf in sorted({re.sub(r"^T\d+", "", f) for f in todos_min}):                      # v5.0: um ensemble por limiar de EV ("" = hoje)
     fmin = sorted([f for f in todos_min if re.sub(r"^T\d+", "", f) == suf], key=lambda f: int(re.match(r"T(\d+)", f).group(1))); nivel = suf.lstrip("_") or "EV0"
     EN_s = EN[EN.nivel_ev == nivel] if "nivel_ev" in EN.columns else EN
     lv_es = lv_e[lv_e.nivel_ev == nivel] if (lv_e is not None and "nivel_ev" in lv_e.columns) else lv_e
     for (jan, conj, regra), lv in LV[LV.funil.isin(fmin)].groupby(["janela", "conjunto", "regra"]):
        jss = [set(JC[(JC.funil == f) & (JC.janela == jan) & (JC.conjunto == conj)].gameid) for f in fmin]
        if any(j != jss[0] for j in jss): erros9.append(f"{jan}/{conj}: jogos avaliados diferentes entre os minutos"); continue
        nj = len(jss[0]); ap_m = lv.sort_values(["gameid", "t"], kind="mergesort"); ap_f = ap_m.groupby("gameid", sort=False).head(1)
        for pol, ap in (("FIRST", ap_f), ("MULTI", ap_m)):
            n9 += 1; e = EN_s[(EN_s.politica_ensemble == pol) & (EN_s.janela == jan) & (EN_s.conjunto == conj) & (EN_s.regra == regra)]
            if len(e) != 1: erros9.append(f"{pol}/{jan}/{conj}/{regra}: linha do ensemble ausente"); continue
            e = e.iloc[0]
            if abs(ap.lucro.sum() / nj - e.ppg) > 1e-9: erros9.append(f"{pol}/{jan}/{conj}/{regra}: refeito {ap.lucro.sum() / nj:+.6f} ≠ ensemble {e.ppg:+.6f}")
            if int(len(ap)) != int(e.n_apostas): erros9.append(f"{pol}/{jan}/{conj}/{regra}: {len(ap)} apostas refeitas ≠ {int(e.n_apostas)}")
            if lv_es is not None:
                x = lv_es[(lv_es.politica_ensemble == pol) & (lv_es.janela == jan) & (lv_es.conjunto == conj) & (lv_es.regra == regra)]
                if not (len(x) == len(ap) and np.array_equal(x.sort_values(["gameid", "t"]).gameid.values, ap.gameid.values) and abs(x.lucro.sum() - ap.lucro.sum()) < 1e-9):
                    erros9.append(f"{pol}/{jan}/{conj}/{regra}: livro do ensemble ≠ refeito")
    checa("A9 ensemble por minuto refeito do livro dos funis T10…T35 (FIRST = 1º minuto com aposta; MULTI = todos) == ensemble_minutos.csv", not erros9 and n9 > 0,
          f"{n9} séries · minutos {fmin}" + ("; " + "; ".join(erros9[:5]) if erros9 else ""))


# ---------- v5.0: A11–A15 ----------
fp_od = os.path.join(PASTA, "odds_por_minuto.csv.gz")
if os.path.exists(fp_lv) and os.path.exists(fp_od) and "clv_fech" in pd.read_csv(fp_lv, nrows=1).columns:
    LV = pd.read_csv(fp_lv, low_memory=False); OD = pd.read_csv(fp_od)
    lado = lambda x: pd.to_numeric(x, errors="coerce").astype("Int64").astype(str) if pd.to_numeric(x, errors="coerce").notna().all() else x.astype(str)
    LV["_s"], OD["_s"] = lado(LV["side"]), lado(OD["side"]); HZ = [int(h) for h in cfg.get("horizontes_clv", [5, 10, 15, 20, 25])]
    e11 = []
    if (OD.gameid >= L0).any(): e11.append("tabela de odds com jogo >= lockbox")
    q_t = LV.merge(OD.rename(columns={"q": "_qt"})[["gameid", "_s", "t", "_qt"]], on=["gameid", "_s", "t"], how="left")["_qt"].values
    if np.nanmax(np.abs(q_t - LV.q_devig.values)) > 1e-9 or np.isnan(q_t).any(): e11.append("q da aposta ≠ q da tabela de odds no minuto da aposta")
    for h in HZ:                                                                              # q exatamente h minutos DEPOIS, no mesmo jogo e lado
        x = LV[["gameid", "_s", "t"]].assign(t=LV.t + h).merge(OD[["gameid", "_s", "t", "q"]], on=["gameid", "_s", "t"], how="left")["q"].values
        dif = np.nanmax(np.abs(x - LV[f"q_mais{h}"].values)) if np.isfinite(x).any() else 0.0
        if not (np.array_equal(np.isnan(x), np.isnan(LV[f"q_mais{h}"].values)) and dif < 1e-12): e11.append(f"q em t+{h} refeito ≠ livro (dif {dif:.2e})")
    ult = OD.sort_values("t").groupby(["gameid", "_s"]).tail(1)[["gameid", "_s", "t", "q", "odd"]].rename(columns={"t": "_mf", "q": "_qf", "odd": "_of"})
    y = LV.merge(ult, on=["gameid", "_s"], how="left"); tem = (y._mf > y.t).values
    qf = np.where(tem, y._qf, np.nan); mf = np.where(tem, y._mf, -1)
    if not (np.array_equal(np.isnan(qf), np.isnan(LV.q_fech.values)) and np.nanmax(np.abs(qf - LV.q_fech.values)) < 1e-12 and np.array_equal(mf.astype(int), LV.minuto_fech.astype(int).values)):
        e11.append("fechamento refeito (última odd do jogo depois da aposta) ≠ livro")
    clv_r = LV.odd.values * qf - 1.0
    if np.nanmax(np.abs(clv_r - LV.clv_fech.values)) > 1e-9 or not np.array_equal(np.isnan(clv_r), np.isnan(LV.clv_fech.values)): e11.append("CLV refeito ≠ livro")
    sup_r = np.where(tem, (qf > LV.q_devig.values).astype(float), np.nan)
    if not np.array_equal(np.nan_to_num(sup_r, nan=-1), np.nan_to_num(LV.supera_fech.values, nan=-1)): e11.append("supera o fechamento refeito ≠ livro")
    if (LV.minuto_fech[LV.minuto_fech >= 0] <= LV.t[LV.minuto_fech >= 0]).any(): e11.append("fechamento num minuto <= o da aposta")
    checa("A11 CLV/markout refeitos da tabela de odds (q em t+h do mesmo jogo/lado; fechamento = última odd do jogo DEPOIS da aposta) == livro", not e11,
          f"{len(LV)} apostas · {int(tem.sum())} com fechamento · horizontes {HZ}" + ("; " + "; ".join(e11) if e11 else ""))
    # A12: EV calibrado só com apostas de jogos ANTERIORES ao par
    MINC = int(cfg.get("min_calib", 30)); c_ok = LV.calib_ate_jogo >= 0
    e12 = int((LV.calib_ate_jogo[c_ok] >= LV.par_a[c_ok] - EMB).sum())                         # REVISÃO: < início do par − EMBARGO_ESCOLHA
    e12b = int((np.abs(LV.ev_cal[LV.calib_n < MINC] - LV.ev[LV.calib_n < MINC]) > 1e-12).sum())
    e12c = int((np.abs(LV.ev - (LV.p * LV.odd - 1)) > 1e-9).sum())
    checa("A12 EV calibrado: última aposta usada na calibração < início do par − embargo; sem histórico suficiente = EV do modelo; EV = p·odd − 1", e12 == 0 and e12b == 0 and e12c == 0,
          f"{int(c_ok.sum())} apostas com histórico de calibração · violações {e12}/{e12b}/{e12c}")
    # A15: limiar de EV e FIRST = no máximo 1 aposta por jogo
    e15 = []
    for fu, lv in LV.groupby("funil"):
        m_ = re.search(r"_EV(\d+)$", str(fu)); thr = int(m_.group(1)) / 100 if m_ else 0.0
        if (lv.ev <= thr).any(): e15.append(f"{fu}: {int((lv.ev <= thr).sum())} apostas com EV <= {thr}")
        if (lv.edge <= 0).any(): e15.append(f"{fu}: aposta com edge <= 0")
        if str(fu).startswith("FIRST") and lv.duplicated(["janela", "conjunto", "regra", "gameid"]).any(): e15.append(f"{fu}: FIRST com 2 apostas no mesmo jogo")
    checa("A15 regra de entrada: EV > limiar do funil (0 / 5% / 10%), edge > 0, FIRST com no máximo 1 aposta por jogo", not e15, f"{LV.funil.nunique()} funis" + ("; " + "; ".join(e15[:5]) if e15 else ""))
    # A14: ROI, ROI esperado, CLV e nº de apostas do resumo == livro
    e14, n14 = [], 0
    for (fu, jan, conj, regra), lv in LV.groupby(["funil", "janela", "conjunto", "regra"]):
        r = R[(R.funil == fu) & (R.janela == jan) & (R.conjunto == conj) & (R.regra == regra) & (R.cadencia == f"a_cada_{PASSO_E}")]
        if not len(r) or "roi_apostas" not in r.columns: continue
        r = r.iloc[0]; n14 += 1; tm = lv.clv_fech.notna()
        for nm, v in (("roi_apostas", lv.lucro.mean()), ("ev_medio_modelo", lv.ev.mean()), ("ev_medio_calibrado", lv.ev_cal.mean()), ("n_apostas_serie", len(lv)),
                      ("clv_fech_medio", lv.clv_fech[tm].mean() if tm.any() else np.nan), ("supera_fech_taxa", lv.supera_fech[tm].mean() if tm.any() else np.nan)):
            if not ((pd.isna(v) and pd.isna(r[nm])) or abs(float(v) - float(r[nm])) < 1e-9): e14.append(f"{fu}/{jan}/{conj}/{regra}: {nm} livro {v} ≠ resumo {r[nm]}")
    checa("A14 ROI, ROI esperado (EV do modelo e calibrado), CLV, % supera e nº de apostas do resumo == recalculados do livro", not e14 and n14 > 0, f"{n14} séries" + ("; " + "; ".join(e14[:5]) if e14 else ""))
fp_sb, fp_se = os.path.join(PASTA, "sombra.csv"), os.path.join(PASTA, "sombra_eventos.csv.gz")
if os.path.exists(fp_sb) and os.path.exists(fp_se) and os.path.getsize(fp_se) > 30:
    SB, SE = pd.read_csv(fp_sb), pd.read_csv(fp_se); NS, MA = int(cfg.get("n_sombra", 50)), float(cfg.get("min_amostra_sombra", 10)); e13, n13 = [], 0
    pr = SE[SE.evento == "promovida"]; rj = SE[SE.evento == "rejeitada"]
    if (pr.jogos_em_sombra < NS).any() or (rj.jogos_em_sombra < NS).any(): e13.append("decisão de sombra com menos de N_SOMBRA jogos")
    if ((pr.sombra_desde >= pr.a) | (rj.sombra_desde >= rj.a)).any(): e13.append("sombra começando no par da decisão ou depois")
    if (pr.metrica_sombra <= pr.metrica_producao).any() or (rj.metrica_sombra > rj.metrica_producao + 1e-12).any(): e13.append("promoção/rejeição incoerente com as métricas")
    if ((pr.amostra_sombra < MA) | (pr.amostra_producao < MA)).any(): e13.append("promoção com amostra < MIN_AMOSTRA_SOMBRA")
    for (fu, jan, conj, regra, cr), ev in SE.groupby(["funil", "janela", "conjunto", "regra", "criterio"]):
        n13 += 1; tr = t2[(t2.funil == fu) & (t2.janela == jan) & (t2.conjunto == conj) & (t2.regra == regra)].sort_values("a")
        ini = ev[ev.evento == "inicio_sombra"].merge(tr[["a", "escolhida"]], on="a", how="left")
        if not ((ini.sombra == ini.escolhida) & (ini.sombra != ini.producao)).all(): e13.append(f"{fu}/{jan}/{conj}/{regra}/{cr}: sombra iniciada sem a regra querer trocar"); continue
        prod, k = None, 0; mudancas = ev[ev.evento.isin(["inicio_producao", "promovida"])].sort_values("a")[["a", "evento", "producao", "sombra"]].values
        prod_par = []
        for a in tr.a.values:                                                                 # produção por par refeita só dos eventos
            while k < len(mudancas) and mudancas[k][0] <= a:
                prod = mudancas[k][2] if mudancas[k][1] == "inicio_producao" else mudancas[k][3]; k += 1
            prod_par.append(prod)
        js = JC[(JC.funil == fu) & (JC.janela == jan) & (JC.conjunto == conj)].gameid.values; lj = LJ[LJ.funil == fu].set_index(["opcao", "gameid"]).lucro
        ini_a = tr.a.values; sg = np.searchsorted(ini_a, js, side="right") - 1
        nao = set(tr.a[tr.escolhida == "NAO_APOSTA"].values)                                   # REVISÃO: a regra decidiu não apostar → a produção não aposta no par
        luc = np.array([lj.get((prod_par[i], g), 0.0) if (i >= 0 and prod_par[i] is not None and ini_a[i] not in nao) else 0.0 for g, i in zip(js, sg)])
        rep_ = SB[(SB.funil == fu) & (SB.janela == jan) & (SB.conjunto == conj) & (SB.regra == regra) & (SB.criterio == cr)].ppg
        if len(rep_) and abs(luc.mean() - float(rep_.iloc[0])) > 1e-9: e13.append(f"{fu}/{jan}/{conj}/{regra}/{cr}: PPG refeito {luc.mean():+.6f} ≠ {float(rep_.iloc[0]):+.6f}")
    ind = SE[SE.evento == "indeterminada"]
    if (ind.jogos_em_sombra < NS).any() or ((ind.amostra_sombra >= MA) & (ind.amostra_producao >= MA)).any(): e13.append("evento 'indeterminada' incoerente")
    checa("A13 (CONSISTÊNCIA, não prova de não-vazamento — essa vem da sabotagem) MODO SOMBRA refeito dos eventos: decisões só com >= N_SOMBRA jogos de sombra, sombra começa antes do par, promoção ⇔ métrica da sombra > produção; "
          "produção por par refeita dos eventos dá o PPG do sombra.csv", not e13 and n13 > 0, f"{n13} séries · {len(pr)} promoções · {len(rj)} rejeições" + ("; " + "; ".join(e13[:5]) if e13 else ""))

# ---------- A7: índices de decisões sorteadas ----------
rows = []
sub = t2[(t2.regra == "ppg") & (t2.janela == "propria") & (t2.conjunto == "todas")]
for _, r in sub.sample(n=min(15, len(sub)), random_state=SEMENTE).sort_values(["funil", "a"]).iterrows():
    el = elegiveis_em(int(r.a), "todas"); hist = U[U < int(r.a) - EMB]
    cand = None if pd.isna(r.escolhida) or r.escolhida == "NAO_APOSTA" else r.escolhida.split("|")[0]
    ft = F[(F.cand == re.sub(r"_(CALR|CAL)$", "", cand)) & (F.pa == par_ini(int(r.a)))] if cand else F.iloc[0:0]
    w_ = int(O.loc[O.nome == r.escolhida, "w"].iloc[0]) if cand else None
    rows.append(dict(funil=r.funil, decisao_a=int(r.a), par_jogos=f"{int(r.a)}–{int(r.b) - 1}", escolhida=r.escolhida, janela_historico=f"[{w_}, {int(r.a) - EMB})" if cand else "",
                     ultimo_jogo_no_historico=int(hist.max()) if len(hist) else None, jogos_no_historico_da_escolhida=int(((hist >= w_)).sum()) if cand else None,
                     opcoes_elegiveis=int(el.sum()), treino_da_previsao_usada_ate=int(ft.gid_treino_max.max()) if len(ft) else None))
A7 = pd.DataFrame(rows); A7.to_csv(os.path.join(PASTA, "auditoria_indices.csv"), index=False)
ok7 = bool(((A7.ultimo_jogo_no_historico < A7.decisao_a) | A7.ultimo_jogo_no_historico.isna()).all() and ((A7.treino_da_previsao_usada_ate < A7.decisao_a) | A7.treino_da_previsao_usada_ate.isna()).all())
checa("A7 índices: último jogo do histórico e do treino < início do par", ok7, "tabela em auditoria_indices.csv")
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 20); diz(A7.to_string(index=False))
diz(f"RESULTADO: {'TUDO OK' if not falhas else 'FALHAS: ' + ', '.join(falhas)}")
open(os.path.join(PASTA, "AUDITORIA_SAIDAS.txt"), "w", encoding="utf-8").write("\n".join(linhas_txt))
sys.exit(1 if falhas else 0)
