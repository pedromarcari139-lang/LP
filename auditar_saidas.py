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
