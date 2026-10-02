# -*- coding: utf-8 -*-
"""
validar_funil.py — VALIDAÇÃO DO FUNIL (escolha do modelo + política FIRST/MULTI) com RE-TREINO e RE-ESCOLHA a cada 2 jogos.

O QUE ESTE SCRIPT FAZ (e o que muda em relação ao rodar_holdout.py / comparar_lockbox.py)
  1) Previsões: o MESMO motor (backtest_sujo.py, sem nenhuma edição), com o treino refeito a cada PASSO = 2 gameids
     (RETREINO_DEV_JOGOS = 2): o par [a, a+2) é previsto por um modelo treinado só com jogos < a (os 2 jogos do par
     acontecem ao mesmo tempo: o 2º é previsto sem o resultado do 1º). Depois das previsões, conferir_retreino() PROVA isso
     linha a linha pelo log de ajustes (aborta se algum treino alcançar o par que ele prevê ou se um treino cobrir > 2 jogos).
  2) C FIXO (um valor só, C_FIXO abaixo) — não roda a grade de C.
  3) Minutos 5, 40 e 45 incluídos (além de 10–35), nos modelos, no critério de escolha e nas políticas.
     As features dos minutos novos seguem o padrão do ATOM/ENX do motor (ver ATOM_RECON/ENX no motor e o manifesto
     features_por_minuto.csv que este script grava): por checkpoint τ ≤ t entram g_τ, D_k_τ, M_τ; D_drake_τ só com τ ≥ 10;
     D_tt_τ só com τ ≥ 15; fixos: side, M_open. Draft/WR (pré-jogo) são os mesmos em todo minuto.
  4) RE-ESCOLHA DO MODELO a cada PASSO_ESCOLHA = 2 jogos (em vez de a cada bloco de 150): antes de cada par [a, a+2), a regra
     (mesma da etapa 1: maior critério acumulado, desempate pelo nome) é aplicada ao histórico ACUMULADO [HIST_INICIO, a).
     A mesma regra re-escolhendo a cada bloco de 150 (o procedimento atual) é calculada nos MESMOS jogos, para comparação pareada.
  5) Funis FIRST e MULTI (e, como referência, os mesmos funis só com 10–35, que é o funil antigo).

O QUE NÃO FAZ
  - Não prevê, não lê resultado e não avalia nenhum jogo >= LOCKBOX_SERIO (8448). O banco de previsões só guarda desenvolvimento.
  - Não escolhe C, não grava DECISAO, não mexe no PROTOCOLO_SELECAO.json nem em nada da etapa 1.

BANCO DE PREVISÕES: os minutos 10–35 são previstos com EXATAMENTE a configuração da etapa 1 (mesmos TEMPOS, MIN_TREINO_JOGOS,
mesma impressão digital dos dados): se C_FIXO for um dos C da grade já rodada, eles vêm do BANCO_PREVISOES sem treinar de novo.
Os minutos 5/40/45 são um grupo separado (chave própria no banco): só eles são treinados do zero (e ficam guardados para a próxima vez).

USO:  python validar_funil.py            (ou dois cliques no RODAR_FUNIL.bat, na MESMA pasta do RODAR.bat)
Saídas em OUT_FUNIL\\ — o essencial está em OUT_FUNIL\\RESUMO_FUNIL.txt.
"""
import os, sys, re, json, time, pickle, hashlib, importlib.util
import numpy as np
import pandas as pd

# ================================ CONFIG ================================
ARQUIVO = "backtest_sujo.py"        # motor (visão oficial: SUJO NO LIMPO). "backtest_limpo.py" também funciona (LIMPO NO LIMPO)
C_FIXO = 1.0                        # C ÚNICO para validar o funil. 1.0 = valor A PRIORI (padrão do sklearn; está na grade da etapa 1 → banco).
                                    # Se trocar pelo C escolhido na etapa 1 (config_escolhida.json): ele foi escolhido OLHANDO estes mesmos
                                    # jogos (6628–8447) → o PPG do funil sai otimista. Para validar o funil, prefira um C fixado antes.
LOCKBOX_SERIO = 8448                # 1º jogo do lockbox sério: NADA >= este gid é previsto (igual ao LOCKBOX_SERIO do rodar_tudo.py)
PASSO = 2                           # re-treino a cada PASSO gameids (2 = 2 partidas ao mesmo tempo, como na operação)
PASSO_ESCOLHA = 2                   # re-escolha do modelo a cada PASSO_ESCOLHA gameids (múltiplo de PASSO; 2 = a cada par)
TEMPOS_NOVOS = [5, 40, 45]          # minutos acrescentados aos do motor (10–35)
MIN_TREINO_NOVOS = {5: 150, 40: 120, 45: 120}   # mínimo de jogos de treino nos minutos novos (5 = o do 10; 40/45 = o do 35)
INICIO_PREVISAO = None              # None = G_CLEAN (6028): os blocos do backfill (700–6027) não entram na escolha, então não são previstos
HIST_INICIO = None                  # None = G_CLEAN: o histórico da escolha começa aqui (igual à etapa 1, visão sujo-no-limpo)
INICIO_AVAL = None                  # None = SELECAO_INICIO_AVALIACAO do motor (6628)
MIN_HIST = None                     # None = SELECAO_MIN_HIST_JOGOS do motor (300): par com histórico menor não é avaliado
CRITERIOS = ["bsskill_minutos", "ppg"]   # 1º = PRÉ-DECLARADO (o da etapa 1). "ppg" = PPG acumulado da PRÓPRIA política do funil (sensibilidade)
_T_TODOS = [5, 10, 15, 20, 25, 30, 35, 40, 45]
_T_ANT = [10, 15, 20, 25, 30, 35]
FUNIS = [   # nome, política, minutos em que pode apostar (cascata FIRST / todas as entradas MULTI), minutos do critério de escolha
    dict(nome="FIRST_5a45",  politica="FIRST", tempos=_T_TODOS, minutos_criterio=_T_TODOS),
    dict(nome="MULTI_5a45",  politica="MULTI", tempos=_T_TODOS, minutos_criterio=_T_TODOS),
    dict(nome="FIRST_10a35", politica="FIRST", tempos=_T_ANT,   minutos_criterio=_T_ANT),    # referência: o funil antigo (mesmas previsões)
    dict(nome="MULTI_10a35", politica="MULTI", tempos=_T_ANT,   minutos_criterio=_T_ANT),
]
CORTE_TESTE = 7729                  # só descritivo: o resultado é mostrado também antes / a partir deste jogo (janela do lockbox TESTE)
USAR_BANCO = True                   # reaproveita/grava o BANCO_PREVISOES (não muda nenhum número)
N_PROCESSOS = 0                     # processos ao mesmo tempo para treinar (0 = automático pelo paralelo_v90; 1 = em série). Não muda resultado
GB_POR_PROCESSO = 2.5
VERIFICAR_LIMPO = True              # confere zzl{t} == zz{t} nos minutos que tiverem zzl (os minutos novos normalmente não têm: fica registrado)
B_BOOT = 5000                       # réplicas do bootstrap por cluster (10 gameids) dos intervalos de PPG
SABOTAGEM_N = 40                    # teste de vazamento da re-escolha: em N pares sorteados, embaralha o futuro (>= a) e confere que a escolha não muda
PASTA_SAIDA = "OUT_FUNIL"
CFG_EXTRA = {}                      # vazio no uso normal (só para testes)
# ========================================================================

AQUI = os.path.dirname(os.path.abspath(__file__))
os.chdir(AQUI)                      # PASTA_DADOS do motor é "." = a pasta deste script (a mesma do RODAR.bat e dos zz)
OUT = os.path.join(AQUI, PASTA_SAIDA)
FILHO = (int(sys.argv[sys.argv.index("--filho") + 1]), int(sys.argv[sys.argv.index("--filho") + 2])) if "--filho" in sys.argv else None
SUF_CAL = r"_(CALR|CAL)$"


def carregar_motor():
    sp = importlib.util.spec_from_file_location("bt_funil", os.path.join(AQUI, ARQUIVO))
    B = importlib.util.module_from_spec(sp); sp.loader.exec_module(B); return B


def configurar(B):
    """Mesma preparação que o main() do motor faz antes do walk-forward (modo, famílias, pastas), com C fixo e treino a cada PASSO."""
    B.CFG.update(CFG_EXTRA)
    modo = "sujo" if "sujo" in os.path.basename(ARQUIVO) else ("limpo" if "limpo" in os.path.basename(ARQUIVO) else B.CFG["MODO"])
    B.CFG["MODO"] = modo
    if modo == "sujo": B.CFG["FAMILIAS"] = [f for f in B.CFG["FAMILIAS"] if f not in ("V6L", "V6WRL", "V6LE", "V6WRLE")]   # igual ao main()
    assert PASSO >= 1 and PASSO_ESCOLHA % PASSO == 0, "PASSO_ESCOLHA tem de ser múltiplo de PASSO"
    B.CFG.update(C_FIXO=float(C_FIXO), RETREINO_DEV_JOGOS=int(PASSO), LOCKBOX_GID_MIN=int(LOCKBOX_SERIO), LOCKBOX_ETAPA="desenvolvimento",
                 LOCKBOX_TREINO="prequencial", PLACEBO=None, PASTA_OUT=OUT)
    B._reset_rngs(); B._pastas(); os.makedirs(OUT, exist_ok=True)
    tant = [int(t) for t in B.CFG["TEMPOS"]]                       # os minutos da etapa 1, NA ORDEM do arquivo (entra na chave do banco)
    tnov = [int(t) for t in TEMPOS_NOVOS if int(t) not in tant]
    for t in tnov: assert t in B.CHECKPOINTS, f"minuto {t} não está nos CHECKPOINTS do motor {B.CHECKPOINTS}"
    assert int(PASSO) == int(B.CFG["RETREINO_DEV_JOGOS"])
    return dict(modo=modo, tant=tant, tnov=tnov, todos=sorted(tant + tnov), min0=dict(B.CFG["MIN_TREINO_JOGOS"]),
                min_nov={**B.CFG["MIN_TREINO_JOGOS"], **{int(k): int(v) for k, v in MIN_TREINO_NOVOS.items()}},
                g_clean=int(B.CFG["G_CLEAN"]), L0=int(LOCKBOX_SERIO))


def carregar_dados(B, K, conferir=True):
    """Lê os zz{t} dos minutos todos (cache do motor) e prepara. Com conferir=True faz as conferências ANTES de preparar/treinar."""
    brutos = {}
    for t in K["todos"]:
        fp = os.path.join(B.CFG["PASTA_DADOS"], f"{B.CFG['PREFIXO_GRANDE']}{t}.xlsx")
        if not os.path.exists(fp): raise FileNotFoundError(f"falta {fp} — para os minutos {K['tnov']} copie zz5/zz40/zz45 (mesmo banco dos outros zz) para esta pasta")
        brutos[t] = B.carregar_bruto(t); B.log(f"t{t}: {B.ORIGEM_DADOS.get(t)} · {len(brutos[t])} linhas · gameid máx {int(pd.to_numeric(brutos[t]['gameid']).max())}")
    if conferir:
        conferir_colunas_brutas(B, K, brutos)
        conferir_tempos(B, brutos)
        if VERIFICAR_LIMPO:
            for t in K["todos"]:
                if os.path.exists(os.path.join(B.CFG["PASTA_DADOS"], f"{B.CFG['PREFIXO_LIMPO']}{t}.xlsx")): B.log(f"t{t}: limpo == grande: {B.verificar_limpo(t, brutos[t])}")
                else: B.log(f"t{t}: sem {B.CFG['PREFIXO_LIMPO']}{t}.xlsx — verificação limpo == grande PULADA neste minuto (registrado)")
    frames = {t: B.preparar(brutos[t], t) for t in K["todos"]}
    for t in K["todos"]: B.log(f"t{t}: {frames[t]['gameid'].nunique()} jogos com os 2 lados")
    return brutos, frames


def _brutas_da_feature(f):
    """coluna(s) da planilha de onde o preparar() tira cada feature do ATOM/ENX"""
    m = re.fullmatch(r"(g|D_k|M|D_drake|D_tt)(\d+)", f)
    if not m: return [f] if f not in ("M_open",) else ["Z", "VSZ"]
    k, tau = m.group(1), m.group(2)
    return {"g": [f"g{tau}"], "D_k": [f"k{tau}m", f"d{tau}m"], "M": [f"Z{tau}", f"VSZ{tau}"], "D_drake": [f"drake{tau}", f"opdrake{tau}"], "D_tt": [f"tt{tau}", f"optt{tau}"]}[k]


def conferir_colunas_brutas(B, K, brutos):
    """Antes de preparar: cada zz{t} tem as colunas que o preparar() e as features de TODOS os candidatos precisam? Para com a lista exata."""
    draft_raw = ["SOLOQ", "SOLOQ25", "SOLOQ30", "SOLOQ35", "SOLOQ35MAIS", "PRO", "PRO25", "PRO30", "PRO35", "PRO35MAIS"] + B.GOLD5 + B.IMP5
    falt = {}
    for t, G in brutos.items():
        req = ["gameid", "side", "resultado", "Z", "VSZ", f"odd{t}", f"vsodd{t}"] + draft_raw + list(B.COLS_WRPROBS) + (list(B.COLS_LOLDRAFT) if K["modo"] == "limpo" else [])
        for (_, fam, var) in B.candidatos():
            for f in B.features(fam, var, t):
                if f in ("side",) or f.startswith("l_") or f in ("GOLDSUM", "IMPSUM"): continue
                req += _brutas_da_feature(f)
        fl = sorted(set(c for c in req if c not in G.columns))
        if fl: falt[t] = fl
    if falt:
        raise KeyError("colunas AUSENTES nos zz (nada foi treinado):\n  - " + "\n  - ".join(f"zz{t}: {v}" for t, v in falt.items())
                       + "\n  As features dos minutos novos seguem o padrão dos outros zz (g, k/d, Z/VSZ, drake/opdrake, tt/optt por checkpoint ≤ t).")
    B.log("colunas dos zz: OK (todas as que as features e o preparar precisam existem em todos os minutos)")


def conferir_tempos(B, brutos, ref=10):
    """Os zz de minutos diferentes vêm do MESMO banco: nas linhas (gameid, side) em comum, as colunas pré-jogo/comuns são iguais.
    Um gameid renumerado (bug antigo) quebra isto. Minutos > ref: todo jogo tem de existir no zz{ref} (quem chega ao minuto 40 passou do 10)."""
    R = brutos[ref]; cols = [c for c in ["resultado", "odd", "vsodd", "Z", "VSZ", "g5", "k5m", "d5m", "timeA", "timeB", "PATCH"] if c in R.columns]; linhas = []
    assert not R.duplicated(["gameid", "side"]).any(), f"zz{ref}: (gameid, side) duplicado"
    for t, G in sorted(brutos.items()):
        assert not G.duplicated(["gameid", "side"]).any(), f"zz{t}: (gameid, side) duplicado"
        cc = [c for c in cols if c in G.columns]
        m = G[["gameid", "side"] + cc].merge(R[["gameid", "side"] + cc], on=["gameid", "side"], suffixes=("_t", "_r"), how="inner", validate="one_to_one")
        ruins = [c for c in cc if not B._iguais(m[c + "_t"], m[c + "_r"])]
        assert not ruins, f"zz{t} × zz{ref}: valores diferentes em {ruins} nas mesmas (gameid, side) — gameid renumerado? Nada foi treinado"
        so_t = len(G) - len(m); so_r = len(R) - len(m)
        if t > ref: assert so_t == 0, f"zz{t}: {so_t} linhas que não existem no zz{ref} (todo jogo que chega ao minuto {t} passou do {ref}) — nada foi treinado"
        linhas.append(f"t{t}: {len(m)} linhas em comum com o zz{ref} ({len(cc)} colunas iguais) · só no zz{t}: {so_t} · só no zz{ref}: {so_r}")
    B.log("conferência entre minutos OK:\n    " + "\n    ".join(linhas))


def conferir_features(B, K, frames):
    """Features de todos os candidatos existem depois do preparar; com atomg.py, ele tem de bater com o padrão também nos minutos novos."""
    if str(B.ORIGEM_ATOM).startswith("atomg"):
        for t in K["tnov"]:
            try: ok = set(B.ATOM_BASE(t)) == set(B.ATOM_RECON(t))
            except Exception as e: raise RuntimeError(f"atomg.py não gera o ATOM do minuto {t} ({type(e).__name__}: {e}) — decida qual padrão vale antes de rodar")
            if not ok: raise RuntimeError(f"atomg.py DIFERE do padrão no minuto {t}: só no atomg {sorted(set(B.ATOM_BASE(t)) - set(B.ATOM_RECON(t)))} · "
                                          f"só no padrão {sorted(set(B.ATOM_RECON(t)) - set(B.ATOM_BASE(t)))} — decida qual vale antes de rodar")
    man = []
    for (n, fam, var) in B.candidatos():
        for t in K["todos"]:
            fs = B.features(fam, var, t); fl = [c for c in fs if c not in frames[t].columns]
            if fl: raise KeyError(f"[{n} t{t}] features ausentes depois do preparar: {fl}")
            man.append(dict(cand=n, familia=fam, variante=var, t=t, n_feats=len(fs), feats=" ".join(fs)))
    man = pd.DataFrame(man); man.to_csv(os.path.join(OUT, "features_por_minuto.csv"), index=False)
    for t in K["todos"]:
        B.log(f"features V6_MOM t{t} ({len(B.features('V6', 'MOM', t))}): {' '.join(B.features('V6', 'MOM', t))}")
    return man


def blocos_previsao(B, K, frames):
    """Blocos de avaliação do motor (os mesmos da etapa 1), só os de desenvolvimento a partir de INICIO_PREVISAO."""
    gid_max = max(int(f["gameid"].max()) for f in frames.values())
    BL = B.blocos_teste(gid_max); ini = int(INICIO_PREVISAO or K["g_clean"])
    dev = [b for b in BL if b[1] < K["L0"] and b[2] <= K["L0"]]
    bp = [b for b in dev if b[1] >= ini]
    assert bp and all(b[2] <= K["L0"] for b in bp), "nenhum bloco de desenvolvimento para prever"
    return bp


def prever(B, K, brutos, frames, blocos, cands):
    """Walk-forward do motor com treino a cada PASSO, em 2 grupos de minutos (10–35 = chave da etapa 1 → banco; 5/40/45 = chave própria)."""
    P, F = [], []
    for grupo, tempos, mt in (("etapa1", K["tant"], K["min0"]), ("novos", K["tnov"], K["min_nov"])):
        if not tempos: continue
        B.CFG["TEMPOS"] = list(tempos); B.CFG["MIN_TREINO_JOGOS"] = dict(mt)
        banco = dict(pasta=B.CFG["PASTA_PREVISOES"], cadeia=B.hash_cadeia({t: brutos[t] for t in tempos})) if USAR_BANCO else None
        B.log(f"=== previsões, minutos {tempos} ({grupo}) · C = {B.CFG['C_FIXO']} · treino a cada {B.CFG['RETREINO_DEV_JOGOS']} jogo(s) · {len(cands)} candidato(s) · "
              f"jogos {blocos[0][1]}–{blocos[-1][2] - 1}" + (" · com banco" if banco else " · SEM banco"))
        p, f = B.walk_forward_dev({t: frames[t] for t in tempos}, blocos, cands, banco=banco)
        P.append(p); F.append(f.assign(grupo=grupo) if len(f) else f)
    B.CFG["TEMPOS"] = list(K["todos"]); B.CFG["MIN_TREINO_JOGOS"] = dict(K["min_nov"])
    return pd.concat(P, ignore_index=True), pd.concat(F, ignore_index=True)


def _cortes(blocos):
    """os cortes FIXOS dos pares de treino (os mesmos do blocos_de_treino do motor): início/fim de cada bloco + múltiplos de PASSO"""
    s = set()
    for (_, g0, g1) in blocos:
        g0, g1 = int(g0), int(g1); s.update([g0, g1]); s.update(range((g0 // PASSO + 1) * PASSO, g1, PASSO))
    return np.array(sorted(s), dtype=np.int64)


def conferir_retreino(B, preds, fits, blocos):
    """PROVA, pelo log de ajustes, que o teste é atualizado a cada PASSO jogos e sem olhar o par previsto:
    (1) todo treino OK prevê jogos de UM par [a, b) com b − a <= PASSO; (2) o treino termina ANTES do par (gid_treino_max < a);
    (3) um treino por (minuto, candidato, par); (4) TODA previsão (inclusive _CALR) tem o treino do seu par. Aborta se algo falhar."""
    cortes = _cortes(blocos); f = fits[fits["status"] == "OK"].copy()
    for c in ("gid_treino_max", "gid_teste_min", "gid_teste_max", "n_treino_jogos", "n_feats"): f[c] = pd.to_numeric(f[c]).astype("int64")
    i = np.searchsorted(cortes, f["gid_teste_min"].values, side="right") - 1
    f["par_a"] = cortes[i]; f["par_b"] = cortes[np.minimum(i + 1, len(cortes) - 1)]
    erros = []
    def _conta(msg, m):
        n = int(np.sum(m))
        if n: erros.append(f"{n} treino(s): {msg}")
    _conta("treino alcança o par que ele prevê (gid_treino_max >= início do par)", f["gid_treino_max"] >= f["par_a"])
    _conta("prevê jogo FORA do seu par", f["gid_teste_max"] >= f["par_b"])
    _conta(f"par com mais de {PASSO} gameids", (f["par_b"] - f["par_a"]) > PASSO)
    _conta("mais de um treino no mesmo (minuto, candidato, par)", f.duplicated(["t", "cand", "par_a"]))
    p0 = preds[["t", "cand", "gameid"]].copy(); p0["cand"] = p0["cand"].str.replace(SUF_CAL, "", regex=True)
    p0["par_a"] = cortes[np.searchsorted(cortes, p0["gameid"].values, side="right") - 1]
    m = p0.drop_duplicates(["t", "cand", "par_a"]).merge(f[["t", "cand", "par_a", "gid_treino_max"]], on=["t", "cand", "par_a"], how="left")
    _conta("previsões sem o treino do próprio par no log de ajustes", m["gid_treino_max"].isna())
    _conta("previsões cujo treino alcança o par", m["gid_treino_max"] >= m["par_a"])
    f["folga"] = f["par_a"] - f["gid_treino_max"]; f["jogos_previstos"] = f["gid_teste_max"] - f["gid_teste_min"] + 1
    f["EPV_calc"] = f["n_treino_jogos"] / (f["n_feats"] + 1.0)
    tab = f.groupby("t").agg(treinos=("par_a", "size"), candidatos=("cand", "nunique"), pares_distintos=("par_a", "nunique"),
                             par_max_gameids=("par_b", lambda s_: int((s_ - f.loc[s_.index, "par_a"]).max())), jogos_previstos_por_treino_max=("jogos_previstos", "max"),
                             folga_mediana=("folga", "median"), folga_max=("folga", "max"), treino_jogos_min=("n_treino_jogos", "min"),
                             treino_jogos_mediana=("n_treino_jogos", "median"), treino_jogos_max=("n_treino_jogos", "max"), n_feats_max=("n_feats", "max"),
                             EPV_min=("EPV_calc", "min"), pct_treinos_EPV_baixo=("EPV_calc", lambda s_: float((s_ < float(B.CFG["EPV_ALERTA"])).mean())),
                             gid_teste_min=("gid_teste_min", "min"), gid_teste_max=("gid_teste_max", "max")).reset_index()
    tab.to_csv(os.path.join(OUT, "conferencia_retreino.csv"), index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    B.log("CONFERÊNCIA DO RE-TREINO (folga = início do par − último jogo do treino; 1 = treinou até o jogo anterior ao par):\n" + tab.to_string(index=False))
    if erros: raise AssertionError("RE-TREINO A CADA %d JOGOS: FALHOU — nada foi avaliado:\n  - " % PASSO + "\n  - ".join(erros))
    B.log(f"RE-TREINO A CADA {PASSO} JOGOS: OK — {len(f)} treinos; cada um prevê só o seu par [a, a+{PASSO}) e treina só com jogos < a; toda previsão tem o seu treino")
    return tab


def montar_base_funil(B, preds, frames):
    """igual ao montar_base do motor (edge, scores, filtro de odd/probabilidade válida), sem as colunas de flags (que os zz novos podem não ter)"""
    info = pd.concat([frames[t][["t", "gameid", "side", "y", "odd_t", "vsodd_t", "q_devig_t", "q_raw_t"]] for t in sorted(frames)], ignore_index=True)
    base = preds.merge(info, on=["t", "gameid", "side"], how="left", validate="many_to_one")
    base["edge"] = base["p"] - base["q_raw_t"]; base["ev"] = base["p"] * base["odd_t"] - 1.0
    base["ll_mod"] = -(base["y"] * np.log(np.clip(base["p"], 1e-6, 1)) + (1 - base["y"]) * np.log(np.clip(1 - base["p"], 1e-6, 1)))
    ok = np.isfinite(base["p"]) & base["q_devig_t"].gt(0) & base["q_devig_t"].lt(1) & np.isfinite(base["odd_t"]) & base["odd_t"].gt(1)
    if (~ok).any(): B.log(f"base: {int((~ok).sum())} linhas com probabilidade/odd inválida removidas")
    base = base[ok].copy(); assert base["p"].between(0, 1).all()
    qc = base["q_devig_t"].clip(1e-6, 1 - 1e-6)
    base["ll_mkt"] = -(base["y"] * np.log(qc) + (1 - base["y"]) * np.log(1 - qc))
    base["bs_mod"] = (base["p"] - base["y"]) ** 2; base["bs_mkt"] = (base["q_devig_t"] - base["y"]) ** 2
    base["cluster"] = B._clu(base["gameid"]).astype(int)
    return base


def escolhas(G, X, inicios, hist_ini, min_hist):
    """A REGRA, aplicada antes de cada segmento que começa em a: média do critério por candidato nos jogos do universo em [hist_ini, a)
    (acumulado: só jogos que JÁ terminaram); escolhe o maior; empate (|dif| <= 1e-12) → 1º em ordem alfabética (colunas de X já vêm ordenadas;
    C é fixo, então o desempate 'menor C' da etapa 1 não se aplica). Devolve (índice escolhido ou −1, valor, 2º valor, nº de jogos no histórico)."""
    fin = np.isfinite(X); m = X.shape[1]
    cs = np.vstack([np.zeros((1, m)), np.cumsum(np.where(fin, X, 0.0), axis=0)])
    cn = np.vstack([np.zeros((1, m)), np.cumsum(fin, axis=0)])
    cr = np.r_[0, np.cumsum(fin.any(axis=1))]
    lo = int(np.searchsorted(G, hist_ini, side="left")); hi = np.searchsorted(G, np.asarray(inicios), side="left")     # jogos com gid < a
    N = cn[hi] - cn[lo]
    with np.errstate(invalid="ignore", divide="ignore"): M = (cs[hi] - cs[lo]) / N
    M[N == 0] = np.nan
    n_hist = cr[hi] - cr[lo]
    tem = np.isfinite(M).any(axis=1)
    best = np.full(len(hi), np.nan); best[tem] = np.nanmax(M[tem], axis=1)
    idx = np.where(tem, np.argmax(np.nan_to_num(M, nan=-np.inf) >= (best[:, None] - 1e-12), axis=1), -1)
    seg = -np.sort(-np.nan_to_num(M, nan=-np.inf), axis=1)[:, 1] if m > 1 else np.full(len(hi), np.nan)
    idx = np.where(tem & (n_hist >= min_hist), idx, -1)
    return idx, best, np.where(np.isfinite(seg), seg, np.nan), n_hist


def procedimento(G, X, LUC, nomes, segmentos, hist_ini, min_hist):
    """Re-escolhe no início de cada segmento [a, b) (só com o passado) e aposta com o escolhido nos jogos do universo em [a, b).
    Devolve (lucro por jogo avaliado, escolhido por jogo, trilha)."""
    ini = np.array([a for a, _ in segmentos], dtype=np.int64); fim = np.array([b for _, b in segmentos], dtype=np.int64)
    idx, val, seg, nh = escolhas(G, X, ini, hist_ini, min_hist)
    s = np.searchsorted(ini, G, side="right") - 1
    dentro = (s >= 0) & (G < fim[np.clip(s, 0, None)])
    s_ok = np.where(dentro, s, -1); esc = np.where(s_ok >= 0, idx[np.clip(s_ok, 0, None)], -1)
    av = esc >= 0
    lucro = pd.Series(LUC[np.where(av)[0], esc[av]], index=G[av])
    escolhido = pd.Series(np.array(nomes, dtype=object)[esc[av]], index=G[av])
    njs = np.bincount(s_ok[dentro], minlength=len(ini)); luc_seg = np.bincount(s[av], weights=lucro.values, minlength=len(ini))
    tr = pd.DataFrame(dict(a=ini, b=fim, n_hist=nh, escolhido=[nomes[i] if i >= 0 else None for i in idx], valor_hist=val, segundo_valor_hist=seg,
                           margem=val - seg, jogos_no_segmento=njs, lucro_no_segmento=luc_seg))
    return lucro, escolhido, tr


def teste_sabotagem(G, X, segmentos, hist_ini, min_hist, n, rng):
    """VAZAMENTO na re-escolha: para pares sorteados, troca TODO o critério dos jogos >= a por ruído e confere que as escolhas de todos os
    segmentos que começam até a NÃO mudam. Pega erro de índice (ex.: incluir o próprio par ou o jogo a no histórico)."""
    ini = np.array([a for a, _ in segmentos], dtype=np.int64); base_idx = escolhas(G, X, ini, hist_ini, min_hist)[0]
    cand = np.where(base_idx >= 0)[0]
    if not len(cand): return 0, 0
    falhas = 0; amostra = rng.choice(cand, size=min(n, len(cand)), replace=False)
    for s in amostra:
        X2 = X.copy(); fut = G >= ini[s]; X2[fut] = rng.normal(0, 10, size=X2[fut].shape)
        idx2 = escolhas(G, X2, ini, hist_ini, min_hist)[0]; ate = ini <= ini[s]
        if not np.array_equal(idx2[ate], base_idx[ate]): falhas += 1
    return len(amostra), falhas


def main_filho(i, n):
    """processo filho: treina só os candidatos cands[i::n] (os dois grupos de minutos) e grava as previsões/ajustes em OUT/_partes"""
    B = carregar_motor(); K = configurar(B)
    brutos, frames = carregar_dados(B, K, conferir=False)
    blocos = blocos_previsao(B, K, frames); cands = B.candidatos()[i::n]
    p, f = prever(B, K, brutos, frames, blocos, cands)
    os.makedirs(os.path.join(OUT, "_partes"), exist_ok=True); arq = os.path.join(OUT, "_partes", f"parte_{i}_de_{n}.pkl")
    with open(arq + ".tmp", "wb") as fh: pickle.dump(dict(preds=p, fits=f, cands=[c[0] for c in cands]), fh, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(arq + ".tmp", arq)
    try:
        import paralelo_v90 as PAR; B.log(f"pico de memória desta rodada: {PAR.pico_memoria_mb()} MB")
    except Exception: pass


def main():
    t0 = time.time(); B = carregar_motor(); K = configurar(B)
    hist_ini = int(HIST_INICIO or K["g_clean"]); aval = int(INICIO_AVAL or B.CFG["SELECAO_INICIO_AVALIACAO"]); min_hist = int(MIN_HIST if MIN_HIST is not None else B.CFG["SELECAO_MIN_HIST_JOGOS"])
    B.log(f"VALIDAR FUNIL · motor {ARQUIVO} (modo {K['modo']}) · C fixo = {B.CFG['C_FIXO']} · treino a cada {PASSO} · re-escolha a cada {PASSO_ESCOLHA} · "
          f"minutos {K['todos']} (etapa 1: {K['tant']}; novos: {K['tnov']}) · lockbox sério {K['L0']}+ NUNCA previsto · histórico da escolha desde {hist_ini} · "
          f"avaliação desde {aval} · histórico mínimo {min_hist} jogos · critérios {CRITERIOS} (1º pré-declarado) · ATOM: {B.ORIGEM_ATOM}")
    brutos, frames = carregar_dados(B, K, conferir=True)
    man = conferir_features(B, K, frames)
    blocos = blocos_previsao(B, K, frames); cands = B.candidatos()
    B.log(f"{len(cands)} candidatos base ({', '.join(c[0] for c in cands)}) + CAL: {B.CFG.get('CAL_PARA')} · {len(blocos)} blocos de avaliação "
          f"({blocos[0][1]}–{blocos[-1][2] - 1}) → {len(_cortes(blocos)) - 1} pares de treino de {PASSO} jogos")
    # ---------------- previsões (em série ou em paralelo por candidato) ----------------
    n_pr = 1
    if N_PROCESSOS != 1:
        try:
            import paralelo_v90 as PAR
            n_pr, motivo = PAR.escolher_n(len(cands), N_PROCESSOS, GB_POR_PROCESSO); PAR.anunciar("funil (treino dividido por candidato)", n_pr, n_pr, motivo)
        except ImportError:
            n_pr = 1
    if n_pr <= 1:
        preds, fits = prever(B, K, brutos, frames, blocos, cands)
    else:
        del brutos, frames                                  # os filhos leem do cache do motor (já aquecido acima); o pai libera a memória
        pastas = os.path.join(OUT, "_partes"); os.makedirs(pastas, exist_ok=True)
        for fn in os.listdir(pastas): os.remove(os.path.join(pastas, fn))
        PAR.rodar_em_paralelo([(f"parte{i}", [os.path.abspath(__file__), "--filho", str(i), str(n_pr)]) for i in range(n_pr)], n_pr, "funil", OUT, cwd=AQUI)
        partes = [pickle.load(open(os.path.join(pastas, f"parte_{i}_de_{n_pr}.pkl"), "rb")) for i in range(n_pr)]
        assert sorted(c for p_ in partes for c in p_["cands"]) == sorted(c[0] for c in cands), "partes não cobrem todos os candidatos"
        preds = pd.concat([p_["preds"] for p_ in partes], ignore_index=True); fits = pd.concat([p_["fits"] for p_ in partes], ignore_index=True)
        brutos, frames = carregar_dados(B, K, conferir=False)
    assert int(preds["gameid"].max()) < K["L0"], "previsão de jogo do lockbox — ABORTADO"
    fits = B._alerta_epv_atual(fits)
    preds.to_csv(os.path.join(OUT, "previsoes_funil.csv.gz"), index=False); fits.to_csv(os.path.join(OUT, "ajustes_funil.csv.gz"), index=False)
    conferir_retreino(B, preds, fits, blocos)
    # ---------------- base, universo, critério ----------------
    base = montar_base_funil(B, preds, frames)
    B.CFG["TEMPOS"] = list(K["todos"]); B.CFG["TEMPOS_DEPLOY"] = sorted({int(t) for f_ in FUNIS for t in f_["tempos"]})
    uv, _ = B.universo_executavel(frames)
    G = np.array(sorted(int(g) for g in uv if hist_ini <= int(g) < K["L0"]), dtype=np.int64)
    nomes = sorted(base["cand"].unique())
    lj = {c: B.linhas_jogo(base, c) for c in nomes}
    longo = pd.concat([lj[c][["t", "gameid", "bsskill"]].assign(cand=c) for c in nomes]); longo = longo[longo["gameid"].isin(set(G))]
    SK = longo.pivot_table(index=["gameid", "t"], columns="cand", values="bsskill").dropna(how="any").sort_index()
    faltam = sorted(set(nomes) - set(SK.columns))
    if faltam: B.log(f"ATENÇÃO: candidatos sem linha no conjunto comum (ficam fora da escolha): {faltam}")
    nomes = sorted(SK.columns)
    B.log(f"universo executável (minutos {B.CFG['TEMPOS_DEPLOY']}) {hist_ini}–{K['L0'] - 1}: {len(G)} jogos (denominador do PPG) · skill por (jogo, minuto), "
          f"conjunto comum: {len(SK)} linhas, por minuto {SK.groupby(level=1).size().to_dict()}")
    SK.to_csv(os.path.join(OUT, "skill_matriz_minutos_funil.csv.gz"))
    # segmentos: pares fixos do treino (re-escolha a cada PASSO_ESCOLHA) e blocos de 150 do motor (o procedimento atual)
    cortes = _cortes(blocos); cortes = cortes[(cortes >= aval) | (cortes == cortes[-1])]
    if PASSO_ESCOLHA != PASSO: cortes = np.array(sorted({int(c) for c in cortes if (c - cortes[0]) % PASSO_ESCOLHA == 0} | {int(cortes[-1])}), dtype=np.int64)
    seg2 = [(int(a), int(b)) for a, b in zip(cortes[:-1], cortes[1:]) if a >= aval]
    seg150 = [(int(g0), int(g1)) for (_, g0, g1) in blocos if g0 >= aval]
    assert seg2 and seg150 and seg2[0][0] == seg150[0][0], "segmentos de avaliação desalinhados"
    rng = np.random.default_rng(int(B.CFG["SEED"]))
    linhas, pareados, rows_fixos, aps_proc = [], [], [], []
    for fu in FUNIS:
        LUC_df, bets_c = {}, {}
        for c in nomes:
            b_, _ = B.apostas(base, c, fu["politica"], tempos=list(fu["tempos"]))
            bets_c[c] = b_; LUC_df[c] = b_.groupby("gameid")["lucro"].sum().reindex(G).fillna(0.0)
        LUC = pd.DataFrame(LUC_df)[nomes].values
        for crit in CRITERIOS:
            if crit == "bsskill_minutos":
                sm = SK[SK.index.get_level_values(1).isin(fu["minutos_criterio"])].groupby(level=0).mean()
                X = sm.reindex(G)[nomes].values
            elif crit == "ppg":
                X = LUC.copy()
            else:
                raise ValueError(f"critério desconhecido: {crit}")
            nome_fc = f"{fu['nome']}__{crit}"
            n_sab, f_sab = teste_sabotagem(G, X, seg2, hist_ini, min_hist, SABOTAGEM_N, rng)
            if f_sab: raise AssertionError(f"{nome_fc}: a escolha MUDOU quando o futuro foi embaralhado em {f_sab}/{n_sab} pares — VAZAMENTO na re-escolha; nada foi gravado")
            l2, e2, tr2 = procedimento(G, X, LUC, nomes, seg2, hist_ini, min_hist)
            l150, e150, tr150 = procedimento(G, X, LUC, nomes, seg150, hist_ini, min_hist)
            tr2.to_csv(os.path.join(OUT, f"trilha_{nome_fc}_a_cada_{PASSO_ESCOLHA}.csv"), index=False); tr150.to_csv(os.path.join(OUT, f"trilha_{nome_fc}_bloco150.csv"), index=False)
            js = np.array(sorted(set(l2.index) & set(l150.index)), dtype=np.int64)
            def bt(s_):
                s_ = s_.reindex(js) if len(js) else s_
                if len(s_) < 30: return dict(media=np.nan, ic_lo=np.nan, ic_hi=np.nan, p_valor=np.nan, n=len(s_))
                r_ = B.boot_media(s_.values.astype(float), B._clu(s_.index.values), B=B_BOOT); r_["n"] = len(s_); return r_
            def bt_parte(s_, m_):
                s_ = s_.reindex(js[m_])
                if len(s_) < 30: return dict(media=np.nan, ic_lo=np.nan, ic_hi=np.nan, p_valor=np.nan, n=len(s_))
                r_ = B.boot_media(s_.values.astype(float), B._clu(s_.index.values), B=B_BOOT); r_["n"] = len(s_); return r_
            tot = {c: float(LUC_df[c].reindex(js).sum()) for c in nomes}; c_ret = max(tot, key=tot.get)
            ref = {f"re-escolha a cada {PASSO_ESCOLHA} jogos": l2, "re-escolha a cada bloco de 150 (atual)": l150,
                   f"melhor fixo EM RETROSPECTO ({c_ret}) — ENVIESADO": LUC_df[c_ret], "média dos candidatos (escolher ao acaso)": pd.DataFrame(LUC_df).mean(axis=1)}
            for nome_r, s_ in ref.items():
                r_ = bt(s_); m_ant = js < CORTE_TESTE
                ra, rd = bt_parte(s_, m_ant), bt_parte(s_, ~m_ant)
                trocas = (int((e2.reindex(js) != e2.reindex(js).shift()).iloc[1:].sum()) if nome_r.startswith("re-escolha a cada " + str(PASSO_ESCOLHA))
                          else (int((e150.reindex(js) != e150.reindex(js).shift()).iloc[1:].sum()) if "150" in nome_r else None))
                linhas.append(dict(funil=fu["nome"], politica=fu["politica"], tempos_aposta=str(fu["tempos"]), criterio=crit, pre_declarado=(crit == CRITERIOS[0]),
                                   estrategia=nome_r, n_jogos=r_["n"], ppg=r_["media"], ic_lo=r_["ic_lo"], ic_hi=r_["ic_hi"], p_unilateral=r_["p_valor"],
                                   ppg_antes_corte=ra["media"], n_antes=ra["n"], ppg_desde_corte=rd["media"], n_desde=rd["n"], trocas_de_modelo=trocas,
                                   sabotagem_pares_testados=n_sab, sabotagem_falhas=f_sab))
            d_ = (l2.reindex(js) - l150.reindex(js)); rdd = bt(d_)
            pareados.append(dict(funil=fu["nome"], criterio=crit, n_jogos=len(js), delta_ppg_2_menos_150=rdd["media"], ic_lo=rdd["ic_lo"], ic_hi=rdd["ic_hi"],
                                 p_unilateral_2_melhor=rdd["p_valor"], jogos_com_escolha_diferente=int((e2.reindex(js) != e150.reindex(js)).sum())))
            if crit == CRITERIOS[0]:
                for c in nomes:
                    rows_fixos.append(dict(funil=fu["nome"], cand=c, n_jogos=len(js), ppg=float(LUC_df[c].reindex(js).mean()),
                                           pct_jogos_escolhido_a_cada_2=float((e2.reindex(js) == c).mean()), pct_jogos_escolhido_bloco150=float((e150.reindex(js) == c).mean())))
                esc_g = e2.reindex(js)
                for c in nomes:
                    gs = set(esc_g.index[esc_g.values == c])
                    if gs: aps_proc.append(bets_c[c][bets_c[c]["gameid"].isin(gs)].assign(cand=c, funil=fu["nome"], criterio=crit))
            B.log(f"{nome_fc}: a cada {PASSO_ESCOLHA} → PPG {bt(l2)['media']:+.4f} · bloco 150 → {bt(l150)['media']:+.4f} · Δ {rdd['media']:+.4f} "
                  f"[{rdd['ic_lo']:+.4f}; {rdd['ic_hi']:+.4f}] · sabotagem {n_sab} pares, {f_sab} falhas")
    R = pd.DataFrame(linhas); PR = pd.DataFrame(pareados); FX = pd.DataFrame(rows_fixos)
    R.to_csv(os.path.join(OUT, "resumo_funil.csv"), index=False); PR.to_csv(os.path.join(OUT, "pareado_2_vs_150.csv"), index=False); FX.to_csv(os.path.join(OUT, "candidatos_fixos.csv"), index=False)
    cols_ap = ["funil", "criterio", "cand", "gameid", "t", "side", "p", "q_devig_t", "odd_t", "edge", "y", "lucro"]
    APP = pd.concat(aps_proc, ignore_index=True)[cols_ap] if aps_proc else pd.DataFrame(columns=cols_ap); APP.to_csv(os.path.join(OUT, "apostas_procedimento.csv.gz"), index=False)
    # ---------------- RESUMO legível ----------------
    f_ = lambda v, fmt="+.4f": "n/d" if v is None or (isinstance(v, float) and not np.isfinite(v)) else format(v, fmt)
    txt = [f"VALIDAÇÃO DO FUNIL — {time.strftime('%Y-%m-%d %H:%M')} · motor {ARQUIVO} ({K['modo']}) · C fixo {B.CFG['C_FIXO']} · {len(nomes)} candidatos",
           f"Treino a cada {PASSO} jogos (conferido: conferencia_retreino.csv) · re-escolha a cada {PASSO_ESCOLHA} jogos com o critério ACUMULADO desde {hist_ini} · "
           f"avaliação {seg2[0][0]}–{seg2[-1][1] - 1} · lockbox sério {K['L0']}+ não previsto",
           f"PPG = lucro ÷ jogos executáveis (minutos {B.CFG['TEMPOS_DEPLOY']}, universo comum a todos os funis) · IC 95% bootstrap por cluster de 10 gameids · "
           f"p unilateral (PPG > 0) · 'antes/desde' = jogos < / >= {CORTE_TESTE}", ""]
    for fu in FUNIS:
        for crit in CRITERIOS:
            r_ = R[(R.funil == fu["nome"]) & (R.criterio == crit)]; p_ = PR[(PR.funil == fu["nome"]) & (PR.criterio == crit)].iloc[0]
            txt.append(f"{fu['nome']} ({fu['politica']}, apostas em {fu['tempos']}) · critério {crit}{' (PRÉ-DECLARADO)' if crit == CRITERIOS[0] else ' (sensibilidade)'}")
            for _, x in r_.iterrows():
                txt.append(f"   {x.estrategia:<58} PPG {f_(x.ppg)} [{f_(x.ic_lo)}; {f_(x.ic_hi)}] p={f_(x.p_unilateral, '.4f')} n={x.n_jogos} · antes {f_(x.ppg_antes_corte)} · desde {f_(x.ppg_desde_corte)}"
                           + (f" · trocas {int(x.trocas_de_modelo)}" if pd.notna(x.trocas_de_modelo) else ""))
            txt.append(f"   Δ (a cada {PASSO_ESCOLHA} − bloco 150), pareado: {f_(p_.delta_ppg_2_menos_150)} [{f_(p_.ic_lo)}; {f_(p_.ic_hi)}] p={f_(p_.p_unilateral_2_melhor, '.4f')} "
                       f"· escolha diferente em {p_.jogos_com_escolha_diferente}/{p_.n_jogos} jogos")
            if crit == CRITERIOS[0] and len(FX):
                top = FX[FX.funil == fu["nome"]].sort_values("pct_jogos_escolhido_a_cada_2", ascending=False).head(5)
                txt.append("   mais escolhidos (a cada %d): " % PASSO_ESCOLHA + " · ".join(f"{r.cand} {r.pct_jogos_escolhido_a_cada_2:.0%}" for r in top.itertuples() if r.pct_jogos_escolhido_a_cada_2 > 0))
                if len(APP):
                    am = APP[(APP.funil == fu["nome"]) & (APP.criterio == crit)]
                    if len(am): txt.append("   apostas do procedimento por minuto: " + " · ".join(f"t{int(t)} {n}" for t, n in am.groupby("t").size().items()))
            txt.append("")
    txt += ["LEIA COM CUIDADO:",
            " - Desenvolvimento: estes jogos já foram vistos em análises; isto mede o MÉTODO (o funil), não confirma nada. A confirmação é o lockbox (8448+).",
            " - 'melhor fixo em retrospecto' escolhe olhando os próprios jogos avaliados: é teto enviesado, não estratégia.",
            " - Muitos funis × critérios foram calculados: não escolha o funil pelo maior PPG desta tabela sem corrigir por seleção múltipla.",
            f" - C fixo = {B.CFG['C_FIXO']}: se ele veio da etapa 1 (escolhido com estes jogos), os PPG acima estão otimistas.",
            f"pronto em {(time.time() - t0) / 60:.1f} min"]
    open(os.path.join(OUT, "RESUMO_FUNIL.txt"), "w", encoding="utf-8").write("\n".join(txt))
    json.dump(dict(arquivo=ARQUIVO, motor_sha256=hashlib.sha256(open(os.path.join(AQUI, ARQUIVO), "rb").read()).hexdigest(), script_sha256=hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest(),
                   C_FIXO=C_FIXO, PASSO=PASSO, PASSO_ESCOLHA=PASSO_ESCOLHA, tempos=K["todos"], min_treino=K["min_nov"], lockbox_serio=K["L0"], hist_inicio=hist_ini, inicio_aval=aval,
                   min_hist=min_hist, criterios=CRITERIOS, funis=FUNIS, n_candidatos=len(nomes), candidatos=nomes, atom=B.ORIGEM_ATOM),
              open(os.path.join(OUT, "config_funil.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
    print("\n".join(txt), flush=True)


if __name__ == "__main__":
    if FILHO: main_filho(*FILHO)
    else: main()
