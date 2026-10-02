# -*- coding: utf-8 -*-
"""
validar_funil.py — v2 (02/10/2026) — VALIDAÇÃO DO FUNIL com re-treino E re-escolha a cada 2 jogos, opções (modelo × flag),
16 regras de escolha ("rule scorers") e Brier / log loss olhados de várias formas.

O QUE FAZ
  1) PREVISÕES (motor sem edição; treino refeito a cada PASSO = 2 gameids; conferir_retreino() PROVA pelo log de ajustes):
     - REGIME_DRAFT = "limpo" (padrão): DRAFT / TUDO / DRAFTR / V6L / V6WRL (famílias "so_limpo") vêm do backtest_limpo.py — treinam SÓ
       com jogos >= G_CLEAN (6028) e só entram como opção a partir do INICIO_TESTE (6628, os 600 jogos limpos de treino do desenho limpo).
       V6 / V6WR / V6E / V6WRE vêm do backtest_sujo.py: para essas famílias a previsão é a MESMA nos dois modos (o modo só mascara
       colunas do loldraft/PRO, que elas não usam), e assim o BANCO_PREVISOES da etapa 1 (sujo, 700–8447) é reaproveitado.
     - Minutos 5, 40 e 45 incluídos. Eles NÃO existem no certax/atomg: as features desses minutos são SEMPRE o padrão reconstruído
       (ATOM_RECON/ENX do motor): por checkpoint τ <= t entram g_τ, D_k_τ, M_τ; D_drake_τ só com τ >= 10 (no t5 NÃO há drake5);
       D_tt_τ só com τ >= 15; fixos side, M_open. Se houver atomg.py, ele continua valendo nos minutos 10–35 (os do certax).
  2) OPÇÕES do funil = (modelo, flag). Flag = só aposta se o LADO escolhido tiver flag == 1 (mesma regra do filtro_col do motor).
       fase suja (jogos < 6028): só modelos sem draft; flags cWRgrande, cPROBSgrande (e "sem flag")
       fase limpa: todos os modelos; flags acima + bisolo, bipro, bisolo30, bipro30, bisolo35mais, bipro35mais
     A disponibilidade é por gameid (FLAG_INICIO / INICIO_OPCOES_DRAFT): antes do início, a opção não existe e o flag vale "não".
  3) RE-ESCOLHA a cada PASSO_ESCOLHA = 2 jogos, com o histórico ACUMULADO, por CADA uma das 16 regras (SCORERS abaixo), em duas
     janelas (JANELAS): "propria" (cada opção é medida desde que existe) e "comum" (todas as opções elegíveis medidas na mesma janela de
     tempo, a partir da mais nova). A mesma regra re-escolhendo a cada bloco do motor (300/600/150 jogos) é calculada nos mesmos jogos.
  4) BRIER / LOG LOSS (metricas_previsao_*.csv): modelo, mercado de-vigado e skill (mercado − modelo, positivo = modelo melhor) por
     período e minuto, nos recortes: todas as previsões; flag = 1; flag = 0; entrou / não entrou (FIRST e MULTI, dentro do universo da
     opção); e o lado apostado (p do lado contra o resultado).

NÃO prevê, não lê resultado e não avalia jogo >= LOCKBOX_SERIO (8448). Não escolhe C, não grava DECISAO, não toca na etapa 1.
USO: python validar_funil.py  (ou RODAR_FUNIL.bat, na pasta do RODAR.bat). Essencial em OUT_FUNIL\\RESUMO_FUNIL.txt.
"""
import os, sys, re, json, time, pickle, hashlib, importlib.util
import numpy as np
import pandas as pd

# ================================ CONFIG ================================
REGIME_DRAFT = "limpo"              # "limpo" (padrão, ver acima) | "sujo" (tudo do backtest_sujo.py: DRAFT/TUDO treinam com o backfill; entram a partir do G_CLEAN)
ARQ_SUJO, ARQ_LIMPO = "backtest_sujo.py", "backtest_limpo.py"
C_FIXO = 1.0                        # C ÚNICO. 1.0 = valor a priori (padrão do sklearn; está na grade → banco). O C da etapa 1 foi escolhido com estes jogos (otimista)
LOCKBOX_SERIO = 8448                # nada >= este gid é previsto
PASSO = 2                           # re-treino a cada PASSO gameids
PASSO_ESCOLHA = 2                   # re-escolha a cada PASSO_ESCOLHA gameids (múltiplo de PASSO)
TEMPOS_NOVOS = [5, 40, 45]
MIN_TREINO_NOVOS = {5: 150, 40: 120, 45: 120}   # mínimo de jogos de treino nos minutos novos (5 = o do 10; 40/45 = o do 35)
HIST_INICIO = None                  # None = INICIO_TESTE_SUJO do motor (700): a fase suja entra no funil (só opções sem draft, flags de WR)
INICIO_OPCOES_DRAFT = None          # None = INICIO_TESTE (6628) no regime limpo; G_CLEAN (6028) no regime sujo
MIN_HIST = None                     # None = SELECAO_MIN_HIST_JOGOS (300): opção só é elegível com >= isto de jogos executáveis na própria janela
MIN_LINHAS = 30                     # regra que depende de linhas/apostas (todas*, entradas*, roi, sharpe, sortino, maxdd, seq_ruim, calmar) exige >= isto
FLAGS_SUJO = ["cWRgrande", "cPROBSgrande"]
FLAGS_LIMPO = ["bisolo", "bipro", "bisolo30", "bipro30", "bisolo35mais", "bipro35mais"]
FLAG_INICIO = None                  # None = WR: desde HIST_INICIO (cWR/cPROBS só com o passado); bisolo*: G_CLEAN; bipro*: PRO_LIVE_FROM (6115: PRO de 6028–6114 é backfill)
SCORERS = ["bsskill_todas", "llskill_todas", "brier_todas", "ll_todas",            # todas as previsões da opção (no universo do flag), minutos do critério
           "bsskill_entradas", "llskill_entradas", "brier_entradas", "ll_entradas",  # só o lado apostado
           "ppg", "lucro", "roi", "sharpe", "sortino", "calmar", "maxdd", "seq_ruim"]  # lucro das apostas
SINAL = {s: 1 for s in SCORERS}; SINAL.update(brier_todas=-1, ll_todas=-1, brier_entradas=-1, ll_entradas=-1, maxdd=-1, seq_ruim=-1)
JANELAS = ["propria", "comum"]
_T_TODOS = [5, 10, 15, 20, 25, 30, 35, 40, 45]
_T_ANT = [10, 15, 20, 25, 30, 35]
FUNIS = [   # nome, política, minutos em que pode apostar, minutos das regras "todas"
    dict(nome="FIRST_5a45",  politica="FIRST", tempos=_T_TODOS, minutos_criterio=_T_TODOS),
    dict(nome="MULTI_5a45",  politica="MULTI", tempos=_T_TODOS, minutos_criterio=_T_TODOS),
    dict(nome="FIRST_10a35", politica="FIRST", tempos=_T_ANT,   minutos_criterio=_T_ANT),
    dict(nome="MULTI_10a35", politica="MULTI", tempos=_T_ANT,   minutos_criterio=_T_ANT),
]
METRICAS_TEMPOS = _T_TODOS          # minutos das políticas usadas no recorte entrou/não entrou das métricas
CORTE_TESTE = 7729                  # só descritivo: períodos separados neste jogo
USAR_BANCO = True
N_PROCESSOS = 0                     # 0 = automático; 1 = em série. Não muda resultado
GB_POR_PROCESSO = 2.5
VERIFICAR_LIMPO = True
B_BOOT = 5000
CONFERENCIA_N = 12                  # decisões sorteadas por (funil, janela) para conferir a conta rápida contra a conta direta
SABOTAGEM_N = 3                     # decisões sorteadas por (funil, janela) em que o futuro é trocado por ruído (a escolha não pode mudar)
PASTA_SAIDA = "OUT_FUNIL"
CFG_EXTRA = {}                      # vazio no uso normal (só para testes)
# ========================================================================

AQUI = os.path.dirname(os.path.abspath(__file__))
os.chdir(AQUI)
OUT = os.path.join(AQUI, PASTA_SAIDA)
FILHO = (int(sys.argv[sys.argv.index("--filho") + 1]), int(sys.argv[sys.argv.index("--filho") + 2])) if "--filho" in sys.argv else None
SUF_CAL = r"_(CALR|CAL)$"
FLAGS = FLAGS_SUJO + FLAGS_LIMPO


def log(*a): print(time.strftime("%H:%M:%S"), *a, flush=True)


# ---------------------------------------------------------------- motores ----------------------------------------------------------------
def carregar_motor(arq, tag):
    sp = importlib.util.spec_from_file_location(f"bt_funil_{tag}", os.path.join(AQUI, arq))
    B = importlib.util.module_from_spec(sp); sp.loader.exec_module(B); return B


def fixar_atom(B, tant):
    """5/40/45 não existem no certax/atomg: nesses minutos vale SEMPRE o padrão (ATOM_RECON). Nos minutos do certax fica o ATOM do motor."""
    orig, recon = B.ATOM_BASE, B.ATOM_RECON
    if str(B.ORIGEM_ATOM).startswith("atomg"):
        for t in TEMPOS_NOVOS:
            try:
                d = sorted(set(orig(t)) ^ set(recon(t))); msg = "igual ao padrão" if not d else f"DIFERE do padrão em {d} — usando o PADRÃO"
            except Exception as e:
                msg = f"não gera este minuto ({type(e).__name__}) — usando o PADRÃO"
            log(f"atomg.py no minuto {t}: {msg}")
    B.ATOM_BASE = lambda t, _o=orig, _r=recon, _a=frozenset(int(x) for x in tant): (_o(t) if int(t) in _a else _r(t))


def configurar(B, tag):
    """mesma preparação do main() do motor antes do walk-forward; famílias do motor conforme REGIME_DRAFT"""
    B.CFG.update(CFG_EXTRA); B.CFG["MODO"] = tag
    fams = list(B.CFG["FAMILIAS"])
    if tag == "sujo": fams = [f for f in fams if f not in ("V6L", "V6WRL", "V6LE", "V6WRLE")]      # igual ao main() no modo sujo
    if REGIME_DRAFT == "limpo": fams = [f for f in fams if bool(B.FAM_DEF[f]["so_limpo"]) == (tag == "limpo")]
    B.CFG["FAMILIAS"] = fams
    assert PASSO >= 1 and PASSO_ESCOLHA % PASSO == 0, "PASSO_ESCOLHA tem de ser múltiplo de PASSO"
    B.CFG.update(C_FIXO=float(C_FIXO), RETREINO_DEV_JOGOS=int(PASSO), LOCKBOX_GID_MIN=int(LOCKBOX_SERIO), LOCKBOX_ETAPA="desenvolvimento",
                 LOCKBOX_TREINO="prequencial", PLACEBO=None, PASTA_OUT=OUT)
    B._reset_rngs(); B._pastas(); os.makedirs(OUT, exist_ok=True)
    tant = [int(t) for t in B.CFG["TEMPOS"]]                       # os minutos da etapa 1, NA ORDEM do arquivo (entra na chave do banco)
    tnov = [int(t) for t in TEMPOS_NOVOS if int(t) not in tant]
    for t in tnov: assert t in B.CHECKPOINTS, f"minuto {t} não está nos CHECKPOINTS do motor"
    fixar_atom(B, tant)
    return dict(tag=tag, tant=tant, tnov=tnov, todos=sorted(tant + tnov), min0=dict(B.CFG["MIN_TREINO_JOGOS"]),
                min_nov={**B.CFG["MIN_TREINO_JOGOS"], **{int(k): int(v) for k, v in MIN_TREINO_NOVOS.items()}}, L0=int(LOCKBOX_SERIO))


def motores():
    """{'sujo': (B, K)} e, no regime limpo, {'limpo': (B, K)}"""
    M = {}
    for tag, arq in (("sujo", ARQ_SUJO), ("limpo", ARQ_LIMPO)):
        if tag == "limpo" and REGIME_DRAFT != "limpo": continue
        B = carregar_motor(arq, tag); M[tag] = (B, configurar(B, tag))
    return M


# ---------------------------------------------------------------- dados ----------------------------------------------------------------
def carregar_brutos(B, K, conferir=True):
    brutos = {}
    for t in K["todos"]:
        fp = os.path.join(B.CFG["PASTA_DADOS"], f"{B.CFG['PREFIXO_GRANDE']}{t}.xlsx")
        if not os.path.exists(fp): raise FileNotFoundError(f"falta {fp} — copie zz5/zz40/zz45 (mesmo banco dos outros zz, até o jogo {LOCKBOX_SERIO - 1}) para esta pasta")
        brutos[t] = B.carregar_bruto(t); log(f"t{t}: {B.ORIGEM_DADOS.get(t)} · {len(brutos[t])} linhas · gameid máx {int(pd.to_numeric(brutos[t]['gameid']).max())}")
    if conferir:
        conferir_colunas_brutas(B, K, brutos); conferir_tempos(B, brutos)
        if VERIFICAR_LIMPO:
            for t in K["todos"]:
                if os.path.exists(os.path.join(B.CFG["PASTA_DADOS"], f"{B.CFG['PREFIXO_LIMPO']}{t}.xlsx")): log(f"t{t}: limpo == grande: {B.verificar_limpo(t, brutos[t])}")
                else: log(f"t{t}: sem {B.CFG['PREFIXO_LIMPO']}{t}.xlsx — verificação limpo == grande PULADA neste minuto (registrado)")
    return brutos


def _brutas_da_feature(f):
    m = re.fullmatch(r"(g|D_k|M|D_drake|D_tt)(\d+)", f)
    if not m: return ["Z", "VSZ"] if f == "M_open" else [f]
    k, tau = m.group(1), m.group(2)
    return {"g": [f"g{tau}"], "D_k": [f"k{tau}m", f"d{tau}m"], "M": [f"Z{tau}", f"VSZ{tau}"], "D_drake": [f"drake{tau}", f"opdrake{tau}"], "D_tt": [f"tt{tau}", f"optt{tau}"]}[k]


def conferir_colunas_brutas(B, K, brutos, Bs_todos=None):
    """cada zz{t} tem as colunas que o preparar, as features de TODOS os candidatos (dos 2 motores) e os flags precisam?"""
    draft_raw = ["SOLOQ", "SOLOQ25", "SOLOQ30", "SOLOQ35", "SOLOQ35MAIS", "PRO", "PRO25", "PRO30", "PRO35", "PRO35MAIS"] + B.GOLD5 + B.IMP5
    falt = {}
    for t, G in brutos.items():
        req = ["gameid", "side", "resultado", "Z", "VSZ", f"odd{t}", f"vsodd{t}"] + draft_raw + list(B.COLS_WRPROBS) + list(B.COLS_LOLDRAFT) + FLAGS
        for B_ in (Bs_todos or [B]):
            for (_, fam, var) in B_.candidatos():
                for f in B_.features(fam, var, t):
                    if f == "side" or f.startswith("l_") or f in ("GOLDSUM", "IMPSUM"): continue
                    req += _brutas_da_feature(f)
        fl = sorted(set(c for c in req if c not in G.columns))
        if fl: falt[t] = fl
    if falt: raise KeyError("colunas AUSENTES nos zz (nada foi treinado):\n  - " + "\n  - ".join(f"zz{t}: {v}" for t, v in falt.items()))
    log("colunas dos zz: OK (features dos candidatos, preparar e flags, em todos os minutos)")


def conferir_tempos(B, brutos, ref=10):
    """zz de minutos diferentes vêm do MESMO banco: mesmas (gameid, side) → mesmos valores nas colunas comuns; minuto > 10 ⊆ zz10."""
    R = brutos[ref]; cols = [c for c in ["resultado", "odd", "vsodd", "Z", "VSZ", "g5", "k5m", "d5m", "timeA", "timeB", "PATCH"] + FLAGS if c in R.columns]; linhas = []
    assert not R.duplicated(["gameid", "side"]).any(), f"zz{ref}: (gameid, side) duplicado"
    for t, G in sorted(brutos.items()):
        assert not G.duplicated(["gameid", "side"]).any(), f"zz{t}: (gameid, side) duplicado"
        cc = [c for c in cols if c in G.columns]
        m = G[["gameid", "side"] + cc].merge(R[["gameid", "side"] + cc], on=["gameid", "side"], suffixes=("_t", "_r"), how="inner", validate="one_to_one")
        ruins = [c for c in cc if not B._iguais(m[c + "_t"], m[c + "_r"])]
        assert not ruins, f"zz{t} × zz{ref}: valores diferentes em {ruins} nas mesmas (gameid, side) — gameid renumerado? Nada foi treinado"
        so_t, so_r = len(G) - len(m), len(R) - len(m)
        if t > ref: assert so_t == 0, f"zz{t}: {so_t} linhas que não existem no zz{ref} — nada foi treinado"
        linhas.append(f"t{t}: {len(m)} linhas em comum com o zz{ref} ({len(cc)} colunas iguais) · só no zz{t}: {so_t} · só no zz{ref}: {so_r}")
    log("conferência entre minutos OK:\n    " + "\n    ".join(linhas))


def preparar_frames(B, K, brutos):
    fr = {t: B.preparar(brutos[t], t) for t in K["todos"]}
    for t in K["todos"]:
        fs = [c for (_, fam, var) in B.candidatos() for c in B.features(fam, var, t) if c not in fr[t].columns]
        if fs: raise KeyError(f"[motor {K['tag']} t{t}] features ausentes depois do preparar: {sorted(set(fs))}")
    return fr


def manifesto_features(M):
    man = [dict(motor=tag, cand=n, familia=fam, variante=var, t=t, n_feats=len(B.features(fam, var, t)), feats=" ".join(B.features(fam, var, t)))
           for tag, (B, K) in M.items() for (n, fam, var) in B.candidatos() for t in K["todos"]]
    pd.DataFrame(man).to_csv(os.path.join(OUT, "features_por_minuto.csv"), index=False)
    B = M["sujo"][0]
    for t in M["sujo"][1]["todos"]: log(f"features V6_MOM t{t} ({len(B.features('V6', 'MOM', t))}): {' '.join(B.features('V6', 'MOM', t))}")


def blocos_motor(B, K, frames, ini):
    gid_max = max(int(f["gameid"].max()) for f in frames.values())
    bp = [b for b in B.blocos_teste(gid_max) if b[1] < K["L0"] and b[2] <= K["L0"] and b[1] >= ini]
    assert bp, f"motor {K['tag']}: nenhum bloco de desenvolvimento"
    return bp


def prever(B, K, brutos, frames, blocos, cands):
    """walk-forward do motor com treino a cada PASSO, em 2 grupos de minutos (10–35 = chave da etapa 1 → banco; 5/40/45 = chave própria)"""
    P, F = [], []
    for grupo, tempos, mt in (("etapa1", K["tant"], K["min0"]), ("novos", K["tnov"], K["min_nov"])):
        if not tempos or not cands: continue
        B.CFG["TEMPOS"] = list(tempos); B.CFG["MIN_TREINO_JOGOS"] = dict(mt)
        banco = dict(pasta=B.CFG["PASTA_PREVISOES"], cadeia=B.hash_cadeia({t: brutos[t] for t in tempos})) if USAR_BANCO else None
        log(f"=== previsões motor {K['tag']}, minutos {tempos} ({grupo}) · C = {B.CFG['C_FIXO']} · treino a cada {B.CFG['RETREINO_DEV_JOGOS']} · "
            f"{len(cands)} candidato(s) · jogos {blocos[0][1]}–{blocos[-1][2] - 1}" + (" · com banco" if banco else " · SEM banco"))
        p, f = B.walk_forward_dev({t: frames[t] for t in tempos}, blocos, cands, banco=banco)
        P.append(p.assign(motor=K["tag"])); F.append(f.assign(grupo=grupo, motor=K["tag"]) if len(f) else f)
    B.CFG["TEMPOS"] = list(K["todos"]); B.CFG["MIN_TREINO_JOGOS"] = dict(K["min_nov"])
    return (pd.concat(P, ignore_index=True) if P else pd.DataFrame()), (pd.concat(F, ignore_index=True) if F else pd.DataFrame())


def _cortes(blocos):
    s = set()
    for (_, g0, g1) in blocos:
        g0, g1 = int(g0), int(g1); s.update([g0, g1]); s.update(range((g0 // PASSO + 1) * PASSO, g1, PASSO))
    return np.array(sorted(s), dtype=np.int64)


def conferir_retreino(B, tag, preds, fits, blocos):
    """PROVA pelo log de ajustes: cada treino OK prevê UM par [a, b) com b − a <= PASSO e termina antes de a; 1 treino por (minuto,
    candidato, par); toda previsão (inclusive _CALR) tem o treino do seu par. Aborta se falhar."""
    cortes = _cortes(blocos); f = fits[fits["status"] == "OK"].copy()
    for c in ("gid_treino_max", "gid_teste_min", "gid_teste_max", "n_treino_jogos", "n_feats"): f[c] = pd.to_numeric(f[c]).astype("int64")
    i = np.searchsorted(cortes, f["gid_teste_min"].values, side="right") - 1
    f["par_a"] = cortes[i]; f["par_b"] = cortes[np.minimum(i + 1, len(cortes) - 1)]
    erros = []
    def _conta(msg, m):
        n = int(np.sum(m))
        if n: erros.append(f"{n} treino(s): {msg}")
    _conta("treino alcança o par que ele prevê", f["gid_treino_max"] >= f["par_a"])
    _conta("prevê jogo FORA do seu par", f["gid_teste_max"] >= f["par_b"])
    _conta(f"par com mais de {PASSO} gameids", (f["par_b"] - f["par_a"]) > PASSO)
    _conta("mais de um treino no mesmo (minuto, candidato, par)", f.duplicated(["t", "cand", "par_a"]))
    p0 = preds[["t", "cand", "gameid"]].copy(); p0["cand"] = p0["cand"].str.replace(SUF_CAL, "", regex=True)
    p0["par_a"] = cortes[np.searchsorted(cortes, p0["gameid"].values, side="right") - 1]
    m = p0.drop_duplicates(["t", "cand", "par_a"]).merge(f[["t", "cand", "par_a", "gid_treino_max"]], on=["t", "cand", "par_a"], how="left")
    _conta("previsões sem o treino do próprio par", m["gid_treino_max"].isna())
    _conta("previsões cujo treino alcança o par", m["gid_treino_max"] >= m["par_a"])
    f["folga"] = f["par_a"] - f["gid_treino_max"]; f["jogos_previstos"] = f["gid_teste_max"] - f["gid_teste_min"] + 1; f["EPV_calc"] = f["n_treino_jogos"] / (f["n_feats"] + 1.0)
    tab = f.groupby("t").agg(treinos=("par_a", "size"), candidatos=("cand", "nunique"), pares_distintos=("par_a", "nunique"),
                             jogos_previstos_por_treino_max=("jogos_previstos", "max"), folga_mediana=("folga", "median"), folga_max=("folga", "max"),
                             treino_jogos_min=("n_treino_jogos", "min"), treino_jogos_mediana=("n_treino_jogos", "median"), n_feats_max=("n_feats", "max"),
                             EPV_min=("EPV_calc", "min"), pct_treinos_EPV_baixo=("EPV_calc", lambda s_: float((s_ < float(B.CFG["EPV_ALERTA"])).mean())),
                             gid_teste_min=("gid_teste_min", "min"), gid_teste_max=("gid_teste_max", "max")).reset_index().assign(motor=tag)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    log(f"CONFERÊNCIA DO RE-TREINO, motor {tag} (folga = início do par − último jogo do treino):\n" + tab.to_string(index=False))
    if erros: raise AssertionError(f"RE-TREINO A CADA {PASSO} JOGOS (motor {tag}): FALHOU — nada foi avaliado:\n  - " + "\n  - ".join(erros))
    log(f"RE-TREINO A CADA {PASSO} JOGOS (motor {tag}): OK — {len(f)} treinos; cada um prevê só o seu par e treina só com jogos < início do par")
    return tab


# ---------------------------------------------------------------- base ----------------------------------------------------------------
def montar_base_funil(B, preds, frames, flag_ini):
    """igual ao montar_base do motor (edge, scores, filtro de odd/probabilidade válida) + flags por lado (vazios antes do início do flag)"""
    info = pd.concat([frames[t][["t", "gameid", "side", "y", "odd_t", "vsodd_t", "q_devig_t", "q_raw_t"] + FLAGS] for t in sorted(frames)], ignore_index=True)
    for F in FLAGS:
        info[F] = pd.to_numeric(info[F], errors="coerce"); info.loc[info["gameid"] < flag_ini[F], F] = np.nan
    base = preds.merge(info, on=["t", "gameid", "side"], how="left", validate="many_to_one")
    base["edge"] = base["p"] - base["q_raw_t"]; base["ev"] = base["p"] * base["odd_t"] - 1.0
    base["ll_mod"] = -(base["y"] * np.log(np.clip(base["p"], 1e-6, 1)) + (1 - base["y"]) * np.log(np.clip(1 - base["p"], 1e-6, 1)))
    ok = np.isfinite(base["p"]) & base["q_devig_t"].gt(0) & base["q_devig_t"].lt(1) & np.isfinite(base["odd_t"]) & base["odd_t"].gt(1)
    if (~ok).any(): log(f"base: {int((~ok).sum())} linhas com probabilidade/odd inválida removidas")
    base = base[ok].copy(); assert base["p"].between(0, 1).all()
    qc = base["q_devig_t"].clip(1e-6, 1 - 1e-6)
    base["ll_mkt"] = -(base["y"] * np.log(qc) + (1 - base["y"]) * np.log(1 - qc))
    base["bs_mod"] = (base["p"] - base["y"]) ** 2; base["bs_mkt"] = (base["q_devig_t"] - base["y"]) ** 2
    base["cluster"] = B._clu(base["gameid"]).astype(int)
    return base, info


def _periodo(g, P):
    out = np.full(len(g), "", dtype=object)
    for nome, a, b in P: out[(g >= a) & (g < b)] = nome
    return out


# ---------------------------------------------------------------- Brier / log loss ----------------------------------------------------------------
def _stats(df, P):
    """por (período, minuto) com 'TODOS' e 'TOTAL': n, médias de modelo e mercado, skill (mercado − modelo) e erro-padrão por cluster de 10 gameids"""
    if not len(df): return pd.DataFrame()
    d = df[["t", "gameid", "cluster", "bs_mod", "bs_mkt", "ll_mod", "ll_mkt"]].copy(); d["periodo"] = _periodo(d["gameid"].values, P); d = d[d["periodo"] != ""]
    d["t"] = d["t"].astype(object)
    d = pd.concat([d, d.assign(t="TODOS")], ignore_index=True); d = pd.concat([d, d.assign(periodo="TOTAL")], ignore_index=True)
    d["dbs"] = d["bs_mkt"] - d["bs_mod"]; d["dll"] = d["ll_mkt"] - d["ll_mod"]
    k = ["periodo", "t"]
    c = d.groupby(k + ["cluster"], sort=False).agg(n=("dbs", "size"), sdbs=("dbs", "sum"), sdll=("dll", "sum")).reset_index()
    c["n2"] = c["n"] ** 2; c["s2b"] = c["sdbs"] ** 2; c["s2l"] = c["sdll"] ** 2; c["nsb"] = c["n"] * c["sdbs"]; c["nsl"] = c["n"] * c["sdll"]
    a = c.groupby(k, sort=False).agg(N=("n", "sum"), G=("n", "size"), Sb=("sdbs", "sum"), Sl=("sdll", "sum"), A_b=("s2b", "sum"), A_l=("s2l", "sum"),
                                      Bb=("nsb", "sum"), Bl=("nsl", "sum"), C2=("n2", "sum"))
    m = d.groupby(k, sort=False).agg(n_jogos=("gameid", "nunique"), brier_mod=("bs_mod", "mean"), brier_mkt=("bs_mkt", "mean"), ll_mod=("ll_mod", "mean"), ll_mkt=("ll_mkt", "mean"))
    r = a.join(m).reset_index()
    for s_, S, A, Bx in (("bsskill", "Sb", "A_b", "Bb"), ("llskill", "Sl", "A_l", "Bl")):
        mu = r[S] / r["N"]; num = (r[A] - 2 * mu * r[Bx] + mu ** 2 * r["C2"]).clip(lower=0)
        se = np.sqrt(r["G"] / (r["G"] - 1).clip(lower=1) * num) / r["N"]
        r[s_] = mu; r[s_ + "_se"] = se; r[s_ + "_z"] = mu / se.replace(0, np.nan)
    return r.rename(columns={"N": "n_linhas"})[["periodo", "t", "n_linhas", "n_jogos", "brier_mod", "brier_mkt", "bsskill", "bsskill_se", "bsskill_z",
                                                 "ll_mod", "ll_mkt", "llskill", "llskill_se", "llskill_z"]]


def metricas_previsao(GMf, apostas_de, modelos, flag_ini, start_mod, P):
    """Brier/log loss de várias formas. Linha = (jogo, minuto) com a média dos 2 lados (como o linhas_jogo do motor), exceto 'lado_apostado'.
    Recortes por (modelo, flag): todas | flag1 | flag0 | entrou_<POL> | nao_entrou_<POL> (FIRST: só os minutos até a aposta, que são os
    em que a política decidiu) | lado_apostado_<POL> (p do lado apostado × resultado; mercado = q de-vigado do mesmo lado)."""
    saida = []
    for c in modelos:
        g = GMf[c]
        for F in [None] + FLAGS:
            ini = max(start_mod[c], flag_ini[F] if F else 0)
            if F is None: U = g; rec = [("todas", g)]
            else:
                av = g[g["gameid"] >= ini]
                if not len(av): continue
                U = av[av[F]]; rec = [("flag1", U), ("flag0", av[~av[F]])]
            for pol in ("FIRST", "MULTI"):
                b = apostas_de(c, F, pol, tuple(METRICAS_TEMPOS)); b = b[b["gameid"] >= ini]
                u = U[U["t"].isin(METRICAS_TEMPOS)]
                ent = u.merge(b[["t", "gameid"]].drop_duplicates(), on=["t", "gameid"], how="left", indicator=True)["_merge"].eq("both").values
                if pol == "FIRST":
                    tb = b.groupby("gameid")["t"].min(); lim = u["gameid"].map(tb).fillna(np.inf).values
                    dec = u["t"].values <= lim
                else:
                    dec = np.ones(len(u), bool)
                rec += [(f"entrou_{pol}", u[ent]), (f"nao_entrou_{pol}", u[dec & ~ent])]
                lado = b.assign(bs_mod=(b["p"] - b["y"]) ** 2, bs_mkt=(b["q_devig_t"] - b["y"]) ** 2,
                                ll_mod=-(b["y"] * np.log(np.clip(b["p"], 1e-6, 1)) + (1 - b["y"]) * np.log(np.clip(1 - b["p"], 1e-6, 1))),
                                ll_mkt=-(b["y"] * np.log(b["q_devig_t"]) + (1 - b["y"]) * np.log(1 - b["q_devig_t"])), cluster=(b["gameid"].values - 1) // 10)
                rec.append((f"lado_apostado_{pol}", lado))
            for nome, df in rec:
                s = _stats(df, P)
                if len(s): saida.append(s.assign(modelo=c, flag=F or "sem_flag", recorte=nome))
    R = pd.concat(saida, ignore_index=True)
    return R[["modelo", "flag", "recorte"] + [c for c in R.columns if c not in ("modelo", "flag", "recorte")]]


# ---------------------------------------------------------------- funil ----------------------------------------------------------------
def dados_funil(G, GMf, apostas_de, opcoes, fu):
    """por opção, somas POR JOGO (índice = posição em G): T = linhas 'todas' [n, ll_mod, ll_mkt, bs_mod, bs_mkt] nos minutos do critério e no
    universo do flag; E = o mesmo no lado apostado; Lg/L2/N2 = lucro, lucro², min(lucro,0)²; e as apostas em ordem (gameid, minuto)."""
    nG, O = len(G), len(opcoes); mins = list(fu["minutos_criterio"])
    T = np.zeros((O, nG, 5)); E = np.zeros((O, nG, 5)); Lg = np.zeros((O, nG)); L2 = np.zeros((O, nG)); N2 = np.zeros((O, nG)); BETS = []
    for j, op in enumerate(opcoes):
        r = GMf[op["cand"]]; r = r[r["t"].isin(mins) & (r["gameid"] >= op["w"])]
        if op["flag"]: r = r[r[op["flag"]]]
        ix = np.searchsorted(G, r["gameid"].values); assert np.all(G[np.minimum(ix, nG - 1)] == r["gameid"].values), "linha fora do universo"
        for k, col in enumerate([None, "ll_mod", "ll_mkt", "bs_mod", "bs_mkt"]):
            T[j, :, k] = np.bincount(ix, weights=None if col is None else r[col].values, minlength=nG)
        b = apostas_de(op["cand"], op["flag"], fu["politica"], tuple(fu["tempos"])); b = b[b["gameid"] >= op["w"]].sort_values(["gameid", "t"], kind="mergesort")
        g_, L, p, y, q, t_ = (b[c].values for c in ("gameid", "lucro", "p", "y", "q_devig_t", "t"))
        ib = np.searchsorted(G, g_); assert np.all(G[np.minimum(ib, nG - 1)] == g_), "aposta fora do universo executável"
        llm = -(y * np.log(np.clip(p, 1e-6, 1)) + (1 - y) * np.log(np.clip(1 - p, 1e-6, 1))); llk = -(y * np.log(q) + (1 - y) * np.log(1 - q))
        for k, v in enumerate([None, llm, llk, (p - y) ** 2, (q - y) ** 2]): E[j, :, k] = np.bincount(ib, weights=v, minlength=nG)
        Lg[j] = np.bincount(ib, weights=L, minlength=nG); L2[j] = np.bincount(ib, weights=L ** 2, minlength=nG); N2[j] = np.bincount(ib, weights=np.minimum(L, 0) ** 2, minlength=nG)
        BETS.append(dict(g=g_.astype(np.int64), L=L.astype(float), p=p, y=y, q=q, t=t_))
    return dict(G=G, T=T, E=E, Lg=Lg, L2=L2, N2=N2, bets=BETS)


def _valores_de_somas(t, e, SL, SL2, SN2, nu, mdd, sq):
    """as 16 regras a partir das somas da janela (mesmas definições do scorers() do motor; 'todas' no lugar de 'geral')"""
    nr, nb = t[..., 0], e[..., 0]
    with np.errstate(invalid="ignore", divide="ignore"):
        okr, okb = nr >= MIN_LINHAS, nb >= MIN_LINHAS
        V = dict(ll_todas=np.where(okr, t[..., 1] / nr, np.nan), brier_todas=np.where(okr, t[..., 3] / nr, np.nan),
                 llskill_todas=np.where(okr, (t[..., 2] - t[..., 1]) / nr, np.nan), bsskill_todas=np.where(okr, (t[..., 4] - t[..., 3]) / nr, np.nan),
                 ll_entradas=np.where(okb, e[..., 1] / nb, np.nan), brier_entradas=np.where(okb, e[..., 3] / nb, np.nan),
                 llskill_entradas=np.where(okb, (e[..., 2] - e[..., 1]) / nb, np.nan), bsskill_entradas=np.where(okb, (e[..., 4] - e[..., 3]) / nb, np.nan),
                 lucro=np.asarray(SL, float), ppg=np.where(nu > 0, SL / np.maximum(nu, 1), np.nan))
        roi = np.where(okb, SL / nb, np.nan); sd = np.sqrt(np.clip((SL2 - SL ** 2 / np.maximum(nb, 1)) / np.maximum(nb - 1, 1), 0, None)); down = np.sqrt(SN2 / np.maximum(nb, 1))
        V.update(roi=roi, sharpe=np.where(okb & (sd > 0), roi / sd, np.nan), sortino=np.where(okb & (down > 0), roi / down, np.nan),
                 maxdd=np.where(okb, mdd, np.nan), seq_ruim=np.where(okb, sq, np.nan), calmar=np.where(okb & (mdd > 0), SL / np.where(mdd > 0, mdd, 1), np.nan))
    return V


def valores_rapido(D, j, a_dec, w_dec):
    """as 16 regras da opção j em cada decisão (janela [w, a) em gameid), por somas acumuladas + caminho (maxdd, seq_ruim) por início de janela"""
    G = D["G"]; hi = np.searchsorted(G, a_dec, "left"); lo = np.minimum(np.searchsorted(G, w_dec, "left"), hi)
    CT = np.vstack([np.zeros((1, 5)), np.cumsum(D["T"][j], axis=0)]); CE = np.vstack([np.zeros((1, 5)), np.cumsum(D["E"][j], axis=0)])
    cl = lambda x: np.r_[0.0, np.cumsum(x)]
    C1, C2, C3 = cl(D["Lg"][j]), cl(D["L2"][j]), cl(D["N2"][j])
    bg, L = D["bets"][j]["g"], D["bets"][j]["L"]; mdd = np.zeros(len(a_dec)); sq = np.zeros(len(a_dec))
    for w in np.unique(w_dec):
        m = w_dec == w; sel = bg >= w; Lw, gw = L[sel], bg[sel]
        if not len(Lw): continue
        c = np.cumsum(Lw); dd = np.maximum.accumulate(np.maximum.accumulate(np.r_[0.0, c])[1:] - c)
        neg = Lw < 0; ii = np.arange(len(Lw)); st = np.maximum.accumulate(np.where(neg, ii - np.maximum.accumulate(np.where(~neg, ii, -1)), 0))
        k = np.searchsorted(gw, a_dec[m], "left"); kk = np.maximum(k - 1, 0)
        mdd[m] = np.where(k > 0, dd[kk], 0.0); sq[m] = np.where(k > 0, st[kk], 0)
    return _valores_de_somas(CT[hi] - CT[lo], CE[hi] - CE[lo], C1[hi] - C1[lo], C2[hi] - C2[lo], C3[hi] - C3[lo], hi - lo, mdd, sq)


def valores_direto(B, D, j, a, w):
    """mesma coisa pela conta DIRETA (máscara de jogos e apostas na janela, maxdd/seq_ruim do próprio motor) — confere a conta rápida"""
    G = D["G"]; m = (G >= w) & (G < a); bt = D["bets"][j]; mb = (bt["g"] >= w) & (bt["g"] < a)
    p, y, q, L = bt["p"][mb], bt["y"][mb], bt["q"][mb], bt["L"][mb]
    e = np.array([len(L), np.sum(-(y * np.log(np.clip(p, 1e-6, 1)) + (1 - y) * np.log(np.clip(1 - p, 1e-6, 1)))), np.sum(-(y * np.log(q) + (1 - y) * np.log(1 - q))),
                  np.sum((p - y) ** 2), np.sum((q - y) ** 2)])
    V = _valores_de_somas(D["T"][j][m].sum(0), e, L.sum(), (L ** 2).sum(), (np.minimum(L, 0) ** 2).sum(), int(m.sum()),
                          B._maxdd(L) if len(L) else 0.0, B._seq_ruim(L) if len(L) else 0)
    if len(L) >= MIN_LINHAS: V["sharpe"] = (L.mean() / L.std(ddof=1)) if L.std(ddof=1) > 0 else np.nan     # motor: roi / sd (ddof=1)
    return {k: float(v) for k, v in V.items()}


def elegiveis(G, opcoes, a_dec, min_hist):
    """opção elegível na decisão a: existe (w < a) e tem >= min_hist jogos executáveis na própria janela [w, a)"""
    hi = np.searchsorted(G, a_dec, "left"); W = np.array([op["w"] for op in opcoes], dtype=np.int64); lo = np.searchsorted(G, W, "left")
    return ((hi[:, None] - lo[None, :]) >= min_hist) & (W[None, :] < a_dec[:, None])


def janela_dec(opcoes, El, janela):
    """início da janela de cada (decisão, opção): própria = w da opção; comum = o maior w entre as elegíveis naquela decisão"""
    W = np.array([op["w"] for op in opcoes], dtype=np.int64)
    if janela == "propria": return np.tile(W, (El.shape[0], 1))
    wc = np.where(El, W[None, :], -1).max(axis=1); wc = np.where(wc < 0, W.min(), wc)
    return np.tile(wc[:, None], (1, len(W)))


def matriz_valores(D, opcoes, a_dec, Wd):
    V = {s: np.full((len(a_dec), len(opcoes)), np.nan) for s in SCORERS}
    for j in range(len(opcoes)):
        vj = valores_rapido(D, j, a_dec, Wd[:, j])
        for s in SCORERS: V[s][:, j] = vj[s]
    return V


def escolher(Vs, El):
    """maior valor × SINAL entre as elegíveis com valor finito; empate (|dif| <= 1e-12) → 1ª opção em ordem alfabética (as opções vêm ordenadas)"""
    ok = El & np.isfinite(Vs); X = np.where(ok, Vs, -np.inf); best = X.max(axis=1); tem = ok.any(axis=1)
    idx = np.argmax(X >= (best[:, None] - 1e-12), axis=1)
    return np.where(tem, idx, -1), np.where(tem, best, np.nan)


def serie_procedimento(G, Lg, segs, idx_seg):
    ini = np.array([a for a, _ in segs], dtype=np.int64); fim = np.array([b for _, b in segs], dtype=np.int64)
    s = np.searchsorted(ini, G, side="right") - 1; dentro = (s >= 0) & (G < fim[np.clip(s, 0, None)])
    esc = np.where(dentro, idx_seg[np.clip(s, 0, None)], -1); av = esc >= 0
    return pd.Series(Lg[esc[av], np.where(av)[0]], index=G[av]), esc


def _bt(B, s):
    if len(s) < 30: return dict(media=np.nan, ic_lo=np.nan, ic_hi=np.nan, p_valor=np.nan, n=len(s))
    r = B.boot_media(s.values.astype(float), B._clu(s.index.values), B=B_BOOT); r["n"] = len(s); return r


def _media_se(s):
    """média e erro-padrão por cluster de 10 gameids (para os recortes por período, sem bootstrap)"""
    if len(s) < 2: return (float(s.mean()) if len(s) else np.nan), np.nan
    c = pd.DataFrame(dict(x=s.values, c=(s.index.values - 1) // 10)).groupby("c")["x"].agg(["sum", "size"]); m = c["sum"].sum() / c["size"].sum(); Gc = len(c)
    return float(m), float(np.sqrt(((c["sum"] - c["size"] * m) ** 2).sum() * Gc / max(Gc - 1, 1)) / c["size"].sum())


# ---------------------------------------------------------------- processos ----------------------------------------------------------------
def itens_de_treino(M):
    return [(tag, c) for tag, (B, K) in M.items() for c in B.candidatos()]


def blocos_por_motor(M, frames, hist_ini):
    return {tag: blocos_motor(B, K, frames[tag], hist_ini if tag == "sujo" else 0) for tag, (B, K) in M.items()}


def main_filho(i, n):
    M = motores(); BS, KS = M["sujo"]
    brutos = carregar_brutos(BS, KS, conferir=False); frames = {tag: preparar_frames(B, K, brutos) for tag, (B, K) in M.items()}
    hist_ini = int(HIST_INICIO or BS.CFG["INICIO_TESTE_SUJO"]); BL = blocos_por_motor(M, frames, hist_ini)
    meus = itens_de_treino(M)[i::n]; P, F = [], []
    for tag, (B, K) in M.items():
        cs = [c for (tg, c) in meus if tg == tag]
        if cs:
            p, f = prever(B, K, brutos, frames[tag], BL[tag], cs); P.append(p); F.append(f)
    os.makedirs(os.path.join(OUT, "_partes"), exist_ok=True); arq = os.path.join(OUT, "_partes", f"parte_{i}_de_{n}.pkl")
    with open(arq + ".tmp", "wb") as fh:
        pickle.dump(dict(preds=pd.concat(P, ignore_index=True) if P else pd.DataFrame(), fits=pd.concat(F, ignore_index=True) if F else pd.DataFrame(),
                         itens=[f"{tg}:{c[0]}" for tg, c in meus]), fh, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(arq + ".tmp", arq)
    try:
        import paralelo_v90 as PAR; log(f"pico de memória desta rodada: {PAR.pico_memoria_mb()} MB")
    except Exception: pass


# ---------------------------------------------------------------- principal ----------------------------------------------------------------
def main():
    t0 = time.time(); M = motores(); BS, KS = M["sujo"]
    G_CLEAN, L0 = int(BS.CFG["G_CLEAN"]), KS["L0"]
    hist_ini = int(HIST_INICIO or BS.CFG["INICIO_TESTE_SUJO"]); min_hist = int(MIN_HIST if MIN_HIST is not None else BS.CFG["SELECAO_MIN_HIST_JOGOS"])
    ini_draft = int(INICIO_OPCOES_DRAFT or (BS.CFG["INICIO_TESTE"] if REGIME_DRAFT == "limpo" else G_CLEAN))
    flag_ini = dict(FLAG_INICIO) if FLAG_INICIO else {}
    for F in FLAGS:
        flag_ini.setdefault(F, hist_ini if F in FLAGS_SUJO else (int(BS.CFG.get("PRO_LIVE_FROM") or G_CLEAN) if F.startswith("bipro") else G_CLEAN))
    P = [(n, a, b) for n, a, b in (("sujo", hist_ini, G_CLEAN), (f"limpo_{G_CLEAN}_{BS.CFG['INICIO_TESTE'] - 1}", G_CLEAN, int(BS.CFG["INICIO_TESTE"])),
                                   (f"limpo_{BS.CFG['INICIO_TESTE']}_{CORTE_TESTE - 1}", int(BS.CFG["INICIO_TESTE"]), CORTE_TESTE), (f"limpo_{CORTE_TESTE}_{L0 - 1}", CORTE_TESTE, L0)) if b > a]
    log(f"VALIDAR FUNIL v2 · regime draft {REGIME_DRAFT} · C fixo {C_FIXO} · treino a cada {PASSO} · re-escolha a cada {PASSO_ESCOLHA} · minutos {KS['todos']} · "
        f"histórico desde {hist_ini} · opções com draft desde {ini_draft} · flags (início): {flag_ini} · lockbox {L0}+ NUNCA previsto · períodos {P}")
    for tag, (B, K) in M.items(): log(f"motor {tag}: famílias {B.CFG['FAMILIAS']} · CAL {[c for c in B.CFG['CAL_PARA'] if c.split('_')[0] in B.CFG['FAMILIAS']]} · ATOM {B.ORIGEM_ATOM}")
    brutos = carregar_brutos(BS, KS, conferir=True)
    conferir_colunas_brutas(BS, KS, brutos, [B for (B, _) in M.values()])
    frames = {tag: preparar_frames(B, K, brutos) for tag, (B, K) in M.items()}
    manifesto_features(M)
    BL = blocos_por_motor(M, frames, hist_ini); itens = itens_de_treino(M)
    for tag in BL: log(f"motor {tag}: {len(BL[tag])} blocos de avaliação ({BL[tag][0][1]}–{BL[tag][-1][2] - 1}) → {len(_cortes(BL[tag])) - 1} pares de treino")
    # ---------------- previsões ----------------
    n_pr = 1
    if N_PROCESSOS != 1:
        try:
            import paralelo_v90 as PAR
            n_pr, motivo = PAR.escolher_n(len(itens), N_PROCESSOS, GB_POR_PROCESSO); PAR.anunciar("funil (treino dividido por candidato)", n_pr, n_pr, motivo)
        except ImportError: n_pr = 1
    if n_pr <= 1:
        PS, FS = [], []
        for tag, (B, K) in M.items():
            p, f = prever(B, K, brutos, frames[tag], BL[tag], B.candidatos()); PS.append(p); FS.append(f)
        preds, fits = pd.concat(PS, ignore_index=True), pd.concat(FS, ignore_index=True)
    else:
        del frames
        pasta = os.path.join(OUT, "_partes"); os.makedirs(pasta, exist_ok=True)
        for fn in os.listdir(pasta): os.remove(os.path.join(pasta, fn))
        PAR.rodar_em_paralelo([(f"parte{i}", [os.path.abspath(__file__), "--filho", str(i), str(n_pr)]) for i in range(n_pr)], n_pr, "funil", OUT, cwd=AQUI)
        partes = [pickle.load(open(os.path.join(pasta, f"parte_{i}_de_{n_pr}.pkl"), "rb")) for i in range(n_pr)]
        assert sorted(x for p_ in partes for x in p_["itens"]) == sorted(f"{tg}:{c[0]}" for tg, c in itens), "partes não cobrem todos os candidatos"
        preds = pd.concat([p_["preds"] for p_ in partes if len(p_["preds"])], ignore_index=True); fits = pd.concat([p_["fits"] for p_ in partes if len(p_["fits"])], ignore_index=True)
        frames = {tag: preparar_frames(B, K, brutos) for tag, (B, K) in M.items()}
    assert int(preds["gameid"].max()) < L0, "previsão de jogo do lockbox — ABORTADO"
    fits = BS._alerta_epv_atual(fits)
    for tag in M: conferir_retreino(M[tag][0], tag, preds[preds["motor"] == tag], fits[fits["motor"] == tag], BL[tag])
    preds.to_csv(os.path.join(OUT, "previsoes_funil.csv.gz"), index=False); fits.to_csv(os.path.join(OUT, "ajustes_funil.csv.gz"), index=False)
    # modelos de famílias so_limpo (draft/TUDO/DRAFTR/V6L/V6WRL) só existem a partir de ini_draft (fase limpa)
    so_limpo = {n for tag, (B, K) in M.items() for (n, fam, _) in B.candidatos() if B.FAM_DEF[fam]["so_limpo"]}
    base_c = preds["cand"].str.replace(SUF_CAL, "", regex=True)
    antes = int((base_c.isin(so_limpo) & (preds["gameid"] < ini_draft)).sum()); preds = preds[~(base_c.isin(so_limpo) & (preds["gameid"] < ini_draft))]
    log(f"opções com draft/só-limpo: {antes} previsões antes do jogo {ini_draft} descartadas (fase suja/aquecimento)")
    # ---------------- base, universo, linhas por (jogo, minuto) ----------------
    base, info = montar_base_funil(BS, preds, frames["sujo"], flag_ini)
    BS.CFG["TEMPOS"] = list(KS["todos"]); BS.CFG["TEMPOS_DEPLOY"] = sorted({int(t) for f_ in FUNIS for t in f_["tempos"]} | {int(t) for t in METRICAS_TEMPOS})
    uv, _ = BS.universo_executavel(frames["sujo"])
    G = np.array(sorted(int(g) for g in uv if hist_ini <= int(g) < L0), dtype=np.int64); Gs = set(G.tolist())
    modelos = sorted(base["cand"].unique())
    FLG = info.groupby(["t", "gameid"])[FLAGS].max().eq(1).reset_index()          # (jogo, minuto) com algum lado com flag == 1 (já vazio antes do início do flag)
    base_por_c = {c: d for c, d in base.groupby("cand")}
    GMf, start_mod = {}, {}
    for c in modelos:
        g = BS.linhas_jogo(base_por_c[c], c); g = g[g["gameid"].isin(Gs)]
        GMf[c] = g.merge(FLG, on=["t", "gameid"], how="left").fillna({F: False for F in FLAGS}); start_mod[c] = int(g["gameid"].min()) if len(g) else L0
    memo = {}
    def apostas_de(c, F, pol, tempos):
        k = (c, F, pol, tempos)
        if k not in memo:
            b, _ = BS.apostas(base_por_c[c], c, pol, tempos=list(tempos), filtro_col=F)
            memo[k] = b[b["gameid"].isin(Gs)][["gameid", "t", "side", "p", "y", "q_devig_t", "odd_t", "edge", "lucro"]].copy()
        return memo[k]
    log(f"universo executável (minutos {BS.CFG['TEMPOS_DEPLOY']}) {hist_ini}–{L0 - 1}: {len(G)} jogos (denominador do PPG) · {len(modelos)} modelos")
    # ---------------- Brier / log loss de várias formas ----------------
    log("métricas de Brier / log loss (todas, flag=1, flag=0, entrou/não entrou, lado apostado)…")
    MT = metricas_previsao(GMf, apostas_de, modelos, flag_ini, start_mod, P)
    MT.to_csv(os.path.join(OUT, "metricas_previsao_por_minuto.csv.gz"), index=False)
    MT[MT["t"].astype(str) == "TODOS"].to_csv(os.path.join(OUT, "metricas_previsao_resumo.csv"), index=False)
    # ---------------- opções do funil ----------------
    opcoes = []
    for c in modelos:
        for F in [None] + FLAGS:
            st = max(start_mod[c], flag_ini[F] if F else 0, hist_ini)
            if st < L0: opcoes.append(dict(nome=c if F is None else f"{c}|{F}", cand=c, flag=F, w=int(st)))
    opcoes = sorted(opcoes, key=lambda o: o["nome"]); nomes = [o["nome"] for o in opcoes]
    log(f"{len(opcoes)} opções (modelo × flag) · início mais cedo {min(o['w'] for o in opcoes)} · {sum(o['w'] < G_CLEAN for o in opcoes)} existem na fase suja")
    cortes = _cortes(BL["sujo"]); cortes = cortes[cortes >= hist_ini]
    if PASSO_ESCOLHA != PASSO: cortes = np.array(sorted({int(c) for c in cortes if (c - cortes[0]) % PASSO_ESCOLHA == 0} | {int(cortes[-1])}), dtype=np.int64)
    SEG = {f"a_cada_{PASSO_ESCOLHA}": [(int(a), int(b)) for a, b in zip(cortes[:-1], cortes[1:])], "bloco_motor": [(int(g0), int(g1)) for (_, g0, g1) in BL["sujo"]]}
    a_all = np.array(sorted({a for s in SEG.values() for a, _ in s}), dtype=np.int64); pos = {int(a): i for i, a in enumerate(a_all)}
    El = elegiveis(G, opcoes, a_all, min_hist); rng = np.random.default_rng(int(BS.CFG["SEED"]))
    linhas, pareados, trilhas, conf = [], [], [], []
    for fu in FUNIS:
        log(f"funil {fu['nome']}: montando as somas por jogo de {len(opcoes)} opções…"); D = dados_funil(G, GMf, apostas_de, opcoes, fu)
        for jan in JANELAS:
            Wd = janela_dec(opcoes, El, jan); V = matriz_valores(D, opcoes, a_all, Wd)
            # conferência: conta rápida == conta direta; e o futuro trocado por ruído não muda a escolha
            amostra = rng.choice(np.where(El.any(axis=1))[0], size=min(CONFERENCIA_N, int(El.any(axis=1).sum())), replace=False)
            dif_max, esc_dif = 0.0, 0
            for d in amostra:
                for j in np.where(El[d])[0]:
                    vd = valores_direto(BS, D, j, a_all[d], Wd[d, j])
                    for s in SCORERS:
                        x, y_ = V[s][d, j], vd[s]
                        if np.isfinite(x) or np.isfinite(y_):
                            dif = abs(x - y_) / max(1.0, abs(y_)) if (np.isfinite(x) and np.isfinite(y_)) else np.inf; dif_max = max(dif_max, dif)
            if dif_max > 1e-7: raise AssertionError(f"{fu['nome']}/{jan}: conta rápida ≠ conta direta (dif relativa {dif_max:.2e}) — nada foi gravado")
            for d in amostra[:SABOTAGEM_N]:
                a_s = a_all[d]; D2 = dict(D, T=D["T"].copy(), E=D["E"].copy(), Lg=D["Lg"].copy(), L2=D["L2"].copy(), N2=D["N2"].copy(), bets=[dict(b) for b in D["bets"]])
                fut = G >= a_s; nf = int(fut.sum())
                D2["T"][:, fut, 1:] = rng.normal(0, 5, size=D2["T"][:, fut, 1:].shape); D2["E"][:, fut, 1:] = rng.normal(0, 5, size=D2["E"][:, fut, 1:].shape)
                D2["Lg"][:, fut] = rng.normal(0, 5, size=(len(opcoes), nf))
                for b in D2["bets"]:
                    mf = b["g"] >= a_s; b["L"] = b["L"].copy(); b["L"][mf] = rng.normal(0, 5, size=int(mf.sum()))
                V2 = matriz_valores(D2, opcoes, a_all, Wd); ate = a_all <= a_s
                for s in SCORERS:
                    if not np.array_equal(escolher(V[s][ate] * SINAL[s], El[ate])[0], escolher(V2[s][ate] * SINAL[s], El[ate])[0]): esc_dif += 1
            if esc_dif: raise AssertionError(f"{fu['nome']}/{jan}: a escolha MUDOU com o futuro trocado por ruído ({esc_dif} casos) — VAZAMENTO; nada foi gravado")
            conf.append(dict(funil=fu["nome"], janela=jan, decisoes_conferidas=len(amostra), dif_relativa_max=dif_max, sabotagens=min(SABOTAGEM_N, len(amostra)), escolhas_alteradas=esc_dif))
            log(f"{fu['nome']}/{jan}: conta rápida × direta em {len(amostra)} decisões (dif máx {dif_max:.1e}) · sabotagem do futuro em {min(SABOTAGEM_N, len(amostra))} decisões: 0 escolhas alteradas")
            series = {}
            for s in SCORERS:
                idx_all, val_all = escolher(V[s] * SINAL[s], El)
                for cad, segs in SEG.items():
                    ii = np.array([pos[a] for a, _ in segs]); lu, esc_g = serie_procedimento(G, D["Lg"], segs, idx_all[ii]); series[(s, cad)] = (lu, esc_g)
                    for (a, b), d in zip(segs, ii):
                        trilhas.append((fu["nome"], jan, s, cad, a, b, int(El[d].sum()), nomes[idx_all[d]] if idx_all[d] >= 0 else None, val_all[d] * SINAL[s] if idx_all[d] >= 0 else np.nan))
            comuns = None
            for (lu, _) in series.values(): comuns = set(lu.index) if comuns is None else comuns & set(lu.index)
            js = np.array(sorted(comuns), dtype=np.int64)
            ii2 = np.array([pos[a] for a, _ in SEG[f"a_cada_{PASSO_ESCOLHA}"]]); sg2 = SEG[f"a_cada_{PASSO_ESCOLHA}"]
            ini2 = np.array([a for a, _ in sg2]); sgi = np.clip(np.searchsorted(ini2, js, side="right") - 1, 0, None); gi = np.searchsorted(G, js)
            Elg = El[ii2[sgi]]
            media = pd.Series(np.where(Elg, D["Lg"][:, gi].T, 0.0).sum(1) / np.maximum(Elg.sum(1), 1), index=js)
            sempre = np.where(Elg.all(axis=0))[0]
            refs = {"média das opções elegíveis (escolher ao acaso)": media}
            if len(sempre):
                jb = sempre[np.argmax(D["Lg"][sempre][:, gi].sum(1))]
                refs[f"melhor fixo EM RETROSPECTO ({nomes[jb]}) — ENVIESADO"] = pd.Series(D["Lg"][jb, gi], index=js)
            for (s, cad), (lu, esc_g) in list(series.items()) + [((r, "-"), (sr, None)) for r, sr in refs.items()]:
                r_ = _bt(BS, lu); rc = _bt(BS, lu.reindex(js)) if cad != "-" else _bt(BS, lu)
                per = {}
                for (pn, pa, pb) in P:
                    x = lu.reindex(js[(js >= pa) & (js < pb)]); m_, se_ = _media_se(x); per[f"ppg_{pn}"] = m_; per[f"se_{pn}"] = se_; per[f"n_{pn}"] = len(x)
                extra = {}
                if esc_g is not None:
                    ev = esc_g[np.isin(G, js)]
                    extra = dict(trocas=int((ev[1:] != ev[:-1]).sum()), pct_jogos_com_flag=float(np.mean([opcoes[k]["flag"] is not None for k in ev])) if len(ev) else np.nan,
                                 mais_escolhidas=" · ".join(f"{nomes[k]} {v:.0%}" for k, v in (pd.Series(ev).value_counts(normalize=True).head(3)).items()))
                    am = []
                    for k in np.unique(ev):
                        gs = js[ev == k]; bt = D["bets"][k]; am.append(bt["t"][np.isin(bt["g"], gs)])
                    tt = pd.Series(np.concatenate(am) if am else []).value_counts().sort_index(); extra["apostas_por_minuto"] = " · ".join(f"t{int(t)} {n}" for t, n in tt.items())
                linhas.append(dict(funil=fu["nome"], politica=fu["politica"], janela=jan, regra=s, cadencia=cad, n_jogos=r_["n"], ppg=r_["media"], ic_lo=r_["ic_lo"], ic_hi=r_["ic_hi"],
                                   p_unilateral=r_["p_valor"], n_jogos_comuns=rc["n"], ppg_comum=rc["media"], ic_lo_comum=rc["ic_lo"], ic_hi_comum=rc["ic_hi"], p_comum=rc["p_valor"], **per, **extra))
            for s in SCORERS:
                a2, b2 = series[(s, f"a_cada_{PASSO_ESCOLHA}")][0].reindex(js), series[(s, "bloco_motor")][0].reindex(js); rd = _bt(BS, a2 - b2)
                pareados.append(dict(funil=fu["nome"], janela=jan, regra=s, n_jogos=len(js), delta_ppg_2_menos_bloco=rd["media"], ic_lo=rd["ic_lo"], ic_hi=rd["ic_hi"], p_unilateral_2_melhor=rd["p_valor"]))
            log(f"{fu['nome']}/{jan}: {len(js)} jogos comuns · " + " · ".join(f"{s} {series[(s, f'a_cada_{PASSO_ESCOLHA}')][0].reindex(js).mean():+.3f}" for s in SCORERS[:4] + ["ppg"]))
        del D
    R = pd.DataFrame(linhas); PR = pd.DataFrame(pareados)
    R.to_csv(os.path.join(OUT, "resumo_funil.csv"), index=False); PR.to_csv(os.path.join(OUT, "pareado_2_vs_bloco.csv"), index=False)
    pd.DataFrame(conf).to_csv(os.path.join(OUT, "conferencia_regras.csv"), index=False)
    pd.DataFrame(trilhas, columns=["funil", "janela", "regra", "cadencia", "a", "b", "opcoes_elegiveis", "escolhida", "valor_historico"]).to_csv(os.path.join(OUT, "trilhas.csv.gz"), index=False)
    pd.DataFrame(opcoes).to_csv(os.path.join(OUT, "opcoes.csv"), index=False)
    escrever_resumo(R, PR, MT, P, dict(regime=REGIME_DRAFT, C=C_FIXO, hist_ini=hist_ini, ini_draft=ini_draft, flag_ini=flag_ini, n_opcoes=len(opcoes), n_G=len(G), t0=t0))
    json.dump(dict(regime=REGIME_DRAFT, motores={tag: hashlib.sha256(open(os.path.join(AQUI, ARQ_SUJO if tag == "sujo" else ARQ_LIMPO), "rb").read()).hexdigest() for tag in M},
                   script_sha256=hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest(), C_FIXO=C_FIXO, PASSO=PASSO, PASSO_ESCOLHA=PASSO_ESCOLHA,
                   tempos=KS["todos"], min_treino=KS["min_nov"], lockbox_serio=L0, hist_inicio=hist_ini, inicio_opcoes_draft=ini_draft, flag_inicio=flag_ini, min_hist=min_hist,
                   min_linhas=MIN_LINHAS, scorers=SCORERS, janelas=JANELAS, funis=FUNIS, periodos=P, opcoes=nomes),
              open(os.path.join(OUT, "config_funil.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)


def escrever_resumo(R, PR, MT, P, info):
    f_ = lambda v, fmt="+.4f": "n/d" if v is None or (isinstance(v, float) and not np.isfinite(v)) else format(v, fmt)
    cad2 = f"a_cada_{PASSO_ESCOLHA}"
    txt = [f"VALIDAÇÃO DO FUNIL v2 — {time.strftime('%Y-%m-%d %H:%M')} · regime draft {info['regime']} · C fixo {info['C']} · {info['n_opcoes']} opções (modelo × flag) · "
           f"{info['n_G']} jogos executáveis desde {info['hist_ini']}",
           f"Treino a cada {PASSO} jogos (conferencia_retreino no log) · re-escolha a cada {PASSO_ESCOLHA} com o histórico ACUMULADO · opções com draft desde {info['ini_draft']} · "
           f"início dos flags {info['flag_ini']}",
           "PPG = lucro ÷ jogos executáveis · 'comum' = mesmos jogos para todas as regras e cadências do funil · IC 95% bootstrap por cluster de 10 gameids · p unilateral (PPG > 0)", ""]
    for (fu, jan), r in R.groupby(["funil", "janela"], sort=False):
        txt.append(f"=== {fu} · janela {jan} ({int(r['n_jogos_comuns'].max())} jogos comuns) — PPG comum [IC] · por período · Δ(a cada {PASSO_ESCOLHA} − bloco) · trocas · % jogos com flag · mais escolhidas")
        for s in SCORERS:
            a = r[(r.regra == s) & (r.cadencia == cad2)].iloc[0]; b = r[(r.regra == s) & (r.cadencia == "bloco_motor")].iloc[0]
            pr = PR[(PR.funil == fu) & (PR.janela == jan) & (PR.regra == s)].iloc[0]
            per = " ".join(f_(a[f"ppg_{pn}"], "+.3f") for pn, _, _ in P)
            txt.append(f"   {s:<17} a cada {PASSO_ESCOLHA}: {f_(a.ppg_comum)} [{f_(a.ic_lo_comum)}; {f_(a.ic_hi_comum)}] | bloco: {f_(b.ppg_comum)} | Δ {f_(pr.delta_ppg_2_menos_bloco)} "
                       f"p={f_(pr.p_unilateral_2_melhor, '.3f')} | períodos {per} | trocas {int(a.trocas)} | flag {f_(a.pct_jogos_com_flag, '.0%')} | {a.mais_escolhidas}")
        for _, x in r[r.cadencia == "-"].iterrows(): txt.append(f"   {x.regra}: {f_(x.ppg)} [{f_(x.ic_lo)}; {f_(x.ic_hi)}]")
        txt.append(f"   (períodos: {', '.join(pn for pn, _, _ in P)})"); txt.append("")
    t = MT[(MT["t"].astype(str) == "TODOS") & (MT["periodo"] == "TOTAL")]
    txt.append("BRIER / LOG LOSS (todas as previsões, todos os minutos, período TOTAL; skill = mercado − modelo, + = modelo melhor; z = skill / erro-padrão por cluster):")
    for _, x in t[t.recorte == "todas"].sort_values("bsskill", ascending=False).iterrows():
        txt.append(f"   {x.modelo:<16} n={int(x.n_linhas):>7} · Brier {x.brier_mod:.4f} (mercado {x.brier_mkt:.4f}) skill {x.bsskill:+.5f} z={f_(x.bsskill_z, '+.2f')} · "
                   f"log loss {x.ll_mod:.4f} (mercado {x.ll_mkt:.4f}) skill {x.llskill:+.5f} z={f_(x.llskill_z, '+.2f')}")
    txt.append("   flags (V6_MOM, se existir): flag=1 × flag=0 × entrou/não entrou → metricas_previsao_resumo.csv")
    for _, x in t[(t.modelo == "V6_MOM") & (t.flag != "sem_flag") & t.recorte.isin(["flag1", "flag0"])].iterrows():
        txt.append(f"   V6_MOM {x.flag:<13} {x.recorte}: n={int(x.n_linhas)} Brier skill {x.bsskill:+.5f} (z {f_(x.bsskill_z, '+.2f')}) · log loss skill {x.llskill:+.5f} (z {f_(x.llskill_z, '+.2f')})")
    txt += ["", "LEIA COM CUIDADO:",
            " - Desenvolvimento: jogos já vistos em análises; mede o MÉTODO, não confirma nada. A confirmação é o lockbox (8448+).",
            f" - {len(FUNIS)} funis × {len(JANELAS)} janelas × {len(SCORERS)} regras × 2 cadências: escolher a regra pelo maior PPG desta tabela é seleção múltipla.",
            " - Regras 'todas'/'entradas' comparam opções em LINHAS diferentes (cada flag tem o seu universo): o skill é relativo ao mercado nas mesmas linhas, o Brier/log loss cru não.",
            " - Na janela 'comum', lucro e ppg escolhem igual (mesmo denominador). 'melhor fixo em retrospecto' é teto enviesado.",
            f"pronto em {(time.time() - info['t0']) / 60:.1f} min"]
    open(os.path.join(OUT, "RESUMO_FUNIL.txt"), "w", encoding="utf-8").write("\n".join(txt)); print("\n".join(txt), flush=True)


if __name__ == "__main__":
    if FILHO: main_filho(*FILHO)
    else: main()
