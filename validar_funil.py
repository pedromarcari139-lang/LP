# -*- coding: utf-8 -*-
"""
validar_funil.py — v5.0 (03/10/2026: + EV>0/5%/10%, CLV e markout, modo sombra, ROI × esperado, variância, rolling/alarmes, gráficos; v4.9: + modo CONTINUAÇÃO com os jogos do antigo lockbox; v4.8: + PRÓXIMA escolha e últimas escolhas de cada regra; v4.7: + escolha POR MINUTO e ensemble do melhor de cada minuto; v4.6: avaliação sem depender da versão do motor — roda com o motor v9.0; minutos 10–35; auditoria de vazamento, livro de apostas, INICIO_PREVISOES, paralelismo próprio) — VALIDAÇÃO DO FUNIL com re-treino E re-escolha a cada 2 jogos, opções (modelo × flag),
MÉTRICAS do histórico de cada opção, REGRAS de escolha (simples ou compostas) e avaliação de cada regra por lucro, RISCO,
VOLATILIDADE e OVERFIT; Brier / log loss olhados de várias formas.

CAMADAS
  1) PREVISÕES (motor sem edição; treino refeito a cada PASSO = 2 gameids; conferir_retreino() PROVA pelo log de ajustes):
     REGIME_DRAFT = "sujo" (padrão = o método do app): TODOS os modelos treinam com o DataFrame inteiro (do início até o jogo anterior);
       os modelos com draft (DRAFT/TUDO/DRAFTR, e as _CALR deles) só viram OPÇÃO quando a fase limpa começa (G_CLEAN = 6028), porque a
       partir dali o draft é capturado ao vivo — o histórico deles para a escolha só tem jogos limpos.
     REGIME_DRAFT = "limpo" (alternativa): as famílias "so_limpo" vêm do backtest_limpo.py (treinam só com jogos >= 6028, opção a partir
       de 6628); V6/V6WR(E) vêm do sujo (previsão idêntica nos dois motores, conferida).
     Minutos 5/40/45 não existem no certax/atomg: features = padrão (ATOM_RECON): por checkpoint τ <= t entram g_τ, D_k_τ, M_τ;
       D_drake_τ só com τ >= 10 (no t5 não há drake5); D_tt_τ só com τ >= 15; fixos side, M_open.
  2) OPÇÕES = (modelo, flag). Flag = só aposta se o lado escolhido tiver flag == 1 (regra do filtro_col do motor).
       fase suja (< 6028): modelos sem draft; flags cWRgrande, cPROBSgrande (e "sem flag")
       fase limpa: todos; + bisolo, bipro, bisolo30, bipro30, bisolo35mais, bipro35mais
  3) MÉTRICAS (METRICAS): calculadas em cada decisão, só com o histórico [w, a) de cada opção (ver dicionário METRICAS abaixo).
  4) REGRAS (REGRAS + REGRAS_EXTRAS): cada uma escolhe UMA opção por decisão a partir das métricas. Três formas:
       "bsskill_todas"                                  → a opção com o melhor valor dessa métrica (SENTIDO diz se maior ou menor é melhor)
       dict(nome=..., ordenar="brier_todas", filtros=[("roi", ">=", 0.02), ("n_apostas", ">=", 100)], sem_candidato=...)
            → a melhor em "ordenar" ENTRE as que passam em todos os filtros. Se nenhuma passa (sem_candidato):
              "melhor_sem_filtro" = a melhor ignorando os filtros · "relaxar" = tira os filtros do último para o 1º até alguém passar ·
              "nao_apostar" = não aposta naquele par (lucro 0, o jogo conta no PPG)
       dict(nome=..., funcao=f)                          → f(M, El) devolve o índice da opção por decisão (−1 = sem decisão, −2 = não apostar);
            M = {métrica: matriz decisões × opções}, El = elegíveis (decisões × opções); opções em ordem alfabética (opcoes.csv)
     Re-escolha a cada PASSO_ESCOLHA = 2 jogos com o histórico ACUMULADO, em 2 janelas ("propria" / "comum"); a mesma regra por bloco do
     motor (300/600/150) nos mesmos jogos, para comparação pareada.
  5) AVALIAÇÃO de cada regra: PPG [IC], por período; ROI, nº de apostas; volatilidade, Sharpe/Sortino por jogo, max drawdown, pior
     sequência, calmar, % de blocos de 150 jogos positivos e pior bloco (RISCO/VOLATILIDADE); otimismo = PPG histórico prometido pela
     escolha − PPG realizado, percentil da escolha entre as elegíveis no bloco seguinte, % de blocos acima da escolha ao acaso, trocas
     (OVERFIT/estabilidade). COMPARAÇÃO entre regras: Δ par a par (comparar_regras.csv), Model Confidence Set e Reality Check contra a
     escolha ao acaso (funções do motor).
  5b) REFERÊNCIAS nos mesmos jogos: BASE (V6_MOM fixo), TODOS os modelos fixos sem flag (MOM e INI) e MOM − INI de cada um, favorito,
     zebra, vig do mercado, acaso (média) e a DISTRIBUIÇÃO DO ACASO (N_ACASO sequências sorteando uma opção elegível a cada par: cada regra
     ganha o seu percentil e p). Tudo também com as regras escolhendo só entre opções INI (CONJUNTOS["so_INI"]).
  6) BRIER / LOG LOSS (metricas_previsao_*.csv): modelo × mercado × skill por período e minuto, nos recortes todas | flag1 | flag0 |
     entrou / não entrou (FIRST e MULTI) | lado apostado.
NÃO prevê, não lê resultado e não avalia jogo >= LOCKBOX_SERIO (8448). Não escolhe C, não grava DECISAO, não toca na etapa 1.
USO: python validar_funil.py  (ou RODAR_FUNIL.bat, na pasta do RODAR.bat). Essencial em OUT_FUNIL\\RESUMO_FUNIL.txt.
"""
import os, sys, re, json, time, pickle, hashlib, importlib.util, operator, zlib
import numpy as np
import pandas as pd

# ================================ CONFIG ================================
REGIME_DRAFT = "sujo"               # "sujo" (padrão: DataFrame inteiro no treino, draft vira opção na fase limpa) | "limpo" (ver acima)
ARQ_SUJO, ARQ_LIMPO = "backtest_sujo.py", "backtest_limpo.py"
C_FIXO = 1.0                        # C ÚNICO. 1.0 = valor a priori (padrão do sklearn; está na grade → banco). O C da etapa 1 foi escolhido com estes jogos (otimista)
LOCKBOX_SERIO = 8448                # nada >= este gid é previsto
PASSO = 2                           # re-treino a cada PASSO gameids
PASSO_ESCOLHA = 2                   # re-escolha a cada PASSO_ESCOLHA gameids (múltiplo de PASSO)
TEMPOS_NOVOS = []                   # v4.5 (Pedro): só os minutos do motor (10–35). Para acrescentar minutos: ex. [5, 40] (precisa de zz5/zz40)
MIN_TREINO_NOVOS = {5: 150, 40: 120, 45: 120}
INICIO_PREVISOES = None             # 1º jogo PREVISTO (motor sujo). None = INICIO_TESTE_SUJO do motor (700, o da etapa 1: o banco é reaproveitado).
                                    # Menor (par, ex. 300) = o funil começa antes, com os 1ºs modelos treinados em menos jogos (MIN_TREINO_JOGOS ainda vale);
                                    # os treinos a partir do 700 continuam vindo do banco (os pares são os mesmos)
HIST_INICIO = None                  # None = INICIO_PREVISOES: a fase suja entra no funil (opções sem draft, flags de WR)
INICIO_OPCOES_DRAFT = None          # None = max(G_CLEAN, PRO_LIVE_FROM) = 6115 no regime sujo (AUDITORIA 02/10: no modo sujo o motor NÃO mascara o PRO de
                                    #        backfill 6028–6114 e todo modelo com draft usa l_PRO*; antes de 6115 a previsão deles usa PRO de backfill);
                                    #        INICIO_TESTE (6628) no regime limpo
EMBARGO_ESCOLHA = 0                 # a escolha antes do par [a, a+2) usa só jogos < a − EMBARGO_ESCOLHA (0 = supõe que o par anterior já terminou;
                                    # use 2 ou mais se houver mais de 2 jogos ao mesmo tempo). O TREINO dos modelos tem o seu próprio (EMBARGO_JOGOS do motor)
MIN_HIST = None                     # None = SELECAO_MIN_HIST_JOGOS (300): opção só é elegível com >= isto de jogos executáveis na própria janela
MIN_LINHAS = 30                     # métricas de linhas/apostas (todas*, entradas*, roi, sharpe, sortino, maxdd, seq_ruim, calmar, t_*) exigem >= isto
JANELA_RECENTE = 300                # métricas "_recente": só os últimos N jogos executáveis da janela
FLAGS_SUJO = ["cWRgrande", "cPROBSgrande"]
FLAGS_LIMPO = ["bisolo", "bipro", "bisolo30", "bipro30", "bisolo35mais", "bipro35mais"]
FLAG_INICIO = None                  # None = WR: desde HIST_INICIO; bisolo*: G_CLEAN; bipro*: PRO_LIVE_FROM (6115: PRO de 6028–6114 é backfill)

# MÉTRICAS do histórico de cada opção (todas usadas como regra simples, filtro ou ordenação). SENTIDO: +1 = maior é melhor, −1 = menor
METRICAS = {
    "bsskill_todas": +1, "llskill_todas": +1, "brier_todas": -1, "ll_todas": -1,            # todas as previsões (universo do flag, minutos do critério); skill = mercado − modelo
    "bsskill_entradas": +1, "llskill_entradas": +1, "brier_entradas": -1, "ll_entradas": -1,  # só o lado apostado
    "ppg": +1, "lucro": +1, "roi": +1, "sharpe": +1, "sortino": +1, "calmar": +1,            # lucro das apostas (sharpe/sortino por aposta, como o motor)
    "maxdd": -1, "seq_ruim": -1,                                                              # risco (por aposta, ordem jogo → minuto)
    "n_apostas": +1, "n_linhas": +1,                                                          # tamanho da amostra (bom para filtros)
    "vol_ppg": -1,                                                                            # volatilidade do lucro por jogo executável
    "cobertura_modelo": +1,                                                                   # fração dos jogos executáveis da janela em que o MODELO (sem flag) tem previsão
    "t_ppg": +1, "t_bsskill_todas": +1,                                                       # média ÷ erro-padrão (consistência; sem cluster — use como filtro, não como p-valor)
    "ppg_recente": +1, "bsskill_recente": +1, "queda_ppg": +1,                                # últimos JANELA_RECENTE jogos; queda_ppg = ppg_recente − ppg (negativo = piorando)
    "clv_medio": +1, "clv_mk": +1, "n_clv": +1,                                               # v5.0: CLV das apostas da janela: média de odd·q_fech − 1; média de
}                                                                                             #       q_fech − q (movimento do mercado a favor, referência 0); nº com fechamento
SENTIDO = dict(METRICAS)
REGRAS = ["bsskill_todas", "llskill_todas", "brier_todas", "ll_todas", "bsskill_entradas", "llskill_entradas", "brier_entradas", "ll_entradas",
          "ppg", "lucro", "roi", "sharpe", "sortino", "calmar", "maxdd", "seq_ruim",               # as 16 regras simples de antes (uma por métrica)
          "clv_medio", "clv_mk",                                                                   # v5.0: CLV (fechamento) do histórico da opção
          dict(nome="clv_mk_com_skill", ordenar="clv_mk", filtros=[("bsskill_todas", ">", 0.0)], sem_candidato="relaxar")]
                                    # v5.0 (pedido: "métrica de decisão = superação do fechamento + ganho sobre a linha de base do mercado"): o maior movimento
                                    # médio do mercado a favor até o fechamento (clv_mk) ENTRE as opções com Brier skill > 0; ninguém → sem o filtro.
                                    # REVISÃO 03/10: a TAXA de superação (clv_taxa) NÃO tem 50% como referência — num mercado sem viés ela depende do preço
                                    # (zebra ≈ 37%, favorito > 50%), então não decide nada; clv_mk = média de q_fech − q tem referência 0 para qualquer preço
REGRAS_EXTRAS = [                                                                              # regras COMPOSTAS (vazio = só as simples). Exemplos:
    # dict(nome="brier_com_roi>=2%", ordenar="brier_todas", filtros=[("roi", ">=", 0.02)], sem_candidato="melhor_sem_filtro"),
    # dict(nome="bsskill_com_t_ppg>=1_e_100apostas", ordenar="bsskill_todas", filtros=[("t_ppg", ">=", 1.0), ("n_apostas", ">=", 100)], sem_candidato="relaxar"),
    # dict(nome="ppg_se_nao_esta_piorando", ordenar="ppg", filtros=[("queda_ppg", ">=", 0.0)], sem_candidato="nao_apostar"),
    # dict(nome="minha_regra", funcao=lambda M, El: ...),
]
JANELAS = ["propria", "comum"]
CONJUNTOS = {                       # conjuntos de opções em que as regras escolhem (cada um é avaliado separado, nos mesmos jogos)
    "todas": lambda op: True,
    "so_INI": lambda op: bool(re.search(r"_INI(_|$)", op["cand"])),     # só modelos SEM a odd do momento (variante INI: só abertura)
}
SALVAR_APOSTAS = [("propria", "todas"), ("comum", "todas")]   # (janela, conjunto) cujo LIVRO DE APOSTAS do procedimento (cadência a cada 2) é gravado
                                    # em apostas_procedimento.csv.gz: cada aposta de cada regra, com o par, a opção escolhida, odd, p, resultado, lucro e acumulado
N_ACASO = 2000                      # sequências sorteadas da DISTRIBUIÇÃO DO ACASO (a cada par, uma opção elegível ao acaso)
_T_TODOS = [5, 10, 15, 20, 25, 30, 35, 40, 45]
_T_ANT = [10, 15, 20, 25, 30, 35]
LIMIARES_EV = [None, 0.05, 0.10]    # v5.0: TODAS as técnicas com 3 regras de entrada. None = a de hoje: edge = p − 1/odd > 0 (EDGE_MIN 0, LIMIAR_EM "edge"
                                    # do motor), que é EXATAMENTE EV = p·odd − 1 > 0 (odd > 0). 0.05 = só aposta se, além disso, EV > 5%; 0.10 = EV > 10%.
                                    # O lado continua o de maior edge (CRITERIO_LADO do motor); o limiar vale para esse lado. Funis com sufixo _EV5 / _EV10
_SUF_EV = lambda ev: "" if ev is None else f"_EV{int(round(ev * 100))}"
FUNIS_BASE = [   # nome, política, minutos em que pode apostar, minutos das métricas "todas"
    dict(nome="FIRST_10a35", politica="FIRST", tempos=_T_ANT,   minutos_criterio=_T_ANT),
    dict(nome="MULTI_10a35", politica="MULTI", tempos=_T_ANT,   minutos_criterio=_T_ANT),
]
FUNIS = [dict(f, nome=f["nome"] + _SUF_EV(ev), ev_min=ev) for ev in LIMIARES_EV for f in FUNIS_BASE]
FUNIS_POR_MINUTO = True             # v4.7: + um funil POR MINUTO (T10…T35): cada regra escolhe o modelo × flag SÓ com o histórico daquele minuto
                                    # (métricas "todas" = linhas daquele minuto; lucro/roi/entradas = apostas naquele minuto) e aposta SÓ nele.
                                    # No fim, o ENSEMBLE (ensemble_minutos.csv): em cada par, cada minuto usa a SUA escolha; FIRST = no jogo, a aposta do
                                    # 1º minuto (10→35) em que a escolhida daquele minuto entra; MULTI = as apostas de todos os minutos
FUNIS_MIN = [dict(nome=f"T{t}" + _SUF_EV(ev), politica="FIRST", tempos=[t], minutos_criterio=[t], minuto=t, ev_min=ev) for ev in LIMIARES_EV for t in _T_ANT] if FUNIS_POR_MINUTO else []
# v5.0 — CLV / MARKOUT (só AVALIAÇÃO e métricas do HISTÓRICO; o futuro de um jogo só é usado depois que o jogo acabou, como o lucro):
#   no lado apostado, q = probabilidade do mercado sem margem (de-vig proporcional, a mesma do motor) e odd = odd do lado.
#   markout h minutos depois (h em HORIZONTES_CLV): mk = q(t+h) − q(t) (> 0 = o mercado veio na direção da aposta = "acerto"); "ROI do markout" =
#   odd(t)·q(t+h) − 1 (EV da aposta medido pelo mercado de t+h); "green-up" = odd(t)/odd(t+h) − 1 (sair pela odd de t+h, sem o spread de lay).
#   FECHAMENTO = a ÚLTIMA odd disponível do jogo DEPOIS do minuto da aposta (minutos 10–35 e, se existirem, MINUTOS_SO_ODDS): CLV = odd(t)·q_fech − 1;
#   "supera o fechamento" = q_fech > q(t). Sem minuto depois (jogo acabou / falta dado) = sem CLV (contado e mostrado; viés de sobrevivência).
HORIZONTES_CLV = [5, 10, 15, 20, 25]
MINUTOS_SO_ODDS = [40, 45]          # zz40/zz45: se estiverem na pasta, só as ODDS deles entram no markout/fechamento (nada treina nem aposta neles)
# v5.0 — MODO SOMBRA ("cada mudança roda em sombra por N eventos; só vira produção se a CLV/Brier em sombra vencer a produção congelada"):
#   para cada regra, quando ela quer TROCAR de opção, a produção continua com a opção atual (congelada) e a nova roda em sombra a partir daquele par;
#   depois de N_SOMBRA jogos executáveis, compara (só jogos já terminados, < início do par) a nova × a produção pelo critério: "clv" = CLV médio das
#   apostas, "brier" = Brier skill (mercado − modelo) das previsões. Venceu → vira produção; não venceu → a sombra recomeça (se a regra ainda quiser trocar).
N_SOMBRA = 50
CRITERIOS_SOMBRA = ["clv", "brier"]
MIN_AMOSTRA_SOMBRA = 10             # mínimo de apostas (clv) / linhas (brier) de CADA lado na janela da sombra para decidir
SOMBRA_NOS_MINUTOS = False          # True = também nos funis por minuto (mais lento)
# v5.0 — ROI ESPERADO, VARIÂNCIA, ROLLING, ALARMES, GRÁFICOS
BINS_EV = [0.0, 0.02, 0.05, 0.10, 0.20, np.inf]   # faixas FIXAS de EV para a calibração (decididas antes; sem olhar resultado)
MIN_CALIB = 30                      # EV calibrado de uma aposta = ROI realizado pelas apostas ANTERIORES (jogos < início do par) da MESMA opção na mesma
                                    # faixa de EV; com menos de MIN_CALIB apostas na faixa, usa o EV do modelo (contado em pct_ev_cal_sem_historico)
COLUNA_DATA = None                  # coluna de DATA nos zz (None = procura "data"/"date"/"datetime"/"dia"); sem data, as janelas são em JOGOS:
JANELA_ROLL_CURTA, JANELA_ROLL_LONGA = 300, 900     # "30 dias" / "90 dias" em jogos executáveis (NÃO sei quantos jogos há por dia: ajuste aqui)
DIAS_ROLL_CURTA, DIAS_ROLL_LONGA = 30, 90           # usados só se houver coluna de data
TOP_GRAFICOS = 5                    # regras com maior PPG em cada (funil, janela, conjunto) nos gráficos (SVG) + BASE
B_BOOT_APOSTAS = 1000               # bootstrap por cluster dos IC de ROI e de CLV
METRICAS_TEMPOS = _T_ANT            # minutos das políticas no recorte entrou/não entrou das métricas de Brier/log loss
CORTE_TESTE = 7729                  # só descritivo: períodos separados neste jogo
BASELINE_MODELO = "V6_MOM"          # benchmark "modelo fixo": esta opção (sem flag) em TODOS os jogos, com a política do funil
BLOCO_RISCO = 150                   # tamanho do bloco (jogos avaliados consecutivos) para % de blocos positivos e pior bloco
USAR_BANCO = True
N_PROCESSOS = 0                     # 0 = automático; 1 = em série. Não muda resultado
GB_POR_PROCESSO = 2.5
VERIFICAR_LIMPO = True
B_BOOT = 5000                       # bootstrap por cluster de 10 gameids dos IC de PPG
B_MCS = 2000                        # réplicas do Model Confidence Set entre as regras
B_RC = 5000                         # réplicas do Reality Check (alguma regra bate a escolha ao acaso?)
ALPHA_MCS = 0.10
CONFERENCIA_N = 12                  # decisões sorteadas por (funil, janela) para conferir a conta rápida contra a conta direta
SABOTAGEM_N = 3                     # decisões sorteadas por (funil, janela) em que o futuro é trocado por ruído (nenhuma regra pode mudar a escolha)
PASTA_SAIDA = "OUT_FUNIL"
CFG_EXTRA = {}                      # vazio no uso normal (só para testes)
# ========================================================================
# v4.9 — CONTINUAÇÃO (RODAR_FUNIL_CONTINUACAO.bat põe FUNIL_CONTINUACAO=1): usa TAMBÉM os jogos que eram lockbox (>= 8448) como uma continuação
# normal: re-treino e re-escolha a cada 2 jogos até o último jogo dos zz. O lockbox deixa de ser uma confirmação independente.
CONTINUACAO = os.environ.get("FUNIL_CONTINUACAO") == "1"
CORTE_ANTIGO = LOCKBOX_SERIO                                         # onde começava o lockbox (vira só um corte de período / de bloco)
if CONTINUACAO:
    LOCKBOX_SERIO = 10 ** 7                                          # sem lockbox: todo jogo dos zz entra
    PASTA_SAIDA = PASTA_SAIDA + "_CONTINUACAO"

AQUI = os.path.dirname(os.path.abspath(__file__))
os.chdir(AQUI)
OUT = os.path.join(AQUI, PASTA_SAIDA)
FILHO = (int(sys.argv[sys.argv.index("--filho") + 1]), int(sys.argv[sys.argv.index("--filho") + 2])) if "--filho" in sys.argv else None
SUF_CAL = r"_(CALR|CAL)$"
FLAGS = FLAGS_SUJO + FLAGS_LIMPO
OPS = {">=": operator.ge, ">": operator.gt, "<=": operator.le, "<": operator.lt}


def log(*a): print(time.strftime("%H:%M:%S"), *a, flush=True)


# ---------------------------------------------------------------- avaliação: CÓPIA LITERAL do motor v9.0.1 ----------------------------------------------------------------
# v4.6 (02/10/2026): o PC da Amanda tem o motor v9.0, SEM a seção 10 da v9.0.1 (_alerta_epv_atual não existe lá → AttributeError).
# Para a AVALIAÇÃO não depender da versão do motor, as funções que o funil usava do motor fora da previsão (seções 5–8, _rng_de/_clu do
# topo e _alerta_epv_atual da seção 10) estão copiadas AQUI, letra por letra, do backtest_sujo.py v9.0.1. Elas leem o CFG do motor
# (CFG = BS.CFG, o MESMO dicionário, atribuído em main()). conferir_motor() mostra no log, função por função, se a do motor da pasta é
# idêntica a esta cópia (o funil usa SEMPRE a cópia). A PREVISÃO (carga, preparar, features, candidatos, walk-forward, banco) continua
# sendo a do motor da pasta.
CFG = None
WAIT_LOG = []
EXEC_POR_MINUTO = {}


def _rng_de(tag, *objs):
    """v7.2: gerador DERIVADO de (SEED, nome da análise, conteúdo da entrada). A mesma análise sobre os mesmos dados sorteia sempre
    as mesmas réplicas, em qualquer ordem e com qualquer número de outras análises (v7.1: fluxo sequencial dentro de cada componente)."""
    import zlib
    h = zlib.crc32(f"{CFG['SEED']}|{tag}".encode())
    for o in objs:
        if o is None: continue
        if hasattr(o, "columns"): h = zlib.crc32("|".join(map(str, o.columns)).encode(), h)
        a = np.asarray(o.values if hasattr(o, "values") else o)
        try: h = zlib.crc32(np.ascontiguousarray(a).tobytes(), h)
        except Exception: h = zlib.crc32(str(a.tolist()).encode(), h)
    return np.random.default_rng(np.random.SeedSequence([int(CFG["SEED"]), int(h)]))


def _clu(g):
    """Cluster de CLUSTER_GIDS gameids consecutivos: 1–10, 11–20, … (v7: (g−1)//K; antes g//K deixava o 1º cluster com 9)."""
    return (np.asarray(g, dtype=np.int64) - 1) // CFG["CLUSTER_GIDS"]


def linhas_jogo(base, cand):
    """1 linha por (t, gameid): scores médios dos 2 lados (simétrico p/ Brier e log)."""
    b = base[base["cand"] == cand]
    g = b.groupby(["t","gameid","bloco","cluster"], as_index=False).agg(
        ll_mod=("ll_mod","mean"), ll_mkt=("ll_mkt","mean"), bs_mod=("bs_mod","mean"), bs_mkt=("bs_mkt","mean"))
    g["llskill"] = g["ll_mkt"] - g["ll_mod"]; g["bsskill"] = g["bs_mkt"] - g["bs_mod"]
    return g


def _escolha(b, edge_min="cfg"):
    """Para cada (t, gameid): lado de maior edge e se passa a regra de entrada (edge_min=None => sem limiar)."""
    if edge_min == "cfg": edge_min = CFG["EDGE_MIN"]
    g_ = b.groupby(["t","gameid"])["side"]; b = b[(g_.transform("size") == 2) & (g_.transform("nunique") == 2)]      # v7.1: 2 lados DIFERENTES
    idx = b.groupby(["t","gameid"])[CFG.get("CRITERIO_LADO", "edge")].idxmax()
    e = b.loc[idx].copy()
    e["ok"] = (e["odd_t"] >= CFG["ODD_MIN"]) & (e["odd_t"] <= CFG["ODD_MAX"])
    if edge_min is not None: e["ok"] &= (e[CFG.get("LIMIAR_EM", "edge")] > edge_min)
    return e


def apostas(base, cand, politica="FIRST", X=None, tempos=None, permitido=None, edge_min="cfg", filtro_col=None, dec_modo="acao", registrar_wait=False):
    """Retorna as apostas (1 linha por aposta) e as linhas de decisão (1 por jogo-minuto avaliado).
    politica: FIRST (1ª entrada em tempos), MULTI (todas), WAIT (espera se odd < X).
    permitido(t, odd) -> bool: filtro extra (célula), avaliado com informação só do passado."""
    tempos = tempos or CFG["TEMPOS_DEPLOY"]
    b = base[(base["cand"] == cand) & (base["t"].isin(tempos))]
    e = _escolha(b, edge_min).sort_values(["gameid","t"])
    if permitido is not None:
        e["ok"] = e["ok"] & np.array([bool(permitido(t, o)) for t, o in zip(e["t"], e["odd_t"])], dtype=bool)
    if filtro_col is not None:
        e["ok"] = e["ok"] & (e[filtro_col] == 1)
    if politica == "MULTI":
        dec = e; bets = e[e["ok"]]
    elif politica == "FIRST":
        bets = e[e["ok"]].groupby("gameid").head(1)  # aposta = 1º tempo com entrada válida
        # v7.1: dec_modo="acao" (padrão) = o jogo-minuto em que a aposta ACONTECE → métricas 'geral' alinhadas ao lucro.
        #       dec_modo="universo" = 1º checkpoint de cada jogo, independe da política → base do teste PRIMÁRIO de skill.
        #       (v7 usava sempre o 1º checkpoint: 22–42% das apostas de produção aconteciam num minuto posterior)
        dec = bets if dec_modo == "acao" else e.groupby("gameid").head(1)
    elif politica == "WAIT":
        rows, decs = [], []
        for gid, grp in e.groupby("gameid", sort=False):
            grp = grp.sort_values("t"); decs.append(grp.iloc[0]); esperando = False; t_sinal = None; apostou = False
            for _, r in grp.iterrows():
                if not r["ok"]: continue
                if r["odd_t"] < X and not esperando: esperando = True; t_sinal = r["t"]; continue   # 1º sinal barato: espera
                if esperando and r["odd_t"] < X: continue                             # ainda barato: segue esperando
                rows.append(r); apostou = True; break
            if registrar_wait:                                                         # v7.2: registra também quem EXPIROU esperando
                WAIT_LOG.append(dict(cand=cand, X=X, gameid=gid, status="aposta" if apostou else ("expirou_esperando" if esperando else "sem_sinal"), t_primeiro_sinal=t_sinal))
        bets = pd.DataFrame(rows) if rows else e.iloc[0:0]; dec = bets if dec_modo == "acao" else pd.DataFrame(decs)
    else:
        raise ValueError(politica)
    bets = bets.copy()
    bets["lucro"] = np.where(bets["y"] == 1, (bets["odd_t"] - 1) * CFG["STAKE"], -CFG["STAKE"])
    return bets, dec


def _maxdd(lucros):
    c = np.cumsum(lucros); pico = np.maximum.accumulate(np.concatenate([[0], c]))[1:]
    return float(np.max(pico - c)) if len(c) else 0.0


def _seq_ruim(lucros):
    best = cur = 0
    for l in lucros:
        cur = cur + 1 if l < 0 else 0; best = max(best, cur)
    return int(best)


def _cluster_sums(x, clusters):
    """Somas e tamanhos por cluster (x pode ser 1-D ou 2-D: obs × colunas)."""
    codes, uniq = pd.factorize(np.asarray(clusters)); n = len(uniq)
    x = np.asarray(x, float); tam = np.bincount(codes, minlength=n).astype(float)
    if x.ndim == 1: soma = np.bincount(codes, weights=x, minlength=n)
    else: soma = np.vstack([np.bincount(codes, weights=x[:, j], minlength=n) for j in range(x.shape[1])]).T
    return soma, tam, n


def boot_media(x, clusters, B=None, rng=None):
    """IC por bootstrap de clusters (média das observações). Vetorizado: sorteio multinomial de clusters."""
    rng = rng or _rng_de("boot", x, clusters); B = B or CFG["B_BOOT"]
    x = np.asarray(x, float); soma, tam, n = _cluster_sums(x, clusters)
    cnt = rng.multinomial(n, np.full(n, 1.0 / n), size=B).astype(float)          # B × n clusters
    out = (cnt @ soma) / (cnt @ tam)
    m = x.mean(); se = out.std(ddof=1)
    # p_gt0 é PROBABILIDADE BOOTSTRAP de efeito positivo (percentil), não um p-valor frequentista.
    # p_valor é o unilateral com bootstrap centrado na nula, com a correção (b+1)/(B+1) (nunca 0).
    p_nulo = float((np.sum((out - m) >= m) + 1) / (B + 1))
    p_neg = float((np.sum((out - m) <= m) + 1) / (B + 1))                         # v8.4: H1 média < 0 (mesma distribuição centrada)
    return dict(media=m, se=se, t=(m/se if se>0 else np.nan), ic_lo=np.quantile(out,0.025), ic_hi=np.quantile(out,0.975),
                p_gt0=(out>0).mean(), p_valor=p_nulo, p_valor_neg=p_neg,
                lb95_basico=2 * m - np.quantile(out, 0.95), ub95_basico=2 * m - np.quantile(out, 0.05))   # v8.4: duais do p centrado (p<0,05 ⇔ lb95>0)


def mcs(perdas, clusters, alpha=None, B=None, rng=None):
    """Model Confidence Set (Hansen-Lunde-Nason 2011, estatística T_max, bootstrap por cluster, vetorizado).
    perdas: DataFrame (obs × modelos) — MENOR = MELHOR (use −skill). Retorna sobreviventes e p-valores de eliminação."""
    rng = rng or _rng_de("mcs", perdas, clusters); alpha = alpha or CFG["ALPHA_MCS"]; B = B or CFG.get("B_MCS", CFG["B_BOOT"])
    M = list(perdas.columns); L_all = perdas.values.astype(float)
    soma_all, tam, n = _cluster_sums(L_all, clusters)                              # n clusters × m
    cnt = rng.multinomial(n, np.full(n, 1.0 / n), size=B).astype(float); N_b = cnt @ tam   # B × 1
    elim = []; p_acum = 0.0
    while len(M) > 1:
        cols = [perdas.columns.get_loc(c) for c in M]
        L = L_all[:, cols]; d = L - L.mean(axis=1, keepdims=True); dbar = d.mean(axis=0)
        soma = soma_all[:, cols]; soma_d = soma - soma.mean(axis=1, keepdims=True)     # somas por cluster de d_ij
        boot = (cnt @ soma_d) / N_b[:, None]                                          # B × |M|
        se = boot.std(axis=0, ddof=1); se[se == 0] = np.nan
        tstat = dbar / se
        if np.all(np.isnan(tstat)): break                                  # guarda: candidatos idênticos
        T = np.nanmax(tstat)
        Tb = np.nanmax((boot - dbar) / se, axis=1)
        p_etapa = float((np.sum(Tb >= T) + 1) / (B + 1))                   # (b+1)/(B+1): p nunca é 0
        p_acum = max(p_acum, p_etapa)                                      # HLN: p do MCS é o MÁXIMO acumulado
        pior = M[int(np.nanargmax(tstat))]
        if p_acum < alpha:
            elim.append(dict(eliminado=pior, p=p_acum, p_etapa=p_etapa, t=float(T))); M.remove(pior)
        else:
            elim.append(dict(eliminado=None, p=p_acum, p_etapa=p_etapa, t=float(T)))
            break
    return M, pd.DataFrame(elim)


def nula_empirica_max(M_series, clusters, B=None, rng=None, escopo=""):
    """Bootstrap empírico e studentizado do MÁXIMO, clusterizado (estilo White 2000 Reality Check / Hansen 2005 SPA).
    M_series: obs × séries (skill por jogo, ou lucro por jogo). H0: nenhuma série tem média > 0.
    p_reality_check: recentra TODAS as séries (configuração menos favorável — conservador com muitas séries ruins).
    p_SPA_c: não recentra as séries claramente piores (t ≤ −√(2 ln ln n)) — menos conservador, mais poder.
    p_max_mean: mesma lógica sem studentizar. ATENÇÃO: o p é do MÁXIMO — responde 'existe alguma série > 0 no escopo?',
    não certifica um candidato (para isso: MCS e StepM). O escopo vai escrito no resultado."""
    rng = rng or _rng_de("rc", M_series, clusters); B = int(B or CFG.get("B_RC", 50000))
    X = M_series.values.astype(float); n_obs = X.shape[0]; soma, tam, n = _cluster_sums(X, clusters)
    meds = []
    for b0 in range(0, B, 5000):
        bb = min(5000, B - b0); cnt = rng.multinomial(n, np.full(n, 1.0 / n), size=bb).astype(float)
        meds.append((cnt @ soma) / (cnt @ tam)[:, None])
    med = np.vstack(meds); obs = X.mean(axis=0); se = med.std(axis=0, ddof=1); se[se == 0] = np.nan
    t_obs = obs / se
    if np.all(np.isnan(t_obs)):
        return dict(escopo=escopo, k_series=X.shape[1], erro="todas as séries constantes"), pd.Series(t_obs, index=M_series.columns)
    maxt = np.nanmax((med - obs) / se, axis=1); t_max = float(np.nanmax(t_obs)); i = int(np.nanargmax(t_obs))
    thr = -np.sqrt(2 * np.log(np.log(max(n_obs, 16)))); mu_c = np.where(t_obs <= thr, obs, 0.0)
    maxt_spa = np.maximum(np.nanmax((med - obs + mu_c) / se, axis=1), 0.0); T_spa = max(t_max, 0.0)
    maxm = np.nanmax(med - obs, axis=1); m_max = float(np.nanmax(obs)); exc = int(np.sum(maxt >= t_max))
    return dict(escopo=escopo, k_series=X.shape[1], n_obs=n_obs, n_clusters=n, B=B,
                barra_p95=float(np.quantile(maxt, 0.95)), barra_p99=float(np.quantile(maxt, 0.99)),
                melhor=M_series.columns[i], t_melhor=t_max, n_excedencias=exc, p_min=1.0 / (B + 1),
                p_reality_check=float((exc + 1) / (B + 1)),
                p_SPA_c_aprox=float((np.sum(maxt_spa >= T_spa) + 1) / (B + 1)), limiar_SPA=float(thr), n_series_nao_recentradas=int(np.sum(t_obs <= thr)),
                p_max_mean=float((np.sum(maxm >= m_max) + 1) / (B + 1)),
                dispersao_t=float(np.nanstd(t_obs, ddof=1))), pd.Series(t_obs, index=M_series.columns)


def universo_executavel(frames):
    """v8.3 (lógica idêntica ao bloco v7.2/v7.5/v7.6, só virou função): jogos com os 2 lados e odd válida no MESMO minuto de decisão
    e algum lado apostável pela política (ODD_MIN–ODD_MAX). Usa só lados e odds — nunca o resultado. Devolve (executáveis, observáveis).
    v8.6: guarda também o conjunto executável POR MINUTO em EXEC_POR_MINUTO (mesma regra de odds do PPG)."""
    uv, uobs = set(), set(); EXEC_POR_MINUTO.clear()
    for t_ in CFG["TEMPOS_DEPLOY"]:
        f_ = frames[t_]; dois_ = f_.groupby("gameid")["side"].transform("nunique").eq(2)
        ambos_ = (np.isfinite(f_["odd_t"]) & (f_["odd_t"] > 1)).groupby(f_["gameid"]).transform("all")
        ambos_ &= f_["odd_t"].between(CFG["ODD_MIN"], CFG["ODD_MAX"]).groupby(f_["gameid"]).transform("any")
        uobs |= set(f_.loc[dois_, "gameid"].unique()); ex_t = set(f_.loc[dois_ & ambos_, "gameid"].unique()); uv |= ex_t; EXEC_POR_MINUTO[t_] = ex_t
    return np.array(sorted(uv)), uobs


def _alerta_epv_atual(fits):
    """alerta_EPV recalculado com o EPV_ALERTA DESTA rodada (os treinos reaproveitados do banco traziam o da rodada que os gravou).
    Mesma conta do walk_forward: EPV = jogos de treino ÷ (nº de features + 1), sem arredondar."""
    if fits is None or not len(fits) or not {"alerta_EPV", "EPV", "n_treino_jogos", "n_feats"} <= set(fits.columns): return fits
    m = fits["EPV"].notna() & fits["n_treino_jogos"].notna() & fits["n_feats"].notna()
    if m.any():
        epv = fits.loc[m, "n_treino_jogos"].astype(float) / (fits.loc[m, "n_feats"].astype(float) + 1.0).clip(lower=1.0)
        fits.loc[m, "alerta_EPV"] = [bool(x) for x in (epv < float(CFG["EPV_ALERTA"])).values]
    return fits



COPIAS_MOTOR = ["_rng_de", "_clu", "linhas_jogo", "_escolha", "apostas", "_maxdd", "_seq_ruim", "_cluster_sums", "boot_media", "mcs", "nula_empirica_max",
                "universo_executavel", "_alerta_epv_atual"]
# o que o validar_funil usa do motor da pasta (PREVISÃO e constantes) e as chaves do CFG que ele LÊ: falta qualquer um → para logo no início
ATRIB_MOTOR = ["CFG", "carregar_bruto", "preparar", "features", "candidatos", "blocos_teste", "walk_forward_dev", "hash_cadeia", "_pastas", "_reset_rngs",
               "_iguais", "FAM_DEF", "ATOM_BASE", "ATOM_RECON", "ORIGEM_ATOM", "ORIGEM_DADOS", "GOLD5", "IMP5", "COLS_WRPROBS", "COLS_LOLDRAFT", "CHECKPOINTS"]
CHAVES_CFG = ["G_CLEAN", "FAMILIAS", "CAL_PARA", "INICIO_TESTE", "INICIO_TESTE_SUJO", "MIN_TREINO_JOGOS", "TEMPOS", "PASTA_DADOS", "PASTA_PREVISOES",
              "PREFIXO_GRANDE", "PREFIXO_LIMPO", "SEED", "SELECAO_MIN_HIST_JOGOS", "EPV_ALERTA", "CLUSTER_GIDS", "EDGE_MIN", "ODD_MIN", "ODD_MAX", "STAKE"]
# SHA-256 do trecho de PREVISÃO (topo do motor até "# 5. BASE LONGA", sem o CFG; mesma conta do próprio motor), igual na v9.0 e na v9.0.1
# segundo o docs/MUDANCAS_v9.0.1.md do pacote (conferido: é o valor do backtest_sujo.py/backtest_limpo.py v9.0.1 com que este script foi testado)
SHA_PREVISAO_V90 = {"sujo": "3693c7c9a42e26216d114899f0f651c4ae1c40616c1030948b15fd5b74b98ed5", "limpo": "9cfd046f92425e34a3d7e9f9cf8ff95f1df38ed8b21542409a069544a73fe864"}


def _sha_previsao(arq):
    """mesma conta do motor (_sha_codigo_previsao da v9.0 / _sha_previsao_de da v9.0.1): SHA-256 do topo até "# 5. BASE LONGA", sem o CFG"""
    import ast
    src = open(arq, encoding="utf-8").read(); linhas = src.splitlines(keepends=True)
    for n in ast.parse(src).body:
        if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", None) == "CFG":
            linhas = linhas[:n.lineno - 1] + linhas[n.end_lineno:]; break
    txt = "".join(linhas); i = txt.find("\n# 5. BASE LONGA")
    return hashlib.sha256(txt[:i].encode("utf-8")).hexdigest() if i > 0 else None


def conferir_motor(B, tag, arq):
    """v4.6: o motor da pasta tem tudo o que o validar_funil usa? Qual versão é? O trecho de previsão é o da v9.0/v9.0.1? As funções da
    avaliação do motor são iguais às cópias deste script? Para no início (antes de qualquer treino) se faltar algo."""
    import inspect
    falta = [a for a in ATRIB_MOTOR if not hasattr(B, a)]
    falta_cfg = [k for k in CHAVES_CFG if k not in (getattr(B, "CFG", None) or {})]
    if falta or falta_cfg:
        raise SystemExit(f"motor {arq}: esta versão não tem o que o validar_funil precisa — faltam {falta + ['CFG[' + k + ']' for k in falta_cfg]}. Nada foi treinado.")
    versao = "v9.0.1" if hasattr(B, "_alerta_epv_atual") else "SEM a seção 10 da v9.0.1 (v9.0 ou anterior)"
    try: sha = _sha_previsao(os.path.join(AQUI, arq))
    except Exception as e: sha = None; log(f"motor {tag}: não consegui calcular o SHA do trecho de previsão ({type(e).__name__}: {e})")
    ref = SHA_PREVISAO_V90.get(tag); igual = bool(sha) and sha == ref
    ig, dif, aus = [], [], []
    for nm in COPIAS_MOTOR:
        if not hasattr(B, nm): aus.append(nm); continue
        try: (ig if inspect.getsource(getattr(B, nm)) == inspect.getsource(globals()[nm]) else dif).append(nm)
        except (OSError, TypeError): dif.append(nm + " (sem fonte)")
    s10 = ""                                                                                    # v5.0 (achado da verificação): na v9.0.1 a seção 10 redefine o banco/cadeia
    if hasattr(B, "_SHA_S10_REF"):
        s10_ok = getattr(B, "_SHA_S10_IMPORT", None) == B._SHA_S10_REF
        s10 = " · seção 10 (banco/cadeia da v9.0.1) " + ("= a de referência" if s10_ok else "≠ a de referência (ATENÇÃO: o banco/walk-forward executados não são os testados)")
    log(f"motor {tag} ({arq}): versão {versao}{s10} · trecho de PREVISÃO SHA-256 {(sha or 'n/d')[:8]}…{(sha or 'n/d')[-4:]} "
        + ("= o da v9.0/v9.0.1 (o mesmo código com que este script foi testado)" if igual else
           f"≠ o da v9.0/v9.0.1 ({ref[:8]}…{ref[-4:]}): ATENÇÃO — o funil usa as previsões DESTE motor (o da etapa 1 desta pasta), mas este código não foi testado com ele")
        + f" · avaliação: o funil usa as {len(COPIAS_MOTOR)} funções copiadas da v9.0.1 (no motor: {len(ig)} idênticas"
        + (f", DIFERENTES {dif}" if dif else "") + (f", ausentes {aus}" if aus else "") + ")")
    return dict(versao=versao, sha_previsao=sha, previsao_igual_v90_v901=igual, copias_identicas=ig, copias_diferentes=dif, ausentes_no_motor=aus)


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
    if INICIO_PREVISOES is not None:
        assert int(INICIO_PREVISOES) % PASSO == 0 and int(INICIO_PREVISOES) < int(B.CFG["G_CLEAN"]), "INICIO_PREVISOES tem de ser múltiplo de PASSO e antes de G_CLEAN"
        B.CFG["INICIO_TESTE_SUJO"] = int(INICIO_PREVISOES)
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
        B = carregar_motor(arq, tag); info = conferir_motor(B, tag, arq); K = configurar(B, tag); K["motor_info"] = info; M[tag] = (B, K)
    return M


# ---------------------------------------------------------------- dados ----------------------------------------------------------------
def carregar_brutos(B, K, conferir=True):
    brutos = {}
    for t in K["todos"]:
        fp = os.path.join(B.CFG["PASTA_DADOS"], f"{B.CFG['PREFIXO_GRANDE']}{t}.xlsx")
        if not os.path.exists(fp): raise FileNotFoundError(f"falta {fp} — copie o zz que falta (mesmo banco dos outros zz, até o jogo {LOCKBOX_SERIO - 1}) para esta pasta")
        G_ = B.carregar_bruto(t); n0 = len(G_); gid_ = pd.to_numeric(G_["gameid"], errors="coerce")
        brutos[t] = G_[(gid_ < K["L0"]).values].reset_index(drop=True).infer_objects()           # AUDITORIA: linhas do lockbox (gid >= L0) saem AQUI e nada mais as vê
        if CONTINUACAO:                                                                         # v4.9: jogo ainda sem resultado não treina nem é avaliado
            g_ = pd.to_numeric(brutos[t]["gameid"], errors="coerce"); sem = set(g_[pd.to_numeric(brutos[t]["resultado"], errors="coerce").isna()].dropna().astype(int))
            if sem:
                brutos[t] = brutos[t][~g_.isin(sem).values].reset_index(drop=True)
                log(f"t{t}: {len(sem)} jogo(s) sem resultado descartados (ex.: {sorted(sem)[-5:]})")
            log(f"t{t}: CONTINUAÇÃO — {int((pd.to_numeric(brutos[t]['gameid']) >= CORTE_ANTIGO).sum())} linhas de jogos >= {CORTE_ANTIGO} (o antigo lockbox) · último jogo {int(pd.to_numeric(brutos[t]['gameid']).max())}")
        assert int(pd.to_numeric(brutos[t]["gameid"]).max()) < K["L0"], "linha do lockbox na memória"
        log(f"t{t}: {B.ORIGEM_DADOS.get(t)} · {len(brutos[t])} linhas de desenvolvimento (gid < {K['L0']}); {n0 - len(brutos[t])} linhas do lockbox (ou sem gameid) descartadas na leitura")
    if conferir:
        conferir_colunas_brutas(B, K, brutos); conferir_tempos(B, brutos)
        if VERIFICAR_LIMPO:
            for t in K["todos"]:
                if os.path.exists(os.path.join(B.CFG["PASTA_DADOS"], f"{B.CFG['PREFIXO_LIMPO']}{t}.xlsx")): log(f"t{t}: limpo == grande (só gid < {K['L0']}): {conferir_limpo(B, t, brutos[t], K['L0'])}")
                else: log(f"t{t}: sem {B.CFG['PREFIXO_LIMPO']}{t}.xlsx — verificação limpo == grande PULADA neste minuto (registrado)")
    return brutos


def conferir_limpo(B, t, G, L0):
    """zzl{t} (+ G_CLEAN − 1 no gameid) == zz{t} em todas as colunas comuns, SÓ nas linhas com gid < L0 (as do lockbox são descartadas ao ler)"""
    L = pd.read_excel(os.path.join(B.CFG["PASTA_DADOS"], f"{B.CFG['PREFIXO_LIMPO']}{t}.xlsx")); L = L.loc[:, [c for c in L.columns if not str(c).startswith("Unnamed")]].copy()
    L["gameid"] = pd.to_numeric(L["gameid"], errors="coerce") + (int(B.CFG["G_CLEAN"]) - 1); L = L[L["gameid"] < L0]
    comuns = [c for c in L.columns if c in G.columns and c != "ordem"]
    m = L[comuns].merge(G[comuns], on=["gameid", "side"], suffixes=("_L", "_G"), validate="one_to_one")
    assert len(m) == len(L), f"t{t}: {len(L) - len(m)} linhas do limpo (gid < {L0}) não casam no grande"
    ruins = [c for c in comuns if c not in ("gameid", "side") and not B._iguais(m[c + "_L"], m[c + "_G"])]
    assert not ruins, f"t{t}: colunas diferentes entre limpo e grande: {ruins}"
    return dict(t=t, linhas_limpo=len(L), colunas_verificadas=len(comuns) - 2)


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
        linhas.append(f"t{t}: {len(m)} linhas em comum com o zz{ref} ({len(cc)} colunas iguais)" + ("" if t == ref else f" · só no zz{t}: {so_t} · só no zz{ref}: {so_r}"))
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
    bl = B.blocos_teste(gid_max)
    if CONTINUACAO:                                                                             # v4.9: mantém o corte de bloco no antigo lockbox (mesmos blocos de antes)
        bl = [(i, a, b) for i, (a, b) in enumerate([x for (_, a, b) in bl for x in ([(a, CORTE_ANTIGO), (CORTE_ANTIGO, b)] if a < CORTE_ANTIGO < b else [(a, b)])])]
    bp = [b for b in bl if b[1] < K["L0"] and b[2] <= K["L0"] and b[1] >= ini]
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
    base["cluster"] = _clu(base["gameid"]).astype(int)
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




# ---------------------------------------------------------------- funil: somas por jogo ----------------------------------------------------------------
def dados_funil(G, GMf, apostas_de, opcoes, fu):
    """por opção, somas POR JOGO (índice = posição em G): T = linhas 'todas' [n, ll_mod, ll_mkt, bs_mod, bs_mkt, (bs_mkt − bs_mod)²] nos
    minutos do critério e no universo do flag; E = [n, ll_mod, ll_mkt, bs_mod, bs_mkt] no lado apostado; Lg/L2/N2 = lucro do jogo,
    Σ lucro² e Σ min(lucro, 0)² por aposta; e as apostas em ordem (gameid, minuto)."""
    nG, O = len(G), len(opcoes); mins = list(fu["minutos_criterio"])
    T = np.zeros((O, nG, 6)); E = np.zeros((O, nG, 5)); Lg = np.zeros((O, nG)); L2 = np.zeros((O, nG)); N2 = np.zeros((O, nG)); BETS = []; COV = np.zeros((O, nG))
    CS = np.zeros((O, nG)); CN = np.zeros((O, nG)); CB = np.zeros((O, nG))                      # v5.0: CLV por jogo (soma de odd·q_fech − 1, nº com fechamento, soma de q_fech − q)
    for j, op in enumerate(opcoes):
        r = GMf[op["cand"]]; r = r[r["t"].isin(mins) & (r["gameid"] >= op["w"])]
        COV[j] = np.bincount(np.searchsorted(G, r["gameid"].values), minlength=nG) > 0                 # o modelo previu algum minuto deste jogo
        if op["flag"]: r = r[r[op["flag"]]]
        ix = np.searchsorted(G, r["gameid"].values); assert np.all(G[np.minimum(ix, nG - 1)] == r["gameid"].values), "linha fora do universo"
        for k, v in enumerate([None, r["ll_mod"].values, r["ll_mkt"].values, r["bs_mod"].values, r["bs_mkt"].values, (r["bs_mkt"].values - r["bs_mod"].values) ** 2]):
            T[j, :, k] = np.bincount(ix, weights=v, minlength=nG)
        b = apostas_de(op["cand"], op["flag"], fu["politica"], tuple(fu["tempos"]), fu.get("ev_min")); b = b[b["gameid"] >= op["w"]].sort_values(["gameid", "t"], kind="mergesort")
        g_, L, p, y, q, t_, sd_, od_, ed_ = (b[c].values for c in ("gameid", "lucro", "p", "y", "q_devig_t", "t", "side", "odd_t", "edge"))
        ib = np.searchsorted(G, g_); assert np.all(G[np.minimum(ib, nG - 1)] == g_), "aposta fora do universo executável"
        llm = -(y * np.log(np.clip(p, 1e-6, 1)) + (1 - y) * np.log(np.clip(1 - p, 1e-6, 1))); llk = -(y * np.log(q) + (1 - y) * np.log(1 - q))
        for k, v in enumerate([None, llm, llk, (p - y) ** 2, (q - y) ** 2]): E[j, :, k] = np.bincount(ib, weights=v, minlength=nG)
        Lg[j] = np.bincount(ib, weights=L, minlength=nG); L2[j] = np.bincount(ib, weights=L ** 2, minlength=nG); N2[j] = np.bincount(ib, weights=np.minimum(L, 0) ** 2, minlength=nG)
        cv = clv_campos(g_, sd_, t_, od_, q); tem = np.isfinite(cv["clv"])                     # v5.0: CLV (fechamento) de cada aposta
        CS[j] = np.bincount(ib[tem], weights=cv["clv"][tem], minlength=nG); CN[j] = np.bincount(ib[tem], minlength=nG); CB[j] = np.bincount(ib[tem], weights=cv["mk_fech"][tem], minlength=nG)
        BETS.append(dict(g=g_.astype(np.int64), L=L.astype(float), p=p, y=y, q=q, t=t_, side=sd_, odd=od_, edge=ed_, clv=cv["clv"], supera=cv["supera"], mkf=cv["mk_fech"]))
    return dict(G=G, T=T, E=E, Lg=Lg, L2=L2, N2=N2, bets=BETS, COV=COV, CS=CS, CN=CN, CB=CB)


# ---------------------------------------------------------------- funil: métricas do histórico ----------------------------------------------------------------
def _metricas(t, e, SL, SL2, SN2, nu, mdd, sq, SG2, SC=0.0, NC=0.0, BC=0.0):
    """MÉTRICAS a partir das somas de uma janela (mesmas definições do scorers() do motor; 'todas' no lugar de 'geral')"""
    nr, nb = t[..., 0], e[..., 0]
    with np.errstate(invalid="ignore", divide="ignore"):
        okr, okb = nr >= MIN_LINHAS, nb >= MIN_LINHAS
        ppg = np.where(nu > 0, SL / np.maximum(nu, 1), np.nan)
        V = dict(ll_todas=np.where(okr, t[..., 1] / nr, np.nan), brier_todas=np.where(okr, t[..., 3] / nr, np.nan),
                 llskill_todas=np.where(okr, (t[..., 2] - t[..., 1]) / nr, np.nan), bsskill_todas=np.where(okr, (t[..., 4] - t[..., 3]) / nr, np.nan),
                 ll_entradas=np.where(okb, e[..., 1] / nb, np.nan), brier_entradas=np.where(okb, e[..., 3] / nb, np.nan),
                 llskill_entradas=np.where(okb, (e[..., 2] - e[..., 1]) / nb, np.nan), bsskill_entradas=np.where(okb, (e[..., 4] - e[..., 3]) / nb, np.nan),
                 lucro=np.asarray(SL, float) + 0.0, ppg=ppg, n_apostas=np.asarray(nb, float) + 0.0, n_linhas=np.asarray(nr, float) + 0.0)
        roi = np.where(okb, SL / nb, np.nan); sd = np.sqrt(np.clip((SL2 - SL ** 2 / np.maximum(nb, 1)) / np.maximum(nb - 1, 1), 0, None)); down = np.sqrt(SN2 / np.maximum(nb, 1))
        V.update(roi=roi, sharpe=np.where(okb & (sd > 0), roi / sd, np.nan), sortino=np.where(okb & (down > 0), roi / down, np.nan),
                 maxdd=np.where(okb, mdd, np.nan), seq_ruim=np.where(okb, sq, np.nan), calmar=np.where(okb & (mdd > 0), SL / np.where(mdd > 0, mdd, 1), np.nan))
        sdg = np.sqrt(np.clip((SG2 - nu * np.nan_to_num(ppg) ** 2) / np.maximum(nu - 1, 1), 0, None))
        V["vol_ppg"] = np.where(nu > 1, sdg, np.nan); V["t_ppg"] = np.where((nu > 1) & (sdg > 0), ppg / (sdg / np.sqrt(np.maximum(nu, 1))), np.nan)
        mu = (t[..., 4] - t[..., 3]) / np.maximum(nr, 1); sdr = np.sqrt(np.clip((t[..., 5] - nr * mu ** 2) / np.maximum(nr - 1, 1), 0, None))
        V["t_bsskill_todas"] = np.where(okr & (sdr > 0), mu / (sdr / np.sqrt(np.maximum(nr, 1))), np.nan)
        okc = np.asarray(NC) >= MIN_LINHAS                                                      # v5.0: CLV das apostas da janela (com fechamento)
        V["clv_medio"] = np.where(okc, SC / np.maximum(NC, 1), np.nan); V["clv_mk"] = np.where(okc, BC / np.maximum(NC, 1), np.nan); V["n_clv"] = np.asarray(NC, float) + 0.0
    return V


def _caminho(bg, L, w, a_arr):
    """maxdd e pior sequência de apostas perdidas desde w, avaliados antes de cada a (apostas em ordem jogo → minuto)"""
    sel = bg >= w; Lw, gw = L[sel], bg[sel]; mdd = np.zeros(len(a_arr)); sq = np.zeros(len(a_arr))
    if len(Lw):
        c = np.cumsum(Lw); dd = np.maximum.accumulate(np.maximum.accumulate(np.r_[0.0, c])[1:] - c)
        neg = Lw < 0; ii = np.arange(len(Lw)); st = np.maximum.accumulate(np.where(neg, ii - np.maximum.accumulate(np.where(~neg, ii, -1)), 0))
        k = np.searchsorted(gw, a_arr, "left"); kk = np.maximum(k - 1, 0); mdd = np.where(k > 0, dd[kk], 0.0); sq = np.where(k > 0, st[kk], 0)
    return mdd, sq


def valores_rapido(D, j, a_dec, w_dec):
    """MÉTRICAS da opção j em cada decisão (janela [w, a) em gameid): somas acumuladas + caminho por início de janela + janela recente"""
    G = D["G"]; hi = np.searchsorted(G, a_dec, "left"); lo = np.minimum(np.searchsorted(G, w_dec, "left"), hi); lr = np.maximum(lo, hi - JANELA_RECENTE)
    cs = lambda x: np.concatenate([np.zeros((1,) + x.shape[1:]), np.cumsum(x, axis=0)])
    CT, CE, C1, C2, C3, CG = cs(D["T"][j]), cs(D["E"][j]), cs(D["Lg"][j]), cs(D["L2"][j]), cs(D["N2"][j]), cs(D["Lg"][j] ** 2)
    mdd, sq = np.zeros(len(a_dec)), np.zeros(len(a_dec))
    for w in np.unique(w_dec):
        m = w_dec == w; mdd[m], sq[m] = _caminho(D["bets"][j]["g"], D["bets"][j]["L"], w, a_dec[m])
    K1, K2, K3 = cs(D["CS"][j]), cs(D["CN"][j]), cs(D["CB"][j])
    V = _metricas(CT[hi] - CT[lo], CE[hi] - CE[lo], C1[hi] - C1[lo], C2[hi] - C2[lo], C3[hi] - C3[lo], hi - lo, mdd, sq, CG[hi] - CG[lo], K1[hi] - K1[lo], K2[hi] - K2[lo], K3[hi] - K3[lo])
    R = _metricas(CT[hi] - CT[lr], CE[hi] - CE[lr], C1[hi] - C1[lr], C2[hi] - C2[lr], C3[hi] - C3[lr], hi - lr, mdd * 0, sq * 0, CG[hi] - CG[lr])
    V["ppg_recente"], V["bsskill_recente"] = R["ppg"], R["bsskill_todas"]; V["queda_ppg"] = V["ppg_recente"] - V["ppg"]
    CV = cs(D["COV"][j]); nu = hi - lo
    with np.errstate(invalid="ignore", divide="ignore"): V["cobertura_modelo"] = np.where(nu > 0, (CV[hi] - CV[lo]) / np.maximum(nu, 1), np.nan)
    return V


def valores_direto(B, D, j, a, w):
    """as mesmas MÉTRICAS pela conta DIRETA (máscara de jogos e apostas na janela; maxdd/seq_ruim do próprio motor) — confere a rápida"""
    G = D["G"]; bt = D["bets"][j]
    def janela(g0):
        m = (G >= g0) & (G < a); mb = (bt["g"] >= g0) & (bt["g"] < a); p, y, q, L = bt["p"][mb], bt["y"][mb], bt["q"][mb], bt["L"][mb]
        e = np.array([len(L), np.sum(-(y * np.log(np.clip(p, 1e-6, 1)) + (1 - y) * np.log(np.clip(1 - p, 1e-6, 1)))), np.sum(-(y * np.log(q) + (1 - y) * np.log(1 - q))),
                      np.sum((p - y) ** 2), np.sum((q - y) ** 2)])
        lg = pd.Series(L).groupby(bt["g"][mb]).sum()                                                    # lucro por jogo, das apostas cruas
        cl_, su_ = bt["clv"][mb], bt["mkf"][mb]; tc = np.isfinite(cl_)                         # v5.0: CLV direto das apostas cruas
        V = _metricas(D["T"][j][m].sum(0), e, L.sum(), (L ** 2).sum(), (np.minimum(L, 0) ** 2).sum(), int(m.sum()),
                      _maxdd(L) if len(L) else 0.0, _seq_ruim(L) if len(L) else 0, float((lg ** 2).sum()), float(cl_[tc].sum()), float(tc.sum()), float(su_[tc].sum()))
        if len(L) >= MIN_LINHAS: V["sharpe"] = (L.mean() / L.std(ddof=1)) if L.std(ddof=1) > 0 else np.nan   # motor: roi / sd (ddof=1)
        return V, m
    V, m = janela(w)
    gj = G[m]; R, _ = janela(int(gj[-JANELA_RECENTE]) if len(gj) > JANELA_RECENTE else w)
    V["ppg_recente"], V["bsskill_recente"] = R["ppg"], R["bsskill_todas"]; V["queda_ppg"] = V["ppg_recente"] - V["ppg"]
    V["cobertura_modelo"] = float(D["COV"][j][m].sum() / m.sum()) if m.sum() > 0 else np.nan
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


def matriz_metricas(D, opcoes, a_dec, Wd):
    M = {s: np.full((len(a_dec), len(opcoes)), np.nan) for s in METRICAS}
    for j in range(len(opcoes)):
        vj = valores_rapido(D, j, a_dec, Wd[:, j])
        for s in METRICAS: M[s][:, j] = vj[s]
    return M


# ---------------------------------------------------------------- funil: regras ----------------------------------------------------------------
def nome_regra(rg): return rg if isinstance(rg, str) else rg["nome"]


def _melhor(X, ok, sentido):
    """melhor valor (no SENTIDO da métrica) entre as opções ok; empate (|dif| <= 1e-12) → 1ª em ordem alfabética; −1 = nenhuma"""
    Y = np.where(ok, X * sentido, -np.inf); best = Y.max(axis=1); tem = ok.any(axis=1)
    return np.where(tem, np.argmax(Y >= (best[:, None] - 1e-12), axis=1), -1)


def aplicar_regra(rg, M, El):
    """índice da opção escolhida em cada decisão: >= 0 opção · −1 sem decisão (nenhuma elegível com valor) · −2 a regra decidiu não apostar"""
    if isinstance(rg, str): rg = dict(nome=rg, ordenar=rg)
    if "funcao" in rg:
        idx = np.asarray(rg["funcao"](M, El), dtype=np.int64)
        assert idx.shape == (El.shape[0],) and np.all((idx >= -2) & (idx < El.shape[1])), f"regra {rg['nome']}: índice inválido"
        assert np.all(El[np.arange(len(idx))[idx >= 0], idx[idx >= 0]]), f"regra {rg['nome']}: escolheu opção NÃO elegível"
        return idx
    X, s = M[rg["ordenar"]], SENTIDO[rg["ordenar"]]; base_ok = El & np.isfinite(X); filtros = list(rg.get("filtros", []))
    for (met, op, _) in filtros: assert met in M and op in OPS, f"regra {rg['nome']}: filtro inválido ({met} {op})"
    def passa(fs):
        m = base_ok.copy()
        for (met, op, v) in fs: Y = M[met]; m &= np.isfinite(Y) & OPS[op](np.nan_to_num(Y, nan=-np.inf if op in (">=", ">") else np.inf), v)
        return m
    idx = _melhor(X, passa(filtros), s); falta = (idx < 0) & base_ok.any(axis=1); modo = rg.get("sem_candidato", "melhor_sem_filtro")
    if modo == "melhor_sem_filtro": idx = np.where(falta, _melhor(X, base_ok, s), idx)
    elif modo == "relaxar":
        for k in range(len(filtros) - 1, -1, -1):
            if not falta.any(): break
            idx = np.where(falta, _melhor(X, passa(filtros[:k]), s), idx); falta = (idx < 0) & base_ok.any(axis=1)
    elif modo == "nao_apostar": idx = np.where(falta, -2, idx)
    else: raise ValueError(f"regra {rg['nome']}: sem_candidato '{modo}' desconhecido")
    return idx


def serie_procedimento(G, D, segs, idx_seg):
    """lucro e nº de apostas por jogo avaliado do procedimento (−2 = não apostou: lucro 0, o jogo conta) + escolha e segmento por jogo"""
    ini = np.array([a for a, _ in segs], dtype=np.int64); fim = np.array([b for _, b in segs], dtype=np.int64)
    s = np.searchsorted(ini, G, side="right") - 1; dentro = (s >= 0) & (G < fim[np.clip(s, 0, None)])
    esc = np.where(dentro, idx_seg[np.clip(s, 0, None)], -1); av = (esc >= 0) | (esc == -2); k = np.where(av)[0]; e_ = esc[av]; op = np.maximum(e_, 0)
    lu = np.where(e_ >= 0, D["Lg"][op, k], 0.0); nb = np.where(e_ >= 0, D["E"][op, k, 0], 0.0)
    return pd.Series(lu, index=G[av]), pd.Series(nb, index=G[av]), esc, np.where(dentro, s, -1)


# ---------------------------------------------------------------- funil: avaliação ----------------------------------------------------------------
def _bt(B, s):
    if len(s) < 30: return dict(media=np.nan, ic_lo=np.nan, ic_hi=np.nan, p_valor=np.nan, n=len(s))
    r = boot_media(s.values.astype(float), _clu(s.index.values), B=B_BOOT); r["n"] = len(s); return r


def _media_se(s):
    """média e erro-padrão por cluster de 10 gameids (recortes por período e comparações par a par, sem bootstrap)"""
    if len(s) < 2: return (float(s.mean()) if len(s) else np.nan), np.nan
    c = pd.DataFrame(dict(x=s.values, c=(s.index.values - 1) // 10)).groupby("c")["x"].agg(["sum", "size"]); m = c["sum"].sum() / c["size"].sum(); Gc = len(c)
    return float(m), float(np.sqrt(((c["sum"] - c["size"] * m) ** 2).sum() * Gc / max(Gc - 1, 1)) / c["size"].sum())


def risco_por_aposta(D, esc, js):
    """AUDITORIA 02/10: max drawdown e pior sequência de perdas na ordem das APOSTAS (jogo → minuto). O risco por jogo soma as apostas do jogo
    e esconde perdas dentro dele (MULTI). esc = opção usada em cada jogo de js (−1/−2 = sem aposta)."""
    g_, t_, L_ = [], [], []
    for k in np.unique(esc[esc >= 0]):
        b = D["bets"][k]; m = np.isin(b["g"], js[esc == k]); g_.append(b["g"][m]); t_.append(b["t"][m]); L_.append(b["L"][m])
    if not g_ or not sum(len(x) for x in L_): return dict(maxdd_por_aposta=0.0, pior_seq_apostas_perdidas=0)
    g_, t_, L_ = np.concatenate(g_), np.concatenate(t_), np.concatenate(L_); o = np.lexsort((t_, g_)); L_ = L_[o]
    c = np.cumsum(L_); run = best = 0
    for v in L_:
        run = run + 1 if v < 0 else 0; best = max(best, run)
    return dict(maxdd_por_aposta=float(np.max(np.maximum.accumulate(np.r_[0.0, c])[1:] - c)), pior_seq_apostas_perdidas=int(best))


def risco_serie(lu, nb):
    """RISCO / VOLATILIDADE da série de lucro por jogo avaliado (em ordem de gameid)"""
    x = lu.values.astype(float); n = len(x)
    if n < 2: return {}
    c = np.cumsum(x); mdd = float(np.max(np.maximum.accumulate(np.r_[0.0, c])[1:] - c)); sd = float(x.std(ddof=1)); down = float(np.sqrt(np.mean(np.minimum(x, 0) ** 2)))
    run = best = 0
    for v in x:
        run = run + 1 if v < 0 else 0; best = max(best, run)
    blocos = [x[i:i + BLOCO_RISCO].mean() for i in range(0, n - BLOCO_RISCO + 1, BLOCO_RISCO)]
    return dict(lucro=float(c[-1]), n_apostas=int(nb.sum()), roi=float(c[-1] / nb.sum()) if nb.sum() > 0 else np.nan, vol_jogo=sd,
                sharpe_jogo=float(x.mean() / sd) if sd > 0 else np.nan, sortino_jogo=float(x.mean() / down) if down > 0 else np.nan, maxdd=mdd,
                calmar=float(c[-1] / mdd) if mdd > 0 else np.nan, pior_seq_jogos_negativos=int(best),
                pct_blocos_positivos=float(np.mean(np.array(blocos) > 0)) if blocos else np.nan, pior_bloco_ppg=float(np.min(blocos)) if blocos else np.nan)


def overfit_serie(lu, esc_g, seg_g, G, js, M_ppg, ii, D, El, pos, blocos):
    """OVERFIT / ESTABILIDADE: otimismo = PPG histórico da opção escolhida (o que a escolha 'prometia') − PPG realizado; percentil da
    escolha entre as opções elegíveis no início de cada bloco do motor (0,5 = escolher ao acaso) e % de blocos acima da média delas."""
    gi = np.searchsorted(G, js); e_ = esc_g[gi]; sg = seg_g[gi]; ok = e_ >= 0
    prom = np.full(len(js), np.nan); prom[ok] = M_ppg[ii[sg[ok]], e_[ok]]
    out = dict(ppg_prometido=float(np.nanmean(prom)) if ok.any() else np.nan)
    out["otimismo"] = out["ppg_prometido"] - float(lu.reindex(js[ok]).mean()) if ok.any() else np.nan
    pct, acima = [], []
    for (_, g0, g1) in blocos:
        mb = (js >= g0) & (js < g1)
        if g0 not in pos or not mb.any(): continue
        el = np.where(El[pos[g0]])[0]
        if len(el) < 2: continue
        lo_ = D["Lg"][el][:, gi[mb]].sum(axis=1); pr = float(lu.reindex(js[mb]).sum())
        pct.append(float(np.mean(lo_ < pr) + 0.5 * np.mean(lo_ == pr))); acima.append(pr > lo_.mean())
    out.update(percentil_oos_medio=float(np.mean(pct)) if pct else np.nan, pct_blocos_acima_do_acaso=float(np.mean(acima)) if acima else np.nan, blocos_avaliados=len(pct))
    return out


def benchmarks(info, G, fu, cfg, D, nomes):
    """BENCHMARKS (lucro e nº de apostas POR JOGO, posição em G), com a política e os minutos do funil e a faixa de odd do motor:
    BASE_<modelo> = a opção BASELINE_MODELO (sem flag) em todos os jogos · FIXO_<modelo> = cada modelo (sem flag) fixo, inclusive os INI (sem a odd
    do momento); um modelo que só existe na fase limpa tem lucro 0 antes de existir · FAVORITO / ZEBRA = 1 u no lado de menor / maior odd (FIRST: no
    1º minuto do funil em que essa odd está na faixa ODD_MIN–ODD_MAX; MULTI: em todo minuto em que está) · VIG_MERCADO = dutching (1 u dividida
    na proporção 1/odd: lucro 1/S − 1 = −margem da casa, igual qualquer que seja o vencedor; FIRST: 1º minuto executável; MULTI: todo minuto executável)."""
    nG = len(G); out = {}
    if BASELINE_MODELO in nomes:
        j = nomes.index(BASELINE_MODELO); out[f"BASE_{BASELINE_MODELO} (fixo, sem flag)"] = (D["Lg"][j].copy(), D["E"][j, :, 0].copy(), j)
    for j, nm in enumerate(nomes):                                                 # TODOS os modelos fixos, sem flag (MOM e INI): quanto rende cada um sozinho
        if "|" not in nm: out[f"FIXO_{nm} (sem flag)"] = (D["Lg"][j].copy(), D["E"][j, :, 0].copy(), j)
    x = info[info["t"].isin(list(fu["tempos"])) & info["gameid"].isin(set(G.tolist()))][["t", "gameid", "side", "y", "odd_t"]].copy()
    x["odd_t"] = pd.to_numeric(x["odd_t"], errors="coerce"); x = x[np.isfinite(x["odd_t"]) & (x["odd_t"] > 1)]
    x = x[x.groupby(["t", "gameid"])["side"].transform("nunique") == 2].sort_values(["gameid", "t", "odd_t", "side"], kind="mergesort")
    def por_jogo(d, L, n):
        ix = np.searchsorted(G, d["gameid"].values); return np.bincount(ix, weights=L, minlength=nG), np.bincount(ix, weights=n, minlength=nG)
    for nome, lado in (("FAVORITO (todos os jogos)", x.groupby(["gameid", "t"]).head(1)), ("ZEBRA (todos os jogos)", x.groupby(["gameid", "t"]).tail(1))):
        d = lado[lado["odd_t"].between(cfg["ODD_MIN"], cfg["ODD_MAX"])].sort_values(["gameid", "t"], kind="mergesort")
        if fu["politica"] == "FIRST": d = d.groupby("gameid").head(1)
        out[nome] = por_jogo(d, np.where(d["y"].values == 1, d["odd_t"].values - 1.0, -1.0), np.ones(len(d))) + (None,)
    # VIG_MERCADO (AUDITORIA 02/10 — antes era ½ u em cada lado, cujo retorno VARIA com o vencedor): "dutching" = 1 u dividida entre os 2 lados
    # na proporção 1/odd, de modo que qualquer vencedor paga o mesmo: retorno = 1/S, S = 1/odd_A + 1/odd_B ⇒ lucro = 1/S − 1 = −margem (negativo
    # sempre que as odds têm margem, S > 1; S < 1 seria odd de arbitragem — o log avisa quantos (jogo, minuto) estão assim).
    # Só em (jogo, minuto) que o universo executável aceitaria: os 2 lados com odd > 1 e pelo menos um na faixa ODD_MIN–ODD_MAX.
    S = x.groupby(["gameid", "t"]).agg(S=("odd_t", lambda o: float(np.sum(1.0 / o.values))), na_faixa=("odd_t", lambda o: bool(o.between(cfg["ODD_MIN"], cfg["ODD_MAX"]).any()))).reset_index()
    S = S[S["na_faixa"]].sort_values(["gameid", "t"], kind="mergesort")
    n_arb = int((S["S"] < 1).sum())
    if n_arb: log(f"ATENÇÃO {fu['nome']}: {n_arb} (jogo, minuto) executáveis com 1/odd_A + 1/odd_B < 1 (odds SEM margem/arbitragem) — confira as odds desses jogos")
    if fu["politica"] == "FIRST": S = S.groupby("gameid").head(1)
    out["VIG_MERCADO (dutching: perde a margem 1/S − 1)"] = por_jogo(S, 1.0 / S["S"].values - 1.0, np.ones(len(S))) + (None,)
    return out


def _brier_proc(D, ks, gi):
    """Brier / log loss / skill das previsões 'todas' da opção escolhida em cada jogo (mesmas linhas que as métricas 'todas' usam)"""
    ok = ks >= 0
    if not ok.any(): return {}
    t = D["T"][ks[ok], gi[ok]].sum(axis=0)
    if t[0] <= 0: return {}
    return dict(n_linhas_previsao=int(t[0]), brier_escolhida=t[3] / t[0], brier_mercado_mesmas_linhas=t[4] / t[0], bsskill_escolhida=(t[4] - t[3]) / t[0],
                ll_escolhida=t[1] / t[0], ll_mercado_mesmas_linhas=t[2] / t[0], llskill_escolhida=(t[2] - t[1]) / t[0])



# ---------------------------------------------------------------- v5.0: CLV / MARKOUT, EV calibrado, estatísticas por aposta ----------------------------------------------------------------
ODDS = None                          # tabela (jogo, lado) × minuto com q (sem margem) e odd — montada em main(); só para AVALIAÇÃO e métricas do histórico


def _lado(x):
    """chave do lado (mesmo tipo dos dois lados do merge): inteiro se for número, senão texto"""
    v = pd.to_numeric(pd.Series(np.asarray(x)), errors="coerce")
    return v.astype("int64").values if v.notna().all() else pd.Series(np.asarray(x)).astype(str).values


def odds_so_minuto(B, t, L0):
    """zz{t} de um minuto SEM modelo (40/45): só gameid, lado, odd e q sem margem (mesma conta do preparar do motor); None se o arquivo não existir"""
    fp = os.path.join(B.CFG["PASTA_DADOS"], f"{B.CFG['PREFIXO_GRANDE']}{t}.xlsx")
    if not os.path.exists(fp): return None
    G_ = B.carregar_bruto(t); g_ = pd.to_numeric(G_["gameid"], errors="coerce"); G_ = G_[(g_ < L0).values].copy()
    if f"odd{t}" not in G_.columns or f"vsodd{t}" not in G_.columns: log(f"zz{t}: sem odd{t}/vsodd{t} — minuto ignorado no CLV"); return None
    G_ = G_[G_.groupby("gameid")["side"].transform("nunique") == 2]
    oa, ob = pd.to_numeric(G_[f"odd{t}"], errors="coerce"), pd.to_numeric(G_[f"vsodd{t}"], errors="coerce")
    val = np.isfinite(oa) & np.isfinite(ob) & (oa > 1) & (ob > 1); ia, ib = (1.0 / oa).where(val), (1.0 / ob).where(val)
    return pd.DataFrame(dict(gameid=pd.to_numeric(G_["gameid"]).astype("int64").values, side=G_["side"].values, odd=oa.where(val).values, q=(ia / (ia + ib)).values))


def montar_odds(frames_sujo, extras):
    """ODDS: para cada (jogo, lado) e minuto, q sem margem e odd. Minutos com modelo vêm dos frames (as MESMAS colunas q_devig_t/odd_t das apostas);
    os de 'extras' (40/45) só das odds. Só linhas com odd válida (> 1) e 0 < q < 1."""
    partes = [pd.DataFrame(dict(gameid=pd.to_numeric(f["gameid"]).astype("int64").values, side=f["side"].values, t=int(t), odd=pd.to_numeric(f["odd_t"], errors="coerce").values,
                                q=pd.to_numeric(f["q_devig_t"], errors="coerce").values)) for t, f in frames_sujo.items()]
    partes += [d.assign(t=int(t)) for t, d in extras.items()]
    L = pd.concat(partes, ignore_index=True); L["side"] = _lado(L["side"])
    L = L[np.isfinite(L["q"]) & (L["q"] > 0) & (L["q"] < 1) & np.isfinite(L["odd"]) & (L["odd"] > 1)]
    assert not L.duplicated(["gameid", "side", "t"]).any(), "ODDS: (jogo, lado, minuto) repetido"
    mins = np.array(sorted(L["t"].unique()), dtype=np.int64)
    Wq = L.pivot(index=["gameid", "side"], columns="t", values="q").reindex(columns=mins); Wo = L.pivot(index=["gameid", "side"], columns="t", values="odd").reindex(index=Wq.index, columns=mins)
    Q, O = Wq.values.astype(float), Wo.values.astype(float); v = np.isfinite(Q)
    ultimo = np.where(v.any(1), v.shape[1] - 1 - np.argmax(v[:, ::-1], axis=1), -1)
    lut = np.full(1000, -1, dtype=np.int64); lut[mins] = np.arange(len(mins))
    gid_ = Wq.index.get_level_values(0).values; parceiro = np.full(len(gid_), -1, dtype=np.int64)   # linha do OUTRO lado do mesmo jogo (green-up com hedge)
    ini_ = np.r_[0, np.flatnonzero(np.diff(gid_)) + 1]; fim_ = np.r_[ini_[1:], len(gid_)]
    dois = (fim_ - ini_) == 2; parceiro[ini_[dois]] = ini_[dois] + 1; parceiro[ini_[dois] + 1] = ini_[dois]
    return dict(idx=Wq.index, Q=Q, O=O, min=mins, lut=lut, ultimo=ultimo, parceiro=parceiro, longo=L.sort_values(["gameid", "side", "t"]).reset_index(drop=True))


def clv_campos(g, side, t, odd, q):
    """CLV e markout de cada aposta (lado apostado). Usa só minutos DEPOIS do minuto da aposta, do MESMO jogo. NaN = sem dado naquele minuto."""
    n = len(g); out = {}; odd = np.asarray(odd, float); q = np.asarray(q, float); t = np.asarray(t, np.int64)
    if ODDS is None or n == 0:
        for k_ in ["q_fech", "odd_fech", "clv", "supera", "supera_bruto", "mk_fech"] + [f"{a}{h}" for h in HORIZONTES_CLV for a in ("q", "mk", "ev", "gr")]: out[k_] = np.full(n, np.nan)
        out["minuto_fech"] = np.full(n, -1); return out
    ri = ODDS["idx"].get_indexer(pd.MultiIndex.from_arrays([np.asarray(g, np.int64), _lado(side)])); ok = ri >= 0; ri0 = np.maximum(ri, 0)
    for h in HORIZONTES_CLV:
        c = ODDS["lut"][np.clip(t + h, 0, len(ODDS["lut"]) - 1)]; m = ok & (c >= 0)
        qh = np.full(n, np.nan); vh = np.full(n, np.nan); qh[m] = ODDS["Q"][ri0[m], c[m]]
        pa_ = ODDS["parceiro"][ri0]; mp_ = m & (pa_ >= 0); vh[mp_] = ODDS["O"][pa_[mp_], c[mp_]]         # odd do OUTRO lado em t+h
        out[f"q{h}"] = qh; out[f"mk{h}"] = qh - q; out[f"ev{h}"] = odd * qh - 1.0
        out[f"gr{h}"] = odd * (1.0 - 1.0 / vh) - 1.0                                          # REVISÃO: green-up REAL (hedge apostando no outro lado na odd da casa)
    uc = np.where(ok, ODDS["ultimo"][ri0], -1); mc = np.where(uc >= 0, ODDS["min"][np.maximum(uc, 0)], -1); tem = ok & (uc >= 0) & (mc > t)
    qc = np.full(n, np.nan); oc = np.full(n, np.nan); qc[tem] = ODDS["Q"][ri0[tem], uc[tem]]; oc[tem] = ODDS["O"][ri0[tem], uc[tem]]
    out.update(q_fech=qc, odd_fech=oc, minuto_fech=np.where(tem, mc, -1), clv=odd * qc - 1.0, mk_fech=qc - q,
               supera=np.where(tem, (qc > q).astype(float), np.nan), supera_bruto=np.where(tem, (odd > oc).astype(float), np.nan))
    return out


def _wilson(k, n, z=1.96):
    if n <= 0: return np.nan, np.nan
    f = k / n; den = 1 + z ** 2 / n; c = (f + z ** 2 / (2 * n)) / den; hw = z * np.sqrt(f * (1 - f) / n + z ** 2 / (4 * n ** 2)) / den
    return float(c - hw), float(c + hw)


def _boot_razao(num, den, gid, tag, B=None):
    """IC 95% de Σnum/Σden por bootstrap de clusters de 10 gameids (sorteio multinomial dos clusters, como o boot_media do motor)"""
    num, den, gid = np.asarray(num, float), np.asarray(den, float), np.asarray(gid)
    if len(num) < 30 or den.sum() <= 0: return np.nan, np.nan
    rng = _rng_de(tag, num, den, gid); X = np.column_stack([num, den]); soma, tam, n = _cluster_sums(X, _clu(gid))
    cnt = rng.multinomial(n, np.full(n, 1.0 / n), size=int(B or B_BOOT_APOSTAS)).astype(float); S = cnt @ soma
    r = S[:, 0] / np.where(S[:, 1] > 0, S[:, 1], np.nan)
    return float(np.nanquantile(r, 0.025)), float(np.nanquantile(r, 0.975))


def calibrador_ev(bt):
    """para UMA opção: faixas fixas de EV (BINS_EV) e somas acumuladas, NA ORDEM (jogo, minuto), de lucro e nº de apostas por faixa"""
    ev = bt["p"] * bt["odd"] - 1.0; fx = np.clip(np.searchsorted(BINS_EV, ev, side="right") - 1, 0, len(BINS_EV) - 2)
    nb = len(BINS_EV) - 1; L = np.zeros((nb, len(ev) + 1)); N = np.zeros((nb, len(ev) + 1))
    for b in range(nb):
        m = fx == b; L[b, 1:] = np.cumsum(np.where(m, bt["L"], 0.0)); N[b, 1:] = np.cumsum(m)
    return dict(g=bt["g"], L=L, N=N)


def apostas_serie(D, ks, js, a_par, cal):
    """as apostas de um procedimento: em cada jogo de js, as apostas da opção ks[i] (−1/−2 = nenhuma). a_par = início do par de cada jogo.
    Acrescenta EV do modelo, EV CALIBRADO (só apostas da MESMA opção em jogos < início do par) e CLV/markout."""
    cols = ["g", "t", "side", "odd", "p", "y", "q", "L", "edge", "clv", "supera"]; partes = []
    for k in np.unique(ks[ks >= 0]):
        bt = D["bets"][k]; gk = js[ks == k]; m = np.isin(bt["g"], gk)
        if not m.any(): continue
        d = {c: np.asarray(bt[c])[m] for c in cols}; d["k"] = np.full(int(m.sum()), k); d["a_par"] = a_par[np.searchsorted(js, d["g"])]
        if k not in cal: cal[k] = calibrador_ev(bt)
        c_ = cal[k]; i_ = np.searchsorted(c_["g"], d["a_par"] - int(EMBARGO_ESCOLHA), side="left")   # apostas da opção com jogo < a_par − EMBARGO
        ev = d["p"] * d["odd"] - 1.0; fx = np.clip(np.searchsorted(BINS_EV, ev, side="right") - 1, 0, len(BINS_EV) - 2)
        nL, nN = c_["L"][fx, i_], c_["N"][fx, i_]
        d["ev"] = ev; d["calib_n"] = nN; d["ev_cal"] = np.where(nN >= MIN_CALIB, nL / np.maximum(nN, 1), ev)
        d["calib_ate_jogo"] = np.where(i_ > 0, c_["g"][np.maximum(i_ - 1, 0)], -1)
        partes.append(d)
    if not partes: return None
    A = {c: np.concatenate([d[c] for d in partes]) for c in partes[0]}
    o = np.lexsort((A["t"], A["g"])); A = {c: v[o] for c, v in A.items()}
    A.update(clv_campos(A["g"], A["side"], A["t"], A["odd"], A["q"]))
    return A


def stats_apostas(A, tag):
    """ROI realizado × esperado (EV do modelo e EV calibrado), regressão lucro ~ EV, variância realizada × esperada, CLV e markout"""
    if A is None or not len(A["L"]): return {}, []
    L, p, odd, g = A["L"], A["p"], A["odd"], A["g"]; n = len(L); ev = A["ev"]
    d = dict(n_apostas_serie=n, roi_apostas=float(L.mean()), ev_medio_modelo=float(ev.mean()), ev_medio_calibrado=float(A["ev_cal"].mean()),
             pct_ev_cal_sem_historico=float(np.mean(A["calib_n"] < MIN_CALIB)))
    d["roi_ic_lo"], d["roi_ic_hi"] = _boot_razao(L, np.ones(n), g, tag + "|roi")
    d["roi_menos_ev_modelo"] = d["roi_apostas"] - d["ev_medio_modelo"]; d["roi_menos_ev_calibrado"] = d["roi_apostas"] - d["ev_medio_calibrado"]
    if n >= 100 and np.std(ev) > 0:                                                         # descritivo: ideal intercepto 0 e inclinação 1
        X = np.column_stack([np.ones(n), ev]); beta = np.linalg.lstsq(X, L, rcond=None)[0]; Hi = np.linalg.inv(X.T @ X)
        cod, _ = pd.factorize(_clu(g)); Gm = np.zeros((cod.max() + 1, 2)); np.add.at(Gm, cod, X * (L - X @ beta)[:, None])
        d.update(ev_reg_intercepto=float(beta[0]), ev_reg_inclinacao=float(beta[1]), ev_reg_inclinacao_se=float(np.sqrt(np.diag(Hi @ (Gm.T @ Gm) @ Hi))[1]))
    var_esp = p * (odd - 1.0 - ev) ** 2 + (1 - p) * (-1.0 - ev) ** 2
    d.update(dp_lucro_aposta=float(L.std(ddof=1)) if n > 1 else np.nan, dp_esperado_modelo=float(np.sqrt(var_esp.mean())),
             razao_variancia_real_esperada=float(L.var(ddof=1) / var_esp.mean()) if n > 1 and var_esp.mean() > 0 else np.nan)
    tem = np.isfinite(A["clv"]); nc = int(tem.sum())
    d.update(n_com_fechamento=nc, pct_sem_fechamento=float(1 - nc / n), clv_fech_medio=float(A["clv"][tem].mean()) if nc else np.nan,
             supera_fech_taxa=float(np.nanmean(A["supera"])) if nc else np.nan, supera_fech_bruto_taxa=float(np.nanmean(A["supera_bruto"])) if nc else np.nan,
             mk_fech_medio=float(np.nanmean(A["mk_fech"])) if nc else np.nan)
    if nc:
        d["clv_ic_lo"], d["clv_ic_hi"] = _boot_razao(A["clv"][tem], np.ones(nc), g[tem], tag + "|clv")
        d["supera_ic_lo"], d["supera_ic_hi"] = _wilson(float(np.nansum(A["supera"])), nc)
    hz = []
    for h in HORIZONTES_CLV:
        mk = A[f"mk{h}"]; m = np.isfinite(mk); k_ = int(m.sum())
        r = dict(horizonte=f"+{h}min", n=k_, perdidas=n - k_, pct_perdidas=float(1 - k_ / n))
        if k_:
            fav, con = mk[m] > 0, mk[m] < 0
            r.update(taxa_a_favor=float(fav.mean()), taxa_contra=float(con.mean()), taxa_parada=float((mk[m] == 0).mean()), mk_medio=float(mk[m].mean()),
                     mk_medio_quando_a_favor=float(mk[m][fav].mean()) if fav.any() else np.nan, mk_medio_quando_contra=float(mk[m][con].mean()) if con.any() else np.nan,
                     mk_p10=float(np.quantile(mk[m], .1)), mk_p50=float(np.quantile(mk[m], .5)), mk_p90=float(np.quantile(mk[m], .9)),
                     roi_markout=float(A[f"ev{h}"][m].mean()), green_up_medio=float(np.nanmean(A[f"gr{h}"][m])))
            r["taxa_a_favor_ic_lo"], r["taxa_a_favor_ic_hi"] = _wilson(float(fav.sum()), k_)
            r["roi_markout_ic_lo"], r["roi_markout_ic_hi"] = _boot_razao(A[f"ev{h}"][m], np.ones(k_), g[m], tag + f"|mk{h}")
            if h == 5: d.update(mk5_taxa_a_favor=r["taxa_a_favor"], mk5_roi=r["roi_markout"])
        hz.append(r)
    if nc:
        fav, con = A["mk_fech"][tem] > 0, A["mk_fech"][tem] < 0
        hz.append(dict(horizonte="fechamento", n=nc, perdidas=n - nc, pct_perdidas=float(1 - nc / n), taxa_a_favor=float(fav.mean()), taxa_contra=float(con.mean()),
                       taxa_parada=float((A["mk_fech"][tem] == 0).mean()), mk_medio=d["mk_fech_medio"], mk_medio_quando_a_favor=float(A["mk_fech"][tem][fav].mean()) if fav.any() else np.nan,
                       mk_medio_quando_contra=float(A["mk_fech"][tem][con].mean()) if con.any() else np.nan, roi_markout=d["clv_fech_medio"],
                       roi_markout_ic_lo=d.get("clv_ic_lo"), roi_markout_ic_hi=d.get("clv_ic_hi"), taxa_a_favor_ic_lo=d.get("supera_ic_lo"), taxa_a_favor_ic_hi=d.get("supera_ic_hi"),
                       green_up_medio=float(np.nanmean(A["odd"][tem] / A["odd_fech"][tem] - 1.0))))
    return d, hz


# ---------------------------------------------------------------- v5.0: MODO SOMBRA ----------------------------------------------------------------
def acumulados_sombra(D):
    """somas acumuladas POR JOGO (posição em G) de cada opção: Brier skill das linhas 'todas' e CLV das apostas (com fechamento)"""
    cs = lambda x: np.concatenate([np.zeros((x.shape[0], 1)), np.cumsum(x, axis=1)], axis=1)
    return dict(brier=(cs(D["T"][:, :, 4] - D["T"][:, :, 3]), cs(D["T"][:, :, 0])), clv=(cs(D["CB"]), cs(D["CN"])))   # clv = média de q_fech − q (ref. 0)


def sombra(idx_pares, a_pares, fim_pares, G, AC, crit, n_sombra, min_amostra):
    """produção de cada par com o MODO SOMBRA sobre as escolhas da regra (idx_pares, uma por par, na ordem). Só usa jogos < início do par.
    Devolve a opção de produção por par e os eventos (início de sombra, promoção, rejeição)."""
    S, N = AC[crit]; prod = -1; cand = -1; s0 = None; out = np.full(len(a_pares), -1, dtype=np.int64); ev = []; avisou = False
    for i, (a, fim, r) in enumerate(zip(a_pares, fim_pares, idx_pares)):                # fim = início do par − EMBARGO_ESCOLHA (fim exclusivo do histórico)
        if prod < 0:
            if r >= 0: prod = r; ev.append(dict(a=int(a), evento="inicio_producao", producao=int(r)))
        elif r >= 0 and r != prod:
            if cand != r:
                cand, s0, avisou = r, int(a), False; ev.append(dict(a=int(a), evento="inicio_sombra", producao=int(prod), sombra=int(r), sombra_desde=s0))
            else:
                lo, hi = np.searchsorted(G, s0, "left"), np.searchsorted(G, fim, "left")
                if hi - lo >= n_sombra:
                    nc_, np_ = N[cand, hi] - N[cand, lo], N[prod, hi] - N[prod, lo]
                    if nc_ >= min_amostra and np_ >= min_amostra:
                        mc, mp = (S[cand, hi] - S[cand, lo]) / nc_, (S[prod, hi] - S[prod, lo]) / np_
                        e_ = dict(a=int(a), producao=int(prod), sombra=int(cand), sombra_desde=s0, jogos_em_sombra=int(hi - lo), metrica_sombra=float(mc), metrica_producao=float(mp),
                                  amostra_sombra=float(nc_), amostra_producao=float(np_))
                        if mc > mp + 1e-12: ev.append(dict(e_, evento="promovida")); prod, cand = cand, -1   # empate NÃO vence (tolerância numérica)
                        else: ev.append(dict(e_, evento="rejeitada")); s0, avisou = int(a), False
                    elif not avisou:                                                            # REVISÃO: amostra insuficiente → registra (1× por janela)
                        ev.append(dict(a=int(a), evento="indeterminada", producao=int(prod), sombra=int(cand), sombra_desde=s0, jogos_em_sombra=int(hi - lo),
                                       amostra_sombra=float(nc_), amostra_producao=float(np_))); avisou = True
        elif r == prod: cand = -1
        out[i] = -2 if (r == -2 and prod >= 0) else prod                                        # REVISÃO: a regra decidiu não apostar → produção não aposta neste par
    return out, ev


# ---------------------------------------------------------------- v5.0: ROLLING, ALARMES, GRÁFICOS ----------------------------------------------------------------
def eixo_tempo(js, datas):
    """eixo para as janelas rolantes: dias (se houver data por jogo) ou nº do jogo executável; e as janelas curta/longa"""
    if datas is not None:
        d = pd.Series(js).map(datas)
        if d.notna().all():
            x = np.maximum.accumulate((pd.to_datetime(d.values) - pd.Timestamp("2000-01-01")).days.values.astype(float))
            return x, DIAS_ROLL_CURTA, DIAS_ROLL_LONGA, "dias"
    return np.arange(len(js), dtype=float), JANELA_ROLL_CURTA, JANELA_ROLL_LONGA, "jogos"


def rolling_janela(x, num, den, W):
    """Σnum/Σden nos jogos com x em (x_i − W, x_i]; NaN se Σden = 0 ou se a janela ainda não está cheia (x_i − x_0 < W − 1)"""
    if not len(x): return np.array([])
    c1, c2 = np.concatenate([[0.0], np.cumsum(num)]), np.concatenate([[0.0], np.cumsum(den)])
    lo = np.searchsorted(x, x - W, side="right"); hi = np.arange(1, len(x) + 1)
    dd = c2[hi] - c2[lo]
    return np.where((dd > 0) & (x - x[0] >= W - 1), (c1[hi] - c1[lo]) / np.where(dd > 0, dd, 1), np.nan)


def _svg_painel(series, titulo, w=1140, h=230, y0=None):
    from xml.sax.saxutils import escape as esc
    """um painel SVG simples (sem matplotlib): linhas {nome: (x, y)}"""
    cores = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b", "#7f7f7f"]
    xs = np.concatenate([np.asarray(x, float) for x, _ in series.values()]) if series else np.array([0, 1.0])
    ys = np.concatenate([np.asarray(y, float)[np.isfinite(y)] for _, y in series.values()]) if series else np.array([0, 1.0])
    if not len(ys): ys = np.array([0, 1.0])
    x0, x1 = float(np.nanmin(xs)), float(np.nanmax(xs)); ya, yb = float(np.nanmin(ys)), float(np.nanmax(ys))
    if y0 is not None: ya, yb = min(ya, y0), max(yb, y0)
    if x1 <= x0: x1 = x0 + 1
    if yb <= ya: yb = ya + 1
    L_, R_, T_, B_ = 60, 330, 22, 24; fx = lambda v: L_ + (v - x0) / (x1 - x0) * (w - L_ - R_); fy = lambda v: T_ + (yb - v) / (yb - ya) * (h - T_ - B_)
    out = [f'<g><text x="{L_}" y="15" font-size="13" font-family="sans-serif">{esc(titulo)}</text>',
           f'<rect x="{L_}" y="{T_}" width="{w - L_ - R_}" height="{h - T_ - B_}" fill="none" stroke="#999"/>']
    for v in (ya, yb) + ((y0,) if y0 is not None else ()):
        out.append(f'<text x="{L_ - 4}" y="{fy(v) + 4:.1f}" font-size="10" text-anchor="end" font-family="sans-serif">{v:.3g}</text>')
    if y0 is not None: out.append(f'<line x1="{L_}" x2="{w - R_}" y1="{fy(y0):.1f}" y2="{fy(y0):.1f}" stroke="#bbb" stroke-dasharray="4,3"/>')
    for i, (nm, (x, y)) in enumerate(series.items()):
        x, y = np.asarray(x, float), np.asarray(y, float); ok = np.isfinite(y)
        passo = max(1, int(ok.sum() // 1500)); pts = " ".join(f"{fx(a):.1f},{fy(b):.1f}" for a, b in list(zip(x[ok], y[ok]))[::passo])
        out.append(f'<polyline fill="none" stroke="{cores[i % len(cores)]}" stroke-width="1.3" points="{pts}"/>')
        out.append(f'<text x="{w - R_ + 6}" y="{T_ + 12 + 13 * i}" font-size="10" fill="{cores[i % len(cores)]}" font-family="sans-serif">{esc(nm[:58])}</text>')
    out.append(f'<text x="{L_}" y="{h - 6}" font-size="10" font-family="sans-serif">{x0:.0f}</text><text x="{w - R_}" y="{h - 6}" font-size="10" text-anchor="end" font-family="sans-serif">{x1:.0f}</text></g>')
    return out, h


def gravar_svg(arq, paineis, titulo):
    corpo, y = [], 30
    for (series, tit, y0) in paineis:
        p_, h_ = _svg_painel(series, tit, y0=y0); corpo.append(f'<g transform="translate(0,{y})">' + "".join(p_) + "</g>"); y += h_ + 10
    open(arq, "w", encoding="utf-8").write(f'<svg xmlns="http://www.w3.org/2000/svg" width="1140" height="{y + 10}" style="background:#fff">'
                                           f'<text x="10" y="18" font-size="14" font-weight="bold" font-family="sans-serif">{__import__("xml.sax.saxutils", fromlist=["escape"]).escape(titulo)}</text>' + "".join(corpo) + "</svg>")


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


# ---- paralelismo PRÓPRIO (v4.4): não depende da versão do paralelo_v90.py que estiver na pasta ----
def _memoria_total_gb():
    """memória física total em GB (Windows via kernel32; Linux via sysconf); None se não der para medir"""
    try:
        import ctypes
        class _M(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong), ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong), ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong), ("sullAvailExtendedVirtual", ctypes.c_ulonglong)]
        m = _M(); m.dwLength = ctypes.sizeof(_M)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)): return m.ullTotalPhys / 2 ** 30
    except Exception:
        pass
    try: return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2 ** 30
    except Exception: return None


def escolher_processos(n_tarefas):
    """quantos processos ao mesmo tempo: N_PROCESSOS > 0 fixo; 0 = min(núcleos − 1, memória total ÷ GB_POR_PROCESSO), no máximo o nº de candidatos"""
    nuc = os.cpu_count() or 2; mem = _memoria_total_gb()
    if N_PROCESSOS and int(N_PROCESSOS) > 0:
        n = max(1, min(int(N_PROCESSOS), n_tarefas)); motivo = f"N_PROCESSOS = {int(N_PROCESSOS)} fixo"
    else:
        teto_mem = int(mem // GB_POR_PROCESSO) if mem else nuc - 1
        n = max(1, min(nuc - 1, teto_mem, n_tarefas)); motivo = f"automático: núcleos − 1 = {nuc - 1} · memória total {mem:.1f} GB ÷ {GB_POR_PROCESSO:g} = {teto_mem}" if mem else f"automático: núcleos − 1 = {nuc - 1} (memória não medida)"
    return n, motivo + f" · {n_tarefas} candidatos"


def rodar_filhos(n):
    """roda os n processos filhos (cada um treina os candidatos itens[i::n], com 1 thread numérica); progresso a cada 5 min; se algum falhar,
    espera os outros terminarem (eles gravam o banco de previsões) e para com erro mostrando o fim do log de quem falhou"""
    import subprocess
    env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
    procs = []
    for i in range(n):
        lg = os.path.join(OUT, f"funil_parte{i}.log"); fh = open(lg, "w", encoding="utf-8")
        procs.append((i, subprocess.Popen([sys.executable, "-u", os.path.abspath(__file__), "--filho", str(i), str(n)], stdout=fh, stderr=subprocess.STDOUT, env=env, cwd=AQUI), fh, lg))
    log(f"{n} processos iniciados (logs em {os.path.join(OUT, 'funil_parte*.log')})"); t0 = t_prog = time.time()
    while any(p.poll() is None for _, p, _, _ in procs):
        time.sleep(15)
        if time.time() - t_prog >= 300:
            t_prog = time.time()
            for i, p, _, lg in procs:
                if p.poll() is not None: continue
                try: ult = [l.strip() for l in open(lg, encoding="utf-8", errors="replace") if "treinos feitos" in l][-1:] or ["(começando)"]
                except Exception: ult = ["(sem log ainda)"]
                log(f"   progresso parte {i} ({(time.time() - t0) / 60:.0f} min): {ult[0][:150]}")
    falhas = []
    for i, p, fh, lg in procs:
        fh.close()
        if p.returncode != 0: falhas.append((i, p.returncode, lg))
    for i, rc, lg in falhas:
        try: cauda = open(lg, encoding="utf-8", errors="replace").read().splitlines()[-25:]
        except Exception: cauda = ["(não consegui ler o log)"]
        print("\n".join(cauda), flush=True)
    if falhas: raise SystemExit(f"{len(falhas)} processo(s) falharam: " + "; ".join(f"parte {i} (código {rc}, ver {lg})" for i, rc, lg in falhas)
                                + " — os que terminaram gravaram o treino no BANCO_PREVISOES (na próxima vez vem de lá)")
    banco = []
    for i, _, _, lg in procs:
        try: banco += [f"parte {i}: " + l.strip()[9:] for l in open(lg, encoding="utf-8", errors="replace")
                       if "banco" in l and any(s_ in l for s_ in ("reaproveitados", "DESLIGADO", "NÃO consegui", "chave diferente", "ilegível", "mudou"))]
        except Exception: pass
    if banco: log(f"BANCO_PREVISOES (dos logs dos processos; {len(banco)} linha(s)):\n    " + "\n    ".join(banco[:60]) + ("\n    …" if len(banco) > 60 else ""))
    log(f"processos terminados em {(time.time() - t0) / 60:.1f} min")



# ---------------------------------------------------------------- principal ----------------------------------------------------------------
def main():
    global CFG
    t0 = time.time(); M = motores(); BS, KS = M["sujo"]
    CFG = BS.CFG                                                                             # v4.6: as cópias da avaliação leem o CFG do motor sujo (o mesmo dicionário)
    G_CLEAN, L0 = int(BS.CFG["G_CLEAN"]), KS["L0"]
    hist_ini = int(HIST_INICIO or BS.CFG["INICIO_TESTE_SUJO"]); min_hist = int(MIN_HIST if MIN_HIST is not None else BS.CFG["SELECAO_MIN_HIST_JOGOS"])
    ini_draft = int(INICIO_OPCOES_DRAFT or (BS.CFG["INICIO_TESTE"] if REGIME_DRAFT == "limpo" else max(G_CLEAN, int(BS.CFG.get("PRO_LIVE_FROM") or G_CLEAN))))
    flag_ini = dict(FLAG_INICIO) if FLAG_INICIO else {}
    for F in FLAGS:
        flag_ini.setdefault(F, hist_ini if F in FLAGS_SUJO else (int(BS.CFG.get("PRO_LIVE_FROM") or G_CLEAN) if F.startswith("bipro") else G_CLEAN))
    regras = list(REGRAS) + list(REGRAS_EXTRAS); nomes_r = [nome_regra(r) for r in regras]
    assert len(set(nomes_r)) == len(nomes_r), f"nomes de regra repetidos: {nomes_r}"
    for r in regras:
        if isinstance(r, str): assert r in METRICAS, f"regra '{r}' não é uma métrica de METRICAS"
        elif "funcao" not in r: assert r["ordenar"] in METRICAS, f"regra {r['nome']}: ordenar '{r['ordenar']}' não é métrica"
    P = [(n, a, b) for n, a, b in (("sujo", hist_ini, G_CLEAN), (f"limpo_{G_CLEAN}_{BS.CFG['INICIO_TESTE'] - 1}", G_CLEAN, int(BS.CFG["INICIO_TESTE"])),
                                   (f"limpo_{BS.CFG['INICIO_TESTE']}_{CORTE_TESTE - 1}", int(BS.CFG["INICIO_TESTE"]), CORTE_TESTE), (f"limpo_{CORTE_TESTE}_{CORTE_ANTIGO - 1}", CORTE_TESTE, CORTE_ANTIGO)) if b > a]
    if CONTINUACAO: P.append((f"continuacao_{CORTE_ANTIGO}_fim", CORTE_ANTIGO, L0))             # v4.9: os jogos que eram lockbox
    PX = P + [(f"desde_{ini_draft}", ini_draft, L0)]                                            # v4.7: + período só da fase limpa com draft (PPG/ranking; sobrepõe os limpos)
    log(f"VALIDAR FUNIL v5.0 · regime draft {REGIME_DRAFT} · C fixo {C_FIXO} · treino a cada {PASSO} · re-escolha a cada {PASSO_ESCOLHA} · minutos {KS['todos']} · "
        f"histórico desde {hist_ini} · opções com draft desde {ini_draft} · flags (início): {flag_ini} · {len(regras)} regras · " + (f"CONTINUAÇÃO: SEM lockbox, jogos >= {CORTE_ANTIGO} entram normalmente" if CONTINUACAO else f"lockbox {L0}+ NUNCA previsto") + f" · períodos {P}")
    for tag, (B, K) in M.items(): log(f"motor {tag}: famílias {B.CFG['FAMILIAS']} · CAL {[c for c in B.CFG['CAL_PARA'] if c.split('_')[0] in B.CFG['FAMILIAS']]} · ATOM {B.ORIGEM_ATOM}")
    brutos = carregar_brutos(BS, KS, conferir=True)
    conferir_colunas_brutas(BS, KS, brutos, [B for (B, _) in M.values()])
    frames = {tag: preparar_frames(B, K, brutos) for tag, (B, K) in M.items()}
    manifesto_features(M)
    BL = blocos_por_motor(M, frames, hist_ini); itens = itens_de_treino(M)
    for tag in BL: log(f"motor {tag}: {len(BL[tag])} blocos de avaliação ({BL[tag][0][1]}–{BL[tag][-1][2] - 1}) → {len(_cortes(BL[tag])) - 1} pares de treino")
    # ---------------- previsões ----------------
    n_pr, motivo = escolher_processos(len(itens)) if N_PROCESSOS != 1 else (1, "N_PROCESSOS = 1 (em série)")
    log(f"treino: {n_pr} processo(s) ao mesmo tempo ({motivo})")
    if n_pr <= 1:
        PS, FS = [], []
        for tag, (B, K) in M.items():
            p, f = prever(B, K, brutos, frames[tag], BL[tag], B.candidatos()); PS.append(p); FS.append(f)
        preds, fits = pd.concat(PS, ignore_index=True), pd.concat(FS, ignore_index=True)
    else:
        del frames
        pasta = os.path.join(OUT, "_partes"); os.makedirs(pasta, exist_ok=True)
        for fn in os.listdir(pasta): os.remove(os.path.join(pasta, fn))
        rodar_filhos(n_pr)
        partes = [pickle.load(open(os.path.join(pasta, f"parte_{i}_de_{n_pr}.pkl"), "rb")) for i in range(n_pr)]
        assert sorted(x for p_ in partes for x in p_["itens"]) == sorted(f"{tg}:{c[0]}" for tg, c in itens), "partes não cobrem todos os candidatos"
        preds = pd.concat([p_["preds"] for p_ in partes if len(p_["preds"])], ignore_index=True); fits = pd.concat([p_["fits"] for p_ in partes if len(p_["fits"])], ignore_index=True)
        frames = {tag: preparar_frames(B, K, brutos) for tag, (B, K) in M.items()}
    assert int(preds["gameid"].max()) < L0, "previsão de jogo do lockbox — ABORTADO"
    fits = _alerta_epv_atual(fits)
    for tag in M: conferir_retreino(M[tag][0], tag, preds[preds["motor"] == tag], fits[fits["motor"] == tag], BL[tag])
    preds.to_csv(os.path.join(OUT, "previsoes_funil.csv.gz"), index=False); fits.to_csv(os.path.join(OUT, "ajustes_funil.csv.gz"), index=False)
    # modelos das famílias so_limpo (com draft; no regime limpo também V6L/V6WRL) só viram opção a partir de ini_draft (fase limpa)
    so_limpo = {n for tag, (B, K) in M.items() for (n, fam, _) in B.candidatos() if B.FAM_DEF[fam]["so_limpo"]}
    base_c = preds["cand"].str.replace(SUF_CAL, "", regex=True); fora = base_c.isin(so_limpo) & (preds["gameid"] < ini_draft)
    log(f"modelos com draft ({sorted(so_limpo)}): {int(fora.sum())} previsões antes do jogo {ini_draft} ficam FORA (fase suja: treinam, mas não são opção)")
    preds = preds[~fora]
    # ---------------- base, universo, linhas por (jogo, minuto) ----------------
    base, info = montar_base_funil(BS, preds, frames["sujo"], flag_ini)
    BS.CFG["TEMPOS"] = list(KS["todos"]); BS.CFG["TEMPOS_DEPLOY"] = sorted({int(t) for f_ in FUNIS for t in f_["tempos"]} | {int(t) for t in METRICAS_TEMPOS})
    uv, _ = universo_executavel(frames["sujo"])
    G = np.array(sorted(int(g) for g in uv if hist_ini <= int(g) < L0), dtype=np.int64); Gs = set(G.tolist())
    assert len(G) and int(G.max()) < L0 and int(base["gameid"].max()) < L0 and int(info["gameid"].max()) < L0, "AUDITORIA: jogo do lockbox no universo/base — ABORTADO"
    modelos = sorted(base["cand"].unique())
    FLG = info.groupby(["t", "gameid"])[FLAGS].max().eq(1).reset_index()          # (jogo, minuto) com algum lado com flag == 1 (vazio antes do início do flag)
    base_por_c = {c: d.copy() for c, d in base.groupby("cand")}
    GMf, start_mod = {}, {}
    for c in modelos:
        g = linhas_jogo(base_por_c[c], c); g = g[g["gameid"].isin(Gs)]
        GMf[c] = g.merge(FLG, on=["t", "gameid"], how="left").fillna({F: False for F in FLAGS}); start_mod[c] = int(g["gameid"].min()) if len(g) else L0
    memo = {}
    def apostas_de(c, F, pol, tempos, ev_min=None):
        k = (c, F, pol, tempos, ev_min)
        if k not in memo:
            col = F
            if ev_min is not None:                                                              # v5.0: limiar de EV no lado escolhido, pelo filtro_col do motor
                col = f"__ev{ev_min}_{F or 'sem_flag'}"; bc = base_por_c[c]                     # (a função apostas copiada NÃO muda): ok = edge > 0 E flag E EV > limiar
                if col not in bc.columns: bc[col] = (((bc[F] == 1) if F else True) & (bc["ev"] > float(ev_min))).astype(int)
            b, _ = apostas(base_por_c[c], c, pol, tempos=list(tempos), filtro_col=col)
            memo[k] = b[b["gameid"].isin(Gs)][["gameid", "t", "side", "p", "y", "q_devig_t", "odd_t", "edge", "lucro"]].copy()
        return memo[k]
    log(f"universo executável (minutos {BS.CFG['TEMPOS_DEPLOY']}) {hist_ini}–{L0 - 1}: {len(G)} jogos (denominador do PPG) · {len(modelos)} modelos")
    global ODDS                                                                                 # v5.0: odds por minuto para CLV / markout (só avaliação e histórico)
    extras = {}
    for t in MINUTOS_SO_ODDS:
        if t in KS["todos"]: continue
        d_ = odds_so_minuto(BS, t, L0)
        if d_ is None: log(f"zz{t}: não está na pasta — markout/fechamento só com os minutos {KS['todos']}"); continue
        chave10 = set(zip(pd.to_numeric(frames["sujo"][min(KS["todos"])]["gameid"]).astype("int64"), _lado(frames["sujo"][min(KS["todos"])]["side"])))
        fora_ = [k_ not in chave10 for k_ in zip(d_["gameid"], _lado(d_["side"]))]
        if any(fora_): log(f"zz{t}: {sum(fora_)} linhas sem par no zz{min(KS['todos'])} descartadas (só odds)"); d_ = d_[~np.array(fora_)]
        extras[t] = d_
    ODDS = montar_odds({t: frames["sujo"][t] for t in KS["todos"]}, extras)
    ODDS["longo"].to_csv(os.path.join(OUT, "odds_por_minuto.csv.gz"), index=False)
    log(f"ODDS para CLV/markout: minutos {list(ODDS['min'])} ({'+ só odds: ' + str(sorted(extras)) if extras else 'sem zz40/zz45'}) · {len(ODDS['idx'])} (jogo, lado) · jogos < {L0}")
    DATAS = None; f_d = frames["sujo"][min(KS["todos"])]
    for c_ in ([COLUNA_DATA] if COLUNA_DATA else [c for c in f_d.columns if str(c).strip().lower() in ("data", "date", "datetime", "dia", "data_jogo")]):
        if c_ in f_d.columns:
            d_ = pd.to_datetime(f_d[c_], errors="coerce", dayfirst=True)
            if d_.notna().mean() > 0.95: DATAS = pd.Series(d_.values, index=pd.to_numeric(f_d["gameid"]).astype("int64").values).groupby(level=0).first(); break
    log(f"janelas rolantes: " + (f"por DATA (coluna {c_}): {DIAS_ROLL_CURTA} e {DIAS_ROLL_LONGA} dias" if DATAS is not None else
                                 f"SEM coluna de data nos zz → em JOGOS executáveis: {JANELA_ROLL_CURTA} e {JANELA_ROLL_LONGA} (ajuste JANELA_ROLL_* no topo)"))
    # ---------------- Brier / log loss de várias formas ----------------
    log("métricas de Brier / log loss (todas, flag=1, flag=0, entrou/não entrou, lado apostado)…")
    MT = metricas_previsao(GMf, apostas_de, modelos, flag_ini, start_mod, P)
    MT.to_csv(os.path.join(OUT, "metricas_previsao_por_minuto.csv.gz"), index=False)
    MT[MT["t"].astype(str) == "TODOS"].to_csv(os.path.join(OUT, "metricas_previsao_resumo.csv"), index=False)
    # ---------------- opções, decisões ----------------
    opcoes = []
    for c in modelos:
        for F in [None] + FLAGS:
            st = max(start_mod[c], flag_ini[F] if F else 0, hist_ini)
            if st < L0: opcoes.append(dict(nome=c if F is None else f"{c}|{F}", cand=c, flag=F, w=int(st)))
    opcoes = sorted(opcoes, key=lambda o: o["nome"]); nomes = [o["nome"] for o in opcoes]
    log(f"{len(opcoes)} opções (modelo × flag) · {sum(o['w'] < G_CLEAN for o in opcoes)} existem na fase suja · as outras entram entre {G_CLEAN} e {max(o['w'] for o in opcoes)}")
    cortes = _cortes(BL["sujo"]); cortes = cortes[cortes >= hist_ini]
    if PASSO_ESCOLHA != PASSO: cortes = np.array(sorted({int(c) for c in cortes if (c - cortes[0]) % PASSO_ESCOLHA == 0} | {int(cortes[-1])}), dtype=np.int64)
    cad2 = f"a_cada_{PASSO_ESCOLHA}"
    SEG = {cad2: [(int(a), int(b)) for a, b in zip(cortes[:-1], cortes[1:])], "bloco_motor": [(int(g0), int(g1)) for (_, g0, g1) in BL["sujo"]]}
    a_prox = int(cortes[-1])                                                                 # v4.8: PRÓXIMA decisão (fim do último par, <= L0): só com o histórico < a_prox
    a_all = np.array(sorted({a for s in SEG.values() for a, _ in s} | {a_prox}), dtype=np.int64); pos = {int(a): i for i, a in enumerate(a_all)}
    assert a_prox <= L0 and a_prox not in {a for s in SEG.values() for a, _ in s}, "próxima decisão inválida"
    a_hist = a_all - int(EMBARGO_ESCOLHA)                                                    # fim (exclusivo) do histórico de cada decisão
    assert all(b <= L0 for sg in SEG.values() for _, b in sg) and np.all(a_hist <= a_all), "AUDITORIA: segmento além do lockbox ou histórico depois do início do par — ABORTADO"
    El = elegiveis(G, opcoes, a_hist, min_hist); rng = np.random.default_rng(int(BS.CFG["SEED"]))
    linhas, pareados, trilhas, conf, comp, mcs_rows, proximas = [], [], [], [], [], [], []; jogos_comuns, dist_rows, momini, lucro_jogo, livro = [], [], [], [], []
    SER, ESC, BETS_MIN, JSM = {}, {}, {}, {}                                                 # v4.7: séries por jogo de cada funil e escolhas por minuto (ensemble)
    clv_rows, sombra_rows, sombra_ev, monitor_rows, amostra_rows, rolling_rows = [], [], [], [], [], []   # v5.0
    os.makedirs(os.path.join(OUT, "graficos"), exist_ok=True)
    for fu in FUNIS + FUNIS_MIN:
        log(f"funil {fu['nome']}: somas por jogo de {len(opcoes)} opções…"); D = dados_funil(G, GMf, apostas_de, opcoes, fu); memo.clear()
        AC = acumulados_sombra(D) if (CRITERIOS_SOMBRA and ("minuto" not in fu or SOMBRA_NOS_MINUTOS)) else None; cal = {}   # v5.0
        BEN = benchmarks(info, G, fu, BS.CFG, D, nomes)
        oj, gj = np.nonzero(D["E"][:, :, 0])                                                     # (opção, jogo) com aposta
        lucro_jogo.append(pd.DataFrame(dict(funil=fu["nome"], opcao=np.array(nomes, dtype=object)[oj], gameid=G[gj], lucro=D["Lg"][oj, gj], n_apostas=D["E"][oj, gj, 0].astype(int))))
        for jan in JANELAS:
          for conj, filtro_conj in CONJUNTOS.items():
            mask_c = np.array([bool(filtro_conj(op)) for op in opcoes]); Elc = El & mask_c[None, :]; tag = f"{fu['nome']}/{jan}/{conj}"
            if not Elc.any(): log(f"{tag}: nenhuma opção neste conjunto — pulado"); continue
            Wd = janela_dec(opcoes, Elc, jan); Mx = matriz_metricas(D, opcoes, a_hist, Wd)
            # -- conferência 1: conta rápida == conta direta (todas as métricas)
            com_el = np.where(Elc.any(axis=1))[0]; com_el = com_el[a_all[com_el] != a_prox]; amostra = rng.choice(com_el, size=min(CONFERENCIA_N, len(com_el)), replace=False); dif_max = 0.0
            for d in amostra:
                for j in np.where(Elc[d])[0]:
                    vd = valores_direto(BS, D, j, a_hist[d], Wd[d, j])
                    for s_ in METRICAS:
                        x, y_ = Mx[s_][d, j], vd[s_]
                        if np.isfinite(x) or np.isfinite(y_):
                            dif_max = max(dif_max, abs(x - y_) / max(1.0, abs(y_)) if (np.isfinite(x) and np.isfinite(y_)) else np.inf)
            if dif_max > 1e-7: raise AssertionError(f"{tag}: conta rápida ≠ conta direta (dif relativa {dif_max:.2e}) — nada foi gravado")
            # -- conferência 2: o futuro trocado por ruído não muda a escolha de NENHUMA regra
            esc_dif = 0; idx_base = {nome_regra(r): aplicar_regra(r, Mx, Elc) for r in regras}
            segs2 = SEG[cad2]; ii2s = np.array([pos[a] for a, _ in segs2]); a2s = np.array([a for a, _ in segs2], dtype=np.int64); f2s = a_hist[ii2s]
            somb = {}                                                                           # v5.0: MODO SOMBRA de cada regra (e critério)
            if AC is not None:
                for r in regras:
                    for cr in CRITERIOS_SOMBRA: somb[(nome_regra(r), cr)] = sombra(idx_base[nome_regra(r)][ii2s], a2s, f2s, G, AC, cr, N_SOMBRA, MIN_AMOSTRA_SOMBRA)
            for d in amostra[:SABOTAGEM_N]:
                a_s = a_all[d]; fut = G >= a_hist[d]; nf = int(fut.sum())
                D2 = dict(D, T=D["T"].copy(), E=D["E"].copy(), Lg=D["Lg"].copy(), L2=D["L2"].copy(), N2=D["N2"].copy(), COV=D["COV"].copy(), bets=[dict(b) for b in D["bets"]],
                          CS=D["CS"].copy(), CN=D["CN"].copy(), CB=D["CB"].copy())
                D2["CS"][:, fut] = rng.normal(0, 5, size=(len(opcoes), nf)); D2["CN"][:, fut] = rng.integers(0, 4, size=(len(opcoes), nf)); D2["CB"][:, fut] = rng.normal(0, 5, size=(len(opcoes), nf))
                D2["L2"][:, fut] = rng.random((len(opcoes), nf)) * 9; D2["N2"][:, fut] = rng.random((len(opcoes), nf)) * 9     # REVISÃO: todos os canais
                D2["T"][:, fut, 0] = rng.integers(0, 9, size=(len(opcoes), nf)); D2["E"][:, fut, 0] = rng.integers(0, 9, size=(len(opcoes), nf))
                D2["T"][:, fut, 1:] = rng.normal(0, 5, size=D2["T"][:, fut, 1:].shape); D2["E"][:, fut, 1:] = rng.normal(0, 5, size=D2["E"][:, fut, 1:].shape)
                D2["Lg"][:, fut] = rng.normal(0, 5, size=(len(opcoes), nf)); D2["COV"][:, fut] = rng.random((len(opcoes), nf)) < 0.5
                for b in D2["bets"]:
                    mf = b["g"] >= a_hist[d]; b["L"] = b["L"].copy(); b["L"][mf] = rng.normal(0, 5, size=int(mf.sum()))
                    b["clv"] = b["clv"].copy(); b["clv"][mf] = rng.normal(0, 5, size=int(mf.sum())); b["mkf"] = b["mkf"].copy(); b["mkf"][mf] = rng.normal(0, 5, size=int(mf.sum()))
                M2 = matriz_metricas(D2, opcoes, a_hist, Wd); ate = a_all <= a_s
                AC2 = acumulados_sombra(D2) if AC is not None else None; ate2 = a2s <= a_s
                for r in regras:
                    i2 = aplicar_regra(r, M2, Elc)
                    if not np.array_equal(i2[ate], idx_base[nome_regra(r)][ate]): esc_dif += 1
                    for cr in (CRITERIOS_SOMBRA if AC2 is not None else []):                    # v5.0: a SOMBRA também não pode mudar com o futuro trocado
                        if not np.array_equal(sombra(i2[ii2s], a2s, f2s, G, AC2, cr, N_SOMBRA, MIN_AMOSTRA_SOMBRA)[0][ate2], somb[(nome_regra(r), cr)][0][ate2]): esc_dif += 1
            if esc_dif: raise AssertionError(f"{tag}: escolha MUDOU com o futuro trocado por ruído ({esc_dif} casos) — VAZAMENTO; nada foi gravado")
            # -- v4.8: ÚLTIMAS escolhas (pares a cada 2) e a PRÓXIMA (para o jogo a_prox em diante, histórico < a_prox − embargo; nada do lockbox)
            ip = pos[a_prox]; segs2 = SEG[cad2]
            for r in regras:
                nr_ = nome_regra(r); idx_r = idx_base[nr_]; ordem = r if isinstance(r, str) else r.get("ordenar")
                k = int(idx_r[ip]); d_ = dict(funil=fu["nome"], janela=jan, conjunto=conj, regra=nr_, decisao_para_jogo=a_prox, historico_ate_jogo=int(a_hist[ip]) - 1,
                                              opcoes_elegiveis=int(Elc[ip].sum()), proxima_escolha=nomes[k] if k >= 0 else ("NAO_APOSTA" if k == -2 else None),
                                              valor_metrica=float(Mx[ordem][ip, k]) if (k >= 0 and ordem) else np.nan, ppg_historico=float(Mx["ppg"][ip, k]) if k >= 0 else np.nan,
                                              n_apostas_historico=float(Mx["n_apostas"][ip, k]) if k >= 0 else np.nan)
                if ordem:                                                                       # 2ª e 3ª colocadas pela mesma métrica (só elegíveis com valor)
                    X_ = np.where(Elc[ip] & np.isfinite(Mx[ordem][ip]), Mx[ordem][ip] * SENTIDO[ordem], -np.inf); o_ = [j for j in np.argsort(-X_, kind="stable") if np.isfinite(X_[j]) and j != k][:2]
                    for n_, j in enumerate(o_, 2): d_[f"opcao_{n_}"] = nomes[j]; d_[f"valor_{n_}"] = float(Mx[ordem][ip, j])
                ult = [(a, idx_r[pos[a]]) for a, _ in segs2[-10:]]
                d_["ultimas_10_escolhas"] = " · ".join(f"{a}:{nomes[k_] if k_ >= 0 else ('NAO_APOSTA' if k_ == -2 else '-')}" for a, k_ in ult)
                d_["ultima_escolha"] = nomes[ult[-1][1]] if ult and ult[-1][1] >= 0 else None
                proximas.append(d_)
            conf.append(dict(funil=fu["nome"], janela=jan, conjunto=conj, decisoes_conferidas=len(amostra), dif_relativa_max=dif_max, sabotagens=min(SABOTAGEM_N, len(amostra)), escolhas_alteradas=esc_dif))
            log(f"{tag}: {int(mask_c.sum())} opções · conta rápida × direta em {len(amostra)} decisões (dif máx {dif_max:.1e}) · sabotagem do futuro em {min(SABOTAGEM_N, len(amostra))}: 0 escolhas alteradas")
            # -- procedimento de cada regra, nas 2 cadências
            series = {}
            for r in regras:
                nr_ = nome_regra(r); idx_all = idx_base[nr_]; ordem = r if isinstance(r, str) else r.get("ordenar")
                for cad, segs in SEG.items():
                    ii = np.array([pos[a] for a, _ in segs]); lu, nb, esc_g, seg_g = serie_procedimento(G, D, segs, idx_all[ii]); series[(nr_, cad)] = (lu, nb, esc_g, seg_g, ii)
                    ia_ = np.searchsorted(G, [a for a, _ in segs], "left"); ib_ = np.searchsorted(G, [b for _, b in segs], "left")
                    for (a, b), d, i0, i1 in zip(segs, ii, ia_, ib_):
                        k = idx_all[d]
                        trilhas.append((fu["nome"], jan, conj, nr_, cad, a, b, int(Elc[d].sum()), nomes[k] if k >= 0 else ("NAO_APOSTA" if k == -2 else None),
                                        float(Mx[ordem][d, k]) if (k >= 0 and ordem) else np.nan, float(Mx["ppg"][d, k]) if k >= 0 else np.nan,
                                        int(i1 - i0), float(D["Lg"][k, i0:i1].sum()) if k >= 0 else 0.0, int(D["E"][k, i0:i1, 0].sum()) if k >= 0 else 0))
            # AUDITORIA 02/10: antes = INTERSEÇÃO dos jogos avaliados por todas as regras (acrescentar uma regra mudava o PPG das outras). Agora o conjunto
            # é FIXO: jogos dos pares (a cada PASSO_ESCOLHA) com >= 1 opção elegível neste conjunto; regra que ainda não consegue decidir ali = sem aposta (0)
            ii2_ = np.array([pos[a] for a, _ in SEG[cad2]]); ini2_ = np.array([a for a, _ in SEG[cad2]]); fim2_ = np.array([b for _, b in SEG[cad2]])
            sg_ = np.searchsorted(ini2_, G, side="right") - 1; dentro_ = (sg_ >= 0) & (G < fim2_[np.clip(sg_, 0, None)])
            js = G[dentro_ & Elc[ii2_[np.clip(sg_, 0, None)]].any(axis=1)]; gi = np.searchsorted(G, js)
            jogos_comuns.append(pd.DataFrame(dict(funil=fu["nome"], janela=jan, conjunto=conj, gameid=js)))
            # referências: acaso (média), fixo em retrospecto, benchmarks e modelos fixos (sem flag)
            ii2 = np.array([pos[a] for a, _ in SEG[cad2]]); ini2 = np.array([a for a, _ in SEG[cad2]]); sgi = np.clip(np.searchsorted(ini2, js, side="right") - 1, 0, None)
            Elg = Elc[ii2[sgi]]; ne = np.maximum(Elg.sum(1), 1); a_par_js = ini2[sgi]             # v5.0: início do par de cada jogo avaliado
            refs = {"ACASO (média das elegíveis)": (pd.Series(np.where(Elg, D["Lg"][:, gi].T, 0.0).sum(1) / ne, index=js), pd.Series(np.where(Elg, D["E"][:, gi, 0].T, 0.0).sum(1) / ne, index=js))}
            ref_j = {}
            sempre = np.where(Elg.all(axis=0))[0]
            if len(sempre):
                jb = sempre[np.argmax(D["Lg"][sempre][:, gi].sum(1))]; nm = f"FIXO_RETROSPECTO ({nomes[jb]}) — ENVIESADO"
                refs[nm] = (pd.Series(D["Lg"][jb, gi], index=js), pd.Series(D["E"][jb, gi, 0], index=js)); ref_j[nm] = jb
            for bn, (lg_, nb_, j_) in BEN.items():
                if j_ is not None and not mask_c[j_] and not bn.startswith("BASE_"): continue          # modelo fixo fora do conjunto (ex.: MOM no "so_INI")
                refs[bn] = (pd.Series(lg_[gi], index=js), pd.Series(nb_[gi], index=js))
                if j_ is not None: ref_j[bn] = j_
            # -- DISTRIBUIÇÃO DO ACASO: N_ACASO sequências que, a cada par, sorteiam UMA opção elegível (mesma cadência, mesmas opções)
            rg_a = np.random.default_rng([int(BS.CFG["SEED"]), zlib.crc32(tag.encode())])
            El2 = Elc[ii2]; n_el = El2.sum(1); ordem_el = np.argsort(~El2, axis=1, kind="stable")
            sg_js = sgi; acaso_ppg = np.zeros(N_ACASO); acaso_dd = np.zeros(N_ACASO); acaso_sh = np.zeros(N_ACASO)
            for b0 in range(0, N_ACASO, 250):
                nb_ = min(250, N_ACASO - b0); r_ = np.floor(rg_a.random((nb_, len(ii2))) * np.maximum(n_el, 1)).astype(np.int64)
                opt = ordem_el[np.arange(len(ii2))[None, :], r_]                                   # nb_ × pares
                Lsim = D["Lg"][opt[:, sg_js], gi[None, :]]                                        # nb_ × jogos
                acaso_ppg[b0:b0 + nb_] = Lsim.mean(1); c_ = np.cumsum(Lsim, axis=1)
                acaso_dd[b0:b0 + nb_] = np.max(np.maximum.accumulate(np.concatenate([np.zeros((nb_, 1)), c_], axis=1), axis=1)[:, 1:] - c_, axis=1)
                sd_ = Lsim.std(axis=1, ddof=1); acaso_sh[b0:b0 + nb_] = np.where(sd_ > 0, Lsim.mean(1) / np.where(sd_ > 0, sd_, 1), np.nan)
            dist_rows.append(dict(funil=fu["nome"], janela=jan, conjunto=conj, n_sorteios=N_ACASO, n_jogos=len(js), ppg_media=acaso_ppg.mean(), ppg_dp=acaso_ppg.std(ddof=1),
                                  ppg_p05=np.quantile(acaso_ppg, .05), ppg_p50=np.quantile(acaso_ppg, .5), ppg_p95=np.quantile(acaso_ppg, .95),
                                  maxdd_p05=np.quantile(acaso_dd, .05), maxdd_p50=np.quantile(acaso_dd, .5), maxdd_p95=np.quantile(acaso_dd, .95),
                                  sharpe_jogo_p50=np.nanquantile(acaso_sh, .5), sharpe_jogo_p95=np.nanquantile(acaso_sh, .95)))
            MON = {}
            for key, val in list(series.items()) + [((rn, "-"), sv + (None, None, None)) for rn, sv in refs.items()]:
                (rn, cad), (lu, nb, esc_g, seg_g, ii) = key, val
                lc, nc = lu.reindex(js).fillna(0.0), nb.reindex(js).fillna(0.0); r_ = _bt(BS, lc)
                per = {}
                for (pn, pa, pb) in PX:
                    m_, se_ = _media_se(lc[(js >= pa) & (js < pb)]); per[f"ppg_{pn}"] = m_; per[f"se_{pn}"] = se_
                extra = dict(risco_serie(lc, nc))
                if esc_g is not None: extra.update(_brier_proc(D, esc_g[gi], gi)); extra.update(risco_por_aposta(D, esc_g[gi], js))
                elif rn in ref_j: extra.update(_brier_proc(D, np.full(len(gi), ref_j[rn]), gi)); extra.update(risco_por_aposta(D, np.full(len(gi), ref_j[rn]), js))
                A_ = None                                                                       # v5.0: estatísticas POR APOSTA (ROI × esperado, variância, CLV, markout)
                if (esc_g is not None and cad == cad2) or (esc_g is None and rn in ref_j):
                    ks_ = esc_g[gi] if esc_g is not None else np.full(len(gi), ref_j[rn])
                    A_ = apostas_serie(D, ks_, js, a_par_js, cal); sa_, hz_ = stats_apostas(A_, f"{tag}|{rn}|{cad}")
                    extra.update(sa_); clv_rows.extend(dict(funil=fu["nome"], janela=jan, conjunto=conj, regra=rn, cadencia=cad, **h_) for h_ in hz_)
                    if A_ is not None:                                                          # REVISÃO: métricas novas também POR PERÍODO (inclui o antigo lockbox)
                        for (pn, pa, pb) in PX:
                            mp_ = (A_["g"] >= pa) & (A_["g"] < pb); tc_ = mp_ & np.isfinite(A_["clv"])
                            extra.update({f"n_apostas_{pn}": int(mp_.sum()), f"roi_apostas_{pn}": float(A_["L"][mp_].mean()) if mp_.any() else np.nan,
                                          f"ev_medio_{pn}": float(A_["ev"][mp_].mean()) if mp_.any() else np.nan, f"clv_fech_{pn}": float(A_["clv"][tc_].mean()) if tc_.any() else np.nan,
                                          f"mk_fech_{pn}": float(A_["mk_fech"][tc_].mean()) if tc_.any() else np.nan})
                    if esc_g is not None or rn.startswith("BASE_"): MON[rn] = (lc, nc, A_, ks_)
                    if esc_g is not None and (jan, conj) in SALVAR_APOSTAS and A_ is not None:  # LIVRO DE APOSTAS do procedimento desta regra
                        livro.append(pd.DataFrame(dict(funil=fu["nome"], janela=jan, conjunto=conj, regra=rn, par_a=A_["a_par"], gameid=A_["g"], t=A_["t"], side=A_["side"],
                                                       opcao=np.array(nomes, dtype=object)[A_["k"]], p=A_["p"], q_devig=A_["q"], odd=A_["odd"], edge=A_["edge"], y=A_["y"], lucro=A_["L"],
                                                       ev=A_["ev"], ev_cal=A_["ev_cal"], calib_n=A_["calib_n"], calib_ate_jogo=A_["calib_ate_jogo"], clv_fech=A_["clv"], supera_fech=A_["supera"],
                                                       q_fech=A_["q_fech"], odd_fech=A_["odd_fech"], minuto_fech=A_["minuto_fech"], **{f"q_mais{h}": A_[f"q{h}"] for h in HORIZONTES_CLV})))
                pp = float(lc.mean())
                extra.update(percentil_no_acaso=float(np.mean(acaso_ppg < pp) + 0.5 * np.mean(acaso_ppg == pp)), p_vs_acaso=float((np.sum(acaso_ppg >= pp) + 1) / (N_ACASO + 1)),
                             percentil_maxdd_no_acaso=float(np.mean(acaso_dd > extra.get("maxdd", np.nan))) if "maxdd" in extra else np.nan)
                if esc_g is not None:
                    ev = esc_g[gi]; evo = ev[ev >= 0]
                    extra.update(overfit_serie(lu, esc_g, seg_g, G, js, Mx["ppg"], ii, D, Elc, pos, BL["sujo"]))
                    extra.update(trocas=int((ev[1:] != ev[:-1]).sum()), opcoes_distintas=int(len(np.unique(evo))), pct_jogos_sem_aposta_por_regra=float(np.mean(ev == -2)),
                                 pct_jogos_sem_decisao=float(np.mean(ev == -1)),
                                 pct_jogos_com_flag=float(np.mean([opcoes[k]["flag"] is not None for k in evo])) if len(evo) else np.nan,
                                 mais_escolhidas=" · ".join(f"{nomes[k]} {v:.0%}" for k, v in pd.Series(evo).value_counts(normalize=True).head(3).items()))
                    tl = [(D["bets"][k]["t"][mk], D["bets"][k]["L"][mk]) for k in np.unique(evo) for mk in [np.isin(D["bets"][k]["g"], js[ev == k])]]
                    tt = pd.Series(np.concatenate([a_ for a_, _ in tl]) if tl else []).value_counts().sort_index()
                    extra["apostas_por_minuto"] = " · ".join(f"t{int(t)} {n}" for t, n in tt.items())
                    lm_ = pd.Series(np.concatenate([b_ for _, b_ in tl]) if tl else [], index=np.concatenate([a_ for a_, _ in tl]) if tl else []).groupby(level=0).sum()
                    extra["lucro_por_minuto"] = " · ".join(f"t{int(t)} {v:+.1f}" for t, v in lm_.items())   # v4.7: de que minuto veio o lucro
                linhas.append(dict(funil=fu["nome"], politica=fu["politica"], janela=jan, conjunto=conj, regra=rn, cadencia=cad, n_jogos=r_["n"], ppg=r_["media"], ic_lo=r_["ic_lo"],
                                   ic_hi=r_["ic_hi"], p_unilateral=r_["p_valor"], **per, **extra))
            # -- comparação entre regras (cadência a cada 2, mesmos jogos): par a par, contra cada referência, MCS e Reality Check contra o acaso
            Lr = pd.DataFrame({rn: series[(rn, cad2)][0].reindex(js).fillna(0.0).values for rn in nomes_r}, index=js); acaso = refs["ACASO (média das elegíveis)"][0].values
            for r1 in nomes_r:
                for r2 in nomes_r:
                    if r1 >= r2: continue
                    m_, se_ = _media_se(Lr[r1] - Lr[r2]); comp.append(dict(funil=fu["nome"], janela=jan, conjunto=conj, regra_1=r1, regra_2=r2, n_jogos=len(js), delta_ppg=m_, se=se_, z=m_ / se_ if se_ and se_ > 0 else np.nan))
                for rn_, (sr_, _nb) in refs.items():
                    m_, se_ = _media_se(Lr[r1] - sr_.reindex(js)); comp.append(dict(funil=fu["nome"], janela=jan, conjunto=conj, regra_1=r1, regra_2=rn_.split(" ")[0], n_jogos=len(js), delta_ppg=m_, se=se_, z=m_ / se_ if se_ and se_ > 0 else np.nan))
            cl = _clu(js)
            try:
                vivos, _ = mcs(-Lr, cl, alpha=ALPHA_MCS, B=B_MCS)
            except Exception as e_: vivos = []; log(f"MCS não calculado ({type(e_).__name__}: {e_})")
            try:
                rc, _ = nula_empirica_max(Lr.sub(acaso, axis=0), cl, B=B_RC, escopo=f"{tag}: PPG de cada regra − escolha ao acaso")
            except Exception as e_: rc = {}; log(f"Reality Check não calculado ({type(e_).__name__}: {e_})")
            mcs_rows.append(dict(funil=fu["nome"], janela=jan, conjunto=conj, n_jogos=len(js), n_regras=len(nomes_r), mcs_alpha=ALPHA_MCS, mcs_regras=" · ".join(vivos),
                                 rc_melhor=rc.get("melhor"), rc_p_reality_check=rc.get("p_reality_check"), rc_p_spa=rc.get("p_SPA_c_aprox")))
            for rn in nomes_r: linhas[[i for i, x in enumerate(linhas) if x["funil"] == fu["nome"] and x["janela"] == jan and x["conjunto"] == conj and x["regra"] == rn and x["cadencia"] == cad2][0]]["no_MCS"] = rn in vivos
            for rn in nomes_r:
                rd = _bt(BS, series[(rn, cad2)][0].reindex(js).fillna(0.0) - series[(rn, "bloco_motor")][0].reindex(js).fillna(0.0))
                pareados.append(dict(funil=fu["nome"], janela=jan, conjunto=conj, regra=rn, n_jogos=len(js), delta_ppg_2_menos_bloco=rd["media"], ic_lo=rd["ic_lo"], ic_hi=rd["ic_hi"], p_unilateral_2_melhor=rd["p_valor"]))
            # -- MOM × INI: o mesmo modelo fixo com e sem a odd do momento (mesmos jogos)
            for bn in refs:
                if bn.startswith("FIXO_") and "_MOM" in bn and not bn.startswith("FIXO_RETRO"):
                    bi = bn.replace("_MOM", "_INI")
                    if bi in refs:
                        m_, se_ = _media_se(refs[bn][0].reindex(js) - refs[bi][0].reindex(js))
                        b1, b2 = _brier_proc(D, np.full(len(gi), ref_j[bn]), gi), _brier_proc(D, np.full(len(gi), ref_j[bi]), gi)
                        momini.append(dict(funil=fu["nome"], janela=jan, conjunto=conj, modelo_MOM=bn.split(" ")[0], modelo_INI=bi.split(" ")[0], n_jogos=len(js),
                                           ppg_MOM=float(refs[bn][0].reindex(js).mean()), ppg_INI=float(refs[bi][0].reindex(js).mean()), delta_ppg_MOM_menos_INI=m_, se=se_,
                                           z=m_ / se_ if se_ and se_ > 0 else np.nan, bsskill_MOM=b1.get("bsskill_escolhida"), bsskill_INI=b2.get("bsskill_escolhida")))
            # -- v5.0: MODO SOMBRA (cada regra × critério) nos mesmos jogos
            for (rn, cr), (prod_p, evs) in somb.items():
                lu_s, nb_s, esc_s, _seg_s = serie_procedimento(G, D, segs2, prod_p)
                lc_s, nc_s = lu_s.reindex(js).fillna(0.0), nb_s.reindex(js).fillna(0.0); base_r = series[(rn, cad2)][0].reindex(js).fillna(0.0)
                r_s = _bt(BS, lc_s); m_, se_ = _media_se(lc_s - base_r); ks_s = esc_s[gi]
                sa_s, _ = stats_apostas(apostas_serie(D, ks_s, js, a_par_js, cal), f"{tag}|{rn}|sombra_{cr}"); rk_s = risco_serie(lc_s, nc_s)
                sombra_rows.append(dict(funil=fu["nome"], politica=fu["politica"], janela=jan, conjunto=conj, regra=rn, criterio=cr, n_sombra=N_SOMBRA, n_jogos=len(js),
                                        ppg=r_s["media"], ic_lo=r_s["ic_lo"], ic_hi=r_s["ic_hi"], ppg_sem_sombra=float(base_r.mean()), delta_vs_sem_sombra=m_, se=se_,
                                        z=m_ / se_ if se_ and se_ > 0 else np.nan, trocas_producao=int((ks_s[1:] != ks_s[:-1]).sum()),
                                        trocas_sem_sombra=int((series[(rn, cad2)][2][gi][1:] != series[(rn, cad2)][2][gi][:-1]).sum()),
                                        promocoes=sum(e["evento"] == "promovida" for e in evs), rejeicoes=sum(e["evento"] == "rejeitada" for e in evs),
                                        indeterminadas=sum(e["evento"] == "indeterminada" for e in evs),
                                        pct_pares_na_1a_escolha=float(np.mean(prod_p[prod_p >= 0] == prod_p[prod_p >= 0][0])) if (prod_p >= 0).any() else np.nan,
                                        maxdd=rk_s.get("maxdd"), roi=sa_s.get("roi_apostas"), clv_fech_medio=sa_s.get("clv_fech_medio"), supera_fech_taxa=sa_s.get("supera_fech_taxa"),
                                        n_apostas=sa_s.get("n_apostas_serie")))
                if (jan, conj) in SALVAR_APOSTAS:
                    for e in evs: sombra_ev.append(dict(e, funil=fu["nome"], janela=jan, conjunto=conj, regra=rn, criterio=cr, producao=nomes[e["producao"]],
                                                        sombra=nomes[e["sombra"]] if "sombra" in e else None))
            # -- v5.0: MONITOR (ROI / CLV / Brier skill rolantes e alarmes), amostra de apostas e GRÁFICOS
            x_t, W1, W2, unid = eixo_tempo(js, DATAS); top_ = [x["regra"] for x in sorted([x for x in linhas if x["funil"] == fu["nome"] and x["janela"] == jan and x["conjunto"] == conj
                                                                                            and x["cadencia"] == cad2], key=lambda x: -np.nan_to_num(x["ppg"], nan=-9))][:TOP_GRAFICOS]
            graf_full, graf_ult = {}, {}
            for rn_, (lc_, nc_, A_, ks_) in MON.items():
                L_g, n_g = lc_.values.astype(float), nc_.values.astype(float); cum = np.cumsum(L_g); dd = cum - np.maximum.accumulate(np.r_[0.0, cum])[1:]
                roi1, roi2 = rolling_janela(x_t, L_g, n_g, W1), rolling_janela(x_t, L_g, n_g, W2)
                okk = ks_ >= 0; ksg = np.maximum(ks_, 0)
                sk_g = np.where(okk, D["T"][ksg, gi, 4] - D["T"][ksg, gi, 3], 0.0); nl_g = np.where(okk, D["T"][ksg, gi, 0], 0.0); bss1 = rolling_janela(x_t, sk_g, nl_g, W1)
                cs_g, cn_g, cb_g, cm_g = np.zeros(len(js)), np.zeros(len(js)), np.zeros(len(js)), np.zeros(len(js))
                if A_ is not None:
                    tm = np.isfinite(A_["clv"]); gp = np.searchsorted(js, A_["g"][tm])
                    cs_g, cn_g, cb_g, cm_g = (np.bincount(gp, weights=w_, minlength=len(js)) for w_ in (A_["clv"][tm], np.ones(int(tm.sum())), A_["supera"][tm], A_["mk_fech"][tm]))
                sup1, clv1, mk1 = rolling_janela(x_t, cb_g, cn_g, W1), rolling_janela(x_t, cs_g, cn_g, W1), rolling_janela(x_t, cm_g, cn_g, W1)
                pct_ = lambda v, f: float(np.mean(f(v[np.isfinite(v)]))) if np.isfinite(v).any() else np.nan     # REVISÃO: só pontos com janela cheia
                ult = lambda v: float(v[np.isfinite(v)][-1]) if np.isfinite(v).any() else np.nan
                monitor_rows.append(dict(funil=fu["nome"], janela=jan, conjunto=conj, regra=rn_, unidade_janela=unid, janela_curta=W1, janela_longa=W2,
                                         roi_curta_atual=ult(roi1), roi_longa_atual=ult(roi2), supera_fech_curta_atual=ult(sup1), clv_curta_atual=ult(clv1), mk_fech_curta_atual=ult(mk1),
                                         bsskill_curta_atual=ult(bss1), roi_curta_min=float(np.nanmin(roi1)) if np.isfinite(roi1).any() else np.nan,
                                         pct_tempo_roi_curta_negativo=pct_(roi1, lambda v: v < 0), pct_tempo_mk_fech_curta_negativo=pct_(mk1, lambda v: v < 0),
                                         pct_tempo_bsskill_curta_negativo=pct_(bss1, lambda v: v < 0),
                                         drawdown_atual=float(dd[-1]) if len(dd) else np.nan, maxdd=float(-dd.min()) if len(dd) else np.nan,
                                         ALARME_ROI=bool(ult(roi1) < 0), ALARME_CLV=bool(ult(mk1) < 0), ALARME_BRIER=bool(ult(bss1) < 0)))
                if rn_ in top_ or rn_.startswith("BASE_"):
                    graf_full[rn_] = (x_t, cum, dd, roi1); ult_ = x_t >= x_t[-1] - W2 if len(x_t) else x_t > 0
                    graf_ult[rn_] = (x_t[ult_], cum[ult_] - (cum[ult_][0] - L_g[ult_][0] if ult_.any() else 0), roi1[ult_])
                    if "minuto" not in fu:
                        rolling_rows.append(pd.DataFrame(dict(funil=fu["nome"], janela=jan, conjunto=conj, regra=rn_, gameid=js, lucro_acumulado=cum, drawdown=dd,
                                                              roi_rolante_curta=roi1, roi_rolante_longa=roi2, supera_fech_rolante_curta=sup1, bsskill_rolante_curta=bss1)))
                    if rn_ in top_[:3] and A_ is not None and (jan, conj) in SALVAR_APOSTAS and "minuto" not in fu:     # AMOSTRA: 20 apostas sorteadas + as 10 últimas
                        n_a = len(A_["L"]); ra = np.random.default_rng([int(BS.CFG["SEED"]), zlib.crc32(f"{tag}|{rn_}".encode())])
                        sel = np.unique(np.r_[ra.choice(n_a, size=min(20, n_a), replace=False), np.arange(max(0, n_a - 10), n_a)])
                        amostra_rows.append(pd.DataFrame({**dict(funil=fu["nome"], janela=jan, conjunto=conj, regra=rn_, tipo=np.where(sel >= n_a - 10, "ultimas_10", "sorteada")),
                                                          **{c_: A_[c_][sel] for c_ in ("a_par", "g", "t", "side", "odd", "p", "q", "ev", "ev_cal", "calib_n", "clv", "supera", "minuto_fech", "y", "L")},
                                                          "opcao": np.array(nomes, dtype=object)[A_["k"][sel]]}))
            if "minuto" not in fu and graf_full:
                nm_g = re.sub(r"[^A-Za-z0-9_]+", "_", f"{fu['nome']}_{jan}_{conj}")
                legenda = {rn_: next((f"{rn_} PPG {x['ppg']:+.3f} [{x['ic_lo']:+.3f};{x['ic_hi']:+.3f}] ROI {x.get('roi_apostas', np.nan):+.3f}" for x in linhas
                                      if x["funil"] == fu["nome"] and x["janela"] == jan and x["conjunto"] == conj and x["regra"] == rn_ and x["cadencia"] in (cad2, "-")), rn_) for rn_ in graf_full}
                gravar_svg(os.path.join(OUT, "graficos", f"{nm_g}.svg"),
                           [({legenda[k]: (v[0], v[1]) for k, v in graf_full.items()}, f"lucro acumulado (u) · eixo = {unid}", 0.0),
                            ({k: (v[0], v[2]) for k, v in graf_full.items()}, "drawdown (u)", 0.0),
                            ({k: (v[0], v[3]) for k, v in graf_full.items()}, f"ROI rolante ({W1} {unid})", 0.0)], f"{tag} — histórico inteiro")
                gravar_svg(os.path.join(OUT, "graficos", f"{nm_g}_ultimos.svg"),
                           [({k: (v[0], v[1]) for k, v in graf_ult.items()}, f"lucro acumulado nos últimos {W2} {unid}", 0.0),
                            ({k: (v[0], v[2]) for k, v in graf_ult.items()}, f"ROI rolante ({W1} {unid}) dentro dos últimos {W2} {unid}", 0.0)], f"{tag} — janela longa final")
            for rn in nomes_r: SER[(fu["nome"], jan, conj, rn)] = series[(rn, cad2)][0].reindex(js).fillna(0.0)          # v4.7
            for bn in refs:
                if bn.startswith("BASE_"): SER[(fu["nome"], jan, conj, "BASE")] = refs[bn][0].reindex(js).fillna(0.0)
            if "minuto" in fu:
                for rn in nomes_r: ESC[(jan, conj, rn, int(fu["minuto"]), fu.get("ev_min"))] = series[(rn, cad2)][2]
                if (jan, conj) in JSM: assert np.array_equal(JSM[(jan, conj)], js), "jogos avaliados diferentes entre os minutos"
                JSM[(jan, conj)] = js
            log(f"{tag}: {len(js)} jogos comuns · acaso PPG {acaso_ppg.mean():+.4f} [{np.quantile(acaso_ppg, .05):+.4f}; {np.quantile(acaso_ppg, .95):+.4f}] · "
                f"MCS {vivos} · Reality Check p={rc.get('p_reality_check')}")
        if "minuto" in fu: BETS_MIN[(int(fu["minuto"]), fu.get("ev_min"))] = D["bets"]           # v4.7: apostas de cada opção NESTE minuto (ensemble)
        del D
    R = pd.DataFrame(linhas); PR = pd.DataFrame(pareados); MC = pd.DataFrame(mcs_rows)
    R.to_csv(os.path.join(OUT, "resumo_funil.csv"), index=False); PR.to_csv(os.path.join(OUT, "pareado_2_vs_bloco.csv"), index=False)
    CP = pd.DataFrame(comp); CP.to_csv(os.path.join(OUT, "comparar_regras.csv"), index=False); MC.to_csv(os.path.join(OUT, "mcs_reality_check.csv"), index=False)
    pd.DataFrame(conf).to_csv(os.path.join(OUT, "conferencia_regras.csv"), index=False)
    pd.DataFrame(dist_rows).to_csv(os.path.join(OUT, "distribuicao_acaso.csv"), index=False); pd.DataFrame(momini).to_csv(os.path.join(OUT, "mom_vs_ini.csv"), index=False)
    pd.DataFrame(trilhas, columns=["funil", "janela", "conjunto", "regra", "cadencia", "a", "b", "opcoes_elegiveis", "escolhida", "valor_ordenacao", "ppg_historico_escolhida",
                                   "jogos_no_par", "lucro_no_par", "apostas_no_par"]).to_csv(os.path.join(OUT, "trilhas.csv.gz"), index=False)
    pd.DataFrame(opcoes).to_csv(os.path.join(OUT, "opcoes.csv"), index=False)
    CLV_T = pd.DataFrame(clv_rows); CLV_T.to_csv(os.path.join(OUT, "clv_markout.csv"), index=False)                 # v5.0
    evc = []                                                                                    # REVISÃO: EV5/EV10 − EV0 PAREADO (mesmos jogos), z por cluster de 10 gameids
    for f0 in [f for f in FUNIS + FUNIS_MIN if f.get("ev_min") is None]:
        for f1 in [f for f in FUNIS + FUNIS_MIN if f.get("ev_min") is not None and f["politica"] == f0["politica"] and f.get("minuto") == f0.get("minuto")
                   and f["tempos"] == f0["tempos"]]:
            for (fn, jan, conj, rn), s0_ in SER.items():
                if fn != f0["nome"] or (f1["nome"], jan, conj, rn) not in SER: continue
                s1_ = SER[(f1["nome"], jan, conj, rn)].reindex(s0_.index).fillna(0.0); m_, se_ = _media_se(s1_ - s0_)
                evc.append(dict(funil_ev0=f0["nome"], funil_ev=f1["nome"], limiar_ev=f1["ev_min"], janela=jan, conjunto=conj, regra=rn, n_jogos=len(s0_),
                                ppg_ev0=float(s0_.mean()), ppg_ev=float(s1_.mean()), delta=m_, se=se_, z=m_ / se_ if se_ and se_ > 0 else np.nan))
    EVC = pd.DataFrame(evc); EVC.to_csv(os.path.join(OUT, "ev_comparacao_pareada.csv"), index=False)
    SB = pd.DataFrame(sombra_rows); SB.to_csv(os.path.join(OUT, "sombra.csv"), index=False)
    pd.DataFrame(sombra_ev).to_csv(os.path.join(OUT, "sombra_eventos.csv.gz"), index=False)
    MO = pd.DataFrame(monitor_rows); MO.to_csv(os.path.join(OUT, "monitor_alarmes.csv"), index=False)
    if amostra_rows: pd.concat(amostra_rows, ignore_index=True).to_csv(os.path.join(OUT, "amostra_apostas.csv"), index=False)
    if rolling_rows: pd.concat(rolling_rows, ignore_index=True).to_csv(os.path.join(OUT, "rolling_top.csv.gz"), index=False)
    PXE = pd.DataFrame(proximas); PXE.to_csv(os.path.join(OUT, "proximas_escolhas.csv"), index=False)              # v4.8
    lg_ = [f"PRÓXIMA ESCOLHA de cada regra para o jogo {a_prox} em diante (histórico até o jogo {a_prox - 1 - int(EMBARGO_ESCOLHA)}; " + ("continuação: todos os jogos dos zz" if CONTINUACAO else "nenhum dado do lockbox") + ") — "
           f"também as 10 últimas escolhas. Tudo em proximas_escolhas.csv (inclui funis por minuto, 2ª/3ª colocadas e valores)."]
    for (fu_, jan_, cj_), x in PXE[PXE.funil.isin([f["nome"] for f in FUNIS])].groupby(["funil", "janela", "conjunto"], sort=False):
        lg_.append(f"=== {fu_} · janela {jan_} · opções {cj_}")
        for _, r_ in x.iterrows():
            lg_.append(f"   {r_.regra:<26} próxima: {str(r_.proxima_escolha):<28} (última: {str(r_.ultima_escolha):<28}) · métrica {r_.valor_metrica:+.5f} · PPG hist {r_.ppg_historico:+.4f} · "
                       f"2ª {r_.get('opcao_2', '-')} · 3ª {r_.get('opcao_3', '-')}")
    open(os.path.join(OUT, "PROXIMAS_ESCOLHAS.txt"), "w", encoding="utf-8").write("\n".join(lg_)); log("\n".join(lg_[:1]))
    if livro:
        LV = pd.concat(livro, ignore_index=True).sort_values(["funil", "janela", "conjunto", "regra", "gameid", "t"], kind="mergesort")
        LV["lucro_acumulado"] = LV.groupby(["funil", "janela", "conjunto", "regra"])["lucro"].cumsum()
        LV.to_csv(os.path.join(OUT, "apostas_procedimento.csv.gz"), index=False)                   # o LIVRO: cada aposta feita por cada regra, na ordem
    pd.DataFrame(dict(gameid=G)).to_csv(os.path.join(OUT, "universo.csv.gz"), index=False)                     # AUDITORIA: denominador do PPG e contagem do histórico
    pd.concat(lucro_jogo, ignore_index=True).to_csv(os.path.join(OUT, "lucro_por_jogo.csv.gz"), index=False)   # AUDITORIA: lucro e nº de apostas de cada opção por jogo
    pd.concat(jogos_comuns, ignore_index=True).to_csv(os.path.join(OUT, "jogos_comuns.csv.gz"), index=False)
    EN = pd.DataFrame()
    if FUNIS_MIN:                                                                                 # v4.7: ENSEMBLE do melhor de cada minuto
        EN, LVE = ensemble_minutos(G, ESC, BETS_MIN, JSM, SER, nomes, nomes_r, PX)
        EN.to_csv(os.path.join(OUT, "ensemble_minutos.csv"), index=False)
        if len(LVE): LVE.to_csv(os.path.join(OUT, "apostas_ensemble.csv.gz"), index=False)
    escrever_resumo(R, PR, MC, MT, P, CP, pd.DataFrame(dist_rows), pd.DataFrame(momini), dict(SB=SB, MO=MO, CLV=CLV_T, EVC=EVC, odds_min=list(ODDS["min"]), datas=DATAS is not None, PX=PX, regime=REGIME_DRAFT, C=C_FIXO, hist_ini=hist_ini, ini_draft=ini_draft, flag_ini=flag_ini, n_opcoes=len(opcoes), n_G=len(G), t0=t0, regras=nomes_r), EN)
    json.dump(dict(regime=REGIME_DRAFT, motor_info={tag: K["motor_info"] for tag, (B, K) in M.items()}, motores={tag: hashlib.sha256(open(os.path.join(AQUI, ARQ_SUJO if tag == "sujo" else ARQ_LIMPO), "rb").read()).hexdigest() for tag in M},
                   script_sha256=hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest(), C_FIXO=C_FIXO, PASSO=PASSO, PASSO_ESCOLHA=PASSO_ESCOLHA,
                   tempos=KS["todos"], min_treino=KS["min_nov"], lockbox_serio=L0, hist_inicio=hist_ini, inicio_opcoes_draft=ini_draft, embargo_escolha=EMBARGO_ESCOLHA, flag_inicio=flag_ini, min_hist=min_hist, conjuntos=list(CONJUNTOS), n_acaso=N_ACASO,
                   min_linhas=MIN_LINHAS, janela_recente=JANELA_RECENTE, metricas=METRICAS, regras=[r if isinstance(r, str) else {k: (v if k != "funcao" else "função") for k, v in r.items()} for r in regras],
                   janelas=JANELAS, funis=FUNIS, funis_minuto=FUNIS_MIN, periodos=P, opcoes=nomes, limiares_ev=LIMIARES_EV, horizontes_clv=HORIZONTES_CLV,
                   minutos_odds=[int(x) for x in ODDS["min"]], n_sombra=N_SOMBRA, criterios_sombra=CRITERIOS_SOMBRA, min_amostra_sombra=MIN_AMOSTRA_SOMBRA, bins_ev=[float(x) for x in BINS_EV],
                   min_calib=MIN_CALIB, janelas_rolantes=dict(curta=JANELA_ROLL_CURTA, longa=JANELA_ROLL_LONGA, dias=[DIAS_ROLL_CURTA, DIAS_ROLL_LONGA], com_data=DATAS is not None), continuacao=CONTINUACAO),
              open(os.path.join(OUT, "config_funil.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)


def ensemble_minutos(G, ESC, BETS_MIN, JSM, SER, nomes, nomes_r, P):
    """v4.7: ENSEMBLE do melhor de cada minuto. Para cada (janela, conjunto, regra): em cada par, o minuto t usa a opção que a regra escolheu
    no funil T{t} (só com o histórico daquele minuto, < início do par). FIRST = em cada jogo, a aposta do 1º minuto (10→35) em que a escolhida
    daquele minuto entra; MULTI = as apostas de todos os minutos. Mesmos jogos avaliados dos outros funis; jogo sem aposta = 0."""
    linhas, livro = [], []
    for evl in sorted({f.get("ev_min") for f in FUNIS_MIN}, key=lambda v: -1 if v is None else v):   # v5.0: um ensemble por regra de entrada (EV)
     mins = sorted(int(f["minuto"]) for f in FUNIS_MIN if f.get("ev_min") == evl); suf = _SUF_EV(evl)
     for (jan, conj), js in JSM.items():
        for rn in nomes_r:
            if any((jan, conj, rn, t, evl) not in ESC for t in mins): continue
            rows = []
            for t in mins:
                esc = ESC[(jan, conj, rn, t, evl)]
                for k in np.unique(esc[esc >= 0]):
                    b = BETS_MIN[(t, evl)][k]; m = np.isin(b["g"], G[esc == k])
                    if m.any(): rows.append(pd.DataFrame(dict(gameid=b["g"][m], t=b["t"][m], opcao=nomes[k], side=b["side"][m], odd=b["odd"][m], p=b["p"][m], y=b["y"][m], lucro=b["L"][m],
                                                              clv=b["clv"][m], supera=b["supera"][m])))
            todas = (pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["gameid", "t", "opcao", "side", "odd", "p", "y", "lucro", "clv", "supera"])).sort_values(["gameid", "t"], kind="mergesort")
            assert set(todas["gameid"]) <= set(js.tolist()), "ensemble: aposta fora dos jogos avaliados"
            for pol in ("FIRST", "MULTI"):
                ap = todas.groupby("gameid", sort=False).head(1) if pol == "FIRST" else todas
                lc = ap.groupby("gameid")["lucro"].sum().reindex(js).fillna(0.0); nc = ap.groupby("gameid").size().reindex(js).fillna(0).astype(float)
                r_ = _bt(None, lc); L_ = ap["lucro"].values.astype(float); c_ = np.cumsum(L_); run = best = 0
                for v in L_:
                    run = run + 1 if v < 0 else 0; best = max(best, run)
                d = dict(politica_ensemble=pol, nivel_ev=suf.lstrip("_") or "EV0", janela=jan, conjunto=conj, regra=rn, n_jogos=len(js), ppg=r_["media"], ic_lo=r_["ic_lo"], ic_hi=r_["ic_hi"], p_unilateral=r_["p_valor"],
                         ev_medio_modelo=float((ap["p"] * ap["odd"] - 1).mean()) if len(ap) else np.nan, clv_fech_medio=float(np.nanmean(ap["clv"])) if np.isfinite(ap["clv"].astype(float)).any() else np.nan,
                         supera_fech_taxa=float(np.nanmean(ap["supera"])) if np.isfinite(ap["supera"].astype(float)).any() else np.nan)
                for (pn, pa, pb) in P:
                    m_, se_ = _media_se(lc[(js >= pa) & (js < pb)]); d[f"ppg_{pn}"] = m_; d[f"se_{pn}"] = se_
                d.update(risco_serie(lc, nc)); d.update(maxdd_por_aposta=float(np.max(np.maximum.accumulate(np.r_[0.0, c_])[1:] - c_)) if len(L_) else 0.0, pior_seq_apostas_perdidas=int(best))
                junto = next((f["nome"] for f in FUNIS if f["politica"] == pol and f.get("ev_min") == evl), None)
                for nm, key in ((f"junto_{pol}", (junto, jan, conj, rn)), ("BASE", (junto, jan, conj, "BASE"))):
                    if key in SER:
                        m_, se_ = _media_se(lc - SER[key]); d[f"ppg_{nm}"] = float(SER[key].mean()); d[f"delta_vs_{nm}"] = m_; d[f"z_vs_{nm}"] = m_ / se_ if se_ and se_ > 0 else np.nan
                for t in mins:
                    key = (f"T{t}{suf}", jan, conj, rn)
                    if key in SER: d[f"ppg_so_T{t}"] = float(SER[key].mean())
                d["apostas_por_minuto"] = " · ".join(f"t{int(t)} {n}" for t, n in ap.groupby("t").size().items())
                d["lucro_por_minuto"] = " · ".join(f"t{int(t)} {v:+.1f}" for t, v in ap.groupby("t")["lucro"].sum().items())
                linhas.append(d)
                if (jan, conj) in SALVAR_APOSTAS:
                    livro.append(ap.assign(politica_ensemble=pol, nivel_ev=suf.lstrip("_") or "EV0", janela=jan, conjunto=conj, regra=rn))
    LV = pd.concat(livro, ignore_index=True) if livro else pd.DataFrame()
    if len(LV): LV = LV[["politica_ensemble", "nivel_ev", "janela", "conjunto", "regra", "gameid", "t", "opcao", "side", "odd", "p", "y", "lucro", "clv", "supera"]]
    log(f"ENSEMBLE por minuto: {len(linhas)} séries · livro {len(LV)} apostas")
    return pd.DataFrame(linhas), LV


def escrever_resumo(R, PR, MC, MT, P, CP, DA, MI, info, EN=None):
    f_ = lambda v, fmt="+.4f": "n/d" if v is None or (isinstance(v, float) and not np.isfinite(v)) else format(v, fmt)
    cad2 = f"a_cada_{PASSO_ESCOLHA}"
    txt = [f"VALIDAÇÃO DO FUNIL v5.0 — {time.strftime('%Y-%m-%d %H:%M')} · regime draft {info['regime']} · C fixo {info['C']} · {info['n_opcoes']} opções (modelo × flag) · "
           f"{info['n_G']} jogos executáveis desde {info['hist_ini']} · opções com draft desde {info['ini_draft']}",
           f"Treino e re-escolha a cada {PASSO}/{PASSO_ESCOLHA} jogos, histórico ACUMULADO · PPG = lucro ÷ jogos executáveis · tudo nos MESMOS jogos dentro de cada (funil, janela) · "
           f"IC 95% bootstrap por cluster de 10 gameids",
           f"Colunas: PPG [IC] · Δ vs mesma regra por bloco · ROI · Sharpe/jogo · maxDD (u) · % blocos de {BLOCO_RISCO} jogos positivos · otimismo (PPG prometido − realizado) · "
           "percentil OOS (0,5 = acaso) · trocas · MCS (✓ = não se distingue da melhor) · Δ PPG contra cada benchmark (z por cluster) · Brier skill das previsões escolhidas · mais escolhidas",
           f"Benchmarks [ref]: BASE_{BASELINE_MODELO} = esse modelo sem flag em todos os jogos · MODELOS FIXOS = cada modelo sozinho (MOM e INI) · acaso pct/p = posição da regra na DISTRIBUIÇÃO DO ACASO · FAVORITO / ZEBRA = 1 u no lado de menor / maior odd · VIG_MERCADO = dutching (lucro = −margem, igual para qualquer vencedor) "
           "(custo da margem) · ACASO = média das opções elegíveis · FIXO_RETROSPECTO = melhor opção olhando o resultado (teto ENVIESADO). Mesma política, minutos e faixa de odd do funil.", ""]
    nomes_min = {f["nome"] for f in FUNIS_MIN}; nomes_ev = {f["nome"] for f in FUNIS if f.get("ev_min") is not None}
    txt.append(f"v5.0: blocos completos abaixo = regra de entrada de HOJE (edge > 0 ⇔ EV > 0). EV > 5% / > 10%: tabela 'LIMIAR DE EV' mais abaixo (detalhes no resumo_funil.csv, funis _EV5/_EV10). "
               f"Cada linha de regra também traz: ROI das apostas [IC] × ROI ESPERADO (EV médio do modelo / EV calibrado no passado da opção) · CLV = odd·q_fech − 1 e % que superou o fechamento.")
    txt.append("")
    for (fu, jan, cj), r in R.groupby(["funil", "janela", "conjunto"], sort=False):
        if fu in nomes_min or fu in nomes_ev: continue                                            # v4.7/v5.0: por minuto e EV5/EV10 → seções próprias abaixo
        f3 = lambda X: X[(X.funil == fu) & (X.janela == jan) & (X.conjunto == cj)]
        mc = f3(MC).iloc[0]; da = f3(DA).iloc[0]
        txt.append(f"=== {fu} · janela {jan} · opções: {cj} · {int(mc.n_jogos)} jogos · Reality Check (alguma regra > acaso, corrigido pelas {int(mc.n_regras)} regras): p={f_(mc.rc_p_reality_check, '.4f')} "
                   f"(SPA {f_(mc.rc_p_spa, '.4f')}) · MCS {mc.mcs_alpha:.0%}: {mc.mcs_regras or 'n/d'}")
        txt.append(f"    DISTRIBUIÇÃO DO ACASO ({int(da.n_sorteios)} sequências, uma opção elegível sorteada a cada par): PPG mediana {f_(da.ppg_p50)} [p5 {f_(da.ppg_p05)}; p95 {f_(da.ppg_p95)}] · "
                   f"maxDD mediana {f_(da.maxdd_p50, '.1f')} [p5 {f_(da.maxdd_p05, '.1f')}; p95 {f_(da.maxdd_p95, '.1f')}]")
        a2 = r[r.cadencia == cad2].sort_values("ppg", ascending=False)
        for _, a in a2.iterrows():
            pr = f3(PR)[f3(PR).regra == a.regra].iloc[0]
            txt.append(f"   {a.regra:<28} {f_(a.ppg)} [{f_(a.ic_lo)}; {f_(a.ic_hi)}] · Δbloco {f_(pr.delta_ppg_2_menos_bloco, '+.3f')} · ROI {f_(a.get('roi'), '+.3f')} · "
                       f"Sh {f_(a.get('sharpe_jogo'), '+.3f')} · DD jogo/aposta {f_(a.get('maxdd'), '.1f')}/{f_(a.get('maxdd_por_aposta'), '.1f')} · sem decisão {f_(a.get('pct_jogos_sem_decisao'), '.0%')} · blocos+ {f_(a.get('pct_blocos_positivos'), '.0%')} · otim {f_(a.get('otimismo'), '+.3f')} · "
                       f"pctOOS {f_(a.get('percentil_oos_medio'), '.2f')} · acaso pct {f_(a.get('percentil_no_acaso'), '.2f')} p={f_(a.get('p_vs_acaso'), '.3f')} · trocas {int(a.trocas)} · {'✓' if a.get('no_MCS') else ' '} · "
                       + " · ".join(f"Δ{b_} {f_(c_.delta_ppg, '+.3f')} (z {f_(c_.z, '+.1f')})" for b_ in (f"BASE_{BASELINE_MODELO}", "FAVORITO", "ZEBRA", "VIG_MERCADO", "ACASO")
                                    for _, c_ in f3(CP)[(f3(CP).regra_1 == a.regra) & (f3(CP).regra_2 == b_)].iterrows())
                       + f" · Brier skill das escolhidas {f_(a.get('bsskill_escolhida'), '+.5f')} · {a.mais_escolhidas}"
                       + f" · ROI apostas {f_(a.get('roi_apostas'), '+.3f')} [{f_(a.get('roi_ic_lo'), '+.3f')}; {f_(a.get('roi_ic_hi'), '+.3f')}] × esperado {f_(a.get('ev_medio_modelo'), '+.3f')} (calibrado {f_(a.get('ev_medio_calibrado'), '+.3f')})"
                       + f" [sem histórico p/ calibrar {f_(a.get('pct_ev_cal_sem_historico'), '.0%')}] · CLV {f_(a.get('clv_fech_medio'), '+.3f')} mk_fech {f_(a.get('mk_fech_medio'), '+.4f')} supera {f_(a.get('supera_fech_taxa'), '.0%')} (n={f_(a.get('n_com_fechamento'), '.0f')}) · dp aposta real/esperado {f_(a.get('dp_lucro_aposta'), '.2f')}/{f_(a.get('dp_esperado_modelo'), '.2f')}")
        fixos = r[(r.cadencia == "-") & r.regra.str.startswith("FIXO_") & ~r.regra.str.startswith("FIXO_RETRO")].sort_values("ppg", ascending=False)
        if len(fixos):
            txt.append(f"   [ref] MODELOS FIXOS sem flag ({len(fixos)}; PPG · Brier skill · percentil no acaso): "
                       + " | ".join(f"{x.regra.split(' ')[0][5:]} {f_(x.ppg, '+.3f')} {f_(x.get('bsskill_escolhida'), '+.4f')} {f_(x.get('percentil_no_acaso'), '.2f')}" for _, x in fixos.iterrows()))
        mi = f3(MI) if len(MI) else MI
        if len(mi):
            txt.append("   MOM − INI (mesmo modelo fixo com × sem a odd do momento; ΔPPG (z) · Brier skill MOM/INI): "
                       + " | ".join(f"{x.modelo_MOM[5:].replace('_MOM', '')} {f_(x.delta_ppg_MOM_menos_INI, '+.3f')} ({f_(x.z, '+.1f')}) {f_(x.bsskill_MOM, '+.4f')}/{f_(x.bsskill_INI, '+.4f')}" for _, x in mi.iterrows()))
        for _, x in r[(r.cadencia == "-") & ~(r.regra.str.startswith("FIXO_") & ~r.regra.str.startswith("FIXO_RETRO"))].iterrows():
            txt.append(f"   [ref] {x.regra:<40} {f_(x.ppg)} [{f_(x.ic_lo)}; {f_(x.ic_hi)}] · ROI {f_(x.get('roi'), '+.3f')} · Sh {f_(x.get('sharpe_jogo'), '+.3f')} · DD {f_(x.get('maxdd'), '.1f')}"
                       + (f" · Brier skill {f_(x.get('bsskill_escolhida'), '+.5f')}" if pd.notna(x.get("bsskill_escolhida", np.nan)) else ""))
        txt.append("   PPG por período (" + ", ".join(pn for pn, _, _ in P) + "): " + " | ".join(f"{a.regra} " + "/".join(f_(a[f'ppg_{pn}'], '+.3f') for pn, _, _ in P) for _, a in a2.head(5).iterrows()))
        pn_x, pa_x, _pb = info["PX"][-1]; cx, sx = f"ppg_{pn_x}", f"se_{pn_x}"                   # v4.7: RANKING só da fase limpa (desde o início das opções com draft)
        if cx in r.columns:
            rk = r[(r.cadencia == cad2) | ((r.cadencia == "-") & ~r.regra.str.startswith("FIXO_RETRO"))].sort_values(cx, ascending=False)
            rk = rk[(rk.cadencia == cad2) | ~rk.regra.str.startswith("FIXO_") | (rk.index.isin(rk[rk.regra.str.startswith("FIXO_")].head(3).index))]
            txt.append(f"   RANKING SÓ jogos >= {pa_x} (PPG ± erro-padrão por cluster; [ref] = benchmark; FIXO = os 3 melhores modelos fixos aqui, escolhidos OLHANDO o resultado): "
                       + " | ".join(f"{'[ref] ' if x.cadencia == '-' else ''}{x.regra.split(' ')[0]} {f_(x[cx], '+.3f')}±{f_(x[sx], '.3f')}" for _, x in rk.iterrows()))
        txt.append("")
    if nomes_min and EN is not None and len(EN):                                                  # v4.7: escolha POR MINUTO e ENSEMBLE
        EN = EN[EN["nivel_ev"] == "EV0"] if "nivel_ev" in EN.columns else EN                       # v5.0: aqui só a regra de entrada de hoje (EV5/EV10 no ensemble_minutos.csv)
        mins = [f["nome"] for f in FUNIS_MIN if f.get("ev_min") is None]
        txt.append("ESCOLHA POR MINUTO: em T10…T35 cada regra escolhe modelo × flag SÓ com o histórico daquele minuto e aposta SÓ nele. ENSEMBLE = cada minuto com a "
                   "sua escolha (FIRST: 1º minuto em que a escolhida do minuto entra; MULTI: todos). 'junto' = os funis acima (uma escolha para os 6 minutos). PPG nos MESMOS jogos.")
        for (jan, cj), en in EN.groupby(["janela", "conjunto"], sort=False):
            f3 = lambda X, fu: X[(X.funil == fu) & (X.janela == jan) & (X.conjunto == cj)]
            txt.append(f"=== janela {jan} · opções: {cj}")
            for fu in mins:
                mc, da = f3(MC, fu), f3(DA, fu)
                if not len(mc): continue
                fx = R[(R.funil == fu) & (R.janela == jan) & (R.conjunto == cj) & (R.cadencia == "-") & R.regra.str.startswith("FIXO_") & ~R.regra.str.startswith("FIXO_RETRO")].sort_values("ppg", ascending=False).head(1)
                txt.append(f"   {fu}: acaso mediana {f_(da.ppg_p50.iloc[0])} [p5 {f_(da.ppg_p05.iloc[0])}; p95 {f_(da.ppg_p95.iloc[0])}] · Reality Check p={f_(mc.rc_p_reality_check.iloc[0], '.4f')} · "
                           f"MCS {mc.mcs_regras.iloc[0] or 'n/d'}" + (f" · melhor modelo fixo {fx.regra.iloc[0].split(' ')[0][5:]} {f_(fx.ppg.iloc[0])}" if len(fx) else ""))
            cab = "   " + f"{'regra':<22}" + "".join(f"{m:>9}" for m in mins) + f"{'ENS_FIRST':>11}{'(z vs junto)':>13}{'ENS_MULTI':>11}{'(z vs junto)':>13}{'junto_FIRST':>13}{'junto_MULTI':>13}"
            txt.append(cab)
            ef, em = en[en.politica_ensemble == "FIRST"].set_index("regra"), en[en.politica_ensemble == "MULTI"].set_index("regra")
            for rn in info["regras"]:
                if rn not in ef.index: continue
                a, b = ef.loc[rn], em.loc[rn]
                txt.append("   " + f"{rn:<22}" + "".join(f"{f_(a.get(f'ppg_so_{m}', np.nan)):>9}" for m in mins)
                           + f"{f_(a.ppg):>11}{'(' + f_(a.get('z_vs_junto_FIRST'), '+.1f') + ')':>13}{f_(b.ppg):>11}{'(' + f_(b.get('z_vs_junto_MULTI'), '+.1f') + ')':>13}"
                           + f"{f_(a.get('ppg_junto_FIRST')):>13}{f_(b.get('ppg_junto_MULTI')):>13}")
            txt.append("")
        txt.append("   Detalhes de cada minuto (IC, risco, overfit, MCS, acaso, benchmarks no minuto) → resumo_funil.csv (funil T10…T35); ensemble (IC, risco, períodos, "
                   "de que minuto veio cada aposta/lucro) → ensemble_minutos.csv; livro do ensemble → apostas_ensemble.csv.gz")
        txt.append("")
    # ---------------- v5.0: LIMIAR DE EV, CLV/MARKOUT, SOMBRA, MONITOR ----------------
    pol_ev = [(f["politica"], f.get("ev_min"), f["nome"]) for f in FUNIS]
    if len({e for _, e, _ in pol_ev}) > 1:
        txt.append("LIMIAR DE EV (mesmas regras, mesmos jogos; só muda a regra de ENTRADA: EV>0 = hoje · EV>5% · EV>10%) — PPG / ROI das apostas / nº de apostas · "
                   "Δ PPG contra EV>0 PAREADO nos mesmos jogos (z por cluster). ATENÇÃO: escolher o limiar por esta tabela é seleção nos mesmos dados:")
        for pol in [f["politica"] for f in FUNIS_BASE]:
            nm_ = [(e, n_) for p_, e, n_ in pol_ev if p_ == pol]
            for (jan, cj), _x in R[R.funil == nm_[0][1]].groupby(["janela", "conjunto"], sort=False):
                txt.append(f"=== {pol} · janela {jan} · opções {cj}   (colunas: " + " | ".join(("EV>0" if e is None else f"EV>{e:.0%}") for e, _ in nm_) + ")")
                for rn in info["regras"] + [f"BASE_{BASELINE_MODELO} (fixo, sem flag)"]:
                    cel = []
                    for e, n_ in nm_:
                        x = R[(R.funil == n_) & (R.janela == jan) & (R.conjunto == cj) & (R.regra == rn) & (R.cadencia.isin([cad2, "-"]))]
                        cel.append(f"{f_(x.ppg.iloc[0], '+.4f')} / {f_(x.get('roi_apostas', pd.Series([np.nan])).iloc[0], '+.3f')} / {f_(x.get('n_apostas_serie', pd.Series([np.nan])).iloc[0], '.0f')}" if len(x) else "n/d")
                    EVC_ = info.get("EVC"); dz = ""
                    if EVC_ is not None and len(EVC_):
                        y_ = EVC_[(EVC_.funil_ev0 == nm_[0][1]) & (EVC_.janela == jan) & (EVC_.conjunto == cj) & (EVC_.regra == (rn if not rn.startswith("BASE_") else "BASE"))]
                        dz = " · " + " ".join(f"Δ{x_.funil_ev.split('_')[-1]} {f_(x_.delta, '+.4f')} (z {f_(x_.z, '+.1f')})" for _, x_ in y_.iterrows())
                    txt.append(f"   {rn.split(' ')[0]:<26} " + " | ".join(f"{c_:>26}" for c_ in cel) + dz)
        txt.append("")
    CLV = info.get("CLV")
    if CLV is not None and len(CLV):
        txt.append("CLV / MARKOUT do lado apostado (regra de entrada de hoje; janela própria, todas as opções; as 5 regras de maior PPG + BASE). mk = q(t+h) − q(t) (q sem margem; + = o mercado "
                   "veio para o lado da aposta; num mercado sem viés a média é 0) · a favor = % mk > 0 (NÃO compare com 50%: depende do preço — zebra fica abaixo de 50% mesmo sem viés) · "
                   "ROI markout = odd(t)·q(t+h) − 1 · fechamento = última odd do jogo depois da aposta (ao vivo isso fica PERTO DO RESULTADO e o horizonte depende da duração do "
                   "jogo: CLV de fechamento acompanha o lucro; os horizontes fixos +5/+10 são o sinal menos ruidoso) · perdidas = sem odd em t+h (minuto não existe nos dados ou jogo acabou):")
        for f0 in [f["nome"] for f in FUNIS if f.get("ev_min") is None]:
            rr = R[(R.funil == f0) & (R.janela == "propria") & (R.conjunto == "todas") & (R.cadencia == cad2)].sort_values("ppg", ascending=False).head(5).regra.tolist() + [f"BASE_{BASELINE_MODELO} (fixo, sem flag)"]
            txt.append(f"=== {f0}")
            for rn in rr:
                x = CLV[(CLV.funil == f0) & (CLV.janela == "propria") & (CLV.conjunto == "todas") & (CLV.regra == rn)]
                if not len(x): continue
                txt.append(f"   {rn.split(' ')[0]:<24} " + " | ".join(f"{h.horizonte}: n {int(h.n)} (perd {h.pct_perdidas:.0%}) a favor {f_(h.get('taxa_a_favor'), '.0%')} mk {f_(h.get('mk_medio'), '+.4f')} "
                                                                   f"[+{f_(h.get('mk_medio_quando_a_favor'), '.3f')}/{f_(h.get('mk_medio_quando_contra'), '+.3f')}] ROI {f_(h.get('roi_markout'), '+.3f')}" for _, h in x.iterrows()))
        txt.append("   (todas as regras, funis e IC → clv_markout.csv)")
        txt.append("")
    SB = info.get("SB")
    if SB is not None and len(SB):
        txt.append(f"MODO SOMBRA (troca só depois de {N_SOMBRA} jogos em sombra vencendo a produção congelada; critério clv = movimento médio do mercado até o fechamento (q_fech − q), "
                   f"brier = Brier skill) — PPG com sombra × sem sombra (Δ, z) · trocas · promoções/rejeições/indeterminadas (sem amostra) · % dos pares ainda na 1ª escolha:")
        for (f0, jan, cj), x in SB[SB.funil.isin([f["nome"] for f in FUNIS if f.get("ev_min") is None])].groupby(["funil", "janela", "conjunto"], sort=False):
            txt.append(f"=== {f0} · janela {jan} · opções {cj}")
            for rn, y in x.groupby("regra", sort=False):
                txt.append(f"   {rn:<26} sem sombra {f_(y.ppg_sem_sombra.iloc[0])} · " + " · ".join(f"{c.criterio}: {f_(c.ppg)} (Δ {f_(c.delta_vs_sem_sombra, '+.4f')}, z {f_(c.z, '+.1f')}) trocas {int(c.trocas_producao)}/{int(c.trocas_sem_sombra)} prom {int(c.promocoes)} rej {int(c.rejeicoes)} indet {int(c.indeterminadas)} 1ª {f_(c.pct_pares_na_1a_escolha, '.0%')}" for _, c in y.iterrows()))
        txt.append("   (EV5/EV10 e eventos de cada troca → sombra.csv / sombra_eventos.csv.gz)")
        txt.append("")
    MO = info.get("MO")
    if MO is not None and len(MO):
        txt.append("MONITOR (para alarme, NÃO para decidir): ROI / movimento do mercado até o fechamento (mk_fech) / Brier skill rolantes na janela curta, no FIM do período, e % do tempo "
                   "(só com a janela cheia) em alarme — regra de entrada de hoje; janela própria, todas:")
        for f0 in [f["nome"] for f in FUNIS if f.get("ev_min") is None]:
            x = MO[(MO.funil == f0) & (MO.janela == "propria") & (MO.conjunto == "todas")]
            if not len(x): continue
            u = x.iloc[0]; txt.append(f"=== {f0} · janela curta {u.janela_curta} {u.unidade_janela} · longa {u.janela_longa} {u.unidade_janela}")
            for _, m in x.iterrows():
                al = ",".join(k_[7:] for k_ in ("ALARME_ROI", "ALARME_CLV", "ALARME_BRIER") if bool(m[k_])) or "—"
                txt.append(f"   {m.regra.split(' ')[0]:<26} ROI curta {f_(m.roi_curta_atual, '+.3f')} longa {f_(m.roi_longa_atual, '+.3f')} · mk_fech {f_(m.mk_fech_curta_atual, '+.4f')} (supera {f_(m.supera_fech_curta_atual, '.0%')}) · Brier skill {f_(m.bsskill_curta_atual, '+.4f')} · "
                           f"DD atual {f_(m.drawdown_atual, '.1f')} (máx {f_(m.maxdd, '.1f')}) · tempo com ROI curta < 0: {f_(m.pct_tempo_roi_curta_negativo, '.0%')} · alarmes agora: {al}")
        txt.append("   (todas → monitor_alarmes.csv · séries → rolling_top.csv.gz · gráficos → graficos/*.svg)")
        txt.append("")
    t = MT[(MT["t"].astype(str) == "TODOS") & (MT["periodo"] == "TOTAL")]
    txt.append("BRIER / LOG LOSS (todas as previsões, todos os minutos, período TOTAL; skill = mercado − modelo, + = modelo melhor; z = skill / erro-padrão por cluster):")
    for _, x in t[t.recorte == "todas"].sort_values("bsskill", ascending=False).iterrows():
        txt.append(f"   {x.modelo:<16} n={int(x.n_linhas):>7} · Brier {x.brier_mod:.4f} (mercado {x.brier_mkt:.4f}) skill {x.bsskill:+.5f} z={f_(x.bsskill_z, '+.2f')} · "
                   f"log loss {x.ll_mod:.4f} (mercado {x.ll_mkt:.4f}) skill {x.llskill:+.5f} z={f_(x.llskill_z, '+.2f')}")
    txt.append("   flag=1 × flag=0 × entrou/não entrou × lado apostado, por modelo, período e minuto → metricas_previsao_resumo.csv / _por_minuto.csv.gz")
    txt += ["", "LEIA COM CUIDADO:",
            " - " + (f"CONTINUAÇÃO: os jogos >= {CORTE_ANTIGO} (antigo lockbox) entraram como jogos normais — não há mais confirmação fora da amostra; o período continuacao_{CORTE_ANTIGO}_fim é o único ainda não usado para ESCOLHER regras." if CONTINUACAO else "Desenvolvimento: jogos já vistos em análises; mede o MÉTODO, não confirma nada. A confirmação é o lockbox (8448+)."),
            f" - {len(info['regras'])} regras × {len(JANELAS)} janelas × {len(FUNIS)} funis: use o Reality Check/MCS (já corrigem pelas regras dentro de cada funil/janela) "
            "antes de dizer que uma regra é melhor; escolher pelo maior PPG desta tabela é seleção múltipla. Regras novas entram na mesma conta.",
            " - Métricas 'todas'/'entradas' comparam opções em LINHAS diferentes (cada flag tem o seu universo): o skill é relativo ao mercado nas mesmas linhas; Brier/log loss cru não.",
            " - Na janela 'comum', lucro e ppg escolhem igual. FIXO_RETROSPECTO é teto enviesado. t_ppg/t_bsskill não têm correção de cluster: use como filtro, não como p-valor.",
            f"pronto em {(time.time() - info['t0']) / 60:.1f} min"]
    open(os.path.join(OUT, "RESUMO_FUNIL.txt"), "w", encoding="utf-8").write("\n".join(txt)); print("\n".join(txt), flush=True)


if __name__ == "__main__":
    if FILHO: main_filho(*FILHO)
    else: main()
