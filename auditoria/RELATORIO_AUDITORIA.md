# Auditoria de vazamento — validar_funil.py (v4 → v4.1) — 02/10/2026

Escala usada: **VERIFICADO** (li o código e testei, com prova) · **ALTA CONFIANÇA** (entendo a lógica, não testei cada detalhe) ·
**INCERTO** · **NÃO SEI**. Toda evidência empírica abaixo é com **dados SINTÉTICOS** (`gerar_zz_sinteticos.py`); nenhum dado real foi usado.

## 1. Como treino, teste e escolha funcionam (e o que mudou)

Não há um "treino" e um "teste" separados. É **walk-forward** (prequencial), em pares fixos de gameid [a, a+2):

| passo | o que acontece | só usa | prova |
|---|---|---|---|
| 1. treino | cada modelo (por minuto) é reajustado com **todas as linhas com gameid < a** | gameid < a | `assert tr.gameid.max() < g_lim <= g0` no motor (linha 705) + `conferir_retreino` + auditor A2/A3 |
| 2. previsão | o modelo prevê os 2 jogos do par (o 2º sem o resultado do 1º) | features do próprio minuto | sabotagem de ponta a ponta (R1) |
| 3. escolha | cada regra calcula as métricas de cada opção sobre os jogos executáveis em [w, a − EMBARGO_ESCOLHA) — **previsões fora da amostra** feitas no seu tempo | gameid < a | auditor A5 (recalcula do zero), sabotagem interna e de ponta a ponta |
| 4. aposta | a opção escolhida aposta no par; o lucro entra na série do FUNIL | resultado do par só no lucro | auditor A6 |
| 5. avança | o par vira histórico do próximo | — | — |

**O que mudou em relação ao que vocês faziam:**
- v8.x: um modelo por **bloco de 150 jogos** (treino até o início do bloco, previa os 150).
- v9.0 (etapa 1): modelo **a cada 2 jogos**, mas a ESCOLHA do modelo/C era refeita **a cada bloco de 150** (rodar_holdout) ou **congelada** no 7729 (comparar_lockbox, "COM lockbox").
- agora: modelo **e** escolha a cada 2 jogos.

Encurtar o "teste" para 2 jogos **não cria vazamento por si só** (cada par continua previsto e escolhido só com jogos anteriores): só usa a informação mais cedo, como na operação. O que PODE criar vazamento é a premissa de ordem: **supõe-se que todos os jogos com gameid < a terminaram antes de o jogo a começar**. Se houver mais de 2 jogos simultâneos, um jogo ainda em andamento entra no treino/histórico. **NÃO SEI** se isso acontece nos seus dados (não há horário de início/fim). Mitigação disponível: `EMBARGO_ESCOLHA` (escolha) e `EMBARGO_JOGOS` do motor (treino; muda a chave do banco → retreina).

**gameid:** os 2 lados de um jogo têm o mesmo gameid e caem sempre no mesmo par (os cortes são por gameid) — **VERIFICADO** pelo motor (`assert (tr.groupby("gameid").size() == 2).all()`, linha 706) e pela conferência entre minutos (`conferir_tempos`: mesmas (gameid, side) → mesmos valores em resultado/odds/flags entre os 9 zz).

## 2. Achados

| # | achado | origem | situação | certeza |
|---|---|---|---|---|
| 1 | **PRO de backfill (6028–6114) entrava na escolha**: no modo sujo o motor não mascara o PRO desses jogos (linhas 500–501 só valem no modo limpo) e todo modelo com draft usa `l_PRO*` (linhas 147/566). A v4 deixava as opções com draft entrarem em 6028. | eu (releitura) | **corrigido**: opções com draft a partir de `max(G_CLEAN, PRO_LIVE_FROM)` = 6115 | VERIFICADO no código; R0 mostra "previsões antes do jogo 420 (PRO_LIVE_FROM do teste) ficam FORA" |
| 1b | O mesmo vale para a **etapa 1 oficial** (rodar_holdout, modo sujo, histórico desde 6028): os candidatos com draft têm previsões com PRO de backfill em 6028–6114 no histórico do critério. | eu | **não mexi** (é o pacote oficial) — decidir com você | ALTA CONFIANÇA (leitura do código; não rodei) |
| 2 | Linhas do **lockbox** (gid ≥ 8448) eram lidas e comparadas (só igualdade) em `conferir_tempos`/`verificar_limpo` | eu | **corrigido**: descartadas na leitura; conferência limpo==grande refeita só com gid < 8448; asserts no universo/base/segmentos | VERIFICADO (R2: saídas idênticas com e sem as linhas do lockbox; mutação M2 pega pelo auditor A1) |
| 3 | **Conjunto de avaliação = interseção dos jogos de TODAS as regras**: acrescentar uma regra mudava o PPG das outras | auditoria externa (T07) | **corrigido**: conjunto FIXO = jogos dos pares com ≥ 1 opção elegível; regra sem decisão ali = sem aposta (0), com `pct_jogos_sem_decisao` | VERIFICADO no código; concordo com o achado |
| 4 | **Drawdown só por jogo** esconde perdas dentro do jogo (MULTI) | externa (T08) | **corrigido**: `maxdd_por_aposta` e `pior_seq_apostas_perdidas` | VERIFICADO (teste unitário: −1,+1 no mesmo jogo → DD por jogo 0, por aposta 1) |
| 5 | **VIG_MERCADO (½ u em cada lado) não perde a margem constante** e não respeitava a faixa de odd. Eu tinha escrito "perde exatamente a margem" — **estava errado**. | externa (T09–T10) | **corrigido**: dutching (stake ∝ 1/odd) ⇒ lucro = 1/S − 1 sempre; só (jogo, minuto) executáveis | VERIFICADO (teste unitário com valores calculados à mão) |
| 6 | **Calibração `_CALR` sem índices próprios no log** | externa | **não muda o motor** (mudaria a chave do banco). Li as linhas 715–741: o calibrador usa só `tr` (gid < g_lim) em dobras rolantes e é aplicado a `te` com o q do próprio minuto. Cobertura empírica: R1 compara também as previsões _CALR | VERIFICADO por leitura; empírico só no sintético |
| 7 | **Opção com previsões ausentes pode ser elegível e vencer por PPG ≈ 0** | externa (T17) | **parcial**: não é vazamento nem erro de conta — sem previsão, o modelo também não apostaria na operação. Mas pode escolher opção "inativa". Acrescentei a métrica `cobertura_modelo` para usar como filtro (ex.: `("cobertura_modelo", ">=", 0.8)`); **não pus filtro padrão** (o limiar seria uma escolha a mais feita olhando dados) | ALTA CONFIANÇA |
| 8 | **A sabotagem INTERNA só testa a camada de escolha** | externa | **concordo**. Por isso existem a sabotagem de PONTA A PONTA (R1: altera os zz antes do treino) e o auditor independente | VERIFICADO |
| 9 | **As conferências internas usam as mesmas variáveis do código**: a mutação M1 (histórico incluindo o 1º jogo do par) passou pelo script e só o **auditor independente** pegou | eu (testes de mutação) | **corrigido**: o RODAR_FUNIL.bat roda o `auditar_saidas.py` sempre no fim e para com erro se algo falhar | VERIFICADO (M1 → A5: 3.953 divergências) |
| 10 | `.bat`: `%ERRORLEVEL%` dentro de bloco `( )` e `)` sem escape — uma falha do auditor apareceria como sucesso | eu (revisão) | **corrigido** com `goto` | ALTA CONFIANÇA (não tenho Windows para rodar) |

## 3. Riscos que NÃO são resolvidos por código
- **Premissa de ordem por gameid** (jogos simultâneos) e **sincronia odd × estado** no mesmo minuto: o próprio motor registra "RISCO NÃO MEDIDO". **NÃO SEI**.
- **cWR/cPROBS/roles e flags cWRgrande/cPROBSgrande calculados só com o passado**: está no CFG como "confirmado pelo Pedro"; eu não verifiquei.
- **Regime sujo**: modelos com draft TREINAM com o backfill (contaminação do treino; não é vazamento do resultado do jogo testado, mas pode distorcer coeficientes).
- **Overfitting de pesquisa**: os jogos 6628–8447 já foram vistos várias vezes; famílias, features, flags, minutos, EDGE_MIN, faixa de odd foram decididos olhando dados. A validação do funil nesses jogos é **otimista** por construção.
- **Comparações múltiplas**: 20 regras × 4 funis × 2 janelas × 2 conjuntos × 2 cadências + referências. MCS/Reality Check corrigem só **dentro** de cada (funil, janela, conjunto).
- **Bootstrap por cluster de 10 gameids** supõe independência entre clusters.

## 4. Testes executados (todos com dados SINTÉTICOS; código final v4.1)

| teste | o que prova | resultado | log |
|---|---|---|---|
| R0 base (regime sujo) + auditor A1–A7 | índices de treino/previsão/escolha; recálculo independente | **TUDO OK**: 16.168 treinos, folga mínima 1; 65.674 previsões com o treino do seu par; 10.800 decisões recalculadas do zero, 0 divergências | `logs/rodada_R0.log`, `logs/auditor_R0.log` |
| R1 sabotagem de ponta a ponta: jogos ≥ 600 com TODAS as colunas trocadas (odds, flags, resultado, estado), 30% desses jogos apagados, texto lixo no lockbox | nada do futuro chega ao passado (treino, features, calibração _CALR, escolha) | **0 diferenças**: 52.336 previsões < 600 com Δp = 0; 66.240 decisões com a ≤ 600 idênticas · controle: 15.993/16.000 decisões depois do 600 mudaram | `logs/comparacao_R1_sabotagem_vs_R0.log` |
| R2 sem as linhas do lockbox (gid ≥ 700 no teste) | o lockbox não influi em NADA | **idêntico**: previsões, 82.240 decisões, resumo (848×55), métricas de Brier (2.694×17), comparações (7.200×9), distribuição do acaso — Δ máx 0 | `logs/comparacao_R2_lockbox_vs_R0.log` |
| R3 regime limpo + auditor | o regime alternativo também passa | TUDO OK | `logs/auditor_R3.log` |
| R4 EMBARGO_ESCOLHA = 2 + auditor | o embargo funciona e o auditor o respeita | TUDO OK | `logs/auditor_R4.log` |
| F código final (só texto + aviso) × R0 | a última edição não mudou números | idêntico (Δ máx 0) | `logs/comparacao_F_final_vs_R0.log` |
| Mutação M1: histórico inclui o 1º jogo do par | o auditor pega vazamento que as conferências internas não pegam | script NÃO pegou; **auditor pegou** (A5: 3.953 divergências) | `logs/mutacao_M1_auditor.log` |
| Mutação M2: jogos do lockbox no universo | idem | script NÃO pegou (antes dos asserts novos); **auditor pegou** (A1) | `logs/mutacao_M2_auditor.log` |
| Mutação M3: treino alcança o par | guarda do motor | **motor abortou** ("LEAK: treino alcança o bloco de teste") | `logs/mutacao_M3_rodada_fim.log` |
| Mutação M4: escolha com o período inteiro | conferência interna | **script abortou** (conta rápida ≠ direta) | `logs/mutacao_M4_rodada_fim.log` |
| Teste unitário (15 verificações) | regras simples/compostas/função, "não apostar", DD por aposta, VIG dutching, faixa de odd | 15/15 | `logs/teste_unitario_regras.log` |

Arquivos: `configs/` (config_funil.json e o validar_funil.py EXATO de cada rodada, com md5 em `md5_scripts_usados.txt`), `dados/` (zz sintéticos base, sabotado e truncado),
`resultados_R0/` (resumo, auditoria de índices, Brier, comparações). Geradores: `gerar_zz_sinteticos.py`, `gerar_variantes.py`; comparação: `comparar_rodadas.py`.

## 5. Grau de confiança (honesto)
- **VERIFICADO (95–99%) — no sintético**: treino só com jogos anteriores ao par; escolha só com o histórico anterior; o futuro (inclusive odds e calibração) não muda nada do passado; o lockbox não influi; as contas das regras ppg/lucro/roi batem com um recálculo independente.
- **ALTA CONFIANÇA (80–90%)**: as mesmas propriedades valem nos dados reais — o código é o mesmo, mas os dados reais podem ter formatos que o sintético não tem (texto em colunas, jogos com 1 lado, odds estranhas). Por isso o auditor roda sozinho no fim da rodada real.
- **NÃO SEI**: se há jogos simultâneos além do par (ordem por gameid ≠ ordem real de término); se cWR/cPROBS/flags são mesmo só com o passado; se o PRO de backfill 6028–6114 carrega informação do resultado.
- **0% de verificação empírica nos seus dados reais** — isso não é uma estimativa de probabilidade de leak; é só que não rodei nada com eles.

## 6. v4.2 — resultados de 2 em 2 GUARDADOS (livro de apostas)
Lacuna: até a v4.1 a escolha de cada par era gravada (trilhas), mas o lucro de cada par e as apostas do procedimento ficavam só na memória
(a v1 gravava; a v2 deixou de gravar). Agora:
- `trilhas.csv.gz` ganha `jogos_no_par`, `apostas_no_par`, `lucro_no_par` (da opção escolhida naquele par);
- `apostas_procedimento.csv.gz` = LIVRO: cada aposta de cada regra (par, opção escolhida, minuto, lado, odd, p, q, edge, resultado, lucro, acumulado),
  para as (janela, conjunto) de `SALVAR_APOSTAS` (padrão: própria e comum, todas as opções), cadência a cada 2.
Provas (sintético): auditor **A8** — livro == lucro por par das trilhas == PPG e nº de apostas do resumo, 160 séries, 87.007 apostas, OK;
`refazer_livro_a_mao.py` refaz as apostas dos zz crus + previsões sem usar o código do script: **iguais** nos 4 funis (185, 879, 186, 710 apostas);
números do resumo idênticos à R0 (`logs/comparacao_G_vs_R0.log`).

## 7. v4.3 — período avaliado = histórico inteiro (pergunta do Pedro)
Já era assim desde a v2 (o 6628 era da etapa 1 e da v1): `HIST_INICIO` = 1º jogo previsto (700 na configuração real). Prova no sintético
(onde 200 equivale ao 700): 1ª decisão no par 250, última no 698; 0 de 27.261 escolhas de opções com draft antes do início permitido; 0 escolhas
de flag antes do início do flag (8 flags). Novo: `INICIO_PREVISOES` permite começar antes do 700. Teste H1 (INICIO_PREVISOES = 100 no sintético):
previsões desde o jogo 100, previsões a partir do 200 IDÊNTICAS às do padrão (treinos vieram do banco: 2.000/2.400), 1ª decisão no par 152,
auditor OK. Padrão (H0) idêntico à rodada G (`logs/comparacao_H0_vs_G.log`).

## 8. v4.4 — erro no PC da Amanda: `paralelo_v90` sem `escolher_n`
Causa: o validar_funil chamava `paralelo_v90.escolher_n`, que só existe na versão v9.0.1 desse arquivo; a pasta da Amanda tem outra versão.
Erro MEU (dependência desnecessária). Correção: o paralelismo agora é do próprio validar_funil (`escolher_processos`, `rodar_filhos`), sem
importar o paralelo_v90. Testes: P1 com um paralelo_v90 antigo SEM escolher_n na pasta (simula a Amanda) e P2 sem o arquivo — os dois
idênticos à rodada H0 e auditor OK; P3 com um processo filho quebrado de propósito — a rodada para com erro e mostra o log de quem falhou.
O log real da Amanda mostrou que, com os dados reais, a leitura e TODAS as conferências passaram (lockbox: 0 linhas; 9 zz consistentes com o
zz10 em 19 colunas; limpo == grande em 169 colunas; colunas/features presentes; 3.874 pares de treino 700–8447).

## 9. v4.5 — só os minutos 10 a 35 (decisão do Pedro)
Motivo: no log real, t45 tem 156 jogos e t40 685 em todo o histórico → modelos de 40/45 muito sobreajustados (EPV ≈ 2–3 no t45).
Mudança: `TEMPOS_NOVOS = []`, funis FIRST_10a35 e MULTI_10a35, métricas nos minutos 10–35; o RODAR_FUNIL.bat confere zz10…zz35 (não pede mais
zz5/40/45). Teste T (sintético, SEM os arquivos zz5/40/45 na pasta): roda; auditor A1–A8 TUDO OK; livro refeito à mão igual (183 e 704
apostas); previsões 10–35 idênticas às da rodada com 5/40/45 (49.428 linhas). Os PPG mudam em relação às rodadas anteriores porque o universo
de jogos executáveis passa a ser o dos minutos 10–35 (antes incluía jogos executáveis só no 5/40/45).
