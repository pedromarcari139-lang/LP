# validar_funil v5.0 — relatório para revisão (03/10/2026)

Escala: **VERIFICADO** (li o código e testei com prova) · **ALTA CONFIANÇA** (lógica entendida, sem teste de cada detalhe) · **INCERTO** · **NÃO SEI**.
Todas as evidências empíricas são com **dados SINTÉTICOS**. Nenhum dado real foi rodado por mim.
Estado: revisão adversarial independente CONCLUÍDA (6 revisores + céticos); achados e correções na seção 6.

## 1. O que o script faz (pipeline)
1. **Motor do usuário** (`backtest_sujo.py`, v9.0 no PC de produção). O script importa o motor e usa dele só a PREVISÃO: carga, `preparar`, features, candidatos, `blocos_teste`, `walk_forward_dev` e banco de previsões. As funções de AVALIAÇÃO são cópias literais da v9.0.1 dentro do script; o log confere, função por função, se as do motor são idênticas.
2. **Walk-forward das previsões:**
   - os jogos são divididos em pares de 2 gameids [a, a+2), de 700 até o último jogo antes do lockbox (8448);
   - para cada par, cada candidato (23 modelos) e cada minuto (10, 15, …, 35), o motor treina com gameid < a e prevê só a e a+1;
   - C é fixo em 1.0;
   - `conferir_retreino()` aborta se algum treino alcançar o par que prevê; o auditor A2/A3 confere de novo.
3. **Opções = modelo × flag** (207 no real). Uma opção existe a partir de w = max(início do modelo, início do flag, 700). Modelos com draft só a partir de 6115 (PRO capturado ao vivo).
4. **Apostas de cada opção (função copiada do motor):**
   - em cada (jogo, minuto), o lado de maior edge (p − 1/odd);
   - aposta se edge > 0, a odd está entre 1,01 e 7, o flag é 1 (se a opção tiver flag) e, na v5.0, EV > limiar (0 / 5% / 10%);
   - FIRST = 1º minuto que passa; MULTI = todos os que passam.
5. **Universo G:** jogos com os 2 lados com odd válida e algum lado na faixa, em algum minuto 10–35. É o denominador do PPG.
6. **Escolha a cada par (cada regra):** métricas de cada opção só com o histórico [w ou W comum, a − EMBARGO), com a mesma conta rápida (somas acumuladas por jogo) e uma conta direta de conferência em decisões sorteadas. Elegível = ≥ 300 jogos executáveis no histórico próprio. A regra escolhe a melhor elegível, e a opção escolhida aposta em a, a+1.
7. **Avaliação:**
   - PPG nos mesmos jogos para todas as regras, com IC por bootstrap de clusters de 10 gameids;
   - comparação contra a mesma regra por bloco, contra benchmarks (V6_MOM fixo, modelos fixos, favorito, zebra, VIG por dutching), contra o acaso (média e distribuição) e contra as outras regras (comparação par a par);
   - MCS e Reality Check;
   - risco por jogo e por aposta, e overfit (otimismo, percentil fora da amostra).

## 2. Novidades da v5.0 (pedido do usuário)
- **EV > 0 / 5% / 10%.** A entrada de hoje é edge = p − 1/odd > 0. Para odd > 0, isso é exatamente EV = p·odd − 1 > 0 (multiplicar por odd > 0 preserva a desigualdade); VERIFICADO na configuração do motor v9.0.1: EDGE_MIN 0.0, LIMIAR_EM "edge". Os limiares de 5% e 10% são aplicados pelo `filtro_col` do motor, com a coluna (flag==1) & (EV > limiar), no lado já escolhido (maior edge), sem mudar a função copiada.
- **CLV / markout.** Para cada aposta (jogo, minuto t, lado), com q = probabilidade sem margem e odd:
  - em t+h (h = 5…25): mk = q(t+h) − q(t), % a favor, "ROI markout" = odd·q(t+h) − 1 e green-up = odd/odd(t+h) − 1;
  - fechamento = última odd do jogo DEPOIS de t, nos minutos 10–35 e, se existirem, zz40/zz45 só como odds: CLV = odd·q_fech − 1; supera = q_fech > q(t);
  - novas métricas de histórico (clv_medio, clv_taxa, n_clv) e regras (clv_medio, clv_mk = média de q_fech − q, clv_mk_com_skill). A taxa de superação é só descritiva (ver seção 6).
- **Modo sombra.** Quando a regra quer trocar, a produção fica congelada e a nova roda em sombra. Depois de N = 50 jogos executáveis, compara (só jogos < início do par − embargo) CLV médio ou Brier skill. Vence → promove; perde → a sombra recomeça.
- **ROI × esperado:**
  - EV médio do modelo;
  - EV calibrado = ROI realizado pelas apostas ANTERIORES (jogos < início do par) da mesma opção na mesma faixa fixa de EV, voltando ao EV do modelo se houver menos de 30;
  - regressão lucro ~ a + b·EV com erro-padrão sanduíche por cluster (descritiva);
  - variância por aposta real × esperada pelo modelo;
  - IC de ROI e de CLV por bootstrap de clusters.
- **Monitor (só alarme, não decide):** ROI / CLV / Brier skill rolantes. Não há coluna de data conhecida, então as janelas são de 300/900 jogos; se houver coluna de data, 30/90 dias. Inclui drawdown, alarmes, gráficos SVG e amostra de apostas.

## 3. Proteções contra vazamento (com prova)
| proteção | onde | prova |
|---|---|---|
| treino < par | motor + `conferir_retreino` + auditor A2/A3 | sintético: 11.688 treinos, folga mínima 1 jogo |
| linhas do lockbox fora da memória | `carregar_brutos` | descartadas na leitura + assert; A1 nas saídas |
| escolha só com o passado | `valores_rapido` (somas até searchsorted(G, a − embargo)) | sabotagem interna: futuro trocado por ruído em decisões sorteadas → 0 escolhas mudam; aborta se mudar. Cobre também CLV (CS/CN/CB, clv/supera das apostas) e a sombra |
| sabotagem de ponta a ponta | zz com jogos ≥ 600 trocados (odds, flags, resultado), 30% apagados, texto no lockbox | 457.056 decisões a ≤ 600 idênticas; livro (172.593 apostas, com CLV/fechamento/EV calibrado) e eventos da sombra idênticos |
| CLV de um jogo passado | `clv_campos` usa só minutos > t do MESMO jogo | A11 refaz das odds (222.315 apostas, 0 diferenças). O jogo já terminou antes da decisão (embargo 0 = supõe o par anterior terminado) |
| EV calibrado | `apostas_serie`: searchsorted(g da opção, a_par) | A12: última aposta usada < início do par em todas as apostas |
| sombra | janela [s0, a − embargo) | teste unitário (futuro adulterado não muda a produção) + sabotagem interna + A13 refaz dos eventos |

## 4. Testes executados (sintéticos)
- **Unitários:** `teste_v50.py` 14 OK (CLV, fechamento, sombra, EV calibrado, rolante, limiar de EV no FIRST) e `teste_regras.py` 15 OK.
- **Rodadas finais (código depois da revisão):** base A1–A15 TUDO OK; continuação A1–A15 TUDO OK; continuação × normal: 567.456 decisões antes do antigo lockbox idênticas.
- **Regressão:** regras e benchmarks antigos com EV > 0 têm diferença 0 em relação à v4.8 (trilhas e resumo).
- **Sabotagem de ponta a ponta (código final):** 457.056 decisões, 172.626 apostas do livro (CLV, fechamento, EV calibrado) e 11.109 eventos da sombra antes do corte idênticos.

## 5. O que NÃO foi verificado / ressalvas
- **Dados reais e motor real:** nada rodou com os dados reais nem com o motor v9.0 real; usei o v9.0.1 sem a seção 10.
- **Tempo de execução:** no real é NÃO SEI. O sintético levou 3,6× a v4.8 (24 funis).
- **"Fechamento" in-play é uma definição minha** (última odd disponível do jogo); não há fechamento natural ao vivo. Viés de sobrevivência: aposta sem minuto posterior fica sem CLV (a porcentagem é mostrada).
- **"30/90 dias" virou 300/900 jogos:** não sei quantos jogos há por dia.
- **Green-up usa a odd de back em t+h,** sem o spread de lay (otimista).
- **Comparações de flags diferentes:** Brier skill de opções com flags diferentes é medido em linhas diferentes (cada flag tem o seu universo).
- **Muitas regras × funis × janelas:** usar Reality Check/MCS. Os jogos são de desenvolvimento, já vistos em análises.

## 6. Revisão adversarial (03/10) — achados e o que foi feito
| achado | gravidade | verificado? | correção |
|---|---|---|---|
| "% que supera o fechamento" não tem 50% como referência (depende do preço; simulação martingale: q 0,15–0,30 → 37%) — as regras clv_taxa e o alarme favoreciam favoritos | ALTA | sim (cético 0,85) | decisão, sombra "clv" e alarme passam a usar o movimento médio q_fech − q (referência 0); a taxa fica só descritiva; regras clv_mk / clv_mk_com_skill |
| % do tempo em alarme contava janelas incompletas como "sem alarme" | média | sim (0,95) | % só sobre pontos com janela cheia |
| sombra: produção sem amostra nunca era comparada e nada registrava | média | sim (0,55) | evento "indeterminada" (1× por janela) e contagem por série |
| tabela de EV sem diferença pareada | média | sim (0,70) | ev_comparacao_pareada.csv e Δ (z) no RESUMO |
| fechamento in-play perto do resultado | média | parcialmente (0,6 / outro cético: não é defeito) | aviso no RESUMO e LEIA; horizontes fixos recomendados |
| EV calibrado ignorava EMBARGO_ESCOLHA (só se > 0) | baixa | — | corte em início do par − embargo; A12 idem |
| sabotagem interna não perturbava L2/N2 e contagens | info | — | agora perturba todos os canais |
| green-up com a odd de back do mesmo lado (otimista) | baixa | — | green-up real: hedge no outro lado, na odd da casa: odd·(1 − 1/odd_outro) − 1 |
| −2 (não apostar) virava "manter produção" na sombra | baixa | — | produção não aposta naquele par; A13 idem |
| métricas novas sem corte por período (antigo lockbox) | baixa | — | colunas por período (n, ROI, EV, CLV, mk) |
| SVG sem escape de texto | baixa | — | escape XML |
| A13 é consistência, não prova de não-vazamento | baixa | — | rótulo corrigido (a prova é a sabotagem) |
| jogos simultâneos na fronteira dos pares (embargo 0) | — | cético: premissa do usuário, não defeito | LEIA recomenda rodar também com EMBARGO_ESCOLHA = 2 |
| EV calibrado com viés de seleção (winner's curse) nas regras | baixa | — | documentado: compare com BASE (opção fixa, sem esse viés) |
