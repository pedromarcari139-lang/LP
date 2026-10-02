# Uso: python simular_etapa1.py backtest_sujo.py sujo   |   python simular_etapa1.py backtest_limpo.py limpo  (simula o main() da etapa 1 / comparar_lockbox limpo nos zz sintéticos, para testar o reaproveitamento do banco)
import importlib.util, os, sys
arq, modo = sys.argv[1], sys.argv[2]
sp = importlib.util.spec_from_file_location("bt_holdout", arq); B = importlib.util.module_from_spec(sp); sp.loader.exec_module(B)
B.CFG.update(dict(G_CLEAN=400, INICIO_TESTE=500, INICIO_TESTE_SUJO=200, BLOCO_JOGOS=50, BLOCO_JOGOS_SUJO=100, MIN_TREINO_JOGOS={10:60,15:60,20:60,25:60,30:60,35:50}, FAMILIAS=['V6','TUDO','V6E','V6WR'], CAL_PARA=['TUDO_INI'], SELECAO_INICIO_AVALIACAO=500, SELECAO_MIN_HIST_JOGOS=50, PRO_LIVE_FROM=420))
B.CFG.update(dict(B_RC=500, B_MCS=200, B_PERM=50, CELL_POLICY=False, C_FIXO=1.0, INFERENCIA_GID_MIN=None, LOCKBOX_GID_MIN=700, VERIFICAR_LIMPO=False, B_BOOT=200, B_CELULAS=200))
B.CFG["LOCKBOX_ETAPA"] = "desenvolvimento"; B.CFG["PASTA_OUT"] = "OUT_SIM_" + modo; os.makedirs(B.CFG["PASTA_OUT"], exist_ok=True); B.main()
