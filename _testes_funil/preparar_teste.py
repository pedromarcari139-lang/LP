# Gera, na pasta atual, uma cópia de ../validar_funil.py com parâmetros PEQUENOS para os zz sintéticos (gera_zz_sinteticos.py).
# Uso (numa pasta com backtest_sujo.py, paralelo_v90.py e os zz sintéticos):
#   python gera_zz_sinteticos.py && python preparar_teste.py && NPR=1 python validar_funil.py
import re, os
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "validar_funil.py"), encoding="utf-8").read()
rep = {
 r"^LOCKBOX_SERIO = 8448": "LOCKBOX_SERIO = 700",
 r"^MIN_HIST = None": "MIN_HIST = 50",
 r"^MIN_TREINO_NOVOS = \{5: 150, 40: 120, 45: 120\}": "MIN_TREINO_NOVOS = {5: 60, 40: 60, 45: 40}",
 r"^N_PROCESSOS = 0": "N_PROCESSOS = int(__import__('os').environ.get('NPR', '1'))",
 r"^VERIFICAR_LIMPO = True": "VERIFICAR_LIMPO = False",
 r"^B_BOOT = 5000": "B_BOOT = 300",
 r"^CORTE_TESTE = 7729": "CORTE_TESTE = 600",
 r"^CFG_EXTRA = \{\}": "CFG_EXTRA = dict(G_CLEAN=400, INICIO_TESTE=500, INICIO_TESTE_SUJO=200, BLOCO_JOGOS=50, BLOCO_JOGOS_SUJO=100, MIN_TREINO_JOGOS={10:60,15:60,20:60,25:60,30:60,35:50}, FAMILIAS=['V6','TUDO','V6E','V6WR'], CAL_PARA=['TUDO_INI'], SELECAO_INICIO_AVALIACAO=500, SELECAO_MIN_HIST_JOGOS=50, PRO_LIVE_FROM=420)",
}
for k, v in rep.items():
    src, n = re.subn(k, v, src, flags=re.M); assert n == 1, k
open("validar_funil.py", "w", encoding="utf-8").write(src)
print("validar_funil.py de teste gravado")
