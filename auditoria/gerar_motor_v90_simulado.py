# Gera um motor "tipo v9.0" a partir do motor v9.0.1: tira a seção 10 inteira (do cabeçalho "# 10. v9.0.1" até antes do
# 'if __name__ == "__main__":'). É o que falta no motor do PC da Amanda (erro: module 'bt_funil_sujo' has no attribute '_alerta_epv_atual').
# Segundo o docs/MUDANCAS_v9.0.1.md, o topo (até "# 5. BASE LONGA", sem o CFG) é byte a byte o da v9.0 e as correções da v9.0.1 estão todas
# na seção 10; as seções 5–9 da v9.0 real NÃO estão disponíveis aqui (por isso o validar_funil v4.6 usa cópias próprias das funções de avaliação).
# Uso: python gerar_motor_v90_simulado.py PASTA_COM_O_MOTOR_v901 PASTA_DESTINO
import os, sys, hashlib
orig, dest = sys.argv[1], sys.argv[2]; os.makedirs(dest, exist_ok=True)
for arq in ("backtest_sujo.py", "backtest_limpo.py"):
    src = open(os.path.join(orig, arq), encoding="utf-8").read()
    i = src.find("\n# =============================================================================\n# 10. v9.0.1")
    j = src.find('\nif __name__ == "__main__":')
    assert 0 < i < j, f"{arq}: marcadores da seção 10 não encontrados"
    novo = src[:i + 1] + src[j + 1:]
    assert "_alerta_epv_atual(fits)" not in novo.split("\ndef main(")[0] and "def _alerta_epv_atual" not in novo
    open(os.path.join(dest, arq), "w", encoding="utf-8").write(novo)
    print(f"{arq}: {src.count(chr(10))} → {novo.count(chr(10))} linhas · md5 {hashlib.md5(novo.encode('utf-8')).hexdigest()}")
