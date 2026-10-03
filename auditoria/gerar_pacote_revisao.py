# Junta num ÚNICO .txt (para outro agente revisar): o relatório do método/vazamento/testes (auditoria/RELATORIO_REVISAO_v5_0.md) e o CÓDIGO COMPLETO
# do validar_funil.py, auditar_saidas.py, .bat, LEIA, testes e ferramentas de auditoria — e, se passado, o TRECHO DE PREVISÃO do motor (topo até a seção 5).
# Uso: python gerar_pacote_revisao.py SAIDA.txt [caminho/do/backtest_sujo.py]
import os, sys, hashlib
AQUI = os.path.dirname(os.path.abspath(__file__)); RAIZ = os.path.dirname(AQUI)
saida = sys.argv[1]; motor = sys.argv[2] if len(sys.argv) > 2 else None
arquivos = ["auditoria/RELATORIO_REVISAO_v5_0.md", "validar_funil.py", "auditar_saidas.py", "RODAR_FUNIL.bat", "RODAR_FUNIL_CONTINUACAO.bat", "LEIA_FUNIL.txt",
            "ver_escolhas.py", "_testes_funil/teste_v50.py", "_testes_funil/teste_regras.py", "_testes_funil/preparar_teste.py",
            "auditoria/comparar_rodadas.py", "auditoria/comparar_todas_saidas.py", "auditoria/comparar_v50_sabotagem.py", "auditoria/gerar_variantes.py",
            "auditoria/gerar_motor_v90_simulado.py", "auditoria/refazer_livro_a_mao.py"]
partes = ["PACOTE DE REVISÃO — validar_funil v5.0 (gerado por auditoria/gerar_pacote_revisao.py)", "Conteúdo: " + ", ".join(arquivos) + (" + trecho de previsão do motor" if motor else ""), ""]
for a in arquivos:
    fp = os.path.join(RAIZ, a)
    if not os.path.exists(fp): partes.append(f"\n{'=' * 100}\n### {a}: NÃO ENCONTRADO\n"); continue
    txt = open(fp, encoding="utf-8", errors="replace").read()
    partes.append(f"\n{'=' * 100}\n### ARQUIVO: {a}  ·  {txt.count(chr(10))} linhas  ·  SHA-256 {hashlib.sha256(open(fp, 'rb').read()).hexdigest()}\n{'=' * 100}\n{txt}")
if motor and os.path.exists(motor):
    src = open(motor, encoding="utf-8").read(); i = src.find("\n# 5. BASE LONGA")
    partes.append(f"\n{'=' * 100}\n### MOTOR {os.path.basename(motor)} — TRECHO DE PREVISÃO (topo até '# 5. BASE LONGA', onde estão carga, preparar, features, ajuste, "
                  f"walk-forward e banco; SHA-256 do arquivo {hashlib.sha256(src.encode('utf-8')).hexdigest()})\n{'=' * 100}\n{src[:i]}")
open(saida, "w", encoding="utf-8").write("\n".join(partes)); print(f"gravado {saida} ({os.path.getsize(saida) / 1e6:.2f} MB)")
