@echo off
setlocal EnableExtensions
cd /d "%~dp0"
title Validar funil - CONTINUACAO (usa os jogos do antigo lockbox) - NAO FECHE ESTA JANELA
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set FUNIL_CONTINUACAO=1
chcp 65001 >nul

rem Mesma pasta do RODAR.bat. Os zz10..zz35 precisam ter os jogos NOVOS (>= 8448), exportados do mesmo banco.
if not exist ".venv\Scripts\python.exe" (
  echo  Nao achei o .venv desta pasta. Copie os arquivos para a pasta do RODAR.bat e rode de novo.
  pause
  exit /b 1
)
for %%F in (zz10.xlsx zz15.xlsx zz20.xlsx zz25.xlsx zz30.xlsx zz35.xlsx) do (
  if not exist "%%F" (
    echo  Falta %%F nesta pasta. Copie e rode de novo.
    pause
    exit /b 1
  )
)
".venv\Scripts\python.exe" -u validar_funil.py
if errorlevel 1 goto :falhou
echo.
echo  AUDITORIA INDEPENDENTE DAS SAIDAS ^(auditar_saidas.py^)...
".venv\Scripts\python.exe" -u auditar_saidas.py OUT_FUNIL_CONTINUACAO
if errorlevel 1 goto :falhou
echo.
echo  DIAGNOSTICO DE ODDS BAIXAS ^(analisar_odds_baixas.py; so descritivo, nao para o resto se falhar^)...
".venv\Scripts\python.exe" -u analisar_odds_baixas.py OUT_FUNIL_CONTINUACAO
echo.
echo  TERMINOU ^(auditoria OK^). Traga a pasta OUT_FUNIL_CONTINUACAO ^(o essencial: RESUMO_FUNIL.txt, PROXIMAS_ESCOLHAS.txt, AUDITORIA_SAIDAS.txt e ODDS_BAIXAS.txt^).
pause
exit /b 0

:falhou
echo.
echo  PAROU com erro. Tire uma foto/copie a mensagem acima e traga a pasta OUT_FUNIL_CONTINUACAO.
pause
exit /b 1
