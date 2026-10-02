@echo off
setlocal EnableExtensions
cd /d "%~dp0"
title Validar funil (re-treino e re-escolha a cada 2 jogos) - NAO FECHE ESTA JANELA
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
chcp 65001 >nul

rem Tem de ficar na MESMA pasta do RODAR.bat (usa o .venv, o backtest_sujo.py, os zz e a BANCO_PREVISOES de la).
if not exist ".venv\Scripts\python.exe" (
  echo  Nao achei o .venv desta pasta. Copie validar_funil.py e RODAR_FUNIL.bat para a pasta do RODAR.bat
  echo  ^(a mesma do backtest_sujo.py e da BANCO_PREVISOES^) e rode de novo.
  pause
  exit /b 1
)
for %%F in (zz5.xlsx zz40.xlsx zz45.xlsx) do (
  if not exist "%%F" (
    echo  Falta %%F nesta pasta ^(mesmo banco dos outros zz^). Copie e rode de novo.
    pause
    exit /b 1
  )
)
".venv\Scripts\python.exe" -u validar_funil.py
if errorlevel 1 goto :falhou
echo.
echo  AUDITORIA INDEPENDENTE DAS SAIDAS ^(auditar_saidas.py^)...
".venv\Scripts\python.exe" -u auditar_saidas.py OUT_FUNIL
if errorlevel 1 goto :falhou
echo.
echo  TERMINOU ^(auditoria OK^). Traga a pasta OUT_FUNIL ^(o essencial esta em OUT_FUNIL\RESUMO_FUNIL.txt e OUT_FUNIL\AUDITORIA_SAIDAS.txt^).
pause
exit /b 0

:falhou
echo.
echo  PAROU com erro. Tire uma foto/copie a mensagem acima e traga a pasta OUT_FUNIL.
pause
exit /b 1
