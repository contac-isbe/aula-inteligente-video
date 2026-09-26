@echo off
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  echo Ejecuta instalar.cmd primero.
  pause
  exit /b 1
)
.venv\Scripts\python.exe -m aula.local %*
if errorlevel 1 pause
