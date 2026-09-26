@echo off
cd /d "%~dp0"
py -3.11 -m venv .venv
if errorlevel 1 goto error
.venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 goto error
echo Instalacion terminada. Ejecuta iniciar.cmd.
pause
exit /b 0
:error
echo No se pudo instalar. Comprueba Python 3.11 y la conexion a Internet.
pause
exit /b 1
