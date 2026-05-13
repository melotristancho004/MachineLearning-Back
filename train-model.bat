@echo off
setlocal

set "ROOT=%~dp0"
set "PYTHON=%ROOT%.venv\Scripts\python.exe"
set "SCRIPT=%ROOT%scripts\train_model.py"
set "DEFAULT_DATASET=%ROOT%datos\med_dataset_v8.csv"

if not exist "%PYTHON%" (
  echo [ERROR] No existe: "%PYTHON%"
  echo Crea el entorno virtual en la raiz del proyecto.
  pause
  exit /b 1
)

if not exist "%DEFAULT_DATASET%" (
  echo [ERROR] No existe el dataset por defecto:
  echo "%DEFAULT_DATASET%"
  pause
  exit /b 1
)

REM Si no pasan argumentos, usa dataset por defecto
if "%~1"=="" (
  "%PYTHON%" "%SCRIPT%" --dataset "%DEFAULT_DATASET%"
) else (
  "%PYTHON%" "%SCRIPT%" %*
)

set "RC=%errorlevel%"
echo.
echo Proceso finalizado con codigo %RC%.
pause
exit /b %RC%