@echo off
setlocal
title MED - Reentrenamiento con correcciones

set "ROOT=%~dp0"
set "PROJECT_ROOT=%ROOT%.."
set "PYTHON=%PROJECT_ROOT%\.venv\Scripts\python.exe"
set "SCRIPT=%ROOT%scripts\retrain_from_correcciones.py"
set "BASE_DATASET=%ROOT%datos\med_dataset_v8.csv"
set "CORRECTIONS=%ROOT%datos\correcciones.csv"
set "START_TIME=%time%"

echo ==============================================
echo  MED - Reentrenamiento con correcciones
echo ==============================================
echo [INFO] Iniciando ejecucion...
echo [INFO] Script : %SCRIPT%
echo [INFO] Dataset : %BASE_DATASET%
echo [INFO] Correcciones : %CORRECTIONS%
echo.

if not exist "%PYTHON%" (
  echo [ERROR] No existe: "%PYTHON%"
  echo Crea el entorno virtual en la raiz del proyecto.
  pause
  exit /b 1
)

if not exist "%BASE_DATASET%" (
  echo [ERROR] No existe el dataset base:
  echo "%BASE_DATASET%"
  pause
  exit /b 1
)

if not exist "%CORRECTIONS%" (
  echo [ERROR] No existe el CSV de correcciones:
  echo "%CORRECTIONS%"
  echo Crea el archivo con columnas text y emotion.
  pause
  exit /b 1
)

REM Si no pasan argumentos, usa el dataset base y el CSV de correcciones por defecto
if "%~1"=="" (
  echo [RUNNING] Reentrenando con correcciones por defecto...
  "%PYTHON%" "%SCRIPT%" --base-dataset "%BASE_DATASET%" --corrections "%CORRECTIONS%" --keep-merged --overwrite
  ) else (
  echo [RUNNING] Reentrenando con los parametros proporcionados...
  "%PYTHON%" "%SCRIPT%" %*
)

set "RC=%errorlevel%"
set "END_TIME=%time%"
echo.
echo [INFO] Inicio : %START_TIME%
echo [INFO] Final  : %END_TIME%
echo Proceso finalizado con codigo %RC%.
if "%RC%"=="0" (
  echo [OK] Reentrenamiento completado correctamente.
  ) else (
  echo [ERROR] El reentrenamiento termino con errores.
)
pause
exit /b %RC%
