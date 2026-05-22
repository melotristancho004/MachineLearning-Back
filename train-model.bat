@echo off
setlocal
title MED - Entrenamiento en curso

set "ROOT=%~dp0"
set "PROJECT_ROOT=%ROOT%.."
set "PYTHON=%PROJECT_ROOT%\.venv\Scripts\python.exe"
set "SCRIPT=%ROOT%scripts\train_model.py"
set "DEFAULT_DATASET=%ROOT%datos\med_dataset_v8.csv"
set "START_TIME=%time%"

echo ==============================================
echo  MED - Entrenamiento del modelo
echo ==============================================
echo [INFO] Iniciando ejecucion...
echo [INFO] Script : %SCRIPT%
echo [INFO] Dataset : %DEFAULT_DATASET%
echo.

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
  echo [RUNNING] Entrenando con el dataset por defecto...
  "%PYTHON%" "%SCRIPT%" --dataset "%DEFAULT_DATASET%"
) else (
  echo [RUNNING] Entrenando con los parametros proporcionados...
  "%PYTHON%" "%SCRIPT%" %*
)

set "RC=%errorlevel%"
set "END_TIME=%time%"
echo.
echo [INFO] Inicio : %START_TIME%
echo [INFO] Final  : %END_TIME%
echo Proceso finalizado con codigo %RC%.
if "%RC%"=="0" (
  echo [OK] Entrenamiento completado correctamente.
) else (
  echo [ERROR] El entrenamiento termino con errores.
)
pause
exit /b %RC%