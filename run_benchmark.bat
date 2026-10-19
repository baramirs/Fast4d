@echo off
REM Official run of the Fast4D error benchmark (docs\PROTOCOLO_BENCHMARK_ERRORES.md)
REM Uses the SAME Python as run_gui.bat (conda env py4dstem-01419).
setlocal
set CONDA_PREFIX=C:\Users\jtapiaca.ASURITE\.conda\envs\py4dstem-01419
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set QT_QPA_PLATFORM=offscreen
cd /d "%~dp0"

set PY=%CONDA_PREFIX%\python.exe
if not exist "%PY%" (
    echo [ERROR] Python not found: %PY%
    pause
    exit /b 1
)

set CFG=%1
if "%CFG%"=="" set CFG=tools\benchmark\config_level1.json

"%PY%" tools\benchmark\benchmark_errors.py "%CFG%"
if errorlevel 1 (
    echo.
    echo [ERROR] Benchmark aborted ^(code %errorlevel%^). No valid results.
    pause
    exit /b %errorlevel%
)
echo.
echo Done. Results in benchmark_results\
pause
