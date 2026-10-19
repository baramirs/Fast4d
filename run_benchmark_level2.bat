@echo off
REM Level-2 benchmark (drawn 4D patterns + real detection). Same Python as run_gui.bat.
REM Usage: run_benchmark_level2.bat tools\benchmark\level2\<config>.json
REM For the official run set "detect_cuda": true in the config (GUI uses the GPU).
setlocal
set CONDA_PREFIX=C:\Users\jtapiaca.ASURITE\.conda\envs\py4dstem-01419
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set QT_QPA_PLATFORM=offscreen
cd /d "%~dp0"
set PY=%CONDA_PREFIX%\python.exe
if "%~1"=="" (
    echo Usage: run_benchmark_level2.bat tools\benchmark\level2\config_L2_stageB_rotation.json
    pause
    exit /b 1
)
"%PY%" tools\benchmark\level2\run_level2.py "%~1"
if errorlevel 1 (
    echo [ERROR] Benchmark aborted ^(code %errorlevel%^).
    pause
    exit /b %errorlevel%
)
echo Done. Run: "%PY%" tools\benchmark\level2\analyze_level2.py benchmark_results\^<folder^>
pause
