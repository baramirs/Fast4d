@echo off
setlocal
set TCL_LIBRARY=C:\Users\jtapiaca.ASURITE\.conda\envs\py4dstem-01419\Library\lib\tcl8.6
set TK_LIBRARY=C:\Users\jtapiaca.ASURITE\.conda\envs\py4dstem-01419\Library\lib\tk8.6
set CONDA_PREFIX=C:\Users\jtapiaca.ASURITE\.conda\envs\py4dstem-01419
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
cd /d "%~dp0"

set PY=C:\Users\jtapiaca.ASURITE\.conda\envs\py4dstem-01419\python.exe
if not exist "%PY%" (
    echo [ERROR] Python not found: %PY%
    echo Install or fix the conda env py4dstem-01419.
    pause
    exit /b 1
)

"%PY%" app.py
if errorlevel 1 (
    echo.
    echo [ERROR] Fast4D exited with code %errorlevel%.
    echo Check the traceback above.
    pause
    exit /b %errorlevel%
)
