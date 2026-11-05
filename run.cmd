@echo off
setlocal
cd /d "%~dp0"
set PYTHONDONTWRITEBYTECODE=1
if not defined PYTHON set "PYTHON=python"
"%PYTHON%" -B "%~dp0python\app.py" %*
exit /b %errorlevel%
