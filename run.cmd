@echo off
setlocal
cd /d "%~dp0"
set PYTHONDONTWRITEBYTECODE=1
"%~dp0assets\runtime\Scripts\python.exe" -B "%~dp0python\app.py" %*
exit /b %errorlevel%
