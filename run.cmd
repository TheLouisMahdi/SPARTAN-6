@echo off
setlocal
cd /d "%~dp0"
set PYTHONDONTWRITEBYTECODE=1
if /i "%~1"=="demo" goto demo
"%~dp0assets\runtime\Scripts\python.exe" -B "%~dp0python\app.py" %*
exit /b %errorlevel%
:demo
"%~dp0assets\runtime\Scripts\python.exe" -B "%~dp0python\demo.py" %*
exit /b %errorlevel%
