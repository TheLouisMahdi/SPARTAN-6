@echo off
setlocal
set "ROOT=%~dp0.."
cd /d "%ROOT%"
set "LOCALPY=%ROOT%\RESULTS\runtime\Scripts\python.exe"
if not exist "%LOCALPY%" (
  echo Project Python environment is missing. See RUN.md for setup. 1>&2
  exit /b 2
)
"%LOCALPY%" -B "%ROOT%\PC\app.py" %*
exit /b %errorlevel%
