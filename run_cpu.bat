@echo off
setlocal
cd /d "%~dp0"
if /i "%~1"=="pretrained" goto pretrained
if /i "%~1"=="aeee2" goto run
if /i "%~1"=="grayscale2" goto run
echo Usage: run_cpu.bat pretrained^|aeee2^|grayscale2
exit /b 2
:run
call "%~dp0run.cmd" live --backend cpu --encoder "%~1" --manifest "assets\models\%~1\manifest.json"
exit /b %errorlevel%
:pretrained
call "%~dp0run.cmd" experiment cew --mode cpu
exit /b %errorlevel%
