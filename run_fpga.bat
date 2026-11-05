@echo off
setlocal
cd /d "%~dp0"
if "%~2"=="" goto usage
if /i "%~1"=="pretrained" goto pretrained
if /i "%~1"=="aeee2" goto run
if /i "%~1"=="grayscale2" goto run
:usage
echo Usage: run_fpga.bat pretrained^|aeee2^|grayscale2 COM26
exit /b 2
:run
call "%~dp0run.cmd" fpga --encoder "%~1" --manifest "assets\models\%~1\manifest.json" --port "%~2"
exit /b %errorlevel%
:pretrained
call "%~dp0run.cmd" experiment cew --mode fpga --port "%~2"
exit /b %errorlevel%
