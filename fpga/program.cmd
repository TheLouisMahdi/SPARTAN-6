@echo off
setlocal
cd /d "%~dp0"
if not exist ..\assets\logs mkdir ..\assets\logs
if not defined XSDB set "XSDB=xsdb.bat"
call "%XSDB%" program.tcl > ..\assets\logs\program.log 2>&1
set "result=%errorlevel%"
type ..\assets\logs\program.log
findstr /c:"PASS: actual CNN/UART" ..\assets\logs\program.log >nul
if errorlevel 1 exit /b 1
exit /b %result%
