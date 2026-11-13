@echo off
setlocal
cd /d "%~dp0"
if not exist ..\RESULTS\logs mkdir ..\RESULTS\logs
call C:\Xilinx\Vivado\2023.2\bin\xsdb.bat program.tcl > ..\RESULTS\logs\program.log 2>&1
set "result=%errorlevel%"
type ..\RESULTS\logs\program.log
findstr /c:"PASS: actual CNN/UART" ..\RESULTS\logs\program.log >nul
if errorlevel 1 exit /b 1
exit /b %result%
