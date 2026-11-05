@echo off
setlocal
cd /d "%~dp0"
if not defined ISE_SETTINGS set "ISE_SETTINGS=C:\Xilinx\14.7\ISE_DS\settings64.bat"
if not exist "%ISE_SETTINGS%" exit /b 1
call "%ISE_SETTINGS%"
if errorlevel 1 exit /b 1
xtclsh create_project.tcl > synthesize.log 2>&1
if errorlevel 1 exit /b 1
xtclsh build.tcl > build.log 2>&1
if errorlevel 1 exit /b 1
impact -batch generate_svf.cmd > generate_svf.log 2>&1
exit /b %errorlevel%
