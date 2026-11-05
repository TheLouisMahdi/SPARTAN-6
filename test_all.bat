@echo off
setlocal
cd /d "%~dp0"
call "%~dp0run.cmd" test-all %*
exit /b %errorlevel%
