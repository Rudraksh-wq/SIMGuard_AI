@echo off
rem Stops the SIMGuard AI API and frontend started by setup.bat.
call "%~dp0setup.bat" --stop --no-pause %*
echo.
pause
