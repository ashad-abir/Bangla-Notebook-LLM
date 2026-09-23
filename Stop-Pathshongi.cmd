@echo off
title Stop Pathshongi
cd /d "%~dp0"
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\stop-pathshongi.ps1"
pause
