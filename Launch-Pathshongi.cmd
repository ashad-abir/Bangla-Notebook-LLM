@echo off
title Pathshongi Launcher
cd /d "%~dp0"
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\launch-pathshongi.ps1"
if errorlevel 1 pause
