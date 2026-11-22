@echo off
title CNN Live Attendance System
echo ========================================================
echo         Launching CNN Live Attendance System...
echo ========================================================
python app.py
if %errorlevel% neq 0 (
    echo.
    echo An error occurred while running the application.
    pause
)
