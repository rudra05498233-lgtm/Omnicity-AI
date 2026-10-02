@echo off
setlocal enabledelayedexpansion
title OmniCity AI — Starting...
color 0B
cls

:: ── This .bat lives INSIDE the omni folder alongside backend.py ──────────
set "OMNIDIR=%~dp0"
if "%OMNIDIR:~-1%"=="\" set "OMNIDIR=%OMNIDIR:~0,-1%"

:: README is also in the same folder
set "READMEFILE=%OMNIDIR%\README.html"

echo.
echo  =====================================================================
echo    OmniCity AI  ^|  Autonomous Urban Operating System  ^|  v4.2.0
echo  =====================================================================
echo.
echo  Running from: %OMNIDIR%
echo.

:: ── Sanity check ─────────────────────────────────────────────────────────
if not exist "%OMNIDIR%\backend.py" (
    echo  [!] ERROR: Cannot find backend.py in this folder.
    echo      Make sure START_OMNICITYAI.bat is in the same folder as backend.py
    echo      Current location: %OMNIDIR%
    echo.
    pause
    exit /b 1
)
echo  [OK] Project files found.
echo.

:: ── Check Python ─────────────────────────────────────────────────────────
echo  [1/9] Checking Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo.
    echo  [!] ERROR: Python not found on this machine.
    echo      Download from: https://www.python.org/downloads/
    echo      During install tick "Add Python to PATH" then run this again.
    echo.
    pause
    exit /b 1
)
for /f "tokens=*" %%v in ('python --version 2^>^&1') do echo  [OK] %%v detected.
echo.

:: ── Install packages ─────────────────────────────────────────────────────
echo  [2/9] Installing core server  ^(fastapi, uvicorn, sqlalchemy, httpx^)...
pip install fastapi uvicorn sqlalchemy httpx --disable-pip-version-check --quiet 2>nul
echo  [OK] Done.
echo.

echo  [3/9] Installing image tools  ^(pillow, opencv-python, numpy^)...
pip install pillow opencv-python numpy --disable-pip-version-check --quiet 2>nul
echo  [OK] Done.
echo.

echo  [4/9] Installing ML libraries  ^(scikit-learn, pandas, networkx^)...
pip install scikit-learn pandas networkx --disable-pip-version-check --quiet 2>nul
echo  [OK] Done.
echo.

echo  [5/9] Installing PyTorch  ^(~800MB — please wait, do NOT close^)...
pip install torch torchvision --disable-pip-version-check --quiet 2>nul
echo  [OK] Done.
echo.

echo  [6/9] Installing FaceNet + MTCNN  ^(face recognition^)...
pip install facenet-pytorch --disable-pip-version-check --quiet 2>nul
echo  [OK] Done.
echo.

echo  [7/9] Installing YOLOv8  ^(object detection^)...
pip install ultralytics --disable-pip-version-check --quiet 2>nul
echo  [OK] Done.
echo.

echo  [8/9] Installing CLIP + Transformers  ^(HuggingFace^)...
pip install transformers huggingface_hub --disable-pip-version-check --quiet 2>nul
echo  [OK] Done.
echo.

echo  [9/9] Installing EasyOCR  ^(Aadhaar card reader^)...
pip install easyocr --disable-pip-version-check --quiet 2>nul
echo  [OK] Done.
echo.

echo  =====================================================================
echo   All packages ready.  Launching OmniCity AI...
echo  =====================================================================
echo.

:: ── Launch SERVER in its own window ──────────────────────────────────────
echo  [*] Starting backend server...
start "OmniCity AI — SERVER (keep open)" cmd /k "title OmniCity AI - SERVER && color 0B && cd /d "%OMNIDIR%" && echo. && echo  Server starting — wait for Uvicorn ready message below. && echo  DO NOT close this window while using the site. && echo. && python backend.py"

echo  [*] Waiting for server to boot up...
timeout /t 7 /nobreak >nul

:: ── Open README from same folder ─────────────────────────────────────────
echo  [*] Opening README guide...
if exist "%READMEFILE%" (
    start "" "%READMEFILE%"
) else (
    echo  [!] README.html not found, skipping.
)
timeout /t 2 /nobreak >nul

:: ── Open the site ────────────────────────────────────────────────────────
echo  [*] Opening OmniCity AI in browser...
start "" "http://localhost:8000/index.html"
timeout /t 1 /nobreak >nul

:: ── Permanent green links window ─────────────────────────────────────────
start "OmniCity AI — ALL LINKS" cmd /k "title OmniCity AI — LINKS && color 0A && cls && echo. && echo  ================================================================ && echo    OmniCity AI is RUNNING — use these links in your browser: && echo  ================================================================ && echo. && echo    [1]  http://localhost:8000/index.html          ^<-- START HERE && echo. && echo    [2]  http://localhost:8000/dashboard.html       City Dashboard && echo. && echo    [3]  http://localhost:8000/cctv_monitor.html    CCTV Monitor && echo. && echo    [4]  http://localhost:8000/citizen_walker.html  Walker View && echo. && echo  ---------------------------------------------------------------- && echo    DEMO LOGIN  ^(no registration needed^): && echo. && echo      Name           :  AER && echo      Aadhaar last 4 :  9012 && echo. && echo      Open index.html then click SIGN IN && echo. && echo  ---------------------------------------------------------------- && echo    Keep the blue SERVER window open in the background. && echo    You can minimise this window anytime. && echo."

echo.
echo  [OK] All done! Closing this window in 5 seconds...
timeout /t 5 /nobreak >nul
endlocal
exit
