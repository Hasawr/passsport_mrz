@echo off
setlocal

rem Always run from the folder containing this script.
cd /d "%~dp0"

set "PYTHON_COMMAND=python"
if exist ".venv\Scripts\python.exe" (
    set "PYTHON_COMMAND=%~dp0.venv\Scripts\python.exe"
)

%PYTHON_COMMAND% --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python was not found.
    echo Install Python 3.10 or newer and try again.
    pause
    exit /b 1
)

if not exist ".env" (
    copy /y ".env.example" ".env" >nul
    %PYTHON_COMMAND% -c "from pathlib import Path; import secrets; p=Path('.env'); p.write_text(p.read_text(encoding='utf-8').replace('replace-with-a-long-random-key', secrets.token_urlsafe(48)), encoding='utf-8')"
    if errorlevel 1 (
        echo [ERROR] Failed to generate a secure API key.
        exit /b 1
    )
    echo [INFO] Created .env with a random API key.
)

%PYTHON_COMMAND% -c "import fastapi, uvicorn, streamlit, cv2, paddle; from services.passport.detector import PassportDetector" >nul 2>&1
if errorlevel 1 (
    echo [INFO] Installing missing dependencies...
    %PYTHON_COMMAND% -m pip install -r requirements.txt
    if errorlevel 1 (
        echo [ERROR] Dependency installation failed.
        pause
        exit /b 1
    )
)

if /i "%~1"=="--check" (
    echo [OK] Configuration and dependencies are ready.
    exit /b 0
)

echo Starting Passport OCR backend at http://127.0.0.1:8000
echo [SECURITY] Streamlit is local-only. Do not expose port 8501 publicly.

start "Passport OCR API Backend" cmd /k ""%PYTHON_COMMAND%" -m uvicorn api.main:app --host 127.0.0.1 --port 8000"

echo Waiting for the Passport OCR backend to become healthy...
for /l %%I in (1,1,60) do (
    powershell -NoProfile -Command "try { $response = Invoke-WebRequest -UseBasicParsing -Uri 'http://127.0.0.1:8000/health' -TimeoutSec 1; if ($response.StatusCode -eq 200) { exit 0 }; exit 1 } catch { exit 1 }" >nul 2>&1
    if not errorlevel 1 goto backend_ready
    timeout /t 1 /nobreak >nul
)

echo [ERROR] Passport OCR backend did not become healthy within 60 seconds.
echo Review the Passport OCR API Backend window for startup errors.
pause
exit /b 1

:backend_ready
echo Starting Streamlit frontend at http://127.0.0.1:8501
start "Passport OCR Streamlit Frontend" cmd /k ""%PYTHON_COMMAND%" -m streamlit run demos\streamlit_app.py --server.address 127.0.0.1 --server.port 8501"

echo.
echo Both applications were opened in separate terminal windows.
echo Close those windows to stop the applications.
endlocal
