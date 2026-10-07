@echo off
REM One-click demo start (Windows): API on :8000 + UI dev server on :5173, then opens the browser.
REM Works offline in demo mode - no model weights needed.
cd /d "%~dp0"

where python >nul 2>nul || (echo Python was not found. Install Python 3.10+ and re-run. & pause & exit /b 1)
where npm >nul 2>nul || (echo Node.js/npm was not found. Install Node 20+ and re-run. & pause & exit /b 1)

python -c "import fastapi, uvicorn, multipart" >nul 2>nul || (
  echo Installing API dependencies...
  python -m pip install -r requirements_api.txt || (echo pip install failed & pause & exit /b 1)
)
if not exist "frontend\node_modules" (
  echo Installing UI dependencies ^(first run only^)...
  pushd frontend && call npm install && popd
)

start "DeepTrace API (port 8000)" cmd /k "cd /d %~dp0 && python -m uvicorn api:app --port 8000"
start "DeepTrace UI (port 5173)" cmd /k "cd /d %~dp0frontend && npm run dev -- --host 127.0.0.1 --port 5173"

echo Waiting for the servers to start...
timeout /t 8 /nobreak >nul
start "" http://localhost:5173
echo.
echo Opened http://localhost:5173  -  close the two console windows to stop.
echo Single-command alternative: cd frontend ^&^& npm run build ^&^& cd .. ^&^& python -m uvicorn api:app --port 8000  ^(then open http://localhost:8000^)
