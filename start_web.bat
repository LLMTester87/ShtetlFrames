@echo off
cd /d "%~dp0"
set PYTHONPATH=%~dp0src
title ShtetlFrames

echo.
echo  Closing any old ShtetlFrames window...
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$pids = @(); " ^
  "try { $pids += (Get-NetTCPConnection -LocalPort 8787 -ErrorAction SilentlyContinue | Select-Object -ExpandProperty OwningProcess -Unique) } catch {}; " ^
  "Get-CimInstance Win32_Process -ErrorAction SilentlyContinue | Where-Object { $_.CommandLine -match 'serve\.py' } | ForEach-Object { $pids += $_.ProcessId }; " ^
  "$pids = $pids | Where-Object { $_ -and $_ -ne 0 } | Select-Object -Unique; " ^
  "foreach ($p in $pids) { try { Stop-Process -Id $p -Force -ErrorAction SilentlyContinue } catch {} }"

timeout /t 1 /nobreak >nul

start "" cmd /c "timeout /t 2 /nobreak >nul & start http://127.0.0.1:8787/"
".venv\Scripts\python.exe" src\serve.py
echo.
echo  Closed.
pause
